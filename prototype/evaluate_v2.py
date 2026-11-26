import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import cv2
import faiss
import numpy as np
from PIL import Image
import torch

from config import (
    ANNOTATION_FILE,
    EVALUATION_DIR,
    INDEX_DIR,
    TOP_K,
    VIDEO_DIR,
)
from clip_model import CLIPEncoder
from temporal import calculate_iou


# ============================================================
# V2 Pipeline Hyperparameters
# ============================================================

SAMPLING_FPS = 1.0
EMBEDDING_DIM = 512
MERGE_GAP_SEC = 2.0
SMOOTHING_WINDOW = 3
MIN_SCORE_THRESHOLD = 0.20
NMS_IOU_THRESHOLD = 0.50


# ============================================================
# Encoder Adapter (Works with CLIPEncoder & newer transformers)
# ============================================================

def _extract_tensor(outputs: Any) -> torch.Tensor:
    if isinstance(outputs, torch.Tensor):
        return outputs
    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        return outputs.pooler_output
    if hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
        return outputs.image_embeds
    if hasattr(outputs, "text_embeds") and outputs.text_embeds is not None:
        return outputs.text_embeds
    return outputs[0]


def encode_pil_image(encoder: CLIPEncoder, pil_img: Image.Image) -> np.ndarray:
    if hasattr(encoder, "encode_image"):
        vec = encoder.encode_image(pil_img)
    else:
        inputs = encoder.processor(images=pil_img, return_tensors="pt").to(encoder.device)
        with torch.no_grad():
            outputs = encoder.model.get_image_features(**inputs)
        vec = _extract_tensor(outputs).detach().cpu().numpy()

    vec = np.asarray(vec, dtype=np.float32).reshape(1, -1)
    faiss.normalize_L2(vec)
    return vec


def encode_query_text(encoder: CLIPEncoder, query: str) -> np.ndarray:
    if hasattr(encoder, "encode_text"):
        vec = encoder.encode_text(query)
    else:
        inputs = encoder.processor(text=[query], return_tensors="pt", padding=True).to(encoder.device)
        with torch.no_grad():
            outputs = encoder.model.get_text_features(**inputs)
        vec = _extract_tensor(outputs).detach().cpu().numpy()

    vec = np.asarray(vec, dtype=np.float32).reshape(1, -1)
    faiss.normalize_L2(vec)
    return vec


# ============================================================
# V2 Image Enhancement & CLIP Preprocessing
# ============================================================

def enhance_frame(bgr: np.ndarray) -> np.ndarray:
    """Denoise -> CLAHE -> Sharpen for low-quality/noisy frames."""
    h, w = bgr.shape[:2]
    if w > 960:
        bgr = cv2.resize(bgr, (960, max(1, int(h * 960 / w))), interpolation=cv2.INTER_AREA)

    denoised = cv2.fastNlMeansDenoisingColored(
        bgr, None, h=5, hColor=5, templateWindowSize=7, searchWindowSize=21
    )
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l_chan)
    enhanced = cv2.cvtColor(cv2.merge([l_eq, a_chan, b_chan]), cv2.COLOR_LAB2BGR)
    blur = cv2.GaussianBlur(enhanced, (0, 0), 1.5)
    return cv2.addWeighted(enhanced, 1.4, blur, -0.4, 0)


def branch_clip_pad_square(rgb: np.ndarray, size: int = 224) -> Image.Image:
    """[1:1 Pad] branch: preserves aspect ratio with neutral gray padding."""
    h, w, _ = rgb.shape
    scale = size / max(h, w, 1)
    nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return Image.fromarray(canvas)


# ============================================================
# Find Video & V2 Index Paths
# ============================================================

def find_video(video_id: str) -> Path:
    extensions = [".mp4", ".avi", ".mov", ".mkv"]
    for extension in extensions:
        path = VIDEO_DIR / f"{video_id}{extension}"
        if path.exists():
            return path

    for path in VIDEO_DIR.rglob("*"):
        if path.is_file() and path.stem == video_id:
            return path

    raise FileNotFoundError(f"Video not found: {video_id}")


def get_v2_index_paths(video_id: str) -> Tuple[Path, Path]:
    safe_id = hashlib.md5(video_id.encode()).hexdigest()[:12]
    index_path = INDEX_DIR / f"{safe_id}_v2.index"
    metadata_path = INDEX_DIR / f"{safe_id}_v2.json"
    return index_path, metadata_path


# ============================================================
# V2 Index Builder (FFmpeg Coarse -> Sharpest Window -> Enhance -> Pad)
# ============================================================

def build_index_v2(
    video_path: Path,
    index_path: Path,
    metadata_path: Path,
    encoder: CLIPEncoder,
    sampling_fps: float = SAMPLING_FPS,
) -> None:
    coarse_fps = max(1.0, sampling_fps * 2.0)

    with tempfile.TemporaryDirectory(prefix="chronovision_v2_") as tmp_dir:
        raw_temp_dir = Path(tmp_dir)

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(video_path),
                "-vf",
                f"fps={coarse_fps}",
                str(raw_temp_dir / "raw_%05d.jpg"),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )

        raw_files = sorted(raw_temp_dir.glob("raw_*.jpg"))
        if not raw_files:
            raise ValueError(f"FFmpeg extracted zero frames from: {video_path}")

        # Sharpest-frame selection inside each 2-frame window
        selected: List[Tuple[float, Path, float]] = []
        for i in range(0, len(raw_files), 2):
            window = raw_files[i:i + 2]
            best_path, best_sharpness, best_ts = window[0], -1.0, round(i / coarse_fps, 2)
            for j, fp in enumerate(window):
                img = cv2.imread(str(fp))
                if img is None:
                    continue
                sharp = float(
                    cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
                )
                if sharp > best_sharpness:
                    best_sharpness = sharp
                    best_path = fp
                    best_ts = round((i + j) / coarse_fps, 2)
            selected.append((best_ts, best_path, best_sharpness))

        index = faiss.IndexFlatIP(EMBEDDING_DIM)
        metadata: List[Dict[str, Any]] = []

        for ts, fp, sharpness in selected:
            bgr = cv2.imread(str(fp))
            if bgr is None:
                continue

            enhanced_bgr = enhance_frame(bgr)
            rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
            sq_img = branch_clip_pad_square(rgb)
            vec = encode_pil_image(encoder, sq_img)

            faiss_idx = index.ntotal
            index.add(vec)
            metadata.append(
                {
                    "faiss_idx": faiss_idx,
                    "timestamp": ts,
                    "sharpness": round(sharpness, 2),
                }
            )

        if index.ntotal == 0:
            raise ValueError(f"No valid frames indexed for: {video_path}")

        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(index_path))
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "pipeline": "v2",
                    "sampling_fps": sampling_fps,
                    "frames": metadata,
                },
                f,
                indent=2,
            )


# ============================================================
# V2 Search & Temporal Grounding
# ============================================================

class VideoSearcherV2:
    def __init__(self, index_path: Path, metadata_path: Path, encoder: CLIPEncoder):
        self.index = faiss.read_index(str(index_path))
        with open(metadata_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        if isinstance(payload, list):
            self.records = payload
            self.sampling_fps = SAMPLING_FPS
        else:
            self.records = payload.get("frames", [])
            self.sampling_fps = float(payload.get("sampling_fps", SAMPLING_FPS))

        self.encoder = encoder

    def search(
        self,
        query: str,
        top_k: int = TOP_K,
        merge_gap_sec: float = MERGE_GAP_SEC,
    ) -> List[Dict[str, Any]]:
        if not self.records or self.index.ntotal == 0:
            return []

        query_vec = encode_query_text(self.encoder, query)
        scores, indices = self.index.search(query_vec, self.index.ntotal)
        clip_map = {int(i): float(s) for i, s in zip(indices[0], scores[0]) if i >= 0}

        scored_frames = []
        for idx, rec in enumerate(self.records):
            faiss_idx = int(rec.get("faiss_idx", idx))
            ts = float(rec.get("timestamp", rec.get("timestamp_sec", 0.0)))
            sim = max(0.0, clip_map.get(faiss_idx, 0.0))
            scored_frames.append({"timestamp": ts, "score": sim})

        # Temporal smoothing + adaptive threshold clustering
        raw = np.array([f["score"] for f in scored_frames], dtype=np.float32)
        kernel = np.ones(SMOOTHING_WINDOW, dtype=np.float32) / float(SMOOTHING_WINDOW)
        smooth = np.convolve(raw, kernel, mode="same")
        thresh = max(float(np.mean(smooth) + 0.5 * np.std(smooth)), MIN_SCORE_THRESHOLD)

        clusters: List[List[Dict[str, Any]]] = []
        current: List[Dict[str, Any]] = []
        for i, frame in enumerate(scored_frames):
            if smooth[i] >= thresh:
                if current and (frame["timestamp"] - current[-1]["timestamp"] > merge_gap_sec):
                    clusters.append(current)
                    current = [frame]
                else:
                    current.append(frame)
        if current:
            clusters.append(current)

        frame_step = 1.0 / max(self.sampling_fps, 0.1)
        proposals: List[Dict[str, Any]] = []

        for cluster in clusters:
            start_t = max(0.0, cluster[0]["timestamp"] - frame_step * 0.5)
            end_t = cluster[-1]["timestamp"] + frame_step * 0.5
            peak = max(cluster, key=lambda x: x["score"])
            confidence = round(min(1.0, peak["score"] * 2.5), 3)

            proposals.append(
                {
                    "start": round(start_t, 2),
                    "end": round(end_t, 2),
                    "peak_timestamp": round(peak["timestamp"], 2),
                    "similarity_score": round(peak["score"], 4),
                    "confidence": confidence,
                }
            )

        # 1D Temporal NMS (IoU > 0.5)
        proposals.sort(key=lambda x: x["confidence"], reverse=True)
        kept: List[Dict[str, Any]] = []
        for prop in proposals:
            if all(
                calculate_iou([prop["start"], prop["end"]], [k["start"], k["end"]]) <= NMS_IOU_THRESHOLD
                for k in kept
            ):
                kept.append(prop)
            if len(kept) >= top_k:
                break

        return kept


# ============================================================
# Evaluate Single Query (R@1 uses top-1 proposal)
# ============================================================

def evaluate_query_v2(
    searcher: VideoSearcherV2,
    query: str,
    ground_truth: List[float],
) -> Dict[str, Any]:
    proposals = searcher.search(query, top_k=TOP_K)
    if not proposals:
        return {
            "prediction": None,
            "iou": 0.0,
            "score": None,
            "confidence": None,
        }

    top1 = proposals[0]
    predicted_segment = [top1["start"], top1["end"]]
    iou = calculate_iou(predicted_segment, ground_truth)

    return {
        "prediction": predicted_segment,
        "iou": iou,
        "score": top1["similarity_score"],
        "confidence": top1["confidence"],
    }


# ============================================================
# Save JSON & CSV
# ============================================================

def save_json(data: Any, filename: str) -> None:
    path = EVALUATION_DIR / filename
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)
    print(f"Saved: {path}")


def save_csv(results: List[Dict[str, Any]], filename: str = "v2_results.csv") -> None:
    path = EVALUATION_DIR / filename
    fields = [
        "video",
        "query",
        "ground_truth_start",
        "ground_truth_end",
        "predicted_start",
        "predicted_end",
        "iou",
        "similarity_score",
        "confidence",
    ]
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved: {path}")


# ============================================================
# Main
# ============================================================

def main() -> None:
    print("=" * 70)
    print("ChronoVision V2 Temporal Grounding Evaluation")
    print("Pipeline: FFmpeg 2x -> Sharpest Window -> Denoise/CLAHE/Sharpen -> 1:1 Pad -> Smooth + NMS")
    print("=" * 70)

    encoder = CLIPEncoder()

    with open(ANNOTATION_FILE, "r", encoding="utf-8") as file:
        annotations = list(csv.DictReader(file))

    print(f"Queries found: {len(annotations)}")

    videos: Dict[str, List[Dict[str, str]]] = {}
    for row in annotations:
        video_id = row["video"].strip()
        videos.setdefault(video_id, []).append(row)

    print(f"Videos found: {len(videos)}")

    total_queries = 0
    videos_evaluated = 0
    correct_03 = 0
    correct_05 = 0
    correct_07 = 0
    all_ious: List[float] = []
    detailed_results: List[Dict[str, Any]] = []

    for video_number, (video_id, queries) in enumerate(videos.items(), start=1):
        print("\n" + "=" * 70)
        print(f"[{video_number}/{len(videos)}] Video: {video_id}")

        try:
            video_path = find_video(video_id)
        except FileNotFoundError as error:
            print(f"WARNING: {error}")
            continue

        index_path, metadata_path = get_v2_index_paths(video_id)

        if not index_path.exists() or not metadata_path.exists():
            print("Building V2 FAISS index (FFmpeg + enhancement + 1:1 pad)...")
            build_index_v2(video_path, index_path, metadata_path, encoder, SAMPLING_FPS)
        else:
            print("Using existing V2 FAISS index.")

        searcher = VideoSearcherV2(index_path, metadata_path, encoder)
        videos_evaluated += 1

        for row in queries:
            query = row["query"].strip()
            ground_truth = [float(row["start"]), float(row["end"])]

            result = evaluate_query_v2(searcher, query, ground_truth)
            iou = result["iou"]
            prediction: Optional[List[float]] = result["prediction"]
            similarity_score = result.get("score")
            confidence = result.get("confidence")

            total_queries += 1
            all_ious.append(iou)

            if iou >= 0.3:
                correct_03 += 1
            if iou >= 0.5:
                correct_05 += 1
            if iou >= 0.7:
                correct_07 += 1

            predicted_start = prediction[0] if prediction else None
            predicted_end = prediction[1] if prediction else None

            detailed_results.append(
                {
                    "video": video_id,
                    "query": query,
                    "ground_truth_start": ground_truth[0],
                    "ground_truth_end": ground_truth[1],
                    "predicted_start": predicted_start,
                    "predicted_end": predicted_end,
                    "iou": iou,
                    "similarity_score": similarity_score,
                    "confidence": confidence,
                }
            )

            print(f"\nQuery: {query}")
            print(f"GT: [{ground_truth[0]:.2f}, {ground_truth[1]:.2f}]")
            print(f"Prediction: {prediction} (conf={confidence})")
            print(f"IoU: {iou:.4f}")

    if total_queries == 0:
        print("\nNo queries evaluated.")
        return

    recall_03 = (correct_03 / total_queries) * 100
    recall_05 = (correct_05 / total_queries) * 100
    recall_07 = (correct_07 / total_queries) * 100
    mean_iou = sum(all_ious) / len(all_ious)

    summary = {
        "pipeline": "v2",
        "model": "CLIP ViT-B/32",
        "preprocessing": {
            "coarse_extraction": "ffmpeg 2x fps",
            "frame_selection": "laplacian sharpest-frame window",
            "image_enhancement": "fastNlMeansDenoisingColored + LAB CLAHE + unsharp mask",
            "aspect_handling": "224x224 1:1 neutral-gray letterbox pad (114)",
        },
        "grounding": {
            "smoothing_window": SMOOTHING_WINDOW,
            "threshold": "max(mean + 0.5*std, 0.20)",
            "merge_gap_sec": MERGE_GAP_SEC,
            "nms_iou": NMS_IOU_THRESHOLD,
        },
        "retrieval": "FAISS IndexFlatIP",
        "embedding_dimension": EMBEDDING_DIM,
        "sampling_fps": SAMPLING_FPS,
        "top_k": TOP_K,
        "queries_evaluated": total_queries,
        "videos_evaluated": videos_evaluated,
        "recall_at_1_iou_0.3": recall_03,
        "recall_at_1_iou_0.5": recall_05,
        "recall_at_1_iou_0.7": recall_07,
        "mean_iou": mean_iou,
    }

    save_json(summary, "v2_summary.json")
    save_json(detailed_results, "v2_results.json")
    save_csv(detailed_results, "v2_results.csv")

    print("\n" + "=" * 70)
    print("FINAL V2 RESULTS")
    print("=" * 70)
    print(f"Queries evaluated : {total_queries}")
    print(f"R@1 IoU@0.3       : {recall_03:.2f}%")
    print(f"R@1 IoU@0.5       : {recall_05:.2f}%")
    print(f"R@1 IoU@0.7       : {recall_07:.2f}%")
    print(f"Mean IoU          : {mean_iou:.4f}")

    # Side-by-side comparison if V1 baseline_summary.json is present
    baseline_path = EVALUATION_DIR / "baseline_summary.json"
    if baseline_path.exists():
        with open(baseline_path, "r", encoding="utf-8") as f:
            v1 = json.load(f)
        print("-" * 70)
        print("DELTA VS V1 BASELINE")
        print("-" * 70)
        print(f"R@1 IoU@0.3       : {recall_03 - v1['recall_at_1_iou_0.3']:+.2f}%")
        print(f"R@1 IoU@0.5       : {recall_05 - v1['recall_at_1_iou_0.5']:+.2f}%")
        print(f"R@1 IoU@0.7       : {recall_07 - v1['recall_at_1_iou_0.7']:+.2f}%")
        print(f"Mean IoU          : {mean_iou - v1['mean_iou']:+.4f}")

    print("=" * 70)


if __name__ == "__main__":
    main()