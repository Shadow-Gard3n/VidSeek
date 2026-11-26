import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import asyncio
import base64
import csv
import hashlib
from io import BytesIO
import json
from pathlib import Path
import random
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import faiss
import numpy as np
from ollama import AsyncClient
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
# V6 Validation Defaults
# ============================================================

DEFAULT_SAMPLING_FPS = 1.0
DEFAULT_MERGE_GAP_SEC = 2.0
DEFAULT_SAMPLE_FRACTION = 0.20  
DEFAULT_SEED = 42
DEFAULT_OLLAMA_MODEL = os.getenv(
    "OLLAMA_VISION_MODEL", os.getenv("OLLAMA_MODEL", "gemma3:4b")
)
DEFAULT_OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_TIMEOUT_SEC = 35.0


# ============================================================
# Encoder Adapter
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
# Video & Index Paths
# ============================================================

def find_video(video_id: str) -> Path:
    for ext in (".mp4", ".avi", ".mov", ".mkv"):
        path = VIDEO_DIR / f"{video_id}{ext}"
        if path.exists():
            return path

    for path in VIDEO_DIR.rglob("*"):
        if path.is_file() and path.stem == video_id:
            return path

    raise FileNotFoundError(f"Video not found: {video_id}")


def get_v6_paths(video_id: str) -> Tuple[Path, Path, Path]:
    safe_id = hashlib.md5(video_id.encode()).hexdigest()[:12]
    index_path = INDEX_DIR / f"{safe_id}_v6.index"
    metadata_path = INDEX_DIR / f"{safe_id}_v6.json"
    frames_dir = INDEX_DIR / "v6_frames" / safe_id
    return index_path, metadata_path, frames_dir


# ============================================================
# V6 Ingestion: FFmpeg + Pure CLIP Index + Cached Keyframes
# ============================================================

def build_index_v6(
    video_path: Path,
    index_path: Path,
    metadata_path: Path,
    frames_dir: Path,
    encoder: CLIPEncoder,
    sampling_fps: float = DEFAULT_SAMPLING_FPS,
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Unable to open video file: {video_path}")
    native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_sec = round(total_frames / native_fps, 2) if total_frames > 0 else 0.0
    cap.release()

    frames_dir.mkdir(parents=True, exist_ok=True)
    for old_fp in frames_dir.glob("frame_v6_*.jpg"):
        old_fp.unlink(missing_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"fps={sampling_fps}",
        "-q:v",
        "2",
        str(frames_dir / "frame_v6_%06d.jpg"),
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    raw_files = sorted(frames_dir.glob("frame_v6_*.jpg"))
    if not raw_files:
        raise ValueError(f"FFmpeg extracted 0 frames from: {video_path}")

    vecs: List[np.ndarray] = []
    frame_records: List[Dict[str, Any]] = []

    for idx, raw_fp in enumerate(raw_files):
        with Image.open(raw_fp) as img:
            rgb = img.convert("RGB")
            vec = encode_pil_image(encoder, rgb)
        vecs.append(vec)

        ts = round(idx / sampling_fps, 2)
        frame_records.append(
            {
                "faiss_idx": idx,
                "timestamp_sec": ts,
                "frame_name": raw_fp.name,
            }
        )

    mat = np.vstack(vecs).astype(np.float32)
    faiss.normalize_L2(mat)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    index = faiss.IndexFlatIP(mat.shape[1])
    index.add(mat)
    faiss.write_index(index, str(index_path))

    payload = {
        "pipeline": "v6",
        "duration_sec": duration_sec,
        "native_fps": round(native_fps, 2),
        "sampling_fps": sampling_fps,
        "frames": frame_records,
    }
    metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ============================================================
# V6 Boundary Proposals & Ollama VLM Verification
# ============================================================

def _encode_image_b64(path: Path, max_dim: int = 512) -> Optional[str]:
    if not path.exists():
        return None
    with Image.open(path) as img:
        rgb = img.convert("RGB")
        rgb.thumbnail((max_dim, max_dim))
        buf = BytesIO()
        rgb.save(buf, format="JPEG", quality=82)
        return base64.b64encode(buf.getvalue()).decode("utf-8")


def _propose_clip_boundaries(
    frames_dir: Path,
    frames: List[Dict[str, Any]],
    sims: np.ndarray,
    duration_sec: float,
    sampling_fps: float,
    top_k: int,
    max_half_window_sec: float = 4.0,
    min_peak_sep_sec: float = 4.0,
) -> List[Dict[str, Any]]:
    n = len(frames)
    if n == 0:
        return []

    if n >= 3:
        smooth = np.convolve(
            sims, np.array([0.25, 0.50, 0.25], dtype=np.float32), mode="same"
        )
    else:
        smooth = sims.copy()

    ranked_indices = np.argsort(smooth)[::-1]
    min_sep_frames = max(1, int(round(min_peak_sep_sec * sampling_fps)))
    max_expand_frames = max(1, int(round(max_half_window_sec * sampling_fps)))

    chosen_peaks: List[int] = []
    for idx in ranked_indices:
        idx_int = int(idx)
        if all(abs(idx_int - p) >= min_sep_frames for p in chosen_peaks):
            chosen_peaks.append(idx_int)
        if len(chosen_peaks) >= max(top_k * 2, 6):
            break

    baseline = float(np.median(smooth))
    proposals: List[Dict[str, Any]] = []

    for peak_idx in chosen_peaks:
        peak_score = float(smooth[peak_idx])
        cutoff = baseline + 0.55 * max(0.0, peak_score - baseline)

        left = peak_idx
        while (
            left > 0
            and (peak_idx - left) < max_expand_frames
            and float(smooth[left - 1]) >= cutoff
        ):
            left -= 1

        right = peak_idx
        while (
            right + 1 < n
            and (right - peak_idx) < max_expand_frames
            and float(smooth[right + 1]) >= cutoff
        ):
            right += 1

        if left == right:
            left = max(0, peak_idx - 1)
            right = min(n - 1, peak_idx + 1)

        step = 1.0 / max(sampling_fps, 0.25)
        start_time = round(float(frames[left]["timestamp_sec"]), 2)
        end_raw = float(frames[right]["timestamp_sec"]) + step
        end_time = round(min(duration_sec, end_raw) if duration_sec > 0 else end_raw, 2)
        peak_ts = round(float(frames[peak_idx]["timestamp_sec"]), 2)

        sample_indices = sorted({left, peak_idx, right})
        kf_paths = [frames_dir / frames[i]["frame_name"] for i in sample_indices]
        kf_timestamps = [round(float(frames[i]["timestamp_sec"]), 2) for i in sample_indices]

        proposals.append(
            {
                "start_time": start_time,
                "end_time": end_time,
                "peak_timestamp": peak_ts,
                "similarity_score": round(float(sims[peak_idx]), 4),
                "_kf_paths": kf_paths,
                "_kf_timestamps": kf_timestamps,
            }
        )

    return proposals


async def _call_ollama_v6(
    client: AsyncClient,
    model_name: str,
    prompt: str,
    image_paths: List[Path],
    timeout_sec: float,
) -> Dict[str, Any]:
    b64_imgs = [
        b for p in image_paths if (b := _encode_image_b64(p, max_dim=512)) is not None
    ]
    msg: Dict[str, Any] = {"role": "user", "content": prompt}
    if b64_imgs:
        msg["images"] = b64_imgs

    resp = await asyncio.wait_for(
        client.chat(
            model=model_name,
            messages=[msg],
            format="json",
            keep_alive="30m",
            options={"num_ctx": 4096, "temperature": 0.1},
        ),
        timeout=timeout_sec,
    )
    content = resp["message"]["content"] if isinstance(resp, dict) else resp.message.content
    return json.loads(content or "{}")


async def _llm_verify_and_refine_v6(
    client: AsyncClient,
    model_name: str,
    query: str,
    candidates: List[Dict[str, Any]],
    timeout_sec: float,
) -> List[Dict[str, Any]]:
    sem = asyncio.Semaphore(1)  # Sequential for local Ollama stability
    total = len(candidates)

    async def verify_one(idx: int, cand: Dict[str, Any]) -> Dict[str, Any]:
        async with sem:
            t0 = time.perf_counter()
            frame_times = cand.get("_kf_timestamps", [])
            prompt = (
                "You are a strict video moment verifier.\n"
                f'User Query: "{query}"\n'
                f"Candidate Window: {cand['start_time']}s to {cand['end_time']}s (peak at {cand['peak_timestamp']}s).\n"
                f"Attached images correspond in order to timestamps (seconds): {frame_times}.\n"
                "Rules:\n"
                "1. Only set matched=true and relevance >= 0.65 if the visual evidence clearly shows what the user asked for.\n"
                "2. If the frames do NOT show the queried action/object/event, set matched=false and relevance <= 0.30.\n"
                "3. Provide tight refined_start_time and refined_end_time within the candidate window.\n"
                'Return ONLY JSON: {"matched": false, "relevance": 0.0, "refined_start_time": 0.0, "refined_end_time": 0.0, "reason": "brief visual explanation"}'
            )
            try:
                res = await _call_ollama_v6(
                    client=client,
                    model_name=model_name,
                    prompt=prompt,
                    image_paths=cand.get("_kf_paths", []),
                    timeout_sec=timeout_sec,
                )
                rel = max(0.0, min(1.0, float(res.get("relevance", 0.0))))
                matched = bool(res.get("matched", rel >= 0.65))
                reason = str(res.get("reason") or "").strip()

                r_start = res.get("refined_start_time")
                r_end = res.get("refined_end_time")
                if r_start is not None and r_end is not None:
                    rs, re_ = float(r_start), float(r_end)
                    if cand["start_time"] <= rs <= re_ <= cand["end_time"] and (re_ - rs) >= 0.5:
                        cand["start_time"] = round(rs, 2)
                        cand["end_time"] = round(re_, 2)

                cand["llm_relevance"] = rel
                cand["llm_selected"] = matched
                cand["llm_reason"] = reason
                cand["llm_latency_sec"] = round(time.perf_counter() - t0, 2)
            except Exception as exc:
                print(f"    [Ollama Error] cand {idx + 1}/{total}: {exc}")
                cand["llm_relevance"] = None
                cand["llm_selected"] = False
                cand["llm_reason"] = f"error: {exc}"
                cand["llm_latency_sec"] = round(time.perf_counter() - t0, 2)
            return cand

    verified = await asyncio.gather(*(verify_one(i, c) for i, c in enumerate(candidates)))
    verified.sort(
        key=lambda x: (
            1 if x.get("llm_selected") else 0,
            x["llm_relevance"] if x.get("llm_relevance") is not None else -1.0,
            x["similarity_score"],
        ),
        reverse=True,
    )
    return list(verified)


class VideoSearcherV6:
    def __init__(
        self,
        index_path: Path,
        metadata_path: Path,
        frames_dir: Path,
        encoder: CLIPEncoder,
        ollama_client: AsyncClient,
        ollama_model: str,
        timeout_sec: float = DEFAULT_TIMEOUT_SEC,
        rerank: bool = True,
    ):
        self.index = faiss.read_index(str(index_path))
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.frames = payload.get("frames", [])
        self.duration_sec = float(payload.get("duration_sec") or 0.0)
        self.sampling_fps = float(payload.get("sampling_fps") or DEFAULT_SAMPLING_FPS)
        self.frames_dir = frames_dir
        self.encoder = encoder
        self.ollama_client = ollama_client
        self.ollama_model = ollama_model
        self.timeout_sec = timeout_sec
        self.rerank = rerank

    async def search(
        self,
        query: str,
        top_k: int = TOP_K,
        merge_gap_sec: float = DEFAULT_MERGE_GAP_SEC,
    ) -> List[Dict[str, Any]]:
        if not self.frames or self.index.ntotal == 0:
            return []

        q_vec = encode_query_text(self.encoder, query)
        scores, indices = self.index.search(q_vec, self.index.ntotal)

        sims = np.zeros(len(self.frames), dtype=np.float32)
        for sc, idx in zip(scores[0], indices[0]):
            if 0 <= int(idx) < len(self.frames):
                sims[int(idx)] = float(sc)

        candidates = _propose_clip_boundaries(
            frames_dir=self.frames_dir,
            frames=self.frames,
            sims=sims,
            duration_sec=self.duration_sec,
            sampling_fps=self.sampling_fps,
            top_k=top_k,
            max_half_window_sec=4.0,
            min_peak_sep_sec=max(4.0, merge_gap_sec * 2.0),
        )

        # Matches V6's Ollama pool_limit = min(len(candidates), min(top_k, 3))
        pool_limit = min(len(candidates), min(top_k, 3))
        if self.rerank and candidates:
            verified = await _llm_verify_and_refine_v6(
                client=self.ollama_client,
                model_name=self.ollama_model,
                query=query,
                candidates=candidates[:pool_limit],
                timeout_sec=self.timeout_sec,
            )
            final_list = verified + candidates[pool_limit:]
        else:
            final_list = candidates

        results: List[Dict[str, Any]] = []
        for rank, item in enumerate(final_list[:top_k], start=1):
            llm_rel = item.get("llm_relevance")
            if llm_rel is not None:
                conf = round(float(llm_rel), 3)
            else:
                conf = round(
                    min(1.0, max(0.05, (float(item["similarity_score"]) - 0.15) / 0.20)),
                    3,
                )
            results.append(
                {
                    "rank": rank,
                    "start": item["start_time"],
                    "end": item["end_time"],
                    "peak_timestamp": item["peak_timestamp"],
                    "similarity_score": item["similarity_score"],
                    "confidence": conf,
                    "llm_selected": bool(item.get("llm_selected", False)),
                    "llm_relevance": llm_rel,
                    "llm_reason": item.get("llm_reason"),
                }
            )
        return results


# ============================================================
# Ollama Pre-Flight Verification
# ============================================================

async def verify_ollama_ready(host: str, model_name: str) -> AsyncClient:
    client = AsyncClient(host=host)
    try:
        listing = await client.list()
    except Exception as exc:
        raise RuntimeError(
            f"Cannot connect to Ollama at {host}. Start Ollama (`ollama serve` or open the Ollama app) first.\nDetail: {exc}"
        )

    models_raw = listing.get("models", []) if isinstance(listing, dict) else getattr(listing, "models", [])
    available = []
    for m in models_raw:
        name = m.get("model") or m.get("name") if isinstance(m, dict) else getattr(m, "model", None)
        if name:
            available.append(name)

    if available and not any(
        m == model_name or m.startswith(f"{model_name}:") for m in available
    ):
        print(
            f"WARNING: Model '{model_name}' not listed in `ollama list` ({', '.join(available)}). "
            f"Run `ollama pull {model_name}` if chat fails."
        )
    return client


# ============================================================
# Subset Comparison Helper (Apples-to-Apples vs V1 / V2)
# ============================================================

def compute_subset_baseline(
    filename: str,
    evaluated_pairs: set,
) -> Optional[Dict[str, float]]:
    path = EVALUATION_DIR / filename
    if not path.exists():
        return None

    rows = json.loads(path.read_text(encoding="utf-8"))
    matched = [
        r for r in rows if (r["video"].strip(), r["query"].strip()) in evaluated_pairs
    ]
    if not matched:
        return None

    n = len(matched)
    r03 = sum(1 for r in matched if float(r["iou"]) >= 0.3) / n * 100
    r05 = sum(1 for r in matched if float(r["iou"]) >= 0.5) / n * 100
    r07 = sum(1 for r in matched if float(r["iou"]) >= 0.7) / n * 100
    miou = sum(float(r["iou"]) for r in matched) / n
    return {
        "n": n,
        "r03": r03,
        "r05": r05,
        "r07": r07,
        "mean_iou": miou,
    }


# ============================================================
# Save Outputs
# ============================================================

def save_json(data: Any, filename: str) -> None:
    path = EVALUATION_DIR / filename
    path.write_text(json.dumps(data, indent=4), encoding="utf-8")
    print(f"Saved: {path}")


def save_csv(results: List[Dict[str, Any]], filename: str) -> None:
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
        "llm_selected",
        "llm_relevance",
        "llm_reason",
        "query_latency_sec",
    ]
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved: {path}")


# ============================================================
# Main Async Runner
# ============================================================

async def async_main(args: argparse.Namespace) -> None:
    print("=" * 72)
    print("ChronoVision V6 Validation (CLIP Peak Boundaries + Ollama VLM Verifier)")
    print(
        f"Dataset Fraction: {args.fraction:.0%} (seed={args.seed}) | "
        f"Ollama Model: {args.model} | Rerank: {not args.no_rerank}"
    )
    print("=" * 72)

    ollama_client = await verify_ollama_ready(args.host, args.model)
    encoder = CLIPEncoder()

    with open(ANNOTATION_FILE, "r", encoding="utf-8") as file:
        annotations = list(csv.DictReader(file))

    all_videos: Dict[str, List[Dict[str, str]]] = {}
    for row in annotations:
        vid = row["video"].strip()
        all_videos.setdefault(vid, []).append(row)

    video_keys = sorted(all_videos.keys())
    if args.fraction < 1.0:
        rng = random.Random(args.seed)
        sample_count = max(1, int(round(len(video_keys) * args.fraction)))
        selected_keys = sorted(rng.sample(video_keys, sample_count))
    else:
        selected_keys = video_keys

    videos = {k: all_videos[k] for k in selected_keys}
    planned_queries = sum(len(q) for q in videos.values())

    print(
        f"Selected {len(videos)}/{len(all_videos)} videos "
        f"({planned_queries}/{len(annotations)} queries)"
    )

    total_queries = 0
    videos_evaluated = 0
    correct_03 = 0
    correct_05 = 0
    correct_07 = 0
    all_ious: List[float] = []
    latencies: List[float] = []
    detailed_results: List[Dict[str, Any]] = []
    evaluated_pairs = set()

    t_eval_start = time.perf_counter()

    for video_number, (video_id, queries) in enumerate(videos.items(), start=1):
        print("\n" + "=" * 72)
        print(f"[{video_number}/{len(videos)}] Video: {video_id}")

        try:
            video_path = find_video(video_id)
        except FileNotFoundError as error:
            print(f"WARNING: {error}")
            continue

        index_path, metadata_path, frames_dir = get_v6_paths(video_id)
        cached_frames_ok = frames_dir.exists() and any(frames_dir.glob("frame_v6_*.jpg"))

        if not index_path.exists() or not metadata_path.exists() or not cached_frames_ok:
            print("Building V6 CLIP index & extracting frames via FFmpeg...")
            build_index_v6(
                video_path=video_path,
                index_path=index_path,
                metadata_path=metadata_path,
                frames_dir=frames_dir,
                encoder=encoder,
                sampling_fps=DEFAULT_SAMPLING_FPS,
            )
        else:
            print("Using cached V6 FAISS index & keyframes.")

        searcher = VideoSearcherV6(
            index_path=index_path,
            metadata_path=metadata_path,
            frames_dir=frames_dir,
            encoder=encoder,
            ollama_client=ollama_client,
            ollama_model=args.model,
            timeout_sec=args.timeout,
            rerank=not args.no_rerank,
        )
        videos_evaluated += 1

        for row in queries:
            query = row["query"].strip()
            ground_truth = [float(row["start"]), float(row["end"])]

            t_q = time.perf_counter()
            proposals = await searcher.search(query, top_k=TOP_K)
            q_latency = round(time.perf_counter() - t_q, 2)
            latencies.append(q_latency)

            if proposals:
                top1 = proposals[0]
                prediction = [top1["start"], top1["end"]]
                iou = calculate_iou(prediction, ground_truth)
                sim_score = top1["similarity_score"]
                conf = top1["confidence"]
                llm_selected = top1["llm_selected"]
                llm_rel = top1["llm_relevance"]
                llm_reason = top1["llm_reason"]
            else:
                prediction = None
                iou = 0.0
                sim_score = None
                conf = None
                llm_selected = False
                llm_rel = None
                llm_reason = None

            total_queries += 1
            all_ious.append(iou)
            evaluated_pairs.add((video_id, query))

            if iou >= 0.3:
                correct_03 += 1
            if iou >= 0.5:
                correct_05 += 1
            if iou >= 0.7:
                correct_07 += 1

            detailed_results.append(
                {
                    "video": video_id,
                    "query": query,
                    "ground_truth_start": ground_truth[0],
                    "ground_truth_end": ground_truth[1],
                    "predicted_start": prediction[0] if prediction else None,
                    "predicted_end": prediction[1] if prediction else None,
                    "iou": iou,
                    "similarity_score": sim_score,
                    "confidence": conf,
                    "llm_selected": llm_selected,
                    "llm_relevance": llm_rel,
                    "llm_reason": llm_reason,
                    "query_latency_sec": q_latency,
                }
            )

            print(f"\nQuery      : {query}")
            print(f"GT         : [{ground_truth[0]:.2f}, {ground_truth[1]:.2f}]")
            print(
                f"Prediction : {prediction} "
                f"(CLIP={sim_score}, LLM_sel={llm_selected}, LLM_rel={llm_rel}, {q_latency}s)"
            )
            if llm_reason:
                print(f"LLM Reason : {llm_reason}")
            print(f"IoU        : {iou:.4f}")

    if total_queries == 0:
        print("\nNo queries evaluated.")
        return

    recall_03 = (correct_03 / total_queries) * 100
    recall_05 = (correct_05 / total_queries) * 100
    recall_07 = (correct_07 / total_queries) * 100
    mean_iou = sum(all_ious) / len(all_ious)
    avg_latency = sum(latencies) / len(latencies)
    total_elapsed = round(time.perf_counter() - t_eval_start, 2)

    summary = {
        "pipeline": "v6",
        "clip_model": "CLIP ViT-B/32",
        "llm_provider": "ollama",
        "llm_model": args.model,
        "rerank_enabled": not args.no_rerank,
        "dataset_fraction": args.fraction,
        "seed": args.seed,
        "sampling_fps": DEFAULT_SAMPLING_FPS,
        "top_k": TOP_K,
        "queries_evaluated": total_queries,
        "videos_evaluated": videos_evaluated,
        "recall_at_1_iou_0.3": recall_03,
        "recall_at_1_iou_0.5": recall_05,
        "recall_at_1_iou_0.7": recall_07,
        "mean_iou": mean_iou,
        "avg_query_latency_sec": round(avg_latency, 2),
        "total_runtime_sec": total_elapsed,
    }

    save_json(summary, "v6_val_summary.json")
    save_json(detailed_results, "v6_val_results.json")
    save_csv(detailed_results, "v6_val_results.csv")

    print("\n" + "=" * 72)
    print(f"FINAL V6 VALIDATION RESULTS ({total_queries} queries / {videos_evaluated} videos)")
    print("=" * 72)
    print(f"R@1 IoU@0.3       : {recall_03:.2f}%")
    print(f"R@1 IoU@0.5       : {recall_05:.2f}%")
    print(f"R@1 IoU@0.7       : {recall_07:.2f}%")
    print(f"Mean IoU          : {mean_iou:.4f}")
    print(f"Avg Query Latency : {avg_latency:.2f}s (Total: {total_elapsed:.1f}s)")

    # Apples-to-apples comparison on the exact same 1/5 subset
    v1_sub = compute_subset_baseline("baseline_results.json", evaluated_pairs)
    v2_sub = compute_subset_baseline("v2_results.json", evaluated_pairs)

    if v1_sub or v2_sub:
        print("-" * 72)
        print("APPLES-TO-APPLES COMPARISON ON THIS EXACT 1/5 SUBSET")
        print("-" * 72)
        if v1_sub:
            print(
                f"V1 (Baseline)     : R@0.3={v1_sub['r03']:.2f}% | "
                f"R@0.5={v1_sub['r05']:.2f}% | R@0.7={v1_sub['r07']:.2f}% | "
                f"mIoU={v1_sub['mean_iou']:.4f}"
            )
        if v2_sub:
            print(
                f"V2 (Enhance+NMS)  : R@0.3={v2_sub['r03']:.2f}% | "
                f"R@0.5={v2_sub['r05']:.2f}% | R@0.7={v2_sub['r07']:.2f}% | "
                f"mIoU={v2_sub['mean_iou']:.4f}"
            )
        print(
            f"V6 (CLIP+Ollama)  : R@0.3={recall_03:.2f}% | "
            f"R@0.5={recall_05:.2f}% | R@0.7={recall_07:.2f}% | "
            f"mIoU={mean_iou:.4f}"
        )
        if v1_sub:
            print(
                f"Delta (V6 - V1)   : R@0.3={recall_03 - v1_sub['r03']:+.2f}% | "
                f"R@0.5={recall_05 - v1_sub['r05']:+.2f}% | "
                f"R@0.7={recall_07 - v1_sub['r07']:+.2f}% | "
                f"mIoU={mean_iou - v1_sub['mean_iou']:+.4f}"
            )

    print("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate ChronoVision V6 on Ollama")
    parser.add_argument(
        "--fraction",
        type=float,
        default=DEFAULT_SAMPLE_FRACTION,
        help="Fraction of videos to evaluate (default: 0.20 = 1/5 dataset)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for deterministic dataset sampling (default: 42)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_OLLAMA_MODEL,
        help="Ollama vision model name (default: gemma3:4b)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=DEFAULT_OLLAMA_HOST,
        help="Ollama server host URL (default: http://localhost:11434)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SEC,
        help="Per-candidate Ollama timeout in seconds (default: 35.0)",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Skip Ollama VLM verification to measure pure V6 CLIP peak proposals",
    )
    args = parser.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()