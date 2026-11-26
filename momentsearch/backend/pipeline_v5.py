import asyncio
import base64
from contextlib import contextmanager
from functools import lru_cache
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
from dotenv import load_dotenv
import faiss
import numpy as np
from PIL import Image
import torch

from config import DEFAULT_SAMPLING_FPS, FRAMES_DIR, INDEXES_DIR
import database as db
from embedding import embedding_service

load_dotenv(Path(__file__).resolve().parent / ".env")

_siglip_model = None
_siglip_processor = None
_text_embedder = None
_ocr_reader = None
_det_model = None

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when",
    "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "to", "from",
    "up", "down", "in", "out", "on", "off", "over", "under", "again",
    "further", "once", "here", "there", "where", "why", "how", "all",
    "any", "both", "each", "few", "more", "most", "other", "some", "such",
    "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "can", "will", "just", "should", "now", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "having", "do", "does",
    "did", "doing", "of", "it", "its", "this", "that", "these", "those",
}


def log_v5(msg: str) -> None:
    print(msg, flush=True)


@contextmanager
def timed(label: str):
    t = time.perf_counter()
    yield
    log_v5(f"[TIME] {label}: {time.perf_counter() - t:.2f}s")


def _get_device() -> str:
    force_cpu = os.getenv("V5_FORCE_CPU", "0") == "1"
    if force_cpu:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ============================================================
# CACHED INDEXES & STARTUP WARMUP
# ============================================================

@lru_cache(maxsize=8)
def _load_index_cached(path_str: str, mtime_ns: int) -> faiss.Index:
    return faiss.read_index(path_str)


def get_faiss_index(path: Path) -> faiss.Index:
    stat = path.stat()
    return _load_index_cached(str(path), stat.st_mtime_ns)


def warmup_v5_models() -> None:
    """Warm up visual and speech text encoders at server startup."""
    try:
        log_v5("[V5 STARTUP] Warming up visual and speech text encoders...")
        with timed("V5 startup warmup"):
            encode_text_visual_v5(["warmup"])
            _encode_speech_texts(["warmup"])
        log_v5("[V5 STARTUP] Warmup complete.")
    except Exception as exc:
        log_v5(f"[V5 Warmup Warning] {exc}")


# ============================================================
# VISUAL & TEXT ENCODERS (NO CLAHE)
# ============================================================

def pad_to_square_pil(bgr_or_rgb: np.ndarray, size: int = 384, is_bgr: bool = True) -> Image.Image:
    rgb = cv2.cvtColor(bgr_or_rgb, cv2.COLOR_BGR2RGB) if is_bgr else bgr_or_rgb
    h, w, _ = rgb.shape
    scale = size / max(h, w, 1)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), 128, dtype=np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top : top + nh, left : left + nw] = resized
    return Image.fromarray(canvas)


def _get_visual_encoder():
    global _siglip_model, _siglip_processor
    model_name = os.getenv("V5_VISUAL_MODEL", "").strip()
    if not model_name:
        embedding_service.load()
        return None, None

    if _siglip_model is None or _siglip_processor is None:
        from transformers import AutoModel, AutoProcessor

        device = _get_device()
        dtype = torch.float16 if device in {"mps", "cuda"} else torch.float32
        log_v5(f"[V5] Loading visual model '{model_name}' on {device} ({dtype})...")
        _siglip_processor = AutoProcessor.from_pretrained(model_name)
        _siglip_model = AutoModel.from_pretrained(model_name, dtype=dtype).to(device)
        _siglip_model.eval()
    return _siglip_model, _siglip_processor


def encode_images_batch_v5(images: List[Image.Image], batch_size: int = 64) -> np.ndarray:
    model, processor = _get_visual_encoder()
    device = _get_device()
    all_vecs: List[np.ndarray] = []

    if model is None or processor is None:
        embedding_service.load()
        assert embedding_service.model is not None and embedding_service.processor is not None
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            inputs = embedding_service.processor(images=batch, return_tensors="pt").to(embedding_service.device)
            with torch.no_grad():
                feats = embedding_service.model.get_image_features(**inputs)
                if not isinstance(feats, torch.Tensor):
                    feats = getattr(feats, "image_embeds", None) or getattr(feats, "pooler_output", feats[0])
            arr = np.float32(feats.detach().cpu().numpy())
            faiss.normalize_L2(arr)
            all_vecs.append(arr)
        return np.vstack(all_vecs)

    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size]
        inputs = processor(images=batch, return_tensors="pt").to(device)
        with torch.no_grad():
            feats = model.get_image_features(**inputs)
            if not isinstance(feats, torch.Tensor):
                feats = getattr(feats, "image_embeds", None) or getattr(feats, "pooler_output", feats[0])
        arr = np.float32(feats.detach().cpu().numpy())
        faiss.normalize_L2(arr)
        all_vecs.append(arr)
    return np.vstack(all_vecs)


def encode_text_visual_v5(queries: List[str]) -> np.ndarray:
    model, processor = _get_visual_encoder()
    device = _get_device()
    if model is None or processor is None:
        vecs = [embedding_service.encode_text(q) for q in queries]
        return np.vstack(vecs)

    inputs = processor(text=queries, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        feats = model.get_text_features(**inputs)
        if not isinstance(feats, torch.Tensor):
            feats = getattr(feats, "text_embeds", None) or getattr(feats, "pooler_output", feats[0])
    arr = np.float32(feats.detach().cpu().numpy())
    faiss.normalize_L2(arr)
    return arr


def _encode_speech_texts(texts: List[str]) -> np.ndarray:
    global _text_embedder
    model_id = os.getenv("V5_TEXT_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    if _text_embedder is None:
        from transformers import AutoModel, AutoTokenizer

        device = _get_device()
        log_v5(f"[V5] Loading speech text embedder '{model_id}' on {device}...")
        tok = AutoTokenizer.from_pretrained(model_id)
        mod = AutoModel.from_pretrained(model_id).to(device)
        mod.eval()
        _text_embedder = (tok, mod)

    tok, mod = _text_embedder
    device = _get_device()
    out_vecs = []
    for i in range(0, len(texts), 32):
        batch = texts[i : i + 32]
        enc = tok(batch, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        with torch.no_grad():
            res = mod(**enc)
            mask = enc["attention_mask"].unsqueeze(-1).expand(res.last_hidden_state.size()).float()
            pooled = torch.sum(res.last_hidden_state * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
        arr = np.float32(pooled.detach().cpu().numpy())
        faiss.normalize_L2(arr)
        out_vecs.append(arr)
    return np.vstack(out_vecs)


# ============================================================
# BACKGROUND OCR & DETECTOR SINGLETONS (OUTSIDE SEARCH PATH)
# ============================================================

def _get_ocr():
    global _ocr_reader
    if _ocr_reader is None:
        try:
            import easyocr

            _ocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        except ImportError:
            _ocr_reader = False
    return _ocr_reader if _ocr_reader is not False else None


def _get_det():
    global _det_model
    if _det_model is None:
        try:
            from ultralytics import YOLO

            _det_model = YOLO(os.getenv("V5_DET_MODEL", "yolov8n.pt"))
        except ImportError:
            _det_model = False
    return _det_model if _det_model is not False else None


def _is_text_likely(bgr: np.ndarray) -> bool:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 200)
    return float(np.mean(edges > 0)) >= 0.03 and float(np.std(gray)) >= 22.0


def build_background_ocr_and_det_v5(video_id: str) -> None:
    """Runs in background after speech index so search never blocks on CPU OCR/YOLO."""
    frames = db.get_frame_meta_v5(video_id)
    if not frames:
        return

    ocr_reader = _get_ocr()
    det_model = _get_det()
    if ocr_reader is None and det_model is None:
        return

    with timed(f"V5 background OCR+Det ({len(frames)} frames)"):
        db.upsert_index_status_v5(
            video_id,
            ocr_status="processing" if ocr_reader else "unavailable",
            det_status="processing" if det_model else "unavailable",
        )
        updates: List[Tuple[int, str, str, int, int]] = []
        for row in frames:
            f_idx = int(row["faiss_idx"])
            ocr_done = int(row.get("ocr_done") or 0)
            det_done = int(row.get("det_done") or 0)
            ocr_val = str(row.get("ocr_text") or "")
            det_val = str(row.get("det_labels_json") or "[]")

            if ocr_done == 1 and det_done == 1:
                updates.append((f_idx, ocr_val, det_val, 1, 1))
                continue

            fp = FRAMES_DIR / video_id / row["thumbnail_name"]
            if not fp.exists():
                updates.append((f_idx, ocr_val, det_val, 1, 1))
                continue

            if ocr_reader is not None and ocr_done == 0:
                bgr = cv2.imread(str(fp))
                if bgr is not None and _is_text_likely(bgr):
                    hits = [str(t).strip() for t in ocr_reader.readtext(str(fp), detail=0) if str(t).strip()]
                    ocr_val = " ".join(hits)
                ocr_done = 1

            if det_model is not None and det_done == 0:
                res = det_model(str(fp), verbose=False)[0]
                names = res.names or {}
                labels = sorted(
                    {str(names.get(int(b.cls[0]), "")).lower() for b in res.boxes if float(b.conf[0]) >= 0.3}
                )
                det_val = json.dumps(labels)
                det_done = 1

            updates.append((f_idx, ocr_val, det_val, ocr_done, det_done))

        db.update_frame_lazy_v5(video_id, updates)
        db.upsert_index_status_v5(
            video_id,
            ocr_status="ready" if ocr_reader else "unavailable",
            det_status="ready" if det_model else "unavailable",
        )


# ============================================================
# INGESTION [A] ONE FFMPEG PASS + [B] SHOT/DEDUP + [C] VISUAL + [D] SPEECH
# ============================================================

def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _detect_shots_and_dedup(
    raw_frames: List[Tuple[float, Path]],
    vecs: np.ndarray,
    shot_cos_thresh: float = 0.82,
    dup_cos_thresh: float = 0.98,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    n = len(raw_frames)
    if n == 0:
        return [], []

    shot_ids = [0] * n
    curr_shot = 0
    for i in range(1, n):
        cos_sim = float(np.dot(vecs[i - 1], vecs[i]))
        if cos_sim < shot_cos_thresh:
            curr_shot += 1
        shot_ids[i] = curr_shot

    kept_items: List[Dict[str, Any]] = []
    for i, (ts, fp) in enumerate(raw_frames):
        step = (raw_frames[i + 1][0] - ts) if i + 1 < n else 1.0
        if not kept_items:
            kept_items.append(
                {
                    "raw_idx": i,
                    "shot_idx": shot_ids[i],
                    "start_time": ts,
                    "end_time": round(ts + max(0.25, step), 2),
                    "peak_timestamp": ts,
                    "path": fp,
                    "vec": vecs[i],
                }
            )
            continue

        prev = kept_items[-1]
        cos_to_prev = float(np.dot(prev["vec"], vecs[i]))
        if prev["shot_idx"] == shot_ids[i] and cos_to_prev > dup_cos_thresh:
            prev["end_time"] = round(ts + max(0.25, step), 2)
        else:
            kept_items.append(
                {
                    "raw_idx": i,
                    "shot_idx": shot_ids[i],
                    "start_time": ts,
                    "end_time": round(ts + max(0.25, step), 2),
                    "peak_timestamp": ts,
                    "path": fp,
                    "vec": vecs[i],
                }
            )

    shots_map: Dict[int, Dict[str, Any]] = {}
    for item in kept_items:
        sid = item["shot_idx"]
        if sid not in shots_map:
            shots_map[sid] = {
                "shot_idx": sid,
                "start_time": item["start_time"],
                "end_time": item["end_time"],
                "keyframe_path": item["path"],
            }
        else:
            shots_map[sid]["end_time"] = max(shots_map[sid]["end_time"], item["end_time"])

    return list(shots_map.values()), kept_items


def _build_speech_index_v5(video_id: str, audio_wav_path: Path) -> None:
    if not audio_wav_path.exists() or audio_wav_path.stat().st_size < 4096:
        db.upsert_index_status_v5(video_id, speech_status="skipped_no_audio")
        return

    try:
        with timed(f"V5 speech index ({video_id[:8]})"):
            db.upsert_index_status_v5(video_id, speech_status="processing")
            segments_raw: List[Dict[str, Any]] = []

            try:
                import mlx_whisper

                mlx_model = os.getenv("MLX_WHISPER_MODEL", "mlx-community/whisper-small-mlx")
                res = mlx_whisper.transcribe(str(audio_wav_path), path_or_hf_repo=mlx_model)
                segments_raw = res.get("segments", [])
            except ImportError:
                try:
                    from faster_whisper import WhisperModel

                    wm = WhisperModel("small", device="cpu", compute_type="int8")
                    segs, _ = wm.transcribe(str(audio_wav_path), vad_filter=True)
                    segments_raw = [{"start": s.start, "end": s.end, "text": s.text} for s in segs]
                except ImportError:
                    db.upsert_index_status_v5(video_id, speech_status="unavailable")
                    return

            if not segments_raw:
                db.upsert_index_status_v5(video_id, speech_status="empty")
                return

            max_end = max(float(s["end"]) for s in segments_raw)
            win_len, stride = 25.0, 12.5
            windows_data: List[Tuple[float, float, str]] = []
            t = 0.0
            while t <= max_end:
                w_end = t + win_len
                texts = [
                    str(s["text"]).strip()
                    for s in segments_raw
                    if float(s["end"]) >= t and float(s["start"]) <= w_end and str(s["text"]).strip()
                ]
                if texts:
                    windows_data.append((round(t, 2), round(w_end, 2), " ".join(texts)))
                if w_end >= max_end:
                    break
                t += stride

            if not windows_data:
                db.upsert_index_status_v5(video_id, speech_status="empty")
                return

            dense_vecs = _encode_speech_texts([w[2] for w in windows_data])
            speech_index = faiss.IndexFlatIP(dense_vecs.shape[1])
            speech_index.add(dense_vecs)
            faiss.write_index(speech_index, str(INDEXES_DIR / f"{video_id}_v5_speech.faiss"))

            db_rows = [(video_id, idx, st, en, txt) for idx, (st, en, txt) in enumerate(windows_data)]
            db.save_speech_windows_v5(video_id, db_rows)
            db.upsert_index_status_v5(video_id, speech_status="ready")
    except Exception as exc:
        log_v5(f"[V5][D] Speech index failed: {exc}")
        db.upsert_index_status_v5(video_id, speech_status="failed")


def process_video_task_v5(
    video_id: str,
    video_path: Path,
    sampling_fps: float = DEFAULT_SAMPLING_FPS,
) -> None:
    """
    [A] ONE ffmpeg pass (scale ~384px short side, base fps + 16kHz mono wav)
    [B] Shot detection + near-dup collapse
    [C] Streamed-batch pad-to-square (NO CLAHE) -> FAISS + SQLite ready immediately
    [D] Background speech index
    [E] Background OCR & detection (outside search path)
    """
    work_dir = FRAMES_DIR / video_id / "_v5_tmp"
    try:
        log_v5(f"[V5 INGEST] Starting pipeline v5 for video={video_id[:8]}...")
        db.update_video_status(
            video_id,
            status="processing",
            progress=2.0,
            sampling_fps=sampling_fps,
            pipeline="v5",
            error_message=None,
        )
        file_hash = _file_sha256(video_path)
        db.upsert_index_status_v5(
            video_id,
            file_hash=file_hash,
            visual_status="processing",
            speech_status="queued",
            ocr_status="queued",
            det_status="queued",
        )

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError("Unable to open video file.")
        native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration_sec = round(total_frames / native_fps, 2) if total_frames > 0 else 0.0
        cap.release()

        video_frames_dir = FRAMES_DIR / video_id
        video_frames_dir.mkdir(parents=True, exist_ok=True)
        if work_dir.exists():
            shutil.rmtree(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

        audio_wav = video_frames_dir / "audio_16k.wav"
        base_fps = max(1.0, sampling_fps)

        with timed(f"V5 FFmpeg extract ({video_id[:8]})"):
            cmd = [
                "ffmpeg",
                "-y",
                "-hwaccel",
                "auto",
                "-i",
                str(video_path),
                "-map",
                "0:v:0",
                "-vf",
                f"fps={base_fps},scale='if(gt(iw,ih),-2,384)':'if(gt(iw,ih),384,-2)'",
                "-q:v",
                "3",
                str(work_dir / "raw_%06d.jpg"),
                "-map",
                "0:a:0?",
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                str(audio_wav),
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        raw_files = sorted(work_dir.glob("raw_*.jpg"))
        if not raw_files:
            raise ValueError("FFmpeg extracted 0 frames.")

        db.update_video_status(
            video_id,
            status="processing",
            progress=25.0,
            duration_sec=duration_sec,
            fps=round(native_fps, 2),
        )

        raw_pairs: List[Tuple[float, Path]] = []
        vec_chunks: List[np.ndarray] = []
        batch_imgs: List[Image.Image] = []

        with timed(f"V5 streamed visual embed ({len(raw_files)} frames)"):
            for idx, fp in enumerate(raw_files):
                bgr = cv2.imread(str(fp))
                if bgr is None:
                    continue
                ts = round(idx / base_fps, 2)
                raw_pairs.append((ts, fp))
                batch_imgs.append(pad_to_square_pil(bgr, size=384, is_bgr=True))
                if len(batch_imgs) == 64:
                    vec_chunks.append(encode_images_batch_v5(batch_imgs, batch_size=64))
                    batch_imgs.clear()
            if batch_imgs:
                vec_chunks.append(encode_images_batch_v5(batch_imgs, batch_size=64))
                batch_imgs.clear()

        if not vec_chunks:
            raise ValueError("No readable extracted frames.")

        all_vecs = np.vstack(vec_chunks)
        db.update_video_status(video_id, status="processing", progress=60.0)

        shots, kept_items = _detect_shots_and_dedup(raw_pairs, all_vecs)

        shot_rows = []
        for s in shots:
            sh_name = f"shot_v5_{s['shot_idx']:04d}.jpg"
            shutil.copy2(s["keyframe_path"], video_frames_dir / sh_name)
            shot_rows.append((video_id, s["shot_idx"], s["start_time"], s["end_time"], sh_name))

        kept_vecs = np.vstack([item["vec"] for item in kept_items]).astype(np.float32)
        visual_index = faiss.IndexFlatIP(kept_vecs.shape[1])
        visual_index.add(kept_vecs)
        faiss.write_index(visual_index, str(INDEXES_DIR / f"{video_id}_v5_visual.faiss"))

        frame_rows = []
        for f_idx, item in enumerate(kept_items):
            kf_name = f"frame_v5_{f_idx:05d}.jpg"
            shutil.copy2(item["path"], video_frames_dir / kf_name)
            frame_rows.append(
                (
                    video_id,
                    f_idx,
                    item["shot_idx"],
                    item["start_time"],
                    item["end_time"],
                    item["peak_timestamp"],
                    kf_name,
                )
            )

        db.save_v5_visual_data(video_id, shot_rows, frame_rows)
        db.upsert_index_status_v5(video_id, visual_status="ready")

        db.update_video_status(
            video_id,
            status="completed",
            progress=100.0,
            duration_sec=duration_sec,
            fps=round(native_fps, 2),
            frame_count=len(frame_rows),
            sampling_fps=base_fps,
            pipeline="v5",
            error_message=None,
        )
        log_v5(f"[V5 INGEST] Visual index ready ({len(frame_rows)} keyframes); starting background speech & OCR/Det...")

        _build_speech_index_v5(video_id, audio_wav)
        if os.getenv("V5_BACKGROUND_OCR_DET", "1") == "1":
            build_background_ocr_and_det_v5(video_id)

        log_v5(f"[V5 INGEST] All background tasks finished for video={video_id[:8]}.")

    except Exception as exc:
        log_v5(f"[V5] Ingestion error: {exc}")
        db.upsert_index_status_v5(video_id, visual_status="failed")
        db.update_video_status(video_id, status="failed", progress=0.0, error_message=str(exc))
    finally:
        if work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)


# ============================================================
# SEARCH: FAST FUSION (<1s) + OPTIONAL VLM RERANK (CONTACT SHEET + TIMEOUT)
# ============================================================

def _temporal_iou(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def _zscore_nonzero(arr: np.ndarray) -> np.ndarray:
    """Normalize only over active (> 0) bins so sparse speech/OCR hits don't explode."""
    out = np.zeros_like(arr, dtype=np.float32)
    nz = arr > 0
    if int(np.sum(nz)) < 2:
        out[nz] = np.clip(arr[nz], 0.0, 2.0)
        return out
    vals = arr[nz]
    std = float(np.std(vals))
    if std < 1e-6:
        out[nz] = 1.0
        return out
    out[nz] = np.clip((vals - float(np.mean(vals))) / std, -1.0, 3.0)
    return out


def _build_contact_sheet_b64(paths: List[Path], tile_size: int = 256) -> Optional[str]:
    """Combines up to 4 frames into a single 2x2 contact sheet JPEG for fast 1-image VLM calls."""
    valid = [p for p in paths if p.exists()][:4]
    if not valid:
        return None
    if len(valid) == 1:
        with Image.open(valid[0]) as img:
            rgb = img.convert("RGB")
            rgb.thumbnail((384, 384))
            buf = BytesIO()
            rgb.save(buf, format="JPEG", quality=75)
            return base64.b64encode(buf.getvalue()).decode("utf-8")

    cols, rows = 2, 2
    sheet = Image.new("RGB", (cols * tile_size, rows * tile_size), (30, 30, 30))
    for idx, p in enumerate(valid):
        with Image.open(p) as img:
            tile = img.convert("RGB")
            tile.thumbnail((tile_size, tile_size))
            r, c = divmod(idx, cols)
            x = c * tile_size + (tile_size - tile.width) // 2
            y = r * tile_size + (tile_size - tile.height) // 2
            sheet.paste(tile, (x, y))

    buf = BytesIO()
    sheet.save(buf, format="JPEG", quality=75)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _build_contact_sheet_pil(paths: List[Path], tile_size: int = 256) -> Optional[Image.Image]:
    b64 = _build_contact_sheet_b64(paths, tile_size=tile_size)
    if not b64:
        return None
    return Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")


async def _call_llm_json_v5(
    prompt: str,
    images: Optional[List[Path]] = None,
    llm_provider: str = "ollama",
    llm_model: Optional[str] = None,
    timeout_sec: float = 20.0,
) -> Dict[str, Any]:
    provider = (llm_provider or "ollama").lower()

    if provider == "gemini":
        import google.genai as genai
        from google.genai import types

        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set.")
        client = genai.Client(api_key=api_key)
        model_name = llm_model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        contents: List[Any] = [prompt]
        sheet_pil = _build_contact_sheet_pil(images or [])
        if sheet_pil is not None:
            contents.append(sheet_pil)

        resp = await asyncio.wait_for(
            asyncio.to_thread(
                client.models.generate_content,
                model=model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1,
                ),
            ),
            timeout=timeout_sec,
        )
        return json.loads(resp.text or "{}")

    from ollama import AsyncClient

    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    model_name = llm_model or os.getenv(
        "OLLAMA_VISION_MODEL", os.getenv("OLLAMA_MODEL", "gemma3:4b")
    )
    client = AsyncClient(host=host)
    msg: Dict[str, Any] = {"role": "user", "content": prompt}
    sheet_b64 = _build_contact_sheet_b64(images or [])
    if sheet_b64:
        msg["images"] = [sheet_b64]

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
    return json.loads(resp["message"]["content"])


def _parse_query_v5(query: str) -> Dict[str, Any]:
    """[1] Word-boundary query parsing with stopword removal."""
    quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'', query)
    quotes = [a or b for a, b in quoted if (a or b).strip()]
    digits = re.findall(r"\b\d+(?::\d+)*\b", query)
    all_tokens = re.findall(r"[a-z0-9]+", query.lower())
    content_tokens = [t for t in all_tokens if len(t) >= 2 and t not in STOPWORDS]
    token_set = set(content_tokens)

    text_heavy = bool(quotes or digits) or bool(
        token_set & {"text", "sign", "plate", "number", "timestamp", "label", "board", "written"}
    )
    object_heavy = bool(
        token_set
        & {"person", "people", "man", "woman", "bag", "car", "bike", "phone", "knife", "weapon", "crowd", "vehicle"}
    )

    phrasings = [query.strip()]
    short_ver = " ".join(content_tokens[:5])
    if short_ver and short_ver.lower() != query.strip().lower():
        phrasings.append(short_ver)

    return {
        "phrasings": phrasings[:2],
        "quotes": quotes,
        "digits": digits,
        "tokens": content_tokens,
        "text_heavy": text_heavy,
        "object_heavy": object_heavy,
    }


def _fuse_1s_timeline_and_propose(
    video_id: str,
    duration_sec: float,
    frames: List[Dict[str, Any]],
    visual_scores: Dict[int, float],
    speech_windows: List[Dict[str, Any]],
    speech_scores: Dict[int, float],
    ocr_scores: Dict[int, float],
    det_scores: Dict[int, float],
    merge_gap_sec: float = 2.0,
    max_candidates: int = 8,
) -> List[Dict[str, Any]]:
    T = max(1, int(np.ceil(max(duration_sec, 1.0))))
    vis_tl = np.zeros(T, dtype=np.float32)
    sp_tl = np.zeros(T, dtype=np.float32)
    ocr_tl = np.zeros(T, dtype=np.float32)
    det_tl = np.zeros(T, dtype=np.float32)

    bin_to_frame: Dict[int, Dict[str, Any]] = {}
    for row in frames:
        f_idx = int(row["faiss_idx"])
        s_bin = max(0, min(T - 1, int(np.floor(float(row["start_time"])))))
        e_bin = max(s_bin, min(T - 1, int(np.ceil(float(row["end_time"])))))
        v_sc = visual_scores.get(f_idx, 0.0)
        o_sc = ocr_scores.get(f_idx, 0.0)
        d_sc = det_scores.get(f_idx, 0.0)
        for b in range(s_bin, e_bin + 1):
            if v_sc >= vis_tl[b]:
                vis_tl[b] = v_sc
                bin_to_frame[b] = row
            ocr_tl[b] = max(ocr_tl[b], o_sc)
            det_tl[b] = max(det_tl[b], d_sc)

    bin_to_speech: Dict[int, str] = {}
    for w in speech_windows:
        w_id = int(w["id"])
        sc = speech_scores.get(w_id, 0.0)
        if sc <= 0:
            continue
        s_bin = max(0, min(T - 1, int(np.floor(float(w["start_time"])))))
        e_bin = max(s_bin, min(T - 1, int(np.ceil(float(w["end_time"])))))
        for b in range(s_bin, e_bin + 1):
            if sc > sp_tl[b]:
                sp_tl[b] = sc
                bin_to_speech[b] = str(w["transcript"])

    z_vis = _zscore_nonzero(vis_tl)
    z_sp = _zscore_nonzero(sp_tl)
    z_ocr = _zscore_nonzero(ocr_tl)
    z_det = _zscore_nonzero(det_tl)

    fused = 0.60 * z_vis + 0.20 * z_sp + 0.10 * z_ocr + 0.10 * z_det
    active_count = (
        (z_vis > 0.3).astype(np.float32)
        + (z_sp > 0.3).astype(np.float32)
        + (z_ocr > 0.3).astype(np.float32)
        + (z_det > 0.3).astype(np.float32)
    )
    fused = fused + 0.25 * np.maximum(0.0, active_count - 1.0)

    smooth = np.convolve(fused, np.ones(3) / 3.0, mode="same")
    thresh = float(np.mean(smooth) + 0.65 * np.std(smooth))

    above = np.where(smooth >= thresh)[0]
    if above.size == 0:
        above = np.sort(np.argsort(smooth)[::-1][: min(6, T)])

    raw_groups: List[List[int]] = [[int(above[0])]]
    for b in above[1:]:
        b_int = int(b)
        if b_int - raw_groups[-1][-1] <= max(1, int(round(merge_gap_sec))):
            raw_groups[-1].append(b_int)
        else:
            raw_groups.append([b_int])

    proposals: List[Dict[str, Any]] = []
    base_level = float(np.percentile(smooth, 25))

    for grp in raw_groups:
        peak_bin = max(grp, key=lambda b: float(smooth[b]))
        peak_val = float(smooth[peak_bin])
        half_height = base_level + 0.5 * max(0.0, peak_val - base_level)

        left = grp[0]
        while left > 0 and float(smooth[left - 1]) >= half_height and (peak_bin - left) < 15:
            left -= 1
        right = grp[-1]
        while right + 1 < T and float(smooth[right + 1]) >= half_height and (right - peak_bin) < 15:
            right += 1

        sample_bins = np.linspace(left, right, num=min(4, max(1, right - left + 1)), dtype=int)
        kf_paths: List[Path] = []
        kf_urls: List[str] = []
        seen_names = set()
        peak_row = bin_to_frame.get(peak_bin) or (frames[0] if frames else None)

        for sb in sample_bins:
            fr = bin_to_frame.get(int(sb))
            if fr and fr["thumbnail_name"] not in seen_names:
                seen_names.add(fr["thumbnail_name"])
                kf_paths.append(FRAMES_DIR / video_id / fr["thumbnail_name"])
                kf_urls.append(f"/api/videos/{video_id}/frames/{fr['thumbnail_name']}")

        if not kf_urls and peak_row:
            kf_paths.append(FRAMES_DIR / video_id / peak_row["thumbnail_name"])
            kf_urls.append(f"/api/videos/{video_id}/frames/{peak_row['thumbnail_name']}")

        raw_vis = float(vis_tl[peak_bin])
        ocr_snip = (peak_row.get("ocr_text") if peak_row else "") or None
        det_objs = json.loads((peak_row.get("det_labels_json") if peak_row else "[]") or "[]")
        sp_snip = bin_to_speech.get(peak_bin)

        proposals.append(
            {
                "start_time": float(left),
                "end_time": float(min(duration_sec or (right + 1), right + 1)),
                "peak_timestamp": float(peak_bin),
                "fused_score": round(peak_val, 4),
                "similarity_score": round(raw_vis, 4),
                "thumbnail_url": kf_urls[0] if kf_urls else "",
                "_kf_paths": kf_paths,
                "keyframes": kf_urls,
                "detected_objects": det_objs,
                "ocr_text": ocr_snip,
                "speech_snippet": sp_snip,
            }
        )

    proposals.sort(key=lambda x: x["fused_score"], reverse=True)
    kept: List[Dict[str, Any]] = []
    for prop in proposals:
        if all(
            _temporal_iou((prop["start_time"], prop["end_time"]), (k["start_time"], k["end_time"])) <= 0.5
            for k in kept
        ):
            kept.append(prop)
        if len(kept) >= max_candidates:
            break

    return kept


async def _pointwise_vlm_rerank_v5(
    query: str,
    candidates: List[Dict[str, Any]],
    llm_provider: str,
    llm_model: Optional[str],
) -> List[Dict[str, Any]]:
    """[6] Pointwise VLM rerank using 2x2 contact sheet + 20s per-call timeout."""
    concurrency = 1 if (llm_provider or "").lower() == "ollama" else 4
    sem = asyncio.Semaphore(concurrency)
    total = len(candidates)
    log_v5(
        f"[V5 SEARCH 6/6] Starting VLM rerank on {total} candidates "
        f"(provider={llm_provider}, concurrency={concurrency})..."
    )

    async def score_one(i: int, cand: Dict[str, Any]) -> Dict[str, Any]:
        async with sem:
            t0 = time.perf_counter()
            log_v5(
                f"[V5 SEARCH 6/6] Reranking candidate {i + 1}/{total} "
                f"({cand['start_time']}s–{cand['end_time']}s, peak={cand['peak_timestamp']}s)..."
            )
            evidence = {
                "start_time": cand["start_time"],
                "end_time": cand["end_time"],
                "peak_timestamp": cand["peak_timestamp"],
                "visual_similarity": cand["similarity_score"],
                "transcript_snippet": cand.get("speech_snippet"),
                "ocr_text": cand.get("ocr_text"),
                "detected_objects": cand.get("detected_objects", []),
            }
            prompt = (
                "Verify whether this video segment matches the user's query.\n"
                f"Query: \"{query}\"\n"
                f"Segment Evidence: {json.dumps(evidence)}\n"
                "Inspect the attached 2x2 frame contact sheet (left-to-right, top-to-bottom across the window).\n"
                'Return ONLY valid JSON: {"relevance": 0.0, "matched": true, "reason": "short explanation"}'
            )
            try:
                res = await _call_llm_json_v5(
                    prompt=prompt,
                    images=cand.get("_kf_paths", [])[:4],
                    llm_provider=llm_provider,
                    llm_model=llm_model,
                    timeout_sec=20.0,
                )
                rel = max(0.0, min(1.0, float(res.get("relevance", 0.0))))
                cand["vlm_score"] = rel
                cand["llm_selected"] = bool(res.get("matched", rel >= 0.6))
                cand["llm_reason"] = str(res.get("reason") or "")
                log_v5(
                    f"[V5 SEARCH 6/6] Candidate {i + 1}/{total} finished in {time.perf_counter() - t0:.2f}s "
                    f"(relevance={rel:.2f}, matched={cand['llm_selected']})"
                )
            except Exception as exc:
                log_v5(
                    f"[V5 SEARCH 6/6] Candidate {i + 1}/{total} fallback after {time.perf_counter() - t0:.2f}s: {exc}"
                )
                cand["vlm_score"] = None
            return cand

    scored = await asyncio.gather(*(score_one(i, c) for i, c in enumerate(candidates)))
    scored.sort(
        key=lambda x: (
            x["vlm_score"] if x.get("vlm_score") is not None else -1.0,
            x["fused_score"],
        ),
        reverse=True,
    )
    return list(scored)


async def search_video_index_v5(
    *,
    video_id: str,
    query: str,
    top_k: int = 5,
    merge_gap_sec: float = 2.0,
    llm_provider: str = "ollama",
    llm_model: Optional[str] = None,
    rerank: bool = True,
) -> List[Dict[str, Any]]:
    t_total = time.perf_counter()
    log_v5(
        f"[V5 SEARCH] video={video_id[:8]} query=\"{query}\" "
        f"provider={llm_provider} rerank={rerank} top_k={top_k}"
    )

    vis_index_path = INDEXES_DIR / f"{video_id}_v5_visual.faiss"
    if not vis_index_path.exists():
        log_v5(f"[V5 SEARCH] Missing visual index at {vis_index_path}")
        raise FileNotFoundError("V5 visual index not found. Index the video with V5 first.")

    record = db.get_video(video_id) or {}
    duration_sec = float(record.get("duration_sec") or 0.0)
    frames = db.get_frame_meta_v5(video_id)
    if not frames:
        log_v5("[V5 SEARCH] No frame_meta_v5 entries found; returning empty list.")
        return []

    log_v5(f"[V5 SEARCH 1/6] Parsing query & loading {len(frames)} frame metadata rows...")
    with timed("V5 [1] Query parse"):
        q_info = _parse_query_v5(query)
    log_v5(
        f"[V5 SEARCH 1/6] Tokens={q_info['tokens']} phrasings={q_info['phrasings']} "
        f"text_heavy={q_info['text_heavy']} obj_heavy={q_info['object_heavy']}"
    )

    log_v5("[V5 SEARCH 2/6] Querying visual FAISS, speech BM25/FAISS, and ready OCR/Det metadata...")
    with timed("V5 [2] Visual + Speech + Ready OCR/Det search"):
        t_sub = time.perf_counter()
        vis_idx = get_faiss_index(vis_index_path)
        log_v5(
            f"[V5 SEARCH 2a] Loaded visual FAISS ({vis_idx.ntotal} vecs) in {time.perf_counter() - t_sub:.2f}s; "
            f"encoding visual query text..."
        )

        t_sub = time.perf_counter()
        q_vecs = encode_text_visual_v5(q_info["phrasings"])
        scores, indices = vis_idx.search(q_vecs, vis_idx.ntotal)
        vis_scores: Dict[int, float] = {}
        for row_s, row_i in zip(scores, indices):
            for s, i in zip(row_s, row_i):
                if i >= 0:
                    vis_scores[int(i)] = max(vis_scores.get(int(i), 0.0), float(s))
        log_v5(
            f"[V5 SEARCH 2b] Visual search done in {time.perf_counter() - t_sub:.2f}s "
            f"({len(vis_scores)} hits). Loading speech windows..."
        )

        t_sub = time.perf_counter()
        speech_windows = db.get_speech_windows_v5(video_id)
        speech_scores: Dict[int, float] = {}
        log_v5(
            f"[V5 SEARCH 2c] Loaded {len(speech_windows)} speech windows in {time.perf_counter() - t_sub:.2f}s."
        )

        if speech_windows:
            t_sub = time.perf_counter()
            fts_terms = " OR ".join(f'"{t}"' for t in q_info["tokens"][:6])
            bm25_map = db.search_speech_bm25_v5(video_id, fts_terms)
            log_v5(
                f"[V5 SEARCH 2d] BM25 done in {time.perf_counter() - t_sub:.2f}s ({len(bm25_map)} hits)."
            )

            dense_map: Dict[int, float] = {}
            sp_idx_path = INDEXES_DIR / f"{video_id}_v5_speech.faiss"
            if sp_idx_path.exists():
                t_sub = time.perf_counter()
                log_v5("[V5 SEARCH 2e] Encoding speech query & searching speech FAISS...")
                sp_idx = get_faiss_index(sp_idx_path)
                qv = _encode_speech_texts([query])
                sc, ix = sp_idx.search(qv, sp_idx.ntotal)
                fidx_to_id = {int(w["faiss_idx"]): int(w["id"]) for w in speech_windows}
                for s, i in zip(sc[0], ix[0]):
                    if int(i) in fidx_to_id:
                        dense_map[fidx_to_id[int(i)]] = max(0.0, float(s))
                log_v5(
                    f"[V5 SEARCH 2e] Speech FAISS done in {time.perf_counter() - t_sub:.2f}s "
                    f"({len(dense_map)} hits)."
                )

            bm25_rank = {wid: r + 1 for r, wid in enumerate(sorted(bm25_map, key=bm25_map.get, reverse=True))}
            dense_rank = {wid: r + 1 for r, wid in enumerate(sorted(dense_map, key=dense_map.get, reverse=True))}
            for w in speech_windows:
                wid = int(w["id"])
                sc_val = (1.0 / (60.0 + bm25_rank[wid]) if wid in bm25_rank else 0.0) + (
                    1.0 / (60.0 + dense_rank[wid]) if wid in dense_rank else 0.0
                )
                if sc_val > 0:
                    speech_scores[wid] = sc_val

        t_sub = time.perf_counter()
        ocr_scores: Dict[int, float] = {}
        det_scores: Dict[int, float] = {}
        terms = q_info["tokens"] + [x.lower() for x in q_info["quotes"] + q_info["digits"]]
        for r in frames:
            f_idx = int(r["faiss_idx"])
            otxt = (r.get("ocr_text") or "").lower()
            if otxt and terms:
                hits = sum(1 for t in terms if re.search(rf"\b{re.escape(t)}\b", otxt))
                if hits > 0:
                    ocr_scores[f_idx] = float(hits)
            labels = json.loads(r.get("det_labels_json") or "[]")
            if labels and terms:
                dhits = sum(1 for lbl in labels if lbl in terms)
                if dhits > 0:
                    det_scores[f_idx] = float(dhits)
        log_v5(
            f"[V5 SEARCH 2f] OCR/Det scan done in {time.perf_counter() - t_sub:.2f}s."
        )

    log_v5(
        f"[V5 SEARCH 2/6] Hits -> visual={len(vis_scores)}, speech={len(speech_scores)}, "
        f"ocr={len(ocr_scores)}, det={len(det_scores)}"
    )

    log_v5("[V5 SEARCH 3-4/6] Fusing 1s timeline, refining boundaries, and running 1D temporal NMS...")
    with timed("V5 [3-4] 1s timeline fusion + NMS"):
        candidates = _fuse_1s_timeline_and_propose(
            video_id=video_id,
            duration_sec=duration_sec,
            frames=frames,
            visual_scores=vis_scores,
            speech_windows=speech_windows,
            speech_scores=speech_scores,
            ocr_scores=ocr_scores,
            det_scores=det_scores,
            merge_gap_sec=merge_gap_sec,
            max_candidates=8,
        )
    log_v5(f"[V5 SEARCH 4/6] Proposed {len(candidates)} grounded segments.")

    if not candidates:
        log_v5("[V5 SEARCH] No candidate segments found; returning empty list.")
        return []

    pool_limit = 3 if (llm_provider or "").lower() == "ollama" else 5
    rerank_pool = candidates[: min(len(candidates), pool_limit)]
    if rerank and (llm_provider or "none").lower() in {"ollama", "gemini"} and rerank_pool:
        with timed(f"V5 [6] VLM rerank ({llm_provider}, n={len(rerank_pool)})"):
            reranked = await _pointwise_vlm_rerank_v5(
                query=query,
                candidates=rerank_pool,
                llm_provider=llm_provider,
                llm_model=llm_model,
            )
            final_pool = reranked + candidates[len(rerank_pool) :]
    else:
        log_v5(f"[V5 SEARCH 6/6] Skipping VLM rerank (rerank={rerank}, provider={llm_provider}).")
        final_pool = candidates

    output: List[Dict[str, Any]] = []
    for rank, item in enumerate(final_pool[:top_k], start=1):
        vlm_sc = item.get("vlm_score")
        if vlm_sc is not None:
            conf = round(float(vlm_sc), 3)
        else:
            conf = round(min(1.0, max(0.05, 0.5 + 0.15 * float(item["fused_score"]))), 3)

        review_flag = (
            "verified"
            if conf >= 0.70
            else ("needs_review" if conf >= 0.45 else "low_confidence")
        )
        reason_parts = []
        if item.get("llm_reason"):
            reason_parts.append(item["llm_reason"])
        if item.get("speech_snippet"):
            reason_parts.append(f"Speech: \"{item['speech_snippet'][:120]}\"")

        output.append(
            {
                "rank": rank,
                "start_time": item["start_time"],
                "end_time": item["end_time"],
                "peak_timestamp": item["peak_timestamp"],
                "similarity_score": item["similarity_score"],
                "thumbnail_url": item["thumbnail_url"],
                "confidence": conf,
                "ai_review_flag": review_flag,
                "detected_objects": item.get("detected_objects", []),
                "ocr_text": item.get("ocr_text"),
                "keyframes": item.get("keyframes", []),
                "llm_selected": bool(item.get("llm_selected") or (rank == 1 and vlm_sc is not None)),
                "llm_reason": " | ".join(reason_parts) if reason_parts else None,
            }
        )

    log_v5(
        f"[V5 SEARCH] Done in {time.perf_counter() - t_total:.2f}s "
        f"(rerank={rerank}, returning {len(output)} matches)."
    )
    return output