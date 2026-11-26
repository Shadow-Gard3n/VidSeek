import asyncio
import base64
from functools import lru_cache
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any, Dict, List, Optional

import cv2
import faiss
import numpy as np
from PIL import Image

from config import DEFAULT_SAMPLING_FPS, FRAMES_DIR, INDEXES_DIR
import database as db
from embedding import embedding_service


def log_v6(msg: str) -> None:
    print(msg, flush=True)


@lru_cache(maxsize=8)
def _load_index_cached(path_str: str, mtime_ns: int) -> faiss.Index:
    return faiss.read_index(path_str)


def get_faiss_index(path: Path) -> faiss.Index:
    stat = path.stat()
    return _load_index_cached(str(path), stat.st_mtime_ns)


# ============================================================
# INGESTION: PURE CLIP FRAME INDEXING
# ============================================================

def process_video_task_v6(
    video_id: str,
    video_path: Path,
    sampling_fps: float = DEFAULT_SAMPLING_FPS,
) -> None:
    """
    V6 Ingestion:
    1. Extract frames with FFmpeg at sampling_fps.
    2. Encode every frame with CLIP (embedding_service).
    3. Save FAISS index ({video_id}_v6.faiss) + frame_metadata_v6 rows.
    """
    t0 = time.perf_counter()
    video_frames_dir = FRAMES_DIR / video_id
    tmp_dir = video_frames_dir / "_v6_tmp"

    try:
        log_v6(f"[V6 INGEST] Starting CLIP-only indexing for video={video_id[:8]} at {sampling_fps} fps...")
        db.update_video_status(
            video_id,
            status="processing",
            progress=5.0,
            sampling_fps=sampling_fps,
            pipeline="v6",
            error_message=None,
        )

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError("Unable to open video file.")
        native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration_sec = round(total_frames / native_fps, 2) if total_frames > 0 else 0.0
        cap.release()

        video_frames_dir.mkdir(parents=True, exist_ok=True)
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vf",
            f"fps={sampling_fps}",
            "-q:v",
            "2",
            str(tmp_dir / "raw_%06d.jpg"),
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        raw_files = sorted(tmp_dir.glob("raw_*.jpg"))
        if not raw_files:
            raise ValueError("FFmpeg extracted 0 frames.")

        db.update_video_status(
            video_id,
            status="processing",
            progress=30.0,
            duration_sec=duration_sec,
            fps=round(native_fps, 2),
        )

        embedding_service.load()
        vecs: List[np.ndarray] = []
        records: List[tuple] = []

        for idx, raw_fp in enumerate(raw_files):
            with Image.open(raw_fp) as img:
                rgb = img.convert("RGB")
                vec = embedding_service.encode_image(rgb)
            vecs.append(vec)

            ts = round(idx / sampling_fps, 2)
            out_name = f"frame_v6_{idx:05d}.jpg"
            shutil.copy2(raw_fp, video_frames_dir / out_name)
            records.append((video_id, idx, ts, out_name))

            if (idx + 1) % 25 == 0 or (idx + 1) == len(raw_files):
                pct = 30.0 + 60.0 * ((idx + 1) / len(raw_files))
                db.update_video_status(video_id, status="processing", progress=round(pct, 1))
                log_v6(f"[V6 INGEST] Encoded {idx + 1}/{len(raw_files)} frames...")

        mat = np.vstack(vecs).astype(np.float32)
        faiss.normalize_L2(mat)

        INDEXES_DIR.mkdir(parents=True, exist_ok=True)
        index = faiss.IndexFlatIP(mat.shape[1])
        index.add(mat)
        faiss.write_index(index, str(INDEXES_DIR / f"{video_id}_v6.faiss"))

        db.save_frame_records_v6(video_id, records)
        db.update_video_status(
            video_id,
            status="completed",
            progress=100.0,
            duration_sec=duration_sec,
            fps=round(native_fps, 2),
            frame_count=len(records),
            sampling_fps=sampling_fps,
            pipeline="v6",
            error_message=None,
        )
        log_v6(
            f"[V6 INGEST] Done in {time.perf_counter() - t0:.2f}s "
            f"({len(records)} frames indexed)."
        )
    except Exception as exc:
        log_v6(f"[V6 INGEST] Failed: {exc}")
        db.update_video_status(video_id, status="failed", progress=0.0, error_message=str(exc))
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)


# ============================================================
# SEARCH: CLIP PEAK BOUNDARY DETECTION + LLM VERIFICATION
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
    video_id: str,
    frames: List[Dict[str, Any]],
    sims: np.ndarray,
    duration_sec: float,
    sampling_fps: float,
    top_k: int,
    max_half_window_sec: float = 4.0,
    min_peak_sep_sec: float = 6.0,
) -> List[Dict[str, Any]]:
    """
    Finds distinct local peaks in CLIP similarity and builds tight boundaries
    around each peak without collapsing the whole video into one blob.
    """
    n = len(frames)
    if n == 0:
        return []

    if n >= 3:
        smooth = np.convolve(sims, np.array([0.25, 0.50, 0.25], dtype=np.float32), mode="same")
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
        while left > 0 and (peak_idx - left) < max_expand_frames and float(smooth[left - 1]) >= cutoff:
            left -= 1

        right = peak_idx
        while right + 1 < n and (right - peak_idx) < max_expand_frames and float(smooth[right + 1]) >= cutoff:
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
        kf_paths = [FRAMES_DIR / video_id / frames[i]["thumbnail_name"] for i in sample_indices]
        kf_urls = [f"/api/videos/{video_id}/frames/{frames[i]['thumbnail_name']}" for i in sample_indices]
        peak_thumb_url = f"/api/videos/{video_id}/frames/{frames[peak_idx]['thumbnail_name']}"

        proposals.append(
            {
                "start_time": start_time,
                "end_time": end_time,
                "peak_timestamp": peak_ts,
                "similarity_score": round(float(sims[peak_idx]), 4),
                "thumbnail_url": peak_thumb_url,
                "keyframes": kf_urls,
                "_kf_paths": kf_paths,
                "_kf_timestamps": [round(float(frames[i]["timestamp_sec"]), 2) for i in sample_indices],
            }
        )

    return proposals


async def _call_vlm_v6(
    prompt: str,
    image_paths: List[Path],
    llm_provider: str,
    llm_model: Optional[str],
    timeout_sec: float = 25.0,
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
        for p in image_paths:
            if p.exists():
                with Image.open(p) as im:
                    rgb = im.convert("RGB")
                    rgb.thumbnail((512, 512))
                    contents.append(rgb.copy())

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
    model_name = llm_model or os.getenv("OLLAMA_VISION_MODEL", os.getenv("OLLAMA_MODEL", "gemma3:4b"))
    client = AsyncClient(host=host)

    b64_imgs = [b for p in image_paths if (b := _encode_image_b64(p, max_dim=512)) is not None]
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
    return json.loads(resp["message"]["content"])


async def _llm_verify_and_refine_v6(
    query: str,
    candidates: List[Dict[str, Any]],
    llm_provider: str,
    llm_model: Optional[str],
) -> List[Dict[str, Any]]:
    concurrency = 1 if (llm_provider or "").lower() == "ollama" else 4
    sem = asyncio.Semaphore(concurrency)
    total = len(candidates)

    async def verify_one(idx: int, cand: Dict[str, Any]) -> Dict[str, Any]:
        async with sem:
            t0 = time.perf_counter()
            log_v6(
                f"[V6 SEARCH 3/3] LLM verifying candidate {idx + 1}/{total} "
                f"({cand['start_time']}s–{cand['end_time']}s, peak={cand['peak_timestamp']}s)..."
            )
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
                res = await _call_vlm_v6(
                    prompt=prompt,
                    image_paths=cand.get("_kf_paths", []),
                    llm_provider=llm_provider,
                    llm_model=llm_model,
                    timeout_sec=25.0,
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
                log_v6(
                    f"[V6 SEARCH 3/3] Candidate {idx + 1}/{total} done in {time.perf_counter() - t0:.2f}s "
                    f"(matched={matched}, relevance={rel:.2f}, window={cand['start_time']}s–{cand['end_time']}s)"
                )
            except Exception as exc:
                log_v6(f"[V6 SEARCH 3/3] Candidate {idx + 1}/{total} LLM error: {exc}")
                cand["llm_relevance"] = None
                cand["llm_selected"] = False
                cand["llm_reason"] = None
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


async def search_video_index_v6(
    *,
    video_id: str,
    query: str,
    top_k: int = 5,
    merge_gap_sec: float = 2.0,
    llm_provider: str = "ollama",
    llm_model: Optional[str] = None,
    rerank: bool = True,
) -> List[Dict[str, Any]]:
    t_start = time.perf_counter()
    log_v6(
        f"[V6 SEARCH] video={video_id[:8]} query=\"{query}\" "
        f"provider={llm_provider} rerank={rerank} top_k={top_k}"
    )

    index_path = INDEXES_DIR / f"{video_id}_v6.faiss"
    if not index_path.exists():
        raise FileNotFoundError("V6 CLIP index not found. Index the video with v6 first.")

    record = db.get_video(video_id) or {}
    duration_sec = float(record.get("duration_sec") or 0.0)
    sampling_fps = float(record.get("sampling_fps") or DEFAULT_SAMPLING_FPS)
    frames = db.get_frame_records_v6(video_id)
    if not frames:
        return []

    t0 = time.perf_counter()
    embedding_service.load()
    q_vec = embedding_service.encode_text(query).astype(np.float32)
    faiss.normalize_L2(q_vec)

    index = get_faiss_index(index_path)
    scores, indices = index.search(q_vec, index.ntotal)

    sims = np.zeros(len(frames), dtype=np.float32)
    for sc, idx in zip(scores[0], indices[0]):
        if 0 <= int(idx) < len(frames):
            sims[int(idx)] = float(sc)

    log_v6(
        f"[V6 SEARCH 1/3] CLIP similarity computed across {len(frames)} frames in {time.perf_counter() - t0:.2f}s "
        f"(min={float(sims.min()):.3f}, max={float(sims.max()):.3f})."
    )

    t0 = time.perf_counter()
    candidates = _propose_clip_boundaries(
        video_id=video_id,
        frames=frames,
        sims=sims,
        duration_sec=duration_sec,
        sampling_fps=sampling_fps,
        top_k=top_k,
        max_half_window_sec=4.0,
        min_peak_sep_sec=max(4.0, merge_gap_sec * 2.0),
    )
    log_v6(
        f"[V6 SEARCH 2/3] Proposed {len(candidates)} distinct CLIP peak boundaries in {time.perf_counter() - t0:.2f}s: "
        + ", ".join(f"[{c['start_time']}s-{c['end_time']}s @ {c['similarity_score']:.3f}]" for c in candidates)
    )

    pool_limit = min(len(candidates), max(top_k, 4) if (llm_provider or "").lower() == "gemini" else min(top_k, 3))
    if rerank and (llm_provider or "none").lower() in {"ollama", "gemini"} and candidates:
        verified = await _llm_verify_and_refine_v6(
            query=query,
            candidates=candidates[:pool_limit],
            llm_provider=llm_provider,
            llm_model=llm_model,
        )
        final_list = verified + candidates[pool_limit:]
    else:
        log_v6(f"[V6 SEARCH 3/3] Skipping LLM verification (rerank={rerank}, provider={llm_provider}).")
        final_list = candidates

    results: List[Dict[str, Any]] = []
    for rank, item in enumerate(final_list[:top_k], start=1):
        llm_rel = item.get("llm_relevance")
        if llm_rel is not None:
            conf = round(float(llm_rel), 3)
        else:
            conf = round(min(1.0, max(0.05, (float(item["similarity_score"]) - 0.15) / 0.20)), 3)

        flag = "verified" if conf >= 0.65 else ("needs_review" if conf >= 0.40 else "low_confidence")
        results.append(
            {
                "rank": rank,
                "start_time": item["start_time"],
                "end_time": item["end_time"],
                "peak_timestamp": item["peak_timestamp"],
                "similarity_score": item["similarity_score"],
                "thumbnail_url": item["thumbnail_url"],
                "confidence": conf,
                "ai_review_flag": flag,
                "detected_objects": [],
                "ocr_text": None,
                "keyframes": item.get("keyframes", []),
                "llm_selected": bool(item.get("llm_selected", False)),
                "llm_reason": item.get("llm_reason"),
            }
        )

    log_v6(f"[V6 SEARCH] Finished in {time.perf_counter() - t_start:.2f}s (returning {len(results)} matches).")
    return results