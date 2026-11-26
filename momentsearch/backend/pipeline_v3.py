import base64
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import faiss

from config import DEFAULT_SAMPLING_FPS, FRAMES_DIR, INDEXES_DIR
import database as db
from embedding import embedding_service
from pipeline_v2 import process_video_task_v2

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")


# ============================================================
# TEMPORAL CLUSTERING & 1D IoU NMS
# ============================================================

def _temporal_iou(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def _encode_image_base64(image_path: Path, max_side: int = 512) -> str:
    with Image.open(image_path) as img:
        rgb = img.convert("RGB")
        w, h = rgb.size
        scale = min(1.0, max_side / max(w, h, 1))
        if scale < 1.0:
            rgb = rgb.resize((max(1, int(w * scale)), max(1, int(h * scale))))
        from io import BytesIO
        buf = BytesIO()
        rgb.save(buf, format="JPEG", quality=80)
        return base64.b64encode(buf.getvalue()).decode("utf-8")


def _load_pil_image(image_path: Path, max_side: int = 512) -> Image.Image:
    with Image.open(image_path) as img:
        rgb = img.convert("RGB")
        w, h = rgb.size
        scale = min(1.0, max_side / max(w, h, 1))
        if scale < 1.0:
            rgb = rgb.resize((max(1, int(w * scale)), max(1, int(h * scale))))
        return rgb.copy()


def generate_candidate_segments_v3(
    video_id: str,
    query: str,
    top_k: int = 5,
    rerank_pool_size: int = 4,
    merge_gap_sec: float = 2.5,
    min_score_floor: float = 0.23,
    sampling_fps: float = 1.0,
) -> List[Dict[str, Any]]:
    index_path = INDEXES_DIR / f"{video_id}_v2.faiss"
    if not index_path.exists():
        raise FileNotFoundError(
            "V2 index not found for this video. Process or re-index with V2 first so V3 can reuse its embeddings."
        )

    index = faiss.read_index(str(index_path))
    records = db.get_frame_records_v2(video_id)
    if not records or index.ntotal == 0:
        return []

    query_vec = embedding_service.encode_text(query)
    scores, indices = index.search(query_vec, index.ntotal)
    clip_map = {int(i): float(s) for i, s in zip(indices[0], scores[0]) if i >= 0}

    scored_frames = []
    for rec in records:
        idx = int(rec["faiss_idx"])
        sim = max(0.0, clip_map.get(idx, 0.0))
        scored_frames.append(
            {
                "timestamp": float(rec["timestamp_sec"]),
                "score": sim,
                "thumbnail_name": rec["thumbnail_name"],
            }
        )

    # 1. Temporal Clustering: smooth + stricter threshold to cut noisy results
    raw = np.array([f["score"] for f in scored_frames])
    smooth = np.convolve(raw, np.ones(3) / 3.0, mode="same")
    thresh = max(float(np.mean(smooth) + 0.85 * np.std(smooth)), min_score_floor)

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

    # Fallback: if threshold is too strict and returns 0 clusters, take top smoothed peaks
    if not clusters:
        top_indices = np.argsort(smooth)[::-1][: min(rerank_pool_size, len(scored_frames))]
        clusters = [[scored_frames[int(i)]] for i in top_indices if smooth[int(i)] > 0]

    frame_step = 1.0 / max(sampling_fps, 0.1)
    proposals: List[Dict[str, Any]] = []

    for cluster in clusters:
        start_t = max(0.0, cluster[0]["timestamp"] - frame_step * 0.5)
        end_t = cluster[-1]["timestamp"] + frame_step * 0.5
        peak = max(cluster, key=lambda x: x["score"])
        confidence = round(min(1.0, peak["score"] * 2.5), 3)

        kf_names = list(
            dict.fromkeys(
                [
                    cluster[0]["thumbnail_name"],
                    peak["thumbnail_name"],
                    cluster[-1]["thumbnail_name"],
                ]
            )
        )
        keyframes = [f"/api/videos/{video_id}/frames/{name}" for name in kf_names]
        review_flag = (
            "verified"
            if confidence >= 0.65
            else ("needs_review" if confidence >= 0.45 else "low_confidence")
        )

        proposals.append(
            {
                "candidate_id": len(proposals) + 1,
                "start_time": round(start_t, 2),
                "end_time": round(end_t, 2),
                "peak_timestamp": round(peak["timestamp"], 2),
                "similarity_score": round(peak["score"], 4),
                "thumbnail_url": f"/api/videos/{video_id}/frames/{peak['thumbnail_name']}",
                "_peak_frame_path": FRAMES_DIR / video_id / peak["thumbnail_name"],
                "confidence": confidence,
                "ai_review_flag": review_flag,
                "detected_objects": [],
                "ocr_text": None,
                "keyframes": keyframes,
                "llm_selected": False,
                "llm_reason": None,
            }
        )

    # 2. Temporal NMS (IoU > 0.5) + hard cap before LLM
    proposals.sort(key=lambda x: x["similarity_score"], reverse=True)
    nms_kept: List[Dict[str, Any]] = []
    for prop in proposals:
        if all(
            _temporal_iou((prop["start_time"], prop["end_time"]), (k["start_time"], k["end_time"])) <= 0.5
            for k in nms_kept
        ):
            nms_kept.append(prop)
        if len(nms_kept) >= max(top_k, rerank_pool_size):
            break

    return nms_kept


# ============================================================
# LLM RERANKER (OLLAMA OR GEMINI) OVER TOP CANDIDATES
# ============================================================

def _build_rerank_prompt(query: str, candidates: List[Dict[str, Any]]) -> str:
    compact = [
        {
            "candidate_id": c["candidate_id"],
            "start_time": c["start_time"],
            "end_time": c["end_time"],
            "peak_timestamp": c["peak_timestamp"],
            "clip_similarity": c["similarity_score"],
        }
        for c in candidates
    ]
    return (
        "You are a video moment verification assistant.\n"
        f"User Search Query: \"{query}\"\n\n"
        "Below are the top candidate video segments retrieved by CLIP + FAISS with their peak frame images in order:\n"
        f"{json.dumps(compact, indent=2)}\n\n"
        "Select which candidate_id best matches the user's query based on the attached peak frames.\n"
        "Return ONLY valid JSON in this exact format:\n"
        "{\n"
        '  "best_candidate_id": 1,\n'
        '  "ranked_ids": [1, 2],\n'
        '  "confidence": 0.85,\n'
        '  "reason": "brief visual explanation"\n'
        "}"
    )

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

async def _rerank_with_ollama(
    query: str,
    candidates: List[Dict[str, Any]],
    model: Optional[str] = None,
    host: Optional[str] = None,
) -> Dict[str, Any]:
    from ollama import AsyncClient

    ollama_host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
    ollama_model = model or os.getenv("OLLAMA_VISION_MODEL", os.getenv("OLLAMA_MODEL", "gemma3:4b"))
    client = AsyncClient(host=ollama_host)

    images_b64 = [
        _encode_image_base64(c["_peak_frame_path"])
        for c in candidates
        if c["_peak_frame_path"].exists()
    ]

    resp = await client.chat(
        model=ollama_model,
        messages=[
            {
                "role": "user",
                "content": _build_rerank_prompt(query, candidates),
                "images": images_b64,
            }
        ],
        format="json",
    )
    content = resp["message"]["content"]
    return json.loads(content)


def _rerank_with_gemini(
    query: str,
    candidates: List[Dict[str, Any]],
    model: Optional[str] = None,
) -> Dict[str, Any]:
    from google import genai
    from google.genai import types

    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key) if api_key else genai.Client()
    gemini_model = model or os.getenv("GEMINI_MODEL", "gemini-3.5-flash")

    contents: List[Any] = [_build_rerank_prompt(query, candidates)]
    for c in candidates:
        p = c["_peak_frame_path"]
        if p.exists():
            contents.append(f"Candidate {c['candidate_id']} ({c['peak_timestamp']}s):")
            contents.append(_load_pil_image(p))

    response = client.models.generate_content(
        model=gemini_model,
        contents=contents,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.1,
        ),
    )
    return json.loads(response.text or "{}")


async def search_video_index_v3(
    *,
    video_id: str,
    query: str,
    top_k: int = 5,
    rerank_top_k: int = 4,
    merge_gap_sec: float = 2.5,
    sampling_fps: float = 1.0,
    llm_provider: str = "ollama",  # "ollama" | "gemini" | "none"
    llm_model: Optional[str] = None,
) -> List[Dict[str, Any]]:
    pool_size = min(max(1, rerank_top_k), 6)  # Keep LLM frame count small (<= 6)
    candidates = generate_candidate_segments_v3(
        video_id=video_id,
        query=query,
        top_k=top_k,
        rerank_pool_size=pool_size,
        merge_gap_sec=merge_gap_sec,
        sampling_fps=sampling_fps,
    )
    if not candidates:
        return []

    rerank_subset = candidates[:pool_size]
    provider = (llm_provider or "none").lower()

    if provider in {"ollama", "gemini"}:
        try:
            verdict = (
                await _rerank_with_ollama(query, rerank_subset, model=llm_model)
                if provider == "ollama"
                else _rerank_with_gemini(query, rerank_subset, model=llm_model)
            )
            best_id = verdict.get("best_candidate_id")
            ranked_ids = [int(x) for x in verdict.get("ranked_ids", []) if str(x).isdigit()]
            llm_conf = verdict.get("confidence")
            reason = verdict.get("reason")

            order_map = {cid: idx for idx, cid in enumerate(ranked_ids)}
            if best_id is not None:
                order_map[int(best_id)] = -1

            def sort_key(item: Dict[str, Any]) -> Tuple[int, float]:
                cid = item["candidate_id"]
                return (order_map.get(cid, 999), -item["similarity_score"])

            rerank_subset.sort(key=sort_key)
            for item in rerank_subset:
                is_best = best_id is not None and int(item["candidate_id"]) == int(best_id)
                item["llm_selected"] = is_best
                item["llm_reason"] = reason if is_best else None
                if is_best and isinstance(llm_conf, (int, float)):
                    item["confidence"] = round(float(llm_conf), 3)
                    item["ai_review_flag"] = (
                        "verified" if float(llm_conf) >= 0.65 else "needs_review"
                    )

            merged_order = rerank_subset + candidates[pool_size:]
        except Exception as exc:
            print(f"[V3 LLM Rerank fallback to CLIP order] {exc}")
            merged_order = candidates
    else:
        merged_order = candidates

    final_results: List[Dict[str, Any]] = []
    for rank, item in enumerate(merged_order[:top_k], start=1):
        clean_item = {k: v for k, v in item.items() if not k.startswith("_")}
        clean_item.pop("candidate_id", None)
        clean_item["rank"] = rank
        final_results.append(clean_item)

    return final_results


process_video_task_v3 = process_video_task_v2