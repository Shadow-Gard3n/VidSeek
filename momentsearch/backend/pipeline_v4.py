import base64
from io import BytesIO
import json
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
from PIL import Image

from config import FRAMES_DIR, INDEXES_DIR
import database as db
from embedding import embedding_service
from pipeline_v2 import process_video_task_v2

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")


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


async def _call_llm_json(
    prompt: str,
    images_paths: Optional[List[Path]] = None,
    llm_provider: str = "ollama",
    llm_model: Optional[str] = None,
) -> Dict[str, Any]:
    provider = (llm_provider or "ollama").lower()
    image_paths = [p for p in (images_paths or []) if p.exists()]

    if provider == "gemini":
        import google.genai as genai
        from google.genai import types

        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is not set.")
        client = genai.Client(api_key=api_key)
        model_name = llm_model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

        contents: List[Any] = [prompt]
        for idx, p in enumerate(image_paths, start=1):
            contents.append(f"Frame {idx}:")
            contents.append(_load_pil_image(p))

        resp = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1,
            ),
        )
        return json.loads(resp.text or "{}")

    from ollama import AsyncClient

    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    model_name = llm_model or os.getenv(
        "OLLAMA_VISION_MODEL", os.getenv("OLLAMA_MODEL", "gemma3:4b")
    )
    client = AsyncClient(host=host)
    msg: Dict[str, Any] = {"role": "user", "content": prompt}
    if image_paths:
        msg["images"] = [_encode_image_base64(p) for p in image_paths]

    resp = await client.chat(model=model_name, messages=[msg], format="json")
    return json.loads(resp["message"]["content"])


def _heuristic_modality_plan(query: str) -> Dict[str, Any]:
    q = query.lower()
    ocr_keywords = (
        "text", "sign", "plate", "number", "timestamp", "time", "date",
        "label", "board", "receipt", "bill", "name", "written", "read", "price",
    )
    yolo_keywords = (
        "person", "man", "woman", "people", "crowd", "car", "bike", "motorcycle",
        "truck", "bus", "bag", "phone", "knife", "weapon", "dog", "animal",
        "chair", "table", "counter", "door", "vehicle", "object",
    )
    use_ocr = any(k in q for k in ocr_keywords)
    use_yolo = any(k in q for k in yolo_keywords)
    return {
        "use_clip": True,  # Always enabled for high recall
        "use_yolo": use_yolo,
        "use_ocr": use_ocr,
        "target_objects": [k for k in yolo_keywords if k in q][:5],
        "target_text_hints": [],
        "reason": "Heuristic routing plan",
    }


async def decide_modalities_v4(
    video_id: str,
    query: str,
    llm_provider: str = "ollama",
    llm_model: Optional[str] = None,
) -> Dict[str, Any]:
    record = db.get_video(video_id) or {}
    preview_frames = sorted((FRAMES_DIR / video_id).glob("frame_*.jpg"))[:2]
    prompt = (
        "You are an ingestion/retrieval planner for a video search system.\n"
        f"Video filename: {record.get('filename', 'unknown')}\n"
        f"Duration (sec): {record.get('duration_sec', 0)}\n"
        f"User Query: \"{query}\"\n\n"
        "CLIP is always active so no important moment is missed.\n"
        "Decide whether YOLO (object detection) or OCR (on-screen text/numbers/signs/timestamps) "
        "is needed to help find timings for this query.\n"
        "Return ONLY valid JSON:\n"
        "{\n"
        '  "use_yolo": true,\n'
        '  "use_ocr": false,\n'
        '  "target_objects": ["person"],\n'
        '  "target_text_hints": [],\n'
        '  "reason": "short reason"\n'
        "}"
    )
    try:
        plan = await _call_llm_json(
            prompt=prompt,
            images_paths=preview_frames,
            llm_provider=llm_provider,
            llm_model=llm_model,
        )
        base = _heuristic_modality_plan(query)
        return {
            "use_clip": True,
            "use_yolo": bool(plan.get("use_yolo", base["use_yolo"])),
            "use_ocr": bool(plan.get("use_ocr", base["use_ocr"])),
            "target_objects": list(plan.get("target_objects") or base["target_objects"]),
            "target_text_hints": list(plan.get("target_text_hints") or []),
            "reason": str(plan.get("reason") or "LLM routing decision"),
        }
    except Exception as exc:
        print(f"[V4 Routing fallback to heuristic] {exc}")
        return _heuristic_modality_plan(query)


def _clip_timings_v4(
    video_id: str,
    records: List[Dict[str, Any]],
    clip_map: Dict[int, float],
    sampling_fps: float,
    merge_gap_sec: float,
    min_score_floor: float = 0.18,  # Recall-safe floor so CLIP doesn't miss subtle moments
) -> List[Dict[str, Any]]:
    scored = []
    for rec in records:
        idx = int(rec["faiss_idx"])
        scored.append(
            {
                "timestamp": float(rec["timestamp_sec"]),
                "score": max(0.0, clip_map.get(idx, 0.0)),
                "thumbnail_name": rec["thumbnail_name"],
            }
        )
    if not scored:
        return []

    raw = np.array([x["score"] for x in scored])
    smooth = np.convolve(raw, np.ones(3) / 3.0, mode="same")
    # Recall-friendly threshold: mean + 0.35*std or min_score_floor
    thresh = max(float(np.mean(smooth) + 0.35 * np.std(smooth)), min_score_floor)

    clusters: List[List[Dict[str, Any]]] = []
    curr: List[Dict[str, Any]] = []
    for i, item in enumerate(scored):
        if smooth[i] >= thresh:
            if curr and (item["timestamp"] - curr[-1]["timestamp"] > merge_gap_sec):
                clusters.append(curr)
                curr = [item]
            else:
                curr.append(item)
    if curr:
        clusters.append(curr)

    # Guarantee top peaks are included even if scores are below threshold
    if len(clusters) < 3:
        top_idx = np.argsort(smooth)[::-1][:4]
        for idx in top_idx:
            if smooth[int(idx)] > 0:
                clusters.append([scored[int(idx)]])

    step = 1.0 / max(sampling_fps, 0.1)
    proposals: List[Dict[str, Any]] = []
    for cl in clusters:
        peak = max(cl, key=lambda x: x["score"])
        proposals.append(
            {
                "source": "clip",
                "start_time": round(max(0.0, cl[0]["timestamp"] - step * 0.5), 2),
                "end_time": round(cl[-1]["timestamp"] + step * 0.5, 2),
                "peak_timestamp": round(peak["timestamp"], 2),
                "similarity_score": round(peak["score"], 4),
                "thumbnail_name": peak["thumbnail_name"],
                "detected_objects": [],
                "ocr_text": None,
            }
        )
    return proposals


def _yolo_timings_v4(
    video_id: str,
    records: List[Dict[str, Any]],
    target_objects: List[str],
    sampling_fps: float,
    merge_gap_sec: float,
) -> List[Dict[str, Any]]:
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[V4 YOLO skipped] ultralytics not installed.")
        return []

    model_path = os.getenv("YOLO_MODEL", "yolov8n.pt")
    model = YOLO(model_path)
    targets = {t.lower().strip() for t in target_objects if t.strip()}
    step = 1.0 / max(sampling_fps, 0.1)
    hits: List[Dict[str, Any]] = []

    for rec in records:
        fp = FRAMES_DIR / video_id / rec["thumbnail_name"]
        if not fp.exists():
            continue
        res = model(str(fp), verbose=False)[0]
        names = res.names or {}
        matched_objs: List[str] = []
        max_conf = 0.0
        for box in res.boxes:
            conf = float(box.conf[0])
            cls_name = str(names.get(int(box.cls[0]), "")).lower()
            if not targets or any(t in cls_name or cls_name in t for t in targets):
                matched_objs.append(cls_name)
                max_conf = max(max_conf, conf)
        if matched_objs:
            hits.append(
                {
                    "timestamp": float(rec["timestamp_sec"]),
                    "score": max_conf,
                    "thumbnail_name": rec["thumbnail_name"],
                    "objs": sorted(set(matched_objs)),
                }
            )

    if not hits:
        return []

    clusters: List[List[Dict[str, Any]]] = [[hits[0]]]
    for h in hits[1:]:
        if h["timestamp"] - clusters[-1][-1]["timestamp"] <= merge_gap_sec:
            clusters[-1].append(h)
        else:
            clusters.append([h])

    proposals: List[Dict[str, Any]] = []
    for cl in clusters:
        peak = max(cl, key=lambda x: x["score"])
        all_objs = sorted({o for item in cl for o in item["objs"]})
        proposals.append(
            {
                "source": "yolo",
                "start_time": round(max(0.0, cl[0]["timestamp"] - step * 0.5), 2),
                "end_time": round(cl[-1]["timestamp"] + step * 0.5, 2),
                "peak_timestamp": round(peak["timestamp"], 2),
                "similarity_score": round(min(1.0, 0.22 + 0.2 * peak["score"]), 4),
                "thumbnail_name": peak["thumbnail_name"],
                "detected_objects": all_objs,
                "ocr_text": None,
            }
        )
    return proposals


def _ocr_timings_v4(
    video_id: str,
    records: List[Dict[str, Any]],
    query: str,
    target_hints: List[str],
    sampling_fps: float,
    merge_gap_sec: float,
) -> List[Dict[str, Any]]:
    try:
        import easyocr
    except ImportError:
        print("[V4 OCR skipped] easyocr not installed.")
        return []

    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    terms = [
        tok.lower()
        for tok in (target_hints + re.findall(r"[a-z0-9]+", query.lower()))
        if len(tok) >= 2
    ]
    step = 1.0 / max(sampling_fps, 0.1)
    hits: List[Dict[str, Any]] = []

    for rec in records[::2]:  # Every 2nd indexed frame for speed
        fp = FRAMES_DIR / video_id / rec["thumbnail_name"]
        if not fp.exists():
            continue
        texts = [str(t).strip() for t in reader.readtext(str(fp), detail=0) if str(t).strip()]
        joined = " ".join(texts).strip()
        if not joined:
            continue
        low = joined.lower()
        score = 0.24 if (not terms or any(term in low for term in terms)) else 0.19
        hits.append(
            {
                "timestamp": float(rec["timestamp_sec"]),
                "score": score,
                "thumbnail_name": rec["thumbnail_name"],
                "text": joined,
            }
        )

    if not hits:
        return []

    clusters: List[List[Dict[str, Any]]] = [[hits[0]]]
    for h in hits[1:]:
        if h["timestamp"] - clusters[-1][-1]["timestamp"] <= max(merge_gap_sec, 3.0):
            clusters[-1].append(h)
        else:
            clusters.append([h])

    proposals: List[Dict[str, Any]] = []
    for cl in clusters:
        peak = max(cl, key=lambda x: x["score"])
        merged_txt = " | ".join(dict.fromkeys(x["text"] for x in cl))[:200]
        proposals.append(
            {
                "source": "ocr",
                "start_time": round(max(0.0, cl[0]["timestamp"] - step), 2),
                "end_time": round(cl[-1]["timestamp"] + step, 2),
                "peak_timestamp": round(peak["timestamp"], 2),
                "similarity_score": round(peak["score"], 4),
                "thumbnail_name": peak["thumbnail_name"],
                "detected_objects": [],
                "ocr_text": merged_txt,
            }
        )
    return proposals


def _ground_and_nms_v4(
    video_id: str,
    proposals: List[Dict[str, Any]],
    merge_gap_sec: float,
    top_k: int,
    rerank_pool_size: int,
) -> List[Dict[str, Any]]:
    if not proposals:
        return []

    # Sort by time to ground/merge overlapping or adjacent multi-modal proposals
    by_time = sorted(proposals, key=lambda x: (x["start_time"], x["end_time"]))
    grounded: List[Dict[str, Any]] = []

    for p in by_time:
        if grounded and (p["start_time"] - grounded[-1]["end_time"] <= merge_gap_sec):
            prev = grounded[-1]
            prev["start_time"] = round(min(prev["start_time"], p["start_time"]), 2)
            prev["end_time"] = round(max(prev["end_time"], p["end_time"]), 2)
            prev_sources = set(prev["_sources"]) | {p["source"]}
            prev["_sources"] = sorted(prev_sources)
            # Boost slightly when multiple modalities (CLIP/YOLO/OCR) agree on a moment
            bonus = 0.03 * (len(prev_sources) - 1)
            if p["similarity_score"] + bonus > prev["similarity_score"]:
                prev["peak_timestamp"] = p["peak_timestamp"]
                prev["thumbnail_name"] = p["thumbnail_name"]
                prev["similarity_score"] = round(min(1.0, p["similarity_score"] + bonus), 4)
            prev["detected_objects"] = sorted(
                set(prev["detected_objects"]) | set(p.get("detected_objects") or [])
            )
            if p.get("ocr_text"):
                prev["ocr_text"] = p["ocr_text"]
        else:
            item = dict(p)
            item["_sources"] = [p["source"]]
            grounded.append(item)

    # 1D Temporal NMS (IoU > 0.5)
    grounded.sort(key=lambda x: x["similarity_score"], reverse=True)
    kept: List[Dict[str, Any]] = []
    for cand in grounded:
        if all(
            _temporal_iou(
                (cand["start_time"], cand["end_time"]),
                (k["start_time"], k["end_time"]),
            )
            <= 0.5
            for k in kept
        ):
            peak_path = FRAMES_DIR / video_id / cand["thumbnail_name"]
            conf = round(min(1.0, cand["similarity_score"] * 2.5), 3)
            kept.append(
                {
                    "candidate_id": len(kept) + 1,
                    "start_time": cand["start_time"],
                    "end_time": cand["end_time"],
                    "peak_timestamp": cand["peak_timestamp"],
                    "similarity_score": cand["similarity_score"],
                    "thumbnail_url": f"/api/videos/{video_id}/frames/{cand['thumbnail_name']}",
                    "_peak_frame_path": peak_path,
                    "_sources": cand["_sources"],
                    "confidence": conf,
                    "ai_review_flag": (
                        "verified"
                        if conf >= 0.65
                        else ("needs_review" if conf >= 0.45 else "low_confidence")
                    ),
                    "detected_objects": cand["detected_objects"],
                    "ocr_text": cand["ocr_text"],
                    "keyframes": [f"/api/videos/{video_id}/frames/{cand['thumbnail_name']}"],
                    "llm_selected": False,
                    "llm_reason": None,
                }
            )
        if len(kept) >= max(top_k, rerank_pool_size):
            break

    return kept


async def search_video_index_v4(
    *,
    video_id: str,
    query: str,
    top_k: int = 5,
    rerank_top_k: int = 4,
    merge_gap_sec: float = 2.5,
    sampling_fps: float = 1.0,
    llm_provider: str = "ollama",
    llm_model: Optional[str] = None,
) -> List[Dict[str, Any]]:
    index_path = INDEXES_DIR / f"{video_id}_v2.faiss"
    if not index_path.exists():
        raise FileNotFoundError(
            "V2/V4 index not found for this video. Index the video first."
        )

    index = faiss.read_index(str(index_path))
    records = db.get_frame_records_v2(video_id)
    if not records or index.ntotal == 0:
        return []

    # Step 1: Initial LLM gate decides whether YOLO / OCR should run alongside CLIP
    plan = await decide_modalities_v4(
        video_id=video_id,
        query=query,
        llm_provider=llm_provider,
        llm_model=llm_model,
    )

    # Step 2: CLIP always runs (high recall); YOLO & OCR run only if selected by routing LLM
    query_vec = embedding_service.encode_text(query)
    scores, indices = index.search(query_vec, index.ntotal)
    clip_map = {int(i): float(s) for i, s in zip(indices[0], scores[0]) if i >= 0}

    proposals: List[Dict[str, Any]] = _clip_timings_v4(
        video_id=video_id,
        records=records,
        clip_map=clip_map,
        sampling_fps=sampling_fps,
        merge_gap_sec=merge_gap_sec,
    )

    if plan["use_yolo"]:
        proposals.extend(
            _yolo_timings_v4(
                video_id=video_id,
                records=records,
                target_objects=plan["target_objects"],
                sampling_fps=sampling_fps,
                merge_gap_sec=merge_gap_sec,
            )
        )

    if plan["use_ocr"]:
        proposals.extend(
            _ocr_timings_v4(
                video_id=video_id,
                records=records,
                query=query,
                target_hints=plan["target_text_hints"],
                sampling_fps=sampling_fps,
                merge_gap_sec=merge_gap_sec,
            )
        )

    # Step 3: Grounding + NMS down to small top candidate set
    pool_size = min(max(1, rerank_top_k), 5)
    candidates = _ground_and_nms_v4(
        video_id=video_id,
        proposals=proposals,
        merge_gap_sec=merge_gap_sec,
        top_k=top_k,
        rerank_pool_size=pool_size,
    )
    if not candidates:
        return []

    # Step 4: Final LLM reranker over top candidate frames
    rerank_subset = candidates[:pool_size]
    provider = (llm_provider or "none").lower()

    if provider in {"ollama", "gemini"}:
        compact = [
            {
                "candidate_id": c["candidate_id"],
                "start_time": c["start_time"],
                "end_time": c["end_time"],
                "peak_timestamp": c["peak_timestamp"],
                "score": c["similarity_score"],
                "modalities": c["_sources"],
                "objects": c["detected_objects"],
                "ocr_text": c["ocr_text"],
            }
            for c in rerank_subset
        ]
        rerank_prompt = (
            "You are the final moment verifier for a video search pipeline.\n"
            f"User Query: \"{query}\"\n"
            f"Routing plan used: CLIP=True, YOLO={plan['use_yolo']}, OCR={plan['use_ocr']}\n\n"
            f"Candidate timings (in order of attached peak frames):\n{json.dumps(compact, indent=2)}\n\n"
            "Pick the best matching candidate_id and return ONLY valid JSON:\n"
            "{\n"
            '  "best_candidate_id": 1,\n'
            '  "ranked_ids": [1, 2],\n'
            '  "confidence": 0.85,\n'
            '  "reason": "concise visual verification explanation"\n'
            "}"
        )
        try:
            verdict = await _call_llm_json(
                prompt=rerank_prompt,
                images_paths=[c["_peak_frame_path"] for c in rerank_subset],
                llm_provider=provider,
                llm_model=llm_model,
            )
            best_id = verdict.get("best_candidate_id")
            ranked_ids = [int(x) for x in verdict.get("ranked_ids", []) if str(x).isdigit()]
            llm_conf = verdict.get("confidence")
            reason = verdict.get("reason")

            order_map = {cid: idx for idx, cid in enumerate(ranked_ids)}
            if best_id is not None:
                order_map[int(best_id)] = -1

            rerank_subset.sort(
                key=lambda item: (order_map.get(item["candidate_id"], 999), -item["similarity_score"])
            )
            for item in rerank_subset:
                is_best = best_id is not None and int(item["candidate_id"]) == int(best_id)
                item["llm_selected"] = is_best
                item["llm_reason"] = reason if is_best else None
                if is_best and isinstance(llm_conf, (int, float)):
                    item["confidence"] = round(float(llm_conf), 3)
                    item["ai_review_flag"] = (
                        "verified" if float(llm_conf) >= 0.65 else "needs_review"
                    )

            merged = rerank_subset + candidates[pool_size:]
        except Exception as exc:
            print(f"[V4 Final LLM Rerank fallback] {exc}")
            merged = candidates
    else:
        merged = candidates

    results: List[Dict[str, Any]] = []
    for rank, item in enumerate(merged[:top_k], start=1):
        clean = {k: v for k, v in item.items() if not k.startswith("_")}
        clean.pop("candidate_id", None)
        clean["rank"] = rank
        results.append(clean)
    return results


async def process_video_task_v4(
    video_id: str,
    video_path: Path,
    sampling_fps: float = 1.0,
) -> None:
    process_video_task_v2(video_id=video_id, video_path=video_path, sampling_fps=sampling_fps)
    record = db.get_video(video_id)
    if record and record.get("status") == "completed":
        db.update_video_status(
            video_id,
            status="completed",
            progress=100.0,
            pipeline="v4",
            error_message=None,
        )