from contextlib import asynccontextmanager
import os
from pathlib import Path
import re
import shutil
from typing import Any, Optional
import uuid

from dotenv import load_dotenv
from fastapi import BackgroundTasks, Body, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

load_dotenv(Path(__file__).resolve().parent / ".env")

# Must execute before importing torch or faiss on macOS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import config as app_config
from config import ALLOWED_EXTENSIONS, DEFAULT_SAMPLING_FPS, FRAMES_DIR, INDEXES_DIR
import database as db
from embedding import embedding_service
from pipeline_v2 import process_video_task_v2
import pipeline_v3 as _p3
from pipeline_v4 import search_video_index_v4
from pipeline_v5 import process_video_task_v5, search_video_index_v5, warmup_v5_models
from pipeline_v6 import process_video_task_v6, search_video_index_v6
from schemas import (
    PipelineType,
    ProcessRequest,
    SearchMatch,
    SearchRequest,
    SearchResponse,
    VideoMetadata,
)
from search_index import search_video_index
from video_processor import process_video_task

UPLOAD_DIR: Path = (
    getattr(app_config, "UPLOAD_DIR", None)
    or getattr(app_config, "UPLOADS_DIR", None)
    or getattr(app_config, "VIDEOS_DIR", None)
    or (FRAMES_DIR.parent / "uploads")
)

rerank_candidates_v3 = getattr(
    _p3,
    "rerank_candidates_v3",
    getattr(_p3, "rerank_video_index_v3", getattr(_p3, "search_video_index_v3", None)),
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init_db()
    embedding_service.load()
    warmup_v5_models()
    yield


app = FastAPI(
    title="MomentSearch / ChronoVision API",
    version="6.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _find_video_file(video_id: str, stored_ext: Optional[str] = None) -> Optional[Path]:
    if stored_ext:
        p = UPLOAD_DIR / f"{video_id}{stored_ext}"
        if p.exists():
            return p
    matches = list(UPLOAD_DIR.glob(f"{video_id}.*"))
    return matches[0] if matches else None


def _run_ingestion_job(
    video_id: str,
    video_path: Path,
    sampling_fps: float,
    pipeline: PipelineType,
) -> None:
    if pipeline == "v6":
        process_video_task_v6(
            video_id=video_id,
            video_path=video_path,
            sampling_fps=sampling_fps,
        )
    elif pipeline == "v5":
        process_video_task_v5(
            video_id=video_id,
            video_path=video_path,
            sampling_fps=sampling_fps,
        )
    elif pipeline in {"v2", "v3", "v4"}:
        process_video_task_v2(
            video_id=video_id,
            video_path=video_path,
            sampling_fps=sampling_fps,
            pipeline=pipeline,
        )
    else:
        process_video_task(
            video_id=video_id,
            video_path=video_path,
            sampling_fps=sampling_fps,
            pipeline="v1",
        )


@app.get("/api/health")
def health_check():
    return {"status": "ok", "version": "6.0.0"}


@app.post("/api/videos/upload", response_model=VideoMetadata)
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    sampling_fps: float = Form(default=DEFAULT_SAMPLING_FPS),
    pipeline: PipelineType = Form(default="v6"),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename.")

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file extension '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    if sampling_fps < 0.25 or sampling_fps > 5.0:
        raise HTTPException(status_code=400, detail="sampling_fps must be between 0.25 and 5.0.")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    video_id = str(uuid.uuid4())
    dest_path = UPLOAD_DIR / f"{video_id}{ext}"

    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    db.create_video(video_id=video_id, filename=file.filename, stored_ext=ext, pipeline=pipeline)
    background_tasks.add_task(_run_ingestion_job, video_id, dest_path, sampling_fps, pipeline)

    record = db.get_video(video_id)
    assert record is not None
    return VideoMetadata(**record)


@app.post("/api/videos/{video_id}/index", response_model=VideoMetadata)
async def trigger_index(
    video_id: str,
    background_tasks: BackgroundTasks,
    payload: Optional[Any] = Body(default=None),
):
    record = db.get_video(video_id)
    if not record:
        raise HTTPException(status_code=404, detail="Video not found.")

    video_path = _find_video_file(video_id, record.get("stored_ext"))
    if not video_path or not video_path.exists():
        raise HTTPException(status_code=404, detail="Source video file missing on disk.")

    if isinstance(payload, dict):
        parsed = ProcessRequest(**payload) if payload else None
    elif isinstance(payload, ProcessRequest):
        parsed = payload
    else:
        parsed = None

    target_pipeline: PipelineType = (
        parsed.pipeline if parsed and parsed.pipeline else record.get("pipeline") or "v6"
    )
    fps = (
        float(parsed.sampling_fps)
        if parsed and parsed.sampling_fps is not None
        else float(record.get("sampling_fps") or DEFAULT_SAMPLING_FPS)
    )

    db.update_video_status(
        video_id=video_id,
        status="processing",
        progress=1.0,
        sampling_fps=fps,
        pipeline=target_pipeline,
        error_message=None,
    )
    background_tasks.add_task(_run_ingestion_job, video_id, video_path, fps, target_pipeline)

    updated = db.get_video(video_id)
    assert updated is not None
    return VideoMetadata(**updated)


@app.post("/api/videos/{video_id}/process", response_model=VideoMetadata)
async def process_video_alias(
    video_id: str,
    background_tasks: BackgroundTasks,
    payload: Optional[Any] = Body(default=None),
):
    return await trigger_index(
        video_id=video_id,
        background_tasks=background_tasks,
        payload=payload,
    )


@app.get("/api/videos", response_model=list[VideoMetadata])
def list_videos():
    rows = db.list_videos()
    return [VideoMetadata(**r) for r in rows]


@app.get("/api/videos/{video_id}/status", response_model=VideoMetadata)
def get_video_status(video_id: str):
    record = db.get_video(video_id)
    if not record:
        raise HTTPException(status_code=404, detail="Video not found.")
    return VideoMetadata(**record)


@app.get("/api/videos/{video_id}/v5-status")
def get_v5_index_status(video_id: str):
    record = db.get_video(video_id)
    if not record:
        raise HTTPException(status_code=404, detail="Video not found.")
    return db.get_index_status_v5(video_id)


@app.get("/api/videos/{video_id}/frames/{frame_name}")
def get_frame_image(video_id: str, frame_name: str):
    if "/" in frame_name or "\\" in frame_name or ".." in frame_name:
        raise HTTPException(status_code=400, detail="Invalid frame name.")
    frame_path = FRAMES_DIR / video_id / frame_name
    if not frame_path.exists():
        raise HTTPException(status_code=404, detail="Frame image not found.")
    return FileResponse(frame_path, media_type="image/jpeg")


@app.get("/api/videos/{video_id}/stream")
async def stream_video(video_id: str, request: Request):
    record = db.get_video(video_id)
    if not record:
        raise HTTPException(status_code=404, detail="Video not found.")

    video_path = _find_video_file(video_id, record.get("stored_ext"))
    if not video_path or not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found on disk.")

    file_size = video_path.stat().st_size
    ext = video_path.suffix.lower()
    mime_map = {
        ".mp4": "video/mp4",
        ".mov": "video/quicktime",
        ".avi": "video/x-msvideo",
        ".mkv": "video/x-matroska",
        ".webm": "video/webm",
    }
    content_type = mime_map.get(ext, "application/octet-stream")
    range_header = request.headers.get("range")

    if not range_header:
        def iter_full():
            with open(video_path, "rb") as f:
                while chunk := f.read(1024 * 1024):
                    yield chunk

        return StreamingResponse(
            iter_full(),
            media_type=content_type,
            headers={"Accept-Ranges": "bytes", "Content-Length": str(file_size)},
        )

    match = re.match(r"bytes=(\d+)-(\d*)", range_header)
    if not match:
        raise HTTPException(status_code=416, detail="Invalid Range header.")

    start = int(match.group(1))
    end = int(match.group(2)) if match.group(2) else file_size - 1
    if start >= file_size or end >= file_size or start > end:
        raise HTTPException(status_code=416, detail="Requested range not satisfiable.")

    chunk_size = end - start + 1

    def iter_range():
        with open(video_path, "rb") as f:
            f.seek(start)
            remaining = chunk_size
            while remaining > 0:
                data = f.read(min(1024 * 1024, remaining))
                if not data:
                    break
                remaining -= len(data)
                yield data

    return StreamingResponse(
        iter_range(),
        status_code=206,
        media_type=content_type,
        headers={
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(chunk_size),
        },
    )


@app.post("/api/videos/{video_id}/search", response_model=SearchResponse)
async def search_video(
    video_id: str,
    payload: SearchRequest,
    pipeline: Optional[PipelineType] = Query(default=None),
):
    record = db.get_video(video_id)
    if not record:
        raise HTTPException(status_code=404, detail="Video not found.")
    if record.get("status") != "completed":
        raise HTTPException(
            status_code=409,
            detail=f"Video is not ready for search (current status: {record.get('status')}).",
        )

    active_pipeline: PipelineType = pipeline or payload.pipeline or record.get("pipeline") or "v6"
    vid = video_id
    rerank_enabled = getattr(payload, "rerank", True)

    if active_pipeline == "v6":
        try:
            matches = await search_video_index_v6(
                video_id=vid,
                query=payload.query,
                top_k=payload.top_k,
                merge_gap_sec=payload.merge_gap_sec,
                llm_provider=payload.llm_provider,
                llm_model=payload.llm_model,
                rerank=rerank_enabled,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
    elif active_pipeline == "v5":
        try:
            matches = await search_video_index_v5(
                video_id=vid,
                query=payload.query,
                top_k=payload.top_k,
                merge_gap_sec=payload.merge_gap_sec,
                llm_provider=payload.llm_provider,
                llm_model=payload.llm_model,
                rerank=rerank_enabled,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
    elif active_pipeline == "v4":
        try:
            matches = await search_video_index_v4(
                video_id=vid,
                query=payload.query,
                top_k=payload.top_k,
                merge_gap_sec=payload.merge_gap_sec,
                llm_provider=payload.llm_provider,
                llm_model=payload.llm_model,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
    elif active_pipeline == "v3":
        if rerank_candidates_v3 is None:
            raise HTTPException(
                status_code=500,
                detail="V3 reranker function not found in pipeline_v3.py.",
            )
        v2_idx = INDEXES_DIR / f"{vid}_v2.faiss"
        v1_idx = INDEXES_DIR / f"{vid}.faiss"
        if not v2_idx.exists() and not v1_idx.exists():
            raise HTTPException(status_code=409, detail="No visual index found for this video.")
        raw_candidates = search_video_index(
            video_id=vid,
            query=payload.query,
            top_k=max(payload.top_k * 2, 8),
            merge_gap_sec=payload.merge_gap_sec,
            pipeline="v2" if v2_idx.exists() else "v1",
        )
        matches = await rerank_candidates_v3(
            video_id=vid,
            query=payload.query,
            candidates=raw_candidates,
            top_k=payload.top_k,
            llm_provider=payload.llm_provider,
            llm_model=payload.llm_model,
        )
    elif active_pipeline == "v2":
        v2_idx = INDEXES_DIR / f"{vid}_v2.faiss"
        if not v2_idx.exists():
            raise HTTPException(status_code=409, detail="V2 index not found. Re-index video with v2.")
        matches = search_video_index(
            video_id=vid,
            query=payload.query,
            top_k=payload.top_k,
            merge_gap_sec=payload.merge_gap_sec,
            pipeline="v2",
        )
    else:
        v1_idx = INDEXES_DIR / f"{vid}.faiss"
        if not v1_idx.exists():
            raise HTTPException(status_code=409, detail="V1 index not found. Re-index video with v1.")
        matches = search_video_index(
            video_id=vid,
            query=payload.query,
            top_k=payload.top_k,
            merge_gap_sec=payload.merge_gap_sec,
            pipeline="v1",
        )

    return SearchResponse(
        video_id=vid,
        query=payload.query,
        pipeline=active_pipeline,
        results=[SearchMatch(**m) for m in matches],
        disclaimer="AI-generated temporal localization; verify critical events directly in the video stream.",
    )

    

# from contextlib import asynccontextmanager
# import os
# from pathlib import Path
# import re
# import shutil
# from typing import Any, Optional
# import uuid

# from dotenv import load_dotenv
# from fastapi import BackgroundTasks, Body, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import FileResponse, StreamingResponse

# load_dotenv(Path(__file__).resolve().parent / ".env")

# # Must execute before importing torch or faiss on macOS
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# os.environ["OMP_NUM_THREADS"] = "1"

# import config as app_config
# from config import ALLOWED_EXTENSIONS, DEFAULT_SAMPLING_FPS, FRAMES_DIR, INDEXES_DIR
# import database as db
# from embedding import embedding_service
# from pipeline_v2 import process_video_task_v2
# import pipeline_v3 as _p3
# from pipeline_v4 import search_video_index_v4
# from pipeline_v5 import process_video_task_v5, search_video_index_v5, warmup_v5_models
# from schemas import (
#     PipelineType,
#     ProcessRequest,
#     SearchMatch,
#     SearchRequest,
#     SearchResponse,
#     VideoMetadata,
# )
# from search_index import search_video_index
# from video_processor import process_video_task

# UPLOAD_DIR: Path = (
#     getattr(app_config, "UPLOAD_DIR", None)
#     or getattr(app_config, "UPLOADS_DIR", None)
#     or getattr(app_config, "VIDEOS_DIR", None)
#     or (FRAMES_DIR.parent / "uploads")
# )

# rerank_candidates_v3 = getattr(
#     _p3,
#     "rerank_candidates_v3",
#     getattr(_p3, "rerank_video_index_v3", getattr(_p3, "search_video_index_v3", None)),
# )


# @asynccontextmanager
# async def lifespan(_: FastAPI):
#     db.init_db()
#     embedding_service.load()
#     warmup_v5_models()
#     yield


# app = FastAPI(
#     title="MomentSearch / ChronoVision API",
#     version="5.0.0",
#     lifespan=lifespan,
# )

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# def _find_video_file(video_id: str, stored_ext: Optional[str] = None) -> Optional[Path]:
#     if stored_ext:
#         p = UPLOAD_DIR / f"{video_id}{stored_ext}"
#         if p.exists():
#             return p
#     matches = list(UPLOAD_DIR.glob(f"{video_id}.*"))
#     return matches[0] if matches else None


# def _run_ingestion_job(
#     video_id: str,
#     video_path: Path,
#     sampling_fps: float,
#     pipeline: PipelineType,
# ) -> None:
#     if pipeline == "v5":
#         process_video_task_v5(video_id=video_id, video_path=video_path, sampling_fps=sampling_fps)
#     elif pipeline in {"v2", "v3", "v4"}:
#         process_video_task_v2(
#             video_id=video_id,
#             video_path=video_path,
#             sampling_fps=sampling_fps,
#             pipeline=pipeline,
#         )
#     else:
#         process_video_task(
#             video_id=video_id,
#             video_path=video_path,
#             sampling_fps=sampling_fps,
#             pipeline="v1",
#         )


# @app.get("/api/health")
# def health_check():
#     return {"status": "ok", "version": "5.0.0"}

# @app.post("/api/videos/{video_id}/process", response_model=VideoMetadata)
# async def process_video_alias(
#     video_id: str,
#     background_tasks: BackgroundTasks,
#     payload: Optional[Any] = Body(default=None),
# ):
#     return await trigger_index(
#         video_id=video_id,
#         background_tasks=background_tasks,
#         payload=payload,
#     )


# @app.post("/api/videos/upload", response_model=VideoMetadata)
# async def upload_video(
#     background_tasks: BackgroundTasks,
#     file: UploadFile = File(...),
#     sampling_fps: float = Form(default=DEFAULT_SAMPLING_FPS),
#     pipeline: PipelineType = Form(default="v5"),
# ):
#     if not file.filename:
#         raise HTTPException(status_code=400, detail="Missing filename.")

#     ext = Path(file.filename).suffix.lower()
#     if ext not in ALLOWED_EXTENSIONS:
#         raise HTTPException(
#             status_code=400,
#             detail=f"Unsupported file extension '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
#         )

#     if sampling_fps < 0.25 or sampling_fps > 5.0:
#         raise HTTPException(status_code=400, detail="sampling_fps must be between 0.25 and 5.0.")

#     UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
#     video_id = str(uuid.uuid4())
#     dest_path = UPLOAD_DIR / f"{video_id}{ext}"

#     with open(dest_path, "wb") as buffer:
#         shutil.copyfileobj(file.file, buffer)

#     db.create_video(video_id=video_id, filename=file.filename, stored_ext=ext, pipeline=pipeline)
#     background_tasks.add_task(_run_ingestion_job, video_id, dest_path, sampling_fps, pipeline)

#     record = db.get_video(video_id)
#     assert record is not None
#     return VideoMetadata(**record)


# @app.post("/api/videos/{video_id}/index", response_model=VideoMetadata)
# async def trigger_index(
#     video_id: str,
#     background_tasks: BackgroundTasks,
#     payload: Optional[Any] = Body(default=None),
# ):
#     record = db.get_video(video_id)
#     if not record:
#         raise HTTPException(status_code=404, detail="Video not found.")

#     video_path = _find_video_file(video_id, record.get("stored_ext"))
#     if not video_path or not video_path.exists():
#         raise HTTPException(status_code=404, detail="Source video file missing on disk.")

#     if isinstance(payload, dict):
#         parsed = ProcessRequest(**payload) if payload else None
#     elif isinstance(payload, ProcessRequest):
#         parsed = payload
#     else:
#         parsed = None

#     target_pipeline: PipelineType = (
#         parsed.pipeline if parsed and parsed.pipeline else record.get("pipeline") or "v5"
#     )
#     fps = (
#         float(parsed.sampling_fps)
#         if parsed and parsed.sampling_fps is not None
#         else float(record.get("sampling_fps") or DEFAULT_SAMPLING_FPS)
#     )

#     db.update_video_status(
#         video_id=video_id,
#         status="processing",
#         progress=1.0,
#         sampling_fps=fps,
#         pipeline=target_pipeline,
#         error_message=None,
#     )
#     background_tasks.add_task(_run_ingestion_job, video_id, video_path, fps, target_pipeline)

#     updated = db.get_video(video_id)
#     assert updated is not None
#     return VideoMetadata(**updated)


# @app.get("/api/videos", response_model=list[VideoMetadata])
# def list_videos():
#     rows = db.list_videos()
#     return [VideoMetadata(**r) for r in rows]


# @app.get("/api/videos/{video_id}/status", response_model=VideoMetadata)
# def get_video_status(video_id: str):
#     record = db.get_video(video_id)
#     if not record:
#         raise HTTPException(status_code=404, detail="Video not found.")
#     return VideoMetadata(**record)


# @app.get("/api/videos/{video_id}/v5-status")
# def get_v5_index_status(video_id: str):
#     record = db.get_video(video_id)
#     if not record:
#         raise HTTPException(status_code=404, detail="Video not found.")
#     return db.get_index_status_v5(video_id)


# @app.get("/api/videos/{video_id}/frames/{frame_name}")
# def get_frame_image(video_id: str, frame_name: str):
#     if "/" in frame_name or "\\" in frame_name or ".." in frame_name:
#         raise HTTPException(status_code=400, detail="Invalid frame name.")
#     frame_path = FRAMES_DIR / video_id / frame_name
#     if not frame_path.exists():
#         raise HTTPException(status_code=404, detail="Frame image not found.")
#     return FileResponse(frame_path, media_type="image/jpeg")


# @app.get("/api/videos/{video_id}/stream")
# async def stream_video(video_id: str, request: Request):
#     record = db.get_video(video_id)
#     if not record:
#         raise HTTPException(status_code=404, detail="Video not found.")

#     video_path = _find_video_file(video_id, record.get("stored_ext"))
#     if not video_path or not video_path.exists():
#         raise HTTPException(status_code=404, detail="Video file not found on disk.")

#     file_size = video_path.stat().st_size
#     ext = video_path.suffix.lower()
#     mime_map = {
#         ".mp4": "video/mp4",
#         ".mov": "video/quicktime",
#         ".avi": "video/x-msvideo",
#         ".mkv": "video/x-matroska",
#         ".webm": "video/webm",
#     }
#     content_type = mime_map.get(ext, "application/octet-stream")
#     range_header = request.headers.get("range")

#     if not range_header:
#         def iter_full():
#             with open(video_path, "rb") as f:
#                 while chunk := f.read(1024 * 1024):
#                     yield chunk

#         return StreamingResponse(
#             iter_full(),
#             media_type=content_type,
#             headers={"Accept-Ranges": "bytes", "Content-Length": str(file_size)},
#         )

#     match = re.match(r"bytes=(\d+)-(\d*)", range_header)
#     if not match:
#         raise HTTPException(status_code=416, detail="Invalid Range header.")

#     start = int(match.group(1))
#     end = int(match.group(2)) if match.group(2) else file_size - 1
#     if start >= file_size or end >= file_size or start > end:
#         raise HTTPException(status_code=416, detail="Requested range not satisfiable.")

#     chunk_size = end - start + 1

#     def iter_range():
#         with open(video_path, "rb") as f:
#             f.seek(start)
#             remaining = chunk_size
#             while remaining > 0:
#                 data = f.read(min(1024 * 1024, remaining))
#                 if not data:
#                     break
#                 remaining -= len(data)
#                 yield data

#     return StreamingResponse(
#         iter_range(),
#         status_code=206,
#         media_type=content_type,
#         headers={
#             "Content-Range": f"bytes {start}-{end}/{file_size}",
#             "Accept-Ranges": "bytes",
#             "Content-Length": str(chunk_size),
#         },
#     )


# @app.post("/api/videos/{video_id}/search", response_model=SearchResponse)
# async def search_video(
#     video_id: str,
#     payload: SearchRequest,
#     pipeline: Optional[PipelineType] = Query(default=None),
# ):
#     record = db.get_video(video_id)
#     if not record:
#         raise HTTPException(status_code=404, detail="Video not found.")
#     if record.get("status") != "completed":
#         raise HTTPException(
#             status_code=409,
#             detail=f"Video is not ready for search (current status: {record.get('status')}).",
#         )

#     active_pipeline: PipelineType = pipeline or payload.pipeline or record.get("pipeline") or "v5"
#     vid = video_id
#     rerank_enabled = getattr(payload, "rerank", True)

#     if active_pipeline == "v5":
#         try:
#             matches = await search_video_index_v5(
#                 video_id=vid,
#                 query=payload.query,
#                 top_k=payload.top_k,
#                 merge_gap_sec=payload.merge_gap_sec,
#                 llm_provider=payload.llm_provider,
#                 llm_model=payload.llm_model,
#                 rerank=rerank_enabled,
#             )
#         except FileNotFoundError as exc:
#             raise HTTPException(status_code=409, detail=str(exc))
#     elif active_pipeline == "v4":
#         try:
#             matches = await search_video_index_v4(
#                 video_id=vid,
#                 query=payload.query,
#                 top_k=payload.top_k,
#                 merge_gap_sec=payload.merge_gap_sec,
#                 llm_provider=payload.llm_provider,
#                 llm_model=payload.llm_model,
#             )
#         except FileNotFoundError as exc:
#             raise HTTPException(status_code=409, detail=str(exc))
#     elif active_pipeline == "v3":
#         if rerank_candidates_v3 is None:
#             raise HTTPException(
#                 status_code=500,
#                 detail="V3 reranker function not found in pipeline_v3.py.",
#             )
#         v2_idx = INDEXES_DIR / f"{vid}_v2.faiss"
#         v1_idx = INDEXES_DIR / f"{vid}.faiss"
#         if not v2_idx.exists() and not v1_idx.exists():
#             raise HTTPException(status_code=409, detail="No visual index found for this video.")
#         raw_candidates = search_video_index(
#             video_id=vid,
#             query=payload.query,
#             top_k=max(payload.top_k * 2, 8),
#             merge_gap_sec=payload.merge_gap_sec,
#             pipeline="v2" if v2_idx.exists() else "v1",
#         )
#         matches = await rerank_candidates_v3(
#             video_id=vid,
#             query=payload.query,
#             candidates=raw_candidates,
#             top_k=payload.top_k,
#             llm_provider=payload.llm_provider,
#             llm_model=payload.llm_model,
#         )
#     elif active_pipeline == "v2":
#         v2_idx = INDEXES_DIR / f"{vid}_v2.faiss"
#         if not v2_idx.exists():
#             raise HTTPException(status_code=409, detail="V2 index not found. Re-index video with v2.")
#         matches = search_video_index(
#             video_id=vid,
#             query=payload.query,
#             top_k=payload.top_k,
#             merge_gap_sec=payload.merge_gap_sec,
#             pipeline="v2",
#         )
#     else:
#         v1_idx = INDEXES_DIR / f"{vid}.faiss"
#         if not v1_idx.exists():
#             raise HTTPException(status_code=409, detail="V1 index not found. Re-index video with v1.")
#         matches = search_video_index(
#             video_id=vid,
#             query=payload.query,
#             top_k=payload.top_k,
#             merge_gap_sec=payload.merge_gap_sec,
#             pipeline="v1",
#         )

#     return SearchResponse(
#         video_id=vid,
#         query=payload.query,
#         pipeline=active_pipeline,
#         results=[SearchMatch(**m) for m in matches],
#         disclaimer="AI-generated temporal localization; verify critical events directly in the video stream.",
#     )

# import os

# from dotenv import load_dotenv
# from pathlib import Path


# load_dotenv(Path(__file__).resolve().parent / ".env")

# # Must execute before importing torch or faiss on macOS
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# os.environ["OMP_NUM_THREADS"] = "1"

# from contextlib import asynccontextmanager
# from typing import List
# import uuid

# from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import FileResponse, StreamingResponse

# from config import (
#     ALLOWED_EXTENSIONS,
#     DEFAULT_SAMPLING_FPS,
#     FRAMES_DIR,
#     MAX_UPLOAD_BYTES,
#     VIDEOS_DIR,
# )
# import database as db
# from embedding import embedding_service
# from pipeline_v2 import process_video_task_v2, search_video_index_v2
# from pipeline_v3 import process_video_task_v3, search_video_index_v3
# from pipeline_v4 import process_video_task_v4, search_video_index_v4
# from pipeline_v5 import process_video_task_v5, search_video_index_v5
# from schemas import ProcessRequest, SearchRequest, SearchResponse, VideoMetadata
# from search_index import search_video_index
# from video_processor import process_video_task

# VALID_PIPELINES = {"v1", "v2", "v3", "v4", "v5"}


# @asynccontextmanager
# async def lifespan(_: FastAPI):
#     db.init_db()
#     embedding_service.load()
#     yield


# app = FastAPI(title="MomentSearch API", version="5.0.0", lifespan=lifespan)

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# def validate_uuid(video_id: str) -> str:
#     try:
#         return str(uuid.UUID(video_id))
#     except ValueError:
#         raise HTTPException(status_code=400, detail="Invalid video ID format.")


# def _get_processor_for_pipeline(pipeline: str):
#     if pipeline == "v5":
#         return process_video_task_v5
#     if pipeline == "v4":
#         return process_video_task_v4
#     if pipeline == "v3":
#         return process_video_task_v3
#     if pipeline == "v2":
#         return process_video_task_v2
#     return process_video_task


# @app.post("/api/videos/upload", response_model=VideoMetadata)
# async def upload_video(
#     background_tasks: BackgroundTasks,
#     file: UploadFile = File(...),
#     pipeline: str = Form("v1"),
# ):
#     if pipeline not in VALID_PIPELINES:
#         raise HTTPException(
#             status_code=400,
#             detail=f"Invalid pipeline '{pipeline}'. Allowed: v1, v2, v3, v4, v5.",
#         )
#     if not file.filename:
#         raise HTTPException(status_code=400, detail="Missing filename.")

#     ext = Path(file.filename).suffix.lower()
#     if ext not in ALLOWED_EXTENSIONS:
#         raise HTTPException(
#             status_code=400,
#             detail=f"Unsupported file format '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
#         )

#     video_id = str(uuid.uuid4())
#     dest_path = VIDEOS_DIR / f"{video_id}{ext}"

#     bytes_written = 0
#     try:
#         with open(dest_path, "wb") as out_file:
#             while chunk := await file.read(1024 * 1024):
#                 bytes_written += len(chunk)
#                 if bytes_written > MAX_UPLOAD_BYTES:
#                     out_file.close()
#                     dest_path.unlink(missing_ok=True)
#                     raise HTTPException(status_code=413, detail="File exceeds 500 MB upload limit.")
#                 out_file.write(chunk)
#     except HTTPException:
#         raise
#     except Exception as exc:
#         dest_path.unlink(missing_ok=True)
#         raise HTTPException(status_code=500, detail=f"Upload failed: {exc}")

#     db.create_video(video_id=video_id, filename=file.filename, stored_ext=ext, pipeline=pipeline)
#     task_fn = _get_processor_for_pipeline(pipeline)
#     background_tasks.add_task(task_fn, video_id, dest_path, DEFAULT_SAMPLING_FPS)

#     record = db.get_video(video_id)
#     assert record is not None
#     return record


# @app.get("/api/videos", response_model=List[VideoMetadata])
# def get_videos():
#     return db.list_videos()


# @app.get("/api/videos/{video_id}", response_model=VideoMetadata)
# def get_video_metadata(video_id: str):
#     vid = validate_uuid(video_id)
#     record = db.get_video(vid)
#     if not record:
#         raise HTTPException(status_code=404, detail="Video not found.")
#     return record


# @app.post("/api/videos/{video_id}/process", response_model=VideoMetadata)
# def start_video_processing(
#     video_id: str,
#     background_tasks: BackgroundTasks,
#     payload: ProcessRequest = ProcessRequest(),
# ):
#     vid = validate_uuid(video_id)
#     record = db.get_video(vid)
#     if not record:
#         raise HTTPException(status_code=404, detail="Video not found.")
#     if payload.pipeline not in VALID_PIPELINES:
#         raise HTTPException(
#             status_code=400,
#             detail=f"Invalid pipeline '{payload.pipeline}'. Allowed: v1, v2, v3, v4, v5.",
#         )

#     video_path = VIDEOS_DIR / f"{vid}{record['stored_ext']}"
#     if not video_path.exists():
#         raise HTTPException(status_code=404, detail="Video file missing from storage.")

#     db.update_video_status(
#         vid,
#         status="processing",
#         progress=0.0,
#         pipeline=payload.pipeline,
#         error_message=None,
#     )
#     task_fn = _get_processor_for_pipeline(payload.pipeline)
#     background_tasks.add_task(task_fn, vid, video_path, payload.sampling_fps)

#     updated = db.get_video(vid)
#     assert updated is not None
#     return updated


# @app.get("/api/videos/{video_id}/status", response_model=VideoMetadata)
# def get_video_status(video_id: str):
#     vid = validate_uuid(video_id)
#     record = db.get_video(vid)
#     if not record:
#         raise HTTPException(status_code=404, detail="Video not found.")
#     return record


# @app.post("/api/videos/{video_id}/search", response_model=SearchResponse)
# async def search_video(video_id: str, payload: SearchRequest):
#     vid = validate_uuid(video_id)
#     record = db.get_video(vid)
#     if not record:
#         raise HTTPException(status_code=404, detail="Video not found.")
#     if record["status"] != "completed":
#         raise HTTPException(
#             status_code=409,
#             detail=f"Video is not ready for search (status: {record['status']}).",
#         )

#     active_pipeline = payload.pipeline or record.get("pipeline") or "v1"
#     if active_pipeline not in VALID_PIPELINES:
#         raise HTTPException(status_code=400, detail=f"Invalid pipeline '{active_pipeline}'.")

#     sampling_fps = float(record.get("sampling_fps") or DEFAULT_SAMPLING_FPS)

#     if active_pipeline == "v5":
#         try:
#             matches = await search_video_index_v5(
#                 video_id=vid,
#                 query=payload.query,
#                 top_k=payload.top_k,
#                 merge_gap_sec=payload.merge_gap_sec,
#                 llm_provider=payload.llm_provider,
#                 llm_model=payload.llm_model,
#             )
#         except FileNotFoundError as exc:
#             raise HTTPException(status_code=409, detail=str(exc))
#     elif active_pipeline == "v4":
#         try:
#             matches = await search_video_index_v4(
#                 video_id=vid,
#                 query=payload.query,
#                 top_k=payload.top_k,
#                 rerank_top_k=min(payload.top_k, 4),
#                 merge_gap_sec=payload.merge_gap_sec,
#                 sampling_fps=sampling_fps,
#                 llm_provider=payload.llm_provider,
#                 llm_model=payload.llm_model,
#             )
#         except FileNotFoundError as exc:
#             raise HTTPException(status_code=409, detail=str(exc))
#     elif active_pipeline == "v3":
#         try:
#             matches = await search_video_index_v3(
#                 video_id=vid,
#                 query=payload.query,
#                 top_k=payload.top_k,
#                 rerank_top_k=min(payload.top_k, 4),
#                 merge_gap_sec=payload.merge_gap_sec,
#                 sampling_fps=sampling_fps,
#                 llm_provider=payload.llm_provider,
#                 llm_model=payload.llm_model,
#             )
#         except FileNotFoundError as exc:
#             raise HTTPException(status_code=409, detail=str(exc))
#     elif active_pipeline == "v2":
#         try:
#             matches = search_video_index_v2(
#                 video_id=vid,
#                 query=payload.query,
#                 top_k=payload.top_k,
#                 merge_gap_sec=payload.merge_gap_sec,
#                 sampling_fps=sampling_fps,
#             )
#         except FileNotFoundError as exc:
#             raise HTTPException(status_code=409, detail=str(exc))
#     else:
#         query_vec = embedding_service.encode_text(payload.query)
#         matches = search_video_index(
#             video_id=vid,
#             query_vec=query_vec,
#             top_k=payload.top_k,
#             merge_gap_sec=payload.merge_gap_sec,
#             sampling_fps=sampling_fps,
#         )

#     return SearchResponse(
#         video_id=vid,
#         query=payload.query,
#         pipeline=active_pipeline,
#         disclaimer="Retrieval scores indicate visual/multimodal similarity, not proof that an incident occurred. Human review is required.",
#         results=matches,
#     )


# @app.get("/api/videos/{video_id}/frames/{frame_name}")
# def get_frame_thumbnail(video_id: str, frame_name: str):
#     vid = validate_uuid(video_id)
#     safe_name = Path(frame_name).name
#     if safe_name != frame_name or not safe_name.endswith(".jpg"):
#         raise HTTPException(status_code=400, detail="Invalid frame filename.")

#     frame_path = FRAMES_DIR / vid / safe_name
#     if not frame_path.exists():
#         raise HTTPException(status_code=404, detail="Frame thumbnail not found.")
#     return FileResponse(frame_path, media_type="image/jpeg")


# @app.get("/api/videos/{video_id}/stream")
# def stream_video(video_id: str, request: Request):
#     vid = validate_uuid(video_id)
#     record = db.get_video(vid)
#     if not record:
#         raise HTTPException(status_code=404, detail="Video not found.")

#     video_path = VIDEOS_DIR / f"{vid}{record['stored_ext']}"
#     if not video_path.exists():
#         raise HTTPException(status_code=404, detail="Video file not found.")

#     file_size = video_path.stat().st_size
#     range_header = request.headers.get("range")
#     media_type = "video/quicktime" if record["stored_ext"] == ".mov" else "video/mp4"

#     if not range_header:
#         return FileResponse(video_path, media_type=media_type)

#     range_val = range_header.strip().replace("bytes=", "")
#     start_str, end_str = range_val.split("-")
#     start = int(start_str) if start_str else 0
#     end = int(end_str) if end_str else file_size - 1
#     end = min(end, file_size - 1)

#     if start >= file_size or start > end:
#         raise HTTPException(status_code=416, detail="Requested range not satisfiable.")

#     chunk_length = end - start + 1

#     def iter_file(path: Path, offset: int, length: int, chunk_size: int = 1024 * 1024):
#         with open(path, "rb") as f:
#             f.seek(offset)
#             remaining = length
#             while remaining > 0:
#                 data = f.read(min(chunk_size, remaining))
#                 if not data:
#                     break
#                 remaining -= len(data)
#                 yield data

#     return StreamingResponse(
#         iter_file(video_path, start, chunk_length),
#         status_code=206,
#         headers={
#             "Content-Range": f"bytes {start}-{end}/{file_size}",
#             "Accept-Ranges": "bytes",
#             "Content-Length": str(chunk_length),
#         },
#         media_type=media_type,
#     )

# import os

# # Must execute before importing torch or faiss on macOS
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# os.environ["OMP_NUM_THREADS"] = "1"

# from contextlib import asynccontextmanager
# from pathlib import Path
# from typing import List
# import uuid

# from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import FileResponse, StreamingResponse

# from config import (
#     ALLOWED_EXTENSIONS,
#     DEFAULT_SAMPLING_FPS,
#     FRAMES_DIR,
#     MAX_UPLOAD_BYTES,
#     VIDEOS_DIR,
# )
# import database as db
# from embedding import embedding_service
# from pipeline_v2 import process_video_task_v2, search_video_index_v2
# from pipeline_v3 import process_video_task_v3, search_video_index_v3
# from pipeline_v4 import process_video_task_v4, search_video_index_v4
# from schemas import ProcessRequest, SearchRequest, SearchResponse, VideoMetadata
# from search_index import search_video_index
# from video_processor import process_video_task

# VALID_PIPELINES = {"v1", "v2", "v3", "v4"}


# @asynccontextmanager
# async def lifespan(_: FastAPI):
#     db.init_db()
#     embedding_service.load()
#     yield


# app = FastAPI(title="MomentSearch API", version="4.0.0", lifespan=lifespan)

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# def validate_uuid(video_id: str) -> str:
#     try:
#         return str(uuid.UUID(video_id))
#     except ValueError:
#         raise HTTPException(status_code=400, detail="Invalid video ID format.")


# def _get_processor_for_pipeline(pipeline: str):
#     if pipeline == "v4":
#         return process_video_task_v4
#     if pipeline == "v3":
#         return process_video_task_v3
#     if pipeline == "v2":
#         return process_video_task_v2
#     return process_video_task


# @app.post("/api/videos/upload", response_model=VideoMetadata)
# async def upload_video(
#     background_tasks: BackgroundTasks,
#     file: UploadFile = File(...),
#     pipeline: str = Form("v1"),
# ):
#     if pipeline not in VALID_PIPELINES:
#         raise HTTPException(
#             status_code=400,
#             detail=f"Invalid pipeline '{pipeline}'. Allowed: v1, v2, v3, v4.",
#         )
#     if not file.filename:
#         raise HTTPException(status_code=400, detail="Missing filename.")

#     ext = Path(file.filename).suffix.lower()
#     if ext not in ALLOWED_EXTENSIONS:
#         raise HTTPException(
#             status_code=400,
#             detail=f"Unsupported file format '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
#         )

#     video_id = str(uuid.uuid4())
#     dest_path = VIDEOS_DIR / f"{video_id}{ext}"

#     bytes_written = 0
#     try:
#         with open(dest_path, "wb") as out_file:
#             while chunk := await file.read(1024 * 1024):
#                 bytes_written += len(chunk)
#                 if bytes_written > MAX_UPLOAD_BYTES:
#                     out_file.close()
#                     dest_path.unlink(missing_ok=True)
#                     raise HTTPException(status_code=413, detail="File exceeds 500 MB upload limit.")
#                 out_file.write(chunk)
#     except HTTPException:
#         raise
#     except Exception as exc:
#         dest_path.unlink(missing_ok=True)
#         raise HTTPException(status_code=500, detail=f"Upload failed: {exc}")

#     db.create_video(video_id=video_id, filename=file.filename, stored_ext=ext, pipeline=pipeline)
#     task_fn = _get_processor_for_pipeline(pipeline)
#     background_tasks.add_task(task_fn, video_id, dest_path, DEFAULT_SAMPLING_FPS)

#     record = db.get_video(video_id)
#     assert record is not None
#     return record


# @app.get("/api/videos", response_model=List[VideoMetadata])
# def get_videos():
#     return db.list_videos()


# @app.get("/api/videos/{video_id}", response_model=VideoMetadata)
# def get_video_metadata(video_id: str):
#     vid = validate_uuid(video_id)
#     record = db.get_video(vid)
#     if not record:
#         raise HTTPException(status_code=404, detail="Video not found.")
#     return record


# @app.post("/api/videos/{video_id}/process", response_model=VideoMetadata)
# def start_video_processing(
#     video_id: str,
#     background_tasks: BackgroundTasks,
#     payload: ProcessRequest = ProcessRequest(),
# ):
#     vid = validate_uuid(video_id)
#     record = db.get_video(vid)
#     if not record:
#         raise HTTPException(status_code=404, detail="Video not found.")
#     if payload.pipeline not in VALID_PIPELINES:
#         raise HTTPException(
#             status_code=400,
#             detail=f"Invalid pipeline '{payload.pipeline}'. Allowed: v1, v2, v3, v4.",
#         )

#     video_path = VIDEOS_DIR / f"{vid}{record['stored_ext']}"
#     if not video_path.exists():
#         raise HTTPException(status_code=404, detail="Video file missing from storage.")

#     db.update_video_status(
#         vid,
#         status="processing",
#         progress=0.0,
#         pipeline=payload.pipeline,
#         error_message=None,
#     )
#     task_fn = _get_processor_for_pipeline(payload.pipeline)
#     background_tasks.add_task(task_fn, vid, video_path, payload.sampling_fps)

#     updated = db.get_video(vid)
#     assert updated is not None
#     return updated


# @app.get("/api/videos/{video_id}/status", response_model=VideoMetadata)
# def get_video_status(video_id: str):
#     vid = validate_uuid(video_id)
#     record = db.get_video(vid)
#     if not record:
#         raise HTTPException(status_code=404, detail="Video not found.")
#     return record


# @app.post("/api/videos/{video_id}/search", response_model=SearchResponse)
# async def search_video(video_id: str, payload: SearchRequest):
#     vid = validate_uuid(video_id)
#     record = db.get_video(vid)
#     if not record:
#         raise HTTPException(status_code=404, detail="Video not found.")
#     if record["status"] != "completed":
#         raise HTTPException(
#             status_code=409,
#             detail=f"Video is not ready for search (status: {record['status']}).",
#         )

#     active_pipeline = payload.pipeline or record.get("pipeline") or "v1"
#     if active_pipeline not in VALID_PIPELINES:
#         raise HTTPException(status_code=400, detail=f"Invalid pipeline '{active_pipeline}'.")

#     sampling_fps = float(record.get("sampling_fps") or DEFAULT_SAMPLING_FPS)

#     if active_pipeline == "v4":
#         try:
#             matches = await search_video_index_v4(
#                 video_id=vid,
#                 query=payload.query,
#                 top_k=payload.top_k,
#                 rerank_top_k=min(payload.top_k, 4),
#                 merge_gap_sec=payload.merge_gap_sec,
#                 sampling_fps=sampling_fps,
#                 llm_provider=payload.llm_provider,
#                 llm_model=payload.llm_model,
#             )
#         except FileNotFoundError as exc:
#             raise HTTPException(status_code=409, detail=str(exc))
#     elif active_pipeline == "v3":
#         try:
#             matches = await search_video_index_v3(
#                 video_id=vid,
#                 query=payload.query,
#                 top_k=payload.top_k,
#                 rerank_top_k=min(payload.top_k, 4),
#                 merge_gap_sec=payload.merge_gap_sec,
#                 sampling_fps=sampling_fps,
#                 llm_provider=payload.llm_provider,
#                 llm_model=payload.llm_model,
#             )
#         except FileNotFoundError as exc:
#             raise HTTPException(status_code=409, detail=str(exc))
#     elif active_pipeline == "v2":
#         try:
#             matches = search_video_index_v2(
#                 video_id=vid,
#                 query=payload.query,
#                 top_k=payload.top_k,
#                 merge_gap_sec=payload.merge_gap_sec,
#                 sampling_fps=sampling_fps,
#             )
#         except FileNotFoundError as exc:
#             raise HTTPException(status_code=409, detail=str(exc))
#     else:
#         query_vec = embedding_service.encode_text(payload.query)
#         matches = search_video_index(
#             video_id=vid,
#             query_vec=query_vec,
#             top_k=payload.top_k,
#             merge_gap_sec=payload.merge_gap_sec,
#             sampling_fps=sampling_fps,
#         )

#     return SearchResponse(
#         video_id=vid,
#         query=payload.query,
#         pipeline=active_pipeline,
#         disclaimer="Retrieval scores indicate visual/multimodal similarity, not proof that an incident occurred. Human review is required.",
#         results=matches,
#     )


# @app.get("/api/videos/{video_id}/frames/{frame_name}")
# def get_frame_thumbnail(video_id: str, frame_name: str):
#     vid = validate_uuid(video_id)
#     safe_name = Path(frame_name).name
#     if safe_name != frame_name or not safe_name.endswith(".jpg"):
#         raise HTTPException(status_code=400, detail="Invalid frame filename.")

#     frame_path = FRAMES_DIR / vid / safe_name
#     if not frame_path.exists():
#         raise HTTPException(status_code=404, detail="Frame thumbnail not found.")
#     return FileResponse(frame_path, media_type="image/jpeg")


# @app.get("/api/videos/{video_id}/stream")
# def stream_video(video_id: str, request: Request):
#     vid = validate_uuid(video_id)
#     record = db.get_video(vid)
#     if not record:
#         raise HTTPException(status_code=404, detail="Video not found.")

#     video_path = VIDEOS_DIR / f"{vid}{record['stored_ext']}"
#     if not video_path.exists():
#         raise HTTPException(status_code=404, detail="Video file not found.")

#     file_size = video_path.stat().st_size
#     range_header = request.headers.get("range")
#     media_type = "video/quicktime" if record["stored_ext"] == ".mov" else "video/mp4"

#     if not range_header:
#         return FileResponse(video_path, media_type=media_type)

#     range_val = range_header.strip().replace("bytes=", "")
#     start_str, end_str = range_val.split("-")
#     start = int(start_str) if start_str else 0
#     end = int(end_str) if end_str else file_size - 1
#     end = min(end, file_size - 1)

#     if start >= file_size or start > end:
#         raise HTTPException(status_code=416, detail="Requested range not satisfiable.")

#     chunk_length = end - start + 1

#     def iter_file(path: Path, offset: int, length: int, chunk_size: int = 1024 * 1024):
#         with open(path, "rb") as f:
#             f.seek(offset)
#             remaining = length
#             while remaining > 0:
#                 data = f.read(min(chunk_size, remaining))
#                 if not data:
#                     break
#                 remaining -= len(data)
#                 yield data

#     return StreamingResponse(
#         iter_file(video_path, start, chunk_length),
#         status_code=206,
#         headers={
#             "Content-Range": f"bytes {start}-{end}/{file_size}",
#             "Accept-Ranges": "bytes",
#             "Content-Length": str(chunk_length),
#         },
#         media_type=media_type,
#     )

# import os

# # Must execute before importing torch or faiss on macOS
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# os.environ["OMP_NUM_THREADS"] = "1"

# from contextlib import asynccontextmanager
# from pathlib import Path
# from typing import List
# import uuid

# from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import FileResponse, StreamingResponse

# from config import (
#     ALLOWED_EXTENSIONS,
#     DEFAULT_SAMPLING_FPS,
#     FRAMES_DIR,
#     MAX_UPLOAD_BYTES,
#     VIDEOS_DIR,
# )
# import database as db
# from embedding import embedding_service
# from pipeline_v2 import process_video_task_v2, search_video_index_v2
# from pipeline_v3 import process_video_task_v3, search_video_index_v3
# from schemas import ProcessRequest, SearchRequest, SearchResponse, VideoMetadata
# from search_index import search_video_index
# from video_processor import process_video_task

# VALID_PIPELINES = {"v1", "v2", "v3"}


# @asynccontextmanager
# async def lifespan(_: FastAPI):
#     db.init_db()
#     embedding_service.load()
#     yield


# app = FastAPI(title="MomentSearch API", version="3.0.0", lifespan=lifespan)

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# def validate_uuid(video_id: str) -> str:
#     try:
#         return str(uuid.UUID(video_id))
#     except ValueError:
#         raise HTTPException(status_code=400, detail="Invalid video ID format.")


# def _get_processor_for_pipeline(pipeline: str):
#     if pipeline == "v3":
#         return process_video_task_v3
#     if pipeline == "v2":
#         return process_video_task_v2
#     return process_video_task


# @app.post("/api/videos/upload", response_model=VideoMetadata)
# async def upload_video(
#     background_tasks: BackgroundTasks,
#     file: UploadFile = File(...),
#     pipeline: str = Form("v1"),
# ):
#     if pipeline not in VALID_PIPELINES:
#         raise HTTPException(
#             status_code=400,
#             detail=f"Invalid pipeline '{pipeline}'. Allowed: v1, v2, v3.",
#         )
#     if not file.filename:
#         raise HTTPException(status_code=400, detail="Missing filename.")

#     ext = Path(file.filename).suffix.lower()
#     if ext not in ALLOWED_EXTENSIONS:
#         raise HTTPException(
#             status_code=400,
#             detail=f"Unsupported file format '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
#         )

#     video_id = str(uuid.uuid4())
#     dest_path = VIDEOS_DIR / f"{video_id}{ext}"

#     bytes_written = 0
#     try:
#         with open(dest_path, "wb") as out_file:
#             while chunk := await file.read(1024 * 1024):
#                 bytes_written += len(chunk)
#                 if bytes_written > MAX_UPLOAD_BYTES:
#                     out_file.close()
#                     dest_path.unlink(missing_ok=True)
#                     raise HTTPException(status_code=413, detail="File exceeds 500 MB upload limit.")
#                 out_file.write(chunk)
#     except HTTPException:
#         raise
#     except Exception as exc:
#         dest_path.unlink(missing_ok=True)
#         raise HTTPException(status_code=500, detail=f"Upload failed: {exc}")

#     db.create_video(video_id=video_id, filename=file.filename, stored_ext=ext, pipeline=pipeline)
#     task_fn = _get_processor_for_pipeline(pipeline)
#     background_tasks.add_task(task_fn, video_id, dest_path, DEFAULT_SAMPLING_FPS)

#     record = db.get_video(video_id)
#     assert record is not None
#     return record


# @app.get("/api/videos", response_model=List[VideoMetadata])
# def get_videos():
#     return db.list_videos()


# @app.get("/api/videos/{video_id}", response_model=VideoMetadata)
# def get_video_metadata(video_id: str):
#     vid = validate_uuid(video_id)
#     record = db.get_video(vid)
#     if not record:
#         raise HTTPException(status_code=404, detail="Video not found.")
#     return record


# @app.post("/api/videos/{video_id}/process", response_model=VideoMetadata)
# def start_video_processing(
#     video_id: str,
#     background_tasks: BackgroundTasks,
#     payload: ProcessRequest = ProcessRequest(),
# ):
#     vid = validate_uuid(video_id)
#     record = db.get_video(vid)
#     if not record:
#         raise HTTPException(status_code=404, detail="Video not found.")
#     if payload.pipeline not in VALID_PIPELINES:
#         raise HTTPException(
#             status_code=400,
#             detail=f"Invalid pipeline '{payload.pipeline}'. Allowed: v1, v2, v3.",
#         )

#     video_path = VIDEOS_DIR / f"{vid}{record['stored_ext']}"
#     if not video_path.exists():
#         raise HTTPException(status_code=404, detail="Video file missing from storage.")

#     db.update_video_status(
#         vid,
#         status="processing",
#         progress=0.0,
#         pipeline=payload.pipeline,
#         error_message=None,
#     )
#     task_fn = _get_processor_for_pipeline(payload.pipeline)
#     background_tasks.add_task(task_fn, vid, video_path, payload.sampling_fps)

#     updated = db.get_video(vid)
#     assert updated is not None
#     return updated


# @app.get("/api/videos/{video_id}/status", response_model=VideoMetadata)
# def get_video_status(video_id: str):
#     vid = validate_uuid(video_id)
#     record = db.get_video(vid)
#     if not record:
#         raise HTTPException(status_code=404, detail="Video not found.")
#     return record


# @app.post("/api/videos/{video_id}/search", response_model=SearchResponse)
# async def search_video(video_id: str, payload: SearchRequest):
#     vid = validate_uuid(video_id)
#     record = db.get_video(vid)
#     if not record:
#         raise HTTPException(status_code=404, detail="Video not found.")
#     if record["status"] != "completed":
#         raise HTTPException(
#             status_code=409,
#             detail=f"Video is not ready for search (status: {record['status']}).",
#         )

#     active_pipeline = payload.pipeline or record.get("pipeline") or "v1"
#     if active_pipeline not in VALID_PIPELINES:
#         raise HTTPException(status_code=400, detail=f"Invalid pipeline '{active_pipeline}'.")

#     sampling_fps = float(record.get("sampling_fps") or DEFAULT_SAMPLING_FPS)

#     if active_pipeline == "v3":
#         try:
#             matches = await search_video_index_v3(
#                 video_id=vid,
#                 query=payload.query,
#                 top_k=payload.top_k,
#                 rerank_top_k=min(payload.top_k, 4),
#                 merge_gap_sec=payload.merge_gap_sec,
#                 sampling_fps=sampling_fps,
#                 llm_provider=payload.llm_provider,
#                 llm_model=payload.llm_model,
#             )
#         except FileNotFoundError as exc:
#             raise HTTPException(status_code=409, detail=str(exc))
#     elif active_pipeline == "v2":
#         try:
#             matches = search_video_index_v2(
#                 video_id=vid,
#                 query=payload.query,
#                 top_k=payload.top_k,
#                 merge_gap_sec=payload.merge_gap_sec,
#                 sampling_fps=sampling_fps,
#             )
#         except FileNotFoundError as exc:
#             raise HTTPException(status_code=409, detail=str(exc))
#     else:
#         query_vec = embedding_service.encode_text(payload.query)
#         matches = search_video_index(
#             video_id=vid,
#             query_vec=query_vec,
#             top_k=payload.top_k,
#             merge_gap_sec=payload.merge_gap_sec,
#             sampling_fps=sampling_fps,
#         )

#     return SearchResponse(
#         video_id=vid,
#         query=payload.query,
#         pipeline=active_pipeline,
#         disclaimer="Retrieval scores indicate visual/multimodal similarity, not proof that an incident occurred. Human review is required.",
#         results=matches,
#     )


# @app.get("/api/videos/{video_id}/frames/{frame_name}")
# def get_frame_thumbnail(video_id: str, frame_name: str):
#     vid = validate_uuid(video_id)
#     safe_name = Path(frame_name).name
#     if safe_name != frame_name or not safe_name.endswith(".jpg"):
#         raise HTTPException(status_code=400, detail="Invalid frame filename.")

#     frame_path = FRAMES_DIR / vid / safe_name
#     if not frame_path.exists():
#         raise HTTPException(status_code=404, detail="Frame thumbnail not found.")
#     return FileResponse(frame_path, media_type="image/jpeg")


# @app.get("/api/videos/{video_id}/stream")
# def stream_video(video_id: str, request: Request):
#     vid = validate_uuid(video_id)
#     record = db.get_video(vid)
#     if not record:
#         raise HTTPException(status_code=404, detail="Video not found.")

#     video_path = VIDEOS_DIR / f"{vid}{record['stored_ext']}"
#     if not video_path.exists():
#         raise HTTPException(status_code=404, detail="Video file not found.")

#     file_size = video_path.stat().st_size
#     range_header = request.headers.get("range")
#     media_type = "video/quicktime" if record["stored_ext"] == ".mov" else "video/mp4"

#     if not range_header:
#         return FileResponse(video_path, media_type=media_type)

#     range_val = range_header.strip().replace("bytes=", "")
#     start_str, end_str = range_val.split("-")
#     start = int(start_str) if start_str else 0
#     end = int(end_str) if end_str else file_size - 1
#     end = min(end, file_size - 1)

#     if start >= file_size or start > end:
#         raise HTTPException(status_code=416, detail="Requested range not satisfiable.")

#     chunk_length = end - start + 1

#     def iter_file(path: Path, offset: int, length: int, chunk_size: int = 1024 * 1024):
#         with open(path, "rb") as f:
#             f.seek(offset)
#             remaining = length
#             while remaining > 0:
#                 data = f.read(min(chunk_size, remaining))
#                 if not data:
#                     break
#                 remaining -= len(data)
#                 yield data

#     return StreamingResponse(
#         iter_file(video_path, start, chunk_length),
#         status_code=206,
#         headers={
#             "Content-Range": f"bytes {start}-{end}/{file_size}",
#             "Accept-Ranges": "bytes",
#             "Content-Length": str(chunk_length),
#         },
#         media_type=media_type,
#     )

# import os

# # Must execute before importing torch or faiss on macOS
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# os.environ["OMP_NUM_THREADS"] = "1"

# from contextlib import asynccontextmanager
# from pathlib import Path
# from typing import List
# import uuid

# from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
# from pipeline_v2 import process_video_task_v2, search_video_index_v2
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import FileResponse, StreamingResponse

# from config import (
#     ALLOWED_EXTENSIONS,
#     DEFAULT_SAMPLING_FPS,
#     FRAMES_DIR,
#     MAX_UPLOAD_BYTES,
#     VIDEOS_DIR,
# )
# import database as db
# from embedding import embedding_service
# from schemas import ProcessRequest, SearchRequest, SearchResponse, VideoMetadata
# from search_index import search_video_index
# from video_processor import process_video_task
# from pipeline_v3 import process_video_task_v3, search_video_index_v3

# @asynccontextmanager
# async def lifespan(_: FastAPI):
#     db.init_db()
#     embedding_service.load()
#     yield


# app = FastAPI(title="MomentSearch API", version="1.0.0", lifespan=lifespan)

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# def validate_uuid(video_id: str) -> str:
#     try:
#         return str(uuid.UUID(video_id))
#     except ValueError:
#         raise HTTPException(status_code=400, detail="Invalid video ID format.")
    

# @app.post("/api/videos/upload", response_model=VideoMetadata)
# async def upload_video(
#     background_tasks: BackgroundTasks,
#     file: UploadFile = File(...),
#     pipeline: str = Form("v1"),
# ):
#     if pipeline not in {"v1", "v2"}:
#         raise HTTPException(status_code=400, detail="Pipeline must be 'v1' or 'v2'.")
#     if not file.filename:
#         raise HTTPException(status_code=400, detail="Missing filename.")

#     ext = Path(file.filename).suffix.lower()
#     if ext not in ALLOWED_EXTENSIONS:
#         raise HTTPException(
#             status_code=400,
#             detail=f"Unsupported file format '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
#         )

#     video_id = str(uuid.uuid4())
#     dest_path = VIDEOS_DIR / f"{video_id}{ext}"

#     bytes_written = 0
#     try:
#         with open(dest_path, "wb") as out_file:
#             while chunk := await file.read(1024 * 1024):
#                 bytes_written += len(chunk)
#                 if bytes_written > MAX_UPLOAD_BYTES:
#                     out_file.close()
#                     dest_path.unlink(missing_ok=True)
#                     raise HTTPException(status_code=413, detail="File exceeds 500 MB upload limit.")
#                 out_file.write(chunk)
#     except HTTPException:
#         raise
#     except Exception as exc:
#         dest_path.unlink(missing_ok=True)
#         raise HTTPException(status_code=500, detail=f"Upload failed: {exc}")

#     db.create_video(video_id=video_id, filename=file.filename, stored_ext=ext, pipeline=pipeline)
#     task_fn = process_video_task_v2 if pipeline == "v2" else process_video_task
#     background_tasks.add_task(task_fn, video_id, dest_path, DEFAULT_SAMPLING_FPS)

#     record = db.get_video(video_id)
#     assert record is not None
#     return record


# # @app.post("/api/videos/upload", response_model=VideoMetadata)
# # async def upload_video(
# #     background_tasks: BackgroundTasks,
# #     file: UploadFile = File(...),
# # ):
# #     if not file.filename:
# #         raise HTTPException(status_code=400, detail="Missing filename.")

# #     ext = Path(file.filename).suffix.lower()
# #     if ext not in ALLOWED_EXTENSIONS:
# #         raise HTTPException(
# #             status_code=400,
# #             detail=f"Unsupported file format '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
# #         )

# #     video_id = str(uuid.uuid4())
# #     dest_path = VIDEOS_DIR / f"{video_id}{ext}"

# #     bytes_written = 0
# #     try:
# #         with open(dest_path, "wb") as out_file:
# #             while chunk := await file.read(1024 * 1024):
# #                 bytes_written += len(chunk)
# #                 if bytes_written > MAX_UPLOAD_BYTES:
# #                     out_file.close()
# #                     dest_path.unlink(missing_ok=True)
# #                     raise HTTPException(status_code=413, detail="File exceeds 500 MB upload limit.")
# #                 out_file.write(chunk)
# #     except HTTPException:
# #         raise
# #     except Exception as exc:
# #         dest_path.unlink(missing_ok=True)
# #         raise HTTPException(status_code=500, detail=f"Upload failed: {exc}")

# #     db.create_video(video_id=video_id, filename=file.filename, stored_ext=ext)
# #     background_tasks.add_task(process_video_task, video_id, dest_path, DEFAULT_SAMPLING_FPS)

# #     record = db.get_video(video_id)
# #     assert record is not None
# #     return record


# @app.get("/api/videos", response_model=List[VideoMetadata])
# def get_videos():
#     return db.list_videos()


# @app.get("/api/videos/{video_id}", response_model=VideoMetadata)
# def get_video_metadata(video_id: str):
#     vid = validate_uuid(video_id)
#     record = db.get_video(vid)
#     if not record:
#         raise HTTPException(status_code=404, detail="Video not found.")
#     return record

# @app.post("/api/videos/{video_id}/process", response_model=VideoMetadata)
# def start_video_processing(
#     video_id: str,
#     background_tasks: BackgroundTasks,
#     payload: ProcessRequest = ProcessRequest(),
# ):
#     vid = validate_uuid(video_id)
#     record = db.get_video(vid)
#     if not record:
#         raise HTTPException(status_code=404, detail="Video not found.")

#     video_path = VIDEOS_DIR / f"{vid}{record['stored_ext']}"
#     if not video_path.exists():
#         raise HTTPException(status_code=404, detail="Video file missing from storage.")

#     db.update_video_status(vid, status="processing", progress=0.0, pipeline=payload.pipeline, error_message=None)
#     task_fn = process_video_task_v2 if payload.pipeline == "v2" else process_video_task
#     background_tasks.add_task(task_fn, vid, video_path, payload.sampling_fps)

#     updated = db.get_video(vid)
#     assert updated is not None
#     return updated


# # @app.post("/api/videos/{video_id}/process", response_model=VideoMetadata)
# # def start_video_processing(
# #     video_id: str,
# #     background_tasks: BackgroundTasks,
# #     payload: ProcessRequest = ProcessRequest(),
# # ):
# #     vid = validate_uuid(video_id)
# #     record = db.get_video(vid)
# #     if not record:
# #         raise HTTPException(status_code=404, detail="Video not found.")

# #     video_path = VIDEOS_DIR / f"{vid}{record['stored_ext']}"
# #     if not video_path.exists():
# #         raise HTTPException(status_code=404, detail="Video file missing from storage.")

# #     db.update_video_status(vid, status="processing", progress=0.0, error_message=None)
# #     background_tasks.add_task(process_video_task, vid, video_path, payload.sampling_fps)

# #     updated = db.get_video(vid)
# #     assert updated is not None
# #     return updated


# @app.get("/api/videos/{video_id}/status", response_model=VideoMetadata)
# def get_video_status(video_id: str):
#     vid = validate_uuid(video_id)
#     record = db.get_video(vid)
#     if not record:
#         raise HTTPException(status_code=404, detail="Video not found.")
#     return record

# @app.post("/api/videos/{video_id}/search", response_model=SearchResponse)
# def search_video(video_id: str, payload: SearchRequest):
#     vid = validate_uuid(video_id)
#     record = db.get_video(vid)
#     if not record:
#         raise HTTPException(status_code=404, detail="Video not found.")
#     if record["status"] != "completed":
#         raise HTTPException(
#             status_code=409,
#             detail=f"Video is not ready for search (status: {record['status']}).",
#         )

#     active_pipeline = payload.pipeline or record.get("pipeline") or "v1"
#     sampling_fps = float(record.get("sampling_fps") or DEFAULT_SAMPLING_FPS)

#     if active_pipeline == "v2":
#         try:
#             matches = search_video_index_v2(
#                 video_id=vid,
#                 query=payload.query,
#                 top_k=payload.top_k,
#                 merge_gap_sec=payload.merge_gap_sec,
#                 sampling_fps=sampling_fps,
#             )
#         except FileNotFoundError as exc:
#             raise HTTPException(status_code=409, detail=str(exc))
#     else:
#         query_vec = embedding_service.encode_text(payload.query)
#         matches = search_video_index(
#             video_id=vid,
#             query_vec=query_vec,
#             top_k=payload.top_k,
#             merge_gap_sec=payload.merge_gap_sec,
#             sampling_fps=sampling_fps,
#         )

#     return SearchResponse(
#         video_id=vid,
#         query=payload.query,
#         pipeline=active_pipeline,
#         disclaimer="Retrieval scores indicate multimodal AI similarity, not proof that an incident occurred. Human review is required.",
#         results=matches,
#     )


# # @app.post("/api/videos/{video_id}/search", response_model=SearchResponse)
# # def search_video(video_id: str, payload: SearchRequest):
# #     vid = validate_uuid(video_id)
# #     record = db.get_video(vid)
# #     if not record:
# #         raise HTTPException(status_code=404, detail="Video not found.")
# #     if record["status"] != "completed":
# #         raise HTTPException(
# #             status_code=409,
# #             detail=f"Video is not ready for search (status: {record['status']}).",
# #         )

# #     query_vec = embedding_service.encode_text(payload.query)
# #     matches = search_video_index(
# #         video_id=vid,
# #         query_vec=query_vec,
# #         top_k=payload.top_k,
# #         merge_gap_sec=payload.merge_gap_sec,
# #         sampling_fps=float(record.get("sampling_fps") or DEFAULT_SAMPLING_FPS),
# #     )

# #     return SearchResponse(
# #         video_id=vid,
# #         query=payload.query,
# #         disclaimer="Retrieval similarity scores indicate CLIP visual-text similarity, not proof that an incident occurred. Human review is required.",
# #         results=matches,
# #     )


# @app.get("/api/videos/{video_id}/frames/{frame_name}")
# def get_frame_thumbnail(video_id: str, frame_name: str):
#     vid = validate_uuid(video_id)
#     safe_name = Path(frame_name).name
#     if safe_name != frame_name or not safe_name.endswith(".jpg"):
#         raise HTTPException(status_code=400, detail="Invalid frame filename.")

#     frame_path = FRAMES_DIR / vid / safe_name
#     if not frame_path.exists():
#         raise HTTPException(status_code=404, detail="Frame thumbnail not found.")
#     return FileResponse(frame_path, media_type="image/jpeg")


# @app.get("/api/videos/{video_id}/stream")
# def stream_video(video_id: str, request: Request):
#     vid = validate_uuid(video_id)
#     record = db.get_video(vid)
#     if not record:
#         raise HTTPException(status_code=404, detail="Video not found.")

#     video_path = VIDEOS_DIR / f"{vid}{record['stored_ext']}"
#     if not video_path.exists():
#         raise HTTPException(status_code=404, detail="Video file not found.")

#     file_size = video_path.stat().st_size
#     range_header = request.headers.get("range")
#     media_type = "video/quicktime" if record["stored_ext"] == ".mov" else "video/mp4"

#     if not range_header:
#         return FileResponse(video_path, media_type=media_type)

#     range_val = range_header.strip().replace("bytes=", "")
#     start_str, end_str = range_val.split("-")
#     start = int(start_str) if start_str else 0
#     end = int(end_str) if end_str else file_size - 1
#     end = min(end, file_size - 1)

#     if start >= file_size or start > end:
#         raise HTTPException(status_code=416, detail="Requested range not satisfiable.")

#     chunk_length = end - start + 1

#     def iter_file(path: Path, offset: int, length: int, chunk_size: int = 1024 * 1024):
#         with open(path, "rb") as f:
#             f.seek(offset)
#             remaining = length
#             while remaining > 0:
#                 data = f.read(min(chunk_size, remaining))
#                 if not data:
#                     break
#                 remaining -= len(data)
#                 yield data

#     return StreamingResponse(
#         iter_file(video_path, start, chunk_length),
#         status_code=206,
#         headers={
#             "Content-Range": f"bytes {start}-{end}/{file_size}",
#             "Accept-Ranges": "bytes",
#             "Content-Length": str(chunk_length),
#         },
#         media_type=media_type,
#     )