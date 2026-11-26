from datetime import datetime, timezone
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

from config import DB_PATH, DEFAULT_SAMPLING_FPS


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), timeout=30.0)
    conn.row_factory = sqlite3.Row
    return conn


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with get_conn() as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS videos (
                video_id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                stored_ext TEXT NOT NULL,
                pipeline TEXT NOT NULL DEFAULT 'v1',
                status TEXT NOT NULL DEFAULT 'uploaded',
                progress REAL NOT NULL DEFAULT 0.0,
                duration_sec REAL NOT NULL DEFAULT 0.0,
                fps REAL NOT NULL DEFAULT 0.0,
                frame_count INTEGER NOT NULL DEFAULT 0,
                sampling_fps REAL NOT NULL DEFAULT 1.0,
                error_message TEXT,
                created_at TEXT NOT NULL
            )
            """
        )

        # Backward-compatible migrations for existing SQLite files
        existing_cols = {
            row["name"] for row in conn.execute("PRAGMA table_info(videos)").fetchall()
        }
        if "pipeline" not in existing_cols:
            conn.execute("ALTER TABLE videos ADD COLUMN pipeline TEXT NOT NULL DEFAULT 'v1'")
        if "sampling_fps" not in existing_cols:
            conn.execute(
                f"ALTER TABLE videos ADD COLUMN sampling_fps REAL NOT NULL DEFAULT {DEFAULT_SAMPLING_FPS}"
            )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS frame_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id TEXT NOT NULL,
                faiss_idx INTEGER NOT NULL,
                timestamp_sec REAL NOT NULL,
                thumbnail_name TEXT NOT NULL,
                FOREIGN KEY (video_id) REFERENCES videos(video_id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS frame_metadata_v2 (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id TEXT NOT NULL,
                faiss_idx INTEGER NOT NULL,
                timestamp_sec REAL NOT NULL,
                thumbnail_name TEXT NOT NULL,
                ocr_text TEXT NOT NULL DEFAULT '',
                det_labels_json TEXT NOT NULL DEFAULT '[]',
                FOREIGN KEY (video_id) REFERENCES videos(video_id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS video_shots_v5 (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id TEXT NOT NULL,
                shot_idx INTEGER NOT NULL,
                start_time REAL NOT NULL,
                end_time REAL NOT NULL,
                keyframe_name TEXT NOT NULL,
                FOREIGN KEY (video_id) REFERENCES videos(video_id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS frame_metadata_v5 (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id TEXT NOT NULL,
                faiss_idx INTEGER NOT NULL,
                shot_idx INTEGER NOT NULL DEFAULT 0,
                start_time REAL NOT NULL,
                end_time REAL NOT NULL,
                peak_timestamp REAL NOT NULL,
                thumbnail_name TEXT NOT NULL,
                ocr_text TEXT NOT NULL DEFAULT '',
                det_labels_json TEXT NOT NULL DEFAULT '[]',
                ocr_done INTEGER NOT NULL DEFAULT 0,
                det_done INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY (video_id) REFERENCES videos(video_id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS speech_windows_v5 (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id TEXT NOT NULL,
                faiss_idx INTEGER NOT NULL,
                start_time REAL NOT NULL,
                end_time REAL NOT NULL,
                transcript TEXT NOT NULL DEFAULT '',
                FOREIGN KEY (video_id) REFERENCES videos(video_id)
            )
            """
        )
        conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS speech_fts_v5
            USING fts5(video_id UNINDEXED, window_id UNINDEXED, transcript)
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS video_index_status_v5 (
                video_id TEXT PRIMARY KEY,
                file_hash TEXT NOT NULL DEFAULT '',
                visual_status TEXT NOT NULL DEFAULT 'pending',
                speech_status TEXT NOT NULL DEFAULT 'pending',
                ocr_status TEXT NOT NULL DEFAULT 'pending',
                det_status TEXT NOT NULL DEFAULT 'pending',
                updated_at TEXT NOT NULL,
                FOREIGN KEY (video_id) REFERENCES videos(video_id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS frame_metadata_v6 (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id TEXT NOT NULL,
                faiss_idx INTEGER NOT NULL,
                timestamp_sec REAL NOT NULL,
                thumbnail_name TEXT NOT NULL,
                FOREIGN KEY (video_id) REFERENCES videos(video_id)
            )
            """
        )
        conn.commit()


def create_video(
    video_id: str,
    filename: str,
    stored_ext: str,
    pipeline: str = "v1",
) -> None:
    now = _utc_now_iso()
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO videos (
                video_id, filename, stored_ext, pipeline, status, progress,
                duration_sec, fps, frame_count, sampling_fps, error_message, created_at
            )
            VALUES (?, ?, ?, ?, 'uploaded', 0.0, 0.0, 0.0, 0, ?, NULL, ?)
            """,
            (video_id, filename, stored_ext, pipeline, DEFAULT_SAMPLING_FPS, now),
        )
        conn.commit()


def update_video_status(
    video_id: str,
    status: str,
    progress: float,
    duration_sec: Optional[float] = None,
    fps: Optional[float] = None,
    frame_count: Optional[int] = None,
    sampling_fps: Optional[float] = None,
    pipeline: Optional[str] = None,
    error_message: Optional[str] = None,
) -> None:
    fields = ["status = ?", "progress = ?", "error_message = ?"]
    params: List[Any] = [status, progress, error_message]

    if duration_sec is not None:
        fields.append("duration_sec = ?")
        params.append(duration_sec)
    if fps is not None:
        fields.append("fps = ?")
        params.append(fps)
    if frame_count is not None:
        fields.append("frame_count = ?")
        params.append(frame_count)
    if sampling_fps is not None:
        fields.append("sampling_fps = ?")
        params.append(sampling_fps)
    if pipeline is not None:
        fields.append("pipeline = ?")
        params.append(pipeline)

    params.append(video_id)
    with get_conn() as conn:
        conn.execute(f"UPDATE videos SET {', '.join(fields)} WHERE video_id = ?", params)
        conn.commit()


def get_video(video_id: str) -> Optional[Dict[str, Any]]:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM videos WHERE video_id = ?", (video_id,)).fetchone()
        return dict(row) if row else None


def list_videos() -> List[Dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM videos ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]


def save_frame_records(video_id: str, records: List[Tuple[str, int, float, str]]) -> None:
    with get_conn() as conn:
        conn.execute("DELETE FROM frame_metadata WHERE video_id = ?", (video_id,))
        conn.executemany(
            """
            INSERT INTO frame_metadata (video_id, faiss_idx, timestamp_sec, thumbnail_name)
            VALUES (?, ?, ?, ?)
            """,
            records,
        )
        conn.commit()


def get_frame_records(video_id: str) -> List[Dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT faiss_idx, timestamp_sec, thumbnail_name
            FROM frame_metadata
            WHERE video_id = ?
            ORDER BY faiss_idx ASC
            """,
            (video_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def save_frame_records_v2(
    video_id: str, records: List[Tuple[str, int, float, str, str, str]]
) -> None:
    with get_conn() as conn:
        conn.execute("DELETE FROM frame_metadata_v2 WHERE video_id = ?", (video_id,))
        conn.executemany(
            """
            INSERT INTO frame_metadata_v2 (
                video_id, faiss_idx, timestamp_sec, thumbnail_name, ocr_text, det_labels_json
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            records,
        )
        conn.commit()


def get_frame_records_v2(video_id: str) -> List[Dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT faiss_idx, timestamp_sec, thumbnail_name, ocr_text, det_labels_json
            FROM frame_metadata_v2
            WHERE video_id = ?
            ORDER BY faiss_idx ASC
            """,
            (video_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def upsert_index_status_v5(
    video_id: str,
    file_hash: Optional[str] = None,
    visual_status: Optional[str] = None,
    speech_status: Optional[str] = None,
    ocr_status: Optional[str] = None,
    det_status: Optional[str] = None,
) -> None:
    now = _utc_now_iso()
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM video_index_status_v5 WHERE video_id = ?", (video_id,)
        ).fetchone()
        if not row:
            conn.execute(
                """
                INSERT INTO video_index_status_v5 (
                    video_id, file_hash, visual_status, speech_status, ocr_status, det_status, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    video_id,
                    file_hash or "",
                    visual_status or "pending",
                    speech_status or "pending",
                    ocr_status or "pending",
                    det_status or "pending",
                    now,
                ),
            )
        else:
            updates = ["updated_at = ?"]
            params: List[Any] = [now]
            if file_hash is not None:
                updates.append("file_hash = ?")
                params.append(file_hash)
            if visual_status is not None:
                updates.append("visual_status = ?")
                params.append(visual_status)
            if speech_status is not None:
                updates.append("speech_status = ?")
                params.append(speech_status)
            if ocr_status is not None:
                updates.append("ocr_status = ?")
                params.append(ocr_status)
            if det_status is not None:
                updates.append("det_status = ?")
                params.append(det_status)
            params.append(video_id)
            conn.execute(
                f"UPDATE video_index_status_v5 SET {', '.join(updates)} WHERE video_id = ?", params
            )
        conn.commit()


def get_index_status_v5(video_id: str) -> Dict[str, Any]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM video_index_status_v5 WHERE video_id = ?", (video_id,)
        ).fetchone()
        if row:
            return dict(row)
        return {
            "video_id": video_id,
            "file_hash": "",
            "visual_status": "pending",
            "speech_status": "pending",
            "ocr_status": "pending",
            "det_status": "pending",
            "updated_at": _utc_now_iso(),
        }


def save_v5_visual_data(
    video_id: str,
    shot_rows: List[Tuple[str, int, float, float, str]],
    frame_rows: List[Tuple[str, int, int, float, float, float, str]],
) -> None:
    with get_conn() as conn:
        conn.execute("DELETE FROM video_shots_v5 WHERE video_id = ?", (video_id,))
        conn.execute("DELETE FROM frame_metadata_v5 WHERE video_id = ?", (video_id,))
        if shot_rows:
            conn.executemany(
                """
                INSERT INTO video_shots_v5 (video_id, shot_idx, start_time, end_time, keyframe_name)
                VALUES (?, ?, ?, ?, ?)
                """,
                shot_rows,
            )
        if frame_rows:
            conn.executemany(
                """
                INSERT INTO frame_metadata_v5 (
                    video_id, faiss_idx, shot_idx, start_time, end_time, peak_timestamp, thumbnail_name
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                frame_rows,
            )
        conn.commit()


def get_frame_meta_v5(video_id: str) -> List[Dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT faiss_idx, shot_idx, start_time, end_time, peak_timestamp,
                   thumbnail_name, ocr_text, det_labels_json, ocr_done, det_done
            FROM frame_metadata_v5
            WHERE video_id = ?
            ORDER BY faiss_idx ASC
            """,
            (video_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def update_frame_lazy_v5(
    video_id: str,
    updates: List[Tuple[int, str, str, int, int]],
) -> None:
    with get_conn() as conn:
        for f_idx, ocr_txt, det_json, ocr_done, det_done in updates:
            conn.execute(
                """
                UPDATE frame_metadata_v5
                SET ocr_text = ?, det_labels_json = ?, ocr_done = ?, det_done = ?
                WHERE video_id = ? AND faiss_idx = ?
                """,
                (ocr_txt, det_json, ocr_done, det_done, video_id, f_idx),
            )
        conn.commit()


def save_speech_windows_v5(
    video_id: str,
    rows: List[Tuple[str, int, float, float, str]],
) -> None:
    with get_conn() as conn:
        conn.execute("DELETE FROM speech_windows_v5 WHERE video_id = ?", (video_id,))
        conn.execute("DELETE FROM speech_fts_v5 WHERE video_id = ?", (video_id,))
        for vid, f_idx, st, en, txt in rows:
            cur = conn.execute(
                """
                INSERT INTO speech_windows_v5 (video_id, faiss_idx, start_time, end_time, transcript)
                VALUES (?, ?, ?, ?, ?)
                """,
                (vid, f_idx, st, en, txt),
            )
            win_id = cur.lastrowid
            conn.execute(
                "INSERT INTO speech_fts_v5 (video_id, window_id, transcript) VALUES (?, ?, ?)",
                (vid, win_id, txt),
            )
        conn.commit()


def get_speech_windows_v5(video_id: str) -> List[Dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT id, faiss_idx, start_time, end_time, transcript
            FROM speech_windows_v5
            WHERE video_id = ?
            ORDER BY start_time ASC
            """,
            (video_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def search_speech_bm25_v5(video_id: str, fts_query: str) -> Dict[int, float]:
    if not fts_query.strip():
        return {}
    with get_conn() as conn:
        try:
            rows = conn.execute(
                """
                SELECT CAST(window_id AS INTEGER) AS wid, bm25(speech_fts_v5) AS score
                FROM speech_fts_v5
                WHERE video_id = ? AND speech_fts_v5 MATCH ?
                ORDER BY score ASC
                LIMIT 20
                """,
                (video_id, fts_query),
            ).fetchall()
            out: Dict[int, float] = {}
            for r in rows:
                out[int(r["wid"])] = max(0.01, -float(r["score"]))
            return out
        except sqlite3.OperationalError:
            return {}


def save_frame_records_v6(video_id: str, records: List[Tuple[str, int, float, str]]) -> None:
    with get_conn() as conn:
        conn.execute("DELETE FROM frame_metadata_v6 WHERE video_id = ?", (video_id,))
        conn.executemany(
            """
            INSERT INTO frame_metadata_v6 (video_id, faiss_idx, timestamp_sec, thumbnail_name)
            VALUES (?, ?, ?, ?)
            """,
            records,
        )
        conn.commit()


def get_frame_records_v6(video_id: str) -> List[Dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT faiss_idx, timestamp_sec, thumbnail_name
            FROM frame_metadata_v6
            WHERE video_id = ?
            ORDER BY faiss_idx ASC
            """,
            (video_id,),
        ).fetchall()
        return [dict(r) for r in rows]

        
# import sqlite3
# from typing import Any, Dict, List, Optional, Tuple

# from config import DB_PATH


# def get_conn() -> sqlite3.Connection:
#     DB_PATH.parent.mkdir(parents=True, exist_ok=True)
#     conn = sqlite3.connect(str(DB_PATH))
#     conn.row_factory = sqlite3.Row
#     return conn


# def init_db() -> None:
#     with get_conn() as conn:
#         conn.execute(
#             """
#             CREATE TABLE IF NOT EXISTS videos (
#                 video_id TEXT PRIMARY KEY,
#                 filename TEXT NOT NULL,
#                 stored_ext TEXT NOT NULL,
#                 pipeline TEXT NOT NULL DEFAULT 'v1',
#                 status TEXT NOT NULL DEFAULT 'uploaded',
#                 progress REAL NOT NULL DEFAULT 0.0,
#                 duration_sec REAL NOT NULL DEFAULT 0.0,
#                 fps REAL NOT NULL DEFAULT 0.0,
#                 frame_count INTEGER NOT NULL DEFAULT 0,
#                 sampling_fps REAL NOT NULL DEFAULT 1.0,
#                 error_message TEXT,
#                 created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
#             )
#             """
#         )
#         # Safe migration if videos table already existed without 'pipeline'
#         cols = [r["name"] for r in conn.execute("PRAGMA table_info(videos)").fetchall()]
#         if "pipeline" not in cols:
#             conn.execute("ALTER TABLE videos ADD COLUMN pipeline TEXT NOT NULL DEFAULT 'v1'")

#         conn.execute(
#             """
#             CREATE TABLE IF NOT EXISTS frame_metadata (
#                 id INTEGER PRIMARY KEY AUTOINCREMENT,
#                 video_id TEXT NOT NULL,
#                 faiss_idx INTEGER NOT NULL,
#                 timestamp_sec REAL NOT NULL,
#                 thumbnail_name TEXT NOT NULL,
#                 FOREIGN KEY (video_id) REFERENCES videos(video_id)
#             )
#             """
#         )
#         conn.execute(
#             """
#             CREATE TABLE IF NOT EXISTS frame_metadata_v2 (
#                 id INTEGER PRIMARY KEY AUTOINCREMENT,
#                 video_id TEXT NOT NULL,
#                 faiss_idx INTEGER NOT NULL,
#                 timestamp_sec REAL NOT NULL,
#                 thumbnail_name TEXT NOT NULL,
#                 sharpness REAL NOT NULL DEFAULT 0.0,
#                 objects_json TEXT NOT NULL DEFAULT '[]',
#                 ocr_text TEXT NOT NULL DEFAULT '',
#                 FOREIGN KEY (video_id) REFERENCES videos(video_id)
#             )
#             """
#         )
#         conn.execute(
#             """
#             CREATE TABLE IF NOT EXISTS index_status_v5 (
#                 video_id TEXT PRIMARY KEY,
#                 file_hash TEXT,
#                 visual_status TEXT NOT NULL DEFAULT 'pending',
#                 speech_status TEXT NOT NULL DEFAULT 'pending',
#                 ocr_status TEXT NOT NULL DEFAULT 'pending',
#                 det_status TEXT NOT NULL DEFAULT 'pending',
#                 updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
#                 FOREIGN KEY (video_id) REFERENCES videos(video_id)
#             )
#             """
#         )
#         conn.execute(
#             """
#             CREATE TABLE IF NOT EXISTS segments_v5 (
#                 id INTEGER PRIMARY KEY AUTOINCREMENT,
#                 video_id TEXT NOT NULL,
#                 shot_idx INTEGER NOT NULL,
#                 start_time REAL NOT NULL,
#                 end_time REAL NOT NULL,
#                 keyframe_name TEXT NOT NULL,
#                 FOREIGN KEY (video_id) REFERENCES videos(video_id)
#             )
#             """
#         )
#         conn.execute(
#             """
#             CREATE TABLE IF NOT EXISTS frame_meta_v5 (
#                 id INTEGER PRIMARY KEY AUTOINCREMENT,
#                 video_id TEXT NOT NULL,
#                 faiss_idx INTEGER NOT NULL,
#                 shot_idx INTEGER NOT NULL,
#                 start_time REAL NOT NULL,
#                 end_time REAL NOT NULL,
#                 peak_timestamp REAL NOT NULL,
#                 thumbnail_name TEXT NOT NULL,
#                 ocr_text TEXT NOT NULL DEFAULT '',
#                 det_labels_json TEXT NOT NULL DEFAULT '[]',
#                 ocr_done INTEGER NOT NULL DEFAULT 0,
#                 det_done INTEGER NOT NULL DEFAULT 0,
#                 FOREIGN KEY (video_id) REFERENCES videos(video_id)
#             )
#             """
#         )
#         # Safe migration for existing frame_meta_v5 tables
#         v5_cols = [r["name"] for r in conn.execute("PRAGMA table_info(frame_meta_v5)").fetchall()]
#         if "ocr_done" not in v5_cols:
#             conn.execute("ALTER TABLE frame_meta_v5 ADD COLUMN ocr_done INTEGER NOT NULL DEFAULT 0")
#         if "det_done" not in v5_cols:
#             conn.execute("ALTER TABLE frame_meta_v5 ADD COLUMN det_done INTEGER NOT NULL DEFAULT 0")

#         conn.execute(
#             """
#             CREATE TABLE IF NOT EXISTS speech_windows_v5 (
#                 id INTEGER PRIMARY KEY AUTOINCREMENT,
#                 video_id TEXT NOT NULL,
#                 faiss_idx INTEGER NOT NULL,
#                 start_time REAL NOT NULL,
#                 end_time REAL NOT NULL,
#                 transcript TEXT NOT NULL,
#                 FOREIGN KEY (video_id) REFERENCES videos(video_id)
#             )
#             """
#         )
#         conn.execute(
#             """
#             CREATE VIRTUAL TABLE IF NOT EXISTS speech_fts_v5 USING fts5(
#                 video_id UNINDEXED,
#                 window_id UNINDEXED,
#                 transcript
#             )
#             """
#         )
#         conn.execute(
#             """
#             CREATE VIRTUAL TABLE IF NOT EXISTS ocr_fts_v5 USING fts5(
#                 video_id UNINDEXED,
#                 faiss_idx UNINDEXED,
#                 ocr_text
#             )
#             """
#         )
#         conn.commit()


# def create_video(video_id: str, filename: str, stored_ext: str, pipeline: str = "v1") -> None:
#     with get_conn() as conn:
#         conn.execute(
#             "INSERT INTO videos (video_id, filename, stored_ext, pipeline, status) VALUES (?, ?, ?, ?, 'uploaded')",
#             (video_id, filename, stored_ext, pipeline),
#         )
#         conn.commit()


# def update_video_status(
#     video_id: str,
#     status: str,
#     progress: float = 0.0,
#     duration_sec: Optional[float] = None,
#     fps: Optional[float] = None,
#     frame_count: Optional[int] = None,
#     sampling_fps: Optional[float] = None,
#     pipeline: Optional[str] = None,
#     error_message: Optional[str] = None,
# ) -> None:
#     with get_conn() as conn:
#         conn.execute(
#             """
#             UPDATE videos
#             SET status = ?,
#                 progress = ?,
#                 duration_sec = COALESCE(?, duration_sec),
#                 fps = COALESCE(?, fps),
#                 frame_count = COALESCE(?, frame_count),
#                 sampling_fps = COALESCE(?, sampling_fps),
#                 pipeline = COALESCE(?, pipeline),
#                 error_message = ?
#             WHERE video_id = ?
#             """,
#             (
#                 status,
#                 progress,
#                 duration_sec,
#                 fps,
#                 frame_count,
#                 sampling_fps,
#                 pipeline,
#                 error_message,
#                 video_id,
#             ),
#         )
#         conn.commit()


# def get_video(video_id: str) -> Optional[Dict[str, Any]]:
#     with get_conn() as conn:
#         row = conn.execute("SELECT * FROM videos WHERE video_id = ?", (video_id,)).fetchone()
#         return dict(row) if row else None


# def list_videos() -> List[Dict[str, Any]]:
#     with get_conn() as conn:
#         rows = conn.execute("SELECT * FROM videos ORDER BY created_at DESC").fetchall()
#         return [dict(r) for r in rows]


# def save_frame_records(video_id: str, records: List[tuple]) -> None:
#     with get_conn() as conn:
#         conn.execute("DELETE FROM frame_metadata WHERE video_id = ?", (video_id,))
#         conn.executemany(
#             """
#             INSERT INTO frame_metadata (video_id, faiss_idx, timestamp_sec, thumbnail_name)
#             VALUES (?, ?, ?, ?)
#             """,
#             records,
#         )
#         conn.commit()


# def get_frame_records(video_id: str) -> List[Dict[str, Any]]:
#     with get_conn() as conn:
#         rows = conn.execute(
#             """
#             SELECT faiss_idx, timestamp_sec, thumbnail_name
#             FROM frame_metadata
#             WHERE video_id = ?
#             ORDER BY faiss_idx ASC
#             """,
#             (video_id,),
#         ).fetchall()
#         return [dict(r) for r in rows]


# def save_frame_records_v2(video_id: str, records: List[tuple]) -> None:
#     with get_conn() as conn:
#         conn.execute("DELETE FROM frame_metadata_v2 WHERE video_id = ?", (video_id,))
#         conn.executemany(
#             """
#             INSERT INTO frame_metadata_v2
#                 (video_id, faiss_idx, timestamp_sec, thumbnail_name, sharpness, objects_json, ocr_text)
#             VALUES (?, ?, ?, ?, ?, ?, ?)
#             """,
#             records,
#         )
#         conn.commit()


# def get_frame_records_v2(video_id: str) -> List[Dict[str, Any]]:
#     with get_conn() as conn:
#         rows = conn.execute(
#             """
#             SELECT faiss_idx, timestamp_sec, thumbnail_name, sharpness, objects_json, ocr_text
#             FROM frame_metadata_v2
#             WHERE video_id = ?
#             ORDER BY faiss_idx ASC
#             """,
#             (video_id,),
#         ).fetchall()
#         return [dict(r) for r in rows]


# def upsert_index_status_v5(video_id: str, **kwargs: Any) -> None:
#     with get_conn() as conn:
#         conn.execute(
#             "INSERT OR IGNORE INTO index_status_v5 (video_id) VALUES (?)",
#             (video_id,),
#         )
#         for col, val in kwargs.items():
#             if col in {"file_hash", "visual_status", "speech_status", "ocr_status", "det_status"}:
#                 conn.execute(
#                     f"UPDATE index_status_v5 SET {col} = ?, updated_at = CURRENT_TIMESTAMP WHERE video_id = ?",
#                     (val, video_id),
#                 )
#         conn.commit()


# def get_index_status_v5(video_id: str) -> Dict[str, Any]:
#     with get_conn() as conn:
#         row = conn.execute(
#             "SELECT * FROM index_status_v5 WHERE video_id = ?", (video_id,)
#         ).fetchone()
#         return (
#             dict(row)
#             if row
#             else {
#                 "video_id": video_id,
#                 "visual_status": "pending",
#                 "speech_status": "pending",
#                 "ocr_status": "pending",
#                 "det_status": "pending",
#             }
#         )


# def save_v5_visual_data(
#     video_id: str,
#     segments: List[tuple],
#     frames: List[tuple],
# ) -> None:
#     with get_conn() as conn:
#         conn.execute("DELETE FROM segments_v5 WHERE video_id = ?", (video_id,))
#         conn.execute("DELETE FROM frame_meta_v5 WHERE video_id = ?", (video_id,))
#         conn.executemany(
#             """
#             INSERT INTO segments_v5 (video_id, shot_idx, start_time, end_time, keyframe_name)
#             VALUES (?, ?, ?, ?, ?)
#             """,
#             segments,
#         )
#         conn.executemany(
#             """
#             INSERT INTO frame_meta_v5
#                 (video_id, faiss_idx, shot_idx, start_time, end_time, peak_timestamp, thumbnail_name)
#             VALUES (?, ?, ?, ?, ?, ?, ?)
#             """,
#             frames,
#         )
#         conn.commit()


# def get_frame_meta_v5(video_id: str) -> List[Dict[str, Any]]:
#     with get_conn() as conn:
#         rows = conn.execute(
#             "SELECT * FROM frame_meta_v5 WHERE video_id = ? ORDER BY faiss_idx ASC",
#             (video_id,),
#         ).fetchall()
#         return [dict(r) for r in rows]


# def save_speech_windows_v5(video_id: str, windows: List[tuple]) -> None:
#     with get_conn() as conn:
#         conn.execute("DELETE FROM speech_windows_v5 WHERE video_id = ?", (video_id,))
#         conn.execute("DELETE FROM speech_fts_v5 WHERE video_id = ?", (video_id,))
#         for w in windows:
#             cur = conn.execute(
#                 """
#                 INSERT INTO speech_windows_v5 (video_id, faiss_idx, start_time, end_time, transcript)
#                 VALUES (?, ?, ?, ?, ?)
#                 """,
#                 w,
#             )
#             conn.execute(
#                 "INSERT INTO speech_fts_v5 (video_id, window_id, transcript) VALUES (?, ?, ?)",
#                 (video_id, cur.lastrowid, w[4]),
#             )
#         conn.commit()


# def get_speech_windows_v5(video_id: str) -> List[Dict[str, Any]]:
#     with get_conn() as conn:
#         rows = conn.execute(
#             "SELECT * FROM speech_windows_v5 WHERE video_id = ? ORDER BY faiss_idx ASC",
#             (video_id,),
#         ).fetchall()
#         return [dict(r) for r in rows]


# def search_speech_bm25_v5(video_id: str, fts_query: str, limit: int = 25) -> Dict[int, float]:
#     if not fts_query.strip():
#         return {}
#     with get_conn() as conn:
#         rows = conn.execute(
#             """
#             SELECT window_id, bm25(speech_fts_v5) AS rank_score
#             FROM speech_fts_v5
#             WHERE video_id = ? AND speech_fts_v5 MATCH ?
#             ORDER BY rank_score ASC
#             LIMIT ?
#             """,
#             (video_id, fts_query, limit),
#         ).fetchall()
#         return {int(r["window_id"]): float(-r["rank_score"]) for r in rows}


# def update_frame_lazy_v5(
#     video_id: str,
#     updates: List[Tuple[int, str, str, int, int]],  # (faiss_idx, ocr_text, det_labels_json, ocr_done, det_done)
# ) -> None:
#     with get_conn() as conn:
#         conn.execute("DELETE FROM ocr_fts_v5 WHERE video_id = ?", (video_id,))
#         for faiss_idx, ocr_text, det_json, ocr_done, det_done in updates:
#             conn.execute(
#                 """
#                 UPDATE frame_meta_v5
#                 SET ocr_text = ?, det_labels_json = ?, ocr_done = ?, det_done = ?
#                 WHERE video_id = ? AND faiss_idx = ?
#                 """,
#                 (ocr_text, det_json, ocr_done, det_done, video_id, faiss_idx),
#             )
#             if ocr_text.strip():
#                 conn.execute(
#                     "INSERT INTO ocr_fts_v5 (video_id, faiss_idx, ocr_text) VALUES (?, ?, ?)",
#                     (video_id, faiss_idx, ocr_text),
#                 )
#         conn.commit()

# import sqlite3
# from typing import Any, Dict, List, Optional, Tuple
# from config import DB_PATH


# def get_conn() -> sqlite3.Connection:
#     DB_PATH.parent.mkdir(parents=True, exist_ok=True)
#     conn = sqlite3.connect(str(DB_PATH))
#     conn.row_factory = sqlite3.Row
#     return conn


# def init_db() -> None:
#     with get_conn() as conn:
#         conn.execute(
#             """
#             CREATE TABLE IF NOT EXISTS videos (
#                 video_id TEXT PRIMARY KEY,
#                 filename TEXT NOT NULL,
#                 stored_ext TEXT NOT NULL,
#                 pipeline TEXT NOT NULL DEFAULT 'v1',
#                 status TEXT NOT NULL DEFAULT 'uploaded',
#                 progress REAL NOT NULL DEFAULT 0.0,
#                 duration_sec REAL NOT NULL DEFAULT 0.0,
#                 fps REAL NOT NULL DEFAULT 0.0,
#                 frame_count INTEGER NOT NULL DEFAULT 0,
#                 sampling_fps REAL NOT NULL DEFAULT 1.0,
#                 error_message TEXT,
#                 created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
#             )
#             """
#         )
#         # Safe migration if videos table already existed without 'pipeline'
#         cols = [r["name"] for r in conn.execute("PRAGMA table_info(videos)").fetchall()]
#         if "pipeline" not in cols:
#             conn.execute("ALTER TABLE videos ADD COLUMN pipeline TEXT NOT NULL DEFAULT 'v1'")

#         conn.execute(
#             """
#             CREATE TABLE IF NOT EXISTS frame_metadata (
#                 id INTEGER PRIMARY KEY AUTOINCREMENT,
#                 video_id TEXT NOT NULL,
#                 faiss_idx INTEGER NOT NULL,
#                 timestamp_sec REAL NOT NULL,
#                 thumbnail_name TEXT NOT NULL,
#                 FOREIGN KEY (video_id) REFERENCES videos(video_id)
#             )
#             """
#         )
#         conn.execute(
#             """
#             CREATE TABLE IF NOT EXISTS frame_metadata_v2 (
#                 id INTEGER PRIMARY KEY AUTOINCREMENT,
#                 video_id TEXT NOT NULL,
#                 faiss_idx INTEGER NOT NULL,
#                 timestamp_sec REAL NOT NULL,
#                 thumbnail_name TEXT NOT NULL,
#                 sharpness REAL NOT NULL DEFAULT 0.0,
#                 objects_json TEXT NOT NULL DEFAULT '[]',
#                 ocr_text TEXT NOT NULL DEFAULT '',
#                 FOREIGN KEY (video_id) REFERENCES videos(video_id)
#             )
#             """
#         )
#         # Add inside init_db() in database.py:
#         conn.execute(
#             """
#             CREATE TABLE IF NOT EXISTS index_status_v5 (
#                 video_id TEXT PRIMARY KEY,
#                 file_hash TEXT,
#                 visual_status TEXT NOT NULL DEFAULT 'pending',
#                 speech_status TEXT NOT NULL DEFAULT 'pending',
#                 ocr_status TEXT NOT NULL DEFAULT 'pending',
#                 det_status TEXT NOT NULL DEFAULT 'pending',
#                 updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
#                 FOREIGN KEY (video_id) REFERENCES videos(video_id)
#             )
#             """
#         )
#         conn.execute(
#             """
#             CREATE TABLE IF NOT EXISTS segments_v5 (
#                 id INTEGER PRIMARY KEY AUTOINCREMENT,
#                 video_id TEXT NOT NULL,
#                 shot_idx INTEGER NOT NULL,
#                 start_time REAL NOT NULL,
#                 end_time REAL NOT NULL,
#                 keyframe_name TEXT NOT NULL,
#                 FOREIGN KEY (video_id) REFERENCES videos(video_id)
#             )
#             """
#         )
#         conn.execute(
#             """
#             CREATE TABLE IF NOT EXISTS frame_meta_v5 (
#                 id INTEGER PRIMARY KEY AUTOINCREMENT,
#                 video_id TEXT NOT NULL,
#                 faiss_idx INTEGER NOT NULL,
#                 shot_idx INTEGER NOT NULL,
#                 start_time REAL NOT NULL,
#                 end_time REAL NOT NULL,
#                 peak_timestamp REAL NOT NULL,
#                 thumbnail_name TEXT NOT NULL,
#                 ocr_text TEXT NOT NULL DEFAULT '',
#                 det_labels_json TEXT NOT NULL DEFAULT '[]',
#                 FOREIGN KEY (video_id) REFERENCES videos(video_id)
#             )
#             """
#         )
#         conn.execute(
#             """
#             CREATE TABLE IF NOT EXISTS speech_windows_v5 (
#                 id INTEGER PRIMARY KEY AUTOINCREMENT,
#                 video_id TEXT NOT NULL,
#                 faiss_idx INTEGER NOT NULL,
#                 start_time REAL NOT NULL,
#                 end_time REAL NOT NULL,
#                 transcript TEXT NOT NULL,
#                 FOREIGN KEY (video_id) REFERENCES videos(video_id)
#             )
#             """
#         )
#         conn.execute(
#             """
#             CREATE VIRTUAL TABLE IF NOT EXISTS speech_fts_v5 USING fts5(
#                 video_id UNINDEXED,
#                 window_id UNINDEXED,
#                 transcript
#             )
#             """
#         )
#         conn.execute(
#             """
#             CREATE VIRTUAL TABLE IF NOT EXISTS ocr_fts_v5 USING fts5(
#                 video_id UNINDEXED,
#                 faiss_idx UNINDEXED,
#                 ocr_text
#             )
#             """
#         )
#         conn.commit()


# def create_video(video_id: str, filename: str, stored_ext: str, pipeline: str = "v1") -> None:
#     with get_conn() as conn:
#         conn.execute(
#             "INSERT INTO videos (video_id, filename, stored_ext, pipeline, status) VALUES (?, ?, ?, ?, 'uploaded')",
#             (video_id, filename, stored_ext, pipeline),
#         )
#         conn.commit()


# def update_video_status(
#     video_id: str,
#     status: str,
#     progress: float = 0.0,
#     duration_sec: Optional[float] = None,
#     fps: Optional[float] = None,
#     frame_count: Optional[int] = None,
#     sampling_fps: Optional[float] = None,
#     pipeline: Optional[str] = None,
#     error_message: Optional[str] = None,
# ) -> None:
#     with get_conn() as conn:
#         conn.execute(
#             """
#             UPDATE videos
#             SET status = ?,
#                 progress = ?,
#                 duration_sec = COALESCE(?, duration_sec),
#                 fps = COALESCE(?, fps),
#                 frame_count = COALESCE(?, frame_count),
#                 sampling_fps = COALESCE(?, sampling_fps),
#                 pipeline = COALESCE(?, pipeline),
#                 error_message = ?
#             WHERE video_id = ?
#             """,
#             (status, progress, duration_sec, fps, frame_count, sampling_fps, pipeline, error_message, video_id),
#         )
#         conn.commit()


# def get_video(video_id: str) -> Optional[Dict[str, Any]]:
#     with get_conn() as conn:
#         row = conn.execute("SELECT * FROM videos WHERE video_id = ?", (video_id,)).fetchone()
#         return dict(row) if row else None


# def list_videos() -> List[Dict[str, Any]]:
#     with get_conn() as conn:
#         rows = conn.execute("SELECT * FROM videos ORDER BY created_at DESC").fetchall()
#         return [dict(r) for r in rows]


# def save_frame_records(video_id: str, records: List[tuple]) -> None:
#     with get_conn() as conn:
#         conn.execute("DELETE FROM frame_metadata WHERE video_id = ?", (video_id,))
#         conn.executemany(
#             """
#             INSERT INTO frame_metadata (video_id, faiss_idx, timestamp_sec, thumbnail_name)
#             VALUES (?, ?, ?, ?)
#             """,
#             records,
#         )
#         conn.commit()


# def get_frame_records(video_id: str) -> List[Dict[str, Any]]:
#     with get_conn() as conn:
#         rows = conn.execute(
#             """
#             SELECT faiss_idx, timestamp_sec, thumbnail_name
#             FROM frame_metadata
#             WHERE video_id = ?
#             ORDER BY faiss_idx ASC
#             """,
#             (video_id,),
#         ).fetchall()
#         return [dict(r) for r in rows]


# def save_frame_records_v2(video_id: str, records: List[tuple]) -> None:
#     with get_conn() as conn:
#         conn.execute("DELETE FROM frame_metadata_v2 WHERE video_id = ?", (video_id,))
#         conn.executemany(
#             """
#             INSERT INTO frame_metadata_v2
#                 (video_id, faiss_idx, timestamp_sec, thumbnail_name, sharpness, objects_json, ocr_text)
#             VALUES (?, ?, ?, ?, ?, ?, ?)
#             """,
#             records,
#         )
#         conn.commit()


# def get_frame_records_v2(video_id: str) -> List[Dict[str, Any]]:
#     with get_conn() as conn:
#         rows = conn.execute(
#             """
#             SELECT faiss_idx, timestamp_sec, thumbnail_name, sharpness, objects_json, ocr_text
#             FROM frame_metadata_v2
#             WHERE video_id = ?
#             ORDER BY faiss_idx ASC
#             """,
#             (video_id,),
#         ).fetchall()
#         return [dict(r) for r in rows]
    
# def upsert_index_status_v5(video_id: str, **kwargs: Any) -> None:
#     with get_conn() as conn:
#         conn.execute(
#             "INSERT OR IGNORE INTO index_status_v5 (video_id) VALUES (?)",
#             (video_id,),
#         )
#         for col, val in kwargs.items():
#             if col in {"file_hash", "visual_status", "speech_status", "ocr_status", "det_status"}:
#                 conn.execute(
#                     f"UPDATE index_status_v5 SET {col} = ?, updated_at = CURRENT_TIMESTAMP WHERE video_id = ?",
#                     (val, video_id),
#                 )
#         conn.commit()


# def get_index_status_v5(video_id: str) -> Dict[str, Any]:
#     with get_conn() as conn:
#         row = conn.execute(
#             "SELECT * FROM index_status_v5 WHERE video_id = ?", (video_id,)
#         ).fetchone()
#         return dict(row) if row else {
#             "video_id": video_id,
#             "visual_status": "pending",
#             "speech_status": "pending",
#             "ocr_status": "pending",
#             "det_status": "pending",
#         }


# def save_v5_visual_data(
#     video_id: str,
#     segments: List[tuple],
#     frames: List[tuple],
# ) -> None:
#     with get_conn() as conn:
#         conn.execute("DELETE FROM segments_v5 WHERE video_id = ?", (video_id,))
#         conn.execute("DELETE FROM frame_meta_v5 WHERE video_id = ?", (video_id,))
#         conn.executemany(
#             """
#             INSERT INTO segments_v5 (video_id, shot_idx, start_time, end_time, keyframe_name)
#             VALUES (?, ?, ?, ?, ?)
#             """,
#             segments,
#         )
#         conn.executemany(
#             """
#             INSERT INTO frame_meta_v5
#                 (video_id, faiss_idx, shot_idx, start_time, end_time, peak_timestamp, thumbnail_name)
#             VALUES (?, ?, ?, ?, ?, ?, ?)
#             """,
#             frames,
#         )
#         conn.commit()


# def get_frame_meta_v5(video_id: str) -> List[Dict[str, Any]]:
#     with get_conn() as conn:
#         rows = conn.execute(
#             "SELECT * FROM frame_meta_v5 WHERE video_id = ? ORDER BY faiss_idx ASC",
#             (video_id,),
#         ).fetchall()
#         return [dict(r) for r in rows]


# def save_speech_windows_v5(video_id: str, windows: List[tuple]) -> None:
#     with get_conn() as conn:
#         conn.execute("DELETE FROM speech_windows_v5 WHERE video_id = ?", (video_id,))
#         conn.execute("DELETE FROM speech_fts_v5 WHERE video_id = ?", (video_id,))
#         for w in windows:
#             cur = conn.execute(
#                 """
#                 INSERT INTO speech_windows_v5 (video_id, faiss_idx, start_time, end_time, transcript)
#                 VALUES (?, ?, ?, ?, ?)
#                 """,
#                 w,
#             )
#             conn.execute(
#                 "INSERT INTO speech_fts_v5 (video_id, window_id, transcript) VALUES (?, ?, ?)",
#                 (video_id, cur.lastrowid, w[4]),
#             )
#         conn.commit()


# def get_speech_windows_v5(video_id: str) -> List[Dict[str, Any]]:
#     with get_conn() as conn:
#         rows = conn.execute(
#             "SELECT * FROM speech_windows_v5 WHERE video_id = ? ORDER BY faiss_idx ASC",
#             (video_id,),
#         ).fetchall()
#         return [dict(r) for r in rows]


# def search_speech_bm25_v5(video_id: str, fts_query: str, limit: int = 25) -> Dict[int, float]:
#     if not fts_query.strip():
#         return {}
#     with get_conn() as conn:
#         rows = conn.execute(
#             """
#             SELECT window_id, bm25(speech_fts_v5) AS rank_score
#             FROM speech_fts_v5
#             WHERE video_id = ? AND speech_fts_v5 MATCH ?
#             ORDER BY rank_score ASC
#             LIMIT ?
#             """,
#             (video_id, fts_query, limit),
#         ).fetchall()
#         # FTS5 bm25() returns negative scores where more negative = better match
#         return {int(r["window_id"]): float(-r["rank_score"]) for r in rows}


# def update_frame_lazy_v5(
#     video_id: str,
#     updates: List[Tuple[int, str, str]],  # (faiss_idx, ocr_text, det_labels_json)
# ) -> None:
#     with get_conn() as conn:
#         conn.execute("DELETE FROM ocr_fts_v5 WHERE video_id = ?", (video_id,))
#         for faiss_idx, ocr_text, det_json in updates:
#             conn.execute(
#                 """
#                 UPDATE frame_meta_v5
#                 SET ocr_text = ?, det_labels_json = ?
#                 WHERE video_id = ? AND faiss_idx = ?
#                 """,
#                 (ocr_text, det_json, video_id, faiss_idx),
#             )
#             if ocr_text.strip():
#                 conn.execute(
#                     "INSERT INTO ocr_fts_v5 (video_id, faiss_idx, ocr_text) VALUES (?, ?, ?)",
#                     (video_id, faiss_idx, ocr_text),
#                 )
#         conn.commit()

# import sqlite3
# from typing import Any, Dict, List, Optional
# from config import DB_PATH


# def get_conn() -> sqlite3.Connection:
#     DB_PATH.parent.mkdir(parents=True, exist_ok=True)
#     conn = sqlite3.connect(str(DB_PATH))
#     conn.row_factory = sqlite3.Row
#     return conn


# def init_db() -> None:
#     with get_conn() as conn:
#         conn.execute(
#             """
#             CREATE TABLE IF NOT EXISTS videos (
#                 video_id TEXT PRIMARY KEY,
#                 filename TEXT NOT NULL,
#                 stored_ext TEXT NOT NULL,
#                 status TEXT NOT NULL DEFAULT 'uploaded',
#                 progress REAL NOT NULL DEFAULT 0.0,
#                 duration_sec REAL NOT NULL DEFAULT 0.0,
#                 fps REAL NOT NULL DEFAULT 0.0,
#                 frame_count INTEGER NOT NULL DEFAULT 0,
#                 sampling_fps REAL NOT NULL DEFAULT 1.0,
#                 error_message TEXT,
#                 created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
#             )
#             """
#         )
#         conn.execute(
#             """
#             CREATE TABLE IF NOT EXISTS frame_metadata (
#                 id INTEGER PRIMARY KEY AUTOINCREMENT,
#                 video_id TEXT NOT NULL,
#                 faiss_idx INTEGER NOT NULL,
#                 timestamp_sec REAL NOT NULL,
#                 thumbnail_name TEXT NOT NULL,
#                 FOREIGN KEY (video_id) REFERENCES videos(video_id)
#             )
#             """
#         )
#         conn.commit()


# def create_video(video_id: str, filename: str, stored_ext: str) -> None:
#     with get_conn() as conn:
#         conn.execute(
#             "INSERT INTO videos (video_id, filename, stored_ext, status) VALUES (?, ?, ?, 'uploaded')",
#             (video_id, filename, stored_ext),
#         )
#         conn.commit()


# def update_video_status(
#     video_id: str,
#     status: str,
#     progress: float = 0.0,
#     duration_sec: Optional[float] = None,
#     fps: Optional[float] = None,
#     frame_count: Optional[int] = None,
#     sampling_fps: Optional[float] = None,
#     error_message: Optional[str] = None,
# ) -> None:
#     with get_conn() as conn:
#         conn.execute(
#             """
#             UPDATE videos
#             SET status = ?,
#                 progress = ?,
#                 duration_sec = COALESCE(?, duration_sec),
#                 fps = COALESCE(?, fps),
#                 frame_count = COALESCE(?, frame_count),
#                 sampling_fps = COALESCE(?, sampling_fps),
#                 error_message = ?
#             WHERE video_id = ?
#             """,
#             (status, progress, duration_sec, fps, frame_count, sampling_fps, error_message, video_id),
#         )
#         conn.commit()


# def get_video(video_id: str) -> Optional[Dict[str, Any]]:
#     with get_conn() as conn:
#         row = conn.execute("SELECT * FROM videos WHERE video_id = ?", (video_id,)).fetchone()
#         return dict(row) if row else None


# def list_videos() -> List[Dict[str, Any]]:
#     with get_conn() as conn:
#         rows = conn.execute("SELECT * FROM videos ORDER BY created_at DESC").fetchall()
#         return [dict(r) for r in rows]


# def save_frame_records(video_id: str, records: List[tuple]) -> None:
#     with get_conn() as conn:
#         conn.execute("DELETE FROM frame_metadata WHERE video_id = ?", (video_id,))
#         conn.executemany(
#             """
#             INSERT INTO frame_metadata (video_id, faiss_idx, timestamp_sec, thumbnail_name)
#             VALUES (?, ?, ?, ?)
#             """,
#             records,
#         )
#         conn.commit()


# def get_frame_records(video_id: str) -> List[Dict[str, Any]]:
#     with get_conn() as conn:
#         rows = conn.execute(
#             """
#             SELECT faiss_idx, timestamp_sec, thumbnail_name
#             FROM frame_metadata
#             WHERE video_id = ?
#             ORDER BY faiss_idx ASC
#             """,
#             (video_id,),
#         ).fetchall()
#         return [dict(r) for r in rows]