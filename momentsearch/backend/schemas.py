from typing import List, Literal, Optional
from pydantic import BaseModel, Field

PipelineType = Literal["v1", "v2", "v3", "v4", "v5", "v6"]
LLMProviderType = Literal["ollama", "gemini", "none"]


class VideoMetadata(BaseModel):
    video_id: str
    filename: str
    pipeline: str = "v1"
    status: str  # uploaded | processing | completed | failed
    progress: float = 0.0
    duration_sec: float = 0.0
    fps: float = 0.0
    frame_count: int = 0
    error_message: Optional[str] = None
    created_at: str


class ProcessRequest(BaseModel):
    sampling_fps: float = Field(default=1.0, ge=0.25, le=5.0)
    pipeline: PipelineType = Field(default="v1")

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=300)
    top_k: int = Field(default=5, ge=1, le=25)
    merge_gap_sec: float = Field(default=2.5, ge=0.0, le=10.0)
    pipeline: Optional[PipelineType] = Field(default=None)
    llm_provider: LLMProviderType = Field(default="ollama")
    llm_model: Optional[str] = Field(default=None)
    rerank: bool = Field(default=True)


# class SearchRequest(BaseModel):
#     query: str = Field(..., min_length=2, max_length=300)
#     top_k: int = Field(default=5, ge=1, le=25)
#     merge_gap_sec: float = Field(default=2.5, ge=0.0, le=10.0)
#     pipeline: Optional[PipelineType] = Field(default=None)
#     llm_provider: LLMProviderType = Field(default="ollama")
#     llm_model: Optional[str] = Field(default=None)


class SearchMatch(BaseModel):
    rank: int
    start_time: float
    end_time: float
    peak_timestamp: float
    similarity_score: float
    thumbnail_url: str
    confidence: Optional[float] = None
    ai_review_flag: Optional[str] = None
    detected_objects: List[str] = Field(default_factory=list)
    ocr_text: Optional[str] = None
    keyframes: List[str] = Field(default_factory=list)
    llm_selected: Optional[bool] = None
    llm_reason: Optional[str] = None


class SearchResponse(BaseModel):
    video_id: str
    query: str
    pipeline: str = "v1"
    disclaimer: str
    results: List[SearchMatch]

# from typing import List, Literal, Optional
# from pydantic import BaseModel, Field

# PipelineType = Literal["v1", "v2", "v3"]
# LLMProviderType = Literal["ollama", "gemini", "none"]


# class VideoMetadata(BaseModel):
#     video_id: str
#     filename: str
#     pipeline: str = "v1"
#     status: str  # uploaded | processing | completed | failed
#     progress: float = 0.0
#     duration_sec: float = 0.0
#     fps: float = 0.0
#     frame_count: int = 0
#     error_message: Optional[str] = None
#     created_at: str


# class ProcessRequest(BaseModel):
#     sampling_fps: float = Field(default=1.0, ge=0.25, le=5.0)
#     pipeline: PipelineType = Field(default="v1")

# class SearchRequest(BaseModel):
#     query: str = Field(..., min_length=2, max_length=300)
#     top_k: int = Field(default=5, ge=1, le=25)
#     merge_gap_sec: float = Field(default=2.5, ge=0.0, le=10.0)
#     pipeline: Optional[PipelineType] = Field(default=None)
#     llm_provider: LLMProviderType = Field(default="ollama")
#     llm_model: Optional[str] = Field(default=None)


# class SearchMatch(BaseModel):
#     rank: int
#     start_time: float
#     end_time: float
#     peak_timestamp: float
#     similarity_score: float
#     thumbnail_url: str
#     confidence: Optional[float] = None
#     ai_review_flag: Optional[str] = None
#     detected_objects: List[str] = Field(default_factory=list)
#     ocr_text: Optional[str] = None
#     keyframes: List[str] = Field(default_factory=list)
#     llm_selected: Optional[bool] = None
#     llm_reason: Optional[str] = None


# # class SearchRequest(BaseModel):
# #     query: str = Field(..., min_length=2, max_length=300)
# #     top_k: int = Field(default=5, ge=1, le=25)
# #     merge_gap_sec: float = Field(default=2.0, ge=0.0, le=10.0)
# #     pipeline: Optional[PipelineType] = Field(default=None)  # Defaults to video's indexed pipeline if None


# # class SearchMatch(BaseModel):
# #     rank: int
# #     start_time: float
# #     end_time: float
# #     peak_timestamp: float
# #     similarity_score: float
# #     thumbnail_url: str
# #     # V2 Multimodal Evidence & Grounding Fields
# #     confidence: Optional[float] = None
# #     ai_review_flag: Optional[str] = None  # verified | needs_review | low_confidence
# #     detected_objects: List[str] = Field(default_factory=list)
# #     ocr_text: Optional[str] = None
# #     keyframes: List[str] = Field(default_factory=list)


# class SearchResponse(BaseModel):
#     video_id: str
#     query: str
#     pipeline: str = "v1"
#     disclaimer: str
#     results: List[SearchMatch]

# # from typing import List, Optional
# # from pydantic import BaseModel, Field


# # class VideoMetadata(BaseModel):
# #     video_id: str
# #     filename: str
# #     status: str  # uploaded | processing | completed | failed
# #     progress: float = 0.0
# #     duration_sec: float = 0.0
# #     fps: float = 0.0
# #     frame_count: int = 0
# #     error_message: Optional[str] = None
# #     created_at: str


# # class ProcessRequest(BaseModel):
# #     sampling_fps: float = Field(default=1.0, ge=0.25, le=5.0)


# # class SearchRequest(BaseModel):
# #     query: str = Field(..., min_length=2, max_length=300)
# #     top_k: int = Field(default=5, ge=1, le=25)
# #     merge_gap_sec: float = Field(default=2.0, ge=0.0, le=10.0)


# # class SearchMatch(BaseModel):
# #     rank: int
# #     start_time: float
# #     end_time: float
# #     peak_timestamp: float
# #     similarity_score: float
# #     thumbnail_url: str


# # class SearchResponse(BaseModel):
# #     video_id: str
# #     query: str
# #     disclaimer: str
# #     results: List[SearchMatch]