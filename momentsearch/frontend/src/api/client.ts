import axios from "axios";
import type {
  LLMProviderType,
  PipelineType,
  SearchResponse,
  VideoMetadata,
} from "../types";

const api = axios.create({
  baseURL: "http://localhost:8000",
});

export async function listVideos(): Promise<VideoMetadata[]> {
  const { data } = await api.get<VideoMetadata[]>("/api/videos");
  return data;
}

export async function getVideoStatus(videoId: string): Promise<VideoMetadata> {
  const { data } = await api.get<VideoMetadata>(`/api/videos/${videoId}/status`);
  return data;
}

export async function uploadVideo(
  file: File,
  pipeline: PipelineType = "v6"
): Promise<VideoMetadata> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("pipeline", pipeline);

  const { data } = await api.post<VideoMetadata>("/api/videos/upload", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
}

export async function processVideo(
  videoId: string,
  pipeline: PipelineType = "v6",
  samplingFps = 1.0
): Promise<VideoMetadata> {
  const { data } = await api.post<VideoMetadata>(`/api/videos/${videoId}/process`, {
    pipeline,
    sampling_fps: samplingFps,
  });
  return data;
}

export async function searchVideo(
  videoId: string,
  query: string,
  pipeline?: PipelineType,
  topK = 5,
  mergeGapSec = 2.5,
  llmProvider: LLMProviderType = "ollama",
  llmModel?: string,
  rerank = true
): Promise<SearchResponse> {
  const { data } = await api.post<SearchResponse>(`/api/videos/${videoId}/search`, {
    query,
    pipeline,
    top_k: topK,
    merge_gap_sec: mergeGapSec,
    llm_provider: llmProvider,
    llm_model: llmModel || null,
    rerank,
  });
  return data;
}

export function getVideoStreamUrl(videoId: string): string {
  return `${api.defaults.baseURL}/api/videos/${videoId}/stream`;
}

export const getStreamUrl = getVideoStreamUrl;

export function resolveAssetUrl(pathOrUrl: string): string {
  if (!pathOrUrl) return "";
  if (pathOrUrl.startsWith("http")) return pathOrUrl;
  return `${api.defaults.baseURL}${pathOrUrl}`;
}

// import axios from "axios";
// import type {
//   LLMProviderType,
//   PipelineType,
//   SearchResponse,
//   VideoMetadata,
// } from "../types";

// const api = axios.create({
//   baseURL: "http://localhost:8000",
// });

// export async function listVideos(): Promise<VideoMetadata[]> {
//   const { data } = await api.get<VideoMetadata[]>("/api/videos");
//   return data;
// }

// export async function getVideoStatus(videoId: string): Promise<VideoMetadata> {
//   const { data } = await api.get<VideoMetadata>(`/api/videos/${videoId}/status`);
//   return data;
// }

// export async function uploadVideo(
//   file: File,
//   pipeline: PipelineType = "v1"
// ): Promise<VideoMetadata> {
//   const formData = new FormData();
//   formData.append("file", file);
//   formData.append("pipeline", pipeline);

//   const { data } = await api.post<VideoMetadata>("/api/videos/upload", formData, {
//     headers: { "Content-Type": "multipart/form-data" },
//   });
//   return data;
// }

// export async function processVideo(
//   videoId: string,
//   pipeline: PipelineType = "v1",
//   samplingFps = 1.0
// ): Promise<VideoMetadata> {
//   const { data } = await api.post<VideoMetadata>(`/api/videos/${videoId}/process`, {
//     pipeline,
//     sampling_fps: samplingFps,
//   });
//   return data;
// }

// export async function searchVideo(
//   videoId: string,
//   query: string,
//   pipeline?: PipelineType,
//   topK = 5,
//   mergeGapSec = 2.5,
//   llmProvider: LLMProviderType = "ollama",
//   llmModel?: string,
//   rerank = true
// ): Promise<SearchResponse> {
//   const { data } = await api.post<SearchResponse>(`/api/videos/${videoId}/search`, {
//     query,
//     pipeline,
//     top_k: topK,
//     merge_gap_sec: mergeGapSec,
//     llm_provider: llmProvider,
//     llm_model: llmModel || null,
//     rerank,
//   });
//   return data;
// }

// // export async function searchVideo(
// //   videoId: string,
// //   query: string,
// //   pipeline?: PipelineType,
// //   topK = 5,
// //   mergeGapSec = 2.5,
// //   llmProvider: LLMProviderType = "ollama",
// //   llmModel?: string
// // ): Promise<SearchResponse> {
// //   const { data } = await api.post<SearchResponse>(`/api/videos/${videoId}/search`, {
// //     query,
// //     pipeline,
// //     top_k: topK,
// //     merge_gap_sec: mergeGapSec,
// //     llm_provider: llmProvider,
// //     llm_model: llmModel || null,
// //   });
// //   return data;
// // }

// export function getVideoStreamUrl(videoId: string): string {
//   return `${api.defaults.baseURL}/api/videos/${videoId}/stream`;
// }

// export const getStreamUrl = getVideoStreamUrl;

// export function resolveAssetUrl(pathOrUrl: string): string {
//   if (pathOrUrl.startsWith("http")) return pathOrUrl;
//   return `${api.defaults.baseURL}${pathOrUrl}`;
// }

// import axios from "axios";
// import type {
//   LLMProviderType,
//   PipelineType,
//   SearchResponse,
//   VideoMetadata,
// } from "../types";

// const api = axios.create({
//   baseURL: "http://localhost:8000",
// });

// export async function listVideos(): Promise<VideoMetadata[]> {
//   const { data } = await api.get<VideoMetadata[]>("/api/videos");
//   return data;
// }

// export async function getVideoStatus(videoId: string): Promise<VideoMetadata> {
//   const { data } = await api.get<VideoMetadata>(`/api/videos/${videoId}/status`);
//   return data;
// }

// export async function uploadVideo(
//   file: File,
//   pipeline: PipelineType = "v1"
// ): Promise<VideoMetadata> {
//   const formData = new FormData();
//   formData.append("file", file);
//   formData.append("pipeline", pipeline);

//   const { data } = await api.post<VideoMetadata>("/api/videos/upload", formData, {
//     headers: { "Content-Type": "multipart/form-data" },
//   });
//   return data;
// }

// export async function processVideo(
//   videoId: string,
//   pipeline: PipelineType = "v1",
//   samplingFps = 1.0
// ): Promise<VideoMetadata> {
//   const { data } = await api.post<VideoMetadata>(`/api/videos/${videoId}/process`, {
//     pipeline,
//     sampling_fps: samplingFps,
//   });
//   return data;
// }

// export async function searchVideo(
//   videoId: string,
//   query: string,
//   pipeline?: PipelineType,
//   topK = 5,
//   mergeGapSec = 2.5,
//   llmProvider: LLMProviderType = "ollama",
//   llmModel?: string
// ): Promise<SearchResponse> {
//   const { data } = await api.post<SearchResponse>(`/api/videos/${videoId}/search`, {
//     query,
//     pipeline,
//     top_k: topK,
//     merge_gap_sec: mergeGapSec,
//     llm_provider: llmProvider,
//     llm_model: llmModel || null,
//   });
//   return data;
// }

// export function getVideoStreamUrl(videoId: string): string {
//   return `${api.defaults.baseURL}/api/videos/${videoId}/stream`;
// }

// export const getStreamUrl = getVideoStreamUrl;

// export function resolveAssetUrl(pathOrUrl: string): string {
//   if (pathOrUrl.startsWith("http")) return pathOrUrl;
//   return `${api.defaults.baseURL}${pathOrUrl}`;
// }

// import axios from "axios";
// import type { PipelineType, SearchResponse, VideoMetadata } from "../types";

// const api = axios.create({
//   baseURL: "http://localhost:8000",
// });

// export async function listVideos(): Promise<VideoMetadata[]> {
//   const { data } = await api.get<VideoMetadata[]>("/api/videos");
//   return data;
// }

// export async function getVideoStatus(videoId: string): Promise<VideoMetadata> {
//   const { data } = await api.get<VideoMetadata>(`/api/videos/${videoId}/status`);
//   return data;
// }

// export async function uploadVideo(
//   file: File,
//   pipeline: PipelineType = "v1"
// ): Promise<VideoMetadata> {
//   const formData = new FormData();
//   formData.append("file", file);
//   formData.append("pipeline", pipeline);

//   const { data } = await api.post<VideoMetadata>("/api/videos/upload", formData, {
//     headers: { "Content-Type": "multipart/form-data" },
//   });
//   return data;
// }

// export async function processVideo(
//   videoId: string,
//   pipeline: PipelineType = "v1",
//   samplingFps = 1.0
// ): Promise<VideoMetadata> {
//   const { data } = await api.post<VideoMetadata>(`/api/videos/${videoId}/process`, {
//     pipeline,
//     sampling_fps: samplingFps,
//   });
//   return data;
// }

// export async function searchVideo(
//   videoId: string,
//   query: string,
//   pipeline?: PipelineType,
//   topK = 5,
//   mergeGapSec = 2.0
// ): Promise<SearchResponse> {
//   const { data } = await api.post<SearchResponse>(`/api/videos/${videoId}/search`, {
//     query,
//     pipeline,
//     top_k: topK,
//     merge_gap_sec: mergeGapSec,
//   });
//   return data;
// }

// export function getVideoStreamUrl(videoId: string): string {
//   return `${api.defaults.baseURL}/api/videos/${videoId}/stream`;
// }

// export const getStreamUrl = getVideoStreamUrl;

// export function resolveAssetUrl(pathOrUrl: string): string {
//   if (pathOrUrl.startsWith("http")) return pathOrUrl;
//   return `${api.defaults.baseURL}${pathOrUrl}`;
// }

// import type { SearchResponse, VideoMetadata } from '../types';

// export const API_BASE = 'http://localhost:8000';

// export async function fetchVideos(): Promise<VideoMetadata[]> {
//   const res = await fetch(`${API_BASE}/api/videos`);
//   if (!res.ok) throw new Error('Failed to load video library.');
//   return res.json();
// }

// export async function uploadVideo(file: File): Promise<VideoMetadata> {
//   const formData = new FormData();
//   formData.append('file', file);

//   const res = await fetch(`${API_BASE}/api/videos/upload`, {
//     method: 'POST',
//     body: formData,
//   });
//   const data = await res.json();
//   if (!res.ok) throw new Error(data.detail || 'Video upload failed.');
//   return data;
// }

// export async function reprocessVideo(videoId: string): Promise<VideoMetadata> {
//   const res = await fetch(`${API_BASE}/api/videos/${videoId}/process`, {
//     method: 'POST',
//   });
//   const data = await res.json();
//   if (!res.ok) throw new Error(data.detail || 'Failed to restart processing.');
//   return data;
// }

// export async function searchVideo(
//   videoId: string,
//   query: string,
//   topK: number
// ): Promise<SearchResponse> {
//   const res = await fetch(`${API_BASE}/api/videos/${videoId}/search`, {
//     method: 'POST',
//     headers: { 'Content-Type': 'application/json' },
//     body: JSON.stringify({
//       query,
//       top_k: topK,
//       merge_gap_sec: 2.0,
//     }),
//   });
//   const data = await res.json();
//   if (!res.ok) throw new Error(data.detail || 'Search failed.');
//   return data;
// }

// export function getVideoStreamUrl(videoId: string): string {
//   return `${API_BASE}/api/videos/${videoId}/stream`;
// }

// export function getThumbnailUrl(relativeUrl: string): string {
//   return `${API_BASE}${relativeUrl}`;
// }