export type PipelineType = "v1" | "v2" | "v3" | "v4" | "v5" | "v6";
export type LLMProviderType = "ollama" | "gemini" | "none";

export interface VideoMetadata {
  video_id: string;
  filename: string;
  pipeline: PipelineType;
  status: "uploaded" | "processing" | "completed" | "failed" | string;
  progress: number;
  duration_sec: number;
  fps: number;
  frame_count: number;
  error_message?: string | null;
  created_at: string;
}

export interface SearchMatch {
  rank: number;
  start_time: number;
  end_time: number;
  peak_timestamp: number;
  similarity_score: number;
  thumbnail_url: string;
  confidence?: number | null;
  ai_review_flag?: "verified" | "needs_review" | "low_confidence" | string | null;
  detected_objects?: string[];
  ocr_text?: string | null;
  keyframes?: string[];
  llm_selected?: boolean | null;
  llm_reason?: string | null;
}

export interface SearchResponse {
  video_id: string;
  query: string;
  pipeline: PipelineType;
  disclaimer: string;
  results: SearchMatch[];
}

// export type PipelineType = "v1" | "v2" | "v3" | "v4";
// export type LLMProviderType = "ollama" | "gemini" | "none";

// export interface VideoMetadata {
//   video_id: string;
//   filename: string;
//   pipeline: PipelineType;
//   status: "uploaded" | "processing" | "completed" | "failed" | string;
//   progress: number;
//   duration_sec: number;
//   fps: number;
//   frame_count: number;
//   error_message?: string | null;
//   created_at: string;
// }

// export interface SearchMatch {
//   rank: number;
//   start_time: number;
//   end_time: number;
//   peak_timestamp: number;
//   similarity_score: number;
//   thumbnail_url: string;
//   confidence?: number | null;
//   ai_review_flag?: "verified" | "needs_review" | "low_confidence" | string | null;
//   detected_objects?: string[];
//   ocr_text?: string | null;
//   keyframes?: string[];
//   llm_selected?: boolean | null;
//   llm_reason?: string | null;
// }

// export interface SearchResponse {
//   video_id: string;
//   query: string;
//   pipeline: PipelineType;
//   disclaimer: string;
//   results: SearchMatch[];
// }

// export type PipelineType = "v1" | "v2" | "v3";
// export type LLMProviderType = "ollama" | "gemini" | "none";

// export interface VideoMetadata {
//   video_id: string;
//   filename: string;
//   pipeline: PipelineType;
//   status: "uploaded" | "processing" | "completed" | "failed" | string;
//   progress: number;
//   duration_sec: number;
//   fps: number;
//   frame_count: number;
//   error_message?: string | null;
//   created_at: string;
// }

// export interface SearchMatch {
//   rank: number;
//   start_time: number;
//   end_time: number;
//   peak_timestamp: number;
//   similarity_score: number;
//   thumbnail_url: string;
//   confidence?: number | null;
//   ai_review_flag?: "verified" | "needs_review" | "low_confidence" | string | null;
//   detected_objects?: string[];
//   ocr_text?: string | null;
//   keyframes?: string[];
//   llm_selected?: boolean | null;
//   llm_reason?: string | null;
// }

// export interface SearchResponse {
//   video_id: string;
//   query: string;
//   pipeline: PipelineType;
//   disclaimer: string;
//   results: SearchMatch[];
// }

// // export type PipelineType = "v1" | "v2";

// // export interface VideoMetadata {
// //   video_id: string;
// //   filename: string;
// //   pipeline: PipelineType;
// //   status: "uploaded" | "processing" | "completed" | "failed" | string;
// //   progress: number;
// //   duration_sec: number;
// //   fps: number;
// //   frame_count: number;
// //   error_message?: string | null;
// //   created_at: string;
// // }

// // export interface SearchMatch {
// //   rank: number;
// //   start_time: number;
// //   end_time: number;
// //   peak_timestamp: number;
// //   similarity_score: number;
// //   thumbnail_url: string;
// //   confidence?: number | null;
// //   ai_review_flag?: "verified" | "needs_review" | "low_confidence" | string | null;
// //   detected_objects?: string[];
// //   ocr_text?: string | null;
// //   keyframes?: string[];
// // }

// // export interface SearchResponse {
// //   video_id: string;
// //   query: string;
// //   pipeline: PipelineType;
// //   disclaimer: string;
// //   results: SearchMatch[];
// // }

// // export interface VideoMetadata {
// //   video_id: string;
// //   filename: string;
// //   status: 'uploaded' | 'processing' | 'completed' | 'failed';
// //   progress: number;
// //   duration_sec: number;
// //   fps: number;
// //   frame_count: number;
// //   error_message?: string | null;
// //   created_at: string;
// // }

// // export interface SearchMatch {
// //   rank: number;
// //   start_time: number;
// //   end_time: number;
// //   peak_timestamp: number;
// //   similarity_score: number;
// //   thumbnail_url: string;
// // }

// // export interface SearchResponse {
// //   video_id: string;
// //   query: string;
// //   disclaimer: string;
// //   results: SearchMatch[];
// // }