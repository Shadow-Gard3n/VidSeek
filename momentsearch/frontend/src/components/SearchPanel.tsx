import React, { useState } from "react";
import { processVideo, resolveAssetUrl, searchVideo } from "../api/client";
import type {
  LLMProviderType,
  PipelineType,
  SearchMatch,
  VideoMetadata,
} from "../types";

interface SearchPanelProps {
  video: VideoMetadata | null;
  pipeline: PipelineType;
  onPipelineChange: (p: PipelineType) => void;
  onSelectTimestamp: (seconds: number) => void;
  onVideoUpdated: (video: VideoMetadata) => void;
}

export const SearchPanel: React.FC<SearchPanelProps> = ({
  video,
  pipeline,
  onPipelineChange,
  onSelectTimestamp,
  onVideoUpdated,
}) => {
  const [query, setQuery] = useState("");
  const [llmProvider, setLlmProvider] = useState<LLMProviderType>("ollama");
  const [llmModel, setLlmModel] = useState("");
  const [matches, setMatches] = useState<SearchMatch[]>([]);
  const [disclaimer, setDisclaimer] = useState<string | null>(null);
  const [searching, setSearching] = useState(false);
  const [reranking, setReranking] = useState(false);
  const [reprocessing, setReprocessing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const usesLlm =
    pipeline === "v3" || pipeline === "v4" || pipeline === "v5" || pipeline === "v6";

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!video || !query.trim()) return;

    setSearching(true);
    setReranking(false);
    setError(null);
    const cleanQuery = query.trim();

    try {
      const fastRes = await searchVideo(
        video.video_id,
        cleanQuery,
        pipeline,
        5,
        2.5,
        llmProvider,
        llmModel.trim() || undefined,
        false
      );
      setMatches(fastRes.results);
      setDisclaimer(fastRes.disclaimer);
      setSearching(false);

      if (usesLlm && llmProvider !== "none" && fastRes.results.length > 0) {
        setReranking(true);
        const rerankedRes = await searchVideo(
          video.video_id,
          cleanQuery,
          pipeline,
          5,
          2.5,
          llmProvider,
          llmModel.trim() || undefined,
          true
        );
        setMatches(rerankedRes.results);
        setDisclaimer(rerankedRes.disclaimer);
      }
    } catch (err: any) {
      setError(err?.response?.data?.detail || "Search failed.");
    } finally {
      setSearching(false);
      setReranking(false);
    }
  };

  const handleReprocessWithPipeline = async () => {
    if (!video) return;
    setReprocessing(true);
    setError(null);
    try {
      const updated = await processVideo(video.video_id, pipeline);
      onVideoUpdated(updated);
    } catch (err: any) {
      setError(err?.response?.data?.detail || "Re-processing failed.");
    } finally {
      setReprocessing(false);
    }
  };

  return (
    <div className="space-y-4 p-4 border rounded-lg bg-white">
      <div className="flex flex-wrap items-center justify-between gap-2 pb-2 border-b">
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-sm font-semibold">Search Pipeline:</span>
          <select
            value={pipeline}
            onChange={(e) => onPipelineChange(e.target.value as PipelineType)}
            className="border rounded px-2.5 py-1 text-sm bg-gray-50"
          >
            <option value="v1">Old Pipeline (v1 — CLIP)</option>
            <option value="v2">V2 Pipeline (CLIP + Grounding)</option>
            <option value="v3">V3 Pipeline (CLIP + NMS + LLM Rerank)</option>
            <option value="v4">V4 Pipeline (LLM Gate + CLIP/YOLO/OCR + Rerank)</option>
            <option value="v5">v5 (No-CLAHE Multi-Index)</option>
            <option value="v6">v6 (CLIP + LLM Boundary)</option>
          </select>

          {usesLlm && (
            <>
              <select
                value={llmProvider}
                onChange={(e) => setLlmProvider(e.target.value as LLMProviderType)}
                className="border rounded px-2.5 py-1 text-sm bg-gray-50"
              >
                <option value="ollama">LLM: Local Ollama</option>
                <option value="gemini">LLM: Gemini API</option>
                <option value="none">LLM: None (CLIP only)</option>
              </select>

              {llmProvider !== "none" && (
                <input
                  type="text"
                  value={llmModel}
                  onChange={(e) => setLlmModel(e.target.value)}
                  placeholder={
                    llmProvider === "ollama"
                      ? "Model (e.g., gemma3:4b)"
                      : "Model (e.g., gemini-2.5-flash)"
                  }
                  className="border rounded px-2.5 py-1 text-xs w-44"
                />
              )}
            </>
          )}
        </div>

        {video && (
          <button
            type="button"
            onClick={handleReprocessWithPipeline}
            disabled={reprocessing}
            className="text-xs px-3 py-1.5 rounded bg-gray-800 text-white disabled:opacity-50"
          >
            {reprocessing ? "Indexing..." : `Index Video with ${pipeline.toUpperCase()}`}
          </button>
        )}
      </div>

      <form onSubmit={handleSearch} className="flex gap-2">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          disabled={!video || video.status !== "completed" || searching || reranking}
          placeholder={
            !video
              ? "Select a video first..."
              : video.status !== "completed"
              ? `Video status: ${video.status}...`
              : "Search moments (e.g., person near counter, timestamp or sign)..."
          }
          className="flex-1 border rounded px-3 py-2 text-sm"
        />
        <button
          type="submit"
          disabled={
            !video ||
            video.status !== "completed" ||
            searching ||
            reranking ||
            !query.trim()
          }
          className="px-4 py-2 rounded bg-blue-600 text-white text-sm font-medium disabled:opacity-50"
        >
          {searching ? "Searching..." : reranking ? "LLM Verifying..." : "Search"}
        </button>
      </form>

      {error && <p className="text-xs text-red-600">{error}</p>}
      {disclaimer && <p className="text-xs text-gray-500 italic">{disclaimer}</p>}

      <div className="space-y-3">
        {matches.map((item) => (
          <div
            key={`${item.rank}-${item.start_time}`}
            onClick={() => onSelectTimestamp(item.start_time)}
            className={`p-3 border rounded-lg hover:border-blue-500 cursor-pointer transition space-y-2 ${
              item.llm_selected ? "border-emerald-500 bg-emerald-50/40" : ""
            }`}
          >
            <div className="flex items-center justify-between gap-2">
              <div className="flex items-center gap-3">
                <img
                  src={resolveAssetUrl(item.thumbnail_url)}
                  alt={`Match ${item.rank}`}
                  className="w-20 h-12 object-cover rounded"
                />
                <div>
                  <div className="font-semibold text-sm flex items-center gap-2">
                    <span>
                      #{item.rank} • {item.start_time}s – {item.end_time}s
                    </span>
                    {item.llm_selected && (
                      <span className="text-[11px] px-2 py-0.5 rounded bg-emerald-600 text-white font-semibold">
                        BEST MATCH ({llmProvider.toUpperCase()})
                      </span>
                    )}
                  </div>
                  <div className="text-xs text-gray-500">
                    Peak: {item.peak_timestamp}s • Score: {item.similarity_score}
                    {item.confidence != null
                      ? ` • Conf: ${(item.confidence * 100).toFixed(1)}%`
                      : ""}
                  </div>
                </div>
              </div>

              {item.ai_review_flag && (
                <span
                  className={`text-xs px-2 py-0.5 rounded font-semibold ${
                    item.ai_review_flag === "verified"
                      ? "bg-emerald-100 text-emerald-800"
                      : "bg-amber-100 text-amber-800"
                  }`}
                >
                  {item.ai_review_flag.replace("_", " ")}
                </span>
              )}
            </div>

            {item.detected_objects && item.detected_objects.length > 0 && (
              <div className="text-xs text-gray-700">
                <strong>Objects:</strong> {item.detected_objects.join(", ")}
              </div>
            )}

            {item.ocr_text && (
              <div className="text-xs text-gray-700">
                <strong>OCR:</strong> "{item.ocr_text}"
              </div>
            )}

            {item.llm_reason && (
              <div className="text-xs text-emerald-900 bg-emerald-100/70 rounded px-2.5 py-1.5">
                <strong>LLM Reason:</strong> {item.llm_reason}
              </div>
            )}

            {item.keyframes && item.keyframes.length > 0 && (
              <div className="flex gap-2 pt-1">
                {item.keyframes.map((kf, i) => (
                  <img
                    key={i}
                    src={resolveAssetUrl(kf)}
                    alt={`Keyframe ${i + 1}`}
                    className="w-16 h-10 object-cover rounded border"
                  />
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

// import React, { useState } from "react";
// import { processVideo, resolveAssetUrl, searchVideo } from "../api/client";
// import type {
//   LLMProviderType,
//   PipelineType,
//   SearchMatch,
//   VideoMetadata,
// } from "../types";

// interface SearchPanelProps {
//   video: VideoMetadata | null;
//   pipeline: PipelineType;
//   onPipelineChange: (p: PipelineType) => void;
//   onSelectTimestamp: (seconds: number) => void;
//   onVideoUpdated: (video: VideoMetadata) => void;
// }

// export const SearchPanel: React.FC<SearchPanelProps> = ({
//   video,
//   pipeline,
//   onPipelineChange,
//   onSelectTimestamp,
//   onVideoUpdated,
// }) => {
//   const [query, setQuery] = useState("");
//   const [llmProvider, setLlmProvider] = useState<LLMProviderType>("ollama");
//   const [llmModel, setLlmModel] = useState("");
//   const [matches, setMatches] = useState<SearchMatch[]>([]);
//   const [disclaimer, setDisclaimer] = useState<string | null>(null);
//   const [searching, setSearching] = useState(false);
//   const [reprocessing, setReprocessing] = useState(false);
//   const [error, setError] = useState<string | null>(null);

//   const usesLlm = pipeline === "v3" || pipeline === "v4" || pipeline === "v5";

//   const [reranking, setReranking] = useState(false);

// const handleSearch = async (e: React.FormEvent) => {
//   e.preventDefault();
//   if (!video || !query.trim()) return;

//   setSearching(true);
//   setReranking(false);
//   setError(null);
//   const cleanQuery = query.trim();

//   try {
//     // Phase 1: Instant retrieval & timeline fusion (< 1s)
//     const fastRes = await searchVideo(
//       video.video_id,
//       cleanQuery,
//       pipeline,
//       5,
//       2.5,
//       llmProvider,
//       llmModel.trim() || undefined,
//       false
//     );
//     setMatches(fastRes.results);
//     setDisclaimer(fastRes.disclaimer);
//     setSearching(false);

//     // Phase 2: Background VLM rerank if enabled
//     if (usesLlm && llmProvider !== "none" && fastRes.results.length > 0) {
//       setReranking(true);
//       const rerankedRes = await searchVideo(
//         video.video_id,
//         cleanQuery,
//         pipeline,
//         5,
//         2.5,
//         llmProvider,
//         llmModel.trim() || undefined,
//         true
//       );
//       setMatches(rerankedRes.results);
//       setDisclaimer(rerankedRes.disclaimer);
//     }
//   } catch (err: any) {
//     setError(err?.response?.data?.detail || "Search failed.");
//   } finally {
//     setSearching(false);
//     setReranking(false);
//   }
// };

//   // const handleSearch = async (e: React.FormEvent) => {
//   //   e.preventDefault();
//   //   if (!video || !query.trim()) return;

//   //   setSearching(true);
//   //   setError(null);
//   //   try {
//   //     const res = await searchVideo(
//   //       video.video_id,
//   //       query.trim(),
//   //       pipeline,
//   //       5,
//   //       2.5,
//   //       llmProvider,
//   //       llmModel.trim() || undefined
//   //     );
//   //     setMatches(res.results);
//   //     setDisclaimer(res.disclaimer);
//   //   } catch (err: any) {
//   //     setError(err?.response?.data?.detail || "Search failed.");
//   //   } finally {
//   //     setSearching(false);
//   //   }
//   // };

//   const handleReprocessWithPipeline = async () => {
//     if (!video) return;
//     setReprocessing(true);
//     setError(null);
//     try {
//       const updated = await processVideo(video.video_id, pipeline);
//       onVideoUpdated(updated);
//     } catch (err: any) {
//       setError(err?.response?.data?.detail || "Re-processing failed.");
//     } finally {
//       setReprocessing(false);
//     }
//   };

//   return (
//     <div className="space-y-4 p-4 border rounded-lg bg-white">
//       {/* Pipeline & LLM Controls */}
//       <div className="flex flex-wrap items-center justify-between gap-2 pb-2 border-b">
//         <div className="flex flex-wrap items-center gap-2">
//           <span className="text-sm font-semibold">Search Pipeline:</span>
//           <select
//             value={pipeline}
//             onChange={(e) => onPipelineChange(e.target.value as PipelineType)}
//             className="border rounded px-2.5 py-1 text-sm bg-gray-50"
//           >
//             <option value="v1">Old Pipeline (v1 — CLIP)</option>
//             <option value="v2">V2 Pipeline (CLIP + Grounding)</option>
//             <option value="v3">V3 Pipeline (CLIP + NMS + LLM Rerank)</option>
//             <option value="v4">V4 Pipeline (LLM Gate + CLIP/YOLO/OCR + Rerank)</option>
//             <option value="v5">v5 (No-CLAHE Multi-Index)</option>
//           </select>

//           {usesLlm && (
//             <>
//               <select
//                 value={llmProvider}
//                 onChange={(e) => setLlmProvider(e.target.value as LLMProviderType)}
//                 className="border rounded px-2.5 py-1 text-sm bg-gray-50"
//               >
//                 <option value="ollama">LLM: Local Ollama</option>
//                 <option value="gemini">LLM: Gemini API</option>
//                 <option value="none">LLM: None (Heuristic/CLIP only)</option>
//               </select>

//               {llmProvider !== "none" && (
//                 <input
//                   type="text"
//                   value={llmModel}
//                   onChange={(e) => setLlmModel(e.target.value)}
//                   placeholder={
//                     llmProvider === "ollama"
//                       ? "Model (e.g., gemma3:4b)"
//                       : "Model (e.g., gemini-2.5-flash)"
//                   }
//                   className="border rounded px-2.5 py-1 text-xs w-44"
//                 />
//               )}
//             </>
//           )}
//         </div>

//         {video && (
//           <button
//             type="button"
//             onClick={handleReprocessWithPipeline}
//             disabled={reprocessing}
//             className="text-xs px-3 py-1.5 rounded bg-gray-800 text-white disabled:opacity-50"
//           >
//             {reprocessing ? "Indexing..." : `Index Video with ${pipeline.toUpperCase()}`}
//           </button>
//         )}
//       </div>

//       {/* Search Form */}
//       <form onSubmit={handleSearch} className="flex gap-2">
//         <input
//           type="text"
//           value={query}
//           onChange={(e) => setQuery(e.target.value)}
//           disabled={!video || video.status !== "completed" || searching}
//           placeholder={
//             !video
//               ? "Select a video first..."
//               : video.status !== "completed"
//               ? `Video status: ${video.status}...`
//               : "Search moments (e.g., person near counter, timestamp or sign)..."
//           }
//           className="flex-1 border rounded px-3 py-2 text-sm"
//         />
//         <button
//           type="submit"
//           disabled={!video || video.status !== "completed" || searching || !query.trim()}
//           className="px-4 py-2 rounded bg-blue-600 text-white text-sm font-medium disabled:opacity-50"
//         >
//           {searching ? "Searching..." : "Search"}
//         </button>
//       </form>

//       {error && <p className="text-xs text-red-600">{error}</p>}
//       {disclaimer && <p className="text-xs text-gray-500 italic">{disclaimer}</p>}

//       {/* Results Timeline */}
//       <div className="space-y-3">
//         {matches.map((item) => (
//           <div
//             key={`${item.rank}-${item.start_time}`}
//             onClick={() => onSelectTimestamp(item.start_time)}
//             className={`p-3 border rounded-lg hover:border-blue-500 cursor-pointer transition space-y-2 ${
//               item.llm_selected ? "border-emerald-500 bg-emerald-50/40" : ""
//             }`}
//           >
//             <div className="flex items-center justify-between gap-2">
//               <div className="flex items-center gap-3">
//                 <img
//                   src={resolveAssetUrl(item.thumbnail_url)}
//                   alt={`Match ${item.rank}`}
//                   className="w-20 h-12 object-cover rounded"
//                 />
//                 <div>
//                   <div className="font-semibold text-sm flex items-center gap-2">
//                     <span>
//                       #{item.rank} • {item.start_time}s – {item.end_time}s
//                     </span>
//                     {item.llm_selected && (
//                       <span className="text-[11px] px-2 py-0.5 rounded bg-emerald-600 text-white font-semibold">
//                         BEST MATCH ({llmProvider.toUpperCase()})
//                       </span>
//                     )}
//                   </div>
//                   <div className="text-xs text-gray-500">
//                     Peak: {item.peak_timestamp}s • Score: {item.similarity_score}
//                     {item.confidence != null
//                       ? ` • Conf: ${(item.confidence * 100).toFixed(1)}%`
//                       : ""}
//                   </div>
//                 </div>
//               </div>

//               {item.ai_review_flag && (
//                 <span
//                   className={`text-xs px-2 py-0.5 rounded font-semibold ${
//                     item.ai_review_flag === "verified"
//                       ? "bg-emerald-100 text-emerald-800"
//                       : "bg-amber-100 text-amber-800"
//                   }`}
//                 >
//                   {item.ai_review_flag.replace("_", " ")}
//                 </span>
//               )}
//             </div>

//             {item.detected_objects && item.detected_objects.length > 0 && (
//               <div className="text-xs text-gray-700">
//                 <strong>Objects:</strong> {item.detected_objects.join(", ")}
//               </div>
//             )}

//             {item.ocr_text && (
//               <div className="text-xs text-gray-700">
//                 <strong>OCR:</strong> "{item.ocr_text}"
//               </div>
//             )}

//             {item.llm_reason && (
//               <div className="text-xs text-emerald-900 bg-emerald-100/70 rounded px-2.5 py-1.5">
//                 <strong>LLM Reason:</strong> {item.llm_reason}
//               </div>
//             )}

//             {item.keyframes && item.keyframes.length > 0 && (
//               <div className="flex gap-2 pt-1">
//                 {item.keyframes.map((kf, i) => (
//                   <img
//                     key={i}
//                     src={resolveAssetUrl(kf)}
//                     alt={`Keyframe ${i + 1}`}
//                     className="w-16 h-10 object-cover rounded border"
//                   />
//                 ))}
//               </div>
//             )}
//           </div>
//         ))}
//       </div>
//     </div>
//   );
// };

// import React, { useState } from "react";
// import { processVideo, resolveAssetUrl, searchVideo } from "../api/client";
// import type {
//   LLMProviderType,
//   PipelineType,
//   SearchMatch,
//   VideoMetadata,
// } from "../types";

// interface SearchPanelProps {
//   video: VideoMetadata | null;
//   pipeline: PipelineType;
//   onPipelineChange: (p: PipelineType) => void;
//   onSelectTimestamp: (seconds: number) => void;
//   onVideoUpdated: (video: VideoMetadata) => void;
// }

// export const SearchPanel: React.FC<SearchPanelProps> = ({
//   video,
//   pipeline,
//   onPipelineChange,
//   onSelectTimestamp,
//   onVideoUpdated,
// }) => {
//   const [query, setQuery] = useState("");
//   const [llmProvider, setLlmProvider] = useState<LLMProviderType>("ollama");
//   const [llmModel, setLlmModel] = useState("");
//   const [matches, setMatches] = useState<SearchMatch[]>([]);
//   const [disclaimer, setDisclaimer] = useState<string | null>(null);
//   const [searching, setSearching] = useState(false);
//   const [reprocessing, setReprocessing] = useState(false);
//   const [error, setError] = useState<string | null>(null);

//   const usesLlm = pipeline === "v3" || pipeline === "v4";

//   const handleSearch = async (e: React.FormEvent) => {
//     e.preventDefault();
//     if (!video || !query.trim()) return;

//     setSearching(true);
//     setError(null);
//     try {
//       const res = await searchVideo(
//         video.video_id,
//         query.trim(),
//         pipeline,
//         5,
//         2.5,
//         llmProvider,
//         llmModel.trim() || undefined
//       );
//       setMatches(res.results);
//       setDisclaimer(res.disclaimer);
//     } catch (err: any) {
//       setError(err?.response?.data?.detail || "Search failed.");
//     } finally {
//       setSearching(false);
//     }
//   };

//   const handleReprocessWithPipeline = async () => {
//     if (!video) return;
//     setReprocessing(true);
//     setError(null);
//     try {
//       const updated = await processVideo(video.video_id, pipeline);
//       onVideoUpdated(updated);
//     } catch (err: any) {
//       setError(err?.response?.data?.detail || "Re-processing failed.");
//     } finally {
//       setReprocessing(false);
//     }
//   };

//   return (
//     <div className="space-y-4 p-4 border rounded-lg bg-white">
//       {/* Pipeline & LLM Controls */}
//       <div className="flex flex-wrap items-center justify-between gap-2 pb-2 border-b">
//         <div className="flex flex-wrap items-center gap-2">
//           <span className="text-sm font-semibold">Search Pipeline:</span>
//           <select
//             value={pipeline}
//             onChange={(e) => onPipelineChange(e.target.value as PipelineType)}
//             className="border rounded px-2.5 py-1 text-sm bg-gray-50"
//           >
//             <option value="v1">Old Pipeline (v1 — CLIP)</option>
//             <option value="v2">V2 Pipeline (CLIP + Grounding)</option>
//             <option value="v3">V3 Pipeline (CLIP + NMS + LLM Rerank)</option>
//             <option value="v4">New Pipeline (v4 — LLM Gate + CLIP/YOLO/OCR + Rerank)</option>
//           </select>

//           {usesLlm && (
//             <>
//               <select
//                 value={llmProvider}
//                 onChange={(e) => setLlmProvider(e.target.value as LLMProviderType)}
//                 className="border rounded px-2.5 py-1 text-sm bg-gray-50"
//               >
//                 <option value="ollama">LLM: Local Ollama</option>
//                 <option value="gemini">LLM: Gemini API</option>
//                 <option value="none">LLM: None (Heuristic/CLIP only)</option>
//               </select>

//               {llmProvider !== "none" && (
//                 <input
//                   type="text"
//                   value={llmModel}
//                   onChange={(e) => setLlmModel(e.target.value)}
//                   placeholder={
//                     llmProvider === "ollama"
//                       ? "Model (e.g., gemma3:4b)"
//                       : "Model (e.g., gemini-2.5-flash)"
//                   }
//                   className="border rounded px-2.5 py-1 text-xs w-44"
//                 />
//               )}
//             </>
//           )}
//         </div>

//         {video && (
//           <button
//             type="button"
//             onClick={handleReprocessWithPipeline}
//             disabled={reprocessing}
//             className="text-xs px-3 py-1.5 rounded bg-gray-800 text-white disabled:opacity-50"
//           >
//             {reprocessing ? "Indexing..." : `Index Video with ${pipeline.toUpperCase()}`}
//           </button>
//         )}
//       </div>

//       {/* Search Form */}
//       <form onSubmit={handleSearch} className="flex gap-2">
//         <input
//           type="text"
//           value={query}
//           onChange={(e) => setQuery(e.target.value)}
//           disabled={!video || video.status !== "completed" || searching}
//           placeholder={
//             !video
//               ? "Select a video first..."
//               : video.status !== "completed"
//               ? `Video status: ${video.status}...`
//               : "Search moments (e.g., person near counter, timestamp or sign)..."
//           }
//           className="flex-1 border rounded px-3 py-2 text-sm"
//         />
//         <button
//           type="submit"
//           disabled={!video || video.status !== "completed" || searching || !query.trim()}
//           className="px-4 py-2 rounded bg-blue-600 text-white text-sm font-medium disabled:opacity-50"
//         >
//           {searching ? "Searching..." : "Search"}
//         </button>
//       </form>

//       {error && <p className="text-xs text-red-600">{error}</p>}
//       {disclaimer && <p className="text-xs text-gray-500 italic">{disclaimer}</p>}

//       {/* Results Timeline */}
//       <div className="space-y-3">
//         {matches.map((item) => (
//           <div
//             key={`${item.rank}-${item.start_time}`}
//             onClick={() => onSelectTimestamp(item.start_time)}
//             className={`p-3 border rounded-lg hover:border-blue-500 cursor-pointer transition space-y-2 ${
//               item.llm_selected ? "border-emerald-500 bg-emerald-50/40" : ""
//             }`}
//           >
//             <div className="flex items-center justify-between gap-2">
//               <div className="flex items-center gap-3">
//                 <img
//                   src={resolveAssetUrl(item.thumbnail_url)}
//                   alt={`Match ${item.rank}`}
//                   className="w-20 h-12 object-cover rounded"
//                 />
//                 <div>
//                   <div className="font-semibold text-sm flex items-center gap-2">
//                     <span>
//                       #{item.rank} • {item.start_time}s – {item.end_time}s
//                     </span>
//                     {item.llm_selected && (
//                       <span className="text-[11px] px-2 py-0.5 rounded bg-emerald-600 text-white font-semibold">
//                         BEST MATCH ({llmProvider.toUpperCase()})
//                       </span>
//                     )}
//                   </div>
//                   <div className="text-xs text-gray-500">
//                     Peak: {item.peak_timestamp}s • Score: {item.similarity_score}
//                     {item.confidence != null
//                       ? ` • Conf: ${(item.confidence * 100).toFixed(1)}%`
//                       : ""}
//                   </div>
//                 </div>
//               </div>

//               {item.ai_review_flag && (
//                 <span
//                   className={`text-xs px-2 py-0.5 rounded font-semibold ${
//                     item.ai_review_flag === "verified"
//                       ? "bg-emerald-100 text-emerald-800"
//                       : "bg-amber-100 text-amber-800"
//                   }`}
//                 >
//                   {item.ai_review_flag.replace("_", " ")}
//                 </span>
//               )}
//             </div>

//             {item.detected_objects && item.detected_objects.length > 0 && (
//               <div className="text-xs text-gray-700">
//                 <strong>Objects:</strong> {item.detected_objects.join(", ")}
//               </div>
//             )}

//             {item.ocr_text && (
//               <div className="text-xs text-gray-700">
//                 <strong>OCR:</strong> "{item.ocr_text}"
//               </div>
//             )}

//             {item.llm_reason && (
//               <div className="text-xs text-emerald-900 bg-emerald-100/70 rounded px-2.5 py-1.5">
//                 <strong>LLM Reason:</strong> {item.llm_reason}
//               </div>
//             )}

//             {item.keyframes && item.keyframes.length > 0 && (
//               <div className="flex gap-2 pt-1">
//                 {item.keyframes.map((kf, i) => (
//                   <img
//                     key={i}
//                     src={resolveAssetUrl(kf)}
//                     alt={`Keyframe ${i + 1}`}
//                     className="w-16 h-10 object-cover rounded border"
//                   />
//                 ))}
//               </div>
//             )}
//           </div>
//         ))}
//       </div>
//     </div>
//   );
// };

// import React, { useState } from "react";
// import { processVideo, resolveAssetUrl, searchVideo } from "../api/client";
// import type {
//   LLMProviderType,
//   PipelineType,
//   SearchMatch,
//   VideoMetadata,
// } from "../types";

// interface SearchPanelProps {
//   video: VideoMetadata | null;
//   pipeline: PipelineType;
//   onPipelineChange: (p: PipelineType) => void;
//   onSelectTimestamp: (seconds: number) => void;
//   onVideoUpdated: (video: VideoMetadata) => void;
// }

// export const SearchPanel: React.FC<SearchPanelProps> = ({
//   video,
//   pipeline,
//   onPipelineChange,
//   onSelectTimestamp,
//   onVideoUpdated,
// }) => {
//   const [query, setQuery] = useState("");
//   const [llmProvider, setLlmProvider] = useState<LLMProviderType>("ollama");
//   const [llmModel, setLlmModel] = useState("");
//   const [matches, setMatches] = useState<SearchMatch[]>([]);
//   const [disclaimer, setDisclaimer] = useState<string | null>(null);
//   const [searching, setSearching] = useState(false);
//   const [reprocessing, setReprocessing] = useState(false);
//   const [error, setError] = useState<string | null>(null);

//   const handleSearch = async (e: React.FormEvent) => {
//     e.preventDefault();
//     if (!video || !query.trim()) return;

//     setSearching(true);
//     setError(null);
//     try {
//       const res = await searchVideo(
//         video.video_id,
//         query.trim(),
//         pipeline,
//         5,
//         2.5,
//         llmProvider,
//         llmModel.trim() || undefined
//       );
//       setMatches(res.results);
//       setDisclaimer(res.disclaimer);
//     } catch (err: any) {
//       setError(err?.response?.data?.detail || "Search failed.");
//     } finally {
//       setSearching(false);
//     }
//   };

//   const handleReprocessWithPipeline = async () => {
//     if (!video) return;
//     setReprocessing(true);
//     setError(null);
//     try {
//       const updated = await processVideo(video.video_id, pipeline);
//       onVideoUpdated(updated);
//     } catch (err: any) {
//       setError(err?.response?.data?.detail || "Re-processing failed.");
//     } finally {
//       setReprocessing(false);
//     }
//   };

//   return (
//     <div className="space-y-4 p-4 border rounded-lg bg-white">
//       {/* Pipeline & LLM Reranker Controls */}
//       <div className="flex flex-wrap items-center justify-between gap-2 pb-2 border-b">
//         <div className="flex flex-wrap items-center gap-2">
//           <span className="text-sm font-semibold">Search Pipeline:</span>
//           <select
//             value={pipeline}
//             onChange={(e) => onPipelineChange(e.target.value as PipelineType)}
//             className="border rounded px-2.5 py-1 text-sm bg-gray-50"
//           >
//             <option value="v1">Old Pipeline (v1 — CLIP)</option>
//             <option value="v2">V2 Pipeline (CLIP + Grounding)</option>
//             <option value="v3">New Pipeline (v3 — CLIP + NMS + LLM Rerank)</option>
//           </select>

//           {pipeline === "v3" && (
//             <>
//               <select
//                 value={llmProvider}
//                 onChange={(e) => setLlmProvider(e.target.value as LLMProviderType)}
//                 className="border rounded px-2.5 py-1 text-sm bg-gray-50"
//               >
//                 <option value="ollama">LLM: Local Ollama</option>
//                 <option value="gemini">LLM: Gemini API</option>
//                 <option value="none">LLM: None (CLIP + NMS only)</option>
//               </select>

//               {llmProvider !== "none" && (
//                 <input
//                   type="text"
//                   value={llmModel}
//                   onChange={(e) => setLlmModel(e.target.value)}
//                   placeholder={
//                     llmProvider === "ollama" ? "Model (e.g., qwen2.5vl:7b)" : "Model (e.g., gemini-2.5-flash)"
//                   }
//                   className="border rounded px-2.5 py-1 text-xs w-44"
//                 />
//               )}
//             </>
//           )}
//         </div>

//         {video && (
//           <button
//             type="button"
//             onClick={handleReprocessWithPipeline}
//             disabled={reprocessing}
//             className="text-xs px-3 py-1.5 rounded bg-gray-800 text-white disabled:opacity-50"
//           >
//             {reprocessing ? "Indexing..." : `Index Video with ${pipeline.toUpperCase()}`}
//           </button>
//         )}
//       </div>

//       {/* Search Form */}
//       <form onSubmit={handleSearch} className="flex gap-2">
//         <input
//           type="text"
//           value={query}
//           onChange={(e) => setQuery(e.target.value)}
//           disabled={!video || video.status !== "completed" || searching}
//           placeholder={
//             !video
//               ? "Select a video first..."
//               : video.status !== "completed"
//               ? `Video status: ${video.status}...`
//               : "Search moments (e.g., person near counter)..."
//           }
//           className="flex-1 border rounded px-3 py-2 text-sm"
//         />
//         <button
//           type="submit"
//           disabled={!video || video.status !== "completed" || searching || !query.trim()}
//           className="px-4 py-2 rounded bg-blue-600 text-white text-sm font-medium disabled:opacity-50"
//         >
//           {searching ? "Searching..." : "Search"}
//         </button>
//       </form>

//       {error && <p className="text-xs text-red-600">{error}</p>}
//       {disclaimer && <p className="text-xs text-gray-500 italic">{disclaimer}</p>}

//       {/* Results Timeline */}
//       <div className="space-y-3">
//         {matches.map((item) => (
//           <div
//             key={`${item.rank}-${item.start_time}`}
//             onClick={() => onSelectTimestamp(item.start_time)}
//             className={`p-3 border rounded-lg hover:border-blue-500 cursor-pointer transition space-y-2 ${
//               item.llm_selected ? "border-emerald-500 bg-emerald-50/40" : ""
//             }`}
//           >
//             <div className="flex items-center justify-between gap-2">
//               <div className="flex items-center gap-3">
//                 <img
//                   src={resolveAssetUrl(item.thumbnail_url)}
//                   alt={`Match ${item.rank}`}
//                   className="w-20 h-12 object-cover rounded"
//                 />
//                 <div>
//                   <div className="font-semibold text-sm flex items-center gap-2">
//                     <span>
//                       #{item.rank} • {item.start_time}s – {item.end_time}s
//                     </span>
//                     {item.llm_selected && (
//                       <span className="text-[11px] px-2 py-0.5 rounded bg-emerald-600 text-white font-semibold">
//                         BEST MATCH ({llmProvider.toUpperCase()})
//                       </span>
//                     )}
//                   </div>
//                   <div className="text-xs text-gray-500">
//                     Peak: {item.peak_timestamp}s • CLIP Sim: {item.similarity_score}
//                     {item.confidence != null ? ` • Conf: ${(item.confidence * 100).toFixed(1)}%` : ""}
//                   </div>
//                 </div>
//               </div>

//               {item.ai_review_flag && (
//                 <span
//                   className={`text-xs px-2 py-0.5 rounded font-semibold ${
//                     item.ai_review_flag === "verified"
//                       ? "bg-emerald-100 text-emerald-800"
//                       : "bg-amber-100 text-amber-800"
//                   }`}
//                 >
//                   {item.ai_review_flag.replace("_", " ")}
//                 </span>
//               )}
//             </div>

//             {item.llm_reason && (
//               <div className="text-xs text-emerald-900 bg-emerald-100/70 rounded px-2.5 py-1.5">
//                 <strong>LLM Reason:</strong> {item.llm_reason}
//               </div>
//             )}

//             {item.keyframes && item.keyframes.length > 0 && (
//               <div className="flex gap-2 pt-1">
//                 {item.keyframes.map((kf, i) => (
//                   <img
//                     key={i}
//                     src={resolveAssetUrl(kf)}
//                     alt={`Keyframe ${i + 1}`}
//                     className="w-16 h-10 object-cover rounded border"
//                   />
//                 ))}
//               </div>
//             )}
//           </div>
//         ))}
//       </div>
//     </div>
//   );
// };

// import React, { useState } from "react";
// import { processVideo, resolveAssetUrl, searchVideo } from "../api/client";
// import type { PipelineType, SearchMatch, VideoMetadata } from "../types";

// interface SearchPanelProps {
//   video: VideoMetadata | null;
//   pipeline: PipelineType;
//   onPipelineChange: (p: PipelineType) => void;
//   onSelectTimestamp: (seconds: number) => void;
//   onVideoUpdated: (video: VideoMetadata) => void;
// }

// export const SearchPanel: React.FC<SearchPanelProps> = ({
//   video,
//   pipeline,
//   onPipelineChange,
//   onSelectTimestamp,
//   onVideoUpdated,
// }) => {
//   const [query, setQuery] = useState("");
//   const [matches, setMatches] = useState<SearchMatch[]>([]);
//   const [disclaimer, setDisclaimer] = useState<string | null>(null);
//   const [searching, setSearching] = useState(false);
//   const [reprocessing, setReprocessing] = useState(false);
//   const [error, setError] = useState<string | null>(null);

//   const handleSearch = async (e: React.FormEvent) => {
//     e.preventDefault();
//     if (!video || !query.trim()) return;

//     setSearching(true);
//     setError(null);
//     try {
//       const res = await searchVideo(video.video_id, query.trim(), pipeline);
//       setMatches(res.results);
//       setDisclaimer(res.disclaimer);
//     } catch (err: any) {
//       setError(err?.response?.data?.detail || "Search failed.");
//     } finally {
//       setSearching(false);
//     }
//   };

//   const handleReprocessWithPipeline = async () => {
//     if (!video) return;
//     setReprocessing(true);
//     setError(null);
//     try {
//       const updated = await processVideo(video.video_id, pipeline);
//       onVideoUpdated(updated);
//     } catch (err: any) {
//       setError(err?.response?.data?.detail || "Re-processing failed.");
//     } finally {
//       setReprocessing(false);
//     }
//   };

//   return (
//     <div className="space-y-4 p-4 border rounded-lg bg-white">
//       {/* Pipeline Toggle + Re-index Button */}
//       <div className="flex flex-wrap items-center justify-between gap-2 pb-2 border-b">
//         <div className="flex items-center gap-2">
//           <span className="text-sm font-semibold">Search Pipeline:</span>
//           <select
//             value={pipeline}
//             onChange={(e) => onPipelineChange(e.target.value as PipelineType)}
//             className="border rounded px-2.5 py-1 text-sm bg-gray-50"
//           >
//             <option value="v1">Old Pipeline (v1 — CLIP)</option>
//             <option value="v2">New Pipeline (v2 — ChronoVision)</option>
//           </select>
//         </div>

//         {video && (
//           <button
//             type="button"
//             onClick={handleReprocessWithPipeline}
//             disabled={reprocessing}
//             className="text-xs px-3 py-1.5 rounded bg-gray-800 text-white disabled:opacity-50"
//           >
//             {reprocessing ? "Indexing..." : `Index Video with ${pipeline.toUpperCase()}`}
//           </button>
//         )}
//       </div>

//       {/* Search Form */}
//       <form onSubmit={handleSearch} className="flex gap-2">
//         <input
//           type="text"
//           value={query}
//           onChange={(e) => setQuery(e.target.value)}
//           disabled={!video || video.status !== "completed" || searching}
//           placeholder={
//             !video
//               ? "Select a video first..."
//               : video.status !== "completed"
//               ? `Video status: ${video.status}...`
//               : "Search moments (e.g., person near EXIT sign)..."
//           }
//           className="flex-1 border rounded px-3 py-2 text-sm"
//         />
//         <button
//           type="submit"
//           disabled={!video || video.status !== "completed" || searching || !query.trim()}
//           className="px-4 py-2 rounded bg-blue-600 text-white text-sm font-medium disabled:opacity-50"
//         >
//           {searching ? "Searching..." : "Search"}
//         </button>
//       </form>

//       {error && <p className="text-xs text-red-600">{error}</p>}
//       {disclaimer && <p className="text-xs text-gray-500 italic">{disclaimer}</p>}

//       {/* Results Timeline */}
//       <div className="space-y-3">
//         {matches.map((item) => (
//           <div
//             key={`${item.rank}-${item.start_time}`}
//             onClick={() => onSelectTimestamp(item.start_time)}
//             className="p-3 border rounded-lg hover:border-blue-500 cursor-pointer transition space-y-2"
//           >
//             <div className="flex items-center justify-between gap-2">
//               <div className="flex items-center gap-3">
//                 <img
//                   src={resolveAssetUrl(item.thumbnail_url)}
//                   alt={`Match ${item.rank}`}
//                   className="w-20 h-12 object-cover rounded"
//                 />
//                 <div>
//                   <div className="font-semibold text-sm">
//                     #{item.rank} • {item.start_time}s – {item.end_time}s
//                   </div>
//                   <div className="text-xs text-gray-500">
//                     Peak: {item.peak_timestamp}s • Score: {item.similarity_score}
//                     {item.confidence != null ? ` • Conf: ${(item.confidence * 100).toFixed(1)}%` : ""}
//                   </div>
//                 </div>
//               </div>

//               {item.ai_review_flag && (
//                 <span
//                   className={`text-xs px-2 py-0.5 rounded font-semibold ${
//                     item.ai_review_flag === "verified"
//                       ? "bg-emerald-100 text-emerald-800"
//                       : "bg-amber-100 text-amber-800"
//                   }`}
//                 >
//                   {item.ai_review_flag.replace("_", " ")}
//                 </span>
//               )}
//             </div>

//             {/* V2 Multimodal Evidence */}
//             {(item.detected_objects?.length || item.ocr_text || item.keyframes?.length) && (
//               <div className="text-xs text-gray-700 space-y-1 pt-1 border-t">
//                 {item.detected_objects && item.detected_objects.length > 0 && (
//                   <div>
//                     <strong>Objects:</strong> {item.detected_objects.join(", ")}
//                   </div>
//                 )}
//                 {item.ocr_text && (
//                   <div>
//                     <strong>OCR:</strong> "{item.ocr_text}"
//                   </div>
//                 )}
//                 {item.keyframes && item.keyframes.length > 0 && (
//                   <div className="flex gap-2 pt-1">
//                     {item.keyframes.map((kf, i) => (
//                       <img
//                         key={i}
//                         src={resolveAssetUrl(kf)}
//                         alt={`Keyframe ${i + 1}`}
//                         className="w-16 h-10 object-cover rounded border"
//                       />
//                     ))}
//                   </div>
//                 )}
//               </div>
//             )}
//           </div>
//         ))}
//       </div>
//     </div>
//   );
// };

// // import { Play, Search, ShieldAlert } from 'lucide-react';
// // import { getThumbnailUrl } from '../api/client';
// // import type { SearchMatch } from '../types';
// // import { formatTimestamp } from './VideoPlayer';

// // interface SearchPanelProps {
// //   matches: SearchMatch[];
// //   disclaimer: string | null;
// //   activeRank: number | null;
// //   onSelectMatch: (match: SearchMatch) => void;
// // }

// // export default function SearchPanel({
// //   matches,
// //   disclaimer,
// //   activeRank,
// //   onSelectMatch,
// // }: SearchPanelProps) {
// //   return (
// //     <div className="flex flex-col gap-3 h-full">
// //       <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-400">
// //         Retrieved Moments ({matches.length})
// //       </h2>

// //       {disclaimer && (
// //         <div className="p-3 rounded-xl bg-amber-950/30 border border-amber-800/50 flex items-start gap-2.5 text-xs text-amber-200/90">
// //           <ShieldAlert className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
// //           <p>{disclaimer}</p>
// //         </div>
// //       )}

// //       {matches.length === 0 ? (
// //         <div className="flex-1 flex flex-col items-center justify-center text-center text-slate-500 px-6 py-12">
// //           <Search className="w-8 h-8 text-slate-700 mb-2" />
// //           <p className="text-sm text-slate-400">No retrieved moments yet</p>
// //           <p className="text-xs text-slate-500 mt-1">
// //             Enter a natural-language event description below the video player to search.
// //           </p>
// //         </div>
// //       ) : (
// //         <div className="space-y-3 overflow-y-auto pr-1">
// //           {matches.map((match) => {
// //             const isActive = activeRank === match.rank;
// //             return (
// //               <div
// //                 key={match.rank}
// //                 onClick={() => onSelectMatch(match)}
// //                 className={`group rounded-xl border p-3 transition cursor-pointer flex gap-3 ${
// //                   isActive
// //                     ? 'bg-indigo-950/40 border-indigo-500'
// //                     : 'bg-slate-900/80 border-slate-800 hover:border-slate-700'
// //                 }`}
// //               >
// //                 <div className="relative w-32 h-20 rounded-lg overflow-hidden bg-black shrink-0 border border-slate-800">
// //                   <img
// //                     src={getThumbnailUrl(match.thumbnail_url)}
// //                     alt={`Match #${match.rank}`}
// //                     className="w-full h-full object-cover group-hover:scale-105 transition duration-200"
// //                   />
// //                   <span className="absolute top-1 left-1 px-1.5 py-0.5 rounded bg-black/75 text-[10px] font-mono text-white">
// //                     #{match.rank}
// //                   </span>
// //                 </div>

// //                 <div className="flex-1 min-w-0 flex flex-col justify-between">
// //                   <div>
// //                     <div className="flex items-center justify-between gap-2">
// //                       <span className="font-mono text-xs font-semibold text-indigo-300">
// //                         {formatTimestamp(match.start_time)} – {formatTimestamp(match.end_time)}
// //                       </span>
// //                       <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-300">
// //                         Sim: {match.similarity_score.toFixed(3)}
// //                       </span>
// //                     </div>
// //                     <p className="text-[11px] text-slate-400 mt-1">
// //                       Peak frame at {formatTimestamp(match.peak_timestamp)}
// //                     </p>
// //                   </div>

// //                   <button
// //                     type="button"
// //                     onClick={(e) => {
// //                       e.stopPropagation();
// //                       onSelectMatch(match);
// //                     }}
// //                     className="mt-2 w-full py-1.5 px-3 rounded-lg bg-slate-800 group-hover:bg-indigo-600 text-xs font-medium text-slate-200 group-hover:text-white flex items-center justify-center gap-1.5 transition"
// //                   >
// //                     <Play className="w-3.5 h-3.5 fill-current" /> Play from{' '}
// //                     {formatTimestamp(match.start_time)}
// //                   </button>
// //                 </div>
// //               </div>
// //             );
// //           })}
// //         </div>
// //       )}
// //     </div>
// //   );
// // }