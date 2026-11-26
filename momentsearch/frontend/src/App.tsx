import { useEffect, useState } from "react";
import { getVideoStatus, listVideos } from "./api/client";
import { SearchPanel } from "./components/SearchPanel";
import VideoPlayer from "./components/VideoPlayer";
import { VideoUpload } from "./components/VideoUpload";
import type { PipelineType, VideoMetadata } from "./types";

function formatPipelineLabel(pipeline: PipelineType): string {
  if (pipeline === "v6") return "v6 (CLIP + LLM Boundary)";
  if (pipeline === "v5") return "v5 (No-CLAHE Multi-Index)";
  if (pipeline === "v4") return "New (v4 — LLM Gate + CLIP/YOLO/OCR + Rerank)";
  if (pipeline === "v3") return "V3 (CLIP + NMS + LLM Rerank)";
  if (pipeline === "v2") return "V2 (CLIP + Temporal Grounding)";
  return "Old (v1 — CLIP)";
}

export default function App() {
  const [pipeline, setPipeline] = useState<PipelineType>("v6");
  const [videos, setVideos] = useState<VideoMetadata[]>([]);
  const [selectedVideo, setSelectedVideo] = useState<VideoMetadata | null>(null);
  const [seekTime, setSeekTime] = useState<number | null>(null);

  const refreshVideos = async () => {
    const items = await listVideos();
    setVideos(items);
    if (!selectedVideo && items.length > 0) {
      setSelectedVideo(items[0]);
    } else if (selectedVideo) {
      const fresh = items.find((item) => item.video_id === selectedVideo.video_id);
      if (fresh) setSelectedVideo(fresh);
    }
  };

  useEffect(() => {
    refreshVideos();
  }, []);

  useEffect(() => {
    if (!selectedVideo || selectedVideo.status !== "processing") return;
    const timer = setInterval(async () => {
      const updated = await getVideoStatus(selectedVideo.video_id);
      setSelectedVideo(updated);
      setVideos((prev) =>
        prev.map((v) => (v.video_id === updated.video_id ? updated : v))
      );
    }, 2000);
    return () => clearInterval(timer);
  }, [selectedVideo?.video_id, selectedVideo?.status]);

  const handleVideoUpdated = (updated: VideoMetadata) => {
    setSelectedVideo(updated);
    setVideos((prev) =>
      prev.map((v) => (v.video_id === updated.video_id ? updated : v))
    );
  };

  return (
    <div className="min-h-screen bg-gray-50 p-6 max-w-6xl mx-auto space-y-6">
      <header className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">ChronoVision / MomentSearch</h1>
          <p className="text-sm text-gray-600">
            Active Pipeline: <strong>{formatPipelineLabel(pipeline)}</strong>
          </p>
        </div>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="space-y-4">
          <VideoUpload
            pipeline={pipeline}
            onPipelineChange={setPipeline}
            onUploaded={(vid) => {
              setVideos((prev) => [vid, ...prev]);
              setSelectedVideo(vid);
            }}
          />

          <div className="p-4 border rounded-lg bg-white space-y-2">
            <h2 className="font-semibold text-sm">Videos</h2>
            {videos.map((v) => {
              const videoPipeline = (v.pipeline || "v1").toUpperCase();
              return (
                <button
                  key={v.video_id}
                  type="button"
                  onClick={() => setSelectedVideo(v)}
                  className={`w-full text-left p-2 rounded border text-xs ${
                    selectedVideo?.video_id === v.video_id
                      ? "border-blue-600 bg-blue-50"
                      : "border-gray-200"
                  }`}
                >
                  <div className="font-medium truncate">{v.filename}</div>
                  <div className="text-gray-500">
                    {v.status} ({v.progress}%) • {videoPipeline}
                  </div>
                </button>
              );
            })}
          </div>
        </div>

        <div className="lg:col-span-2 space-y-4">
          {selectedVideo && (
            <VideoPlayer video={selectedVideo} seekToSeconds={seekTime} />
          )}

          <SearchPanel
            video={selectedVideo}
            pipeline={pipeline}
            onPipelineChange={setPipeline}
            onSelectTimestamp={(sec) => setSeekTime(sec)}
            onVideoUpdated={handleVideoUpdated}
          />
        </div>
      </div>
    </div>
  );
}

// import { useEffect, useState } from "react";
// import { getVideoStatus, listVideos } from "./api/client";
// import { SearchPanel } from "./components/SearchPanel";
// import VideoPlayer from "./components/VideoPlayer";
// import { VideoUpload } from "./components/VideoUpload";
// import type { PipelineType, VideoMetadata } from "./types";

// function formatPipelineLabel(pipeline: PipelineType): string {
//   if (pipeline === "v4") return "New (v4 — LLM Gate + CLIP/YOLO/OCR + Rerank)";
//   if (pipeline === "v3") return "V3 (CLIP + NMS + LLM Rerank)";
//   if (pipeline === "v2") return "V2 (CLIP + Temporal Grounding)";
//   return "Old (v1 — CLIP)";
// }

// export default function App() {
//   const [pipeline, setPipeline] = useState<PipelineType>("v4");
//   const [videos, setVideos] = useState<VideoMetadata[]>([]);
//   const [selectedVideo, setSelectedVideo] = useState<VideoMetadata | null>(null);
//   const [seekTime, setSeekTime] = useState<number | null>(null);

//   const refreshVideos = async () => {
//     const items = await listVideos();
//     setVideos(items);
//     if (!selectedVideo && items.length > 0) {
//       setSelectedVideo(items[0]);
//     } else if (selectedVideo) {
//       const fresh = items.find((item) => item.video_id === selectedVideo.video_id);
//       if (fresh) setSelectedVideo(fresh);
//     }
//   };

//   useEffect(() => {
//     refreshVideos();
//   }, []);

//   useEffect(() => {
//     if (!selectedVideo || selectedVideo.status !== "processing") return;
//     const timer = setInterval(async () => {
//       const updated = await getVideoStatus(selectedVideo.video_id);
//       setSelectedVideo(updated);
//       setVideos((prev) =>
//         prev.map((v) => (v.video_id === updated.video_id ? updated : v))
//       );
//     }, 2000);
//     return () => clearInterval(timer);
//   }, [selectedVideo?.video_id, selectedVideo?.status]);

//   const handleVideoUpdated = (updated: VideoMetadata) => {
//     setSelectedVideo(updated);
//     setVideos((prev) =>
//       prev.map((v) => (v.video_id === updated.video_id ? updated : v))
//     );
//   };

//   return (
//     <div className="min-h-screen bg-gray-50 p-6 max-w-6xl mx-auto space-y-6">
//       <header className="flex items-center justify-between">
//         <div>
//           <h1 className="text-2xl font-bold">ChronoVision / MomentSearch</h1>
//           <p className="text-sm text-gray-600">
//             Active Pipeline: <strong>{formatPipelineLabel(pipeline)}</strong>
//           </p>
//         </div>
//       </header>

//       <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
//         <div className="space-y-4">
//           <VideoUpload
//             pipeline={pipeline}
//             onPipelineChange={setPipeline}
//             onUploaded={(vid) => {
//               setVideos((prev) => [vid, ...prev]);
//               setSelectedVideo(vid);
//             }}
//           />

//           <div className="p-4 border rounded-lg bg-white space-y-2">
//             <h2 className="font-semibold text-sm">Videos</h2>
//             {videos.map((v) => {
//               const videoPipeline = (v.pipeline || "v1").toUpperCase();
//               return (
//                 <button
//                   key={v.video_id}
//                   type="button"
//                   onClick={() => setSelectedVideo(v)}
//                   className={`w-full text-left p-2 rounded border text-xs ${
//                     selectedVideo?.video_id === v.video_id
//                       ? "border-blue-600 bg-blue-50"
//                       : "border-gray-200"
//                   }`}
//                 >
//                   <div className="font-medium truncate">{v.filename}</div>
//                   <div className="text-gray-500">
//                     {v.status} ({v.progress}%) • {videoPipeline}
//                   </div>
//                 </button>
//               );
//             })}
//           </div>
//         </div>

//         <div className="lg:col-span-2 space-y-4">
//           {selectedVideo && (
//             <VideoPlayer video={selectedVideo} seekToSeconds={seekTime} />
//           )}

//           <SearchPanel
//             video={selectedVideo}
//             pipeline={pipeline}
//             onPipelineChange={setPipeline}
//             onSelectTimestamp={(sec) => setSeekTime(sec)}
//             onVideoUpdated={handleVideoUpdated}
//           />
//         </div>
//       </div>
//     </div>
//   );
// }

// import { useEffect, useState } from "react";
// import { getVideoStatus, listVideos } from "./api/client";
// import { SearchPanel } from "./components/SearchPanel";
// import VideoPlayer from "./components/VideoPlayer";
// import { VideoUpload } from "./components/VideoUpload";
// import type { PipelineType, VideoMetadata } from "./types";

// export default function App() {
//   const [pipeline, setPipeline] = useState<PipelineType>("v2");
//   const [videos, setVideos] = useState<VideoMetadata[]>([]);
//   const [selectedVideo, setSelectedVideo] = useState<VideoMetadata | null>(null);
//   const [seekTime, setSeekTime] = useState<number | null>(null);

//   const refreshVideos = async () => {
//     const items = await listVideos();
//     setVideos(items);
//     if (!selectedVideo && items.length > 0) {
//       setSelectedVideo(items[0]);
//     }
//   };

//   useEffect(() => {
//     refreshVideos();
//   }, []);

//   useEffect(() => {
//     if (!selectedVideo || selectedVideo.status !== "processing") return;
//     const timer = setInterval(async () => {
//       const updated = await getVideoStatus(selectedVideo.video_id);
//       setSelectedVideo(updated);
//       setVideos((prev) => prev.map((v) => (v.video_id === updated.video_id ? updated : v)));
//     }, 2000);
//     return () => clearInterval(timer);
//   }, [selectedVideo?.video_id, selectedVideo?.status]);

//   return (
//     <div className="min-h-screen bg-gray-50 p-6 max-w-6xl mx-auto space-y-6">
//       <header className="flex items-center justify-between">
//         <div>
//           <h1 className="text-2xl font-bold">ChronoVision / MomentSearch</h1>
//           <p className="text-sm text-gray-600">
//             Active Pipeline: <strong>{pipeline === "v2" ? "New (v2)" : "Old (v1)"}</strong>
//           </p>
//         </div>
//       </header>

//       <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
//         <div className="space-y-4">
//           <VideoUpload
//             pipeline={pipeline}
//             onPipelineChange={setPipeline}
//             onUploaded={(vid) => {
//               setVideos((prev) => [vid, ...prev]);
//               setSelectedVideo(vid);
//             }}
//           />

//           <div className="p-4 border rounded-lg bg-white space-y-2">
//             <h2 className="font-semibold text-sm">Videos</h2>
//             {videos.map((v) => (
//               <button
//                 key={v.video_id}
//                 type="button"
//                 onClick={() => setSelectedVideo(v)}
//                 className={`w-full text-left p-2 rounded border text-xs ${
//                   selectedVideo?.video_id === v.video_id ? "border-blue-600 bg-blue-50" : "border-gray-200"
//                 }`}
//               >
//                 <div className="font-medium truncate">{v.filename}</div>
//                 <div className="text-gray-500">
//                   {v.status} ({v.progress}%) • {v.pipeline?.toUpperCase() || "V1"}
//                 </div>
//               </button>
//             ))}
//           </div>
//         </div>

//         <div className="lg:col-span-2 space-y-4">
//           {selectedVideo && (
//             <VideoPlayer video={selectedVideo} seekToSeconds={seekTime} />
//           )}

//           <SearchPanel
//             video={selectedVideo}
//             pipeline={pipeline}
//             onPipelineChange={setPipeline}
//             onSelectTimestamp={(sec) => setSeekTime(sec)}
//             onVideoUpdated={(updated) => {
//               setSelectedVideo(updated);
//               setVideos((prev) => prev.map((v) => (v.video_id === updated.video_id ? updated : v)));
//             }}
//           />
//         </div>
//       </div>
//     </div>
//   );
// }

// import React, { useEffect, useState } from 'react';
// import {
//   AlertTriangle,
//   CheckCircle2,
//   Film,
//   Loader2,
//   RefreshCw,
//   Search,
//   SlidersHorizontal,
// } from 'lucide-react';
// import { fetchVideos, reprocessVideo, searchVideo } from './api/client';
// import SearchPanel from './components/SearchPanel';
// import VideoPlayer from './components/VideoPlayer';
// import VideoUpload from './components/VideoUpload';
// import type { SearchMatch, VideoMetadata } from './types';

// export default function App() {
//   const [videos, setVideos] = useState<VideoMetadata[]>([]);
//   const [selectedVideoId, setSelectedVideoId] = useState<string | null>(null);

//   const [query, setQuery] = useState('');
//   const [topK, setTopK] = useState(5);
//   const [searching, setSearching] = useState(false);
//   const [searchError, setSearchError] = useState<string | null>(null);

//   const [matches, setMatches] = useState<SearchMatch[]>([]);
//   const [disclaimer, setDisclaimer] = useState<string | null>(null);
//   const [activeRank, setActiveRank] = useState<number | null>(null);
//   const [seekRequest, setSeekRequest] = useState<{ time: number; nonce: number } | null>(null);

//   const selectedVideo = videos.find((v) => v.video_id === selectedVideoId) || null;

//   const loadLibrary = async () => {
//     try {
//       const list = await fetchVideos();
//       setVideos(list);
//       setSelectedVideoId((prev) => prev ?? (list.length > 0 ? list[0].video_id : null));
//     } catch (err) {
//       console.error(err);
//     }
//   };

//   useEffect(() => {
//     loadLibrary();
//   }, []);

//   // Poll status while any video is indexing
//   useEffect(() => {
//     const busy = videos.some((v) => v.status === 'processing' || v.status === 'uploaded');
//     if (!busy) return;
//     const timer = setInterval(loadLibrary, 1500);
//     return () => clearInterval(timer);
//   }, [videos]);

//   // Clear previous search results when switching videos
//   useEffect(() => {
//     setMatches([]);
//     setDisclaimer(null);
//     setSearchError(null);
//     setActiveRank(null);
//     setSeekRequest(null);
//   }, [selectedVideoId]);

//   const handleUploaded = (video: VideoMetadata) => {
//     setVideos((prev) => [video, ...prev]);
//     setSelectedVideoId(video.video_id);
//   };

//   const handleRetry = async (e: React.MouseEvent, videoId: string) => {
//     e.stopPropagation();
//     try {
//       await reprocessVideo(videoId);
//       await loadLibrary();
//     } catch (err) {
//       console.error(err);
//     }
//   };

//   const handleSearch = async (e: React.FormEvent) => {
//     e.preventDefault();
//     if (!selectedVideoId || !query.trim()) return;

//     setSearching(true);
//     setSearchError(null);
//     setActiveRank(null);

//     try {
//       const data = await searchVideo(selectedVideoId, query.trim(), topK);
//       setMatches(data.results);
//       setDisclaimer(data.disclaimer);
//     } catch (err: unknown) {
//       setSearchError(err instanceof Error ? err.message : 'Search failed.');
//     } finally {
//       setSearching(false);
//     }
//   };

//   const handleSelectMatch = (match: SearchMatch) => {
//     setActiveRank(match.rank);
//     setSeekRequest({ time: match.start_time, nonce: Date.now() });
//   };

//   return (
//     <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans">
//       <header className="h-14 border-b border-slate-800 bg-slate-900/80 backdrop-blur px-6 flex items-center justify-between">
//         <div className="flex items-center gap-3">
//           <div className="h-8 w-8 rounded-lg bg-indigo-600 flex items-center justify-center font-bold text-white shadow-md shadow-indigo-500/20">
//             M
//           </div>
//           <div>
//             <h1 className="text-base font-semibold tracking-tight text-white">MomentSearch</h1>
//             <p className="text-[11px] text-slate-400">
//               CLIP ViT-B/32 + FAISS Temporal Video Retrieval
//             </p>
//           </div>
//         </div>
//         <span className="text-xs text-slate-400 hidden sm:inline">
//           1.0 FPS Frame Sampling • Cosine Similarity
//         </span>
//       </header>

//       <div className="flex-1 grid grid-cols-1 lg:grid-cols-12 overflow-hidden">
//         {/* Left Sidebar: Upload & Video Library */}
//         <aside className="lg:col-span-3 border-r border-slate-800 bg-slate-900/40 p-4 flex flex-col gap-4 overflow-y-auto">
//           <VideoUpload onUploaded={handleUploaded} />

//           <div className="flex-1 flex flex-col min-h-0">
//             <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2.5 flex items-center gap-1.5">
//               <Film className="w-3.5 h-3.5" /> Video Library ({videos.length})
//             </h2>

//             <div className="space-y-2 overflow-y-auto pr-1">
//               {videos.length === 0 && (
//                 <p className="text-xs text-slate-500 py-6 text-center">
//                   No videos uploaded yet.
//                 </p>
//               )}

//               {videos.map((v) => {
//                 const isSelected = v.video_id === selectedVideoId;
//                 return (
//                   <div
//                     key={v.video_id}
//                     onClick={() => setSelectedVideoId(v.video_id)}
//                     className={`p-3 rounded-xl border transition cursor-pointer ${
//                       isSelected
//                         ? 'bg-indigo-950/40 border-indigo-500/60'
//                         : 'bg-slate-900/60 border-slate-800 hover:border-slate-700'
//                     }`}
//                   >
//                     <div className="flex items-start justify-between gap-2">
//                       <p className="text-xs font-medium text-slate-200 truncate" title={v.filename}>
//                         {v.filename}
//                       </p>
//                       {v.status === 'completed' && (
//                         <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
//                       )}
//                       {(v.status === 'processing' || v.status === 'uploaded') && (
//                         <Loader2 className="w-4 h-4 text-amber-400 animate-spin shrink-0" />
//                       )}
//                       {v.status === 'failed' && (
//                         <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0" />
//                       )}
//                     </div>

//                     <div className="mt-1.5 flex items-center justify-between text-[11px] text-slate-400">
//                       <span>
//                         {v.duration_sec > 0 ? `${v.duration_sec}s` : 'Pending'} • {v.frame_count}{' '}
//                         frames
//                       </span>
//                       <span className="capitalize text-slate-300">{v.status}</span>
//                     </div>

//                     {(v.status === 'processing' || v.status === 'uploaded') && (
//                       <div className="mt-2 w-full bg-slate-800 rounded-full h-1.5 overflow-hidden">
//                         <div
//                           className="bg-indigo-500 h-full transition-all duration-300"
//                           style={{ width: `${Math.max(8, v.progress)}%` }}
//                         />
//                       </div>
//                     )}

//                     {v.status === 'failed' && (
//                       <div className="mt-2 flex items-center justify-between gap-2">
//                         <span className="text-[11px] text-rose-400 truncate">
//                           {v.error_message || 'Indexing failed'}
//                         </span>
//                         <button
//                           type="button"
//                           onClick={(e) => handleRetry(e, v.video_id)}
//                           className="text-[11px] px-2 py-0.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-200 flex items-center gap-1 shrink-0"
//                         >
//                           <RefreshCw className="w-3 h-3" /> Retry
//                         </button>
//                       </div>
//                     )}
//                   </div>
//                 );
//               })}
//             </div>
//           </div>
//         </aside>

//         {/* Center Workspace: Player & Search Input */}
//         <main className="lg:col-span-5 p-6 flex flex-col gap-5 border-r border-slate-800 overflow-y-auto">
//           <VideoPlayer
//             video={selectedVideo}
//             seekRequest={seekRequest}
//             matches={matches}
//             activeRank={activeRank}
//             onSelectMatch={handleSelectMatch}
//           />

//           {selectedVideo && (
//             <form onSubmit={handleSearch} className="space-y-3">
//               <label className="block text-xs font-semibold uppercase tracking-wider text-slate-400">
//                 Natural-Language Event Query
//               </label>
//               <div className="flex gap-2">
//                 <input
//                   type="text"
//                   value={query}
//                   onChange={(e) => setQuery(e.target.value)}
//                   placeholder='e.g., "a person wearing a red shirt enters the room"'
//                   className="flex-1 bg-slate-900 border border-slate-700 focus:border-indigo-500 rounded-xl px-4 py-2.5 text-sm text-white placeholder-slate-500 outline-none"
//                 />
//                 <button
//                   type="submit"
//                   disabled={searching || selectedVideo.status !== 'completed' || !query.trim()}
//                   className="px-5 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 disabled:bg-slate-800 disabled:text-slate-500 text-white text-sm font-medium flex items-center gap-2 transition cursor-pointer"
//                 >
//                   {searching ? (
//                     <Loader2 className="w-4 h-4 animate-spin" />
//                   ) : (
//                     <Search className="w-4 h-4" />
//                   )}
//                   Search
//                 </button>
//               </div>

//               <div className="flex items-center justify-between text-xs text-slate-400">
//                 <div className="flex items-center gap-2">
//                   <SlidersHorizontal className="w-3.5 h-3.5" />
//                   <span>Max results:</span>
//                   <select
//                     value={topK}
//                     onChange={(e) => setTopK(Number(e.target.value))}
//                     className="bg-slate-900 border border-slate-700 rounded-lg px-2 py-1 text-slate-200 outline-none"
//                   >
//                     {[3, 5, 10, 15].map((k) => (
//                       <option key={k} value={k}>
//                         Top {k}
//                       </option>
//                     ))}
//                   </select>
//                 </div>

//                 {selectedVideo.status !== 'completed' && (
//                   <span className="text-amber-400">
//                     Indexing ({selectedVideo.progress}%) — search unlocks when complete
//                   </span>
//                 )}
//               </div>

//               {searchError && (
//                 <div className="p-3 rounded-xl bg-rose-950/50 border border-rose-900 text-xs text-rose-300">
//                   {searchError}
//                 </div>
//               )}
//             </form>
//           )}
//         </main>

//         {/* Right Panel: Retrieved Timestamps & Frame Previews */}
//         <section className="lg:col-span-4 bg-slate-900/30 p-5 overflow-y-auto">
//           <SearchPanel
//             matches={matches}
//             disclaimer={disclaimer}
//             activeRank={activeRank}
//             onSelectMatch={handleSelectMatch}
//           />
//         </section>
//       </div>
//     </div>
//   );
// }