import React, { useState } from "react";
import { uploadVideo } from "../api/client";
import type { PipelineType, VideoMetadata } from "../types";

interface VideoUploadProps {
  pipeline: PipelineType;
  onPipelineChange: (p: PipelineType) => void;
  onUploaded: (video: VideoMetadata) => void;
}

export const VideoUpload: React.FC<VideoUploadProps> = ({
  pipeline,
  onPipelineChange,
  onUploaded,
}) => {
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) return;

    setUploading(true);
    setError(null);
    try {
      const created = await uploadVideo(file, pipeline);
      onUploaded(created);
      setFile(null);
    } catch (err: any) {
      setError(err?.response?.data?.detail || "Video upload failed.");
    } finally {
      setUploading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-3 p-4 border rounded-lg bg-white">
      <div className="space-y-2">
        <span className="text-sm font-semibold block">Ingestion Pipeline</span>
        <div className="grid grid-cols-2 gap-1.5 text-xs">
          <button
            type="button"
            onClick={() => onPipelineChange("v1")}
            className={`px-2 py-1.5 rounded border transition ${
              pipeline === "v1"
                ? "bg-blue-600 text-white border-blue-600 font-semibold"
                : "bg-gray-100 text-gray-700 border-gray-200 hover:bg-gray-200"
            }`}
          >
            v1 (CLIP)
          </button>
          <button
            type="button"
            onClick={() => onPipelineChange("v2")}
            className={`px-2 py-1.5 rounded border transition ${
              pipeline === "v2"
                ? "bg-indigo-600 text-white border-indigo-600 font-semibold"
                : "bg-gray-100 text-gray-700 border-gray-200 hover:bg-gray-200"
            }`}
          >
            v2 (Grounding)
          </button>
          <button
            type="button"
            onClick={() => onPipelineChange("v3")}
            className={`px-2 py-1.5 rounded border transition ${
              pipeline === "v3"
                ? "bg-teal-600 text-white border-teal-600 font-semibold"
                : "bg-gray-100 text-gray-700 border-gray-200 hover:bg-gray-200"
            }`}
          >
            v3 (NMS + LLM)
          </button>
          <button
            type="button"
            onClick={() => onPipelineChange("v4")}
            className={`px-2 py-1.5 rounded border transition ${
              pipeline === "v4"
                ? "bg-emerald-600 text-white border-emerald-600 font-semibold"
                : "bg-gray-100 text-gray-700 border-gray-200 hover:bg-gray-200"
            }`}
          >
            v4 (Gate + Multi)
          </button>
          <button
            type="button"
            onClick={() => onPipelineChange("v5")}
            className={`px-2 py-1.5 rounded border transition ${
              pipeline === "v5"
                ? "bg-violet-600 text-white border-violet-600 font-semibold"
                : "bg-gray-100 text-gray-700 border-gray-200 hover:bg-gray-200"
            }`}
          >
            v5 (No-CLAHE Multi-Index)
          </button>
          <button
            type="button"
            onClick={() => onPipelineChange("v6")}
            className={`px-2 py-1.5 rounded border transition ${
              pipeline === "v6"
                ? "bg-amber-600 text-white border-amber-600 font-semibold"
                : "bg-gray-100 text-gray-700 border-gray-200 hover:bg-gray-200"
            }`}
          >
            v6 (CLIP + LLM Boundary)
          </button>
        </div>
      </div>

      <input
        type="file"
        accept=".mp4,.mov,.avi,.mkv,video/*"
        onChange={(e) => setFile(e.target.files?.[0] || null)}
        className="block w-full text-sm"
      />

      {error && <p className="text-xs text-red-600">{error}</p>}

      <button
        type="submit"
        disabled={!file || uploading}
        className="w-full py-2 px-4 rounded bg-blue-600 text-white text-sm font-medium disabled:opacity-50"
      >
        {uploading
          ? `Uploading (${pipeline.toUpperCase()})...`
          : `Upload & Process (${pipeline.toUpperCase()})`}
      </button>
    </form>
  );
};

// import React, { useState } from "react";
// import { uploadVideo } from "../api/client";
// import type { PipelineType, VideoMetadata } from "../types";

// interface VideoUploadProps {
//   pipeline: PipelineType;
//   onPipelineChange: (p: PipelineType) => void;
//   onUploaded: (video: VideoMetadata) => void;
// }

// export const VideoUpload: React.FC<VideoUploadProps> = ({
//   pipeline,
//   onPipelineChange,
//   onUploaded,
// }) => {
//   const [file, setFile] = useState<File | null>(null);
//   const [uploading, setUploading] = useState(false);
//   const [error, setError] = useState<string | null>(null);

//   const handleSubmit = async (e: React.FormEvent) => {
//     e.preventDefault();
//     if (!file) return;

//     setUploading(true);
//     setError(null);
//     try {
//       const created = await uploadVideo(file, pipeline);
//       onUploaded(created);
//       setFile(null);
//     } catch (err: any) {
//       setError(err?.response?.data?.detail || "Video upload failed.");
//     } finally {
//       setUploading(false);
//     }
//   };

//   return (
//     <form onSubmit={handleSubmit} className="space-y-3 p-4 border rounded-lg bg-white">
//       <div className="space-y-2">
//         <span className="text-sm font-semibold block">Ingestion Pipeline</span>
//         <div className="grid grid-cols-2 gap-1.5 text-xs">
//           <button
//             type="button"
//             onClick={() => onPipelineChange("v1")}
//             className={`px-2 py-1.5 rounded border transition ${
//               pipeline === "v1"
//                 ? "bg-blue-600 text-white border-blue-600 font-semibold"
//                 : "bg-gray-100 text-gray-700 border-gray-200 hover:bg-gray-200"
//             }`}
//           >
//             v1 (CLIP)
//           </button>
//           <button
//             type="button"
//             onClick={() => onPipelineChange("v2")}
//             className={`px-2 py-1.5 rounded border transition ${
//               pipeline === "v2"
//                 ? "bg-indigo-600 text-white border-indigo-600 font-semibold"
//                 : "bg-gray-100 text-gray-700 border-gray-200 hover:bg-gray-200"
//             }`}
//           >
//             v2 (Grounding)
//           </button>
//           <button
//             type="button"
//             onClick={() => onPipelineChange("v3")}
//             className={`px-2 py-1.5 rounded border transition ${
//               pipeline === "v3"
//                 ? "bg-teal-600 text-white border-teal-600 font-semibold"
//                 : "bg-gray-100 text-gray-700 border-gray-200 hover:bg-gray-200"
//             }`}
//           >
//             v3 (NMS + LLM)
//           </button>
//           <button
//             type="button"
//             onClick={() => onPipelineChange("v4")}
//             className={`px-2 py-1.5 rounded border transition ${
//               pipeline === "v4"
//                 ? "bg-emerald-600 text-white border-emerald-600 font-semibold"
//                 : "bg-gray-100 text-gray-700 border-gray-200 hover:bg-gray-200"
//             }`}
//           >
//             v4 (Gate + Multi)
//           </button>
//           <button
//             type="button"
//             onClick={() => onPipelineChange("v5")}
//             className={`col-span-2 px-2 py-1.5 rounded border transition ${
//               pipeline === "v5"
//                 ? "bg-violet-600 text-white border-violet-600 font-semibold"
//                 : "bg-gray-100 text-gray-700 border-gray-200 hover:bg-gray-200"
//             }`}
//           >
//             v5 (No-CLAHE Multi-Index)
//           </button>
//         </div>
//       </div>

//       <input
//         type="file"
//         accept=".mp4,.mov,.avi,.mkv,video/*"
//         onChange={(e) => setFile(e.target.files?.[0] || null)}
//         className="block w-full text-sm"
//       />

//       {error && <p className="text-xs text-red-600">{error}</p>}

//       <button
//         type="submit"
//         disabled={!file || uploading}
//         className="w-full py-2 px-4 rounded bg-blue-600 text-white text-sm font-medium disabled:opacity-50"
//       >
//         {uploading
//           ? `Uploading (${pipeline.toUpperCase()})...`
//           : `Upload & Process (${pipeline.toUpperCase()})`}
//       </button>
//     </form>
//   );
// };

// import React, { useState } from "react";
// import { uploadVideo } from "../api/client";
// import type { PipelineType, VideoMetadata } from "../types";

// interface VideoUploadProps {
//   pipeline: PipelineType;
//   onPipelineChange: (p: PipelineType) => void;
//   onUploaded: (video: VideoMetadata) => void;
// }

// export const VideoUpload: React.FC<VideoUploadProps> = ({
//   pipeline,
//   onPipelineChange,
//   onUploaded,
// }) => {
//   const [file, setFile] = useState<File | null>(null);
//   const [uploading, setUploading] = useState(false);
//   const [error, setError] = useState<string | null>(null);

//   const handleSubmit = async (e: React.FormEvent) => {
//     e.preventDefault();
//     if (!file) return;

//     setUploading(true);
//     setError(null);
//     try {
//       const created = await uploadVideo(file, pipeline);
//       onUploaded(created);
//       setFile(null);
//     } catch (err: any) {
//       setError(err?.response?.data?.detail || "Video upload failed.");
//     } finally {
//       setUploading(false);
//     }
//   };

//   return (
//     <form onSubmit={handleSubmit} className="space-y-3 p-4 border rounded-lg bg-white">
//       <div className="space-y-2">
//         <span className="text-sm font-semibold block">Ingestion Pipeline</span>
//         <div className="grid grid-cols-2 gap-1.5 text-xs">
//           <button
//             type="button"
//             onClick={() => onPipelineChange("v1")}
//             className={`px-2 py-1.5 rounded border transition ${
//               pipeline === "v1"
//                 ? "bg-blue-600 text-white border-blue-600 font-semibold"
//                 : "bg-gray-100 text-gray-700 border-gray-200 hover:bg-gray-200"
//             }`}
//           >
//             v1 (CLIP)
//           </button>
//           <button
//             type="button"
//             onClick={() => onPipelineChange("v2")}
//             className={`px-2 py-1.5 rounded border transition ${
//               pipeline === "v2"
//                 ? "bg-indigo-600 text-white border-indigo-600 font-semibold"
//                 : "bg-gray-100 text-gray-700 border-gray-200 hover:bg-gray-200"
//             }`}
//           >
//             v2 (Grounding)
//           </button>
//           <button
//             type="button"
//             onClick={() => onPipelineChange("v3")}
//             className={`px-2 py-1.5 rounded border transition ${
//               pipeline === "v3"
//                 ? "bg-teal-600 text-white border-teal-600 font-semibold"
//                 : "bg-gray-100 text-gray-700 border-gray-200 hover:bg-gray-200"
//             }`}
//           >
//             v3 (NMS + LLM)
//           </button>
//           <button
//             type="button"
//             onClick={() => onPipelineChange("v4")}
//             className={`px-2 py-1.5 rounded border transition ${
//               pipeline === "v4"
//                 ? "bg-emerald-600 text-white border-emerald-600 font-semibold"
//                 : "bg-gray-100 text-gray-700 border-gray-200 hover:bg-gray-200"
//             }`}
//           >
//             v4 (Gate + Multi)
//           </button>
//         </div>
//       </div>

//       <input
//         type="file"
//         accept=".mp4,.mov,.avi,.mkv,video/*"
//         onChange={(e) => setFile(e.target.files?.[0] || null)}
//         className="block w-full text-sm"
//       />

//       {error && <p className="text-xs text-red-600">{error}</p>}

//       <button
//         type="submit"
//         disabled={!file || uploading}
//         className="w-full py-2 px-4 rounded bg-blue-600 text-white text-sm font-medium disabled:opacity-50"
//       >
//         {uploading
//           ? `Uploading (${pipeline.toUpperCase()})...`
//           : `Upload & Process (${pipeline.toUpperCase()})`}
//       </button>
//     </form>
//   );
// };

// import React, { useState } from "react";
// import { uploadVideo } from "../api/client";
// import type { PipelineType, VideoMetadata } from "../types";

// interface VideoUploadProps {
//   pipeline: PipelineType;
//   onPipelineChange: (p: PipelineType) => void;
//   onUploaded: (video: VideoMetadata) => void;
// }

// export const VideoUpload: React.FC<VideoUploadProps> = ({
//   pipeline,
//   onPipelineChange,
//   onUploaded,
// }) => {
//   const [file, setFile] = useState<File | null>(null);
//   const [uploading, setUploading] = useState(false);
//   const [error, setError] = useState<string | null>(null);

//   const handleSubmit = async (e: React.FormEvent) => {
//     e.preventDefault();
//     if (!file) return;

//     setUploading(true);
//     setError(null);
//     try {
//       const created = await uploadVideo(file, pipeline);
//       onUploaded(created);
//       setFile(null);
//     } catch (err: any) {
//       setError(err?.response?.data?.detail || "Video upload failed.");
//     } finally {
//       setUploading(false);
//     }
//   };

//   return (
//     <form onSubmit={handleSubmit} className="space-y-3 p-4 border rounded-lg bg-white">
//       <div className="space-y-2">
//         <span className="text-sm font-semibold block">Ingestion Pipeline</span>
//         <div className="grid grid-cols-3 rounded border overflow-hidden text-xs">
//           <button
//             type="button"
//             onClick={() => onPipelineChange("v1")}
//             className={`px-2 py-1.5 transition ${
//               pipeline === "v1"
//                 ? "bg-blue-600 text-white font-semibold"
//                 : "bg-gray-100 text-gray-700 hover:bg-gray-200"
//             }`}
//           >
//             Old (v1: CLIP)
//           </button>
//           <button
//             type="button"
//             onClick={() => onPipelineChange("v2")}
//             className={`px-2 py-1.5 transition ${
//               pipeline === "v2"
//                 ? "bg-indigo-600 text-white font-semibold"
//                 : "bg-gray-100 text-gray-700 hover:bg-gray-200"
//             }`}
//           >
//             V2 (Grounding)
//           </button>
//           <button
//             type="button"
//             onClick={() => onPipelineChange("v3")}
//             className={`px-2 py-1.5 transition ${
//               pipeline === "v3"
//                 ? "bg-emerald-600 text-white font-semibold"
//                 : "bg-gray-100 text-gray-700 hover:bg-gray-200"
//             }`}
//           >
//             New (v3: LLM)
//           </button>
//         </div>
//       </div>

//       <input
//         type="file"
//         accept=".mp4,.mov,.avi,.mkv,video/*"
//         onChange={(e) => setFile(e.target.files?.[0] || null)}
//         className="block w-full text-sm"
//       />

//       {error && <p className="text-xs text-red-600">{error}</p>}

//       <button
//         type="submit"
//         disabled={!file || uploading}
//         className="w-full py-2 px-4 rounded bg-blue-600 text-white text-sm font-medium disabled:opacity-50"
//       >
//         {uploading
//           ? `Uploading (${pipeline.toUpperCase()})...`
//           : `Upload & Process (${pipeline.toUpperCase()})`}
//       </button>
//     </form>
//   );
// };

// // import React, { useState } from "react";
// // import { uploadVideo } from "../api/client";
// // import type { PipelineType, VideoMetadata } from "../types";

// // interface VideoUploadProps {
// //   pipeline: PipelineType;
// //   onPipelineChange: (p: PipelineType) => void;
// //   onUploaded: (video: VideoMetadata) => void;
// // }

// // export const VideoUpload: React.FC<VideoUploadProps> = ({
// //   pipeline,
// //   onPipelineChange,
// //   onUploaded,
// // }) => {
// //   const [file, setFile] = useState<File | null>(null);
// //   const [uploading, setUploading] = useState(false);
// //   const [error, setError] = useState<string | null>(null);

// //   const handleSubmit = async (e: React.FormEvent) => {
// //     e.preventDefault();
// //     if (!file) return;

// //     setUploading(true);
// //     setError(null);
// //     try {
// //       const created = await uploadVideo(file, pipeline);
// //       onUploaded(created);
// //       setFile(null);
// //     } catch (err: any) {
// //       setError(err?.response?.data?.detail || "Video upload failed.");
// //     } finally {
// //       setUploading(false);
// //     }
// //   };

// //   return (
// //     <form onSubmit={handleSubmit} className="space-y-3 p-4 border rounded-lg bg-white">
// //       <div className="flex flex-wrap items-center justify-between gap-2">
// //         <span className="text-sm font-semibold">Ingestion Pipeline</span>
// //         <div className="inline-flex rounded border overflow-hidden text-xs">
// //           <button
// //             type="button"
// //             onClick={() => onPipelineChange("v1")}
// //             className={`px-3 py-1.5 ${
// //               pipeline === "v1" ? "bg-blue-600 text-white font-semibold" : "bg-gray-100 text-gray-700"
// //             }`}
// //           >
// //             Old (v1: CLIP)
// //           </button>
// //           <button
// //             type="button"
// //             onClick={() => onPipelineChange("v2")}
// //             className={`px-3 py-1.5 ${
// //               pipeline === "v2" ? "bg-emerald-600 text-white font-semibold" : "bg-gray-100 text-gray-700"
// //             }`}
// //           >
// //             New (v2: CLIP + YOLO + OCR)
// //           </button>
// //         </div>
// //       </div>

// //       <input
// //         type="file"
// //         accept=".mp4,.mov,.avi,.mkv,video/*"
// //         onChange={(e) => setFile(e.target.files?.[0] || null)}
// //         className="block w-full text-sm"
// //       />

// //       {error && <p className="text-xs text-red-600">{error}</p>}

// //       <button
// //         type="submit"
// //         disabled={!file || uploading}
// //         className="w-full py-2 px-4 rounded bg-blue-600 text-white text-sm font-medium disabled:opacity-50"
// //       >
// //         {uploading ? `Uploading (${pipeline.toUpperCase()})...` : `Upload & Process (${pipeline.toUpperCase()})`}
// //       </button>
// //     </form>
// //   );
// // };

// // // import React, { useRef, useState } from 'react';
// // // import { Loader2, Upload } from 'lucide-react';
// // // import { uploadVideo } from '../api/client';
// // // import type { VideoMetadata } from '../types';

// // // interface VideoUploadProps {
// // //   onUploaded: (video: VideoMetadata) => void;
// // // }

// // // export default function VideoUpload({ onUploaded }: VideoUploadProps) {
// // //   const [uploading, setUploading] = useState(false);
// // //   const [error, setError] = useState<string | null>(null);
// // //   const inputRef = useRef<HTMLInputElement>(null);

// // //   const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
// // //     const file = e.target.files?.[0];
// // //     if (!file) return;

// // //     setUploading(true);
// // //     setError(null);
// // //     try {
// // //       const created = await uploadVideo(file);
// // //       onUploaded(created);
// // //     } catch (err: unknown) {
// // //       setError(err instanceof Error ? err.message : 'Upload failed.');
// // //     } finally {
// // //       setUploading(false);
// // //       if (inputRef.current) inputRef.current.value = '';
// // //     }
// // //   };

// // //   return (
// // //     <div>
// // //       <input
// // //         ref={inputRef}
// // //         type="file"
// // //         accept=".mp4,.mov,.avi,.mkv"
// // //         className="hidden"
// // //         onChange={handleFileChange}
// // //       />
// // //       <button
// // //         type="button"
// // //         disabled={uploading}
// // //         onClick={() => inputRef.current?.click()}
// // //         className="w-full py-3.5 px-4 rounded-xl border border-dashed border-slate-700 hover:border-indigo-500 bg-slate-900/90 hover:bg-slate-900 transition flex flex-col items-center justify-center gap-1.5 text-sm font-medium text-slate-200 disabled:opacity-50 cursor-pointer"
// // //       >
// // //         {uploading ? (
// // //           <>
// // //             <Loader2 className="w-5 h-5 text-indigo-400 animate-spin" />
// // //             <span>Uploading footage...</span>
// // //           </>
// // //         ) : (
// // //           <>
// // //             <Upload className="w-5 h-5 text-indigo-400" />
// // //             <span>Upload CCTV Footage</span>
// // //             <span className="text-[11px] text-slate-500">
// // //               MP4, MOV, AVI, MKV (max 500 MB)
// // //             </span>
// // //           </>
// // //         )}
// // //       </button>

// // //       {error && (
// // //         <p className="mt-2 text-xs text-rose-300 bg-rose-950/60 border border-rose-900 rounded-lg p-2">
// // //           {error}
// // //         </p>
// // //       )}
// // //     </div>
// // //   );
// // // }