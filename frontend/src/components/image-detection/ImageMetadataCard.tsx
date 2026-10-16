'use client';

import React, { useState } from 'react';
import { ImageDetectionResponse } from '@/types/image-detection';

interface ImageMetadataCardProps {
  data: ImageDetectionResponse;
}

export function ImageMetadataCard({ data }: ImageMetadataCardProps) {
  const [showRaw, setShowRaw] = useState(false);
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(JSON.stringify(data, null, 2));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const meta = data.image_metadata;

  return (
    <div className="bg-white rounded-2xl border border-gray-200/90 shadow-sm p-6 space-y-5">
      <div className="flex items-center justify-between border-b border-gray-100 pb-3">
        <h3 className="text-sm font-bold text-gray-900 uppercase tracking-wider flex items-center gap-2">
          <span>🔬</span> Technical Forensic Attributes
        </h3>
        <span className="text-[11px] font-mono text-gray-600 bg-gray-100 px-2 py-0.5 rounded">
          {data.model_version}
        </span>
      </div>

      {/* Grid of Attributes */}
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 text-xs">
        <div className="bg-gray-50 rounded-xl p-3 border border-gray-100">
          <p className="text-gray-600 font-medium">Resolution</p>
          <p className="font-bold text-gray-900 mt-0.5">
            {meta.width} × {meta.height} px
          </p>
        </div>

        <div className="bg-gray-50 rounded-xl p-3 border border-gray-100">
          <p className="text-gray-600 font-medium">Color Channels</p>
          <p className="font-bold text-gray-900 mt-0.5">
            {meta.channels} Channels (RGB)
          </p>
        </div>

        <div className="bg-gray-50 rounded-xl p-3 border border-gray-100">
          <p className="text-gray-600 font-medium">Source Format</p>
          <p className="font-bold text-gray-900 mt-0.5 uppercase">
            {meta.format || 'Image Stream'}
          </p>
        </div>

        <div className="bg-gray-50 rounded-xl p-3 border border-gray-100">
          <p className="text-gray-600 font-medium">Inference Latency</p>
          <p className="font-bold text-blue-600 mt-0.5">
            {data.latency_ms.toFixed(1)} ms
          </p>
        </div>

        <div className="bg-gray-50 rounded-xl p-3 border border-gray-100">
          <p className="text-gray-600 font-medium">Hardware Target</p>
          <p className="font-bold text-emerald-600 mt-0.5 uppercase">
            {data.device}
          </p>
        </div>

        <div className="bg-gray-50 rounded-xl p-3 border border-gray-100">
          <p className="text-gray-600 font-medium">Evidence Filename</p>
          <p className="font-bold text-gray-900 mt-0.5 truncate" title={data.filename}>
            {data.filename}
          </p>
        </div>
      </div>

      {/* Request ID & Action */}
      <div className="pt-2 border-t border-gray-100 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 text-xs">
        <div className="flex items-center gap-1.5 text-gray-600 font-mono text-[11px] truncate">
          <span>UUID:</span>
          <span className="truncate">{data.request_id}</span>
        </div>

        <button
          type="button"
          onClick={() => setShowRaw(!showRaw)}
          className="text-blue-600 hover:text-blue-700 font-semibold cursor-pointer text-left"
        >
          {showRaw ? 'Hide Raw Audit JSON ▲' : 'View Raw Audit JSON ▼'}
        </button>
      </div>

      {/* Raw JSON Inspector */}
      {showRaw && (
        <div className="mt-3 p-3 bg-gray-900 rounded-xl text-gray-200 font-mono text-xs relative">
          <button
            type="button"
            onClick={handleCopy}
            className="absolute top-2.5 right-2.5 px-2 py-1 bg-gray-800 hover:bg-gray-700 text-gray-300 rounded text-[10px] font-sans border border-gray-700 cursor-pointer"
          >
            {copied ? '✓ Copied' : 'Copy JSON'}
          </button>
          <pre className="overflow-x-auto max-h-48 pt-4">
            {JSON.stringify(data, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
}
