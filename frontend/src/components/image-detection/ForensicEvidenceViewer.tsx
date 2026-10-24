'use client';

/**
 * ForensicEvidenceViewer.tsx — Multi-Channel Decomposition & Frequency Spectrum Inspector
 *
 * Student: Divisha Manak Bohra (23ESKCA038)
 * Sprint: Sprint 2 — Baseline Detection Service
 * Task 4: Designing image upload and result interaction components (Week 6)
 */

import React, { useState, useEffect, useRef } from 'react';
import { SelectedImageFile } from '@/types/image-detection';

interface ForensicEvidenceViewerProps {
  selectedFile: SelectedImageFile;
  verdict?: 'REAL' | 'AI_GENERATED';
}

type ViewMode = 'rgb' | 'red' | 'green' | 'blue' | 'luminance' | 'highpass';

export function ForensicEvidenceViewer({
  selectedFile,
  verdict,
}: ForensicEvidenceViewerProps) {
  const [activeMode, setActiveMode] = useState<ViewMode>('rgb');
  const [zoomLevel, setZoomLevel] = useState<number>(1);
  const [cursorPos, setCursorPos] = useState<{ x: number; y: number } | null>(
    null
  );
  const [pixelData, setPixelData] = useState<{
    r: number;
    g: number;
    b: number;
  } | null>(null);

  // Channel canvas URLs
  const [channelUrls, setChannelUrls] = useState<Record<ViewMode, string>>({
    rgb: selectedFile.previewUrl,
    red: '',
    green: '',
    blue: '',
    luminance: '',
    highpass: '',
  });

  const hiddenCanvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);

  // Extract channel decomposition in-memory using canvas
  useEffect(() => {
    const img = new window.Image();
    img.crossOrigin = 'anonymous';
    img.src = selectedFile.previewUrl;

    img.onload = () => {
      const w = img.naturalWidth || 224;
      const h = img.naturalHeight || 224;

      const canvas = document.createElement('canvas');
      canvas.width = w;
      canvas.height = h;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      ctx.drawImage(img, 0, 0, w, h);
      const imgData = ctx.getImageData(0, 0, w, h);
      const data = imgData.data;

      // Helper creating processed data URLs
      const createFilteredUrl = (
        filterFn: (r: number, g: number, b: number) => [number, number, number]
      ): string => {
        const outCanvas = document.createElement('canvas');
        outCanvas.width = w;
        outCanvas.height = h;
        const outCtx = outCanvas.getContext('2d')!;
        const outImgData = outCtx.createImageData(w, h);
        const outData = outImgData.data;

        for (let i = 0; i < data.length; i += 4) {
          const [nr, ng, nb] = filterFn(data[i], data[i + 1], data[i + 2]);
          outData[i] = nr;
          outData[i + 1] = ng;
          outData[i + 2] = nb;
          outData[i + 3] = 255;
        }

        outCtx.putImageData(outImgData, 0, 0);
        return outCanvas.toDataURL('image/png');
      };

      // 1. Red Channel
      const redUrl = createFilteredUrl((r) => [r, 0, 0]);
      // 2. Green Channel
      const greenUrl = createFilteredUrl((_, g) => [0, g, 0]);
      // 3. Blue Channel
      const blueUrl = createFilteredUrl((_, __, b) => [0, 0, b]);
      // 4. Luminance (Rec. 601 grayscale)
      const lumUrl = createFilteredUrl((r, g, b) => {
        const y = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
        return [y, y, y];
      });
      // 5. High-Pass Spatial Frequency Filter (Emphasizes pixel grid & latent noise)
      const hpCanvas = document.createElement('canvas');
      hpCanvas.width = w;
      hpCanvas.height = h;
      const hpCtx = hpCanvas.getContext('2d')!;
      const hpImgData = hpCtx.createImageData(w, h);
      const hpData = hpImgData.data;

      for (let y = 1; y < h - 1; y++) {
        for (let x = 1; x < w - 1; x++) {
          const idx = (y * w + x) * 4;
          const leftIdx = (y * w + (x - 1)) * 4;
          const topIdx = ((y - 1) * w + x) * 4;

          const diffR = Math.abs(data[idx] - (data[leftIdx] + data[topIdx]) / 2) * 3;
          const diffG = Math.abs(data[idx + 1] - (data[leftIdx + 1] + data[topIdx + 1]) / 2) * 3;
          const diffB = Math.abs(data[idx + 2] - (data[leftIdx + 2] + data[topIdx + 2]) / 2) * 3;

          hpData[idx] = Math.min(255, diffR);
          hpData[idx + 1] = Math.min(255, diffG);
          hpData[idx + 2] = Math.min(255, diffB);
          hpData[idx + 3] = 255;
        }
      }
      hpCtx.putImageData(hpImgData, 0, 0);
      const hpUrl = hpCanvas.toDataURL('image/png');

      setChannelUrls({
        rgb: selectedFile.previewUrl,
        red: redUrl,
        green: greenUrl,
        blue: blueUrl,
        luminance: lumUrl,
        highpass: hpUrl,
      });
    };
  }, [selectedFile.previewUrl]);

  // Handle probe cursor over image
  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = Math.floor(e.clientX - rect.left);
    const y = Math.floor(e.clientY - rect.top);
    setCursorPos({ x, y });

    if (imageRef.current && hiddenCanvasRef.current) {
      const canvas = hiddenCanvasRef.current;
      const ctx = canvas.getContext('2d');
      if (ctx) {
        try {
          const pixel = ctx.getImageData(
            Math.floor((x / rect.width) * canvas.width),
            Math.floor((y / rect.height) * canvas.height),
            1,
            1
          ).data;
          setPixelData({ r: pixel[0], g: pixel[1], b: pixel[2] });
        } catch {
          // Cross-origin fallback
        }
      }
    }
  };

  const modes: { id: ViewMode; label: string; icon: string; desc: string }[] = [
    { id: 'rgb', label: 'Full RGB Spectrum', icon: '🎨', desc: 'Standard composite visible color domain' },
    { id: 'luminance', label: 'Luminance (Y)', icon: '☀️', desc: 'Grayscale intensity distribution' },
    { id: 'highpass', label: 'Spatial High-Pass', icon: '⚡', desc: 'Latent grid & compression artifact map' },
    { id: 'red', label: 'Red Channel', icon: '🔴', desc: 'Long-wavelength chromatic layer' },
    { id: 'green', label: 'Green Channel', icon: '🟢', desc: 'Bayer sensor primary filter channel' },
    { id: 'blue', label: 'Blue Channel', icon: '🔵', desc: 'Short-wavelength chromatic layer' },
  ];

  return (
    <div className="bg-white rounded-2xl border border-gray-200/90 shadow-sm p-6 space-y-5">
      {/* Hidden canvas for pixel probe */}
      <canvas ref={hiddenCanvasRef} className="hidden" />

      {/* Header & Controls */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 border-b border-gray-100 pb-3">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-base">🔬</span>
            <h3 className="text-sm font-bold text-gray-900 uppercase tracking-wider">
              Forensic Evidence Visualizer & Channel Decomposition
            </h3>
          </div>
          <p className="text-xs text-gray-500 mt-0.5">
            Inspect frequency domain residuals, Bayer filter noise, and chromatic decomposition.
          </p>
        </div>

        {/* Zoom Controls */}
        <div className="flex items-center gap-1.5 self-start sm:self-auto bg-gray-100 p-1 rounded-xl text-xs">
          {[1, 1.5, 2].map((lvl) => (
            <button
              key={lvl}
              type="button"
              onClick={() => setZoomLevel(lvl)}
              className={`px-2.5 py-1 font-bold rounded-lg transition-all cursor-pointer ${
                zoomLevel === lvl
                  ? 'bg-white text-blue-600 shadow-2xs'
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              {lvl * 100}%
            </button>
          ))}
        </div>
      </div>

      {/* View Mode Tabs */}
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-2">
        {modes.map((m) => {
          const isActive = activeMode === m.id;
          return (
            <button
              key={m.id}
              type="button"
              onClick={() => setActiveMode(m.id)}
              className={`p-2 rounded-xl border text-center transition-all cursor-pointer flex flex-col items-center justify-center gap-1 ${
                isActive
                  ? 'border-blue-500 bg-blue-50/70 shadow-2xs'
                  : 'border-gray-200 bg-gray-50/40 hover:bg-gray-100 hover:border-gray-300'
              }`}
            >
              <span className="text-sm">{m.icon}</span>
              <span
                className={`text-[11px] font-bold ${
                  isActive ? 'text-blue-700' : 'text-gray-700'
                }`}
              >
                {m.label}
              </span>
            </button>
          );
        })}
      </div>

      {/* Image Display Canvas Container */}
      <div className="relative rounded-2xl bg-gray-950 border border-gray-800 overflow-hidden min-h-[300px] flex items-center justify-center">
        <div
          onMouseMove={handleMouseMove}
          onMouseLeave={() => {
            setCursorPos(null);
            setPixelData(null);
          }}
          className="relative overflow-auto max-h-[460px] w-full flex items-center justify-center p-4 cursor-crosshair"
        >
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            ref={imageRef}
            src={channelUrls[activeMode] || selectedFile.previewUrl}
            alt="Forensic Evidence Inspection"
            style={{
              transform: `scale(${zoomLevel})`,
              transformOrigin: 'center center',
              transition: 'transform 0.2s ease-out',
            }}
            className="max-h-[400px] max-w-full rounded-lg object-contain shadow-2xl"
          />
        </div>

        {/* HUD Overlay Info */}
        <div className="absolute bottom-3 left-3 bg-gray-900/90 backdrop-blur-xs text-white px-3 py-1.5 rounded-lg border border-gray-700 text-[11px] flex items-center gap-3">
          <span className="font-semibold uppercase tracking-wider text-blue-400">
            {activeMode} domain
          </span>
          {cursorPos && (
            <span className="font-mono text-gray-300">
              X:{cursorPos.x} Y:{cursorPos.y}
            </span>
          )}
          {pixelData && (
            <span className="font-mono text-gray-300">
              RGB({pixelData.r},{pixelData.g},{pixelData.b})
            </span>
          )}
        </div>

        {verdict && (
          <div className="absolute top-3 right-3">
            <span
              className={`px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider shadow-md ${
                verdict === 'AI_GENERATED'
                  ? 'bg-rose-600 text-white'
                  : 'bg-emerald-600 text-white'
              }`}
            >
              {verdict === 'AI_GENERATED' ? 'Synthetic AI Artifacts' : 'Authentic PRNU Valid'}
            </span>
          </div>
        )}
      </div>

      {/* Footer Mode Explanation */}
      <div className="p-3 bg-gray-50 rounded-xl border border-gray-200 text-xs text-gray-600 flex items-center justify-between">
        <p>
          <strong className="text-gray-900">Active Mode: </strong>
          {modes.find((m) => m.id === activeMode)?.desc}
        </p>
        <span className="text-[10px] text-gray-400 font-mono">
          Resolution: {selectedFile.dimensions?.width || 224}×{selectedFile.dimensions?.height || 224} px
        </span>
      </div>
    </div>
  );
}
