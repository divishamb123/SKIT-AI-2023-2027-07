'use client';

import React, { useState, useRef, useCallback } from 'react';
import { SelectedImageFile } from '@/types/image-detection';

interface ImageUploadDropzoneProps {
  files: SelectedImageFile[];
  onFilesSelected: (files: SelectedImageFile[]) => void;
  onFileRemoved: (id: string) => void;
  onClearAll: () => void;
  isMultiple?: boolean;
  maxFiles?: number;
  maxSizeBytes?: number; // default 15 MB
  disabled?: boolean;
  onSampleLoad?: (type: 'real' | 'ai') => void;
}

const ALLOWED_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.webp'];
const ALLOWED_MIME_TYPES = ['image/png', 'image/jpeg', 'image/webp'];
const DEFAULT_MAX_SIZE = 15 * 1024 * 1024; // 15 MB
const DEFAULT_MAX_FILES = 32;

export function ImageUploadDropzone({
  files,
  onFilesSelected,
  onFileRemoved,
  onClearAll,
  isMultiple = false,
  maxFiles = DEFAULT_MAX_FILES,
  maxSizeBytes = DEFAULT_MAX_SIZE,
  disabled = false,
  onSampleLoad,
}: ImageUploadDropzoneProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [validationError, setValidationError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Validate and read file dimensions
  const processFiles = useCallback(
    async (incomingFiles: File[]) => {
      setValidationError(null);

      if (!incomingFiles || incomingFiles.length === 0) return;

      // Limit count if batch
      const targetCount = isMultiple
        ? Math.min(incomingFiles.length, maxFiles - files.length)
        : 1;

      if (isMultiple && files.length + incomingFiles.length > maxFiles) {
        setValidationError(
          `Maximum batch size is ${maxFiles} images. Only the first ${targetCount} were added.`
        );
      }

      const filesToProcess = incomingFiles.slice(0, targetCount);
      const newSelectedFiles: SelectedImageFile[] = [];

      for (const file of filesToProcess) {
        // 1. Extension check
        const ext = '.' + file.name.split('.').pop()?.toLowerCase();
        if (!ALLOWED_EXTENSIONS.includes(ext) && !ALLOWED_MIME_TYPES.includes(file.type)) {
          setValidationError(
            `"${file.name}" has an unsupported format. Allowed: PNG, JPEG, WEBP.`
          );
          continue;
        }

        // 2. Size check
        if (file.size > maxSizeBytes) {
          setValidationError(
            `"${file.name}" exceeds the 15 MB size limit (${(file.size / (1024 * 1024)).toFixed(1)} MB).`
          );
          continue;
        }

        // 3. Create preview and read dimensions
        const previewUrl = URL.createObjectURL(file);
        const dims = await new Promise<{ width: number; height: number } | undefined>(
          (resolve) => {
            const img = new Image();
            img.onload = () => resolve({ width: img.naturalWidth, height: img.naturalHeight });
            img.onerror = () => resolve(undefined);
            img.src = previewUrl;
          }
        );

        newSelectedFiles.push({
          file,
          previewUrl,
          id: `${file.name}-${file.size}-${Date.now()}-${Math.random()}`,
          dimensions: dims,
        });
      }

      if (newSelectedFiles.length > 0) {
        onFilesSelected(newSelectedFiles);
      }
    },
    [isMultiple, maxFiles, files.length, maxSizeBytes, onFilesSelected]
  );

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    if (!disabled) setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (disabled) return;
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      processFiles(Array.from(e.dataTransfer.files));
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      processFiles(Array.from(e.target.files));
    }
    // reset input so the same file can be re-selected if removed
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="w-full space-y-4">
      {/* Drop Area */}
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => !disabled && fileInputRef.current?.click()}
        className={`relative border-2 border-dashed rounded-2xl p-6 sm:p-8 text-center transition-all cursor-pointer ${
          disabled
            ? 'opacity-60 cursor-not-allowed border-gray-200 bg-gray-50'
            : isDragging
            ? 'border-blue-500 bg-blue-50/70 shadow-sm'
            : 'border-gray-300 hover:border-blue-400 bg-gray-50/40 hover:bg-gray-50/80'
        }`}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept={ALLOWED_EXTENSIONS.join(',')}
          multiple={isMultiple}
          disabled={disabled}
          onChange={handleInputChange}
          className="sr-only"
        />

        <div className="flex flex-col items-center justify-center space-y-3">
          <div className="w-14 h-14 rounded-2xl bg-blue-50 text-blue-600 flex items-center justify-center shadow-xs">
            <svg
              className="w-7 h-7"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
              />
            </svg>
          </div>

          <div>
            <p className="text-sm font-bold text-gray-900">
              {isMultiple
                ? 'Drop image files here, or click to browse multiple'
                : 'Drop an image file here, or click to browse'}
            </p>
            <p className="text-xs text-gray-500 mt-1">
              Supports PNG, JPEG, WEBP up to 15 MB {isMultiple ? `(Max ${maxFiles} images/batch)` : ''}
            </p>
          </div>
        </div>
      </div>

      {/* Inline Validation Alert */}
      {validationError && (
        <div className="p-3 rounded-xl bg-amber-50 border border-amber-200 text-amber-800 text-xs flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span>⚠️</span>
            <span>{validationError}</span>
          </div>
          <button
            type="button"
            onClick={() => setValidationError(null)}
            className="text-amber-700 hover:text-amber-900 font-bold ml-2 cursor-pointer"
          >
            ✕
          </button>
        </div>
      )}

      {/* Preset Buttons */}
      {onSampleLoad && (
        <div className="flex flex-wrap items-center justify-between gap-2 pt-1">
          <span className="text-xs text-gray-500 font-medium">Quick Pre-loaded Samples:</span>
          <div className="flex items-center gap-2">
            <button
              type="button"
              disabled={disabled}
              onClick={() => onSampleLoad('real')}
              className="px-3 py-1.5 text-xs font-semibold text-emerald-700 bg-emerald-50 hover:bg-emerald-100 border border-emerald-200 rounded-lg transition-colors cursor-pointer shadow-2xs disabled:opacity-50"
            >
              📷 Load Real Photo Sample
            </button>
            <button
              type="button"
              disabled={disabled}
              onClick={() => onSampleLoad('ai')}
              className="px-3 py-1.5 text-xs font-semibold text-rose-700 bg-rose-50 hover:bg-rose-100 border border-rose-200 rounded-lg transition-colors cursor-pointer shadow-2xs disabled:opacity-50"
            >
              ⚡ Load AI Diffusion Sample
            </button>
          </div>
        </div>
      )}

      {/* Selected Files Preview Queue */}
      {files.length > 0 && (
        <div className="bg-white rounded-2xl border border-gray-200 p-4 space-y-3">
          <div className="flex items-center justify-between border-b border-gray-100 pb-2">
            <span className="text-xs font-bold text-gray-700 uppercase tracking-wider">
              Selected Evidence ({files.length} {files.length === 1 ? 'file' : 'files'})
            </span>
            <button
              type="button"
              disabled={disabled}
              onClick={onClearAll}
              className="text-xs text-gray-500 hover:text-red-600 font-semibold cursor-pointer disabled:opacity-50"
            >
              Clear All
            </button>
          </div>

          <div
            className={`grid gap-2.5 ${
              isMultiple
                ? 'grid-cols-2 sm:grid-cols-3 md:grid-cols-4 max-h-64 overflow-y-auto pr-1'
                : 'grid-cols-1'
            }`}
          >
            {files.map((item) => (
              <div
                key={item.id}
                className="relative group bg-gray-50 rounded-xl p-2 border border-gray-200 flex items-center gap-3 overflow-hidden"
              >
                {/* Thumbnail */}
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={item.previewUrl}
                  alt={item.file.name}
                  className="w-12 h-12 rounded-lg object-cover bg-gray-200 shrink-0 border border-gray-200"
                />

                {/* Details */}
                <div className="flex-1 min-w-0 pr-6">
                  <p className="text-xs font-bold text-gray-900 truncate" title={item.file.name}>
                    {item.file.name}
                  </p>
                  <p className="text-[10px] text-gray-500 mt-0.5">
                    {(item.file.size / 1024).toFixed(0)} KB
                    {item.dimensions
                      ? ` • ${item.dimensions.width}×${item.dimensions.height}`
                      : ''}
                  </p>
                </div>

                {/* Remove Button */}
                <button
                  type="button"
                  disabled={disabled}
                  onClick={(e) => {
                    e.stopPropagation();
                    onFileRemoved(item.id);
                  }}
                  className="absolute top-2 right-2 w-6 h-6 rounded-full bg-gray-200 hover:bg-red-500 hover:text-white text-gray-600 text-xs flex items-center justify-center transition-colors cursor-pointer disabled:opacity-50"
                  title="Remove image"
                >
                  ✕
                </button>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
