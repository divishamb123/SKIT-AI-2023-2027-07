'use client';

import React, { useState, useRef } from 'react';

interface UploadDropzoneProps {
  onFileSelect: (file: File) => void;
  acceptedTypes?: string;
  modality: 'text' | 'image' | 'audio';
}

export function UploadDropzone({ onFileSelect, acceptedTypes = "*/*", modality }: UploadDropzoneProps) {
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      onFileSelect(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      onFileSelect(e.target.files[0]);
    }
  };

  return (
    <div
      className={`mt-1 flex justify-center px-6 pt-5 pb-6 border-2 border-dashed rounded-md transition-colors ${
        isDragging ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-gray-400'
      }`}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      onClick={() => fileInputRef.current?.click()}
    >
      <div className="space-y-1 text-center cursor-pointer">
        <svg className="mx-auto h-12 w-12 text-gray-400" stroke="currentColor" fill="none" viewBox="0 0 48 48" aria-hidden="true">
          <path d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
        <div className="flex text-sm text-gray-600 justify-center">
          <label className="relative cursor-pointer rounded-md font-medium text-blue-600 hover:text-blue-500 focus-within:outline-none">
            <span>Upload a file</span>
            <input ref={fileInputRef} type="file" className="sr-only" accept={acceptedTypes} onChange={handleChange} />
          </label>
          <p className="pl-1">or drag and drop</p>
        </div>
        <p className="text-xs text-gray-500">Supported for {modality} analysis</p>
      </div>
    </div>
  );
}
