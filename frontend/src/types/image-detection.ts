/**
 * image-detection.ts — TypeScript Schemas for Sprint 2 Week 5 Image Detection Interface
 *
 * Student: Divisha Manak Bohra (23ESKCA038)
 * Sprint: Sprint 2 — Baseline Detection Service
 * Task 4: Designing image upload and result interaction components (Week 5)
 */

export interface ImageMetadata {
  width: number;
  height: number;
  channels: number;
  format: string;
}

export interface ImageProbabilities {
  real: number;
  ai_generated: number;
}

export interface ImageDetectionResponse {
  filename: string;
  verdict: 'REAL' | 'AI_GENERATED';
  is_ai: boolean;
  confidence: number;
  probabilities: ImageProbabilities;
  image_metadata: ImageMetadata;
  latency_ms: number;
  model_version: string;
  device: string;
  request_id: string;
}

export interface BatchImageItem {
  index: number;
  filename: string;
  verdict: 'REAL' | 'AI_GENERATED';
  is_ai: boolean;
  confidence: number;
  probabilities: ImageProbabilities;
  image_metadata: ImageMetadata;
  latency_ms: number;
}

export interface BatchImageErrorItem {
  index: number;
  filename: string;
  error_code: string;
  error_message: string;
}

export interface BatchImageDetectionResponse {
  batch_size: number;
  successful_count: number;
  failed_count: number;
  results: BatchImageItem[];
  errors: BatchImageErrorItem[];
  total_latency_ms: number;
  avg_latency_ms: number;
  throughput_fps: number;
  device: string;
  request_id: string;
}

export interface SelectedImageFile {
  file: File;
  previewUrl: string;
  id: string;
  dimensions?: { width: number; height: number };
}
