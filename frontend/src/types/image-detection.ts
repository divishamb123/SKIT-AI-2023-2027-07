/**
 * image-detection.ts — TypeScript Schemas for Sprint 2 Image Detection Interface
 *
 * Student: Divisha Manak Bohra (23ESKCA038)
 * Sprint: Sprint 2 — Baseline Detection Service
 * Tasks:
 *   - Task 4: Designing image upload and result interaction components (Weeks 5 & 6)
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

// --------------------------------------------------------------------------
// Sprint 2 Week 6: Advanced Result Interaction & Evidence Inspection Schemas
// --------------------------------------------------------------------------

export type ConfidenceTier =
  | 'HIGH_AI'
  | 'SUSPICIOUS_AI'
  | 'AMBIGUOUS'
  | 'PLAUSIBLE_REAL'
  | 'HIGH_REAL';

export interface ThresholdSimulationResult {
  threshold: number; // e.g. 0.50
  adjustedVerdict: 'REAL' | 'AI_GENERATED';
  margin: number;
  confidenceTier: ConfidenceTier;
  confidenceLabel: string;
  riskLevel: 'critical' | 'high' | 'moderate' | 'low' | 'minimal';
}

export interface ChannelDecomposition {
  redUrl: string;
  greenUrl: string;
  blueUrl: string;
  luminanceUrl: string;
  highPassUrl: string;
}

export interface ForensicAuditReport {
  caseId: string;
  evidenceHashSha256: string;
  evidenceFileName: string;
  timestamp: string;
  analyst: string;
  department: string;
  institution: string;
  rawVerdict: 'REAL' | 'AI_GENERATED';
  calibratedVerdict: 'REAL' | 'AI_GENERATED';
  thresholdUsed: number;
  confidencePercentage: number;
  probabilities: ImageProbabilities;
  imageMetadata: ImageMetadata;
  latencyMs: number;
  modelVersion: string;
  hardwareDevice: string;
  status: 'CERTIFIED' | 'PRELIMINARY';
}
