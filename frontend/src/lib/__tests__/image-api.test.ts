import { describe, it, expect, vi, beforeEach } from 'vitest';
import { detectImage, detectImageBase64, detectImageBatch, imageApiClient } from '../api';
import { ImageDetectionResponse, BatchImageDetectionResponse } from '@/types/image-detection';

describe('Image Detection API client (Member 1: Divisha Manak Bohra)', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it('detectImage uploads a single file via multipart form-data and returns ImageDetectionResponse', async () => {
    const mockResponse: ImageDetectionResponse = {
      filename: 'sample.png',
      verdict: 'AI_GENERATED',
      is_ai: true,
      confidence: 0.9942,
      probabilities: { real: 0.0058, ai_generated: 0.9942 },
      image_metadata: { width: 512, height: 512, channels: 3, format: 'PNG' },
      latency_ms: 14.5,
      device: 'mps',
      model_version: 'baseline-resnet18-v1.0',
      request_id: 'test-req-001',
    };

    const mockPost = vi.spyOn(imageApiClient, 'post').mockResolvedValueOnce({
      data: mockResponse,
    });

    const file = new File(['fake-image-bytes'], 'sample.png', { type: 'image/png' });
    const result = await detectImage(file);

    expect(mockPost).toHaveBeenCalledWith(
      '/api/detect/image',
      expect.any(FormData),
      expect.objectContaining({
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 30000,
      })
    );
    expect(result).toEqual(mockResponse);
    expect(result.verdict).toBe('AI_GENERATED');
    expect(result.confidence).toBe(0.9942);
    expect(result.model_version).toBe('baseline-resnet18-v1.0');
    expect(result.request_id).toBe('test-req-001');
  });

  it('detectImageBase64 sends JSON payload with Base64 string and filename', async () => {
    const mockResponse: ImageDetectionResponse = {
      filename: 'test-photo.png',
      verdict: 'REAL',
      is_ai: false,
      confidence: 0.9123,
      probabilities: { real: 0.9123, ai_generated: 0.0877 },
      image_metadata: { width: 256, height: 256, channels: 3, format: 'JPEG' },
      latency_ms: 8.2,
      device: 'mps',
      model_version: 'baseline-resnet18-v1.0',
      request_id: 'test-req-002',
    };

    const mockPost = vi.spyOn(imageApiClient, 'post').mockResolvedValueOnce({
      data: mockResponse,
    });

    const base64Data =
      'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=';
    const result = await detectImageBase64(base64Data, 'test-photo.png');

    expect(mockPost).toHaveBeenCalledWith(
      '/api/detect/image/base64',
      {
        image_data: base64Data,
        filename: 'test-photo.png',
      },
      expect.objectContaining({
        headers: { 'Content-Type': 'application/json' },
        timeout: 30000,
      })
    );
    expect(result.verdict).toBe('REAL');
    expect(result.confidence).toBe(0.9123);
    expect(result.is_ai).toBe(false);
  });

  it('detectImageBatch uploads multiple files via multipart form-data with batch ceiling verification', async () => {
    const mockBatchResponse: BatchImageDetectionResponse = {
      batch_size: 2,
      successful_count: 2,
      failed_count: 0,
      results: [
        {
          index: 0,
          filename: 'image_1.png',
          verdict: 'REAL',
          is_ai: false,
          confidence: 0.92,
          probabilities: { real: 0.92, ai_generated: 0.08 },
          image_metadata: { width: 128, height: 128, channels: 3, format: 'PNG' },
          latency_ms: 6.4,
        },
        {
          index: 1,
          filename: 'image_2.png',
          verdict: 'AI_GENERATED',
          is_ai: true,
          confidence: 0.98,
          probabilities: { real: 0.02, ai_generated: 0.98 },
          image_metadata: { width: 128, height: 128, channels: 3, format: 'PNG' },
          latency_ms: 6.8,
        },
      ],
      errors: [],
      total_latency_ms: 13.2,
      avg_latency_ms: 6.6,
      throughput_fps: 151.5,
      device: 'mps',
      request_id: 'test-batch-001',
    };

    const mockPost = vi.spyOn(imageApiClient, 'post').mockResolvedValueOnce({
      data: mockBatchResponse,
    });

    const file1 = new File(['bytes1'], 'image_1.png', { type: 'image/png' });
    const file2 = new File(['bytes2'], 'image_2.png', { type: 'image/png' });

    const result = await detectImageBatch([file1, file2]);

    expect(mockPost).toHaveBeenCalledWith(
      '/api/detect/image/batch',
      expect.any(FormData),
      expect.objectContaining({
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 60000,
      })
    );
    expect(result.batch_size).toBe(2);
    expect(result.successful_count).toBe(2);
    expect(result.results).toHaveLength(2);
    expect(result.errors).toHaveLength(0);
    expect(result.throughput_fps).toBe(151.5);
  });
});
