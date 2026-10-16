import axios from 'axios';
import { DetectionVerdict } from '@/components/types';
import {
  ImageDetectionResponse,
  BatchImageDetectionResponse,
} from '@/types/image-detection';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api';
const IMAGE_SERVICE_URL =
  process.env.NEXT_PUBLIC_IMAGE_SERVICE_URL || 'http://localhost:8004';

export const apiClient = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const imageApiClient = axios.create({
  baseURL: IMAGE_SERVICE_URL,
});

// --------------------------------------------------------------------------
// Text Detection (Member 2: Aryansh Agarwal)
// --------------------------------------------------------------------------
export async function detectText(text: string): Promise<DetectionVerdict> {
  const response = await apiClient.post('/v1/detect/text', { text });
  const { label, confidence, model_version } = response.data;
  return {
    isAI: label === 'ai',
    confidence,
    label: label === 'ai' ? 'AI Generated' : 'Human Written',
    modelVersion: model_version,
  };
}

// --------------------------------------------------------------------------
// Image Detection (Member 1: Divisha Manak Bohra - Sprint 2 Task 4)
// --------------------------------------------------------------------------
/**
 * Analyze a single image file via multipart form-data upload.
 */
export async function detectImage(file: File): Promise<ImageDetectionResponse> {
  const formData = new FormData();
  formData.append('file', file, file.name);

  const response = await imageApiClient.post<ImageDetectionResponse>(
    '/api/detect/image',
    formData,
    {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      timeout: 30000,
    }
  );
  return response.data;
}

/**
 * Analyze a Base64-encoded image string or RFC 2397 Data URI.
 */
export async function detectImageBase64(
  imageBase64: string,
  filename: string = 'upload.png'
): Promise<ImageDetectionResponse> {
  const response = await imageApiClient.post<ImageDetectionResponse>(
    '/api/detect/image/base64',
    {
      image_data: imageBase64,
      filename,
    },
    {
      headers: {
        'Content-Type': 'application/json',
      },
      timeout: 30000,
    }
  );
  return response.data;
}

/**
 * Analyze multiple image files concurrently via batched tensor inference.
 */
export async function detectImageBatch(
  files: File[]
): Promise<BatchImageDetectionResponse> {
  const formData = new FormData();
  for (const file of files) {
    formData.append('files', file, file.name);
  }

  const response = await imageApiClient.post<BatchImageDetectionResponse>(
    '/api/detect/image/batch',
    formData,
    {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      timeout: 60000,
    }
  );
  return response.data;
}
