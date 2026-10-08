import axios from 'axios';
import { DetectionVerdict } from '@/components/types';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api';

export const apiClient = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  // timeout: 10000,
});

export async function detectText(text: string): Promise<DetectionVerdict> {
  const response = await apiClient.post('/v1/detect/text', { text });
  const { label, confidence } = response.data;
  return {
    isAI: label === 'ai',
    confidence,
    label: label === 'ai' ? 'AI Generated' : 'Human Written',
  };
}

