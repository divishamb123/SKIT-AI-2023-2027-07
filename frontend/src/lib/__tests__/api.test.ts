import { describe, it, expect, vi, beforeEach } from 'vitest';
import axios, { InternalAxiosRequestConfig } from 'axios';
import { detectText, apiClient } from '../api';

describe('detectText adapter', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it('maps "ai" label to isAI=true and "AI Generated", passing confidence and modelVersion', async () => {
    const mockPost = vi.spyOn(apiClient, 'post').mockResolvedValueOnce({
      data: {
        label: 'ai',
        confidence: 0.985,
        model_version: 'microsoft/deberta-v3-base-pretrained',
      },
    });

    const result = await detectText('This is some test AI generated text here.');

    expect(mockPost).toHaveBeenCalledWith('/v1/detect/text', {
      text: 'This is some test AI generated text here.',
    });
    expect(result).toEqual({
      isAI: true,
      confidence: 0.985,
      label: 'AI Generated',
      modelVersion: 'microsoft/deberta-v3-base-pretrained',
    });
  });

  it('maps "human" label to isAI=false and "Human Written", passing confidence and modelVersion', async () => {
    const mockPost = vi.spyOn(apiClient, 'post').mockResolvedValueOnce({
      data: {
        label: 'human',
        confidence: 0.92,
        model_version: 'microsoft/deberta-v3-base-pretrained',
      },
    });

    const result = await detectText('This is completely human-written research text.');

    expect(mockPost).toHaveBeenCalledWith('/v1/detect/text', {
      text: 'This is completely human-written research text.',
    });
    expect(result).toEqual({
      isAI: false,
      confidence: 0.92,
      label: 'Human Written',
      modelVersion: 'microsoft/deberta-v3-base-pretrained',
    });
  });

  it('passes confidence and modelVersion correctly through the adapter', async () => {
    vi.spyOn(apiClient, 'post').mockResolvedValueOnce({
      data: {
        label: 'ai',
        confidence: 0.7654,
        model_version: 'custom-finetuned-deberta-v1',
      },
    });

    const result = await detectText('Sample evaluation text');
    expect(result.confidence).toBe(0.7654);
    expect(result.modelVersion).toBe('custom-finetuned-deberta-v1');
  });

  it('propagates an Axios 429 rate limit error', async () => {
    const error429 = new axios.AxiosError(
      'Request failed with status code 429',
      'ERR_BAD_REQUEST',
      undefined,
      undefined,
      {
        status: 429,
        statusText: 'Too Many Requests',
        data: { detail: 'Rate limit exceeded: 30 requests per minute.' },
        headers: {},
        config: { headers: {} } as unknown as InternalAxiosRequestConfig,
      }
    );

    vi.spyOn(apiClient, 'post').mockRejectedValueOnce(error429);

    await expect(detectText('Testing rate limit propagation')).rejects.toThrow(error429);
  });
});
