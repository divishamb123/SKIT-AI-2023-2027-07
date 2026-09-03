import React from 'react';
import Link from 'next/link';
import { Button } from '@/components/ui/Button';

export default function HomePage() {
  return (
    <div className="bg-white">
      <div className="max-w-7xl mx-auto py-16 px-4 sm:py-24 sm:px-6 lg:px-8">
        <div className="text-center">
          <h1 className="text-4xl font-extrabold text-gray-900 sm:text-5xl sm:tracking-tight lg:text-6xl">
            Detect AI-Generated Content
          </h1>
          <p className="max-w-xl mt-5 mx-auto text-xl text-gray-500">
            A comprehensive multi-model system for verifying the authenticity of text, images, and audio. Ensure trust in digital media.
          </p>
          <div className="mt-8 flex justify-center gap-4">
            <Link href="/dashboard">
              <Button variant="primary" className="text-lg px-8 py-3">
                Go to Dashboard
              </Button>
            </Link>
            <Link href="/register">
              <Button variant="outline" className="text-lg px-8 py-3">
                Create Account
              </Button>
            </Link>
          </div>
        </div>
      </div>
      
      <div className="bg-gray-50 py-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 text-center">
            <div className="bg-white p-6 rounded-lg shadow-sm">
              <div className="text-blue-500 text-4xl mb-4">📝</div>
              <h3 className="text-lg font-medium text-gray-900 mb-2">Text Analysis</h3>
              <p className="text-gray-500">Detect LLM-generated text from GPT, Llama, and other modern language models.</p>
            </div>
            <div className="bg-white p-6 rounded-lg shadow-sm">
              <div className="text-blue-500 text-4xl mb-4">🖼️</div>
              <h3 className="text-lg font-medium text-gray-900 mb-2">Image Analysis</h3>
              <p className="text-gray-500">Identify AI imagery from Diffusion models and GANs with robust classifiers.</p>
            </div>
            <div className="bg-white p-6 rounded-lg shadow-sm">
              <div className="text-blue-500 text-4xl mb-4">🎙️</div>
              <h3 className="text-lg font-medium text-gray-900 mb-2">Audio Analysis</h3>
              <p className="text-gray-500">Verify voice authenticity and detect deepfake synthetic audio samples.</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
