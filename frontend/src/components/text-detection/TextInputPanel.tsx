'use client';
import React, { useState } from 'react';
import { Button } from '@/components/ui/Button';

interface TextInputPanelProps {
  onSubmit: (text: string) => void;
  isLoading: boolean;
}

export function TextInputPanel({ onSubmit, isLoading }: TextInputPanelProps) {
  const [text, setText] = useState('');
  const [error, setError] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const wordCount = text.trim().split(/\s+/).filter(word => word.length > 0).length;
    if (wordCount < 10) {
      setError(`Please enter at least 10 words. (Currently: ${wordCount})`);
      return;
    }
    setError('');
    onSubmit(text);
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4 h-full flex flex-col">
      <div className="flex-grow flex flex-col">
        <label htmlFor="text-input" className="block text-sm font-medium text-gray-700 mb-2">
          Text to Analyze
        </label>
        <textarea
          id="text-input"
          className="w-full flex-grow min-h-[200px] p-4 border border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 text-gray-900 resize-none"
          placeholder="Paste or type at least 10 words here to detect if it's AI-generated..."
          value={text}
          onChange={(e) => {
            setText(e.target.value);
            if (error) setError('');
          }}
          disabled={isLoading}
        />
        {error && <p className="mt-2 text-sm text-red-600 font-medium">{error}</p>}
      </div>
      <Button type="submit" disabled={isLoading} className="w-full sm:w-auto py-3 text-base">
        {isLoading ? 'Analyzing Text...' : 'Analyze Text'}
      </Button>
    </form>
  );
}
