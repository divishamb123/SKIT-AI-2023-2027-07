import React from 'react';
import { DetectionVerdict } from '../types';
import { Card, CardContent, CardHeader } from '@/components/ui/Card';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';

interface VerdictDisplayProps {
  verdict: DetectionVerdict | null;
  isLoading: boolean;
}

export function VerdictDisplay({ verdict, isLoading }: VerdictDisplayProps) {
  if (isLoading) {
    return (
      <Card className="w-full h-full flex items-center justify-center min-h-[300px]">
        <CardContent className="flex flex-col items-center justify-center p-8">
          <LoadingSpinner className="h-10 w-10 text-blue-600 mb-6" />
          <p className="text-gray-600 font-medium text-lg animate-pulse">Analyzing text patterns...</p>
        </CardContent>
      </Card>
    );
  }

  if (!verdict) {
    return (
      <Card className="w-full h-full flex items-center justify-center min-h-[300px] bg-gray-50 border-dashed border-2">
        <CardContent className="text-center p-8">
          <p className="text-gray-500 text-lg">Submit text to view the detection verdict here.</p>
        </CardContent>
      </Card>
    );
  }

  const confidencePercentage = Math.round(verdict.confidence * 100);
  const barColor = verdict.isAI ? 'bg-red-500' : 'bg-green-500';
  const headerColor = verdict.isAI ? 'text-red-700' : 'text-green-700';

  return (
    <Card className="w-full h-full flex flex-col overflow-hidden min-h-[300px]">
      <CardHeader className="bg-gray-50 border-b border-gray-100 flex-shrink-0">
        <h3 className="text-lg font-medium text-gray-900">Analysis Result</h3>
      </CardHeader>
      <CardContent className="p-8 flex-grow flex flex-col justify-center">
        <div className="flex flex-col space-y-8">
          <div className="text-center">
            <span className="text-sm font-semibold uppercase tracking-wider text-gray-500 mb-2 block">Verdict</span>
            <h4 className={`text-4xl font-extrabold ${headerColor}`}>{verdict.label}</h4>
          </div>
          
          <div className="space-y-4">
            <div className="flex justify-between text-sm font-medium">
              <span className="text-gray-600">Confidence Score</span>
              <span className="text-gray-900 text-lg">{confidencePercentage}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-4 overflow-hidden">
              <div 
                className={`${barColor} h-4 rounded-full transition-all duration-1000 ease-out`}
                style={{ width: `${confidencePercentage}%` }}
              />
            </div>
            <p className="text-sm text-gray-500 text-center mt-4">
              Our model is <span className="font-semibold">{confidencePercentage}%</span> confident that this text was written by <span className="font-semibold">{verdict.isAI ? 'an AI' : 'a human'}</span>.
            </p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
