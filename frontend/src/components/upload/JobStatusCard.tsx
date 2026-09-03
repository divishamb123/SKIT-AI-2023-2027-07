import React from 'react';
import { Card, CardContent } from '../ui/Card';
import { LoadingSpinner } from '../ui/LoadingSpinner';

export type JobStatus = 'pending' | 'processing' | 'completed' | 'failed';

interface JobStatusCardProps {
  jobId: string;
  status: JobStatus;
  modality: 'text' | 'image' | 'audio';
  result?: string;
  error?: string;
}

export function JobStatusCard({ jobId, status, modality, result, error }: JobStatusCardProps) {
  const statusColors = {
    pending: 'text-gray-500',
    processing: 'text-blue-500',
    completed: 'text-green-500',
    failed: 'text-red-500',
  };

  return (
    <Card className="w-full">
      <CardContent className="flex flex-col space-y-4">
        <div className="flex justify-between items-center">
          <div className="text-sm font-medium text-gray-900">Job ID: {jobId}</div>
          <div className={`text-sm font-semibold flex items-center space-x-2 ${statusColors[status]}`}>
            {status === 'processing' && <LoadingSpinner className="h-4 w-4" />}
            <span className="capitalize">{status}</span>
          </div>
        </div>
        <div className="text-sm text-gray-500 capitalize">Modality: {modality}</div>
        {status === 'completed' && result && (
          <div className="mt-2 p-3 bg-gray-50 rounded-md border border-gray-200 text-sm">
            <strong>Result:</strong> {result}
          </div>
        )}
        {status === 'failed' && error && (
          <div className="mt-2 p-3 bg-red-50 rounded-md border border-red-200 text-sm text-red-700">
            <strong>Error:</strong> {error}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
