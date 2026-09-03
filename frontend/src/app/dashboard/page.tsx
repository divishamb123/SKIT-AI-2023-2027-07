'use client';

import React, { useState } from 'react';
import { UploadDropzone } from '@/components/upload/UploadDropzone';
import { JobStatusCard } from '@/components/upload/JobStatusCard';

export default function DashboardPage() {
  const [activeTab, setActiveTab] = useState<'text' | 'image' | 'audio'>('text');
  
  const tabs = [
    { id: 'text', label: 'Text Analysis' },
    { id: 'image', label: 'Image Analysis' },
    { id: 'audio', label: 'Audio Analysis' },
  ] as const;

  return (
    <div className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
      <div className="px-4 py-6 sm:px-0">
        <h1 className="text-3xl font-bold text-gray-900 mb-8">Forensics Dashboard</h1>
        
        <div className="mb-8">
          <div className="sm:hidden">
            <select
              id="tabs"
              name="tabs"
              className="block w-full focus:ring-blue-500 focus:border-blue-500 border-gray-300 rounded-md"
              value={activeTab}
              onChange={(e) => setActiveTab(e.target.value as 'text' | 'image' | 'audio')}
            >
              {tabs.map((tab) => (
                <option key={tab.id} value={tab.id}>{tab.label}</option>
              ))}
            </select>
          </div>
          <div className="hidden sm:block">
            <div className="border-b border-gray-200">
              <nav className="-mb-px flex space-x-8" aria-label="Tabs">
                {tabs.map((tab) => (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={`
                      whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm
                      ${activeTab === tab.id
                        ? 'border-blue-500 text-blue-600'
                        : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                      }
                    `}
                  >
                    {tab.label}
                  </button>
                ))}
              </nav>
            </div>
          </div>
        </div>

        <div className="bg-white shadow px-4 py-5 sm:rounded-lg sm:p-6 mb-8">
          <div className="md:grid md:grid-cols-3 md:gap-6">
            <div className="md:col-span-1">
              <h3 className="text-lg font-medium leading-6 text-gray-900">Upload new evidence</h3>
              <p className="mt-1 text-sm text-gray-500">
                Select a file for AI generation analysis. 
                Currently supporting {activeTab} modalities.
              </p>
            </div>
            <div className="mt-5 md:mt-0 md:col-span-2">
              <UploadDropzone
                modality={activeTab}
                onFileSelect={(file) => console.log('File selected:', file)}
                acceptedTypes={activeTab === 'image' ? 'image/*' : activeTab === 'audio' ? 'audio/*' : '.txt,.csv,.json'}
              />
            </div>
          </div>
        </div>

        <div className="space-y-4">
          <h3 className="text-lg font-medium leading-6 text-gray-900">Recent Analysis Jobs</h3>
          {/* Placeholders for actual backend data */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <JobStatusCard jobId="job-001" status="completed" modality="image" result="98% Real (CIFAKE)" />
            <JobStatusCard jobId="job-002" status="processing" modality="text" />
            <JobStatusCard jobId="job-003" status="failed" modality="audio" error="Unsupported encoding format" />
          </div>
        </div>
      </div>
    </div>
  );
}
