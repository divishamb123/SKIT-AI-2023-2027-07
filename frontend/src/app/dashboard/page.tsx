'use client';

import React, { useState } from 'react';
import Link from 'next/link';
import { UploadDropzone } from '@/components/upload/UploadDropzone';
import { JobStatusCard } from '@/components/upload/JobStatusCard';
import { useAuth } from '@/context/AuthContext';
import { useIsClient } from '@/lib/useIsClient';

export default function DashboardPage() {
  const { user, isAuthenticated } = useAuth();
  const [activeTab, setActiveTab] = useState<'text' | 'image' | 'audio'>('image');
  const isClient = useIsClient();

  const showAuthUser = isClient && isAuthenticated && user;

  const tabs = [
    { id: 'image', label: 'Image Analysis (Sprint 1-3)' },
    { id: 'text', label: 'Text Analysis' },
    { id: 'audio', label: 'Audio Analysis' },
  ] as const;

  return (
    <div className="max-w-7xl mx-auto py-8 px-4 sm:px-6 lg:px-8">
      {/* Session greeting bar */}
      <div className="mb-6 p-4 rounded-xl bg-white border border-gray-200/80 shadow-xs flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-full bg-blue-600 text-white flex items-center justify-center font-bold text-sm shadow-xs">
            {showAuthUser ? user.name.charAt(0) : 'A'}
          </div>
          <div>
            <h2 className="text-base font-bold text-gray-900">
              {showAuthUser
                ? `Welcome back, ${user.name}`
                : 'SKIT Forensic Research Console'}
            </h2>
            <p className="text-xs text-gray-500">
              {showAuthUser
                ? `${user.role} • ${user.department}`
                : 'Viewing in guest session mode. Sign in to track ongoing forensic jobs.'}
            </p>
          </div>
        </div>

        {!showAuthUser && (
          <div className="flex items-center gap-2">
            <Link
              href="/login"
              className="px-3.5 py-1.5 text-xs font-semibold text-blue-600 bg-blue-50 hover:bg-blue-100 rounded-lg transition-colors"
            >
              Sign in
            </Link>
            <Link
              href="/register"
              className="px-3.5 py-1.5 text-xs font-semibold text-white bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors shadow-xs"
            >
              Create Account
            </Link>
          </div>
        )}
      </div>

      <div className="px-0 sm:px-0">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between mb-8 pb-4 border-b border-gray-200">
          <div>
            <h1 className="text-3xl font-extrabold text-gray-900 tracking-tight">
              Forensics Dashboard
            </h1>
            <p className="text-sm text-gray-500 mt-1">
              Multi-modal synthetic content evaluation suite (Text, Image & Audio).
            </p>
          </div>

          <div className="mt-3 sm:mt-0 flex items-center gap-2">
            <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-emerald-100 text-emerald-800">
              Dataset Splits: 132,000 Loaded
            </span>
          </div>
        </div>

        <div className="mb-8">
          <div className="sm:hidden">
            <select
              id="tabs"
              name="tabs"
              className="block w-full focus:ring-blue-500 focus:border-blue-500 border-gray-300 rounded-lg p-2.5 bg-white text-sm"
              value={activeTab}
              onChange={(e) =>
                setActiveTab(e.target.value as 'text' | 'image' | 'audio')
              }
            >
              {tabs.map((tab) => (
                <option key={tab.id} value={tab.id}>
                  {tab.label}
                </option>
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
                      whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm transition-colors cursor-pointer
                      ${
                        activeTab === tab.id
                          ? 'border-blue-600 text-blue-600 font-semibold'
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

        <div className="bg-white shadow-sm border border-gray-200/80 rounded-2xl p-6 sm:p-8 mb-8">
          <div className="md:grid md:grid-cols-3 md:gap-6">
            <div className="md:col-span-1">
              <h3 className="text-lg font-semibold text-gray-900">
                Upload new evidence
              </h3>
              <p className="mt-2 text-sm text-gray-500 leading-relaxed">
                Select a media file for synthetic generator classification.
                Currently viewing <span className="font-semibold text-gray-700 capitalize">{activeTab}</span> modality.
              </p>
              <div className="mt-4 p-3 bg-blue-50/50 rounded-lg border border-blue-100 text-xs text-blue-800 space-y-1">
                <p className="font-semibold">Sprint 1 Partition Baseline:</p>
                <p>• 90k Train / 10k Val / 20k Test / 12k Holdout</p>
                <p>• CIFAKE & GenImage generators</p>
              </div>
            </div>
            <div className="mt-5 md:mt-0 md:col-span-2">
              <UploadDropzone
                modality={activeTab}
                onFileSelect={(file) => console.log('File selected:', file)}
                acceptedTypes={
                  activeTab === 'image'
                    ? 'image/*'
                    : activeTab === 'audio'
                    ? 'audio/*'
                    : '.txt,.csv,.json'
                }
              />
            </div>
          </div>
        </div>

        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-semibold text-gray-900">
              Recent Forensic Jobs
            </h3>
            <span className="text-xs text-gray-500">Live Mock Pipeline</span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <JobStatusCard
              jobId="job-img-001"
              status="completed"
              modality="image"
              result="99.4% Real Photographic (CIFAKE / CIFAR-10 baseline)"
            />
            <JobStatusCard
              jobId="job-img-002"
              status="completed"
              modality="image"
              result="98.1% AI-Generated (Stable Diffusion v1.4)"
            />
            <JobStatusCard
              jobId="job-txt-001"
              status="processing"
              modality="text"
            />
            <JobStatusCard
              jobId="job-aud-001"
              status="failed"
              modality="audio"
              error="Awaiting Sprint 2 Speech Model Integration"
            />
          </div>
        </div>
      </div>
    </div>
  );
}
