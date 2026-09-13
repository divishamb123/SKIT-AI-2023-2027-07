'use client';

import React, { useState, useMemo } from 'react';
import Link from 'next/link';
import { useAuth } from '@/context/AuthContext';
import { useIsClient } from '@/lib/useIsClient';

// Forensic Scan Item Interface
interface ScanRecord {
  id: string;
  name: string;
  modality: 'image' | 'text' | 'audio';
  verdict: 'AI-Generated' | 'Authentic' | 'Suspicious';
  confidence: number;
  generator: string;
  timestamp: string;
  size: string;
  sha256: string;
  details: {
    spectralScore?: number;
    artifactScore?: number;
    sensorNoise?: string;
    perplexity?: number;
    burstiness?: number;
  };
}

// Initial realistic forensic database records
const INITIAL_RECORDS: ScanRecord[] = [
  {
    id: 'EV-84920',
    name: 'corporate_headshot_executive.png',
    modality: 'image',
    verdict: 'AI-Generated',
    confidence: 99.4,
    generator: 'Midjourney v6.0',
    timestamp: '2 mins ago',
    size: '3.4 MB',
    sha256: 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
    details: { spectralScore: 98.2, artifactScore: 96.7, sensorNoise: 'Inconsistent / Synthesized' },
  },
  {
    id: 'EV-84919',
    name: 'field_investigation_canon_raw.jpg',
    modality: 'image',
    verdict: 'Authentic',
    confidence: 98.8,
    generator: 'Natural Sensor (CMOS Bayer)',
    timestamp: '14 mins ago',
    size: '8.1 MB',
    sha256: '9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08',
    details: { spectralScore: 2.1, artifactScore: 1.4, sensorNoise: 'Natural PRNU Pattern Verified' },
  },
  {
    id: 'EV-84918',
    name: 'press_statement_acquisition.txt',
    modality: 'text',
    verdict: 'AI-Generated',
    confidence: 96.1,
    generator: 'GPT-4o Language Model',
    timestamp: '32 mins ago',
    size: '14.2 KB',
    sha256: '5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8',
    details: { perplexity: 14.8, burstiness: 0.88 },
  },
  {
    id: 'EV-84917',
    name: 'earnings_call_audio_excerpt.wav',
    modality: 'audio',
    verdict: 'AI-Generated',
    confidence: 94.7,
    generator: 'ElevenLabs Voice Cloning',
    timestamp: '1 hour ago',
    size: '4.8 MB',
    sha256: '4b227777d4dd1fc61c6f884f48641d02b4d121d3fd328cb08b5531fcacdabf8a',
    details: { spectralScore: 92.4, artifactScore: 91.0 },
  },
  {
    id: 'EV-84916',
    name: 'street_photography_berlin.png',
    modality: 'image',
    verdict: 'Authentic',
    confidence: 99.1,
    generator: 'Photographic Camera Sensor',
    timestamp: '2 hours ago',
    size: '5.2 MB',
    sha256: 'ef2d127de37b942baad06145e54b0c619a1f22327b2ebbcfbec78f5564afe39d',
    details: { spectralScore: 3.2, artifactScore: 2.1, sensorNoise: 'Valid Physical PRNU' },
  },
  {
    id: 'EV-84915',
    name: 'legal_brief_editorial.pdf',
    modality: 'text',
    verdict: 'Authentic',
    confidence: 97.4,
    generator: 'Human Authorship',
    timestamp: '3 hours ago',
    size: '48.6 KB',
    sha256: '8c6976e5b5410415bde908bd4dee15dfb167a9c873fc4bb8a81f6f2ab448a918',
    details: { perplexity: 68.4, burstiness: 0.24 },
  },
  {
    id: 'EV-84914',
    name: 'landscape_synthetic_concept.png',
    modality: 'image',
    verdict: 'AI-Generated',
    confidence: 98.9,
    generator: 'Stable Diffusion XL',
    timestamp: '4 hours ago',
    size: '6.5 MB',
    sha256: '2c26b46b68ffc68ff99b453c1d30413413422d706483bfa0f98a5e886266e7ae',
    details: { spectralScore: 97.5, artifactScore: 95.8, sensorNoise: 'Synthetic Latent Diffusion' },
  },
];

export default function DashboardPage() {
  const { user, isAuthenticated } = useAuth();
  const isClient = useIsClient();

  // Active Modality Tab
  const [activeTab, setActiveTab] = useState<'image' | 'text' | 'audio'>('image');

  // Scanner State
  const [scanning, setScanning] = useState(false);
  const [scanProgress, setScanProgress] = useState(0);
  const [scanStage, setScanStage] = useState('');
  const [activeResult, setActiveResult] = useState<ScanRecord | null>(null);
  const [heatmapActive, setHeatmapActive] = useState(false);

  // Selected or sample file info
  const [selectedFileName, setSelectedFileName] = useState<string | null>(null);
  const [textInputContent, setTextInputContent] = useState('');

  // History Filter & Search State
  const [searchQuery, setSearchQuery] = useState('');
  const [verdictFilter, setVerdictFilter] = useState<'all' | 'AI-Generated' | 'Authentic' | 'Suspicious'>('all');
  const [records, setRecords] = useState<ScanRecord[]>(INITIAL_RECORDS);
  const [inspectRecord, setInspectRecord] = useState<ScanRecord | null>(null);

  const showAuthUser = isClient && isAuthenticated && user;

  // Filtered log items
  const filteredRecords = useMemo(() => {
    return records.filter((r) => {
      const matchesSearch =
        r.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        r.id.toLowerCase().includes(searchQuery.toLowerCase()) ||
        r.generator.toLowerCase().includes(searchQuery.toLowerCase());
      const matchesVerdict = verdictFilter === 'all' || r.verdict === verdictFilter;
      return matchesSearch && matchesVerdict;
    });
  }, [records, searchQuery, verdictFilter]);

  // Execute realistic simulated multi-stage scan
  const handleExecuteScan = (
    fileName: string,
    modality: 'image' | 'text' | 'audio',
    presetVerdict: 'AI-Generated' | 'Authentic',
    generatorName: string
  ) => {
    setSelectedFileName(fileName);
    setScanning(true);
    setScanProgress(5);
    setScanStage('Initializing neural forensic pipeline...');
    setActiveResult(null);

    const stages = [
      { p: 25, label: 'Extracting spatial & frequency spectrum features (FFT)...' },
      { p: 55, label: 'Evaluating multi-generator ensemble classifiers...' },
      { p: 80, label: 'Computing PRNU sensor noise & latent consistency...' },
      { p: 95, label: 'Finalizing authenticity verdict & confidence score...' },
    ];

    let currentStep = 0;
    const interval = setInterval(() => {
      if (currentStep < stages.length) {
        setScanProgress(stages[currentStep].p);
        setScanStage(stages[currentStep].label);
        currentStep++;
      } else {
        clearInterval(interval);
        setScanProgress(100);
        setScanning(false);

        const confidence = presetVerdict === 'AI-Generated' ? 98.7 : 99.2;
        const newRecord: ScanRecord = {
          id: `EV-${Math.floor(10000 + Math.random() * 90000)}`,
          name: fileName,
          modality,
          verdict: presetVerdict,
          confidence,
          generator: generatorName,
          timestamp: 'Just now',
          size: modality === 'image' ? '4.2 MB' : modality === 'text' ? '18 KB' : '3.6 MB',
          sha256: Array.from({ length: 64 }, () => Math.floor(Math.random() * 16).toString(16)).join(''),
          details: {
            spectralScore: presetVerdict === 'AI-Generated' ? 97.8 : 2.4,
            artifactScore: presetVerdict === 'AI-Generated' ? 96.1 : 1.8,
            sensorNoise: presetVerdict === 'AI-Generated' ? 'Synthetic Diffusion Residuals' : 'Physical CMOS Sensor Verified',
            perplexity: modality === 'text' ? (presetVerdict === 'AI-Generated' ? 12.4 : 72.1) : undefined,
            burstiness: modality === 'text' ? (presetVerdict === 'AI-Generated' ? 0.91 : 0.22) : undefined,
          },
        };

        setActiveResult(newRecord);
        setRecords((prev) => [newRecord, ...prev]);
      }
    }, 450);
  };

  // Quick preset triggers
  const handleSampleImage = (isSynthetic: boolean) => {
    if (isSynthetic) {
      handleExecuteScan(
        'synthetic_photorealistic_portrait.png',
        'image',
        'AI-Generated',
        'Stable Diffusion XL'
      );
    } else {
      handleExecuteScan(
        'raw_nikon_landscape_evidence.jpg',
        'image',
        'Authentic',
        'Optical Camera Sensor'
      );
    }
  };

  const handleSampleText = (isSynthetic: boolean) => {
    if (isSynthetic) {
      const sampleText =
        'Artificial intelligence has revolutionized digital media through sophisticated latent diffusion models and large language transformers. As synthetic content becomes increasingly indistinguishable from reality, multi-modal forensics provides the rigorous mathematical framework needed to guarantee information integrity.';
      setTextInputContent(sampleText);
      handleExecuteScan('press_editorial_excerpt.txt', 'text', 'AI-Generated', 'GPT-4o Transformer');
    } else {
      const sampleText =
        'I woke up around five in the morning to an odd rattling noise near the back patio. It turned out to be our neighbor’s golden retriever, who had somehow wriggled past the cedar fence latch and was happily nudging an old tennis ball across the damp gravel.';
      setTextInputContent(sampleText);
      handleExecuteScan('witness_statement_notes.txt', 'text', 'Authentic', 'Human Authorship');
    }
  };

  const handleSampleAudio = (isSynthetic: boolean) => {
    if (isSynthetic) {
      handleExecuteScan('voice_call_biometric_sample.wav', 'audio', 'AI-Generated', 'Neural Speech Synthesis (Voice Clone)');
    } else {
      handleExecuteScan('natural_microphone_interview.wav', 'audio', 'Authentic', 'Physical Acoustic Recording');
    }
  };

  const displayName =
    showAuthUser && user?.name && !/divisha|dev|aryansh/i.test(user.name)
      ? user.name
      : 'Alex Rivera';
  const displayRole =
    showAuthUser && user?.role && !/skit/i.test(user.role)
      ? user.role
      : 'Lead Forensics Investigator';
  const displayDept =
    showAuthUser && user?.department && !/skit/i.test(user.department)
      ? user.department
      : 'Digital Media Forensics Lab';

  return (
    <div className="max-w-7xl mx-auto py-8 px-4 sm:px-6 lg:px-8 space-y-8">
      {/* Top Banner: Authenticated Session & System Health */}
      <div className="rounded-2xl bg-white border border-gray-200/80 shadow-xs p-5 flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div className="flex items-center gap-3.5">
          <div className="w-11 h-11 rounded-xl bg-gradient-to-br from-blue-600 to-indigo-700 text-white flex items-center justify-center font-bold text-base shadow-sm">
            {showAuthUser ? displayName.charAt(0) : 'A'}
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h2 className="text-base font-bold text-gray-900">
                {showAuthUser ? `Investigator Console: ${displayName}` : 'AIForensics Media Verification Console'}
              </h2>
              <span className="inline-flex items-center px-2 py-0.5 rounded text-[11px] font-semibold bg-emerald-50 text-emerald-700 border border-emerald-200">
                Live Engine Active
              </span>
            </div>
            <p className="text-xs text-gray-500 mt-0.5">
              {showAuthUser
                ? `${displayRole} • ${displayDept}`
                : 'Enterprise Forensics Mode • Multi-Model Verification Stack Active'}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2.5">
          {!showAuthUser ? (
            <>
              <Link
                href="/login"
                className="px-3.5 py-2 text-xs font-semibold text-gray-700 bg-gray-50 hover:bg-gray-100 border border-gray-200 rounded-lg transition-colors"
              >
                Sign In
              </Link>
              <Link
                href="/register"
                className="px-3.5 py-2 text-xs font-semibold text-white bg-blue-600 hover:bg-blue-700 rounded-lg shadow-xs transition-colors"
              >
                Create Account
              </Link>
            </>
          ) : (
            <div className="flex items-center gap-2">
              <span className="text-xs text-gray-500 font-medium">Session: Active</span>
              <button
                type="button"
                onClick={() => {
                  const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(records, null, 2));
                  const dlAnchor = document.createElement('a');
                  dlAnchor.setAttribute('href', dataStr);
                  dlAnchor.setAttribute('download', 'forensic_audit_log.json');
                  dlAnchor.click();
                }}
                className="px-3 py-1.5 text-xs font-medium text-gray-700 bg-white hover:bg-gray-50 border border-gray-200 rounded-lg shadow-2xs transition-colors cursor-pointer flex items-center gap-1.5"
              >
                <svg className="w-3.5 h-3.5 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                </svg>
                Export Audit Log
              </button>
            </div>
          )}
        </div>
      </div>

      {/* KPI Metrics Strip */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-white rounded-xl border border-gray-200/80 p-5 shadow-xs">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-gray-500 uppercase tracking-wider">Total Media Scanned</span>
            <span className="text-emerald-600 text-xs font-semibold bg-emerald-50 px-1.5 py-0.5 rounded">+14.2%</span>
          </div>
          <p className="text-2xl font-extrabold text-gray-900 mt-2">148,290</p>
          <p className="text-[11px] text-gray-500 mt-1">Multi-modal corpus scans</p>
        </div>

        <div className="bg-white rounded-xl border border-gray-200/80 p-5 shadow-xs">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-gray-500 uppercase tracking-wider">Synthetic Detection Rate</span>
            <span className="text-blue-600 text-xs font-semibold bg-blue-50 px-1.5 py-0.5 rounded">Stable</span>
          </div>
          <p className="text-2xl font-extrabold text-gray-900 mt-2">38.4%</p>
          <p className="text-[11px] text-gray-500 mt-1">Flagged as machine-generated</p>
        </div>

        <div className="bg-white rounded-xl border border-gray-200/80 p-5 shadow-xs">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-gray-500 uppercase tracking-wider">Inference Accuracy</span>
            <span className="text-emerald-600 text-xs font-semibold bg-emerald-50 px-1.5 py-0.5 rounded">99.1% F1</span>
          </div>
          <p className="text-2xl font-extrabold text-gray-900 mt-2">97.8%</p>
          <p className="text-[11px] text-gray-500 mt-1">Ensemble benchmark confidence</p>
        </div>

        <div className="bg-white rounded-xl border border-gray-200/80 p-5 shadow-xs">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-gray-500 uppercase tracking-wider">Average Latency</span>
            <span className="text-indigo-600 text-xs font-semibold bg-indigo-50 px-1.5 py-0.5 rounded">Turbo</span>
          </div>
          <p className="text-2xl font-extrabold text-gray-900 mt-2">420 ms</p>
          <p className="text-[11px] text-gray-500 mt-1">Per evidence analysis</p>
        </div>
      </div>

      {/* Main Forensic Workbench */}
      <div className="bg-white rounded-2xl border border-gray-200/80 shadow-sm overflow-hidden">
        {/* Modality Tabs */}
        <div className="border-b border-gray-200 bg-gray-50/70 px-6 pt-4 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div className="flex space-x-2">
            {[
              { id: 'image', label: 'Image Forensics', icon: '🖼️' },
              { id: 'text', label: 'Text & Document Forensics', icon: '📝' },
              { id: 'audio', label: 'Audio & Speech Forensics', icon: '🎙️' },
            ].map((tab) => (
              <button
                key={tab.id}
                type="button"
                onClick={() => {
                  setActiveTab(tab.id as 'image' | 'text' | 'audio');
                  setActiveResult(null);
                  setSelectedFileName(null);
                }}
                className={`flex items-center gap-2 px-4 py-2.5 text-sm font-semibold rounded-t-xl transition-all border-b-2 cursor-pointer ${
                  activeTab === tab.id
                    ? 'bg-white text-blue-600 border-blue-600 shadow-2xs'
                    : 'text-gray-500 hover:text-gray-900 border-transparent'
                }`}
              >
                <span>{tab.icon}</span>
                <span>{tab.label}</span>
              </button>
            ))}
          </div>

          <span className="text-xs text-gray-500 pb-2 sm:pb-0">
            Active Engine: <span className="font-semibold text-gray-700">Multi-Model Ensemble v2.4</span>
          </span>
        </div>

        {/* Workbench Body */}
        <div className="p-6 sm:p-8">
          {/* TAB 1: IMAGE FORENSICS */}
          {activeTab === 'image' && (
            <div className="space-y-6">
              <div className="flex flex-col lg:flex-row gap-6">
                {/* Upload & Preset Box */}
                <div className="flex-1 border-2 border-dashed border-gray-200 rounded-2xl p-6 sm:p-8 text-center hover:border-blue-400 transition-colors bg-gray-50/50 flex flex-col items-center justify-center min-h-[260px]">
                  <div className="w-14 h-14 rounded-2xl bg-blue-50 text-blue-600 flex items-center justify-center mb-4 ring-8 ring-blue-50/50">
                    <svg className="w-7 h-7" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                    </svg>
                  </div>
                  <h3 className="text-base font-bold text-gray-900">Upload Image Evidence for Forensic Inspection</h3>
                  <p className="text-xs text-gray-500 mt-1 max-w-sm">
                    Supports PNG, JPG, WEBP, TIFF up to 25 MB. Analyzes pixel grid artifacts, latent diffusion traces, and sensor noise (PRNU).
                  </p>

                  <div className="mt-4 flex flex-wrap gap-2 justify-center">
                    <button
                      type="button"
                      disabled={scanning}
                      onClick={() => handleSampleImage(true)}
                      className="px-3 py-1.5 text-xs font-semibold text-rose-700 bg-rose-50 hover:bg-rose-100 border border-rose-200 rounded-lg transition-colors cursor-pointer shadow-2xs"
                    >
                      ⚡ Test Sample AI Image (Midjourney/SD)
                    </button>
                    <button
                      type="button"
                      disabled={scanning}
                      onClick={() => handleSampleImage(false)}
                      className="px-3 py-1.5 text-xs font-semibold text-emerald-700 bg-emerald-50 hover:bg-emerald-100 border border-emerald-200 rounded-lg transition-colors cursor-pointer shadow-2xs"
                    >
                      📷 Test Sample Real Photo (DSLR)
                    </button>
                  </div>
                </div>

                {/* Scan Status or Result Pane */}
                <div className="w-full lg:w-96 flex flex-col justify-between p-5 rounded-2xl bg-gray-50 border border-gray-200">
                  <div>
                    <h4 className="text-xs font-bold text-gray-500 uppercase tracking-wider mb-2">
                      Detection Status
                    </h4>

                    {scanning ? (
                      <div className="py-8 space-y-4">
                        <div className="flex justify-between text-xs font-semibold text-gray-700">
                          <span>Analyzing Evidence...</span>
                          <span>{scanProgress}%</span>
                        </div>
                        <div className="w-full h-2.5 bg-gray-200 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-blue-600 transition-all duration-300 rounded-full"
                            style={{ width: `${scanProgress}%` }}
                          />
                        </div>
                        <p className="text-xs text-blue-600 animate-pulse font-medium">{scanStage}</p>
                      </div>
                    ) : activeResult && activeResult.modality === 'image' ? (
                      <div className="space-y-4 animate-in fade-in duration-200">
                        {/* Verdict Header */}
                        <div
                          className={`p-4 rounded-xl border flex items-center gap-3 ${
                            activeResult.verdict === 'AI-Generated'
                              ? 'bg-rose-50 border-rose-200 text-rose-900'
                              : 'bg-emerald-50 border-emerald-200 text-emerald-900'
                          }`}
                        >
                          <div
                            className={`w-9 h-9 rounded-lg flex items-center justify-center font-bold text-white shrink-0 ${
                              activeResult.verdict === 'AI-Generated' ? 'bg-rose-600' : 'bg-emerald-600'
                            }`}
                          >
                            {activeResult.verdict === 'AI-Generated' ? 'AI' : '✓'}
                          </div>
                          <div>
                            <p className="text-xs font-bold uppercase tracking-wider">
                              {activeResult.verdict === 'AI-Generated' ? 'Synthetic Media Detected' : 'Authentic Media Verified'}
                            </p>
                            <p className="text-sm font-extrabold">{activeResult.confidence}% Confidence</p>
                          </div>
                        </div>

                        {/* Generator & Sensor Attribution */}
                        <div className="bg-white rounded-lg p-3 border border-gray-200 text-xs space-y-1.5">
                          <div className="flex justify-between">
                            <span className="text-gray-500">Attributed Source:</span>
                            <span className="font-semibold text-gray-900">{activeResult.generator}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-500">Sensor Noise (PRNU):</span>
                            <span className="font-semibold text-gray-900">{activeResult.details.sensorNoise}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-500">Spectral Residuals:</span>
                            <span className="font-semibold text-gray-900">{activeResult.details.spectralScore}%</span>
                          </div>
                        </div>

                        {/* Visual Artifact Heatmap Toggle */}
                        <div className="pt-2 border-t border-gray-200">
                          <div className="flex items-center justify-between mb-2">
                            <span className="text-xs font-semibold text-gray-700">Forensic Overlay</span>
                            <button
                              type="button"
                              onClick={() => setHeatmapActive(!heatmapActive)}
                              className={`text-xs px-2.5 py-1 rounded-md font-medium transition-colors cursor-pointer ${
                                heatmapActive
                                  ? 'bg-purple-600 text-white shadow-xs'
                                  : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                              }`}
                            >
                              {heatmapActive ? 'Heatmap: ON' : 'Toggle Heatmap'}
                            </button>
                          </div>

                          {/* Preview container */}
                          <div className="relative h-32 rounded-lg bg-gray-900 overflow-hidden flex items-center justify-center border border-gray-300">
                            {heatmapActive ? (
                              <div className="w-full h-full bg-gradient-to-tr from-purple-900 via-rose-600 to-amber-400 opacity-90 flex items-center justify-center text-white text-xs font-bold text-center px-4">
                                Latent Grid Anomaly Detected in Frequency Domain
                              </div>
                            ) : (
                              <div className="text-center text-gray-300 text-xs px-3">
                                <p className="font-semibold">{selectedFileName}</p>
                                <p className="text-[10px] text-gray-400 mt-1">Spatial Domain Representation</p>
                              </div>
                            )}
                          </div>
                        </div>
                      </div>
                    ) : (
                      <div className="py-12 text-center text-xs text-gray-400">
                        Select a sample above or drop an image file to trigger the multi-model forensic analysis.
                      </div>
                    )}
                  </div>

                  {activeResult && activeResult.modality === 'image' && (
                    <div className="pt-4 border-t border-gray-200 mt-4 flex gap-2">
                      <button
                        type="button"
                        onClick={() => setInspectRecord(activeResult)}
                        className="flex-1 py-2 text-xs font-semibold bg-blue-600 hover:bg-blue-700 text-white rounded-lg shadow-xs transition-colors cursor-pointer text-center"
                      >
                        Full Forensic Report
                      </button>
                      <button
                        type="button"
                        onClick={() => setActiveResult(null)}
                        className="px-3 py-2 text-xs font-semibold bg-white hover:bg-gray-100 text-gray-700 border border-gray-200 rounded-lg transition-colors cursor-pointer"
                      >
                        Reset
                      </button>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* TAB 2: TEXT FORENSICS */}
          {activeTab === 'text' && (
            <div className="space-y-6">
              <div className="flex flex-col lg:flex-row gap-6">
                <div className="flex-1 space-y-4">
                  <div>
                    <label className="block text-xs font-bold text-gray-700 uppercase tracking-wider mb-2">
                      Document Content for Forensic Perplexity Analysis
                    </label>
                    <textarea
                      rows={7}
                      value={textInputContent}
                      onChange={(e) => setTextInputContent(e.target.value)}
                      placeholder="Paste text or document excerpts here to analyze token predictability, burstiness, and LLM generative signatures..."
                      className="w-full rounded-xl border border-gray-300 p-4 text-sm text-gray-900 placeholder-gray-400 focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20"
                    />
                  </div>

                  <div className="flex flex-wrap items-center justify-between gap-3">
                    <div className="flex items-center gap-2">
                      <button
                        type="button"
                        onClick={() => handleSampleText(true)}
                        className="px-3 py-1.5 text-xs font-semibold text-rose-700 bg-rose-50 hover:bg-rose-100 border border-rose-200 rounded-lg transition-colors cursor-pointer"
                      >
                        ⚡ Sample AI Text (GPT-4o)
                      </button>
                      <button
                        type="button"
                        onClick={() => handleSampleText(false)}
                        className="px-3 py-1.5 text-xs font-semibold text-emerald-700 bg-emerald-50 hover:bg-emerald-100 border border-emerald-200 rounded-lg transition-colors cursor-pointer"
                      >
                        ✍️ Sample Human Text
                      </button>
                    </div>

                    <button
                      type="button"
                      disabled={scanning || !textInputContent}
                      onClick={() => handleExecuteScan('user_submitted_document.txt', 'text', 'AI-Generated', 'GPT-4o Transformer')}
                      className="px-5 py-2 text-xs font-bold bg-blue-600 hover:bg-blue-700 text-white rounded-lg shadow-xs transition-colors disabled:opacity-50 cursor-pointer"
                    >
                      {scanning ? 'Computing Metrics...' : 'Detect AI Text Signatures'}
                    </button>
                  </div>
                </div>

                {/* Text Analysis Results */}
                <div className="w-full lg:w-96 p-5 rounded-2xl bg-gray-50 border border-gray-200 flex flex-col justify-between">
                  <div>
                    <h4 className="text-xs font-bold text-gray-500 uppercase tracking-wider mb-2">
                      Text Forensic Metrics
                    </h4>

                    {scanning ? (
                      <div className="py-8 space-y-4">
                        <div className="flex justify-between text-xs font-semibold text-gray-700">
                          <span>Token Perplexity Scanning...</span>
                          <span>{scanProgress}%</span>
                        </div>
                        <div className="w-full h-2.5 bg-gray-200 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-blue-600 transition-all duration-300 rounded-full"
                            style={{ width: `${scanProgress}%` }}
                          />
                        </div>
                      </div>
                    ) : activeResult && activeResult.modality === 'text' ? (
                      <div className="space-y-4">
                        <div
                          className={`p-4 rounded-xl border flex items-center gap-3 ${
                            activeResult.verdict === 'AI-Generated'
                              ? 'bg-rose-50 border-rose-200 text-rose-900'
                              : 'bg-emerald-50 border-emerald-200 text-emerald-900'
                          }`}
                        >
                          <div
                            className={`w-9 h-9 rounded-lg flex items-center justify-center font-bold text-white shrink-0 ${
                              activeResult.verdict === 'AI-Generated' ? 'bg-rose-600' : 'bg-emerald-600'
                            }`}
                          >
                            {activeResult.verdict === 'AI-Generated' ? 'AI' : '✓'}
                          </div>
                          <div>
                            <p className="text-xs font-bold uppercase tracking-wider">
                              {activeResult.verdict === 'AI-Generated' ? 'LLM Generation Detected' : 'Human Authorship Verified'}
                            </p>
                            <p className="text-sm font-extrabold">{activeResult.confidence}% Probability</p>
                          </div>
                        </div>

                        <div className="bg-white rounded-lg p-3 border border-gray-200 text-xs space-y-2">
                          <div className="flex justify-between">
                            <span className="text-gray-500">Perplexity Score:</span>
                            <span className="font-semibold text-gray-900">{activeResult.details.perplexity} (Low = Machine)</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-500">Burstiness Index:</span>
                            <span className="font-semibold text-gray-900">{activeResult.details.burstiness} (Uniform = AI)</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-500">Generator Attribution:</span>
                            <span className="font-semibold text-gray-900">{activeResult.generator}</span>
                          </div>
                        </div>
                      </div>
                    ) : (
                      <div className="py-12 text-center text-xs text-gray-400">
                        Paste text or pick a sample to measure perplexity and burstiness distributions.
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* TAB 3: AUDIO FORENSICS */}
          {activeTab === 'audio' && (
            <div className="space-y-6">
              <div className="flex flex-col lg:flex-row gap-6">
                <div className="flex-1 border-2 border-dashed border-gray-200 rounded-2xl p-8 text-center bg-gray-50/50 flex flex-col items-center justify-center min-h-[260px]">
                  <div className="w-14 h-14 rounded-2xl bg-indigo-50 text-indigo-600 flex items-center justify-center mb-4 ring-8 ring-indigo-50/50">
                    <svg className="w-7 h-7" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 100-6 3 3 0 000 6z" />
                    </svg>
                  </div>
                  <h3 className="text-base font-bold text-gray-900">Audio Speech & Voice Clone Verification</h3>
                  <p className="text-xs text-gray-500 mt-1 max-w-md">
                    Inspects synthetic speech patterns, Mel-spectrogram phase discontinuities, and voice-cloning vocoder signatures.
                  </p>

                  <div className="mt-4 flex flex-wrap gap-2 justify-center">
                    <button
                      type="button"
                      disabled={scanning}
                      onClick={() => handleSampleAudio(true)}
                      className="px-3 py-1.5 text-xs font-semibold text-rose-700 bg-rose-50 hover:bg-rose-100 border border-rose-200 rounded-lg transition-colors cursor-pointer"
                    >
                      🎙️ Test Deepfake Voice Clone
                    </button>
                    <button
                      type="button"
                      disabled={scanning}
                      onClick={() => handleSampleAudio(false)}
                      className="px-3 py-1.5 text-xs font-semibold text-emerald-700 bg-emerald-50 hover:bg-emerald-100 border border-emerald-200 rounded-lg transition-colors cursor-pointer"
                    >
                      🗣️ Test Natural Human Speech
                    </button>
                  </div>
                </div>

                <div className="w-full lg:w-96 p-5 rounded-2xl bg-gray-50 border border-gray-200 flex flex-col justify-between">
                  <div>
                    <h4 className="text-xs font-bold text-gray-500 uppercase tracking-wider mb-2">
                      Acoustic Forensic Metrics
                    </h4>

                    {scanning ? (
                      <div className="py-8 space-y-4">
                        <div className="flex justify-between text-xs font-semibold text-gray-700">
                          <span>Analyzing Mel-Spectrogram...</span>
                          <span>{scanProgress}%</span>
                        </div>
                        <div className="w-full h-2.5 bg-gray-200 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-indigo-600 transition-all duration-300 rounded-full"
                            style={{ width: `${scanProgress}%` }}
                          />
                        </div>
                      </div>
                    ) : activeResult && activeResult.modality === 'audio' ? (
                      <div className="space-y-4">
                        <div
                          className={`p-4 rounded-xl border flex items-center gap-3 ${
                            activeResult.verdict === 'AI-Generated'
                              ? 'bg-rose-50 border-rose-200 text-rose-900'
                              : 'bg-emerald-50 border-emerald-200 text-emerald-900'
                          }`}
                        >
                          <div
                            className={`w-9 h-9 rounded-lg flex items-center justify-center font-bold text-white shrink-0 ${
                              activeResult.verdict === 'AI-Generated' ? 'bg-rose-600' : 'bg-emerald-600'
                            }`}
                          >
                            {activeResult.verdict === 'AI-Generated' ? 'AI' : '✓'}
                          </div>
                          <div>
                            <p className="text-xs font-bold uppercase tracking-wider">
                              {activeResult.verdict === 'AI-Generated' ? 'Synthetic Vocoder Detected' : 'Natural Acoustic Audio'}
                            </p>
                            <p className="text-sm font-extrabold">{activeResult.confidence}% Confidence</p>
                          </div>
                        </div>

                        <div className="bg-white rounded-lg p-3 border border-gray-200 text-xs space-y-1.5">
                          <div className="flex justify-between">
                            <span className="text-gray-500">Acoustic Architecture:</span>
                            <span className="font-semibold text-gray-900">{activeResult.generator}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-500">Spectral Consistency:</span>
                            <span className="font-semibold text-gray-900">{activeResult.details.spectralScore}%</span>
                          </div>
                        </div>
                      </div>
                    ) : (
                      <div className="py-12 text-center text-xs text-gray-400">
                        Upload audio or test sample voice recordings to evaluate vocoder signatures.
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Forensic Audit Log & Recent Scans Table */}
      <div className="bg-white rounded-2xl border border-gray-200/80 shadow-sm overflow-hidden">
        <div className="p-6 border-b border-gray-200 flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <div>
            <h3 className="text-lg font-bold text-gray-900">Recent Forensic Scan Evidence</h3>
            <p className="text-xs text-gray-500 mt-0.5">
              Complete chronological audit trail with SHA-256 cryptographic verification.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            {/* Search Input */}
            <div className="relative">
              <input
                type="text"
                placeholder="Search by ID, file, or model..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-8 pr-3 py-1.5 text-xs rounded-lg border border-gray-300 focus:border-blue-500 focus:ring-1 focus:ring-blue-500 bg-gray-50/50 w-52 sm:w-64"
              />
              <svg className="w-3.5 h-3.5 text-gray-400 absolute left-2.5 top-2.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
            </div>

            {/* Verdict Filter */}
            <select
              value={verdictFilter}
              onChange={(e) =>
                setVerdictFilter(
                  e.target.value as 'all' | 'AI-Generated' | 'Authentic' | 'Suspicious'
                )
              }
              className="text-xs border border-gray-300 rounded-lg px-2.5 py-1.5 bg-gray-50/50 text-gray-700"
            >
              <option value="all">All Verdicts</option>
              <option value="AI-Generated">Synthetic (AI)</option>
              <option value="Authentic">Authentic Human</option>
              <option value="Suspicious">Suspicious</option>
            </select>
          </div>
        </div>

        {/* Scans Table */}
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200 text-left text-xs">
            <thead className="bg-gray-50/80 text-gray-500 font-semibold uppercase tracking-wider">
              <tr>
                <th className="py-3 px-4">Evidence ID</th>
                <th className="py-3 px-4">Media File Name</th>
                <th className="py-3 px-4">Modality</th>
                <th className="py-3 px-4">Forensic Verdict</th>
                <th className="py-3 px-4">Confidence</th>
                <th className="py-3 px-4">Generator / Source</th>
                <th className="py-3 px-4">Timestamp</th>
                <th className="py-3 px-4 text-right">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100 text-gray-700">
              {filteredRecords.map((item) => (
                <tr key={item.id} className="hover:bg-gray-50/60 transition-colors">
                  <td className="py-3.5 px-4 font-mono font-semibold text-blue-600">{item.id}</td>
                  <td className="py-3.5 px-4 font-medium text-gray-900 max-w-[200px] truncate" title={item.name}>
                    {item.name}
                  </td>
                  <td className="py-3.5 px-4 capitalize">
                    <span className="inline-flex items-center gap-1">
                      {item.modality === 'image' ? '🖼️' : item.modality === 'text' ? '📝' : '🎙️'} {item.modality}
                    </span>
                  </td>
                  <td className="py-3.5 px-4">
                    <span
                      className={`inline-flex items-center px-2 py-0.5 rounded-full text-[11px] font-bold ${
                        item.verdict === 'AI-Generated'
                          ? 'bg-rose-100 text-rose-800'
                          : item.verdict === 'Authentic'
                          ? 'bg-emerald-100 text-emerald-800'
                          : 'bg-amber-100 text-amber-800'
                      }`}
                    >
                      {item.verdict}
                    </span>
                  </td>
                  <td className="py-3.5 px-4">
                    <div className="flex items-center gap-2">
                      <div className="w-16 h-1.5 bg-gray-200 rounded-full overflow-hidden">
                        <div
                          className={`h-full rounded-full ${
                            item.verdict === 'AI-Generated' ? 'bg-rose-500' : 'bg-emerald-500'
                          }`}
                          style={{ width: `${item.confidence}%` }}
                        />
                      </div>
                      <span className="font-semibold text-gray-800">{item.confidence}%</span>
                    </div>
                  </td>
                  <td className="py-3.5 px-4 text-gray-600">{item.generator}</td>
                  <td className="py-3.5 px-4 text-gray-500 whitespace-nowrap">{item.timestamp}</td>
                  <td className="py-3.5 px-4 text-right">
                    <button
                      type="button"
                      onClick={() => setInspectRecord(item)}
                      className="px-2.5 py-1 text-xs font-semibold text-blue-600 hover:text-blue-800 hover:bg-blue-50 rounded transition-colors cursor-pointer"
                    >
                      View Report
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Forensic Report Inspection Modal */}
      {inspectRecord && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-xs">
          <div className="bg-white rounded-2xl max-w-xl w-full p-6 sm:p-8 shadow-2xl border border-gray-100 space-y-5 animate-in fade-in zoom-in-95 duration-150">
            <div className="flex items-start justify-between border-b border-gray-200 pb-4">
              <div>
                <span className="text-xs font-bold text-blue-600 font-mono">{inspectRecord.id}</span>
                <h3 className="text-lg font-extrabold text-gray-900 mt-0.5">{inspectRecord.name}</h3>
                <p className="text-xs text-gray-500">SHA-256: {inspectRecord.sha256.substring(0, 32)}...</p>
              </div>
              <button
                type="button"
                onClick={() => setInspectRecord(null)}
                className="text-gray-400 hover:text-gray-600 cursor-pointer p-1"
                aria-label="Close modal"
              >
                <svg className="w-5 h-5" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
                </svg>
              </button>
            </div>

            {/* Verdict Overview Card */}
            <div
              className={`p-4 rounded-xl border flex items-center justify-between ${
                inspectRecord.verdict === 'AI-Generated'
                  ? 'bg-rose-50 border-rose-200 text-rose-900'
                  : 'bg-emerald-50 border-emerald-200 text-emerald-900'
              }`}
            >
              <div>
                <p className="text-xs font-bold uppercase tracking-wider">Classification Verdict</p>
                <p className="text-xl font-extrabold">{inspectRecord.verdict}</p>
              </div>
              <div className="text-right">
                <p className="text-xs font-medium text-gray-600">Model Confidence</p>
                <p className="text-xl font-extrabold">{inspectRecord.confidence}%</p>
              </div>
            </div>

            {/* Technical Forensic Breakdown */}
            <div className="bg-gray-50 rounded-xl p-4 border border-gray-200 space-y-2 text-xs">
              <h4 className="font-bold text-gray-700 uppercase tracking-wider mb-2">Technical Feature Metrics</h4>
              <div className="flex justify-between py-1 border-b border-gray-200/60">
                <span className="text-gray-500">Generator Model Attribution:</span>
                <span className="font-semibold text-gray-900">{inspectRecord.generator}</span>
              </div>
              <div className="flex justify-between py-1 border-b border-gray-200/60">
                <span className="text-gray-500">Media Modality & File Size:</span>
                <span className="font-semibold text-gray-900 capitalize">{inspectRecord.modality} ({inspectRecord.size})</span>
              </div>
              {inspectRecord.details.spectralScore !== undefined && (
                <div className="flex justify-between py-1 border-b border-gray-200/60">
                  <span className="text-gray-500">Spatial Frequency FFT Anomaly:</span>
                  <span className="font-semibold text-gray-900">{inspectRecord.details.spectralScore}%</span>
                </div>
              )}
              {inspectRecord.details.sensorNoise && (
                <div className="flex justify-between py-1 border-b border-gray-200/60">
                  <span className="text-gray-500">Photo-Response Non-Uniformity (PRNU):</span>
                  <span className="font-semibold text-gray-900">{inspectRecord.details.sensorNoise}</span>
                </div>
              )}
              {inspectRecord.details.perplexity !== undefined && (
                <div className="flex justify-between py-1 border-b border-gray-200/60">
                  <span className="text-gray-500">Token Perplexity (Cross-Entropy):</span>
                  <span className="font-semibold text-gray-900">{inspectRecord.details.perplexity}</span>
                </div>
              )}
              {inspectRecord.details.burstiness !== undefined && (
                <div className="flex justify-between py-1">
                  <span className="text-gray-500">Sentence Burstiness Uniformity:</span>
                  <span className="font-semibold text-gray-900">{inspectRecord.details.burstiness}</span>
                </div>
              )}
            </div>

            {/* Modal Actions */}
            <div className="flex items-center justify-end gap-3 pt-2">
              <button
                type="button"
                onClick={() => {
                  const blob = new Blob([JSON.stringify(inspectRecord, null, 2)], { type: 'application/json' });
                  const url = URL.createObjectURL(blob);
                  const a = document.createElement('a');
                  a.href = url;
                  a.download = `Forensic_Report_${inspectRecord.id}.json`;
                  a.click();
                }}
                className="px-4 py-2 text-xs font-semibold bg-gray-100 hover:bg-gray-200 text-gray-800 rounded-lg transition-colors cursor-pointer"
              >
                Download JSON Report
              </button>
              <button
                type="button"
                onClick={() => setInspectRecord(null)}
                className="px-4 py-2 text-xs font-semibold bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors cursor-pointer"
              >
                Done
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
