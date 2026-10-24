'use client';

import React, { useState, useMemo } from 'react';
import {
  BatchImageDetectionResponse,
  BatchImageItem,
  BatchImageErrorItem,
} from '@/types/image-detection';

interface BatchResultsGalleryProps {
  data: BatchImageDetectionResponse;
  onReset?: () => void;
  onInspectItem?: (item: BatchImageItem) => void;
}

export function BatchResultsGallery({
  data,
  onReset,
  onInspectItem,
}: BatchResultsGalleryProps) {
  const [filter, setFilter] = useState<'all' | 'ai' | 'real' | 'errors'>('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [sortBy, setSortBy] = useState<'index' | 'conf_desc' | 'conf_asc' | 'latency'>('index');
  const [batchThreshold, setBatchThreshold] = useState<number>(0.5);
  const [selectedItem, setSelectedItem] = useState<BatchImageItem | null>(null);

  // Compute counts under active batchThreshold
  const aiCount = useMemo(
    () =>
      data.results.filter(
        (r) => r.probabilities.ai_generated >= batchThreshold
      ).length,
    [data.results, batchThreshold]
  );
  const realCount = useMemo(
    () =>
      data.results.filter(
        (r) => r.probabilities.ai_generated < batchThreshold
      ).length,
    [data.results, batchThreshold]
  );

  const filteredResults = useMemo(() => {
    let res = data.results.filter((item) => {
      const isAI = item.probabilities.ai_generated >= batchThreshold;
      const matchesSearch = item.filename
        .toLowerCase()
        .includes(searchQuery.toLowerCase());
      if (!matchesSearch) return false;

      if (filter === 'ai') return isAI;
      if (filter === 'real') return !isAI;
      if (filter === 'errors') return false;
      return true;
    });

    if (sortBy === 'conf_desc') {
      res = [...res].sort((a, b) => b.confidence - a.confidence);
    } else if (sortBy === 'conf_asc') {
      res = [...res].sort((a, b) => a.confidence - b.confidence);
    } else if (sortBy === 'latency') {
      res = [...res].sort((a, b) => b.latency_ms - a.latency_ms);
    } else {
      res = [...res].sort((a, b) => a.index - b.index);
    }

    return res;
  }, [data.results, filter, searchQuery, sortBy, batchThreshold]);

  const handleExportJson = () => {
    const dataStr =
      'data:text/json;charset=utf-8,' +
      encodeURIComponent(JSON.stringify(data, null, 2));
    const dlAnchor = document.createElement('a');
    dlAnchor.setAttribute('href', dataStr);
    dlAnchor.setAttribute(
      'download',
      `image_batch_forensic_report_${data.request_id || Date.now()}.json`
    );
    dlAnchor.click();
  };

  const handleExportCsv = () => {
    const headers = [
      'Index',
      'Filename',
      'Verdict',
      'Confidence',
      'Prob_Real',
      'Prob_AI',
      'Width',
      'Height',
      'Format',
      'Latency_ms',
    ];
    const rows = data.results.map((r) => [
      r.index + 1,
      `"${r.filename}"`,
      r.probabilities.ai_generated >= batchThreshold ? 'AI_GENERATED' : 'REAL',
      r.confidence.toFixed(4),
      r.probabilities.real.toFixed(4),
      r.probabilities.ai_generated.toFixed(4),
      r.image_metadata.width,
      r.image_metadata.height,
      r.image_metadata.format,
      r.latency_ms.toFixed(2),
    ]);

    const csvContent =
      'data:text/csv;charset=utf-8,' +
      [headers.join(','), ...rows.map((e) => e.join(','))].join('\n');
    const dlAnchor = document.createElement('a');
    dlAnchor.setAttribute('href', encodeURI(csvContent));
    dlAnchor.setAttribute(
      'download',
      `batch_forensic_manifest_${Date.now()}.csv`
    );
    dlAnchor.click();
  };

  return (
    <div className="bg-white rounded-2xl border border-gray-200/90 shadow-sm p-6 space-y-6">
      {/* Top Header & Export */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 border-b border-gray-100 pb-4">
        <div>
          <div className="flex items-center gap-2">
            <h3 className="text-base font-bold text-gray-900">
              Batch Inference Results Gallery & Evidence Inspector
            </h3>
            <span className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-bold bg-blue-50 text-blue-700 uppercase border border-blue-200">
              {data.device}
            </span>
          </div>
          <p className="text-xs text-gray-500 mt-0.5">
            Analyzed {data.batch_size} images using vectorized batched tensor collation.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            onClick={handleExportCsv}
            className="px-3 py-1.5 text-xs font-semibold text-gray-700 bg-gray-50 hover:bg-gray-100 border border-gray-200 rounded-lg transition-colors cursor-pointer flex items-center gap-1.5 shadow-2xs"
          >
            <span>📊</span> Export CSV
          </button>
          <button
            type="button"
            onClick={handleExportJson}
            className="px-3 py-1.5 text-xs font-semibold text-gray-700 bg-gray-50 hover:bg-gray-100 border border-gray-200 rounded-lg transition-colors cursor-pointer flex items-center gap-1.5 shadow-2xs"
          >
            <span>📥</span> Export JSON
          </button>
          {onReset && (
            <button
              type="button"
              onClick={onReset}
              className="px-3 py-1.5 text-xs font-semibold text-red-600 hover:bg-red-50 border border-red-200 rounded-lg transition-colors cursor-pointer"
            >
              Reset Batch
            </button>
          )}
        </div>
      </div>

      {/* Summary KPI Cards */}
      <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-5 gap-3">
        <div className="bg-gray-50 rounded-xl p-3 border border-gray-100">
          <span className="text-[10px] font-bold uppercase text-gray-400">Total Images</span>
          <p className="text-xl font-extrabold text-gray-900 mt-1">{data.batch_size}</p>
          <span className="text-[10px] text-gray-500">Collation Size</span>
        </div>

        <div className="bg-rose-50/60 rounded-xl p-3 border border-rose-100">
          <span className="text-[10px] font-bold uppercase text-rose-500">AI Flagged</span>
          <p className="text-xl font-extrabold text-rose-700 mt-1">{aiCount}</p>
          <span className="text-[10px] text-rose-600">
            {((aiCount / (data.successful_count || 1)) * 100).toFixed(0)}% at $\tau={batchThreshold}$
          </span>
        </div>

        <div className="bg-emerald-50/60 rounded-xl p-3 border border-emerald-100">
          <span className="text-[10px] font-bold uppercase text-emerald-500">Authentic Real</span>
          <p className="text-xl font-extrabold text-emerald-700 mt-1">{realCount}</p>
          <span className="text-[10px] text-emerald-600">
            {((realCount / (data.successful_count || 1)) * 100).toFixed(0)}% at $\tau={batchThreshold}$
          </span>
        </div>

        <div className="bg-blue-50/60 rounded-xl p-3 border border-blue-100">
          <span className="text-[10px] font-bold uppercase text-blue-500">Throughput</span>
          <p className="text-xl font-extrabold text-blue-700 mt-1">
            {data.throughput_fps.toFixed(1)} <span className="text-xs font-semibold">FPS</span>
          </p>
          <span className="text-[10px] text-blue-600">
            {data.avg_latency_ms.toFixed(1)} ms / img
          </span>
        </div>

        <div className="bg-gray-50 rounded-xl p-3 border border-gray-100 col-span-2 sm:col-span-4 lg:col-span-1">
          <span className="text-[10px] font-bold uppercase text-gray-400">Batch Latency</span>
          <p className="text-xl font-extrabold text-gray-900 mt-1">
            {data.total_latency_ms.toFixed(0)} <span className="text-xs font-semibold">ms</span>
          </p>
          <span className="text-[10px] text-gray-500">
            {data.failed_count > 0 ? `${data.failed_count} errors isolated` : 'Zero errors'}
          </span>
        </div>
      </div>

      {/* Interactive Controls Bar: Threshold Slider, Search, Sort */}
      <div className="p-4 bg-gray-50/80 rounded-xl border border-gray-200 flex flex-col md:flex-row md:items-center md:justify-between gap-4 text-xs">
        {/* Search Input */}
        <div className="flex-1">
          <input
            type="text"
            placeholder="Search evidence by filename..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full px-3 py-1.5 rounded-lg border border-gray-300 bg-white placeholder-gray-400 focus:border-blue-500"
          />
        </div>

        {/* Dynamic Threshold Slider */}
        <div className="flex items-center gap-2 shrink-0">
          <span className="text-gray-600 font-medium whitespace-nowrap">
            Threshold $\tau$: <strong className="font-mono text-blue-600">{batchThreshold.toFixed(2)}</strong>
          </span>
          <input
            type="range"
            min="0.10"
            max="0.90"
            step="0.05"
            value={batchThreshold}
            onChange={(e) => setBatchThreshold(parseFloat(e.target.value))}
            className="w-24 h-1.5 bg-gray-300 rounded appearance-none cursor-pointer accent-blue-600"
          />
        </div>

        {/* Sort Select */}
        <div className="flex items-center gap-2 shrink-0">
          <span className="text-gray-600 font-medium">Sort:</span>
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value as any)}
            className="px-2.5 py-1.5 bg-white border border-gray-300 rounded-lg text-xs"
          >
            <option value="index">Input Order (#)</option>
            <option value="conf_desc">Confidence (High → Low)</option>
            <option value="conf_asc">Confidence (Low → High)</option>
            <option value="latency">Latency (Fastest)</option>
          </select>
        </div>
      </div>

      {/* Filter Tabs */}
      <div className="flex items-center gap-2 border-b border-gray-100 pb-2 overflow-x-auto text-xs">
        <button
          type="button"
          onClick={() => setFilter('all')}
          className={`px-3 py-1.5 font-bold rounded-lg transition-colors cursor-pointer ${
            filter === 'all'
              ? 'bg-blue-600 text-white shadow-2xs'
              : 'text-gray-600 hover:bg-gray-100'
          }`}
        >
          All Items ({data.batch_size})
        </button>
        <button
          type="button"
          onClick={() => setFilter('ai')}
          className={`px-3 py-1.5 font-bold rounded-lg transition-colors cursor-pointer ${
            filter === 'ai'
              ? 'bg-rose-600 text-white shadow-2xs'
              : 'text-rose-700 hover:bg-rose-50'
          }`}
        >
          AI-Generated ({aiCount})
        </button>
        <button
          type="button"
          onClick={() => setFilter('real')}
          className={`px-3 py-1.5 font-bold rounded-lg transition-colors cursor-pointer ${
            filter === 'real'
              ? 'bg-emerald-600 text-white shadow-2xs'
              : 'text-emerald-700 hover:bg-emerald-50'
          }`}
        >
          Authentic Real ({realCount})
        </button>
        {data.failed_count > 0 && (
          <button
            type="button"
            onClick={() => setFilter('errors')}
            className={`px-3 py-1.5 font-bold rounded-lg transition-colors cursor-pointer ${
              filter === 'errors'
                ? 'bg-amber-600 text-white shadow-2xs'
                : 'text-amber-700 hover:bg-amber-50'
            }`}
          >
            Errors ({data.failed_count})
          </button>
        )}
      </div>

      {/* Gallery Grid */}
      {filter !== 'errors' ? (
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
          {filteredResults.map((item) => {
            const isAI = item.probabilities.ai_generated >= batchThreshold;
            const confPct = item.confidence > 1 ? item.confidence : item.confidence * 100;

            return (
              <div
                key={item.index}
                onClick={() => {
                  setSelectedItem(item);
                  if (onInspectItem) onInspectItem(item);
                }}
                className="group border border-gray-200 rounded-xl p-3 hover:border-blue-400 hover:shadow-sm transition-all cursor-pointer bg-white flex flex-col justify-between"
              >
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-[10px] font-mono text-gray-600 bg-gray-100 px-1.5 py-0.5 rounded">
                      #{item.index + 1}
                    </span>
                    <span
                      className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${
                        isAI
                          ? 'bg-rose-100 text-rose-800'
                          : 'bg-emerald-100 text-emerald-800'
                      }`}
                    >
                      {isAI ? 'AI' : 'REAL'}
                    </span>
                  </div>

                  <p className="text-xs font-bold text-gray-900 truncate" title={item.filename}>
                    {item.filename}
                  </p>
                  <p className="text-[10px] text-gray-600 mt-0.5">
                    {item.image_metadata.width}×{item.image_metadata.height} px • {item.image_metadata.format}
                  </p>
                </div>

                <div className="mt-3 pt-2 border-t border-gray-100 space-y-1">
                  <div className="flex justify-between text-[11px] font-semibold">
                    <span className="text-gray-600">Confidence</span>
                    <span className={isAI ? 'text-rose-600 font-bold' : 'text-emerald-600 font-bold'}>
                      {confPct.toFixed(1)}%
                    </span>
                  </div>
                  <div className="w-full h-1.5 bg-gray-100 rounded-full overflow-hidden">
                    <div
                      style={{ width: `${confPct}%` }}
                      className={`h-full ${isAI ? 'bg-rose-500' : 'bg-emerald-500'}`}
                    />
                  </div>
                  <div className="flex justify-between text-[10px] text-gray-600 pt-0.5">
                    <span>Latency</span>
                    <span>{item.latency_ms.toFixed(1)} ms</span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      ) : null}

      {/* Errors Section */}
      {(filter === 'errors' || (filter === 'all' && data.errors.length > 0)) && (
        <div className="space-y-2">
          {filter === 'all' && (
            <h4 className="text-xs font-bold text-amber-800 uppercase tracking-wider pt-2">
              Isolated Batch Errors ({data.errors.length})
            </h4>
          )}
          <div className="space-y-2">
            {data.errors.map((err: BatchImageErrorItem) => (
              <div
                key={err.index}
                className="p-3 rounded-xl bg-amber-50 border border-amber-200 text-xs flex flex-col sm:flex-row sm:items-center sm:justify-between gap-1"
              >
                <div className="flex items-center gap-2">
                  <span className="font-mono bg-amber-200 text-amber-900 px-1.5 py-0.5 rounded text-[10px]">
                    #{err.index + 1}
                  </span>
                  <span className="font-bold text-gray-900">{err.filename}</span>
                  <span className="text-amber-800 font-mono text-[10px]">
                    [{err.error_code}]
                  </span>
                </div>
                <span className="text-amber-900 text-xs">{err.error_message}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Inspect Item Modal */}
      {selectedItem && (
        <div
          role="dialog"
          aria-modal="true"
          className="fixed inset-0 z-50 bg-black/50 backdrop-blur-xs flex items-center justify-center p-4"
          onClick={() => setSelectedItem(null)}
        >
          <div
            className="bg-white rounded-2xl max-w-md w-full p-6 space-y-4 shadow-xl"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between border-b border-gray-100 pb-3">
              <div>
                <span className="text-[10px] font-mono text-gray-600 bg-gray-100 px-1.5 py-0.5 rounded">
                  Item #{selectedItem.index + 1}
                </span>
                <h4 className="text-sm font-bold text-gray-900 mt-1 truncate">
                  {selectedItem.filename}
                </h4>
              </div>
              <button
                type="button"
                onClick={() => setSelectedItem(null)}
                className="text-gray-600 hover:text-gray-900 text-lg font-bold p-1 cursor-pointer"
              >
                ✕
              </button>
            </div>

            {/* Verdict Chip */}
            <div
              className={`p-3 rounded-xl border text-center font-bold text-sm ${
                selectedItem.probabilities.ai_generated >= batchThreshold
                  ? 'bg-rose-50 text-rose-800 border-rose-200'
                  : 'bg-emerald-50 text-emerald-800 border-emerald-200'
              }`}
            >
              {selectedItem.probabilities.ai_generated >= batchThreshold
                ? '⚠️ AI-GENERATED MEDIA'
                : '🛡️ AUTHENTIC REAL PHOTOGRAPH'}
              <p className="text-xs font-normal mt-0.5">
                Confidence: {(selectedItem.confidence * 100).toFixed(1)}% (Threshold: {batchThreshold.toFixed(2)})
              </p>
            </div>

            {/* Probability Breakdown */}
            <div className="bg-gray-50 rounded-xl p-3 text-xs space-y-2 border border-gray-100">
              <div className="flex justify-between font-semibold">
                <span className="text-emerald-700">
                  Real: {(selectedItem.probabilities.real * 100).toFixed(2)}%
                </span>
                <span className="text-rose-700">
                  AI: {(selectedItem.probabilities.ai_generated * 100).toFixed(2)}%
                </span>
              </div>
              <div className="w-full h-2 bg-gray-200 rounded-full overflow-hidden flex">
                <div
                  style={{ width: `${selectedItem.probabilities.real * 100}%` }}
                  className="bg-emerald-500 h-full"
                />
                <div
                  style={{ width: `${selectedItem.probabilities.ai_generated * 100}%` }}
                  className="bg-rose-500 h-full"
                />
              </div>
            </div>

            {/* Technical Metadata */}
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div className="bg-gray-50 p-2.5 rounded-lg border border-gray-100">
                <span className="text-gray-600 text-[10px]">Dimensions</span>
                <p className="font-bold text-gray-900 mt-0.5">
                  {selectedItem.image_metadata.width} × {selectedItem.image_metadata.height}
                </p>
              </div>
              <div className="bg-gray-50 p-2.5 rounded-lg border border-gray-100">
                <span className="text-gray-600 text-[10px]">Inference Latency</span>
                <p className="font-bold text-blue-600 mt-0.5">
                  {selectedItem.latency_ms.toFixed(1)} ms
                </p>
              </div>
            </div>

            <button
              type="button"
              onClick={() => setSelectedItem(null)}
              className="w-full py-2 bg-gray-100 hover:bg-gray-200 text-gray-800 rounded-xl text-xs font-bold transition-colors cursor-pointer"
            >
              Close Forensic Inspection
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
