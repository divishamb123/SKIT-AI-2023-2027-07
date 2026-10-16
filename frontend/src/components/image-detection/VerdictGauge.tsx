'use client';

import React from 'react';

interface VerdictGaugeProps {
  verdict: 'REAL' | 'AI_GENERATED';
  confidence: number; // 0.0 to 1.0 or 0 to 100
  probabilities?: {
    real: number;
    ai_generated: number;
  };
  latencyMs?: number;
  device?: string;
}

export function VerdictGauge({
  verdict,
  confidence,
  probabilities,
  latencyMs,
  device,
}: VerdictGaugeProps) {
  // Normalize confidence to 0-100 percentage
  const confPercent = confidence > 1 ? confidence : confidence * 100;
  const isAI = verdict === 'AI_GENERATED';

  // Probabilities in percentage
  const realPct = probabilities
    ? (probabilities.real > 1 ? probabilities.real : probabilities.real * 100).toFixed(1)
    : (!isAI ? confPercent : 100 - confPercent).toFixed(1);
  const aiPct = probabilities
    ? (probabilities.ai_generated > 1
        ? probabilities.ai_generated
        : probabilities.ai_generated * 100
      ).toFixed(1)
    : (isAI ? confPercent : 100 - confPercent).toFixed(1);

  // SVG Gauge Math (radius 70, circumference 2 * pi * 70 = ~439.82)
  const radius = 70;
  const circumference = 2 * Math.PI * radius;
  // Use a 270-degree arc gauge (0.75 of circumference)
  const arcLength = circumference * 0.75;
  const strokeOffset = arcLength - (confPercent / 100) * arcLength;

  return (
    <div className="bg-white rounded-2xl border border-gray-200/90 shadow-sm p-6 flex flex-col items-center text-center transition-all">
      {/* Top Tag */}
      <div className="flex items-center justify-between w-full mb-4">
        <span className="text-[11px] font-semibold text-gray-500 uppercase tracking-wider">
          Forensic Verdict Gauge
        </span>
        {device && (
          <span className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-bold bg-gray-100 text-gray-700 uppercase border border-gray-200">
            Hardware: {device}
          </span>
        )}
      </div>

      {/* Radial Gauge SVG */}
      <div className="relative w-48 h-48 flex items-center justify-center">
        <svg
          className="w-48 h-48 transform -rotate-135"
          viewBox="0 0 160 160"
        >
          {/* Background Track */}
          <circle
            cx="80"
            cy="80"
            r={radius}
            fill="transparent"
            stroke="#f1f5f9"
            strokeWidth="14"
            strokeDasharray={`${arcLength} ${circumference}`}
            strokeLinecap="round"
          />

          {/* Value Progress Arc */}
          <circle
            cx="80"
            cy="80"
            r={radius}
            fill="transparent"
            stroke={isAI ? '#f43f5e' : '#10b981'}
            strokeWidth="14"
            strokeDasharray={`${arcLength} ${circumference}`}
            strokeDashoffset={strokeOffset}
            strokeLinecap="round"
            className="transition-all duration-700 ease-out"
          />
        </svg>

        {/* Center Text */}
        <div className="absolute inset-0 flex flex-col items-center justify-center pt-2">
          <span
            className={`text-3xl font-extrabold tracking-tight ${
              isAI ? 'text-rose-600' : 'text-emerald-600'
            }`}
          >
            {confPercent.toFixed(1)}%
          </span>
          <span className="text-[11px] font-bold uppercase tracking-wider text-gray-600 mt-0.5">
            Confidence
          </span>
          {latencyMs !== undefined && (
            <span className="text-[10px] text-gray-600 mt-1">
              {latencyMs.toFixed(1)} ms
            </span>
          )}
        </div>
      </div>

      {/* Main Verdict Badge */}
      <div
        className={`w-full mt-3 py-2.5 px-4 rounded-xl font-bold text-sm tracking-wide flex items-center justify-center gap-2 border ${
          isAI
            ? 'bg-rose-50 text-rose-800 border-rose-200'
            : 'bg-emerald-50 text-emerald-800 border-emerald-200'
        }`}
      >
        <span className="text-base">{isAI ? '⚠️' : '🛡️'}</span>
        <span>{isAI ? 'AI-GENERATED MEDIA' : 'AUTHENTIC PHOTOGRAPH'}</span>
      </div>

      {/* Probability Dual Breakdown Bar */}
      <div className="w-full mt-6 space-y-2">
        <div className="flex justify-between text-xs font-semibold text-gray-700">
          <span className="text-emerald-700">Real: {realPct}%</span>
          <span className="text-rose-700">AI: {aiPct}%</span>
        </div>
        <div className="w-full h-3 bg-gray-100 rounded-full overflow-hidden flex shadow-inner">
          <div
            style={{ width: `${realPct}%` }}
            className="bg-emerald-500 h-full transition-all duration-500"
            title={`Real Probability: ${realPct}%`}
          />
          <div
            style={{ width: `${aiPct}%` }}
            className="bg-rose-500 h-full transition-all duration-500"
            title={`AI-Generated Probability: ${aiPct}%`}
          />
        </div>
        <div className="flex justify-between text-[10px] text-gray-600 pt-0.5">
          <span>Natural Sensor Distribution</span>
          <span>Synthetic Latent Traces</span>
        </div>
      </div>
    </div>
  );
}
