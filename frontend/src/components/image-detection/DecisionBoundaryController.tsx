'use client';

/**
 * DecisionBoundaryController.tsx — Dynamic Threshold Calibration & Margin Simulator
 *
 * Student: Divisha Manak Bohra (23ESKCA038)
 * Sprint: Sprint 2 — Baseline Detection Service
 * Task 4: Designing image upload and result interaction components (Week 6)
 */

import React, { useMemo } from 'react';
import {
  ImageProbabilities,
  ThresholdSimulationResult,
  ConfidenceTier,
} from '@/types/image-detection';

interface DecisionBoundaryControllerProps {
  probabilities: ImageProbabilities;
  currentThreshold: number;
  onThresholdChange: (threshold: number) => void;
  onResetThreshold?: () => void;
}

export function computeThresholdSimulation(
  pAI: number,
  threshold: number
): ThresholdSimulationResult {
  const adjustedVerdict: 'REAL' | 'AI_GENERATED' =
    pAI >= threshold ? 'AI_GENERATED' : 'REAL';
  const margin = Math.abs(pAI - threshold);

  let confidenceTier: ConfidenceTier = 'AMBIGUOUS';
  let confidenceLabel = 'Borderline / Inconclusive Boundary';
  let riskLevel: 'critical' | 'high' | 'moderate' | 'low' | 'minimal' =
    'moderate';

  if (pAI >= 0.85) {
    confidenceTier = 'HIGH_AI';
    confidenceLabel = 'High-Confidence Synthetic Generation';
    riskLevel = 'critical';
  } else if (pAI >= threshold && margin >= 0.05) {
    confidenceTier = 'SUSPICIOUS_AI';
    confidenceLabel = 'Synthetic Diffusion Traces Detected';
    riskLevel = 'high';
  } else if (pAI <= 0.15) {
    confidenceTier = 'HIGH_REAL';
    confidenceLabel = 'High-Confidence Natural Optical Sensor';
    riskLevel = 'minimal';
  } else if (pAI < threshold && margin >= 0.05) {
    confidenceTier = 'PLAUSIBLE_REAL';
    confidenceLabel = 'Plausible Natural Photographic Capture';
    riskLevel = 'low';
  } else {
    confidenceTier = 'AMBIGUOUS';
    confidenceLabel = 'Borderline Ambiguous Evidence Zone';
    riskLevel = 'moderate';
  }

  return {
    threshold,
    adjustedVerdict,
    margin,
    confidenceTier,
    confidenceLabel,
    riskLevel,
  };
}

export function DecisionBoundaryController({
  probabilities,
  currentThreshold,
  onThresholdChange,
  onResetThreshold,
}: DecisionBoundaryControllerProps) {
  // Normalize AI probability to 0.0 - 1.0 range
  const pAI =
    probabilities.ai_generated > 1
      ? probabilities.ai_generated / 100
      : probabilities.ai_generated;

  const simulation = useMemo(
    () => computeThresholdSimulation(pAI, currentThreshold),
    [pAI, currentThreshold]
  );

  const presets = [
    { label: 'High Sensitivity (0.35)', value: 0.35, desc: 'Prioritize catching all AI' },
    { label: 'Balanced Baseline (0.50)', value: 0.50, desc: 'Equal error cost benchmark' },
    { label: 'High Specificity (0.75)', value: 0.75, desc: 'Prevent false alarms on real' },
  ];

  const isAI = simulation.adjustedVerdict === 'AI_GENERATED';

  return (
    <div className="bg-white rounded-2xl border border-gray-200/90 shadow-sm p-6 space-y-5">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 border-b border-gray-100 pb-3">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-base">⚖️</span>
            <h3 className="text-sm font-bold text-gray-900 uppercase tracking-wider">
              Decision Boundary & Threshold Simulator
            </h3>
          </div>
          <p className="text-xs text-gray-500 mt-0.5">
            Calibrate classification sensitivity in real-time and observe forensic margin shifts.
          </p>
        </div>

        {onResetThreshold && currentThreshold !== 0.5 && (
          <button
            type="button"
            onClick={onResetThreshold}
            className="text-xs text-blue-600 hover:text-blue-700 font-semibold cursor-pointer text-left"
          >
            Reset to Default (0.50)
          </button>
        )}
      </div>

      {/* Dynamic Threshold Slider */}
      <div className="space-y-2">
        <div className="flex justify-between items-center text-xs">
          <span className="font-semibold text-gray-700">
            Active Decision Threshold ($\tau$):{' '}
            <strong className="text-blue-600 font-mono text-sm">
              {currentThreshold.toFixed(2)}
            </strong>
          </span>
          <span className="text-gray-500 text-[11px]">
            Model $P(AI)$: <strong className="font-mono">{(pAI * 100).toFixed(1)}%</strong>
          </span>
        </div>

        <input
          type="range"
          min="0.10"
          max="0.90"
          step="0.01"
          value={currentThreshold}
          onChange={(e) => onThresholdChange(parseFloat(e.target.value))}
          className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
        />

        <div className="flex justify-between text-[10px] text-gray-600 px-0.5">
          <span>0.10 (Aggressive Detection)</span>
          <span>0.50 (Standard)</span>
          <span>0.90 (High Confidence Only)</span>
        </div>
      </div>

      {/* Preset Buttons */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 pt-1">
        {presets.map((preset) => {
          const isActive = Math.abs(currentThreshold - preset.value) < 0.005;
          return (
            <button
              key={preset.value}
              type="button"
              onClick={() => onThresholdChange(preset.value)}
              className={`p-2.5 rounded-xl border text-left transition-all cursor-pointer ${
                isActive
                  ? 'border-blue-500 bg-blue-50/70 shadow-2xs'
                  : 'border-gray-200 bg-gray-50/50 hover:bg-gray-100 hover:border-gray-300'
              }`}
            >
              <p
                className={`text-xs font-bold ${
                  isActive ? 'text-blue-700' : 'text-gray-800'
                }`}
              >
                {preset.label}
              </p>
              <p className="text-[10px] text-gray-500 mt-0.5">{preset.desc}</p>
            </button>
          );
        })}
      </div>

      {/* Real-Time Simulated Outcome Strip */}
      <div
        className={`p-4 rounded-xl border flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 ${
          isAI
            ? 'bg-rose-50/80 border-rose-200 text-rose-900'
            : 'bg-emerald-50/80 border-emerald-200 text-emerald-900'
        }`}
      >
        <div className="flex items-center gap-3">
          <div
            className={`w-9 h-9 rounded-lg flex items-center justify-center font-bold text-white shrink-0 ${
              isAI ? 'bg-rose-600' : 'bg-emerald-600'
            }`}
          >
            {isAI ? 'AI' : '✓'}
          </div>
          <div>
            <p className="text-xs font-bold uppercase tracking-wider">
              Calibrated Verdict: {isAI ? 'AI-GENERATED' : 'AUTHENTIC REAL'}
            </p>
            <p className="text-[11px] font-medium opacity-90">
              {simulation.confidenceLabel}
            </p>
          </div>
        </div>

        <div className="text-right sm:text-right text-xs">
          <span className="text-[10px] font-semibold uppercase opacity-80 block">
            Boundary Margin ($\Delta$)
          </span>
          <span className="font-mono font-bold text-sm">
            {(simulation.margin * 100).toFixed(1)}%
          </span>
          <span className="text-[10px] block opacity-80">
            {pAI >= currentThreshold ? 'Above threshold' : 'Below threshold'}
          </span>
        </div>
      </div>
    </div>
  );
}
