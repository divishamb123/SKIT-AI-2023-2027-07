import React from 'react';

export function Footer() {
  return (
    <footer className="bg-white border-t border-gray-200 mt-auto">
      <div className="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-4">
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
          <span className="text-xs text-gray-500 font-medium">
            Forensics Inference Engine Active • Neural Multi-Model Stack
          </span>
        </div>
        <p className="text-center text-xs text-gray-500">
          &copy; {new Date().getFullYear()} AIForensics Platform. Enterprise AI-Generated Content Detection & Media Verification.
        </p>
      </div>
    </footer>
  );
}
