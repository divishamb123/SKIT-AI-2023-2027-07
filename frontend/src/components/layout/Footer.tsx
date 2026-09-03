import React from 'react';

export function Footer() {
  return (
    <footer className="bg-white border-t border-gray-200 mt-auto">
      <div className="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
        <p className="text-center text-sm text-gray-500">
          &copy; {new Date().getFullYear()} AI Content Forensics System. Sprint 1 - Aryansh Agarwal (23ESKCA021).
        </p>
      </div>
    </footer>
  );
}
