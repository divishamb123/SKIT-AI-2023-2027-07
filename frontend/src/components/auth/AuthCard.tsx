import React from 'react';

interface AuthCardProps {
  title: string;
  subtitle: string;
  badge?: string;
  children: React.ReactNode;
  footer?: React.ReactNode;
  className?: string;
}

export function AuthCard({
  title,
  subtitle,
  badge = 'SKIT AI Forensics Lab',
  children,
  footer,
  className = '',
}: AuthCardProps) {
  return (
    <div className={`w-full max-w-md mx-auto ${className}`}>
      <div className="bg-white rounded-2xl shadow-xl border border-gray-100/80 overflow-hidden transition-all duration-300">
        {/* Header decoration bar */}
        <div className="h-1.5 bg-gradient-to-r from-blue-600 via-indigo-600 to-cyan-500" />

        <div className="p-8 sm:p-10">
          {/* Brand header */}
          <div className="flex flex-col items-center text-center mb-8">
            <div className="inline-flex items-center justify-center w-12 h-12 rounded-xl bg-blue-50 text-blue-600 mb-3 shadow-inner ring-1 ring-blue-100">
              <svg
                className="w-6 h-6"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
                <path d="m9 12 2 2 4-4" />
              </svg>
            </div>

            {badge && (
              <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold bg-blue-50 text-blue-700 mb-2 border border-blue-100/80">
                {badge}
              </span>
            )}

            <h1 className="text-2xl font-bold tracking-tight text-gray-900">
              {title}
            </h1>
            <p className="mt-1.5 text-sm text-gray-500 max-w-xs">{subtitle}</p>
          </div>

          {/* Body Content */}
          <div>{children}</div>
        </div>

        {/* Optional Footer */}
        {footer && (
          <div className="px-8 py-4 bg-gray-50/80 border-t border-gray-100 text-center text-sm text-gray-600">
            {footer}
          </div>
        )}
      </div>
    </div>
  );
}
