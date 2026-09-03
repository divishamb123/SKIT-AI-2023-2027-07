import React from 'react';

export function Card({ className = '', children }: { className?: string, children: React.ReactNode }) {
  return (
    <div className={`bg-white rounded-lg border border-gray-200 shadow-sm overflow-hidden ${className}`}>
      {children}
    </div>
  );
}

export function CardHeader({ className = '', children }: { className?: string, children: React.ReactNode }) {
  return <div className={`px-6 py-4 border-b border-gray-200 ${className}`}>{children}</div>;
}

export function CardContent({ className = '', children }: { className?: string, children: React.ReactNode }) {
  return <div className={`px-6 py-4 ${className}`}>{children}</div>;
}

export function CardFooter({ className = '', children }: { className?: string, children: React.ReactNode }) {
  return <div className={`px-6 py-4 border-t border-gray-200 bg-gray-50 ${className}`}>{children}</div>;
}
