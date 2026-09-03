import React from 'react';
import { Card, CardContent, CardHeader } from '@/components/ui/Card';

export default function HealthPage() {
  return (
    <div className="max-w-7xl mx-auto py-12 px-4 sm:px-6 lg:px-8">
      <Card className="max-w-3xl mx-auto">
        <CardHeader>
          <h2 className="text-2xl font-bold text-gray-900">System Health Status</h2>
        </CardHeader>
        <CardContent>
          <div className="space-y-6">
            <div className="flex items-center justify-between py-3 border-b border-gray-200">
              <div className="flex items-center">
                <div className="h-2.5 w-2.5 rounded-full bg-green-500 mr-2"></div>
                <span className="text-sm font-medium text-gray-900">Frontend Service</span>
              </div>
              <span className="text-sm text-gray-500">Operational</span>
            </div>
            <div className="flex items-center justify-between py-3 border-b border-gray-200">
              <div className="flex items-center">
                <div className="h-2.5 w-2.5 rounded-full bg-gray-300 mr-2"></div>
                <span className="text-sm font-medium text-gray-900">API Backend</span>
              </div>
              <span className="text-sm text-gray-500">Not Connected (Placeholder)</span>
            </div>
            <div className="flex items-center justify-between py-3 border-b border-gray-200">
              <div className="flex items-center">
                <div className="h-2.5 w-2.5 rounded-full bg-gray-300 mr-2"></div>
                <span className="text-sm font-medium text-gray-900">Image Model Microservice</span>
              </div>
              <span className="text-sm text-gray-500">Not Connected (Placeholder)</span>
            </div>
            <div className="flex items-center justify-between py-3">
              <div className="flex items-center">
                <div className="h-2.5 w-2.5 rounded-full bg-gray-300 mr-2"></div>
                <span className="text-sm font-medium text-gray-900">Text Model Microservice</span>
              </div>
              <span className="text-sm text-gray-500">Not Connected (Placeholder)</span>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
