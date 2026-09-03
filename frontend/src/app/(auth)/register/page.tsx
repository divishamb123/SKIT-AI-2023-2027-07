'use client';

import React from 'react';
import { Card, CardContent, CardHeader } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import Link from 'next/link';

export default function RegisterPage() {
  return (
    <div className="min-h-full flex items-center justify-center py-12 px-4 sm:px-6 lg:px-8">
      <Card className="max-w-md w-full space-y-8">
        <CardHeader>
          <h2 className="mt-6 text-center text-3xl font-extrabold text-gray-900">Create your account</h2>
        </CardHeader>
        <CardContent>
          <form className="mt-8 space-y-6" onSubmit={(e) => e.preventDefault()}>
            <div className="rounded-md shadow-sm -space-y-px">
              <div>
                <label htmlFor="name" className="sr-only">Full Name</label>
                <input id="name" name="name" type="text" required className="appearance-none rounded-none relative block w-full px-3 py-2 border border-gray-300 placeholder-gray-500 text-gray-900 rounded-t-md focus:outline-none focus:ring-blue-500 focus:border-blue-500 focus:z-10 sm:text-sm" placeholder="Full Name" />
              </div>
              <div>
                <label htmlFor="email-address" className="sr-only">Email address</label>
                <input id="email-address" name="email" type="email" required className="appearance-none rounded-none relative block w-full px-3 py-2 border border-gray-300 placeholder-gray-500 text-gray-900 focus:outline-none focus:ring-blue-500 focus:border-blue-500 focus:z-10 sm:text-sm" placeholder="Email address" />
              </div>
              <div>
                <label htmlFor="password" className="sr-only">Password</label>
                <input id="password" name="password" type="password" required className="appearance-none rounded-none relative block w-full px-3 py-2 border border-gray-300 placeholder-gray-500 text-gray-900 rounded-b-md focus:outline-none focus:ring-blue-500 focus:border-blue-500 focus:z-10 sm:text-sm" placeholder="Password" />
              </div>
            </div>
            <div>
              <Button type="submit" className="w-full flex justify-center py-2 px-4">
                Register
              </Button>
            </div>
            <div className="text-center mt-4">
              <Link href="/login" className="text-sm font-medium text-blue-600 hover:text-blue-500">
                Already have an account? Sign in
              </Link>
            </div>
          </form>
        </CardContent>
      </Card>
    </div>
  );
}
