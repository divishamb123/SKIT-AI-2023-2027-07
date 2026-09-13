'use client';

import React, { useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { AuthCard } from '@/components/auth/AuthCard';
import { Input } from '@/components/ui/Input';
import { Button } from '@/components/ui/Button';
import { FormAlert } from '@/components/ui/FormAlert';
import { validateEmail } from '@/lib/validation';
import { useAuth } from '@/context/AuthContext';

export default function LoginPage() {
  const router = useRouter();
  const { login, fillDemoCredentials, isAuthenticated, user } = useAuth();

  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [rememberMe, setRememberMe] = useState(true);

  // Field validation error states
  const [emailError, setEmailError] = useState<string | null>(null);
  const [passwordError, setPasswordError] = useState<string | null>(null);

  // Form level states
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [formError, setFormError] = useState<string | null>(null);
  const [formSuccess, setFormSuccess] = useState<string | null>(null);

  // Forgot password modal state
  const [showForgotModal, setShowForgotModal] = useState(false);
  const [forgotEmail, setForgotEmail] = useState('');
  const [forgotStatus, setForgotStatus] = useState<string | null>(null);

  // Handle demo autofill
  const handleQuickDemo = (type: 'analyst' | 'security' | 'auditor' | string) => {
    const creds = fillDemoCredentials(type);
    setEmail(creds.email);
    setPassword(creds.password);
    setEmailError(null);
    setPasswordError(null);
    setFormError(null);
  };

  const handleBlurEmail = () => {
    if (email) {
      setEmailError(validateEmail(email));
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setFormError(null);
    setFormSuccess(null);

    // Validate inputs
    const eErr = validateEmail(email);
    const pErr = !password ? 'Password is required to sign in.' : null;

    setEmailError(eErr);
    setPasswordError(pErr);

    if (eErr || pErr) {
      return;
    }

    setIsSubmitting(true);

    try {
      const result = await login(email, password, rememberMe);
      if (!result.success) {
        setFormError(result.error || 'Authentication failed. Please verify credentials.');
        setIsSubmitting(false);
      } else {
        setFormSuccess('Authentication successful! Redirecting to Forensic Dashboard...');
        setTimeout(() => {
          router.push('/dashboard');
        }, 800);
      }
    } catch {
      setFormError('An unexpected network error occurred. Please try again.');
      setIsSubmitting(false);
    }
  };

  const handleForgotSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const err = validateEmail(forgotEmail);
    if (err) {
      setForgotStatus(`Error: ${err}`);
      return;
    }
    setForgotStatus(
      `Password reset instructions have been dispatched to ${forgotEmail}. Please check your institutional inbox.`
    );
  };

  return (
    <div className="flex-1 flex items-center justify-center py-12 px-4 sm:px-6 lg:px-8 bg-gradient-to-b from-gray-50 to-gray-100">
      <AuthCard
        title="Sign in to your account"
        subtitle="Enterprise Synthetic Media & Content Authenticity Platform"
        badge="Enterprise Forensics Console"
        footer={
          <p>
            Don&apos;t have an account yet?{' '}
            <Link
              href="/register"
              className="font-semibold text-blue-600 hover:text-blue-500 hover:underline transition-colors"
            >
              Create an account
            </Link>
          </p>
        }
      >
        {/* If already authenticated notice */}
        {isAuthenticated && user && !formSuccess && (
          <div className="mb-6 p-3 bg-blue-50 border border-blue-200 rounded-lg text-xs text-blue-800 flex items-center justify-between">
            <div>
              <p className="font-semibold">Currently signed in as:</p>
              <p className="text-gray-700">{user.name} ({user.email})</p>
            </div>
            <Link
              href="/dashboard"
              className="px-2.5 py-1 bg-blue-600 text-white rounded font-medium hover:bg-blue-700 transition-colors"
            >
              Go to Dashboard
            </Link>
          </div>
        )}

        {/* Form Alerts */}
        {formError && (
          <FormAlert
            type="error"
            message={formError}
            className="mb-6"
            onDismiss={() => setFormError(null)}
          />
        )}

        {formSuccess && (
          <FormAlert
            type="success"
            message={formSuccess}
            className="mb-6"
          />
        )}

        <form onSubmit={handleSubmit} className="space-y-5" noValidate>
          <Input
            id="login-email"
            name="email"
            type="email"
            label="Corporate / Organization Email"
            placeholder="e.g., analyst@aiforensics.io"
            value={email}
            onChange={(e) => {
              setEmail(e.target.value);
              if (emailError) setEmailError(null);
            }}
            onBlur={handleBlurEmail}
            error={emailError || undefined}
            required
            disabled={isSubmitting}
            autoComplete="email"
            leftIcon={
              <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z" />
                <polyline points="22,6 12,13 2,6" />
              </svg>
            }
          />

          <div>
            <Input
              id="login-password"
              name="password"
              type="password"
              label="Password"
              placeholder="••••••••"
              value={password}
              onChange={(e) => {
                setPassword(e.target.value);
                if (passwordError) setPasswordError(null);
              }}
              error={passwordError || undefined}
              required
              showPasswordToggle
              disabled={isSubmitting}
              autoComplete="current-password"
              leftIcon={
                <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <rect x="3" y="11" width="18" height="11" rx="2" ry="2" />
                  <path d="M7 11V7a5 5 0 0 1 10 0v4" />
                </svg>
              }
            />

            <div className="flex items-center justify-between mt-3 text-xs">
              <label className="flex items-center text-gray-600 cursor-pointer select-none">
                <input
                  type="checkbox"
                  checked={rememberMe}
                  onChange={(e) => setRememberMe(e.target.checked)}
                  className="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500 cursor-pointer"
                />
                <span className="ml-2">Remember me for 30 days</span>
              </label>

              <button
                type="button"
                onClick={() => setShowForgotModal(true)}
                className="font-medium text-blue-600 hover:text-blue-500 hover:underline cursor-pointer"
              >
                Forgot password?
              </button>
            </div>
          </div>

          <Button
            type="submit"
            variant="primary"
            size="lg"
            fullWidth
            isLoading={isSubmitting}
          >
            {isSubmitting ? 'Authenticating...' : 'Sign in to Console'}
          </Button>

          {/* Quick Demo Credentials Panel */}
          <div className="pt-4 mt-4 border-t border-gray-100">
            <p className="text-[11px] font-semibold text-gray-400 uppercase tracking-wider text-center mb-2.5">
              Quick Sign-In Demo Accounts
            </p>
            <div className="grid grid-cols-3 gap-2">
              <button
                type="button"
                onClick={() => handleQuickDemo('analyst')}
                className="px-2 py-1.5 text-xs font-medium text-blue-700 bg-blue-50 hover:bg-blue-100 border border-blue-200 rounded-lg transition-colors cursor-pointer text-center"
              >
                Lead Analyst
              </button>
              <button
                type="button"
                onClick={() => handleQuickDemo('security')}
                className="px-2 py-1.5 text-xs font-medium text-indigo-700 bg-indigo-50 hover:bg-indigo-100 border border-indigo-200 rounded-lg transition-colors cursor-pointer text-center"
              >
                Security Lead
              </button>
              <button
                type="button"
                onClick={() => handleQuickDemo('auditor')}
                className="px-2 py-1.5 text-xs font-medium text-teal-700 bg-teal-50 hover:bg-teal-100 border border-teal-200 rounded-lg transition-colors cursor-pointer text-center"
              >
                Compliance
              </button>
            </div>
          </div>
        </form>
      </AuthCard>

      {/* Forgot Password Modal */}
      {showForgotModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50 backdrop-blur-xs">
          <div className="bg-white rounded-2xl max-w-md w-full p-6 shadow-2xl border border-gray-100 animate-in fade-in zoom-in-95 duration-150">
            <div className="flex justify-between items-start mb-4">
              <h3 className="text-lg font-bold text-gray-900">Reset Password</h3>
              <button
                type="button"
                onClick={() => {
                  setShowForgotModal(false);
                  setForgotStatus(null);
                }}
                className="text-gray-400 hover:text-gray-600 cursor-pointer"
                aria-label="Close modal"
              >
                <svg className="w-5 h-5" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
                </svg>
              </button>
            </div>

            {forgotStatus ? (
              <div className="space-y-4">
                <FormAlert
                  type={forgotStatus.startsWith('Error') ? 'error' : 'success'}
                  message={forgotStatus}
                />
                <Button
                  fullWidth
                  variant="outline"
                  onClick={() => {
                    setShowForgotModal(false);
                    setForgotStatus(null);
                  }}
                >
                  Back to Sign In
                </Button>
              </div>
            ) : (
              <form onSubmit={handleForgotSubmit} className="space-y-4">
                <p className="text-sm text-gray-600">
                  Enter your registered institutional email address and we will issue password reset instructions.
                </p>
                <Input
                  id="forgot-email"
                  type="email"
                  label="Email address"
                  placeholder="analyst@aiforensics.io"
                  value={forgotEmail}
                  onChange={(e) => setForgotEmail(e.target.value)}
                  required
                />
                <div className="flex gap-2 justify-end pt-2">
                  <Button
                    type="button"
                    variant="ghost"
                    onClick={() => setShowForgotModal(false)}
                  >
                    Cancel
                  </Button>
                  <Button type="submit" variant="primary">
                    Send Reset Link
                  </Button>
                </div>
              </form>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
