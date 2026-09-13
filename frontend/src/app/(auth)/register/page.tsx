'use client';

import React, { useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { AuthCard } from '@/components/auth/AuthCard';
import { Input } from '@/components/ui/Input';
import { Button } from '@/components/ui/Button';
import { FormAlert } from '@/components/ui/FormAlert';
import { PasswordRequirements, evaluatePasswordRules } from '@/components/auth/PasswordRequirements';
import {
  validateEmail,
  validateName,
  validatePasswordMatch,
} from '@/lib/validation';
import { useAuth } from '@/context/AuthContext';

export default function RegisterPage() {
  const router = useRouter();
  const { register } = useAuth();

  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [agreeTerms, setAgreeTerms] = useState(false);

  // Field validation errors
  const [nameError, setNameError] = useState<string | null>(null);
  const [emailError, setEmailError] = useState<string | null>(null);
  const [passwordError, setPasswordError] = useState<string | null>(null);
  const [confirmError, setConfirmError] = useState<string | null>(null);
  const [termsError, setTermsError] = useState<string | null>(null);

  // Form states
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [formError, setFormError] = useState<string | null>(null);
  const [formSuccess, setFormSuccess] = useState<string | null>(null);

  // Autofill sample data for quick evaluation
  const handleAutofillDemo = () => {
    const randomId = Math.floor(100 + Math.random() * 900);
    setName('Divisha Bohra');
    setEmail(`analyst_${randomId}@skit.ac.in`);
    setPassword('SecurePass@2026');
    setConfirmPassword('SecurePass@2026');
    setAgreeTerms(true);
    setNameError(null);
    setEmailError(null);
    setPasswordError(null);
    setConfirmError(null);
    setTermsError(null);
    setFormError(null);
  };

  const handleBlurEmail = () => {
    if (email) setEmailError(validateEmail(email));
  };

  const handleBlurName = () => {
    if (name) setNameError(validateName(name));
  };

  const handleBlurConfirm = () => {
    if (confirmPassword) {
      setConfirmError(validatePasswordMatch(password, confirmPassword));
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setFormError(null);
    setFormSuccess(null);

    // Validate all fields
    const nErr = validateName(name);
    const eErr = validateEmail(email);
    const { isSatisfied } = evaluatePasswordRules(password);
    const pErr = isSatisfied ? null : 'Password does not meet all security requirements.';
    const cErr = validatePasswordMatch(password, confirmPassword);
    const tErr = !agreeTerms ? 'You must agree to the research & ethics guidelines.' : null;

    setNameError(nErr);
    setEmailError(eErr);
    setPasswordError(pErr);
    setConfirmError(cErr);
    setTermsError(tErr);

    if (nErr || eErr || pErr || cErr || tErr) {
      return;
    }

    setIsSubmitting(true);

    try {
      const result = await register(name, email, password);
      if (!result.success) {
        setFormError(result.error || 'Failed to complete registration.');
        setIsSubmitting(false);
      } else {
        setFormSuccess(
          'Account successfully created! Redirecting to Forensic Dashboard...'
        );
        setTimeout(() => {
          router.push('/dashboard');
        }, 900);
      }
    } catch {
      setFormError('An unexpected server error occurred during registration.');
      setIsSubmitting(false);
    }
  };

  return (
    <div className="flex-1 flex items-center justify-center py-12 px-4 sm:px-6 lg:px-8 bg-gradient-to-b from-gray-50 to-gray-100">
      <AuthCard
        title="Create your account"
        subtitle="Join the SKIT Multi-Model AI Detection & Forensics Lab"
        badge="Sprint 1 — Registration Interface"
        footer={
          <p>
            Already have an account?{' '}
            <Link
              href="/login"
              className="font-semibold text-blue-600 hover:text-blue-500 hover:underline transition-colors"
            >
              Sign in
            </Link>
          </p>
        }
      >
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

        <form onSubmit={handleSubmit} className="space-y-4" noValidate>
          <Input
            id="register-name"
            name="name"
            type="text"
            label="Full Name"
            placeholder="Divisha Manak Bohra"
            value={name}
            onChange={(e) => {
              setName(e.target.value);
              if (nameError) setNameError(null);
            }}
            onBlur={handleBlurName}
            error={nameError || undefined}
            required
            disabled={isSubmitting}
            autoComplete="name"
            leftIcon={
              <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
                <circle cx="12" cy="7" r="4" />
              </svg>
            }
          />

          <Input
            id="register-email"
            name="email"
            type="email"
            label="Institutional Email"
            placeholder="divisha@skit.ac.in"
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
              id="register-password"
              name="password"
              type="password"
              label="Password"
              placeholder="Create a strong password"
              value={password}
              onChange={(e) => {
                setPassword(e.target.value);
                if (passwordError) setPasswordError(null);
                if (confirmPassword && confirmError) {
                  setConfirmError(validatePasswordMatch(e.target.value, confirmPassword));
                }
              }}
              error={passwordError || undefined}
              required
              showPasswordToggle
              disabled={isSubmitting}
              autoComplete="new-password"
              leftIcon={
                <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <rect x="3" y="11" width="18" height="11" rx="2" ry="2" />
                  <path d="M7 11V7a5 5 0 0 1 10 0v4" />
                </svg>
              }
            />

            {/* Dynamic Password Strength & Requirements Checklist */}
            <PasswordRequirements password={password} />
          </div>

          <Input
            id="register-confirm"
            name="confirmPassword"
            type="password"
            label="Confirm Password"
            placeholder="Re-enter your password"
            value={confirmPassword}
            onChange={(e) => {
              setConfirmPassword(e.target.value);
              if (confirmError) setConfirmError(null);
            }}
            onBlur={handleBlurConfirm}
            error={confirmError || undefined}
            required
            showPasswordToggle
            disabled={isSubmitting}
            autoComplete="new-password"
            leftIcon={
              <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
              </svg>
            }
          />

          {/* Terms checkbox */}
          <div>
            <label className="flex items-start text-xs text-gray-600 cursor-pointer select-none">
              <input
                type="checkbox"
                checked={agreeTerms}
                onChange={(e) => {
                  setAgreeTerms(e.target.checked);
                  if (termsError) setTermsError(null);
                }}
                disabled={isSubmitting}
                className="mt-0.5 w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500 cursor-pointer"
              />
              <span className="ml-2">
                I agree to the{' '}
                <span className="font-semibold text-gray-800">
                  SKIT AI Research Code of Ethics
                </span>{' '}
                and terms of academic data handling.
              </span>
            </label>
            {termsError && (
              <p className="mt-1 text-xs text-red-600 font-medium">{termsError}</p>
            )}
          </div>

          <Button
            type="submit"
            variant="primary"
            size="lg"
            fullWidth
            isLoading={isSubmitting}
            className="mt-2"
          >
            {isSubmitting ? 'Creating Account...' : 'Complete Registration'}
          </Button>

          {/* Demo Autofill Shortcut */}
          <div className="pt-3 border-t border-gray-100 text-center">
            <button
              type="button"
              onClick={handleAutofillDemo}
              className="text-xs text-blue-600 hover:text-blue-700 font-medium hover:underline cursor-pointer"
            >
              Fill with sample evaluator data
            </button>
          </div>
        </form>
      </AuthCard>
    </div>
  );
}
