import React from 'react';

export interface PasswordRule {
  id: string;
  label: string;
  met: boolean;
}

export function evaluatePasswordRules(password: string): {
  rules: PasswordRule[];
  score: number;
  strengthLabel: 'Too Weak' | 'Weak' | 'Moderate' | 'Strong' | 'Very Strong';
  isSatisfied: boolean;
} {
  const rules: PasswordRule[] = [
    {
      id: 'length',
      label: 'At least 8 characters',
      met: password.length >= 8,
    },
    {
      id: 'uppercase',
      label: 'At least one uppercase letter (A-Z)',
      met: /[A-Z]/.test(password),
    },
    {
      id: 'lowercase',
      label: 'At least one lowercase letter (a-z)',
      met: /[a-z]/.test(password),
    },
    {
      id: 'number',
      label: 'At least one number (0-9)',
      met: /[0-9]/.test(password),
    },
    {
      id: 'special',
      label: 'At least one special symbol (!@#$%^&*)',
      met: /[^A-Za-z0-9]/.test(password),
    },
  ];

  const metCount = rules.filter((r) => r.met).length;

  let strengthLabel: 'Too Weak' | 'Weak' | 'Moderate' | 'Strong' | 'Very Strong' =
    'Too Weak';
  if (metCount === 0 || password.length === 0) strengthLabel = 'Too Weak';
  else if (metCount <= 2) strengthLabel = 'Weak';
  else if (metCount === 3) strengthLabel = 'Moderate';
  else if (metCount === 4) strengthLabel = 'Strong';
  else if (metCount === 5) strengthLabel = 'Very Strong';

  return {
    rules,
    score: metCount,
    strengthLabel,
    isSatisfied: metCount === 5,
  };
}

interface PasswordRequirementsProps {
  password: string;
  className?: string;
  showChecklist?: boolean;
}

export function PasswordRequirements({
  password,
  className = '',
  showChecklist = true,
}: PasswordRequirementsProps) {
  const { rules, score, strengthLabel } = evaluatePasswordRules(password);

  const getBarColor = (scoreValue: number) => {
    switch (scoreValue) {
      case 1:
      case 2:
        return 'bg-red-500';
      case 3:
        return 'bg-amber-500';
      case 4:
        return 'bg-blue-500';
      case 5:
        return 'bg-emerald-500';
      default:
        return 'bg-gray-200';
    }
  };

  const getTextColor = (scoreValue: number) => {
    switch (scoreValue) {
      case 1:
      case 2:
        return 'text-red-600';
      case 3:
        return 'text-amber-600';
      case 4:
        return 'text-blue-600';
      case 5:
        return 'text-emerald-600';
      default:
        return 'text-gray-400';
    }
  };

  return (
    <div className={`mt-2 space-y-2.5 ${className}`}>
      {/* Strength Bar */}
      <div>
        <div className="flex justify-between items-center text-xs mb-1">
          <span className="font-medium text-gray-600">Password Strength:</span>
          <span className={`font-semibold ${getTextColor(score)}`}>
            {password ? strengthLabel : 'Not entered'}
          </span>
        </div>
        <div className="grid grid-cols-5 gap-1.5 h-1.5 w-full">
          {[1, 2, 3, 4, 5].map((step) => (
            <div
              key={step}
              className={`h-full rounded-full transition-all duration-300 ${
                score >= step ? getBarColor(score) : 'bg-gray-200'
              }`}
            />
          ))}
        </div>
      </div>

      {/* Checklist items */}
      {showChecklist && (
        <ul className="space-y-1.5 pt-1 text-xs text-gray-600">
          {rules.map((rule) => (
            <li
              key={rule.id}
              className={`flex items-center gap-2 transition-colors duration-200 ${
                rule.met ? 'text-emerald-600 font-medium' : 'text-gray-500'
              }`}
            >
              {rule.met ? (
                <svg
                  className="w-3.5 h-3.5 text-emerald-500 shrink-0"
                  viewBox="0 0 20 20"
                  fill="currentColor"
                >
                  <path
                    fillRule="evenodd"
                    d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                    clipRule="evenodd"
                  />
                </svg>
              ) : (
                <span className="w-3.5 h-3.5 flex items-center justify-center text-gray-400 shrink-0">
                  <span className="w-1.5 h-1.5 rounded-full bg-gray-300" />
                </span>
              )}
              <span>{rule.label}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
