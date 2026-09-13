import { evaluatePasswordRules } from '@/components/auth/PasswordRequirements';

export const EMAIL_REGEX = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/;

export function validateEmail(email: string): string | null {
  if (!email || email.trim() === '') {
    return 'Email address is required.';
  }
  if (!EMAIL_REGEX.test(email.trim())) {
    return 'Please enter a valid email address (e.g., analyst@aiforensics.io).';
  }
  return null;
}

export function validateName(name: string): string | null {
  if (!name || name.trim() === '') {
    return 'Full name is required.';
  }
  if (name.trim().length < 2) {
    return 'Name must be at least 2 characters long.';
  }
  if (!/^[a-zA-Z\s.\-()']+$/.test(name.trim())) {
    return 'Name can only contain letters, spaces, dots, or hyphens.';
  }
  return null;
}

export function validatePassword(password: string): string | null {
  if (!password) {
    return 'Password is required.';
  }
  const { isSatisfied, rules } = evaluatePasswordRules(password);
  if (!isSatisfied) {
    const unmet = rules.filter((r) => !r.met).map((r) => r.label);
    return `Password does not meet all requirements (${unmet.length} missing).`;
  }
  return null;
}

export function validatePasswordMatch(password: string, confirmPassword: string): string | null {
  if (!confirmPassword) {
    return 'Please confirm your password.';
  }
  if (password !== confirmPassword) {
    return 'Passwords do not match.';
  }
  return null;
}
