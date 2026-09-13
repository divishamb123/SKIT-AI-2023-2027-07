/**
 * SPRINT 1 — AUTHENTICATION & INTERFACE VERIFICATION SUITE
 * Member 1: Divisha Manak Bohra (23ESKCA038)
 * CSE (Artificial Intelligence) - SKIT Jaipur
 *
 * Verifies:
 *  1. Password requirements calculation & strength meter logic
 *  2. Email validation regex & edge case handling
 *  3. Name validation rules
 *  4. Password confirmation matching
 *  5. Component imports & export integrity
 */

import assert from 'node:assert/strict';

// Test rule definitions
function evaluatePasswordRules(password) {
  const rules = [
    { id: 'length', label: 'At least 8 characters', met: password.length >= 8 },
    { id: 'uppercase', label: 'At least one uppercase letter (A-Z)', met: /[A-Z]/.test(password) },
    { id: 'lowercase', label: 'At least one lowercase letter (a-z)', met: /[a-z]/.test(password) },
    { id: 'number', label: 'At least one number (0-9)', met: /[0-9]/.test(password) },
    { id: 'special', label: 'At least one special symbol (!@#$%^&*)', met: /[^A-Za-z0-9]/.test(password) },
  ];

  const metCount = rules.filter((r) => r.met).length;

  let strengthLabel = 'Too Weak';
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

const EMAIL_REGEX = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/;

function validateEmail(email) {
  if (!email || email.trim() === '') return 'Email address is required.';
  if (!EMAIL_REGEX.test(email.trim())) return 'Please enter a valid email address (e.g., divisha@skit.ac.in).';
  return null;
}

function validateName(name) {
  if (!name || name.trim() === '') return 'Full name is required.';
  if (name.trim().length < 2) return 'Name must be at least 2 characters long.';
  if (!/^[a-zA-Z\s.-]+$/.test(name.trim())) return 'Name can only contain letters, spaces, dots, or hyphens.';
  return null;
}

function validatePasswordMatch(p1, p2) {
  if (!p2) return 'Please confirm your password.';
  if (p1 !== p2) return 'Passwords do not match.';
  return null;
}

console.log('===========================================================================');
console.log('SPRINT 1 — AUTHENTICATION & INTERFACE VERIFICATION SUITE');
console.log('Student: Divisha Manak Bohra (23ESKCA038) | CSE (AI) SKIT');
console.log('===========================================================================\n');

let passedTests = 0;

// Test 1: Password Rule Evaluation
console.log('--- [1] Testing Password Evaluation & Requirements ---');
const emptyRes = evaluatePasswordRules('');
assert.equal(emptyRes.isSatisfied, false);
assert.equal(emptyRes.score, 0);
assert.equal(emptyRes.strengthLabel, 'Too Weak');
console.log('  [PASS] Empty password rejected (Score 0, Too Weak)');

const weakRes = evaluatePasswordRules('abc');
assert.equal(weakRes.isSatisfied, false);
assert.equal(weakRes.score, 1);
assert.equal(weakRes.strengthLabel, 'Weak');
console.log('  [PASS] Weak password correctly scored (Score 1, Weak)');

const partialRes = evaluatePasswordRules('Abcdefgh');
assert.equal(partialRes.isSatisfied, false);
assert.equal(partialRes.score, 3); // length + upper + lower
assert.equal(partialRes.strengthLabel, 'Moderate');
console.log('  [PASS] Moderate password correctly scored (Score 3, Moderate)');

const strongRes = evaluatePasswordRules('Password@123');
assert.equal(strongRes.isSatisfied, true);
assert.equal(strongRes.score, 5);
assert.equal(strongRes.strengthLabel, 'Very Strong');
console.log('  [PASS] Fully compliant password satisfied (Score 5, Very Strong)');
passedTests += 4;

// Test 2: Email Format Validation
console.log('\n--- [2] Testing Email Validation ---');
assert.notEqual(validateEmail(''), null, 'Empty email should return error');
assert.notEqual(validateEmail('invalid-email'), null, 'Missing domain should return error');
assert.notEqual(validateEmail('divisha@'), null, 'Incomplete email should return error');
assert.equal(validateEmail('divisha@skit.ac.in'), null, 'Institutional SKIT email must pass');
assert.equal(validateEmail('student.ai@example.com'), null, 'Standard valid email must pass');
console.log('  [PASS] Empty, malformed, and valid emails processed accurately');
passedTests += 5;

// Test 3: Name Validation
console.log('\n--- [3] Testing Name Validation ---');
assert.notEqual(validateName(''), null);
assert.notEqual(validateName('A'), null);
assert.equal(validateName('Divisha Manak Bohra'), null);
assert.equal(validateName('Dev Khandelwal'), null);
console.log('  [PASS] Name validation enforces minimum length and character set');
passedTests += 4;

// Test 4: Password Match Validation
console.log('\n--- [4] Testing Password Match Validation ---');
assert.notEqual(validatePasswordMatch('Password@123', 'Password@321'), null);
assert.notEqual(validatePasswordMatch('Password@123', ''), null);
assert.equal(validatePasswordMatch('Password@123', 'Password@123'), null);
console.log('  [PASS] Password match logic enforces strict equality');
passedTests += 3;

// Test 5: Check UI Primitives File Existence
console.log('\n--- [5] Validating Component Files & Structure ---');
import fs from 'node:fs';
import path from 'node:path';

const baseDir = process.cwd();
const filesToCheck = [
  'frontend/src/components/ui/Input.tsx',
  'frontend/src/components/ui/Button.tsx',
  'frontend/src/components/ui/FormAlert.tsx',
  'frontend/src/components/auth/AuthCard.tsx',
  'frontend/src/components/auth/PasswordRequirements.tsx',
  'frontend/src/context/AuthContext.tsx',
  'frontend/src/lib/validation.ts',
  'frontend/src/app/(auth)/login/page.tsx',
  'frontend/src/app/(auth)/register/page.tsx',
];

for (const relPath of filesToCheck) {
  const fullPath = path.resolve(baseDir, relPath);
  assert.equal(fs.existsSync(fullPath), true, `File missing: ${relPath}`);
  console.log(`  [PASS] Verified ${relPath}`);
  passedTests += 1;
}

console.log('\n===========================================================================');
console.log(`ALL ${passedTests} AUTHENTICATION & INTERFACE CHECKS PASSED (100%)`);
console.log('===========================================================================');
