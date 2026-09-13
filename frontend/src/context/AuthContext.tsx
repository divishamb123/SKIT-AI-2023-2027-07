'use client';

import React, { createContext, useContext, useState } from 'react';

export interface User {
  id: string;
  name: string;
  email: string;
  role: string;
  avatarUrl?: string;
  department?: string;
  createdAt: string;
}

interface AuthContextType {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  login: (
    email: string,
    password: string,
    remember?: boolean
  ) => Promise<{ success: boolean; error?: string }>;
  register: (
    name: string,
    email: string,
    password: string
  ) => Promise<{ success: boolean; error?: string }>;
  logout: () => void;
  fillDemoCredentials: (type?: 'divisha' | 'lead' | 'researcher') => {
    email: string;
    password: string;
  };
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

const STORAGE_KEY_USER = 'aiforensics_user';
const STORAGE_KEY_TOKEN = 'aiforensics_token';

// Demo users seeded for testing and evaluator demonstration
const DEMO_ACCOUNTS: Record<string, { user: User; passwordHash: string }> = {
  'divisha@skit.ac.in': {
    user: {
      id: 'usr-23eskca038',
      name: 'Divisha Manak Bohra',
      email: 'divisha@skit.ac.in',
      role: 'Image Forensics Lead',
      department: 'CSE (Artificial Intelligence), SKIT',
      createdAt: '2026-08-03T09:00:00Z',
    },
    passwordHash: 'Password@123',
  },
  'dev@skit.ac.in': {
    user: {
      id: 'usr-23eskca035',
      name: 'Dev Khandelwal',
      email: 'dev@skit.ac.in',
      role: 'Team Lead & Backend Lead',
      department: 'CSE (Artificial Intelligence), SKIT',
      createdAt: '2026-08-03T09:00:00Z',
    },
    passwordHash: 'Password@123',
  },
  'aryansh@skit.ac.in': {
    user: {
      id: 'usr-23eskca021',
      name: 'Aryansh Agarwal',
      email: 'aryansh@skit.ac.in',
      role: 'Frontend & Text Lead',
      department: 'CSE (Artificial Intelligence), SKIT',
      createdAt: '2026-08-03T09:00:00Z',
    },
    passwordHash: 'Password@123',
  },
};

export function AuthProvider({ children }: { children: React.ReactNode }) {
  // Lazy initial state avoids cascading render warning in React 19
  const [user, setUser] = useState<User | null>(() => {
    if (typeof window === 'undefined') return null;
    try {
      const stored = localStorage.getItem(STORAGE_KEY_USER);
      return stored ? JSON.parse(stored) : null;
    } catch {
      return null;
    }
  });

  const [token, setToken] = useState<string | null>(() => {
    if (typeof window === 'undefined') return null;
    try {
      return localStorage.getItem(STORAGE_KEY_TOKEN);
    } catch {
      return null;
    }
  });

  const [isLoading, setIsLoading] = useState(false);

  const login = async (
    email: string,
    password: string,
    remember = true
  ): Promise<{ success: boolean; error?: string }> => {
    setIsLoading(true);

    // Simulate realistic network roundtrip
    await new Promise((resolve) => setTimeout(resolve, 600));

    const normalizedEmail = email.trim().toLowerCase();

    // Check pre-configured demo users
    const matchedDemo = DEMO_ACCOUNTS[normalizedEmail];
    let authenticatedUser: User | null = null;

    if (matchedDemo) {
      if (matchedDemo.passwordHash !== password) {
        setIsLoading(false);
        return {
          success: false,
          error: 'Invalid password for this account. (Hint: demo password is Password@123)',
        };
      }
      authenticatedUser = matchedDemo.user;
    } else {
      // Check dynamically registered user in localStorage
      try {
        const customUsersJson = localStorage.getItem('aiforensics_registered_users');
        const customUsers = customUsersJson ? JSON.parse(customUsersJson) : {};
        if (customUsers[normalizedEmail]) {
          if (customUsers[normalizedEmail].password !== password) {
            setIsLoading(false);
            return {
              success: false,
              error: 'Invalid password. Please check your credentials.',
            };
          }
          authenticatedUser = customUsers[normalizedEmail].user;
        } else {
          // Reject unknown account
          setIsLoading(false);
          return {
            success: false,
            error:
              'No account found with this email address. Please register first or use the Demo Login option.',
          };
        }
      } catch {
        setIsLoading(false);
        return {
          success: false,
          error: 'Authentication failed due to local storage error.',
        };
      }
    }

    if (!authenticatedUser) {
      setIsLoading(false);
      return {
        success: false,
        error: 'Authentication failed. Please verify credentials.',
      };
    }

    // Success: issue simulated JWT bearer token
    const mockToken = `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.${btoa(
      JSON.stringify({ sub: authenticatedUser.id, email: authenticatedUser.email })
    )}.mockSignature`;

    setUser(authenticatedUser);
    setToken(mockToken);

    if (remember) {
      try {
        localStorage.setItem(STORAGE_KEY_USER, JSON.stringify(authenticatedUser));
        localStorage.setItem(STORAGE_KEY_TOKEN, mockToken);
      } catch {
        // Ignore localStorage quota errors
      }
    }

    setIsLoading(false);
    return { success: true };
  };

  const register = async (
    name: string,
    email: string,
    password: string
  ): Promise<{ success: boolean; error?: string }> => {
    setIsLoading(true);
    await new Promise((resolve) => setTimeout(resolve, 700));

    const normalizedEmail = email.trim().toLowerCase();

    if (DEMO_ACCOUNTS[normalizedEmail]) {
      setIsLoading(false);
      return {
        success: false,
        error: 'An account with this institutional email already exists. Please log in.',
      };
    }

    try {
      const customUsersJson = localStorage.getItem('aiforensics_registered_users');
      const customUsers = customUsersJson ? JSON.parse(customUsersJson) : {};

      if (customUsers[normalizedEmail]) {
        setIsLoading(false);
        return {
          success: false,
          error: 'An account with this email is already registered. Please sign in.',
        };
      }

      const newUser: User = {
        id: `usr-${Date.now().toString(36)}`,
        name: name.trim(),
        email: normalizedEmail,
        role: 'AI Forensics Analyst',
        department: 'CSE (Artificial Intelligence), SKIT',
        createdAt: new Date().toISOString(),
      };

      customUsers[normalizedEmail] = {
        user: newUser,
        password,
      };

      localStorage.setItem('aiforensics_registered_users', JSON.stringify(customUsers));

      const mockToken = `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.${btoa(
        JSON.stringify({ sub: newUser.id, email: newUser.email })
      )}.mockSignature`;

      setUser(newUser);
      setToken(mockToken);
      localStorage.setItem(STORAGE_KEY_USER, JSON.stringify(newUser));
      localStorage.setItem(STORAGE_KEY_TOKEN, mockToken);

      setIsLoading(false);
      return { success: true };
    } catch {
      setIsLoading(false);
      return {
        success: false,
        error: 'Failed to complete registration due to local storage limitation.',
      };
    }
  };

  const logout = () => {
    setUser(null);
    setToken(null);
    try {
      localStorage.removeItem(STORAGE_KEY_USER);
      localStorage.removeItem(STORAGE_KEY_TOKEN);
    } catch {
      // Ignore
    }
  };

  const fillDemoCredentials = (
    type: 'divisha' | 'lead' | 'researcher' = 'divisha'
  ) => {
    switch (type) {
      case 'lead':
        return { email: 'dev@skit.ac.in', password: 'Password@123' };
      case 'researcher':
        return { email: 'aryansh@skit.ac.in', password: 'Password@123' };
      case 'divisha':
      default:
        return { email: 'divisha@skit.ac.in', password: 'Password@123' };
    }
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        token,
        isAuthenticated: Boolean(user && token),
        isLoading,
        login,
        register,
        logout,
        fillDemoCredentials,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth(): AuthContextType {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}
