/**
 * Authentication Service
 * Handles login, token management, and user authentication state
 */

interface LoginRequest {
  email: string;
  password: string;
}

interface LoginResponse {
  data: {
    access_token: string;
    refresh_token?: string;
    has_details?: boolean;
  };
  message?: string;
}

interface AuthResponse {
  message: string;
}

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8082/api/v1';
const TOKEN_KEY = 'access_token';
const REFRESH_TOKEN_KEY = 'refresh_token';
const HAS_DETAILS_KEY = 'has_details';

const extractApiErrorMessage = (payload: unknown): string | null => {
  if (!payload) {
    return null;
  }

  if (typeof payload === 'string') {
    return payload;
  }

  if (Array.isArray(payload)) {
    for (const item of payload) {
      const message = extractApiErrorMessage(item);
      if (message) {
        return message;
      }
    }
    return null;
  }

  if (typeof payload === 'object') {
    const record = payload as Record<string, unknown>;

    for (const key of ['message', 'detail', 'error']) {
      const value = record[key];
      if (typeof value === 'string' && value.trim()) {
        return value;
      }
    }

    for (const key of ['errors', 'data']) {
      const message = extractApiErrorMessage(record[key]);
      if (message) {
        return message;
      }
    }

    for (const value of Object.values(record)) {
      const message = extractApiErrorMessage(value);
      if (message) {
        return message;
      }
    }
  }

  return null;
};

export const authService = {
  /**
   * Login user with email and password
   */
  login: async (credentials: LoginRequest): Promise<LoginResponse> => {
    try {
      const response = await fetch(`${API_BASE_URL}/token/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(credentials),
      });

      if (!response.ok) {
        let errorMessage = 'Login failed';

        try {
          const error = await response.json();
          errorMessage = extractApiErrorMessage(error) || errorMessage;
        } catch {
          errorMessage = `Login failed (${response.status})`;
        }

        throw new Error(errorMessage);
      }

      return await response.json();
    } catch (error) {
      throw error;
    }
  },

  /**
   * Store user feedback for a conversation/response
   */
  storeFeedback: async (feedbackData: Record<string, unknown>) => {
    const token = authService.getToken();

    const response = await fetch("/api/store-feedback", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
      body: JSON.stringify(feedbackData),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(errorText || "Failed to store feedback");
    }

    return response.json();
  },

  /**
   * Register new user
   */
  register: async (data: { email: string; password: string; name?: string }): Promise<AuthResponse> => {
    try {
      const response = await fetch(`${API_BASE_URL}/users/register/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
      });

      if (!response.ok) {
        let errorMessage = 'Registration failed';
        try {
          const error = await response.json();
          errorMessage = error.message || (error.data ? JSON.stringify(error.data) : errorMessage);
          if (error.errors) {
            errorMessage += ': ' + JSON.stringify(error.errors);
          }
          console.error('Registration error (JSON):', error);
        } catch {
          const text = await response.text();
          console.error('Registration error (Non-JSON):', text.substring(0, 500)); // Log first 500 chars
          errorMessage = `Server Error (${response.status}): ${response.statusText}`;
        }
        throw new Error(errorMessage);
      }

      return await response.json();
    } catch (error) {
      throw error;
    }
  },

  /**
   * Create a password for an existing account
   */
  createPassword: async (data: { email: string; password: string }): Promise<AuthResponse> => {
    const response = await fetch('/api/auth/create-password', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => null);
      throw new Error(extractApiErrorMessage(error) || 'Password creation failed');
    }

    return await response.json();
  },

  /**
   * Set access token in localStorage
   */
  setToken: (token: string) => {
    if (typeof window !== 'undefined') {
      localStorage.setItem(TOKEN_KEY, token);
    }
  },

  /**
   * Get access token from localStorage
   */
  getToken: (): string | null => {
    if (typeof window !== 'undefined') {
      return localStorage.getItem(TOKEN_KEY);
    }
    return null;
  },

  /**
   * Set refresh token in localStorage
   */
  setRefreshToken: (token: string) => {
    if (typeof window !== 'undefined') {
      localStorage.setItem(REFRESH_TOKEN_KEY, token);
    }
  },

  /**
   * Get refresh token from localStorage
   */
  getRefreshToken: (): string | null => {
    if (typeof window !== 'undefined') {
      return localStorage.getItem(REFRESH_TOKEN_KEY);
    }
    return null;
  },

  /**
   * Set user details flag
   */
  setHasDetails: (hasDetails: boolean) => {
    if (typeof window !== 'undefined') {
      localStorage.setItem(HAS_DETAILS_KEY, JSON.stringify(hasDetails));
    }
  },

  /**
   * Get user details flag
   */
  getHasDetails: (): boolean => {
    if (typeof window !== 'undefined') {
      const value = localStorage.getItem(HAS_DETAILS_KEY);
      return value ? JSON.parse(value) : false;
    }
    return false;
  },

  /**
   * Check if user is authenticated
   */
  isAuthenticated: (): boolean => {
    return !!authService.getToken();
  },

  /**
   * Check whether the user profile is already completed
   */
  isProfileCompleted: (): boolean => {
    return authService.getHasDetails();
  },

  /**
   * Get Authorization header
   */
  getAuthHeader: (): Partial<Record<'Authorization', string>> => {
    const token = authService.getToken();
    if (token) {
      return { Authorization: `Bearer ${token}` };
    }
    return {};
  },

  /**
   * Logout user and clear tokens
   */
  logout: () => {
    if (typeof window !== 'undefined') {
      localStorage.removeItem(TOKEN_KEY);
      localStorage.removeItem(REFRESH_TOKEN_KEY);
      localStorage.removeItem(HAS_DETAILS_KEY);
    }
  },

  /**
   * Refresh access token using refresh token
   */
  refreshToken: async (): Promise<{ access_token: string }> => {
    try {
      const refreshToken = authService.getRefreshToken();
      if (!refreshToken) {
        throw new Error('No refresh token available');
      }

      const response = await fetch(`${API_BASE_URL}/users/refresh-token`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${authService.getToken()}`,
        },
        body: JSON.stringify({ refresh_token: refreshToken }),
      });

      if (!response.ok) {
        throw new Error('Token refresh failed');
      }

      const data = await response.json();
      authService.setToken(data.access_token);
      return data;
    } catch (error) {
      authService.logout();
      throw error;
    }
  },

  /**
   * Google Sign-In with credential
   */
  googleSignIn: async (credential: string): Promise<LoginResponse> => {
    try {
      const response = await fetch(`${API_BASE_URL}/token/oauth/callback/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ token: credential }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.message || 'Google sign-in failed');
      }

      return await response.json();
    } catch (error) {
      throw error;
    }
  },
};
