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

export const authService = {
  /**
   * Login user with email and password
   */
  login: async (credentials: LoginRequest): Promise<LoginResponse> => {
    try {
      const response = await fetch(`${API_BASE_URL}/users/login`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(credentials),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.message || 'Login failed');
      }

      return await response.json();
    } catch (error) {
      throw error;
    }
  },

  /**
   * Register new user
   */
  register: async (data: { email: string; password: string; name?: string }): Promise<AuthResponse> => {
    try {
      const response = await fetch(`${API_BASE_URL}/users/register`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.message || 'Registration failed');
      }

      return await response.json();
    } catch (error) {
      throw error;
    }
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
      const response = await fetch(`${API_BASE_URL}/users/google-signin`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ credential }),
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
