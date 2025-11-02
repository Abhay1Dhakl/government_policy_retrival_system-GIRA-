'use client';
import { useState } from 'react'
import { useRouter } from 'next/navigation'
import LoginForm from '@/components/auth/LoginForm'
import { authService } from '@/lib/auth'

export default function LoginPage() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const router = useRouter();

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError('');

    try {
      const response = await authService.login({ email, password });
      
      const { access_token, refresh_token, has_details } = response.data;
      console.log('Login response data:', response.data);

      if (!access_token) {
        throw new Error('No access token received');
      }

      authService.setToken(access_token);
      if (refresh_token) {
        authService.setRefreshToken(refresh_token);
      }
      if (typeof has_details !== 'undefined') {
        authService.setHasDetails(has_details);
      }

      console.log('Login successful:', response.message, 'has_details:', has_details);

      if (has_details === true) {
        router.push('/chat');
      } else {
        router.push('/users');
      }
    } catch (error) {
      console.error('Login failed:', error);
      setError(error instanceof Error ? error.message : 'Login failed. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleGoogleSignIn = async (credential: string) => {
    setIsLoading(true);
    setError('');

    try {
      const response = await authService.googleSignIn(credential);
      
      const { access_token, refresh_token, has_details } = response.data;

      if (!access_token) {
        throw new Error('No access token received');
      }

      console.log('Google sign-in successful:', response.message, 'has_details:', has_details);

      // Redirect based on profile completion
      if (has_details === true) {
        router.push('/chat');
      } else {
        router.push('/users');
      }
    } catch (error) {
      console.error('Google sign-in failed:', error);
      setError(error instanceof Error ? error.message : 'Google sign-in failed. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <LoginForm
      email={email}
      password={password}
      onEmailChange={setEmail}
      onPasswordChange={setPassword}
      onLogin={handleLogin}
      onGoogleSignIn={handleGoogleSignIn}
      isLoading={isLoading}
      error={error}
    />
  );
}