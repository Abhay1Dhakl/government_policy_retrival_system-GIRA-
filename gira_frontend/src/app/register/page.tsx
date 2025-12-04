'use client';
import { useState } from 'react'
import { useRouter } from 'next/navigation'
import RegisterForm from '@/components/auth/RegisterForm'
import { authService } from '@/lib/auth'

export default function RegisterPage() {
  const [formData, setFormData] = useState({
    email: '',
    password: '',
    passwordConfirm: '',
    firstName: '',
    lastName: ''
  });
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const router = useRouter();

  const handleRegister = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError('');

    // Validate passwords match
    if (formData.password !== formData.passwordConfirm) {
      setError('Passwords do not match');
      setIsLoading(false);
      return;
    }

    try {
      const response = await authService.register({
        email: formData.email,
        password: formData.password,
        password_confirm: formData.passwordConfirm,
        first_name: formData.firstName,
        last_name: formData.lastName
      });
      
      const { tokens } = response.data;
      const { access, refresh } = tokens;

      if (!access) {
        throw new Error('No access token received');
      }

      authService.setToken(access);
      if (refresh) {
        authService.setRefreshToken(refresh);
      }

      console.log('Registration successful:', response.message);

      // Redirect to users page to complete profile
      router.push('/users');
    } catch (error: any) {
      console.error('Registration failed:', error);
      
      // Handle validation errors from backend
      if (error.response?.data?.data) {
        const errors = error.response.data.data;
        const errorMessages = Object.entries(errors)
          .map(([field, messages]: [string, any]) => {
            const fieldName = field.replace('_', ' ');
            return `${fieldName}: ${Array.isArray(messages) ? messages.join(', ') : messages}`;
          })
          .join('\n');
        setError(errorMessages);
      } else {
        setError(error.response?.data?.message || error.message || 'Registration failed. Please try again.');
      }
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
    <RegisterForm
      formData={formData}
      onFormDataChange={setFormData}
      onRegister={handleRegister}
      onGoogleSignIn={handleGoogleSignIn}
      isLoading={isLoading}
      error={error}
    />
  );
}
