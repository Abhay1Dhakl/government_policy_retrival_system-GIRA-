"use client"

import { useRouter } from "next/navigation"
import { authService } from '@/lib/auth'
import CreatePasswordForm from "@/components/auth/CreatePasswordForm"

export default function CreatePasswordPage() {
  const router = useRouter()

  const handleSubmit = async (data: {
    email: string
    password: string
  }) => {
    try {
      await authService.createPassword(data)
      router.push("/login")
    } catch (err) {
      console.error("Password creation failed:", err)
    }
  }

  const handleGoogleSignIn = async (credential: string) => {
    try {
      console.log('Google credential received:', credential.substring(0, 50) + '...')
      await authService.googleSignIn(credential)
      
      // After successful Google sign-in, redirect based on has_details
      if (authService.isProfileCompleted()) {
        router.push("/chat")
      } else {
        router.push("/users")
      }
    } catch (err) {
      console.error("Google sign-in failed:", err)
    }
  }

  return (
    <CreatePasswordForm 
      onSubmit={handleSubmit}
      onGoogleSignIn={handleGoogleSignIn}
    />
  )
}