"use client"

import { useState, useEffect, useRef } from "react"
import { Eye, EyeOff } from "lucide-react"
import Image from "next/image"

interface CreatePasswordFormProps {
  onSubmit: (data: {
    email: string
    password: string
  }) => Promise<void>
  onGoogleSignIn: (credential: string) => Promise<void>
}

export default function CreatePasswordForm({ onSubmit, onGoogleSignIn }: CreatePasswordFormProps) {
  const [formData, setFormData] = useState({
    email: "",
    password: "",
    confirm_password: "",
  })
  const [errors, setErrors] = useState<Record<string, string>>({})
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [showPasswords, setShowPasswords] = useState({
    password: false,
    confirm_password: false,
  })
  const googleButtonRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const initializeGoogleSignIn = () => {
      const { google } = window as any
      
      if (!google || !googleButtonRef.current) {
        // Retry after a short delay if not loaded yet
        setTimeout(initializeGoogleSignIn, 100)
        return
      }

      try {
        google.accounts.id.initialize({
          client_id: process.env.NEXT_PUBLIC_GOOGLE_CLIENT_ID,
          callback: async (response: any) => {
            setIsSubmitting(true)
            try {
              await onGoogleSignIn(response.credential)
            } catch (error) {
              console.error("Google sign-in failed:", error)
            } finally {
              setIsSubmitting(false)
            }
          },
        })

        google.accounts.id.renderButton(
          googleButtonRef.current,
          { 
            theme: "outline", 
            size: "large",
            width: googleButtonRef.current.offsetWidth,
            text: "continue_with",
          }
        )
      } catch (error) {
        console.error("Error initializing Google Sign-In:", error)
      }
    }

    // Start initialization immediately
    initializeGoogleSignIn()
  }, [onGoogleSignIn])

  const validate = () => {
    const newErrors: Record<string, string> = {}

    if (!formData.email) {
      newErrors.email = "Email is required"
    } else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(formData.email)) {
      newErrors.email = "Please enter a valid email address"
    }

    if (!formData.password) {
      newErrors.password = "Password is required"
    } else if (formData.password.length < 8) {
      newErrors.password = "Password must be at least 8 characters"
    }

    if (formData.password !== formData.confirm_password) {
      newErrors.confirm_password = "Passwords do not match"
    }

    setErrors(newErrors)
    return Object.keys(newErrors).length === 0
  }

  const handleSubmit = async () => {
    if (validate()) {
      setIsSubmitting(true)
      try {
        await onSubmit({
          email: formData.email,
          password: formData.password,
        })
      } finally {
        setIsSubmitting(false)
      }
    }
  }

  const handleChange = (name: string, value: string) => {
    setFormData((prev) => ({ ...prev, [name]: value }))
    if (errors[name]) {
      setErrors((prev) => ({ ...prev, [name]: "" }))
    }
  }

  const togglePasswordVisibility = (field: keyof typeof showPasswords) => {
    setShowPasswords((prev) => ({ ...prev, [field]: !prev[field] }))
  }

  const handleGoogleSignIn = () => {
    // This is now handled by the renderButton callback
  }

  return (
    <div className="min-h-screen bg-gray-100 flex flex-col justify-center py-12 sm:px-6 lg:px-8">
      <div className="sm:mx-auto sm:w-full sm:max-w-md">
        {/* Logo */}
        <div className="flex justify-center mb-8">
          <div className="flex items-center">
            <Image src="/logo.svg" alt="GENSIGHTS" width={120} height={120} className="mr-2" />
          </div>
        </div>

        {/* Form Container */}
        <div className="bg-white py-12 px-8 shadow-lg rounded-lg border border-gray-200">
          <div className="space-y-6">
            {/* Google Sign In Button - Rendered by Google */}
            <div ref={googleButtonRef} className="w-full"></div>

            {/* Divider */}
            <div className="relative">
              <div className="absolute inset-0 flex items-center">
                <div className="w-full border-t border-gray-300"></div>
              </div>
              <div className="relative flex justify-center text-sm">
                <span className="px-2 bg-white text-gray-500">Or create password</span>
              </div>
            </div>

            {/* Email */}
            <div>
              <input
                type="email"
                className={`block w-full px-4 py-4 border ${
                  errors.email ? "border-red-500" : "border-gray-300"
                } rounded-md placeholder-gray-500 focus:outline-none focus:ring-0 focus:border-gray-400 text-sm`}
                value={formData.email}
                onChange={(e) => handleChange("email", e.target.value)}
                placeholder="Email Address"
              />
              {errors.email && <p className="mt-1 text-sm text-red-600">{errors.email}</p>}
            </div>

            {/* New Password */}
            <div>
              <div className="relative">
                <input
                  type={showPasswords.password ? "text" : "password"}
                  className={`block w-full px-4 py-4 border ${
                    errors.password ? "border-red-500" : "border-gray-300"
                  } rounded-md placeholder-gray-500 focus:outline-none focus:ring-0 focus:border-gray-400 text-sm`}
                  value={formData.password}
                  onChange={(e) => handleChange("password", e.target.value)}
                  placeholder="New Password"
                />
                <button
                  type="button"
                  className="absolute inset-y-0 right-0 pr-4 flex items-center"
                  onClick={() => togglePasswordVisibility("password")}
                >
                  {showPasswords.password ? (
                    <EyeOff className="h-5 w-5 text-gray-400" />
                  ) : (
                    <Eye className="h-5 w-5 text-gray-400" />
                  )}
                </button>
              </div>
              {errors.password && <p className="mt-1 text-sm text-red-600">{errors.password}</p>}
            </div>

            {/* Confirm Password */}
            <div>
              <div className="relative">
                <input
                  type={showPasswords.confirm_password ? "text" : "password"}
                  className={`block w-full px-4 py-4 border ${
                    errors.confirm_password ? "border-red-500" : "border-gray-300"
                  } rounded-md placeholder-gray-500 focus:outline-none focus:ring-0 focus:border-gray-400 text-sm`}
                  value={formData.confirm_password}
                  onChange={(e) => handleChange("confirm_password", e.target.value)}
                  placeholder="Confirm Password"
                />
                <button
                  type="button"
                  className="absolute inset-y-0 right-0 pr-4 flex items-center"
                  onClick={() => togglePasswordVisibility("confirm_password")}
                >
                  {showPasswords.confirm_password ? (
                    <EyeOff className="h-5 w-5 text-gray-400" />
                  ) : (
                    <Eye className="h-5 w-5 text-gray-400" />
                  )}
                </button>
              </div>
              {errors.confirm_password && <p className="mt-1 text-sm text-red-600">{errors.confirm_password}</p>}
            </div>

            {/* Submit Button */}
            <div className="pt-2">
              <button
                type="button"
                disabled={isSubmitting}
                onClick={handleSubmit}
                className={`w-full flex justify-center py-4 px-4 border border-transparent rounded-md text-sm font-medium text-white bg-blue-800 hover:bg-blue-900 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 transition-colors ${
                  isSubmitting ? "opacity-50 cursor-not-allowed" : ""
                }`}
              >
                {isSubmitting ? "Creating Password..." : "Create Password"}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}