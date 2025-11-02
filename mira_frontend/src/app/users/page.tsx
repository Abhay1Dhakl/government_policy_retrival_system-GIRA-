"use client"

import { useState, useEffect } from "react"
import { useRouter } from "next/navigation"
import PersonalInformationForm from "./components/PersonalInformationForm"
import AdditionalInformationForm from "./components/AdditionalInformationForm"
import { authService } from "@/lib/auth"
import type { PersonalInfo, AdditionalInfo, FormError } from "@/lib/types"

const redirectToLogin = (router: any) => router.push("/login")
const redirectToChat = (router: any) => router.push("/chat")
const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8082/api/v1"
const USER_UPDATE_URL = `${API_BASE_URL}/users/update/`

console.log("API_BASE_URL:", API_BASE_URL)
console.log("User Update URL:", USER_UPDATE_URL)

export default function UsersPage() {
  const router = useRouter()
  const [currentStep, setCurrentStep] = useState(1)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<FormError | null>(null)
  const [personalInfo, setPersonalInfo] = useState<PersonalInfo>({
    firstName: "",
    lastName: "",
    phoneNumber: "",
    email: "",
    address: "",
  })
  const [additionalInfo, setAdditionalInfo] = useState<AdditionalInfo>({
    country: "",
    city: "",
    institution: "",
    zipCode: "",
  })

  useEffect(() => {
    const checkUserAuth = async () => {
      try {
        if (!authService.isAuthenticated()) {
          redirectToLogin(router)
          return
        }

        const hasDetails = localStorage.getItem("has_details") === "true"
        if (hasDetails) {
          console.log("Profile already completed (has_details: true), redirecting to chat")
          redirectToChat(router)
          return
        }

        console.log("Profile not completed (has_details: false), showing form")
        setLoading(false)
      } catch (err) {
        console.error("Error checking authentication:", err)
        redirectToLogin(router)
      }
    }

    checkUserAuth()
  }, [router])

  const handlePersonalInfoSubmit = (data: PersonalInfo) => {
    setPersonalInfo(data)
    setCurrentStep(2)
  }

  const handleAdditionalInfoSubmit = async (data: AdditionalInfo) => {
    setError(null)

    try {
      if (!authService.isAuthenticated()) {
        redirectToLogin(router)
        return
      }

      const updateData = {
        first_name: personalInfo.firstName,
        last_name: personalInfo.lastName,
        phone_number: personalInfo.phoneNumber,
        address: personalInfo.address,
        country: data.country,
        city: data.city,
        institution: data.institution,
        zip_code: data.zipCode,
      }

      console.log("Submitting user profile update")
      const response = await fetch(USER_UPDATE_URL, {
        method: "PATCH",
        headers: {
          ...authService.getAuthHeader(),
          "Content-Type": "application/json",
        },
        body: JSON.stringify(updateData),
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.message || "Failed to update user profile")
      }

      console.log("Profile updated successfully, redirecting to chat")
      localStorage.setItem("has_details", "true")
      redirectToChat(router)
    } catch (err) {
      console.error("Error updating user profile:", err)
      setError({
        message: err instanceof Error ? err.message : "Failed to update profile. Please try again.",
      })
    }
  }

  const handleBack = () => setCurrentStep(1)

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-md mx-auto bg-white rounded-lg shadow-md p-6">
        {error && <div className="mb-4 p-3 bg-red-100 border border-red-400 text-red-700 rounded">{error.message}</div>}
        {currentStep === 1 ? (
          <PersonalInformationForm initialData={personalInfo} onNext={handlePersonalInfoSubmit} />
        ) : (
          <AdditionalInformationForm
            initialData={additionalInfo}
            onNext={handleAdditionalInfoSubmit}
            onBack={handleBack}
          />
        )}
      </div>
    </div>
  )
}
