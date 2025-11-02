"use client"

import type React from "react"

import { useState } from "react"
import type { AdditionalInfo } from "@/lib/types"
import StepIndicator from "./StepIndicator"
import FormInput from "./FormInput"
import NavigationButtons from "./NavigationButtons"

interface AdditionalInformationFormProps {
  initialData: AdditionalInfo
  onNext: (data: AdditionalInfo) => void
  onBack: () => void
}

export default function AdditionalInformationForm({ initialData, onNext, onBack }: AdditionalInformationFormProps) {
  const [formData, setFormData] = useState<AdditionalInfo>(initialData)
  const [isSubmitting, setIsSubmitting] = useState(false)

  const handleInputChange = (field: keyof AdditionalInfo, value: string) => {
    setFormData((prev) => ({ ...prev, [field]: value }))
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setIsSubmitting(true)
    try {
      onNext(formData)
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      <StepIndicator currentStep={2} />

      <div>
        <h2 className="text-2xl font-semibold text-gray-900 mb-8">Additional Information</h2>

        <div className="space-y-6">
          <FormInput
            label="Country"
            type="text"
            value={formData.country}
            onChange={(value) => handleInputChange("country", value)}
            placeholder="Enter country name"
            required
          />

          <FormInput
            label="City"
            type="text"
            value={formData.city}
            onChange={(value) => handleInputChange("city", value)}
            placeholder="Enter city name"
            required
          />

          <FormInput
            label="Institution"
            type="text"
            value={formData.institution}
            onChange={(value) => handleInputChange("institution", value)}
            placeholder="Enter institution name"
            required
          />

          <FormInput
            label="Zip Code"
            type="text"
            value={formData.zipCode}
            onChange={(value) => handleInputChange("zipCode", value)}
            placeholder="Zip code"
            required
          />
        </div>
      </div>

      <NavigationButtons
        showBack={true}
        nextText={isSubmitting ? "Saving..." : "Complete Profile"}
        onBack={onBack}
        onNext={() => onNext(formData)}
        type="submit"
      />
    </form>
  )
}
