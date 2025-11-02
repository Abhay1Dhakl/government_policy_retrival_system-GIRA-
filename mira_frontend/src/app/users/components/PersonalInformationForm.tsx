"use client"

import type React from "react"

import { useState } from "react"
import type { PersonalInfo } from "@/lib/types"
import FormInput from "./FormInput"
import NavigationButtons from "./NavigationButtons"
import StepIndicator from "./StepIndicator"

interface PersonalInformationFormProps {
  initialData: PersonalInfo
  onNext: (data: PersonalInfo) => void
}

export default function PersonalInformationForm({ initialData, onNext }: PersonalInformationFormProps) {
  const [formData, setFormData] = useState<PersonalInfo>(initialData)

  const handleInputChange = (field: keyof PersonalInfo, value: string) => {
    setFormData((prev) => ({ ...prev, [field]: value }))
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    onNext(formData)
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      <StepIndicator currentStep={1} />

      <div>
        <h2 className="text-2xl font-semibold text-gray-900 mb-8">Personal Information</h2>

        <div className="space-y-6">
          <FormInput
            label="First Name"
            type="text"
            value={formData.firstName}
            onChange={(value) => handleInputChange("firstName", value)}
            placeholder="Enter first name"
            required
          />

          <FormInput
            label="Last Name"
            type="text"
            value={formData.lastName}
            onChange={(value) => handleInputChange("lastName", value)}
            placeholder="Enter last name"
            required
          />

          <FormInput
            label="Phone Number"
            type="tel"
            value={formData.phoneNumber}
            onChange={(value) => handleInputChange("phoneNumber", value)}
            placeholder="(__)-___-____"
            required
          />

          <FormInput
            label="Email"
            type="email"
            value={formData.email}
            onChange={(value) => handleInputChange("email", value)}
            placeholder="Enter email"
            required
          />

          <FormInput
            label="Address"
            type="text"
            value={formData.address}
            onChange={(value) => handleInputChange("address", value)}
            placeholder="Your address"
            required
          />
        </div>
      </div>

      <NavigationButtons showBack={false} nextText="Next" onNext={() => onNext(formData)} type="submit" />
    </form>
  )
}
