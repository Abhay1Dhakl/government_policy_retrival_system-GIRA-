"use client"

import { useState } from "react"
import { X, ArrowLeft } from "lucide-react"
import { Button } from "@/components/ui/Button"
import { Textarea } from "@/components/ui/TextArea"

interface FeedbackModalProps {
  isOpen: boolean
  onClose: () => void
  feedbackType: "like" | "dislike"
  onSubmit: (reason: string) => void
}

const likeOptions = ["Up to date", "Accurate", "Helpful", "Followed instructions", "Good sources", "Other..."]

const dislikeOptions = ["Out of date", "Inaccurate", "Wrong sources", "Too long", "Too short", "Other..."]

export default function FeedbackModal({ isOpen, onClose, feedbackType, onSubmit }: FeedbackModalProps) {
  const [showTextInput, setShowTextInput] = useState(false)
  const [customFeedback, setCustomFeedback] = useState("")

  if (!isOpen) return null

  const options = feedbackType === "like" ? likeOptions : dislikeOptions
  const title =
    feedbackType === "like"
      ? "What did you like about this response?"
      : "What didn't you like about this response?"

  const handleOptionClick = (option: string) => {
    if (option === "Other...") {
      setShowTextInput(true)
    } else {
      onSubmit(option)
      onClose()
    }
  }

  const handleCustomSubmit = () => {
    onSubmit(customFeedback.trim() || "Other")
    onClose()
    setCustomFeedback("")
    setShowTextInput(false)
  }

  const handleBack = () => {
    setShowTextInput(false)
    setCustomFeedback("")
  }

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 w-full max-w-2xl mx-4">
        <div className="flex items-center justify-between mb-6">
          {showTextInput && (
            <Button variant="ghost" size="sm" onClick={handleBack} className="p-1">
              <ArrowLeft className="h-4 w-4" />
            </Button>
          )}
          <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100 flex-1">
            {title}
          </h3>
          <Button variant="ghost" size="sm" onClick={onClose} className="p-1">
            <X className="h-4 w-4" />
          </Button>
        </div>

        {showTextInput ? (
          <div className="space-y-4">
            <Textarea
              placeholder="Add specific details"
              value={customFeedback}
              onChange={(e) => setCustomFeedback(e.target.value)}
              className="min-h-[100px] resize-none"
            />
            <div className="flex justify-end">
              <Button onClick={handleCustomSubmit} className="bg-blue-600 hover:bg-blue-700 text-white">
                Submit
              </Button>
            </div>
          </div>
        ) : (
          <div className="grid grid-cols-3 gap-3">
            {options.map((option) => (
              <Button
                key={option}
                variant="outline"
                onClick={() => handleOptionClick(option)}
                className="rounded-full px-4 py-2 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700 border-gray-300 dark:border-gray-600 h-auto whitespace-nowrap"
              >
                {option}
              </Button>
            ))}
          </div>
        )}
      </div>
    </div>
  )
};