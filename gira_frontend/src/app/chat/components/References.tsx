"use client"
import { ExternalLink, ChevronDown, ChevronUp } from "lucide-react"
import { useState } from "react"

interface Reference {
  page_number: number
  text: string
  source: string
  answer_segment?: string
  original_text?: string
  answer_snippet?: string
  chunk_index?: number
  reference_number?: string
}

interface ReferencesProps {
  references: Reference[]
  onReferenceClick: (source: string, pageNumber: number, highlightText: string, targetPageNumber: number) => void
  isStreaming?: boolean
}

export default function References({ references, onReferenceClick, isStreaming = false }: ReferencesProps) {
  const [isOpen, setIsOpen] = useState(false)
  const [expandedIndexes, setExpandedIndexes] = useState<Record<number, boolean>>({})

  // Show thinking state when streaming is active but no references yet
  if (isStreaming && (!references || references.length === 0)) {
    return (
      <div className="mt-4 pt-4 border-t border-gray-200">
        <button
          onClick={() => setIsOpen(!isOpen)}
          className="flex items-center justify-between w-full text-left group"
        >
          <h4 className="text-sm font-semibold text-gray-700">
            References
          </h4>
          {isOpen ? (
            <ChevronUp className="h-4 w-4 text-gray-500 group-hover:text-gray-700" />
          ) : (
            <ChevronDown className="h-4 w-4 text-gray-500 group-hover:text-gray-700" />
          )}
        </button>
        
        {isOpen && (
          <div className="space-y-2 mt-2">
            <div className="p-4 bg-gray-50 rounded-lg border border-gray-200 text-center">
              <div className="flex items-center justify-center gap-2">
                <div className="animate-pulse flex space-x-1">
                  <div className="h-2 w-2 bg-gray-400 rounded-full animate-bounce"></div>
                  <div className="h-2 w-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                  <div className="h-2 w-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                </div>
                <span className="text-sm text-gray-500">Thinking...</span>
              </div>
            </div>
          </div>
        )}
      </div>
    )
  }

  // Don't show anything if no references and not streaming
  if (!references || references.length === 0) {
    return null
  }

  return (
    <div className="mt-4 pt-4 border-t border-gray-200">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center justify-between w-full text-left group"
      >
        <h4 className="text-sm font-semibold text-gray-700">
          References ({references.length})
        </h4>
        {isOpen ? (
          <ChevronUp className="h-4 w-4 text-gray-500 group-hover:text-gray-700" />
        ) : (
          <ChevronDown className="h-4 w-4 text-gray-500 group-hover:text-gray-700" />
        )}
      </button>
      
      {isOpen && (
        <div className="space-y-2 mt-2">
          {references.map((reference, index) => {
            const textToShow = reference.original_text || reference.text
            const textToHighlight = reference.original_text || reference.text
            const citationLabel =
              reference.reference_number ??
              (reference.chunk_index !== undefined
                ? `${reference.page_number}.${reference.chunk_index}`
                : `${reference.page_number}`)
            const isExpanded = !!expandedIndexes[index]

            return (
              <div
                key={index}
                className="p-2 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors border border-gray-200"
              >
                <div className="flex items-start justify-between gap-2">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <button
                        aria-expanded={isExpanded}
                        aria-controls={`ref-details-${index}`}
                        onClick={(e) => {
                          e.stopPropagation()
                          setExpandedIndexes((prev) => ({ ...prev, [index]: !prev[index] }))
                        }}
                        className="inline-flex items-center gap-2 px-2 py-0.5 rounded-md hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-offset-1 focus:ring-blue-300"
                        title={isExpanded ? "Hide details" : "Show details"}
                      >
                        <span className="text-xs font-medium text-blue-600">{citationLabel}</span>
                        {isExpanded ? (
                          <ChevronUp className="h-3 w-3 text-gray-500" />
                        ) : (
                          <ChevronDown className="h-3 w-3 text-gray-500" />
                        )}
                      </button>

                      <div className="flex items-center gap-2">
                        <span className="text-xs font-medium text-blue-600">{reference.source}</span>
                        <span className="text-xs text-gray-500">Page {reference.page_number}</span>
                        {reference.chunk_index !== undefined && (
                          <span className="text-xs text-gray-500">Chunk {reference.chunk_index}</span>
                        )}
                      </div>
                    </div>

                    <div
                      id={`ref-details-${index}`}
                      className={`text-xs text-gray-600 ${isExpanded ? "" : "line-clamp-2"}`}
                    >
                      <p className="mb-1">
                        {isExpanded ? textToShow : `${textToShow.substring(0, 150)}${textToShow.length > 150 ? "..." : ""}`}
                      </p>

                      {isExpanded && reference.answer_segment && (
                        <p className="text-xs text-gray-500 italic">Answer segment: {reference.answer_segment}</p>
                      )}

                      {isExpanded && reference.original_text && (
                        <p className="text-xs text-gray-500 mt-1">Original: {reference.original_text}</p>
                      )}
                    </div>
                  </div>

                  <div className="flex-shrink-0 flex items-center gap-2">
                    <button
                      onClick={() => onReferenceClick(reference.source, reference.page_number, textToHighlight, reference.page_number)}
                      className="p-1 rounded hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-offset-1 focus:ring-blue-300"
                      title="Open reference"
                    >
                      <ExternalLink className="h-4 w-4 text-gray-400" />
                    </button>
                  </div>
                </div>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
};
