"use client"

import React, { useEffect, useRef, useState } from "react"
import { Copy, Edit, ThumbsUp, ThumbsDown, RotateCcw, TriangleAlert, Check, X } from "lucide-react"
import References from "./References"
import FeedbackModal from "./FeedbackModal"
import authService from "@/lib/auth"

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

interface Message {
  id: string
  content: string
  sender: "user" | "assistant"
  timestamp: Date
  references?: Reference[]
  flagging_value?: string
  conversation_id?: string
  paragraph_feedback?: ("like" | "dislike" | null)[]
  isStreaming?: boolean
}

interface ChatAreaProps {
  messages: Message[]
  isLoading?: boolean
  onReferenceClick: (source: string, pageNumber: number, highlightText: string, targetPageNumber: number, conversationId: string) => void
  onRegenerateResponse?: (conversationId: string) => void
  onEditMessage?: (messageId: string, newContent: string) => void
  pageId?: string
  isDarkMode: boolean;
}

const splitContentIntoParagraphs = (content: string): string[] => {
  return content.split("\n").filter((p) => p.trim() !== "")
}

export default function ChatArea({
  messages,
  isLoading,
  onReferenceClick,
  onRegenerateResponse,
  onEditMessage,
  pageId,
}: ChatAreaProps) {
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const editTextareaRef = useRef<HTMLTextAreaElement>(null)
  const [feedbackModal, setFeedbackModal] = useState<{
    isOpen: boolean
    feedbackType: "like" | "dislike"
    messageId: string
    conversationId?: string
    userQuery: string
    assistantResponse: string
    paragraphKey?: string
  }>({
    isOpen: false,
    feedbackType: "like",
    messageId: "",
    conversationId: "",
    userQuery: "",
    assistantResponse: "",
  })

  const [paragraphFeedback, setParagraphFeedback] = useState<Record<string, "like" | "dislike" | null>>({})
  const [editingMessageId, setEditingMessageId] = useState<string | null>(null)
  const [editingContent, setEditingContent] = useState("")

  // Initialize paragraph feedback from messages (for persistence)
  useEffect(() => {
    const initialFeedback: Record<string, "like" | "dislike" | null> = {}

    messages.forEach((message) => {
      if (message.sender === "assistant" && message.paragraph_feedback) {
        message.paragraph_feedback.forEach((feedback, index) => {
          if (feedback) {
            initialFeedback[`${message.id}-${index}`] = feedback
          }
        })
      }
    })

    setParagraphFeedback(initialFeedback)
  }, [messages])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages, isLoading])

  // Auto-resize edit textarea
  useEffect(() => {
    if (editTextareaRef.current) {
      editTextareaRef.current.style.height = 'auto'
      editTextareaRef.current.style.height = `${editTextareaRef.current.scrollHeight}px`
    }
  }, [editingContent])

  // Find the latest assistant message
  const getLatestAssistantMessage = () => {
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].sender === "assistant") {
        return messages[i]
      }
    }
    return null
  }

  const latestAssistantMessage = getLatestAssistantMessage()
  
  // Check if we should show thinking for the latest assistant message
  const shouldShowThinking = () => {
    if (!latestAssistantMessage) return false
    
    // Show thinking if:
    // 1. The message is empty AND we're loading
    // 2. OR the message is marked as streaming
    const isEmpty = !latestAssistantMessage.content || latestAssistantMessage.content.trim() === ""
    return (isEmpty && isLoading) || latestAssistantMessage.isStreaming
  }

  const handleMessageAction = (action: string, messageId: string, messageContent: string, conversationId?: string) => {
    const messageIndex = messages.findIndex((msg) => msg.id === messageId)
    let userQuery = ""

    for (let i = messageIndex - 1; i >= 0; i--) {
      if (messages[i].sender === "user") {
        userQuery = messages[i].content
        break
      }
    }

    switch (action) {
      case "copy":
        navigator.clipboard.writeText(messageContent)
        break
      case "edit":
        setEditingMessageId(messageId)
        setEditingContent(messageContent)
        break
      case "like":
      case "dislike":
        setFeedbackModal({
          isOpen: true,
          feedbackType: action as "like" | "dislike",
          messageId,
          conversationId: conversationId || "",
          userQuery,
          assistantResponse: messageContent,
        })
        break
      case "regenerate":
        if (conversationId && onRegenerateResponse) {
          onRegenerateResponse(conversationId)
        } else {
          console.error("Cannot regenerate: missing conversation_id or callback")
        }
        break
    }
  }

  const handleEditSave = () => {
    if (editingMessageId && editingContent.trim() && onEditMessage) {
      onEditMessage(editingMessageId, editingContent.trim())
    }
    setEditingMessageId(null)
    setEditingContent("")
  }

  const handleEditCancel = () => {
    setEditingMessageId(null)
    setEditingContent("")
  }

  const handleEditKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleEditSave()
    } else if (e.key === 'Escape') {
      handleEditCancel()
    }
  }

  // Handle paragraph feedback with toggle + modal sync
  const handleParagraphFeedback = (
    action: "like" | "dislike",
    messageId: string,
    paragraphIndex: number,
    messageContent: string,
    conversationId?: string
  ) => {
    const key = `${messageId}-${paragraphIndex}`
    const current = paragraphFeedback[key]

    if (current === action) {
      // Deselect
      setParagraphFeedback((prev) => ({ ...prev, [key]: null }))
      setFeedbackModal((prev) => ({ ...prev, isOpen: false }))
    } else {
      // Select new
      setParagraphFeedback((prev) => ({ ...prev, [key]: action }))
      handleMessageAction(action, messageId, messageContent, conversationId)
      setFeedbackModal((prev) => ({ ...prev, paragraphKey: key }))
    }
  }

  const handleFeedbackSubmit = async (reason: string, paragraphKey?: string) => {
    try {
      const [messageId, paragraphIndexStr] = paragraphKey ? paragraphKey.split("-") : []
      const paragraphIndex = paragraphIndexStr ? parseInt(paragraphIndexStr) : undefined
      const feedbackValue = feedbackModal.feedbackType === "like" ? 1 : -1

      // Find the message to get its conversation_id
      const message = messages.find(msg => msg.id === (messageId || feedbackModal.messageId))
      const conversationId = message?.conversation_id || feedbackModal.conversationId

      const feedbackData = {
        conversation_id: conversationId || "",
        user_query: feedbackModal.userQuery,
        assistant_response: feedbackModal.assistantResponse,
        feedback: feedbackValue,
        feedback_reason: reason,
        page_id: pageId || "",
        paragraph_index: paragraphIndex,
      }

      await authService.storeFeedback(feedbackData)
    } catch (error) {
      console.error("Error submitting feedback:", error)
    } finally {
      setFeedbackModal((prev) => ({ ...prev, isOpen: false }))
    }
  }

  const handleFeedbackClose = () => {
    setFeedbackModal((prev) => ({ ...prev, isOpen: false }))
  }

  // Render streaming cursor for the currently streaming message
  const renderStreamingCursor = () => (
    <span className="inline-block w-2 h-4 bg-blue-500 ml-1 animate-pulse"></span>
  )

  // Render thinking indicator
  const renderThinkingIndicator = () => (
    <div className="flex items-center gap-2 text-gray-500">
      <div className="flex space-x-1">
        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: "0.1s" }}></div>
        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: "0.2s" }}></div>
      </div>
      <span className="text-sm">Thinking...</span>
    </div>
  )

  return (
    <>
      <div className="flex-1 overflow-y-auto p-6">
        <div className="max-w-4xl mx-auto space-y-6">
          {messages.length === 0 ? (
            <div className="text-center text-gray-500 mt-20">
              <h2 className="text-xl font-semibold mb-2">Welcome to Chat</h2>
              <p>Start a conversation by typing a message or using voice commands.</p>
            </div>
          ) : (
            messages.map((message) => {
              const paragraphs = splitContentIntoParagraphs(message.content)
              const isEditing = editingMessageId === message.id
              const isStreaming = message.isStreaming
              const isEmpty = !message.content || message.content.trim() === ""
              
              // Show thinking for assistant messages that are empty and either streaming or loading
              const showThinking = message.sender === "assistant" && isEmpty && (isStreaming || (isLoading && message === latestAssistantMessage))

              return (
                <div key={message.id} className={`flex ${message.sender === "user" ? "justify-end" : "justify-start"}`}>
                  <div className="max-w-3xl w-full dark:bg-gray-800">
                    <div
                      className={`p-4 rounded-lg ${
                        message.sender === "user"
                          ? "bg-blue-600 text-white ml-auto max-w-fit"
                          : "bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 border border-gray-200 dark:border-gray-700"
                      }`}
                    >
                      {message.sender === "assistant" && (
                        <div className="flex items-start gap-3">
                          <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center flex-shrink-0">
                            <span className="text-blue-600 text-sm font-semibold">A</span>
                          </div>
                          <div className="flex-1">
                            {message.flagging_value && message.flagging_value.trim() !== "" && (
                              <div className="mb-3 p-3 bg-yellow-50 border-l-4 border-yellow-400 rounded-r-md">
                                <div className="flex">
                                  <div className="flex-shrink-0">
                                    <TriangleAlert className="h-5 w-5 text-yellow-400" />
                                  </div>
                                  <div className="ml-3">
                                    <p className="text-sm text-yellow-700">
                                      <strong>Warning:</strong> {message.flagging_value}
                                    </p>
                                  </div>
                                </div>
                              </div>
                            )}

                            {/* Show thinking indicator for empty assistant messages */}
                            {showThinking && (
                              <div className="prose prose-sm max-w-none">
                                {renderThinkingIndicator()}
                              </div>
                            )}

                            {/* Show the actual content when it exists */}
                            {!isEmpty && (
                              <div className="prose prose-sm max-w-none">
                                {paragraphs.map((paragraph, index) => {
                                  const key = `${message.id}-${index}`
                                  const feedback = paragraphFeedback[key]
                                  const isLastParagraph = index === paragraphs.length - 1

                                  const renderParagraphWithCitations = () => {
                                    const text = paragraph.trim()
                                    const citationRegex = /\[(\d+)\.(\d+)\]/g
                                    const nodes: (string | React.ReactNode)[] = []
                                    let lastIndex = 0
                                    let match: RegExpExecArray | null
                                    
                                    // Group references by document (source)
                                    const refsByDoc: { [key: string]: Reference[] } = {}
                                    if (message.references) {
                                      message.references.forEach(ref => {
                                        if (!refsByDoc[ref.source]) {
                                          refsByDoc[ref.source] = []
                                        }
                                        refsByDoc[ref.source].push(ref)
                                      })
                                    }
                                    
                                    // Get unique document sources in order
                                    const docSources = message.references 
                                      ? Array.from(new Set(message.references.map(r => r.source)))
                                      : []

                                    while ((match = citationRegex.exec(text)) !== null) {
                                      const matchStart = match.index
                                      const matchEnd = citationRegex.lastIndex

                                      if (matchStart > lastIndex) {
                                        nodes.push(text.slice(lastIndex, matchStart))
                                      }

                                      // Parse citation: [docNum.refNum]
                                      const docNum = parseInt(match[1], 10)
                                      const refNum = parseInt(match[2], 10)
                                      
                                      // Get the document source (docNum is 1-indexed)
                                      const docSource = docSources[docNum - 1]
                                      
                                      // Get the reference from that document (refNum is 1-indexed)
                                      if (docSource && refsByDoc[docSource] && refsByDoc[docSource][refNum - 1]) {
                                        const ref = refsByDoc[docSource][refNum - 1]
                                        const btnKey = `${key}-citation-${match[0]}`

                                        // Build a concise snippet around this citation from the answer text
                                        const localStart = matchStart
                                        const localEnd = matchEnd
                                        const SNIPPET_PAD = 160
                                        const leftBound = Math.max(0, localStart - SNIPPET_PAD)
                                        const rightBound = Math.min(text.length, localEnd + SNIPPET_PAD)
                                        const windowText = text.slice(leftBound, rightBound)
                                        const localIdx = localStart - leftBound
                                        const prevStop = Math.max(
                                          windowText.lastIndexOf('.', localIdx),
                                          windowText.lastIndexOf('\n', localIdx)
                                        )
                                        const nextStopCandidates = [
                                          windowText.indexOf('.', localIdx),
                                          windowText.indexOf('\n', localIdx),
                                        ].filter((v) => v !== -1)
                                        const nextStop = nextStopCandidates.length ? Math.min(...nextStopCandidates) : windowText.length
                                        let snippet = windowText.slice(prevStop >= 0 ? prevStop + 1 : 0, nextStop).trim()
                                        if (snippet.length > 220) snippet = snippet.slice(0, 220) + '...'

                                        nodes.push(
                                          <button
                                            key={btnKey}
                                            type="button"
                                            onClick={() =>
                                              onReferenceClick(
                                                ref.source,
                                                ref.page_number,
                                                (ref as any).answer_snippet || snippet || ref.original_text || ref.text || "",
                                                ref.page_number,
                                                message.conversation_id || ""
                                              )
                                            }
                                            className="text-blue-600 underline text-sm ml-1 mr-1"
                                            title={`Open reference ${match[0]}`}
                                          >
                                            {match[0]}
                                          </button>,
                                        )
                                      } else {
                                        nodes.push(match[0])
                                      }

                                      lastIndex = matchEnd
                                    }

                                    if (lastIndex < text.length) {
                                      nodes.push(text.slice(lastIndex))
                                    }

                                    if (nodes.length === 0) {
                                      return (
                                        <>
                                          {text}
                                          {isStreaming && isLastParagraph && renderStreamingCursor()}
                                        </>
                                      )
                                    }

                                    return (
                                      <>
                                        {nodes.map((n, i) => 
                                          typeof n === "string" ? <span key={`${key}-t-${i}`}>{n}</span> : n
                                        )}
                                        {isStreaming && isLastParagraph && renderStreamingCursor()}
                                      </>
                                    )
                                  }

                                  return (
                                    <div key={index} className={index > 0 ? "mt-3" : ""}>
                                      <p>{renderParagraphWithCitations()}</p>
                                      {!isStreaming && (
                                        <div className="flex items-center gap-1 mt-1">
                                          <button
                                            onClick={() =>
                                              handleParagraphFeedback(
                                                "like",
                                                message.id,
                                                index,
                                                message.content,
                                                message.conversation_id
                                              )
                                            }
                                            className={`p-1 rounded-md transition-colors ${
                                              feedback === "like"
                                                ? "text-blue-600 bg-blue-50"
                                                : "text-gray-400 hover:text-blue-600 hover:bg-blue-50"
                                            }`}
                                            title="Like this paragraph"
                                          >
                                            <ThumbsUp className="h-3 w-3" />
                                          </button>
                                          <button
                                            onClick={() =>
                                              handleParagraphFeedback(
                                                "dislike",
                                                message.id,
                                                index,
                                                message.content,
                                                message.conversation_id
                                              )
                                            }
                                            className={`p-1 rounded-md transition-colors ${
                                              feedback === "dislike"
                                                ? "text-red-600 bg-red-50"
                                                : "text-gray-400 hover:text-red-600 hover:bg-red-50"
                                            }`}
                                            title="Dislike this paragraph"
                                          >
                                            <ThumbsDown className="h-3 w-3" />
                                          </button>
                                        </div>
                                      )}
                                    </div>
                                  )
                                })}
                              </div>
                            )}

                            {message.references && message.references.length > 0 && (
                              <References
                                references={message.references}
                                onReferenceClick={(source, pageNumber, originalText) => {
                                  onReferenceClick(
                                    source,
                                    pageNumber,
                                    originalText,
                                    pageNumber,
                                    message.conversation_id || ""
                                  )
                                }}
                                isStreaming={isStreaming}
                              />
                            )}
                          </div>
                        </div>
                      )}

                      {message.sender === "user" && (
                        <div className="text-right">
                          <div className="inline-flex items-center gap-2">
                            <span className="text-sm">👤</span>
                            <div className="text-left">
                              {isEditing ? (
                                <div className="bg-white text-black rounded-lg p-3 border border-gray-200 min-w-[300px]">
                                  <textarea
                                    ref={editTextareaRef}
                                    value={editingContent}
                                    onChange={(e) => setEditingContent(e.target.value)}
                                    onKeyDown={handleEditKeyDown}
                                    className="w-full resize-none border-none outline-none bg-transparent text-gray-900 min-h-[40px] max-h-32"
                                    placeholder="Edit your message..."
                                    autoFocus
                                  />
                                  <div className="flex justify-end gap-2 mt-2 pt-2 border-t border-gray-100">
                                    <button
                                      onClick={handleEditCancel}
                                      className="p-1.5 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-md transition-colors"
                                      title="Cancel edit"
                                    >
                                      <X className="h-4 w-4" />
                                    </button>
                                    <button
                                      onClick={handleEditSave}
                                      disabled={!editingContent.trim()}
                                      className={`p-1.5 rounded-md transition-colors ${
                                        editingContent.trim()
                                          ? 'text-green-600 hover:text-green-700 hover:bg-green-50'
                                          : 'text-gray-300 cursor-not-allowed'
                                      }`}
                                      title="Save edit"
                                    >
                                      <Check className="h-4 w-4" />
                                    </button>
                                  </div>
                                </div>
                              ) : (
                                paragraphs.map((paragraph, index) => (
                                  <p key={index} className={index > 0 ? "mt-3" : ""}>
                                    {paragraph.trim()}
                                  </p>
                                ))
                              )}
                            </div>
                          </div>
                        </div>
                      )}
                    </div>

                    {/* Action buttons - only show for non-empty, non-streaming messages */}
                    {message.sender === "assistant" && !isEditing && !isEmpty && !isStreaming && (
                      <div className="flex items-center gap-1 mt-2 ml-11">
                        <button
                          onClick={() => handleMessageAction("copy", message.id, message.content)}
                          className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-md transition-colors"
                          title="Copy message"
                        >
                          <Copy className="h-4 w-4" />
                        </button>
                        <button
                          onClick={() => handleMessageAction("edit", message.id, message.content)}
                          className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-md transition-colors"
                          title="Edit message"
                        >
                          <Edit className="h-4 w-4" />
                        </button>
                        {message.conversation_id && (
                          <button
                            onClick={() =>
                              handleMessageAction("regenerate", message.id, message.content, message.conversation_id)
                            }
                            className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-md transition-colors"
                            title="Regenerate response"
                          >
                            <RotateCcw className="h-4 w-4" />
                          </button>
                        )}
                      </div>
                    )}

                    {message.sender === "user" && !isEditing && (
                      <div className="flex items-center gap-1 mt-2 justify-end mr-11">
                        <button
                          onClick={() => handleMessageAction("copy", message.id, message.content)}
                          className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-md transition-colors"
                          title="Copy message"
                        >
                          <Copy className="h-4 w-4" />
                        </button>
                        <button
                          onClick={() => handleMessageAction("edit", message.id, message.content)}
                          className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-md transition-colors"
                          title="Edit message"
                        >
                          <Edit className="h-4 w-4" />
                        </button>
                      </div>
                    )}
                  </div>
                </div>
              )
            })
          )}

          <div ref={messagesEndRef} />
        </div>
      </div>

      <FeedbackModal
        isOpen={feedbackModal.isOpen}
        onClose={handleFeedbackClose}
        feedbackType={feedbackModal.feedbackType}
        onSubmit={(reason) => handleFeedbackSubmit(reason, feedbackModal.paragraphKey)}
      />
    </>
  )
}
