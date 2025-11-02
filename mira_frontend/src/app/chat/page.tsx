"use client"

import { useState, useEffect, useRef } from "react"
import Sidebar from "./components/Sidebar"
import ChatArea from "./components/ChatArea"
import MessageInput from "./components/MessageInput"
import PdfViewerClient from "./components/PdfViewer"
import authService from "@/lib/auth"
import PlusMenu from "./components/PlusMenu"
import { documentTypeOptions, llmOptions } from "./utils/types"
import { DarkModeProvider, useDarkMode } from "@/context/DarkModeContext"

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
}

interface PdfViewerState {
  isOpen: boolean
  pdfUrl: string
  pageNumber: number
  highlightText: string
  targetPageNumber: number
}

interface Conversation {
  page_id: string
  page_title: string
  last_activity: string
  message_count: number
  first_message: string
}

export default function ChatPage() {
  const { isDarkMode } = useDarkMode()
  const [pageId, setPageId] = useState<string>("")
  const [pageTitle, setPageTitle] = useState<string>("")
  const [messages, setMessages] = useState<Message[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [pdfViewer, setPdfViewer] = useState<PdfViewerState>({
    isOpen: false,
    pdfUrl: "",
    pageNumber: 1,
    highlightText: "",
    targetPageNumber: 1,
  })
  const [chatHistory, setChatHistory] = useState<Conversation[]>([])
  const [selectedTools, setSelectedTools] = useState<string[]>([])
  const [selectedLlm, setSelectedLlm] = useState<string>(() => {
    const defaultLlm = llmOptions.find((option) => option.selected)
    return defaultLlm?.apiName || "openai"
  })
  const [isPlusMenuOpen, setIsPlusMenuOpen] = useState(false)
  const [hasUserSentMessage, setHasUserSentMessage] = useState(false)
  const [documentToggleStates, setDocumentToggleStates] = useState<Record<string, boolean>>(() => {
    return documentTypeOptions.reduce(
      (acc, option) => ({
        ...acc,
        [option.id]: option.enabled,
      }),
      {} as Record<string, boolean>,
    )
  })

  const plusButtonRef = useRef<HTMLButtonElement>(null)

  const extractJSONFromMarkdown = (text: string): any | null => {
    let jsonText = text
    const jsonMatch = text.match(/```json\s*([\s\S]*?)\s*```/)
    if (jsonMatch) {
      jsonText = jsonMatch[1]
    } else if (!text.trim().startsWith("{")) {
      return null
    }

    try {
      jsonText = jsonText
        .trim()
        .replace(/\n/g, " ")
        .replace(/\r/g, " ")
        .replace(/\t/g, " ")
        .replace(/\s+/g, " ")
        .replace(/,(\s*[}\]])/g, "$1")

      return JSON.parse(jsonText)
    } catch (e) {
      console.error("Error parsing JSON:", e)
      console.error("Problematic JSON text:", jsonText.substring(0, 500))
      try {
        const cleanedJson = jsonText.replace(/[\x00-\x1F\x7F]/g, "")
        return JSON.parse(cleanedJson)
      } catch (e2) {
        console.error("Secondary JSON parsing also failed:", e2)
        return null
      }
    }
  }

  // Fetch chat history 
  const fetchChatHistory = async () => {
    try {
      const token = authService.getToken()

      const apiUrl = `${process.env.NEXT_PUBLIC_CHAT_API_BASE_URL}/pages/get_user_chat_sessions`
      const response = await fetch(apiUrl, {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
      })

      if (response.ok) {
        const contentType = response.headers.get("content-type")
        if (!contentType || !contentType.includes("application/json")) {
          return
        }

        const data = await response.json()

        if (data.sessions && Array.isArray(data.sessions)) {
          const sessions = (data.sessions as Conversation[])
            .map((session: Conversation) => {
              const effectiveTitle = backendPageTitles[session.page_id] || session.page_title
              return {
                ...session,
                page_title: effectiveTitle
              }
            })
            .sort((a: Conversation, b: Conversation) =>
              new Date(b.last_activity).getTime() - new Date(a.last_activity).getTime())

          setChatHistory(sessions)
        } else {
          setChatHistory([])
        }
      } else {
        const contentType = response.headers.get("content-type")
        if (contentType && contentType.includes("application/json")) {
          const errorData = await response.json()

        } else {
          const textResponse = await response.text()
        }
      }
    } catch (error) {
      console.error("Error fetching chat history:", error)
      if (error instanceof SyntaxError && error.message.includes("Unexpected token")) {
        console.error(
          `- Check if ${process.env.NEXT_PUBLIC_CHAT_API_BASE_URL}/pages/get_user_chat_sessions is accessible`,
        )
      }
    }
  }

  useEffect(() => {
    if (!pageId) {
      setMessages([])
    }
  }, [pageId])

  useEffect(() => {
    fetchChatHistory()
  }, [])

  useEffect(() => {
    const activeTools = documentTypeOptions
      .filter((option) => documentToggleStates[option.id])
      .map((option) => option.toolName)
    setSelectedTools(activeTools)
  }, [documentToggleStates])

  useEffect(() => {
    if (pageId) {
      localStorage.setItem("current_page_id", pageId)
    }
  }, [pageId])

  const handlePlusMenuSelect = (option: string, type?: "document" | "llm") => {
    if (type === "document") {
      const toolsArray = option.split(",").filter((tool) => tool.trim() !== "")
      setSelectedTools(toolsArray)

      const newToggleStates = documentTypeOptions.reduce(
        (acc, docOption) => ({
          ...acc,
          [docOption.id]: toolsArray.includes(docOption.toolName),
        }),
        {} as Record<string, boolean>,
      )
      setDocumentToggleStates(newToggleStates)

      console.log("Selected document types:", toolsArray)
    } else if (type === "llm") {
      setSelectedLlm(option)
      console.log("Selected LLM:", option)
    }
    setIsPlusMenuOpen(false)
  }

  const handlePlusMenuClose = () => {
    setIsPlusMenuOpen(false)
  }

  const containsSensitiveInfo = (text: string): boolean => {
    const lower = text.toLowerCase()
    const keywords = [
      "passport",
      "citizenship number",
      "social security",
      "ssn",
      "aadhaar",
      "aadhar",
      "national id",
      "id number",
      "driver's license",
      "driving licence",
      "credit card",
      "debit card",
      "cvv",
      "bank account",
      "account number",
      "routing number",
      "iban",
      "swift",
      "pan number",
      "pan card",
    ]
    if (keywords.some((k) => lower.includes(k))) return true

    const nameWithTitlePattern =
      /\b(Dr\.?|Mr\.?|Mrs\.?|Ms\.?|Prof\.?|Professor|Doctor)\s+[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?\b/
    if (nameWithTitlePattern.test(text)) return true

    const ssnPattern = /\b\d{3}-\d{2}-\d{4}\b/
    const ccPattern = /\b(?:\d{4}[ -]?){3}\d{4}\b/
    if (ssnPattern.test(text) || ccPattern.test(text)) return true

    return false
  }

  const loadConversation = async (sessionId: string) => {
    try {
      const session = chatHistory.find((sess) => sess.page_id === sessionId)

      if (!session) {
        console.error("Session not found:", sessionId)
        return
      }

      setPageId(session.page_id)
      setPageTitle(session.page_title)
      localStorage.setItem("current_page_id", session.page_id)

      try {
        const token = authService.getToken()
        const apiUrl = `${process.env.NEXT_PUBLIC_CHAT_API_BASE_URL}/pages/get_stored_chat_history?page_id=${session.page_id}`
        const response = await fetch(apiUrl, {
          method: "GET",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${token}`,
          },
        })

        if (response.ok) {
          const contentType = response.headers.get("content-type")
          if (!contentType || !contentType.includes("application/json")) {
            console.error("Expected JSON but got:", contentType)
            const textResponse = await response.text()
            console.error("Response body:", textResponse.substring(0, 500))
            throw new Error("API returned HTML instead of JSON - check if endpoint exists")
          }

          const data = await response.json()
          if (data.conversations && Array.isArray(data.conversations)) {
            const sessionConversations = data.conversations
              .sort((a: any, b: any) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime())

            if (sessionConversations.length > 0) {
              const allMessages: Message[] = []

              for (const conv of sessionConversations) {
                allMessages.push({
                  id: `user_${conv.conversation_id}`,
                  content: conv.user_query,
                  sender: "user",
                  timestamp: new Date(conv.created_at),
                })

                let assistantContent = conv.assistant_response
                let references: Reference[] = []

                const parsedJSON = extractJSONFromMarkdown(conv.assistant_response)
                if (parsedJSON && parsedJSON.answer) {
                  assistantContent = parsedJSON.answer
                  if (parsedJSON.references && Array.isArray(parsedJSON.references)) {
                    references = parsedJSON.references.map((ref: any) => ({
                      page_number: ref.page_number,
                      text: ref.original_text || ref.text || "",
                      source: ref.source,
                      answer_segment: ref.answer_segment,
                      original_text: ref.original_text,
                      answer_snippet: ref.answer_snippet,
                      chunk_index: ref.chunk_index,
                      reference_number: ref.reference_number,
                    }))
                  }
                }

                allMessages.push({
                  id: `assistant_${conv.conversation_id}`,
                  content: assistantContent,
                  sender: "assistant",
                  timestamp: new Date(conv.created_at),
                  references: references.length > 0 ? references : undefined,
                  conversation_id: conv.conversation_id,
                })
              }

              setMessages(allMessages)
              setHasUserSentMessage(true)
              return
            } else {
              console.log("No conversations found for page_id:", session.page_id)
            }
          } else {
            console.log("No conversations array in API response")
          }
        } else {
          const contentType = response.headers.get("content-type")
          if (contentType && contentType.includes("application/json")) {
            const errorData = await response.json()
            console.error("API error:", errorData)
          } else {
            const textResponse = await response.text()
            console.error("Non-JSON error response:", textResponse.substring(0, 500))
          }
        }
      } catch (error) {
        console.error("Error fetching session conversations:", error)
      }
      setMessages([
        {
          id: `user_${session.page_id}`,
          content: session.first_message,
          sender: "user",
          timestamp: new Date(session.last_activity),
        }
      ])

      setHasUserSentMessage(true)
    } catch (error) {
      console.error("Error loading session:", error)
    }
  }

  const startNewChat = () => {
    setPageId("")
    setPageTitle("")
    setMessages([])
    setHasUserSentMessage(false)

    localStorage.removeItem("current_page_id")
  }

  const updateChatHistoryWithNewConversation = (newPageId: string, newPageTitle: string, firstMessage: string) => {
    const newConversation: Conversation = {
      page_id: newPageId,
      page_title: newPageTitle,
      last_activity: new Date().toISOString(),
      message_count: 1,
      first_message: firstMessage
    }

    setChatHistory(prevHistory => {
      const existingIndex = prevHistory.findIndex(conv => conv.page_id === newPageId)

      if (existingIndex >= 0) {
        const updatedHistory = [...prevHistory]
        updatedHistory[existingIndex] = {
          ...updatedHistory[existingIndex],
          page_title: newPageTitle,
          last_activity: new Date().toISOString(),
          message_count: updatedHistory[existingIndex].message_count + 1
        }
        return updatedHistory.sort((a, b) =>
          new Date(b.last_activity).getTime() - new Date(a.last_activity).getTime())
      } else {
        return [newConversation, ...prevHistory]
      }
    })
  }

  const [backendPageTitles, setBackendPageTitles] = useState<Record<string, string>>(() => {
    if (typeof window === "undefined") return {}
    try {
      const raw = localStorage.getItem("backend_page_titles")
      return raw ? JSON.parse(raw) : {}
    } catch (e) {
      console.error("Error reading backend_page_titles from localStorage:", e)
      return {}
    }
  })

  const persistBackendPageTitles = (titles: Record<string, string>) => {
    if (typeof window === "undefined") return
    try {
      localStorage.setItem("backend_page_titles", JSON.stringify(titles))
    } catch (e) {
      console.error("Error saving backend_page_titles to localStorage:", e)
    }
  }

  const updateBackendPageTitle = (id: string, title: string) => {
    setBackendPageTitles((prev) => {
      const next = { ...prev, [id]: title }
      persistBackendPageTitles(next)
      return next
    })
  }

  const handleRegenerateResponse = async (conversationId: string) => {
    try {
      setIsLoading(true)

      const regenerateData = {
        page_id: pageId,
        conversation_id: conversationId,
        tools: selectedTools,
        llm: selectedLlm,
      }

      const response = await authService.regenerateResponse(regenerateData)
      let assistantContent = ""
      let references: Reference[] = []
      let flaggingValue = ""
      const newConversationId = response.conversation_id || conversationId

      if (response.answer && response.references) {
        assistantContent = response.answer
        references = response.references.map((ref: any) => ({
          page_number: ref.page_number,
          text: ref.original_text || ref.text || "",
          source: ref.source,
          answer_segment: ref.answer_segment,
          original_text: ref.original_text,
          chunk_index: ref.chunk_index,
          reference_number: ref.reference_number,
        }))
        flaggingValue = response.flagging_value || ""
      } else if (response.response && typeof response.response === "string") {
        const parsedJSON = extractJSONFromMarkdown(response.response)
        if (parsedJSON && parsedJSON.answer && parsedJSON.references) {
          assistantContent = parsedJSON.answer
          references = parsedJSON.references.map((ref: any) => ({
            page_number: ref.page_number,
            text: ref.original_text || ref.text || "",
            source: ref.source,
            answer_segment: ref.answer_segment,
            original_text: ref.original_text,
            chunk_index: ref.chunk_index,
            reference_number: ref.reference_number,
          }))
          flaggingValue = parsedJSON.flagging_value || ""
        } else {
          assistantContent = response.response
          flaggingValue = response.flagging_value || ""
        }
      } else {
        assistantContent = response.response || response.message || "I couldn't regenerate the response properly."
        flaggingValue = response.flagging_value || ""
      }

      setMessages((prevMessages) => {
        const updatedMessages = prevMessages.map((message) => {
          if (message.conversation_id === conversationId && message.sender === "assistant") {
            return {
              ...message,
              content: assistantContent,
              references: references.length > 0 ? references : undefined,
              flagging_value: flaggingValue,
              conversation_id: newConversationId,
              timestamp: new Date(),
            }
          }
          return message
        })
        return updatedMessages
      })

    } catch (error) {
      console.error("Error regenerating response:", error)
      const errorMessage: Message = {
        id: Date.now().toString(),
        content: `Sorry, I couldn't regenerate the response: ${error instanceof Error ? error.message : "Unknown error"}`,
        sender: "assistant",
        timestamp: new Date(),
      }
      setMessages((prev) => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  const handleSendMessage = async (content: string) => {
    const isFirstMessage = !hasUserSentMessage

    if (!hasUserSentMessage) {
      setHasUserSentMessage(true)
    }

    const newMessage: Message = {
      id: Date.now().toString(),
      content,
      sender: "user",
      timestamp: new Date(),
    }
    setMessages((prev) => [...prev, newMessage])

    if (containsSensitiveInfo(content)) {
      const warningMessage: Message = {
        id: `${Date.now()}_warn`,
        content:
          "Privacy notice: Your message appears to include another person's name or personal information (e.g., passport or ID numbers). Please avoid sharing sensitive personal data about yourself or others.",
        sender: "assistant",
        timestamp: new Date(),
      }
      setMessages((prev) => [...prev, warningMessage])
      return
    }
    setIsLoading(true)

    const assistantMessageId = (Date.now() + 1).toString()
    const initialAssistantMessage: Message = {
      id: assistantMessageId,
      content: "",
      sender: "assistant",
      timestamp: new Date(),
    }
    setMessages((prev) => [...prev, initialAssistantMessage])

    try {
      const token = authService.getToken()
      const payload: any = {
        user_query: content,
        tools: selectedTools,
        llm: selectedLlm,
        stream_type: "token",
        token,
      }
      if (pageId) {
        payload.page_id = pageId
      }

      const response = await fetch(`${process.env.NEXT_PUBLIC_CHAT_API_BASE_URL}/query/stream_query`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify(payload),
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        console.error("API error response:", errorData)
        throw new Error(`HTTP error! status: ${response.status}, details: ${JSON.stringify(errorData)}`)
      }

      if (!response.body) {
        throw new Error("Response body is null")
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ""
      let streamedContent = ""
      let metadata: any = null
      let conversationId = ""
      let finalReferences: Reference[] = []
      let flaggingValue = ""

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split("\n")
        buffer = lines.pop() || ""

        for (const line of lines) {
          if (!line.trim() || !line.startsWith("data: ")) continue

          try {
            const jsonStr = line.slice(6)
            const data = JSON.parse(jsonStr)

            if (data.type === "metadata") {
              metadata = data
              conversationId = data.conversation_id || ""

              if (data.page_id && data.page_title) {
                setPageId(data.page_id)
                setPageTitle(data.page_title)
                updateBackendPageTitle(data.page_id, data.page_title)
                localStorage.setItem("current_page_id", data.page_id)

                if (isFirstMessage) {
                  updateChatHistoryWithNewConversation(data.page_id, data.page_title, content)
                }
              }
            } else if (data.type === "chunk") {
              streamedContent += data.content

              setMessages((prev) =>
                prev.map((msg) =>
                  msg.id === assistantMessageId
                    ? { ...msg, content: streamedContent }
                    : msg
                )
              )
            } else if (data.type === "references") {
              const references = data.references.map((ref: any) => ({
                page_number: ref.page_number,
                text: ref.original_text || ref.text || "",
                source: ref.source,
                answer_segment: ref.answer_segment,
                original_text: ref.original_text,
                answer_snippet: ref.answer_snippet,
                chunk_index: ref.chunk_index,
                reference_number: ref.reference_number,
              }))

              setMessages((prev) =>
                prev.map((msg) =>
                  msg.id === assistantMessageId
                    ? {
                      ...msg,
                      references,
                      conversation_id: conversationId,
                      flagging_value: data.flagging_value || ""
                    }
                    : msg
                )
              )
            } else if (data.type === "final") {
              // Capture references from the final data type
              if (data.references && Array.isArray(data.references)) {
                finalReferences = data.references.map((ref: any) => ({
                  page_number: ref.page_number,
                  text: ref.original_text || ref.text || "",
                  source: ref.source,
                  answer_segment: ref.answer_segment,
                  original_text: ref.original_text,
                  answer_snippet: ref.answer_snippet,
                  chunk_index: ref.chunk_index,
                  reference_number: ref.reference_number,
                }))
                flaggingValue = data.flagging_value || ""

                // Update the message with final references
                setMessages((prev) =>
                  prev.map((msg) =>
                    msg.id === assistantMessageId
                      ? {
                        ...msg,
                        references: finalReferences,
                        conversation_id: conversationId,
                        flagging_value: flaggingValue
                      }
                      : msg
                  )
                )
              }
            } else if (data.type === "error") {
              throw new Error(data.message || "Stream error occurred")
            }
          } catch (parseError) {
            console.error("Error parsing SSE line:", parseError, line)
          }
        }
      }

      // Final update with any references from final type if not already set
      if (finalReferences.length > 0) {
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === assistantMessageId
              ? {
                ...msg,
                references: finalReferences,
                conversation_id: conversationId,
                flagging_value: flaggingValue
              }
              : msg
          )
        )
      }

      if (!streamedContent) {
        throw new Error("No content received from stream")
      }

    } catch (error: unknown) {
      console.error("Error sending message:", error)
      const errorMessageContent = error instanceof Error ? error.message : "Unknown error occurred"

      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === assistantMessageId
            ? {
              ...msg,
              content: `Sorry, I'm having trouble connecting to the server: ${errorMessageContent}. Please try again later.`,
            }
            : msg
        )
      )
    } finally {
      setIsLoading(false)
      if (isFirstMessage) {
        setTimeout(() => {
          fetchChatHistory()
        }, 1000)
      }
    }
  }

  const handleEditMessage = async (messageId: string, newContent: string) => {
    const messageIndex = messages.findIndex(msg => msg.id === messageId)
    if (messageIndex === -1) {
      console.error("Message not found:", messageId)
      return
    }

    const editedMessage = messages[messageIndex]

    if (editedMessage.sender !== 'user') {
      console.error("Can only edit user messages")
      return
    }

    const updatedMessages = [...messages]
    updatedMessages[messageIndex] = {
      ...editedMessage,
      content: newContent,
    }

    const messagesToKeep = updatedMessages.slice(0, messageIndex + 1)
    setMessages(messagesToKeep)

    setIsLoading(true)

    const assistantMessageId = (Date.now() + 1).toString()
    const initialAssistantMessage: Message = {
      id: assistantMessageId,
      content: "",
      sender: "assistant",
      timestamp: new Date(),
    }
    setMessages((prev) => [...prev, initialAssistantMessage])

    try {
      const token = authService.getToken()
      const payload: any = {
        user_query: newContent,
        tools: selectedTools,
        llm: selectedLlm,
        stream_type: "token",
        token,
      }
      if (pageId) {
        payload.page_id = pageId
      }

      const response = await fetch(`${process.env.NEXT_PUBLIC_CHAT_API_BASE_URL}/query/stream_query`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify(payload),
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        console.error("API error response:", errorData)
        throw new Error(`HTTP error! status: ${response.status}, details: ${JSON.stringify(errorData)}`)
      }

      if (!response.body) {
        throw new Error("Response body is null")
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ""
      let streamedContent = ""
      let metadata: any = null
      let conversationId = ""
      let finalReferences: Reference[] = []
      let flaggingValue = ""

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split("\n")
        buffer = lines.pop() || ""

        for (const line of lines) {
          if (!line.trim() || !line.startsWith("data: ")) continue

          try {
            const jsonStr = line.slice(6)
            const data = JSON.parse(jsonStr)

            if (data.type === "metadata") {
              metadata = data
              conversationId = data.conversation_id || ""

              if (data.page_id && data.page_title) {
                setPageId(data.page_id)
                setPageTitle(data.page_title)
                updateBackendPageTitle(data.page_id, data.page_title)
                localStorage.setItem("current_page_id", data.page_id)
              }
            } else if (data.type === "chunk") {
              streamedContent += data.content

              setMessages((prev) =>
                prev.map((msg) =>
                  msg.id === assistantMessageId
                    ? { ...msg, content: streamedContent }
                    : msg
                )
              )
            } else if (data.type === "references") {
              const references = data.references.map((ref: any) => ({
                page_number: ref.page_number,
                text: ref.original_text || ref.text || "",
                source: ref.source,
                answer_segment: ref.answer_segment,
                original_text: ref.original_text,
                chunk_index: ref.chunk_index,
                reference_number: ref.reference_number,
              }))

              setMessages((prev) =>
                prev.map((msg) =>
                  msg.id === assistantMessageId
                    ? {
                      ...msg,
                      references,
                      conversation_id: conversationId,
                      flagging_value: data.flagging_value || ""
                    }
                    : msg
                )
              )
            } else if (data.type === "final") {
              // Capture references from the final data type
              if (data.references && Array.isArray(data.references)) {
                finalReferences = data.references.map((ref: any) => ({
                  page_number: ref.page_number,
                  text: ref.original_text || ref.text || "",
                  source: ref.source,
                  answer_segment: ref.answer_segment,
                  original_text: ref.original_text,
                  chunk_index: ref.chunk_index,
                  reference_number: ref.reference_number,
                }))
                flaggingValue = data.flagging_value || ""

                // Update the message with final references
                setMessages((prev) =>
                  prev.map((msg) =>
                    msg.id === assistantMessageId
                      ? {
                        ...msg,
                        references: finalReferences,
                        conversation_id: conversationId,
                        flagging_value: flaggingValue
                      }
                      : msg
                  )
                )
              }
            } else if (data.type === "error") {
              throw new Error(data.message || "Stream error occurred")
            }
          } catch (parseError) {
            console.error("Error parsing SSE line:", parseError, line)
          }
        }
      }

      // Final update with any references from final type if not already set
      if (finalReferences.length > 0) {
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === assistantMessageId
              ? {
                ...msg,
                references: finalReferences,
                conversation_id: conversationId,
                flagging_value: flaggingValue
              }
              : msg
          )
        )
      }

      if (!streamedContent) {
        throw new Error("No content received from stream")
      }

    } catch (error: unknown) {
      console.error("Error sending edited message:", error)
      const errorMessageContent = error instanceof Error ? error.message : "Unknown error occurred"

      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === assistantMessageId
            ? {
              ...msg,
              content: `Sorry, I'm having trouble processing your edited message: ${errorMessageContent}. Please try again.`,
            }
            : msg
        )
      )
    } finally {
      setIsLoading(false)
    }
  }

  const handleReferenceClick = async (
    source: string,
    pageNumber: number,
    highlightText: string,
    targetPageNumber: number,
    conversationId: string
  ) => {
    try {
      const token = authService.getToken()

      const currentMessage = messages.find((msg) => msg.conversation_id === conversationId)

      if (!currentMessage || !currentMessage.references) {
        console.error("Could not find the message or it has no references:", conversationId)
        return
      }

      const textsToHighlight = currentMessage.references
        .filter((ref) => ref.source === source)
        .flatMap((ref) => {
          const pick = [
            (ref as any).answer_snippet,
            ref.original_text,
            ref.text,
          ].filter((v): v is string => typeof v === 'string' && v.trim().length >= 20)

          // De-duplicate by normalized content
          const seen = new Set<string>()
          const unique = pick.filter((t) => {
            const key = t.toLowerCase().replace(/\s+/g, ' ').trim()
            if (seen.has(key)) return false
            seen.add(key)
            return true
          })

          const pageIdx = Math.max(0, Math.floor(Number(ref.page_number || 1)) - 1)
          return unique.map((t) => ({ text: t, page: pageIdx }))
        })

      if (textsToHighlight.length > 0) {
        let pdfPath = source
        if (!pdfPath.startsWith("/") && !pdfPath.startsWith("http")) {
          pdfPath = `/${pdfPath}`
        }
        const safeSourceName = source.replace(/[^a-zA-Z0-9-_\.]/g, "_")
        const outputFilename = `highlighted_${safeSourceName}`

        const highlightResponse = await fetch(`${process.env.NEXT_PUBLIC_CHAT_API_BASE_URL}/pdf/highlight_pdf_text`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${token}`,
          },
          body: JSON.stringify({
            pdf_path: pdfPath,
            texts_to_highlight: textsToHighlight,
            output_filename: outputFilename,
            auto_cleanup: true,
            cleanup_delay: 3600,
          }),
        })

        if (highlightResponse.ok) {
          const highlightedPdfBlob = await highlightResponse.blob()
          const highlightedPdfUrl = URL.createObjectURL(highlightedPdfBlob)

          setPdfViewer({
            isOpen: true,
            pdfUrl: highlightedPdfUrl,
            pageNumber,
            highlightText,
            targetPageNumber,
          })
          return
        } else {
          console.error("Failed to get highlighted PDF from backend:", highlightResponse.status)
          const errorText = await highlightResponse.text()
          console.error("Error details:", errorText)
        }
      } else {
        console.log("No texts to highlight found for source:", source, "in conversation:", conversationId)
      }
    } catch (error) {
      console.error("Error getting highlighted PDF from backend:", error)
    }

    console.log("No fallback available - backend must provide the PDF")
  }

  const closePdfViewer = () => {
    setPdfViewer((prev) => ({ ...prev, isOpen: false }))
  }

  return (
    <DarkModeProvider>
      <div className="flex h-screen bg-gray-50 dark:bg-gray-900">
        <Sidebar
          chatHistory={chatHistory}
          onConversationClick={loadConversation}
          onNewChatClick={startNewChat}
          backendPageTitles={backendPageTitles}
          selectedPageId={pageId}
        />
        <div className="flex-1 flex dark:text-white dark:border-gray-700">
          <div className={`flex flex-col transition-all duration-300 ${pdfViewer.isOpen ? "w-1/2" : "flex-1"}`}>
            <div className="flex items-center justify-between p-4 border-b border-gray-200">
              <div className="relative">
                {isPlusMenuOpen && (
                  <PlusMenu
                    onSelect={handlePlusMenuSelect}
                    onClose={handlePlusMenuClose}
                    anchorRef={plusButtonRef}
                    documentToggleStates={documentToggleStates}
                    setDocumentToggleStates={setDocumentToggleStates}
                    selectedLlm={selectedLlm}
                    setSelectedLlm={setSelectedLlm}
                  />
                )}
              </div>
            </div>
            <ChatArea
              messages={messages}
              isLoading={isLoading}
              onReferenceClick={handleReferenceClick}
              onRegenerateResponse={handleRegenerateResponse}
              onEditMessage={handleEditMessage}
              pageId={pageId}
              isDarkMode={isDarkMode}
            />
            <MessageInput
              onSendMessage={handleSendMessage}
              isLoading={isLoading}
              selectedLlm={selectedLlm}
              setSelectedLlm={setSelectedLlm}
              documentToggleStates={documentToggleStates}
              setDocumentToggleStates={setDocumentToggleStates}
              onPlusMenuSelect={handlePlusMenuSelect}
            />
          </div>
          {pdfViewer.isOpen && (
            <div className="w-1/2 border-l border-gray-200 transition-all duration-300">
              <PdfViewerClient
                pdfUrl={pdfViewer.pdfUrl}
                pageNumber={pdfViewer.pageNumber}
                highlightText={pdfViewer.highlightText}
                targetPageNumber={pdfViewer.targetPageNumber}
                onClose={closePdfViewer}
              />
            </div>
          )}
        </div>
      </div>
    </DarkModeProvider>
  )
};
