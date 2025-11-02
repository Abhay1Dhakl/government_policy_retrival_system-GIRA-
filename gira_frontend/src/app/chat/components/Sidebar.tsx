"use client"

import { PlusCircle, Search, MessageSquare, Moon, Sun } from "lucide-react"
import { useRouter } from "next/navigation"
import authService from "@/lib/auth"
import { useDarkMode } from "@/context/DarkModeContext"

interface Session {
  page_id: string
  page_title: string
  last_activity: string
  message_count: number
  first_message: string
}

interface SidebarProps {
  chatHistory: Session[]
  onConversationClick?: (pageId: string) => void
  onNewChatClick?: () => void
  backendPageTitles?: Record<string, string>
  selectedPageId?: string
}

const sidebarItems = [
  { label: "New Chat", icon: PlusCircle, color: "text-blue-500" },
  { label: "Search Conversation", icon: Search, color: "text-gray-500" },
]

const formatDistanceToNow = (date: Date): string => {
  const now = new Date()
  const diffInSeconds = Math.floor((now.getTime() - date.getTime()) / 1000)

  if (diffInSeconds < 60) return "just now"
  if (diffInSeconds < 3600) return `${Math.floor(diffInSeconds / 60)}m ago`
  if (diffInSeconds < 86400) return `${Math.floor(diffInSeconds / 3600)}h ago`
  if (diffInSeconds < 2592000) return `${Math.floor(diffInSeconds / 86400)}d ago`
  return `${Math.floor(diffInSeconds / 2592000)}mo ago`
}

export default function Sidebar({ 
  chatHistory, 
  onConversationClick, 
  onNewChatClick, 
  backendPageTitles = {},
  selectedPageId,
}: SidebarProps) {
  const router = useRouter()
  const { isDarkMode, toggleDarkMode } = useDarkMode()

  const handleConversationClick = (pageId: string) => {
    if (onConversationClick) {
      onConversationClick(pageId)
    }
  }

  const handleLogout = () => {
    authService.logout()

    if (typeof window !== "undefined") {
      localStorage.removeItem("current_page_id")
    }

    // Redirect to login
    router.push("/login")
  }

  const getSessionTitle = (session: Session): string => {
    const backendTitle = backendPageTitles[session.page_id]
    if (backendTitle) {
      return backendTitle
    }
    
    const title = session.page_title || session.first_message
    return title
  }

  return (
    <div className="w-72 bg-gray-50 border-r border-gray-200 flex flex-col h-screen dark:bg-gray-900 dark:border-gray-700">
      {/* Top Section */}
      <div className="p-6 space-y-4">
        <h2 className="text-xl font-bold text-gray-800 dark:text-gray-100">MIRA</h2>
        {sidebarItems.map((item, index) => (
          <button
            key={index}
            onClick={() => {
              if (item.label === "New Chat" && onNewChatClick) {
                onNewChatClick()
              }
            }}
            className="w-full flex items-center gap-3 p-3 text-sm text-gray-700 hover:bg-gray-100 rounded-lg transition-colors duration-200"
          >
            <item.icon className={`w-5 h-5 ${item.color}`} />
            <span className="font-medium">{item.label}</span>
          </button>
        ))}
      </div>

      {/* Conversations List */}
      <div className="flex-1 px-6 py-4 overflow-y-auto">
        <h3 className="text-sm font-semibold text-gray-500 mb-3">Past Conversations</h3>
        <div className="space-y-2">
          {chatHistory.length === 0 ? (
            <p className="text-sm text-gray-400">No conversations yet.</p>
          ) : (
            chatHistory.map((session) => (
              <button
                  key={session.page_id}
                  onClick={() => handleConversationClick(session.page_id)}
                  // highlight selected conversation
                  className={`w-full text-left p-3 text-sm rounded-lg transition-colors duration-200 flex items-start gap-3 ${
                    selectedPageId === session.page_id
                      ? 'bg-gray-100 dark:bg-gray-800 text-gray-900 dark:text-gray-100 font-medium'
                      : 'text-gray-600 hover:bg-gray-100'
                  }`}
                  aria-current={selectedPageId === session.page_id ? 'true' : undefined}
                >
                <MessageSquare className="w-5 h-5 text-gray-400 flex-shrink-0 mt-0.5" />
                <div className="flex-1 min-w-0">
                  <p
                    className="font-medium text-gray-800 truncate whitespace-nowrap overflow-hidden"
                    title={getSessionTitle(session)}
                  >
                    {getSessionTitle(session)}
                  </p>
                  <div className="flex items-center justify-between mt-1">
                    <p className="text-xs text-gray-400">
                      {formatDistanceToNow(new Date(session.last_activity))}
                    </p>
                    {session.message_count > 1 && (
                      <span className="text-xs text-gray-500 bg-gray-200 px-2 py-1 rounded-full">
                        {session.message_count} msgs
                      </span>
                    )}
                  </div>
                </div>
              </button>
            ))
          )}
        </div>
      </div>

      {/* Bottom Section */}

      {/* Dark Mode Toggle */}
      <div className="p-2 border-t border-gray-200">
        <button
          onClick={toggleDarkMode}
          className="w-full flex items-center gap-3 px-4 py-2 rounded-lg text-sm font-medium 
                     bg-gray-100 hover:bg-gray-200 dark:bg-gray-800 dark:hover:bg-gray-700 
                     text-gray-700 dark:text-gray-300 transition-colors duration-200"
        >
          {isDarkMode ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
          <span>{isDarkMode ? "Light Mode" : "Dark Mode"}</span>
        </button>
      </div>

      <div className="p-2">
        <button
          onClick={handleLogout}
          className="w-full bg-gray-600 text-white px-4 py-2 rounded-lg text-sm font-medium hover:bg-gray-700 transition-colors duration-200"
        >
          Logout
        </button>
      </div>
    </div>
  )
};