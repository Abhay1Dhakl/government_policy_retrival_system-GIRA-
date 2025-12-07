"use client"

import { Pause, Play, Square } from "lucide-react"

interface StreamControlsProps {
  isLoading: boolean
  isPaused: boolean
  onPause: () => void
  onResume: () => void
  onStop: () => void
}

export default function StreamControls({
  isLoading,
  isPaused,
  onPause,
  onResume,
  onStop,
}: StreamControlsProps) {
  if (!isLoading) return null

  return (
    <div className="flex items-center gap-2 px-4 py-2 bg-blue-50 dark:bg-blue-900/20 border-b border-blue-200 dark:border-blue-800">
      <div className="flex items-center gap-2 flex-1">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></div>
          <span className="text-sm text-blue-700 dark:text-blue-300 font-medium">
            {isPaused ? "Stream Paused" : "Generating response..."}
          </span>
        </div>
      </div>
      
      <div className="flex items-center gap-2">
        {!isPaused ? (
          <button
            onClick={onPause}
            className="flex items-center gap-1 px-3 py-1.5 text-sm bg-white dark:bg-gray-800 hover:bg-gray-50 dark:hover:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg transition-colors"
            title="Pause streaming"
          >
            <Pause className="w-4 h-4" />
            <span>Pause</span>
          </button>
        ) : (
          <button
            onClick={onResume}
            className="flex items-center gap-1 px-3 py-1.5 text-sm bg-blue-500 hover:bg-blue-600 text-white rounded-lg transition-colors"
            title="Resume streaming"
          >
            <Play className="w-4 h-4" />
            <span>Resume</span>
          </button>
        )}
        
        <button
          onClick={onStop}
          className="flex items-center gap-1 px-3 py-1.5 text-sm bg-red-500 hover:bg-red-600 text-white rounded-lg transition-colors"
          title="Stop generation"
        >
          <Square className="w-4 h-4" />
          <span>Stop</span>
        </button>
      </div>
    </div>
  )
}
