"use client"
import { useState, useEffect, useCallback } from "react"
import { Worker, Viewer } from "@react-pdf-viewer/core"
import { highlightPlugin, type RenderHighlightTargetProps } from "@react-pdf-viewer/highlight"
import { defaultLayoutPlugin } from "@react-pdf-viewer/default-layout"
import { searchPlugin } from "@react-pdf-viewer/search"
import { pageNavigationPlugin } from "@react-pdf-viewer/page-navigation"

import "@react-pdf-viewer/core/lib/styles/index.css"
import "@react-pdf-viewer/default-layout/lib/styles/index.css"
import "@react-pdf-viewer/highlight/lib/styles/index.css"
import "@react-pdf-viewer/search/lib/styles/index.css"

interface PDFViewerProps {
  pdfUrl: string
  pageNumber?: number
  highlightText?: string
  targetPageNumber?: number
  onClose: () => void
}

export default function PdfViewerClient({ pdfUrl, pageNumber = 1, highlightText, targetPageNumber, onClose }: PDFViewerProps) {
  const [isClient, setIsClient] = useState(false)
  const [documentLoaded, setDocumentLoaded] = useState(false)
  const [currentSearchTerm, setCurrentSearchTerm] = useState("")
  const [searchAttempted, setSearchAttempted] = useState(false)
  const [currentPage, setCurrentPage] = useState(pageNumber)

  useEffect(() => setIsClient(true), [])

  // Plugins
  const pageNavigationPluginInstance = pageNavigationPlugin()

  const renderHighlightTarget = (props: RenderHighlightTargetProps) => (
    <div
      style={{
        background: "rgba(255, 255, 0, 0.6)",
        border: "1px solid rgba(255, 193, 7, 0.8)",
        borderRadius: "2px",
        mixBlendMode: "multiply",
      }}
    />
  )
  const highlightPluginInstance = highlightPlugin({ renderHighlightTarget })
  const searchPluginInstance = searchPlugin()
  const defaultLayoutPluginInstance = defaultLayoutPlugin({ sidebarTabs: () => [] })

const extractSearchTerms = (text: string): string[] => {
  if (!text) return []

  const cleanedText = text
    .replace(/[\r\n]+/g, " ") 
    .replace(/\s+/g, " ")     
    .replace(/[""]/g, '"') 
    .replace(/['']/g, "'") 
    .trim()

  const searchTerms: string[] = []

  if (cleanedText.length <= 300) {
    searchTerms.push(cleanedText)
  }

  if (cleanedText.includes(" ")) {
    const flexibleSpacing = cleanedText.replace(/\s+/g, "\\s+")
    searchTerms.push(flexibleSpacing)
  }

  if (cleanedText.length > 200) {
    const sentences = cleanedText.split(/[.!?]+/).filter(s => s.trim().length > 20)
    for (const sentence of sentences.slice(0, 2)) {
      const trimmedSentence = sentence.trim()
      if (trimmedSentence.length <= 150) {
        searchTerms.push(trimmedSentence)
      }
    }
  }

  const uniqueTerms = [...new Set(searchTerms)]
  return uniqueTerms
}

  const performAutoSearch = useCallback(async () => {
    if (!highlightText || !documentLoaded || searchAttempted) return

    setSearchAttempted(true)

    const searchTerms = extractSearchTerms(highlightText)
    if (searchTerms.length === 0) return

    const { highlight } = searchPluginInstance

    if (targetPageNumber && targetPageNumber > 0) {
      pageNavigationPluginInstance.jumpToPage(targetPageNumber - 1)
      setCurrentPage(targetPageNumber)
      await new Promise(resolve => setTimeout(resolve, 1000))
    }

    const searchOnCurrentPage = async (term: string): Promise<number> => {
      try {
        const allMatches = await highlight(term)
        
        const pageMatches = allMatches.filter(match => 
          match.pageIndex === currentPage - 1
        )
        
        return pageMatches.length
      } catch (error) {
        console.error("[PDFViewer] Search error for term:", term.substring(0, 30), error)
        return 0
      }
    }

    for (let i = 0; i < searchTerms.length; i++) {
      const term = searchTerms[i]
      if (term.length < 4) continue 

      try {
        const matchesCount = await searchOnCurrentPage(term)
        
        if (matchesCount > 0) {
          setCurrentSearchTerm(term)
          return 
        }
      } catch (error) {
        console.error("[PDFViewer] Search error for term:", term.substring(0, 30), error)
      }
            await new Promise(resolve => setTimeout(resolve, 200))
    }
    
  }, [highlightText, documentLoaded, searchAttempted, searchPluginInstance, targetPageNumber, pageNavigationPluginInstance, currentPage])

  useEffect(() => {
    setSearchAttempted(false)
    setCurrentSearchTerm("")
  }, [highlightText])

  useEffect(() => {
    if (documentLoaded && highlightText && !searchAttempted) {
      const timer = setTimeout(() => {
        performAutoSearch()
      }, 1200)

      return () => clearTimeout(timer)
    }
  }, [documentLoaded, highlightText, searchAttempted, performAutoSearch])

  const handleDocumentLoad = useCallback(() => {
    setDocumentLoaded(true)
    setCurrentPage(pageNumber)
    if (pageNumber > 1) {
      setTimeout(() => pageNavigationPluginInstance.jumpToPage(pageNumber - 1), 500)
    }
  }, [pageNumber, pageNavigationPluginInstance])

  if (!isClient) return <div>Loading PDF...</div>

  return (
    <div className="h-full flex flex-col bg-white">
      <div className="flex items-center justify-between p-4 border-b bg-gray-50">
        <h3 className="font-semibold text-gray-900">PDF Viewer</h3>
        <button
          onClick={onClose}
          className="px-3 py-1 text-sm bg-gray-200 hover:bg-gray-300 rounded transition-colors"
        >
          Close
        </button>
      </div>

      {currentSearchTerm && (
        <div className="px-4 py-2 bg-yellow-100 border-b text-sm">
          <span className="text-yellow-800">
            Highlighted on page {currentPage}: "{currentSearchTerm.substring(0, 100)}..."
          </span>
        </div>
      )}


      <div className="flex-1 overflow-hidden">
        <Worker workerUrl="https://unpkg.com/pdfjs-dist@3.11.174/build/pdf.worker.min.js">
          <Viewer
            fileUrl={pdfUrl}
            plugins={[searchPluginInstance, highlightPluginInstance, defaultLayoutPluginInstance, pageNavigationPluginInstance]}
            initialPage={pageNumber - 1}
            onDocumentLoad={handleDocumentLoad}
            theme="light"
          />
        </Worker>
      </div>
    </div>
  )
}