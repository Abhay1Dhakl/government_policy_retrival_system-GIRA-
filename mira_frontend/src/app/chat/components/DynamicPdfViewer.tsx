import dynamic from 'next/dynamic'

const DynamicPDFViewer = dynamic(
  () => import('./PdfViewer'),
  { 
    ssr: false,
    loading: () => (
      <div className="h-full flex flex-col bg-white">
        <div className="flex items-center justify-between p-4 border-b bg-gray-50">
          <h3 className="font-semibold text-gray-900">Loading PDF Viewer...</h3>
        </div>
        <div className="flex-1 flex items-center justify-center">
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 border-2 border-blue-600 border-t-transparent rounded-full animate-spin"></div>
            <span className="text-gray-600">Loading PDF...</span>
          </div>
        </div>
      </div>
    )
  }
)

interface PDFViewerProps {
  pdfUrl: string
  pageNumber?: number
  highlightText?: string
  onClose: () => void
}

export default function PDFViewer(props: PDFViewerProps) {
  return <DynamicPDFViewer {...props} />
}