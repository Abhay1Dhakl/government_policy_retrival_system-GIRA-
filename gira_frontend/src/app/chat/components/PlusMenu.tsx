'use client';
import { FileText, Bot, ChevronRight, ChevronLeft, GripVertical, FileSearch2 } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { documentTypeOptions, llmOptions} from '../utils/types';

interface PlusMenuProps {
  onSelect: (option: string, type?: 'document' | 'llm') => void;
  onClose: () => void;
  anchorRef: React.RefObject<HTMLButtonElement | null>;
  documentToggleStates?: Record<string, boolean>;
  setDocumentToggleStates?: React.Dispatch<React.SetStateAction<Record<string, boolean>>>;
  selectedLlm: string;
  setSelectedLlm: React.Dispatch<React.SetStateAction<string>>;
}

const mainMenuOptions = [
  {
    id: 'choose-document-type',
    label: 'Choose Document Type',
    icon: FileText
  },
  {
    id: 'deep-research',
    label: 'Deep Research',
    icon: FileSearch2
  },
];

export default function PlusMenu({ 
  onSelect, 
  onClose, 
  anchorRef, 
  documentToggleStates = {},
  setDocumentToggleStates,
  selectedLlm,
  setSelectedLlm
}: PlusMenuProps) {
  const menuRef = useRef<HTMLDivElement>(null);
  const [currentView, setCurrentView] = useState<'main' | 'document-type'>('main');
  
  // State for reorderable document types
  const [documentOrder, setDocumentOrder] = useState([...documentTypeOptions]);
  const [draggedDocIndex, setDraggedDocIndex] = useState<number | null>(null);
  
  // (LLM selection removed) -- this menu only supports document types and deep research
  
  // Local state for document toggles as fallback
  const [localDocumentStates, setLocalDocumentStates] = useState(() => {
    return documentTypeOptions.reduce((acc, option) => ({
      ...acc,
      [option.id]: option.enabled
    }), {} as Record<string, boolean>);
  });
  
  // Use provided states if available, otherwise use local state
  const safeDocumentToggleStates = documentToggleStates && Object.keys(documentToggleStates).length > 0 
    ? documentToggleStates 
    : localDocumentStates;
  
  const handleMainMenuClick = (optionId: string) => {
    if (optionId === 'choose-document-type') {
      setCurrentView('document-type');
    } else {
      // deep-research or other actions
      onSelect(optionId);
    }
  };

  const handleDocumentToggle = (optionId: string) => {
    const newStates = {
      ...safeDocumentToggleStates,
      [optionId]: !safeDocumentToggleStates[optionId]
    };
    
    // Update the provided state if available and we're using it
    if (setDocumentToggleStates && documentToggleStates && Object.keys(documentToggleStates).length > 0) {
      setDocumentToggleStates(newStates);
    } else {
      // Otherwise update local state
      setLocalDocumentStates(newStates);
    }
  };

  // LLM selection removed — no-op

  const handleBack = () => {
    setCurrentView('main');
  };

  const handleDocumentTypeSelectionComplete = () => {
    // Get all selected document types as tool names in the current order
    const selectedTools = documentOrder
      .filter(option => safeDocumentToggleStates[option.id])
      .map(option => option.toolName);
    
    // Send the selected tools as a comma-separated string
    onSelect(selectedTools.join(','), 'document');
    setCurrentView('main');
  };

  // Document drag handlers
  const handleDocumentDragStart = (e: React.DragEvent, index: number) => {
    setDraggedDocIndex(index);
    e.dataTransfer.effectAllowed = 'move';
  };

  const handleDocumentDragOver = (e: React.DragEvent, index: number) => {
    e.preventDefault();
    if (draggedDocIndex === null || draggedDocIndex === index) return;
    
    const newOrder = [...documentOrder];
    const draggedItem = newOrder[draggedDocIndex];
    newOrder.splice(draggedDocIndex, 1);
    newOrder.splice(index, 0, draggedItem);
    
    setDocumentOrder(newOrder);
    setDraggedDocIndex(index);
  };

  const handleDocumentDragEnd = () => {
    setDraggedDocIndex(null);
  };

  // (LLM handlers removed)

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        menuRef.current &&
        !menuRef.current.contains(event.target as Node) &&
        anchorRef.current &&
        !anchorRef.current.contains(event.target as Node)
      ) {
        onClose();
      }
    };

    const handleEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        if (currentView !== 'main') {
          setCurrentView('main');
        } else {
          onClose();
        }
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    document.addEventListener('keydown', handleEscape);

    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
      document.removeEventListener('keydown', handleEscape);
    };
  }, [onClose, currentView, anchorRef]);

  return (
    <div
      ref={menuRef}
      className="absolute bottom-full left-0 mb-2 w-64 bg-white border border-gray-200 rounded-lg shadow-lg z-50"
    >
      {currentView === 'main' && (
        <div className="py-2">
          {mainMenuOptions.map((option) => (
            <button
              key={option.id}
              onClick={() => handleMainMenuClick(option.id)}
              className="w-full flex items-center justify-between px-4 py-3 text-left hover:bg-gray-50 transition-colors text-gray-700"
            >
              <div className="flex items-center gap-3">
                <span className="text-gray-500">
                  <option.icon size={18} />
                </span>
                <span className="text-sm font-medium">{option.label}</span>
              </div>
              <ChevronRight size={16} className="text-gray-400" />
            </button>
          ))}
        </div>
      )}

      {currentView === 'document-type' && (
        <div className="py-2">
          <div className="flex items-center gap-2 px-4 py-2 border-b border-gray-100">
            <button 
              onClick={handleBack}
              className="text-gray-500 hover:text-gray-700"
            >
              <ChevronLeft size={16} />
            </button>
            <span className="text-sm font-semibold text-gray-700">Choose Document Type</span>
            <button
              onClick={handleDocumentTypeSelectionComplete}
              className="ml-auto text-sm text-blue-600 font-medium"
            >
              Done
            </button>
          </div>
          {documentOrder.map((option, index) => (
            <div
              key={option.id}
              draggable
              onDragStart={(e) => handleDocumentDragStart(e, index)}
              onDragOver={(e) => handleDocumentDragOver(e, index)}
              onDragEnd={handleDocumentDragEnd}
              className={`flex items-center justify-between px-4 py-3 hover:bg-gray-50 transition-colors cursor-grab active:cursor-grabbing ${
                draggedDocIndex === index ? 'opacity-50 bg-gray-100' : ''
              }`}
            >
              <div className="flex items-center gap-2">
                <GripVertical size={16} className="text-gray-400 cursor-grab" />
                <span 
                  className="text-sm font-medium text-gray-700 cursor-pointer flex-1"
                  onClick={() => handleDocumentToggle(option.id)}
                >
                  {option.label}
                </span>
              </div>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  handleDocumentToggle(option.id);
                }}
                className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 ${
                  safeDocumentToggleStates[option.id] ? 'bg-blue-600' : 'bg-gray-200'
                }`}
              >
                <span
                  className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                    safeDocumentToggleStates[option.id] ? 'translate-x-6' : 'translate-x-1'
                  }`}
                />
              </button>
            </div>
          ))}
        </div>
      )}

      {/* LLM view removed - only document-type and deep-research are supported here */}
    </div>
  );
}