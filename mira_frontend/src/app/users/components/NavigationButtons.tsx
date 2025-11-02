'use client';

import React from 'react';

interface NavigationButtonsProps {
  showBack: boolean;
  nextText: string;
  onBack?: () => void;
  onNext: () => void;
  type?: 'button' | 'submit';
  disabled?: boolean;
}

export default function NavigationButtons({
  showBack,
  nextText,
  onBack,
  onNext,
  type = 'button',
  disabled = false
}: NavigationButtonsProps) {
  return (
    <div className="flex justify-between items-center pt-6">
      {showBack ? (
        <button
          type="button"
          onClick={onBack}
          className="flex items-center px-4 py-2 text-sm font-medium text-gray-500 
                   hover:text-gray-700 focus:outline-none focus:ring-2 
                   focus:ring-offset-2 focus:ring-blue-500"
        >
          <svg 
            className="w-4 h-4 mr-2" 
            fill="none" 
            stroke="currentColor" 
            viewBox="0 0 24 24"
          >
            <path 
              strokeLinecap="round" 
              strokeLinejoin="round" 
              strokeWidth={2} 
              d="M15 19l-7-7 7-7" 
            />
          </svg>
          Back
        </button>
      ) : (
        <div />
      )}
      
      <button
        type={type}
        onClick={type === 'button' ? onNext : undefined}
        disabled={disabled}
        className={`flex items-center px-6 py-2 text-sm font-medium rounded-md 
                 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500
                 ${disabled 
                   ? 'bg-gray-400 text-gray-200 cursor-not-allowed' 
                   : 'bg-blue-600 text-white hover:bg-blue-700'
                 }`}
      >
        {nextText}
        {type === 'button' && (
          <svg 
            className="w-4 h-4 ml-2" 
            fill="none" 
            stroke="currentColor" 
            viewBox="0 0 24 24"
          >
            <path 
              strokeLinecap="round" 
              strokeLinejoin="round" 
              strokeWidth={2} 
              d="M9 5l7 7-7 7" 
            />
          </svg>
        )}
      </button>
    </div>
  );
}