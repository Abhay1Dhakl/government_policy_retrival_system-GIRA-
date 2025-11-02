'use client';

import { useState, useRef, useEffect } from 'react';
import PlusMenu from './PlusMenu';
import { Mic, Plus, Send } from "lucide-react";

interface MessageInputProps {
  onSendMessage: (message: string) => void;
  selectedLlm?: string;
  setSelectedLlm?: React.Dispatch<React.SetStateAction<string>>;
  documentToggleStates?: Record<string, boolean>;
  setDocumentToggleStates?: React.Dispatch<React.SetStateAction<Record<string, boolean>>>;
  onPlusMenuSelect?: (option: string, type?: 'document' | 'llm') => void;
  isLoading?: boolean; 
}

export default function MessageInput({ 
  onSendMessage, 
  selectedLlm, 
  setSelectedLlm, 
  documentToggleStates, 
  setDocumentToggleStates, 
  onPlusMenuSelect,
  isLoading = false 
}: MessageInputProps) {
  const [message, setMessage] = useState('');
  const [showPlusMenu, setShowPlusMenu] = useState(false);
  const [isRecording, setIsRecording] = useState(false);

  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const plusButtonRef = useRef<HTMLButtonElement>(null);

  // Voice recording refs
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const silenceTimerRef = useRef<NodeJS.Timeout | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const dataArrayRef = useRef<Uint8Array<ArrayBuffer> | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const isMonitoringRef = useRef<boolean>(false);
  
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (message.trim() && !isLoading) {
      onSendMessage(message.trim());
      setMessage('');
      setShowPlusMenu(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey && !isLoading) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
    }
  }, [message]);

  const handlePlusMenuSelect = (option: string, type?: 'document' | 'llm') => {
    setShowPlusMenu(false);
    if (onPlusMenuSelect) onPlusMenuSelect(option, type);
  };

  // Monitor silence and auto-stop
  const monitorSilence = () => {
    if (!analyserRef.current || !dataArrayRef.current) return;
    
    isMonitoringRef.current = true;

    const checkSilence = () => {
      if (!isMonitoringRef.current || !analyserRef.current || !dataArrayRef.current) return;
      
      analyserRef.current.getByteFrequencyData(dataArrayRef.current as Uint8Array<ArrayBuffer>);
      const avgVolume =
        (dataArrayRef.current as Uint8Array<ArrayBuffer>).reduce((a, b) => a + b, 0) /
        (dataArrayRef.current as Uint8Array<ArrayBuffer>).length;

      if (avgVolume < 10) {
        if (!silenceTimerRef.current) {
          silenceTimerRef.current = setTimeout(() => stopRecording(), 1500);
        }
      } else if (silenceTimerRef.current) {
        clearTimeout(silenceTimerRef.current);
        silenceTimerRef.current = null;
      }

      if (isMonitoringRef.current) {
        animationFrameRef.current = requestAnimationFrame(checkSilence);
      }
    };

    checkSilence();
  };

  // Stop recording safely
  const stopRecording = () => {
    isMonitoringRef.current = false;
    
    if (animationFrameRef.current !== null) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }

    mediaRecorderRef.current?.stop();
    setIsRecording(false);

    if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
      audioContextRef.current.close().catch(() => {});
      audioContextRef.current = null;
    }
    sourceRef.current = null;
    analyserRef.current = null;
    dataArrayRef.current = null;

    if (silenceTimerRef.current) {
      clearTimeout(silenceTimerRef.current);
      silenceTimerRef.current = null;
    }
  };

  // Handle voice recording and transcription
  const handleVoiceInput = async () => {
    if (isRecording) {
      stopRecording();
      return;
    }

    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const hasMic = devices.some(device => device.kind === 'audioinput');
      if (!hasMic) {
        alert('No microphone detected. Please connect a microphone.');
        return;
      }

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      // AudioContext for silence detection
      const audioContext = new AudioContext();
      const source = audioContext.createMediaStreamSource(stream);
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 256;
      const bufferLength = analyser.frequencyBinCount;

      // TypeScript-safe Uint8Array
      const dataArray = new Uint8Array(bufferLength) as Uint8Array<ArrayBuffer>;

      source.connect(analyser);

      audioContextRef.current = audioContext;
      analyserRef.current = analyser;
      sourceRef.current = source;
      dataArrayRef.current = dataArray;

      monitorSilence();

      mediaRecorder.ondataavailable = (event) => {
        audioChunksRef.current.push(event.data);
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        const formData = new FormData();
        formData.append('file', audioBlob, 'recording.webm');

        try {
          const res = await fetch('/api/transcribe', { method: 'POST', body: formData });
          const data = await res.json();
          if (data.text) {
            setMessage((prev) => (prev ? `${prev} ${data.text}` : data.text));
          } else if (data.error) {
            alert('Transcription error: ' + data.error);
          } else {
            alert('Transcription failed. Please try again.');
          }
        } catch (err: any) {
          console.error('Transcription API error:', err);
          alert('Transcription API error: ' + err.message);
        }
      };

      mediaRecorder.start();
      setIsRecording(true);
    } catch (err: any) {
      console.error('Microphone error:', err);
      if (err.name === 'NotFoundError') {
        alert('No microphone found. Please connect a microphone and allow access.');
      } else if (err.name === 'NotAllowedError') {
        alert('Microphone access denied. Please allow microphone permission.');
      } else {
        alert('Microphone error: ' + err.message);
      }
    }
  };

  return (
    <div className="bg-white border-t border-gray-200">
      <div className="p-4">
        <div className="max-w-4xl mx-auto">
          <form onSubmit={handleSubmit} className="relative">
            <div
              className={`flex items-center gap-3 p-3 border rounded-full bg-white shadow-sm transition-all ${
                isLoading
                  ? 'border-gray-200 opacity-60'
                  : 'border-gray-300 focus-within:border-blue-500 focus-within:ring-1 focus-within:ring-blue-500'
              }`}
            >
              {/* Plus Button */}
              <div className="relative">
                <button
                  ref={plusButtonRef}
                  type="button"
                  onClick={() => !isLoading && setShowPlusMenu(!showPlusMenu)}
                  disabled={isLoading}
                  className={`p-2 rounded-full transition-colors ${
                    isLoading
                      ? 'text-gray-300 cursor-not-allowed'
                      : 'text-gray-500 hover:text-gray-700 hover:bg-gray-100'
                  }`}
                  title={isLoading ? 'Please wait...' : 'Add attachments'}
                >
                  <Plus className="h-5 w-5" />
                </button>

                {showPlusMenu && !isLoading && (
                  <PlusMenu
                    onSelect={handlePlusMenuSelect}
                    onClose={() => setShowPlusMenu(false)}
                    anchorRef={plusButtonRef}
                    selectedLlm={selectedLlm || 'openai'}
                    setSelectedLlm={setSelectedLlm || (() => {})}
                    documentToggleStates={documentToggleStates}
                    setDocumentToggleStates={setDocumentToggleStates}
                  />
                )}
              </div>

              {/* Textarea */}
              <div className="flex-1 flex flex-col">
                {isRecording && (
                  <span className="text-sm text-red-600 mb-1 font-medium">
                    🎤 Listening...
                  </span>
                )}
                <textarea
                  ref={textareaRef}
                  value={message}
                  onChange={(e) => !isLoading && !isRecording && setMessage(e.target.value)}
                  onKeyDown={handleKeyDown}
                  disabled={isLoading || isRecording}
                  placeholder={
                    isLoading
                      ? 'Waiting for response...'
                      : 'Enter your medical query here…'
                  }
                  className={`flex-1 resize-none border-none outline-none bg-transparent min-h-[36px] max-h-32 text-base py-2 transition-colors
                    ${
                      isLoading
                        ? 'cursor-not-allowed text-gray-400 placeholder:text-gray-300'
                        : 'placeholder:text-gray-500'
                    }
                    ${message ? 'text-left' : 'text-center placeholder:text-center'}
                  `}
                  rows={1}
                />
              </div>

              {/* Right Buttons */}
              <div className="flex items-center gap-2">
                {/* Voice Button */}
                <button
                  type="button"
                  onClick={handleVoiceInput}
                  disabled={isLoading}
                  className={`p-2 rounded-full transition-all ${
                    isRecording
                      ? 'bg-red-100 text-red-600 animate-pulse'
                      : isLoading
                      ? 'text-gray-300 cursor-not-allowed'
                      : 'text-gray-500 hover:text-gray-700 hover:bg-gray-100'
                  }`}
                  title={isRecording ? 'Recording...' : 'Start voice input'}
                >
                  <Mic className="h-5 w-5" />
                </button>

                {/* Send Button */}
                <button
                  type="submit"
                  disabled={!message.trim() || isLoading}
                  className={`p-2 rounded-full transition-all ${
                    message.trim() && !isLoading
                      ? 'bg-blue-600 text-white hover:bg-blue-700 shadow-sm'
                      : 'bg-gray-100 text-gray-400 cursor-not-allowed'
                  }`}
                  title={
                    isLoading
                      ? 'Please wait...'
                      : !message.trim()
                      ? 'Enter a message'
                      : 'Send message'
                  }
                >
                  <Send className="h-5 w-5" />
                </button>
              </div>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
}