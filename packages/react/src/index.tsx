import React, { useState, useEffect, useRef, useCallback } from 'react';

export interface VoiceWidgetProps {
  /** WebSocket server URL (e.g., wss://voice.example.com/ws) */
  serverUrl: string;
  /** API key for authentication */
  apiKey?: string;
  /** Enable continuous conversation mode */
  continuousMode?: boolean;
  /** Callback when transcript is received */
  onTranscript?: (text: string) => void;
  /** Callback when AI responds */
  onResponse?: (text: string) => void;
  /** Callback on error */
  onError?: (error: string) => void;
  /** Custom button style */
  buttonStyle?: React.CSSProperties;
  /** Custom container style */
  style?: React.CSSProperties;
  /** Button size in pixels */
  size?: number;
  /** Primary color */
  color?: string;
}

type ConnectionStatus = 'disconnected' | 'connecting' | 'connected' | 'error';

export function VoiceWidget({
  serverUrl,
  apiKey,
  continuousMode = false,
  onTranscript,
  onResponse,
  onError,
  buttonStyle,
  style,
  size = 80,
  color = '#ff6b35',
}: VoiceWidgetProps) {
  const [status, setStatus] = useState<ConnectionStatus>('disconnected');
  const [isListening, setIsListening] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  
  const wsRef = useRef<WebSocket | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);

  // Connect to WebSocket
  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;
    
    setStatus('connecting');
    
    const url = apiKey 
      ? `${serverUrl}?api_key=${apiKey}`
      : serverUrl;
    
    const ws = new WebSocket(url);
    
    ws.onopen = () => {
      setStatus('connected');
    };
    
    ws.onclose = (event) => {
      setStatus('disconnected');
      if (event.code === 4001) {
        onError?.('API key required');
      } else if (event.code === 4002) {
        onError?.('Invalid API key');
      } else if (event.code === 4003) {
        onError?.('Rate limit exceeded');
      }
    };
    
    ws.onerror = () => {
      setStatus('error');
      onError?.('Connection failed');
    };
    
    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      handleMessage(msg);
    };
    
    wsRef.current = ws;
  }, [serverUrl, apiKey, onError]);

  // Handle incoming messages
  const handleMessage = useCallback((msg: any) => {
    switch (msg.type) {
      case 'listening_started':
        setIsListening(true);
        break;
      case 'listening_stopped':
        setIsListening(false);
        break;
      case 'transcript':
        onTranscript?.(msg.text);
        break;
      case 'response_text':
        onResponse?.(msg.text);
        break;
      case 'audio_response':
        playAudio(msg.data, msg.sample_rate);
        if (continuousMode) {
          // Auto-start listening after response
          setTimeout(() => startListening(), 500);
        }
        break;
    }
  }, [onTranscript, onResponse, continuousMode]);

  // Play audio response
  const playAudio = useCallback((base64Data: string, sampleRate: number) => {
    setIsSpeaking(true);
    
    const audioCtx = new AudioContext({ sampleRate });
    const binary = atob(base64Data);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) {
      bytes[i] = binary.charCodeAt(i);
    }
    const audioData = new Float32Array(bytes.buffer);
    
    const buffer = audioCtx.createBuffer(1, audioData.length, sampleRate);
    buffer.getChannelData(0).set(audioData);
    
    const source = audioCtx.createBufferSource();
    source.buffer = buffer;
    source.connect(audioCtx.destination);
    source.onended = () => setIsSpeaking(false);
    source.start();
  }, []);

  // Start listening
  const startListening = useCallback(async () => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      connect();
      return;
    }
    
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { sampleRate: 16000, channelCount: 1 }
      });
      
      mediaStreamRef.current = stream;
      audioContextRef.current = new AudioContext({ sampleRate: 16000 });
      
      const source = audioContextRef.current.createMediaStreamSource(stream);
      const processor = audioContextRef.current.createScriptProcessor(4096, 1, 1);
      
      processor.onaudioprocess = (e) => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
          const audioData = e.inputBuffer.getChannelData(0);
          const base64 = btoa(String.fromCharCode(...new Uint8Array(audioData.buffer)));
          wsRef.current.send(JSON.stringify({ type: 'audio', data: base64 }));
        }
      };
      
      source.connect(processor);
      processor.connect(audioContextRef.current.destination);
      processorRef.current = processor;
      
      wsRef.current.send(JSON.stringify({ type: 'start_listening' }));
      
    } catch (err) {
      onError?.('Microphone access denied');
    }
  }, [connect, onError]);

  // Stop listening
  const stopListening = useCallback(() => {
    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current = null;
    }
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach(t => t.stop());
      mediaStreamRef.current = null;
    }
    
    wsRef.current?.send(JSON.stringify({ type: 'stop_listening' }));
  }, []);

  // Connect on mount
  useEffect(() => {
    connect();
    return () => {
      wsRef.current?.close();
    };
  }, [connect]);

  // Button handlers
  const handleMouseDown = () => {
    if (!continuousMode) startListening();
  };
  
  const handleMouseUp = () => {
    if (!continuousMode) stopListening();
  };
  
  const handleClick = () => {
    if (continuousMode) {
      if (isListening) {
        stopListening();
      } else {
        startListening();
      }
    }
  };

  const buttonColor = isListening ? color : isSpeaking ? '#4fc3f7' : color;
  
  return (
    <div style={{ textAlign: 'center', ...style }}>
      <button
        onMouseDown={handleMouseDown}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onClick={handleClick}
        disabled={status !== 'connected'}
        style={{
          width: size,
          height: size,
          borderRadius: '50%',
          border: 'none',
          background: buttonColor,
          color: 'white',
          fontSize: size / 5,
          fontWeight: 600,
          cursor: status === 'connected' ? 'pointer' : 'not-allowed',
          opacity: status === 'connected' ? 1 : 0.5,
          transition: 'all 0.2s',
          boxShadow: isListening ? `0 0 20px ${color}` : 'none',
          ...buttonStyle,
        }}
      >
        {isListening ? '🎙️' : isSpeaking ? '🔊' : '🎤'}
      </button>
      <div style={{ marginTop: 8, fontSize: 12, color: '#888' }}>
        {status === 'connecting' && 'Connecting...'}
        {status === 'connected' && (isListening ? 'Listening...' : continuousMode ? 'Tap to talk' : 'Hold to talk')}
        {status === 'disconnected' && 'Disconnected'}
        {status === 'error' && 'Connection error'}
      </div>
    </div>
  );
}

export default VoiceWidget;
