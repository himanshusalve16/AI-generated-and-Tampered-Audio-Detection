import React, { useState, useRef, useEffect, useCallback } from 'react';
import axios from 'axios';
import { gsap } from 'gsap';
import { Waves, Sparkles } from 'lucide-react';
import UploadCard from './components/UploadCard';
import ResultCard from './components/ResultCard';

function App() {
  const [file, setFile] = useState(null);
  const [audioUrl, setAudioUrl] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  // --- Mic recording state ---
  const [recording, setRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);
  const timerRef = useRef(null);

  const containerRef = useRef(null);

  useEffect(() => {
    if (!containerRef.current) return;
    gsap.fromTo(
      containerRef.current,
      { opacity: 0, y: 40, scale: 0.96 },
      { opacity: 1, y: 0, scale: 1, duration: 0.9, ease: 'power3.out' }
    );
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
        mediaRecorderRef.current.stop();
      }
    };
  }, []);

  const handleFileChange = (event) => {
    const selectedFile = event.target.files?.[0];
    setResult(null);
    setError('');

    if (!selectedFile) {
      setFile(null);
      setAudioUrl('');
      return;
    }

    setFile(selectedFile);
    const url = URL.createObjectURL(selectedFile);
    setAudioUrl(url);
  };

  // --- Mic recording handlers ---
  const startRecording = useCallback(async () => {
    setResult(null);
    setError('');
    setFile(null);
    setAudioUrl('');
    chunksRef.current = [];
    setRecordingTime(0);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

      // Pick a supported MIME type
      const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
        ? 'audio/webm;codecs=opus'
        : MediaRecorder.isTypeSupported('audio/webm')
          ? 'audio/webm'
          : '';

      const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : {});
      mediaRecorderRef.current = recorder;

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      recorder.onstop = () => {
        // Stop all mic tracks
        stream.getTracks().forEach((t) => t.stop());
        if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null; }

        if (chunksRef.current.length === 0) return;

        const blob = new Blob(chunksRef.current, {
          type: recorder.mimeType || 'audio/webm',
        });

        // Determine extension from MIME
        const ext = blob.type.includes('webm') ? 'webm' : blob.type.includes('ogg') ? 'ogg' : 'wav';
        const recorded = new File([blob], `mic_recording.${ext}`, { type: blob.type });
        setFile(recorded);
        setAudioUrl(URL.createObjectURL(blob));
        setRecording(false);
      };

      recorder.start(250); // collect in 250ms chunks
      setRecording(true);

      // Timer for UI display
      timerRef.current = setInterval(() => {
        setRecordingTime((prev) => prev + 1);
      }, 1000);
    } catch (err) {
      console.error('Microphone access error:', err);
      setError('Could not access microphone. Please allow mic permission and try again.');
    }
  }, []);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
    }
  }, []);

  const handleUpload = async () => {
    if (!file) {
      setError('Please select or record an audio file first.');
      return;
    }

    setLoading(true);
    setError('');
    setResult(null);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await axios.post('http://localhost:8000/predict', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      setResult(response.data);
    } catch (err) {
      if (err.response && err.response.data && err.response.data.detail) {
        setError(`Server error: ${err.response.data.detail}`);
      } else {
        setError('Could not get prediction. Please ensure the backend is running on port 8000.');
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center px-4 py-8 bg-transparent font-sans">
      <div
        ref={containerRef}
        className="relative w-full max-w-6xl rounded-3xl bg-slate-900/70 border border-slate-700/60 shadow-2xl shadow-sky-500/10 backdrop-blur-2xl p-6 md:p-8 overflow-hidden"
      >
        <div className="pointer-events-none absolute -top-32 -right-16 h-72 w-72 rounded-full bg-sky-500/10 blur-3xl" />
        <div className="pointer-events-none absolute -bottom-24 -left-10 h-64 w-64 rounded-full bg-cyan-400/10 blur-3xl" />

        <header className="relative z-10 mb-8 flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
          <div>
            <div className="inline-flex items-center gap-2 rounded-full border border-sky-500/30 bg-slate-900/60 px-3 py-1 text-xs font-medium text-sky-200 shadow-glow-blue">
              <Sparkles className="h-3.5 w-3.5 text-sky-300" />
              <span>Deep Learning Lab · ResNet-18 + LSTM Ensemble</span>
            </div>
            <h1 className="mt-4 text-2xl md:text-3xl font-semibold text-slate-50 tracking-tight flex items-center gap-2">
              <Waves className="h-7 w-7 text-sky-400" />
              AI-Generated &amp; Tampered Audio Detection
            </h1>
            <p className="mt-2 text-sm md:text-base text-slate-300 max-w-2xl">
              Upload a file or record from your microphone to analyze mel-spectrogram features with a
              ResNet-18 + LSTM ensemble and detect AI-generated speech.
            </p>
          </div>
        </header>

        <main className="relative z-10 grid gap-6 md:grid-cols-[minmax(0,1fr)_minmax(0,1.4fr)]">
          <UploadCard
            file={file}
            audioUrl={audioUrl}
            loading={loading}
            error={error}
            recording={recording}
            recordingTime={recordingTime}
            onFileChange={handleFileChange}
            onUpload={handleUpload}
            onStartRecording={startRecording}
            onStopRecording={stopRecording}
          />

          <ResultCard result={result} loading={loading} />
        </main>
      </div>
    </div>
  );
}

export default App;
