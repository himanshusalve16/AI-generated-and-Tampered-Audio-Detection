import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { gsap } from 'gsap';
import { Waves, Sparkles } from 'lucide-react';
import UploadCard from './components/UploadCard';
import ResultCard from './components/ResultCard';

function App() {
  const [file, setFile] = useState(null);
  const [audioUrl, setAudioUrl] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);   // full ensemble result object
  const [error, setError] = useState('');

  const containerRef = useRef(null);

  useEffect(() => {
    if (!containerRef.current) return;

    gsap.fromTo(
      containerRef.current,
      { opacity: 0, y: 40, scale: 0.96 },
      {
        opacity: 1,
        y: 0,
        scale: 1,
        duration: 0.9,
        ease: 'power3.out'
      }
    );
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

  const handleUpload = async () => {
    if (!file) {
      setError('Please select an audio file first.');
      return;
    }

    setLoading(true);
    setError('');
    setResult(null);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await axios.post('http://localhost:8000/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });

      // Store the full ensemble response
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
              Upload a short audio clip to analyze mel-spectrogram features with a
              ResNet-18 + LSTM ensemble and estimate whether it is real human speech or AI-generated.
            </p>
          </div>
        </header>

        <main className="relative z-10 grid gap-6 md:grid-cols-[minmax(0,1fr)_minmax(0,1.4fr)]">
          <UploadCard
            file={file}
            audioUrl={audioUrl}
            loading={loading}
            error={error}
            onFileChange={handleFileChange}
            onUpload={handleUpload}
          />

          <ResultCard result={result} loading={loading} />
        </main>
      </div>
    </div>
  );
}

export default App;
