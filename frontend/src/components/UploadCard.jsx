import React, { useRef } from 'react';
import { UploadCloud, FileAudio2, Loader2, Mic, MicOff, CircleDot } from 'lucide-react';
import { gsap } from 'gsap';

function UploadCard({
  file,
  audioUrl,
  loading,
  error,
  recording,
  recordingTime,
  onFileChange,
  onUpload,
  onStartRecording,
  onStopRecording,
}) {
  const buttonRef = useRef(null);

  const handleClick = async () => {
    if (!buttonRef.current) {
      await onUpload();
      return;
    }

    gsap.fromTo(
      buttonRef.current,
      { scale: 1 },
      {
        scale: 0.96,
        duration: 0.08,
        yoyo: true,
        repeat: 1,
        ease: 'power2.inOut',
        onComplete: onUpload,
      }
    );
  };

  // Format seconds as mm:ss
  const formatTime = (secs) => {
    const m = Math.floor(secs / 60).toString().padStart(2, '0');
    const s = (secs % 60).toString().padStart(2, '0');
    return `${m}:${s}`;
  };

  return (
    <section className="rounded-2xl border border-slate-700/70 bg-slate-900/70 p-4 md:p-5 shadow-lg shadow-slate-900/60">
      <div className="flex items-center justify-between gap-2 mb-3">
        <div className="flex items-center gap-2">
          <UploadCloud className="h-5 w-5 text-sky-400" />
          <h2 className="text-sm font-semibold text-slate-50 tracking-tight">Upload or Record</h2>
        </div>
        <span className="rounded-full bg-slate-800/80 px-2 py-0.5 text-[11px] font-medium text-slate-300 border border-slate-700/80">
          .wav · .mp3 · mic
        </span>
      </div>

      {/* --- File upload --- */}
      <label className="group flex flex-col gap-3 rounded-xl border border-dashed border-slate-600/80 bg-slate-900/70 px-4 py-3 text-sm text-slate-300 cursor-pointer hover:border-sky-400/80 hover:bg-slate-900/90 transition-colors">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-full bg-slate-800/80 text-sky-300 group-hover:bg-sky-500/10 group-hover:text-sky-300 transition-colors">
            <FileAudio2 className="h-5 w-5" />
          </div>
          <div className="flex flex-col">
            <span className="text-xs font-medium text-slate-100">
              {file ? 'Change audio file' : 'Select audio file'}
            </span>
            <span className="text-[11px] text-slate-400">
              Click to browse your device and choose a short speech clip.
            </span>
          </div>
        </div>
        <input
          type="file"
          accept=".wav, audio/wav, .mp3, audio/mpeg, .webm, audio/webm, .ogg, audio/ogg, .flac, audio/flac, .m4a"
          onChange={onFileChange}
          className="hidden"
          disabled={recording}
        />
      </label>

      {/* --- Divider --- */}
      <div className="flex items-center gap-3 my-3">
        <div className="flex-1 h-px bg-slate-700/60" />
        <span className="text-[10px] font-medium text-slate-500 uppercase tracking-widest">or</span>
        <div className="flex-1 h-px bg-slate-700/60" />
      </div>

      {/* --- Mic recording --- */}
      {recording ? (
        <button
          type="button"
          onClick={onStopRecording}
          className="group w-full flex items-center gap-3 rounded-xl border border-rose-500/60 bg-rose-950/30 px-4 py-3 text-sm transition-colors hover:bg-rose-950/50"
        >
          <div className="relative flex h-10 w-10 items-center justify-center rounded-full bg-rose-500/20 text-rose-400">
            <MicOff className="h-5 w-5" />
            {/* pulsing ring */}
            <span className="absolute inset-0 rounded-full border-2 border-rose-400 animate-ping opacity-40" />
          </div>
          <div className="flex flex-col items-start">
            <span className="text-xs font-semibold text-rose-300 flex items-center gap-1.5">
              <CircleDot className="h-3 w-3 text-rose-400 animate-pulse" />
              Recording… {formatTime(recordingTime)}
            </span>
            <span className="text-[11px] text-slate-400">Click to stop recording</span>
          </div>
        </button>
      ) : (
        <button
          type="button"
          onClick={onStartRecording}
          disabled={loading}
          className="group w-full flex items-center gap-3 rounded-xl border border-dashed border-slate-600/80 bg-slate-900/70 px-4 py-3 text-sm transition-colors hover:border-emerald-400/60 hover:bg-slate-900/90 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <div className="flex h-10 w-10 items-center justify-center rounded-full bg-slate-800/80 text-emerald-400 group-hover:bg-emerald-500/10 transition-colors">
            <Mic className="h-5 w-5" />
          </div>
          <div className="flex flex-col items-start">
            <span className="text-xs font-medium text-slate-100">Record from microphone</span>
            <span className="text-[11px] text-slate-400">
              Click to start recording live audio from your mic.
            </span>
          </div>
        </button>
      )}

      {/* --- Audio preview --- */}
      {audioUrl && (
        <div className="mt-4 space-y-2">
          <p className="text-xs font-medium text-slate-200 flex items-center gap-1.5">
            <span className="inline-flex h-1.5 w-1.5 rounded-full bg-emerald-400" />
            Audio preview
          </p>
          <audio
            controls
            src={audioUrl}
            className="w-full rounded-lg border border-slate-700/70 bg-slate-900/80"
          >
            Your browser does not support the audio element.
          </audio>
          {file && (
            <p className="text-[11px] text-slate-400 truncate">
              {file.name.startsWith('mic_') ? (
                <>Source: <span className="text-emerald-300">Microphone</span></>
              ) : (
                <>Selected file: <span className="text-slate-200">{file.name}</span></>
              )}
            </p>
          )}
        </div>
      )}

      {/* --- Analyze button --- */}
      <div className="mt-4 space-y-2">
        <button
          ref={buttonRef}
          type="button"
          onClick={handleClick}
          disabled={loading || !file || recording}
          className="inline-flex w-full items-center justify-center gap-2 rounded-full bg-sky-500 px-4 py-2.5 text-sm font-semibold text-slate-950 shadow-lg shadow-sky-500/40 transition-all hover:bg-sky-400 disabled:cursor-not-allowed disabled:bg-slate-700 disabled:text-slate-300 disabled:shadow-none"
        >
          {loading ? (
            <>
              <Loader2 className="h-4 w-4 animate-spin" />
              <span>Analyzing audio...</span>
            </>
          ) : (
            <>
              <UploadCloud className="h-4 w-4" />
              <span>Analyze audio</span>
            </>
          )}
        </button>

        {loading && (
          <p className="text-[11px] text-slate-400">
            Running pre-processing and spectral feature extraction on your audio clip…
          </p>
        )}

        {error && (
          <p className="text-[11px] text-rose-400 border border-rose-500/40 bg-rose-950/40 rounded-lg px-3 py-2">
            {error}
          </p>
        )}
      </div>
    </section>
  );
}

export default UploadCard;
