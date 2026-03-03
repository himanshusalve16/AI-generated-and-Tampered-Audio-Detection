import React, { useEffect, useRef } from 'react';
import { Activity, CheckCircle2, AlertTriangle } from 'lucide-react';
import { gsap } from 'gsap';

function ResultCard({ prediction, confidence, loading }) {
  const resultRef = useRef(null);

  useEffect(() => {
    if (!prediction || !resultRef.current) return;

    gsap.fromTo(
      resultRef.current,
      { opacity: 0, y: 12 },
      {
        opacity: 1,
        y: 0,
        duration: 0.45,
        ease: 'power2.out'
      }
    );
  }, [prediction]);

  const isPositive =
    prediction &&
    typeof prediction === 'string' &&
    prediction.toLowerCase().includes('real');

  return (
    <section className="rounded-2xl border border-slate-700/70 bg-slate-900/70 p-4 md:p-5 shadow-lg shadow-slate-900/60 flex flex-col">
      <div className="flex items-center justify-between gap-2 mb-3">
        <div className="flex items-center gap-2">
          <Activity className="h-5 w-5 text-sky-400" />
          <h2 className="text-sm font-semibold text-slate-50 tracking-tight">
            Model prediction
          </h2>
        </div>
        <span className="rounded-full bg-slate-800/80 px-2 py-0.5 text-[11px] font-medium text-slate-300 border border-slate-700/80">
          Binary classifier · ResNet-18
        </span>
      </div>

      {prediction ? (
        <div ref={resultRef} className="mt-2 space-y-3">
          <div
            className={`flex items-center gap-2 rounded-xl px-3 py-2.5 border ${
              isPositive
                ? 'border-emerald-500/50 bg-emerald-950/40'
                : 'border-amber-400/50 bg-amber-950/40'
            }`}
          >
            {isPositive ? (
              <CheckCircle2 className="h-5 w-5 text-emerald-400" />
            ) : (
              <AlertTriangle className="h-5 w-5 text-amber-300" />
            )}
            <div>
              <p className="text-xs font-medium text-slate-100 uppercase tracking-wide">
                {isPositive ? 'Likely real human speech' : 'Potentially AI-generated or tampered'}
              </p>
              <p className="text-[11px] text-slate-200">
                Model output: <span className="font-semibold">{prediction}</span>
              </p>
            </div>
          </div>

          {typeof confidence === 'number' && (
            <div className="rounded-xl border border-slate-700/80 bg-slate-900/70 px-3 py-2.5 space-y-1.5">
              <div className="flex items-center justify-between gap-2">
                <div className="flex items-center gap-1.5 text-[11px] text-slate-300">
                  <Activity className="h-3.5 w-3.5 text-sky-300" />
                  <span>Confidence score</span>
                </div>
                <span className="text-xs font-semibold text-slate-50">
                  {(confidence * 100).toFixed(2)}%
                </span>
              </div>
              <div className="h-1.5 w-full rounded-full bg-slate-800 overflow-hidden">
                <div
                  className="h-1.5 rounded-full bg-gradient-to-r from-sky-400 via-cyan-300 to-emerald-300"
                  style={{ width: `${Math.min(Math.max(confidence * 100, 4), 100)}%` }}
                />
              </div>
              <p className="text-[10px] text-slate-400">
                Derived from softmax probabilities over the two output classes of the network.
              </p>
            </div>
          )}
        </div>
      ) : (
        <div className="mt-2 rounded-xl border border-dashed border-slate-700/80 bg-slate-900/60 px-3 py-3">
          <p className="text-xs font-medium text-slate-200">Awaiting prediction</p>
          <p className="mt-1 text-[11px] text-slate-400">
            Upload a speech clip on the left and click <span className="font-semibold">Analyze audio</span> to
            run pre-processing, mel-spectrogram extraction, and inference through the ResNet-18 model.
          </p>
          {loading && (
            <p className="mt-1 text-[11px] text-sky-300">
              The model is currently processing your audio…
            </p>
          )}
        </div>
      )}
    </section>
  );
}

export default ResultCard;

