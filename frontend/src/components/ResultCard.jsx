import React, { useEffect, useRef } from 'react';
import { Activity, CheckCircle2, AlertTriangle, Brain, Cpu, Layers } from 'lucide-react';
import { gsap } from 'gsap';

/* ---------------------------------------------------------------
   Single prediction card (reusable for ResNet / LSTM / Ensemble)
--------------------------------------------------------------- */
function PredictionCard({ icon: Icon, title, tag, prediction, confidence, accentClass }) {
  const isReal =
    prediction &&
    typeof prediction === 'string' &&
    prediction.toLowerCase().includes('real');

  const isNA = prediction === 'N/A';

  return (
    <div className="rounded-xl border border-slate-700/70 bg-slate-900/60 p-3 space-y-2">
      {/* Header */}
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-1.5">
          <Icon className={`h-4 w-4 ${accentClass}`} />
          <span className="text-xs font-semibold text-slate-100">{title}</span>
        </div>
        <span className="rounded-full bg-slate-800/80 px-2 py-0.5 text-[10px] font-medium text-slate-400 border border-slate-700/80">
          {tag}
        </span>
      </div>

      {/* Prediction pill */}
      {prediction && !isNA ? (
        <div
          className={`flex items-center gap-2 rounded-lg px-2.5 py-2 border ${
            isReal
              ? 'border-emerald-500/50 bg-emerald-950/30'
              : 'border-amber-400/50 bg-amber-950/30'
          }`}
        >
          {isReal ? (
            <CheckCircle2 className="h-4 w-4 text-emerald-400 shrink-0" />
          ) : (
            <AlertTriangle className="h-4 w-4 text-amber-300 shrink-0" />
          )}
          <div>
            <p className="text-[11px] font-medium text-slate-100 uppercase tracking-wide">
              {isReal ? 'Real human speech' : 'AI-generated / tampered'}
            </p>
            <p className="text-[10px] text-slate-300">
              Output: <span className="font-semibold">{prediction}</span>
            </p>
          </div>
        </div>
      ) : isNA ? (
        <p className="text-[11px] text-slate-500 italic px-1">Model not loaded</p>
      ) : null}

      {/* Confidence bar */}
      {typeof confidence === 'number' && confidence > 0 && !isNA && (
        <div className="space-y-1">
          <div className="flex items-center justify-between text-[10px] text-slate-400">
            <span className="flex items-center gap-1">
              <Activity className="h-3 w-3 text-sky-300" />
              Confidence
            </span>
            <span className="font-semibold text-slate-200">{(confidence * 100).toFixed(2)}%</span>
          </div>
          <div className="h-1.5 w-full rounded-full bg-slate-800 overflow-hidden">
            <div
              className={`h-1.5 rounded-full bg-gradient-to-r ${
                isReal
                  ? 'from-emerald-400 via-emerald-300 to-cyan-300'
                  : 'from-amber-400 via-orange-300 to-rose-400'
              }`}
              style={{ width: `${Math.min(Math.max(confidence * 100, 4), 100)}%` }}
            />
          </div>
        </div>
      )}
    </div>
  );
}

/* ---------------------------------------------------------------
   Main ResultCard — shows spectrogram + 3 prediction cards
--------------------------------------------------------------- */
function ResultCard({ result, loading }) {
  const resultRef = useRef(null);

  useEffect(() => {
    if (!result || !resultRef.current) return;

    gsap.fromTo(
      resultRef.current,
      { opacity: 0, y: 16 },
      {
        opacity: 1,
        y: 0,
        duration: 0.5,
        ease: 'power2.out'
      }
    );
  }, [result]);

  return (
    <section className="rounded-2xl border border-slate-700/70 bg-slate-900/70 p-4 md:p-5 shadow-lg shadow-slate-900/60 flex flex-col">
      <div className="flex items-center justify-between gap-2 mb-3">
        <div className="flex items-center gap-2">
          <Layers className="h-5 w-5 text-sky-400" />
          <h2 className="text-sm font-semibold text-slate-50 tracking-tight">
            Ensemble Predictions
          </h2>
        </div>
        <span className="rounded-full bg-slate-800/80 px-2 py-0.5 text-[11px] font-medium text-slate-300 border border-slate-700/80">
          ResNet-18 + LSTM
        </span>
      </div>

      {result ? (
        <div ref={resultRef} className="space-y-4">
          {/* Spectrogram display */}
          {result.spectrogram_image && (
            <div className="space-y-1.5">
              <p className="text-xs font-medium text-slate-200 flex items-center gap-1.5">
                <span className="inline-flex h-1.5 w-1.5 rounded-full bg-violet-400" />
                Mel Spectrogram (ResNet input)
              </p>
              <div className="rounded-xl overflow-hidden border border-slate-700/70 bg-slate-950/50">
                <img
                  src={result.spectrogram_image}
                  alt="Mel Spectrogram"
                  className="w-full h-auto object-contain"
                />
              </div>
            </div>
          )}

          {/* Three prediction cards */}
          <div className="grid gap-3 sm:grid-cols-1 lg:grid-cols-3">
            <PredictionCard
              icon={Cpu}
              title="ResNet-18"
              tag="CNN"
              prediction={result.resnet_prediction}
              confidence={result.resnet_confidence}
              accentClass="text-sky-400"
            />
            <PredictionCard
              icon={Brain}
              title="LSTM"
              tag="RNN"
              prediction={result.lstm_prediction}
              confidence={result.lstm_confidence}
              accentClass="text-violet-400"
            />
            <PredictionCard
              icon={Layers}
              title="Ensemble"
              tag="Final"
              prediction={result.ensemble_prediction}
              confidence={result.ensemble_confidence}
              accentClass="text-emerald-400"
            />
          </div>

          <p className="text-[10px] text-slate-500 px-1">
            Ensemble prediction uses weighted averaging of softmax probabilities from both models.
          </p>
        </div>
      ) : (
        <div className="mt-2 rounded-xl border border-dashed border-slate-700/80 bg-slate-900/60 px-3 py-3">
          <p className="text-xs font-medium text-slate-200">Awaiting prediction</p>
          <p className="mt-1 text-[11px] text-slate-400">
            Upload a speech clip on the left and click <span className="font-semibold">Analyze audio</span> to
            run preprocessing, mel-spectrogram extraction, and inference through the ResNet-18 + LSTM ensemble.
          </p>
          {loading && (
            <p className="mt-1 text-[11px] text-sky-300">
              Both models are currently processing your audio…
            </p>
          )}
        </div>
      )}
    </section>
  );
}

export default ResultCard;
