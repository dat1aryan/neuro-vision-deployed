import { AnimatePresence, motion } from 'framer-motion';
import { Flame, Sparkles } from 'lucide-react';


function Spinner() {
  return (
    <svg className="h-4 w-4 animate-spin" viewBox="0 0 24 24" fill="none">
      <circle cx="12" cy="12" r="10" stroke="currentColor" strokeOpacity="0.2" strokeWidth="4" />
      <path d="M22 12a10 10 0 0 0-10-10" stroke="currentColor" strokeWidth="4" strokeLinecap="round" />
    </svg>
  );
}


export default function GradCAMView({ previewUrl, gradcamUrl, loading, disabled, onGenerate }) {
  return (
    <section className="glass-card h-full p-6 sm:p-7">
      <div className="relative z-10">
        <div className="mb-5 flex flex-wrap items-start justify-between gap-4">
          <div>
            <p className="section-kicker">Explainable AI</p>
            <h2 className="section-title mt-2">AI Explainability</h2>
            <p className="section-copy mt-2">
              Visualize how the AI model interprets MRI scans using GradCAM heatmaps, highlighting the regions that influence tumor classification decisions.
            </p>
          </div>

          <button type="button" disabled={disabled || loading} onClick={onGenerate} className="neon-button">
            {loading ? <Spinner /> : <Flame className="h-4 w-4" />}
            Generate Heatmap
          </button>
        </div>

        <div className="grid gap-4 lg:grid-cols-2">
          <div className="surface-panel">
            <p className="metric-label">Original MRI</p>
            <div className="mt-3 overflow-hidden rounded-xl border border-slate-700/70 bg-slate-950/50">
              {previewUrl ? (
                <img src={previewUrl} alt="Original MRI" className="h-72 w-full bg-slate-950/85 object-contain" />
              ) : (
                <div className="flex h-72 items-center justify-center px-6 text-center text-sm text-slate-400">
                  Upload an MRI image to view the original scan.
                </div>
              )}
            </div>
          </div>

          <div className="surface-panel">
            <p className="metric-label">GradCAM Output</p>
            <div className="mt-3 overflow-hidden rounded-xl border border-slate-700/70 bg-slate-950/50">
              <AnimatePresence mode="wait">
                {gradcamUrl ? (
                  <motion.img
                    key={gradcamUrl}
                    src={gradcamUrl}
                    alt="GradCAM heatmap"
                    initial={{ opacity: 0, scale: 0.98 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0 }}
                    className="h-72 w-full bg-slate-950/85 object-contain"
                  />
                ) : (
                  <motion.div
                    key="placeholder"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="flex h-72 flex-col items-center justify-center gap-3 px-6 text-center"
                  >
                    <Sparkles className="h-8 w-8 text-ann-cyan" />
                    <p className="text-sm text-slate-400">
                    GradCAM heatmap will appear here once the explainability output is generated.
                    </p>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}