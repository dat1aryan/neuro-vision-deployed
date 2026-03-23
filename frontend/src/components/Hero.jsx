import { motion } from 'framer-motion';
import { ArrowDown, BrainCircuit, Sparkles } from 'lucide-react';
import { useState } from 'react';

export default function Hero({ onStartAnalysis }) {
  const [logoError, setLogoError] = useState(false);

  return (
    <section className="relative mb-8 overflow-hidden px-4 pb-12 pt-7 sm:px-6 lg:px-8">
      <div className="mx-auto max-w-7xl">
        <motion.div
          initial={{ opacity: 0, y: 26 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
          className="glass-card overflow-hidden rounded-[2rem] px-6 py-10 shadow-glow sm:px-10 sm:py-12"
        >
          <div className="absolute inset-0 bg-hero-gradient opacity-35" />
          <div className="absolute inset-0 bg-[length:200%_100%] opacity-45 animate-shimmer [background-image:linear-gradient(120deg,rgba(99,102,241,0.22)_0%,rgba(34,211,238,0.14)_35%,rgba(99,102,241,0.22)_70%)]" />
          <motion.div
            animate={{ x: [0, 20, 0], y: [0, -16, 0] }}
            transition={{ duration: 9, repeat: Infinity, ease: 'easeInOut' }}
            className="absolute -right-14 -top-14 h-56 w-56 rounded-full bg-ann-cyan/20 blur-3xl"
          />

          <div className="relative z-10 grid items-center gap-8 lg:grid-cols-[1.15fr_0.85fr]">
            <div>
              <p className="section-kicker mb-4">AI Neurological Analysis Platform</p>

              <h1 className="font-display text-[3rem] font-bold leading-[1.1] tracking-tight text-white sm:text-[3.5rem] lg:text-[4rem]">
                <span className="gradient-text">Neuro Vision</span>
              </h1>
              <p className="mt-5 max-w-2xl text-lg font-medium leading-8 text-slate-200">
                AI-powered brain health analysis combining MRI diagnostics, cognitive assessment, and explainable AI.
              </p>
              <p className="mt-4 max-w-2xl text-base leading-8 text-slate-400">
                Neuro Vision brings together advanced deep learning and cognitive analysis to assist early neurological diagnosis. By combining MRI tumor detection, cognitive risk evaluation, and explainable AI visualization, Neuro Vision provides a unified platform for intelligent brain health insights.
              </p>

              <div className="mt-6 flex flex-wrap gap-3">
                <button type="button" onClick={onStartAnalysis} className="neon-button">
                  Start Analysis
                  <ArrowDown className="h-4 w-4" />
                </button>

                <div className="ghost-button cursor-default">
                  <Sparkles className="h-4 w-4 text-ann-cyan" />
                  Explainable AI Ready
                </div>
              </div>
            </div>

            <div className="relative">
              <motion.div
                animate={{ scale: [1, 1.04, 1], opacity: [0.55, 0.9, 0.55] }}
                transition={{ duration: 4, repeat: Infinity, ease: 'easeInOut' }}
                className="mx-auto flex h-64 w-64 items-center justify-center rounded-full border border-ann-cyan/40 bg-slate-900/55 shadow-glow"
              >
                <motion.div
                  animate={{ rotate: [0, 360] }}
                  transition={{ duration: 18, repeat: Infinity, ease: 'linear' }}
                  className="absolute inset-4 rounded-full border border-dashed border-ann-highlight/40"
                />
                {!logoError ? (
                  <img
                    src="/assets/neuro_vision_logo.png"
                    alt="Neuro Vision"
                    className="h-48 w-48 rounded-full object-contain"
                    onError={() => setLogoError(true)}
                  />
                ) : (
                  <BrainCircuit className="h-24 w-24 text-ann-cyan" />
                )}
              </motion.div>

              <div className="mx-auto mt-5 max-w-xs rounded-2xl border border-slate-700/80 bg-slate-900/70 px-4 py-3 text-sm leading-7 text-slate-300">
                Neuro Vision unifies MRI tumor detection, cognitive risk scoring, and explainable AI
                into one seamless diagnostic experience.
              </div>
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  );
}