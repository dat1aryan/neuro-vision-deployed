import { motion } from 'framer-motion';
import { Brain, Orbit } from 'lucide-react';


export default function About() {
  return (
    <section className="glass-card p-6 sm:p-7">
      <div className="relative z-10 grid gap-6 lg:grid-cols-[1.05fr_0.95fr]">
        <div>
          <p className="section-kicker">About the Platform</p>
          <h2 className="section-title mt-2">About Neuro Vision</h2>
          <p className="mt-4 text-base leading-8 text-slate-300">
            Neuro Vision is an AI-powered healthcare platform designed to assist early neurological
            diagnosis by combining MRI tumor detection with cognitive risk prediction and explainable
            AI visualization. By unifying these modalities, Neuro Vision delivers actionable brain
            health insights for clinicians and researchers.
          </p>
        </div>

        <motion.div
          whileHover={{ y: -4 }}
          className="hover-lift rounded-3xl border border-slate-700/70 bg-gradient-to-b from-slate-900/70 to-slate-950/70 p-6"
        >
          <div className="grid gap-4 sm:grid-cols-2">
            <div className="surface-panel">
              <Brain className="h-6 w-6 text-ann-cyan" />
              <p className="mt-3 text-sm font-semibold text-white">MRI Intelligence</p>
              <p className="mt-1 text-sm text-slate-300">Tumor pattern analysis with transfer learning.</p>
            </div>

            <div className="surface-panel">
              <Orbit className="h-6 w-6 text-ann-indigo" />
              <p className="mt-3 text-sm font-semibold text-white">Cognitive Fusion</p>
              <p className="mt-1 text-sm text-slate-300">Clinical marker scoring with risk probability.</p>
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  );
}