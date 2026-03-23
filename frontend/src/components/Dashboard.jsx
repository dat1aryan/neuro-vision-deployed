import { AnimatePresence, motion } from 'framer-motion';
import {
  ActivitySquare,
  BrainCircuit,
  CheckCircle2,
  FileText,
  Microscope,
  ScanSearch,
} from 'lucide-react';


const iconMap = {
  mri: ScanSearch,
  cognitive: BrainCircuit,
  gradcam: Microscope,
  report: FileText,
};


function ProgressDots() {
  return (
    <span className="inline-flex items-center gap-1">
      {[0, 1, 2].map((i) => (
        <motion.span
          key={i}
          className="h-1.5 w-1.5 rounded-full bg-ann-cyan"
          animate={{ opacity: [0.25, 1, 0.25] }}
          transition={{ duration: 1.2, repeat: Infinity, delay: i * 0.22, ease: 'easeInOut' }}
        />
      ))}
    </span>
  );
}


export default function Dashboard({ cards }) {
  return (
    <section className="glass-card px-6 py-7 sm:px-7 sm:py-8">
      <div className="relative z-10">
        <div className="mb-6 flex flex-wrap items-center justify-between gap-4">
          <div>
            <p className="section-kicker">Analysis Pipeline</p>
            <h2 className="section-title mt-2">Neuro Vision Dashboard</h2>
            <p className="section-copy mt-2 max-w-2xl">
              Monitor MRI inference, cognitive risk prediction, explainability generation, and report
              synthesis across the full diagnostic pipeline.
            </p>
          </div>
        </div>

        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
          {cards.map((card, index) => {
            const Icon = iconMap[card.id] || ActivitySquare;
            const isRunning = card.phase === 'running';
            const isCompleted = card.phase === 'completed';

            return (
              <motion.article
                key={card.id}
                initial={{ opacity: 0, y: 18 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.06 }}
                whileHover={{ y: -6 }}
                className={`group hover-lift rounded-3xl border bg-gradient-to-b from-slate-900/72 to-slate-950/72 p-5 shadow-soft transition duration-300 ${
                  isRunning
                    ? 'border-ann-cyan/55 shadow-glow'
                    : isCompleted
                    ? 'border-emerald-500/40 hover:border-emerald-400/60'
                    : 'border-slate-700/70 hover:border-ann-cyan/50 hover:shadow-glow'
                }`}
              >
                <div className="mb-5 flex items-center justify-between">
                  <div className={`rounded-2xl border p-3 transition-colors duration-300 ${
                    isRunning
                      ? 'border-ann-cyan/45 bg-ann-cyan/10'
                      : isCompleted
                      ? 'border-emerald-500/35 bg-emerald-500/10'
                      : 'border-slate-700/60 bg-slate-950/55'
                  }`}>
                    <Icon className={`h-5 w-5 transition-colors duration-300 ${
                      isCompleted ? 'text-emerald-400' : 'text-ann-cyan'
                    }`} />
                  </div>

                  <AnimatePresence mode="wait">
                    {isRunning && (
                      <motion.span
                        key="processing-label"
                        initial={{ opacity: 0, x: 6 }}
                        animate={{ opacity: 1, x: 0 }}
                        exit={{ opacity: 0, x: 6 }}
                        className="metric-label text-ann-cyan"
                      >
                        Processing
                      </motion.span>
                    )}
                    {isCompleted && (
                      <motion.span
                        key="complete-label"
                        initial={{ opacity: 0, x: 6 }}
                        animate={{ opacity: 1, x: 0 }}
                        exit={{ opacity: 0, x: 6 }}
                        className="metric-label text-emerald-400"
                      >
                        Complete
                      </motion.span>
                    )}
                  </AnimatePresence>
                </div>

                <h3 className="font-display text-lg font-semibold text-white">{card.title}</h3>
                <p className="mt-2 text-sm leading-6 text-slate-300">{card.copy}</p>

                <AnimatePresence mode="wait">
                  {isRunning ? (
                    <motion.div
                      key="running"
                      initial={{ opacity: 0, y: 6 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -6 }}
                      transition={{ duration: 0.22 }}
                      className="mt-5 rounded-xl border border-ann-cyan/25 bg-ann-cyan/5 px-3 py-2.5"
                    >
                      <div className="flex items-center gap-2">
                        <p className="text-sm font-medium text-ann-cyan">{card.progressText}</p>
                        <ProgressDots />
                      </div>
                    </motion.div>
                  ) : isCompleted ? (
                    <motion.div
                      key="completed"
                      initial={{ opacity: 0, y: 6 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -6 }}
                      transition={{ duration: 0.22 }}
                      className="mt-5 rounded-xl border border-emerald-500/25 bg-emerald-500/8 px-3 py-2.5"
                    >
                      <div className="flex items-center gap-2">
                        <CheckCircle2 className="h-4 w-4 flex-shrink-0 text-emerald-400" />
                        <p className="text-sm font-semibold text-emerald-300">Analysis Complete</p>
                      </div>
                      {card.value && card.value !== 'Generated' && (
                        <p className="mt-1 text-xs capitalize text-slate-400">{card.value}</p>
                      )}
                    </motion.div>
                  ) : (
                    <motion.div
                      key="idle"
                      initial={{ opacity: 0, y: 6 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -6 }}
                      transition={{ duration: 0.22 }}
                      className="mt-5 space-y-1.5"
                    >
                      <p className="metric-label mb-2">Capabilities</p>
                      {card.capabilities.map((cap) => (
                        <div key={cap} className="flex items-start gap-2 text-xs text-slate-400">
                          <span className="mt-0.5 h-1.5 w-1.5 flex-shrink-0 rounded-full bg-ann-cyan/60" />
                          {cap}
                        </div>
                      ))}
                    </motion.div>
                  )}
                </AnimatePresence>
              </motion.article>
            );
          })}
        </div>
      </div>
    </section>
  );
}