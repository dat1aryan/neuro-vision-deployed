import { motion } from 'framer-motion';
import { BrainCircuit } from 'lucide-react';


export default function Navbar() {
  return (
    <header className="sticky top-0 z-40 border-b border-slate-800/60 bg-ann-bg/85 backdrop-blur-xl before:pointer-events-none before:absolute before:inset-x-0 before:bottom-0 before:h-px before:bg-gradient-to-r before:from-transparent before:via-ann-cyan/40 before:to-transparent">
      <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-4 sm:px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, x: -16 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
          className="flex items-center gap-3"
        >
          <div className="flex h-10 w-10 items-center justify-center rounded-xl border border-slate-700/70 bg-gradient-to-br from-ann-indigo/80 to-ann-cyan/85 shadow-soft">
            <BrainCircuit className="h-5 w-5 text-white" />
          </div>
          <span className="font-display text-lg font-semibold tracking-tight text-white">
            Neuro Vision
          </span>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, x: 16 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
          className="hidden items-center gap-2 rounded-2xl border border-slate-700/60 bg-gradient-to-b from-slate-900/70 to-slate-950/70 px-3 py-1.5 text-xs font-medium text-slate-300 sm:flex"
        >
          <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 shadow-[0_0_6px_rgba(52,211,153,0.8)]" />
          AI Healthcare Platform
        </motion.div>
      </div>
    </header>
  );
}
