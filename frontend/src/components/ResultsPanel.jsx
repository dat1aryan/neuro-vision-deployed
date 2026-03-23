import { motion } from 'framer-motion';
import { Brain, ShieldAlert, ShieldCheck } from 'lucide-react';


function ringColor(level) {
  if (level === 'high') {
    return '#f59e0b';
  }
  if (level === 'moderate') {
    return '#fbbf24';
  }
  return '#22d3ee';
}


function normalizeRiskLevel(value) {
  const normalized = String(value || '').toLowerCase();
  if (normalized.includes('very high') || normalized.includes('high')) {
    return 'high';
  }
  if (normalized.includes('moderate')) {
    return 'moderate';
  }
  if (normalized.includes('low')) {
    return 'low';
  }
  return 'unknown';
}


function toPercent(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return 0;
  }
  return numeric * 100;
}


function ProgressRing({ value, color }) {
  const clamped = Math.max(0, Math.min(100, value));
  const radius = 50;
  const circumference = 2 * Math.PI * radius;
  const dashoffset = circumference - (clamped / 100) * circumference;

  return (
    <div className="relative h-32 w-32">
      <svg viewBox="0 0 120 120" className="h-32 w-32">
        <circle cx="60" cy="60" r={radius} fill="none" stroke="rgba(148, 163, 184, 0.18)" strokeWidth="10" />
        <motion.circle
          cx="60"
          cy="60"
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth="10"
          strokeLinecap="round"
          initial={{ strokeDashoffset: circumference }}
          animate={{ strokeDashoffset: dashoffset }}
          transition={{ duration: 0.9, ease: 'easeOut' }}
          style={{ strokeDasharray: circumference }}
          transform="rotate(-90 60 60)"
        />
      </svg>
      <div className="absolute inset-0 flex items-center justify-center text-sm font-semibold text-white">
        {clamped.toFixed(0)}%
      </div>
    </div>
  );
}


function ResultCard({ title, subtitle, value, percent, icon, accentColor }) {
  return (
    <motion.article
      whileHover={{ y: -4 }}
      className="glass-card hover-lift rounded-3xl border-slate-700/70 p-6 sm:p-7"
    >
      <div className="relative z-10 flex items-start justify-between gap-4">
        <div>
          <p className="section-kicker">Diagnostic Results</p>
          <h3 className="section-title mt-2 text-xl sm:text-2xl">{title}</h3>
          <p className="mt-2 text-sm leading-7 text-slate-300">{subtitle}</p>
          <p className="mt-4 font-display text-3xl font-semibold capitalize text-white">{value}</p>
        </div>

        <div className="rounded-2xl border border-slate-700/70 bg-gradient-to-b from-slate-900/75 to-slate-950/75 p-3">{icon}</div>
      </div>

      <div className="relative z-10 mt-5 flex items-center justify-between gap-4">
        <ProgressRing value={percent} color={accentColor} />
        <div className="surface-panel flex-1">
          <p className="metric-label">Confidence Score</p>
          <p className="mt-2 text-sm leading-7 text-slate-300">
            Confidence score updates with each inference run.
          </p>
        </div>
      </div>
    </motion.article>
  );
}


export default function ResultsPanel({ mriResult, cognitiveResult, report }) {
  const tumorPrediction = mriResult?.tumor_prediction || 'Waiting';
  const tumorConfidence = toPercent(mriResult?.confidence);
  const mriUncertainty = toPercent(report?.mri_uncertainty ?? mriResult?.mri_uncertainty);
  const confidenceLevel = String(report?.confidence_level || mriResult?.confidence_level || 'Pending');

  const cognitiveRisk = cognitiveResult?.cognitive_risk || 'Waiting';
  const cognitiveProbability = toPercent(cognitiveResult?.risk_probability);
  const cognitiveRiskLevel = normalizeRiskLevel(cognitiveRisk);

  return (
    <section className="grid gap-6 lg:grid-cols-2">
      <ResultCard
        title="Tumor Prediction"
        subtitle={`Model-based MRI output • Uncertainty ${mriUncertainty.toFixed(1)}% (${confidenceLevel})`}
        value={tumorPrediction}
        percent={tumorConfidence}
        accentColor="#38bdf8"
        icon={<Brain className="h-6 w-6 text-ann-cyan" />}
      />

      <ResultCard
        title="Cognitive Risk"
        subtitle="Model-based cognitive probability"
        value={cognitiveRisk}
        percent={cognitiveProbability}
        accentColor={ringColor(cognitiveRiskLevel)}
        icon={
          cognitiveRiskLevel === 'high' || cognitiveRiskLevel === 'moderate' ? (
            <ShieldAlert className="h-6 w-6 text-amber-300" />
          ) : (
            <ShieldCheck className="h-6 w-6 text-emerald-300" />
          )
        }
      />
    </section>
  );
}