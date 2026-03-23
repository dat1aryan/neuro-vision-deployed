import { motion } from 'framer-motion';
import { BrainCircuit, GraduationCap, MemoryStick, UserRound } from 'lucide-react';
import { Link } from 'react-router-dom';


const fields = [
  {
    key: 'age',
    label: 'Age',
    placeholder: 'Example: 65',
    helperText: 'Enter the current age in years. The trained cognitive dataset mainly covers ages 60 to 90, so values outside that range are clipped before inference.',
    min: 18,
    max: 120,
    icon: UserRound,
  },
  {
    key: 'education',
    label: 'Years of Formal Education',
    placeholder: "Example: 12 (High School), 16 (Bachelor's)",
    helperText: 'Enter years of schooling. The backend maps this to the training dataset\'s four education levels before prediction.',
    min: 0,
    max: 25,
    icon: GraduationCap,
  },
  {
    key: 'memoryScore',
    label: 'Memory Score',
    placeholder: 'Example: 18',
    helperText: 'Score obtained from the 10-minute cognitive memory test. It is aligned to the trained MMSE scale before inference.',
    min: 0,
    max: 25,
    icon: MemoryStick,
  },
  {
    key: 'cognitiveScore',
    label: 'Cognitive Score',
    placeholder: 'Example: 22',
    helperText: 'Score obtained from the full cognitive test evaluating attention, reasoning, and processing speed. It is aligned to the trained functional-assessment scale before inference.',
    min: 0,
    max: 30,
    icon: BrainCircuit,
  },
];


function Spinner() {
  return (
    <svg className="h-4 w-4 animate-spin" viewBox="0 0 24 24" fill="none">
      <circle cx="12" cy="12" r="10" stroke="currentColor" strokeOpacity="0.2" strokeWidth="4" />
      <path d="M22 12a10 10 0 0 0-10-10" stroke="currentColor" strokeWidth="4" strokeLinecap="round" />
    </svg>
  );
}


export default function CognitiveForm({ values, result, loading, onFieldChange, onSubmit }) {
  const probabilityPercent = Number(result?.risk_probability || 0) * 100;

  return (
    <section className="glass-card h-full p-6 sm:p-7">
      <div className="relative z-10">
        <div className="mb-6">
          <p className="section-kicker">Clinical Risk Assessment</p>
          <h2 className="section-title mt-2">Cognitive Health Assessment</h2>
          <p className="section-copy mt-2">
            Enter the four indicators used by the deployed cognitive model. Education years and test scores are aligned to the training schema before prediction.
          </p>
        </div>

        <div className="grid gap-4 sm:grid-cols-2">
          {fields.map((field, index) => {
            const Icon = field.icon;
            return (
              <motion.label
                key={field.key}
                initial={{ opacity: 0, y: 12 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.05 }}
                className="block"
              >
                <span className="metric-label">{field.label}</span>
                <div className="relative mt-2">
                  <Icon className="pointer-events-none absolute left-3.5 top-3.5 h-4 w-4 text-slate-400" />
                  <input
                    value={values[field.key]}
                    onChange={(event) => onFieldChange(field.key, event.target.value)}
                    onKeyDown={(event) => {
                      if (['e', 'E', '+', '-', '.'].includes(event.key)) {
                        event.preventDefault();
                      }
                    }}
                    type="number"
                    inputMode="numeric"
                    min={field.min}
                    max={field.max}
                    step="1"
                    placeholder={field.placeholder}
                    className="field-input pl-11 placeholder:text-slate-500"
                  />
                </div>
                <p className="mt-2 text-xs text-slate-400">{field.helperText}</p>
              </motion.label>
            );
          })}
        </div>

        <div className="surface-panel mt-5">
          <p className="text-sm text-slate-300">
            Don&apos;t know your scores? Take the 10-minute cognitive test.
          </p>
          <Link
            to="/cognitive-test"
            className="mt-3 inline-flex items-center text-sm font-semibold text-ann-cyan underline decoration-2 underline-offset-4 transition hover:text-cyan-200"
          >
            Take Cognitive Test
          </Link>
        </div>

        <div className="surface-panel mt-5">
          <div className="mb-2 flex items-center justify-between">
            <p className="metric-label">Risk probability</p>
            <p className="text-xs font-semibold text-ann-cyan">{probabilityPercent.toFixed(1)}%</p>
          </div>
          <div className="h-2 rounded-full bg-slate-800">
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${Math.max(0, Math.min(probabilityPercent, 100))}%` }}
              transition={{ duration: 0.7, ease: 'easeOut' }}
              className="h-full rounded-full bg-gradient-to-r from-ann-indigo to-ann-cyan"
            />
          </div>
          <p className="mt-3 text-sm text-slate-300">
            {result
              ? `Cognitive risk level: ${result.cognitive_risk}`
              : 'Run the assessment to visualize the cognitive risk score.'}
          </p>
        </div>

        <div className="mt-6 flex justify-end">
          <button type="button" onClick={onSubmit} disabled={loading} className="neon-button">
            {loading ? <Spinner /> : null}
            Predict Cognitive Risk
          </button>
        </div>
      </div>
    </section>
  );
}