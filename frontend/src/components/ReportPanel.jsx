import { AnimatePresence, motion } from 'framer-motion';
import { Download, FileText, LoaderCircle } from 'lucide-react';


function toPercent(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return 0;
  }
  return numeric * 100;
}


export default function ReportPanel({ report, loading, disabled, onGenerate, onExportPDF, exportingPdf }) {
  return (
    <section className="glass-card h-full p-6 sm:p-7">
      <div className="relative z-10">
        <div className="mb-5 flex items-start justify-between gap-4">
          <div>
            <p className="section-kicker">Clinical Intelligence</p>
            <h2 className="section-title mt-2">AI Brain Analysis Report</h2>
            <p className="section-copy mt-2">
              Neuro Vision combines MRI predictions and cognitive indicators to generate a consolidated AI-driven analysis report.
            </p>
          </div>

          <button type="button" onClick={onGenerate} disabled={disabled || loading} className="neon-button">
            {loading ? <LoaderCircle className="h-4 w-4 animate-spin" /> : <FileText className="h-4 w-4" />}
            Generate AI Report
          </button>
        </div>

        <AnimatePresence mode="wait">
          {report ? (
            <motion.div
              key="report"
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="space-y-4"
            >
              <div className="rounded-2xl border border-ann-cyan/40 bg-gradient-to-r from-ann-indigo/30 via-fuchsia-500/20 to-ann-cyan/25 p-5 shadow-glow">
                <p className="metric-label text-slate-200">Platform</p>
                <h3 className="mt-2 font-display text-xl font-semibold text-white">{report.platform}</h3>
              </div>

              <div className="grid gap-3 sm:grid-cols-3">
                <div className="surface-panel hover-lift">
                  <p className="metric-label">Tumor prediction</p>
                  <p className="mt-2 text-lg font-semibold capitalize text-white">
                    {report.tumor_prediction}
                  </p>
                  <p className="mt-1 text-sm text-slate-300">
                    Model confidence: {toPercent(report.tumor_confidence).toFixed(1)}%
                  </p>
                </div>

                <div className="surface-panel hover-lift">
                  <p className="metric-label">Cognitive risk</p>
                  <p className="mt-2 text-lg font-semibold capitalize text-white">
                    {report.cognitive_risk}
                  </p>
                  <p className="mt-1 text-sm text-slate-300">
                    Cognitive model probability: {toPercent(report.cognitive_model_probability).toFixed(1)}%
                  </p>
                </div>

                <div className="surface-panel hover-lift">
                  <p className="metric-label">Final multimodal risk</p>
                  <p className="mt-2 text-lg font-semibold capitalize text-white">
                    {report.final_risk_category || 'Waiting'}
                  </p>
                  <p className="mt-1 text-sm text-slate-300">
                    Fusion score: {toPercent(report.final_risk_score).toFixed(1)}%
                  </p>
                </div>
              </div>

              <div className="surface-panel">
                <p className="metric-label">Top contributing factors</p>
                {Array.isArray(report.top_contributing_factors) && report.top_contributing_factors.length ? (
                  <div className="mt-2 space-y-2">
                    {report.top_contributing_factors.slice(0, 4).map((factor, index) => (
                      <p key={`${factor.factor || 'factor'}-${index}`} className="text-sm leading-7 text-slate-300">
                        {factor.factor}: {factor.detail}
                      </p>
                    ))}
                  </div>
                ) : (
                  <p className="mt-2 text-sm leading-7 text-slate-300">
                    Top drivers become available after multimodal fusion finishes.
                  </p>
                )}
              </div>

              <div className="surface-panel">
                <p className="metric-label">Summary</p>
                <p className="mt-2 text-sm leading-7 text-slate-300">{report.summary}</p>
              </div>

              {report.clinical_disclaimer ? (
                <div className="rounded-2xl border border-amber-500/30 bg-amber-500/10 p-4">
                  <p className="metric-label text-amber-200">Clinical disclaimer</p>
                  <p className="mt-2 text-sm leading-7 text-amber-100/90">{report.clinical_disclaimer}</p>
                </div>
              ) : null}

              <button
                type="button"
                onClick={onExportPDF}
                disabled={exportingPdf}
                className="neon-button w-full justify-center"
              >
                {exportingPdf
                  ? <LoaderCircle className="h-4 w-4 animate-spin" />
                  : <Download className="h-4 w-4" />}
                <span>{exportingPdf ? 'Generating PDF…' : 'Export to PDF'}</span>
              </button>
            </motion.div>
          ) : (
            <motion.div
              key="placeholder"
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="surface-panel-muted p-5"
            >
              <p className="text-sm leading-7 text-slate-300">
                Generate the Neuro Vision brain health report to display tumor prediction, cognitive
                risk, confidence values, and a fused clinical summary.
              </p>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </section>
  );
}