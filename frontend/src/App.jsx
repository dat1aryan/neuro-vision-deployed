import { AnimatePresence, motion } from 'framer-motion';
import { CheckCircle2, XCircle } from 'lucide-react';
import { useEffect, useMemo, useRef, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import {
  buildClinicalPayload,
  formatApiError,
  generateAIReport,
  exportToPDF,
  generateGradCAM,
  getBackendStatus,
  predictCognitive,
  predictMRI,
} from './api';
import About from './components/About';
import CognitiveTestPage from './components/CognitiveTestPage';
import CognitiveForm from './components/CognitiveForm';
import Dashboard from './components/Dashboard';
import GradCAMView from './components/GradCAMView';
import Hero from './components/Hero';
import MRIUpload from './components/MRIUpload';
import Navbar from './components/Navbar';
import ReportPanel from './components/ReportPanel';
import ResultsPanel from './components/ResultsPanel';
import Team from './components/Team';


const initialClinicalState = {
  age: '72',
  education: '14',
  memoryScore: '18',
  cognitiveScore: '22',
};

const reveal = {
  hidden: { opacity: 0, y: 24 },
  show: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.55, ease: [0.22, 1, 0.36, 1] },
  },
};


function toNumber(value, fallback = 0) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}


function normalizeMRIResult(data) {
  const prediction = String(data?.tumor_prediction || data?.prediction || data?.tumor_type || '').toLowerCase();
  const confidence = toNumber(data?.tumor_confidence ?? data?.confidence);
  return {
    tumor_prediction: prediction,
    confidence,
    raw_confidence: toNumber(data?.raw_confidence),
    mri_uncertainty: toNumber(data?.mri_uncertainty ?? data?.uncertainty_score),
    uncertainty_score: toNumber(data?.uncertainty_score ?? data?.mri_uncertainty),
    uncertainty_variance: toNumber(data?.uncertainty_variance),
    confidence_level: String(data?.confidence_level || ''),
    tumor_probability: toNumber(data?.tumor_probability),
    class_probabilities: data?.class_probabilities || {},
  };
}


function normalizeCognitiveResult(data) {
  const cognitiveRisk = String(data?.cognitive_risk || data?.risk_label || '').toLowerCase();
  const riskProbability = toNumber(data?.risk_score ?? data?.risk_probability);
  return {
    cognitive_risk: cognitiveRisk,
    risk_probability: riskProbability,
    risk_percentage: toNumber(data?.risk_percentage),
    model_probability: toNumber(data?.model_probability ?? data?.cognitive_model_probability),
    hybrid_components: data?.hybrid_components || {},
    module_deficit_breakdown: data?.module_deficit_breakdown || {},
    module_reliability_flag: String(data?.module_reliability_flag || ''),
    top_contributing_factors: Array.isArray(data?.top_contributing_factors)
      ? data.top_contributing_factors
      : [],
  };
}

function getFileCacheKey(file) {
  if (!file) {
    return '';
  }
  return [file.name, file.size, file.lastModified, file.type].join('::');
}


function App() {
  const dashboardRef = useRef(null);
  const location = useLocation();
  const navigate = useNavigate();

  const [backendStatus, setBackendStatus] = useState({
    state: 'checking',
    message: 'Checking Neuro Vision backend...',
  });
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState('');
  const [clinicalData, setClinicalData] = useState(initialClinicalState);
  const [mriResult, setMriResult] = useState(null);
  const [cognitiveResult, setCognitiveResult] = useState(null);
  const [gradcamUrl, setGradcamUrl] = useState('');
  const [report, setReport] = useState(null);
  const [lastResult, setLastResult] = useState(null);
  const [toasts, setToasts] = useState([]);
  const [loading, setLoading] = useState({
    mri: false,
    cognitive: false,
    gradcam: false,
    report: false,
    exportPdf: false,
  });

  useEffect(() => {
    let active = true;

    async function probeBackend() {
      try {
        const data = await getBackendStatus();
        if (!active) {
          return;
        }

        setBackendStatus({
          state: 'online',
          message: data.message || 'Neuro Vision — Connected',
        });
      } catch (error) {
        if (!active) {
          return;
        }

        setBackendStatus({
          state: 'offline',
          message: formatApiError(error, 'Neuro Vision backend unreachable'),
        });
      }
    }

    probeBackend();
    return () => {
      active = false;
    };
  }, []);

  useEffect(
    () => () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    },
    [previewUrl],
  );

  useEffect(() => {
    const screeningResult = location.state?.cognitiveTestResult;
    if (!screeningResult) {
      return;
    }

    const memoryScore = Number(screeningResult.memory_score);
    const cognitiveScore = Number(screeningResult.cognitive_score);

    if (Number.isFinite(memoryScore) && Number.isFinite(cognitiveScore)) {
      const reliabilityFlag = String(screeningResult.reliability_flag || '').toLowerCase();
      const reliabilityMessage = String(screeningResult.reliability_message || '').trim();

      setClinicalData((current) => ({
        ...current,
        memoryScore: String(Math.max(0, Math.min(25, Math.round(memoryScore)))),
        cognitiveScore: String(Math.max(0, Math.min(30, Math.round(cognitiveScore)))),
      }));

      setCognitiveResult(null);
      setReport(null);

      notify(
        'success',
        'Cognitive screening complete',
        reliabilityFlag === 'low' && reliabilityMessage
          ? `Scores were auto-filled, but reliability is low. ${reliabilityMessage}`
          : 'Memory Score and Cognitive Score were auto-filled. You can now click Predict Cognitive Risk.',
      );

      window.setTimeout(() => {
        dashboardRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }, 120);
    }

    navigate('/', { replace: true, state: null });
  }, [location.state, navigate]);

  useEffect(() => {
    console.log('Heatmap state:', gradcamUrl);
  }, [gradcamUrl]);

  const dashboardCards = useMemo(
    () => {
      function getPhase(isLoading, hasResult) {
        if (isLoading) return 'running';
        if (hasResult) return 'completed';
        return 'idle';
      }

      return [
        {
          id: 'mri',
          title: 'MRI Analysis',
          copy: 'Deep learning tumor classification with confidence scoring.',
          capabilities: [
            'Glioma detection',
            'Meningioma detection',
            'Pituitary tumor detection',
            'No tumor classification',
          ],
          progressText: 'Analyzing MRI scan',
          phase: getPhase(loading.mri || loading.report, mriResult),
          value: mriResult ? mriResult.tumor_prediction : '',
        },
        {
          id: 'cognitive',
          title: 'Cognitive Risk',
          copy: 'Clinical indicator fusion for neurological risk estimation.',
          capabilities: [
            'Age & education scoring',
            'Memory assessment',
            'Functional analysis',
            'Risk probability scoring',
          ],
          progressText: 'Evaluating cognitive indicators',
          phase: getPhase(loading.cognitive || loading.report, cognitiveResult),
          value: cognitiveResult ? cognitiveResult.cognitive_risk : '',
        },
        {
          id: 'gradcam',
          title: 'AI Explainability',
          copy: 'GradCAM heatmap highlighting regions driving classification.',
          capabilities: [
            'GradCAM heatmap generation',
            'Region-of-interest mapping',
            'Activation visualization',
            'Overlay rendering',
          ],
          progressText: 'Generating GradCAM heatmap',
          phase: getPhase(loading.gradcam || loading.report, gradcamUrl),
          value: gradcamUrl ? 'Generated' : '',
        },
        {
          id: 'report',
          title: 'Brain Health Report',
          copy: 'Consolidated AI-driven analysis combining all modalities.',
          capabilities: [
            'Multimodal result fusion',
            'Natural language summary',
            'Confidence breakdown',
            'Consolidated diagnostics',
          ],
          progressText: 'Compiling multimodal AI report',
          phase: getPhase(loading.report, report),
          value: report ? 'Generated' : '',
        },
      ];
    },
    [mriResult, cognitiveResult, gradcamUrl, report, loading],
  );

  function updateLoadingState(key, value) {
    setLoading((current) => ({ ...current, [key]: value }));
  }

  function dismissToast(id) {
    setToasts((current) => current.filter((toast) => toast.id !== id));
  }

  function notify(kind, title, message) {
    const id = globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random()}`;
    setToasts((current) => [...current, { id, kind, title, message }]);

    window.setTimeout(() => {
      dismissToast(id);
    }, 4200);
  }

  function scrollToDashboard() {
    dashboardRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  function handleFileSelect(file) {
    if (!file) {
      return;
    }

    if (!file.type.startsWith('image/')) {
      notify('error', 'Invalid MRI file', 'Please upload a valid image file.');
      return;
    }

    setSelectedFile(file);
    setUploadedFile(file);
    setReport(null);
    setGradcamUrl('');
    setMriResult(null);

    setPreviewUrl((currentUrl) => {
      if (currentUrl) {
        URL.revokeObjectURL(currentUrl);
      }
      return URL.createObjectURL(file);
    });

    notify('success', 'MRI uploaded', `${file.name} is ready for analysis.`);
  }

  function handleClinicalChange(field, value) {
    setClinicalData((current) => ({ ...current, [field]: value }));
  }

  async function handleAnalyzeMRI() {
    if (loading.mri) {
      return;
    }

    if (!selectedFile) {
      notify('error', 'MRI required', 'Upload an MRI scan before running analysis.');
      return;
    }

    const fileKey = getFileCacheKey(selectedFile);
    if (lastResult?.fileKey === fileKey && lastResult?.mri) {
      setMriResult(lastResult.mri);
      notify('success', 'MRI analysis complete', `Tumor prediction: ${lastResult.mri.tumor_prediction}.`);
      return;
    }

    updateLoadingState('mri', true);

    const predictionPromise = predictMRI(selectedFile);

    // Start GradCAM in parallel to keep UI responsive while prediction returns first.
    const shouldPrefetchHeatmap = !(lastResult?.fileKey === fileKey && lastResult?.heatmapData);
    const heatmapPromise = shouldPrefetchHeatmap ? generateGradCAM(selectedFile) : Promise.resolve(null);

    if (shouldPrefetchHeatmap && !loading.gradcam) {
      updateLoadingState('gradcam', true);
      heatmapPromise
        .then((heatmapData) => {
          if (!heatmapData) {
            return;
          }
          setLastResult((current) => ({
            ...(current || {}),
            fileKey,
            heatmapData,
          }));
        })
        .catch(() => {
          // Ignore background prefetch errors and keep explicit heatmap action unchanged.
        })
        .finally(() => {
          updateLoadingState('gradcam', false);
        });
    }

    try {
      const [data] = await Promise.all([predictionPromise]);
      const normalized = normalizeMRIResult(data);
      setMriResult(normalized);
      setLastResult((current) => ({
        ...(current || {}),
        fileKey,
        mri: normalized,
      }));
      notify('success', 'MRI analysis complete', `Tumor prediction: ${normalized.tumor_prediction}.`);
    } catch (error) {
      notify('error', 'MRI analysis failed', formatApiError(error, 'Unable to analyze MRI scan.'));
    } finally {
      updateLoadingState('mri', false);
    }
  }

  async function handlePredictCognitive() {
    if (loading.cognitive) {
      return;
    }

    const payload = buildClinicalPayload(clinicalData);
    if (!Object.keys(payload).length) {
      notify('error', 'Clinical data required', 'Fill in clinical values before prediction.');
      return;
    }

    updateLoadingState('cognitive', true);
    try {
      const data = await predictCognitive(payload);
      const normalized = normalizeCognitiveResult(data);
      setCognitiveResult(normalized);
      notify('success', 'Cognitive prediction complete', `Risk level: ${normalized.cognitive_risk}.`);
    } catch (error) {
      notify(
        'error',
        'Cognitive prediction failed',
        formatApiError(error, 'Unable to run cognitive risk prediction.'),
      );
    } finally {
      updateLoadingState('cognitive', false);
    }
  }

  async function handleGenerateGradCAM() {
    if (loading.gradcam) {
      return;
    }

    if (!selectedFile) {
      notify('error', 'MRI required', 'Upload an MRI scan before generating GradCAM.');
      return;
    }

    const fileKey = getFileCacheKey(selectedFile);
    if (lastResult?.fileKey === fileKey && lastResult?.heatmapData) {
      setMriResult(normalizeMRIResult(lastResult.heatmapData));
      setGradcamUrl(lastResult.heatmapData.imageUrl);
      notify('success', 'GradCAM ready', 'Heatmap loaded from cache.');
      return;
    }

    updateLoadingState('gradcam', true);
    try {
      const data = await generateGradCAM(selectedFile);
      setMriResult(normalizeMRIResult(data));
      setGradcamUrl(data.imageUrl);
      setLastResult((current) => ({
        ...(current || {}),
        fileKey,
        heatmapData: data,
      }));
      notify('success', 'GradCAM ready', 'Heatmap generated successfully.');
    } catch (error) {
      notify('error', 'GradCAM generation failed', formatApiError(error, 'Unable to generate GradCAM heatmap.'));
    } finally {
      updateLoadingState('gradcam', false);
    }
  }

  async function handleExportPDF() {
    if (!report || loading.exportPdf) return;
    setLoading(prev => ({ ...prev, exportPdf: true }));
    try {
      await exportToPDF({
        ...report,
        gradcam_image: gradcamUrl || null,
      });
    } catch (err) {
      console.error('PDF export failed:', err);
    } finally {
      setLoading(prev => ({ ...prev, exportPdf: false }));
    }
  }

  async function handleGenerateReport() {
    if (loading.report) {
      return;
    }

    if (!uploadedFile) {
      notify('error', 'MRI required', 'Upload an MRI scan before generating the report.');
      return;
    }

    if (!mriResult || !cognitiveResult) {
      notify(
        'error',
        'Run predictions first',
        'Please run MRI Analysis and Cognitive Risk Prediction before generating the report.',
      );
      return;
    }

    const payload = buildClinicalPayload(clinicalData);
    if (!Object.keys(payload).length) {
      notify('error', 'Clinical data required', 'Fill in clinical values before generating report.');
      return;
    }

    updateLoadingState('report', true);
    try {
      const data = await generateAIReport(uploadedFile, payload);

      // Keep report aligned with the already computed MRI and cognitive assessments.
      const stabilizedReport = {
        ...data,
        tumor_prediction: mriResult.tumor_prediction,
        confidence: mriResult.confidence,
        tumor_confidence: mriResult.confidence,
        raw_confidence: mriResult.raw_confidence,
        mri_uncertainty: mriResult.mri_uncertainty,
        uncertainty_score: mriResult.uncertainty_score,
        confidence_level: mriResult.confidence_level,
        tumor_probability: mriResult.tumor_probability,
        class_probabilities: mriResult.class_probabilities || {},
        cognitive_risk: cognitiveResult.cognitive_risk,
        risk_probability: cognitiveResult.risk_probability,
        risk_score: cognitiveResult.risk_probability,
        cognitive_model_probability: cognitiveResult.model_probability,
        gradcam_image: gradcamUrl || data.gradcam_image || data.imageUrl || null,
      };

      setReport(stabilizedReport);
      notify('success', 'Report ready', 'Neuro Vision brain health report is ready.');
    } catch (error) {
      notify('error', 'Report generation failed', formatApiError(error, 'Unable to generate AI report.'));
    } finally {
      updateLoadingState('report', false);
    }
  }

  if (location.pathname === '/cognitive-test') {
    return <CognitiveTestPage />;
  }

  return (
    <div className="relative min-h-screen overflow-x-hidden bg-ann-bg text-slate-100">
      <div className="pointer-events-none fixed inset-0 -z-10">
        <motion.div
          animate={{ x: [0, 24, 0], y: [0, -18, 0] }}
          transition={{ duration: 10, repeat: Infinity, ease: 'easeInOut' }}
          className="absolute left-[-8rem] top-[-6rem] h-80 w-80 rounded-full bg-ann-indigo/30 blur-3xl"
        />
        <motion.div
          animate={{ x: [0, -20, 0], y: [0, 16, 0] }}
          transition={{ duration: 12, repeat: Infinity, ease: 'easeInOut' }}
          className="absolute right-[-8rem] top-[20%] h-80 w-80 rounded-full bg-ann-cyan/25 blur-3xl"
        />
      </div>

      <AnimatePresence>
        {toasts.length ? (
          <div className="fixed right-4 top-4 z-50 flex w-full max-w-sm flex-col gap-3" aria-live="polite" aria-atomic="false">
            {toasts.map((toast) => (
              <motion.div
                key={toast.id}
                layout
                initial={{ opacity: 0, x: 36, scale: 0.96 }}
                animate={{ opacity: 1, x: 0, scale: 1 }}
                exit={{ opacity: 0, x: 28, scale: 0.98 }}
                role="status"
                className={`glass-card px-4 py-4 ${
                  toast.kind === 'error'
                    ? 'border-rose-400/45'
                    : 'border-emerald-400/40'
                }`}
              >
                <div className="relative z-10 flex items-start gap-3">
                  {toast.kind === 'error' ? (
                    <XCircle className="mt-0.5 h-5 w-5 text-rose-400" />
                  ) : (
                    <CheckCircle2 className="mt-0.5 h-5 w-5 text-emerald-400" />
                  )}
                  <div className="min-w-0 flex-1">
                    <p className="font-display text-sm font-semibold text-white">{toast.title}</p>
                    <p className="mt-1 text-sm text-slate-300">{toast.message}</p>
                  </div>
                  <button
                    type="button"
                    onClick={() => dismissToast(toast.id)}
                    className="rounded-full p-1 text-slate-400 transition hover:bg-slate-700/40 hover:text-white"
                  >
                    <span className="sr-only">Dismiss</span>
                    <svg viewBox="0 0 20 20" className="h-4 w-4 fill-current">
                      <path d="M5.22 5.22a.75.75 0 0 1 1.06 0L10 8.94l3.72-3.72a.75.75 0 1 1 1.06 1.06L11.06 10l3.72 3.72a.75.75 0 1 1-1.06 1.06L10 11.06l-3.72 3.72a.75.75 0 1 1-1.06-1.06L8.94 10 5.22 6.28a.75.75 0 0 1 0-1.06Z" />
                    </svg>
                  </button>
                </div>
              </motion.div>
            ))}
          </div>
        ) : null}
      </AnimatePresence>

      <Navbar />
      <Hero backendStatus={backendStatus} onStartAnalysis={scrollToDashboard} />

      <main ref={dashboardRef} className="mx-auto flex w-full max-w-7xl flex-col gap-7 px-4 pb-16 sm:px-6 lg:px-8">
        <motion.section variants={reveal} initial="hidden" whileInView="show" viewport={{ once: true, amount: 0.2 }}>
          <Dashboard cards={dashboardCards} />
        </motion.section>

        <div className="grid gap-6 xl:grid-cols-[1.1fr_0.9fr]">
          <motion.section variants={reveal} initial="hidden" whileInView="show" viewport={{ once: true, amount: 0.2 }}>
            <MRIUpload
              selectedFile={selectedFile}
              previewUrl={previewUrl}
              result={mriResult}
              loading={loading.mri}
              onAnalyze={handleAnalyzeMRI}
              onFileSelect={handleFileSelect}
            />
          </motion.section>

          <motion.section variants={reveal} initial="hidden" whileInView="show" viewport={{ once: true, amount: 0.2 }}>
            <CognitiveForm
              values={clinicalData}
              result={cognitiveResult}
              loading={loading.cognitive}
              onFieldChange={handleClinicalChange}
              onSubmit={handlePredictCognitive}
            />
          </motion.section>
        </div>

        <motion.section variants={reveal} initial="hidden" whileInView="show" viewport={{ once: true, amount: 0.2 }}>
          <ResultsPanel mriResult={mriResult} cognitiveResult={cognitiveResult} report={report} />
        </motion.section>

        <div className="grid items-start gap-6 xl:grid-cols-[1.05fr_0.95fr]">
          <motion.section variants={reveal} initial="hidden" whileInView="show" viewport={{ once: true, amount: 0.2 }}>
            <GradCAMView
              previewUrl={previewUrl}
              gradcamUrl={gradcamUrl}
              loading={loading.gradcam}
              disabled={!selectedFile}
              onGenerate={handleGenerateGradCAM}
            />
          </motion.section>

          <motion.section variants={reveal} initial="hidden" whileInView="show" viewport={{ once: true, amount: 0.2 }}>
            <ReportPanel
              report={report}
              heatmap={gradcamUrl}
              loading={loading.report}
              disabled={!selectedFile}
              onGenerate={handleGenerateReport}
              onExportPDF={handleExportPDF}
              exportingPdf={loading.exportPdf}
            />
          </motion.section>
        </div>

        <motion.section variants={reveal} initial="hidden" whileInView="show" viewport={{ once: true, amount: 0.2 }}>
          <About />
        </motion.section>

        <motion.section variants={reveal} initial="hidden" whileInView="show" viewport={{ once: true, amount: 0.2 }}>
          <Team />
        </motion.section>
      </main>

      <footer className="border-t border-slate-800/80 bg-slate-950/40 px-4 py-8 text-center">
        <p className="text-sm font-medium text-slate-400">
          <span className="gradient-text font-semibold">Neuro Vision</span>
          {' '}— AI-powered brain health analysis platform
        </p>
        <p className="mt-1 text-xs text-slate-600">MRI tumor detection · Cognitive risk assessment · Explainable AI</p>
      </footer>
    </div>
  );
}


export default App;