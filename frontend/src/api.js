import axios from 'axios';

function normalizeBaseUrl(url) {
  return String(url || '').trim().replace(/\/$/, '');
}

const DEFAULT_PRIMARY_API_URL = 'https://api.neuro-vision.me';
const DEFAULT_FALLBACK_API_URL = 'https://neuro-vision-deployed.onrender.com';

const envPrimaryUrl =
  import.meta.env.NEXT_PUBLIC_API_URL ||
  import.meta.env.VITE_API_URL ||
  DEFAULT_PRIMARY_API_URL;
const envFallbackUrl =
  import.meta.env.NEXT_PUBLIC_API_FALLBACK_URL ||
  import.meta.env.VITE_API_FALLBACK_URL ||
  DEFAULT_FALLBACK_API_URL;

const PRIMARY_API_URL = normalizeBaseUrl(envPrimaryUrl);
const FALLBACK_API_URL = normalizeBaseUrl(envFallbackUrl);

const API_BASE_URLS = [...new Set([PRIMARY_API_URL, FALLBACK_API_URL].filter(Boolean))];
let activeApiBaseUrl = PRIMARY_API_URL;

export const API_BASE_URL = PRIMARY_API_URL;
const REQUEST_TIMEOUT_MS = 25000;
const RETRY_COUNT = 1;

const api = axios.create({
  baseURL: PRIMARY_API_URL || undefined,
  timeout: 60000,
});

function shouldFallbackStatus(status) {
  return !status || status >= 500;
}

function shouldFallbackAxiosError(error) {
  if (!error.response) {
    return true;
  }
  return shouldFallbackStatus(error.response.status);
}

api.interceptors.response.use(
  (response) => {
    const responseBase = normalizeBaseUrl(response.config?.baseURL || PRIMARY_API_URL);
    if (responseBase) {
      activeApiBaseUrl = responseBase;
    }
    return response;
  },
  async (error) => {
    const originalConfig = error?.config || {};
    const requestBase = normalizeBaseUrl(originalConfig.baseURL || PRIMARY_API_URL);
    const canRetryOnFallback =
      Boolean(FALLBACK_API_URL) &&
      requestBase === PRIMARY_API_URL &&
      !originalConfig.__isFallbackRetry &&
      shouldFallbackAxiosError(error);

    if (!canRetryOnFallback) {
      return Promise.reject(error);
    }

    try {
      const fallbackResponse = await api.request({
        ...originalConfig,
        baseURL: FALLBACK_API_URL,
        __isFallbackRetry: true,
      });
      activeApiBaseUrl = FALLBACK_API_URL;
      return fallbackResponse;
    } catch (fallbackError) {
      return Promise.reject(fallbackError);
    }
  },
);

async function safeFetch(fn, retries = RETRY_COUNT) {
  try {
    return await fn();
  } catch (error) {
    if (retries > 0) {
      return safeFetch(fn, retries - 1);
    }
    throw error;
  }
}

async function fetchWithTimeout(url, options = {}, timeoutMs = REQUEST_TIMEOUT_MS) {
  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => controller.abort(), timeoutMs);

  try {
    return await fetch(url, {
      ...options,
      signal: controller.signal,
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') {
      throw new Error('Request timed out. Please try again.');
    }
    throw error;
  } finally {
    window.clearTimeout(timeoutId);
  }
}

function toNumber(value, fallback = 0) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function withApiBase(path) {
  if (!activeApiBaseUrl) {
    return path;
  }
  return `${activeApiBaseUrl}${path}`;
}

function buildApiUrl(baseUrl, path) {
  const normalizedPath = path.startsWith('/') ? path : `/${path}`;
  return `${baseUrl}${normalizedPath}`;
}

async function fetchWithFallback(path, options = {}) {
  let lastError;

  for (let index = 0; index < API_BASE_URLS.length; index += 1) {
    const baseUrl = API_BASE_URLS[index];
    const isLastBase = index === API_BASE_URLS.length - 1;

    try {
      const response = await fetchWithTimeout(buildApiUrl(baseUrl, path), options);

      if (response.ok || isLastBase || !shouldFallbackStatus(response.status)) {
        if (response.ok) {
          activeApiBaseUrl = baseUrl;
        }
        return response;
      }
    } catch (error) {
      lastError = error;
      if (isLastBase) {
        throw error;
      }
    }
  }

  if (lastError) {
    throw lastError;
  }

  return fetchWithTimeout(buildApiUrl(PRIMARY_API_URL, path), options);
}

function resolveHeatmapUrl(heatmapValue) {
  const normalized = String(heatmapValue || '').trim();
  if (!normalized) {
    return '';
  }

  if (normalized.startsWith('data:image/')) {
    return normalized;
  }

  if (/^https?:\/\//i.test(normalized)) {
    const cacheBust = `ts=${Date.now()}`;
    return `${normalized}${normalized.includes('?') ? '&' : '?'}${cacheBust}`;
  }

  const publicPath = normalized.startsWith('/') ? normalized : `/${normalized}`;
  return withApiBase(publicPath);
}

export async function getBackendStatus() {
  const { data } = await api.get('/');
  return data;
}

export async function predictMRI(file) {
  const formData = new FormData();
  formData.append('file', file);

  const response = await safeFetch(() =>
    fetchWithFallback('/api/prediction/mri', {
      method: 'POST',
      body: formData,
    }),
  );

  if (!response.ok) {
    let detail = 'MRI request failed';
    try {
      const payload = await response.json();
      detail = payload?.detail || payload?.message || detail;
    } catch {
      // Keep default error detail when response is not JSON.
    }
    throw new Error(String(detail));
  }

  const data = await response.json();

  return {
    ...data,
    prediction: String(data?.prediction || data?.tumor_prediction || '').toLowerCase(),
    tumor_prediction: String(data?.tumor_prediction || data?.prediction || '').toLowerCase(),
    confidence: toNumber(data?.confidence),
    tumor_confidence: toNumber(data?.tumor_confidence ?? data?.confidence),
    raw_confidence: toNumber(data?.raw_confidence),
    mri_uncertainty: toNumber(data?.mri_uncertainty ?? data?.uncertainty_score),
    uncertainty_score: toNumber(data?.uncertainty_score ?? data?.mri_uncertainty),
    tumor_probability: toNumber(data?.tumor_probability),
    confidence_level: String(data?.confidence_level || ''),
    class_probabilities: data?.class_probabilities || {},
  };
}

export async function predictCognitive(payload) {
  const { data } = await api.post('/api/prediction/cognitive', payload, {
    headers: {
      'Content-Type': 'application/json',
    },
  });
  return {
    ...data,
    cognitive_risk: String(data?.cognitive_risk || data?.risk_label || '').toLowerCase(),
    risk_score: toNumber(data?.risk_score ?? data?.risk_probability),
    risk_probability: toNumber(data?.risk_probability ?? data?.risk_score),
    risk_percentage: toNumber(data?.risk_percentage),
    model_probability: toNumber(data?.model_probability),
    hybrid_components: data?.hybrid_components || {},
    module_deficit_breakdown: data?.module_deficit_breakdown || {},
    module_reliability_flag: String(data?.module_reliability_flag || ''),
    top_contributing_factors: Array.isArray(data?.top_contributing_factors)
      ? data.top_contributing_factors
      : [],
  };
}

export async function getDatasetsOverview() {
  const { data } = await api.get('/api/datasets/overview');
  return data;
}

export async function getCognitiveDatasetProfile() {
  const { data } = await api.get('/api/datasets/cognitive-profile');
  return data;
}

export async function getCognitiveWordBank() {
  const { data } = await api.get('/api/cognitive/word-bank');
  return data;
}

export async function getCognitiveReasoningQuestions() {
  const { data } = await api.get('/api/cognitive/reasoning-questions');
  return data;
}

export async function getCognitiveCategoryQuestions() {
  const { data } = await api.get('/api/cognitive/category-questions');
  return data;
}

export async function getCognitiveSpatialRotationQuestions() {
  const { data } = await api.get('/api/cognitive/spatial-rotation');
  return data;
}

export async function submitCognitiveTestResults(payload) {
  const { data } = await api.post('/cognitive-test-results', payload, {
    headers: {
      'Content-Type': 'application/json',
    },
  });
  return {
    memory_score: Number(data.memory_score ?? 0),
    cognitive_score: Number(data.cognitive_score ?? 0),
    duration_minutes: Number(data.duration_minutes ?? 0),
    reliability_flag: data.reliability_flag || 'standard',
    reliability_message: data.reliability_message || '',
    delayed_recall_count: Number(data.delayed_recall_count ?? 0),
    encoded_word_count: Number(data.encoded_word_count ?? 0),
    module_scores: data.module_scores || {},
  };
}

export async function generateGradCAM(file) {
  const formData = new FormData();
  formData.append('file', file);

  const response = await safeFetch(() =>
    fetchWithFallback('/gradcam', {
      method: 'POST',
      body: formData,
    }),
  );

  if (!response.ok) {
    let detail = 'GradCAM request failed';
    try {
      const payload = await response.json();
      detail = payload?.detail || payload?.message || payload?.error || detail;
    } catch {
      // Keep default error detail when response is not JSON.
    }
    throw new Error(String(detail));
  }

  const data = await response.json();
  console.log('API RESPONSE:', data);

  if (!data || !data.heatmap) {
    throw new Error('Heatmap missing from backend response');
  }

  const rawHeatmap = String(data.heatmap || '').trim();
  const heatmapSrc = rawHeatmap.startsWith('data:image/')
    ? rawHeatmap
    : `data:image/png;base64,${rawHeatmap}`;

  const prediction = String(data?.prediction || '').toLowerCase();
  const confidence = toNumber(data?.confidence);

  return {
    ...data,
    prediction,
    tumor_prediction: prediction,
    confidence,
    tumor_confidence: confidence,
    heatmap: heatmapSrc,
    gradcam_image: heatmapSrc,
    imageUrl: heatmapSrc,
  };
}

export async function generateAIReport(file, payload) {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('clinical_json', JSON.stringify(payload));

  const response = await safeFetch(() =>
    fetchWithFallback('/api/prediction/final-risk', {
      method: 'POST',
      body: formData,
    }),
  );

  if (!response.ok) {
    let detail = 'Report request failed';
    try {
      const payloadResponse = await response.json();
      detail = payloadResponse?.detail || payloadResponse?.message || payloadResponse?.error || detail;
    } catch {
      // Keep default error detail when response is not JSON.
    }
    throw new Error(String(detail));
  }

  const data = await response.json();

  const tumorConfidence = toNumber(data?.tumor_confidence ?? data?.confidence);
  const riskProbability = toNumber(data?.risk_probability ?? data?.risk_score);
  const finalRiskScore = toNumber(data?.final_risk_score);

  return {
    ...data,
    platform: data?.platform || 'Neuro Vision',
    confidence: tumorConfidence,
    tumor_confidence: tumorConfidence,
    raw_confidence: toNumber(data?.raw_confidence),
    mri_uncertainty: toNumber(data?.mri_uncertainty ?? data?.uncertainty_score),
    uncertainty_score: toNumber(data?.uncertainty_score ?? data?.mri_uncertainty),
    confidence_level: String(data?.confidence_level || ''),
    temperature: toNumber(data?.temperature, 1.0),
    tumor_probability: toNumber(data?.tumor_probability),
    class_probabilities: data?.class_probabilities || {},
    risk_score: riskProbability,
    risk_probability: riskProbability,
    risk_percentage: toNumber(data?.risk_percentage),
    cognitive_model_probability: toNumber(data?.cognitive_model_probability),
    hybrid_components: data?.hybrid_components || {},
    module_deficit_breakdown: data?.module_deficit_breakdown || {},
    module_reliability_flag: String(data?.module_reliability_flag || ''),
    final_risk_score: finalRiskScore,
    final_risk_percent: toNumber(data?.final_risk_percent, finalRiskScore * 100),
    final_risk_category: String(data?.final_risk_category || ''),
    top_contributing_factors: Array.isArray(data?.top_contributing_factors)
      ? data.top_contributing_factors
      : [],
    explainability: data?.explainability || {},
    clinical_disclaimer: String(data?.clinical_disclaimer || ''),
    summary: data?.summary || data?.report_summary || '',
    imageUrl: resolveHeatmapUrl(data?.gradcam_image),
  };
}

export async function exportToPDF(reportData) {
  const payload = {
    tumor_prediction: reportData.tumor_prediction || '',
    confidence: Number(reportData.tumor_confidence ?? reportData.confidence ?? 0),
    tumor_confidence: Number(reportData.tumor_confidence ?? reportData.confidence ?? 0),
    cognitive_risk: reportData.final_risk_category || reportData.cognitive_risk || '',
    risk_score: Number(reportData.final_risk_score ?? reportData.risk_score ?? reportData.risk_probability ?? 0),
    report_summary: reportData.summary || reportData.report_summary || '',
    gradcam_image: reportData.gradcam_image || null,
  };
  const response = await api.post('/generate-report', payload, {
    responseType: 'blob',
    headers: {
      'Content-Type': 'application/json',
    },
  });
  const url = window.URL.createObjectURL(new Blob([response.data], { type: 'application/pdf' }));
  const link = document.createElement('a');
  link.href = url;
  link.setAttribute('download', 'neurovision_report.pdf');
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.URL.revokeObjectURL(url);
}

export function buildClinicalPayload(values) {
  const mapping = [
    ['age', values.age],
    ['education', values.education],
    ['memory_score', values.memoryScore],
    ['cognitive_score', values.cognitiveScore],
  ];

  return Object.fromEntries(
    mapping
      .filter(([, value]) => value !== '' && value !== null && value !== undefined)
      .map(([key, value]) => [key, Number(value)])
      .filter(([, value]) => Number.isFinite(value)),
  );
}

export function formatApiError(error, fallbackMessage = 'Request failed') {
  if (axios.isAxiosError(error)) {
    const responseData = error.response?.data;
    const detail = responseData?.detail;

    if (typeof responseData?.message === 'string' && responseData.message.trim()) {
      return responseData.message;
    }

    if (typeof responseData?.error === 'string' && responseData.error.trim()) {
      return responseData.error;
    }

    if (typeof detail?.message === 'string' && detail.message.trim()) {
      return detail.message;
    }

    if (typeof detail?.error === 'string' && detail.error.trim()) {
      return detail.error;
    }

    if (typeof detail === 'string' && detail.trim()) {
      return detail;
    }

    return error.message || fallbackMessage;
  }

  if (error instanceof Error) {
    return error.message;
  }

  return fallbackMessage;
}

export default api;