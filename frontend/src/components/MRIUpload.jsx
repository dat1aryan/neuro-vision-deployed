import { motion } from 'framer-motion';
import { Activity, FileImage, UploadCloud } from 'lucide-react';
import { useRef, useState } from 'react';


function Spinner() {
  return (
    <svg className="h-4 w-4 animate-spin" viewBox="0 0 24 24" fill="none">
      <circle cx="12" cy="12" r="10" stroke="currentColor" strokeOpacity="0.2" strokeWidth="4" />
      <path d="M22 12a10 10 0 0 0-10-10" stroke="currentColor" strokeWidth="4" strokeLinecap="round" />
    </svg>
  );
}


export default function MRIUpload({
  selectedFile,
  previewUrl,
  result,
  loading,
  onAnalyze,
  onFileSelect,
}) {
  const fileInputRef = useRef(null);
  const [dragging, setDragging] = useState(false);

  function chooseFile() {
    fileInputRef.current?.click();
  }

  function handleDropzoneKeyDown(event) {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      chooseFile();
    }
  }

  function handleFiles(files) {
    if (files && files.length > 0) {
      onFileSelect(files[0]);
    }
  }

  function onDrop(event) {
    event.preventDefault();
    setDragging(false);
    handleFiles(event.dataTransfer.files);
  }

  const confidencePercent = Number(result?.confidence || 0) * 100;

  return (
    <section className="glass-card p-6 sm:p-7">
      <div className="relative z-10">
        <div className="mb-5 flex items-start justify-between gap-4">
          <div>
            <p className="section-kicker">Deep Learning Analysis</p>
            <h2 className="section-title mt-2">MRI Tumor Detection</h2>
            <p className="section-copy mt-2">
              Upload a <span className="font-semibold">black and white brain MRI scan</span> to analyze potential tumor patterns using our deep learning model. The system accepts only grayscale images and classifies tumor types with confidence scores to support early medical insights.
            </p>
          </div>
        </div>

        <div className="surface-panel mb-6 border-ann-cyan/45 bg-black/55 p-5">
          <div className="flex items-start gap-3">
            <div className="flex-shrink-0">
              <div className="flex h-8 w-8 items-center justify-center rounded-full border border-ann-cyan/55 bg-ann-cyan/15 shadow-[0_0_20px_rgba(34,211,238,0.2)]">
                <span className="text-ann-cyan font-bold text-sm">!</span>
              </div>
            </div>
            <div>
              <p className="font-semibold text-white text-sm">⚠️ IMPORTANT: Black & White Brain MRI Only</p>
              <p className="mt-2 text-xs leading-6 text-slate-200">
                This system accepts <span className="font-semibold text-ann-cyan">ONLY pure black and white (grayscale) brain MRI images</span>. 
                Colored images, photographs, or other image types will be rejected. 
                Please ensure your MRI scan is in grayscale format before uploading.
              </p>
            </div>
          </div>
        </div>

        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={(event) => handleFiles(event.target.files)}
        />

        <motion.div
          layout
          role="button"
          tabIndex={0}
          aria-label="Select an MRI image to upload"
          onClick={chooseFile}
          onKeyDown={handleDropzoneKeyDown}
          onDragOver={(event) => {
            event.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
          className={`cursor-pointer rounded-3xl border border-dashed p-4 transition sm:p-5 ${
            dragging
              ? 'border-ann-cyan bg-ann-cyan/10 shadow-glow'
              : 'border-slate-700/75 bg-slate-900/40'
          }`}
        >
          {previewUrl ? (
            <motion.div initial={{ opacity: 0, scale: 0.98 }} animate={{ opacity: 1, scale: 1 }}>
              <img
                src={previewUrl}
                alt="MRI preview"
                className="h-[21rem] w-full rounded-[1.25rem] object-contain bg-slate-950/85 shadow-soft sm:h-[24rem]"
              />
            </motion.div>
          ) : (
            <div className="flex h-[21rem] flex-col items-center justify-center rounded-[1.25rem] border border-slate-700/55 bg-slate-950/40 px-6 text-center sm:h-[24rem]">
              <div className="rounded-2xl border border-slate-700/70 bg-slate-900/70 p-4">
                <UploadCloud className="h-10 w-10 text-ann-cyan" />
              </div>
              <h3 className="mt-5 font-display text-2xl text-white">Drag and drop your MRI scan</h3>
              <p className="mt-3 max-w-md text-sm leading-7 text-slate-300">
                Click to upload or drag a <span className="font-semibold text-ann-cyan">black and white brain MRI</span> image here to run tumor classification and downstream explainability analysis.
              </p>
              <p className="mt-2 max-w-md rounded-lg border border-ann-cyan/30 bg-ann-cyan/5 px-3 py-2 text-xs text-ann-cyan font-medium">
                ⚠️ Only grayscale brain MRI images accepted. No colored images.
              </p>
            </div>
          )}
        </motion.div>

        <div className="mt-5 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <p className="metric-label">Current MRI file</p>
            <p className="mt-1 text-sm font-semibold text-slate-200">
              {selectedFile ? selectedFile.name : 'No scan selected'}
            </p>
          </div>

          <div className="flex flex-wrap gap-3">
            <button type="button" onClick={chooseFile} className="ghost-button">
              <FileImage className="h-4 w-4" />
              Upload Image
            </button>
            <button type="button" onClick={onAnalyze} disabled={!selectedFile || loading} className="neon-button">
              {loading ? <Spinner /> : <Activity className="h-4 w-4" />}
              Analyze MRI
            </button>
          </div>
        </div>

        <div className="surface-panel mt-5">
          <p className="metric-label">Latest prediction</p>
          <p className="mt-2 text-sm text-slate-300">
            {result
              ? `Tumor type: ${result.tumor_prediction} — ${confidencePercent.toFixed(1)}% confidence`
              : 'Analyze an MRI scan to view tumor classification output.'}
          </p>
        </div>
      </div>
    </section>
  );
}