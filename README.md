# 🧠 Neuro Vision – AI Brain Health Analysis Platform

Neuro Vision is a **multimodal AI platform** for neurological assessment that combines **MRI analysis, explainable AI, and cognitive screening** into a unified system.

It enables users to:

* Upload MRI scans
* Detect brain tumors using deep learning
* Visualize model decisions using GradCAM
* Perform cognitive assessments
* Generate AI-powered medical reports

---

# 🚀 Live Project Showcase

### 🌐 Frontend (Main Application)

👉 https://neuro-vision-deployed.vercel.app

The frontend provides a complete interactive dashboard where users can:

* Upload MRI images
* View tumor predictions
* Generate GradCAM heatmaps
* Take cognitive tests
* Download AI-generated reports

---

### ⚙️ Backend API (FastAPI – Render)

👉 https://neuro-vision-deployed.onrender.com/docs

The backend handles:

* API routing and request handling
* MRI validation and preprocessing
* Communication with ML model (Hugging Face)
* Cognitive risk prediction
* Report generation (PDF)

---

### 🤖 ML Model Service (Hugging Face Spaces)

👉 https://huggingface.co/spaces/dat1aryan/neuro-vision-ml

The ML service performs:

* MRI tumor classification using ResNet CNN
* GradCAM heatmap generation
* Optimized inference for large model deployment

---

# 🧠 Problem Statement

Brain tumor detection and neurological health evaluation require expert analysis of MRI scans and cognitive assessments.

Manual interpretation:

* Is time-consuming
* Depends on clinical expertise
* May lead to variability

AI can assist by providing:

* Faster predictions
* Consistent analysis
* Improved accessibility

---

# 💡 Proposed Solution

Neuro Vision integrates:

* Deep learning MRI analysis
* Explainable AI (GradCAM)
* Cognitive screening system
* Risk prediction models
* Automated report generation

All within a single platform.

---

# 🧪 Dataset

MRI model trained on:

**Brain Tumor MRI Dataset (Kaggle)**
https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset/data

Classes:

* Glioma
* Meningioma
* Pituitary Tumor
* No Tumor

---

# ⚙️ Key Features

### 🧠 MRI Tumor Detection

Classifies MRI images using deep learning CNN models.

### 🔥 Explainable AI (GradCAM)

Highlights regions influencing predictions.

### 🧩 Cognitive Assessment

Interactive test evaluating multiple cognitive domains.

### 📊 Multimodal Risk Analysis

Combines MRI + cognitive inputs for better insights.

### 📄 AI Report Generation

Generates structured clinical-style reports including:

* Prediction
* Confidence
* Heatmap insights
* Cognitive scores

---

# 🏗️ System Architecture

```id="f8w2lo"
Frontend (Vercel)
        ↓
Backend API (Render - FastAPI)
        ↓
Hugging Face ML Service
        ↓
Prediction + GradCAM
        ↓
Cognitive Inputs
        ↓
Risk Prediction
        ↓
AI Report Generation
```

---

# 🧠 Cognitive Screening System

Includes a **10-minute test** evaluating:

* Memory
* Attention
* Executive reasoning
* Processing speed
* Visuospatial ability

These metrics contribute to neurological risk scoring.

---

# 🧰 Tech Stack

## Frontend

* React
* Vite
* Tailwind CSS

## Backend

* FastAPI
* Python
* Uvicorn

## AI & ML

* PyTorch
* ResNet CNN
* GradCAM

## Cloud Deployment

* Vercel (Frontend)
* Render (Backend)
* Hugging Face Spaces (ML)

## Tools

* GitHub
* VS Code
* GitHub Copilot

---

# 📁 Repository Structure

```id="8h3h3x"
backend/        FastAPI backend
frontend/       React UI
training/       Model training scripts
models/         Lightweight backend models
Data/           Dataset (local use)
```

---

# 🔮 Future Improvements

* 3D MRI volumetric analysis
* Larger medical datasets
* Advanced multimodal fusion
* Real-time inference optimization
* Clinical validation

---

# 🎥 Demo Presentation

https://www.genspark.ai/slides?project_id=254028ee-6d15-4d80-854c-82566cbca498

---

# 👥 Team

Aryan Kumar — Full Stack Developer
Krishna Prakash — Machine Learning Developer
Preesha Bhardwaj — Designer & Frontend Developer
Saurabh Mazumdar — Lead Developer

---

# ⚠️ Disclaimer

This project is intended for **educational and research purposes only** and should not be used as a substitute for professional medical diagnosis.
