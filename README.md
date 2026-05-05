# 🏥 MediAI — End-to-End Healthcare AI System

> Disease Prediction · Medicine Recommendation · Adverse Drug Reaction Detection

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Next.js](https://img.shields.io/badge/Next.js-Frontend-black?logo=next.js)](https://nextjs.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A unified AI platform that assists medical professionals with real-time **diagnosis support**, **drug selection**, and **patient safety monitoring** — all from a single interface.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Tech Stack](#tech-stack)
- [Modules](#modules)
  - [Module 1: Disease Prediction](#module-1-disease-prediction)
  - [Module 2: Medicine Recommendation](#module-2-medicine-recommendation)
  - [Module 3: ADR Detection](#module-3-adr-detection)
- [Datasets](#datasets)
- [Performance Results](#performance-results)
- [Installation & Setup](#installation--setup)
- [API Endpoints](#api-endpoints)
- [Project Structure](#project-structure)
- [Limitations & Future Work](#limitations--future-work)
- [Acknowledgements](#acknowledgements)

---

## Overview

MediAI is an end-to-end AI-powered healthcare decision-support system developed as a final-year B.Tech (CSE) project at **KIIT Deemed to be University**, under the guidance of **Dr. Nachiketa Tarasia**.

The platform integrates three independent but chainable AI microservices into a single clinical workflow:

```
Clinical Notes  →  BERT Model  →  Diagnosis
                                      ↓
Medicine Name  →  Cosine Similarity  →  Top-5 Alternatives
                                              ↓
Patient Review  →  NER + Classifier + SIDER  →  ADR Risk Verdict
```

---

## Features

| Feature | Description |
|---|---|
| 🧠 **Disease Classifier** | Fine-tuned BERT predicts 1 of 20 diseases from clinical notes |
| 💊 **Medicine Recommender** | Content-based filtering returns 5 clinically similar alternatives |
| ⚠️ **ADR Detector** | Hybrid NER + ML pipeline extracts drug/symptom entities and classifies adverse reaction risk |
| 🗄️ **SIDER Validation** | Cross-checks detected ADRs against 153,663 FDA/EMA-verified drug–side-effect pairs |
| ⚡ **Real-Time API** | All three models served via FastAPI with in-memory inference (~20–300 ms latency) |
| 🖥️ **Intuitive Dashboard** | Next.js frontend built for non-technical clinical staff |

---

## System Architecture

```
┌─────────────────────────────────────────────────────┐
│                     FRONTEND                        │
│              Next.js + React + Tailwind             │
└───────────────────┬─────────────────────────────────┘
                    │ HTTP (JSON)
┌───────────────────▼─────────────────────────────────┐
│                  BACKEND API                        │
│           FastAPI + Uvicorn + Pydantic              │
├──────────────┬──────────────────┬───────────────────┤
│   /patient   │    /medicine     │      /adr         │
│ BERT Model   │ Cosine Sim Model │ spaCy NER + LR    │
└──────────────┴──────────┬───────┴───────────────────┘
                          │
┌─────────────────────────▼───────────────────────────┐
│                    DATABASE                         │
│         SQLite (SIDER 4.1 — 153,663 pairs)          │
└─────────────────────────────────────────────────────┘
```

---

## Tech Stack

### Backend
| Tool | Purpose |
|---|---|
| Python 3.12 | Core language |
| FastAPI + Uvicorn | REST API framework |
| PyTorch 2.x | Deep learning engine |
| HuggingFace Transformers 4.x | BERT tokenizer & model |
| spaCy 3.x | NER model training & inference |
| scikit-learn 1.x | TF-IDF, Logistic Regression, cosine similarity |
| NLTK 3.x | Stopwords, Porter Stemmer |
| SQLite / sqlite3 | SIDER ADR reference database |
| rapidfuzz | Fuzzy drug-name matching |
| joblib / pickle | Model serialisation |

### Frontend
| Tool | Purpose |
|---|---|
| Next.js | Server-side rendering |
| React.js | UI components |
| Tailwind CSS | Styling |
| TypeScript | Type safety |

---

## Modules

### Module 1: Disease Prediction

Predicts one of **20 diseases** from a free-text clinical note using a fine-tuned `bert-base-uncased` model.

**Pipeline (8 steps):**
1. Data ingestion & validation
2. Text preprocessing (lowercase, remove digits/punctuation, stopword removal)
3. Label encoding (20 diagnosis classes → integers 0–19)
4. 80/20 train–test split (4,000 / 1,000 records)
5. BERT tokenisation (max 512 tokens, attention masks)
6. Fine-tuning (5 epochs, LR 2e-5, batch size 16)
7. Evaluation (per-class F1, confusion matrix)
8. Model serialisation (`patient_model/`, `label_encoder.pkl`)

**Saved artefacts:**
```
patient_model/        ← BERT weights + tokeniser config
label_encoder.pkl     ← Diagnosis label map
patient_model.zip     ← Compressed archive for deployment
```

---

### Module 2: Medicine Recommendation

Returns the **top-5 most similar medicines** for any queried drug using content-based filtering on 9,720 medicine records.

**Pipeline (7 steps):**
1. Data ingestion & EDA
2. Tag construction (merge `Description` + `Reason`)
3. Porter Stemming (root normalisation)
4. Bag-of-Words vectorisation (`CountVectorizer`, 806-word vocab)
5. Cosine similarity matrix (9,720 × 9,720)
6. `recommend(medicine_name)` function
7. Serialisation (float16 compression: ~800 MB → ~200 MB)

**Saved artefacts:**
```
medicine_dict.pkl     ← Medicine DataFrame
model.pkl             ← float16 similarity matrix
new_data.csv          ← Cleaned data (audit / retraining)
```

---

### Module 3: ADR Detection

A **three-stage hybrid pipeline** for adverse drug reaction detection from patient review text.

| Stage | Technique | Purpose |
|---|---|---|
| 1 — Entity Extraction | spaCy NER (trained) + PhraseMatcher | Identify DRUG and SYMPTOM entities |
| 2 — Risk Classification | TF-IDF + Logistic Regression | Predict HIGH / LOW ADR risk |
| 3 — Database Validation | SQLite query (SIDER 4.1) | Confirm known ADR against 153K verified pairs |

**Pipeline (8 steps):**
1. Data loading & EDA (3,107 reviews, 6 EDA charts)
2. Text cleaning & binary ADR labelling (Moderate/Severe/Extremely Severe → 1)
3. NER training data construction + augmentation (3,107 → 12,428 samples)
4. spaCy NER training (35 epochs, dropout 0.35)
5. Hybrid PhraseMatcher override (502 known drug names)
6. TF-IDF + Logistic Regression classifier training (AUC 0.852)
7. SIDER 4.1 SQLite integration (153,663 drug–side-effect pairs)
8. `ClinicalAuditEngine` inference class

**Saved artefacts:**
```
adr_ner_final_certified/      ← Hybrid NER model
adr_classifier_model.pkl      ← TF-IDF + LR pipeline
adr_database.db               ← SIDER SQLite DB (153K pairs)
certified_drug_mappings.csv   ← Fuzzy-matched drug name bridges
```

---

## Datasets

| Dataset | Records | Used For |
|---|---|---|
| `clinical_notes_diagnosis_prediction_5000.csv` | 5,000 clinical notes × 20 diagnoses | Disease prediction training |
| `medicine.csv` | 9,720 medicine entries | Medicine recommendation |
| `drugLibTrain_raw.csv` | 3,107 patient drug reviews | NER training + ADR classifier |
| SIDER 4.1 (GitHub: `dhimmel/SIDER4`) | 153,663 drug–side-effect pairs | ADR ground-truth validation |
| DrugBank (GitHub: `dhimmel/drugbank`) | Drug name mappings | SIDER drug name bridging |

---

## Performance Results

### Disease Prediction
| Metric | Score |
|---|---|
| Test Precision | 1.00 |
| Test Recall | 1.00 |
| Test F1-Score | **1.00** |
| Disease Classes | 20 |
| Training Epochs | 5 |

### Medicine Recommendation
| Metric | Value |
|---|---|
| Medicines Indexed | 9,720 |
| Vocabulary Size | 806 words |
| Query Latency | ~20 ms |
| Recommendations | Top-5 per query |

### ADR Detection
| Metric | Logistic Regression | Random Forest |
|---|---|---|
| AUC | **0.852** | 0.808 |
| NER Precision | **99.5%** | — |
| NER F1-Score | **99.5%** | — |
| ADR Cases Detected (175/232) | **75.4%** | ~69% |
| Overall Accuracy | **79%** | ~75% |

---

## Installation & Setup

### Prerequisites
- Python 3.12+
- Node.js 18+
- Git

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/mediAI.git
cd mediAI
```

### 2. Backend setup
```bash
cd backend
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
```

### 3. Download models
Place the following pre-trained model artefacts in `backend/models/`:
```
patient_model/
label_encoder.pkl
medicine_dict.pkl
model.pkl
adr_ner_final_certified/
adr_classifier_model.pkl
adr_database.db
```

### 4. Start the backend
```bash
# From /backend
uvicorn main:app --reload --host 127.0.0.1 --port 8000
# Or double-click start_backend.bat (Windows)
```

API docs available at: `http://127.0.0.1:8000/docs`

### 5. Frontend setup
```bash
cd ../frontend
npm install
npm run dev
# Or double-click start_frontend.bat (Windows)
```

Frontend available at: `http://localhost:3000`

---

## API Endpoints

### Disease Prediction
```http
POST /api/v1/predict-disease
Content-Type: application/json

{
  "clinical_note": "Patient presents with severe chest pain and shortness of breath..."
}
```
```json
{
  "predicted_diagnosis": "Acute Myocardial Infarction",
  "confidence": 0.94,
  "top_3": [...]
}
```

### Medicine Recommendation
```http
POST /api/v1/recommend-medicine
Content-Type: application/json

{
  "medicine_name": "ACGEL CL NANO Gel 15gm"
}
```
```json
{
  "query": "ACGEL CL NANO Gel 15gm",
  "recommendations": [
    {"rank": 1, "name": "ACGEL NANO Gel 15gm", "similarity": 0.98},
    ...
  ]
}
```

### ADR Detection
```http
POST /api/v1/detect-adr
Content-Type: application/json

{
  "review_text": "I started taking enalapril last week. By day three I had terrible dizziness.",
  "drug_name": "enalapril",
  "condition": "hypertension"
}
```
```json
{
  "verdict": "HIGH RISK (ADR)",
  "adr_probability": 0.592,
  "entities": [
    {"text": "enalapril", "type": "DRUG"},
    {"text": "dizziness", "type": "SYMPTOM"}
  ],
  "sider_verification": {
    "enalapril + dizziness": "CONFIRMED — Known ADR in SIDER 4.1"
  }
}
```

---

## Project Structure

```
mediAI/
├── backend/
│   ├── main.py                    # FastAPI app entry point
│   ├── requirements.txt
│   ├── models/
│   │   ├── patient_model/         # BERT weights
│   │   ├── label_encoder.pkl
│   │   ├── medicine_dict.pkl
│   │   ├── model.pkl              # Similarity matrix
│   │   ├── adr_ner_final_certified/
│   │   ├── adr_classifier_model.pkl
│   │   └── adr_database.db
│   └── routers/
│       ├── disease.py
│       ├── medicine.py
│       └── adr.py
├── frontend/
│   ├── pages/
│   ├── components/
│   └── package.json
├── notebooks/
│   ├── 01_disease_prediction.ipynb
│   ├── 02_medicine_recommendation.ipynb
│   └── 03_adr_detection.ipynb
├── start_backend.bat
├── start_frontend.bat
└── README.md
```

---

## Limitations & Future Work

**Current Limitations:**
- Disease model covers only 20 conditions; real hospitals treat hundreds
- Perfect F1 (1.00) likely reflects limited dataset variety — independent validation on real EHR data is essential before any clinical use
- Medicine recommendations are based on textual similarity, not pharmacological equivalence — pharmacist review required
- SIDER mapping covers only 78 of 502 drugs (15.5%), limiting ADR cross-verification
- ADR classifier misses ~25% of true adverse reactions (false-negative rate)

**Planned Improvements:**
- [ ] Expand disease classes via hospital EHR partnerships
- [ ] Integrate structured pharmacological metadata (DrugBank / ChEMBL ATC codes)
- [ ] Increase SIDER drug-name coverage using MedDRA/UMLS codes
- [ ] Add human-review queue for borderline ADR predictions (40–60% probability range)
- [ ] Cloud deployment (Kubernetes + Horizontal Pod Autoscaler)
- [ ] HIPAA/GDPR compliance review before any clinical pilot

> ⚠️ **Disclaimer:** This system is a research prototype. It is not approved for clinical use and must not be used as a substitute for professional medical advice.

---

## Acknowledgements

We are deeply grateful to **Dr. Nachiketa Tarasia**, KIIT Deemed to be University, for his expert guidance and continuous support throughout this project.

**External resources used:**
- [BERT (Devlin et al., 2019)](https://arxiv.org/abs/1810.04805) — Google AI
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [SIDER 4.1](http://sideeffects.embl.de/) — Kuhn et al.
- [DrugBank](https://go.drugbank.com/) — Wishart et al.
- [spaCy](https://spacy.io/) — Explosion AI
- [scikit-learn](https://scikit-learn.org/) — Pedregosa et al.

---

<p align="center">Made with ❤️ at KIIT Deemed to be University · April 2026</p>
