# 🚀 JanSetu AI Engine

An AI-powered backend engine for the JanSetu platform that automates complaint analysis using Machine Learning, Natural Language Processing, Computer Vision, and Speech Recognition.

> Developed as the AI/ML module for the JanSetu project.

---

## 📌 Features

- 🎤 Speech-to-Text using OpenAI Whisper
- 🖼️ Image Verification using OpenAI CLIP
- 📝 Complaint Category Prediction
- ⚡ Complaint Priority Prediction
- ⏱️ SLA (Service Level Agreement) Prediction
- 🌐 REST API built with FastAPI
- 📖 Interactive Swagger API Documentation

---

## 🛠️ Tech Stack

### Languages
- Python

### Framework
- FastAPI
- Uvicorn

### Machine Learning
- Scikit-learn
- XGBoost
- Transformers
- OpenAI Whisper
- OpenAI CLIP
- PyTorch

### Data Processing
- Pandas
- NumPy

---

## 📂 Project Structure

```
JanSetu-AI-Engine/
│
├── app.py
├── requirements.txt
├── text_utils.py
├── evaluate_sample.py
├── test_clip.py
│
├── data/
├── models/
├── training/
├── utils/
└── weights/
```

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/ANU-AIHUB/JanSetu-AI-Engine.git
```

Move to the project directory

```bash
cd JanSetu-AI-Engine
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate the virtual environment

Windows

```bash
.venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the server

```bash
uvicorn app:app --reload
```

---

## 🌐 API Documentation

After running the server, open

```
http://127.0.0.1:8000/docs
```

---

## 📡 Available APIs

| Endpoint | Description |
|----------|-------------|
| `/` | API Information |
| `/health` | Health Check |
| `/models/status` | Model Status |
| `/transcribe` | Speech to Text |
| `/transcribe-batch` | Batch Speech Transcription |
| `/predict` | Complaint Classification |
| `/predict-sla` | SLA Prediction |

---

## 🤖 AI Models Used

- OpenAI Whisper (Speech Recognition)
- OpenAI CLIP (Image Verification)
- TF-IDF Vectorizer
- LinearSVC Complaint Classifier
- Logistic Regression Priority Classifier
- XGBoost SLA Prediction Model

---

## 📷 Sample Output

### Root Endpoint

```json
{
  "name": "JanSetu AI Engine",
  "version": "2.0",
  "status": "running"
}
```

---

## 📈 Future Improvements

- Multilingual Complaint Classification
- Better Image Validation
- Real-time Model Monitoring
- Docker Deployment
- CI/CD Integration

---

## 👩‍💻 Developer

**Anuradha Verma**

B.Tech – Artificial Intelligence & Machine Learning

GitHub: https://github.com/ANU-AIHUB

LinkedIn: *(Add your LinkedIn profile here)*

---

## ⭐ If you like this project

Please consider giving it a ⭐ on GitHub.
