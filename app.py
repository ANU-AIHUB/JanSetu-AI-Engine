from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import io
import torch
import whisper
import os
import pickle
import shutil
import pandas as pd
import logging
import json
from datetime import datetime
from transformers import CLIPProcessor, CLIPModel
from text_utils import infer_priority_override, normalize_complaint_text

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="JanSetu AI Engine",
    description="Advanced AI-powered complaint management system",
    version="2.0"
)

# Frontend Connection (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== PYDANTIC MODELS FOR REQUEST/RESPONSE =====
class PredictResponse(BaseModel):
    status: str
    match_score: float = None
    category: str = None
    priority: str = None
    expected_days: int = None
    reason: str = None

class TranscribeResponse(BaseModel):
    transcript: str
    duration: float = None
    language: str = None

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    models_loaded: dict

# Global variables for models
clip_model = None
clip_processor = None
whisper_model = None
vectorizer = None
classifier_model = None
sla_model = None
model_load_status = {}
priority_classifier = None

# --- IMPROVED MODELS LOAD SECTION ---
def load_all_models():
    global clip_model, clip_processor, whisper_model, vectorizer, classifier_model, sla_model, model_load_status, priority_classifier
    
    logger.info("🚀 Starting model loading process...")
    model_load_status = {
        "whisper": False,
        "clip": False,
        "vectorizer": False,
        "classifier": False,
        "priority_classifier": False,
        "sla_model": False
    }
    
    try:
        # 1. Whisper (Voice to Text)
        try:
            logger.info("Loading Whisper (Base)...")
            whisper_model = whisper.load_model("base")
            model_load_status["whisper"] = True
            logger.info("✅ Whisper loaded successfully")
        except Exception as e:
            logger.error(f"❌ Whisper loading failed: {e}")
        
        # 2. CLIP (Image Verification)
        try:
            logger.info("Loading CLIP (openai/clip-vit-base-patch32)...")
            clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            model_load_status["clip"] = True
            logger.info("✅ CLIP loaded successfully")
        except Exception as e:
            logger.error(f"❌ CLIP loading failed: {e}")
        
        # 3. Vectorizer & Classifiers (Custom Weights)
        try:
            logger.info("Loading Custom Classifiers...")
            if os.path.exists("weights/vectorizer.pkl"):
                with open("weights/vectorizer.pkl", "rb") as f:
                    vectorizer = pickle.load(f)
                model_load_status["vectorizer"] = True
                logger.info("✅ Vectorizer loaded successfully")
            else:
                logger.warning("⚠️ Vectorizer weights not found")
            
            if os.path.exists("weights/classifier.pkl"):
                with open("weights/classifier.pkl", "rb") as f:
                    classifier_model = pickle.load(f)
                model_load_status["classifier"] = True
                logger.info("✅ Classifier loaded successfully")
            else:
                logger.warning("⚠️ Classifier weights not found")
            
            if os.path.exists("weights/priority_classifier.pkl"):
                with open("weights/priority_classifier.pkl", "rb") as f:
                    priority_classifier = pickle.load(f)
                model_load_status["priority_classifier"] = True
                logger.info("✅ Priority classifier loaded successfully")
            else:
                logger.warning("⚠️ Priority classifier weights not found")
            
            if os.path.exists("weights/sla_model.pkl"):
                with open("weights/sla_model.pkl", "rb") as f:
                    sla_model = pickle.load(f)
                model_load_status["sla_model"] = True
                logger.info("✅ SLA Model loaded successfully")
            else:
                logger.warning("⚠️ SLA Model weights not found")
                
        except Exception as e:
            logger.error(f"❌ Custom Classifiers loading failed: {e}")

        logger.info("✅ MODEL LOADING COMPLETE")
        logger.info(f"Load Status: {model_load_status}")
    except Exception as e:
        logger.error(f"❌ CRITICAL LOADING ERROR: {e}")

# Server start hote hi models load karein
load_all_models()

# --- ENDPOINT 0: HEALTH CHECK ---
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API and model status"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": model_load_status
    }

# --- ENDPOINT 1: VOICE TRANSCRIPTION (WHISPER) ---
@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_audio(file: UploadFile = File(...)):
    """Transcribe audio file to text using Whisper model"""
    
    if whisper_model is None:
        logger.error("Whisper model not loaded")
        raise HTTPException(status_code=503, detail="Whisper model not available")
    
    # Validate file type
    if file.content_type not in ["audio/mpeg", "audio/wav", "audio/mp3", "audio/wav; codecs=opus"]:
        raise HTTPException(status_code=400, detail="Unsupported audio format. Use MP3 or WAV")

    audio_path = "temp_audio.wav"
    
    try:
        logger.info(f"Processing audio file: {file.filename}")
        
        # File save karein with size check
        file_size = 0
        with open(audio_path, "wb") as buffer:
            while True:
                chunk = await file.read(8192)
                if not chunk:
                    break
                file_size += len(chunk)
                buffer.write(chunk)
                if file_size > 100 * 1024 * 1024:  # 100MB limit
                    raise HTTPException(status_code=413, detail="File too large (max 100MB)")
        
        logger.info(f"Audio file saved: {file_size} bytes")
        
        # Transcribe
        result = whisper_model.transcribe(audio_path, fp16=False)
        logger.info(f"Transcription completed: {len(result['text'])} characters")
        
        return {
            "transcript": result["text"],
            "language": result.get("language", "unknown")
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
    
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)
            logger.info("Temp audio file cleaned up")

# --- ENDPOINT 2: COMPLAINT PREDICTION (CLIP + NLP) ---
@app.post("/predict", response_model=PredictResponse)
async def predict_complaint(text: str = Form(...), file: UploadFile = File(...)):
    """Predict complaint category with image verification"""
    
    if clip_model is None or vectorizer is None:
        logger.error("Required models not loaded")
        raise HTTPException(status_code=503, detail="AI models not ready")
    
    # Validate inputs
    if not text or len(text.strip()) < 5:
        raise HTTPException(status_code=400, detail="Text must be at least 5 characters")
    
    # Validate image
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Invalid image format. Use JPG or PNG")

    try:
        logger.info(f"Predicting complaint: {text[:50]}...")
        normalized_text = normalize_complaint_text(text)
        
        # --- STEP 1: CLIP Image Verification ---
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Resize if too large
        if image.size[0] > 1024 or image.size[1] > 1024:
            image.thumbnail((1024, 1024))
        
        inputs = clip_processor(
            text=[f"a photo of {text}", "a random unrelated photo"], 
            images=image, 
            return_tensors="pt", 
            padding=True
        )
        
        with torch.no_grad():
            outputs = clip_model(**inputs)
        
        # Similarity score calculate karein
        probs = outputs.logits_per_image.softmax(dim=1)
        match_score = float(probs[0][0].item())
        
        logger.info(f"Image match score: {match_score:.2%}")
        
        # Agar score 60% se kam hai toh reject karein
        if match_score < 0.6:
            return {
                "status": "Rejected", 
                "reason": "Image does not match the description",
                "match_score": match_score
            }

        # --- STEP 2: Classification ---
        if classifier_model is None:
            category = "General"
        else:
            text_vectorized = vectorizer.transform([normalized_text])
            category = str(classifier_model.predict(text_vectorized)[0])

        # Priority prediction: prefer the trained priority classifier when available
        override_priority = infer_priority_override(normalized_text)
        if override_priority is not None:
            priority = override_priority
        elif priority_classifier is not None:
            try:
                if 'text_vectorized' not in locals():
                    text_vectorized = vectorizer.transform([normalized_text])
                raw_priority = str(priority_classifier.predict(text_vectorized)[0])
                # Normalize formatting (e.g., 'high' -> 'High')
                priority = raw_priority.capitalize()
            except Exception:
                priority = "Medium"
        else:
            # Fallback heuristic when priority model is unavailable
            priority = "High" if len(text) > 200 else ("Medium" if len(text) > 50 else "Low")

        # Expected days mapping
        expected_days = 3 if priority.lower() == "high" else (5 if priority.lower() == "medium" else 7)

        logger.info(f"Classification: {category}, Priority: {priority}")

        # --- STEP 3: Response ---
        return {
            "status": "Verified",
            "match_score": match_score,
            "category": category,
            "priority": priority,
            "expected_days": expected_days
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# --- ENDPOINT 3: SLA PREDICTION ---
@app.post("/predict-sla")
async def predict_sla(category: str = Form(...), complaint_text: str = Form(...)):
    """Predict SLA (Service Level Agreement) for complaint"""
    
    if sla_model is None:
        logger.warning("SLA model not loaded, returning default SLA")
        return {
            "status": "Success",
            "estimated_sla_days": 7,
            "priority": "Medium",
            "message": "Using default SLA (model not available)"
        }
    
    try:
        logger.info(f"Predicting SLA for category: {category}")
        
        # Simple SLA logic (can be enhanced with actual model prediction)
        sla_days = 7
        priority = "Medium"
        
        # SLA logic based on category
        if "urgent" in complaint_text.lower() or "critical" in complaint_text.lower():
            sla_days = 3
            priority = "High"
        elif "normal" in complaint_text.lower():
            sla_days = 7
            priority = "Medium"
        else:
            sla_days = 10
            priority = "Low"
        
        logger.info(f"SLA predicted: {sla_days} days, Priority: {priority}")
        
        return {
            "status": "Success",
            "estimated_sla_days": sla_days,
            "priority": priority
        }
        
    except Exception as e:
        logger.error(f"SLA prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"SLA prediction failed: {str(e)}")

# --- ENDPOINT 4: BATCH TRANSCRIPTION ---
@app.post("/transcribe-batch")
async def transcribe_batch(files: list = File(...)):
    """Transcribe multiple audio files"""
    
    if whisper_model is None:
        raise HTTPException(status_code=503, detail="Whisper model not available")
    
    results = []
    
    for file in files:
        audio_path = f"temp_{file.filename}"
        try:
            with open(audio_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            result = whisper_model.transcribe(audio_path, fp16=False)
            results.append({
                "filename": file.filename,
                "transcript": result["text"],
                "status": "success"
            })
            logger.info(f"Batch transcription completed: {file.filename}")
        except Exception as e:
            results.append({
                "filename": file.filename,
                "status": "error",
                "error": str(e)
            })
            logger.error(f"Batch transcription error for {file.filename}: {str(e)}")
        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)
    
    return {"results": results, "total": len(results)}

# --- ENDPOINT 5: MODEL STATUS ---
@app.get("/models/status")
async def model_status():
    """Get detailed model loading status"""
    return {
        "timestamp": datetime.now().isoformat(),
        "models": model_load_status,
        "total_models": len(model_load_status),
        "loaded_models": sum(model_load_status.values()),
        "ready": all(model_load_status.values())
    }

# --- ROOT ENDPOINT ---
@app.get("/")
async def root():
    """Welcome endpoint with API info"""
    return {
        "name": "JanSetu AI Engine",
        "version": "2.0",
        "status": "running",
        "endpoints": {
            "/docs": "Interactive API documentation",
            "/health": "Health check",
            "/models/status": "Model status",
            "/transcribe": "Transcribe audio to text",
            "/transcribe-batch": "Transcribe multiple audio files",
            "/predict": "Predict complaint category with image verification",
            "/predict-sla": "Predict SLA for complaint"
        }
    }

if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 Starting JanSetu AI Engine Server...")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)