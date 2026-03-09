import os
import logging
from fastapi import FastAPI
from atrium_csd_ner.models import NERRequest, NERResponse
from atrium_csd_ner.core import ArchaeologicalNERService

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

from contextlib import asynccontextmanager

# Global service instance
ner_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global ner_service
    # Use absolute path preference
    model_path = os.getenv("MODEL_PATH", "data/train/final")
    logger.info(f"Startup: Checking model at {os.path.abspath(model_path)}")
    
    if not os.path.exists(model_path):
        logger.error(f"CRITICAL: Model directory NOT FOUND at {model_path}")
    else:
        try:
            ner_service = ArchaeologicalNERService(model_path)
            logger.info("Service successfully initialized and model loaded.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
    
    yield
    # Clean up if necessary
    ner_service = None

# Support reverse proxy paths for Swagger UI
root_path = os.getenv("API_ROOT_PATH", "")
app = FastAPI(title="Atrium Archaeological NER (GLiNER2)", lifespan=lifespan, root_path=root_path)

@app.post("/ner", response_model=NERResponse)
def ner_endpoint(request: NERRequest):
    if ner_service is None:
        return {"sentences": [{"text": "Model not loaded. Ensure training is complete.", "ents": []}]}
    
    results = ner_service.process_text(request.text, request.threshold)
    return {"sentences": results}

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": ner_service is not None}

@app.get("/palette")
def get_palette():
    # Standard color palette for our labels to keep UI consistent
    return {
        "ARTEFACT": "#FFB3B3",
        "PERIOD": "#B3D1FF",
        "LOCATION": "#B3FFB3",
        "CONTEXT": "#FFE6B3",
        "FEATURE": "#D9B3FF",
        "CONTEXT_ID": "#E8F1F2",
        "MATERIAL": "#FFCCB3",
        "SPECIES": "#B3FFE6"
    }

if __name__ == "__main__":
    import uvicorn
    # Allow port to be set via environment variable for production/deployment
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
