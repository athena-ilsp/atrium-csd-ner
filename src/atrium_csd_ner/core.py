import logging
import torch
import transformers.trainer
from typing import List
from gliner import GLiNER
from .models import Entity, SentenceResult

# Fix for transformers compatibility
if not hasattr(transformers.trainer, "ALL_LAYERNORM_LAYERS"):
    import torch.nn as nn
    transformers.trainer.ALL_LAYERNORM_LAYERS = [nn.LayerNorm]

logger = logging.getLogger(__name__)

class ArchaeologicalNERService:
    def __init__(self, model_path: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading GLiNER model from {model_path} on {self.device}")
        self.model = GLiNER.from_pretrained(model_path)
        self.model.to(self.device)
        self.labels = ["ARTEFACT", "PERIOD", "LOCATION", "CONTEXT", "FEATURE", "CONTEXT_ID", "MATERIAL", "SPECIES"]

    def process_text(self, text: str, threshold: float = 0.5) -> List[SentenceResult]:
        # Simple implementation: Treat the whole text as one processing block
        # GLiNER handles long text internally, but we can split by double newlines if needed
        # For now, let's keep it consistent with the existing API structure.
        
        # We simulate the "sentences" structure for backward compatibility with the UI
        predicted = self.model.predict_entities(text, self.labels, threshold=threshold)
        
        ents = [
            Entity(
                start=e["start"],
                end=e["end"],
                label=e["label"],
                text=e["text"],
                score=e.get("score")
            )
            for e in predicted
        ]
        
        return [SentenceResult(text=text, ents=ents)]
