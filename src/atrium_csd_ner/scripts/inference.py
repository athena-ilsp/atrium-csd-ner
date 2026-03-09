import argparse
import json
from gliner import GLiNER
import torch

def run_inference(text, model_path, labels):
    print(f"Loading model from: {model_path}")
    model = GLiNER.from_pretrained(model_path)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    print(f"Predicting entities for text: {text[:100]}...")
    entities = model.predict_entities(text, labels, threshold=0.5)
    
    return entities

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GLiNER2 Archaeology Inference")
    parser.add_argument("--text", type=str, required=True, help="Text to analyze")
    parser.add_argument("--model", type=str, default="data/train/final", help="Path to fine-tuned model")
    
    args = parser.parse_args()
    
    # Standard archaeological labels used in training
    labels = ["ARTEFACT", "PERIOD", "LOCATION", "CONTEXT", "FEATURE", "CONTEXT_ID", "MATERIAL", "SPECIES"]
    
    entities = run_inference(args.text, args.model, labels)
    
    print("\n" + "="*50)
    print("EXTRACTED ENTITIES")
    print("="*50)
    for ent in entities:
        print(f"[{ent['label']}] {ent['text']} (Pos: {ent['start']}-{ent['end']})")
    print("="*50)
