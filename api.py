from fastapi import FastAPI, APIRouter
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from pathlib import Path
import os
import torch
import re

import dotenv
dotenv.load_dotenv( Path.home() / ".env")

torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)] 
# or simply:
import seaborn as sns
from huggingface_hub import hf_hub_download
from flair.splitter import SegtokSentenceSplitter
from flair.models import SequenceTagger
import logging
from logging import getLogger
logger = getLogger()
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


example_text = """Lower fill of posthole transitioning into fill 1036 gradually.
There is a linear delimitation / demarcation from the block behind the back of the head to the left foot.
It is contemporaneous with contexts 626 and 614, which initially corresponded to the inner elements of the cover.
It cover the original northern face of the structure , which appear to be vertical.
The context 609 corresponds to a fine layer of soil that partially covers the stones of the grave's cover , located in the eastern part of the burial.
C913 is the floor of the suspected c903 oven, it is a thin, irregular whitish layer of clay.
In this season we excavate c214 to reach the same level.
For further details, refer to the description provided for context 614.
An urn found in context 526.
Probably Late Mesolithic.
Animal bones in context 631.
""" 

app = FastAPI()

class NERRequest(BaseModel):
    text: str = Field(
        example=example_text,
        description="Text to process for NER."
    )

class Entity(BaseModel):
    start: int
    end: int
    label: str

class SentenceResult(BaseModel):
    text: str
    ents: List[Entity]
    title: Any = None

class NERResponse(BaseModel):
    sentences: List[SentenceResult]

def clean_label(label):
    return label.replace("I-", "").replace("B-", "").replace("E-", "").replace("S-", "") 

model_filenames = ["20250626-atrium-speech-based-ner.pt"]
models = dict()
for model_filename in model_filenames:
    if model_filename not in models:
        logger.info(f"Loading model {model_filename}")
        model_path = hf_hub_download(repo_id="pprokopidis/atrium-speech-based-ner", filename=model_filename)
        model = SequenceTagger.load(model_path)
        models[model_filename] = model 
        logger.info(f"Done loading model {model_filename}")

def get_model_color_palette(model=models[model_filenames[0]]):
    labels = {clean_label(l) for l in model.label_dictionary.get_items() if l != "O"}
    colors = sns.color_palette("husl", len(labels))
    model_color_palette = {label: f"#{int(c[0]*255):02x}{int(c[1]*255):02x}{int(c[2]*255):02x}" for label, c in zip(labels, colors)}
    return model_color_palette


# --- Model and splitter initialization (only once) ---
splitter = SegtokSentenceSplitter()

model_color_palettes = dict()
for model_filename in model_filenames:
    if model_filename not in model_color_palettes:
        model_color_palettes[model_filename] = get_model_color_palette(models[model_filename])

#selected_model_str = st.selectbox("Select a model:", [k for k in model_filenames])
selected_model_str = "20250626-atrium-speech-based-ner.pt"
model_color_palette = model_color_palettes[selected_model_str]

def process_text(text: str):
    sentences = splitter.split(text)
    for sentence in sentences:
        model.predict(sentence)
    doc = [{
        "text": sentence.text,
        "ents": [
            {"start": ent.start_position, "end": ent.end_position, "label": ent.labels[0].value}
            for ent in sentence.get_spans('ner')
        ],
        "title": None,
    } for sentence in sentences]
    return doc

@app.post("/ner", response_model=NERResponse)
def ner_endpoint(request: NERRequest):
    doc = process_text(request.text)
    return {"sentences": doc}

@app.get("/palette", include_in_schema=False)
def get_palette():
    """
    Returns the default color palette for the selected model.
    """
    return model_color_palette

