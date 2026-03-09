import os
import json
import logging
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
import openai
from pathlib import Path

logger = logging.getLogger(__name__)

class Entity(BaseModel):
    text: str = Field(description="The entity text as it appears in the transcript.")
    label: str = Field(description="The label of the entity.")
    start: int = Field(description="The start character offset.")
    end: int = Field(description="The end character offset.")

class Transcript(BaseModel):
    id: str = Field(description="A unique identifier for the transcript.")
    text: str = Field(description="The transcript of the archaeological finding description.")
    entities: List[Entity] = Field(description="List of entities found in the text.")

class SyntheticGenerator:
    """
    Generates synthetic ASR transcripts for archaeological NER using LLMs.
    """
    
    DEFAULT_TEMPLATE = """
You are an archaeologist recording field notes via a voice recorder in the field. 
Your task is to generate realistic transcripts of these spoken notes.

**NATURE OF THE DATA**:
The data is ASR (Automatic Speech Recognition) transcriptions. 
It should sound like SPOKEN language:
- Occasional slight hesitations or repetitions (e.g., "we found... we found a shard").
- Informal but professional archaeological terminology.
- Varied sentence structures (not strictly formulaic).

**LABELSET GUIDELINES**:
- ARTEFACT: Objects found (pottery, bones, tools, etc.).
- PERIOD: Time periods or dates.
- LOCATION: Municipalities, sites, or specific sectors.
- CONTEXT: Human-made features (cesspit, ditch, burial mound).
- MATERIAL: What an artefact is made of (flint, bone, concrete).
- SPECIES: Animal, plant, or human species names.
- CONTEXT_ID: Specific identifier numbers for contexts (e.g., "625", "context 104").

**TASK**:
Generate a synthetic transcript and its NER labels in JSON format.
The labels MUST be character-accurate offsets in the generated text.

{format_instructions}

{seed_context}
"""

    def __init__(self, model_name: Optional[str] = None):
        # Prioritize passed arg, then OPENAI_MODEL_NAME, then default
        self.model = model_name or os.getenv("OPENAI_MODEL_NAME") or os.getenv("ATRIUM_LLM_MODEL", "google/gemma-3-27b-it")
        
        # Standard OpenAI client behavior is to automatically look for 
        # OPENAI_API_KEY and OPENAI_BASE_URL in the environment.
        # We explicitly pass them only if we want to override or ensure they are set.
        self.client = openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL")
        )
        self.parser = PydanticOutputParser(pydantic_object=Transcript)

    def generate_single(self, seed_examples: Optional[List[dict]] = None) -> Transcript:
        """Generates a single synthetic transcript."""
        
        seed_context = ""
        if seed_examples:
            seed_context = "Use these existing examples as style/labeling reference:\n"
            seed_context += json.dumps(seed_examples[:3], indent=2, ensure_ascii=False)
            
        prompt = self.DEFAULT_TEMPLATE.format(
            format_instructions=self.parser.get_format_instructions(),
            seed_context=seed_context
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert archaeological data annotator."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8, # Higher for variety
            )
            raw_content = response.choices[0].message.content
            return self.parser.parse(raw_content)
        except Exception as e:
            logger.error(f"Error generating synthetic data: {e}")
            return None

    def save_to_gliner(self, transcripts: List[Transcript], output_path: str):
        """Converts Transcripts to GLiNER JSON format and saves them."""
        gliner_data = []
        for t in transcripts:
            if not t: continue
            item = {
                "tokenized_text": t.text.split(), # Simple tokenization
                "ner": [[e.start, e.end, e.label] for e in t.entities],
                "text": t.text
            }
            gliner_data.append(item)
            
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(gliner_data, f, indent=2, ensure_ascii=False)
