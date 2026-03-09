from pydantic import BaseModel, Field
from typing import List, Optional

class Entity(BaseModel):
    start: int
    end: int
    label: str
    text: str
    score: Optional[float] = None

class NERRequest(BaseModel):
    text: str = Field(..., description="The raw text to process for NER.", examples=["During the 2018 excavations at Trench V, a notable concentration of Middle Helladic pottery sherds was recovered from the primary fill (c402) of a large, sub-rectangular pit."])
    threshold: float = Field(default=0.5, description="The confidence threshold for extraction.")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "During the 2018 excavations at Trench V, a notable concentration of Middle Helladic pottery sherds was recovered from the primary fill (c402) of a large, sub-rectangular pit.",
                "threshold": 0.5
            }
        }

class SentenceResult(BaseModel):
    text: str
    ents: List[Entity]

class NERResponse(BaseModel):
    sentences: List[SentenceResult]
