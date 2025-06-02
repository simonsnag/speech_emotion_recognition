from enum import Enum
from pydantic import BaseModel
from typing import Optional


class EmotionLabel(str, Enum):
    NEUTRAL = "neutral"
    ANGRY = "angry"
    SAD = "sad"
    POSITIVE = "positive"


class PredictionResult(BaseModel):
    request_id: str
    text: str
    voice_emotion: Optional[EmotionLabel] = None
    details: dict
    text_emotion: Optional[str] = None
    text_label_probability: Optional[float] = None
