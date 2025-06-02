from typing import Optional
from fastapi import APIRouter, HTTPException, File, UploadFile
from text_processing import get_sentiment
from schemas import (
    PredictionResult,
)
from logger import get_logger
from utils import (
    process_audio_input,
    generate_request_id,
)
from models import rf_model, torch_model

router = APIRouter(prefix="/api")
logger = get_logger(__name__)


@router.post("/predict_rf", response_model=PredictionResult)
async def predict_emotion_rf(
    file: Optional[UploadFile] = File(None),
    check_text: bool = False,
):
    """
    Распознавание эмоции в голосе с помощью деревьев
    с информацией о вероятностях
    """
    request_id = generate_request_id()
    try:
        audio_path, text = await process_audio_input(file, request_id)

        prediction = rf_model.predict_with_probabilities(request_id, audio_path)

        if "error" in prediction:
            raise HTTPException(500, detail=prediction["error"])

        text_emotion = None
        text_label_probability = None
        if check_text:
            text_emotion, probs = get_sentiment(text)
            text_label_probability = max(probs) if probs is not None else None

        result = PredictionResult(
            request_id=request_id,
            text=text,
            voice_emotion=prediction["emotion"],
            details=prediction["detail"],
            text_emotion=text_emotion,
            text_label_probability=text_label_probability,
        )

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"{request_id}: Error in detailed prediction: {str(e)}")
        raise HTTPException(500, detail=str(e))


@router.post("/predict_fcnn", response_model=PredictionResult)
async def predict_emotion_ml(
    file: Optional[UploadFile] = File(None),
    check_text: bool = False,
):
    """
    Распознавание эмоции в голосе с помощью полносвязной сети и
    информацией о вероятностях
    """
    request_id = generate_request_id()
    try:
        audio_path, text = await process_audio_input(file, request_id)

        prediction = torch_model.predict_with_probabilities(request_id, audio_path)

        if "error" in prediction:
            raise HTTPException(500, detail=prediction["error"])

        text_emotion = None
        text_label_probability = None
        if check_text:
            text_emotion, probs = get_sentiment(text)
            text_label_probability = max(probs) if probs is not None else None

        result = PredictionResult(
            request_id=request_id,
            text=text,
            voice_emotion=prediction["emotion"],
            details=prediction["detail"],
            text_emotion=text_emotion,
            text_label_probability=text_label_probability,
        )

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"{request_id}: Error in detailed prediction: {str(e)}")
        raise HTTPException(500, detail=str(e))
