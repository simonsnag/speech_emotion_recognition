import multiprocessing
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, Form
from model import emotion_models
from schemas import (
    AudioSource,
    EmotionLabel,
    FitResponse,
    ModelInfoResponse,
    ModelsResponse,
    PredictionResult,
    RandomForestParams,
    SetModelRequest,
    SetModelResponse,
)
from logger import get_logger
from utils import (
    process_audio_input,
    generate_request_id,
    train_model,
)

router = APIRouter(prefix="/api")
logger = get_logger(__name__)


@router.post("/fit", response_model=FitResponse)
async def fit_model(
    audio_source: AudioSource = Form(...),
    emotion_label: EmotionLabel = Form(...),
    learning_params: RandomForestParams = Depends(),
    file: Optional[UploadFile] = File(None),
):
    """
    Дообучение модели новыми данными
    """
    try:
        request_id = generate_request_id()

        audio_path, text, _ = await process_audio_input(
            audio_source=audio_source.value,
            file=file,
            request_id=request_id,
            prefix=audio_source.value,
            check_duration=True,
        )

        params = (
            learning_params.model_dump(exclude_none=True) if learning_params else {}
        )
        queue = multiprocessing.Queue()
        process = multiprocessing.Process(
            target=train_model,
            args=(
                queue,
                audio_path,
                emotion_label,
                params,
                request_id,
            ),
        )

        # Запуск процесса
        process.start()
        process.join(timeout=10)

        if process.is_alive():
            process.terminate()
            logger.warning(f"{request_id}: Training timeout")
            raise HTTPException(408, detail="Training timeout")

        success = queue.get()
        if not success:
            raise HTTPException(500, detail="Model training failed")

        return FitResponse(
            request_id=request_id, detail="Model trained succesfully", status="success"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"{request_id}: Training failed: {str(e)}")
        raise HTTPException(500, detail=str(e))


@router.post("/predict", response_model=PredictionResult)
async def predict_emotion(
    audio_source: AudioSource = Form(...),
    file: Optional[UploadFile] = File(None),
):
    """
    Распознавание эмоции в голосе с информацией о вероятностях
    """
    request_id = generate_request_id()
    try:
        audio_path, text, _ = await process_audio_input(
            audio_source, file, request_id, check_duration=True
        )

        prediction = emotion_models.predict_with_probabilities(request_id, audio_path)

        if "error" in prediction:
            raise HTTPException(500, detail=prediction["error"])

        result = PredictionResult(
            request_id=request_id,
            text=text,
            voice_emotion=prediction["emotion"],
            detail=prediction["detail"],
        )

        return result
    except Exception as e:
        logger.error(f"{request_id}: Error in detailed prediction: {str(e)}")
        raise HTTPException(500, detail=str(e))


@router.get("/models", response_model=ModelsResponse)
async def get_models():
    """
    Получение списка доступных моделей
    """
    try:
        models = emotion_models.get_available_models()
        return ModelsResponse(models=models)
    except Exception as e:
        logger.error(f"Error getting models: {str(e)}")
        raise HTTPException(500, detail=str(e))


@router.get("/models/current", response_model=ModelInfoResponse)
async def get_current_model():
    """
    Получение информации о текущей модели
    """
    try:
        model_info = emotion_models.get_current_model_info()
        return ModelInfoResponse(**model_info)
    except Exception as e:
        logger.error(f"Error getting current model info: {str(e)}")
        raise HTTPException(500, detail=str(e))


@router.post("/models/set", response_model=SetModelResponse)
async def set_model(model_request: SetModelRequest):
    """
    Выбор модели для использования
    """
    try:
        success = emotion_models.set_model(model_request.ml_model_name)
        if success:
            return SetModelResponse(
                message=f"Model '{model_request.ml_model_name}' set as current"
            )
        else:
            raise HTTPException(
                400, detail=f"Failed to set model '{model_request.ml_model_name}'"
            )
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error setting model: {str(e)}")
        raise HTTPException(500, detail=str(e))
