import os
from typing import Optional, Tuple, Dict, Any
from multiprocessing import Queue
import uuid
from fastapi import HTTPException, UploadFile
import aiofiles
from audio_processing import audio_processor
from model import emotion_models
from schemas import EmotionLabel
from logger import get_logger

logger = get_logger(__name__)


async def process_audio_input(
    audio_source: str,
    file: Optional[UploadFile],
    request_id: str,
    prefix: str = "audio",
    check_duration: bool = False,
) -> Tuple[str, str, bool]:
    """
    Функция для обработки аудио в зависимости от запроса
    """
    audio_path = None
    is_uploaded = False

    if audio_source == "upload":
        if not file:
            raise HTTPException(
                400, detail="File is required when audio_source is 'upload'"
            )

        content = await file.read()
        filename = f"{prefix}_{request_id}.wav"
        audio_path = os.path.join(os.path.dirname(__file__), "files", filename)

        async with aiofiles.open(audio_path, "wb") as f:
            await f.write(content)

        await file.close()

        if check_duration:
            is_valid = audio_processor.check_audio_duration(request_id, audio_path)
            if not is_valid:
                raise HTTPException(
                    400, detail="File exceeds maximum compatible duration"
                )

        logger.info(f"{request_id}: File uploaded and saved to {audio_path}")
        is_uploaded = True
    else:
        audio_path = audio_processor.record_audio(request_id)
        logger.info(f"{request_id}: Audio recorded to {audio_path}")

    text = audio_processor.transcribe_audio(request_id, audio_path)
    return audio_path, text, is_uploaded


def generate_request_id() -> str:
    return str(uuid.uuid4())


def train_model(
    queue: Queue,
    audio_path: str,
    emotion_label: EmotionLabel,
    train_params: Dict[str, Any],
    request_id: str,
) -> None:
    """
    Дообучение модели
    """
    try:
        emotion_models.load_backup_model()
        success = emotion_models.partial_fit(
            request_id, audio_path, emotion_label, train_params
        )
        queue.put(success)
    except Exception as e:
        logger.error(f"{request_id}: Training error: {str(e)}")
        queue.put(False)
