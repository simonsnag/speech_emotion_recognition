import os
from typing import Optional, Tuple
import uuid
from fastapi import HTTPException, UploadFile
import aiofiles
from audio_processing import audio_processor
from logger import get_logger
from pydub import AudioSegment


logger = get_logger(__name__)

MAX_DURATION_SECONDS = 10


async def process_audio_input(
    file: Optional[UploadFile],
    request_id: str,
) -> Tuple[str, str]:
    if not file:
        raise HTTPException(status_code=400, detail="File is required")

    if not (file.content_type and file.content_type.startswith("audio/")):
        raise HTTPException(status_code=400, detail="Uploaded file is not an audio")

    raw_dir = os.path.join(os.path.dirname(__file__), "files", "raw")
    wav_dir = os.path.join(os.path.dirname(__file__), "files", "wav")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(wav_dir, exist_ok=True)

    ext = os.path.splitext(file.filename)[-1].lower() or ".tmp"
    raw_path = os.path.join(raw_dir, f"{request_id}{ext}")

    try:
        content = await file.read()
        async with aiofiles.open(raw_path, "wb") as f:
            await f.write(content)
        await file.close()
    except Exception as e:
        logger.error(f"{request_id}: Error saving uploaded file: {e}")
        raise HTTPException(status_code=500, detail="Failed to save uploaded file")

    wav_path = os.path.join(wav_dir, f"{request_id}.wav")
    try:
        if ext != ".wav":
            audio_seg = AudioSegment.from_file(raw_path)
            audio_seg.export(wav_path, format="wav")
            logger.info(f"{request_id}: Converted {raw_path} to {wav_path}")
        else:
            os.replace(raw_path, wav_path)
            logger.info(f"{request_id}: Saved WAV to {wav_path}")
    except Exception as e:
        logger.error(f"{request_id}: Error converting to WAV: {e}")
        raise HTTPException(status_code=400, detail="Failed to convert audio to WAV")

    try:
        audio_seg = AudioSegment.from_wav(wav_path)
        duration_sec = audio_seg.duration_seconds
        if duration_sec > MAX_DURATION_SECONDS:
            logger.warning(
                f"{request_id}: Audio duration {duration_sec:.2f}s exceeds limit"
            )
            raise HTTPException(status_code=400, detail="Audio file exceeds 10 seconds")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"{request_id}: Error checking duration: {e}")
        raise HTTPException(status_code=500, detail="Failed to check audio duration")

    logger.info(
        f"{request_id}: Audio uploaded, converted, and duration OK ({duration_sec:.2f}s)"
    )

    try:
        text = audio_processor.transcribe_audio(request_id, wav_path)
    except Exception as e:
        logger.error(f"{request_id}: Error during transcription: {e}")
        raise HTTPException(status_code=500, detail="Failed to transcribe audio")

    return wav_path, text


def generate_request_id() -> str:
    return str(uuid.uuid4())
