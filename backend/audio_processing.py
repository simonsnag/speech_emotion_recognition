import os
import librosa
import numpy as np
import pandas as pd
import sounddevice as sd
from scipy.io.wavfile import write
import speech_recognition as sr
from logger import get_logger

logger = get_logger(__name__)
FILES_DIR = os.path.join(os.path.dirname(__file__), "files")


class AudioProcessor:
    """
    Класс для обработки аудио данных
    """

    def __init__(self):
        self.SAMPLE_RATE = 22050
        self.DURATION = 10
        logger.info(
            f"AudioProcessor initialized with sample rate: {self.SAMPLE_RATE}, duration: {self.DURATION}",
        )

    def record_audio(self, request_id: str) -> str:
        """
        Запись аудио с микрофона
        """
        filename = f"recording_{request_id}.wav"

        filepath = os.path.join(FILES_DIR, os.path.basename(filename))
        logger.debug(f"{request_id}: Recording audio")
        try:
            audio = sd.rec(
                int(self.SAMPLE_RATE * self.DURATION),
                samplerate=self.SAMPLE_RATE,
                channels=1,
                dtype="int16",
            )
            sd.wait()
            write(filepath, self.SAMPLE_RATE, audio)
            logger.debug(f"{request_id}: Audio recording completed: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"{request_id}: Error recording audio: {str(e)}")
            raise

    def check_audio_duration(self, request_id: str, audio_path: str):
        """
        Проверка длительности аудиофайла
        """
        logger.debug(f"{request_id}: Checking duration of audio")
        try:
            audio_data, sr = librosa.load(audio_path, sr=None)
            duration = librosa.get_duration(y=audio_data, sr=sr)

            is_valid = duration <= self.DURATION

            if not is_valid:
                logger.warning(
                    f"{request_id}: Audio file duration exceeds maximum allowed"
                )
            else:
                logger.debug(f"{request_id}: Audio file duration is valid")

            return is_valid
        except Exception as e:
            logger.error(f"{request_id}: Error checking audio duration: {str(e)}")
            raise

    def extract_features(self, request_id: str, audio_path: str):
        """
        Извлечение MFCC признаков
        """
        logger.debug(f"{request_id}: Extracting features")
        try:
            audio_data, sr = librosa.load(audio_path, sr=self.SAMPLE_RATE)
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
            return pd.DataFrame(
                [np.mean(mfccs, axis=1)], columns=[f"mfcc_{i}" for i in range(1, 14)]
            )
        except Exception as e:
            logger.error(f"{request_id}: Error extracting features: {str(e)}")
            raise

    def transcribe_audio(
        self, request_id: str, audio_path: str, language="ru-RU", use_google=True
    ):
        logger.info(f"{request_id}: Transcribing audio with language: {language}")
        try:
            recognizer = sr.Recognizer()
            with sr.AudioFile(audio_path) as source:
                audio_data = recognizer.record(source)

            try:
                if use_google:
                    text = recognizer.recognize_google(audio_data, language=language)
                else:
                    text = recognizer.recognize_sphinx(audio_data, language="ru-RU")
                logger.debug(f"{request_id}: Transcription successful: {text[:30]}...")
            except sr.UnknownValueError:
                text = "неизвестная речь"
                logger.warning(f"{request_id}: Speech not recognized")
            except sr.RequestError as e:
                logger.error(f"{request_id}: Speech recognition error: {e}")

            return text.strip()
        except Exception as e:
            logger.error(f"{request_id}: Error in transcription process: {str(e)}")
            raise


audio_processor = AudioProcessor()
