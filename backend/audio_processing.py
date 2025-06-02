import os
import librosa
import numpy as np
import pandas as pd
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
            mfcc = librosa.feature.mfcc(
                y=audio_data,
                sr=sr,
                n_mfcc=13,
                n_fft=min(2048, len(audio_data)),
                hop_length=min(512, len(audio_data) // 4),
            )

            width = min(9, mfcc.shape[1])
            if width % 2 == 0:
                width -= 1
            width = max(width, 1)

            mfcc_delta = librosa.feature.delta(mfcc, order=1, width=width)

            def agg(x):
                return np.mean(x, axis=1)

            features = np.hstack(
                [
                    agg(mfcc),
                    agg(mfcc_delta),
                ]
            )
            features = features.reshape(1, -1)
            feats = pd.DataFrame(features, columns=[f"mfcc_{i}" for i in range(1, 27)])
            return feats
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
