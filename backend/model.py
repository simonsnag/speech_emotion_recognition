import glob
import os
import pickle
from typing import Dict, Any, List, Union
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from logger import get_logger
from audio_processing import audio_processor
from schemas import EmotionLabel

logger = get_logger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)


class EmotionModels:
    """
    Класс для работы с моделью
    """

    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.current_model_name = "default"

    def initialize_default_model(self):
        """Инициализация новой модели с дефолтными значениями"""
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit([e.value for e in EmotionLabel])
        logger.info("Default model initialized")

    def load_backup_model(self) -> bool:
        model_path = os.path.join(MODEL_DIR, "random_forest_model.pkl")
        encoder_path = os.path.join(MODEL_DIR, "label_encoder.pkl")

        try:
            if os.path.exists(model_path) and os.path.exists(encoder_path):
                with open(model_path, "rb") as f:
                    self.model = pickle.load(f)
                with open(encoder_path, "rb") as f:
                    self.label_encoder = pickle.load(f)
                self.current_model_name = "random_forest"
                logger.info("Model 'random_forest_model' loaded successfully")
                return True
            else:
                logger.warning(
                    "Model 'random_forest_model' not found, initializing default model"
                )
                self.initialize_default_model()
                self._save_model("default")
                self.current_model_name = "default"
                return False
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            self.initialize_default_model()
            self._save_model("default")
            self.current_model_name = "default"
            return False

    def _load_model(self, model_name: str) -> bool:
        model_path = os.path.join(MODEL_DIR, f"{model_name}_model.pkl")

        try:
            if os.path.exists(model_path):
                with open(model_path, "rb") as f:
                    self.model = pickle.load(f)
                self.current_model_name = model_name
                logger.info(f"Model {model_name} loaded successfully")
                return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False

    def _save_model(self, model_name: str = None) -> bool:
        if model_name is None:
            model_name = self.current_model_name

        model_path = os.path.join(MODEL_DIR, f"{model_name}_model.pkl")

        try:
            with open(model_path, "wb") as f:
                pickle.dump(self.model, f)
            logger.info(f"Model '{model_name}' saved successfully")
            return True
        except Exception as e:
            logger.error(f"Error saving model '{model_name}': {str(e)}")
            return False

    def get_available_models(self) -> List[str]:
        try:
            model_files = glob.glob(os.path.join(MODEL_DIR, "*_model.pkl"))
            model_names = [
                os.path.basename(f).replace("_model.pkl", "") for f in model_files
            ]
            return model_names
        except Exception as e:
            logger.error(f"Error getting available models: {str(e)}")
            return []

    def get_current_model_info(self) -> Dict[str, Any]:
        """Получение информации о текущей модели"""
        if self.model is None:
            return {"error": "No model loaded"}

        try:
            info = {
                "name": self.current_model_name,
                "type": type(self.model).__name__,
                "n_estimators": getattr(self.model, "n_estimators", None),
                "classes": list(self.label_encoder.classes_.astype(str))
                if hasattr(self.label_encoder, "classes_")
                else [],
                "parameters": self.model.get_params(),
            }
            return info
        except Exception as e:
            logger.error(f"Error getting model info: {str(e)}")
            return {"error": str(e)}

    def set_model(self, model_name: str) -> bool:
        """Выбор модели для использования"""
        if model_name == self.current_model_name:
            logger.info(f"Model '{model_name}' is already active")
            return True

        return self._load_model(model_name)

    def predict_with_probabilities(
        self, request_id: str, audio_path: str
    ) -> Dict[str, Union[str, Dict[str, float]]]:
        """
        Предсказание эмоции с вероятностями для каждого класса
        """
        try:
            if self.model is None or self.label_encoder is None:
                logger.warning(f"{request_id}: Model not initialized.")
                return {"error": "Model not initialized."}

            features = audio_processor.extract_features(request_id, audio_path)
            if features is None or features.empty:
                logger.error(f"{request_id}: Failed to extract features from audio")
                return {"error": "Failed to extract features"}

            prediction = self.model.predict(features)

            if len(prediction) == 0:
                logger.error(f"{request_id}: Empty prediction")
                return {"error": "Empty prediction"}

            emotion = self.label_encoder.inverse_transform(prediction)[0]
            probabilities = self.model.predict_proba(features)[0]
            prob_dict = {}
            for i, emotion_class in enumerate(self.label_encoder.classes_):
                prob_dict[emotion_class] = float(probabilities[i])

            result = {"emotion": emotion, "detail": prob_dict}

            logger.info(f"{request_id}: Predicted emotion: {emotion}")
            return result

        except Exception as e:
            logger.error(
                f"{request_id}: Error in prediction with probabilities: {str(e)}"
            )
            return {"error": str(e)}

    def partial_fit(
        self,
        request_id: str,
        audio_path: str,
        emotion_label: EmotionLabel,
        params: RandomForestClassifier = None,
    ) -> bool:
        """
        Дообучение модели новыми данными с кастомизируемыми параметрами
        """
        try:
            if params is None:
                params = {}

            if self.model is None or self.label_encoder is None:
                logger.warning(f"{request_id}: Model not initialized")
                return False

            features = audio_processor.extract_features(request_id, audio_path)
            if features is None or features.empty:
                logger.error(f"{request_id}: Failed to extract features from audio")
                return False

            current_classes = (
                self.label_encoder.classes_
                if hasattr(self.label_encoder, "classes_")
                else []
            )
            if emotion_label.value not in current_classes:
                logger.error(
                    f"{request_id}: Not supported emotion label by current model"
                )
                return False

            y = self.label_encoder.transform([emotion_label.value])
            self.model.set_params(**params)
            self.model.fit(features, y)

            self._save_model()
            logger.info(
                f"{request_id}: Voice emotion model successfully updated with new data"
            )
            return True

        except Exception as e:
            logger.error(f"{request_id}: Error in partial_fit: {str(e)}")
            return False


emotion_models = EmotionModels()
