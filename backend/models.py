import os
import pickle
from typing import Dict, Union

import torch
import torch.nn.functional as F
from torch import nn

from audio_processing import audio_processor
from logger import get_logger

logger = get_logger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)


class RandomForestEmotionModel:
    def __init__(self):
        self.model = None
        self.label_encoder = None
        self._load_model_and_encoder()

    def _load_model_and_encoder(self) -> None:
        model_path = os.path.join(MODEL_DIR, "random_forest_model.pkl")
        encoder_path = os.path.join(MODEL_DIR, "label_encoder.pkl")

        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError("Model file not found after reconstruction")

            with open(model_path, "rb") as f_model:
                self.model = pickle.load(f_model)
            with open(encoder_path, "rb") as f_le:
                self.label_encoder = pickle.load(f_le)
            logger.info("RandomForest model and label encoder loaded successfully")
        except Exception as e:
            logger.error(f"Error loading RandomForest model or encoder: {e}")
            raise

    def predict_with_probabilities(
        self, request_id: str, audio_path: str
    ) -> Dict[str, Union[str, Dict[str, float]]]:
        if self.model is None or self.label_encoder is None:
            logger.error(f"{request_id}: Model or LabelEncoder is not loaded.")
            return {"error": "Model not loaded"}

        try:
            features = audio_processor.extract_features(request_id, audio_path)
            if features is None or features.empty:
                logger.error(f"{request_id}: Failed to extract features from audio")
                return {"error": "Failed to extract features"}

            preds = self.model.predict(features)
            probs = self.model.predict_proba(features)[0]

            if len(preds) == 0:
                logger.error(f"{request_id}: Empty prediction result")
                return {"error": "Empty prediction"}

            emotion = self.label_encoder.inverse_transform(preds)[0]
            prob_dict = {
                label: float(probs[idx])
                for idx, label in enumerate(self.label_encoder.classes_)
            }

            result = {"emotion": emotion, "detail": prob_dict}
            logger.info(f"{request_id}: RF predicted emotion: {emotion}")
            return result

        except Exception as e:
            logger.error(f"{request_id}: Error during RF prediction: {e}")
            return {"error": str(e)}


class EmotionFCNN(nn.Module):
    def __init__(self):
        super(EmotionFCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(26, 416),
            nn.ReLU(),
            nn.BatchNorm1d(416),
            nn.Dropout(0.2),
            nn.Linear(416, 208),
            nn.ReLU(),
            nn.BatchNorm1d(208),
            nn.Dropout(0.1),
            nn.Linear(208, 104),
            nn.ReLU(),
            nn.BatchNorm1d(104),
            nn.Dropout(0.1),
            nn.Linear(104, 52),
            nn.ReLU(),
            nn.BatchNorm1d(52),
            nn.Dropout(0.1),
            nn.Linear(52, 4),
        )

    def forward(self, x):
        return self.net(x)


class TorchEmotionModel:
    def __init__(self, device: str = None):
        self.device = torch.device(device if device else "cpu")
        self.model = None
        self.label_encoder = None
        self._load_model_and_encoder()

    def _load_model_and_encoder(self) -> None:
        model_path = os.path.join(MODEL_DIR, "fcnn_model.pth")
        encoder_path = os.path.join(MODEL_DIR, "label_encoder.pkl")

        try:
            with open(encoder_path, "rb") as f_le:
                self.label_encoder = pickle.load(f_le)

            self.model = EmotionFCNN().to(self.device)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            logger.info("Torch FCNN model and label encoder loaded successfully")
        except Exception as e:
            logger.error(f"Error loading Torch model or encoder: {e}")
            raise

    def predict_with_probabilities(
        self, request_id: str, audio_path: str
    ) -> Dict[str, Union[str, Dict[str, float]]]:
        if self.model is None or self.label_encoder is None:
            logger.error(f"{request_id}: Torch model or LabelEncoder is not loaded.")
            return {"error": "Model not loaded"}

        try:
            features = audio_processor.extract_features(request_id, audio_path)
            if features is None or features.empty:
                logger.error(f"{request_id}: Failed to extract features from audio")
                return {"error": "Failed to extract features"}
            x = torch.tensor(features.values, dtype=torch.float32, device=self.device)
            if x.dim() == 1:
                x = x.unsqueeze(0)

            with torch.no_grad():
                logits = self.model(x)
                probs = F.softmax(logits, dim=1)

            probs_np = probs.cpu().numpy()[0]
            pred_idx = int(probs_np.argmax())
            emotion = self.label_encoder.inverse_transform([pred_idx])[0]

            prob_dict = {
                label: float(probs_np[idx])
                for idx, label in enumerate(self.label_encoder.classes_)
            }

            result = {"emotion": emotion, "detail": prob_dict}
            logger.info(f"{request_id}: TorchCNN predicted emotion: {emotion}")
            return result

        except Exception as e:
            logger.error(f"{request_id}: Error during TorchCNN prediction: {e}")
            return {"error": str(e)}


rf_model = RandomForestEmotionModel()
torch_model = TorchEmotionModel(device="cpu")
