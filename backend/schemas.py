from enum import Enum
from pydantic import BaseModel, Field, ValidationInfo, field_validator
from typing import Any, Dict, List, Optional, Union


class AudioSource(str, Enum):
    UPLOAD = "upload"
    RECORD = "record"


class EmotionLabel(str, Enum):
    NEUTRAL = "neutral"
    ANGRY = "angry"
    SAD = "sad"
    POSITIVE = "positive"


class RandomForestParams(BaseModel):
    n_estimators: Optional[int] = Field(
        default=100, ge=1, description="Количество деревьев в лесу"
    )
    criterion: Optional[str] = Field(
        default="gini", description="Функция для измерения качества разбиения"
    )
    max_depth: Optional[int] = Field(
        default=None, ge=1, description="Максимальная глубина дерева"
    )
    min_samples_split: Optional[Union[int, float]] = Field(
        default=2,
        ge=1,
        description="Минимальное количество выборок для разбиения внутреннего узла",
    )
    min_samples_leaf: Optional[Union[int, float]] = Field(
        default=1, ge=1, description="Минимальное количество выборок в листовом узле"
    )
    max_features: Optional[Union[int, float, str]] = Field(
        default="sqrt",
        description="Количество признаков для поиска наилучшего разбиения",
    )
    bootstrap: Optional[bool] = Field(
        default=True,
        description="Использовать ли бутстрап выборки при построении деревьев",
    )
    random_state: Optional[int] = Field(
        default=None, description="Seed для генератора случайных чисел"
    )

    @field_validator("criterion")
    @classmethod
    def validate_criterion(cls, v, info: ValidationInfo):
        allowed_values = ["gini", "entropy", "log_loss"]
        if v not in allowed_values:
            raise ValueError(f"criterion должен быть одним из {allowed_values}")
        return v

    @field_validator("max_features")
    @classmethod
    def validate_max_features(cls, v, info: ValidationInfo):
        if isinstance(v, str) and v not in ["sqrt", "log2", "auto", None]:
            raise ValueError("max_features должен быть числом, 'sqrt', 'log2' или None")
        return v


class FitResponse(BaseModel):
    request_id: str
    status: str
    detail: str


class PredictionResult(BaseModel):
    request_id: str
    text: str
    voice_emotion: Optional[EmotionLabel] = None
    details: dict


class ModelsResponse(BaseModel):
    models: List[str]


class ModelParameters(BaseModel):
    name: str
    type: str
    n_estimators: Optional[int] = None
    classes: List[str] = []
    parameters: Dict[str, Any] = {}


class ModelInfoResponse(BaseModel):
    name: Optional[str] = None
    type: Optional[str] = None
    n_estimators: Optional[int] = None
    classes: List[str] = []
    parameters: Dict[str, Any] = {}
    error: Optional[str] = None


class SetModelRequest(BaseModel):
    ml_model_name: str


class SetModelResponse(BaseModel):
    message: str
