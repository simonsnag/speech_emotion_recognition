import requests
import os
from typing import Dict, Any
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")


def get_api_url(endpoint: str) -> str:
    return f"{BACKEND_URL}/api{endpoint}"


def predict_emotion_rf(audio_file: bytes, check_text: bool = False) -> Dict[str, Any]:
    """
    Predict emotion using Random Forest model
    """
    try:
        files = None
        if audio_file:
            files = {"file": ("audio.wav", audio_file, "audio/wav")}

        params = {"check_text": check_text}

        response = requests.post(get_api_url("/predict_rf"), files=files, params=params)

        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
            return {"error": response.text}
    except Exception as e:
        st.error(f"Error connecting to backend: {str(e)}")
        return {"error": str(e)}


def predict_emotion_fcnn(audio_file: bytes, check_text: bool = False) -> Dict[str, Any]:
    """
    Predict emotion using Fully Connected Neural Network model
    """
    try:
        files = None
        if audio_file:
            files = {"file": ("audio.wav", audio_file, "audio/wav")}

        params = {"check_text": check_text}

        response = requests.post(
            get_api_url("/predict_fcnn"), files=files, params=params
        )

        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
            return {"error": response.text}
    except Exception as e:
        st.error(f"Error connecting to backend: {str(e)}")
        return {"error": str(e)}


def predict_emotion(
    audio_file: bytes, model_type: str = "rf", check_text: bool = False
) -> Dict[str, Any]:
    """
    Unified function to predict emotion using either RF or FCNN model
    """
    if model_type.lower() == "rf":
        return predict_emotion_rf(audio_file, check_text)
    elif model_type.lower() == "fcnn":
        return predict_emotion_fcnn(audio_file, check_text)
    else:
        st.error(f"Unknown model type: {model_type}")
        return {"error": f"Unknown model type: {model_type}"}
