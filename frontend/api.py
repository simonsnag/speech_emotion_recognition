import requests
import os
from typing import Dict, Any, Optional
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "")


def get_api_url(endpoint: str) -> str:
    return f"{BACKEND_URL}/api{endpoint}"


def predict_emotion(audio_file: bytes, audio_source: str) -> Dict[str, Any]:
    try:
        files = None
        if audio_file:
            files = {"file": ("audio.wav", audio_file, "audio/wav")}

        response = requests.post(
            get_api_url("/predict"), files=files, data={"audio_source": audio_source}
        )

        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
            return {"error": response.text}
    except Exception as e:
        st.error(f"Error connecting to backend: {str(e)}")
        return {"error": str(e)}


def train_model(
    audio_file: bytes,
    audio_source: str,
    emotion_label: str,
    model_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    try:
        files = None
        if audio_file:
            files = {"file": ("audio.wav", audio_file, "audio/wav")}

        data = {"audio_source": audio_source, "emotion_label": emotion_label}

        if model_params:
            for key, value in model_params.items():
                if value is not None:
                    data[key] = value

        response = requests.post(get_api_url("/fit"), files=files, data=data)

        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
            return {"error": response.text}
    except Exception as e:
        st.error(f"Error connecting to backend: {str(e)}")
        return {"error": str(e)}


def get_models_info() -> Dict[str, Any]:
    try:
        response = requests.get(get_api_url("/models"))
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
            return {"error": response.text}
    except Exception as e:
        st.error(f"Error connecting to backend: {str(e)}")
        return {"error": str(e)}


def get_current_model_info() -> Dict[str, Any]:
    try:
        response = requests.get(get_api_url("/models/current"))
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
            return {"error": response.text}
    except Exception as e:
        st.error(f"Error connecting to backend: {str(e)}")
        return {"error": str(e)}


def set_current_model(model_name: str) -> Dict[str, Any]:
    try:
        response = requests.post(
            get_api_url("/models/set"), json={"ml_model_name": model_name}
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
            return {"error": response.text}
    except Exception as e:
        st.error(f"Error connecting to backend: {str(e)}")
        return {"error": str(e)}
