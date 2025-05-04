import io
import wave
import numpy as np
import streamlit as st
import sounddevice as sd


def record_audio(duration=10, sample_rate=2205):
    st.write("Запись...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    with st.spinner(f"Идет запись ({duration} сек)..."):
        sd.wait()

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes((audio * 32767).astype(np.int16).tobytes())
    buffer.seek(0)
    return buffer.read()
