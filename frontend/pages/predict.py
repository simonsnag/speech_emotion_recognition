import streamlit as st

from api import predict_emotion
from utils import record_audio


def show_page():
    st.header("Получение эмоции из речи")
    audio_data = None
    audio_choice = st.radio(
        "Выбери тип источника:", ["Загрузить аудио", "Записать через микрофон"]
    )

    if audio_choice == "Загрузить аудио":
        uploaded_file = st.file_uploader("Выберите WAV файл", type=["wav"])
        if uploaded_file:
            audio_data = uploaded_file.read()
            st.audio(audio_data, format="audio/wav")
    else:
        duration = st.slider("Длительность записи (сек)", 5, 10)
        if st.button("Начать запись"):
            audio_data = record_audio(duration)
            st.audio(audio_data, format="audio/wav")

    if audio_data and st.button("Предположение эмоции"):
        with st.spinner("Анализирую голос..."):
            result = predict_emotion(audio_file=audio_data, audio_source="upload")
            emotion = result.get("voice_emotion", "Unknown")
            st.success(f"Эмоция: {emotion}")
            if "text" in result:
                st.subheader("Транскрипция")
                st.write(result["text"])
            if "details" in result:
                st.subheader("Детали")
                st.json(result["details"])
