import streamlit as st
from api import train_model
from utils import record_audio


def show_page():
    st.header("Тренировка модели")
    audio_choice = st.radio(
        "Выбери тип источника:", ["Загрузить аудио", "Записать через микрофон"]
    )

    audio_source = "upload"
    audio_data = None
    if audio_choice == "Загрузить аудио":
        uploaded_file = st.file_uploader("Upload an audio file", type=["wav"])
        if uploaded_file:
            audio_data = uploaded_file.read()
            st.audio(audio_data, format="audio/wav")
    else:
        duration = st.slider("Длительность записи (сек)", 5, 10)
        if st.button("Начать запись"):
            audio_data = record_audio(duration)
            st.audio(audio_data, format="audio/wav")

    emotion_label = st.selectbox(
        "Выбери эмоцию, которая в записи:", ["neutral", "angry", "sad", "positive"]
    )

    if st.button("Тренировка модели"):
        with st.spinner("Обучение..."):
            result = train_model(
                audio_file=audio_data,
                audio_source=audio_source,
                emotion_label=emotion_label,
            )

        if "error" not in result:
            st.success(f"Модель дообучена! ID: {result.get('request_id')}")
            st.json(result)
        else:
            st.error(f"Обучение завершилось неудачно: {result.get('error')}")
