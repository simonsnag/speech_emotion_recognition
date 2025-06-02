import streamlit as st

from api import predict_emotion


def show_page():
    st.header("Получение эмоции из речи")

    model_type = st.selectbox(
        "Выберите модель для предсказания:",
        options=["rf", "fcnn"],
        format_func=lambda x: "Random Forest"
        if x == "rf"
        else "Fully Connected Neural Network",
    )

    check_text = st.checkbox("Анализировать эмоции в тексте", value=False)

    audio_data = None

    uploaded_file = st.file_uploader("Выберите WAV файл", type=["wav"])
    if uploaded_file:
        audio_data = uploaded_file.read()
        st.audio(audio_data, format="audio/wav")

    if audio_data and st.button("Предсказание эмоции"):
        with st.spinner("Анализирую голос..."):
            result = predict_emotion(
                audio_file=audio_data, model_type=model_type, check_text=check_text
            )

            if "error" in result:
                st.error(f"Ошибка: {result['error']}")
                return

            voice_emotion = result.get("voice_emotion", "Unknown")
            st.success(f"Эмоция в голосе: {voice_emotion}")

            if "text" in result and result["text"]:
                st.subheader("Транскрипция")
                st.write(result["text"])

            if check_text and "text_emotion" in result and result["text_emotion"]:
                text_emotion = result["text_emotion"]
                text_prob = result.get("text_label_probability", 0)
                st.info(
                    f"Эмоция в тексте: {text_emotion} (вероятность: {text_prob:.2f})"
                )

            if "details" in result:
                st.subheader("Детали предсказания")
                st.json(result["details"])

            if "request_id" in result:
                st.caption(f"ID запроса: {result['request_id']}")
