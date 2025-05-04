import streamlit as st
from api import get_models_info, get_current_model_info, set_current_model


def show_page():
    st.title("🍭 Детектор эмоций по голосу")

    st.write("""
    ### Как это работает?
    Приложение использует модель машинного обучения для определения эмоций по аудио.
    Можно записать голос через микрофон или загрузить файл. Поддерживаемые эмоции:
    - 😊 Радость
    - 😢 Грусть
    - 😠 Злость
    - 😐 Нейтрально
    """)

    st.subheader("Текущая модель")
    model_info = get_current_model_info()

    if not model_info.get("error"):
        st.write(f"Название: **{model_info.get('name', '?')}**")
    else:
        st.error("Ошибка загрузки данных")

    st.subheader("Доступные модели")
    models_data = get_models_info()

    if not models_data.get("error"):
        model_list = models_data.get("models", [])
        selected = st.selectbox("Выберите модель:", model_list)

        if st.button("Использовать эту модель"):
            result = set_current_model(selected)
            if not result.get("error"):
                st.success(f"Выбрана модель: {selected}!")
            else:
                st.error("Не удалось изменить модель")

        st.write("### Все модели:")
        for model in models_data.get("models", []):
            st.write(f"✅ {model}")
    else:
        st.error("Не могу загрузить список моделей")
