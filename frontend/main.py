import streamlit as st

st.set_page_config(
    page_title="Speech Emotion Recognition(SER)",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    st.title("Speech Emotion Recognition(SER)")

    st.markdown("""
    ### Что умеет это приложение:
    - Анализировать эмоции в аудио
    - Обучать модель на новых записях
    """)

    st.sidebar.title("Навигация")
    page = st.sidebar.radio(
        "Выбери страницу:", ["Предположение эмоции", "Дообучение модели", "Описание"]
    )

    if page == "Предположение эмоции":
        from pages.predict import show_page

        show_page()
    elif page == "Дообучение модели":
        from pages.train import show_page

        show_page()
    elif page == "Описание":
        from pages.about import show_page

        show_page()


if __name__ == "__main__":
    main()
