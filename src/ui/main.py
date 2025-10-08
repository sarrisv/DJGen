import streamlit as st

from src.ui.models import init_session_state
from src.ui.components import render_sidebar, render_main_content


def main() -> None:
    st.set_page_config(
        page_title="DJP Generator", layout="wide", initial_sidebar_state="expanded"
    )

    init_session_state()
    config = render_sidebar()
    render_main_content(config)


if __name__ == "__main__":
    main()
