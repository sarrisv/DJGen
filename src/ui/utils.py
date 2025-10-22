from typing import Tuple, Any
import streamlit as st


def _render_two_column_layout(left_component, right_component, container: Any = st):
    """Standard two-column layout for paired form elements"""
    col1, col2 = container.columns(2)
    with col1:
        left_result = left_component()
    with col2:
        right_result = right_component()
    return left_result, right_result


def _render_standard_button_pair(
    left_text: str,
    right_text: str,
    left_key: str,
    right_key: str,
    left_icon: str | None = None,
    right_icon: str | None = None,
    right_disabled: bool = False,
    container: Any = st,
) -> Tuple[bool, bool]:
    """Standard button pair layout with consistent styling"""

    def left():
        return st.button(left_text, width="stretch", key=left_key, icon=left_icon)

    def right():
        return st.button(
            right_text,
            disabled=right_disabled,
            width="stretch",
            key=right_key,
            icon=right_icon,
        )

    return _render_two_column_layout(left, right, container=container)


def _render_standard_input_pair(left_component, right_component, container: Any = st):
    """Standard input pair"""

    def left():
        return left_component()

    def right():
        return right_component()

    return _render_two_column_layout(left, right, container=container)
