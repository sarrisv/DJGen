from typing import Tuple
import streamlit as st


def _render_two_column_layout(left_component, right_component):
    """Standard two-column layout for paired form elements"""
    col1, col2 = st.columns(2)
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
    right_disabled: bool = False,
) -> Tuple[bool, bool]:
    """Standard button pair layout with consistent styling"""

    def left():
        return st.button(left_text, use_container_width=True, key=left_key)

    def right():
        return st.button(
            right_text, disabled=right_disabled, use_container_width=True, key=right_key
        )

    return _render_two_column_layout(left, right)


def _render_standard_input_pair(left_component, right_component):
    """Standard input pair"""

    def left():
        return left_component()

    def right():
        return right_component()

    return _render_two_column_layout(left, right)
