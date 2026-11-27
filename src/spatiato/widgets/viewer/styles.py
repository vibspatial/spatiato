from __future__ import annotations

from spatiato.widgets.shared_styles import (
    WIDGET_ACCENT_BORDER_COLOR,
    WIDGET_ACCENT_SOFT_COLOR,
    WIDGET_BORDER_COLOR,
    WIDGET_PANEL_COLOR,
    WIDGET_TEXT_MUTED_COLOR,
    WIDGET_TEXT_SECONDARY_COLOR,
    build_input_control_stylesheet,
)

INPUT_CONTROL_STYLESHEET = build_input_control_stylesheet("QComboBox")
DETAIL_PANEL_STYLESHEET = (
    "QFrame[spatiatoViewerDetailPanel='true'] {"
    f"background-color: {WIDGET_PANEL_COLOR}; "
    f"border: 1px solid {WIDGET_BORDER_COLOR}; "
    "border-radius: 8px;}"
)
CARD_TITLE_STYLESHEET = (
    "QLabel {"
    f"background-color: {WIDGET_ACCENT_SOFT_COLOR}; "
    f"border: 1px solid {WIDGET_ACCENT_BORDER_COLOR}; "
    "border-radius: 8px; "
    f"color: {WIDGET_TEXT_SECONDARY_COLOR}; "
    "font-weight: 700; "
    "padding: 6px 10px;}"
)
SUMMARY_LABEL_STYLESHEET = f"color: {WIDGET_TEXT_SECONDARY_COLOR}; font-weight: 500;"
EMPTY_STATE_STYLESHEET = f"color: {WIDGET_TEXT_MUTED_COLOR}; font-weight: 500;"
# A container that lets the card background show through, such as a list of
# loaded layers.
TRANSPARENT_PANEL_STYLESHEET = "QWidget { background: transparent; }"
# A small heading inside a card, such as the count of loaded layers.
SUBSECTION_LABEL_STYLESHEET = f"color: {WIDGET_TEXT_MUTED_COLOR}; font-size: 11px; font-weight: 600;"
