from __future__ import annotations

import re
from html import escape
from typing import Literal, cast

from qtpy.QtCore import QByteArray, QPointF, QRectF, QSize, Qt, QTimer
from qtpy.QtGui import QColor, QIcon, QPainter, QPainterPath, QPalette, QPen, QPixmap
from qtpy.QtSvg import QSvgRenderer
from qtpy.QtWidgets import (
    QComboBox,
    QLabel,
    QLineEdit,
    QSizePolicy,
    QStyle,
    QStyleOptionComboBox,
    QStylePainter,
    QWidget,
)

from spatiato._resources import get_logo_path

WIDGET_SURFACE_COLOR = "#25272c"
WIDGET_SURFACE_STYLESHEET = f"background-color: {WIDGET_SURFACE_COLOR};"
WIDGET_MIN_WIDTH = 390
WIDGET_PANEL_COLOR = "#30333a"
WIDGET_PANEL_MUTED_COLOR = "#3a3e47"
WIDGET_PANEL_SUBTLE_COLOR = "#2b2e34"
WIDGET_BORDER_COLOR = "#4b515d"
WIDGET_BORDER_STRONG_COLOR = "#67707f"
WIDGET_TEXT_COLOR = "#f2f4f8"
WIDGET_TEXT_MUTED_COLOR = "#aeb6c2"
WIDGET_TEXT_SECONDARY_COLOR = "#d4d9e2"
WIDGET_DISABLED_TEXT_COLOR = "#7f8794"
WIDGET_DISABLED_BORDER_COLOR = "#414651"
WIDGET_ACCENT_COLOR = "#9b7bea"
WIDGET_ACCENT_SOFT_COLOR = "#3c3450"
WIDGET_ACCENT_BORDER_COLOR = "#b196f3"
WIDGET_SELECTION_COLOR = "#4b3f63"
WIDGET_PRESSED_COLOR = "#454a55"
WIDGET_SUCCESS_COLOR = "#70c98a"
WIDGET_SUCCESS_HOVER_COLOR = "#83d99b"
WIDGET_PRIMARY_BUTTON_TEXT_COLOR = "#111827"
WIDGET_WARNING_TEXT_COLOR = "#f0c36a"
WIDGET_WARNING_BACKGROUND_COLOR = "#3f3325"
WIDGET_WARNING_BORDER_COLOR = "#c99845"
WIDGET_WARNING_HOVER_COLOR = "#4f402c"
TOOLTIP_TEXT_COLOR = WIDGET_TEXT_COLOR
DISCLOSURE_CHEVRON_SIZE = 14
VISIBILITY_EYE_ICON_SIZE = 16
WARNING_ICON_SIZE = 16
HEADER_LOGO_WIDTH = 220
LOGO_NAVY_COLOR = "#10244B"
LOGO_DARK_SURFACE_COLOR = WIDGET_TEXT_COLOR
FORM_LABEL_STYLESHEET = (
    f"color: {WIDGET_TEXT_SECONDARY_COLOR}; font-weight: 600; padding-top: 6px; background: transparent;"
)
ACTION_BUTTON_STYLESHEET = (
    "QPushButton {"
    f"background-color: {WIDGET_PANEL_COLOR}; "
    f"border: 1px solid {WIDGET_BORDER_COLOR}; "
    "border-radius: 8px; "
    f"color: {WIDGET_TEXT_COLOR}; "
    "font-weight: 600; "
    "padding: 4px 10px; "
    "min-height: 30px;}"
    f"QPushButton:hover {{ background-color: {WIDGET_PANEL_MUTED_COLOR}; border-color: {WIDGET_BORDER_STRONG_COLOR}; }}"
    f"QPushButton:pressed {{ background-color: {WIDGET_PRESSED_COLOR}; border-color: {WIDGET_BORDER_STRONG_COLOR}; }}"
    f"QPushButton:disabled {{ background-color: {WIDGET_PANEL_SUBTLE_COLOR}; "
    f"border-color: {WIDGET_DISABLED_BORDER_COLOR}; color: {WIDGET_DISABLED_TEXT_COLOR}; }}"
)
SMALL_ACTION_BUTTON_STYLESHEET = (
    "QPushButton {"
    f"background-color: {WIDGET_PANEL_COLOR}; "
    f"border: 1px solid {WIDGET_BORDER_COLOR}; "
    "border-radius: 8px; "
    f"color: {WIDGET_TEXT_COLOR}; "
    "font-weight: 600; "
    "padding: 4px 10px; "
    "min-height: 20px;}"
    f"QPushButton:hover {{ background-color: {WIDGET_PANEL_MUTED_COLOR}; border-color: {WIDGET_BORDER_STRONG_COLOR}; }}"
    f"QPushButton:pressed {{ background-color: {WIDGET_PRESSED_COLOR}; border-color: {WIDGET_BORDER_STRONG_COLOR}; }}"
    f"QPushButton:disabled {{ background-color: {WIDGET_PANEL_SUBTLE_COLOR}; "
    f"border-color: {WIDGET_DISABLED_BORDER_COLOR}; color: {WIDGET_DISABLED_TEXT_COLOR}; }}"
)
CALCULATE_BUTTON_STYLESHEET = ACTION_BUTTON_STYLESHEET
CHECKBOX_STYLESHEET = (
    "QCheckBox {"
    f"color: {WIDGET_TEXT_COLOR}; "
    "font-weight: 500; "
    "spacing: 8px; "
    "background: transparent;}"
    f"QCheckBox:disabled {{ color: {WIDGET_DISABLED_TEXT_COLOR}; }}"
    "QCheckBox::indicator {"
    "width: 16px; "
    "height: 16px; "
    "border-radius: 4px; "
    f"border: 1px solid {WIDGET_BORDER_COLOR}; "
    f"background-color: {WIDGET_PANEL_COLOR};}}"
    f"QCheckBox::indicator:hover {{ border-color: {WIDGET_ACCENT_BORDER_COLOR}; }}"
    "QCheckBox::indicator:checked {"
    f"border-color: {WIDGET_ACCENT_COLOR}; "
    f"background-color: {WIDGET_ACCENT_COLOR};}}"
    "QCheckBox::indicator:disabled {"
    f"border-color: {WIDGET_DISABLED_BORDER_COLOR}; "
    f"background-color: {WIDGET_PANEL_MUTED_COLOR};}}"
)
COMPLETER_POPUP_STYLESHEET = (
    "QAbstractItemView {"
    f"background-color: {WIDGET_PANEL_COLOR}; "
    f"border: 1px solid {WIDGET_BORDER_COLOR}; "
    f"color: {WIDGET_TEXT_COLOR}; "
    f"selection-background-color: {WIDGET_SELECTION_COLOR}; "
    f"selection-color: {WIDGET_TEXT_COLOR}; "
    "outline: 0px;}"
    "QAbstractItemView::item { padding: 4px 8px; }"
)
StatusCardKind = Literal["info", "warning", "success", "error"]
STATUS_CARD_KINDS: tuple[StatusCardKind, ...] = ("info", "warning", "success", "error")
STATUS_CARD_PALETTE: dict[StatusCardKind, dict[str, str]] = {
    "info": {"text": "#9ecbff", "border": "#5f8fd9", "background": "#28384f"},
    "warning": {
        "text": WIDGET_WARNING_TEXT_COLOR,
        "border": WIDGET_WARNING_BORDER_COLOR,
        "background": WIDGET_WARNING_BACKGROUND_COLOR,
    },
    "success": {"text": "#8ee6a6", "border": WIDGET_SUCCESS_COLOR, "background": "#253b30"},
    "error": {"text": "#ff9aa2", "border": "#e06c75", "background": "#4a2b31"},
}
PRIMARY_BUTTON_STYLESHEET = (
    "QPushButton {"
    f"background-color: {WIDGET_SUCCESS_COLOR}; "
    f"color: {WIDGET_PRIMARY_BUTTON_TEXT_COLOR}; "
    f"border: 1px solid {WIDGET_SUCCESS_COLOR}; "
    "border-radius: 6px; "
    "padding: 7px 14px; "
    "font-weight: 600;}"
    f"QPushButton:hover {{ background-color: {WIDGET_SUCCESS_HOVER_COLOR}; border-color: {WIDGET_SUCCESS_HOVER_COLOR}; }}"
)
WARNING_BUTTON_STYLESHEET = (
    "QPushButton {"
    f"background-color: {WIDGET_WARNING_BACKGROUND_COLOR}; "
    f"color: {WIDGET_WARNING_TEXT_COLOR}; "
    f"border: 1px solid {WIDGET_WARNING_BORDER_COLOR}; "
    "border-radius: 6px; "
    "padding: 7px 14px; "
    "font-weight: 600;}"
    f"QPushButton:hover {{ background-color: {WIDGET_WARNING_HOVER_COLOR}; }}"
)
SECONDARY_BUTTON_STYLESHEET = (
    "QPushButton {"
    f"background-color: {WIDGET_PANEL_COLOR}; "
    f"color: {WIDGET_TEXT_COLOR}; "
    f"border: 1px solid {WIDGET_BORDER_COLOR}; "
    "border-radius: 6px; "
    "padding: 7px 14px;}"
    f"QPushButton:hover {{ background-color: {WIDGET_PANEL_MUTED_COLOR}; }}"
)
_INLINE_BACKTICK_RE = re.compile(r"`([^`]+)`")


def validate_status_card_kind(kind: str) -> StatusCardKind:
    """Return a validated status-card kind, or fail loudly for invalid input."""
    if kind not in STATUS_CARD_KINDS:
        allowed_kinds = ", ".join(repr(allowed_kind) for allowed_kind in STATUS_CARD_KINDS)
        raise ValueError(f"Invalid status card kind {kind!r}. Expected one of: {allowed_kinds}.")

    return cast(StatusCardKind, kind)


def create_disclosure_chevron_icon(*, expanded: bool, color: str = WIDGET_TEXT_COLOR) -> QIcon:
    pixmap = QPixmap(DISCLOSURE_CHEVRON_SIZE, DISCLOSURE_CHEVRON_SIZE)
    pixmap.fill(Qt.GlobalColor.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    pen = QPen(QColor(color))
    pen.setWidthF(2.0)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)

    if expanded:
        painter.drawLine(QPointF(3.5, 5.0), QPointF(7.0, 8.5))
        painter.drawLine(QPointF(7.0, 8.5), QPointF(10.5, 5.0))
    else:
        painter.drawLine(QPointF(5.0, 3.5), QPointF(8.5, 7.0))
        painter.drawLine(QPointF(8.5, 7.0), QPointF(5.0, 10.5))

    painter.end()
    return QIcon(pixmap)


def create_visibility_eye_icon(*, visible: bool, color: str = WIDGET_TEXT_COLOR) -> QIcon:
    """Return a compact open-eye or crossed-eye icon."""
    pixmap = QPixmap(VISIBILITY_EYE_ICON_SIZE, VISIBILITY_EYE_ICON_SIZE)
    pixmap.fill(Qt.GlobalColor.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    pen = QPen(QColor(color))
    pen.setWidthF(1.6)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)

    eye = QPainterPath()
    eye.moveTo(1.5, 8.0)
    eye.cubicTo(4.0, 3.4, 7.0, 2.8, 8.0, 2.8)
    eye.cubicTo(9.0, 2.8, 12.0, 3.4, 14.5, 8.0)
    eye.cubicTo(12.0, 12.6, 9.0, 13.2, 8.0, 13.2)
    eye.cubicTo(7.0, 13.2, 4.0, 12.6, 1.5, 8.0)
    painter.drawPath(eye)
    painter.drawEllipse(QPointF(8.0, 8.0), 2.1, 2.1)

    if not visible:
        outline_pen = QPen(QColor(WIDGET_SURFACE_COLOR))
        outline_pen.setWidthF(3.8)
        outline_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(outline_pen)
        painter.drawLine(QPointF(2.2, 2.2), QPointF(13.8, 13.8))
        painter.setPen(pen)
        painter.drawLine(QPointF(2.2, 2.2), QPointF(13.8, 13.8))

    painter.end()
    return QIcon(pixmap)


def create_warning_icon(*, color: str = WIDGET_WARNING_TEXT_COLOR) -> QIcon:
    """Return a compact warning triangle with an exclamation mark."""
    pixmap = QPixmap(WARNING_ICON_SIZE, WARNING_ICON_SIZE)
    pixmap.fill(Qt.GlobalColor.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    pen = QPen(QColor(color))
    pen.setWidthF(1.6)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)

    triangle = QPainterPath()
    triangle.moveTo(8.0, 1.8)
    triangle.lineTo(14.6, 13.8)
    triangle.lineTo(1.4, 13.8)
    triangle.closeSubpath()
    painter.drawPath(triangle)
    painter.drawLine(QPointF(8.0, 6.0), QPointF(8.0, 9.2))
    painter.drawPoint(QPointF(8.0, 11.5))

    painter.end()
    return QIcon(pixmap)


def _render_header_logo_pixmap(device_pixel_ratio: float) -> QPixmap | None:
    try:
        svg = get_logo_path().read_text(encoding="utf-8")
    except OSError:
        return None

    # The brand navy is unreadable on the dark widget surface, so draw it in a light color instead.
    renderer = QSvgRenderer(QByteArray(svg.replace(LOGO_NAVY_COLOR, LOGO_DARK_SURFACE_COLOR).encode("utf-8")))
    if not renderer.isValid():
        return None

    # Render at physical pixels so the logo stays sharp on high-DPI screens.
    default_size = renderer.defaultSize()
    width = round(HEADER_LOGO_WIDTH * device_pixel_ratio)
    height = round(default_size.height() * width / default_size.width())
    pixmap = QPixmap(width, height)
    pixmap.fill(Qt.GlobalColor.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    renderer.render(painter, QRectF(0, 0, width, height))
    painter.end()

    pixmap.setDevicePixelRatio(device_pixel_ratio)
    return pixmap


def create_header_logo(object_name: str) -> QLabel:
    """Return the centered widget header logo, falling back to a text title."""
    logo_label = QLabel()
    logo_label.setObjectName(object_name)
    logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

    logo_pixmap = _render_header_logo_pixmap(logo_label.devicePixelRatioF())
    if logo_pixmap is not None:
        logo_label.setPixmap(logo_pixmap)
        return logo_label

    logo_label.setText("spatiato")
    logo_label.setStyleSheet(f"color: {WIDGET_TEXT_COLOR}; font-size: 18px; font-weight: 600;")
    return logo_label


def build_input_control_stylesheet(control_selector: str) -> str:
    selectors = [selector.strip() for selector in control_selector.split(",")]
    disabled_selector = ", ".join(f"{selector}:disabled" for selector in selectors)
    focus_selector = ", ".join(f"{selector}:focus" for selector in selectors)

    return (
        f"{control_selector} {{"
        f"background-color: {WIDGET_PANEL_COLOR}; "
        f"border: 1px solid {WIDGET_BORDER_COLOR}; "
        "border-radius: 8px; "
        f"color: {WIDGET_TEXT_COLOR}; "
        "padding: 4px 10px; "
        "min-height: 20px;}"
        f"{disabled_selector} {{"
        f"background-color: {WIDGET_PANEL_MUTED_COLOR}; "
        f"border-color: {WIDGET_DISABLED_BORDER_COLOR}; "
        f"color: {WIDGET_DISABLED_TEXT_COLOR};}}"
        f"{focus_selector} {{"
        f"border-color: {WIDGET_ACCENT_COLOR}; "
        f"background-color: {WIDGET_PANEL_COLOR};}}"
        "QComboBox { padding-right: 24px; }"
        "QComboBox::drop-down {"
        "subcontrol-origin: padding; "
        "subcontrol-position: top right; "
        "width: 24px; "
        "border: 0px; "
        "background: transparent;}"
    )


class CompactComboBox(QComboBox):
    """Combo box with a capped width hint for long item names."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        minimum_contents_length: int = 12,
    ) -> None:
        super().__init__(parent)
        self.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.setMinimumContentsLength(minimum_contents_length)
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.currentIndexChanged.connect(self._update_current_text_tooltip)
        self._update_current_text_tooltip()

    def sizeHint(self) -> QSize:
        return self._capped_size_hint(super().sizeHint())

    def minimumSizeHint(self) -> QSize:
        return self._capped_size_hint(super().minimumSizeHint())

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_current_text_tooltip()

    def paintEvent(self, event) -> None:
        del event
        painter = QStylePainter(self)
        option = QStyleOptionComboBox()
        self.initStyleOption(option)
        option.currentText = self._elided_current_text(option)
        if self.currentIndex() < 0 and self.placeholderText():
            placeholder_color = option.palette.color(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text)
            option.palette.setColor(QPalette.ColorRole.ButtonText, placeholder_color)
            option.palette.setColor(QPalette.ColorRole.Text, placeholder_color)
        painter.drawComplexControl(QStyle.ComplexControl.CC_ComboBox, option)
        painter.drawControl(QStyle.ControlElement.CE_ComboBoxLabel, option)

    def _capped_size_hint(self, hint: QSize) -> QSize:
        cap_width = self.fontMetrics().horizontalAdvance("M" * max(1, self.minimumContentsLength())) + 48
        return QSize(min(hint.width(), cap_width), hint.height())

    def _elided_current_text(self, option: QStyleOptionComboBox | None = None) -> str:
        if option is None:
            option = QStyleOptionComboBox()
            self.initStyleOption(option)

        current_text = option.currentText
        if not current_text and self.currentIndex() < 0:
            current_text = self.placeholderText()

        text_rect = self.style().subControlRect(
            QStyle.ComplexControl.CC_ComboBox,
            option,
            QStyle.SubControl.SC_ComboBoxEditField,
            self,
        )
        available_width = max(0, text_rect.width())
        return self.fontMetrics().elidedText(current_text, Qt.TextElideMode.ElideRight, available_width)

    def _update_current_text_tooltip(self, _index: int | None = None) -> None:
        current_text = self.currentText()
        if not current_text:
            self.setToolTip("")
            return

        elided_text = self._elided_current_text()
        self.setToolTip(format_tooltip(current_text) if elided_text != current_text else "")


class CompleterPopupLineEdit(QLineEdit):
    """Line edit that can open its completer popup when entered."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._show_completion_popup_on_entry = False
        self._accepted_completion_text_to_clear: str | None = None
        self._completion_clear_timer = QTimer(self)
        self._completion_clear_timer.setSingleShot(True)
        self._completion_clear_timer.timeout.connect(self._clear_reinserted_completion)

    def set_completion_popup_on_entry_enabled(self, enabled: bool) -> None:
        self._show_completion_popup_on_entry = enabled

    def clear_after_accepted_completion(self, expected_text: str) -> None:
        """Clear search text that produced an accepted completion.

        This is opt-in because some consumers use the completed text as a
        persistent selection. Only text contained in the accepted completion
        is cleared immediately. A guarded deferred clear also handles Qt
        writing the activated completion again after consumer callbacks return.
        """
        normalized_expected_text = expected_text.strip().casefold()
        normalized_current_text = self.text().strip().casefold()
        if normalized_current_text and normalized_current_text in normalized_expected_text:
            self.clear()

        self._accepted_completion_text_to_clear = expected_text
        self._completion_clear_timer.start(0)

    def focusInEvent(self, event) -> None:
        super().focusInEvent(event)
        self.show_completion_popup()

    def mousePressEvent(self, event) -> None:
        super().mousePressEvent(event)
        self.show_completion_popup()

    def show_completion_popup(self) -> None:
        if not self._show_completion_popup_on_entry or not self.isEnabled():
            return

        completer = self.completer()
        if completer is None:
            return

        model = completer.model()
        if model is None or model.rowCount() == 0:
            return

        completer.setCompletionPrefix(self.text())
        completer.complete()

    def _clear_reinserted_completion(self) -> None:
        """Clear the completion Qt may rewrite after activation callbacks.

        On Qt's Cocoa path, popup activation writes the selected item before
        emitting ``QCompleter.activated`` and may write it again after connected
        callbacks return. A consumer clearing synchronously during its callback
        is therefore overwritten. The single-shot timer invokes this method on
        the next event-loop turn. Only the accepted completion is cleared, so
        newer, unrelated input is preserved.
        """
        expected_text = self._accepted_completion_text_to_clear
        self._accepted_completion_text_to_clear = None
        self._completion_clear_timer.stop()
        if expected_text is None:
            return
        if self.text().strip().casefold() == expected_text.strip().casefold():
            self.clear()


def apply_widget_surface(widget: QWidget) -> None:
    widget.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
    widget.setAutoFillBackground(True)
    palette = widget.palette()
    palette.setColor(QPalette.ColorRole.Window, QColor(WIDGET_SURFACE_COLOR))
    widget.setPalette(palette)
    widget.setStyleSheet(WIDGET_SURFACE_STYLESHEET)


def apply_scroll_content_surface(widget: QWidget) -> None:
    widget.setAutoFillBackground(True)
    palette = widget.palette()
    palette.setColor(QPalette.ColorRole.Window, QColor(WIDGET_SURFACE_COLOR))
    widget.setPalette(palette)
    widget.setStyleSheet(WIDGET_SURFACE_STYLESHEET)


def create_form_label(text: str) -> QLabel:
    label = QLabel(text)
    label.setStyleSheet(FORM_LABEL_STYLESHEET)
    label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
    return label


def format_tooltip(message: str) -> str:
    message = format_feedback_text(message)
    escaped_message = (
        escape(message).replace("_", "_&#8203;").replace("/", "/&#8203;").replace("-", "-&#8203;").replace("\n", "<br>")
    )
    return f"<qt><div style='color: {TOOLTIP_TEXT_COLOR}; max-width: 360px;'>{escaped_message}</div></qt>"


class _ElidedLabel(QLabel):
    """Single-line label that shows a tooltip only when the text is elided."""

    def __init__(self, text: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._full_text = text
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setMinimumWidth(0)
        self.setMinimumHeight(36)
        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self._update_elided_text()

    def set_full_text(self, text: str) -> None:
        self._full_text = text
        self._update_elided_text()

    def resizeEvent(self, event: object) -> None:
        super().resizeEvent(event)
        self._update_elided_text()

    def _update_elided_text(self) -> None:
        available_width = max(0, self.contentsRect().width())
        elided_text = self.fontMetrics().elidedText(self._full_text, Qt.TextElideMode.ElideRight, available_width)
        super().setText(elided_text)
        self.setToolTip(format_tooltip(self._full_text) if elided_text != self._full_text else "")


def format_feedback_text(message: str) -> str:
    """Return text with napari-friendly inline identifier styling."""
    return _INLINE_BACKTICK_RE.sub(r'"\1"', message)


def format_feedback_identifier(name: str, *, max_length: int = 56) -> tuple[str, bool]:
    """Return a visible identifier and whether it was shortened."""
    if max_length < 4:
        raise ValueError("`max_length` must be at least 4.")
    if len(name) <= max_length:
        return name, False

    head_length = min(32, max_length - 4)
    tail_length = max_length - head_length - 1
    return f"{name[:head_length]}…{name[-tail_length:]}", True


def set_status_card(
    label: QLabel,
    *,
    title: str,
    lines: list[str],
    kind: StatusCardKind,
    tooltip_message: str | None = None,
) -> None:
    """Render a compact titled status card into a QLabel."""
    palette = STATUS_CARD_PALETTE.get(kind, STATUS_CARD_PALETTE["info"])
    formatted_lines = "<br>".join(f"<span>{escape(format_feedback_text(line), quote=False)}</span>" for line in lines)
    label.setText(
        "<div>"
        f"<span style='font-size: 11px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;'>"
        f"{escape(title, quote=False)}</span><br>"
        f"{formatted_lines}"
        "</div>"
    )
    label.setStyleSheet(
        "font-weight: 500; "
        f"color: {palette['text']}; "
        f"background-color: {palette['background']}; "
        f"border: 1px solid {palette['border']}; "
        "border-radius: 8px; "
        "padding: 10px 12px;"
    )
    label.setToolTip(format_tooltip(tooltip_message) if tooltip_message else "")
    label.setVisible(bool(lines))
