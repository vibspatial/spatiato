"""A row that presents one loaded napari layer in a widget card."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

from qtpy.QtCore import QSignalBlocker, QSize, Qt, Signal
from qtpy.QtWidgets import QHBoxLayout, QLabel, QPushButton, QToolButton, QWidget

from spatiato.viewer.adapter import BaseLayerBinding
from spatiato.widgets.shared_styles import (
    WARNING_ICON_SIZE,
    WIDGET_BORDER_COLOR,
    WIDGET_PANEL_MUTED_COLOR,
    WIDGET_TEXT_COLOR,
    _ElidedLabel,
    create_visibility_eye_icon,
    create_warning_icon,
    format_tooltip,
)

_VISIBILITY_BUTTON_STYLESHEET = (
    "QToolButton {"
    "background: transparent; "
    "border: 1px solid transparent; "
    "border-radius: 5px; "
    "padding: 2px;}"
    f"QToolButton:hover {{ background-color: {WIDGET_PANEL_MUTED_COLOR}; "
    f"border-color: {WIDGET_BORDER_COLOR}; }}"
    f"QToolButton:focus {{ border-color: {WIDGET_TEXT_COLOR}; }}"
)


class _LayerRow(QWidget):
    """Present one loaded layer and bridge user intent with live napari state.

    A row shows an eye toggle, the layer's label and a remove button. Between
    the label and the remove button it can hold extra controls, added with
    `add_control(...)`, and a warning, set with `set_warning(...)`. It accepts
    the binding of any layer type: labels, shapes, points or images.

    A row keeps its construction-time binding for its entire lifetime. Its owner
    must dispose this row and construct another one when the binding changes.

    User actions never mutate the layer directly. Identity-free visibility and
    removal intent returns to the owning widget, which binds its own identity
    while connecting the signals. The owner validates the current live binding
    before assigning ``layer.visible`` or requesting removal through
    ``ViewerAdapter``. The same holds for added controls: a control emits
    intent, and the owner, which connects its signals, mutates the layer.

    A successful property assignment causes napari to emit
    ``layer.events.visible``. Every row bound to that layer receives the native
    event, reads the accepted layer property, and refreshes its presentation.
    Owners do not manually emit napari events or render a requested value as
    though napari had already accepted it.

    Explicit presentation refreshes, with `refresh_presentation()`, are used for
    no-op or rejected assignments, where no native property-change event may be
    emitted. Presentation updates use ``QSignalBlocker`` where needed so
    reflecting napari state cannot emit another user request.
    """

    remove_requested = Signal()
    visibility_change_requested = Signal(bool)

    # Prefix of the object names of the row and its widgets. A subclass sets its
    # own, so its rows keep object names of their own.
    _object_name_prefix: ClassVar[str] = "layer"

    def __init__(
        self,
        binding: BaseLayerBinding,
        *,
        display_label: str,
        accessibility_label: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.binding = binding
        self.display_label = display_label
        self.accessibility_label = accessibility_label
        self._warning_message: str | None = None
        self._disposed = False
        prefix = self._object_name_prefix
        self.setObjectName(f"{prefix}_row")

        self._row_layout = QHBoxLayout(self)
        self._row_layout.setContentsMargins(0, 0, 0, 0)
        self._row_layout.setSpacing(8)

        self.visibility_button = QToolButton(self)
        self.visibility_button.setObjectName(f"{prefix}_visibility_button")
        self.visibility_button.setCheckable(True)
        self.visibility_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.visibility_button.setFixedSize(28, 22)
        self.visibility_button.setIconSize(QSize(16, 16))
        self.visibility_button.setStyleSheet(_VISIBILITY_BUTTON_STYLESHEET)

        self.layer_label = _ElidedLabel(display_label, self)
        self.layer_label.setObjectName(f"{prefix}_label")
        self.layer_label.setAccessibleName(accessibility_label)

        self.warning_label = QLabel(self)
        self.warning_label.setObjectName(f"{prefix}_warning_label")
        self.warning_label.setPixmap(create_warning_icon().pixmap(QSize(WARNING_ICON_SIZE, WARNING_ICON_SIZE)))
        self.warning_label.hide()

        self.remove_button = QPushButton("×")
        self.remove_button.setObjectName(f"{prefix}_remove_button")
        remove_message = f"Remove {accessibility_label} from viewer"
        self.remove_button.setAccessibleName(remove_message)
        self.remove_button.setToolTip(format_tooltip(remove_message))
        self.remove_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.remove_button.setFixedWidth(28)
        self.remove_button.clicked.connect(lambda _checked=False: self.remove_requested.emit())

        self.visibility_button.toggled.connect(self._on_visibility_toggled)

        self._row_layout.addWidget(self.visibility_button)
        self._row_layout.addWidget(self.layer_label, 1)
        self._row_layout.addWidget(self.warning_label)
        self._row_layout.addWidget(self.remove_button)

        self.binding.layer.events.visible.connect(self._on_layer_visible_changed)
        self._apply_visibility(self.binding.layer.visible)

    @property
    def warning_message(self) -> str | None:
        """Return the warning shown on the row, or `None` when there is none."""
        return self._warning_message

    def add_control(self, control: QWidget) -> None:
        """Add `control` to the row, between the label and the warning.

        Controls are laid out in the order they are added. The row only lays
        the control out: the owner or subclass that adds it connects its
        signals.
        """
        self._row_layout.insertWidget(self._row_layout.indexOf(self.warning_label), control)

    def set_warning(self, message: str | None) -> None:
        """Show `message` as a warning icon with a tooltip, or remove the warning for `None`."""
        self._warning_message = message
        if message is None:
            self.warning_label.setToolTip("")
            self.warning_label.setAccessibleName("")
            self.warning_label.hide()
            return
        self.warning_label.setToolTip(format_tooltip(message))
        self.warning_label.setAccessibleName(f"Warning: {message}")
        self.warning_label.show()

    def refresh_presentation(self) -> None:
        """Render the row from the authoritative live layer."""
        self._apply_visibility(self.binding.layer.visible)

    def dispose(self) -> None:
        """Idempotently disconnect this row from its napari layer."""
        if self._disposed:
            return
        self._disposed = True
        self._disconnect_layer_events()

    def _disconnect_layer_events(self) -> None:
        """Disconnect the napari layer events the row listens to; a subclass extends it."""
        self._disconnect_layer_event(self.binding.layer.events.visible, self._on_layer_visible_changed)

    @staticmethod
    def _disconnect_layer_event(event: Any, callback: Callable[[object], None]) -> None:
        try:
            event.disconnect(callback)
        except (TypeError, RuntimeError, ValueError):
            pass

    def _on_layer_visible_changed(self, _event: object) -> None:
        self._apply_visibility(self.binding.layer.visible)

    def _on_visibility_toggled(self, visible: bool) -> None:
        self.visibility_change_requested.emit(visible)

    def _apply_visibility(self, visible: bool) -> None:
        # Reflect napari state without turning this programmatic eye update into
        # another owner visibility-change request.
        with QSignalBlocker(self.visibility_button):
            self.visibility_button.setChecked(visible)
        self.visibility_button.setIcon(create_visibility_eye_icon(visible=visible))
        action = "Hide" if visible else "Show"
        message = f"{action} {self.accessibility_label}"
        self.visibility_button.setAccessibleName(message)
        self.visibility_button.setToolTip(format_tooltip(message))
