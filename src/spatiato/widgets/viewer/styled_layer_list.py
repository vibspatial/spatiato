"""The list of loaded styled layers of one labels or shapes element, shared by their cards."""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QScrollArea, QVBoxLayout, QWidget

from spatiato.core._color_source import ShapeColumnColorSourceSpec, TableColorSourceSpec
from spatiato.viewer.adapter import LabelsLayerBinding, ShapesLayerBinding
from spatiato.widgets.layer_row import _LayerRow
from spatiato.widgets.shared_styles import format_tooltip
from spatiato.widgets.viewer.styles import (
    EMPTY_STATE_STYLESHEET,
    SUBSECTION_LABEL_STYLESHEET,
    TRANSPARENT_PANEL_STYLESHEET,
)

type StyleSpec = TableColorSourceSpec | ShapeColumnColorSourceSpec
type StyledLayerBinding = LabelsLayerBinding | ShapesLayerBinding

_MAX_VISIBLE_STYLED_LAYER_ROWS = 10


def _row_accessibility_label(display_label: str) -> str:
    return f"colored layer {display_label}"


class _StyledLayerList(QWidget):
    """List the styled layers of one element that are loaded in the viewer.

    The list shows a count, a "Remove all" button, and one `_LayerRow` per
    styled layer, in napari layer order. Rows are labelled with the source
    label of their spec, such as `obs: leiden`. When the element has more than
    one linked table, rows of table sources are prefixed with the table name,
    as in `table_multi · obs: leiden`; the owning card sets this with
    `set_show_table_names(...)`.

    The owning card passes the complete set of live styled bindings to
    `set_styled_bindings(...)` after every membership change. A row is kept
    while its binding object is unchanged, and gets a new label when the table
    name prefix is switched on or off; a row whose binding changed is disposed
    and built again. The list never loads, removes or changes napari layers.
    Its rows emit intent, which the list forwards with the row's spec as
    `remove_requested` and `visibility_requested` once it has checked that the
    row still holds the binding it was built for.

    The list also keeps the "source no longer available" warnings, by spec. A
    row is rebuilt whenever its binding object changes, so the warning is
    applied again to each row on every reconcile. A warning is dropped when
    its source is no longer loaded.
    """

    remove_requested = Signal(object)
    remove_all_requested = Signal()
    visibility_requested = Signal(object, bool)

    def __init__(self, *, element_name: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.element_name = element_name
        self._rows_by_spec: dict[StyleSpec, _LayerRow] = {}
        self._spec_order: list[StyleSpec] = []
        self._bindings: tuple[StyledLayerBinding, ...] = ()
        self._show_table_names = False
        self._warnings_by_spec: dict[StyleSpec, str] = {}
        self.setObjectName(f"viewer_widget_styled_layers_{element_name}")
        self.setStyleSheet(TRANSPARENT_PANEL_STYLESHEET)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        summary_layout = QHBoxLayout()
        summary_layout.setContentsMargins(0, 0, 0, 0)
        summary_layout.setSpacing(8)

        self.count_label = QLabel()
        self.count_label.setObjectName(f"viewer_widget_styled_layer_count_{element_name}")
        self.count_label.setStyleSheet(SUBSECTION_LABEL_STYLESHEET)

        self.remove_all_button = QPushButton("Remove all")
        self.remove_all_button.setObjectName(f"viewer_widget_remove_all_styled_layers_button_{element_name}")
        self.remove_all_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.remove_all_button.setToolTip(format_tooltip("Remove all colored layers of this element"))
        self.remove_all_button.clicked.connect(self._emit_remove_all_requested)

        summary_layout.addWidget(self.count_label)
        summary_layout.addStretch(1)
        summary_layout.addWidget(self.remove_all_button)

        self.empty_label = QLabel("No colored layers in viewer")
        self.empty_label.setObjectName(f"viewer_widget_no_styled_layers_label_{element_name}")
        self.empty_label.setStyleSheet(EMPTY_STATE_STYLESHEET)

        self.scroll_area = QScrollArea()
        self.scroll_area.setObjectName(f"viewer_widget_styled_layer_scroll_area_{element_name}")
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setStyleSheet("QScrollArea { border: 0px; background: transparent; }")

        self.list_widget = QWidget()
        self.list_widget.setObjectName(f"viewer_widget_styled_layer_list_{element_name}")
        self.list_widget.setStyleSheet(TRANSPARENT_PANEL_STYLESHEET)
        self.list_layout = QVBoxLayout(self.list_widget)
        self.list_layout.setContentsMargins(0, 0, 0, 0)
        self.list_layout.setSpacing(6)
        self.scroll_area.setWidget(self.list_widget)

        layout.addLayout(summary_layout)
        layout.addWidget(self.empty_label)
        layout.addWidget(self.scroll_area)

        self._refresh_presentation()

    @property
    def rows(self) -> list[_LayerRow]:
        """Return the rows in napari layer order."""
        return [self._rows_by_spec[spec] for spec in self._spec_order]

    @property
    def loaded_specs(self) -> tuple[StyleSpec, ...]:
        """Return the specs of the loaded styled layers, in napari layer order."""
        return tuple(self._spec_order)

    def row_for(self, style_spec: StyleSpec) -> _LayerRow | None:
        """Return the row of `style_spec`, or `None` when its layer is not loaded."""
        return self._rows_by_spec.get(style_spec)

    def warning_for(self, style_spec: StyleSpec) -> str | None:
        """Return the warning of `style_spec`, or `None` when it has none."""
        return self._warnings_by_spec.get(style_spec)

    def validate_styled_bindings(self, bindings: Sequence[StyledLayerBinding]) -> None:
        """Raise `ValueError` unless every binding has a spec and no two have the same one."""
        seen_specs: set[StyleSpec] = set()
        for binding in bindings:
            style_spec = binding.style_spec
            if style_spec is None:
                raise ValueError(f"A styled layer of `{self.element_name}` has no color source.")
            if style_spec in seen_specs:
                raise ValueError(
                    f"Found multiple live styled layers for `{self.element_name}` "
                    f"and source `{style_spec.source_label}`."
                )
            seen_specs.add(style_spec)

    def set_styled_bindings(self, bindings: Sequence[StyledLayerBinding]) -> None:
        """Render one complete snapshot of the live styled bindings, in napari layer order.

        The snapshot is validated with `validate_styled_bindings(...)` before
        any row changes.
        """
        ordered_bindings = tuple(bindings)
        self.validate_styled_bindings(ordered_bindings)
        self._bindings = ordered_bindings
        self._reconcile_rows()

    def set_show_table_names(self, show: bool) -> None:
        """Prefix the rows of table sources with their table name, or stop doing so."""
        if show == self._show_table_names:
            return
        self._show_table_names = show
        self._reconcile_rows()

    def set_styled_source_warning(self, style_spec: StyleSpec, message: str | None) -> None:
        """Show `message` on the row of `style_spec`, or clear its warning for `None`.

        A warning for a source that is not loaded is ignored.
        """
        row = self._rows_by_spec.get(style_spec)
        if row is None:
            return
        if message is None:
            self._warnings_by_spec.pop(style_spec, None)
        else:
            self._warnings_by_spec[style_spec] = message
        row.set_warning(message)

    def refresh_row_presentation(self, style_spec: StyleSpec, binding: StyledLayerBinding) -> None:
        """Refresh the row of `style_spec` only when it holds `binding`."""
        row = self._rows_by_spec.get(style_spec)
        if row is not None and row.binding is binding:
            row.refresh_presentation()

    def dispose(self) -> None:
        """Disconnect every row from its napari layer."""
        for row in self._rows_by_spec.values():
            row.dispose()

    def _display_label(self, style_spec: StyleSpec) -> str:
        if self._show_table_names and isinstance(style_spec, TableColorSourceSpec):
            return f"{style_spec.table_name} · {style_spec.source_label}"
        return style_spec.source_label

    def _reconcile_rows(self) -> None:
        bindings_by_spec: dict[StyleSpec, StyledLayerBinding] = {}
        next_specs: list[StyleSpec] = []
        for binding in self._bindings:
            style_spec = binding.style_spec
            assert style_spec is not None
            bindings_by_spec[style_spec] = binding
            next_specs.append(style_spec)

        for style_spec in tuple(self._rows_by_spec):
            row = self._rows_by_spec[style_spec]
            if bindings_by_spec.get(style_spec) is row.binding:
                continue
            self._rows_by_spec.pop(style_spec)
            row.dispose()
            self.list_layout.removeWidget(row)
            row.deleteLater()

        for style_spec in next_specs:
            binding = bindings_by_spec[style_spec]
            display_label = self._display_label(style_spec)
            row = self._rows_by_spec.get(style_spec)
            if row is None:
                row = _LayerRow(
                    binding,
                    display_label=display_label,
                    accessibility_label=_row_accessibility_label(display_label),
                    parent=self.list_widget,
                )
                row.remove_requested.connect(partial(self._emit_remove_requested, binding))
                row.visibility_change_requested.connect(partial(self._emit_visibility_requested, binding))
                self._rows_by_spec[style_spec] = row
            else:
                if row.display_label != display_label:
                    row.set_display_label(display_label, accessibility_label=_row_accessibility_label(display_label))
                row.refresh_presentation()

        for style_spec in tuple(self._warnings_by_spec):
            if style_spec not in bindings_by_spec:
                del self._warnings_by_spec[style_spec]
        for style_spec, row in self._rows_by_spec.items():
            row.set_warning(self._warnings_by_spec.get(style_spec))

        for row in self._rows_by_spec.values():
            self.list_layout.removeWidget(row)
        for style_spec in next_specs:
            self.list_layout.addWidget(self._rows_by_spec[style_spec])

        self._spec_order = next_specs
        self._refresh_presentation()

    def _emit_remove_requested(self, expected_binding: StyledLayerBinding) -> None:
        style_spec = expected_binding.style_spec
        row = self._rows_by_spec.get(style_spec) if style_spec is not None else None
        if row is None or row.binding is not expected_binding:
            return
        self.remove_requested.emit(style_spec)

    def _emit_visibility_requested(self, expected_binding: StyledLayerBinding, visible: bool) -> None:
        style_spec = expected_binding.style_spec
        row = self._rows_by_spec.get(style_spec) if style_spec is not None else None
        if row is None or row.binding is not expected_binding:
            return
        self.visibility_requested.emit(style_spec, visible)

    def _emit_remove_all_requested(self, _checked: bool = False) -> None:
        if not self._spec_order:
            return
        self.remove_all_requested.emit()

    def _refresh_presentation(self) -> None:
        count = len(self._spec_order)
        self.count_label.setText(f"{count} colored layer" if count == 1 else f"{count} colored layers")
        self.empty_label.setVisible(count == 0)
        self.scroll_area.setVisible(count > 0)
        self.remove_all_button.setVisible(count > 0)
        self._set_scroll_height(self.rows)

    def _set_scroll_height(self, rows: list[_LayerRow]) -> None:
        visible_rows = rows[:_MAX_VISIBLE_STYLED_LAYER_ROWS]
        if not visible_rows:
            self.list_widget.setMinimumHeight(0)
            self.scroll_area.setFixedHeight(0)
            return

        spacing = self.list_layout.spacing()
        margins = self.list_layout.contentsMargins()
        vertical_margins = margins.top() + margins.bottom()
        content_height = sum(row.sizeHint().height() for row in rows)
        content_height += spacing * max(0, len(rows) - 1)
        content_height += vertical_margins
        self.list_widget.setMinimumHeight(content_height)

        visible_height = sum(row.sizeHint().height() for row in visible_rows)
        visible_height += spacing * max(0, len(visible_rows) - 1)
        visible_height += vertical_margins
        visible_height += self.scroll_area.frameWidth() * 2
        # A maximum alone allows the parent layout to compress the viewport and
        # introduce a scrollbar before the configured row cap is reached.
        self.scroll_area.setFixedHeight(visible_height)
