from __future__ import annotations

from dataclasses import dataclass

from qtpy.QtCore import QSignalBlocker, QStringListModel, Qt, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QCompleter,
    QFormLayout,
    QFrame,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
)

from spatiato.core._color_source import (
    ShapeColorSourceKind,
    ShapeColumnColorSourceSpec,
    TableColorSourceKind,
    TableColorSourceSpec,
)
from spatiato.widgets.shared_styles import (
    ACTION_BUTTON_STYLESHEET,
    CHECKBOX_STYLESHEET,
    COMPLETER_POPUP_STYLESHEET,
    CompactComboBox,
    CompleterPopupLineEdit,
    _ElidedLabel,
    build_input_control_stylesheet,
    create_form_label,
)
from spatiato.widgets.viewer._table_color_sources import ColorSourceKindEntry, LinkedTableColorSources
from spatiato.widgets.viewer.styles import (
    CARD_TITLE_STYLESHEET,
    DETAIL_PANEL_STYLESHEET,
    INPUT_CONTROL_STYLESHEET,
    SUMMARY_LABEL_STYLESHEET,
)

_SHAPE_COLUMN_PLACEHOLDER = "Select column"
_OBS_SOURCE_PLACEHOLDER = "Select obs column"
_VAR_SOURCE_PLACEHOLDER = "Select var"
_NO_COLOR_SOURCE_LABEL = "No color source"
_SHAPE_COLUMN_KIND_ENTRY = ColorSourceKindEntry(label="Shapes column", source_kind="shape_column")


@dataclass(frozen=True)
class ShapesLoadRequest:
    shapes_name: str
    table_name: str | None
    selected_source_kind: ShapeColorSourceKind | TableColorSourceKind | None
    selected_color_source: ShapeColumnColorSourceSpec | TableColorSourceSpec | None
    fill_shapes: bool


class _ShapesCardWidget(QFrame):
    """Card UI shell for one shapes element in the selected coordinate system."""

    add_update_requested = Signal(object)

    def __init__(
        self,
        *,
        shapes_name: str,
        shape_column_color_sources: list[ShapeColumnColorSourceSpec],
        linked_tables: dict[str, LinkedTableColorSources],
    ) -> None:
        super().__init__()
        self.shapes_name = shapes_name
        self._shape_column_color_sources = shape_column_color_sources
        self._linked_tables = linked_tables
        self._filtered_color_sources: list[ShapeColumnColorSourceSpec | TableColorSourceSpec] = []
        self._active_source_kind: ShapeColorSourceKind | TableColorSourceKind | None = None
        self.setObjectName(f"viewer_widget_shapes_card_{shapes_name}")
        self.setProperty("spatiatoViewerDetailPanel", True)
        self.setStyleSheet(DETAIL_PANEL_STYLESHEET)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self.title_label = _ElidedLabel(shapes_name, self)
        self.title_label.setObjectName(f"viewer_widget_shapes_card_title_{shapes_name}")
        self.title_label.setStyleSheet(CARD_TITLE_STYLESHEET)
        self.title_label.hide()

        form_layout = QFormLayout()
        form_layout.setContentsMargins(0, 0, 0, 0)
        form_layout.setHorizontalSpacing(8)
        form_layout.setVerticalSpacing(6)

        linked_table_label = create_form_label("Linked table")
        self.linked_table_combo = CompactComboBox(minimum_contents_length=8)
        self.linked_table_combo.setObjectName(f"viewer_widget_shapes_linked_table_combo_{shapes_name}")
        self.linked_table_combo.setStyleSheet(INPUT_CONTROL_STYLESHEET)
        table_names = list(linked_tables)
        if table_names:
            self.linked_table_combo.addItems(table_names)
        else:
            self.linked_table_combo.addItem("No linked tables")
            self.linked_table_combo.setEnabled(False)

        color_source_kind_label = create_form_label("Color source")
        self.color_source_kind_combo = CompactComboBox(minimum_contents_length=8)
        self.color_source_kind_combo.setObjectName(f"viewer_widget_shapes_color_source_kind_combo_{shapes_name}")
        self.color_source_kind_combo.setStyleSheet(INPUT_CONTROL_STYLESHEET)
        self.color_source_kind_combo.addItem(_NO_COLOR_SOURCE_LABEL, None)

        self.color_source_value_label = create_form_label("Shapes column")
        self.color_source_value_input = CompleterPopupLineEdit()
        self.color_source_value_input.setObjectName(f"viewer_widget_shapes_color_source_value_input_{shapes_name}")
        self.color_source_value_input.setStyleSheet(build_input_control_stylesheet("QLineEdit"))
        self.color_source_value_input.setMinimumWidth(0)
        self.color_source_value_input.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.color_source_value_input.setEnabled(False)

        self.fill_toggle = QCheckBox("Fill")
        self.fill_toggle.setObjectName(f"viewer_widget_shapes_fill_toggle_{shapes_name}")
        self.fill_toggle.setStyleSheet(CHECKBOX_STYLESHEET)
        self.fill_toggle.setChecked(False)

        self._color_source_completer_model = QStringListModel(self.color_source_value_input)
        self._color_source_completer = QCompleter(self._color_source_completer_model, self.color_source_value_input)
        self._color_source_completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        self._color_source_completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self._color_source_completer.setFilterMode(Qt.MatchFlag.MatchContains)
        self._color_source_completer.setMaxVisibleItems(10)
        self._color_source_completer.popup().setStyleSheet(COMPLETER_POPUP_STYLESHEET)
        self.color_source_value_input.setCompleter(self._color_source_completer)

        self.action_hint_label = QLabel()
        self.action_hint_label.setObjectName(f"viewer_widget_shapes_action_hint_{shapes_name}")
        self.action_hint_label.setWordWrap(True)
        self.action_hint_label.setStyleSheet(SUMMARY_LABEL_STYLESHEET)

        self.add_update_button = QPushButton("Add / Update in viewer")
        self.add_update_button.setObjectName(f"viewer_widget_add_update_shapes_button_{shapes_name}")
        self.add_update_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.add_update_button.setMinimumHeight(28)
        self.add_update_button.setStyleSheet(ACTION_BUTTON_STYLESHEET)
        self.add_update_button.clicked.connect(self._emit_add_update_request)
        self.add_update_button.setToolTip("")

        form_layout.addRow(linked_table_label, self.linked_table_combo)
        form_layout.addRow(color_source_kind_label, self.color_source_kind_combo)
        form_layout.addRow(self.color_source_value_label, self.color_source_value_input)
        form_layout.addRow(create_form_label("Display"), self.fill_toggle)

        layout.addLayout(form_layout)
        layout.addWidget(self.action_hint_label)
        layout.addWidget(self.add_update_button)

        self.linked_table_combo.currentIndexChanged.connect(self._refresh_color_source_controls)
        self.color_source_kind_combo.currentIndexChanged.connect(self._refresh_color_source_controls)
        self.color_source_value_input.textChanged.connect(self._update_action_status)
        self.color_source_value_input.editingFinished.connect(self._sync_current_source_selection)

        self._refresh_color_source_controls()

    @property
    def selected_table_name(self) -> str | None:
        if not self.linked_table_combo.isEnabled():
            return None

        table_name = self.linked_table_combo.currentText()
        return table_name if table_name in self._linked_tables else None

    @property
    def selected_kind_entry(self) -> ColorSourceKindEntry | None:
        """Return the selected "Color source" entry, or `None` for "No color source"."""
        entry = self.color_source_kind_combo.currentData()
        return entry if isinstance(entry, ColorSourceKindEntry) else None

    @property
    def selected_source_kind(self) -> ShapeColorSourceKind | TableColorSourceKind | None:
        entry = self.selected_kind_entry
        return None if entry is None else entry.source_kind

    @property
    def selected_color_source(self) -> ShapeColumnColorSourceSpec | TableColorSourceSpec | None:
        current_text = self.color_source_value_input.text().strip()
        for source in self._filtered_color_sources:
            if source.display_name == current_text:
                return source
        return None

    @property
    def fill_shapes(self) -> bool:
        return self.fill_toggle.isChecked()

    def set_linked_tables(self, linked_tables: dict[str, LinkedTableColorSources]) -> None:
        previous_table_name = self.selected_table_name
        previous_kind_entry = self.selected_kind_entry
        previous_source_identity = (
            self.selected_color_source.identity if self.selected_color_source is not None else None
        )

        self._linked_tables = linked_tables
        table_names = list(linked_tables)
        with QSignalBlocker(self.linked_table_combo):
            self.linked_table_combo.clear()
            if table_names:
                self.linked_table_combo.addItems(table_names)
                self.linked_table_combo.setEnabled(True)
                next_index = table_names.index(previous_table_name) if previous_table_name in table_names else 0
                self.linked_table_combo.setCurrentIndex(next_index)
            else:
                self.linked_table_combo.addItem("No linked tables")
                self.linked_table_combo.setEnabled(False)
                self.linked_table_combo.setCurrentIndex(0)

        self._refresh_color_source_controls(
            preferred_kind_entry=previous_kind_entry,
            preferred_source_identity=previous_source_identity,
        )

    def _refresh_color_source_controls(
        self,
        _index: int | None = None,
        *,
        preferred_kind_entry: ColorSourceKindEntry | None = None,
        preferred_source_identity: tuple[object, ...] | None = None,
    ) -> None:
        current_kind_entry = preferred_kind_entry if preferred_kind_entry is not None else self.selected_kind_entry
        current_source_kind = None if current_kind_entry is None else current_kind_entry.source_kind
        if preferred_source_identity is not None:
            selected_source_identity = preferred_source_identity
        elif self._active_source_kind == current_source_kind:
            selected_source_identity = (
                self.selected_color_source.identity if self.selected_color_source is not None else None
            )
        else:
            selected_source_identity = None

        self._rebuild_color_source_kind_combo(preferred_kind_entry=current_kind_entry)
        kind_entry = self.selected_kind_entry
        source_kind = self.selected_source_kind
        linked_table = self._selected_linked_table()

        if source_kind == "shape_column":
            self.color_source_value_label.setText("Shapes column")
            placeholder_text = _SHAPE_COLUMN_PLACEHOLDER
        elif source_kind == "obs_column":
            self.color_source_value_label.setText("Observation")
            placeholder_text = _OBS_SOURCE_PLACEHOLDER
        elif source_kind == "var":
            self.color_source_value_label.setText("Var")
            placeholder_text = _VAR_SOURCE_PLACEHOLDER
        else:
            self.color_source_value_label.setText("Value source")
            placeholder_text = "Select a color source kind first"

        if kind_entry is None:
            self._filtered_color_sources = []
        elif kind_entry.source_kind == "shape_column":
            self._filtered_color_sources = list(self._shape_column_color_sources)
        elif linked_table is None:
            self._filtered_color_sources = []
        else:
            self._filtered_color_sources = list(linked_table.sources_for(kind_entry))

        with QSignalBlocker(self.color_source_value_input):
            if source_kind is None:
                self.color_source_value_input.setEnabled(False)
                self.color_source_value_input.clear()
                self.color_source_value_input.setPlaceholderText(placeholder_text)
            else:
                self.color_source_value_input.setEnabled(bool(self._filtered_color_sources))
                if selected_source_identity is not None:
                    matching_source = next(
                        (
                            source
                            for source in self._filtered_color_sources
                            if source.identity == selected_source_identity
                        ),
                        None,
                    )
                    if matching_source is not None:
                        self.color_source_value_input.setText(matching_source.display_name)
                    else:
                        self.color_source_value_input.clear()
                else:
                    self.color_source_value_input.clear()

                self.color_source_value_input.setPlaceholderText(placeholder_text)

            self._color_source_completer_model.setStringList(
                [source.display_name for source in self._filtered_color_sources]
            )

        self.color_source_value_input.set_completion_popup_on_entry_enabled(source_kind is not None)
        self._active_source_kind = source_kind
        self._update_action_status()

    def _selected_linked_table(self) -> LinkedTableColorSources | None:
        table_name = self.selected_table_name
        return None if table_name is None else self._linked_tables[table_name]

    def _kind_entries(self) -> list[ColorSourceKindEntry]:
        kind_entries: list[ColorSourceKindEntry] = []
        if self._shape_column_color_sources:
            kind_entries.append(_SHAPE_COLUMN_KIND_ENTRY)

        linked_table = self._selected_linked_table()
        if linked_table is not None:
            kind_entries.extend(linked_table.kind_entries())
        return kind_entries

    def _rebuild_color_source_kind_combo(
        self,
        *,
        preferred_kind_entry: ColorSourceKindEntry | None = None,
    ) -> None:
        kind_entries = self._kind_entries()

        with QSignalBlocker(self.color_source_kind_combo):
            self.color_source_kind_combo.clear()
            self.color_source_kind_combo.addItem(_NO_COLOR_SOURCE_LABEL, None)
            next_index = 0
            for kind_entry in kind_entries:
                if kind_entry == preferred_kind_entry:
                    next_index = self.color_source_kind_combo.count()
                self.color_source_kind_combo.addItem(kind_entry.label, kind_entry)

            self.color_source_kind_combo.setCurrentIndex(next_index)
            self.color_source_kind_combo.setEnabled(bool(kind_entries))

    def _sync_current_source_selection(self) -> None:
        if not self.color_source_value_input.isEnabled():
            return

        current_text = self.color_source_value_input.text().strip()
        for source in self._filtered_color_sources:
            if source.display_name == current_text:
                self.color_source_value_input.setText(source.display_name)
                break
        self._update_action_status()

    def _update_action_status(self) -> None:
        source_kind = self.selected_source_kind
        table_name = self.selected_table_name
        selected_source = self.selected_color_source
        self._update_fill_toggle_enabled(selected_source is not None)

        if source_kind is None:
            self.action_hint_label.setText("Action: add/update primary shapes layer")
            return

        if source_kind in {"obs_column", "var"} and table_name is None:
            self.action_hint_label.setText("Action: table-backed shapes coloring requires a linked table")
            return

        if selected_source is None:
            if source_kind == "shape_column" and self._filtered_color_sources:
                self.action_hint_label.setText("Action: select a shapes column for a styled shapes layer")
            elif source_kind == "shape_column":
                self.action_hint_label.setText("Action: no colorable shapes columns available")
            elif source_kind == "obs_column" and self._filtered_color_sources:
                self.action_hint_label.setText("Action: select an observation column for a styled shapes layer")
            elif source_kind == "obs_column":
                self.action_hint_label.setText("Action: no colorable observation columns available")
            elif self._filtered_color_sources:
                self.action_hint_label.setText("Action: select a var for a styled shapes layer")
            else:
                self.action_hint_label.setText("Action: no vars available for a styled shapes layer")
            return

        self.action_hint_label.setText(f'Action: add/update styled shapes layer "{selected_source.source_label}"')

    def _update_fill_toggle_enabled(self, enabled: bool) -> None:
        self.fill_toggle.setEnabled(enabled)
        if not enabled:
            self.fill_toggle.setChecked(False)

    def _emit_add_update_request(self, _checked: bool = False) -> None:
        self.add_update_requested.emit(
            ShapesLoadRequest(
                shapes_name=self.shapes_name,
                table_name=self.selected_table_name,
                selected_source_kind=self.selected_source_kind,
                selected_color_source=self.selected_color_source,
                fill_shapes=self.fill_shapes,
            )
        )
