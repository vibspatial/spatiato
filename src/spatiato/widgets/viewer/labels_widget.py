from __future__ import annotations

from collections.abc import Sequence
from functools import partial

from qtpy.QtCore import QSignalBlocker, QStringListModel, Qt, Signal
from qtpy.QtWidgets import (
    QCompleter,
    QFormLayout,
    QFrame,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from spatiato.core._color_source import TableColorSourceKind, TableColorSourceSpec
from spatiato.viewer.adapter import LabelsLayerBinding
from spatiato.widgets.layer_row import _LayerRow
from spatiato.widgets.shared_styles import (
    ACTION_BUTTON_STYLESHEET,
    COMPLETER_POPUP_STYLESHEET,
    WIDGET_WARNING_TEXT_COLOR,
    CompactComboBox,
    CompleterPopupLineEdit,
    _ElidedLabel,
    build_input_control_stylesheet,
    create_form_label,
    format_tooltip,
)
from spatiato.widgets.viewer._table_color_sources import ColorSourceKindEntry, LinkedTableColorSources
from spatiato.widgets.viewer.styled_layer_list import _StyledLayerList
from spatiato.widgets.viewer.styles import (
    CARD_TITLE_STYLESHEET,
    DETAIL_PANEL_STYLESHEET,
    INPUT_CONTROL_STYLESHEET,
    TRANSPARENT_PANEL_STYLESHEET,
)

_OBS_SOURCE_PLACEHOLDER = "Search and add obs columns"
_VAR_SOURCE_PLACEHOLDER = "Search and add vars"
_NO_COLOR_SOURCES_LABEL = "No color sources"
_MEMBERSHIP_WARNING_STYLESHEET = f"color: {WIDGET_WARNING_TEXT_COLOR}; font-weight: 600;"


class _LabelsCardWidget(QFrame):
    """Render one labels card and emit labels-layer user intent.

    The card shows the primary layer of the labels element, the controls to
    pick a color source of a linked table, and the list of styled layers of
    the element that are loaded in the viewer. It does not call
    ``ViewerAdapter`` or mutate napari layers. Its signals carry intent to
    ``ViewerWidget``:

    - ``primary_load_requested`` carries the labels name when the user clicks
      "Load in viewer" while no primary layer is loaded;
    - ``primary_remove_requested`` and ``primary_visibility_requested`` carry
      intent for the primary row;
    - ``styled_source_add_requested`` carries the labels name and the
      `TableColorSourceSpec` as soon as the user accepts a value in the search
      field. There is no apply button;
    - ``styled_source_remove_requested``, ``styled_sources_remove_all_requested``
      and ``styled_source_visibility_requested`` carry intent for the rows of
      the loaded list.

    The "Linked table" and "Color source" combos only choose which sources the
    search field offers. Sources that are already loaded are left out.

    Rows capture their construction-time binding, and the card, or its loaded
    list, forwards a row's intent only while that row still holds the binding.
    This rejects delayed signals from a disposed or replaced row.

    ``ViewerWidget`` validates the active context and performs each mutation.
    Visibility changes return through the layer's native ``visible`` event to
    the row. Membership changes return through
    ``ViewerAdapter.labels_layers_changed``: the Viewer re-queries the live
    labels layers of the element and calls `set_loaded_labels_bindings(...)`.
    """

    primary_load_requested = Signal(str)
    primary_remove_requested = Signal(str)
    primary_visibility_requested = Signal(str, bool)
    styled_source_add_requested = Signal(str, object)
    styled_source_remove_requested = Signal(str, object)
    styled_sources_remove_all_requested = Signal(str)
    styled_source_visibility_requested = Signal(str, object, bool)

    def __init__(
        self,
        *,
        labels_name: str,
        linked_tables: dict[str, LinkedTableColorSources],
    ) -> None:
        super().__init__()
        self.labels_name = labels_name
        self._linked_tables = linked_tables
        # The sources of the selected "Color source" entry, and those of them
        # that are not loaded, which the search field offers.
        self._kind_sources: list[TableColorSourceSpec] = []
        self._available_sources: list[TableColorSourceSpec] = []
        self._active_source_kind: TableColorSourceKind | None = None
        self._primary_row: _LayerRow | None = None
        self._membership_error: str | None = None
        self.setObjectName(f"viewer_widget_labels_card_{labels_name}")
        self.setProperty("spatiatoViewerDetailPanel", True)
        self.setStyleSheet(DETAIL_PANEL_STYLESHEET)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self.title_label = _ElidedLabel(labels_name, self)
        self.title_label.setObjectName(f"viewer_widget_labels_card_title_{labels_name}")
        self.title_label.setStyleSheet(CARD_TITLE_STYLESHEET)
        self.title_label.hide()

        self.membership_warning_label = QLabel()
        self.membership_warning_label.setObjectName(f"viewer_widget_labels_membership_warning_{labels_name}")
        self.membership_warning_label.setWordWrap(True)
        self.membership_warning_label.setStyleSheet(_MEMBERSHIP_WARNING_STYLESHEET)
        self.membership_warning_label.hide()

        form_layout = QFormLayout()
        form_layout.setContentsMargins(0, 0, 0, 0)
        form_layout.setHorizontalSpacing(8)
        form_layout.setVerticalSpacing(6)

        primary_label = create_form_label("Primary layer")
        self.primary_container = QWidget()
        self.primary_container.setObjectName(f"viewer_widget_labels_primary_container_{labels_name}")
        self.primary_container.setStyleSheet(TRANSPARENT_PANEL_STYLESHEET)
        self.primary_layout = QVBoxLayout(self.primary_container)
        self.primary_layout.setContentsMargins(0, 0, 0, 0)
        self.primary_layout.setSpacing(0)

        self.primary_load_button = QPushButton("Load in viewer")
        self.primary_load_button.setObjectName(f"viewer_widget_load_labels_button_{labels_name}")
        self.primary_load_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.primary_load_button.setMinimumHeight(28)
        self.primary_load_button.setStyleSheet(ACTION_BUTTON_STYLESHEET)
        self.primary_load_button.clicked.connect(self._emit_primary_load_request)
        self.primary_layout.addWidget(self.primary_load_button)

        linked_table_label = create_form_label("Linked table")
        self.linked_table_combo = CompactComboBox(minimum_contents_length=8)
        self.linked_table_combo.setObjectName(f"viewer_widget_linked_table_combo_{labels_name}")
        self.linked_table_combo.setStyleSheet(INPUT_CONTROL_STYLESHEET)
        table_names = list(linked_tables)
        if table_names:
            self.linked_table_combo.addItems(table_names)
        else:
            self.linked_table_combo.addItem("No linked tables")
            self.linked_table_combo.setEnabled(False)

        color_source_kind_label = create_form_label("Color source")
        self.color_source_kind_combo = CompactComboBox(minimum_contents_length=8)
        self.color_source_kind_combo.setObjectName(f"viewer_widget_color_source_kind_combo_{labels_name}")
        self.color_source_kind_combo.setStyleSheet(INPUT_CONTROL_STYLESHEET)

        self.color_source_value_label = create_form_label("Value source")
        self.color_source_value_input = CompleterPopupLineEdit()
        self.color_source_value_input.setObjectName(f"viewer_widget_color_source_value_input_{labels_name}")
        self.color_source_value_input.setStyleSheet(build_input_control_stylesheet("QLineEdit"))
        self.color_source_value_input.setMinimumWidth(0)
        self.color_source_value_input.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.color_source_value_input.setEnabled(False)

        self._color_source_completer_model = QStringListModel(self.color_source_value_input)
        self._color_source_completer = QCompleter(self._color_source_completer_model, self.color_source_value_input)
        self._color_source_completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        self._color_source_completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self._color_source_completer.setFilterMode(Qt.MatchFlag.MatchContains)
        self._color_source_completer.setMaxVisibleItems(10)
        self._color_source_completer.popup().setStyleSheet(COMPLETER_POPUP_STYLESHEET)
        self.color_source_value_input.setCompleter(self._color_source_completer)

        form_layout.addRow(primary_label, self.primary_container)
        form_layout.addRow(linked_table_label, self.linked_table_combo)
        form_layout.addRow(color_source_kind_label, self.color_source_kind_combo)
        form_layout.addRow(self.color_source_value_label, self.color_source_value_input)

        self.styled_layer_list = _StyledLayerList(element_name=labels_name, parent=self)
        self.styled_layer_list.set_show_table_names(len(linked_tables) > 1)

        layout.addWidget(self.membership_warning_label)
        layout.addLayout(form_layout)
        layout.addWidget(self.styled_layer_list)

        self.linked_table_combo.currentIndexChanged.connect(self._refresh_color_source_controls)
        self.color_source_kind_combo.currentIndexChanged.connect(self._refresh_color_source_controls)
        self._color_source_completer.activated[str].connect(self._request_source_from_text)
        self.color_source_value_input.returnPressed.connect(self._request_source_from_input)
        self.styled_layer_list.remove_requested.connect(self._emit_styled_source_remove_requested)
        self.styled_layer_list.remove_all_requested.connect(self._emit_styled_sources_remove_all_requested)
        self.styled_layer_list.visibility_requested.connect(self._emit_styled_source_visibility_requested)

        self._refresh_color_source_controls()
        self._refresh_primary_presentation()

    @property
    def selected_table_name(self) -> str | None:
        if not self.linked_table_combo.isEnabled():
            return None

        table_name = self.linked_table_combo.currentText()
        return table_name if table_name in self._linked_tables else None

    @property
    def selected_kind_entry(self) -> ColorSourceKindEntry | None:
        """Return the selected "Color source" entry, or `None` when the linked table offers none."""
        entry = self.color_source_kind_combo.currentData()
        return entry if isinstance(entry, ColorSourceKindEntry) else None

    @property
    def selected_source_kind(self) -> TableColorSourceKind | None:
        entry = self.selected_kind_entry
        return None if entry is None else entry.source_kind

    @property
    def selected_color_source(self) -> TableColorSourceSpec | None:
        """Return the source the search field's text names, among the sources it offers."""
        return self._resolve_source_text(self.color_source_value_input.text())

    @property
    def available_source_names(self) -> tuple[str, ...]:
        """Return the names the search field offers: the sources of the selected entry that are not loaded."""
        return tuple(self._color_source_completer_model.stringList())

    @property
    def primary_row(self) -> _LayerRow | None:
        return self._primary_row

    @property
    def styled_rows(self) -> list[_LayerRow]:
        """Return the rows of the loaded styled layers, in napari layer order."""
        return self.styled_layer_list.rows

    @property
    def loaded_styled_specs(self) -> tuple[TableColorSourceSpec, ...]:
        """Return the specs of the loaded styled layers, in napari layer order."""
        return tuple(spec for spec in self.styled_layer_list.loaded_specs if isinstance(spec, TableColorSourceSpec))

    def set_linked_tables(self, linked_tables: dict[str, LinkedTableColorSources]) -> None:
        previous_table_name = self.selected_table_name
        previous_kind_entry = self.selected_kind_entry
        previous_source_identity = (
            self.selected_color_source.identity if self.selected_color_source is not None else None
        )

        self._linked_tables = linked_tables
        table_names = list(linked_tables)
        with QSignalBlocker(self.linked_table_combo):
            # Rebuild linked-table choices after the sdata table set changed,
            # then restore the previous selection when it is still valid.
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

        self.styled_layer_list.set_show_table_names(len(table_names) > 1)
        self._refresh_color_source_controls(
            preferred_kind_entry=previous_kind_entry,
            preferred_source_identity=previous_source_identity,
        )

    def set_loaded_labels_bindings(
        self,
        *,
        primary_binding: LabelsLayerBinding | None,
        styled_bindings: Sequence[LabelsLayerBinding],
    ) -> None:
        """Render one complete snapshot of the live labels layers of this element.

        ``ViewerWidget`` builds the snapshot from the labels layers in napari
        for the active SpatialData object, labels element and coordinate
        system. This method does not query ``ViewerAdapter`` or inspect the
        napari viewer.

        The snapshot is validated before any row changes: the primary binding
        must be primary, every styled binding styled with a table source, all
        of them for this labels element, and no two styled bindings may have
        the same source. Rows whose binding object is unchanged are kept, new
        bindings get new rows, and the rows of absent bindings are disposed.
        This method never loads, removes or mutates napari layers.
        """
        ordered_bindings = tuple(styled_bindings)
        if primary_binding is not None:
            if primary_binding.labels_role != "primary":
                raise ValueError("The primary labels binding must have the role `primary`.")
            if primary_binding.element_name != self.labels_name:
                raise ValueError(f"The primary labels binding is not for labels `{self.labels_name}`.")
        for binding in ordered_bindings:
            if binding.labels_role != "styled":
                raise ValueError("Every styled labels binding must have the role `styled`.")
            if binding.element_name != self.labels_name:
                raise ValueError(f"A styled labels binding is not for labels `{self.labels_name}`.")
            if not isinstance(binding.style_spec, TableColorSourceSpec):
                raise ValueError("Every styled labels binding must be styled by a table source.")
        self.styled_layer_list.validate_styled_bindings(ordered_bindings)

        self.set_labels_membership_error(None)
        current_primary_row = self._primary_row
        if current_primary_row is not None and current_primary_row.binding is not primary_binding:
            self._primary_row = None
            current_primary_row.dispose()
            self.primary_layout.removeWidget(current_primary_row)
            current_primary_row.deleteLater()

        if primary_binding is not None:
            if self._primary_row is None:
                row = _LayerRow(
                    primary_binding,
                    display_label=self.labels_name,
                    accessibility_label=f"labels layer {self.labels_name}",
                    parent=self.primary_container,
                )
                row.remove_requested.connect(partial(self._emit_primary_remove_requested, primary_binding))
                row.visibility_change_requested.connect(
                    partial(self._emit_primary_visibility_requested, primary_binding)
                )
                self._primary_row = row
                self.primary_layout.addWidget(row)
            else:
                self._primary_row.refresh_presentation()

        self.styled_layer_list.set_styled_bindings(ordered_bindings)
        self._refresh_available_sources()
        self._refresh_primary_presentation()

    def set_labels_membership_error(self, message: str | None) -> None:
        """Set or clear a card-scoped error about the live labels layers of this element."""
        self._membership_error = message
        self.membership_warning_label.setText(message or "")
        self.membership_warning_label.setToolTip(format_tooltip(message) if message else "")
        self.membership_warning_label.setVisible(message is not None)
        self.primary_container.setEnabled(message is None)
        self.styled_layer_list.setEnabled(message is None)
        self._refresh_value_input_enabled()

    def set_styled_source_warning(self, style_spec: TableColorSourceSpec, message: str | None) -> None:
        """Show `message` on the row of `style_spec`, or clear its warning for `None`."""
        self.styled_layer_list.set_styled_source_warning(style_spec, message)

    def refresh_primary_presentation(self, binding: LabelsLayerBinding) -> None:
        """Refresh the primary row only when it holds `binding`."""
        row = self._primary_row
        if row is not None and row.binding is binding:
            row.refresh_presentation()

    def refresh_styled_source_presentation(self, style_spec: TableColorSourceSpec, binding: LabelsLayerBinding) -> None:
        """Refresh the row of `style_spec` only when it holds `binding`."""
        self.styled_layer_list.refresh_row_presentation(style_spec, binding)

    def finish_styled_source_add(self, style_spec: TableColorSourceSpec, *, succeeded: bool) -> None:
        """Clear the search field after `style_spec` was loaded; keep it when the load failed."""
        if succeeded:
            self.color_source_value_input.clear_after_accepted_completion(style_spec.display_name)

    def dispose(self) -> None:
        """Disconnect every row from its napari layer."""
        if self._primary_row is not None:
            self._primary_row.dispose()
        self.styled_layer_list.dispose()

    def _refresh_color_source_controls(
        self,
        _index: int | None = None,
        *,
        preferred_kind_entry: ColorSourceKindEntry | None = None,
        preferred_source_identity: tuple[str, TableColorSourceKind, str] | None = None,
    ) -> None:
        current_kind_entry = preferred_kind_entry if preferred_kind_entry is not None else self.selected_kind_entry
        current_source_kind = None if current_kind_entry is None else current_kind_entry.source_kind
        if preferred_source_identity is not None:
            selected_source_identity = preferred_source_identity
        elif self._active_source_kind == current_source_kind:
            selected_source = self.selected_color_source
            selected_source_identity = selected_source.identity if selected_source is not None else None
        else:
            selected_source_identity = None

        self._rebuild_color_source_kind_combo(preferred_kind_entry=current_kind_entry)
        kind_entry = self.selected_kind_entry
        source_kind = self.selected_source_kind
        linked_table = self._selected_linked_table()

        if source_kind == "obs_column":
            self.color_source_value_label.setText("Observation")
            placeholder_text = _OBS_SOURCE_PLACEHOLDER
        elif source_kind == "var":
            self.color_source_value_label.setText("Var")
            placeholder_text = _VAR_SOURCE_PLACEHOLDER
        else:
            self.color_source_value_label.setText("Value source")
            placeholder_text = _NO_COLOR_SOURCES_LABEL

        if kind_entry is None or linked_table is None:
            self._kind_sources = []
        else:
            self._kind_sources = linked_table.sources_for(kind_entry)
        self._refresh_available_sources()

        with QSignalBlocker(self.color_source_value_input):
            matching_source = None
            if selected_source_identity is not None:
                for source in self._available_sources:
                    if source.identity == selected_source_identity:
                        matching_source = source
                        break
            if matching_source is not None:
                self.color_source_value_input.setText(matching_source.display_name)
            else:
                self.color_source_value_input.clear()
            self.color_source_value_input.setPlaceholderText(placeholder_text)

        self.color_source_value_input.set_completion_popup_on_entry_enabled(source_kind is not None)
        self._active_source_kind = source_kind
        self._refresh_value_input_enabled()

    def _refresh_available_sources(self) -> None:
        """Offer the sources of the selected entry that are not loaded."""
        loaded_specs = set(self.styled_layer_list.loaded_specs)
        self._available_sources = [source for source in self._kind_sources if source not in loaded_specs]
        self._color_source_completer_model.setStringList([source.display_name for source in self._available_sources])

    def _refresh_value_input_enabled(self) -> None:
        # The field stays enabled when every source of the entry is loaded, as
        # the image card's channel search does: it then offers no completions.
        self.color_source_value_input.setEnabled(
            self._membership_error is None and self.selected_source_kind is not None and bool(self._kind_sources)
        )

    def _refresh_primary_presentation(self) -> None:
        self.primary_load_button.setVisible(self._primary_row is None)

    def _selected_linked_table(self) -> LinkedTableColorSources | None:
        table_name = self.selected_table_name
        return None if table_name is None else self._linked_tables[table_name]

    def _rebuild_color_source_kind_combo(
        self,
        *,
        preferred_kind_entry: ColorSourceKindEntry | None = None,
    ) -> None:
        linked_table = self._selected_linked_table()
        kind_entries = [] if linked_table is None else linked_table.kind_entries()

        with QSignalBlocker(self.color_source_kind_combo):
            self.color_source_kind_combo.clear()
            if not kind_entries:
                self.color_source_kind_combo.addItem(_NO_COLOR_SOURCES_LABEL, None)
                self.color_source_kind_combo.setCurrentIndex(0)
                self.color_source_kind_combo.setEnabled(False)
                return

            next_index = 0
            for kind_entry in kind_entries:
                if kind_entry == preferred_kind_entry:
                    next_index = self.color_source_kind_combo.count()
                self.color_source_kind_combo.addItem(kind_entry.label, kind_entry)

            self.color_source_kind_combo.setCurrentIndex(next_index)
            self.color_source_kind_combo.setEnabled(True)

    def _resolve_source_text(self, text: str) -> TableColorSourceSpec | None:
        """Return the offered source named `text`: an exact match, else the only case-insensitive one."""
        normalized_text = text.strip()
        if not normalized_text:
            return None

        for source in self._available_sources:
            if source.display_name == normalized_text:
                return source

        casefolded_text = normalized_text.casefold()
        matches = [source for source in self._available_sources if source.display_name.casefold() == casefolded_text]
        return matches[0] if len(matches) == 1 else None

    def _request_source_from_input(self) -> None:
        self._request_source_from_text(self.color_source_value_input.text())

    def _request_source_from_text(self, text: str) -> None:
        if not self.color_source_value_input.isEnabled():
            return
        source = self._resolve_source_text(text)
        if source is None:
            return
        self.styled_source_add_requested.emit(self.labels_name, source)

    def _emit_primary_load_request(self, _checked: bool = False) -> None:
        if self._primary_row is not None:
            return
        self.primary_load_requested.emit(self.labels_name)

    def _emit_primary_remove_requested(self, expected_binding: LabelsLayerBinding) -> None:
        if self._primary_row is None or self._primary_row.binding is not expected_binding:
            return
        self.primary_remove_requested.emit(self.labels_name)

    def _emit_primary_visibility_requested(self, expected_binding: LabelsLayerBinding, visible: bool) -> None:
        if self._primary_row is None or self._primary_row.binding is not expected_binding:
            return
        self.primary_visibility_requested.emit(self.labels_name, visible)

    def _emit_styled_source_remove_requested(self, style_spec: TableColorSourceSpec) -> None:
        self.styled_source_remove_requested.emit(self.labels_name, style_spec)

    def _emit_styled_sources_remove_all_requested(self) -> None:
        self.styled_sources_remove_all_requested.emit(self.labels_name)

    def _emit_styled_source_visibility_requested(self, style_spec: TableColorSourceSpec, visible: bool) -> None:
        self.styled_source_visibility_requested.emit(self.labels_name, style_spec, visible)
