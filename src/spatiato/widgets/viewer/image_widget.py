from __future__ import annotations

from collections.abc import Sequence
from functools import partial

from qtpy.QtCore import QSignalBlocker, QStringListModel, Qt, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QCompleter,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from spatiato.viewer.adapter import ImageLayerBinding
from spatiato.viewer.image_styling import DEFAULT_OVERLAY_COLORS, ImageDisplayMode
from spatiato.widgets.image_layer_row import (
    _binding_channel_index,
    _binding_channel_name,
    _ImageLayerRow,
    _normalized_color_or_none,
    _solid_color_from_layer,
)
from spatiato.widgets.shared_styles import (
    ACTION_BUTTON_STYLESHEET,
    CHECKBOX_STYLESHEET,
    COMPLETER_POPUP_STYLESHEET,
    WIDGET_WARNING_TEXT_COLOR,
    CompleterPopupLineEdit,
    _ElidedLabel,
    build_input_control_stylesheet,
    format_tooltip,
)
from spatiato.widgets.viewer.styles import (
    CARD_TITLE_STYLESHEET,
    DETAIL_PANEL_STYLESHEET,
    EMPTY_STATE_STYLESHEET,
    SUBSECTION_LABEL_STYLESHEET,
    TRANSPARENT_PANEL_STYLESHEET,
)

_CHANNEL_WARNING_STYLESHEET = f"color: {WIDGET_WARNING_TEXT_COLOR}; font-weight: 600;"
_MAX_VISIBLE_OVERLAY_CHANNELS = 10


class _ImageCardWidget(QFrame):
    """Render one image card and emit image-layer user intent.

    The card owns available-channel search, selected-row presentation, and
    card-local color preferences. It does not call ``ViewerAdapter`` or mutate
    napari layers. Instead, its signals carry stack and focused overlay intent
    to ``ViewerWidget``:

    - ``stack_load_requested`` carries the image name when the user invokes the
      explicit ``Load in viewer`` action while no Stack is loaded;
    - ``overlay_channel_add_requested`` carries the image name, channel index,
      and initial color immediately after the composer accepts one channel.
      Overlay loading has no aggregate apply action;
    - ``overlay_channel_remove_requested`` carries image name and channel
      index;
    - ``overlay_channels_remove_all_requested`` carries image name;
    - ``overlay_channel_visibility_requested`` carries image name, channel
      index, and requested visibility;
    - ``overlay_channel_color_requested`` carries image name, channel index,
      and requested solid color.
    - ``stack_remove_requested``, ``stack_visibility_requested``, and
      ``stack_color_requested`` carry focused intent for the one reconciled
      Stack row.

    When connecting a live row's intent signals, the card uses ``partial`` to
    capture that row's construction-time binding as ``expected_binding``. Each
    intent handler then locates the card's current row and requires
    ``row.binding is expected_binding`` before forwarding the request to
    ``ViewerWidget``. This rejects delayed signals from a disposed or replaced
    row before they can target a newly loaded layer.

    ``ViewerWidget`` validates the active context and performs each mutation.
    It also resolves current live napari membership after the card-local
    identity check. Visibility and color requests then return through the live
    layer's native property event directly to the selected row. Completed
    membership changes instead return through
    ``ViewerAdapter.image_layers_changed``; the Viewer re-queries complete live
    image membership and calls ``set_loaded_image_bindings`` to reconcile this
    card atomically.
    """

    stack_load_requested = Signal(str)
    overlay_channel_add_requested = Signal(str, int, str)
    overlay_channel_remove_requested = Signal(str, int)
    overlay_channels_remove_all_requested = Signal(str)
    overlay_channel_visibility_requested = Signal(str, int, bool)
    overlay_channel_color_requested = Signal(str, int, str)
    stack_remove_requested = Signal(str)
    stack_visibility_requested = Signal(str, bool)
    stack_color_requested = Signal(str, str)

    def __init__(
        self,
        *,
        image_name: str,
        channel_names: list[str],
        channel_error: str | None = None,
    ) -> None:
        super().__init__()
        self.image_name = image_name
        self.channel_names = channel_names
        self.channel_error = channel_error
        self._overlay_rows_by_channel_index: dict[int, _ImageLayerRow] = {}
        self._overlay_channel_order: list[int] = []
        self._last_used_overlay_colors: dict[int, str] = {}
        self._stack_row: _ImageLayerRow | None = None
        self._membership_initialized = False
        self._membership_error: str | None = None
        self._channel_names_by_casefold: dict[str, list[tuple[int, str]]] = {}
        for channel_index, channel_name in enumerate(channel_names):
            self._channel_names_by_casefold.setdefault(channel_name.casefold(), []).append(
                (channel_index, channel_name)
            )

        self.setObjectName(f"viewer_widget_image_card_{image_name}")
        self.setProperty("spatiatoViewerDetailPanel", True)
        self.setStyleSheet(DETAIL_PANEL_STYLESHEET)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self.title_label = _ElidedLabel(image_name, self)
        self.title_label.setObjectName(f"viewer_widget_image_card_title_{image_name}")
        self.title_label.setStyleSheet(CARD_TITLE_STYLESHEET)
        self.title_label.hide()

        mode_layout = QHBoxLayout()
        mode_layout.setContentsMargins(0, 0, 0, 0)
        mode_layout.setSpacing(16)

        self.overlay_toggle = QCheckBox("overlay")
        self.overlay_toggle.setObjectName(f"viewer_widget_overlay_toggle_{image_name}")
        self.overlay_toggle.setStyleSheet(CHECKBOX_STYLESHEET)
        self.overlay_toggle.setChecked(True)

        self.stack_toggle = QCheckBox("stack")
        self.stack_toggle.setObjectName(f"viewer_widget_stack_toggle_{image_name}")
        self.stack_toggle.setStyleSheet(CHECKBOX_STYLESHEET)

        mode_layout.addWidget(self.overlay_toggle)
        mode_layout.addWidget(self.stack_toggle)
        mode_layout.addStretch(1)

        self.channel_warning_label = QLabel()
        self.channel_warning_label.setObjectName(f"viewer_widget_channel_warning_{image_name}")
        self.channel_warning_label.setWordWrap(True)
        self.channel_warning_label.setStyleSheet(_CHANNEL_WARNING_STYLESHEET)
        self.channel_warning_label.hide()

        self.channel_panel = QWidget()
        self.channel_panel.setObjectName(f"viewer_widget_channel_panel_{image_name}")
        self.channel_panel.setStyleSheet(TRANSPARENT_PANEL_STYLESHEET)
        self.channel_panel.setVisible(False)
        channel_layout = QVBoxLayout(self.channel_panel)
        channel_layout.setContentsMargins(24, 10, 0, 0)
        channel_layout.setSpacing(8)

        self.channel_section_label = QLabel("Channels")
        self.channel_section_label.setObjectName(f"viewer_widget_channel_section_label_{image_name}")
        self.channel_section_label.setStyleSheet(SUBSECTION_LABEL_STYLESHEET)
        channel_layout.addWidget(self.channel_section_label)

        self.channel_search_input = CompleterPopupLineEdit()
        self.channel_search_input.setObjectName(f"viewer_widget_channel_search_input_{image_name}")
        self.channel_search_input.setPlaceholderText("Search and add channels")
        self.channel_search_input.setStyleSheet(build_input_control_stylesheet("QLineEdit"))
        self.channel_search_input.setMinimumWidth(0)
        self.channel_search_input.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.channel_search_input.set_completion_popup_on_entry_enabled(True)

        self._channel_completer_model = QStringListModel(self.channel_search_input)
        self._channel_completer = QCompleter(self._channel_completer_model, self.channel_search_input)
        self._channel_completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        self._channel_completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self._channel_completer.setFilterMode(Qt.MatchFlag.MatchContains)
        self._channel_completer.setMaxVisibleItems(10)
        self._channel_completer.popup().setStyleSheet(COMPLETER_POPUP_STYLESHEET)
        self.channel_search_input.setCompleter(self._channel_completer)
        channel_layout.addWidget(self.channel_search_input)

        selected_summary_layout = QHBoxLayout()
        selected_summary_layout.setContentsMargins(0, 0, 0, 0)
        selected_summary_layout.setSpacing(8)

        self.selected_count_label = QLabel("0 channels")
        self.selected_count_label.setObjectName(f"viewer_widget_selected_channel_count_{image_name}")
        self.selected_count_label.setStyleSheet(SUBSECTION_LABEL_STYLESHEET)

        self.remove_all_channels_button = QPushButton("Remove all")
        self.remove_all_channels_button.setObjectName(f"viewer_widget_remove_all_channels_button_{image_name}")
        self.remove_all_channels_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.remove_all_channels_button.setToolTip(format_tooltip("Remove all overlay channels for this image"))
        self.remove_all_channels_button.setVisible(False)

        selected_summary_layout.addWidget(self.selected_count_label)
        selected_summary_layout.addStretch(1)
        selected_summary_layout.addWidget(self.remove_all_channels_button)
        channel_layout.addLayout(selected_summary_layout)

        self.no_selected_channels_label = QLabel("No channels in viewer")
        self.no_selected_channels_label.setObjectName(f"viewer_widget_no_selected_channels_label_{image_name}")
        self.no_selected_channels_label.setStyleSheet(EMPTY_STATE_STYLESHEET)
        channel_layout.addWidget(self.no_selected_channels_label)

        self.channel_scroll_area = QScrollArea()
        self.channel_scroll_area.setObjectName(f"viewer_widget_channel_scroll_area_{image_name}")
        self.channel_scroll_area.setWidgetResizable(True)
        self.channel_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.channel_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.channel_scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.channel_scroll_area.setStyleSheet("QScrollArea { border: 0px; background: transparent; }")
        self.channel_scroll_area.hide()

        self.channel_list_widget = QWidget()
        self.channel_list_widget.setObjectName(f"viewer_widget_channel_list_{image_name}")
        self.channel_list_widget.setStyleSheet(TRANSPARENT_PANEL_STYLESHEET)
        self.channel_list_layout = QVBoxLayout(self.channel_list_widget)
        self.channel_list_layout.setContentsMargins(0, 0, 0, 0)
        self.channel_list_layout.setSpacing(6)
        self.channel_scroll_area.setWidget(self.channel_list_widget)
        channel_layout.addWidget(self.channel_scroll_area)

        self.stack_row_container = QWidget()
        self.stack_row_container.setObjectName(f"viewer_widget_stack_row_container_{image_name}")
        self.stack_row_container.setStyleSheet(TRANSPARENT_PANEL_STYLESHEET)
        self.stack_row_layout = QVBoxLayout(self.stack_row_container)
        self.stack_row_layout.setContentsMargins(24, 10, 0, 0)
        self.stack_row_layout.setSpacing(0)
        self.stack_row_container.hide()

        self.stack_load_button = QPushButton("Load in viewer")
        self.stack_load_button.setObjectName(f"viewer_widget_load_image_stack_button_{image_name}")
        self.stack_load_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.stack_load_button.setMinimumHeight(28)
        self.stack_load_button.setStyleSheet(ACTION_BUTTON_STYLESHEET)
        self.stack_load_button.setToolTip("")

        self.stack_load_button.clicked.connect(self._emit_stack_load_request)
        self.overlay_toggle.toggled.connect(self._on_overlay_toggled)
        self.stack_toggle.toggled.connect(self._on_stack_toggled)
        self._channel_completer.activated[str].connect(self._request_channel_from_text)
        self.channel_search_input.returnPressed.connect(self._request_channel_from_input)
        self.remove_all_channels_button.clicked.connect(self._emit_remove_all_requested)

        layout.addLayout(mode_layout)
        layout.addWidget(self.channel_warning_label)
        layout.addWidget(self.channel_panel)
        layout.addWidget(self.stack_row_container)
        layout.addWidget(self.stack_load_button)

        self._refresh_overlay_availability()
        self._refresh_search_model()
        self._refresh_membership_presentation()

    @property
    def overlay_rows(self) -> list[_ImageLayerRow]:
        """Return live Overlay rows in their rendered napari order."""
        return [self._overlay_rows_by_channel_index[channel_index] for channel_index in self._overlay_channel_order]

    @property
    def stack_row(self) -> _ImageLayerRow | None:
        return self._stack_row

    @property
    def loaded_overlay_channel_indices(self) -> tuple[int, ...]:
        return tuple(self._overlay_channel_order)

    @property
    def loaded_overlay_channel_names(self) -> tuple[str, ...]:
        return tuple(_binding_channel_name(row.binding) for row in self.overlay_rows)

    @property
    def loaded_stack_binding(self) -> ImageLayerBinding | None:
        return self._stack_row.binding if self._stack_row is not None else None

    @property
    def available_channel_names(self) -> tuple[str, ...]:
        return tuple(self._channel_completer_model.stringList())

    def set_loaded_image_bindings(
        self,
        *,
        stack_binding: ImageLayerBinding | None,
        overlay_bindings: Sequence[ImageLayerBinding],
    ) -> None:
        """Render one complete Stack/Overlay membership snapshot.

        ``ViewerWidget`` builds the snapshot from image layers currently
        present in napari for the active SpatialData object, image, and
        coordinate system. This method does not query ``ViewerAdapter`` or
        inspect the napari viewer.

        ``stack_binding`` and ``overlay_bindings`` describe the entire current
        membership, not an incremental addition or removal. Before changing
        any rows, this method validates display modes, that every binding is
        for this image, Overlay channel identity, duplicate channels, and
        mutual exclusion between Stack and Overlay membership.

        A valid snapshot updates presentation only: existing rows are retained
        when their binding identity is unchanged, replaced bindings receive
        new rows, and absent bindings have their rows disposed. This method
        never loads, removes, or mutates napari layers.
        """
        ordered_bindings = tuple(overlay_bindings)
        if stack_binding is not None and stack_binding.image_display_mode != "stack":
            raise ValueError("The stack binding must have image display mode `stack`.")
        if stack_binding is not None and stack_binding.element_name != self.image_name:
            raise ValueError(f"The stack binding is not for image `{self.image_name}`.")
        if stack_binding is not None and ordered_bindings:
            raise ValueError(f"Image `{self.image_name}` cannot have both stack and overlay bindings.")

        seen_channel_indices: set[int] = set()
        for binding in ordered_bindings:
            if binding.image_display_mode != "overlay":
                raise ValueError("Every overlay binding must have image display mode `overlay`.")
            if binding.element_name != self.image_name:
                raise ValueError(f"An overlay binding is not for image `{self.image_name}`.")
            channel_index = _binding_channel_index(binding)
            _binding_channel_name(binding)
            if channel_index in seen_channel_indices:
                raise ValueError(
                    f"Found multiple live overlay bindings for image `{self.image_name}` "
                    f"and channel index {channel_index}."
                )
            seen_channel_indices.add(channel_index)

        was_initialized = self._membership_initialized
        had_stack = self._stack_row is not None
        had_overlays = bool(self._overlay_channel_order)

        self.set_image_membership_error(None)
        current_stack_row = self._stack_row
        if current_stack_row is not None and current_stack_row.binding is not stack_binding:
            self._stack_row = None
            current_stack_row.dispose()
            self.stack_row_layout.removeWidget(current_stack_row)
            current_stack_row.deleteLater()

        if stack_binding is not None:
            if self._stack_row is None:
                is_rgb = stack_binding.layer.rgb
                display_label = "RGB stack" if is_rgb else "Stack"
                row = _ImageLayerRow(
                    stack_binding,
                    display_label=display_label,
                    accessibility_label=f"{display_label} for image {self.image_name}",
                    show_colormap=not is_rgb,
                    parent=self.stack_row_container,
                )
                row.remove_requested.connect(partial(self._emit_stack_remove_requested, stack_binding))
                row.visibility_change_requested.connect(partial(self._emit_stack_visibility_requested, stack_binding))
                row.color_change_requested.connect(partial(self._emit_stack_color_requested, stack_binding))
                self._stack_row = row
                self.stack_row_layout.addWidget(row)
            else:
                self._stack_row.refresh_presentation()

        next_channel_indices = [_binding_channel_index(binding) for binding in ordered_bindings]
        next_bindings_by_channel_index = {_binding_channel_index(binding): binding for binding in ordered_bindings}

        for channel_index in tuple(self._overlay_channel_order):
            row = self._overlay_rows_by_channel_index[channel_index]
            next_binding = next_bindings_by_channel_index.get(channel_index)
            if next_binding is row.binding:
                continue
            self._overlay_rows_by_channel_index.pop(channel_index)
            if next_binding is None:
                self._remember_row_solid_color(row)
            row.dispose()
            self.channel_list_layout.removeWidget(row)
            row.deleteLater()

        for binding in ordered_bindings:
            channel_index = _binding_channel_index(binding)
            row = self._overlay_rows_by_channel_index.get(channel_index)
            if row is None:
                channel_name = _binding_channel_name(binding)
                row = _ImageLayerRow(
                    binding,
                    display_label=channel_name,
                    accessibility_label=f"channel {channel_name}",
                    parent=self.channel_list_widget,
                )
                row.remove_requested.connect(
                    partial(
                        self._emit_overlay_channel_remove_requested,
                        binding,
                    )
                )
                row.visibility_change_requested.connect(
                    partial(
                        self._emit_overlay_channel_visibility_requested,
                        binding,
                    )
                )
                row.color_change_requested.connect(
                    partial(
                        self._emit_overlay_channel_color_requested,
                        binding,
                    )
                )
                self._overlay_rows_by_channel_index[channel_index] = row
            else:
                row.refresh_presentation()

        for row in self._overlay_rows_by_channel_index.values():
            self.channel_list_layout.removeWidget(row)
        for channel_index in next_channel_indices:
            self.channel_list_layout.addWidget(self._overlay_rows_by_channel_index[channel_index])

        self._overlay_channel_order = next_channel_indices
        self._refresh_search_model()
        self._refresh_membership_presentation()

        has_stack = stack_binding is not None
        has_overlays = bool(next_channel_indices)
        if not was_initialized:
            self._select_display_mode("stack" if has_stack else "overlay")
        elif not had_overlays and has_overlays:
            self._select_display_mode("overlay")
        elif not had_stack and has_stack:
            self._select_display_mode("stack")
        self._membership_initialized = True

    def refresh_overlay_channel_presentation(
        self,
        channel_index: int,
        binding: ImageLayerBinding,
    ) -> None:
        """Refresh one row only when it owns the resolved live binding."""
        row = self._overlay_rows_by_channel_index.get(channel_index)
        if row is None or row.binding is not binding:
            return
        row.refresh_presentation()

    def refresh_stack_presentation(self, binding: ImageLayerBinding) -> None:
        """Refresh the Stack row only when it owns the resolved live binding."""
        row = self._stack_row
        if row is not None and row.binding is binding:
            row.refresh_presentation()

    def dispose(self) -> None:
        """Disconnect all live rows from their napari layers."""
        if self._stack_row is not None:
            self._stack_row.dispose()
        for row in self._overlay_rows_by_channel_index.values():
            row.dispose()

    def set_image_membership_error(self, message: str | None) -> None:
        """Set or clear a card-scoped binding invariant error."""
        self._membership_error = message
        self._refresh_overlay_availability()

    def finish_overlay_channel_add(self, channel_index: int, *, succeeded: bool) -> None:
        """Finish one synchronous add intent without clearing unrelated input."""
        if not succeeded:
            return

        self.channel_search_input.clear_after_accepted_completion(self.channel_names[channel_index])

    def finish_overlay_channel_remove(self, channel_index: int, *, succeeded: bool) -> None:
        """Clear stale input after one channel was successfully removed."""
        if succeeded:
            self._clear_channel_input_if_matches(channel_index)

    def cache_overlay_channel_color(self, channel_index: int) -> None:
        """Cache the current live solid color for one selected row."""
        row = self._overlay_rows_by_channel_index.get(channel_index)
        if row is not None:
            self._remember_row_solid_color(row)

    def cache_all_overlay_channel_colors(self) -> None:
        for row in self.overlay_rows:
            self._remember_row_solid_color(row)

    def _refresh_overlay_availability(self) -> None:
        if self.channel_error is not None:
            warning = (
                "Overlay is unavailable because this image has duplicate channel names. "
                'Use "sdata.set_channel_names(...)" to rename them.'
            )
        elif not self.channel_names:
            warning = "No channel axis available for this image."
        else:
            warning = self._membership_error

        is_available = warning is None
        self.stack_load_button.setEnabled(self._membership_error is None)
        self.overlay_toggle.setEnabled(self.channel_error is None and bool(self.channel_names))
        self.channel_search_input.setEnabled(is_available)
        self.remove_all_channels_button.setEnabled(is_available and bool(self._overlay_channel_order))
        if self._stack_row is not None:
            self._stack_row.setEnabled(self._membership_error is None)
        for row in self._overlay_rows_by_channel_index.values():
            row.setEnabled(is_available)

        self.channel_warning_label.setText(warning or "")
        self.channel_warning_label.setToolTip(format_tooltip(warning) if warning else "")
        self.channel_warning_label.setVisible(warning is not None)

    def _select_display_mode(self, mode: ImageDisplayMode) -> None:
        toggle = self.overlay_toggle if mode == "overlay" else self.stack_toggle
        if not toggle.isChecked():
            toggle.setChecked(True)

    def _on_overlay_toggled(self, checked: bool) -> None:
        if checked:
            with QSignalBlocker(self.stack_toggle):
                self.stack_toggle.setChecked(False)
            self._refresh_mode_presentation()
            return

        if not self.stack_toggle.isChecked():
            with QSignalBlocker(self.overlay_toggle):
                self.overlay_toggle.setChecked(True)
        self._refresh_mode_presentation()

    def _on_stack_toggled(self, checked: bool) -> None:
        if checked:
            with QSignalBlocker(self.overlay_toggle):
                self.overlay_toggle.setChecked(False)
            self._refresh_mode_presentation()
            return

        if not self.overlay_toggle.isChecked():
            with QSignalBlocker(self.stack_toggle):
                self.stack_toggle.setChecked(True)
        self._refresh_mode_presentation()

    def _refresh_mode_presentation(self) -> None:
        """Render editor content from selected mode and live Stack membership.

        Selected mode | Stack loaded | Visible content
        Overlay       | either       | Overlay channel panel
        Stack         | no           | Load in viewer button
        Stack         | yes          | Live Stack row

        A Stack layer can be added or removed through napari while the selected
        mode remains unchanged. Refresh this presentation after membership
        updates as well as after mode-toggle events, so Stack switches
        correctly between its pending button and live row.
        """
        stack_selected = self.stack_toggle.isChecked()
        self.channel_panel.setVisible(not stack_selected)
        self.stack_row_container.setVisible(stack_selected and self._stack_row is not None)
        self.stack_load_button.setVisible(stack_selected and self._stack_row is None)

    def display_mode(self) -> str:
        return "overlay" if self.overlay_toggle.isChecked() else "stack"

    def _emit_stack_load_request(self, _checked: bool = False) -> None:
        if self.display_mode() != "stack" or self._stack_row is not None:
            return
        self.stack_load_requested.emit(self.image_name)

    def _request_channel_from_input(self) -> None:
        self._request_channel_from_text(self.channel_search_input.text())

    def _request_channel_from_text(self, text: str) -> None:
        if not self.channel_search_input.isEnabled():
            return
        resolved = self._resolve_channel_text(text)
        if resolved is None:
            return
        channel_index, _channel_name = resolved
        if channel_index in self._overlay_rows_by_channel_index:
            return
        self.overlay_channel_add_requested.emit(
            self.image_name,
            channel_index,
            self._initial_color_for_channel(channel_index),
        )

    def _resolve_channel_text(self, text: str) -> tuple[int, str] | None:
        normalized_text = text.strip()
        if not normalized_text:
            return None

        for channel_index, channel_name in enumerate(self.channel_names):
            if channel_name == normalized_text:
                return channel_index, channel_name

        matches = self._channel_names_by_casefold.get(normalized_text.casefold(), [])
        if len(matches) == 1:
            return matches[0]
        return None

    def _clear_channel_input_if_matches(self, channel_index: int) -> None:
        resolved = self._resolve_channel_text(self.channel_search_input.text())
        if resolved is not None and resolved[0] == channel_index:
            self.channel_search_input.clear()

    def _emit_overlay_channel_remove_requested(
        self,
        expected_binding: ImageLayerBinding,
    ) -> None:
        channel_index = _binding_channel_index(expected_binding)
        row = self._overlay_rows_by_channel_index.get(channel_index)
        if row is None or row.binding is not expected_binding:
            return
        self.overlay_channel_remove_requested.emit(self.image_name, channel_index)

    def _emit_overlay_channel_visibility_requested(
        self,
        expected_binding: ImageLayerBinding,
        visible: bool,
    ) -> None:
        channel_index = _binding_channel_index(expected_binding)
        row = self._overlay_rows_by_channel_index.get(channel_index)
        if row is None or row.binding is not expected_binding:
            return
        self.overlay_channel_visibility_requested.emit(
            self.image_name,
            channel_index,
            visible,
        )

    def _emit_overlay_channel_color_requested(
        self,
        expected_binding: ImageLayerBinding,
        color: str,
    ) -> None:
        channel_index = _binding_channel_index(expected_binding)
        row = self._overlay_rows_by_channel_index.get(channel_index)
        if row is None or row.binding is not expected_binding:
            return
        self.overlay_channel_color_requested.emit(
            self.image_name,
            channel_index,
            color,
        )

    def _emit_stack_remove_requested(self, expected_binding: ImageLayerBinding) -> None:
        if self._stack_row is None or self._stack_row.binding is not expected_binding:
            return
        self.stack_remove_requested.emit(self.image_name)

    def _emit_stack_visibility_requested(
        self,
        expected_binding: ImageLayerBinding,
        visible: bool,
    ) -> None:
        if self._stack_row is None or self._stack_row.binding is not expected_binding:
            return
        self.stack_visibility_requested.emit(self.image_name, visible)

    def _emit_stack_color_requested(
        self,
        expected_binding: ImageLayerBinding,
        color: str,
    ) -> None:
        if self._stack_row is None or self._stack_row.binding is not expected_binding:
            return
        self.stack_color_requested.emit(self.image_name, color)

    def _emit_remove_all_requested(self, _checked: bool = False) -> None:
        if not self._overlay_channel_order:
            return
        self.overlay_channels_remove_all_requested.emit(self.image_name)

    def _refresh_search_model(self) -> None:
        selected_channel_indices = set(self._overlay_channel_order)
        available_names = [
            channel_name
            for channel_index, channel_name in enumerate(self.channel_names)
            if channel_index not in selected_channel_indices
        ]
        self._channel_completer_model.setStringList(available_names)

    def _refresh_membership_presentation(self) -> None:
        selected_count = len(self._overlay_channel_order)
        count_text = f"{selected_count} channel" if selected_count == 1 else f"{selected_count} channels"
        self.selected_count_label.setText(count_text)
        self.no_selected_channels_label.setVisible(selected_count == 0)
        self.channel_scroll_area.setVisible(selected_count > 0)
        self.remove_all_channels_button.setVisible(selected_count > 0)
        self._set_channel_scroll_height(self.overlay_rows)
        self._refresh_overlay_availability()
        self._refresh_mode_presentation()

    def _set_channel_scroll_height(self, channel_rows: list[QWidget]) -> None:
        visible_rows = channel_rows[:_MAX_VISIBLE_OVERLAY_CHANNELS]
        if not visible_rows:
            self.channel_list_widget.setMinimumHeight(0)
            self.channel_scroll_area.setFixedHeight(0)
            return

        spacing = self.channel_list_layout.spacing()
        margins = self.channel_list_layout.contentsMargins()
        vertical_margins = margins.top() + margins.bottom()
        content_height = sum(row.sizeHint().height() for row in channel_rows)
        content_height += spacing * max(0, len(channel_rows) - 1)
        content_height += vertical_margins
        self.channel_list_widget.setMinimumHeight(content_height)

        visible_height = sum(row.sizeHint().height() for row in visible_rows)
        visible_height += spacing * max(0, len(visible_rows) - 1)
        visible_height += vertical_margins
        visible_height += self.channel_scroll_area.frameWidth() * 2
        # A maximum alone allows the parent layout to compress the viewport and
        # introduce a scrollbar before the configured row cap is reached.
        self.channel_scroll_area.setFixedHeight(visible_height)

    def _initial_color_for_channel(self, channel_index: int) -> str:
        cached_color = self._last_used_overlay_colors.get(channel_index)
        if cached_color is not None:
            return cached_color

        used_colors = {
            color for row in self.overlay_rows if (color := _solid_color_from_layer(row.binding.layer)) is not None
        }
        for color in DEFAULT_OVERLAY_COLORS:
            normalized_color = _normalized_color_or_none(color)
            if normalized_color is not None and normalized_color not in used_colors:
                return normalized_color

        return DEFAULT_OVERLAY_COLORS[len(used_colors) % len(DEFAULT_OVERLAY_COLORS)]

    def _remember_row_solid_color(self, row: _ImageLayerRow) -> None:
        color = _solid_color_from_layer(row.binding.layer)
        if color is not None:
            self._last_used_overlay_colors[_binding_channel_index(row.binding)] = color
