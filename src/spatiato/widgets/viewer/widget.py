from __future__ import annotations

import os
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd
from harpy.io import read_zarr
from pandas.api.types import is_bool_dtype, is_numeric_dtype, is_object_dtype, is_string_dtype
from qtpy.QtCore import QSignalBlocker, Qt
from qtpy.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QFrame,
    QLabel,
    QLayout,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from spatiato._app_state import (
    CoordinateSystemChangedEvent,
    ShapesElementReloadedEvent,
    ShapesElementWrittenEvent,
    SpatiatoAppState,
    TableStateChangedEvent,
    get_or_create_app_state,
)
from spatiato.core.color_source_options import get_shape_column_color_source_options
from spatiato.core.spatialdata import (
    get_annotating_table_names,
    get_coordinate_system_names_from_sdata,
    get_image_channel_names_from_sdata,
    get_spatialdata_image_options_for_coordinate_system_from_sdata,
    get_spatialdata_labels_options_for_coordinate_system_from_sdata,
    get_spatialdata_points_options_for_coordinate_system_from_sdata,
    get_spatialdata_shapes_options_for_coordinate_system_from_sdata,
)
from spatiato.viewer.adapter import ImageLayerBinding
from spatiato.viewer.points_styling import PointsLoadResult
from spatiato.widgets.image_layer_row import (
    _normalized_color_or_none,
    _solid_color_from_layer,
)
from spatiato.widgets.shared_styles import (
    ACTION_BUTTON_STYLESHEET,
    WIDGET_MIN_WIDTH,
    CompactComboBox,
    StatusCardKind,
    apply_scroll_content_surface,
    apply_widget_surface,
    create_form_label,
    create_header_logo,
    format_feedback_identifier,
    set_status_card,
)
from spatiato.widgets.spatialdata_replacement_dialog import confirm_spatialdata_replacement
from spatiato.widgets.viewer._table_color_sources import LinkedTableColorSources
from spatiato.widgets.viewer.disclosure import _CollapsibleSectionWidget, _DisclosureElementWidget
from spatiato.widgets.viewer.image_widget import _ImageCardWidget
from spatiato.widgets.viewer.labels_widget import LabelsLoadRequest, _LabelsCardWidget
from spatiato.widgets.viewer.points_controller import PointsController, PointsLoadRequest, PointsValueSource
from spatiato.widgets.viewer.points_widget import PointsValueWidget
from spatiato.widgets.viewer.shapes_widget import ShapesLoadRequest, _ShapesCardWidget
from spatiato.widgets.viewer.status_card import (
    _ViewerStatusCardSpec,
    build_image_loaded_card_spec,
    build_points_layer_card_spec,
    build_primary_labels_loaded_card_spec,
    build_primary_shapes_loaded_card_spec,
    build_styled_labels_card_spec,
    build_styled_shapes_card_spec,
    build_viewer_feedback_card_spec,
)
from spatiato.widgets.viewer.styles import (
    EMPTY_STATE_STYLESHEET,
    INPUT_CONTROL_STYLESHEET,
)
from spatiato.widgets.viewer.tiled_points_controller import TiledPointsController

if TYPE_CHECKING:
    import napari
    from spatialdata import SpatialData


_SUMMARY_COORDINATE_SYSTEM_MAX_LENGTH = 32
_EXPERIMENTAL_TILED_POINTS_ENV = "SPATIATO_EXPERIMENTAL_TILED_POINTS"
_TRUE_ENV_VALUES = frozenset({"1", "on", "true", "yes"})


def _experimental_tiled_points_enabled(explicit: bool | None) -> bool:
    """Resolve the process-start opt-in, allowing an explicit override."""
    if explicit is not None:
        return explicit
    return os.environ.get(_EXPERIMENTAL_TILED_POINTS_ENV, "").strip().casefold() in _TRUE_ENV_VALUES


class ViewerWidget(QWidget):
    """Shared viewer widget backed by `SpatiatoAppState` and `ViewerAdapter`.

    Points use the in-memory backend by default. Pass
    ``experimental_tiled_points=True`` or set
    ``SPATIATO_EXPERIMENTAL_TILED_POINTS=1`` before starting napari to opt
    into the cache-backed tiled renderer for this widget's lifetime.
    """

    def __init__(
        self,
        napari_viewer: napari.Viewer | None = None,
        *,
        experimental_tiled_points: bool | None = None,
    ) -> None:
        super().__init__()
        self.setObjectName("viewer_widget")
        apply_widget_surface(self)
        self.setMinimumWidth(WIDGET_MIN_WIDTH)
        self._viewer = napari_viewer
        self._app_state = get_or_create_app_state(napari_viewer)
        self._experimental_tiled_points = _experimental_tiled_points_enabled(experimental_tiled_points)
        if self._experimental_tiled_points:
            self._points_controller = TiledPointsController(
                self._app_state.viewer_adapter,
                on_state_changed=self._on_points_controller_state_changed,
                on_values_loaded=self._on_points_values_loaded,
            )
        else:
            self._points_controller = PointsController(
                on_state_changed=self._on_points_controller_state_changed,
                on_value_source_loaded=self._on_points_value_source_loaded,
                on_points_loaded=self._on_points_loaded,
            )
        self._last_points_load_request: PointsLoadRequest | None = None
        self._last_points_load_result: PointsLoadResult | None = None
        self._labels_cards: list[_LabelsCardWidget] = []
        self._image_cards: list[_ImageCardWidget] = []
        self._shape_cards: list[_ShapesCardWidget] = []
        self._labels_rows: list[_DisclosureElementWidget] = []
        self._image_rows: list[_DisclosureElementWidget] = []
        self._shape_rows: list[_DisclosureElementWidget] = []
        self._expanded_labels_names: set[str] = set()
        self._expanded_image_names: set[str] = set()
        self._expanded_shapes_names: set[str] = set()

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        self.scroll_area = QScrollArea()
        self.scroll_area.setObjectName("viewer_widget_scroll_area")
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setStyleSheet("QScrollArea { border: 0px; background: transparent; }")

        self.scroll_content = QWidget()
        self.scroll_content.setObjectName("viewer_widget_scroll_content")
        apply_scroll_content_surface(self.scroll_content)
        self.content_layout = QVBoxLayout(self.scroll_content)
        self.content_layout.setContentsMargins(12, 12, 12, 12)
        self.content_layout.setSpacing(10)

        header_logo = create_header_logo("viewer_widget_header_logo")

        self.open_sdata_button = QPushButton("Load SpatialData")
        self.open_sdata_button.setObjectName("viewer_widget_open_sdata_button")
        self.open_sdata_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.open_sdata_button.setMinimumHeight(28)
        self.open_sdata_button.setStyleSheet(ACTION_BUTTON_STYLESHEET)
        self.open_sdata_button.clicked.connect(self._open_spatialdata)

        self.empty_state_label = QLabel(
            'No SpatialData loaded. Use "Interactive(sdata)" for now; an in-widget open action will follow later.'
        )
        self.empty_state_label.setObjectName("viewer_widget_empty_state")
        self.empty_state_label.setWordWrap(True)
        self.empty_state_label.setStyleSheet(EMPTY_STATE_STYLESHEET)

        self.summary_label = QLabel("No SpatialData loaded.")
        self.summary_label.setObjectName("viewer_widget_summary")
        self.summary_label.setWordWrap(True)

        selector_layout = QFormLayout()
        selector_layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        selector_layout.setHorizontalSpacing(12)
        selector_layout.setVerticalSpacing(10)

        self.coordinate_system_combo = CompactComboBox()
        self.coordinate_system_combo.setObjectName("viewer_widget_coordinate_system_combo")
        self.coordinate_system_combo.currentIndexChanged.connect(self._on_coordinate_system_changed)
        self.coordinate_system_combo.setStyleSheet(INPUT_CONTROL_STYLESHEET)
        selector_layout.addRow(create_form_label("Coordinate system"), self.coordinate_system_combo)

        self.global_action_feedback_label = QLabel("")
        self.global_action_feedback_label.setObjectName("viewer_widget_action_feedback")
        self.global_action_feedback_label.setWordWrap(True)
        self.global_action_feedback_label.hide()

        self.images_empty_label = QLabel("No images available in the selected coordinate system.")
        self.images_empty_label.setObjectName("viewer_widget_images_empty_state")
        self.images_empty_label.setWordWrap(True)
        self.images_empty_label.setStyleSheet(EMPTY_STATE_STYLESHEET)

        self.images_section = QWidget()
        self.images_section.setObjectName("viewer_widget_images_section")
        self.images_section_layout = QVBoxLayout(self.images_section)
        self.images_section_layout.setContentsMargins(0, 0, 0, 0)
        self.images_section_layout.setSpacing(8)
        self.images_group = _CollapsibleSectionWidget(
            title="Images",
            object_name="viewer_widget_images_group",
            toggle_object_name="viewer_widget_images_section_toggle",
            expanded=False,
        )
        self.images_group.content_layout.addWidget(self.images_empty_label)
        self.images_group.content_layout.addWidget(self.images_section)
        self.images_section_toggle = self.images_group.toggle_button
        self.images_section_title = self.images_section_toggle

        self.labels_empty_label = QLabel("No labels available in the selected coordinate system.")
        self.labels_empty_label.setObjectName("viewer_widget_labels_empty_state")
        self.labels_empty_label.setWordWrap(True)
        self.labels_empty_label.setStyleSheet(EMPTY_STATE_STYLESHEET)

        self.labels_section = QWidget()
        self.labels_section.setObjectName("viewer_widget_labels_section")
        self.labels_section_layout = QVBoxLayout(self.labels_section)
        self.labels_section_layout.setContentsMargins(0, 0, 0, 0)
        self.labels_section_layout.setSpacing(8)
        self.labels_group = _CollapsibleSectionWidget(
            title="Labels",
            object_name="viewer_widget_labels_group",
            toggle_object_name="viewer_widget_labels_section_toggle",
            expanded=False,
        )
        self.labels_group.content_layout.addWidget(self.labels_empty_label)
        self.labels_group.content_layout.addWidget(self.labels_section)
        self.labels_section_toggle = self.labels_group.toggle_button
        self.labels_section_title = self.labels_section_toggle

        self.shapes_empty_label = QLabel("No shapes available in the selected coordinate system.")
        self.shapes_empty_label.setObjectName("viewer_widget_shapes_empty_state")
        self.shapes_empty_label.setWordWrap(True)
        self.shapes_empty_label.setStyleSheet(EMPTY_STATE_STYLESHEET)

        self.shapes_section = QWidget()
        self.shapes_section.setObjectName("viewer_widget_shapes_section")
        self.shapes_section_layout = QVBoxLayout(self.shapes_section)
        self.shapes_section_layout.setContentsMargins(0, 0, 0, 0)
        self.shapes_section_layout.setSpacing(8)
        self.shapes_group = _CollapsibleSectionWidget(
            title="Shapes",
            object_name="viewer_widget_shapes_group",
            toggle_object_name="viewer_widget_shapes_section_toggle",
            expanded=False,
        )
        self.shapes_group.content_layout.addWidget(self.shapes_empty_label)
        self.shapes_group.content_layout.addWidget(self.shapes_section)
        self.shapes_section_toggle = self.shapes_group.toggle_button
        self.shapes_section_title = self.shapes_section_toggle

        self.points_empty_label = QLabel("No points available in the selected coordinate system.")
        self.points_empty_label.setObjectName("viewer_widget_points_empty_state")
        self.points_empty_label.setWordWrap(True)
        self.points_empty_label.setStyleSheet(EMPTY_STATE_STYLESHEET)

        self.points_widget = PointsValueWidget()
        self.points_widget.source_changed.connect(self._on_points_source_changed)
        self.points_widget.add_update_requested.connect(self._add_or_update_points_selection)
        self.points_group = _CollapsibleSectionWidget(
            title="Points",
            object_name="viewer_widget_points_group",
            toggle_object_name="viewer_widget_points_section_toggle",
            expanded=False,
        )
        self.points_group.content_layout.addWidget(self.points_empty_label)
        self.points_group.content_layout.addWidget(self.points_widget)
        self.points_section_toggle = self.points_group.toggle_button
        self.points_section_title = self.points_section_toggle

        self.content_layout.addWidget(header_logo)
        self.content_layout.addWidget(self.open_sdata_button)
        self.content_layout.addWidget(self.empty_state_label)
        self.content_layout.addWidget(self.summary_label)
        self.content_layout.addLayout(selector_layout)
        self.content_layout.addWidget(self.global_action_feedback_label)
        self.content_layout.addWidget(self.images_group)
        self.content_layout.addWidget(self.labels_group)
        self.content_layout.addWidget(self.shapes_group)
        self.content_layout.addWidget(self.points_group)
        self.content_layout.addStretch(1)

        self.scroll_area.setWidget(self.scroll_content)
        root_layout.addWidget(self.scroll_area)

        self._app_state.sdata_changed.connect(self._on_sdata_changed)
        self._app_state.coordinate_system_changed.connect(self._on_app_state_coordinate_system_changed)
        # Accepted AnnData table-component changes can alter which tables are
        # linked to labels or shapes elements. Listen through shared app state
        # so the Viewer can refresh those cards without depending on the widget
        # that produced the change.
        self._app_state.table_state_changed.connect(self._on_table_state_changed)
        # Annotation can create/update shapes elements inside the active sdata;
        # refresh only the Viewer shapes section when that happens.
        self._app_state.shapes_element_written.connect(self._on_shapes_element_written)
        self._app_state.shapes_element_reloaded.connect(self._on_shapes_element_reloaded)
        self._app_state.viewer_adapter.image_layers_changed.connect(self._on_image_layers_changed)
        self.refresh_from_sdata(self._app_state.sdata)

    def closeEvent(self, event: Any) -> None:
        """Stop points work before the panel is destroyed."""
        self._points_controller.shutdown()
        super().closeEvent(event)

    @property
    def app_state(self) -> SpatiatoAppState:
        """Return the shared Spatiato app state for this widget."""
        return self._app_state

    @property
    def points_visualization_backend(self) -> Literal["in-memory", "tiled-cache"]:
        """Return the points backend fixed for this widget's lifetime."""
        return "tiled-cache" if self._experimental_tiled_points else "in-memory"

    @property
    def labels_cards(self) -> list[_LabelsCardWidget]:
        """Return the currently visible labels cards."""
        return list(self._labels_cards)

    @property
    def image_cards(self) -> list[_ImageCardWidget]:
        """Return the currently visible image cards."""
        return list(self._image_cards)

    @property
    def shape_cards(self) -> list[_ShapesCardWidget]:
        """Return the currently visible shapes cards."""
        return list(self._shape_cards)

    @property
    def image_rows(self) -> list[_DisclosureElementWidget]:
        """Return the currently visible compact image rows."""
        return list(self._image_rows)

    @property
    def labels_rows(self) -> list[_DisclosureElementWidget]:
        """Return the currently visible compact labels rows."""
        return list(self._labels_rows)

    @property
    def shape_rows(self) -> list[_DisclosureElementWidget]:
        """Return the currently visible compact shapes rows."""
        return list(self._shape_rows)

    def _on_sdata_changed(self, sdata: SpatialData | None) -> None:
        """Refresh the widget when the shared loaded `SpatialData` changes."""
        self.refresh_from_sdata(sdata)

    def _on_coordinate_system_changed(self, index: int) -> None:
        """Publish explicit user coordinate-system changes to shared app state."""
        coordinate_system = self.coordinate_system_combo.itemData(index)
        changed = self._app_state.set_coordinate_system(
            coordinate_system if isinstance(coordinate_system, str) else None,
            source="viewer_widget",
        )
        # changed is True
        #     → app state emits coordinate_system_changed
        #     → the shared event handler synchronizes all widgets
        #
        # changed is False
        #     → no event is emitted
        #     → this initiating widget must restore its own combo
        if not changed:
            self._sync_coordinate_system_combo_selection(self._app_state.coordinate_system)

    def _on_app_state_coordinate_system_changed(self, event: CoordinateSystemChangedEvent) -> None:
        """Refresh the combo and cards when the shared coordinate system changes."""
        del event
        self._clear_action_feedback()
        self._sync_coordinate_system_combo_selection(self._app_state.coordinate_system)
        self._refresh_coordinate_system_content()

    def _on_table_state_changed(self, event: object) -> None:
        """Refresh linked-table controls after an accepted table change."""
        if not isinstance(event, TableStateChangedEvent):
            return
        if event.sdata is not self._app_state.sdata:
            return
        # Existing `user_class` edits change row values and palette entries,
        # but they do not change which table color sources the Viewer can
        # discover. The object-classification widget refreshes the active
        # labels layer directly, so rebuilding every linked-table control here
        # only delays that row-scoped visual update. Keep refreshing for the
        # initial `created` event, when `user_class` first becomes available as
        # a Viewer color source.
        if event.source == "object_classification_annotation" and event.change_kind == "updated":
            return

        self._refresh_labels_card_linked_tables()
        self._refresh_shapes_card_linked_tables()

    def _on_shapes_element_written(self, event: object) -> None:
        """Refresh the shapes section after same-session annotation writes."""
        if not isinstance(event, ShapesElementWrittenEvent):
            return
        self._refresh_shapes_section_for_shared_event(event)

    def _on_shapes_element_reloaded(self, event: object) -> None:
        """Refresh the shapes section after persisted geometry is adopted."""
        if not isinstance(event, ShapesElementReloadedEvent):
            return
        self._refresh_shapes_section_for_shared_event(event)

    def _refresh_shapes_section_for_shared_event(
        self,
        event: ShapesElementWrittenEvent | ShapesElementReloadedEvent,
    ) -> None:
        if event.sdata is not self._app_state.sdata:
            return
        if event.coordinate_system != self._app_state.coordinate_system:
            return

        self._refresh_shapes_section()

    def _open_spatialdata(self, _checked: bool = False) -> None:
        selected_path = QFileDialog.getExistingDirectory(
            self,
            "Load SpatialData",
            "",
            QFileDialog.Option.ShowDirsOnly | QFileDialog.Option.DontResolveSymlinks,
        )
        if not selected_path:
            return

        if self._app_state.sdata is not None and not confirm_spatialdata_replacement(self):
            return

        try:
            # harpy's reader opens every table matrix lazily, as Dask arrays
            # that read from the store; `obs`, `var` and `uns` are read into memory.
            sdata = read_zarr(selected_path)
        except Exception as error:  # noqa: BLE001 - this UI boundary must report every load failure.
            self._set_action_feedback(
                title="SpatialData Load Error",
                lines=[f"Could not load SpatialData store: {error}"],
                kind="error",
            )
            return

        self._app_state.set_sdata(sdata, discard_current=True)
        self._clear_action_feedback()

    def refresh_from_sdata(self, sdata: SpatialData | None) -> None:
        """Refresh the viewer widget from the currently loaded `SpatialData`."""
        with QSignalBlocker(self.coordinate_system_combo):
            self.coordinate_system_combo.clear()

            if sdata is None:
                self.empty_state_label.show()
                set_status_card(
                    self.summary_label,
                    title="No SpatialData Loaded",
                    lines=["No SpatialData loaded."],
                    kind="info",
                )
                self.coordinate_system_combo.setEnabled(False)
                self._clear_cards()
                self._update_section_empty_states([], [], [], [])
                return

            coordinate_systems = get_coordinate_system_names_from_sdata(sdata)
            for coordinate_system in coordinate_systems:
                self.coordinate_system_combo.addItem(coordinate_system, coordinate_system)

            self.empty_state_label.hide()
            self.coordinate_system_combo.setEnabled(bool(coordinate_systems))

        self._sync_coordinate_system_combo_selection(self._app_state.coordinate_system)
        self._refresh_coordinate_system_content()

    def _refresh_coordinate_system_content(self) -> None:
        sdata = self._app_state.sdata
        coordinate_system = self._app_state.coordinate_system

        if sdata is None or not coordinate_system:
            self._clear_cards()
            self._update_section_empty_states([], [], [], [])
            if sdata is None:
                set_status_card(
                    self.summary_label,
                    title="No SpatialData Loaded",
                    lines=["No SpatialData loaded."],
                    kind="info",
                )
            else:
                set_status_card(
                    self.summary_label,
                    title="Current View",
                    lines=["No coordinate system selected."],
                    kind="info",
                )
            return

        labels_names = _get_labels_in_coordinate_system(sdata, coordinate_system)
        image_names = _get_images_in_coordinate_system(sdata, coordinate_system)
        shapes_names = _get_shapes_in_coordinate_system(sdata, coordinate_system)
        points_names = _get_points_in_coordinate_system(sdata, coordinate_system)

        self._set_coordinate_system_summary(
            coordinate_system,
            image_names=image_names,
            labels_names=labels_names,
            shapes_names=shapes_names,
            points_names=points_names,
        )
        self._rebuild_image_cards(sdata, image_names)
        self._rebuild_labels_cards(sdata, labels_names)
        self._rebuild_shapes_cards(sdata, shapes_names)
        self._refresh_points_section(sdata, points_names)
        self._update_section_empty_states(image_names, labels_names, shapes_names, points_names)

    def _refresh_shapes_section(self) -> None:
        sdata = self._app_state.sdata
        coordinate_system = self._app_state.coordinate_system
        if sdata is None or not coordinate_system:
            return

        image_names = _get_images_in_coordinate_system(sdata, coordinate_system)
        labels_names = _get_labels_in_coordinate_system(sdata, coordinate_system)
        shapes_names = _get_shapes_in_coordinate_system(sdata, coordinate_system)
        points_names = _get_points_in_coordinate_system(sdata, coordinate_system)

        self._set_coordinate_system_summary(
            coordinate_system,
            image_names=image_names,
            labels_names=labels_names,
            shapes_names=shapes_names,
            points_names=points_names,
        )
        self._rebuild_shapes_cards(sdata, shapes_names)
        self.shapes_group.set_count(len(shapes_names))
        self.shapes_empty_label.setVisible(not shapes_names)
        self.shapes_section.setVisible(bool(shapes_names))

    def _set_coordinate_system_summary(
        self,
        coordinate_system: str,
        *,
        image_names: list[str],
        labels_names: list[str],
        shapes_names: list[str],
        points_names: list[str],
    ) -> None:
        display_coordinate_system, coordinate_system_shortened = format_feedback_identifier(
            coordinate_system,
            max_length=_SUMMARY_COORDINATE_SYSTEM_MAX_LENGTH,
        )
        full_summary = (
            f'In coordinate system "{coordinate_system}": '
            f"{len(image_names)} image element(s), {len(labels_names)} labels element(s), "
            f"{len(shapes_names)} shapes element(s), and {len(points_names)} points element(s)."
        )
        set_status_card(
            self.summary_label,
            title="Current View",
            lines=[
                f'"{display_coordinate_system}": '
                f"{len(image_names)} image element(s), {len(labels_names)} labels element(s), "
                f"{len(shapes_names)} shapes element(s), and {len(points_names)} points element(s)."
            ],
            kind="info",
            tooltip_message=full_summary if coordinate_system_shortened else None,
        )

    def _rebuild_image_cards(self, sdata: SpatialData, image_names: list[str]) -> None:
        self._dispose_image_cards()
        _clear_layout(self.images_section_layout)
        self._image_cards = []
        self._image_rows = []
        self._expanded_image_names.intersection_update(image_names)

        for image_name in image_names:
            channel_error = None
            channel_names: list[str] = []
            try:
                channel_names = get_image_channel_names_from_sdata(sdata, image_name)
            except ValueError as error:
                channel_error = str(error)
            card = _ImageCardWidget(
                image_name=image_name,
                channel_names=channel_names,
                channel_error=channel_error,
            )
            # Stack and Overlay intentionally have separate load intents:
            # Stack loads only through its explicit pending action, while
            # accepting a composer channel immediately requests one focused
            # Overlay load. Do not route them through an aggregate image
            # Add/Update request.
            card.stack_load_requested.connect(self._load_image_stack)
            # Image cards emit intent only. ViewerWidget owns validation and
            # adapter mutations; adapter lifecycle reconciliation renders the
            # resulting live membership back into each card.
            card.overlay_channel_add_requested.connect(self._add_image_overlay_channel)
            card.overlay_channel_remove_requested.connect(self._remove_image_overlay_channel)
            card.overlay_channels_remove_all_requested.connect(self._remove_all_image_overlay_channels)
            card.overlay_channel_visibility_requested.connect(self._change_image_overlay_channel_visibility)
            card.overlay_channel_color_requested.connect(self._change_image_overlay_channel_color)
            card.stack_remove_requested.connect(self._remove_image_stack_layer)
            card.stack_visibility_requested.connect(self._change_image_stack_visibility)
            card.stack_color_requested.connect(self._change_image_stack_color)
            row = _DisclosureElementWidget(
                title=image_name,
                object_name=f"viewer_widget_image_row_{image_name}",
                toggle_object_name=f"viewer_widget_image_row_toggle_{image_name}",
                detail_widget=card,
                expanded=image_name in self._expanded_image_names,
            )
            row.expanded_changed.connect(
                lambda expanded, *, name=image_name: self._on_image_row_expanded(
                    name,
                    expanded,
                )
            )
            self.images_section_layout.addWidget(row)
            self._image_cards.append(card)
            self._image_rows.append(row)
            self._refresh_image_card_membership(card)

    def _rebuild_labels_cards(self, sdata: SpatialData, labels_names: list[str]) -> None:
        _clear_layout(self.labels_section_layout)
        self._labels_cards = []
        self._labels_rows = []
        self._expanded_labels_names.intersection_update(labels_names)

        for labels_name in labels_names:
            table_names = get_annotating_table_names(sdata, labels_name)
            card = _LabelsCardWidget(
                labels_name=labels_name,
                linked_tables=_get_linked_tables(sdata, table_names),
            )
            card.add_update_requested.connect(self._add_or_update_labels_layer)
            row = _DisclosureElementWidget(
                title=labels_name,
                object_name=f"viewer_widget_labels_row_{labels_name}",
                toggle_object_name=f"viewer_widget_labels_row_toggle_{labels_name}",
                detail_widget=card,
                expanded=labels_name in self._expanded_labels_names,
            )
            row.expanded_changed.connect(
                lambda expanded, *, name=labels_name: self._on_labels_row_expanded(
                    name,
                    expanded,
                )
            )
            self.labels_section_layout.addWidget(row)
            self._labels_cards.append(card)
            self._labels_rows.append(row)

    def _refresh_labels_card_linked_tables(self) -> None:
        sdata = self._app_state.sdata
        if sdata is None:
            return

        for card in self._labels_cards:
            table_names = get_annotating_table_names(sdata, card.labels_name)
            card.set_linked_tables(_get_linked_tables(sdata, table_names))

    def _refresh_shapes_card_linked_tables(self) -> None:
        sdata = self._app_state.sdata
        if sdata is None:
            return

        for card in self._shape_cards:
            table_names = get_annotating_table_names(sdata, card.shapes_name)
            card.set_linked_tables(_get_linked_tables(sdata, table_names))

    def _rebuild_shapes_cards(self, sdata: SpatialData, shapes_names: list[str]) -> None:
        _clear_layout(self.shapes_section_layout)
        self._shape_cards = []
        self._shape_rows = []
        self._expanded_shapes_names.intersection_update(shapes_names)

        for shapes_name in shapes_names:
            table_names = get_annotating_table_names(sdata, shapes_name)
            card = _ShapesCardWidget(
                shapes_name=shapes_name,
                shape_column_color_sources=get_shape_column_color_source_options(sdata, shapes_name),
                linked_tables=_get_linked_tables(sdata, table_names),
            )
            card.add_update_requested.connect(self._add_or_update_shapes_layer)
            row = _DisclosureElementWidget(
                title=shapes_name,
                object_name=f"viewer_widget_shapes_row_{shapes_name}",
                toggle_object_name=f"viewer_widget_shapes_row_toggle_{shapes_name}",
                detail_widget=card,
                expanded=shapes_name in self._expanded_shapes_names,
            )
            row.expanded_changed.connect(
                lambda expanded, *, name=shapes_name: self._on_shapes_row_expanded(
                    name,
                    expanded,
                )
            )
            self.shapes_section_layout.addWidget(row)
            self._shape_cards.append(card)
            self._shape_rows.append(row)

    def _refresh_points_section(self, sdata: SpatialData, points_names: list[str]) -> None:
        self.points_widget.set_points_names(points_names)
        self._refresh_points_index_columns(sdata)
        self._bind_points_source()

    def _refresh_points_index_columns(self, sdata: SpatialData | None) -> None:
        points_name = self.points_widget.selected_points_name()
        index_columns = [] if sdata is None or points_name is None else _get_points_index_columns(sdata, points_name)
        self.points_widget.set_index_columns(index_columns)

    def _on_points_source_changed(self) -> None:
        sdata = self._app_state.sdata
        self._refresh_points_index_columns(sdata)
        self._bind_points_source()

    def _bind_points_source(self) -> None:
        sdata = self._app_state.sdata
        coordinate_system = self._app_state.coordinate_system
        points_name = self.points_widget.selected_points_name()
        index_column = self.points_widget.selected_index_column()

        changed = self._points_controller.bind_source(
            sdata,
            points_name,
            coordinate_system,
            index_column,
        )
        if self._experimental_tiled_points:
            if not changed:
                self.points_widget.render_controller_state(self._points_controller)
            return

        if changed:
            self._last_points_load_request = None
            self._last_points_load_result = None
        if changed and self._points_controller.can_load_values:
            self._points_controller.load_value_source()
        else:
            self.points_widget.render_controller_state(self._points_controller)

    def _add_or_update_points_selection(self, values: Sequence[str] | Literal["all"], render_point_budget: int) -> None:
        if self._experimental_tiled_points:
            self._points_controller.apply_selection(
                values,
                render_point_budget=render_point_budget,
            )
            return

        self._last_points_load_request = None
        self._last_points_load_result = None
        self._points_controller.load_selection(
            values,
            render_point_budget=render_point_budget,
        )

    def _on_points_controller_state_changed(self) -> None:
        self.points_widget.render_controller_state(self._points_controller)
        if self._last_points_load_request is not None and self._last_points_load_result is not None:
            self._render_points_loaded_status(self._last_points_load_request, self._last_points_load_result)

    def _on_points_value_source_loaded(self, value_source: PointsValueSource) -> None:
        self.points_widget.set_value_source(value_source)

    def _on_points_values_loaded(self, values: tuple[str, ...]) -> None:
        self.points_widget.set_available_values(values)

    def _on_points_loaded(self, request: PointsLoadRequest) -> None:
        try:
            result = self._app_state.viewer_adapter._ensure_points_layer_from_selection(
                request.identity,
                selection=request.selection,
                categorical_colors=request.selected_value_colors or None,
            )
        except ValueError as error:
            self._set_action_feedback(
                title="Points Layer Error",
                lines=[str(error)],
                kind="error",
            )
            return

        self._last_points_load_request = request
        self._last_points_load_result = result
        self._app_state.viewer_adapter.activate_layer(result.layer)
        self._render_points_loaded_status(request, result)

    def _render_points_loaded_status(
        self,
        request: PointsLoadRequest,
        result: PointsLoadResult,
    ) -> None:
        self._apply_status_card_spec(
            self.global_action_feedback_label,
            build_points_layer_card_spec(request, result),
        )

    def _on_image_row_expanded(
        self,
        image_name: str,
        expanded: bool,
    ) -> None:
        if expanded:
            self._expanded_image_names.add(image_name)
            return

        self._expanded_image_names.discard(image_name)

    def _on_labels_row_expanded(
        self,
        labels_name: str,
        expanded: bool,
    ) -> None:
        if expanded:
            self._expanded_labels_names.add(labels_name)
            return

        self._expanded_labels_names.discard(labels_name)

    def _on_shapes_row_expanded(
        self,
        shapes_name: str,
        expanded: bool,
    ) -> None:
        if expanded:
            self._expanded_shapes_names.add(shapes_name)
            return

        self._expanded_shapes_names.discard(shapes_name)

    def _add_or_update_labels_layer(self, request: LabelsLoadRequest) -> None:
        if request.selected_source_kind is None:
            self._add_or_update_primary_labels_layer(request)
            return

        self._add_or_update_styled_labels_layer(request)

    def _add_or_update_primary_labels_layer(self, request: LabelsLoadRequest) -> None:
        sdata = self._app_state.sdata
        coordinate_system = self._app_state.coordinate_system
        labels_name = request.labels_name

        if sdata is None or not coordinate_system:
            self._set_action_feedback(
                title="Labels Load Error",
                lines=["Load a SpatialData object and select a coordinate system first."],
                kind="error",
            )
            return

        try:
            result = self._app_state.viewer_adapter.ensure_labels_loaded(sdata, labels_name, coordinate_system)
        except ValueError as error:
            self._set_action_feedback(title="Labels Load Error", lines=[str(error)], kind="error")
            return

        self._app_state.viewer_adapter.activate_layer(result.layer)
        self._apply_status_card_spec(
            self.global_action_feedback_label,
            build_primary_labels_loaded_card_spec(request, result),
        )

    def _add_or_update_styled_labels_layer(self, request: LabelsLoadRequest) -> None:
        sdata = self._app_state.sdata
        coordinate_system = self._app_state.coordinate_system

        if sdata is None or not coordinate_system:
            self._set_action_feedback(
                title="Styled Labels Error",
                lines=["Load a SpatialData object and select a coordinate system first."],
                kind="error",
            )
            return

        if request.table_name is None:
            self._set_action_feedback(
                title="Styled Labels Error",
                lines=[f'Labels element "{request.labels_name}" has no linked table for table-driven coloring.'],
                kind="error",
            )
            return

        if request.selected_color_source is None:
            if request.selected_source_kind == "obs_column":
                line = (
                    f"The selected observation column is not available in the linked table for "
                    f'"{request.labels_name}". Pick another observation column.'
                )
            else:
                line = (
                    f'The selected var is not available in the linked table for "{request.labels_name}". '
                    "Pick another var."
                )
            self._set_action_feedback(
                title="Styled Labels Error",
                lines=[line],
                kind="error",
            )
            return

        try:
            result = self._app_state.viewer_adapter.ensure_styled_labels_loaded(
                sdata,
                request.labels_name,
                coordinate_system,
                request.selected_color_source,
            )
        except ValueError as error:
            self._set_action_feedback(title="Styled Labels Error", lines=[str(error)], kind="error")
            return

        self._app_state.viewer_adapter.activate_layer(result.layer)
        self._apply_status_card_spec(
            self.global_action_feedback_label,
            build_styled_labels_card_spec(request, result),
        )

    def _add_or_update_shapes_layer(self, request: ShapesLoadRequest) -> None:
        if request.selected_source_kind is None:
            self._add_or_update_primary_shapes_layer(request)
            return

        self._add_or_update_styled_shapes_layer(request)

    def _add_or_update_primary_shapes_layer(self, request: ShapesLoadRequest) -> None:
        sdata = self._app_state.sdata
        coordinate_system = self._app_state.coordinate_system
        shapes_name = request.shapes_name

        if sdata is None or not coordinate_system:
            self._set_action_feedback(
                title="Shapes Load Error",
                lines=["Load a SpatialData object and select a coordinate system first."],
                kind="error",
            )
            return

        try:
            result = self._app_state.viewer_adapter.ensure_shapes_loaded(sdata, shapes_name, coordinate_system)
        except ValueError as error:
            self._set_action_feedback(title="Shapes Load Error", lines=[str(error)], kind="error")
            return

        self._app_state.viewer_adapter.activate_layer(result.layer)
        self._apply_status_card_spec(
            self.global_action_feedback_label,
            build_primary_shapes_loaded_card_spec(request, result),
        )

    def _add_or_update_styled_shapes_layer(self, request: ShapesLoadRequest) -> None:
        sdata = self._app_state.sdata
        coordinate_system = self._app_state.coordinate_system

        if sdata is None or not coordinate_system:
            self._set_action_feedback(
                title="Styled Shapes Error",
                lines=["Load a SpatialData object and select a coordinate system first."],
                kind="error",
            )
            return

        if request.selected_source_kind in {"obs_column", "var"} and request.table_name is None:
            self._set_action_feedback(
                title="Styled Shapes Error",
                lines=[f'Shapes element "{request.shapes_name}" has no linked table for table-driven coloring.'],
                kind="error",
            )
            return

        if request.selected_color_source is None:
            if request.selected_source_kind == "obs_column":
                line = (
                    f"The selected observation column is not available in the linked table for "
                    f'"{request.shapes_name}". Pick another observation column.'
                )
            elif request.selected_source_kind == "var":
                line = (
                    f'The selected var is not available in the linked table for "{request.shapes_name}". '
                    "Pick another var."
                )
            else:
                line = (
                    f'The selected shapes column is not available for "{request.shapes_name}". '
                    "Pick another shapes column."
                )
            self._set_action_feedback(
                title="Styled Shapes Error",
                lines=[line],
                kind="error",
            )
            return

        try:
            result = self._app_state.viewer_adapter.ensure_styled_shapes_loaded(
                sdata,
                request.shapes_name,
                coordinate_system,
                request.selected_color_source,
                fill=request.fill_shapes,
            )
        except ValueError as error:
            self._set_action_feedback(title="Styled Shapes Error", lines=[str(error)], kind="error")
            return

        self._app_state.viewer_adapter.activate_layer(result.layer)
        self._apply_status_card_spec(
            self.global_action_feedback_label,
            build_styled_shapes_card_spec(request, result),
        )

    def _load_image_stack(self, image_name: str) -> None:
        sdata = self._app_state.sdata
        coordinate_system = self._app_state.coordinate_system

        if sdata is None or not coordinate_system:
            self._set_action_feedback(
                title="Image Load Error",
                lines=["Load a SpatialData object and select a coordinate system first."],
                kind="error",
            )
            return

        try:
            result = self._app_state.viewer_adapter.ensure_image_loaded(
                sdata,
                image_name,
                coordinate_system,
                mode="stack",
            )
        except ValueError as error:
            self._set_action_feedback(title="Image Load Error", lines=[str(error)], kind="error")
            return

        self._app_state.viewer_adapter.activate_layer(result.primary_layer)
        self._apply_status_card_spec(
            self.global_action_feedback_label,
            build_image_loaded_card_spec(image_name, result),
        )

    def _on_image_layers_changed(self) -> None:
        """Re-query complete image membership after adapter lifecycle changes."""
        for card in self._image_cards:
            self._refresh_image_card_membership(card)

    def _refresh_image_card_membership(self, card: _ImageCardWidget) -> None:
        sdata = self._app_state.sdata
        coordinate_system = self._app_state.coordinate_system
        if sdata is None or not coordinate_system:
            card.set_loaded_image_bindings(
                stack_binding=None,
                overlay_bindings=(),
            )
            return

        try:
            stack_binding, overlay_bindings = self._get_live_image_bindings(
                sdata,
                card.image_name,
                coordinate_system,
                channel_names=card.channel_names,
            )
            card.set_loaded_image_bindings(
                stack_binding=stack_binding,
                overlay_bindings=overlay_bindings,
            )
        except ValueError as error:
            message = str(error)
            card.set_image_membership_error(message)
            self._set_action_feedback(
                title="Image Membership Error",
                lines=[message],
                kind="error",
            )

    def _get_live_image_bindings(
        self,
        sdata: SpatialData,
        image_name: str,
        coordinate_system: str,
        *,
        channel_names: Sequence[str],
    ) -> tuple[ImageLayerBinding | None, list[ImageLayerBinding]]:
        """Partition valid image bindings in the viewer's current layer order."""
        adapter = self._app_state.viewer_adapter
        stack_bindings: list[ImageLayerBinding] = []
        overlay_bindings: list[ImageLayerBinding] = []
        seen_channel_indices: set[int] = set()

        for layer in adapter.get_loaded_image_layers(sdata, image_name):
            binding = adapter.layer_bindings.get_binding(layer)
            if not isinstance(binding, ImageLayerBinding):
                continue
            if binding.sdata_id != id(sdata):
                continue
            if binding.element_name != image_name:
                continue
            if binding.coordinate_system != coordinate_system:
                continue
            if binding.image_display_mode == "stack":
                stack_bindings.append(binding)
                continue
            if binding.image_display_mode != "overlay":
                continue

            channel_index = binding.channel_index
            channel_name = binding.channel_name
            if not isinstance(channel_index, int) or channel_index < 0:
                raise ValueError(f"Overlay binding for image `{image_name}` has no valid channel index.")
            if channel_index >= len(channel_names):
                raise ValueError(
                    f"Overlay binding for image `{image_name}` references channel index "
                    f"{channel_index}, but the image has {len(channel_names)} channels."
                )
            if channel_name != channel_names[channel_index]:
                raise ValueError(
                    f"Overlay binding for image `{image_name}` and channel index {channel_index} "
                    "does not match the current channel name."
                )
            if channel_index in seen_channel_indices:
                raise ValueError(
                    f"Found multiple live overlay bindings for image `{image_name}`, "
                    f"coordinate system `{coordinate_system}`, and channel index {channel_index}."
                )

            seen_channel_indices.add(channel_index)
            overlay_bindings.append(binding)

        if len(stack_bindings) > 1:
            raise ValueError(
                f"Found multiple live stack bindings for image `{image_name}` "
                f"and coordinate system `{coordinate_system}`."
            )
        if stack_bindings and overlay_bindings:
            raise ValueError(
                f"Image `{image_name}` cannot have both live stack and overlay bindings "
                f"in coordinate system `{coordinate_system}`."
            )

        stack_binding = stack_bindings[0] if stack_bindings else None
        return stack_binding, overlay_bindings

    def _find_image_card(self, image_name: str) -> _ImageCardWidget | None:
        return next(
            (card for card in self._image_cards if card.image_name == image_name),
            None,
        )

    def _resolve_live_overlay_binding(
        self,
        image_name: str,
        channel_index: int,
    ) -> ImageLayerBinding | None:
        """Resolve one channel against the Viewer's current live context."""
        card = self._find_image_card(image_name)
        sdata = self._app_state.sdata
        coordinate_system = self._app_state.coordinate_system
        if card is None or sdata is None or not coordinate_system:
            return None

        _, bindings = self._get_live_image_bindings(
            sdata,
            image_name,
            coordinate_system,
            channel_names=card.channel_names,
        )
        return next(
            (binding for binding in bindings if binding.channel_index == channel_index),
            None,
        )

    def _resolve_exact_live_stack_binding(
        self,
        image_name: str,
    ) -> ImageLayerBinding | None:
        """Resolve the Stack row against the Viewer's current live context."""
        card = self._find_image_card(image_name)
        sdata = self._app_state.sdata
        coordinate_system = self._app_state.coordinate_system
        if card is None or sdata is None or not coordinate_system:
            return None

        stack_binding, _ = self._get_live_image_bindings(
            sdata,
            image_name,
            coordinate_system,
            channel_names=card.channel_names,
        )
        row = card.stack_row
        if stack_binding is not None and row is not None and row.binding is stack_binding:
            return stack_binding

        self._refresh_image_card_membership(card)
        return None

    def _change_image_stack_visibility(
        self,
        image_name: str,
        visible: bool,
    ) -> None:
        """Apply Stack-row intent; napari's ``visible`` event renders it."""
        card = self._find_image_card(image_name)
        if card is None:
            return

        try:
            binding = self._resolve_exact_live_stack_binding(image_name)
        except ValueError as error:
            self._set_action_feedback(
                title="Image Stack Error",
                lines=[str(error)],
                kind="error",
            )
            return
        if binding is None:
            return

        layer = binding.layer
        try:
            if layer.visible == visible:
                card.refresh_stack_presentation(binding)
                return
            layer.visible = visible
        except (TypeError, RuntimeError, ValueError) as error:
            card.refresh_stack_presentation(binding)
            self._set_action_feedback(
                title="Image Stack Error",
                lines=[f"Could not update Stack visibility: {error}"],
                kind="error",
            )

    def _change_image_stack_color(
        self,
        image_name: str,
        color: str,
    ) -> None:
        """Apply Stack-row intent; napari's ``colormap`` event renders it."""
        card = self._find_image_card(image_name)
        if card is None:
            return

        try:
            binding = self._resolve_exact_live_stack_binding(image_name)
        except ValueError as error:
            self._set_action_feedback(
                title="Image Stack Error",
                lines=[str(error)],
                kind="error",
            )
            return
        if binding is None:
            return

        layer = binding.layer
        if layer.rgb:
            card.refresh_stack_presentation(binding)
            return

        normalized_color = _normalized_color_or_none(color)
        if normalized_color is None:
            card.refresh_stack_presentation(binding)
            self._set_action_feedback(
                title="Image Stack Error",
                lines=[f'Could not update Stack color: "{color}" is not a valid color.'],
                kind="error",
            )
            return

        try:
            if _solid_color_from_layer(layer) == normalized_color:
                card.refresh_stack_presentation(binding)
                return
            layer.colormap = normalized_color
        except (TypeError, RuntimeError, ValueError) as error:
            card.refresh_stack_presentation(binding)
            self._set_action_feedback(
                title="Image Stack Error",
                lines=[f"Could not update Stack color: {error}"],
                kind="error",
            )

    def _remove_image_stack_layer(self, image_name: str) -> None:
        """Remove the exact current Stack binding while preserving other images."""
        card = self._find_image_card(image_name)
        sdata = self._app_state.sdata
        coordinate_system = self._app_state.coordinate_system
        if card is None or sdata is None or not coordinate_system:
            self._set_action_feedback(
                title="Image Stack Error",
                lines=["Load a SpatialData object and select a coordinate system first."],
                kind="error",
            )
            return

        binding: ImageLayerBinding | None = None
        try:
            binding = self._resolve_exact_live_stack_binding(image_name)
            if binding is None:
                return
            removed_layer = self._app_state.viewer_adapter.remove_image_stack_layer(
                sdata,
                image_name,
                coordinate_system,
            )
        except (TypeError, RuntimeError, ValueError) as error:
            if binding is not None:
                card.refresh_stack_presentation(binding)
            self._set_action_feedback(
                title="Image Stack Error",
                lines=[f"Could not remove Stack: {error}"],
                kind="error",
            )
            return

        if removed_layer is None:
            self._refresh_image_card_membership(card)
            return
        self._set_action_feedback(
            title="Image Stack Updated",
            lines=[f'Removed Stack for image "{image_name}" from the viewer.'],
            kind="success",
        )

    def _change_image_overlay_channel_visibility(
        self,
        image_name: str,
        channel_index: int,
        visible: bool,
    ) -> None:
        """Apply row intent; the napari ``visible`` event renders the result."""
        card = self._find_image_card(image_name)
        if card is None:
            return

        try:
            binding = self._resolve_live_overlay_binding(
                image_name,
                channel_index,
            )
        except ValueError as error:
            self._set_action_feedback(
                title="Image Overlay Error",
                lines=[str(error)],
                kind="error",
            )
            return

        if binding is None:
            self._refresh_image_card_membership(card)
            return

        layer = binding.layer
        try:
            if layer.visible == visible:
                card.refresh_overlay_channel_presentation(channel_index, binding)
                return
            layer.visible = visible
        except (TypeError, RuntimeError, ValueError) as error:
            card.refresh_overlay_channel_presentation(channel_index, binding)
            self._set_action_feedback(
                title="Image Overlay Error",
                lines=[f"Could not update channel visibility: {error}"],
                kind="error",
            )
            return

    def _change_image_overlay_channel_color(
        self,
        image_name: str,
        channel_index: int,
        color: str,
    ) -> None:
        """Apply row intent; the napari ``colormap`` event renders the result."""
        card = self._find_image_card(image_name)
        if card is None:
            return

        try:
            binding = self._resolve_live_overlay_binding(
                image_name,
                channel_index,
            )
        except ValueError as error:
            self._set_action_feedback(
                title="Image Overlay Error",
                lines=[str(error)],
                kind="error",
            )
            return

        if binding is None:
            self._refresh_image_card_membership(card)
            return

        normalized_color = _normalized_color_or_none(color)
        if normalized_color is None:
            card.refresh_overlay_channel_presentation(channel_index, binding)
            self._set_action_feedback(
                title="Image Overlay Error",
                lines=[f'Could not update channel color: "{color}" is not a valid color.'],
                kind="error",
            )
            return

        layer = binding.layer
        try:
            if _solid_color_from_layer(layer) == normalized_color:
                card.refresh_overlay_channel_presentation(channel_index, binding)
                return
            layer.colormap = normalized_color
        except (TypeError, RuntimeError, ValueError) as error:
            card.refresh_overlay_channel_presentation(channel_index, binding)
            self._set_action_feedback(
                title="Image Overlay Error",
                lines=[f"Could not update channel color: {error}"],
                kind="error",
            )
            return

    def _add_image_overlay_channel(
        self,
        image_name: str,
        channel_index: int,
        channel_color: str,
    ) -> None:
        """Handle one card add intent through the focused adapter operation."""
        card = self._find_image_card(image_name)
        sdata = self._app_state.sdata
        coordinate_system = self._app_state.coordinate_system
        if card is None or sdata is None or not coordinate_system:
            if card is not None:
                card.finish_overlay_channel_add(channel_index, succeeded=False)
            self._set_action_feedback(
                title="Image Overlay Error",
                lines=["Load a SpatialData object and select a coordinate system first."],
                kind="error",
            )
            return

        try:
            _, live_bindings = self._get_live_image_bindings(
                sdata,
                image_name,
                coordinate_system,
                channel_names=card.channel_names,
            )
            existing_binding = next(
                (binding for binding in live_bindings if binding.channel_index == channel_index),
                None,
            )
            if existing_binding is not None:
                card.set_loaded_image_bindings(
                    stack_binding=None,
                    overlay_bindings=live_bindings,
                )
                card.finish_overlay_channel_add(channel_index, succeeded=True)
                self._app_state.viewer_adapter.activate_layer(existing_binding.layer)
                return

            result = self._app_state.viewer_adapter.ensure_image_overlay_channel_loaded(
                sdata,
                image_name,
                coordinate_system,
                channel=channel_index,
                channel_color=channel_color,
            )
        except ValueError as error:
            card.finish_overlay_channel_add(channel_index, succeeded=False)
            self._set_action_feedback(
                title="Image Overlay Error",
                lines=[str(error)],
                kind="error",
            )
            return

        card.finish_overlay_channel_add(channel_index, succeeded=True)
        self._app_state.viewer_adapter.activate_layer(result.primary_layer)
        self._apply_status_card_spec(
            self.global_action_feedback_label,
            build_image_loaded_card_spec(image_name, result),
        )

    def _remove_image_overlay_channel(
        self,
        image_name: str,
        channel_index: int,
    ) -> None:
        """Handle one card removal intent while preserving sibling layers."""
        card = self._find_image_card(image_name)
        sdata = self._app_state.sdata
        coordinate_system = self._app_state.coordinate_system
        if card is None or sdata is None or not coordinate_system:
            self._set_action_feedback(
                title="Image Overlay Error",
                lines=["Load a SpatialData object and select a coordinate system first."],
                kind="error",
            )
            return

        card.cache_overlay_channel_color(channel_index)
        try:
            removed_layer = self._app_state.viewer_adapter.remove_image_overlay_channel(
                sdata,
                image_name,
                coordinate_system,
                channel_index=channel_index,
            )
        except ValueError as error:
            self._set_action_feedback(
                title="Image Overlay Error",
                lines=[str(error)],
                kind="error",
            )
            return

        if removed_layer is None:
            self._refresh_image_card_membership(card)
            return

        card.finish_overlay_channel_remove(channel_index, succeeded=True)
        self._set_action_feedback(
            title="Image Overlay Updated",
            lines=[f'Removed channel "{card.channel_names[channel_index]}" from the viewer.'],
            kind="success",
        )

    def _remove_all_image_overlay_channels(self, image_name: str) -> None:
        """Handle remove-all intent for one image's current overlay bindings."""
        card = self._find_image_card(image_name)
        sdata = self._app_state.sdata
        coordinate_system = self._app_state.coordinate_system
        if card is None or sdata is None or not coordinate_system:
            self._set_action_feedback(
                title="Image Overlay Error",
                lines=["Load a SpatialData object and select a coordinate system first."],
                kind="error",
            )
            return

        try:
            _, bindings = self._get_live_image_bindings(
                sdata,
                image_name,
                coordinate_system,
                channel_names=card.channel_names,
            )
            card.cache_all_overlay_channel_colors()
            missing_target = False
            for binding in tuple(bindings):
                assert binding.channel_index is not None
                removed_layer = self._app_state.viewer_adapter.remove_image_overlay_channel(
                    sdata,
                    image_name,
                    coordinate_system,
                    channel_index=binding.channel_index,
                )
                missing_target = missing_target or removed_layer is None
                card.finish_overlay_channel_remove(
                    binding.channel_index,
                    succeeded=removed_layer is not None,
                )
        except ValueError as error:
            self._set_action_feedback(
                title="Image Overlay Error",
                lines=[str(error)],
                kind="error",
            )
            return

        if missing_target:
            self._refresh_image_card_membership(card)
        if bindings:
            self._set_action_feedback(
                title="Image Overlay Updated",
                lines=[f'Removed all overlay channels for image "{image_name}".'],
                kind="success",
            )

    def _update_section_empty_states(
        self,
        image_names: list[str],
        labels_names: list[str],
        shapes_names: list[str],
        points_names: list[str],
    ) -> None:
        self.images_group.set_count(len(image_names))
        self.labels_group.set_count(len(labels_names))
        self.shapes_group.set_count(len(shapes_names))
        self.points_group.set_count(len(points_names))
        self.images_empty_label.setVisible(not image_names)
        self.labels_empty_label.setVisible(not labels_names)
        self.shapes_empty_label.setVisible(not shapes_names)
        self.points_empty_label.setVisible(not points_names)
        self.images_section.setVisible(bool(image_names))
        self.labels_section.setVisible(bool(labels_names))
        self.shapes_section.setVisible(bool(shapes_names))
        self.points_widget.setVisible(bool(points_names))

    def _clear_cards(self) -> None:
        self._dispose_image_cards()
        _clear_layout(self.images_section_layout)
        _clear_layout(self.labels_section_layout)
        _clear_layout(self.shapes_section_layout)
        self._image_cards = []
        self._labels_cards = []
        self._shape_cards = []
        self._image_rows = []
        self._labels_rows = []
        self._shape_rows = []
        self._expanded_image_names.clear()
        self._expanded_labels_names.clear()
        self._expanded_shapes_names.clear()
        self.points_widget.set_points_names([])
        self.points_widget.set_index_columns([])
        self.points_widget.set_value_source(None)
        self._last_points_load_request = None
        self._last_points_load_result = None
        self._points_controller.bind_source(None, None, None, None)

    def _dispose_image_cards(self) -> None:
        """Disconnect all image-card rows before their widgets are replaced."""
        for card in tuple(self._image_cards):
            card.dispose()

    def _set_action_feedback(
        self,
        message: str | None = None,
        *,
        title: str | None = None,
        lines: list[str] | None = None,
        kind: StatusCardKind | None = None,
        is_error: bool | None = None,
        tooltip_message: str | None = None,
    ) -> None:
        """Render viewer action feedback as the shared status-card pattern."""
        self._apply_status_card_spec(
            self.global_action_feedback_label,
            build_viewer_feedback_card_spec(
                message,
                title=title,
                lines=lines,
                kind=kind,
                is_error=is_error,
                tooltip_message=tooltip_message,
            ),
        )

    def _apply_status_card_spec(self, label: QLabel, spec: _ViewerStatusCardSpec | None) -> None:
        if spec is None:
            label.clear()
            label.setToolTip("")
            label.setStyleSheet("")
            label.hide()
            return

        set_status_card(
            label,
            title=spec.title,
            lines=list(spec.lines),
            kind=spec.kind,
            tooltip_message=spec.tooltip_message,
        )

    def _clear_action_feedback(self) -> None:
        self._apply_status_card_spec(self.global_action_feedback_label, None)

    def _sync_coordinate_system_combo_selection(self, coordinate_system: str | None) -> None:
        with QSignalBlocker(self.coordinate_system_combo):
            if coordinate_system is None:
                self.coordinate_system_combo.setCurrentIndex(-1)
                return

            index = self.coordinate_system_combo.findData(coordinate_system)
            self.coordinate_system_combo.setCurrentIndex(index)


def _get_linked_tables(sdata: SpatialData, table_names: list[str]) -> dict[str, LinkedTableColorSources]:
    """Return the color sources of each linked table, by table name."""
    return {table_name: LinkedTableColorSources.from_table(sdata, table_name) for table_name in table_names}


def _clear_layout(layout: QLayout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        child_layout = item.layout()
        if widget is not None:
            widget.deleteLater()
        elif child_layout is not None:
            _clear_layout(child_layout)


def _get_labels_in_coordinate_system(sdata: SpatialData, coordinate_system: str) -> list[str]:
    return [
        option.labels_name
        for option in get_spatialdata_labels_options_for_coordinate_system_from_sdata(
            sdata=sdata,
            coordinate_system=coordinate_system,
        )
    ]


def _get_images_in_coordinate_system(sdata: SpatialData, coordinate_system: str) -> list[str]:
    return [
        option.image_name
        for option in get_spatialdata_image_options_for_coordinate_system_from_sdata(
            sdata=sdata,
            coordinate_system=coordinate_system,
        )
    ]


def _get_shapes_in_coordinate_system(sdata: SpatialData, coordinate_system: str) -> list[str]:
    return [
        option.shapes_name
        for option in get_spatialdata_shapes_options_for_coordinate_system_from_sdata(
            sdata=sdata,
            coordinate_system=coordinate_system,
        )
    ]


def _get_points_in_coordinate_system(sdata: SpatialData, coordinate_system: str) -> list[str]:
    return [
        option.points_name
        for option in get_spatialdata_points_options_for_coordinate_system_from_sdata(
            sdata=sdata,
            coordinate_system=coordinate_system,
        )
    ]


def _get_points_index_columns(sdata: SpatialData, points_name: str) -> list[str]:
    points = getattr(sdata, "points", {}).get(points_name)
    if points is None:
        return []

    meta = getattr(points, "_meta", points)
    columns = list(getattr(meta, "columns", ()))
    index_columns: list[str] = []
    for column in columns:
        column_name = str(column)
        if column_name in {"x", "y"}:
            continue

        dtype = getattr(meta[column], "dtype", None)
        if dtype is None or is_bool_dtype(dtype) or is_numeric_dtype(dtype):
            continue
        if isinstance(dtype, pd.CategoricalDtype) or is_string_dtype(dtype) or is_object_dtype(dtype):
            index_columns.append(column_name)

    return index_columns
