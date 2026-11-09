from __future__ import annotations

from collections.abc import Callable
from html import unescape
from types import SimpleNamespace

import dask
import dask.array as da
import dask.dataframe as dd
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_rgba
from napari.layers import Image, Shapes
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor
from qtpy.QtWidgets import QCheckBox, QComboBox, QCompleter, QLabel
from shapely.geometry import LineString, Polygon
from spatialdata import SpatialData
from spatialdata.models import ShapesModel
from spatialdata.transformations import Identity

import spatiato._app_state as app_state_module
import spatiato.widgets.overlay_color_button as overlay_color_button_module
import spatiato.widgets.viewer.widget as viewer_widget_module
from spatiato._app_state import (
    ShapesElementReloadedEvent,
    ShapesElementWrittenEvent,
    TableChangeKind,
    TableStateChangedEvent,
)
from spatiato._points_value_index import PointsValueSelection, PointsValueTable
from spatiato.core._color_source import ShapeColumnColorSourceSpec, TableColorSourceSpec
from spatiato.core.persistence import TableComponentPath
from spatiato.viewer.adapter import ImageLayerBinding, PointsLayerIdentity
from spatiato.viewer.shapes_styling import SHAPES_FACE_ALPHA
from spatiato.widgets.overlay_color_button import OverlayColorButton
from spatiato.widgets.shared_styles import (
    STATUS_CARD_PALETTE,
    WIDGET_MIN_WIDTH,
    CompactComboBox,
    _ElidedLabel,
)
from spatiato.widgets.viewer.disclosure import _CollapsibleSectionWidget, _ElidedToolButton
from spatiato.widgets.viewer.image_widget import _ImageCardWidget
from spatiato.widgets.viewer.points_controller import PointsController, PointsLoadRequest
from spatiato.widgets.viewer.shapes_widget import ShapesLoadRequest
from spatiato.widgets.viewer.tiled_points_controller import TiledPointsController
from spatiato.widgets.viewer.widget import ViewerWidget


def _table_event(
    sdata: object,
    *paths: TableComponentPath,
    table_name: str = "table",
    source: str = "test",
    change_kind: TableChangeKind = "updated",
) -> TableStateChangedEvent:
    return TableStateChangedEvent(
        sdata=sdata,
        table_name=table_name,
        paths=frozenset(paths),
        regions=(),
        change_kind=change_kind,
        source=source,
    )


class DummyEventEmitter:
    def __init__(self) -> None:
        self._callbacks: list[Callable[[object], None]] = []

    def connect(self, callback: Callable[[object], None]) -> None:
        self._callbacks.append(callback)

    def disconnect(self, callback: Callable[[object], None]) -> None:
        self._callbacks.remove(callback)

    def emit(self, value: object | None = None) -> None:
        event = SimpleNamespace(value=value)
        for callback in list(self._callbacks):
            callback(event)


class DummyLayers(list):
    def __init__(self) -> None:
        super().__init__()
        self.selection = SimpleNamespace(active=None, select_only=self._select_only)
        self.events = SimpleNamespace(
            inserted=DummyEventEmitter(),
            removed=DummyEventEmitter(),
            reordered=DummyEventEmitter(),
        )

    def _select_only(self, layer: object) -> None:
        self.selection.active = layer


class DummyViewer:
    def __init__(self) -> None:
        self.layers = DummyLayers()


_FEEDBACK_BACKGROUND_BY_KIND = {kind: palette["background"] for kind, palette in STATUS_CARD_PALETTE.items()}


def _assert_action_feedback_card(widget: ViewerWidget, *, title: str, kind: str) -> None:
    assert title in widget.global_action_feedback_label.text()
    assert f"background-color: {_FEEDBACK_BACKGROUND_BY_KIND[kind]}" in widget.global_action_feedback_label.styleSheet()
    assert not widget.global_action_feedback_label.isHidden()


def _label_text(label) -> str:
    return unescape(label.text())


def _tooltip_text(label) -> str:
    return unescape(label.toolTip()).replace("&#8203;", "").replace("\u200b", "")


def _patch_coordinate_system_names(monkeypatch, coordinate_systems: list[str]) -> None:
    monkeypatch.setattr(
        viewer_widget_module,
        "get_coordinate_system_names_from_sdata",
        lambda sdata: list(coordinate_systems),
    )
    monkeypatch.setattr(
        app_state_module,
        "get_coordinate_system_names_from_sdata",
        lambda sdata: list(coordinate_systems),
    )


def _combo_texts(combo: QComboBox) -> list[str]:
    return [combo.itemText(index) for index in range(combo.count())]


def _select_color_source_kind(card: object, source_kind: str) -> None:
    index = card.color_source_kind_combo.findData(source_kind)
    assert index >= 0
    card.color_source_kind_combo.setCurrentIndex(index)


def _patch_viewer_widget_labels_tables(
    monkeypatch,
    *,
    labels_names: list[str],
    table_names_by_label: dict[str, list[str]],
    color_sources_by_table: dict[str, list[TableColorSourceSpec]] | None = None,
) -> None:
    monkeypatch.setattr(viewer_widget_module, "_get_images_in_coordinate_system", lambda sdata, coordinate_system: [])
    monkeypatch.setattr(viewer_widget_module, "_get_shapes_in_coordinate_system", lambda sdata, coordinate_system: [])
    monkeypatch.setattr(viewer_widget_module, "_get_points_in_coordinate_system", lambda sdata, coordinate_system: [])
    monkeypatch.setattr(
        viewer_widget_module,
        "_get_labels_in_coordinate_system",
        lambda sdata, coordinate_system: list(labels_names),
    )
    monkeypatch.setattr(
        viewer_widget_module,
        "get_annotating_table_names",
        lambda sdata, labels_name: list(table_names_by_label.get(labels_name, [])),
    )
    monkeypatch.setattr(
        viewer_widget_module,
        "get_table_color_source_options",
        lambda sdata, table_name: list((color_sources_by_table or {}).get(table_name, [])),
    )


def _make_shapes_sdata(geodataframe: gpd.GeoDataFrame, shapes_name: str = "cells") -> SimpleNamespace:
    shapes = ShapesModel.parse(geodataframe, transformations={"global": Identity()})
    return SimpleNamespace(shapes={shapes_name: shapes}, tables={})


def _points_dataframe(data: dict[str, object]) -> dd.DataFrame:
    with dask.config.set({"dataframe.convert-string": False}):
        return dd.from_pandas(pd.DataFrame(data), npartitions=1)


def _make_points_sdata(points_name: str = "transcripts") -> SimpleNamespace:
    return SimpleNamespace(
        points={
            points_name: _points_dataframe(
                {
                    "x": [0.0, 1.0, 2.0],
                    "y": [3.0, 4.0, 5.0],
                    "gene": ["AAMP", "AXL", "MALAT1"],
                    "target": ["T1", "T2", "T3"],
                    "score": [0.1, 0.2, 0.3],
                }
            )
        }
    )


def _make_colorable_shapes_sdata(
    *,
    shapes_name: str = "cells",
    cell_type_colors: list[str] | None = None,
    duplicate_index: bool = False,
    include_unsupported_geometry: bool = False,
) -> SimpleNamespace:
    geometries = [
        Polygon([(0, 0), (4, 0), (4, 4), (0, 4), (0, 0)]),
        Polygon([(5, 0), (9, 0), (9, 4), (5, 4), (5, 0)]),
    ]
    if include_unsupported_geometry:
        geometries[1] = LineString([(5, 0), (9, 4)])

    data: dict[str, object] = {
        "cell_type": pd.Categorical(["T", "B"], categories=["T", "B"]),
        "score": [0.0, 1.0],
        "free_text": ["alpha", "beta"],
    }
    if cell_type_colors is not None:
        data["cell_type_colors"] = cell_type_colors

    index = ["cell_1", "cell_1"] if duplicate_index else ["cell_1", "cell_2"]
    geodataframe = gpd.GeoDataFrame(data, geometry=geometries, index=index)
    return _make_shapes_sdata(geodataframe, shapes_name=shapes_name)


def _select_shape_column(card: object, value_key: str) -> None:
    _select_color_source_kind(card, "shape_column")
    card.color_source_value_input.setText(value_key)


def _make_points_load_request(sdata: object) -> PointsLoadRequest:
    value_table = PointsValueTable(
        values=pd.DataFrame(
            {
                "value_id": pd.Series([0], dtype="uint32"),
                "value": ["AAMP"],
                "n_points": pd.Series([2], dtype="uint64"),
            }
        ),
        index_column="gene",
        total_count=2,
    )
    selection = PointsValueSelection(
        coordinates=np.asarray([[3.0, 0.0], [4.0, 1.0]], dtype="float32"),
        features=pd.DataFrame(
            {
                "gene": pd.Categorical(["AAMP", "AAMP"], categories=["AAMP"]),
                "value_id": pd.Series([0, 0], dtype="uint32"),
            }
        ),
        index_column="gene",
        selected_values=("AAMP",),
        selected_value_ids=(0,),
        selection_mode="values",
        total_count=2,
        render_point_budget=100_000,
        is_sampled=False,
        warning=None,
    )
    return PointsLoadRequest(
        identity=PointsLayerIdentity(
            sdata=sdata,
            points_name="transcripts",
            coordinate_system="global",
            index_column="gene",
        ),
        selection=selection,
        value_table=value_table,
    )


@pytest.fixture(autouse=True)
def _disable_experimental_tiled_points_by_default(monkeypatch) -> None:
    monkeypatch.delenv(viewer_widget_module._EXPERIMENTAL_TILED_POINTS_ENV, raising=False)


def test_viewer_widget_can_be_instantiated(qtbot) -> None:
    widget = ViewerWidget()

    qtbot.addWidget(widget)

    assert widget is not None
    assert widget.findChild(QLabel, "viewer_widget_header_logo") is not None
    assert widget.app_state.sdata is None
    assert not widget.empty_state_label.isHidden()
    assert "No SpatialData Loaded" in _label_text(widget.summary_label)
    assert "No SpatialData loaded." in _label_text(widget.summary_label)
    assert widget.coordinate_system_combo.count() == 0
    assert not widget.coordinate_system_combo.isEnabled()
    assert isinstance(widget.coordinate_system_combo, CompactComboBox)
    assert widget.coordinate_system_combo.sizeAdjustPolicy() == (
        QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
    )
    assert widget.image_cards == []
    assert widget.labels_cards == []
    assert widget.shape_cards == []
    assert widget.points_visualization_backend == "in-memory"
    assert isinstance(widget._points_controller, PointsController)


def test_viewer_widget_tiled_points_backend_is_opt_in(qtbot, monkeypatch) -> None:
    monkeypatch.setenv(viewer_widget_module._EXPERIMENTAL_TILED_POINTS_ENV, "1")

    widget = ViewerWidget()
    qtbot.addWidget(widget)

    assert widget.points_visualization_backend == "tiled-cache"
    assert isinstance(widget._points_controller, TiledPointsController)


def test_viewer_widget_explicit_points_backend_overrides_environment(qtbot, monkeypatch) -> None:
    monkeypatch.setenv(viewer_widget_module._EXPERIMENTAL_TILED_POINTS_ENV, "true")

    widget = ViewerWidget(experimental_tiled_points=False)
    qtbot.addWidget(widget)

    assert widget.points_visualization_backend == "in-memory"
    assert isinstance(widget._points_controller, PointsController)


def test_elided_label_only_shows_tooltip_when_text_is_truncated(qtbot, monkeypatch) -> None:
    label = _ElidedLabel("blobs_multiscale_image")

    qtbot.addWidget(label)

    class _FakeRect:
        def __init__(self, width: int) -> None:
            self._width = width

        def width(self) -> int:
            return self._width

    class _FakeFontMetrics:
        def elidedText(self, text: str, mode: object, width: int) -> str:
            del mode
            return text if width >= len(text) else "blobs_multiscale…"

    monkeypatch.setattr(label, "fontMetrics", lambda: _FakeFontMetrics())
    monkeypatch.setattr(label, "contentsRect", lambda: _FakeRect(400))
    label._update_elided_text()

    assert label.toolTip() == ""

    monkeypatch.setattr(label, "contentsRect", lambda: _FakeRect(10))
    label._update_elided_text()

    tooltip = unescape(label.toolTip()).replace("&#8203;", "").replace("\u200b", "")
    assert "blobs_multiscale_image" in tooltip
    assert "..." in label.text() or "\u2026" in label.text()


def test_elided_tool_button_only_shows_tooltip_when_text_is_truncated(qtbot, monkeypatch) -> None:
    button = _ElidedToolButton("blobs_image_long_name_blobs_image_long_name")

    qtbot.addWidget(button)

    class _FakeRect:
        def __init__(self, width: int) -> None:
            self._width = width

        def width(self) -> int:
            return self._width

    class _FakeFontMetrics:
        def elidedText(self, text: str, mode: object, width: int) -> str:
            del mode
            return text if width >= len(text) else "blobs_image..."

    monkeypatch.setattr(button, "fontMetrics", lambda: _FakeFontMetrics())
    monkeypatch.setattr(button, "contentsRect", lambda: _FakeRect(400))
    button.refresh_elision()

    assert button.toolTip() == ""

    monkeypatch.setattr(button, "contentsRect", lambda: _FakeRect(20))
    button.refresh_elision()

    tooltip = unescape(button.toolTip()).replace("&#8203;", "").replace("\u200b", "")
    assert "blobs_image_long_name_blobs_image_long_name" in tooltip
    assert "collapsed" not in tooltip
    assert "..." in button.text() or "\u2026" in button.text()


def test_viewer_disclosure_toggle_uses_compact_metrics(qtbot) -> None:
    section = _CollapsibleSectionWidget(
        title="Images",
        object_name="test_viewer_section",
        toggle_object_name="test_viewer_section_toggle",
    )

    qtbot.addWidget(section)

    assert "min-height: 26px" in section.toggle_button.styleSheet()
    assert "padding: 3px 10px" in section.toggle_button.styleSheet()


def test_overlay_color_button_uses_color_dialog_selection(qtbot, monkeypatch) -> None:
    button = OverlayColorButton("#00FFFF")
    selected_colors: list[str] = []

    qtbot.addWidget(button)
    button.color_selected.connect(selected_colors.append)

    monkeypatch.setattr(
        overlay_color_button_module.QColorDialog,
        "getColor",
        lambda *args, **kwargs: QColor("#123456"),
    )

    button.choose_color()

    assert button.current_color == "#123456"
    assert "background: #123456" in button.styleSheet()
    assert "Current color" in button.toolTip()
    assert selected_colors == ["#123456"]


def test_overlay_color_button_programmatic_gradient_is_silent_and_keeps_picker_seed(
    qtbot,
) -> None:
    button = OverlayColorButton("#00FFFF")
    selected_colors: list[str] = []

    qtbot.addWidget(button)
    button.color_selected.connect(selected_colors.append)

    button.set_colormap_preview(
        "viridis",
        ("#440154", "#31688E", "#35B779", "#FDE725"),
    )

    assert button.current_color == "#00FFFF"
    assert button.gradient_name == "viridis"
    assert "qlineargradient" in button.styleSheet()
    assert "viridis" in _tooltip_text(button)
    assert "viridis" in button.accessibleName()
    assert selected_colors == []


def test_image_card_reconstructs_overlay_row_for_replacement_binding(
    qtbot,
) -> None:
    first_layer = Image(np.zeros((8, 8)), colormap="#00FFFF")
    second_layer = Image(np.zeros((8, 8)), colormap="#FF00FF")
    stable_layer = Image(np.zeros((8, 8)), colormap="#FFFF00")
    first_binding = ImageLayerBinding(
        layer=first_layer,
        element_name="image",
        coordinate_system="global",
        sdata_id=1,
        image_display_mode="overlay",
        channel_index=0,
        channel_name="DAPI",
    )
    second_binding = ImageLayerBinding(
        layer=second_layer,
        element_name="image",
        coordinate_system="global",
        sdata_id=1,
        image_display_mode="overlay",
        channel_index=0,
        channel_name="DAPI",
    )
    stable_binding = ImageLayerBinding(
        layer=stable_layer,
        element_name="image",
        coordinate_system="global",
        sdata_id=1,
        image_display_mode="overlay",
        channel_index=1,
        channel_name="CD3",
    )
    card = _ImageCardWidget(
        image_name="image",
        channel_names=["DAPI", "CD3"],
    )

    qtbot.addWidget(card)
    card.set_loaded_image_bindings(
        stack_binding=None,
        overlay_bindings=[first_binding, stable_binding],
    )
    first_row, stable_row = card.overlay_rows

    assert first_row.visibility_button.isChecked()
    assert first_row.color_button.current_color == "#00FFFF"

    card.set_loaded_image_bindings(
        stack_binding=None,
        overlay_bindings=[first_binding, stable_binding],
    )

    current_rows = card.overlay_rows
    assert current_rows[0] is first_row
    assert current_rows[1] is stable_row

    card.set_loaded_image_bindings(
        stack_binding=None,
        overlay_bindings=[second_binding, stable_binding],
    )
    second_row, current_stable_row = card.overlay_rows

    assert second_row is not first_row
    assert second_row.binding is second_binding
    assert current_stable_row is stable_row
    assert second_row.visibility_button.isChecked()
    assert second_row.color_button.current_color == "#FF00FF"

    stale_visibility_requests: list[tuple[str, int, bool]] = []
    stale_color_requests: list[tuple[str, int, str]] = []
    stale_remove_requests: list[tuple[str, int]] = []
    card.overlay_channel_visibility_requested.connect(
        lambda image_name, channel_index, visible: stale_visibility_requests.append(
            (image_name, channel_index, visible)
        )
    )
    card.overlay_channel_color_requested.connect(
        lambda image_name, channel_index, color: stale_color_requests.append((image_name, channel_index, color))
    )
    card.overlay_channel_remove_requested.connect(
        lambda image_name, channel_index: stale_remove_requests.append((image_name, channel_index))
    )
    first_row.visibility_change_requested.emit(False)
    first_row.color_change_requested.emit("#123456")
    first_row.remove_requested.emit()

    assert stale_visibility_requests == []
    assert stale_color_requests == []
    assert stale_remove_requests == []

    first_layer.visible = False
    first_layer.colormap = "#123456"

    assert second_row.visibility_button.isChecked()
    assert second_row.color_button.current_color == "#FF00FF"

    second_layer.visible = False
    second_layer.colormap = "viridis"

    assert not second_row.visibility_button.isChecked()
    assert second_row.color_button.gradient_name == "viridis"

    card.dispose()
    second_layer.visible = True
    second_layer.colormap = "#ABCDEF"

    assert not second_row.visibility_button.isChecked()
    assert second_row.color_button.gradient_name == "viridis"


def test_image_card_reconciles_membership_atomically_and_tracks_appearances(qtbot) -> None:
    overlay_layer = Image(np.zeros((8, 8)), colormap="#00FFFF")
    overlay_binding = ImageLayerBinding(
        layer=overlay_layer,
        element_name="image",
        coordinate_system="global",
        sdata_id=1,
        image_display_mode="overlay",
        channel_index=0,
        channel_name="DAPI",
    )
    stack_binding = ImageLayerBinding(
        layer=Image(np.zeros((2, 8, 8))),
        element_name="image",
        coordinate_system="global",
        sdata_id=1,
        image_display_mode="stack",
    )
    card = _ImageCardWidget(
        image_name="image",
        channel_names=["DAPI"],
    )
    qtbot.addWidget(card)

    card.set_loaded_image_bindings(
        stack_binding=None,
        overlay_bindings=[overlay_binding],
    )
    overlay_row = card.overlay_rows[0]
    assert card.overlay_toggle.isChecked()

    with pytest.raises(ValueError, match="stack binding"):
        card.set_loaded_image_bindings(
            stack_binding=overlay_binding,
            overlay_bindings=(),
        )
    with pytest.raises(ValueError, match="Every overlay binding"):
        card.set_loaded_image_bindings(
            stack_binding=None,
            overlay_bindings=[stack_binding],
        )
    with pytest.raises(ValueError, match="both stack and overlay"):
        card.set_loaded_image_bindings(
            stack_binding=stack_binding,
            overlay_bindings=[overlay_binding],
        )

    assert card.loaded_stack_binding is None
    assert card.overlay_rows == [overlay_row]

    card.set_loaded_image_bindings(
        stack_binding=None,
        overlay_bindings=(),
    )
    assert card.overlay_toggle.isChecked()

    card.set_loaded_image_bindings(
        stack_binding=stack_binding,
        overlay_bindings=(),
    )
    assert card.loaded_stack_binding is stack_binding
    assert card.stack_toggle.isChecked()

    first_stack_row = card.stack_row
    assert first_stack_row is not None
    replacement_stack_binding = ImageLayerBinding(
        layer=Image(np.zeros((2, 8, 8))),
        element_name="image",
        coordinate_system="global",
        sdata_id=1,
        image_display_mode="stack",
    )
    stale_stack_requests: list[tuple[str, bool]] = []
    card.stack_visibility_requested.connect(
        lambda image_name, visible: stale_stack_requests.append((image_name, visible))
    )
    card.set_loaded_image_bindings(
        stack_binding=replacement_stack_binding,
        overlay_bindings=(),
    )

    assert card.stack_row is not first_stack_row
    assert card.loaded_stack_binding is replacement_stack_binding

    first_stack_row.visibility_change_requested.emit(False)

    assert stale_stack_requests == []

    card.set_loaded_image_bindings(
        stack_binding=None,
        overlay_bindings=(),
    )
    assert card.loaded_stack_binding is None
    assert card.stack_toggle.isChecked()


def test_viewer_widget_refreshes_cards_when_shared_sdata_changes(qtbot, sdata_blobs) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)

    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata_blobs)

    assert widget.app_state.sdata is sdata_blobs
    assert widget.empty_state_label.isHidden()
    assert widget.coordinate_system_combo.count() == 1
    assert widget.coordinate_system_combo.itemText(0) == "global"
    assert len(widget.image_cards) == 2
    assert len(widget.labels_cards) == 2
    assert len(widget.shape_cards) == 3
    assert [card.image_name for card in widget.image_cards] == ["blobs_image", "blobs_multiscale_image"]
    assert [card.labels_name for card in widget.labels_cards] == ["blobs_labels", "blobs_multiscale_labels"]
    assert [card.shapes_name for card in widget.shape_cards] == [
        "blobs_circles",
        "blobs_multipolygons",
        "blobs_polygons",
    ]
    assert widget.image_cards[0].channel_names == ["0", "1", "2"]
    assert widget.image_cards[0].overlay_toggle.text() == "overlay"
    assert widget.image_cards[0].overlay_toggle.isChecked()
    assert widget.image_cards[0].stack_toggle.text() == "stack"
    assert not widget.image_cards[0].stack_toggle.isChecked()
    assert widget.image_cards[0].available_channel_names == ("0", "1", "2")
    assert widget.image_cards[0].loaded_overlay_channel_indices == ()
    assert widget.image_cards[0].selected_count_label.text() == "0 channels"
    assert not widget.image_cards[0].no_selected_channels_label.isHidden()
    assert len(widget.image_rows) == 2
    assert len(widget.labels_rows) == 2
    assert len(widget.shape_rows) == 3
    assert widget.images_section_toggle.text() == "Images (2)"
    assert widget.labels_section_toggle.text() == "Labels (2)"
    assert widget.shapes_section_toggle.text() == "Shapes (3)"
    assert not widget.images_group.is_expanded()
    assert not widget.labels_group.is_expanded()
    assert not widget.shapes_group.is_expanded()
    assert widget.image_rows[0].detail_widget.isHidden()
    assert widget.labels_rows[0].detail_widget.isHidden()
    assert widget.shape_rows[0].detail_widget.isHidden()
    assert widget.shape_cards[0].action_hint_label.text() == "Action: add/update primary shapes layer"
    assert widget.shape_cards[0].add_update_button.isEnabled()
    assert widget.labels_cards[0].linked_table_combo.count() == 1
    assert widget.labels_cards[0].linked_table_combo.itemText(0) == "table"
    assert widget.labels_cards[0].linked_table_combo.sizeAdjustPolicy() == (
        QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
    )
    assert widget.labels_cards[1].linked_table_combo.count() == 1
    assert widget.labels_cards[1].linked_table_combo.itemText(0) == "No linked tables"
    assert not widget.labels_cards[1].linked_table_combo.isEnabled()
    assert "Current View" in _label_text(widget.summary_label)
    assert '"global":' in _label_text(widget.summary_label)
    assert widget.summary_label.toolTip() == ""


def test_viewer_widget_summary_card_shortens_long_coordinate_system(
    qtbot,
    monkeypatch,
    sdata_blobs,
) -> None:
    coordinate_system = "global_long_coordinate_system_name_" + "x" * 80
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    qtbot.addWidget(widget)
    _patch_coordinate_system_names(monkeypatch, [coordinate_system])
    monkeypatch.setattr(viewer_widget_module, "_get_images_in_coordinate_system", lambda sdata, coordinate_system: [])
    monkeypatch.setattr(viewer_widget_module, "_get_labels_in_coordinate_system", lambda sdata, coordinate_system: [])
    monkeypatch.setattr(viewer_widget_module, "_get_shapes_in_coordinate_system", lambda sdata, coordinate_system: [])
    monkeypatch.setattr(viewer_widget_module, "_get_points_in_coordinate_system", lambda sdata, coordinate_system: [])

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata_blobs)

    summary = _label_text(widget.summary_label)
    assert "Current View" in summary
    assert coordinate_system not in summary
    assert "…" in summary
    assert "0 image element(s)" in summary
    tooltip = _tooltip_text(widget.summary_label)
    assert coordinate_system in tooltip
    assert 'In coordinate system "' in tooltip


def test_viewer_widget_points_section_populates_and_starts_value_loading(qtbot, monkeypatch) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    fake_sdata = _make_points_sdata()
    load_value_calls = 0

    qtbot.addWidget(widget)

    _patch_coordinate_system_names(monkeypatch, ["global"])
    monkeypatch.setattr(viewer_widget_module, "_get_images_in_coordinate_system", lambda sdata, coordinate_system: [])
    monkeypatch.setattr(viewer_widget_module, "_get_labels_in_coordinate_system", lambda sdata, coordinate_system: [])
    monkeypatch.setattr(viewer_widget_module, "_get_shapes_in_coordinate_system", lambda sdata, coordinate_system: [])
    monkeypatch.setattr(
        viewer_widget_module, "_get_points_in_coordinate_system", lambda sdata, coordinate_system: ["transcripts"]
    )

    def record_value_loading() -> bool:
        nonlocal load_value_calls
        load_value_calls += 1
        return True

    monkeypatch.setattr(widget._points_controller, "load_value_source", record_value_loading)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(fake_sdata)

    assert widget.points_section_toggle.text() == "Points (1)"
    assert widget.points_empty_label.isHidden()
    assert not widget.points_widget.isHidden()
    assert widget.points_widget.selected_points_name() == "transcripts"
    assert widget.points_widget.selected_index_column() == "gene"
    assert [
        widget.points_widget.index_column_combo.itemText(index)
        for index in range(widget.points_widget.index_column_combo.count())
    ] == ["gene", "target"]
    assert load_value_calls == 1


def test_viewer_widget_opt_in_tiled_points_starts_cache_descriptor_loading(qtbot, monkeypatch) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer, experimental_tiled_points=True)
    fake_sdata = _make_points_sdata()
    bound_sources: list[tuple[object, str | None, str | None, str | None]] = []

    qtbot.addWidget(widget)

    _patch_coordinate_system_names(monkeypatch, ["global"])
    monkeypatch.setattr(viewer_widget_module, "_get_images_in_coordinate_system", lambda sdata, coordinate_system: [])
    monkeypatch.setattr(viewer_widget_module, "_get_labels_in_coordinate_system", lambda sdata, coordinate_system: [])
    monkeypatch.setattr(viewer_widget_module, "_get_shapes_in_coordinate_system", lambda sdata, coordinate_system: [])
    monkeypatch.setattr(
        viewer_widget_module, "_get_points_in_coordinate_system", lambda sdata, coordinate_system: ["transcripts"]
    )

    def record_binding(sdata, points_name, coordinate_system, value_column) -> bool:
        bound_sources.append((sdata, points_name, coordinate_system, value_column))
        return True

    monkeypatch.setattr(widget._points_controller, "bind_source", record_binding)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(fake_sdata)

    assert bound_sources[-1] == (fake_sdata, "transcripts", "global", "gene")


def test_viewer_widget_points_add_update_request_calls_controller(qtbot, monkeypatch) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    recorded_requests: list[tuple[object, int]] = []

    qtbot.addWidget(widget)

    monkeypatch.setattr(
        widget._points_controller,
        "load_selection",
        lambda values, *, render_point_budget, random_state=42: (
            recorded_requests.append((values, render_point_budget)) or True
        ),
    )
    widget.points_widget.set_points_names(["transcripts"])
    widget.points_widget.set_index_columns(["gene"])
    widget.points_widget.set_value_source(
        SimpleNamespace(value_table=SimpleNamespace(values=pd.DataFrame({"value": ["AAMP", "AXL"]})))
    )
    widget.points_widget.render_controller_state(
        SimpleNamespace(
            can_load_values=True,
            can_visualize=True,
            is_loading=False,
            is_loading_values=False,
            status_message="Points: ready.",
            status_kind="success",
        )
    )
    widget.points_widget.value_input.setText("AAMP")
    widget.points_widget.add_value_button.click()
    widget.points_widget.value_input.setText("AXL")
    widget.points_widget.add_value_button.click()
    widget.points_widget.render_point_budget_input.setText("50_000")

    widget.points_widget.add_update_button.click()

    assert recorded_requests == [(("AAMP", "AXL"), 50_000)]


def test_viewer_widget_opt_in_tiled_points_add_update_calls_cache_controller(qtbot, monkeypatch) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer, experimental_tiled_points=True)
    recorded_requests: list[tuple[object, int]] = []

    qtbot.addWidget(widget)

    monkeypatch.setattr(
        widget._points_controller,
        "apply_selection",
        lambda values, *, render_point_budget: recorded_requests.append((values, render_point_budget)) or True,
    )
    widget.points_widget.set_points_names(["transcripts"])
    widget.points_widget.set_index_columns(["gene"])
    widget.points_widget.set_value_source(
        SimpleNamespace(value_table=SimpleNamespace(values=pd.DataFrame({"value": ["AAMP", "AXL"]})))
    )
    widget.points_widget.render_controller_state(
        SimpleNamespace(
            can_load_values=True,
            can_visualize=True,
            is_loading=False,
            is_loading_values=False,
            status_message="Points: ready.",
            status_kind="success",
        )
    )
    widget.points_widget.value_input.setText("AAMP")
    widget.points_widget.add_value_button.click()
    widget.points_widget.value_input.setText("AXL")
    widget.points_widget.add_value_button.click()
    widget.points_widget.render_point_budget_input.setText("50_000")

    widget.points_widget.add_update_button.click()

    assert recorded_requests == [(("AAMP", "AXL"), 50_000)]


def test_viewer_widget_on_points_loaded_applies_layer_and_status(qtbot) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    fake_sdata = object()
    load_request = _make_points_load_request(fake_sdata)

    qtbot.addWidget(widget)

    widget._on_points_loaded(load_request)

    assert len(viewer.layers) == 1
    layer = viewer.layers[0]
    assert layer.name == "transcripts: gene=AAMP"
    assert viewer.layers.selection.active is layer
    assert "Points Layer Created" in widget.global_action_feedback_label.text()
    assert "2 point" in widget.global_action_feedback_label.text()
    assert not widget.global_action_feedback_label.isHidden()


def test_viewer_widget_opt_in_cache_values_populate_existing_points_panel(qtbot) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer, experimental_tiled_points=True)

    qtbot.addWidget(widget)

    widget._on_points_values_loaded(("AAMP", "AXL"))

    assert widget.points_widget._value_completer_model.stringList() == ["AAMP", "AXL"]
    assert len(viewer.layers) == 0


def test_viewer_widget_progressive_disclosure_expands_sections_and_elements(qtbot, sdata_blobs) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)

    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata_blobs)

    first_image_row = widget.image_rows[0]
    second_image_row = widget.image_rows[1]
    first_labels_row = widget.labels_rows[0]
    first_shape_row = widget.shape_rows[0]

    assert widget.images_group.content_widget.isHidden()
    assert widget.labels_group.content_widget.isHidden()
    assert widget.shapes_group.content_widget.isHidden()
    assert first_image_row.detail_widget.isHidden()
    assert first_labels_row.detail_widget.isHidden()
    assert first_shape_row.detail_widget.isHidden()
    assert widget.images_section_toggle.arrowType() == Qt.ArrowType.NoArrow
    assert not widget.images_section_toggle.icon().isNull()

    widget.images_section_toggle.click()

    assert widget.images_group.is_expanded()
    assert not widget.images_group.content_widget.isHidden()
    assert first_image_row.detail_widget.isHidden()

    first_image_row.toggle_button.click()

    assert first_image_row.is_expanded()
    assert not first_image_row.detail_widget.isHidden()
    assert widget.image_cards[0].overlay_toggle.isChecked()

    second_image_row.toggle_button.click()

    assert first_image_row.is_expanded()
    assert not first_image_row.detail_widget.isHidden()
    assert second_image_row.is_expanded()
    assert not second_image_row.detail_widget.isHidden()

    widget.labels_section_toggle.click()
    first_labels_row.toggle_button.click()

    assert widget.labels_group.is_expanded()
    assert first_labels_row.is_expanded()
    assert not first_labels_row.detail_widget.isHidden()
    assert widget.labels_cards[0].linked_table_combo.currentText() == "table"

    widget.shapes_section_toggle.click()
    first_shape_row.toggle_button.click()

    assert widget.shapes_group.is_expanded()
    assert first_shape_row.is_expanded()
    assert not first_shape_row.detail_widget.isHidden()
    assert widget.shape_cards[0].add_update_button.isEnabled()


def test_viewer_widget_expanded_detail_panels_fit_current_minimum_width(qtbot, monkeypatch, sdata_blobs) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)

    qtbot.addWidget(widget)

    monkeypatch.setattr(widget._points_controller, "load_value_source", lambda: None)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata_blobs)

    scrollbar_width = widget.scroll_area.verticalScrollBar().sizeHint().width()
    content_margins = widget.content_layout.contentsMargins()
    content_width = WIDGET_MIN_WIDTH - scrollbar_width - content_margins.left() - content_margins.right()

    for group, row, card in (
        (widget.images_group, widget.image_rows[0], widget.image_cards[0]),
        (widget.labels_group, widget.labels_rows[0], widget.labels_cards[0]),
        (widget.shapes_group, widget.shape_rows[0], widget.shape_cards[0]),
    ):
        group_margins = group.layout().contentsMargins()
        row_margins = row.layout().contentsMargins()
        available_detail_width = (
            content_width - group_margins.left() - group_margins.right() - row_margins.left() - row_margins.right()
        )

        assert card.minimumSizeHint().width() <= available_detail_width
        assert card.sizeHint().width() <= available_detail_width

    points_group_margins = widget.points_group.layout().contentsMargins()
    available_points_width = content_width - points_group_margins.left() - points_group_margins.right()

    assert widget.points_widget.minimumSizeHint().width() <= available_points_width
    assert widget.points_widget.sizeHint().width() <= available_points_width


def test_viewer_widget_progressive_disclosure_actions_still_load_layers(qtbot, sdata_blobs) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)

    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata_blobs)

    widget.images_section_toggle.click()
    widget.image_rows[0].toggle_button.click()
    widget.image_cards[0].stack_toggle.setChecked(True)
    widget.image_cards[0].stack_load_button.click()

    assert len(viewer.layers) == 1
    assert viewer.layers[0].name == "blobs_image"

    widget.labels_section_toggle.click()
    widget.labels_rows[0].toggle_button.click()
    widget.labels_cards[0].add_update_button.click()

    assert len(viewer.layers) == 2
    assert viewer.layers[1].name == "blobs_labels"

    widget.shapes_section_toggle.click()
    widget.shape_rows[0].toggle_button.click()
    widget.shape_cards[0].add_update_button.click()

    assert len(viewer.layers) == 3
    assert viewer.layers[2].name == "blobs_circles"


def test_viewer_widget_shapes_empty_state_appears_when_no_shapes(qtbot, monkeypatch) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    fake_sdata = object()

    qtbot.addWidget(widget)

    _patch_coordinate_system_names(monkeypatch, ["global"])
    monkeypatch.setattr(viewer_widget_module, "_get_images_in_coordinate_system", lambda sdata, coordinate_system: [])
    monkeypatch.setattr(viewer_widget_module, "_get_labels_in_coordinate_system", lambda sdata, coordinate_system: [])
    monkeypatch.setattr(viewer_widget_module, "_get_shapes_in_coordinate_system", lambda sdata, coordinate_system: [])

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(fake_sdata)

    assert widget.shapes_section_toggle.text() == "Shapes (0)"
    assert not widget.shapes_empty_label.isHidden()
    assert widget.shapes_section.isHidden()
    assert widget.shape_cards == []
    assert widget.shape_rows == []


def test_viewer_widget_preserves_expanded_shape_rows_across_refreshes(qtbot, sdata_blobs) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)

    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata_blobs)

    widget.shapes_section_toggle.click()
    widget.shape_rows[0].toggle_button.click()
    expanded_shapes_name = widget.shape_cards[0].shapes_name

    widget.refresh_from_sdata(sdata_blobs)

    refreshed_row = widget.shape_rows[0]
    refreshed_card = widget.shape_cards[0]
    assert refreshed_card.shapes_name == expanded_shapes_name
    assert refreshed_row.is_expanded()
    assert not refreshed_row.detail_widget.isHidden()


def test_viewer_widget_labels_cards_expose_table_driven_coloring_controls(qtbot, sdata_blobs) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)

    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata_blobs)

    first_card = widget.labels_cards[0]
    second_card = widget.labels_cards[1]

    assert first_card.color_source_kind_combo.count() == 3
    assert [first_card.color_source_kind_combo.itemText(index) for index in range(3)] == [
        "No color source",
        "Observations",
        "Vars",
    ]
    assert first_card.color_source_kind_combo.isEnabled()
    assert first_card.color_source_value_input.completer() is not None
    assert not first_card.color_source_value_input.isEnabled()
    assert first_card.action_hint_label.text() == "Action: add/update primary labels layer"

    _select_color_source_kind(first_card, "obs_column")
    assert first_card.color_source_value_input.isEnabled()
    assert first_card.color_source_value_input.text() == ""
    assert first_card.color_source_value_input.placeholderText() == "Select obs column"
    assert first_card.color_source_value_input.completer().model().stringList() == ["instance_id"]
    assert first_card.color_source_value_input.completer().completionMode() == QCompleter.CompletionMode.PopupCompletion
    assert first_card.color_source_value_input.completer().maxVisibleItems() == 10
    assert first_card.selected_color_source is None
    assert first_card.action_hint_label.text() == "Action: select an observation column for a colored overlay"

    _select_color_source_kind(first_card, "x_var")
    assert first_card.color_source_value_input.isEnabled()
    assert first_card.color_source_value_input.text() == ""
    assert first_card.color_source_value_input.placeholderText() == "Select var"
    assert first_card.color_source_value_input.completer().model().stringList() == [
        "channel_0_sum",
        "channel_1_sum",
        "channel_2_sum",
    ]
    first_card.color_source_value_input.show_completion_popup()
    assert first_card.color_source_value_input.completer().completionPrefix() == ""
    assert first_card.color_source_value_input.completer().completionModel().rowCount() == 3
    first_card.color_source_value_input.completer().popup().hide()
    assert first_card.selected_color_source is None
    assert first_card.action_hint_label.text() == "Action: select a var for a colored overlay"

    first_card.color_source_value_input.setText("channel_1_sum")
    assert first_card.action_hint_label.text() == 'Action: add/update colored overlay for X[:, "channel_1_sum"]'

    assert _combo_texts(second_card.color_source_kind_combo) == ["No color source"]
    assert not second_card.color_source_kind_combo.isEnabled()
    assert second_card.color_source_kind_combo.findData("x_var") == -1
    assert second_card.selected_source_kind is None
    assert second_card.selected_color_source is None
    assert second_card.action_hint_label.text() == "Action: add/update primary labels layer"


def test_viewer_widget_labels_card_repopulates_color_sources_when_linked_table_changes(qtbot, monkeypatch) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    fake_sdata = object()

    qtbot.addWidget(widget)

    _patch_coordinate_system_names(monkeypatch, ["global"])
    monkeypatch.setattr(viewer_widget_module, "_get_images_in_coordinate_system", lambda sdata, coordinate_system: [])
    monkeypatch.setattr(
        viewer_widget_module, "_get_labels_in_coordinate_system", lambda sdata, coordinate_system: ["labels"]
    )
    monkeypatch.setattr(
        viewer_widget_module, "get_annotating_table_names", lambda sdata, labels_name: ["table_a", "table_b"]
    )
    monkeypatch.setattr(
        viewer_widget_module,
        "get_table_color_source_options",
        lambda sdata, table_name: (
            [
                TableColorSourceSpec(
                    table_name=table_name, source_kind="obs_column", value_key="cell_type", value_kind="categorical"
                )
            ]
            if table_name == "table_a"
            else [
                TableColorSourceSpec(
                    table_name=table_name, source_kind="x_var", value_key="GeneA", value_kind="continuous"
                )
            ]
        ),
    )

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(fake_sdata)

    card = widget.labels_cards[0]

    assert _combo_texts(card.color_source_kind_combo) == ["No color source", "Observations"]
    assert card.color_source_kind_combo.findData("x_var") == -1

    _select_color_source_kind(card, "obs_column")
    assert card.color_source_value_input.isEnabled()
    assert card.color_source_value_input.completer().model().stringList() == ["cell_type"]
    assert card.color_source_value_input.text() == ""
    assert card.color_source_value_input.placeholderText() == "Select obs column"
    assert card.selected_color_source is None
    assert card.action_hint_label.text() == "Action: select an observation column for a colored overlay"

    card.linked_table_combo.setCurrentIndex(1)
    assert _combo_texts(card.color_source_kind_combo) == ["No color source", "Vars"]
    assert card.color_source_kind_combo.findData("obs_column") == -1
    assert card.selected_source_kind is None
    assert not card.color_source_value_input.isEnabled()
    assert card.action_hint_label.text() == "Action: add/update primary labels layer"

    _select_color_source_kind(card, "x_var")
    assert card.color_source_value_input.isEnabled()
    assert card.color_source_value_input.completer().model().stringList() == ["GeneA"]
    assert card.color_source_value_input.text() == ""
    assert card.color_source_value_input.placeholderText() == "Select var"
    assert card.selected_color_source is None
    assert card.action_hint_label.text() == "Action: select a var for a colored overlay"

    card.color_source_value_input.setText("GeneA")
    assert card.action_hint_label.text() == 'Action: add/update colored overlay for X[:, "GeneA"]'


def test_viewer_widget_ignores_non_feature_matrix_write_events(qtbot, monkeypatch) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    fake_sdata = object()
    table_names_by_label = {"labels": ["table"]}

    qtbot.addWidget(widget)

    _patch_coordinate_system_names(monkeypatch, ["global"])
    _patch_viewer_widget_labels_tables(
        monkeypatch,
        labels_names=["labels"],
        table_names_by_label=table_names_by_label,
    )

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(fake_sdata)

    card = widget.labels_cards[0]
    table_names_by_label["labels"] = ["table", "new_table"]

    widget._on_table_state_changed(object())

    assert _combo_texts(card.linked_table_combo) == ["table"]
    assert card.selected_table_name == "table"


def test_viewer_widget_ignores_feature_matrix_writes_for_other_sdata(qtbot, monkeypatch) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    fake_sdata = object()
    other_sdata = object()
    table_names_by_label = {"labels": ["table"]}

    qtbot.addWidget(widget)

    _patch_coordinate_system_names(monkeypatch, ["global"])
    _patch_viewer_widget_labels_tables(
        monkeypatch,
        labels_names=["labels"],
        table_names_by_label=table_names_by_label,
    )

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(fake_sdata)

    card = widget.labels_cards[0]
    table_names_by_label["labels"] = ["table", "new_table"]

    widget.app_state.record_table_mutation(
        _table_event(
            other_sdata,
            TableComponentPath("obsm", ("features",)),
            table_name="new_table",
            source="feature_extraction",
        )
    )

    assert _combo_texts(card.linked_table_combo) == ["table"]
    assert card.selected_table_name == "table"


def test_viewer_widget_refreshes_labels_card_linked_tables_from_feature_matrix_event(qtbot, monkeypatch) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    fake_sdata = object()
    table_names_by_label = {"labels": ["table"]}

    qtbot.addWidget(widget)

    _patch_coordinate_system_names(monkeypatch, ["global"])
    _patch_viewer_widget_labels_tables(
        monkeypatch,
        labels_names=["labels"],
        table_names_by_label=table_names_by_label,
    )

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(fake_sdata)

    card = widget.labels_cards[0]
    row = widget.labels_rows[0]
    row.set_expanded(True)
    table_names_by_label["labels"] = ["new_table", "table"]

    widget.app_state.record_table_mutation(
        _table_event(
            fake_sdata,
            TableComponentPath("obsm", ("features",)),
            table_name="new_table",
            source="feature_extraction",
        )
    )

    assert _combo_texts(card.linked_table_combo) == ["new_table", "table"]
    assert card.selected_table_name == "table"
    assert row.is_expanded()
    assert len(viewer.layers) == 0


def test_viewer_widget_selects_first_linked_table_when_event_creates_first_table(qtbot, monkeypatch) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    fake_sdata = object()
    table_names_by_label = {"labels": []}
    color_sources_by_table: dict[str, list[TableColorSourceSpec]] = {}

    qtbot.addWidget(widget)

    _patch_coordinate_system_names(monkeypatch, ["global"])
    _patch_viewer_widget_labels_tables(
        monkeypatch,
        labels_names=["labels"],
        table_names_by_label=table_names_by_label,
        color_sources_by_table=color_sources_by_table,
    )

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(fake_sdata)

    card = widget.labels_cards[0]
    assert _combo_texts(card.linked_table_combo) == ["No linked tables"]
    assert card.selected_table_name is None
    assert not card.linked_table_combo.isEnabled()
    assert _combo_texts(card.color_source_kind_combo) == ["No color source"]
    assert not card.color_source_kind_combo.isEnabled()
    assert card.selected_source_kind is None

    table_names_by_label["labels"] = ["new_table"]
    color_sources_by_table["new_table"] = [
        TableColorSourceSpec(
            table_name="new_table",
            source_kind="obs_column",
            value_key="cell_type",
            value_kind="categorical",
        )
    ]
    widget.app_state.record_table_mutation(
        _table_event(
            fake_sdata,
            TableComponentPath("obsm", ("features",)),
            table_name="new_table",
            source="feature_extraction",
        )
    )

    assert _combo_texts(card.linked_table_combo) == ["new_table"]
    assert card.linked_table_combo.isEnabled()
    assert card.selected_table_name == "new_table"
    assert _combo_texts(card.color_source_kind_combo) == ["No color source", "Observations"]
    assert card.color_source_kind_combo.isEnabled()
    assert card.selected_source_kind is None
    assert len(viewer.layers) == 0


def test_viewer_widget_preserves_labels_card_color_source_selection_after_event(qtbot, monkeypatch) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    fake_sdata = object()
    table_names_by_label = {"labels": ["table"]}
    color_sources_by_table = {
        "table": [
            TableColorSourceSpec(
                table_name="table",
                source_kind="obs_column",
                value_key="cell_type",
                value_kind="categorical",
            )
        ]
    }

    qtbot.addWidget(widget)

    _patch_coordinate_system_names(monkeypatch, ["global"])
    _patch_viewer_widget_labels_tables(
        monkeypatch,
        labels_names=["labels"],
        table_names_by_label=table_names_by_label,
        color_sources_by_table=color_sources_by_table,
    )

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(fake_sdata)

    card = widget.labels_cards[0]
    _select_color_source_kind(card, "obs_column")
    card.color_source_value_input.setText("cell_type")
    assert card.selected_color_source == color_sources_by_table["table"][0]

    table_names_by_label["labels"] = ["new_table", "table"]
    color_sources_by_table["new_table"] = [
        TableColorSourceSpec(
            table_name="new_table",
            source_kind="obs_column",
            value_key="other_type",
            value_kind="categorical",
        )
    ]
    widget.app_state.record_table_mutation(
        _table_event(
            fake_sdata,
            TableComponentPath("obsm", ("features",)),
            table_name="new_table",
            source="feature_extraction",
        )
    )

    assert card.selected_table_name == "table"
    assert card.selected_source_kind == "obs_column"
    assert card.selected_color_source == color_sources_by_table["table"][0]
    assert card.action_hint_label.text() == 'Action: add/update colored overlay for obs["cell_type"]'


def test_viewer_widget_refreshes_table_color_sources_when_user_class_is_created(qtbot, monkeypatch) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    fake_sdata = object()
    table_names_by_label = {"labels": ["table"]}
    color_sources_by_table = {
        "table": [
            TableColorSourceSpec(
                table_name="table",
                source_kind="obs_column",
                value_key="cell_type",
                value_kind="categorical",
            )
        ]
    }

    qtbot.addWidget(widget)

    _patch_coordinate_system_names(monkeypatch, ["global"])
    _patch_viewer_widget_labels_tables(
        monkeypatch,
        labels_names=["labels"],
        table_names_by_label=table_names_by_label,
        color_sources_by_table=color_sources_by_table,
    )

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(fake_sdata)

    card = widget.labels_cards[0]
    _select_color_source_kind(card, "obs_column")
    assert card._color_source_completer_model.stringList() == ["cell_type"]

    color_sources_by_table["table"] = [
        *color_sources_by_table["table"],
        TableColorSourceSpec(
            table_name="table",
            source_kind="obs_column",
            value_key="user_class",
            value_kind="categorical",
        ),
    ]

    widget.app_state.record_table_mutation(
        _table_event(
            fake_sdata,
            TableComponentPath("obs", ("user_class",)),
            TableComponentPath("uns", ("user_class_colors",)),
            source="object_classification_annotation",
            change_kind="created",
        )
    )

    assert card._color_source_completer_model.stringList() == ["cell_type", "user_class"]
    assert card.selected_table_name == "table"
    assert card.selected_source_kind == "obs_column"
    assert len(viewer.layers) == 0


def test_viewer_widget_skips_linked_table_refresh_for_updated_user_class_annotation(qtbot, monkeypatch) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    fake_sdata = object()
    refresh_calls: list[str] = []
    emitted_events: list[object] = []

    qtbot.addWidget(widget)
    widget.app_state.sdata = fake_sdata
    monkeypatch.setattr(
        widget,
        "_refresh_labels_card_linked_tables",
        lambda: refresh_calls.append("labels"),
    )
    monkeypatch.setattr(
        widget,
        "_refresh_shapes_card_linked_tables",
        lambda: refresh_calls.append("shapes"),
    )
    widget.app_state.table_state_changed.connect(emitted_events.append)

    event = _table_event(
        fake_sdata,
        TableComponentPath("obs", ("user_class",)),
        TableComponentPath("uns", ("user_class_colors",)),
        source="object_classification_annotation",
    )
    widget.app_state.record_table_mutation(event)

    assert emitted_events == [event]
    assert widget.app_state.is_table_dirty(fake_sdata, "table")
    assert refresh_calls == []


def test_viewer_widget_ignores_classification_table_events_for_other_sdata(qtbot, monkeypatch) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    fake_sdata = object()
    other_sdata = object()
    table_names_by_label = {"labels": ["table"]}
    color_sources_by_table = {
        "table": [
            TableColorSourceSpec(
                table_name="table",
                source_kind="obs_column",
                value_key="cell_type",
                value_kind="categorical",
            )
        ]
    }

    qtbot.addWidget(widget)

    _patch_coordinate_system_names(monkeypatch, ["global"])
    _patch_viewer_widget_labels_tables(
        monkeypatch,
        labels_names=["labels"],
        table_names_by_label=table_names_by_label,
        color_sources_by_table=color_sources_by_table,
    )

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(fake_sdata)

    card = widget.labels_cards[0]
    _select_color_source_kind(card, "obs_column")
    color_sources_by_table["table"] = [
        *color_sources_by_table["table"],
        TableColorSourceSpec(
            table_name="table",
            source_kind="obs_column",
            value_key="user_class",
            value_kind="categorical",
        ),
    ]

    widget.app_state.record_table_mutation(
        _table_event(
            other_sdata,
            TableComponentPath("obs", ("user_class",)),
            source="object_classification",
        )
    )

    assert card._color_source_completer_model.stringList() == ["cell_type"]


@pytest.mark.parametrize(
    ("event_type", "emitter_name"),
    [
        (ShapesElementWrittenEvent, "emit_shapes_element_written"),
        (ShapesElementReloadedEvent, "emit_shapes_element_reloaded"),
    ],
)
def test_viewer_widget_refreshes_only_shapes_section_from_shapes_element_event(
    qtbot,
    monkeypatch,
    event_type,
    emitter_name,
) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    fake_sdata = object()
    names = {
        "images": ["image"],
        "labels": ["labels"],
        "shapes": ["shape_a"],
        "points": ["points"],
    }

    qtbot.addWidget(widget)

    _patch_coordinate_system_names(monkeypatch, ["global"])
    monkeypatch.setattr(
        viewer_widget_module, "_get_images_in_coordinate_system", lambda sdata, coordinate_system: names["images"]
    )
    monkeypatch.setattr(
        viewer_widget_module, "_get_labels_in_coordinate_system", lambda sdata, coordinate_system: names["labels"]
    )
    monkeypatch.setattr(
        viewer_widget_module, "_get_shapes_in_coordinate_system", lambda sdata, coordinate_system: names["shapes"]
    )
    monkeypatch.setattr(
        viewer_widget_module, "_get_points_in_coordinate_system", lambda sdata, coordinate_system: names["points"]
    )
    monkeypatch.setattr(viewer_widget_module, "get_image_channel_names_from_sdata", lambda sdata, image_name: ["c0"])
    monkeypatch.setattr(viewer_widget_module, "get_annotating_table_names", lambda sdata, element_name: [])
    monkeypatch.setattr(viewer_widget_module, "get_table_color_source_options", lambda sdata, table_name: [])
    monkeypatch.setattr(viewer_widget_module, "get_shape_column_color_source_options", lambda sdata, shapes_name: [])

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(fake_sdata)

    assert [card.shapes_name for card in widget.shape_cards] == ["shape_a"]
    image_rows = widget.image_rows
    labels_rows = widget.labels_rows

    def fail_if_rebuilt(*args, **kwargs):
        del args, kwargs
        raise AssertionError("non-shapes sections should not be rebuilt")

    monkeypatch.setattr(widget, "_rebuild_image_cards", fail_if_rebuilt)
    monkeypatch.setattr(widget, "_rebuild_labels_cards", fail_if_rebuilt)
    monkeypatch.setattr(widget, "_refresh_points_section", fail_if_rebuilt)
    names["shapes"] = ["shape_a", "new_regions"]

    getattr(widget.app_state, emitter_name)(
        event_type(
            sdata=fake_sdata,
            shapes_name="new_regions",
            coordinate_system="global",
        )
    )

    assert [card.shapes_name for card in widget.shape_cards] == ["shape_a", "new_regions"]
    assert widget.image_rows == image_rows
    assert widget.labels_rows == labels_rows
    assert widget.shapes_section_title.full_text() == "Shapes (2)"
    assert widget.shapes_empty_label.isHidden()
    assert not widget.shapes_section.isHidden()
    assert "2 shapes element(s)" in widget.summary_label.text()
    assert len(viewer.layers) == 0


def test_viewer_widget_ignores_shapes_element_events_for_other_sdata_or_coordinate_system(
    qtbot,
    monkeypatch,
) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    fake_sdata = object()
    other_sdata = object()
    shapes_names = ["shape_a"]

    qtbot.addWidget(widget)

    _patch_coordinate_system_names(monkeypatch, ["global", "local"])
    monkeypatch.setattr(viewer_widget_module, "_get_images_in_coordinate_system", lambda sdata, coordinate_system: [])
    monkeypatch.setattr(viewer_widget_module, "_get_labels_in_coordinate_system", lambda sdata, coordinate_system: [])
    monkeypatch.setattr(
        viewer_widget_module,
        "_get_shapes_in_coordinate_system",
        lambda sdata, coordinate_system: list(shapes_names),
    )
    monkeypatch.setattr(viewer_widget_module, "_get_points_in_coordinate_system", lambda sdata, coordinate_system: [])
    monkeypatch.setattr(viewer_widget_module, "get_annotating_table_names", lambda sdata, element_name: [])
    monkeypatch.setattr(viewer_widget_module, "get_table_color_source_options", lambda sdata, table_name: [])
    monkeypatch.setattr(viewer_widget_module, "get_shape_column_color_source_options", lambda sdata, shapes_name: [])

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(fake_sdata)

    assert widget.app_state.coordinate_system == "global"
    assert [card.shapes_name for card in widget.shape_cards] == ["shape_a"]

    shapes_names[:] = ["shape_a", "new_regions"]
    widget.app_state.emit_shapes_element_written(
        ShapesElementWrittenEvent(
            sdata=other_sdata,
            shapes_name="new_regions",
            coordinate_system="global",
        )
    )
    widget.app_state.emit_shapes_element_written(
        ShapesElementWrittenEvent(
            sdata=fake_sdata,
            shapes_name="new_regions",
            coordinate_system="local",
        )
    )

    assert [card.shapes_name for card in widget.shape_cards] == ["shape_a"]
    assert widget.shapes_section_title.full_text() == "Shapes (1)"
    assert "1 shapes element(s)" in widget.summary_label.text()


def test_viewer_widget_image_mode_checkboxes_are_mutually_exclusive(qtbot, sdata_blobs) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)

    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata_blobs)

    image_card = widget.image_cards[0]

    assert isinstance(image_card.overlay_toggle, QCheckBox)
    assert isinstance(image_card.stack_toggle, QCheckBox)
    mode_layout = image_card.layout().itemAt(0).layout()
    assert mode_layout.itemAt(0).widget() is image_card.overlay_toggle
    assert mode_layout.itemAt(1).widget() is image_card.stack_toggle
    assert image_card.overlay_toggle.isChecked()
    assert not image_card.stack_toggle.isChecked()
    assert not image_card.channel_panel.isHidden()
    assert image_card.channel_section_label.text() == "Channels"
    assert image_card.stack_load_button.text() == "Load in viewer"
    assert image_card.stack_load_button.isHidden()
    assert not hasattr(image_card, "add_update_button")
    assert not hasattr(image_card, "add_update_requested")

    image_card.overlay_toggle.setChecked(False)

    assert image_card.overlay_toggle.isChecked()
    assert not image_card.stack_toggle.isChecked()

    image_card.stack_toggle.setChecked(True)

    assert not image_card.overlay_toggle.isChecked()
    assert image_card.stack_toggle.isChecked()
    assert image_card.channel_panel.isHidden()
    assert image_card.stack_load_button.isEnabled()

    image_card.overlay_toggle.setChecked(True)

    assert image_card.overlay_toggle.isChecked()
    assert not image_card.stack_toggle.isChecked()
    assert not image_card.channel_panel.isHidden()
    assert image_card.stack_load_button.isHidden()


def test_viewer_widget_overlay_composer_keeps_many_channels_searchable(qtbot, monkeypatch) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    fake_sdata = object()
    many_channels = [f"c{i}" for i in range(12)]

    qtbot.addWidget(widget)

    _patch_coordinate_system_names(monkeypatch, ["global"])
    monkeypatch.setattr(viewer_widget_module, "_get_labels_in_coordinate_system", lambda sdata, coordinate_system: [])
    monkeypatch.setattr(
        viewer_widget_module, "_get_images_in_coordinate_system", lambda sdata, coordinate_system: ["image"]
    )
    monkeypatch.setattr(
        viewer_widget_module, "get_image_channel_names_from_sdata", lambda sdata, image_name: many_channels
    )

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(fake_sdata)

    image_card = widget.image_cards[0]
    image_card.overlay_toggle.setChecked(True)

    assert image_card.available_channel_names == tuple(many_channels)
    assert image_card.loaded_overlay_channel_indices == ()
    assert image_card.selected_count_label.text() == "0 channels"
    assert not image_card.no_selected_channels_label.isHidden()
    assert image_card.channel_scroll_area.isHidden()
    assert image_card.channel_scroll_area.verticalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAsNeeded
    assert image_card.channel_search_input.completer().completionMode() == (QCompleter.CompletionMode.PopupCompletion)


def test_image_card_overlay_viewport_grows_through_ten_rows_then_scrolls(qtbot) -> None:
    channel_names = [f"channel_{index}" for index in range(12)]
    bindings = [
        ImageLayerBinding(
            layer=Image(np.zeros((8, 8))),
            element_name="image",
            coordinate_system="global",
            sdata_id=1,
            image_display_mode="overlay",
            channel_index=index,
            channel_name=channel_name,
        )
        for index, channel_name in enumerate(channel_names)
    ]
    card = _ImageCardWidget(
        image_name="image",
        channel_names=channel_names,
    )
    qtbot.addWidget(card)
    card.show()

    viewport_heights: list[int] = []
    for channel_count in (1, 3, 10):
        card.set_loaded_image_bindings(
            stack_binding=None,
            overlay_bindings=bindings[:channel_count],
        )
        qtbot.wait(1)
        viewport_heights.append(card.channel_scroll_area.height())

    assert viewport_heights[0] < viewport_heights[1] < viewport_heights[2]
    assert card.channel_scroll_area.verticalScrollBar().maximum() == 0

    card.set_loaded_image_bindings(
        stack_binding=None,
        overlay_bindings=bindings[:11],
    )
    qtbot.wait(1)

    assert card.channel_scroll_area.height() == viewport_heights[2]
    assert card.channel_scroll_area.verticalScrollBar().maximum() > 0


def test_viewer_widget_surfaces_duplicate_channel_names_and_disables_overlay(qtbot, monkeypatch) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    fake_sdata = object()

    qtbot.addWidget(widget)

    _patch_coordinate_system_names(monkeypatch, ["global"])
    monkeypatch.setattr(viewer_widget_module, "_get_labels_in_coordinate_system", lambda sdata, coordinate_system: [])
    monkeypatch.setattr(
        viewer_widget_module, "_get_images_in_coordinate_system", lambda sdata, coordinate_system: ["image"]
    )
    monkeypatch.setattr(
        viewer_widget_module,
        "get_image_channel_names_from_sdata",
        lambda sdata, image_name: (_ for _ in ()).throw(
            ValueError(
                "Image element `image` exposes duplicate channel names (`dup`), "
                "which spatiato does not support. "
                "Update the channel names in the SpatialData object with "
                "`sdata.set_channel_names(...)`."
            )
        ),
    )

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(fake_sdata)

    image_card = widget.image_cards[0]

    assert image_card.channel_names == []
    assert image_card.channel_error is not None
    assert not image_card.overlay_toggle.isEnabled()
    assert not image_card.channel_warning_label.isHidden()
    assert "sdata.set_channel_names(...)" in image_card.channel_warning_label.text()
    assert "duplicate channel names" in image_card.channel_warning_label.toolTip()


def test_viewer_widget_filters_cards_by_selected_coordinate_system(qtbot, monkeypatch) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    fake_sdata = object()

    qtbot.addWidget(widget)

    _patch_coordinate_system_names(monkeypatch, ["global", "local"])
    monkeypatch.setattr(
        viewer_widget_module,
        "_get_labels_in_coordinate_system",
        lambda sdata, coordinate_system: ["labels_global"] if coordinate_system == "global" else ["labels_local"],
    )
    monkeypatch.setattr(
        viewer_widget_module,
        "_get_images_in_coordinate_system",
        lambda sdata, coordinate_system: ["image_global"] if coordinate_system == "global" else ["image_local"],
    )
    monkeypatch.setattr(
        viewer_widget_module,
        "_get_shapes_in_coordinate_system",
        lambda sdata, coordinate_system: ["shape_global"] if coordinate_system == "global" else ["shape_local"],
    )
    monkeypatch.setattr(viewer_widget_module, "get_shape_column_color_source_options", lambda sdata, shapes_name: [])
    monkeypatch.setattr(
        viewer_widget_module, "get_image_channel_names_from_sdata", lambda sdata, image_name: ["c0", "c1"]
    )
    monkeypatch.setattr(
        viewer_widget_module,
        "get_annotating_table_names",
        lambda sdata, labels_name: ["table_global"] if labels_name == "labels_global" else ["table_local"],
    )
    monkeypatch.setattr(
        viewer_widget_module,
        "get_table_color_source_options",
        lambda sdata, table_name: [],
    )

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(fake_sdata)

    assert widget.coordinate_system_combo.count() == 2
    assert widget.app_state.coordinate_system == "global"
    assert [card.image_name for card in widget.image_cards] == ["image_global"]
    assert [card.labels_name for card in widget.labels_cards] == ["labels_global"]
    assert [card.shapes_name for card in widget.shape_cards] == ["shape_global"]

    with qtbot.waitSignal(widget.app_state.coordinate_system_changed) as blocker:
        widget.coordinate_system_combo.setCurrentIndex(1)

    assert blocker.args[0].previous_coordinate_system == "global"
    assert blocker.args[0].coordinate_system == "local"
    assert blocker.args[0].source == "viewer_widget"
    assert widget.app_state.coordinate_system == "local"
    assert [card.image_name for card in widget.image_cards] == ["image_local"]
    assert [card.labels_name for card in widget.labels_cards] == ["labels_local"]
    assert [card.shapes_name for card in widget.shape_cards] == ["shape_local"]


def test_viewer_widget_refreshes_from_shared_coordinate_system_changes(qtbot, monkeypatch) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    fake_sdata = object()

    qtbot.addWidget(widget)

    _patch_coordinate_system_names(monkeypatch, ["global", "local"])
    monkeypatch.setattr(
        viewer_widget_module,
        "_get_labels_in_coordinate_system",
        lambda sdata, coordinate_system: ["labels_global"] if coordinate_system == "global" else ["labels_local"],
    )
    monkeypatch.setattr(
        viewer_widget_module,
        "_get_images_in_coordinate_system",
        lambda sdata, coordinate_system: ["image_global"] if coordinate_system == "global" else ["image_local"],
    )
    monkeypatch.setattr(
        viewer_widget_module,
        "_get_shapes_in_coordinate_system",
        lambda sdata, coordinate_system: ["shape_global"] if coordinate_system == "global" else ["shape_local"],
    )
    monkeypatch.setattr(viewer_widget_module, "get_shape_column_color_source_options", lambda sdata, shapes_name: [])
    monkeypatch.setattr(
        viewer_widget_module, "get_image_channel_names_from_sdata", lambda sdata, image_name: ["c0", "c1"]
    )
    monkeypatch.setattr(
        viewer_widget_module,
        "get_annotating_table_names",
        lambda sdata, labels_name: ["table_global"] if labels_name == "labels_global" else ["table_local"],
    )
    monkeypatch.setattr(viewer_widget_module, "get_table_color_source_options", lambda sdata, table_name: [])

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(fake_sdata)

    changed = widget.app_state.set_coordinate_system("local", source="object_classification_widget")

    assert changed is True
    assert widget.coordinate_system_combo.currentText() == "local"
    assert [card.image_name for card in widget.image_cards] == ["image_local"]
    assert [card.labels_name for card in widget.labels_cards] == ["labels_local"]
    assert [card.shapes_name for card in widget.shape_cards] == ["shape_local"]


def test_viewer_widget_coordinate_system_switch_prunes_old_spatiato_layers(qtbot, monkeypatch) -> None:
    global_layer = Image(np.zeros((2, 2), dtype=np.float32), name="global_layer")
    local_layer = Image(np.zeros((2, 2), dtype=np.float32), name="local_layer")
    external_layer = Image(np.zeros((2, 2), dtype=np.float32), name="external_layer")
    viewer = DummyViewer()
    viewer.layers.extend([global_layer, local_layer, external_layer])
    widget = ViewerWidget(viewer)
    fake_sdata = object()

    qtbot.addWidget(widget)

    _patch_coordinate_system_names(monkeypatch, ["global", "local"])
    monkeypatch.setattr(viewer_widget_module, "_get_labels_in_coordinate_system", lambda sdata, coordinate_system: [])
    monkeypatch.setattr(viewer_widget_module, "_get_images_in_coordinate_system", lambda sdata, coordinate_system: [])

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(fake_sdata)

    widget.app_state.viewer_adapter.register_image_layer(
        global_layer,
        sdata=fake_sdata,
        image_name="global_image",
        coordinate_system="global",
    )
    widget.app_state.viewer_adapter.register_image_layer(
        local_layer,
        sdata=fake_sdata,
        image_name="local_image",
        coordinate_system="local",
    )
    widget._set_action_feedback(
        title="Labels Layer Created",
        lines=['Created labels layer for "global_image".'],
        kind="success",
    )

    with qtbot.waitSignal(widget.app_state.coordinate_system_changed):
        widget.coordinate_system_combo.setCurrentIndex(1)

    assert widget.app_state.coordinate_system == "local"
    assert widget.global_action_feedback_label.text() == ""
    assert widget.global_action_feedback_label.isHidden()
    assert list(viewer.layers) == [local_layer, external_layer]
    assert widget.app_state.viewer_adapter.layer_bindings.get_binding(global_layer) is None
    assert widget.app_state.viewer_adapter.layer_bindings.get_binding(local_layer) is not None
    assert widget.app_state.viewer_adapter.layer_bindings.get_binding(external_layer) is None


def test_viewer_widget_open_spatialdata_loads_selected_store(qtbot, monkeypatch, sdata_blobs) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    recorded_paths: list[str] = []
    recorded_sdata: list[object] = []
    original_set_sdata = widget.app_state.set_sdata

    qtbot.addWidget(widget)

    monkeypatch.setattr(
        viewer_widget_module.QFileDialog,
        "getExistingDirectory",
        lambda *args, **kwargs: "/tmp/example.zarr",
    )
    monkeypatch.setattr(
        viewer_widget_module,
        "read_zarr",
        lambda path: recorded_paths.append(path) or sdata_blobs,
    )

    def wrapped_set_sdata(sdata: object, *, discard_current: bool = False) -> None:
        recorded_sdata.append(sdata)
        original_set_sdata(sdata, discard_current=discard_current)

    monkeypatch.setattr(widget.app_state, "set_sdata", wrapped_set_sdata)
    widget._set_action_feedback("Old error", is_error=True)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.open_sdata_button.click()

    assert recorded_paths == ["/tmp/example.zarr"]
    assert recorded_sdata == [sdata_blobs]
    assert widget.app_state.sdata is sdata_blobs
    assert widget.coordinate_system_combo.count() == 1
    assert widget.coordinate_system_combo.itemText(0) == "global"
    assert widget.global_action_feedback_label.text() == ""
    assert widget.global_action_feedback_label.isHidden()


def test_viewer_widget_cancelled_spatialdata_replacement_preserves_current_session(
    qtbot,
    monkeypatch,
    sdata_blobs,
) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    qtbot.addWidget(widget)
    widget.app_state.set_sdata(sdata_blobs)

    monkeypatch.setattr(
        viewer_widget_module.QFileDialog,
        "getExistingDirectory",
        lambda *args, **kwargs: "/tmp/replacement.zarr",
    )
    monkeypatch.setattr(viewer_widget_module, "confirm_spatialdata_replacement", lambda _parent: False)

    def fail_if_read(_path: str) -> object:
        raise AssertionError("Cancel must stop before reading the replacement store.")

    monkeypatch.setattr(
        viewer_widget_module,
        "read_zarr",
        fail_if_read,
    )

    widget.open_sdata_button.click()

    assert widget.app_state.sdata is sdata_blobs


def test_viewer_widget_confirmed_spatialdata_replacement_loads_new_session(
    qtbot,
    monkeypatch,
    sdata_blobs,
) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    qtbot.addWidget(widget)
    widget.app_state.set_sdata(sdata_blobs)
    replacement_sdata = SpatialData()

    monkeypatch.setattr(
        viewer_widget_module.QFileDialog,
        "getExistingDirectory",
        lambda *args, **kwargs: "/tmp/replacement.zarr",
    )
    monkeypatch.setattr(viewer_widget_module, "confirm_spatialdata_replacement", lambda _parent: True)
    monkeypatch.setattr(viewer_widget_module, "read_zarr", lambda _path: replacement_sdata)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.open_sdata_button.click()

    assert widget.app_state.sdata is replacement_sdata


def test_viewer_widget_open_spatialdata_opens_table_matrices_lazily(
    qtbot,
    monkeypatch,
    tmp_path,
    sdata_blobs: SpatialData,
) -> None:
    """Open reads a real store with harpy's reader, so table matrices are Dask arrays."""
    store_path = tmp_path / "blobs.zarr"
    sdata_blobs.write(store_path)
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    qtbot.addWidget(widget)
    monkeypatch.setattr(
        viewer_widget_module.QFileDialog,
        "getExistingDirectory",
        lambda *args, **kwargs: str(store_path),
    )

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.open_sdata_button.click()

    sdata = widget.app_state.sdata
    assert sdata is not None
    assert sdata.is_backed()
    table = sdata.tables["table"]
    assert isinstance(table.X, da.Array)
    assert sorted(table.obsm.keys()) == ["features_1", "features_2"]
    for key in table.obsm:
        assert isinstance(table.obsm[key], da.Array)


@pytest.mark.parametrize("error_type", [ValueError, RuntimeError])
def test_viewer_widget_open_spatialdata_shows_error_when_loading_fails(
    qtbot,
    monkeypatch,
    error_type: type[Exception],
) -> None:
    """Open reports any failure of the reader as feedback, not only `OSError` and `ValueError`."""
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)

    qtbot.addWidget(widget)

    monkeypatch.setattr(
        viewer_widget_module.QFileDialog,
        "getExistingDirectory",
        lambda *args, **kwargs: "/tmp/example.zarr",
    )

    def raise_read_error(path: str) -> object:
        raise error_type(f"bad store at {path}")

    monkeypatch.setattr(viewer_widget_module, "read_zarr", raise_read_error)

    widget.open_sdata_button.click()

    assert widget.app_state.sdata is None
    assert "Could not load SpatialData store" in widget.global_action_feedback_label.text()
    assert "bad store at /tmp/example.zarr" in widget.global_action_feedback_label.text()
    assert not widget.global_action_feedback_label.isHidden()


def test_viewer_widget_add_update_labels_loads_and_activates_layer(qtbot, sdata_blobs) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)

    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata_blobs)

    first_card = widget.labels_cards[0]

    first_card.add_update_button.click()

    assert len(viewer.layers) == 1
    _assert_action_feedback_card(widget, title="Labels Layer Created", kind="success")
    assert 'Created labels layer for "blobs_labels"' in widget.global_action_feedback_label.text()


def test_viewer_widget_add_update_labels_dispatches_to_styled_overlay_path(qtbot, monkeypatch, sdata_blobs) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    recorded_requests: list[object] = []

    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata_blobs)

    monkeypatch.setattr(widget, "_add_or_update_styled_labels_layer", lambda request: recorded_requests.append(request))

    first_card = widget.labels_cards[0]
    _select_color_source_kind(first_card, "x_var")
    first_card.color_source_value_input.setText("channel_1_sum")
    first_card.add_update_button.click()

    assert len(recorded_requests) == 1
    request = recorded_requests[0]
    assert request.labels_name == "blobs_labels"
    assert request.table_name == "table"
    assert request.selected_source_kind == "x_var"
    assert request.selected_color_source is not None
    assert request.selected_color_source.value_key == "channel_1_sum"


def test_viewer_widget_add_update_labels_creates_and_updates_styled_overlay(qtbot, sdata_blobs) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    table = sdata_blobs["table"]
    table.obs["cell_type"] = ["odd" if instance_id % 2 else "even" for instance_id in table.obs["instance_id"]]
    table.obs["cell_type"] = table.obs["cell_type"].astype("category")
    table.uns["cell_type_colors"] = ["#ff0000", "#00ff00"]

    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata_blobs)

    first_card = widget.labels_cards[0]
    _select_color_source_kind(first_card, "obs_column")
    first_card.color_source_value_input.setText("cell_type")

    first_card.add_update_button.click()

    assert len(viewer.layers) == 1
    layer = viewer.layers[0]
    binding = widget.app_state.viewer_adapter.layer_bindings.get_binding(layer)
    assert binding is not None
    assert binding.labels_role == "styled"
    _assert_action_feedback_card(widget, title="Colored Overlay Created", kind="success")
    assert 'Created colored overlay for obs["cell_type"]' in widget.global_action_feedback_label.text()
    assert "stored categorical palette" in widget.global_action_feedback_label.text()

    first_card.add_update_button.click()

    assert len(viewer.layers) == 1
    assert viewer.layers[0] is layer
    _assert_action_feedback_card(widget, title="Colored Overlay Updated", kind="success")
    assert 'Updated colored overlay for obs["cell_type"]' in widget.global_action_feedback_label.text()


def test_viewer_widget_styled_overlay_missing_palette_uses_info_card(qtbot, sdata_blobs) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    table = sdata_blobs["table"]
    table.obs["cell_type"] = ["odd" if instance_id % 2 else "even" for instance_id in table.obs["instance_id"]]
    table.obs["cell_type"] = table.obs["cell_type"].astype("category")

    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata_blobs)

    first_card = widget.labels_cards[0]
    _select_color_source_kind(first_card, "obs_column")
    first_card.color_source_value_input.setText("cell_type")

    first_card.add_update_button.click()

    _assert_action_feedback_card(widget, title="Colored Overlay Created", kind="info")
    assert "no stored palette was present" in widget.global_action_feedback_label.text()


def test_viewer_widget_styled_overlay_instance_key_uses_success_card(qtbot, sdata_blobs) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)

    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata_blobs)

    first_card = widget.labels_cards[0]
    _select_color_source_kind(first_card, "obs_column")
    first_card.color_source_value_input.setText("instance_id")

    first_card.add_update_button.click()

    _assert_action_feedback_card(widget, title="Colored Overlay Created", kind="success")
    assert 'Created colored overlay for obs["instance_id"]' in widget.global_action_feedback_label.text()
    assert "Used instance colors." in widget.global_action_feedback_label.text()
    binding = widget.app_state.viewer_adapter.layer_bindings.get_binding(viewer.layers[0])
    assert binding is not None
    assert binding.style_spec is not None
    assert binding.style_spec.value_kind == "instance"


def test_viewer_widget_styled_overlay_invalid_palette_uses_warning_card(qtbot, sdata_blobs) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    table = sdata_blobs["table"]
    table.obs["cell_type"] = ["odd"] * table.n_obs
    table.obs["cell_type"] = table.obs["cell_type"].astype("category")
    table.uns["cell_type_colors"] = ["#ff0000", "#00ff00"]

    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata_blobs)

    first_card = widget.labels_cards[0]
    _select_color_source_kind(first_card, "obs_column")
    first_card.color_source_value_input.setText("cell_type")

    first_card.add_update_button.click()

    _assert_action_feedback_card(widget, title="Colored Overlay Created With Warning", kind="warning")
    assert "stored categorical palette was invalid" in widget.global_action_feedback_label.text()


def test_viewer_widget_styled_overlay_string_coercion_uses_warning_card(qtbot, sdata_blobs) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    table = sdata_blobs["table"]
    table.obs["sample_type"] = ["odd" if instance_id % 2 else "even" for instance_id in table.obs["instance_id"]]

    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata_blobs)

    first_card = widget.labels_cards[0]
    _select_color_source_kind(first_card, "obs_column")
    first_card.color_source_value_input.setText("sample_type")

    first_card.add_update_button.click()

    _assert_action_feedback_card(widget, title="Colored Overlay Created With Warning", kind="warning")
    assert "Coerced string values to categorical" in widget.global_action_feedback_label.text()


def test_viewer_widget_styled_overlay_precondition_error_uses_error_card(qtbot, sdata_blobs) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)

    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata_blobs)

    first_card = widget.labels_cards[0]
    _select_color_source_kind(first_card, "obs_column")
    first_card.color_source_value_input.setText("not_a_column")

    first_card.add_update_button.click()

    _assert_action_feedback_card(widget, title="Styled Labels Error", kind="error")
    assert "The selected observation column is not available" in widget.global_action_feedback_label.text()


def test_viewer_widget_load_image_stack_creates_layer_and_live_row(qtbot, sdata_blobs) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)

    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata_blobs)

    first_card = widget.image_cards[0]

    first_card.stack_toggle.setChecked(True)
    first_card.stack_load_button.click()

    assert len(viewer.layers) == 1
    layer = viewer.layers[0]
    assert layer.name == "blobs_image"
    binding = widget.app_state.viewer_adapter.layer_bindings.get_binding(layer)
    assert binding is not None
    assert binding.image_display_mode == "stack"
    assert viewer.layers.selection.active is layer
    assert first_card.stack_row is not None
    assert first_card.stack_load_button.isHidden()
    _assert_action_feedback_card(widget, title="Image Layer Created", kind="success")
    assert 'Created image layer for "blobs_image" in stack mode' in widget.global_action_feedback_label.text()


def test_viewer_widget_loaded_stack_has_no_update_action(qtbot, sdata_blobs) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)

    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata_blobs)

    first_card = widget.image_cards[0]

    first_card.stack_toggle.setChecked(True)
    first_card.stack_load_button.click()
    first_layer = viewer.layers[0]

    assert len(viewer.layers) == 1
    assert viewer.layers[0] is first_layer
    assert first_card.stack_row is not None
    assert first_card.stack_load_button.isHidden()


def test_viewer_widget_mode_switch_is_presentation_only_until_first_overlay_is_accepted(
    qtbot,
    sdata_blobs,
) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata_blobs)

    card = widget.image_cards[0]
    card.stack_toggle.setChecked(True)
    card.stack_load_button.click()
    stack_layer = viewer.layers[0]

    card.overlay_toggle.setChecked(True)

    assert list(viewer.layers) == [stack_layer]
    assert card.overlay_toggle.isChecked()
    assert not card.channel_panel.isHidden()

    card.channel_search_input.setText("1")
    card.channel_search_input.returnPressed.emit()

    assert len(viewer.layers) == 1
    assert viewer.layers[0] is not stack_layer
    binding = widget.app_state.viewer_adapter.layer_bindings.get_binding(viewer.layers[0])
    assert isinstance(binding, ImageLayerBinding)
    assert binding.image_display_mode == "overlay"
    assert binding.channel_index == 1
    assert card.loaded_stack_binding is None
    assert card.loaded_overlay_channel_indices == (1,)
    assert card.overlay_toggle.isChecked()


def test_viewer_widget_explicit_stack_load_replaces_overlays_after_mode_switch(
    qtbot,
    sdata_blobs,
) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata_blobs)

    card = widget.image_cards[0]
    card.overlay_toggle.setChecked(True)
    for channel_name in ("0", "2"):
        card.channel_search_input.setText(channel_name)
        card.channel_search_input.returnPressed.emit()
    overlay_layers = list(viewer.layers)

    card.stack_toggle.setChecked(True)

    assert list(viewer.layers) == overlay_layers
    assert card.stack_toggle.isChecked()
    assert not card.stack_load_button.isHidden()

    card.stack_load_button.click()

    assert len(viewer.layers) == 1
    assert viewer.layers[0] not in overlay_layers
    binding = widget.app_state.viewer_adapter.layer_bindings.get_binding(viewer.layers[0])
    assert isinstance(binding, ImageLayerBinding)
    assert binding.image_display_mode == "stack"
    assert card.loaded_overlay_channel_indices == ()
    assert card.loaded_stack_binding is binding
    assert card.stack_row is not None
    assert card.stack_load_button.isHidden()


def test_viewer_widget_overlay_composer_requests_one_channel_with_default_color(qtbot, monkeypatch) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    fake_sdata = object()
    fake_layer = object()
    recorded_calls: list[tuple[object, str, str, int, str]] = []
    activated_layers: list[object] = []

    qtbot.addWidget(widget)

    _patch_coordinate_system_names(monkeypatch, ["global"])
    monkeypatch.setattr(viewer_widget_module, "_get_labels_in_coordinate_system", lambda sdata, coordinate_system: [])
    monkeypatch.setattr(
        viewer_widget_module, "_get_images_in_coordinate_system", lambda sdata, coordinate_system: ["image"]
    )
    monkeypatch.setattr(
        viewer_widget_module,
        "get_image_channel_names_from_sdata",
        lambda sdata, image_name: ["c0", "c1", "c2"],
    )
    monkeypatch.setattr(
        widget.app_state.viewer_adapter,
        "ensure_image_overlay_channel_loaded",
        lambda sdata, image_name, coordinate_system, *, channel, channel_color: (
            recorded_calls.append((sdata, image_name, coordinate_system, channel, channel_color))
            or SimpleNamespace(
                layers=(fake_layer,),
                primary_layer=fake_layer,
                mode="overlay",
                created=True,
                channels=(channel,),
                channel_names=("c2",),
            )
        ),
    )
    monkeypatch.setattr(
        widget.app_state.viewer_adapter,
        "activate_layer",
        lambda layer: activated_layers.append(layer) or True,
    )

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(fake_sdata)

    image_card = widget.image_cards[0]
    image_card.overlay_toggle.setChecked(True)
    image_card.channel_search_input.setText("c")
    image_card.channel_search_input.completer().activated[str].emit("c2")
    # Match Cocoa Qt's final write after activated callbacks return.
    image_card.channel_search_input.setText("c2")
    image_card.channel_search_input._completion_clear_timer.timeout.emit()

    assert recorded_calls == [(fake_sdata, "image", "global", 2, "#00FFFF")]
    assert activated_layers == [fake_layer]
    assert image_card.channel_search_input.text() == ""
    assert image_card.loaded_overlay_channel_indices == ()


def test_viewer_widget_overlay_composer_preserves_input_when_add_fails(
    qtbot,
    monkeypatch,
    sdata_blobs,
) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata_blobs)

    monkeypatch.setattr(
        widget.app_state.viewer_adapter,
        "ensure_image_overlay_channel_loaded",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("Could not load this channel.")),
    )
    image_card = widget.image_cards[0]
    image_card.overlay_toggle.setChecked(True)
    image_card.channel_search_input.setText("1")

    image_card.channel_search_input.returnPressed.emit()

    assert image_card.channel_search_input.text() == "1"
    assert image_card.loaded_overlay_channel_indices == ()
    _assert_action_feedback_card(widget, title="Image Overlay Error", kind="error")
    assert "Could not load this channel." in widget.global_action_feedback_label.text()


def test_viewer_widget_overlay_composer_adds_reuses_and_removes_live_channels(qtbot, sdata_blobs) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)

    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata_blobs)

    image_card = widget.image_cards[0]

    image_card.overlay_toggle.setChecked(True)
    image_card.channel_search_input.setText("0")
    image_card.channel_search_input.returnPressed.emit()
    image_card.channel_search_input.setText("2")
    image_card.channel_search_input.returnPressed.emit()

    assert len(viewer.layers) == 2
    first_layers = list(viewer.layers)
    assert [layer.name for layer in first_layers] == ["blobs_image[0]", "blobs_image[2]"]
    assert viewer.layers.selection.active is first_layers[1]
    assert image_card.loaded_overlay_channel_indices == (0, 2)
    assert image_card.loaded_overlay_channel_names == ("0", "2")
    assert image_card.available_channel_names == ("1",)
    assert image_card.selected_count_label.text() == "2 channels"
    assert 'Created image overlay for "blobs_image"' in widget.global_action_feedback_label.text()

    image_card.channel_search_input.setText("2")
    image_card.channel_search_input.returnPressed.emit()

    assert len(viewer.layers) == 2
    assert list(viewer.layers) == first_layers

    image_card.channel_search_input.setText("0")
    image_card.overlay_rows[0].remove_button.click()

    assert len(viewer.layers) == 1
    assert viewer.layers[0].name == "blobs_image[2]"
    assert image_card.channel_search_input.text() == ""
    assert image_card.loaded_overlay_channel_indices == (2,)
    assert image_card.available_channel_names == ("0", "1")


def test_viewer_widget_overlay_row_syncs_visibility_and_colormap_bidirectionally(
    qtbot,
    monkeypatch,
    sdata_blobs,
) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)

    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata_blobs)

    image_card = widget.image_cards[0]
    image_card.overlay_toggle.setChecked(True)
    image_card.channel_search_input.setText("0")
    image_card.channel_search_input.returnPressed.emit()

    layer = viewer.layers[0]
    row = image_card.overlay_rows[0]
    assert row.visibility_button.isChecked()
    assert "Hide channel 0" in _tooltip_text(row.visibility_button)
    assert row.color_button.current_color == "#00FFFF"
    visibility_presentations: list[bool] = []
    original_apply_visibility = row._apply_visibility

    def record_visibility_presentation(visible: bool) -> None:
        visibility_presentations.append(visible)
        original_apply_visibility(visible)

    row._apply_visibility = record_visibility_presentation  # type: ignore[method-assign]

    row.visibility_button.click()

    assert layer.visible is False
    assert not row.visibility_button.isChecked()
    assert "Show channel 0" in _tooltip_text(row.visibility_button)
    assert image_card.loaded_overlay_channel_indices == (0,)
    assert visibility_presentations == [False]

    layer.visible = True

    assert row.visibility_button.isChecked()
    assert "Hide channel 0" in _tooltip_text(row.visibility_button)
    assert visibility_presentations == [False, True]

    monkeypatch.setattr(
        overlay_color_button_module.QColorDialog,
        "getColor",
        lambda *args, **kwargs: QColor("#123456"),
    )
    row.color_button.choose_color()

    assert layer.colormap.name == "#123456"
    assert row.color_button.current_color == "#123456"
    assert row.color_button.gradient_name is None
    assert visibility_presentations == [False, True]

    layer.colormap = "viridis"

    assert row.color_button.gradient_name == "viridis"
    assert "qlineargradient" in row.color_button.styleSheet()
    assert "viridis" in _tooltip_text(row.color_button)
    assert visibility_presentations == [False, True]


def test_viewer_widget_overlay_property_intent_skips_equal_live_values(
    qtbot,
    sdata_blobs,
) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)

    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata_blobs)

    image_card = widget.image_cards[0]
    image_card.overlay_toggle.setChecked(True)
    image_card.channel_search_input.setText("0")
    image_card.channel_search_input.returnPressed.emit()
    layer = viewer.layers[0]
    visible_events: list[object] = []
    colormap_events: list[object] = []
    layer.events.visible.connect(visible_events.append)
    layer.events.colormap.connect(colormap_events.append)

    image_card.overlay_channel_visibility_requested.emit(
        image_card.image_name,
        0,
        True,
    )
    image_card.overlay_channel_color_requested.emit(
        image_card.image_name,
        0,
        "#00FFFF",
    )

    assert visible_events == []
    assert colormap_events == []


def test_viewer_widget_overlay_property_failure_restores_live_row_state(
    qtbot,
    monkeypatch,
) -> None:
    class _RejectingPresentationLayer:
        def __init__(self) -> None:
            self.events = SimpleNamespace(
                visible=DummyEventEmitter(),
                colormap=DummyEventEmitter(),
            )
            self._visible = True
            self._colormap = SimpleNamespace(
                name="#00FFFF",
                colors=np.asarray(
                    [
                        [0.0, 0.0, 0.0, 1.0],
                        [0.0, 1.0, 1.0, 1.0],
                    ]
                ),
            )

        @property
        def visible(self) -> bool:
            return self._visible

        @visible.setter
        def visible(self, value: bool) -> None:
            del value
            raise ValueError("visibility rejected")

        @property
        def colormap(self) -> object:
            return self._colormap

        @colormap.setter
        def colormap(self, value: str) -> None:
            del value
            raise ValueError("colormap rejected")

    layer = _RejectingPresentationLayer()
    binding = ImageLayerBinding(
        layer=layer,  # type: ignore[arg-type]
        element_name="image",
        coordinate_system="global",
        sdata_id=1,
        image_display_mode="overlay",
        channel_index=0,
        channel_name="DAPI",
    )
    card = _ImageCardWidget(
        image_name="image",
        channel_names=["DAPI"],
    )
    card.set_loaded_image_bindings(
        stack_binding=None,
        overlay_bindings=[binding],
    )
    widget = ViewerWidget()
    widget._image_cards = [card]
    card.overlay_channel_visibility_requested.connect(widget._change_image_overlay_channel_visibility)
    card.overlay_channel_color_requested.connect(widget._change_image_overlay_channel_color)
    monkeypatch.setattr(
        widget,
        "_resolve_live_overlay_binding",
        lambda image_name, channel_index: binding,
    )

    qtbot.addWidget(widget)
    qtbot.addWidget(card)

    row = card.overlay_rows[0]
    row.visibility_button.click()

    assert row.visibility_button.isChecked()
    assert "visibility rejected" in widget.global_action_feedback_label.text()

    monkeypatch.setattr(
        overlay_color_button_module.QColorDialog,
        "getColor",
        lambda *args, **kwargs: QColor("#123456"),
    )
    row.color_button.choose_color()

    assert row.color_button.current_color == "#00FFFF"
    assert "colormap rejected" in widget.global_action_feedback_label.text()


def test_viewer_widget_rebuild_disposes_stale_overlay_rows(
    qtbot,
    sdata_blobs,
) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)

    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata_blobs)

    image_card = widget.image_cards[0]
    image_card.overlay_toggle.setChecked(True)
    image_card.channel_search_input.setText("0")
    image_card.channel_search_input.returnPressed.emit()
    layer = viewer.layers[0]
    stale_row = image_card.overlay_rows[0]

    widget._rebuild_image_cards(
        sdata_blobs,
        ["blobs_image", "blobs_multiscale_image"],
    )
    current_row = widget.image_cards[0].overlay_rows[0]
    layer.visible = False

    assert stale_row.visibility_button.isChecked()
    assert not current_row.visibility_button.isChecked()


def test_viewer_widget_overlay_membership_hydrates_and_tracks_napari_side_removal(
    qtbot,
    sdata_blobs,
) -> None:
    viewer = DummyViewer()
    app_state = app_state_module.get_or_create_app_state(viewer)
    app_state.set_sdata(sdata_blobs)
    result = app_state.viewer_adapter.ensure_image_overlay_channel_loaded(
        sdata_blobs,
        "blobs_image",
        "global",
        channel=1,
        channel_color="#FF00FF",
    )

    widget = ViewerWidget(viewer)
    qtbot.addWidget(widget)
    image_card = widget.image_cards[0]

    assert image_card.loaded_overlay_channel_indices == (1,)
    assert image_card.available_channel_names == ("0", "2")
    assert image_card.overlay_toggle.isChecked()

    layer = result.primary_layer
    viewer.layers.remove(layer)
    viewer.layers.events.removed.emit(layer)

    assert image_card.loaded_overlay_channel_indices == ()
    assert image_card.available_channel_names == ("0", "1", "2")
    assert image_card.overlay_toggle.isChecked()


def test_viewer_widget_reports_mixed_stack_and_overlay_membership_without_mutating_layers(
    qtbot,
    sdata_blobs,
) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)

    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata_blobs)

    image_card = widget.image_cards[0]
    image_card.overlay_toggle.setChecked(True)
    image_card.channel_search_input.setText("0")
    image_card.channel_search_input.returnPressed.emit()
    image_card.channel_search_input.setText("2")
    image_card.channel_search_input.returnPressed.emit()

    stack_layer = Image(np.zeros((2, 8, 8)), name="blobs_image")
    viewer.layers.append(stack_layer)
    widget.app_state.viewer_adapter.register_image_layer(
        stack_layer,
        image_name="blobs_image",
        coordinate_system="global",
        sdata=sdata_blobs,
        image_display_mode="stack",
    )

    assert [layer.name for layer in viewer.layers] == ["blobs_image[0]", "blobs_image[2]", "blobs_image"]

    assert image_card.loaded_overlay_channel_indices == (0, 2)
    assert image_card.loaded_stack_binding is None
    assert [layer.name for layer in viewer.layers] == [
        "blobs_image[0]",
        "blobs_image[2]",
        "blobs_image",
    ]
    assert "cannot have both live stack and overlay bindings" in widget.global_action_feedback_label.text()


def test_viewer_widget_reports_duplicate_stack_membership_without_replacing_safe_state(
    qtbot,
    sdata_blobs,
) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata_blobs)

    image_card = widget.image_cards[0]
    first_stack = Image(np.zeros((2, 8, 8)), name="blobs_image")
    viewer.layers.append(first_stack)
    widget.app_state.viewer_adapter.register_image_layer(
        first_stack,
        image_name="blobs_image",
        coordinate_system="global",
        sdata=sdata_blobs,
        image_display_mode="stack",
    )
    first_binding = image_card.loaded_stack_binding

    second_stack = Image(np.zeros((2, 8, 8)), name="blobs_image duplicate")
    viewer.layers.append(second_stack)
    widget.app_state.viewer_adapter.register_image_layer(
        second_stack,
        image_name="blobs_image",
        coordinate_system="global",
        sdata=sdata_blobs,
        image_display_mode="stack",
    )

    assert image_card.loaded_stack_binding is first_binding
    assert list(viewer.layers) == [first_stack, second_stack]
    assert "multiple live stack bindings" in widget.global_action_feedback_label.text()


def test_viewer_widget_hydrates_and_tracks_stack_membership(qtbot, sdata_blobs) -> None:
    viewer = DummyViewer()
    app_state = app_state_module.get_or_create_app_state(viewer)
    app_state.set_sdata(sdata_blobs)
    result = app_state.viewer_adapter.ensure_image_loaded(
        sdata_blobs,
        "blobs_image",
        "global",
        mode="stack",
    )

    widget = ViewerWidget(viewer)
    qtbot.addWidget(widget)
    image_card = widget.image_cards[0]

    assert image_card.loaded_stack_binding is not None
    assert image_card.loaded_stack_binding.layer is result.primary_layer
    assert image_card.stack_toggle.isChecked()
    assert image_card.stack_row is not None
    assert image_card.stack_row.layer_label.text() == "Stack"
    assert image_card.stack_load_button.isHidden()

    layer = result.primary_layer
    stale_row = image_card.stack_row
    viewer.layers.remove(layer)
    viewer.layers.events.removed.emit(layer)

    assert image_card.loaded_stack_binding is None
    assert image_card.stack_row is None
    assert image_card.stack_toggle.isChecked()
    assert not image_card.stack_load_button.isHidden()

    layer.visible = False
    layer.colormap = "#123456"

    assert stale_row.visibility_button.isChecked()
    assert stale_row.color_button.current_color != "#123456"


def test_viewer_widget_stack_row_syncs_visibility_colormap_and_removal(
    qtbot,
    monkeypatch,
    sdata_blobs,
) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata_blobs)

    image_card = widget.image_cards[0]
    image_card.stack_toggle.setChecked(True)
    image_card.stack_load_button.click()
    layer = viewer.layers[0]
    row = image_card.stack_row
    assert row is not None
    assert row.binding.layer is layer
    assert row.visibility_button.isChecked()
    assert not row.color_button.isHidden()

    visibility_presentations: list[bool] = []
    original_apply_visibility = row._apply_visibility

    def record_visibility_presentation(visible: bool) -> None:
        visibility_presentations.append(visible)
        original_apply_visibility(visible)

    row._apply_visibility = record_visibility_presentation  # type: ignore[method-assign]
    row.visibility_change_requested.emit(True)

    assert layer.visible is True
    assert visibility_presentations == [True]

    row.visibility_button.click()

    assert layer.visible is False
    assert not row.visibility_button.isChecked()
    assert visibility_presentations == [True, False]

    layer.visible = True

    assert row.visibility_button.isChecked()
    assert visibility_presentations == [True, False, True]

    color_presentations: list[str] = []
    original_set_color = row.color_button.set_color

    def record_color_presentation(color: str) -> None:
        color_presentations.append(color)
        original_set_color(color)

    row.color_button.set_color = record_color_presentation  # type: ignore[method-assign]
    current_color = row.color_button.current_color
    original_colormap = layer.colormap
    row.color_change_requested.emit(current_color)

    assert layer.colormap is original_colormap
    assert color_presentations == [current_color]

    monkeypatch.setattr(
        overlay_color_button_module.QColorDialog,
        "getColor",
        lambda *args, **kwargs: QColor("#123456"),
    )
    row.color_button.choose_color()

    assert layer.colormap.name == "#123456"
    assert row.color_button.current_color == "#123456"

    layer.colormap = "viridis"

    assert row.color_button.gradient_name == "viridis"
    assert "qlineargradient" in row.color_button.styleSheet()

    row.remove_button.click()

    assert list(viewer.layers) == []
    assert image_card.stack_row is None
    assert image_card.stack_toggle.isChecked()
    assert not image_card.stack_load_button.isHidden()
    assert "Removed Stack" in widget.global_action_feedback_label.text()


def test_image_card_rgb_stack_row_has_no_colormap_control(qtbot) -> None:
    layer = Image(np.zeros((8, 8, 3)), rgb=True)
    binding = ImageLayerBinding(
        layer=layer,
        element_name="image",
        coordinate_system="global",
        sdata_id=1,
        image_display_mode="stack",
    )
    card = _ImageCardWidget(
        image_name="image",
        channel_names=["R", "G", "B"],
    )
    qtbot.addWidget(card)

    card.set_loaded_image_bindings(
        stack_binding=binding,
        overlay_bindings=(),
    )

    row = card.stack_row
    assert row is not None
    assert row.layer_label.text() == "RGB stack"
    assert row.color_button.isHidden()
    assert not row.visibility_button.isHidden()
    assert not row.remove_button.isHidden()
    assert card.loaded_stack_binding is binding


def test_viewer_widget_stack_property_failure_restores_live_row_state(
    qtbot,
    monkeypatch,
) -> None:
    class _RejectingStackLayer:
        rgb = False

        def __init__(self) -> None:
            self.events = SimpleNamespace(
                visible=DummyEventEmitter(),
                colormap=DummyEventEmitter(),
            )
            self._visible = True
            self._colormap = SimpleNamespace(
                name="#00FFFF",
                colors=np.asarray(
                    [
                        [0.0, 0.0, 0.0, 1.0],
                        [0.0, 1.0, 1.0, 1.0],
                    ]
                ),
            )

        @property
        def visible(self) -> bool:
            return self._visible

        @visible.setter
        def visible(self, value: bool) -> None:
            del value
            raise ValueError("visibility rejected")

        @property
        def colormap(self) -> object:
            return self._colormap

        @colormap.setter
        def colormap(self, value: str) -> None:
            del value
            raise ValueError("colormap rejected")

    layer = _RejectingStackLayer()
    binding = ImageLayerBinding(
        layer=layer,  # type: ignore[arg-type]
        element_name="image",
        coordinate_system="global",
        sdata_id=1,
        image_display_mode="stack",
    )
    card = _ImageCardWidget(
        image_name="image",
        channel_names=["DAPI"],
    )
    card.set_loaded_image_bindings(
        stack_binding=binding,
        overlay_bindings=(),
    )
    widget = ViewerWidget()
    widget._image_cards = [card]
    card.stack_visibility_requested.connect(widget._change_image_stack_visibility)
    card.stack_color_requested.connect(widget._change_image_stack_color)
    monkeypatch.setattr(
        widget,
        "_resolve_exact_live_stack_binding",
        lambda image_name: binding,
    )
    qtbot.addWidget(widget)
    qtbot.addWidget(card)

    row = card.stack_row
    assert row is not None
    row.visibility_button.click()

    assert row.visibility_button.isChecked()
    assert "visibility rejected" in widget.global_action_feedback_label.text()

    monkeypatch.setattr(
        overlay_color_button_module.QColorDialog,
        "getColor",
        lambda *args, **kwargs: QColor("#123456"),
    )
    row.color_button.choose_color()

    assert row.color_button.current_color == "#00FFFF"
    assert "colormap rejected" in widget.global_action_feedback_label.text()


def test_viewer_widget_load_image_stack_uses_selected_coordinate_system(qtbot, monkeypatch) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    fake_sdata = object()
    fake_layer = Shapes([np.asarray([(0, 0), (0, 1), (1, 1), (1, 0)], dtype=float)], shape_type="polygon")
    recorded_calls: list[tuple[object, str, str, str]] = []
    activated_layers: list[object] = []

    qtbot.addWidget(widget)

    _patch_coordinate_system_names(monkeypatch, ["global", "local"])
    monkeypatch.setattr(viewer_widget_module, "_get_labels_in_coordinate_system", lambda sdata, coordinate_system: [])
    monkeypatch.setattr(
        viewer_widget_module,
        "_get_images_in_coordinate_system",
        lambda sdata, coordinate_system: ["image_global"] if coordinate_system == "global" else ["image_local"],
    )
    monkeypatch.setattr(
        viewer_widget_module, "get_image_channel_names_from_sdata", lambda sdata, image_name: ["c0", "c1"]
    )
    monkeypatch.setattr(
        widget.app_state.viewer_adapter,
        "ensure_image_loaded",
        lambda sdata, image_name, coordinate_system, *, mode, channels=None, channel_colors=None: (
            recorded_calls.append((sdata, image_name, coordinate_system, mode))
            or SimpleNamespace(
                layers=(fake_layer,),
                primary_layer=fake_layer,
                mode=mode,
                created=True,
                channels=tuple(channels or ()),
            )
        ),
    )
    monkeypatch.setattr(
        widget.app_state.viewer_adapter,
        "activate_layer",
        lambda layer: activated_layers.append(layer) or True,
    )

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(fake_sdata)

    widget.coordinate_system_combo.setCurrentIndex(1)
    image_card = widget.image_cards[0]

    image_card.stack_toggle.setChecked(True)
    image_card.stack_load_button.click()

    assert recorded_calls == [(fake_sdata, "image_local", "local", "stack")]
    assert activated_layers == [fake_layer]


def test_viewer_widget_shapes_card_exposes_shape_column_controls(qtbot) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    sdata = _make_colorable_shapes_sdata(cell_type_colors=["red", "blue"])

    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata)

    card = widget.shape_cards[0]

    assert [card.color_source_kind_combo.itemText(index) for index in range(card.color_source_kind_combo.count())] == [
        "No color source",
        "Shapes column",
    ]
    assert card.color_source_kind_combo.isEnabled()
    assert card.color_source_kind_combo.findData("obs_column") == -1
    assert card.color_source_kind_combo.findData("x_var") == -1
    assert _combo_texts(card.linked_table_combo) == ["No linked tables"]
    assert not card.linked_table_combo.isEnabled()
    assert card.color_source_value_label.text() == "Value source"
    assert not card.color_source_value_input.isEnabled()
    assert card.fill_toggle.text() == "Fill"
    assert not card.fill_toggle.isEnabled()
    assert not card.fill_toggle.isChecked()
    assert card.action_hint_label.text() == "Action: add/update primary shapes layer"

    _select_color_source_kind(card, "shape_column")

    assert card.color_source_value_input.isEnabled()
    assert card.color_source_value_input.text() == ""
    assert card.color_source_value_input.placeholderText() == "Select column"
    assert not card.fill_toggle.isEnabled()
    assert not card.fill_toggle.isChecked()
    assert card._color_source_completer_model.stringList() == ["cell_type", "score", "free_text"]
    assert card.color_source_value_input.completer().maxVisibleItems() == 10
    card.color_source_value_input.show_completion_popup()
    assert card.color_source_value_input.completer().completionPrefix() == ""
    assert card.color_source_value_input.completer().completionModel().rowCount() == 3
    card.color_source_value_input.completer().popup().hide()
    assert card.action_hint_label.text() == "Action: select a shapes column for a styled shapes layer"

    card.color_source_value_input.setText("cell_type")
    assert card.fill_toggle.isEnabled()
    assert card.action_hint_label.text() == 'Action: add/update styled shapes layer for column "cell_type"'

    card.fill_toggle.setChecked(True)
    _select_color_source_kind(card, "shape_column")
    card.color_source_value_input.setText("not_a_shape_column")

    assert not card.fill_toggle.isEnabled()
    assert not card.fill_toggle.isChecked()
    assert card.action_hint_label.text() == "Action: select a shapes column for a styled shapes layer"


def test_viewer_widget_shapes_card_disables_color_source_when_no_sources(qtbot) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    geodataframe = gpd.GeoDataFrame(
        geometry=[
            Polygon([(0, 0), (4, 0), (4, 4), (0, 4), (0, 0)]),
        ],
        index=["cell_1"],
    )
    sdata = _make_shapes_sdata(geodataframe)

    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata)

    card = widget.shape_cards[0]

    assert _combo_texts(card.linked_table_combo) == ["No linked tables"]
    assert not card.linked_table_combo.isEnabled()
    assert _combo_texts(card.color_source_kind_combo) == ["No color source"]
    assert not card.color_source_kind_combo.isEnabled()
    assert card.selected_source_kind is None
    assert card.selected_color_source is None
    assert not card.color_source_value_input.isEnabled()
    assert card._color_source_completer_model.stringList() == []
    assert not card.fill_toggle.isEnabled()
    assert not card.fill_toggle.isChecked()
    assert card.action_hint_label.text() == "Action: add/update primary shapes layer"


def test_viewer_widget_shapes_card_omits_shape_column_when_only_table_sources(qtbot, monkeypatch) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    geodataframe = gpd.GeoDataFrame(
        geometry=[
            Polygon([(0, 0), (4, 0), (4, 4), (0, 4), (0, 0)]),
        ],
        index=["cell_1"],
    )
    sdata = _make_shapes_sdata(geodataframe)
    table_source = TableColorSourceSpec(
        table_name="table",
        source_kind="obs_column",
        value_key="cell_type",
        value_kind="categorical",
    )

    monkeypatch.setattr(
        viewer_widget_module,
        "get_annotating_table_names",
        lambda sdata, element_name: ["table"] if element_name == "cells" else [],
    )
    monkeypatch.setattr(
        viewer_widget_module,
        "get_table_color_source_options",
        lambda sdata, table_name: [table_source],
    )
    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata)

    card = widget.shape_cards[0]

    assert _combo_texts(card.color_source_kind_combo) == ["No color source", "Observations"]
    assert card.color_source_kind_combo.findData("shape_column") == -1
    assert card.color_source_kind_combo.isEnabled()

    _select_color_source_kind(card, "obs_column")

    assert card.color_source_value_input.isEnabled()
    assert card._color_source_completer_model.stringList() == ["cell_type"]
    assert card.action_hint_label.text() == "Action: select an observation column for a styled shapes layer"


def test_viewer_widget_shape_column_selector_hides_geometry_and_palette_columns(qtbot) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    sdata = _make_colorable_shapes_sdata(cell_type_colors=["red", "blue"])

    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata)

    card = widget.shape_cards[0]
    _select_color_source_kind(card, "shape_column")

    assert "geometry" not in card._color_source_completer_model.stringList()
    assert "cell_type_colors" not in card._color_source_completer_model.stringList()


def test_viewer_widget_shapes_card_exposes_linked_table_sources(qtbot, monkeypatch) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    sdata = _make_colorable_shapes_sdata(cell_type_colors=["red", "blue"])
    color_sources_by_table = {
        "table_a": [
            TableColorSourceSpec(
                table_name="table_a",
                source_kind="obs_column",
                value_key="cell_type",
                value_kind="categorical",
            )
        ],
        "table_b": [
            TableColorSourceSpec(
                table_name="table_b",
                source_kind="x_var",
                value_key="GeneA",
                value_kind="continuous",
            )
        ],
    }

    monkeypatch.setattr(
        viewer_widget_module,
        "get_annotating_table_names",
        lambda sdata, element_name: ["table_a", "table_b"] if element_name == "cells" else [],
    )
    monkeypatch.setattr(
        viewer_widget_module,
        "get_table_color_source_options",
        lambda sdata, table_name: list(color_sources_by_table[table_name]),
    )
    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata)

    card = widget.shape_cards[0]

    assert _combo_texts(card.linked_table_combo) == ["table_a", "table_b"]
    assert card.selected_table_name == "table_a"
    assert _combo_texts(card.color_source_kind_combo) == ["No color source", "Shapes column", "Observations"]
    assert card.color_source_kind_combo.findData("x_var") == -1

    _select_color_source_kind(card, "obs_column")
    assert card.color_source_value_label.text() == "Observation"
    assert card.color_source_value_input.isEnabled()
    assert card.color_source_value_input.text() == ""
    assert card.color_source_value_input.placeholderText() == "Select obs column"
    assert card._color_source_completer_model.stringList() == ["cell_type"]
    assert card.color_source_value_input.completer().completionMode() == QCompleter.CompletionMode.PopupCompletion
    assert card.color_source_value_input.completer().maxVisibleItems() == 10
    assert card.selected_color_source is None
    assert card.action_hint_label.text() == "Action: select an observation column for a styled shapes layer"

    card.color_source_value_input.setText("cell_type")
    assert card.selected_color_source == color_sources_by_table["table_a"][0]
    assert card.action_hint_label.text() == 'Action: add/update styled shapes layer for obs["cell_type"]'

    card.linked_table_combo.setCurrentIndex(1)
    assert _combo_texts(card.color_source_kind_combo) == ["No color source", "Shapes column", "Vars"]
    assert card.color_source_kind_combo.findData("obs_column") == -1
    assert card.selected_source_kind is None
    assert not card.color_source_value_input.isEnabled()
    assert card.action_hint_label.text() == "Action: add/update primary shapes layer"

    _select_color_source_kind(card, "x_var")
    assert card.color_source_value_label.text() == "Var"
    assert card.color_source_value_input.isEnabled()
    assert card.color_source_value_input.text() == ""
    assert card.color_source_value_input.placeholderText() == "Select var"
    assert card._color_source_completer_model.stringList() == ["GeneA"]
    assert card.selected_color_source is None
    assert card.action_hint_label.text() == "Action: select a var for a styled shapes layer"

    card.color_source_value_input.show_completion_popup()
    assert card.color_source_value_input.completer().completionPrefix() == ""
    assert card.color_source_value_input.completer().completionModel().rowCount() == 1
    card.color_source_value_input.completer().popup().hide()
    card.color_source_value_input.setText("GeneA")
    assert card.selected_color_source == color_sources_by_table["table_b"][0]
    assert card.action_hint_label.text() == 'Action: add/update styled shapes layer for X[:, "GeneA"]'


def test_viewer_widget_add_update_shapes_with_table_source_dispatches_to_styled_path(qtbot, monkeypatch) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    sdata = _make_colorable_shapes_sdata(cell_type_colors=["red", "blue"])
    table_source = TableColorSourceSpec(
        table_name="table",
        source_kind="obs_column",
        value_key="cell_type",
        value_kind="categorical",
    )
    recorded_calls: list[tuple[object, str, str, TableColorSourceSpec, bool]] = []
    result_layer = object()

    monkeypatch.setattr(
        viewer_widget_module,
        "get_annotating_table_names",
        lambda sdata, element_name: ["table"] if element_name == "cells" else [],
    )
    monkeypatch.setattr(
        viewer_widget_module,
        "get_table_color_source_options",
        lambda sdata, table_name: [table_source],
    )

    def ensure_styled_shapes_loaded(
        sdata_arg: object,
        shapes_name: str,
        coordinate_system: str,
        style_spec: TableColorSourceSpec,
        *,
        fill: bool = False,
    ) -> SimpleNamespace:
        recorded_calls.append((sdata_arg, shapes_name, coordinate_system, style_spec, fill))
        return SimpleNamespace(
            layer=result_layer,
            created=True,
            value_kind="categorical",
            palette_source="stored",
            coercion_applied=False,
            skipped_geometry_count=0,
            unannotated_source_shape_count=1,
            unannotated_rendered_shape_count=1,
            shapes_rendering_mode="points",
        )

    monkeypatch.setattr(widget.app_state.viewer_adapter, "ensure_styled_shapes_loaded", ensure_styled_shapes_loaded)
    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata)

    card = widget.shape_cards[0]
    _select_color_source_kind(card, "obs_column")
    card.color_source_value_input.setText("cell_type")
    card.fill_toggle.setChecked(True)
    card.add_update_button.click()

    assert recorded_calls == [(sdata, "cells", "global", table_source, True)]
    assert viewer.layers.selection.active is result_layer
    _assert_action_feedback_card(widget, title="Styled Shapes Created", kind="info")
    assert 'Created styled shapes layer for obs["cell_type"]' in widget.global_action_feedback_label.text()
    assert (
        "Rendered point-radius shapes as napari points for faster display."
        in widget.global_action_feedback_label.text()
    )
    assert "Used the stored categorical palette." in widget.global_action_feedback_label.text()
    assert "Rendered 1 shape transparent because it has no row in the linked table." in (
        widget.global_action_feedback_label.text()
    )


def test_viewer_widget_add_update_shapes_with_shape_column_dispatches_to_styled_path(qtbot, monkeypatch) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    sdata = _make_colorable_shapes_sdata(cell_type_colors=["red", "blue"])
    recorded_requests: list[ShapesLoadRequest] = []

    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata)

    monkeypatch.setattr(widget, "_add_or_update_styled_shapes_layer", lambda request: recorded_requests.append(request))
    card = widget.shape_cards[0]
    _select_shape_column(card, "score")
    card.fill_toggle.setChecked(True)

    card.add_update_button.click()

    assert len(recorded_requests) == 1
    request = recorded_requests[0]
    assert request.shapes_name == "cells"
    assert request.table_name is None
    assert request.selected_source_kind == "shape_column"
    assert request.selected_color_source == ShapeColumnColorSourceSpec(
        source_kind="shape_column",
        value_key="score",
        value_kind="continuous",
    )
    assert request.fill_shapes is True


def test_viewer_widget_add_update_styled_shapes_creates_and_updates_layer(qtbot) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    sdata = _make_colorable_shapes_sdata(cell_type_colors=["red", "blue"])

    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata)

    card = widget.shape_cards[0]
    _select_shape_column(card, "cell_type")

    card.add_update_button.click()

    assert len(viewer.layers) == 1
    layer = viewer.layers[0]
    assert layer.name == "cells[shapes_column:cell_type]"
    binding = widget.app_state.viewer_adapter.layer_bindings.get_binding(layer)
    assert binding is not None
    assert binding.element_type == "shapes"
    assert binding.shapes_role == "styled"
    assert binding.style_spec == ShapeColumnColorSourceSpec(
        source_kind="shape_column",
        value_key="cell_type",
        value_kind="categorical",
    )
    np.testing.assert_allclose(layer.face_color[:, 3], np.zeros(len(layer.data)))
    _assert_action_feedback_card(widget, title="Styled Shapes Created", kind="success")
    assert 'Created styled shapes layer for column "cell_type"' in widget.global_action_feedback_label.text()
    assert "Used the stored categorical palette." in widget.global_action_feedback_label.text()

    card.fill_toggle.setChecked(True)
    card.add_update_button.click()

    assert len(viewer.layers) == 1
    assert viewer.layers[0] is layer
    np.testing.assert_allclose(layer.face_color[:, 3], np.full(len(layer.data), SHAPES_FACE_ALPHA))
    _assert_action_feedback_card(widget, title="Styled Shapes Updated", kind="success")
    assert 'Updated styled shapes layer for column "cell_type"' in widget.global_action_feedback_label.text()


def test_viewer_widget_styled_shapes_feedback_reports_missing_palette(qtbot) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    sdata = _make_colorable_shapes_sdata(cell_type_colors=None)

    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata)

    _select_shape_column(widget.shape_cards[0], "cell_type")
    widget.shape_cards[0].add_update_button.click()

    _assert_action_feedback_card(widget, title="Styled Shapes Created", kind="info")
    assert "no stored palette was present" in widget.global_action_feedback_label.text()


def test_viewer_widget_styled_shapes_feedback_reports_invalid_palette(qtbot) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    sdata = _make_colorable_shapes_sdata(cell_type_colors=["red", "not-a-color"])

    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata)

    _select_shape_column(widget.shape_cards[0], "cell_type")
    widget.shape_cards[0].add_update_button.click()

    _assert_action_feedback_card(widget, title="Styled Shapes Created With Warning", kind="warning")
    assert "stored categorical palette was invalid" in widget.global_action_feedback_label.text()


def test_viewer_widget_styled_shapes_feedback_reports_string_coercion(qtbot) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    sdata = _make_colorable_shapes_sdata(cell_type_colors=["red", "blue"])

    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata)

    _select_shape_column(widget.shape_cards[0], "free_text")
    widget.shape_cards[0].add_update_button.click()

    _assert_action_feedback_card(widget, title="Styled Shapes Created With Warning", kind="warning")
    assert "Coerced string values to categorical" in widget.global_action_feedback_label.text()


def test_viewer_widget_styled_shapes_allows_duplicate_source_index(qtbot) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    sdata = _make_colorable_shapes_sdata(cell_type_colors=["red", "blue"], duplicate_index=True)

    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata)

    _select_shape_column(widget.shape_cards[0], "cell_type")
    widget.shape_cards[0].add_update_button.click()

    assert len(viewer.layers) == 1
    assert viewer.layers[0].features["index"].to_list() == ["cell_1", "cell_1"]
    assert viewer.layers[0].features["cell_type"].to_list() == ["T", "B"]
    _assert_action_feedback_card(widget, title="Styled Shapes Created", kind="success")


def test_viewer_widget_table_backed_styled_shapes_without_linked_table_is_feedback(qtbot) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    sdata = _make_colorable_shapes_sdata(cell_type_colors=["red", "blue"])

    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata)

    request = ShapesLoadRequest(
        shapes_name="cells",
        table_name=None,
        selected_source_kind="obs_column",
        selected_color_source=None,
        fill_shapes=False,
    )

    widget._add_or_update_shapes_layer(request)

    _assert_action_feedback_card(widget, title="Styled Shapes Error", kind="error")
    assert "has no linked table for table-driven coloring" in widget.global_action_feedback_label.text()


def test_viewer_widget_table_backed_styled_shapes_missing_source_is_feedback(qtbot) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    sdata = _make_colorable_shapes_sdata(cell_type_colors=["red", "blue"])

    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata)

    request = ShapesLoadRequest(
        shapes_name="cells",
        table_name="table",
        selected_source_kind="obs_column",
        selected_color_source=None,
        fill_shapes=False,
    )

    widget._add_or_update_shapes_layer(request)

    _assert_action_feedback_card(widget, title="Styled Shapes Error", kind="error")
    assert "The selected observation column is not available" in widget.global_action_feedback_label.text()


def test_viewer_widget_table_backed_styled_shapes_alignment_error_is_feedback(qtbot, monkeypatch) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    sdata = _make_colorable_shapes_sdata(cell_type_colors=["red", "blue"])
    style_spec = TableColorSourceSpec(
        table_name="table",
        source_kind="obs_column",
        value_key="cell_type",
        value_kind="categorical",
    )

    def raise_alignment_error(*args: object, **kwargs: object) -> None:
        raise ValueError("Every selected-region table instance must exist in the shapes instance column.")

    monkeypatch.setattr(widget.app_state.viewer_adapter, "ensure_styled_shapes_loaded", raise_alignment_error)
    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata)

    request = ShapesLoadRequest(
        shapes_name="cells",
        table_name="table",
        selected_source_kind="obs_column",
        selected_color_source=style_spec,
        fill_shapes=False,
    )

    widget._add_or_update_shapes_layer(request)

    _assert_action_feedback_card(widget, title="Styled Shapes Error", kind="error")
    assert "Every selected-region table instance must exist" in widget.global_action_feedback_label.text()


def test_viewer_widget_styled_shapes_feedback_reports_skipped_geometry(qtbot) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    sdata = _make_colorable_shapes_sdata(
        cell_type_colors=["red", "blue"],
        include_unsupported_geometry=True,
    )

    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata)

    _select_shape_column(widget.shape_cards[0], "cell_type")
    widget.shape_cards[0].add_update_button.click()

    _assert_action_feedback_card(widget, title="Styled Shapes Created With Warning", kind="warning")
    assert "Skipped 1 empty, invalid, or unsupported geometries" in widget.global_action_feedback_label.text()


def test_viewer_widget_add_update_shapes_loads_layer(qtbot, sdata_blobs) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)

    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata_blobs)

    first_card = widget.shape_cards[0]
    first_card.fill_toggle.setChecked(True)

    first_card.add_update_button.click()

    assert len(viewer.layers) == 1
    layer = viewer.layers[0]
    assert layer.name == "blobs_circles"
    binding = widget.app_state.viewer_adapter.layer_bindings.get_binding(layer)
    assert binding is not None
    assert binding.element_type == "shapes"
    assert binding.element_name == "blobs_circles"
    assert binding.coordinate_system == "global"
    assert viewer.layers.selection.active is layer
    np.testing.assert_allclose(layer.face_color, np.asarray([to_rgba("#00FFFF")] * len(layer.data)))
    _assert_action_feedback_card(widget, title="Shapes Layer Created", kind="success")
    assert 'Created shapes layer for "blobs_circles".' in widget.global_action_feedback_label.text()
    assert "Rendered point-radius shapes as napari points for faster display." in (
        widget.global_action_feedback_label.text()
    )


def test_viewer_widget_add_update_shapes_reuses_existing_layer(qtbot, sdata_blobs) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)

    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata_blobs)

    first_card = widget.shape_cards[0]

    first_card.add_update_button.click()
    first_layer = viewer.layers[0]

    first_card.add_update_button.click()

    assert len(viewer.layers) == 1
    assert viewer.layers[0] is first_layer


def test_viewer_widget_add_update_shapes_uses_selected_coordinate_system(qtbot, monkeypatch) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    fake_sdata = object()
    fake_layer = object()
    recorded_calls: list[tuple[object, str, str]] = []
    activated_layers: list[object] = []

    qtbot.addWidget(widget)

    _patch_coordinate_system_names(monkeypatch, ["global", "local"])
    monkeypatch.setattr(viewer_widget_module, "_get_labels_in_coordinate_system", lambda sdata, coordinate_system: [])
    monkeypatch.setattr(viewer_widget_module, "_get_images_in_coordinate_system", lambda sdata, coordinate_system: [])
    monkeypatch.setattr(
        viewer_widget_module,
        "_get_shapes_in_coordinate_system",
        lambda sdata, coordinate_system: ["shape_global"] if coordinate_system == "global" else ["shape_local"],
    )
    monkeypatch.setattr(viewer_widget_module, "get_shape_column_color_source_options", lambda sdata, shapes_name: [])
    monkeypatch.setattr(viewer_widget_module, "get_annotating_table_names", lambda sdata, element_name: [])
    monkeypatch.setattr(viewer_widget_module, "get_table_color_source_options", lambda sdata, table_name: [])
    monkeypatch.setattr(
        widget.app_state.viewer_adapter,
        "ensure_shapes_loaded",
        lambda sdata, shapes_name, coordinate_system: (
            recorded_calls.append((sdata, shapes_name, coordinate_system))
            or SimpleNamespace(
                layer=fake_layer,
                created=True,
                value_kind=None,
                palette_source=None,
                coercion_applied=False,
                skipped_geometry_count=0,
                shapes_rendering_mode="shapes",
            )
        ),
    )
    monkeypatch.setattr(
        widget.app_state.viewer_adapter,
        "activate_layer",
        lambda layer: activated_layers.append(layer) or True,
    )

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(fake_sdata)

    widget.coordinate_system_combo.setCurrentIndex(1)
    shape_card = widget.shape_cards[0]

    shape_card.add_update_button.click()

    assert recorded_calls == [(fake_sdata, "shape_local", "local")]
    assert activated_layers == [fake_layer]


def test_viewer_widget_add_update_shapes_reports_skipped_geometry_warning(qtbot, monkeypatch) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    fake_sdata = object()
    fake_layer = object()

    qtbot.addWidget(widget)

    _patch_coordinate_system_names(monkeypatch, ["global"])
    monkeypatch.setattr(viewer_widget_module, "_get_labels_in_coordinate_system", lambda sdata, coordinate_system: [])
    monkeypatch.setattr(viewer_widget_module, "_get_images_in_coordinate_system", lambda sdata, coordinate_system: [])
    monkeypatch.setattr(
        viewer_widget_module, "_get_shapes_in_coordinate_system", lambda sdata, coordinate_system: ["cells"]
    )
    monkeypatch.setattr(viewer_widget_module, "get_shape_column_color_source_options", lambda sdata, shapes_name: [])
    monkeypatch.setattr(viewer_widget_module, "get_annotating_table_names", lambda sdata, element_name: [])
    monkeypatch.setattr(viewer_widget_module, "get_table_color_source_options", lambda sdata, table_name: [])
    monkeypatch.setattr(
        widget.app_state.viewer_adapter,
        "ensure_shapes_loaded",
        lambda sdata, shapes_name, coordinate_system: SimpleNamespace(
            layer=fake_layer,
            created=True,
            value_kind=None,
            palette_source=None,
            coercion_applied=False,
            skipped_geometry_count=2,
            shapes_rendering_mode="shapes",
        ),
    )
    monkeypatch.setattr(widget.app_state.viewer_adapter, "activate_layer", lambda layer: True)
    widget.app_state.viewer_adapter.register_shapes_layer(
        fake_layer,
        sdata=fake_sdata,
        shapes_name="cells",
        coordinate_system="global",
        skipped_geometry_count=2,
    )

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(fake_sdata)

    widget.shape_cards[0].add_update_button.click()

    _assert_action_feedback_card(widget, title="Shapes Layer Created With Warning", kind="warning")
    assert "point-radius shapes as napari points" not in widget.global_action_feedback_label.text()
    assert "Skipped 2 empty, invalid, or unsupported geometries" in widget.global_action_feedback_label.text()


def test_viewer_widget_shares_app_state_for_same_viewer(qtbot) -> None:
    viewer = DummyViewer()
    first = ViewerWidget(viewer)
    second = ViewerWidget(viewer)

    qtbot.addWidget(first)
    qtbot.addWidget(second)

    assert first.app_state is second.app_state
