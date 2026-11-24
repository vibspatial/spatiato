from __future__ import annotations

import warnings
from collections.abc import Callable, Iterator
from dataclasses import replace
from html import unescape
from types import SimpleNamespace

import anndata as ad
import bermuda
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from _shapes_regression_fixtures import (
    TRIANGULATION_REGRESSION_FAILED_MOVE_COORDINATE,
    TRIANGULATION_REGRESSION_HOLE_ANCHOR_INDICES,
    TRIANGULATION_REGRESSION_MOVED_VERTEX_INDEX,
    make_concave_simple_polygon_deletion_regression_vertices,
    make_triangulation_regression_pre_drag_vertices,
)
from matplotlib.colors import to_rgba
from napari.layers import Image, Points, Shapes
from napari.layers.base._base_constants import ActionType
from napari.layers.shapes._shapes_constants import Mode
from napari.utils.key_bindings import KeymapHandler, coerce_keybinding
from napari_builtins.io import csv_to_layer_data, napari_write_shapes
from qtpy.QtWidgets import QComboBox
from shapely.geometry import Polygon
from spatialdata import SpatialData, read_zarr
from spatialdata.models import ShapesModel, TableModel
from spatialdata.transformations import Identity, set_transformation

import spatiato._app_state as app_state_module
import spatiato.core.shapes_geometry as shapes_geometry_module
import spatiato.widgets.annotation.widget as annotation_widget_module
import spatiato.widgets.shapes_annotation._edit_guard as shapes_annotation_edit_guard_module
import spatiato.widgets.shapes_annotation._identity_feature_defaults as shapes_annotation_identity_defaults_module
import spatiato.widgets.shapes_annotation._layer_state as shapes_annotation_layer_state_module
import spatiato.widgets.shapes_annotation.widget as shapes_annotation_widget_module
from spatiato._app_state import (
    ShapesElementReloadedEvent,
    ShapesElementWrittenEvent,
    get_or_create_app_state,
)
from spatiato._shapes_triangulation import configure_shapes_triangulation_backend
from spatiato.core._color_source import ShapeColumnColorSourceSpec
from spatiato.core.shapes_annotation import AnnotateShapesElementResult
from spatiato.core.shapes_geometry import (
    NapariPolygonVertexDeletion,
    delete_napari_polygon_vertex,
    insert_napari_polygon_vertex,
    napari_polygon_vertices_to_shapely_polygon,
    napari_polygon_vertices_to_topology,
    shapely_polygon_to_napari_polygon_vertices,
)
from spatiato.core.spatialdata_io import write_shapes_element
from spatiato.viewer.adapter import ShapesLayerBinding
from spatiato.viewer.shapes_styling import (
    _SHAPES_EDGE_COLOR_SYNC_CALLBACK_ATTR,
    _SHAPES_EDGE_WIDTH_SYNC_CALLBACK_ATTR,
    PRIMARY_SHAPES_FACE_COLOR,
    apply_primary_shapes_layer_style,
)
from spatiato.widgets.annotation.models import AnnotationContext
from spatiato.widgets.annotation.widget import AnnotationWidget
from spatiato.widgets.shapes_annotation.widget import ShapesAnnotation

_SPACE_PAN_TIP_TEXT = (
    "Tip: while drawing in polygon, path, polyline or lasso mode, hold Space and drag to pan without ending the shape."
)


def _rgba(color: str) -> np.ndarray:
    return np.asarray(to_rgba(color), dtype=float)


class DummyEventEmitter:
    def __init__(self) -> None:
        self._callbacks: list[Callable[[object], None]] = []

    def connect(self, callback: Callable[[object], None]) -> None:
        self._callbacks.append(callback)

    def emit(self, value: object | None = None) -> None:
        event = SimpleNamespace(value=value)
        for callback in list(self._callbacks):
            callback(event)


class DummySelection:
    def __init__(self) -> None:
        self.events = SimpleNamespace(active=DummyEventEmitter())
        self._active: object | None = None

    @property
    def active(self) -> object | None:
        return self._active

    @active.setter
    def active(self, layer: object | None) -> None:
        self._active = layer
        self.events.active.emit(layer)

    def select_only(self, layer: object) -> None:
        self.active = layer


class ProxyActiveDummySelection(DummySelection):
    @DummySelection.active.setter
    def active(self, layer: object | None) -> None:
        self._active = None if layer is None else SimpleNamespace(__wrapped__=layer)
        self.events.active.emit(layer)


class DummyLayers(list):
    def __init__(self, *, selection: DummySelection | None = None) -> None:
        super().__init__()
        self.selection = DummySelection() if selection is None else selection
        self.events = SimpleNamespace(
            inserted=DummyEventEmitter(),
            removed=DummyEventEmitter(),
            reordered=DummyEventEmitter(),
        )

    def remove(self, layer: object) -> None:
        super().remove(layer)
        self.events.removed.emit(layer)


class DummyViewer:
    def __init__(self) -> None:
        self.layers = DummyLayers()

    def add_layer(self, layer: object) -> object:
        self.layers.append(layer)
        self.layers.events.inserted.emit(layer)
        return layer


class AutoActivatingDummyViewer(DummyViewer):
    def add_layer(self, layer: object) -> object:
        self.layers.append(layer)
        self.layers.events.inserted.emit(layer)
        self.layers.selection.active = layer
        return layer


class ProxyActiveAutoActivatingDummyViewer(AutoActivatingDummyViewer):
    def __init__(self) -> None:
        self.layers = DummyLayers(selection=ProxyActiveDummySelection())


def _combo_texts(combo: QComboBox) -> list[str]:
    return [combo.itemText(index) for index in range(combo.count())]


def _combo_data(combo: QComboBox) -> list[object]:
    return [combo.itemData(index) for index in range(combo.count())]


def _combo_index_for_text(combo: QComboBox, text: str) -> int:
    for index in range(combo.count()):
        if combo.itemText(index) == text:
            return index
    return -1


def _status_text(widget: ShapesAnnotation) -> str:
    return unescape(widget.status_label.text())


def _clean_tooltip_text(text: str) -> str:
    return unescape(text).replace("&#8203;", "").replace("\u200b", "")


def _tooltip_text(widget: ShapesAnnotation) -> str:
    return _clean_tooltip_text(widget.status_label.toolTip())


def _first_current_property_value(layer: Shapes, feature_name: str) -> object:
    values = np.asarray(layer.current_properties[feature_name], dtype=object).ravel()
    assert len(values) == 1
    return values[0]


def _assert_identity_feature_default_missing(layer: Shapes, feature_name: str) -> None:
    assert pd.isna(_first_current_property_value(layer, feature_name))
    assert feature_name in layer.feature_defaults.columns
    assert pd.isna(layer.feature_defaults[feature_name].iloc[0])


def _assert_layer_data_unchanged(layer: Shapes, expected_data: list[np.ndarray]) -> None:
    assert len(layer.data) == len(expected_data)
    for actual_vertices, expected_vertices in zip(layer.data, expected_data, strict=True):
        np.testing.assert_allclose(np.asarray(actual_vertices, dtype=float), expected_vertices)


def _assert_shapes_layer_matches_baseline(
    layer: Shapes,
    baseline: shapes_annotation_layer_state_module._ShapesLayerBaseline,
) -> None:
    assert len(layer.data) == len(baseline.data)
    for actual_vertices, expected_vertices in zip(layer.data, baseline.data, strict=True):
        np.testing.assert_array_equal(actual_vertices, expected_vertices)
    assert tuple(layer.shape_type) == baseline.shape_types
    pd.testing.assert_frame_equal(layer.features, baseline.features)
    pd.testing.assert_frame_equal(layer.feature_defaults, baseline.feature_defaults)
    np.testing.assert_allclose(layer.edge_color, baseline.style.edge_color)
    np.testing.assert_allclose(layer.face_color, baseline.style.face_color)
    assert tuple(layer.edge_width) == baseline.style.edge_width
    assert tuple(layer.z_index) == baseline.style.z_index
    assert layer.opacity == baseline.style.opacity
    np.testing.assert_allclose(to_rgba(layer.current_edge_color), to_rgba(baseline.style.current_edge_color))
    np.testing.assert_allclose(to_rgba(layer.current_face_color), to_rgba(baseline.style.current_face_color))
    assert layer.current_edge_width == baseline.style.current_edge_width
    assert layer.mode == baseline.mode
    assert frozenset(layer.selected_data) == baseline.selected_data
    assert len(layer._data_view.shapes) == len(baseline.data)


def _patch_coordinate_system_names(monkeypatch, coordinate_systems: list[str]) -> None:
    monkeypatch.setattr(
        annotation_widget_module,
        "get_coordinate_system_names_from_sdata",
        lambda sdata: coordinate_systems,
    )
    monkeypatch.setattr(
        shapes_annotation_widget_module,
        "get_coordinate_system_names_from_sdata",
        lambda sdata: coordinate_systems,
    )
    monkeypatch.setattr(
        app_state_module,
        "get_coordinate_system_names_from_sdata",
        lambda sdata: coordinate_systems,
    )


def _create_embedded_shapes_annotation(qtbot, viewer: DummyViewer | None = None) -> ShapesAnnotation:
    parent = AnnotationWidget(viewer)
    qtbot.addWidget(parent)
    child = parent.shapes_annotation
    # Existing edit-behavior tests keep the child as their subject while
    # reaching parent-owned selectors through this explicit test-only handle.
    child._test_parent = parent
    return child


def _create_ready_annotation_widget(qtbot, viewer: DummyViewer, sdata: SpatialData) -> ShapesAnnotation:
    app_state = get_or_create_app_state(viewer)
    app_state.set_sdata(sdata)
    widget = _create_embedded_shapes_annotation(qtbot, viewer)
    widget.name_edit.setText("new_regions")
    return widget


def _add_polygon(layer: Shapes, offset: float = 0.0) -> None:
    layer.add_polygons(
        np.asarray(
            [
                [offset + 0.0, 0.0],
                [offset + 0.0, 2.0],
                [offset + 2.0, 2.0],
                [offset + 2.0, 0.0],
            ],
            dtype=float,
        )
    )


def _native_polygon_layer(name: str, *, affine: np.ndarray | None = None) -> Shapes:
    return Shapes(
        [
            np.asarray(
                [
                    [0.0, 0.0],
                    [0.0, 2.0],
                    [2.0, 2.0],
                    [2.0, 0.0],
                ],
                dtype=float,
            )
        ],
        shape_type="polygon",
        affine=affine,
        name=name,
    )


def _register_shapes_candidate_layer(
    widget: ShapesAnnotation,
    sdata: SpatialData,
    *,
    shapes_name: str = "blobs_polygons",
    coordinate_system: str | None = "global",
    layer: Shapes | Points | None = None,
    shapes_role: str = "primary",
    shapes_rendering_mode: str = "shapes",
    style_spec: object | None = None,
) -> Shapes | Points:
    layer = Shapes([], ndim=2, name=shapes_name) if layer is None else layer
    widget.app_state.viewer_adapter.register_shapes_layer(
        layer,
        sdata=sdata,
        shapes_name=shapes_name,
        coordinate_system=coordinate_system,
        shapes_role=shapes_role,
        shapes_rendering_mode=shapes_rendering_mode,
        style_spec=style_spec,
    )
    return layer


def _yx_to_xy(coordinates_yx: np.ndarray) -> list[tuple[float, float]]:
    return [(float(x), float(y)) for y, x in np.asarray(coordinates_yx, dtype=float)]


def _polygon_hole_roundtrip_fixture() -> tuple[Polygon, Polygon]:
    center_y = 1000.0
    center_x = 2000.0

    polygon_1_shell_yx = np.array(
        [
            [center_y - 350, center_x - 280],
            [center_y - 420, center_x + 120],
            [center_y - 120, center_x + 320],
            [center_y + 180, center_x + 140],
            [center_y + 120, center_x - 240],
        ],
        dtype=float,
    )
    polygon_1_hole_yx = np.array(
        [
            [center_y - 150, center_x - 40],
            [center_y - 170, center_x + 70],
            [center_y - 80, center_x + 130],
            [center_y - 10, center_x + 40],
            [center_y - 50, center_x - 70],
        ],
        dtype=float,
    )
    polygon_2_yx = np.array(
        [
            [center_y + 260, center_x - 40],
            [center_y + 180, center_x + 260],
            [center_y + 420, center_x + 340],
            [center_y + 520, center_x + 40],
            [center_y + 360, center_x - 180],
        ],
        dtype=float,
    )

    polygon_1 = Polygon(
        _yx_to_xy(polygon_1_shell_yx),
        holes=[_yx_to_xy(polygon_1_hole_yx)],
    )
    polygon_2 = Polygon(_yx_to_xy(polygon_2_yx))
    assert polygon_1.is_valid
    assert polygon_2.is_valid
    assert len(polygon_1.interiors) == 1
    # When polygon_1 is encoded for napari, the flat path is:
    # shell[0:5] + shell[0] + hole[0:5] + hole[0] + shell[0].
    # The shell anchor/separator copies are shell[0]; the hole anchor
    # copies are hole[0].
    return polygon_1, polygon_2


def _direct_drag_callback_selecting_vertex(*, moved_vertex_index: int) -> Callable[[Shapes, object], object]:
    """Return a native-press stand-in that must remain suspended on moves."""

    def direct_drag_callback(layer: Shapes, event: object) -> object:
        layer._moving_value = (0, moved_vertex_index)
        layer._drag_start = np.zeros(layer.ndim, dtype=float)
        layer.selected_data = {0}
        yield "press"
        if getattr(event, "type", None) == "mouse_move":
            raise AssertionError("Spatiato resumed the native generator for a guarded polygon move.")

    return direct_drag_callback


def _exhaust_generator(iterator: Iterator[object]) -> list[object]:
    values: list[object] = []
    while True:
        try:
            values.append(next(iterator))
        except StopIteration:
            return values


def _install_direct_drag_callback_for_annotation_guard(
    widget: ShapesAnnotation,
    layer: Shapes,
    *,
    moved_vertex_index: int,
) -> None:
    widget._annotation_edit_guard.disconnect()
    layer._drag_modes = dict(layer._drag_modes)
    layer._drag_modes[Mode.DIRECT] = _direct_drag_callback_selecting_vertex(moved_vertex_index=moved_vertex_index)
    widget._annotation_edit_guard.attach(layer)
    assert layer._drag_modes[Mode.DIRECT] is widget._annotation_edit_guard._wrapped_direct_callback


def _drag_annotation_vertex(
    layer: Shapes,
    *,
    vertex_index: int,
    moved_coordinate: np.ndarray,
) -> np.ndarray:
    event = SimpleNamespace(type="mouse_press", position=tuple(layer.data[0][vertex_index]))
    drag = layer._drag_modes[Mode.DIRECT](layer, event)
    assert next(drag) == "press"
    event.type = "mouse_move"
    event.position = tuple(moved_coordinate)
    assert next(drag) is None
    event.type = "mouse_release"
    with pytest.raises(StopIteration):
        next(drag)
    event.type = "mouse_release"
    try:
        next(drag)
    except StopIteration:
        pass
    edited_vertices = np.asarray(layer.data[0], dtype=float)
    np.testing.assert_allclose(edited_vertices[vertex_index], moved_coordinate)
    return edited_vertices


def _make_polygon_hole_roundtrip_sdata(*, shapes_name: str = "hole_regions") -> SpatialData:
    polygon_1, polygon_2 = _polygon_hole_roundtrip_fixture()
    geodataframe = gpd.GeoDataFrame(
        {"label": ["polygon_with_hole", "simple_polygon"], "score": [1.25, 2.5]},
        geometry=[polygon_1, polygon_2],
        index=pd.Index(["hole_row", "simple_row"], name="region_id"),
    )
    shapes = ShapesModel.parse(geodataframe, transformations={"global": Identity()})
    return SpatialData(shapes={shapes_name: shapes})


def _make_generated_annotation_ids_sdata(*, shapes_name: str = "generated_regions") -> SpatialData:
    polygons = [
        Polygon([(0, 0), (0, 2), (2, 2), (2, 0)]),
        Polygon([(5, 5), (5, 7), (7, 7), (7, 5)]),
    ]
    geodataframe = gpd.GeoDataFrame(
        {"label": ["existing_0", "existing_1"]},
        geometry=polygons,
        index=pd.Index(["__annotation_0", "__annotation_1"], name="instance_id"),
    )
    shapes = ShapesModel.parse(geodataframe, transformations={"global": Identity()})
    return SpatialData(shapes={shapes_name: shapes})


def _make_create_holes_sdata(
    *,
    shapes_name: str = "create_holes_regions",
) -> tuple[SpatialData, Polygon, Polygon, Polygon]:
    shell = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    child = Polygon([(2, 2), (2, 4), (4, 4), (4, 2)])
    unselected = Polygon([(20, 20), (20, 24), (24, 24), (24, 20)])
    geodataframe = gpd.GeoDataFrame(
        {"label": ["shell", "child", "unselected"], "score": [1.0, 2.0, 3.0]},
        geometry=[shell, child, unselected],
        index=pd.Index(["shell_row", "child_row", "unselected_row"], name="region_id"),
    )
    shapes = ShapesModel.parse(geodataframe, transformations={"global": Identity()})
    return SpatialData(shapes={shapes_name: shapes}), shell, child, unselected


def _assert_polygon_hole_geometries_preserved(saved: gpd.GeoDataFrame, polygon_1: Polygon, polygon_2: Polygon) -> None:
    saved_polygon_1 = saved.geometry.iloc[0]
    saved_polygon_2 = saved.geometry.iloc[1]

    assert saved_polygon_1.equals(polygon_1)
    assert len(saved_polygon_1.interiors) == 1
    assert saved_polygon_1.area == polygon_1.area
    assert saved_polygon_1.bounds == polygon_1.bounds
    assert saved_polygon_2.equals(polygon_2)
    assert len(saved_polygon_2.interiors) == 0


def _add_dummy_table_annotating_shapes(sdata: SpatialData, *, shapes_name: str, table_name: str) -> ad.AnnData:
    shapes = sdata.shapes[shapes_name]
    index_values = shapes.index.to_list()
    cell_types = np.resize(np.asarray(["T", "B"], dtype=object), len(index_values))
    table = TableModel.parse(
        ad.AnnData(
            obs=pd.DataFrame(
                {
                    "region": [shapes_name] * len(index_values),
                    "index": index_values,
                    "cell_type": pd.Categorical(cell_types, categories=["T", "B"]),
                },
                index=[f"obs_{index}" for index in index_values],
            )
        ),
        region=shapes_name,
        region_key="region",
        instance_key="index",
    )
    sdata.tables[table_name] = table
    return table


def test_shapes_annotation_child_constructs_without_shared_selectors_or_transition_participation(qtbot) -> None:
    viewer = DummyViewer()
    widget = ShapesAnnotation(viewer)
    qtbot.addWidget(widget)

    assert widget.selected_spatialdata is None
    assert widget.selected_coordinate_system is None
    assert not hasattr(widget, "coordinate_system_combo")
    assert not hasattr(widget, "shapes_combo")
    assert widget.app_state._coordinate_system_change_participant is None


def test_shapes_annotation_reload_is_disabled_without_an_existing_shapes_selection(
    qtbot,
    backed_sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer()
    app_state = get_or_create_app_state(viewer)
    app_state.set_sdata(backed_sdata_blobs)
    widget = _create_embedded_shapes_annotation(qtbot, viewer)

    assert widget._test_parent.annotation_context.shapes_target == (
        shapes_annotation_widget_module._ShapesAnnotationTarget.create_new()
    )
    assert widget.reload_shapes_button.isEnabled() is False

    widget._test_parent.shapes_combo.setCurrentIndex(-1)

    assert widget._test_parent.annotation_context.shapes_target is None
    assert widget.reload_shapes_button.isEnabled() is False


def test_shapes_annotation_child_reapplies_context_for_active_create_new_session(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer()
    widget = _create_ready_annotation_widget(qtbot, viewer, sdata_blobs)
    widget.create_layer_button.click()
    layer = widget._annotation_layer
    context = widget._test_parent.annotation_context

    widget.apply_annotation_context(context)

    assert widget._annotation_layer is layer
    assert widget._annotation_session is not None
    assert widget._annotation_session.mode == "create_new"


def test_shapes_annotation_child_rejects_context_for_another_coordinate_system(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer()
    widget = _create_ready_annotation_widget(qtbot, viewer, sdata_blobs)
    context = replace(widget._test_parent.annotation_context, coordinate_system="another-coordinate-system")

    with pytest.raises(ValueError, match="coordinate system must match the shared app state"):
        widget.apply_annotation_context(context)


def test_shapes_annotation_widget_connects_to_napari_active_layer_event(qtbot) -> None:
    viewer = DummyViewer()
    widget = _create_embedded_shapes_annotation(qtbot, viewer)

    callbacks = viewer.layers.selection.events.active._callbacks

    assert any(
        getattr(callback, "__self__", None) is widget
        and getattr(callback, "__name__", "") == "_on_active_layer_changed"
        for callback in callbacks
    )


def test_shapes_annotation_widget_active_layer_event_routes_to_placeholder(qtbot, monkeypatch) -> None:
    viewer = DummyViewer()
    widget = _create_embedded_shapes_annotation(qtbot, viewer)
    layer = Shapes([], ndim=2)
    routed_layers: list[object] = []
    monkeypatch.setattr(widget, "_maybe_adopt_active_shapes_layer", routed_layers.append)

    viewer.layers.selection.active = layer

    assert routed_layers == [layer]


def test_shapes_annotation_widget_active_layer_event_ignores_current_layer_and_reentrant_calls(
    qtbot, monkeypatch
) -> None:
    viewer = DummyViewer()
    widget = _create_embedded_shapes_annotation(qtbot, viewer)
    layer = Shapes([], ndim=2)
    routed_layers: list[object] = []
    monkeypatch.setattr(widget, "_maybe_adopt_active_shapes_layer", routed_layers.append)

    widget._annotation_layer = layer
    widget._on_active_layer_changed(SimpleNamespace(value=layer))
    widget._annotation_layer = None
    widget._is_handling_active_layer_change = True
    widget._on_active_layer_changed(SimpleNamespace(value=layer))

    assert routed_layers == []


def test_shapes_annotation_widget_active_primary_shapes_candidate_accepts_compatible_layer(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer()
    widget = _create_ready_annotation_widget(qtbot, viewer, sdata_blobs)
    layer = _register_shapes_candidate_layer(widget, sdata_blobs)

    candidate = widget._active_primary_shapes_candidate(layer)

    assert candidate is not None
    assert candidate.layer is layer
    assert candidate.shapes_name == "blobs_polygons"
    assert candidate.coordinate_system == "global"


def test_shapes_annotation_widget_active_primary_shapes_candidate_rejects_incompatible_layers(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer()
    widget = _create_ready_annotation_widget(qtbot, viewer, sdata_blobs)
    style_spec = ShapeColumnColorSourceSpec(
        source_kind="shape_column",
        value_key="label",
    )

    assert widget._active_primary_shapes_candidate(Image(np.zeros((2, 2)))) is None
    assert widget._active_primary_shapes_candidate(Shapes([], ndim=2)) is None

    styled_layer = _register_shapes_candidate_layer(
        widget,
        sdata_blobs,
        layer=Shapes([], ndim=2),
        shapes_role="styled",
        style_spec=style_spec,
    )
    points_layer = _register_shapes_candidate_layer(
        widget,
        sdata_blobs,
        layer=Points(np.empty((0, 2))),
        shapes_rendering_mode="points",
    )
    other_coordinate_layer = _register_shapes_candidate_layer(
        widget,
        sdata_blobs,
        layer=Shapes([], ndim=2),
        coordinate_system="other",
    )
    other_sdata_layer = _register_shapes_candidate_layer(
        widget,
        _make_polygon_hole_roundtrip_sdata(shapes_name="blobs_polygons"),
        layer=Shapes([], ndim=2),
    )
    ineligible_name_layer = _register_shapes_candidate_layer(
        widget,
        sdata_blobs,
        shapes_name="not_eligible",
        layer=Shapes([], ndim=2),
    )

    assert widget._active_primary_shapes_candidate(styled_layer) is None
    assert widget._active_primary_shapes_candidate(points_layer) is None
    assert widget._active_primary_shapes_candidate(other_coordinate_layer) is None
    assert widget._active_primary_shapes_candidate(other_sdata_layer) is None
    assert widget._active_primary_shapes_candidate(ineligible_name_layer) is None


def test_shapes_annotation_widget_active_primary_shapes_layer_selection_opens_annotation_session(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer()
    app_state = get_or_create_app_state(viewer)
    app_state.set_sdata(sdata_blobs)
    widget = _create_embedded_shapes_annotation(qtbot, viewer)
    load_result = app_state.viewer_adapter.ensure_shapes_loaded(sdata_blobs, "blobs_polygons", "global")
    layer = load_result.layer

    assert widget._annotation_layer is None

    # Simulate the user selecting this layer in napari's layer list.
    app_state.viewer_adapter.activate_layer(layer)

    assert widget._annotation_layer is layer
    assert widget._annotation_edit_guard.layer is layer
    assert "_drag_modes" in vars(layer)
    assert layer._drag_modes[Mode.DIRECT] is widget._annotation_edit_guard._wrapped_direct_callback
    assert widget._annotation_session is not None
    assert widget._annotation_session.mode == "edit_existing"
    assert widget._annotation_session.shapes_name == "blobs_polygons"
    assert widget._annotation_session.coordinate_system == "global"
    assert (
        widget._annotation_context.shapes_target
        == shapes_annotation_widget_module._ShapesAnnotationTarget.edit_existing("blobs_polygons")
    )
    assert widget._test_parent.shapes_combo.currentText() == "blobs_polygons"
    assert widget.save_shapes_button.isEnabled() is False
    assert widget.create_holes_button.isEnabled() is True
    assert "Existing Shapes Opened" in _status_text(widget)


def test_shapes_annotation_widget_adopts_auto_active_primary_shapes_after_registration(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = AutoActivatingDummyViewer()
    app_state = get_or_create_app_state(viewer)
    app_state.set_sdata(sdata_blobs)
    widget = _create_embedded_shapes_annotation(qtbot, viewer)

    assert widget._annotation_layer is None
    assert widget._test_parent.shapes_combo.currentText() == "Create shapes..."

    load_result = app_state.viewer_adapter.ensure_shapes_loaded(sdata_blobs, "blobs_polygons", "global")
    layer = load_result.layer

    assert load_result.created is True
    assert viewer.layers.selection.active is layer
    assert widget._annotation_layer is layer
    assert widget._annotation_edit_guard.layer is layer
    assert widget._annotation_session is not None
    assert widget._annotation_session.shapes_name == "blobs_polygons"
    assert widget._test_parent.shapes_combo.currentText() == "blobs_polygons"
    assert widget.save_shapes_button.isEnabled() is False
    assert widget.create_holes_button.isEnabled() is True


def test_shapes_annotation_widget_adopts_proxy_active_primary_shapes_after_registration(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = ProxyActiveAutoActivatingDummyViewer()
    app_state = get_or_create_app_state(viewer)
    app_state.set_sdata(sdata_blobs)
    widget = _create_embedded_shapes_annotation(qtbot, viewer)

    load_result = app_state.viewer_adapter.ensure_shapes_loaded(sdata_blobs, "blobs_polygons", "global")
    layer = load_result.layer

    assert getattr(viewer.layers.selection.active, "__wrapped__", None) is layer
    assert widget._annotation_layer is layer
    assert widget._annotation_edit_guard.layer is layer
    assert widget._annotation_session is not None
    assert widget._annotation_session.shapes_name == "blobs_polygons"
    assert widget._test_parent.shapes_combo.currentText() == "blobs_polygons"


def test_shapes_annotation_widget_space_pan_predicate_requires_active_widget_owned_layer(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = ProxyActiveAutoActivatingDummyViewer()
    widget = _create_ready_annotation_widget(qtbot, viewer, sdata_blobs)
    widget.create_layer_button.click()
    layer = widget._annotation_layer
    assert isinstance(layer, Shapes)
    assert getattr(viewer.layers.selection.active, "__wrapped__", None) is layer

    assert widget._can_annotation_layer_space_pan_draw(layer) is True

    other_layer = Shapes([], ndim=2)
    viewer.layers.selection._active = SimpleNamespace(__wrapped__=other_layer)

    assert widget._can_annotation_layer_space_pan_draw(layer) is False
    assert widget._can_annotation_layer_space_pan_draw(other_layer) is False

    viewer.layers.selection._active = SimpleNamespace(__wrapped__=layer)
    widget.app_state.viewer_adapter.unregister_layer(layer)

    assert widget._can_annotation_layer_space_pan_draw(layer) is False


def test_shapes_annotation_widget_space_key_falls_back_when_annotation_layer_is_inactive(
    qtbot,
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    viewer = AutoActivatingDummyViewer()
    widget = _create_ready_annotation_widget(qtbot, viewer, sdata_blobs)
    widget.create_layer_button.click()
    layer = widget._annotation_layer
    assert isinstance(layer, Shapes)
    assert viewer.layers.selection.active is layer
    layer.mode = Mode.ADD_POLYGON
    layer._is_creating = True
    guard = widget._annotation_edit_guard
    fallback_layers: list[Shapes] = []

    def fallback_pan_zoom_key_hold(fallback_layer: Shapes) -> Iterator[None]:
        fallback_layers.append(fallback_layer)
        yield

    monkeypatch.setattr(guard, "_fallback_pan_zoom_key_hold", fallback_pan_zoom_key_hold)
    viewer.layers.selection._active = Shapes([], ndim=2)
    handler = KeymapHandler()
    handler.keymap_providers = [layer]

    assert handler.press_key("Space") is True

    assert fallback_layers == [layer]
    assert guard._space_pan_key_held is False
    assert guard._space_pan_mouse_gesture_active is False
    assert guard._previous_mouse_pan is None

    assert handler.release_key("Space") is True

    assert guard._space_pan_key_held is False
    assert guard._space_pan_mouse_gesture_active is False
    assert guard._previous_mouse_pan is None


def test_shapes_annotation_widget_primary_shapes_registration_ignores_inactive_layer(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer()
    app_state = get_or_create_app_state(viewer)
    app_state.set_sdata(sdata_blobs)
    widget = _create_embedded_shapes_annotation(qtbot, viewer)

    load_result = app_state.viewer_adapter.ensure_shapes_loaded(sdata_blobs, "blobs_polygons", "global")

    assert load_result.created is True
    assert viewer.layers.selection.active is None
    assert widget._annotation_layer is None
    assert widget._annotation_session is None
    assert widget._test_parent.shapes_combo.currentText() == "Create shapes..."


def test_shapes_annotation_widget_active_registration_ignores_incompatible_primary_shapes(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = AutoActivatingDummyViewer()
    app_state = get_or_create_app_state(viewer)
    app_state.set_sdata(sdata_blobs)
    widget = _create_embedded_shapes_annotation(qtbot, viewer)
    style_spec = ShapeColumnColorSourceSpec(
        source_kind="shape_column",
        value_key="label",
    )

    styled_layer = Shapes([], ndim=2, name="blobs_polygons")
    viewer.add_layer(styled_layer)
    app_state.viewer_adapter.register_shapes_layer(
        styled_layer,
        sdata=sdata_blobs,
        shapes_name="blobs_polygons",
        coordinate_system="global",
        shapes_role="styled",
        style_spec=style_spec,
    )

    assert viewer.layers.selection.active is styled_layer
    assert widget._annotation_layer is None
    assert widget._annotation_session is None

    points_layer = Points(np.empty((0, 2)), name="blobs_polygons")
    viewer.add_layer(points_layer)
    app_state.viewer_adapter.register_shapes_layer(
        points_layer,
        sdata=sdata_blobs,
        shapes_name="blobs_polygons",
        coordinate_system="global",
        shapes_role="primary",
        shapes_rendering_mode="points",
    )

    assert viewer.layers.selection.active is points_layer
    assert widget._annotation_layer is None
    assert widget._annotation_session is None


def test_shapes_annotation_widget_active_primary_shapes_layer_selection_switches_clean_session(
    qtbot,
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    sdata_blobs.shapes["other_polygons"] = sdata_blobs.shapes["blobs_polygons"].copy()
    viewer = DummyViewer()
    app_state = get_or_create_app_state(viewer)
    app_state.set_sdata(sdata_blobs)
    widget = _create_embedded_shapes_annotation(qtbot, viewer)

    widget._test_parent.shapes_combo.setCurrentIndex(
        _combo_index_for_text(widget._test_parent.shapes_combo, "blobs_polygons")
    )
    first_layer = widget._annotation_layer
    assert isinstance(first_layer, Shapes)
    assert widget._annotation_edit_guard.layer is first_layer
    assert "_drag_modes" in vars(first_layer)

    # Clean session switches should close without asking for dirty-session
    # discard confirmation.
    def fail_if_confirmed(*, reason: str) -> bool:
        raise AssertionError(f"Clean active-layer switch should not warn: {reason}")

    monkeypatch.setattr(widget, "_confirm_discard_annotation_layer", fail_if_confirmed)
    other_result = app_state.viewer_adapter.ensure_shapes_loaded(sdata_blobs, "other_polygons", "global")
    other_layer = other_result.layer

    app_state.viewer_adapter.activate_layer(other_layer)

    assert widget._annotation_layer is other_layer
    assert widget._annotation_edit_guard.layer is other_layer
    assert widget._annotation_session is not None
    assert widget._annotation_session.shapes_name == "other_polygons"
    assert (
        widget._annotation_context.shapes_target
        == shapes_annotation_widget_module._ShapesAnnotationTarget.edit_existing("other_polygons")
    )
    assert widget._test_parent.shapes_combo.currentText() == "other_polygons"
    assert list(viewer.layers) == [first_layer, other_layer]
    assert "_drag_modes" not in vars(first_layer)
    assert "_drag_modes" in vars(other_layer)
    assert widget.save_shapes_button.isEnabled() is False
    assert widget.create_holes_button.isEnabled() is True


def test_shapes_annotation_widget_active_primary_shapes_layer_selection_dirty_cancel_keeps_session(
    qtbot,
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    sdata_blobs.shapes["other_polygons"] = sdata_blobs.shapes["blobs_polygons"].copy()
    viewer = DummyViewer()
    app_state = get_or_create_app_state(viewer)
    app_state.set_sdata(sdata_blobs)
    widget = _create_embedded_shapes_annotation(qtbot, viewer)

    widget._test_parent.shapes_combo.setCurrentIndex(
        _combo_index_for_text(widget._test_parent.shapes_combo, "blobs_polygons")
    )
    first_layer = widget._annotation_layer
    assert isinstance(first_layer, Shapes)
    _add_polygon(first_layer, offset=100)
    assert widget._annotation_layer_has_unsaved_changes() is True
    other_result = app_state.viewer_adapter.ensure_shapes_loaded(sdata_blobs, "other_polygons", "global")
    other_layer = other_result.layer
    discard_contexts: list[str] = []

    def cancel_discard(*, reason: str) -> bool:
        discard_contexts.append(reason)
        return False

    monkeypatch.setattr(widget, "_confirm_discard_annotation_layer", cancel_discard)

    app_state.viewer_adapter.activate_layer(other_layer)

    assert discard_contexts == ["shapes_target"]
    qtbot.waitUntil(lambda: viewer.layers.selection.active is first_layer)
    assert viewer.layers.selection.active is first_layer
    assert widget._annotation_layer is first_layer
    assert widget._annotation_edit_guard.layer is first_layer
    assert widget._annotation_session is not None
    assert widget._annotation_session.shapes_name == "blobs_polygons"
    assert (
        widget._annotation_context.shapes_target
        == shapes_annotation_widget_module._ShapesAnnotationTarget.edit_existing("blobs_polygons")
    )
    assert widget._test_parent.shapes_combo.currentText() == "blobs_polygons"
    assert "_drag_modes" in vars(first_layer)
    assert "_drag_modes" not in vars(other_layer)
    assert widget.save_shapes_button.isEnabled() is True
    assert widget.create_holes_button.isEnabled() is True


def test_shapes_annotation_widget_active_primary_shapes_layer_selection_dirty_confirm_adopts_layer(
    qtbot,
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    sdata_blobs.shapes["other_polygons"] = sdata_blobs.shapes["blobs_polygons"].copy()
    viewer = DummyViewer()
    app_state = get_or_create_app_state(viewer)
    app_state.set_sdata(sdata_blobs)
    widget = _create_embedded_shapes_annotation(qtbot, viewer)

    widget._test_parent.shapes_combo.setCurrentIndex(
        _combo_index_for_text(widget._test_parent.shapes_combo, "blobs_polygons")
    )
    first_layer = widget._annotation_layer
    assert isinstance(first_layer, Shapes)
    _add_polygon(first_layer, offset=100)
    assert widget._annotation_layer_has_unsaved_changes() is True
    other_result = app_state.viewer_adapter.ensure_shapes_loaded(sdata_blobs, "other_polygons", "global")
    other_layer = other_result.layer
    discard_contexts: list[str] = []

    def confirm_discard(*, reason: str) -> bool:
        discard_contexts.append(reason)
        return True

    monkeypatch.setattr(widget, "_confirm_discard_annotation_layer", confirm_discard)

    app_state.viewer_adapter.activate_layer(other_layer)

    assert discard_contexts == ["shapes_target"]
    assert viewer.layers.selection.active is other_layer
    assert first_layer not in viewer.layers
    assert widget._annotation_layer is other_layer
    assert widget._annotation_edit_guard.layer is other_layer
    assert widget._annotation_session is not None
    assert widget._annotation_session.shapes_name == "other_polygons"
    assert (
        widget._annotation_context.shapes_target
        == shapes_annotation_widget_module._ShapesAnnotationTarget.edit_existing("other_polygons")
    )
    assert widget._test_parent.shapes_combo.currentText() == "other_polygons"
    assert "_drag_modes" not in vars(first_layer)
    assert "_drag_modes" in vars(other_layer)
    assert widget.save_shapes_button.isEnabled() is False
    assert widget.create_holes_button.isEnabled() is True


def test_annotation_identity_feature_default_guard_reentrant_event_is_ignored() -> None:
    layer = Shapes(
        [
            np.asarray(
                [
                    [0.0, 0.0],
                    [0.0, 2.0],
                    [2.0, 2.0],
                    [2.0, 0.0],
                ],
                dtype=float,
            )
        ],
        shape_type="polygon",
        features=pd.DataFrame({"instance_id": ["__annotation_0"]}),
    )
    guard = shapes_annotation_identity_defaults_module._AnnotationIdentityFeatureDefaultGuard()

    guard.attach(layer, feature_name="instance_id")
    _assert_identity_feature_default_missing(layer, "instance_id")

    guard._is_clearing = True
    layer.current_properties = {"instance_id": np.asarray(["__annotation_0"], dtype=object)}
    guard._is_clearing = False

    assert _first_current_property_value(layer, "instance_id") == "__annotation_0"

    guard._on_identity_feature_default_changed()

    _assert_identity_feature_default_missing(layer, "instance_id")


def test_annotation_identity_feature_default_guard_allows_selecting_stored_and_unsaved_rows() -> None:
    """Select stored and unsaved rows without an ambiguous missing-value comparison."""
    first = np.asarray(
        [[0.0, 0.0], [0.0, 2.0], [2.0, 2.0], [2.0, 0.0]],
        dtype=float,
    )
    layer = Shapes(
        [first],
        shape_type="polygon",
        features=pd.DataFrame({"instance_id": ["__annotation_0"]}),
    )
    guard = shapes_annotation_identity_defaults_module._AnnotationIdentityFeatureDefaultGuard()
    guard.attach(layer, feature_name="instance_id")

    layer.add(first + 4.0, shape_type="polygon")

    assert layer.features["instance_id"].iloc[1] is None
    layer.selected_data = {0}
    layer.selected_data.add(1)

    assert set(layer.selected_data) == {0, 1}
    assert layer.features["instance_id"].tolist() == ["__annotation_0", None]


def test_annotation_layer_edit_guard_delegates_direct_mode_and_restores_instance_mapping() -> None:
    layer = Shapes([], ndim=2)
    layer._drag_modes = dict(layer._drag_modes)
    layer._move_modes = dict(layer._move_modes)
    original_drag_modes = layer._drag_modes
    original_move_modes = layer._move_modes
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def original_direct_callback(*args: object, **kwargs: object) -> str:
        calls.append((args, kwargs))
        return "delegated"

    original_drag_modes[Mode.DIRECT] = original_direct_callback
    original_vertex_remove_callback = original_drag_modes[Mode.VERTEX_REMOVE]
    original_vertex_insert_callback = original_drag_modes[Mode.VERTEX_INSERT]
    original_lasso_move_callback = original_move_modes[Mode.ADD_POLYGON_LASSO]
    original_lasso_drag_callback = original_drag_modes[Mode.ADD_POLYGON_LASSO]
    guard = shapes_annotation_edit_guard_module._AnnotationLayerEditGuard()

    guard.attach(layer)
    wrapped_direct_callback = layer._drag_modes[Mode.DIRECT]
    wrapped_vertex_remove_callback = layer._drag_modes[Mode.VERTEX_REMOVE]
    wrapped_vertex_insert_callback = layer._drag_modes[Mode.VERTEX_INSERT]

    assert guard.layer is layer
    assert layer._drag_modes is not original_drag_modes
    assert layer._move_modes is not original_move_modes
    assert wrapped_direct_callback is not original_direct_callback
    assert wrapped_vertex_remove_callback is not original_vertex_remove_callback
    assert wrapped_vertex_insert_callback is not original_vertex_insert_callback
    assert layer._drag_modes[Mode.ADD_POLYGON_LASSO] is not original_lasso_drag_callback
    assert layer._move_modes[Mode.ADD_POLYGON_LASSO] is not original_lasso_move_callback
    assert wrapped_direct_callback("event", value=3) == "delegated"
    assert calls == [(("event",), {"value": 3})]

    guard.disconnect()

    assert guard.layer is None
    assert layer._drag_modes is original_drag_modes
    assert layer._move_modes is original_move_modes
    assert layer._drag_modes[Mode.DIRECT] is original_direct_callback
    assert layer._drag_modes[Mode.VERTEX_REMOVE] is original_vertex_remove_callback
    assert layer._drag_modes[Mode.VERTEX_INSERT] is original_vertex_insert_callback
    assert layer._drag_modes[Mode.ADD_POLYGON_LASSO] is original_lasso_drag_callback
    assert layer._move_modes[Mode.ADD_POLYGON_LASSO] is original_lasso_move_callback


def test_annotation_layer_edit_guard_attach_is_idempotent_and_restores_class_mapping() -> None:
    layer = Shapes([], ndim=2)
    original_direct_callback = layer._drag_modes[Mode.DIRECT]
    original_vertex_remove_callback = layer._drag_modes[Mode.VERTEX_REMOVE]
    original_vertex_insert_callback = layer._drag_modes[Mode.VERTEX_INSERT]
    original_lasso_move_callback = layer._move_modes[Mode.ADD_POLYGON_LASSO]
    original_lasso_drag_callback = layer._drag_modes[Mode.ADD_POLYGON_LASSO]
    guard = shapes_annotation_edit_guard_module._AnnotationLayerEditGuard()

    guard.attach(layer)
    first_wrapped_direct_callback = layer._drag_modes[Mode.DIRECT]
    first_wrapped_vertex_remove_callback = layer._drag_modes[Mode.VERTEX_REMOVE]
    first_wrapped_vertex_insert_callback = layer._drag_modes[Mode.VERTEX_INSERT]
    first_move_modes = layer._move_modes
    guard.attach(layer)

    assert layer._drag_modes[Mode.DIRECT] is first_wrapped_direct_callback
    assert layer._drag_modes[Mode.VERTEX_REMOVE] is first_wrapped_vertex_remove_callback
    assert layer._drag_modes[Mode.VERTEX_INSERT] is first_wrapped_vertex_insert_callback
    assert layer._move_modes is first_move_modes
    assert layer._drag_modes[Mode.ADD_POLYGON_LASSO] is not original_lasso_drag_callback
    assert layer._move_modes[Mode.ADD_POLYGON_LASSO] is not original_lasso_move_callback
    assert "_drag_modes" in vars(layer)
    assert "_move_modes" in vars(layer)

    guard.disconnect()

    assert guard.layer is None
    assert "_drag_modes" not in vars(layer)
    assert "_move_modes" not in vars(layer)
    assert layer._drag_modes[Mode.DIRECT] is original_direct_callback
    assert layer._drag_modes[Mode.VERTEX_REMOVE] is original_vertex_remove_callback
    assert layer._drag_modes[Mode.VERTEX_INSERT] is original_vertex_insert_callback
    assert layer._drag_modes[Mode.ADD_POLYGON_LASSO] is original_lasso_drag_callback
    assert layer._move_modes[Mode.ADD_POLYGON_LASSO] is original_lasso_move_callback


def test_annotation_layer_edit_guard_replacing_layer_disconnects_previous_layer() -> None:
    first_layer = Shapes([], ndim=2)
    second_layer = Shapes([], ndim=2)
    first_direct_callback = first_layer._drag_modes[Mode.DIRECT]
    first_vertex_remove_callback = first_layer._drag_modes[Mode.VERTEX_REMOVE]
    first_vertex_insert_callback = first_layer._drag_modes[Mode.VERTEX_INSERT]
    first_lasso_move_callback = first_layer._move_modes[Mode.ADD_POLYGON_LASSO]
    first_lasso_drag_callback = first_layer._drag_modes[Mode.ADD_POLYGON_LASSO]
    guard = shapes_annotation_edit_guard_module._AnnotationLayerEditGuard()

    guard.attach(first_layer)
    first_wrapped_direct_callback = first_layer._drag_modes[Mode.DIRECT]
    first_wrapped_vertex_remove_callback = first_layer._drag_modes[Mode.VERTEX_REMOVE]
    first_wrapped_vertex_insert_callback = first_layer._drag_modes[Mode.VERTEX_INSERT]
    first_move_modes = first_layer._move_modes
    # `attach(...)` first calls `disconnect(...)`, so moving the guard to a new
    # layer must restore the previous layer before patching the new one.
    guard.attach(second_layer)

    assert guard.layer is second_layer
    assert first_layer._drag_modes[Mode.DIRECT] is first_direct_callback
    assert first_layer._drag_modes[Mode.VERTEX_REMOVE] is first_vertex_remove_callback
    assert first_layer._drag_modes[Mode.VERTEX_INSERT] is first_vertex_insert_callback
    assert first_layer._drag_modes[Mode.ADD_POLYGON_LASSO] is first_lasso_drag_callback
    assert first_layer._move_modes[Mode.ADD_POLYGON_LASSO] is first_lasso_move_callback
    assert "_drag_modes" not in vars(first_layer)
    assert "_move_modes" not in vars(first_layer)
    assert "_drag_modes" in vars(second_layer)
    assert "_move_modes" in vars(second_layer)
    assert second_layer._drag_modes[Mode.DIRECT] is not first_wrapped_direct_callback
    assert second_layer._drag_modes[Mode.VERTEX_REMOVE] is not first_wrapped_vertex_remove_callback
    assert second_layer._drag_modes[Mode.VERTEX_INSERT] is not first_wrapped_vertex_insert_callback
    assert second_layer._move_modes is not first_move_modes


def test_annotation_layer_edit_guard_space_binding_preserves_existing_binding_and_mouse_state() -> None:
    layer = Shapes([], ndim=2)
    layer.mode = Mode.ADD_POLYGON_LASSO
    space_key = coerce_keybinding("Space")

    def existing_space_keybinding(_layer: Shapes) -> None:
        return None

    layer.bind_key("Space", existing_space_keybinding, overwrite=True)
    original_keymap = dict(layer.keymap)
    original_mouse_pan = layer.mouse_pan
    original_mode = layer.mode
    original_drag_callbacks = list(layer.mouse_drag_callbacks)
    original_move_callbacks = list(layer.mouse_move_callbacks)
    original_resumable_drag_callbacks = {
        mode: layer._drag_modes[mode] for mode in shapes_annotation_edit_guard_module._SPACE_PAN_RESUMABLE_DRAW_MODES
    }
    original_resumable_move_callbacks = {
        mode: layer._move_modes[mode] for mode in shapes_annotation_edit_guard_module._SPACE_PAN_RESUMABLE_DRAW_MODES
    }
    guard = shapes_annotation_edit_guard_module._AnnotationLayerEditGuard()

    guard.attach(layer)

    assert "Space" not in layer.keymap
    assert layer.keymap[space_key] is guard._wrapped_space_keybinding
    assert guard._had_instance_space_keybinding is True
    assert guard._previous_space_keybinding is existing_space_keybinding
    assert layer.mouse_pan is original_mouse_pan
    assert layer.mode == original_mode
    assert original_drag_callbacks[0] not in layer.mouse_drag_callbacks
    assert layer._drag_modes[Mode.ADD_POLYGON_LASSO] in layer.mouse_drag_callbacks
    assert original_move_callbacks[0] not in layer.mouse_move_callbacks
    assert layer._move_modes[Mode.ADD_POLYGON_LASSO] in layer.mouse_move_callbacks
    for mode, callback in original_resumable_drag_callbacks.items():
        assert layer._drag_modes[mode] is not callback
    for mode, callback in original_resumable_move_callbacks.items():
        assert layer._move_modes[mode] is not callback

    guard.disconnect()

    assert dict(layer.keymap) == original_keymap
    assert layer.keymap[space_key] is existing_space_keybinding
    assert layer.mouse_pan is original_mouse_pan
    assert layer.mode == original_mode
    assert list(layer.mouse_drag_callbacks) == original_drag_callbacks
    assert list(layer.mouse_move_callbacks) == original_move_callbacks
    for mode, callback in original_resumable_drag_callbacks.items():
        assert layer._drag_modes[mode] is callback
    for mode, callback in original_resumable_move_callbacks.items():
        assert layer._move_modes[mode] is callback


def test_annotation_layer_edit_guard_replaces_active_supported_draw_callbacks() -> None:
    layer = Shapes([], ndim=2)
    layer.mode = Mode.ADD_POLYGON_LASSO
    original_mode = layer.mode
    original_drag_callback = layer.mouse_drag_callbacks[0]
    original_move_callback = layer.mouse_move_callbacks[0]

    def unrelated_drag_callback(_layer: Shapes, _event: object) -> None:
        return None

    def unrelated_move_callback(_layer: Shapes, _event: object) -> None:
        return None

    layer.mouse_drag_callbacks.append(unrelated_drag_callback)
    layer.mouse_move_callbacks.append(unrelated_move_callback)
    guard = shapes_annotation_edit_guard_module._AnnotationLayerEditGuard()

    guard.attach(layer)

    wrapped_drag_callback = layer._drag_modes[Mode.ADD_POLYGON_LASSO]
    wrapped_move_callback = layer._move_modes[Mode.ADD_POLYGON_LASSO]
    assert layer.mode == original_mode
    assert original_drag_callback not in layer.mouse_drag_callbacks
    assert wrapped_drag_callback in layer.mouse_drag_callbacks
    assert unrelated_drag_callback in layer.mouse_drag_callbacks
    assert original_move_callback not in layer.mouse_move_callbacks
    assert wrapped_move_callback in layer.mouse_move_callbacks
    assert unrelated_move_callback in layer.mouse_move_callbacks

    guard.disconnect()

    assert layer.mode == original_mode
    assert original_drag_callback in layer.mouse_drag_callbacks
    assert wrapped_drag_callback not in layer.mouse_drag_callbacks
    assert unrelated_drag_callback in layer.mouse_drag_callbacks
    assert original_move_callback in layer.mouse_move_callbacks
    assert wrapped_move_callback not in layer.mouse_move_callbacks
    assert unrelated_move_callback in layer.mouse_move_callbacks


@pytest.mark.parametrize(
    "mode",
    [Mode.DIRECT, Mode.VERTEX_REMOVE, Mode.VERTEX_INSERT],
)
def test_annotation_layer_edit_guard_replaces_active_guarded_edit_callback(
    mode: Mode,
) -> None:
    layer = Shapes([], ndim=2)
    layer.mode = mode
    original_callback = layer._drag_modes[mode]

    def unrelated_callback(_layer: Shapes, _event: object) -> None:
        return None

    layer.mouse_drag_callbacks.append(unrelated_callback)
    guard = shapes_annotation_edit_guard_module._AnnotationLayerEditGuard()

    assert original_callback in layer.mouse_drag_callbacks

    guard.attach(layer)

    wrapped_callback = layer._drag_modes[mode]
    assert layer.mode == mode
    assert original_callback not in layer.mouse_drag_callbacks
    assert wrapped_callback in layer.mouse_drag_callbacks
    assert unrelated_callback in layer.mouse_drag_callbacks

    guard.disconnect()

    assert layer.mode == mode
    assert original_callback in layer.mouse_drag_callbacks
    assert wrapped_callback not in layer.mouse_drag_callbacks
    assert unrelated_callback in layer.mouse_drag_callbacks


def test_annotation_layer_edit_guard_leaves_active_callbacks_for_non_resumable_mode() -> None:
    layer = Shapes([], ndim=2)
    layer.mode = Mode.ADD_LINE
    original_mode = layer.mode
    original_drag_callbacks = list(layer.mouse_drag_callbacks)
    original_move_callbacks = list(layer.mouse_move_callbacks)
    guard = shapes_annotation_edit_guard_module._AnnotationLayerEditGuard()

    guard.attach(layer)

    assert layer.mode == original_mode
    assert list(layer.mouse_drag_callbacks) == original_drag_callbacks
    assert list(layer.mouse_move_callbacks) == original_move_callbacks

    guard.disconnect()

    assert layer.mode == original_mode
    assert list(layer.mouse_drag_callbacks) == original_drag_callbacks
    assert list(layer.mouse_move_callbacks) == original_move_callbacks


def test_annotation_layer_edit_guard_space_binding_is_removed_when_guard_created_it() -> None:
    layer = Shapes([], ndim=2)
    space_key = coerce_keybinding("Space")
    guard = shapes_annotation_edit_guard_module._AnnotationLayerEditGuard()

    assert space_key not in layer.keymap

    guard.attach(layer)

    assert "Space" not in layer.keymap
    assert layer.keymap[space_key] is guard._wrapped_space_keybinding
    assert guard._had_instance_space_keybinding is False
    assert guard._previous_space_keybinding is None

    guard.disconnect()

    assert space_key not in layer.keymap


def test_annotation_layer_edit_guard_space_binding_restores_none_like_existing_value() -> None:
    layer = Shapes([], ndim=2)
    space_key = coerce_keybinding("Space")
    layer.keymap[space_key] = None
    guard = shapes_annotation_edit_guard_module._AnnotationLayerEditGuard()

    guard.attach(layer)

    assert layer.keymap[space_key] is guard._wrapped_space_keybinding
    assert guard._had_instance_space_keybinding is True
    assert guard._previous_space_keybinding is None

    guard.disconnect()

    assert space_key in layer.keymap
    assert layer.keymap[space_key] is None


def test_annotation_layer_edit_guard_space_pan_resumable_draw_modes_are_explicit() -> None:
    supported_modes = shapes_annotation_edit_guard_module._SPACE_PAN_RESUMABLE_DRAW_MODES
    assert supported_modes == {
        Mode.ADD_POLYGON_LASSO,
        Mode.ADD_PATH,
        Mode.ADD_POLYGON,
        Mode.ADD_POLYLINE,
    }


def test_annotation_layer_edit_guard_can_space_pan_only_for_active_resumable_draw_modes() -> None:
    guard = shapes_annotation_edit_guard_module._AnnotationLayerEditGuard()

    for mode in shapes_annotation_edit_guard_module._SPACE_PAN_RESUMABLE_DRAW_MODES:
        layer = Shapes([], ndim=2)
        layer.mode = mode

        assert guard._can_space_pan_draw_mode(layer) is False

        layer._is_creating = True

        assert guard._can_space_pan_draw_mode(layer) is True

    for mode in {Mode.ADD_LINE, Mode.ADD_RECTANGLE, Mode.ADD_ELLIPSE}:
        layer = Shapes([], ndim=2)
        layer.mode = mode
        layer._is_creating = True

        assert guard._can_space_pan_draw_mode(layer) is False


def test_annotation_layer_edit_guard_supported_wrappers_delegate_when_drawing_is_not_suspended() -> None:
    layer = Shapes([], ndim=2)
    layer._drag_modes = dict(layer._drag_modes)
    layer._move_modes = dict(layer._move_modes)
    drag_calls: list[str] = []
    move_calls: list[str] = []

    def original_drag_callback(_layer: Shapes, event: object) -> Iterator[str]:
        drag_calls.append(getattr(event, "type", ""))
        yield "press"
        while getattr(event, "type", None) == "mouse_move":
            drag_calls.append(getattr(event, "type", ""))
            yield "move"

    def original_move_callback(_layer: Shapes, event: object) -> str:
        move_calls.append(getattr(event, "type", ""))
        return "moved"

    layer._drag_modes[Mode.ADD_POLYGON] = original_drag_callback
    layer._move_modes[Mode.ADD_POLYGON] = original_move_callback
    guard = shapes_annotation_edit_guard_module._AnnotationLayerEditGuard()
    guard.attach(layer)

    drag_event = SimpleNamespace(type="mouse_press")
    drag = layer._drag_modes[Mode.ADD_POLYGON](layer, drag_event)

    assert next(drag) == "press"
    drag_event.type = "mouse_move"
    assert next(drag) == "move"
    drag_event.type = "mouse_release"
    assert _exhaust_generator(drag) == []
    assert drag_calls == ["mouse_press", "mouse_move"]

    move_event = SimpleNamespace(type="mouse_move")
    assert layer._move_modes[Mode.ADD_POLYGON](layer, move_event) == "moved"
    assert move_calls == ["mouse_move"]


def test_annotation_layer_edit_guard_supported_wrappers_suppress_while_drawing_is_suspended() -> None:
    layer = Shapes([], ndim=2)
    layer._drag_modes = dict(layer._drag_modes)
    layer._move_modes = dict(layer._move_modes)
    original_mouse_pan = layer.mouse_pan
    original_data = list(layer.data)
    calls: list[str] = []

    def original_drag_callback(_layer: Shapes, _event: object) -> None:
        calls.append("drag")
        raise AssertionError("suspended drag should not call the original callback")

    def original_move_callback(_layer: Shapes, _event: object) -> None:
        calls.append("move")
        raise AssertionError("suspended move should not call the original callback")

    layer._drag_modes[Mode.ADD_POLYGON] = original_drag_callback
    layer._move_modes[Mode.ADD_POLYGON] = original_move_callback
    guard = shapes_annotation_edit_guard_module._AnnotationLayerEditGuard()
    guard.attach(layer)
    guard._begin_space_pan_key_hold(layer)

    assert layer._move_modes[Mode.ADD_POLYGON](layer, SimpleNamespace(type="mouse_move")) is None

    drag_event = SimpleNamespace(type="mouse_press")
    drag = layer._drag_modes[Mode.ADD_POLYGON](layer, drag_event)

    assert guard._space_pan_mouse_gesture_active is False
    assert next(drag) is None
    assert guard._space_pan_mouse_gesture_active is True

    drag_event.type = "mouse_move"
    assert next(drag) is None
    drag_event.type = "mouse_release"
    assert _exhaust_generator(drag) == []

    assert guard._space_pan_key_held is True
    assert guard._space_pan_mouse_gesture_active is False
    assert layer.mouse_pan is True
    assert calls == []
    assert list(layer.data) == original_data

    guard._end_space_pan_key_hold(layer)

    assert layer.mouse_pan is original_mouse_pan


def test_annotation_layer_edit_guard_supported_wrappers_restore_if_space_releases_before_mouse() -> None:
    layer = Shapes([], ndim=2)
    original_mouse_pan = layer.mouse_pan
    guard = shapes_annotation_edit_guard_module._AnnotationLayerEditGuard()
    guard.attach(layer)
    guard._begin_space_pan_key_hold(layer)
    drag_event = SimpleNamespace(type="mouse_press")
    drag = layer._drag_modes[Mode.ADD_POLYGON](layer, drag_event)

    assert next(drag) is None

    guard._end_space_pan_key_hold(layer)

    assert guard._space_pan_key_held is False
    assert guard._space_pan_mouse_gesture_active is True
    assert layer.mouse_pan is True

    drag_event.type = "mouse_release"
    assert _exhaust_generator(drag) == []

    assert guard._space_pan_key_held is False
    assert guard._space_pan_mouse_gesture_active is False
    assert layer.mouse_pan is original_mouse_pan


def test_annotation_layer_edit_guard_space_key_uses_custom_branch_for_active_supported_draw(
    monkeypatch,
) -> None:
    layer = Shapes([], ndim=2)
    layer.mode = Mode.ADD_POLYGON
    event = SimpleNamespace(position=(0.0, 0.0), pos=np.asarray([0.0, 0.0]))
    layer._drag_modes[Mode.ADD_POLYGON](layer, event)
    guard = shapes_annotation_edit_guard_module._AnnotationLayerEditGuard(
        can_space_pan_draw=lambda candidate: candidate is layer
    )
    guard.attach(layer)

    original_mouse_pan = layer.mouse_pan

    def fail_if_fallback_is_used(_layer: Shapes) -> Iterator[None]:
        raise AssertionError("active supported drawing should use the custom Space-pan branch")
        yield

    monkeypatch.setattr(guard, "_fallback_pan_zoom_key_hold", fail_if_fallback_is_used)
    handler = KeymapHandler()
    handler.keymap_providers = [layer]

    assert layer._is_creating is True
    assert handler.press_key("Space") is True

    assert layer.mode == Mode.ADD_POLYGON
    assert layer.mouse_pan is True
    assert guard._space_pan_key_held is True
    assert guard._previous_mouse_pan is original_mouse_pan

    assert handler.release_key("Space") is True

    assert layer.mode == Mode.ADD_POLYGON
    assert guard._space_pan_key_held is False
    assert guard._previous_mouse_pan is None
    assert layer.mouse_pan is original_mouse_pan


def test_annotation_layer_edit_guard_space_key_delegates_when_widget_predicate_rejects(
    monkeypatch,
) -> None:
    layer = Shapes([], ndim=2)
    layer.mode = Mode.ADD_POLYGON
    layer._is_creating = True
    fallback_layers: list[Shapes] = []
    guard = shapes_annotation_edit_guard_module._AnnotationLayerEditGuard(can_space_pan_draw=lambda _layer: False)
    guard.attach(layer)

    def fallback_pan_zoom_key_hold(fallback_layer: Shapes) -> Iterator[None]:
        fallback_layers.append(fallback_layer)
        yield

    monkeypatch.setattr(guard, "_fallback_pan_zoom_key_hold", fallback_pan_zoom_key_hold)
    handler = KeymapHandler()
    handler.keymap_providers = [layer]

    assert handler.press_key("Space") is True

    assert fallback_layers == [layer]
    assert guard._space_pan_key_held is False
    assert guard._space_pan_mouse_gesture_active is False
    assert guard._previous_mouse_pan is None

    assert handler.release_key("Space") is True

    assert guard._space_pan_key_held is False
    assert guard._space_pan_mouse_gesture_active is False
    assert guard._previous_mouse_pan is None


def test_annotation_layer_edit_guard_space_key_delegates_when_widget_predicate_is_missing(
    monkeypatch,
) -> None:
    layer = Shapes([], ndim=2)
    layer.mode = Mode.ADD_POLYGON
    layer._is_creating = True
    fallback_layers: list[Shapes] = []
    guard = shapes_annotation_edit_guard_module._AnnotationLayerEditGuard()
    guard.attach(layer)

    def fallback_pan_zoom_key_hold(fallback_layer: Shapes) -> Iterator[None]:
        fallback_layers.append(fallback_layer)
        yield

    monkeypatch.setattr(guard, "_fallback_pan_zoom_key_hold", fallback_pan_zoom_key_hold)
    handler = KeymapHandler()
    handler.keymap_providers = [layer]

    assert handler.press_key("Space") is True

    assert fallback_layers == [layer]
    assert guard._space_pan_key_held is False
    assert guard._space_pan_mouse_gesture_active is False
    assert guard._previous_mouse_pan is None

    assert handler.release_key("Space") is True

    assert guard._space_pan_key_held is False
    assert guard._space_pan_mouse_gesture_active is False
    assert guard._previous_mouse_pan is None


def test_annotation_layer_edit_guard_space_key_delegates_for_unsupported_modes() -> None:
    layer = Shapes([], ndim=2)
    layer.mode = Mode.ADD_LINE
    original_drag_callbacks = list(layer.mouse_drag_callbacks)
    original_move_callbacks = list(layer.mouse_move_callbacks)
    guard = shapes_annotation_edit_guard_module._AnnotationLayerEditGuard()
    guard.attach(layer)
    handler = KeymapHandler()
    handler.keymap_providers = [layer]

    assert handler.press_key("Space") is True

    assert layer.mode == Mode.PAN_ZOOM
    assert guard._space_pan_key_held is False
    assert guard._previous_mouse_pan is None

    assert handler.release_key("Space") is True

    assert layer.mode == Mode.ADD_LINE
    assert guard._space_pan_key_held is False
    assert guard._previous_mouse_pan is None
    assert list(layer.mouse_drag_callbacks) == original_drag_callbacks
    assert list(layer.mouse_move_callbacks) == original_move_callbacks


def test_annotation_layer_edit_guard_space_pan_key_hold_restores_mouse_pan() -> None:
    layer = Shapes([], ndim=2)
    layer.mode = Mode.ADD_POLYGON_LASSO
    original_mouse_pan = layer.mouse_pan
    original_mode = layer.mode
    original_keymap = dict(layer.keymap)
    original_drag_callbacks = list(layer.mouse_drag_callbacks)
    original_move_callbacks = list(layer.mouse_move_callbacks)
    guard = shapes_annotation_edit_guard_module._AnnotationLayerEditGuard()

    assert original_mouse_pan is False
    assert guard._drawing_is_suspended() is False

    guard._begin_space_pan_key_hold(layer)

    assert guard._space_pan_key_held is True
    assert guard._space_pan_mouse_gesture_active is False
    assert guard._previous_mouse_pan is original_mouse_pan
    assert guard._drawing_is_suspended() is True
    assert layer.mouse_pan is True

    guard._end_space_pan_key_hold(layer)

    assert guard._space_pan_key_held is False
    assert guard._space_pan_mouse_gesture_active is False
    assert guard._previous_mouse_pan is None
    assert guard._drawing_is_suspended() is False
    assert layer.mouse_pan is original_mouse_pan
    assert layer.mode == original_mode
    assert dict(layer.keymap) == original_keymap
    assert list(layer.mouse_drag_callbacks) == original_drag_callbacks
    assert list(layer.mouse_move_callbacks) == original_move_callbacks


def test_annotation_layer_edit_guard_space_pan_repeated_begin_keeps_original_mouse_pan() -> None:
    layer = Shapes([], ndim=2)
    layer.mode = Mode.ADD_POLYGON_LASSO
    original_mouse_pan = layer.mouse_pan
    guard = shapes_annotation_edit_guard_module._AnnotationLayerEditGuard()

    assert original_mouse_pan is False

    guard._begin_space_pan_key_hold(layer)
    guard._begin_space_pan_key_hold(layer)
    guard._end_space_pan_key_hold(layer)

    assert guard._previous_mouse_pan is None
    assert layer.mouse_pan is original_mouse_pan


def test_annotation_layer_edit_guard_space_pan_restores_after_space_released_first() -> None:
    layer = Shapes([], ndim=2)
    layer.mode = Mode.ADD_POLYGON_LASSO
    original_mouse_pan = layer.mouse_pan
    guard = shapes_annotation_edit_guard_module._AnnotationLayerEditGuard()

    guard._begin_space_pan_key_hold(layer)
    guard._begin_space_pan_mouse_gesture()
    guard._end_space_pan_key_hold(layer)

    assert guard._space_pan_key_held is False
    assert guard._space_pan_mouse_gesture_active is True
    assert guard._previous_mouse_pan is original_mouse_pan
    assert guard._drawing_is_suspended() is True
    assert layer.mouse_pan is True

    guard._end_space_pan_mouse_gesture(layer)

    assert guard._space_pan_key_held is False
    assert guard._space_pan_mouse_gesture_active is False
    assert guard._previous_mouse_pan is None
    assert guard._drawing_is_suspended() is False
    assert layer.mouse_pan is original_mouse_pan


def test_annotation_layer_edit_guard_space_pan_restores_after_mouse_released_first() -> None:
    layer = Shapes([], ndim=2)
    layer.mode = Mode.ADD_POLYGON_LASSO
    original_mouse_pan = layer.mouse_pan
    guard = shapes_annotation_edit_guard_module._AnnotationLayerEditGuard()

    guard._begin_space_pan_key_hold(layer)
    guard._begin_space_pan_mouse_gesture()
    guard._end_space_pan_mouse_gesture(layer)

    assert guard._space_pan_key_held is True
    assert guard._space_pan_mouse_gesture_active is False
    assert guard._previous_mouse_pan is original_mouse_pan
    assert guard._drawing_is_suspended() is True
    assert layer.mouse_pan is True

    guard._end_space_pan_key_hold(layer)

    assert guard._space_pan_key_held is False
    assert guard._space_pan_mouse_gesture_active is False
    assert guard._previous_mouse_pan is None
    assert guard._drawing_is_suspended() is False
    assert layer.mouse_pan is original_mouse_pan


def test_annotation_layer_edit_guard_space_pan_mouse_gesture_only_does_not_change_mouse_pan() -> None:
    layer = Shapes([], ndim=2)
    layer.mode = Mode.ADD_POLYGON_LASSO
    original_mouse_pan = layer.mouse_pan
    guard = shapes_annotation_edit_guard_module._AnnotationLayerEditGuard()

    guard._begin_space_pan_mouse_gesture()

    assert guard._space_pan_key_held is False
    assert guard._space_pan_mouse_gesture_active is True
    assert guard._previous_mouse_pan is None
    assert guard._drawing_is_suspended() is True
    assert layer.mouse_pan is original_mouse_pan

    guard._end_space_pan_mouse_gesture(layer)

    assert guard._space_pan_key_held is False
    assert guard._space_pan_mouse_gesture_active is False
    assert guard._previous_mouse_pan is None
    assert guard._drawing_is_suspended() is False
    assert layer.mouse_pan is original_mouse_pan


def test_annotation_layer_edit_guard_disconnect_restores_active_space_pan_mouse_state() -> None:
    layer = Shapes([], ndim=2)
    layer.mode = Mode.ADD_POLYGON_LASSO
    original_mouse_pan = layer.mouse_pan
    guard = shapes_annotation_edit_guard_module._AnnotationLayerEditGuard()
    guard.attach(layer)

    guard._begin_space_pan_key_hold(layer)
    guard._begin_space_pan_mouse_gesture()
    guard.disconnect()

    assert guard.layer is None
    assert guard._space_pan_key_held is False
    assert guard._space_pan_mouse_gesture_active is False
    assert guard._previous_mouse_pan is None
    assert layer.mouse_pan is original_mouse_pan


def test_annotation_layer_edit_guard_direct_drag_delegates_non_polygon_vertex() -> None:
    layer = Shapes([np.asarray([[0.0, 0.0], [1.0, 1.0]])], shape_type="line")
    layer._drag_modes = dict(layer._drag_modes)
    native_moves = 0
    native_released = False

    def native_direct_drag(bound_layer: Shapes, event: object) -> Iterator[str]:
        nonlocal native_moves, native_released
        bound_layer._moving_value = (0, 0)
        yield "press"
        while event.type == "mouse_move":
            native_moves += 1
            yield "move"
        native_released = True
        bound_layer._moving_value = (None, None)

    layer._drag_modes[Mode.DIRECT] = native_direct_drag
    guard = shapes_annotation_edit_guard_module._AnnotationLayerEditGuard()
    guard.attach(layer)
    event = SimpleNamespace(type="mouse_press")

    drag = layer._drag_modes[Mode.DIRECT](layer, event)
    assert next(drag) == "press"
    event.type = "mouse_move"
    assert next(drag) == "move"
    event.type = "mouse_release"
    with pytest.raises(StopIteration):
        next(drag)

    assert native_moves == 1
    assert native_released is True
    assert layer._moving_value == (None, None)


def test_annotation_layer_edit_guard_direct_drag_syncs_shell_anchor_group() -> None:
    polygon, _ = _polygon_hole_roundtrip_fixture()
    original_vertices = shapely_polygon_to_napari_polygon_vertices(polygon)
    moved_coordinate = original_vertices[0] + np.asarray([1.0, 1.0])
    layer = Shapes([original_vertices], shape_type="polygon")
    layer._drag_modes = dict(layer._drag_modes)
    layer._drag_modes[Mode.DIRECT] = _direct_drag_callback_selecting_vertex(moved_vertex_index=0)
    guard = shapes_annotation_edit_guard_module._AnnotationLayerEditGuard()
    guard.attach(layer)
    event = SimpleNamespace(type="mouse_press", position=tuple(original_vertices[0]))

    drag = layer._drag_modes[Mode.DIRECT](layer, event)
    assert next(drag) == "press"
    event.type = "mouse_move"
    event.position = tuple(moved_coordinate)
    assert next(drag) is None
    event.type = "mouse_release"
    with pytest.raises(StopIteration):
        next(drag)

    edited_vertices = np.asarray(layer.data[0], dtype=float)
    np.testing.assert_allclose(edited_vertices[[0, 5, 12]], np.repeat(moved_coordinate[None, :], 3, axis=0))


def test_annotation_layer_edit_guard_direct_drag_syncs_shell_separator_group() -> None:
    polygon, _ = _polygon_hole_roundtrip_fixture()
    original_vertices = shapely_polygon_to_napari_polygon_vertices(polygon)
    moved_coordinate = original_vertices[12] + np.asarray([1.0, 1.0])
    layer = Shapes([original_vertices], shape_type="polygon")
    layer._drag_modes = dict(layer._drag_modes)
    layer._drag_modes[Mode.DIRECT] = _direct_drag_callback_selecting_vertex(moved_vertex_index=12)
    guard = shapes_annotation_edit_guard_module._AnnotationLayerEditGuard()
    guard.attach(layer)
    event = SimpleNamespace(type="mouse_press", position=tuple(original_vertices[12]))

    drag = layer._drag_modes[Mode.DIRECT](layer, event)
    assert next(drag) == "press"
    event.type = "mouse_move"
    event.position = tuple(moved_coordinate)
    assert next(drag) is None

    edited_vertices = np.asarray(layer.data[0], dtype=float)
    np.testing.assert_allclose(edited_vertices[[0, 5, 12]], np.repeat(moved_coordinate[None, :], 3, axis=0))


def test_annotation_layer_edit_guard_direct_drag_syncs_hole_anchor_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    polygon, _ = _polygon_hole_roundtrip_fixture()
    original_vertices = shapely_polygon_to_napari_polygon_vertices(polygon)
    moved_coordinate = original_vertices[6] + np.asarray([1.0, 1.0])
    layer = Shapes([original_vertices], shape_type="polygon")
    written_rows: list[np.ndarray] = []
    original_edit = layer._data_view.edit

    def record_edit(row_index: int, vertices: np.ndarray, *args: object, **kwargs: object) -> None:
        written_rows.append(np.asarray(vertices, dtype=float).copy())
        original_edit(row_index, vertices, *args, **kwargs)

    monkeypatch.setattr(layer._data_view, "edit", record_edit)
    layer._drag_modes = dict(layer._drag_modes)
    layer._drag_modes[Mode.DIRECT] = _direct_drag_callback_selecting_vertex(moved_vertex_index=6)
    guard = shapes_annotation_edit_guard_module._AnnotationLayerEditGuard()
    guard.attach(layer)
    event = SimpleNamespace(type="mouse_press", position=tuple(original_vertices[6]))

    drag = layer._drag_modes[Mode.DIRECT](layer, event)
    assert next(drag) == "press"
    event.type = "mouse_move"
    event.position = tuple(moved_coordinate)
    assert next(drag) is None
    event.type = "mouse_release"
    with pytest.raises(StopIteration):
        next(drag)

    edited_vertices = np.asarray(layer.data[0], dtype=float)
    np.testing.assert_allclose(edited_vertices[[6, 11]], np.repeat(moved_coordinate[None, :], 2, axis=0))
    assert len(written_rows) == 1
    np.testing.assert_allclose(written_rows[0][[6, 11]], np.repeat(moved_coordinate[None, :], 2, axis=0))


def test_annotation_layer_edit_guard_direct_drag_leaves_non_anchor_vertex_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    polygon, _ = _polygon_hole_roundtrip_fixture()
    original_vertices = shapely_polygon_to_napari_polygon_vertices(polygon)
    moved_coordinate = original_vertices[8] + np.asarray([1.0, 0.0])
    layer = Shapes([original_vertices], shape_type="polygon")
    data_events: list[object] = []
    written_rows: list[np.ndarray] = []
    thumbnail_updates = 0
    layer.events.data.connect(data_events.append)
    original_edit = layer._data_view.edit
    original_update_thumbnail = layer._update_thumbnail

    def record_edit(row_index: int, vertices: np.ndarray, *args: object, **kwargs: object) -> None:
        written_rows.append(np.asarray(vertices, dtype=float).copy())
        original_edit(row_index, vertices, *args, **kwargs)

    def record_thumbnail_update(*args: object, **kwargs: object) -> None:
        nonlocal thumbnail_updates
        thumbnail_updates += 1
        original_update_thumbnail(*args, **kwargs)

    monkeypatch.setattr(layer._data_view, "edit", record_edit)
    monkeypatch.setattr(layer, "_update_thumbnail", record_thumbnail_update)
    layer._drag_modes = dict(layer._drag_modes)
    layer._drag_modes[Mode.DIRECT] = _direct_drag_callback_selecting_vertex(moved_vertex_index=8)
    guard = shapes_annotation_edit_guard_module._AnnotationLayerEditGuard()
    guard.attach(layer)
    event = SimpleNamespace(type="mouse_press", position=tuple(original_vertices[8]))

    drag = layer._drag_modes[Mode.DIRECT](layer, event)
    assert next(drag) == "press"
    event.type = "mouse_move"
    event.position = tuple(moved_coordinate)
    assert next(drag) is None
    event.type = "mouse_release"
    with pytest.raises(StopIteration):
        next(drag)

    edited_vertices = np.asarray(layer.data[0], dtype=float)
    np.testing.assert_allclose(edited_vertices[8], moved_coordinate)
    np.testing.assert_allclose(edited_vertices[[0, 5, 12]], original_vertices[[0, 5, 12]])
    np.testing.assert_allclose(edited_vertices[[6, 11]], original_vertices[[6, 11]])
    assert len(written_rows) == 1
    assert len(data_events) == 1
    assert data_events[0].action is ActionType.CHANGED
    assert data_events[0].data_indices == (0,)
    assert data_events[0].vertex_indices == (tuple(range(len(edited_vertices))),)
    assert thumbnail_updates == 1


def test_annotation_layer_edit_guard_direct_drag_rejects_invalid_hole_vertex_once_per_drag() -> None:
    polygon, _ = _polygon_hole_roundtrip_fixture()
    original_vertices = shapely_polygon_to_napari_polygon_vertices(polygon)
    invalid_coordinates = [np.asarray([10_000.0, 10_000.0]), np.asarray([20_000.0, 20_000.0])]
    layer = Shapes([original_vertices], shape_type="polygon")
    layer._drag_modes = dict(layer._drag_modes)
    layer._drag_modes[Mode.DIRECT] = _direct_drag_callback_selecting_vertex(moved_vertex_index=8)
    warnings: list[str] = []
    guard = shapes_annotation_edit_guard_module._AnnotationLayerEditGuard(warning_callback=warnings.append)
    guard.attach(layer)
    event = SimpleNamespace(type="mouse_press", position=tuple(original_vertices[8]))

    drag = layer._drag_modes[Mode.DIRECT](layer, event)
    assert next(drag) == "press"
    event.type = "mouse_move"
    event.position = tuple(invalid_coordinates[0])
    assert next(drag) is None
    event.position = tuple(invalid_coordinates[1])
    assert next(drag) is None
    event.type = "mouse_release"
    with pytest.raises(StopIteration):
        next(drag)

    np.testing.assert_allclose(np.asarray(layer.data[0], dtype=float), original_vertices)
    assert warnings == [shapes_annotation_edit_guard_module._INVALID_POLYGON_DRAG_WARNING]
    assert layer._moving_value == (None, None)


def test_annotation_layer_edit_guard_direct_drag_rejects_invalid_hole_anchor() -> None:
    polygon, _ = _polygon_hole_roundtrip_fixture()
    original_vertices = shapely_polygon_to_napari_polygon_vertices(polygon)
    invalid_coordinate = np.asarray([10_000.0, 10_000.0])
    layer = Shapes([original_vertices], shape_type="polygon")
    layer._drag_modes = dict(layer._drag_modes)
    layer._drag_modes[Mode.DIRECT] = _direct_drag_callback_selecting_vertex(moved_vertex_index=6)
    warnings: list[str] = []
    guard = shapes_annotation_edit_guard_module._AnnotationLayerEditGuard(warning_callback=warnings.append)
    guard.attach(layer)
    event = SimpleNamespace(type="mouse_press", position=tuple(original_vertices[6]))

    drag = layer._drag_modes[Mode.DIRECT](layer, event)
    assert next(drag) == "press"
    event.type = "mouse_move"
    event.position = tuple(invalid_coordinate)
    assert next(drag) is None
    event.type = "mouse_release"
    with pytest.raises(StopIteration):
        next(drag)

    edited_vertices = np.asarray(layer.data[0], dtype=float)
    np.testing.assert_allclose(edited_vertices, original_vertices)
    np.testing.assert_allclose(edited_vertices[[6, 11]], original_vertices[[6, 11]])
    assert warnings == [shapes_annotation_edit_guard_module._INVALID_POLYGON_DRAG_WARNING]


def test_annotation_layer_edit_guard_prevalidates_exact_vertex_39_regression_and_allows_next_drag(
    monkeypatch: pytest.MonkeyPatch,
    numba_triangulation_backend: None,
) -> None:
    original_vertices = make_triangulation_regression_pre_drag_vertices()
    layer = Shapes([original_vertices], shape_type="polygon")
    # Napari may normalize coordinate precision while constructing the shape;
    # the transaction baseline is the actual accepted live row after that step.
    original_vertices = np.asarray(layer.data[0], dtype=float).copy()
    layer.mode = Mode.DIRECT
    layer.selected_data = {0}
    warnings: list[str] = []
    data_actions: list[ActionType] = []
    decoded_rows: list[np.ndarray] = []
    hit_vertex_index = TRIANGULATION_REGRESSION_MOVED_VERTEX_INDEX
    layer.events.data.connect(lambda event: data_actions.append(event.action))
    monkeypatch.setattr(layer, "get_value", lambda position, world=True: (0, hit_vertex_index))

    original_decode = shapes_geometry_module.napari_polygon_vertices_to_shapely_polygon

    def record_decoded_row(vertices: np.ndarray) -> Polygon:
        decoded_rows.append(np.asarray(vertices, dtype=float).copy())
        return original_decode(vertices)

    monkeypatch.setattr(
        shapes_geometry_module,
        "napari_polygon_vertices_to_shapely_polygon",
        record_decoded_row,
    )
    guard = shapes_annotation_edit_guard_module._AnnotationLayerEditGuard(warning_callback=warnings.append)
    guard.attach(layer)

    original_edit = layer._data_view.edit
    original_set_meshes = layer._data_view.shapes[0]._set_meshes

    def fail_edit(*args: object, **kwargs: object) -> None:
        raise AssertionError("The invalid synchronized candidate reached `_data_view.edit(...)`.")

    def fail_triangulation(*args: object, **kwargs: object) -> None:
        raise AssertionError("The invalid synchronized candidate reached triangulation.")

    monkeypatch.setattr(layer._data_view, "edit", fail_edit)
    monkeypatch.setattr(layer._data_view.shapes[0], "_set_meshes", fail_triangulation)

    event = SimpleNamespace(
        type="mouse_press",
        position=tuple(original_vertices[TRIANGULATION_REGRESSION_MOVED_VERTEX_INDEX]),
        modifiers=(),
    )
    drag = layer._drag_modes[Mode.DIRECT](layer, event)
    assert next(drag) is None
    event.type = "mouse_move"
    event.position = TRIANGULATION_REGRESSION_FAILED_MOVE_COORDINATE
    assert next(drag) is None
    event.type = "mouse_release"
    with pytest.raises(StopIteration):
        next(drag)

    synchronized_candidate = decoded_rows[-1]
    np.testing.assert_array_equal(
        synchronized_candidate[list(TRIANGULATION_REGRESSION_HOLE_ANCHOR_INDICES)],
        np.repeat(
            np.asarray(TRIANGULATION_REGRESSION_FAILED_MOVE_COORDINATE)[None, :],
            len(TRIANGULATION_REGRESSION_HOLE_ANCHOR_INDICES),
            axis=0,
        ),
    )
    np.testing.assert_array_equal(np.asarray(layer.data[0], dtype=float), original_vertices)
    assert warnings == [shapes_annotation_edit_guard_module._INVALID_POLYGON_DRAG_WARNING]
    assert data_actions == []
    assert layer._moving_value == (None, None)
    assert layer._is_moving is False

    # The rejected regression gesture must not poison the next direct drag.
    monkeypatch.setattr(layer._data_view, "edit", original_edit)
    monkeypatch.setattr(layer._data_view.shapes[0], "_set_meshes", original_set_meshes)
    hit_vertex_index = 1
    valid_coordinate = original_vertices[1] + np.asarray([1.0, 0.0])
    event = SimpleNamespace(
        type="mouse_press",
        position=tuple(original_vertices[1]),
        modifiers=(),
    )
    drag = layer._drag_modes[Mode.DIRECT](layer, event)
    assert next(drag) is None
    event.type = "mouse_move"
    event.position = tuple(valid_coordinate)
    assert next(drag) is None
    event.type = "mouse_release"
    with pytest.raises(StopIteration):
        next(drag)

    np.testing.assert_array_equal(np.asarray(layer.data[0], dtype=float)[1], valid_coordinate)
    assert data_actions == [ActionType.CHANGED]
    assert layer._moving_value == (None, None)


def test_annotation_layer_edit_guard_direct_drag_rejects_invalid_simple_polygon_after_valid_move() -> None:
    """Keep an earlier accepted move when a later move is invalid.

    Move one vertex first to a valid coordinate, then propose an invalid
    coordinate during the same drag. The invalid candidate must not replace or
    discard the last accepted row. On release, retain that earlier valid edit,
    warn once, emit one ``CHANGED`` event for the surviving mutation, and clear
    napari's temporary moving state.
    """
    original_vertices = np.asarray(
        [
            [0.0, 0.0],
            [0.0, 10.0],
            [10.0, 10.0],
            [10.0, 0.0],
        ],
        dtype=float,
    )
    valid_coordinate = np.asarray([0.0, 11.0])
    invalid_coordinate = np.asarray([20.0, 0.0])
    layer = Shapes([original_vertices], shape_type="polygon")
    data_actions: list[ActionType] = []
    layer.events.data.connect(lambda event: data_actions.append(event.action))
    layer._drag_modes = dict(layer._drag_modes)
    layer._drag_modes[Mode.DIRECT] = _direct_drag_callback_selecting_vertex(moved_vertex_index=1)
    warnings: list[str] = []
    guard = shapes_annotation_edit_guard_module._AnnotationLayerEditGuard(warning_callback=warnings.append)
    guard.attach(layer)
    event = SimpleNamespace(type="mouse_press", position=tuple(original_vertices[1]))

    drag = layer._drag_modes[Mode.DIRECT](layer, event)
    assert next(drag) == "press"
    event.type = "mouse_move"
    event.position = tuple(valid_coordinate)
    assert next(drag) is None
    accepted_vertices = np.asarray(layer.data[0], dtype=float).copy()
    np.testing.assert_allclose(accepted_vertices[1], valid_coordinate)
    event.position = tuple(invalid_coordinate)
    assert next(drag) is None
    event.type = "mouse_release"
    with pytest.raises(StopIteration):
        next(drag)

    np.testing.assert_allclose(np.asarray(layer.data[0], dtype=float), accepted_vertices)
    assert warnings == [shapes_annotation_edit_guard_module._INVALID_POLYGON_DRAG_WARNING]
    assert data_actions == [ActionType.CHANGED]
    assert layer._moving_value == (None, None)


def test_annotation_layer_edit_guard_direct_drag_syncs_closed_simple_polygon_endpoints() -> None:
    original_vertices = np.asarray(
        [[0.0, 0.0], [0.0, 10.0], [10.0, 10.0], [10.0, 0.0], [0.0, 0.0]],
        dtype=float,
    )
    moved_coordinate = np.asarray([-1.0, 0.0])
    layer = Shapes([original_vertices], shape_type="polygon")
    layer._drag_modes = dict(layer._drag_modes)
    layer._drag_modes[Mode.DIRECT] = _direct_drag_callback_selecting_vertex(moved_vertex_index=0)
    guard = shapes_annotation_edit_guard_module._AnnotationLayerEditGuard()
    guard.attach(layer)
    event = SimpleNamespace(type="mouse_press", position=tuple(original_vertices[0]))

    drag = layer._drag_modes[Mode.DIRECT](layer, event)
    assert next(drag) == "press"
    event.type = "mouse_move"
    event.position = tuple(moved_coordinate)
    assert next(drag) is None
    event.type = "mouse_release"
    with pytest.raises(StopIteration):
        next(drag)

    edited_vertices = np.asarray(layer.data[0], dtype=float)
    np.testing.assert_array_equal(edited_vertices[[0, -1]], np.repeat(moved_coordinate[None, :], 2, axis=0))
    _ = napari_polygon_vertices_to_shapely_polygon(edited_vertices)


def test_annotation_layer_edit_guard_direct_drag_rejects_anchor_topology_collision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    polygon, _ = _polygon_hole_roundtrip_fixture()
    original_vertices = shapely_polygon_to_napari_polygon_vertices(polygon)
    layer = Shapes([original_vertices], shape_type="polygon")
    layer._drag_modes = dict(layer._drag_modes)
    layer._drag_modes[Mode.DIRECT] = _direct_drag_callback_selecting_vertex(moved_vertex_index=8)
    warnings: list[str] = []

    def fail_edit(*args: object, **kwargs: object) -> None:
        raise AssertionError("A topology-changing candidate reached the live row.")

    monkeypatch.setattr(layer._data_view, "edit", fail_edit)
    guard = shapes_annotation_edit_guard_module._AnnotationLayerEditGuard(warning_callback=warnings.append)
    guard.attach(layer)
    event = SimpleNamespace(type="mouse_press", position=tuple(original_vertices[8]))

    drag = layer._drag_modes[Mode.DIRECT](layer, event)
    assert next(drag) == "press"
    event.type = "mouse_move"
    event.position = tuple(original_vertices[0])
    assert next(drag) is None
    event.type = "mouse_release"
    with pytest.raises(StopIteration):
        next(drag)

    np.testing.assert_array_equal(np.asarray(layer.data[0], dtype=float), original_vertices)
    assert warnings == [shapes_annotation_edit_guard_module._INVALID_POLYGON_DRAG_WARNING]


def test_annotation_layer_edit_guard_direct_drag_noop_does_not_write_or_emit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_vertices = np.asarray(
        [[0.0, 0.0], [0.0, 10.0], [10.0, 10.0], [10.0, 0.0]],
        dtype=float,
    )
    layer = Shapes([original_vertices], shape_type="polygon")
    layer._drag_modes = dict(layer._drag_modes)
    layer._drag_modes[Mode.DIRECT] = _direct_drag_callback_selecting_vertex(moved_vertex_index=1)
    data_actions: list[ActionType] = []
    layer.events.data.connect(lambda event: data_actions.append(event.action))

    def fail_edit(*args: object, **kwargs: object) -> None:
        raise AssertionError("An unchanged polygon candidate reached `_data_view.edit(...)`.")

    def fail_thumbnail(*args: object, **kwargs: object) -> None:
        raise AssertionError("A no-op polygon drag updated the thumbnail.")

    monkeypatch.setattr(layer._data_view, "edit", fail_edit)
    monkeypatch.setattr(layer, "_update_thumbnail", fail_thumbnail)
    guard = shapes_annotation_edit_guard_module._AnnotationLayerEditGuard()
    guard.attach(layer)
    event = SimpleNamespace(type="mouse_press", position=tuple(original_vertices[1]))

    drag = layer._drag_modes[Mode.DIRECT](layer, event)
    assert next(drag) == "press"
    event.type = "mouse_move"
    assert next(drag) is None
    event.type = "mouse_release"
    with pytest.raises(StopIteration):
        next(drag)

    np.testing.assert_array_equal(np.asarray(layer.data[0], dtype=float), original_vertices)
    assert data_actions == []
    assert layer._is_moving is False


def test_annotation_layer_edit_guard_direct_drag_restores_and_continues_after_partial_write_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A restored renderer failure must not end the guarded drag."""
    original_vertices = np.asarray(
        [[0.0, 0.0], [0.0, 10.0], [10.0, 10.0], [10.0, 0.0]],
        dtype=float,
    )
    failed_coordinate = np.asarray([0.0, 11.0])
    later_valid_coordinate = np.asarray([0.0, 12.0])
    layer = Shapes([original_vertices], shape_type="polygon")
    layer._drag_modes = dict(layer._drag_modes)
    layer._drag_modes[Mode.DIRECT] = _direct_drag_callback_selecting_vertex(moved_vertex_index=1)
    warnings: list[str] = []
    data_actions: list[ActionType] = []
    written_rows: list[np.ndarray] = []
    layer.events.data.connect(lambda event: data_actions.append(event.action))
    original_edit = layer._data_view.edit

    def fail_after_first_write(row_index: int, vertices: np.ndarray, *args: object, **kwargs: object) -> None:
        written_rows.append(np.asarray(vertices, dtype=float).copy())
        original_edit(row_index, vertices, *args, **kwargs)
        if len(written_rows) == 1:
            raise RuntimeError("simulated failure after the candidate row was written")

    monkeypatch.setattr(layer._data_view, "edit", fail_after_first_write)
    guard = shapes_annotation_edit_guard_module._AnnotationLayerEditGuard(warning_callback=warnings.append)
    guard.attach(layer)
    event = SimpleNamespace(type="mouse_press", position=tuple(original_vertices[1]))

    drag = layer._drag_modes[Mode.DIRECT](layer, event)
    assert next(drag) == "press"
    event.type = "mouse_move"
    event.position = tuple(failed_coordinate)
    assert next(drag) is None

    # The first valid candidate was partially written before the injected
    # renderer failure. Spatiato restores the accepted baseline and keeps this
    # same gesture alive for another move.
    np.testing.assert_array_equal(np.asarray(layer.data[0], dtype=float), original_vertices)
    event.position = tuple(later_valid_coordinate)
    assert next(drag) is None
    event.type = "mouse_release"
    with pytest.raises(StopIteration):
        next(drag)

    assert len(written_rows) == 3
    np.testing.assert_array_equal(written_rows[0][1], failed_coordinate)
    np.testing.assert_array_equal(written_rows[1], original_vertices)
    np.testing.assert_array_equal(written_rows[2][1], later_valid_coordinate)
    np.testing.assert_array_equal(np.asarray(layer.data[0], dtype=float)[1], later_valid_coordinate)
    assert warnings == [shapes_annotation_edit_guard_module._POLYGON_DRAG_RENDERING_WARNING]
    assert data_actions == [ActionType.CHANGED]
    assert layer._is_moving is False
    assert layer._moving_value == (None, None)


def test_annotation_layer_edit_guard_direct_drag_keeps_accepted_move_after_later_renderer_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A later restored failure must not discard or hide an accepted move.

    Exercise three live row writes during one drag:

    1. write the accepted candidate with vertex 1 at ``[0, 11]``;
    2. write the later candidate with vertex 1 at ``[0, 12]``, then inject the
       renderer failure;
    3. restore the accepted candidate with vertex 1 at ``[0, 11]``.

    The release must retain that earlier accepted position and report its one
    surviving change.
    """
    original_vertices = np.asarray(
        [[0.0, 0.0], [0.0, 10.0], [10.0, 10.0], [10.0, 0.0]],
        dtype=float,
    )
    accepted_coordinate = np.asarray([0.0, 11.0])
    failed_coordinate = np.asarray([0.0, 12.0])
    layer = Shapes([original_vertices], shape_type="polygon")
    layer._drag_modes = dict(layer._drag_modes)
    layer._drag_modes[Mode.DIRECT] = _direct_drag_callback_selecting_vertex(moved_vertex_index=1)
    warnings: list[str] = []
    data_actions: list[ActionType] = []
    written_rows: list[np.ndarray] = []
    layer.events.data.connect(lambda event: data_actions.append(event.action))
    original_edit = layer._data_view.edit

    def fail_after_second_write(row_index: int, vertices: np.ndarray, *args: object, **kwargs: object) -> None:
        written_rows.append(np.asarray(vertices, dtype=float).copy())
        original_edit(row_index, vertices, *args, **kwargs)
        if len(written_rows) == 2:
            raise RuntimeError("simulated failure after the later candidate row was written")

    monkeypatch.setattr(layer._data_view, "edit", fail_after_second_write)
    guard = shapes_annotation_edit_guard_module._AnnotationLayerEditGuard(warning_callback=warnings.append)
    guard.attach(layer)
    event = SimpleNamespace(type="mouse_press", position=tuple(original_vertices[1]))

    drag = layer._drag_modes[Mode.DIRECT](layer, event)
    assert next(drag) == "press"
    event.type = "mouse_move"
    event.position = tuple(accepted_coordinate)
    assert next(drag) is None
    accepted_vertices = np.asarray(layer.data[0], dtype=float).copy()
    event.position = tuple(failed_coordinate)
    assert next(drag) is None
    event.type = "mouse_release"
    with pytest.raises(StopIteration):
        next(drag)

    assert len(written_rows) == 3
    np.testing.assert_array_equal(written_rows[0][1], accepted_coordinate)
    np.testing.assert_array_equal(written_rows[1][1], failed_coordinate)
    np.testing.assert_array_equal(written_rows[2], accepted_vertices)
    np.testing.assert_array_equal(np.asarray(layer.data[0], dtype=float), accepted_vertices)
    assert warnings == [shapes_annotation_edit_guard_module._POLYGON_DRAG_RENDERING_WARNING]
    assert data_actions == [ActionType.CHANGED]
    assert layer._is_moving is False
    assert layer._moving_value == (None, None)


def test_annotation_layer_edit_guard_direct_drag_preserves_application_and_restoration_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Report both transaction failures while still cleaning up the drag."""
    original_vertices = np.asarray(
        [[0.0, 0.0], [0.0, 10.0], [10.0, 10.0], [10.0, 0.0]],
        dtype=float,
    )
    moved_coordinate = np.asarray([0.0, 11.0])
    layer = Shapes([original_vertices], shape_type="polygon")
    layer._drag_modes = dict(layer._drag_modes)
    layer._drag_modes[Mode.DIRECT] = _direct_drag_callback_selecting_vertex(moved_vertex_index=1)
    data_actions: list[ActionType] = []
    finished_calls: list[str] = []
    highlight_calls: list[str] = []
    layer.events.data.connect(lambda event: data_actions.append(event.action))
    original_edit = layer._data_view.edit
    application_error = RuntimeError("simulated candidate application failure")
    restoration_error = RuntimeError("simulated accepted-row restoration failure")
    edit_calls = 0

    def fail_application_and_restoration(
        row_index: int,
        vertices: np.ndarray,
        *args: object,
        **kwargs: object,
    ) -> None:
        nonlocal edit_calls
        edit_calls += 1
        if edit_calls == 1:
            original_edit(row_index, vertices, *args, **kwargs)
            raise application_error
        raise restoration_error

    monkeypatch.setattr(layer._data_view, "edit", fail_application_and_restoration)
    monkeypatch.setattr(layer, "_set_highlight", lambda: highlight_calls.append("highlight"))
    guard = shapes_annotation_edit_guard_module._AnnotationLayerEditGuard(
        polygon_edit_finished_callback=lambda: finished_calls.append("finished")
    )
    guard.attach(layer)
    event = SimpleNamespace(type="mouse_press", position=tuple(original_vertices[1]))

    drag = layer._drag_modes[Mode.DIRECT](layer, event)
    assert next(drag) == "press"
    highlight_calls.clear()
    event.type = "mouse_move"
    event.position = tuple(moved_coordinate)
    with pytest.raises(ExceptionGroup) as caught:
        next(drag)

    assert caught.value.exceptions == (application_error, restoration_error)
    assert caught.value.__cause__ is application_error
    assert edit_calls == 2
    assert data_actions == []
    assert finished_calls == ["finished"]
    assert highlight_calls == ["highlight"]
    assert layer._is_moving is False
    assert layer._moving_value == (None, None)


def test_annotation_layer_edit_guard_direct_drag_warns_for_already_invalid_polygon() -> None:
    source = Polygon(
        [(0, 0), (4, 0), (4, 4), (0, 4)],
        holes=[[(6, 6), (6, 8), (8, 8), (8, 6)]],
    )
    invalid_vertices = shapely_polygon_to_napari_polygon_vertices(source)
    moved_coordinate = invalid_vertices[6] + np.asarray([1.0, 1.0])
    layer = Shapes([invalid_vertices], shape_type="polygon")
    layer._drag_modes = dict(layer._drag_modes)
    layer._drag_modes[Mode.DIRECT] = _direct_drag_callback_selecting_vertex(moved_vertex_index=6)
    warnings: list[str] = []
    guard = shapes_annotation_edit_guard_module._AnnotationLayerEditGuard(warning_callback=warnings.append)
    guard.attach(layer)
    event = SimpleNamespace(type="mouse_press", position=tuple(invalid_vertices[6]))

    drag = layer._drag_modes[Mode.DIRECT](layer, event)
    assert next(drag) == "press"
    event.type = "mouse_move"
    event.position = tuple(moved_coordinate)
    assert next(drag) is None
    event.type = "mouse_release"
    with pytest.raises(StopIteration):
        next(drag)

    edited_vertices = np.asarray(layer.data[0], dtype=float)
    np.testing.assert_allclose(edited_vertices, invalid_vertices)
    assert warnings == [shapes_annotation_edit_guard_module._ALREADY_INVALID_POLYGON_DRAG_WARNING]
    assert layer._moving_value == (None, None)


def test_annotation_layer_edit_guard_direct_drag_does_not_guess_malformed_topology() -> None:
    polygon, _ = _polygon_hole_roundtrip_fixture()
    original_vertices = shapely_polygon_to_napari_polygon_vertices(polygon)[:-1]
    moved_coordinate = np.asarray([1234.0, 2345.0])
    layer = Shapes([original_vertices], shape_type="polygon")
    layer._drag_modes = dict(layer._drag_modes)
    layer._drag_modes[Mode.DIRECT] = _direct_drag_callback_selecting_vertex(moved_vertex_index=0)
    warnings: list[str] = []
    guard = shapes_annotation_edit_guard_module._AnnotationLayerEditGuard(warning_callback=warnings.append)
    guard.attach(layer)
    event = SimpleNamespace(type="mouse_press", position=tuple(original_vertices[0]))

    drag = layer._drag_modes[Mode.DIRECT](layer, event)
    assert next(drag) == "press"
    event.type = "mouse_move"
    event.position = tuple(moved_coordinate)
    assert next(drag) is None
    event.type = "mouse_release"
    with pytest.raises(StopIteration):
        next(drag)

    edited_vertices = np.asarray(layer.data[0], dtype=float)
    np.testing.assert_allclose(edited_vertices, original_vertices)
    assert warnings == [shapes_annotation_edit_guard_module._ALREADY_INVALID_POLYGON_DRAG_WARNING]
    assert layer._moving_value == (None, None)


@pytest.mark.parametrize("target_kind", ["simple", "shell", "later_hole"])
def test_annotation_layer_edit_guard_vertex_insert_commits_valid_polygon_candidate(
    target_kind: str,
) -> None:
    """Guard simple and genuine shell/hole edges through one candidate API."""
    if target_kind == "simple":
        original_vertices = np.asarray(
            [[0.0, 0.0], [0.0, 20.0], [20.0, 20.0], [20.0, 0.0]],
            dtype=float,
        )
        insert_index = 2
    else:
        shell_yx = np.asarray(
            [[0.0, 0.0], [0.0, 20.0], [20.0, 20.0], [20.0, 0.0]],
            dtype=float,
        )
        first_hole_yx = np.asarray(
            [[3.0, 3.0], [3.0, 6.0], [6.0, 6.0], [6.0, 3.0]],
            dtype=float,
        )
        second_hole_yx = np.asarray(
            [[12.0, 12.0], [12.0, 15.0], [15.0, 15.0], [15.0, 12.0]],
            dtype=float,
        )
        polygon = Polygon(
            _yx_to_xy(shell_yx),
            holes=[_yx_to_xy(first_hole_yx), _yx_to_xy(second_hole_yx)],
        )
        original_vertices = shapely_polygon_to_napari_polygon_vertices(polygon)
        topology = napari_polygon_vertices_to_topology(original_vertices)
        insert_index = 2 if target_kind == "shell" else topology.hole_anchor_groups[-1][0] + 2

    inserted_coordinate = np.mean(original_vertices[[insert_index - 1, insert_index]], axis=0)
    original_topology = napari_polygon_vertices_to_topology(original_vertices)
    expected_vertices, expected_topology = insert_napari_polygon_vertex(
        original_vertices,
        original_topology,
        insert_index,
        inserted_coordinate,
    )
    path_vertices = np.asarray([[30.0, 0.0], [32.0, 2.0], [34.0, 0.0]], dtype=float)
    layer = Shapes(
        [original_vertices, path_vertices],
        shape_type=["polygon", "path"],
        features=pd.DataFrame(
            {
                "source_identity": ["polygon-id", "path-id"],
                "label": ["edited", "untouched"],
            }
        ),
    )
    layer.edge_color = np.asarray([_rgba("#112233"), _rgba("#445566")])
    layer.face_color = np.asarray([_rgba("#01020344"), _rgba("#05060744")])
    layer.edge_width = [3, 7]
    layer.z_index = [2, 5]
    layer.opacity = 0.42
    layer._drag_modes = dict(layer._drag_modes)
    original_features = layer.features.copy(deep=True)
    events: list[tuple[ActionType, tuple[int, ...], tuple[tuple[int, ...], ...]]] = []
    finished_calls: list[str] = []

    def original_vertex_insert_callback(*args: object, **kwargs: object) -> None:
        raise AssertionError("polygon insertion should be guarded")

    def record_data_event(event: object) -> None:
        events.append((event.action, event.data_indices, event.vertex_indices))

    layer._drag_modes[Mode.VERTEX_INSERT] = original_vertex_insert_callback
    layer.events.data.connect(record_data_event)
    guard = shapes_annotation_edit_guard_module._AnnotationLayerEditGuard(
        polygon_edit_finished_callback=lambda: finished_calls.append("finished")
    )
    guard.attach(layer)
    layer.mode = Mode.VERTEX_INSERT
    layer.selected_data = {0}
    layer.current_edge_color = "#abcdef"
    layer.current_face_color = "#12345678"
    layer.current_edge_width = 11
    original_feature_defaults = layer.feature_defaults.copy(deep=True)
    original_edge_color = np.asarray(layer.edge_color).copy()
    original_face_color = np.asarray(layer.face_color).copy()
    original_edge_width = list(layer.edge_width)

    layer.mouse_drag_callbacks[0](
        layer,
        SimpleNamespace(position=tuple(inserted_coordinate)),
    )

    np.testing.assert_allclose(np.asarray(layer.data[0], dtype=float), expected_vertices)
    np.testing.assert_allclose(np.asarray(layer.data[1], dtype=float), path_vertices)
    assert napari_polygon_vertices_to_topology(layer.data[0]) == expected_topology
    assert list(layer.shape_type) == ["polygon", "path"]
    pd.testing.assert_frame_equal(layer.features, original_features)
    pd.testing.assert_frame_equal(layer.feature_defaults, original_feature_defaults)
    np.testing.assert_allclose(layer.edge_color, original_edge_color)
    np.testing.assert_allclose(layer.face_color, original_face_color)
    assert layer.edge_width == original_edge_width
    assert layer.z_index == [2, 5]
    assert layer.opacity == 0.42
    np.testing.assert_allclose(to_rgba(layer.current_edge_color), to_rgba("#abcdef"))
    np.testing.assert_allclose(to_rgba(layer.current_face_color), to_rgba("#12345678"))
    assert layer.current_edge_width == 11
    assert layer.mode == Mode.VERTEX_INSERT
    assert set(layer.selected_data) == {0}
    assert layer._data_view._vertices_index.tolist() == [
        0,
        len(expected_vertices),
        len(expected_vertices) + len(path_vertices),
    ]
    inserted_hit = layer.get_value(inserted_coordinate, world=True)
    assert inserted_hit is not None
    assert int(inserted_hit[0]) == 0
    assert events == [
        (ActionType.CHANGING, (0,), ((insert_index,),)),
        (ActionType.CHANGED, (0,), ((insert_index,),)),
    ]
    assert finished_calls == ["finished"]


@pytest.mark.parametrize("rejection_kind", ["bridge", "invalid_simple"])
def test_annotation_layer_edit_guard_vertex_insert_rejects_invalid_candidate_before_mutation(
    monkeypatch: pytest.MonkeyPatch,
    rejection_kind: str,
) -> None:
    """Reject the selected raw edge rather than falling back to native code."""
    if rejection_kind == "bridge":
        shell_yx = np.asarray(
            [[0.0, 0.0], [0.0, 10.0], [10.0, 10.0], [10.0, 0.0]],
            dtype=float,
        )
        hole_yx = np.asarray(
            [[3.0, 3.0], [3.0, 6.0], [6.0, 6.0], [6.0, 3.0]],
            dtype=float,
        )
        original_vertices = shapely_polygon_to_napari_polygon_vertices(
            Polygon(_yx_to_xy(shell_yx), holes=[_yx_to_xy(hole_yx)])
        )
        topology = napari_polygon_vertices_to_topology(original_vertices)
        shell_end = topology.shell_anchor_group[1]
        inserted_coordinate = np.mean(original_vertices[[shell_end, shell_end + 1]], axis=0)
    else:
        original_vertices = np.asarray(
            [[0.0, 0.0], [0.0, 4.0], [4.0, 4.0], [4.0, 0.0]],
            dtype=float,
        )
        inserted_coordinate = np.asarray([0.0, -6.0])

    layer = Shapes([original_vertices], shape_type="polygon")
    layer._drag_modes = dict(layer._drag_modes)
    events: list[object] = []
    warnings: list[str] = []
    finished_calls: list[str] = []

    def original_vertex_insert_callback(*args: object, **kwargs: object) -> None:
        raise AssertionError("rejected polygon insertion must not call native code")

    layer._drag_modes[Mode.VERTEX_INSERT] = original_vertex_insert_callback
    layer.events.data.connect(events.append)
    guard = shapes_annotation_edit_guard_module._AnnotationLayerEditGuard(
        warning_callback=warnings.append,
        polygon_edit_finished_callback=lambda: finished_calls.append("finished"),
    )
    guard.attach(layer)
    layer.mode = Mode.VERTEX_INSERT
    layer.selected_data = {0}
    monkeypatch.setattr(layer, "refresh", lambda: pytest.fail("rejected insertion refreshed the layer"))

    layer.mouse_drag_callbacks[0](
        layer,
        SimpleNamespace(position=tuple(inserted_coordinate)),
    )

    np.testing.assert_allclose(np.asarray(layer.data[0], dtype=float), original_vertices)
    assert warnings == [shapes_annotation_edit_guard_module._INVALID_POLYGON_DRAG_WARNING]
    assert events == []
    assert finished_calls == []


def test_annotation_layer_edit_guard_vertex_insert_rejects_malformed_polygon() -> None:
    polygon, _ = _polygon_hole_roundtrip_fixture()
    malformed_vertices = shapely_polygon_to_napari_polygon_vertices(polygon)[:-1]
    inserted_coordinate = np.mean(malformed_vertices[[0, 1]], axis=0)
    layer = Shapes([malformed_vertices], shape_type="polygon")
    layer._drag_modes = dict(layer._drag_modes)
    events: list[object] = []
    warnings: list[str] = []

    def original_vertex_insert_callback(*args: object, **kwargs: object) -> None:
        raise AssertionError("malformed polygon insertion must not call native code")

    layer._drag_modes[Mode.VERTEX_INSERT] = original_vertex_insert_callback
    layer.events.data.connect(events.append)
    guard = shapes_annotation_edit_guard_module._AnnotationLayerEditGuard(warning_callback=warnings.append)
    guard.attach(layer)
    layer.mode = Mode.VERTEX_INSERT
    layer.selected_data = {0}

    layer.mouse_drag_callbacks[0](
        layer,
        SimpleNamespace(position=tuple(inserted_coordinate)),
    )

    np.testing.assert_allclose(np.asarray(layer.data[0], dtype=float), malformed_vertices)
    assert warnings == [shapes_annotation_edit_guard_module._ALREADY_INVALID_POLYGON_DRAG_WARNING]
    assert events == []


@pytest.mark.parametrize("delegation_kind", ["no_edge", "mixed_selection"])
def test_annotation_layer_edit_guard_vertex_insert_delegates_native_target(
    delegation_kind: str,
) -> None:
    if delegation_kind == "no_edge":
        layer = Shapes([], ndim=2)
        event = SimpleNamespace(position=(0.0, 0.0))
    else:
        polygon_vertices = np.asarray(
            [[20.0, 20.0], [20.0, 24.0], [24.0, 24.0], [24.0, 20.0]],
            dtype=float,
        )
        path_vertices = np.asarray([[0.0, 0.0], [0.0, 4.0], [4.0, 4.0]], dtype=float)
        layer = Shapes(
            [polygon_vertices, path_vertices],
            shape_type=["polygon", "path"],
        )
        layer.selected_data = {0, 1}
        event = SimpleNamespace(position=(0.0, 2.0))

    layer._drag_modes = dict(layer._drag_modes)
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def original_vertex_insert_callback(*args: object, **kwargs: object) -> str:
        calls.append((args, kwargs))
        return "delegated"

    layer._drag_modes[Mode.VERTEX_INSERT] = original_vertex_insert_callback
    guard = shapes_annotation_edit_guard_module._AnnotationLayerEditGuard()
    guard.attach(layer)

    result = layer._drag_modes[Mode.VERTEX_INSERT](layer, event, source="test")

    assert result == "delegated"
    assert calls == [((layer, event), {"source": "test"})]


def test_annotation_layer_edit_guard_vertex_insert_restores_after_commit_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Restore the row-change baseline after a late insertion failure.

    Perform the real longer-row rebuild, fail the final commit refresh, restore
    the original layer baseline, and report the insertion-rendering warning.
    Only ``CHANGING`` remains for the recovered attempt and the shared finished
    callback is withheld. A later valid insertion can still commit.
    """
    original_vertices = np.asarray(
        [[0.0, 0.0], [0.0, 4.0], [4.0, 4.0], [4.0, 0.0]],
        dtype=float,
    )
    path_vertices = np.asarray([[8.0, 0.0], [9.0, 2.0], [10.0, 0.0]], dtype=float)
    insert_index = 1
    inserted_coordinate = np.mean(original_vertices[[0, 1]], axis=0)
    original_topology = napari_polygon_vertices_to_topology(original_vertices)
    expected_inserted_vertices, _ = insert_napari_polygon_vertex(
        original_vertices,
        original_topology,
        insert_index,
        inserted_coordinate,
    )
    layer = Shapes(
        [original_vertices, path_vertices],
        shape_type=["polygon", "path"],
        features=pd.DataFrame(
            {
                "instance_id": ["polygon-0", "path-1"],
                "label": ["first", "second"],
            },
        ),
    )
    identity_guard = shapes_annotation_identity_defaults_module._AnnotationIdentityFeatureDefaultGuard()
    identity_guard.attach(layer, feature_name="instance_id")
    layer._drag_modes = dict(layer._drag_modes)
    original_data = [np.asarray(vertices).copy() for vertices in layer.data]
    original_shape_types = list(layer.shape_type)
    original_features = layer.features.copy(deep=True)
    events: list[ActionType] = []
    warnings: list[str] = []
    finished_calls: list[str] = []

    def original_vertex_insert_callback(*args: object, **kwargs: object) -> None:
        raise AssertionError("polygon insertion should be guarded")

    layer._drag_modes[Mode.VERTEX_INSERT] = original_vertex_insert_callback
    layer.events.data.connect(lambda event: events.append(event.action))
    guard = shapes_annotation_edit_guard_module._AnnotationLayerEditGuard(
        warning_callback=warnings.append,
        polygon_edit_finished_callback=lambda: finished_calls.append("finished"),
    )
    original_replace = guard._replace_shape_row_preserving_layer_state
    original_refresh = layer.refresh
    fail_next_refresh = False

    def replace_then_arm_refresh_failure(
        bound_layer: Shapes,
        row_index: int,
        vertices: np.ndarray,
    ) -> None:
        nonlocal fail_next_refresh
        original_replace(bound_layer, row_index, vertices)
        fail_next_refresh = True

    def fail_once_refresh(*args: object, **kwargs: object) -> None:
        nonlocal fail_next_refresh
        if fail_next_refresh:
            fail_next_refresh = False
            raise RuntimeError("simulated final insertion refresh failure")
        original_refresh(*args, **kwargs)

    monkeypatch.setattr(
        guard,
        "_replace_shape_row_preserving_layer_state",
        replace_then_arm_refresh_failure,
    )
    monkeypatch.setattr(layer, "refresh", fail_once_refresh)
    guard.attach(layer)
    layer.mode = Mode.VERTEX_INSERT
    layer.selected_data = {0}
    layer.feature_defaults = pd.DataFrame(
        {
            "instance_id": [pd.NA],
            "label": ["next-polygon"],
        },
    )

    layer.mouse_drag_callbacks[0](
        layer,
        SimpleNamespace(position=tuple(inserted_coordinate)),
    )

    _assert_layer_data_unchanged(layer, original_data)
    assert list(layer.shape_type) == original_shape_types
    assert napari_polygon_vertices_to_topology(layer.data[0]) == original_topology
    pd.testing.assert_frame_equal(layer.features, original_features)
    _assert_identity_feature_default_missing(layer, "instance_id")
    assert layer.feature_defaults["label"].iloc[0] == "next-polygon"
    assert layer.mode == Mode.VERTEX_INSERT
    assert set(layer.selected_data) == {0}
    assert layer._data_view._vertices_index.tolist() == [
        0,
        len(original_vertices),
        len(original_vertices) + len(path_vertices),
    ]
    restored_hit = layer.get_value(original_vertices[0], world=True)
    assert restored_hit is not None
    assert int(restored_hit[0]) == 0
    assert events == [ActionType.CHANGING]
    assert warnings == [shapes_annotation_edit_guard_module._POLYGON_INSERT_RENDERING_WARNING]
    assert finished_calls == []

    monkeypatch.setattr(guard, "_replace_shape_row_preserving_layer_state", original_replace)
    layer.mouse_drag_callbacks[0](
        layer,
        SimpleNamespace(position=tuple(inserted_coordinate)),
    )

    np.testing.assert_allclose(np.asarray(layer.data[0], dtype=float), expected_inserted_vertices)
    assert events == [
        ActionType.CHANGING,
        ActionType.CHANGING,
        ActionType.CHANGED,
    ]
    assert finished_calls == ["finished"]


def test_annotation_layer_edit_guard_vertex_insert_preserves_application_and_restoration_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve both errors when insertion application and recovery fail."""
    original_vertices = np.asarray(
        [[0.0, 0.0], [0.0, 4.0], [4.0, 4.0], [4.0, 0.0]],
        dtype=float,
    )
    inserted_coordinate = np.mean(original_vertices[[0, 1]], axis=0)
    layer = Shapes([original_vertices], shape_type="polygon")
    layer._drag_modes = dict(layer._drag_modes)
    events: list[ActionType] = []
    warnings: list[str] = []
    finished_calls: list[str] = []

    def original_vertex_insert_callback(*args: object, **kwargs: object) -> None:
        raise AssertionError("polygon insertion should be guarded")

    layer._drag_modes[Mode.VERTEX_INSERT] = original_vertex_insert_callback
    layer.events.data.connect(lambda event: events.append(event.action))
    guard = shapes_annotation_edit_guard_module._AnnotationLayerEditGuard(
        warning_callback=warnings.append,
        polygon_edit_finished_callback=lambda: finished_calls.append("finished"),
    )
    original_replace = guard._replace_shape_row_preserving_layer_state
    application_error = RuntimeError("simulated insertion application failure")
    restoration_error = RuntimeError("simulated insertion baseline restoration failure")

    def fail_after_rebuild(
        bound_layer: Shapes,
        row_index: int,
        vertices: np.ndarray,
    ) -> None:
        original_replace(bound_layer, row_index, vertices)
        raise application_error

    def fail_restoration(
        bound_layer: Shapes,
        baseline: shapes_annotation_layer_state_module._ShapesLayerBaseline,
    ) -> None:
        raise restoration_error

    monkeypatch.setattr(guard, "_replace_shape_row_preserving_layer_state", fail_after_rebuild)
    monkeypatch.setattr(
        shapes_annotation_edit_guard_module,
        "_restore_shapes_layer_baseline",
        fail_restoration,
    )
    guard.attach(layer)
    layer.mode = Mode.VERTEX_INSERT
    layer.selected_data = {0}

    with pytest.raises(ExceptionGroup) as caught:
        layer.mouse_drag_callbacks[0](
            layer,
            SimpleNamespace(position=tuple(inserted_coordinate)),
        )

    assert caught.value.exceptions == (application_error, restoration_error)
    assert caught.value.__cause__ is application_error
    assert events == [ActionType.CHANGING]
    assert warnings == []
    assert finished_calls == []


def test_annotation_layer_edit_guard_vertex_remove_delegates_when_no_vertex(monkeypatch) -> None:
    polygon, _ = _polygon_hole_roundtrip_fixture()
    layer = Shapes([shapely_polygon_to_napari_polygon_vertices(polygon)], shape_type="polygon")
    layer._drag_modes = dict(layer._drag_modes)
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def original_vertex_remove_callback(*args: object, **kwargs: object) -> str:
        calls.append((args, kwargs))
        return "delegated"

    monkeypatch.setattr(layer, "get_value", lambda position, world=True: (0, None))
    layer._drag_modes[Mode.VERTEX_REMOVE] = original_vertex_remove_callback
    guard = shapes_annotation_edit_guard_module._AnnotationLayerEditGuard()
    guard.attach(layer)
    event = SimpleNamespace(position=(0.0, 0.0))

    result = layer._drag_modes[Mode.VERTEX_REMOVE](layer, event)

    assert result == "delegated"
    assert calls == [((layer, event), {})]


def test_annotation_layer_edit_guard_vertex_remove_delegates_non_polygon(monkeypatch) -> None:
    path_vertices = np.asarray([[0.0, 0.0], [1.0, 2.0], [3.0, 3.0]], dtype=float)
    layer = Shapes([path_vertices], shape_type="path")
    layer._drag_modes = dict(layer._drag_modes)
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def original_vertex_remove_callback(*args: object, **kwargs: object) -> str:
        calls.append((args, kwargs))
        return "delegated"

    monkeypatch.setattr(layer, "get_value", lambda position, world=True: (0, 1))
    layer._drag_modes[Mode.VERTEX_REMOVE] = original_vertex_remove_callback
    guard = shapes_annotation_edit_guard_module._AnnotationLayerEditGuard()
    guard.attach(layer)
    event = SimpleNamespace(position=(0.0, 0.0))

    result = layer._drag_modes[Mode.VERTEX_REMOVE](layer, event)

    assert result == "delegated"
    assert calls == [((layer, event), {})]


@pytest.mark.parametrize("unsafe_case", ["invalid_index", "invalid_geometry", "nonfinite", "nonnumeric"])
def test_annotation_layer_edit_guard_vertex_remove_rejects_unsafe_polygon_hit(
    monkeypatch,
    unsafe_case: str,
) -> None:
    vertices = np.asarray([[0.0, 0.0], [0.0, 4.0], [4.0, 4.0], [4.0, 0.0]], dtype=float)
    hit = (0, 0)
    if unsafe_case == "invalid_geometry":
        vertices = np.asarray([[0.0, 0.0], [4.0, 4.0], [0.0, 4.0], [4.0, 0.0]], dtype=float)
    elif unsafe_case == "invalid_index":
        hit = (0, len(vertices))

    layer = Shapes([vertices], shape_type="polygon")
    if unsafe_case in {"nonfinite", "nonnumeric"}:
        unsafe_vertices = np.asarray(vertices, dtype=object if unsafe_case == "nonnumeric" else float)
        unsafe_vertices[0, 0] = "not-a-coordinate" if unsafe_case == "nonnumeric" else np.nan
        # Inject the malformed row after construction so this routing test
        # reaches Spatiato without asking napari to triangulate unsafe input first.
        layer._data_view.shapes[0]._data = unsafe_vertices
    original_vertices = np.asarray(layer.data[0]).copy()
    layer._drag_modes = dict(layer._drag_modes)
    warnings: list[str] = []
    events: list[object] = []
    finished_calls: list[str] = []
    get_value_calls = 0

    def get_value(position: object, world: bool = True) -> tuple[int, int]:
        nonlocal get_value_calls
        get_value_calls += 1
        return hit

    def original_vertex_remove_callback(*args: object, **kwargs: object) -> None:
        raise AssertionError("unsafe polygon deletion should not fall back to napari")

    monkeypatch.setattr(layer, "get_value", get_value)
    monkeypatch.setattr(layer, "refresh", lambda *args, **kwargs: pytest.fail("rejected deletion refreshed the layer"))
    layer._drag_modes[Mode.VERTEX_REMOVE] = original_vertex_remove_callback
    layer.events.data.connect(events.append)
    guard = shapes_annotation_edit_guard_module._AnnotationLayerEditGuard(
        warning_callback=warnings.append,
        polygon_edit_finished_callback=lambda: finished_calls.append("finished"),
    )
    guard.attach(layer)

    result = layer._drag_modes[Mode.VERTEX_REMOVE](layer, SimpleNamespace(position=(0.0, 0.0)))

    assert result is None
    np.testing.assert_array_equal(np.asarray(layer.data[0]), original_vertices)
    assert get_value_calls == 1
    assert warnings == [shapes_annotation_edit_guard_module._ALREADY_INVALID_POLYGON_DRAG_WARNING]
    assert events == []
    assert finished_calls == []


def test_annotation_layer_edit_guard_vertex_remove_guards_valid_explicitly_closed_simple_polygon(monkeypatch) -> None:
    _, polygon = _polygon_hole_roundtrip_fixture()
    original_vertices = shapely_polygon_to_napari_polygon_vertices(polygon)
    expected_vertices = np.delete(original_vertices, 1, axis=0)
    _ = napari_polygon_vertices_to_shapely_polygon(expected_vertices)
    layer = Shapes([original_vertices], shape_type="polygon")
    layer._drag_modes = dict(layer._drag_modes)
    events: list[ActionType] = []

    def original_vertex_remove_callback(*args: object, **kwargs: object) -> None:
        raise AssertionError("simple polygon deletion should be guarded")

    monkeypatch.setattr(layer, "get_value", lambda position, world=True: (0, 1))
    layer._drag_modes[Mode.VERTEX_REMOVE] = original_vertex_remove_callback
    layer.events.data.connect(lambda event: events.append(event.action))
    finished_calls: list[str] = []
    guard = shapes_annotation_edit_guard_module._AnnotationLayerEditGuard(
        polygon_edit_finished_callback=lambda: finished_calls.append("finished"),
    )
    guard.attach(layer)
    event = SimpleNamespace(position=(0.0, 0.0))

    result = layer._drag_modes[Mode.VERTEX_REMOVE](layer, event)

    assert result is None
    np.testing.assert_allclose(np.asarray(layer.data[0], dtype=float), expected_vertices)
    np.testing.assert_allclose(np.asarray(layer.data[0], dtype=float)[0], np.asarray(layer.data[0], dtype=float)[-1])
    assert events == [ActionType.CHANGING, ActionType.CHANGED]
    assert finished_calls == ["finished"]


def test_annotation_layer_edit_guard_vertex_remove_restores_after_commit_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Restore the complete deletion baseline after a late commit failure.

    Perform the real shortened-row rebuild, then inject an exception before the
    final refresh and success notifications. The exception is caught, the
    original layer baseline is restored, and a renderer-failure warning is
    reported. ``CHANGING`` has already been emitted, but ``CHANGED`` and the
    delete-finished callback are withheld. Later movement and deletion remain
    usable.
    """
    polygon, simple_polygon = _polygon_hole_roundtrip_fixture()
    original_vertices = shapely_polygon_to_napari_polygon_vertices(polygon)
    second_vertices = shapely_polygon_to_napari_polygon_vertices(simple_polygon)
    deleted_vertex_index = 1
    layer = Shapes(
        [original_vertices, second_vertices],
        shape_type=["polygon", "polygon"],
        features=pd.DataFrame(
            {
                "instance_id": ["polygon-0", "polygon-1"],
                "label": ["first", "second"],
            },
        ),
    )
    identity_guard = shapes_annotation_identity_defaults_module._AnnotationIdentityFeatureDefaultGuard()
    identity_guard.attach(layer, feature_name="instance_id")
    layer.feature_defaults = pd.DataFrame(
        {
            "instance_id": [pd.NA],
            "label": ["next-polygon"],
        },
    )
    layer.edge_color = np.asarray([_rgba("#112233"), _rgba("#445566")])
    layer.face_color = np.asarray([_rgba("#01020344"), _rgba("#05060744")])
    layer.edge_width = [2, 4]
    layer.z_index = [3, 5]
    layer.opacity = 0.42
    layer.current_edge_color = "#abcdef"
    layer.current_face_color = "#12345678"
    layer.current_edge_width = 11
    layer.mode = Mode.VERTEX_REMOVE
    layer.selected_data = set()
    layer._drag_modes = dict(layer._drag_modes)
    layer._drag_modes[Mode.DIRECT] = _direct_drag_callback_selecting_vertex(moved_vertex_index=2)
    original_data = [np.asarray(vertices).copy() for vertices in layer.data]
    original_shape_types = list(layer.shape_type)
    original_features = layer.features.copy(deep=True)
    original_edge_color = np.asarray(layer.edge_color).copy()
    original_face_color = np.asarray(layer.face_color).copy()
    original_edge_width = list(layer.edge_width)
    original_z_index = list(layer.z_index)
    events: list[ActionType] = []
    warnings: list[str] = []
    finished_calls: list[str] = []

    def original_vertex_remove_callback(*args: object, **kwargs: object) -> None:
        raise AssertionError("polygon deletion should be guarded")

    monkeypatch.setattr(
        layer,
        "get_value",
        lambda position, world=True: (0, deleted_vertex_index),
    )
    layer._drag_modes[Mode.VERTEX_REMOVE] = original_vertex_remove_callback
    layer.events.data.connect(lambda event: events.append(event.action))
    guard = shapes_annotation_edit_guard_module._AnnotationLayerEditGuard(
        warning_callback=warnings.append,
        polygon_edit_finished_callback=lambda: finished_calls.append("finished"),
    )
    original_replace = guard._replace_shape_row_preserving_layer_state
    commit_error = RuntimeError("simulated failure after the deletion row was rebuilt")

    def fail_after_rebuild(bound_layer: Shapes, row_index: int, vertices: np.ndarray) -> None:
        original_replace(bound_layer, row_index, vertices)
        raise commit_error

    monkeypatch.setattr(guard, "_replace_shape_row_preserving_layer_state", fail_after_rebuild)
    guard.attach(layer)

    layer._drag_modes[Mode.VERTEX_REMOVE](
        layer,
        SimpleNamespace(position=tuple(original_vertices[deleted_vertex_index])),
    )

    _assert_layer_data_unchanged(layer, original_data)
    assert list(layer.shape_type) == original_shape_types
    pd.testing.assert_frame_equal(layer.features, original_features)
    _assert_identity_feature_default_missing(layer, "instance_id")
    assert layer.feature_defaults["label"].iloc[0] == "next-polygon"
    np.testing.assert_allclose(layer.edge_color, original_edge_color)
    np.testing.assert_allclose(layer.face_color, original_face_color)
    assert layer.edge_width == original_edge_width
    assert layer.z_index == original_z_index
    assert layer.opacity == 0.42
    np.testing.assert_allclose(to_rgba(layer.current_edge_color), to_rgba("#abcdef"))
    np.testing.assert_allclose(to_rgba(layer.current_face_color), to_rgba("#12345678"))
    assert layer.current_edge_width == 11
    assert layer.mode == Mode.VERTEX_REMOVE
    assert set(layer.selected_data) == set()
    assert layer._data_view._vertices_index.tolist() == [
        0,
        len(original_data[0]),
        len(original_data[0]) + len(original_data[1]),
    ]
    restored_hit = layer.get_value(original_data[0][0], world=True)
    assert restored_hit is not None
    assert int(restored_hit[0]) == 0
    assert warnings == [shapes_annotation_edit_guard_module._POLYGON_DELETE_RENDERING_WARNING]
    assert events == [ActionType.CHANGING]
    assert finished_calls == []

    # Recovery leaves no persistent unsafe state: a later move and deletion can
    # both commit through the same guard.
    monkeypatch.setattr(guard, "_replace_shape_row_preserving_layer_state", original_replace)
    move_event = SimpleNamespace(type="mouse_press", position=tuple(layer.data[0][2]))
    drag = layer._drag_modes[Mode.DIRECT](layer, move_event)
    assert next(drag) == "press"
    move_event.type = "mouse_move"
    move_event.position = tuple(np.asarray(layer.data[0][2], dtype=float) + np.asarray([0.1, 0.1]))
    assert next(drag) is None
    move_event.type = "mouse_release"
    with pytest.raises(StopIteration):
        next(drag)

    layer._drag_modes[Mode.VERTEX_REMOVE](
        layer,
        SimpleNamespace(position=tuple(layer.data[0][deleted_vertex_index])),
    )

    # The recovered deletion contributes CHANGING only. The first CHANGED is
    # from the later successful movement; the final CHANGING/CHANGED pair is
    # from the later successful deletion. The shared finished callback runs
    # once for each of those two successful follow-up polygon edits.
    assert events == [
        ActionType.CHANGING,
        ActionType.CHANGED,
        ActionType.CHANGING,
        ActionType.CHANGED,
    ]
    assert finished_calls == ["finished", "finished"]


def test_annotation_layer_edit_guard_vertex_remove_restores_after_final_refresh_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Include the final layer refresh in the deletion transaction boundary."""
    _, polygon = _polygon_hole_roundtrip_fixture()
    original_vertices = shapely_polygon_to_napari_polygon_vertices(polygon)
    deleted_vertex_index = 1
    layer = Shapes([original_vertices], shape_type="polygon")
    layer._drag_modes = dict(layer._drag_modes)
    events: list[ActionType] = []
    warnings: list[str] = []
    finished_calls: list[str] = []

    def original_vertex_remove_callback(*args: object, **kwargs: object) -> None:
        raise AssertionError("polygon deletion should be guarded")

    monkeypatch.setattr(layer, "get_value", lambda position, world=True: (0, deleted_vertex_index))
    layer._drag_modes[Mode.VERTEX_REMOVE] = original_vertex_remove_callback
    layer.events.data.connect(lambda event: events.append(event.action))
    guard = shapes_annotation_edit_guard_module._AnnotationLayerEditGuard(
        warning_callback=warnings.append,
        polygon_edit_finished_callback=lambda: finished_calls.append("finished"),
    )
    original_replace = guard._replace_shape_row_preserving_layer_state
    original_refresh = layer.refresh
    fail_next_refresh = False

    def replace_then_arm_refresh_failure(
        bound_layer: Shapes,
        row_index: int,
        vertices: np.ndarray,
    ) -> None:
        nonlocal fail_next_refresh
        original_replace(bound_layer, row_index, vertices)
        fail_next_refresh = True

    def fail_once_refresh(*args: object, **kwargs: object) -> None:
        nonlocal fail_next_refresh
        if fail_next_refresh:
            fail_next_refresh = False
            raise RuntimeError("simulated final deletion refresh failure")
        original_refresh(*args, **kwargs)

    monkeypatch.setattr(
        guard,
        "_replace_shape_row_preserving_layer_state",
        replace_then_arm_refresh_failure,
    )
    monkeypatch.setattr(layer, "refresh", fail_once_refresh)
    guard.attach(layer)

    layer._drag_modes[Mode.VERTEX_REMOVE](
        layer,
        SimpleNamespace(position=tuple(original_vertices[deleted_vertex_index])),
    )

    np.testing.assert_allclose(np.asarray(layer.data[0], dtype=float), original_vertices)
    assert warnings == [shapes_annotation_edit_guard_module._POLYGON_DELETE_RENDERING_WARNING]
    assert events == [ActionType.CHANGING]
    assert finished_calls == []


def test_annotation_layer_edit_guard_vertex_remove_preserves_application_and_restoration_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve both errors when deletion application and recovery fail."""
    _, polygon = _polygon_hole_roundtrip_fixture()
    original_vertices = shapely_polygon_to_napari_polygon_vertices(polygon)
    deleted_vertex_index = 1
    layer = Shapes([original_vertices], shape_type="polygon")
    layer._drag_modes = dict(layer._drag_modes)
    events: list[ActionType] = []
    warnings: list[str] = []
    finished_calls: list[str] = []

    def original_vertex_remove_callback(*args: object, **kwargs: object) -> None:
        raise AssertionError("polygon deletion should be guarded")

    monkeypatch.setattr(layer, "get_value", lambda position, world=True: (0, deleted_vertex_index))
    layer._drag_modes[Mode.VERTEX_REMOVE] = original_vertex_remove_callback
    layer.events.data.connect(lambda event: events.append(event.action))
    guard = shapes_annotation_edit_guard_module._AnnotationLayerEditGuard(
        warning_callback=warnings.append,
        polygon_edit_finished_callback=lambda: finished_calls.append("finished"),
    )
    original_replace = guard._replace_shape_row_preserving_layer_state
    application_error = RuntimeError("simulated deletion application failure")
    restoration_error = RuntimeError("simulated deletion baseline restoration failure")

    def fail_after_rebuild(bound_layer: Shapes, row_index: int, vertices: np.ndarray) -> None:
        original_replace(bound_layer, row_index, vertices)
        raise application_error

    def fail_restoration(
        bound_layer: Shapes,
        baseline: shapes_annotation_layer_state_module._ShapesLayerBaseline,
    ) -> None:
        raise restoration_error

    monkeypatch.setattr(guard, "_replace_shape_row_preserving_layer_state", fail_after_rebuild)
    monkeypatch.setattr(
        shapes_annotation_edit_guard_module,
        "_restore_shapes_layer_baseline",
        fail_restoration,
    )
    guard.attach(layer)

    with pytest.raises(ExceptionGroup) as caught:
        layer._drag_modes[Mode.VERTEX_REMOVE](
            layer,
            SimpleNamespace(position=tuple(original_vertices[deleted_vertex_index])),
        )

    assert caught.value.exceptions == (application_error, restoration_error)
    assert caught.value.__cause__ is application_error
    assert warnings == []
    assert events == [ActionType.CHANGING]
    assert finished_calls == []


def test_annotation_layer_edit_guard_vertex_remove_guards_valid_implicitly_closed_simple_polygon(monkeypatch) -> None:
    original_vertices = np.asarray(
        [[0.0, 0.0], [0.0, 4.0], [2.0, 5.0], [4.0, 4.0], [4.0, 0.0]],
        dtype=float,
    )
    expected_vertices = np.delete(original_vertices, 2, axis=0)
    layer = Shapes([original_vertices], shape_type="polygon")
    layer._drag_modes = dict(layer._drag_modes)

    def original_vertex_remove_callback(*args: object, **kwargs: object) -> None:
        raise AssertionError("simple polygon deletion should be guarded")

    monkeypatch.setattr(layer, "get_value", lambda position, world=True: (0, 2))
    layer._drag_modes[Mode.VERTEX_REMOVE] = original_vertex_remove_callback
    guard = shapes_annotation_edit_guard_module._AnnotationLayerEditGuard()
    guard.attach(layer)

    layer._drag_modes[Mode.VERTEX_REMOVE](layer, SimpleNamespace(position=tuple(original_vertices[2])))

    np.testing.assert_allclose(np.asarray(layer.data[0], dtype=float), expected_vertices)
    assert not np.array_equal(layer.data[0][0], layer.data[0][-1])
    _ = napari_polygon_vertices_to_shapely_polygon(layer.data[0])


@pytest.mark.parametrize("deleted_vertex_index", [0, 4])
def test_annotation_layer_edit_guard_vertex_remove_deletes_explicit_closure_endpoint_alias(
    monkeypatch,
    deleted_vertex_index: int,
) -> None:
    original_vertices = np.asarray(
        [[0.0, 0.0], [0.0, 4.0], [4.0, 4.0], [4.0, 0.0], [0.0, 0.0]],
        dtype=float,
    )
    expected_vertices = np.asarray([[0.0, 4.0], [4.0, 4.0], [4.0, 0.0], [0.0, 4.0]], dtype=float)
    layer = Shapes([original_vertices], shape_type="polygon")
    layer._drag_modes = dict(layer._drag_modes)

    def original_vertex_remove_callback(*args: object, **kwargs: object) -> None:
        raise AssertionError("closure endpoint deletion should be guarded")

    monkeypatch.setattr(
        layer,
        "get_value",
        lambda position, world=True: (0, deleted_vertex_index),
    )
    layer._drag_modes[Mode.VERTEX_REMOVE] = original_vertex_remove_callback
    guard = shapes_annotation_edit_guard_module._AnnotationLayerEditGuard()
    guard.attach(layer)

    layer._drag_modes[Mode.VERTEX_REMOVE](
        layer, SimpleNamespace(position=tuple(original_vertices[deleted_vertex_index]))
    )

    np.testing.assert_allclose(np.asarray(layer.data[0], dtype=float), expected_vertices)
    _ = napari_polygon_vertices_to_shapely_polygon(layer.data[0])


def test_annotation_layer_edit_guard_vertex_remove_rejects_invalid_concave_simple_polygon_deletion(
    monkeypatch,
) -> None:
    original_vertices = make_concave_simple_polygon_deletion_regression_vertices()
    layer = Shapes([original_vertices], shape_type="polygon")
    layer._drag_modes = dict(layer._drag_modes)
    warnings: list[str] = []
    events: list[object] = []
    finished_calls: list[str] = []

    def original_vertex_remove_callback(*args: object, **kwargs: object) -> None:
        raise AssertionError("invalid simple polygon deletion should not fall back to napari")

    monkeypatch.setattr(layer, "get_value", lambda position, world=True: (0, 0))
    monkeypatch.setattr(layer, "refresh", lambda *args, **kwargs: pytest.fail("rejected deletion refreshed the layer"))
    layer._drag_modes[Mode.VERTEX_REMOVE] = original_vertex_remove_callback
    layer.events.data.connect(events.append)
    guard = shapes_annotation_edit_guard_module._AnnotationLayerEditGuard(
        warning_callback=warnings.append,
        polygon_edit_finished_callback=lambda: finished_calls.append("finished"),
    )
    guard.attach(layer)

    layer._drag_modes[Mode.VERTEX_REMOVE](layer, SimpleNamespace(position=tuple(original_vertices[0])))

    np.testing.assert_allclose(np.asarray(layer.data[0], dtype=float), original_vertices)
    assert warnings == ["Polygon path cannot be converted to a valid polygon."]
    assert events == []
    assert finished_calls == []


@pytest.mark.parametrize(
    ("explicitly_closed", "deleted_vertex_index"),
    [
        (False, 0),
        (False, 1),
        (False, 2),
        (True, 0),
        (True, 1),
        (True, 2),
        (True, 3),
    ],
)
def test_annotation_layer_edit_guard_vertex_remove_removes_semantic_triangle(
    monkeypatch,
    explicitly_closed: bool,
    deleted_vertex_index: int,
) -> None:
    semantic_vertices = np.asarray([[0.0, 0.0], [0.0, 4.0], [4.0, 0.0]], dtype=float)
    original_vertices = np.vstack([semantic_vertices, semantic_vertices[0]]) if explicitly_closed else semantic_vertices
    layer = Shapes([original_vertices], shape_type="polygon")
    layer.mode = Mode.VERTEX_REMOVE
    layer.selected_data = {0}
    layer._drag_modes = dict(layer._drag_modes)
    events: list[tuple[ActionType, tuple[int, ...], tuple[tuple[int, ...], ...]]] = []
    finished_calls: list[str] = []

    def original_vertex_remove_callback(*args: object, **kwargs: object) -> None:
        raise AssertionError("semantic triangle deletion should be guarded")

    def record_data_event(event: object) -> None:
        events.append((event.action, event.data_indices, event.vertex_indices))

    monkeypatch.setattr(
        layer,
        "get_value",
        lambda position, world=True: (0, deleted_vertex_index),
    )
    layer._drag_modes[Mode.VERTEX_REMOVE] = original_vertex_remove_callback
    layer.events.data.connect(record_data_event)
    guard = shapes_annotation_edit_guard_module._AnnotationLayerEditGuard(
        polygon_edit_finished_callback=lambda: finished_calls.append("finished"),
    )
    guard.attach(layer)

    layer._drag_modes[Mode.VERTEX_REMOVE](layer, SimpleNamespace(position=(0.0, 0.0)))

    assert layer.data == []
    assert layer._data_view._vertices_index.tolist() == [0]
    assert set(layer.selected_data) == set()
    assert layer.mode == Mode.VERTEX_REMOVE
    assert events == [
        (ActionType.CHANGING, (0,), ((deleted_vertex_index,),)),
        (ActionType.CHANGED, (0,), ((deleted_vertex_index,),)),
    ]
    assert finished_calls == ["finished"]


def test_annotation_layer_edit_guard_semantic_triangle_removal_preserves_multirow_layer_state(monkeypatch) -> None:
    rectangle_vertices = np.asarray([[0.0, 0.0], [0.0, 2.0], [2.0, 2.0], [2.0, 0.0]], dtype=float)
    triangle_vertices = np.asarray([[4.0, 0.0], [4.0, 3.0], [7.0, 0.0], [4.0, 0.0]], dtype=float)
    path_vertices = np.asarray([[9.0, 0.0], [10.0, 2.0], [11.0, 0.0]], dtype=float)
    original_surviving_data = [rectangle_vertices, path_vertices]
    layer = Shapes(
        [rectangle_vertices, triangle_vertices, path_vertices],
        shape_type=["rectangle", "polygon", "path"],
        features=pd.DataFrame(
            {
                "source_identity": ["rectangle-id", "triangle-id", "path-id"],
                "label": ["first", "deleted", "last"],
            },
        ),
    )
    layer.edge_color = np.asarray([_rgba("#112233"), _rgba("#445566"), _rgba("#778899")])
    layer.face_color = np.asarray([_rgba("#01020344"), _rgba("#05060744"), _rgba("#090a0b44")])
    layer.edge_width = [2, 4, 6]
    layer.z_index = [3, 5, 7]
    layer.opacity = 0.42
    layer.current_edge_color = "#abcdef"
    layer.current_face_color = "#12345678"
    layer.current_edge_width = 11
    layer.mode = Mode.VERTEX_REMOVE
    layer.selected_data = {0, 1, 2}
    layer._drag_modes = dict(layer._drag_modes)
    expected_features = layer.features.iloc[[0, 2]].reset_index(drop=True)
    expected_edge_color = np.asarray(layer.edge_color)[[0, 2]].copy()
    expected_face_color = np.asarray(layer.face_color)[[0, 2]].copy()
    steps: list[object] = []
    original_refresh = layer.refresh

    def original_vertex_remove_callback(*args: object, **kwargs: object) -> None:
        raise AssertionError("semantic triangle deletion should be guarded")

    def refresh(*args: object, **kwargs: object) -> None:
        steps.append("refresh")
        original_refresh(*args, **kwargs)

    def record_data_event(event: object) -> None:
        steps.append((event.action, event.data_indices, event.vertex_indices))

    monkeypatch.setattr(layer, "get_value", lambda position, world=True: (1, 3))
    monkeypatch.setattr(layer, "refresh", refresh)
    layer._drag_modes[Mode.VERTEX_REMOVE] = original_vertex_remove_callback
    layer.events.data.connect(record_data_event)
    guard = shapes_annotation_edit_guard_module._AnnotationLayerEditGuard(
        polygon_edit_finished_callback=lambda: steps.append("finished"),
    )
    guard.attach(layer)

    layer._drag_modes[Mode.VERTEX_REMOVE](layer, SimpleNamespace(position=tuple(triangle_vertices[-1])))

    _assert_layer_data_unchanged(layer, original_surviving_data)
    assert list(layer.shape_type) == ["rectangle", "path"]
    pd.testing.assert_frame_equal(layer.features, expected_features)
    np.testing.assert_allclose(layer.edge_color, expected_edge_color)
    np.testing.assert_allclose(layer.face_color, expected_face_color)
    assert layer.edge_width == [2, 6]
    assert layer.z_index == [3, 7]
    assert layer.opacity == 0.42
    np.testing.assert_allclose(to_rgba(layer.current_edge_color), to_rgba("#abcdef"))
    np.testing.assert_allclose(to_rgba(layer.current_face_color), to_rgba("#12345678"))
    assert layer.current_edge_width == 11
    assert layer.mode == Mode.VERTEX_REMOVE
    assert set(layer.selected_data) == {0, 1}
    assert layer._data_view._vertices_index.tolist() == [0, 4, 7]
    assert steps[0] == (ActionType.CHANGING, (1,), ((3,),))
    assert steps[-2:] == [(ActionType.CHANGED, (1,), ((3,),)), "finished"]
    assert steps[1:-2]
    assert all(step == "refresh" for step in steps[1:-2])


def test_annotation_layer_edit_guard_semantic_triangle_removal_restores_after_commit_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Restore a removed mixed-layer row and every shifted row attribute."""
    rectangle_vertices = np.asarray([[0.0, 0.0], [0.0, 2.0], [2.0, 2.0], [2.0, 0.0]], dtype=float)
    triangle_vertices = np.asarray([[4.0, 0.0], [4.0, 3.0], [7.0, 0.0], [4.0, 0.0]], dtype=float)
    path_vertices = np.asarray([[9.0, 0.0], [10.0, 2.0], [11.0, 0.0]], dtype=float)
    layer = Shapes(
        [rectangle_vertices, triangle_vertices, path_vertices],
        shape_type=["rectangle", "polygon", "path"],
        features=pd.DataFrame(
            {
                "source_identity": ["rectangle-id", "triangle-id", "path-id"],
                "label": ["first", "deleted", "last"],
            },
        ),
    )
    layer.edge_color = np.asarray([_rgba("#112233"), _rgba("#445566"), _rgba("#778899")])
    layer.face_color = np.asarray([_rgba("#01020344"), _rgba("#05060744"), _rgba("#090a0b44")])
    layer.edge_width = [2, 4, 6]
    layer.z_index = [3, 5, 7]
    layer.opacity = 0.42
    layer.current_edge_color = "#abcdef"
    layer.current_face_color = "#12345678"
    layer.current_edge_width = 11
    layer.mode = Mode.VERTEX_REMOVE
    layer.selected_data = {0, 1, 2}
    layer._drag_modes = dict(layer._drag_modes)
    original_data = [np.asarray(vertices).copy() for vertices in layer.data]
    original_shape_types = list(layer.shape_type)
    original_features = layer.features.copy(deep=True)
    original_edge_color = np.asarray(layer.edge_color).copy()
    original_face_color = np.asarray(layer.face_color).copy()
    events: list[ActionType] = []
    warnings: list[str] = []
    finished_calls: list[str] = []

    def original_vertex_remove_callback(*args: object, **kwargs: object) -> None:
        raise AssertionError("semantic triangle deletion should be guarded")

    monkeypatch.setattr(layer, "get_value", lambda position, world=True: (1, 3))
    layer._drag_modes[Mode.VERTEX_REMOVE] = original_vertex_remove_callback
    layer.events.data.connect(lambda event: events.append(event.action))
    guard = shapes_annotation_edit_guard_module._AnnotationLayerEditGuard(
        warning_callback=warnings.append,
        polygon_edit_finished_callback=lambda: finished_calls.append("finished"),
    )
    original_remove = guard._remove_shape_row_preserving_layer_state
    commit_error = RuntimeError("simulated failure after the polygon row was removed")

    def fail_after_remove(bound_layer: Shapes, row_index: int) -> None:
        original_remove(bound_layer, row_index)
        raise commit_error

    monkeypatch.setattr(guard, "_remove_shape_row_preserving_layer_state", fail_after_remove)
    guard.attach(layer)

    layer._drag_modes[Mode.VERTEX_REMOVE](
        layer,
        SimpleNamespace(position=tuple(triangle_vertices[-1])),
    )

    _assert_layer_data_unchanged(layer, original_data)
    assert list(layer.shape_type) == original_shape_types
    pd.testing.assert_frame_equal(layer.features, original_features)
    np.testing.assert_allclose(layer.edge_color, original_edge_color)
    np.testing.assert_allclose(layer.face_color, original_face_color)
    assert layer.edge_width == [2, 4, 6]
    assert layer.z_index == [3, 5, 7]
    assert layer.opacity == 0.42
    np.testing.assert_allclose(to_rgba(layer.current_edge_color), to_rgba("#abcdef"))
    np.testing.assert_allclose(to_rgba(layer.current_face_color), to_rgba("#12345678"))
    assert layer.current_edge_width == 11
    assert layer.mode == Mode.VERTEX_REMOVE
    assert set(layer.selected_data) == {0, 1, 2}
    assert layer._data_view._vertices_index.tolist() == [0, 4, 8, 11]
    restored_hit = layer.get_value(triangle_vertices[-1], world=True)
    assert restored_hit is not None
    assert int(restored_hit[0]) == 1
    assert warnings == [shapes_annotation_edit_guard_module._POLYGON_DELETE_RENDERING_WARNING]
    assert events == [ActionType.CHANGING]
    assert finished_calls == []


def test_annotation_layer_edit_guard_vertex_remove_uses_helper_for_hole_bearing_polygon(monkeypatch) -> None:
    polygon, _ = _polygon_hole_roundtrip_fixture()
    deleted_vertex_indices = [2, 8, 0, 12, 6]

    for deleted_vertex_index in deleted_vertex_indices:
        original_vertices = shapely_polygon_to_napari_polygon_vertices(polygon)
        topology = napari_polygon_vertices_to_topology(original_vertices)
        expected_deletion = delete_napari_polygon_vertex(original_vertices, topology, deleted_vertex_index)
        assert expected_deletion.vertices is not None
        expected_vertices = expected_deletion.vertices
        layer = Shapes([original_vertices], shape_type="polygon")
        layer._drag_modes = dict(layer._drag_modes)
        event = SimpleNamespace(position=(0.0, 0.0))
        events: list[tuple[ActionType, tuple[int, ...], tuple[tuple[int, ...], ...]]] = []

        def original_vertex_remove_callback(*args: object, **kwargs: object) -> None:
            raise AssertionError("hole-bearing polygon deletion should use the topology helper")

        def record_data_event(
            event: object,
            event_log: list[tuple[ActionType, tuple[int, ...], tuple[tuple[int, ...], ...]]] = events,
        ) -> None:
            event_log.append((event.action, event.data_indices, event.vertex_indices))

        monkeypatch.setattr(
            layer,
            "get_value",
            lambda position, world=True, index=deleted_vertex_index: (0, index),
        )
        layer._drag_modes[Mode.VERTEX_REMOVE] = original_vertex_remove_callback
        layer.events.data.connect(record_data_event)
        guard = shapes_annotation_edit_guard_module._AnnotationLayerEditGuard()
        guard.attach(layer)

        layer._drag_modes[Mode.VERTEX_REMOVE](layer, event)

        np.testing.assert_allclose(np.asarray(layer.data[0], dtype=float), expected_vertices)
        assert events == [
            (ActionType.CHANGING, (0,), ((deleted_vertex_index,),)),
            (ActionType.CHANGED, (0,), ((deleted_vertex_index,),)),
        ]


def test_annotation_layer_edit_guard_vertex_remove_removes_minimal_hole(monkeypatch) -> None:
    shell_yx = np.asarray([[0.0, 0.0], [0.0, 10.0], [10.0, 10.0], [10.0, 0.0]], dtype=float)
    hole_yx = np.asarray([[3.0, 3.0], [3.0, 5.0], [5.0, 4.0]], dtype=float)
    polygon = Polygon(_yx_to_xy(shell_yx), holes=[_yx_to_xy(hole_yx)])
    original_vertices = shapely_polygon_to_napari_polygon_vertices(polygon)
    topology = napari_polygon_vertices_to_topology(original_vertices)
    expected_deletion = delete_napari_polygon_vertex(original_vertices, topology, 5)
    assert expected_deletion.vertices is not None
    assert expected_deletion.topology is not None
    expected_vertices = expected_deletion.vertices
    expected_topology = expected_deletion.topology
    layer = Shapes([original_vertices], shape_type="polygon")
    layer._drag_modes = dict(layer._drag_modes)

    def original_vertex_remove_callback(*args: object, **kwargs: object) -> None:
        raise AssertionError("minimal-hole deletion should use the topology helper")

    monkeypatch.setattr(layer, "get_value", lambda position, world=True: (0, 5))
    layer._drag_modes[Mode.VERTEX_REMOVE] = original_vertex_remove_callback
    guard = shapes_annotation_edit_guard_module._AnnotationLayerEditGuard()
    guard.attach(layer)

    layer._drag_modes[Mode.VERTEX_REMOVE](layer, SimpleNamespace(position=(0.0, 0.0)))

    np.testing.assert_allclose(np.asarray(layer.data[0], dtype=float), expected_vertices)
    assert not expected_topology.synchronized_anchor_groups


def test_annotation_layer_edit_guard_vertex_remove_rebuilds_cache_after_shortening_hole_row() -> None:
    polygon, simple_polygon = _polygon_hole_roundtrip_fixture()
    original_hole_vertices = shapely_polygon_to_napari_polygon_vertices(polygon)
    original_simple_vertices = shapely_polygon_to_napari_polygon_vertices(simple_polygon)
    layer = Shapes(
        [original_hole_vertices, original_simple_vertices],
        shape_type=["polygon", "polygon"],
        features=pd.DataFrame({"instance_id": ["hole_row", "simple_row"]}),
    )
    layer.mode = Mode.VERTEX_REMOVE
    layer.selected_data = {0}
    layer._drag_modes = dict(layer._drag_modes)

    def original_vertex_remove_callback(*args: object, **kwargs: object) -> None:
        raise AssertionError("hole-bearing polygon deletion should use the topology helper")

    layer._drag_modes[Mode.VERTEX_REMOVE] = original_vertex_remove_callback
    original_features = layer.features.copy()
    original_shape_types = list(layer.shape_type)
    events: list[tuple[ActionType, tuple[int, ...], tuple[tuple[int, ...], ...]]] = []

    def record_data_event(event: object) -> None:
        events.append((event.action, event.data_indices, event.vertex_indices))

    layer.events.data.connect(record_data_event)
    guard = shapes_annotation_edit_guard_module._AnnotationLayerEditGuard()
    guard.attach(layer)

    layer._drag_modes[Mode.VERTEX_REMOVE](layer, SimpleNamespace(position=original_hole_vertices[7]))

    shortened_vertices = np.asarray(layer.data[0], dtype=float)
    assert len(shortened_vertices) == len(original_hole_vertices) - 1
    assert layer._data_view._vertices_index.tolist() == [
        0,
        len(shortened_vertices),
        len(shortened_vertices) + len(original_simple_vertices),
    ]
    shell_hit = layer.get_value(shortened_vertices[0], world=True)
    assert shell_hit is not None
    assert tuple(int(index) for index in shell_hit) == (0, len(shortened_vertices) - 1)
    pd.testing.assert_frame_equal(layer.features, original_features)
    assert list(layer.shape_type) == original_shape_types
    assert layer.mode == Mode.VERTEX_REMOVE
    assert set(layer.selected_data) == {0}

    layer._drag_modes[Mode.VERTEX_REMOVE](layer, SimpleNamespace(position=shortened_vertices[0]))

    shell_deleted_vertices = np.asarray(layer.data[0], dtype=float)
    assert len(shell_deleted_vertices) == len(shortened_vertices) - 1
    assert len(napari_polygon_vertices_to_shapely_polygon(shell_deleted_vertices).interiors) == 1
    pd.testing.assert_frame_equal(layer.features, original_features)
    assert list(layer.shape_type) == original_shape_types
    assert layer.mode == Mode.VERTEX_REMOVE
    assert set(layer.selected_data) == {0}
    assert events == [
        (ActionType.CHANGING, (0,), ((7,),)),
        (ActionType.CHANGED, (0,), ((7,),)),
        (ActionType.CHANGING, (0,), ((len(shortened_vertices) - 1,),)),
        (ActionType.CHANGED, (0,), ((len(shortened_vertices) - 1,),)),
    ]


def test_annotation_layer_edit_guard_vertex_remove_preserves_styles_with_stale_napari_color_arrays(monkeypatch) -> None:
    polygon, simple_polygon = _polygon_hole_roundtrip_fixture()
    original_hole_vertices = shapely_polygon_to_napari_polygon_vertices(polygon)
    original_simple_vertices = shapely_polygon_to_napari_polygon_vertices(simple_polygon)
    deleted_vertex_index = 7
    topology = napari_polygon_vertices_to_topology(original_hole_vertices)
    expected_deletion = delete_napari_polygon_vertex(
        original_hole_vertices,
        topology,
        deleted_vertex_index,
    )
    assert expected_deletion.vertices is not None
    expected_vertices = expected_deletion.vertices
    layer = Shapes(
        [original_hole_vertices, original_simple_vertices],
        shape_type=["polygon", "polygon"],
    )
    apply_primary_shapes_layer_style(layer)
    layer.mode = Mode.VERTEX_REMOVE
    layer.selected_data = {0}
    layer._drag_modes = dict(layer._drag_modes)

    current_edge_color = "#aa00aa"
    current_face_color = "#bbccdd44"
    current_edge_width = 9
    layer.current_edge_color = current_edge_color
    layer.current_face_color = current_face_color
    layer.current_edge_width = current_edge_width

    edge_color = np.asarray([_rgba("#00ffff"), _rgba("#123456")], dtype=float)
    face_color = np.asarray([_rgba("#00000000"), _rgba("#65432188")], dtype=float)
    edge_width = [3, 5]
    z_index = [11, 13]
    layer.edge_color = edge_color
    layer.face_color = face_color
    layer.edge_width = edge_width
    layer.z_index = z_index
    layer.opacity = 0.37

    # Simulate the intermittent napari state behind the UI bug: the logical
    # data rows are correct, but the private color arrays have stale extra rows.
    layer._data_view._edge_color = np.vstack([layer._data_view._edge_color, _rgba("#ffff00")])
    layer._data_view._face_color = np.vstack([layer._data_view._face_color, _rgba("#ffff00")])

    def original_vertex_remove_callback(*args: object, **kwargs: object) -> None:
        raise AssertionError("hole-bearing polygon deletion should use the topology helper")

    monkeypatch.setattr(layer, "get_value", lambda position, world=True: (0, deleted_vertex_index))
    layer._drag_modes[Mode.VERTEX_REMOVE] = original_vertex_remove_callback
    guard = shapes_annotation_edit_guard_module._AnnotationLayerEditGuard()
    guard.attach(layer)

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        layer._drag_modes[Mode.VERTEX_REMOVE](layer, SimpleNamespace(position=(0.0, 0.0)))

    style_warnings = [
        str(warning.message)
        for warning in caught_warnings
        if "edge_color" in str(warning.message) or "face_color" in str(warning.message)
    ]
    assert style_warnings == []
    np.testing.assert_allclose(np.asarray(layer.data[0], dtype=float), expected_vertices)
    assert set(layer.selected_data) == {0}
    assert layer.mode == Mode.VERTEX_REMOVE
    assert layer.opacity == 0.37
    assert layer.edge_width == edge_width
    assert layer.z_index == z_index
    np.testing.assert_allclose(layer.edge_color, edge_color)
    np.testing.assert_allclose(layer.face_color, face_color)
    np.testing.assert_allclose(to_rgba(layer.current_edge_color), to_rgba(current_edge_color))
    np.testing.assert_allclose(to_rgba(layer.current_face_color), to_rgba(current_face_color))
    assert layer.current_edge_width == current_edge_width


def test_annotation_layer_edit_guard_vertex_remove_rejects_malformed_topology(monkeypatch) -> None:
    polygon, _ = _polygon_hole_roundtrip_fixture()
    malformed_vertices = shapely_polygon_to_napari_polygon_vertices(polygon)[:-1]
    layer = Shapes([malformed_vertices], shape_type="polygon")
    layer._drag_modes = dict(layer._drag_modes)
    warnings: list[str] = []
    events: list[object] = []

    def original_vertex_remove_callback(*args: object, **kwargs: object) -> None:
        raise AssertionError("malformed polygon deletion should not fall back to napari")

    monkeypatch.setattr(layer, "get_value", lambda position, world=True: (0, 0))
    monkeypatch.setattr(layer, "refresh", lambda *args, **kwargs: pytest.fail("rejected deletion refreshed the layer"))
    layer._drag_modes[Mode.VERTEX_REMOVE] = original_vertex_remove_callback
    layer.events.data.connect(events.append)
    finished_calls: list[str] = []
    guard = shapes_annotation_edit_guard_module._AnnotationLayerEditGuard(
        warning_callback=warnings.append,
        polygon_edit_finished_callback=lambda: finished_calls.append("finished"),
    )
    guard.attach(layer)
    event = SimpleNamespace(position=(0.0, 0.0))

    result = layer._drag_modes[Mode.VERTEX_REMOVE](layer, event)

    assert result is None
    np.testing.assert_allclose(np.asarray(layer.data[0], dtype=float), malformed_vertices)
    assert warnings == [shapes_annotation_edit_guard_module._ALREADY_INVALID_POLYGON_DRAG_WARNING]
    assert events == []
    assert finished_calls == []


def test_annotation_layer_edit_guard_vertex_remove_helper_error_warns_without_mutating_layer(monkeypatch) -> None:
    polygon, _ = _polygon_hole_roundtrip_fixture()
    original_vertices = shapely_polygon_to_napari_polygon_vertices(polygon)
    layer = Shapes([original_vertices], shape_type="polygon")
    layer._drag_modes = dict(layer._drag_modes)
    warnings: list[str] = []
    events: list[object] = []

    def original_vertex_remove_callback(*args: object, **kwargs: object) -> None:
        raise AssertionError("rejected hole deletion should not fall back to napari deletion")

    def reject_delete(*args: object, **kwargs: object) -> NapariPolygonVertexDeletion:
        raise ValueError("Deletion would make the polygon invalid.")

    monkeypatch.setattr(layer, "get_value", lambda position, world=True: (0, 0))
    monkeypatch.setattr(shapes_annotation_edit_guard_module, "delete_napari_polygon_vertex", reject_delete)
    layer._drag_modes[Mode.VERTEX_REMOVE] = original_vertex_remove_callback
    layer.events.data.connect(events.append)
    guard = shapes_annotation_edit_guard_module._AnnotationLayerEditGuard(warning_callback=warnings.append)
    guard.attach(layer)

    layer._drag_modes[Mode.VERTEX_REMOVE](layer, SimpleNamespace(position=(0.0, 0.0)))

    np.testing.assert_allclose(np.asarray(layer.data[0], dtype=float), original_vertices)
    assert warnings == ["Deletion would make the polygon invalid."]
    assert events == []


def test_shapes_annotation_widget_shares_app_state(qtbot) -> None:
    viewer = DummyViewer()

    widget = _create_embedded_shapes_annotation(qtbot, viewer)

    assert widget.app_state is get_or_create_app_state(viewer)


def test_shapes_annotation_widget_evaluates_dirty_state_only_after_data_mutations(
    qtbot,
    monkeypatch,
) -> None:
    widget = ShapesAnnotation()
    qtbot.addWidget(widget)
    evaluations: list[None] = []
    monkeypatch.setattr(
        widget,
        "_refresh_save_and_dirty_state",
        lambda *, layer_actionable: evaluations.append(None),
    )

    for action in (ActionType.ADDING, ActionType.REMOVING, ActionType.CHANGING):
        widget._on_annotation_layer_content_changed(SimpleNamespace(type="data", action=action))

    assert evaluations == []

    for event in (
        SimpleNamespace(type="data", action=ActionType.ADDED),
        SimpleNamespace(type="data", action=ActionType.REMOVED),
        SimpleNamespace(type="data", action=ActionType.CHANGED),
    ):
        widget._on_annotation_layer_content_changed(event)

    assert len(evaluations) == 3

    with pytest.raises(AttributeError):
        widget._on_annotation_layer_content_changed(SimpleNamespace(type="data"))
    with pytest.raises(ValueError, match="Unexpected Shapes annotation event type"):
        widget._on_annotation_layer_content_changed(SimpleNamespace(type="features"))


def test_shapes_annotation_widget_geometry_event_reuses_one_snapshot_without_reload_refresh(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    sdata_blobs: SpatialData,
) -> None:
    """One completed edit updates Save and parent dirty state without disk-facing refresh."""
    viewer = DummyViewer()
    widget = _create_ready_annotation_widget(qtbot, viewer, sdata_blobs)
    widget.create_layer_button.click()
    layer = widget._annotation_layer
    assert isinstance(layer, Shapes)
    real_capture = shapes_annotation_widget_module._capture_annotation_layer_snapshot
    snapshot_count = 0

    def record_snapshot(current_layer: Shapes):
        nonlocal snapshot_count
        snapshot_count += 1
        return real_capture(current_layer)

    def fail_reload_refresh() -> bool:
        raise AssertionError("Geometry events must not refresh Reload readiness.")

    monkeypatch.setattr(shapes_annotation_widget_module, "_capture_annotation_layer_snapshot", record_snapshot)
    monkeypatch.setattr(widget, "_refresh_reload_shapes_state", fail_reload_refresh)

    _add_polygon(layer)

    assert snapshot_count == 1
    assert widget.save_shapes_button.isEnabled() is True
    assert widget._test_parent.annotation_context.has_unsaved_shapes_changes is True


def test_shapes_annotation_widget_exact_geometry_restoration_disables_save(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer()
    app_state = get_or_create_app_state(viewer)
    app_state.set_sdata(sdata_blobs)
    widget = _create_embedded_shapes_annotation(qtbot, viewer)
    widget._test_parent.shapes_combo.setCurrentIndex(
        _combo_index_for_text(widget._test_parent.shapes_combo, "blobs_polygons")
    )
    layer = widget._annotation_layer
    assert isinstance(layer, Shapes)
    original_data = [np.asarray(vertices).copy() for vertices in layer.data]
    edited_data = [vertices.copy() for vertices in original_data]
    edited_data[0][0] += np.asarray([1.0, 1.0])

    layer.data = edited_data

    assert widget.save_shapes_button.isEnabled() is True
    assert widget._test_parent.annotation_context.has_unsaved_shapes_changes is True

    layer.data = original_data

    assert widget.save_shapes_button.isEnabled() is False
    assert widget._test_parent.annotation_context.has_unsaved_shapes_changes is False


def test_shapes_annotation_widget_validates_empty_invalid_and_duplicate_names(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer()
    app_state = get_or_create_app_state(viewer)
    app_state.set_sdata(sdata_blobs)
    widget = _create_embedded_shapes_annotation(qtbot, viewer)
    assert "Shapes element name must not be empty" in _status_text(widget)

    widget.name_edit.setText("bad/name")
    assert widget.create_layer_button.isEnabled() is False
    assert "must be a valid SpatialData name" in _status_text(widget)

    existing_shapes_name = next(iter(sdata_blobs.shapes))
    widget.name_edit.setText(existing_shapes_name)
    assert widget.create_layer_button.isEnabled() is False
    assert "Name Already Exists" in _status_text(widget)

    widget.name_edit.setText(existing_shapes_name.upper())
    assert widget.create_layer_button.isEnabled() is False
    assert "Name Already Exists" in _status_text(widget)


def test_shapes_annotation_widget_status_cards_shorten_long_identifiers(
    qtbot,
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    coordinate_system = "global_long_name_" + "x" * 80
    shapes_name = "annotation_shapes_long_name_" + "y" * 80
    _patch_coordinate_system_names(monkeypatch, [coordinate_system])
    viewer = DummyViewer()
    app_state = get_or_create_app_state(viewer)
    app_state.set_sdata(sdata_blobs)
    widget = _create_embedded_shapes_annotation(qtbot, viewer)

    widget.name_edit.setText(shapes_name)

    status = _status_text(widget)
    assert "Ready" in status
    assert shapes_name not in status
    assert coordinate_system not in status
    assert "…" in status
    tooltip = _tooltip_text(widget)
    assert shapes_name in tooltip
    assert coordinate_system in tooltip

    def fake_create_shapes_element(request, napari_layer):
        # Keep this UI-formatting test independent of geometry conversion and
        # core coordinate-system validation, while still materializing the
        # saved target that the parent must discover after first save.
        del napari_layer
        saved_shapes = request.sdata.shapes["blobs_polygons"].copy()
        set_transformation(saved_shapes, Identity(), to_coordinate_system=request.coordinate_system)
        request.sdata.shapes[request.shapes_name] = saved_shapes
        return AnnotateShapesElementResult(
            shapes_name=request.shapes_name,
            coordinate_system=request.coordinate_system,
            row_count=2,
        )

    monkeypatch.setattr(
        shapes_annotation_widget_module,
        "create_shapes_element_from_napari_shapes_layer",
        fake_create_shapes_element,
    )
    widget.create_layer_button.click()
    layer = widget._annotation_layer
    assert isinstance(layer, Shapes)
    _add_polygon(layer)
    widget.save_shapes_button.click()

    status = _status_text(widget)
    assert "Shapes Saved" in status
    assert shapes_name not in status
    assert coordinate_system not in status
    assert "…" in status
    tooltip = _tooltip_text(widget)
    assert shapes_name in tooltip
    assert coordinate_system in tooltip


def test_shapes_annotation_widget_create_layer_adds_registered_active_empty_shapes_layer(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer()
    widget = _create_ready_annotation_widget(qtbot, viewer, sdata_blobs)

    assert widget.create_layer_button.isEnabled() is True
    assert _SPACE_PAN_TIP_TEXT not in _status_text(widget)
    widget.create_layer_button.click()

    assert len(viewer.layers) == 1
    layer = viewer.layers[0]
    assert isinstance(layer, Shapes)
    assert layer.name == "new_regions"
    assert len(layer.data) == 0
    assert layer.ndim == 2
    assert layer.current_edge_width == 1
    np.testing.assert_allclose(to_rgba(layer.current_edge_color), to_rgba("#00FFFF"))
    np.testing.assert_allclose(to_rgba(layer.current_face_color), to_rgba(PRIMARY_SHAPES_FACE_COLOR))
    assert layer.opacity == 0.8
    assert hasattr(layer, _SHAPES_EDGE_WIDTH_SYNC_CALLBACK_ATTR)
    assert hasattr(layer, _SHAPES_EDGE_COLOR_SYNC_CALLBACK_ATTR)
    status = _status_text(widget)
    assert "Annotation Layer Ready" in status
    assert "Draw shapes in the viewer, then click Save shapes." in status
    assert _SPACE_PAN_TIP_TEXT in status

    binding = widget.app_state.viewer_adapter.layer_bindings.get_binding(layer)
    assert isinstance(binding, ShapesLayerBinding)
    assert binding.element_name == "new_regions"
    assert binding.coordinate_system == "global"
    assert binding.shapes_role == "primary"
    assert binding.shapes_rendering_mode == "shapes"
    assert binding.style_spec is None
    assert binding.source_shapes_index_feature_name == "instance_id"
    assert viewer.layers.selection.active is layer

    assert widget.validated_shapes_name == "new_regions"
    assert widget._annotation_layer is layer
    assert widget._annotation_edit_guard.layer is layer
    assert "_drag_modes" in vars(layer)
    assert layer._drag_modes[Mode.DIRECT] is widget._annotation_edit_guard._wrapped_direct_callback
    assert widget._annotation_shapes_name == "new_regions"
    assert widget._annotation_coordinate_system == "global"
    assert widget._annotation_session is not None
    assert widget.name_edit.isEnabled() is False
    assert widget.create_layer_button.isEnabled() is False
    assert widget.save_shapes_button.isEnabled() is False
    assert "Annotation Layer Ready" in _status_text(widget)


def test_shapes_annotation_widget_invalid_drag_warning_clears_after_release(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer()
    widget = _create_ready_annotation_widget(qtbot, viewer, sdata_blobs)
    widget.create_layer_button.click()
    layer = widget._annotation_layer
    assert isinstance(layer, Shapes)
    polygon, _ = _polygon_hole_roundtrip_fixture()
    original_vertices = shapely_polygon_to_napari_polygon_vertices(polygon)
    layer.add_polygons(original_vertices)
    _install_direct_drag_callback_for_annotation_guard(
        widget,
        layer,
        moved_vertex_index=8,
    )
    event = SimpleNamespace(type="mouse_press", position=tuple(original_vertices[8]))

    drag = layer._drag_modes[Mode.DIRECT](layer, event)
    assert next(drag) == "press"
    event.type = "mouse_move"
    event.position = (10_000.0, 10_000.0)
    assert next(drag) is None

    status = _status_text(widget)
    assert "Edit Rejected" in status
    assert shapes_annotation_edit_guard_module._INVALID_POLYGON_DRAG_WARNING in status

    event.type = "mouse_release"
    with pytest.raises(StopIteration):
        next(drag)

    status = _status_text(widget)
    assert "Annotation Layer Ready" in status
    assert shapes_annotation_edit_guard_module._INVALID_POLYGON_DRAG_WARNING not in status


def test_shapes_annotation_widget_already_invalid_drag_warning_clears_after_rejected_release(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer()
    widget = _create_ready_annotation_widget(qtbot, viewer, sdata_blobs)
    widget.create_layer_button.click()
    layer = widget._annotation_layer
    assert isinstance(layer, Shapes)
    source = Polygon(
        [(0, 0), (4, 0), (4, 4), (0, 4)],
        holes=[[(6, 6), (6, 8), (8, 8), (8, 6)]],
    )
    invalid_vertices = shapely_polygon_to_napari_polygon_vertices(source)
    layer.add_polygons(invalid_vertices)
    _install_direct_drag_callback_for_annotation_guard(
        widget,
        layer,
        moved_vertex_index=6,
    )
    event = SimpleNamespace(type="mouse_press", position=tuple(invalid_vertices[6]))

    drag = layer._drag_modes[Mode.DIRECT](layer, event)
    assert next(drag) == "press"
    event.type = "mouse_move"
    event.position = tuple(invalid_vertices[6] + np.asarray([1.0, 1.0]))
    assert next(drag) is None

    event.type = "mouse_release"
    with pytest.raises(StopIteration):
        next(drag)

    status = _status_text(widget)
    assert "Annotation Layer Ready" in status
    assert shapes_annotation_edit_guard_module._ALREADY_INVALID_POLYGON_DRAG_WARNING not in status


def test_shapes_annotation_widget_annotation_edit_warning_uses_generic_title(qtbot) -> None:
    widget = _create_embedded_shapes_annotation(qtbot, DummyViewer())

    widget._set_annotation_edit_warning("Deletion would make the polygon invalid.")

    status = _status_text(widget)
    assert "Edit Rejected" in status
    assert "Deletion would make the polygon invalid." in status
    assert "Could Not Delete Vertex" not in status


def test_shapes_annotation_widget_successful_vertex_delete_clears_stale_edit_warning(
    qtbot,
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer()
    widget = _create_ready_annotation_widget(qtbot, viewer, sdata_blobs)
    widget.create_layer_button.click()
    layer = widget._annotation_layer
    assert isinstance(layer, Shapes)
    polygon, _ = _polygon_hole_roundtrip_fixture()
    original_vertices = shapely_polygon_to_napari_polygon_vertices(polygon)
    layer.add_polygons(original_vertices)
    original_delete_helper = shapes_annotation_edit_guard_module.delete_napari_polygon_vertex
    hit_values = iter([(0, 0), (0, 8)])
    call_count = 0

    def reject_then_delete(
        vertices: np.ndarray,
        topology: object,
        deleted_vertex_index: int,
    ) -> NapariPolygonVertexDeletion:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ValueError("Polygon holes must be contained by the exterior ring.")
        return original_delete_helper(vertices, topology, deleted_vertex_index)

    monkeypatch.setattr(layer, "get_value", lambda position, world=True: next(hit_values))
    monkeypatch.setattr(shapes_annotation_edit_guard_module, "delete_napari_polygon_vertex", reject_then_delete)

    layer._drag_modes[Mode.VERTEX_REMOVE](layer, SimpleNamespace(position=(0.0, 0.0)))

    status = _status_text(widget)
    assert "Edit Rejected" in status
    assert "Polygon holes must be contained by the exterior ring." in status
    np.testing.assert_allclose(np.asarray(layer.data[0], dtype=float), original_vertices)

    layer._drag_modes[Mode.VERTEX_REMOVE](layer, SimpleNamespace(position=(0.0, 0.0)))

    status = _status_text(widget)
    assert "Annotation Layer Ready" in status
    assert "Edit Rejected" not in status
    assert "Polygon holes must be contained by the exterior ring." not in status
    assert call_count == 2


def test_shapes_annotation_widget_open_existing_target_loads_edit_session_layer(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer()
    app_state = get_or_create_app_state(viewer)
    app_state.set_sdata(sdata_blobs)
    widget = _create_embedded_shapes_annotation(qtbot, viewer)
    widget._test_parent.shapes_combo.setCurrentIndex(
        _combo_index_for_text(widget._test_parent.shapes_combo, "blobs_polygons")
    )

    assert len(viewer.layers) == 1
    layer = viewer.layers[0]
    assert isinstance(layer, Shapes)
    assert widget._annotation_layer is layer
    assert widget._annotation_edit_guard.layer is layer
    assert "_drag_modes" in vars(layer)
    assert layer._drag_modes[Mode.DIRECT] is widget._annotation_edit_guard._wrapped_direct_callback
    assert widget._annotation_shapes_name == "blobs_polygons"
    assert widget._annotation_coordinate_system == "global"
    assert widget._annotation_session is not None
    assert widget._annotation_session.mode == "edit_existing"
    assert widget._annotation_session.shapes_name == "blobs_polygons"
    assert widget._annotation_session.coordinate_system == "global"
    assert widget._annotation_session.source_shapes_index_feature_name == "index"
    assert widget._annotation_session.source_geodataframe is not sdata_blobs.shapes["blobs_polygons"]
    assert widget._annotation_session.source_geodataframe is not None
    assert widget._annotation_session.source_geodataframe.index.equals(sdata_blobs.shapes["blobs_polygons"].index)
    assert widget._annotation_session.source_geodataframe_index_name == sdata_blobs.shapes["blobs_polygons"].index.name
    assert widget._annotation_session.table_linked is False
    assert viewer.layers.selection.active is layer
    assert widget.create_layer_button.isEnabled() is False
    assert widget.save_shapes_button.isEnabled() is False
    assert widget.reload_shapes_button.isEnabled() is False
    assert "only for SpatialData opened from a zarr store" in widget.reload_shapes_button.toolTip()
    assert "Existing Shapes Opened" in _status_text(widget)


def test_shapes_annotation_widget_open_existing_target_adopts_loaded_primary_layer(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer()
    app_state = get_or_create_app_state(viewer)
    app_state.set_sdata(sdata_blobs)
    load_result = app_state.viewer_adapter.ensure_shapes_loaded(sdata_blobs, "blobs_polygons", "global")
    widget = _create_embedded_shapes_annotation(qtbot, viewer)
    widget._test_parent.shapes_combo.setCurrentIndex(
        _combo_index_for_text(widget._test_parent.shapes_combo, "blobs_polygons")
    )

    assert len(viewer.layers) == 1
    assert widget._annotation_layer is load_result.layer
    assert widget._annotation_session is not None
    assert viewer.layers.selection.active is load_result.layer


def test_shapes_annotation_widget_adopts_selected_target_loaded_from_viewer(
    qtbot,
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer()
    app_state = get_or_create_app_state(viewer)
    app_state.set_sdata(sdata_blobs)
    widget = _create_embedded_shapes_annotation(qtbot, viewer)
    target = shapes_annotation_widget_module._ShapesAnnotationTarget.edit_existing("blobs_polygons")
    original_open = widget._open_existing_annotation_layer
    monkeypatch.setattr(widget, "_open_existing_annotation_layer", lambda: None)
    widget._test_parent.shapes_combo.setCurrentIndex(
        _combo_index_for_text(widget._test_parent.shapes_combo, "blobs_polygons")
    )
    monkeypatch.setattr(widget, "_open_existing_annotation_layer", original_open)

    assert widget._annotation_context.shapes_target == target
    assert widget._annotation_layer is None
    assert widget._annotation_session is None
    assert widget.save_shapes_button.isEnabled() is False

    load_result = app_state.viewer_adapter.ensure_shapes_loaded(sdata_blobs, "blobs_polygons", "global")

    assert load_result.created is True
    assert widget._annotation_layer is load_result.layer
    assert widget._annotation_session is not None
    assert widget._annotation_session.shapes_name == "blobs_polygons"
    assert widget._annotation_session.coordinate_system == "global"
    assert widget.save_shapes_button.isEnabled() is False
    assert viewer.layers.selection.active is load_result.layer


def test_shapes_annotation_widget_ignores_viewer_loaded_nonmatching_target(
    qtbot,
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer()
    app_state = get_or_create_app_state(viewer)
    sdata_blobs.shapes["other_polygons"] = sdata_blobs.shapes["blobs_polygons"].copy()
    app_state.set_sdata(sdata_blobs)
    widget = _create_embedded_shapes_annotation(qtbot, viewer)
    original_open = widget._open_existing_annotation_layer
    monkeypatch.setattr(widget, "_open_existing_annotation_layer", lambda: None)
    widget._test_parent.shapes_combo.setCurrentIndex(
        _combo_index_for_text(widget._test_parent.shapes_combo, "blobs_polygons")
    )
    monkeypatch.setattr(widget, "_open_existing_annotation_layer", original_open)

    load_result = app_state.viewer_adapter.ensure_shapes_loaded(sdata_blobs, "other_polygons", "global")

    assert load_result.created is True
    assert widget._annotation_layer is None
    assert widget._annotation_session is None
    assert widget.save_shapes_button.isEnabled() is False


def test_shapes_annotation_widget_viewer_load_does_not_steal_active_session(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer()
    app_state = get_or_create_app_state(viewer)
    sdata_blobs.shapes["other_polygons"] = sdata_blobs.shapes["blobs_polygons"].copy()
    app_state.set_sdata(sdata_blobs)
    widget = _create_embedded_shapes_annotation(qtbot, viewer)
    widget._test_parent.shapes_combo.setCurrentIndex(
        _combo_index_for_text(widget._test_parent.shapes_combo, "blobs_polygons")
    )
    annotation_layer = widget._annotation_layer

    load_result = app_state.viewer_adapter.ensure_shapes_loaded(sdata_blobs, "other_polygons", "global")

    assert load_result.created is True
    assert annotation_layer is not None
    assert widget._annotation_layer is annotation_layer
    assert widget._annotation_session is not None
    assert widget._annotation_session.shapes_name == "blobs_polygons"


def test_shapes_annotation_widget_lists_multipolygon_source_without_eager_validation(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer()
    app_state = get_or_create_app_state(viewer)
    app_state.set_sdata(sdata_blobs)
    widget = _create_embedded_shapes_annotation(qtbot, viewer)

    assert _combo_index_for_text(widget._test_parent.shapes_combo, "blobs_multipolygons") >= 0
    assert list(viewer.layers) == []
    assert widget._annotation_layer is None
    assert widget._annotation_session is None


def test_shapes_annotation_widget_validates_only_the_selected_existing_target(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer()
    app_state = get_or_create_app_state(viewer)
    app_state.set_sdata(sdata_blobs)
    widget = _create_embedded_shapes_annotation(qtbot, viewer)
    parent = widget._test_parent
    selected_source = sdata_blobs.shapes["blobs_polygons"]
    validated_sources: list[object] = []
    real_validate = annotation_widget_module.validate_existing_shapes_source_geodataframe

    def record_validation(source: object) -> gpd.GeoDataFrame:
        validated_sources.append(source)
        return real_validate(source)

    monkeypatch.setattr(
        annotation_widget_module,
        "validate_existing_shapes_source_geodataframe",
        record_validation,
    )
    monkeypatch.setattr(
        shapes_annotation_widget_module,
        "validate_existing_shapes_source_geodataframe",
        record_validation,
    )

    parent.shapes_combo.setCurrentIndex(_combo_index_for_text(parent.shapes_combo, "blobs_polygons"))

    assert len(validated_sources) == 1
    assert validated_sources[0] is selected_source
    assert widget._annotation_session is not None
    assert widget._annotation_session.shapes_name == "blobs_polygons"

    parent._refresh_shapes_targets()

    assert len(validated_sources) == 1


def test_shapes_annotation_widget_rejects_invalid_target_before_releasing_session(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    backed_sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer()
    app_state = get_or_create_app_state(viewer)
    app_state.set_sdata(backed_sdata_blobs)
    widget = _create_embedded_shapes_annotation(qtbot, viewer)
    parent = widget._test_parent
    valid_shapes_name = "blobs_polygons"
    parent.shapes_combo.setCurrentIndex(_combo_index_for_text(parent.shapes_combo, valid_shapes_name))
    layer = widget._annotation_layer
    session = widget._annotation_session
    assert isinstance(layer, Shapes)
    assert session is not None

    # Invalid geometry remains visible because selector refresh is names-only.
    # Validation happens now, before the accepted session can be released.
    parent.shapes_combo.setCurrentIndex(_combo_index_for_text(parent.shapes_combo, "blobs_multipolygons"))

    assert parent.annotation_context.shapes_target == (
        shapes_annotation_widget_module._ShapesAnnotationTarget.edit_existing(valid_shapes_name)
    )
    assert parent.shapes_combo.currentText() == valid_shapes_name
    assert widget._annotation_layer is layer
    assert widget._annotation_session is session
    assert "Could Not Open Shapes" in _status_text(widget)
    assert "Polygon geometries only" in _status_text(widget)

    _add_polygon(layer, offset=100)
    assert parent.annotation_context.has_unsaved_shapes_changes is True
    confirmation_reasons: list[str] = []

    def cancel_reload(*, reason: str) -> bool:
        confirmation_reasons.append(reason)
        return False

    monkeypatch.setattr(widget, "_confirm_discard_annotation_layer", cancel_reload)
    widget.reload_shapes_button.click()

    assert confirmation_reasons == ["reload"]
    assert widget._annotation_layer is layer
    assert parent.annotation_context.has_unsaved_shapes_changes is True


def test_shapes_annotation_widget_edit_existing_save_updates_shapes_element_and_session_snapshot(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer()
    app_state = get_or_create_app_state(viewer)
    app_state.set_sdata(sdata_blobs)
    widget = _create_embedded_shapes_annotation(qtbot, viewer)
    shapes_name = "blobs_polygons"
    original_index = sdata_blobs.shapes[shapes_name].index.to_list()
    emitted_events: list[object] = []
    widget.app_state.shapes_element_written.connect(emitted_events.append)

    widget._test_parent.shapes_combo.setCurrentIndex(
        _combo_index_for_text(widget._test_parent.shapes_combo, shapes_name)
    )
    layer = widget._annotation_layer
    assert isinstance(layer, Shapes)
    assert widget.save_shapes_button.isEnabled() is False

    _add_polygon(layer, offset=100)
    assert widget.save_shapes_button.isEnabled() is True
    layer.features.loc[len(layer.features) - 1, "index"] = None
    widget.save_shapes_button.click()

    saved_geodataframe = sdata_blobs.shapes[shapes_name]
    assert saved_geodataframe.index.to_list() == [*original_index, max(original_index) + 1]
    assert widget._annotation_session is not None
    assert widget._annotation_session.mode == "edit_existing"
    assert widget._annotation_session.source_geodataframe is not saved_geodataframe
    assert widget._annotation_session.source_geodataframe is not None
    assert widget._annotation_session.source_geodataframe.index.equals(saved_geodataframe.index)
    assert widget._annotation_session.source_shapes_index_feature_name == "index"
    assert emitted_events == [
        ShapesElementWrittenEvent(
            sdata=sdata_blobs,
            shapes_name=shapes_name,
            coordinate_system="global",
            source="shapes_annotation_widget",
        )
    ]
    assert "Shapes Saved" in _status_text(widget)


def test_shapes_annotation_widget_identity_default_guard_keeps_new_generated_rows_missing_until_save(
    qtbot,
) -> None:
    shapes_name = "generated_regions"
    sdata = _make_generated_annotation_ids_sdata(shapes_name=shapes_name)
    viewer = DummyViewer()
    app_state = get_or_create_app_state(viewer)
    app_state.set_sdata(sdata)
    widget = _create_embedded_shapes_annotation(qtbot, viewer)

    widget._test_parent.shapes_combo.setCurrentIndex(
        _combo_index_for_text(widget._test_parent.shapes_combo, shapes_name)
    )
    layer = widget._annotation_layer
    assert isinstance(layer, Shapes)
    assert widget._annotation_session is not None
    assert widget._annotation_session.source_shapes_index_feature_name == "instance_id"
    original_features = layer.features.copy(deep=True)
    _assert_identity_feature_default_missing(layer, "instance_id")

    layer.selected_data = {1}
    layer.mode = Mode.SELECT
    current_properties = dict(layer.current_properties)
    current_properties["instance_id"] = np.asarray(["__annotation_1"], dtype=object)
    layer.current_properties = current_properties

    pd.testing.assert_frame_equal(layer.features, original_features)
    _assert_identity_feature_default_missing(layer, "instance_id")

    _add_polygon(layer, offset=20)
    _add_polygon(layer, offset=30)

    assert layer.features["instance_id"].iloc[:2].tolist() == ["__annotation_0", "__annotation_1"]
    assert pd.isna(layer.features["instance_id"].iloc[2])
    assert pd.isna(layer.features["instance_id"].iloc[3])

    widget.save_shapes_button.click()

    expected_index = ["__annotation_0", "__annotation_1", "__annotation_2", "__annotation_3"]
    assert sdata.shapes[shapes_name].index.tolist() == expected_index
    assert layer.features["instance_id"].tolist() == expected_index
    _assert_identity_feature_default_missing(layer, "instance_id")


def test_shapes_annotation_widget_identity_default_guard_uses_fallback_source_feature_name(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    shapes_name = "blobs_polygons"
    viewer = DummyViewer()
    app_state = get_or_create_app_state(viewer)
    app_state.set_sdata(sdata_blobs)
    widget = _create_embedded_shapes_annotation(qtbot, viewer)

    widget._test_parent.shapes_combo.setCurrentIndex(
        _combo_index_for_text(widget._test_parent.shapes_combo, shapes_name)
    )
    layer = widget._annotation_layer
    assert isinstance(layer, Shapes)
    assert widget._annotation_session is not None
    assert widget._annotation_session.source_shapes_index_feature_name == "index"
    original_index_values = layer.features["index"].tolist()
    _assert_identity_feature_default_missing(layer, "index")

    current_properties = dict(layer.current_properties)
    current_properties["index"] = np.asarray([original_index_values[0]], dtype=object)
    layer.current_properties = current_properties

    assert layer.features["index"].tolist() == original_index_values
    _assert_identity_feature_default_missing(layer, "index")

    _add_polygon(layer, offset=100)

    assert layer.features["index"].iloc[:-1].tolist() == original_index_values
    assert pd.isna(layer.features["index"].iloc[-1])


def test_shapes_annotation_widget_identity_default_guard_disconnects_from_cleared_layer(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer()
    app_state = get_or_create_app_state(viewer)
    app_state.set_sdata(sdata_blobs)
    widget = _create_embedded_shapes_annotation(qtbot, viewer)

    widget._test_parent.shapes_combo.setCurrentIndex(
        _combo_index_for_text(widget._test_parent.shapes_combo, "blobs_polygons")
    )
    layer = widget._annotation_layer
    assert isinstance(layer, Shapes)
    _assert_identity_feature_default_missing(layer, "index")

    widget._clear_annotation_state()

    assert widget._annotation_identity_feature_default_guard.layer is None
    assert widget._annotation_identity_feature_default_guard.feature_name is None

    current_properties = dict(layer.current_properties)
    current_properties["index"] = np.asarray(["copied_after_disconnect"], dtype=object)
    layer.current_properties = current_properties

    assert _first_current_property_value(layer, "index") == "copied_after_disconnect"
    assert layer.feature_defaults["index"].iloc[0] == "copied_after_disconnect"


def test_shapes_annotation_widget_edit_existing_preserves_polygon_holes_on_save(qtbot) -> None:
    shapes_name = "hole_regions"
    polygon_1, polygon_2 = _polygon_hole_roundtrip_fixture()
    sdata = _make_polygon_hole_roundtrip_sdata(shapes_name=shapes_name)
    viewer = DummyViewer()
    app_state = get_or_create_app_state(viewer)
    app_state.set_sdata(sdata)
    widget = _create_embedded_shapes_annotation(qtbot, viewer)

    widget._test_parent.shapes_combo.setCurrentIndex(
        _combo_index_for_text(widget._test_parent.shapes_combo, shapes_name)
    )
    layer = widget._annotation_layer
    assert isinstance(layer, Shapes)
    assert widget.save_shapes_button.isEnabled() is False
    edited_polygon_2_vertices = np.asarray(layer.data[1], dtype=float).copy()
    edited_polygon_2_vertices[1] += np.asarray([0.0, 1.0])
    layer.data = [np.asarray(layer.data[0], dtype=float), edited_polygon_2_vertices]
    edited_polygon_2 = napari_polygon_vertices_to_shapely_polygon(edited_polygon_2_vertices)
    assert widget.save_shapes_button.isEnabled() is True

    widget.save_shapes_button.click()

    saved = sdata.shapes[shapes_name]
    assert saved.index.name == "region_id"
    assert saved.index.tolist() == ["hole_row", "simple_row"]
    assert saved["label"].tolist() == ["polygon_with_hole", "simple_polygon"]
    np.testing.assert_allclose(saved["score"].to_numpy(), np.asarray([1.25, 2.5]))
    _assert_polygon_hole_geometries_preserved(saved, polygon_1, edited_polygon_2)
    assert widget._annotation_session is not None
    assert widget._annotation_session.source_geodataframe is not saved
    assert widget._annotation_session.source_geodataframe is not None
    assert widget._annotation_session.source_geodataframe.index.equals(saved.index)
    assert "Shapes Saved" in _status_text(widget)


def test_shapes_annotation_widget_edit_existing_preserves_non_anchor_hole_vertex_edits(qtbot) -> None:
    shapes_name = "hole_regions"
    original_polygon_1, polygon_2 = _polygon_hole_roundtrip_fixture()
    sdata = _make_polygon_hole_roundtrip_sdata(shapes_name=shapes_name)
    viewer = DummyViewer()
    app_state = get_or_create_app_state(viewer)
    app_state.set_sdata(sdata)
    widget = _create_embedded_shapes_annotation(qtbot, viewer)

    widget._test_parent.shapes_combo.setCurrentIndex(
        _combo_index_for_text(widget._test_parent.shapes_combo, shapes_name)
    )
    layer = widget._annotation_layer
    assert isinstance(layer, Shapes)

    edited_vertices = np.asarray(layer.data[0], dtype=float).copy()
    np.testing.assert_allclose(edited_vertices[[0, 5, 12]], edited_vertices[[0, 0, 0]])
    np.testing.assert_allclose(edited_vertices[[6, 11]], edited_vertices[[6, 6]])
    edited_vertices[2] += np.asarray([35.0, -45.0])
    edited_vertices[8] += np.asarray([-25.0, 30.0])
    layer.data = [edited_vertices, np.asarray(layer.data[1], dtype=float)]
    expected_polygon_1 = Polygon(
        _yx_to_xy(edited_vertices[:6]),
        holes=[_yx_to_xy(edited_vertices[6:12])],
    )
    assert expected_polygon_1.is_valid
    assert not expected_polygon_1.equals(original_polygon_1)

    widget.save_shapes_button.click()

    saved = sdata.shapes[shapes_name]
    assert saved.index.name == "region_id"
    assert saved.index.tolist() == ["hole_row", "simple_row"]
    assert saved["label"].tolist() == ["polygon_with_hole", "simple_polygon"]
    np.testing.assert_allclose(saved["score"].to_numpy(), np.asarray([1.25, 2.5]))
    _assert_polygon_hole_geometries_preserved(saved, expected_polygon_1, polygon_2)

    reloaded_viewer = DummyViewer()
    reloaded_layer = (
        get_or_create_app_state(reloaded_viewer)
        .viewer_adapter.ensure_shapes_loaded(
            sdata,
            shapes_name,
            "global",
        )
        .layer
    )
    reloaded_polygon_1 = napari_polygon_vertices_to_shapely_polygon(reloaded_layer.data[0])

    assert reloaded_polygon_1.equals(saved.geometry.iloc[0])
    assert len(reloaded_polygon_1.interiors) == 1


def test_shapes_annotation_widget_create_holes_invalid_selection_warns_without_mutation(qtbot) -> None:
    shapes_name = "create_holes_regions"
    sdata, _shell, _child, _unselected = _make_create_holes_sdata(shapes_name=shapes_name)
    viewer = DummyViewer()
    app_state = get_or_create_app_state(viewer)
    app_state.set_sdata(sdata)
    widget = _create_embedded_shapes_annotation(qtbot, viewer)

    widget._test_parent.shapes_combo.setCurrentIndex(
        _combo_index_for_text(widget._test_parent.shapes_combo, shapes_name)
    )
    layer = widget._annotation_layer
    assert isinstance(layer, Shapes)
    assert widget.create_holes_button.isEnabled() is True
    original_data = [np.asarray(vertices, dtype=float).copy() for vertices in layer.data]
    original_features = layer.features.copy()
    layer.selected_data = {0}

    widget.create_holes_button.click()

    assert "Could Not Create Holes" in _status_text(widget)
    assert "Select one shell polygon" in _status_text(widget)
    _assert_layer_data_unchanged(layer, original_data)
    pd.testing.assert_frame_equal(layer.features, original_features)
    assert widget.save_shapes_button.isEnabled() is False
    assert widget.create_holes_button.isEnabled() is True


def test_shapes_annotation_widget_create_holes_render_failure_restores_layer_and_reports_error(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    restore_triangulation_backend: None,
) -> None:
    configure_shapes_triangulation_backend("bermuda")
    shapes_name = "create_holes_regions"
    sdata, _shell, _child, _unselected = _make_create_holes_sdata(shapes_name=shapes_name)
    viewer = DummyViewer()
    app_state = get_or_create_app_state(viewer)
    app_state.set_sdata(sdata)
    widget = _create_embedded_shapes_annotation(qtbot, viewer)

    widget._test_parent.shapes_combo.setCurrentIndex(
        _combo_index_for_text(widget._test_parent.shapes_combo, shapes_name)
    )
    layer = widget._annotation_layer
    assert isinstance(layer, Shapes)
    layer.mode = Mode.DIRECT
    layer.selected_data = {0, 1}
    baseline = shapes_annotation_layer_state_module._capture_shapes_layer_baseline(layer)
    plan = shapes_annotation_widget_module._create_holes_plan_from_selection(layer)
    expected_candidate = np.asarray(plan.vertices, dtype=np.float32)
    real_triangulate = bermuda.triangulate_polygons_with_edge
    failure_count = 0

    def fail_exact_candidate(polygons: list[np.ndarray]) -> object:
        nonlocal failure_count
        if np.array_equal(polygons[0], expected_candidate):
            failure_count += 1
            raise RuntimeError("synthetic Bermuda failure for Create-holes widget test")
        return real_triangulate(polygons)

    monkeypatch.setattr(bermuda, "triangulate_polygons_with_edge", fail_exact_candidate)

    widget.create_holes_button.click()

    assert failure_count == 1
    _assert_shapes_layer_matches_baseline(layer, baseline)
    assert widget._annotation_layer_has_unsaved_changes() is False
    assert widget._test_parent.annotation_context.has_unsaved_shapes_changes is False
    assert widget.save_shapes_button.isEnabled() is False
    assert widget.create_holes_button.isEnabled() is True
    status = _status_text(widget)
    assert "Could Not Create Holes" in status
    assert "The original annotations were restored." in status
    assert "Created Holes" not in status


def test_shapes_annotation_widget_create_holes_mutates_layer_and_marks_dirty(qtbot) -> None:
    shapes_name = "create_holes_regions"
    sdata, shell, child, unselected = _make_create_holes_sdata(shapes_name=shapes_name)
    viewer = DummyViewer()
    app_state = get_or_create_app_state(viewer)
    app_state.set_sdata(sdata)
    widget = _create_embedded_shapes_annotation(qtbot, viewer)

    widget._test_parent.shapes_combo.setCurrentIndex(
        _combo_index_for_text(widget._test_parent.shapes_combo, shapes_name)
    )
    layer = widget._annotation_layer
    assert isinstance(layer, Shapes)
    assert widget._annotation_layer_has_unsaved_changes() is False
    layer.selected_data = {0, 1}

    widget.create_holes_button.click()

    assert len(layer.data) == 2
    assert list(layer.shape_type) == ["polygon", "polygon"]
    assert layer.features["region_id"].tolist() == ["shell_row", "unselected_row"]
    assert set(layer.selected_data) == {0}
    expected_shell = Polygon(shell.exterior.coords, holes=[child.exterior.coords])
    assert napari_polygon_vertices_to_shapely_polygon(layer.data[0]).equals(expected_shell)
    assert napari_polygon_vertices_to_shapely_polygon(layer.data[1]).equals(unselected)
    assert widget._annotation_layer_has_unsaved_changes() is True
    assert widget.save_shapes_button.isEnabled() is True
    assert widget.create_holes_button.isEnabled() is True
    status = _status_text(widget)
    assert "Created Holes" in status
    assert "Converted 1 selected polygon(s) into hole(s) and removed their shape row(s)." in status


def test_shapes_annotation_widget_create_holes_saves_and_reloads(qtbot) -> None:
    """Integration test for the create-holes, save, and reload path."""
    shapes_name = "create_holes_regions"
    sdata, shell, child, unselected = _make_create_holes_sdata(shapes_name=shapes_name)
    viewer = DummyViewer()
    app_state = get_or_create_app_state(viewer)
    app_state.set_sdata(sdata)
    widget = _create_embedded_shapes_annotation(qtbot, viewer)

    widget._test_parent.shapes_combo.setCurrentIndex(
        _combo_index_for_text(widget._test_parent.shapes_combo, shapes_name)
    )
    layer = widget._annotation_layer
    assert isinstance(layer, Shapes)
    layer.selected_data = {0, 1}

    widget.create_holes_button.click()
    widget.save_shapes_button.click()

    saved = sdata.shapes[shapes_name]
    assert saved.index.name == "region_id"
    assert saved.index.tolist() == ["shell_row", "unselected_row"]
    assert saved["label"].tolist() == ["shell", "unselected"]
    np.testing.assert_allclose(saved["score"].to_numpy(), np.asarray([1.0, 3.0]))
    expected_polygon = Polygon(shell.exterior.coords, holes=[child.exterior.coords])
    assert saved.geometry.iloc[0].equals(expected_polygon)
    assert len(saved.geometry.iloc[0].interiors) == 1
    assert saved.geometry.iloc[1].equals(unselected)
    assert "Shapes Saved" in _status_text(widget)

    reloaded_viewer = DummyViewer()
    reloaded_layer = (
        get_or_create_app_state(reloaded_viewer)
        .viewer_adapter.ensure_shapes_loaded(
            sdata,
            shapes_name,
            "global",
        )
        .layer
    )
    reloaded_polygon = napari_polygon_vertices_to_shapely_polygon(reloaded_layer.data[0])

    assert reloaded_polygon.equals(saved.geometry.iloc[0])
    assert len(reloaded_polygon.interiors) == 1
    assert len(reloaded_layer.data) == 2


def test_shapes_annotation_widget_create_holes_table_linked_warning_is_explicit(qtbot) -> None:
    shapes_name = "create_holes_regions"
    table_name = "create_holes_table"
    sdata, _shell, _child, _unselected = _make_create_holes_sdata(shapes_name=shapes_name)
    table = _add_dummy_table_annotating_shapes(sdata, shapes_name=shapes_name, table_name=table_name)
    original_obs = table.obs.copy(deep=True)
    viewer = DummyViewer()
    app_state = get_or_create_app_state(viewer)
    app_state.set_sdata(sdata)
    widget = _create_embedded_shapes_annotation(qtbot, viewer)

    widget._test_parent.shapes_combo.setCurrentIndex(
        _combo_index_for_text(widget._test_parent.shapes_combo, shapes_name)
    )
    layer = widget._annotation_layer
    assert isinstance(layer, Shapes)
    assert widget._annotation_session is not None
    assert widget._annotation_session.table_linked is True
    layer.selected_data = {0, 1}

    widget.create_holes_button.click()

    assert sdata.tables[table_name] is table
    pd.testing.assert_frame_equal(sdata.tables[table_name].obs, original_obs)
    status = _status_text(widget)
    assert "Created Holes" in status
    assert "Linked tables are not updated automatically" in status
    assert "table annotations may no longer match the shapes rows" in status


# Keep end-to-end anchor-edit coverage focused: one shell-anchor edit on an
# edit-existing layer and one hole-anchor edit on an adopted native layer.
def test_shapes_annotation_widget_edit_existing_shell_anchor_edit_saves_and_reloads_with_hole(qtbot) -> None:
    """Exercise guarded shell-anchor drag through edit-existing save and reload."""
    shapes_name = "hole_regions"
    original_polygon_1, polygon_2 = _polygon_hole_roundtrip_fixture()
    sdata = _make_polygon_hole_roundtrip_sdata(shapes_name=shapes_name)
    viewer = DummyViewer()
    app_state = get_or_create_app_state(viewer)
    app_state.set_sdata(sdata)
    widget = _create_embedded_shapes_annotation(qtbot, viewer)

    widget._test_parent.shapes_combo.setCurrentIndex(
        _combo_index_for_text(widget._test_parent.shapes_combo, shapes_name)
    )
    layer = widget._annotation_layer
    assert isinstance(layer, Shapes)
    original_vertices = np.asarray(layer.data[0], dtype=float)
    moved_coordinate = original_vertices[0] + np.asarray([-25.0, 35.0])
    _install_direct_drag_callback_for_annotation_guard(
        widget,
        layer,
        moved_vertex_index=0,
    )

    edited_vertices = _drag_annotation_vertex(layer, vertex_index=0, moved_coordinate=moved_coordinate)

    np.testing.assert_allclose(edited_vertices[[0, 5, 12]], np.repeat(moved_coordinate[None, :], 3, axis=0))
    expected_polygon_1 = napari_polygon_vertices_to_shapely_polygon(edited_vertices)
    assert expected_polygon_1.is_valid
    assert len(expected_polygon_1.interiors) == 1
    assert not expected_polygon_1.equals(original_polygon_1)

    widget.save_shapes_button.click()

    saved = sdata.shapes[shapes_name]
    assert saved.index.name == "region_id"
    assert saved.index.tolist() == ["hole_row", "simple_row"]
    assert saved["label"].tolist() == ["polygon_with_hole", "simple_polygon"]
    np.testing.assert_allclose(saved["score"].to_numpy(), np.asarray([1.25, 2.5]))
    _assert_polygon_hole_geometries_preserved(saved, expected_polygon_1, polygon_2)

    reloaded_viewer = DummyViewer()
    reloaded_layer = (
        get_or_create_app_state(reloaded_viewer)
        .viewer_adapter.ensure_shapes_loaded(
            sdata,
            shapes_name,
            "global",
        )
        .layer
    )
    reloaded_polygon_1 = napari_polygon_vertices_to_shapely_polygon(reloaded_layer.data[0])

    assert reloaded_polygon_1.equals(saved.geometry.iloc[0])
    assert len(reloaded_polygon_1.interiors) == 1
    assert "Shapes Saved" in _status_text(widget)


def test_shapes_annotation_widget_adopted_native_hole_anchor_edit_saves_and_reloads_with_hole(qtbot) -> None:
    """Exercise guarded hole-anchor drag through native adoption, save, and reload."""
    shapes_name = "native_hole_anchor_roundtrip"
    original_polygon_1, polygon_2 = _polygon_hole_roundtrip_fixture()
    sdata = _make_polygon_hole_roundtrip_sdata(shapes_name="reference_regions")
    native_layer = Shapes(
        [
            shapely_polygon_to_napari_polygon_vertices(original_polygon_1),
            shapely_polygon_to_napari_polygon_vertices(polygon_2),
        ],
        shape_type=["polygon", "polygon"],
        name=shapes_name,
    )
    viewer = DummyViewer()
    app_state = get_or_create_app_state(viewer)
    app_state.set_sdata(sdata)
    widget = _create_embedded_shapes_annotation(qtbot, viewer)

    viewer.add_layer(native_layer)
    qtbot.waitUntil(
        lambda: (
            widget._annotation_session is not None
            and widget._annotation_session.shapes_name == shapes_name
            and widget._annotation_layer is not None
            and widget._annotation_layer is not native_layer
        )
    )
    adopted_layer = widget._annotation_layer
    assert isinstance(adopted_layer, Shapes)
    assert native_layer not in viewer.layers
    original_vertices = np.asarray(adopted_layer.data[0], dtype=float)
    moved_coordinate = original_vertices[6] + np.asarray([15.0, -20.0])
    _install_direct_drag_callback_for_annotation_guard(
        widget,
        adopted_layer,
        moved_vertex_index=6,
    )

    edited_vertices = _drag_annotation_vertex(adopted_layer, vertex_index=6, moved_coordinate=moved_coordinate)

    np.testing.assert_allclose(edited_vertices[[6, 11]], np.repeat(moved_coordinate[None, :], 2, axis=0))
    expected_polygon_1 = napari_polygon_vertices_to_shapely_polygon(edited_vertices)
    assert expected_polygon_1.is_valid
    assert len(expected_polygon_1.interiors) == 1
    assert not expected_polygon_1.equals(original_polygon_1)

    widget.save_shapes_button.click()

    assert shapes_name in sdata.shapes
    saved = sdata.shapes[shapes_name]
    assert saved.index.name == "instance_id"
    assert saved.index.tolist() == ["__annotation_0", "__annotation_1"]
    assert adopted_layer.features["instance_id"].tolist() == ["__annotation_0", "__annotation_1"]
    _assert_polygon_hole_geometries_preserved(saved, expected_polygon_1, polygon_2)
    assert widget._annotation_session is not None
    assert widget._annotation_session.mode == "edit_existing"
    assert widget._annotation_session.shapes_name == shapes_name
    assert widget._test_parent.shapes_combo.currentText() == shapes_name

    reloaded_viewer = DummyViewer()
    reloaded_layer = (
        get_or_create_app_state(reloaded_viewer)
        .viewer_adapter.ensure_shapes_loaded(
            sdata,
            shapes_name,
            "global",
        )
        .layer
    )
    reloaded_polygon_1 = napari_polygon_vertices_to_shapely_polygon(reloaded_layer.data[0])

    assert reloaded_polygon_1.equals(saved.geometry.iloc[0])
    assert len(reloaded_polygon_1.interiors) == 1
    assert "Shapes Saved" in _status_text(widget)


def test_shapes_annotation_widget_table_linked_edit_warns_without_mutating_table(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    shapes_name = "blobs_polygons"
    table_name = "shapes_annotation_table"
    table = _add_dummy_table_annotating_shapes(sdata_blobs, shapes_name=shapes_name, table_name=table_name)
    original_obs = table.obs.copy(deep=True)
    viewer = DummyViewer()
    app_state = get_or_create_app_state(viewer)
    app_state.set_sdata(sdata_blobs)
    widget = _create_embedded_shapes_annotation(qtbot, viewer)

    widget._test_parent.shapes_combo.setCurrentIndex(
        _combo_index_for_text(widget._test_parent.shapes_combo, shapes_name)
    )

    assert widget._annotation_session is not None
    assert widget._annotation_session.table_linked is True
    assert (
        "Linked tables are not updated by Annotation and may go out of sync if rows are added or removed."
        in _status_text(widget)
    )

    layer = widget._annotation_layer
    assert isinstance(layer, Shapes)
    _add_polygon(layer, offset=100)
    layer.features.loc[len(layer.features) - 1, "index"] = None
    widget.save_shapes_button.click()

    assert sdata_blobs.tables[table_name] is table
    pd.testing.assert_frame_equal(sdata_blobs.tables[table_name].obs, original_obs)
    assert (
        "Linked tables are not updated by Annotation and may go out of sync if rows are added or removed."
        in _status_text(widget)
    )


def test_shapes_annotation_widget_clears_annotation_state_when_sdata_is_cleared(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer()
    widget = _create_ready_annotation_widget(qtbot, viewer, sdata_blobs)
    widget.create_layer_button.click()
    layer = viewer.layers[0]

    widget.app_state.clear_sdata(discard_current=True)

    assert list(viewer.layers) == []
    assert widget.app_state.viewer_adapter.layer_bindings.get_binding(layer) is None
    assert widget._annotation_layer is None
    assert widget._annotation_shapes_name is None
    assert widget._annotation_coordinate_system is None
    assert widget.name_edit.isHidden() is True
    assert widget.name_edit.isEnabled() is False
    assert widget.create_layer_button.isEnabled() is False
    assert widget.save_shapes_button.isEnabled() is False
    assert "No SpatialData Loaded" in _status_text(widget)


def test_shapes_annotation_widget_manual_annotation_layer_deletion_clears_state(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer()
    widget = _create_ready_annotation_widget(qtbot, viewer, sdata_blobs)
    widget.create_layer_button.click()
    layer = viewer.layers[0]

    viewer.layers.remove(layer)

    assert list(viewer.layers) == []
    assert widget.app_state.viewer_adapter.layer_bindings.get_binding(layer) is None
    assert widget._annotation_layer is None
    assert widget._annotation_shapes_name is None
    assert widget._annotation_coordinate_system is None
    assert (
        widget._annotation_context.shapes_target == shapes_annotation_widget_module._ShapesAnnotationTarget.create_new()
    )
    assert widget._test_parent.shapes_combo.currentText() == "Create shapes..."
    assert widget.name_edit.isEnabled() is True
    assert widget.create_layer_button.isEnabled() is True
    assert widget.save_shapes_button.isEnabled() is False
    assert "Ready" in _status_text(widget)


def test_shapes_annotation_widget_manual_existing_layer_deletion_resets_selector_and_can_reopen(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer()
    app_state = get_or_create_app_state(viewer)
    app_state.set_sdata(sdata_blobs)
    widget = _create_embedded_shapes_annotation(qtbot, viewer)
    existing_shapes_name = "blobs_polygons"
    widget._test_parent.shapes_combo.setCurrentIndex(
        _combo_index_for_text(widget._test_parent.shapes_combo, existing_shapes_name)
    )
    removed_layer = widget._annotation_layer

    assert removed_layer is not None
    assert widget._annotation_session is not None
    assert widget._annotation_session.shapes_name == existing_shapes_name

    viewer.layers.remove(removed_layer)

    assert widget.app_state.viewer_adapter.layer_bindings.get_binding(removed_layer) is None
    assert widget._annotation_layer is None
    assert widget._annotation_session is None
    assert (
        widget._annotation_context.shapes_target == shapes_annotation_widget_module._ShapesAnnotationTarget.create_new()
    )
    assert widget._test_parent.shapes_combo.currentText() == "Create shapes..."
    assert _combo_index_for_text(widget._test_parent.shapes_combo, existing_shapes_name) >= 0
    assert widget.save_shapes_button.isEnabled() is False

    widget._test_parent.shapes_combo.setCurrentIndex(
        _combo_index_for_text(widget._test_parent.shapes_combo, existing_shapes_name)
    )

    assert widget._annotation_layer is not None
    assert widget._annotation_layer is not removed_layer
    assert widget._annotation_session is not None
    assert widget._annotation_session.shapes_name == existing_shapes_name
    assert widget.save_shapes_button.isEnabled() is False


def test_shapes_annotation_widget_removal_listener_defensively_unregisters_annotation_layer(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer()
    widget = _create_ready_annotation_widget(qtbot, viewer, sdata_blobs)
    widget.create_layer_button.click()
    layer = viewer.layers[0]

    widget._on_viewer_layer_removed(SimpleNamespace(value=layer))

    assert widget.app_state.viewer_adapter.layer_bindings.get_binding(layer) is None
    assert widget._annotation_layer is None
    assert widget._annotation_shapes_name is None
    assert widget._annotation_coordinate_system is None
    assert widget.name_edit.isEnabled() is True
    assert widget.create_layer_button.isEnabled() is True
    assert widget.save_shapes_button.isEnabled() is False


def test_shapes_annotation_widget_ignores_unrelated_layer_removal(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer()
    widget = _create_ready_annotation_widget(qtbot, viewer, sdata_blobs)
    widget.create_layer_button.click()
    annotation_layer = viewer.layers[0]
    unrelated_layer = Shapes(
        [np.asarray([(0, 0), (0, 1), (1, 1)], dtype=float)],
        shape_type="polygon",
        name="unrelated",
    )
    viewer.add_layer(unrelated_layer)

    viewer.layers.remove(unrelated_layer)

    assert list(viewer.layers) == [annotation_layer]
    assert widget.app_state.viewer_adapter.layer_bindings.get_binding(annotation_layer) is not None
    assert widget._annotation_layer is annotation_layer
    assert widget._annotation_shapes_name == "new_regions"
    assert widget._annotation_coordinate_system == "global"
    assert widget.name_edit.isEnabled() is False
    assert widget.create_layer_button.isEnabled() is False
    assert widget.save_shapes_button.isEnabled() is False


def test_shapes_annotation_widget_adopts_native_empty_shapes_layer(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer()
    app_state = get_or_create_app_state(viewer)
    app_state.set_sdata(sdata_blobs)
    widget = _create_embedded_shapes_annotation(qtbot, viewer)
    native_layer = Shapes(
        [],
        shape_type="polygon",
        name="native_shapes",
        affine=np.asarray([[1.0, 0.0, 5.0], [0.0, 1.0, 7.0], [0.0, 0.0, 1.0]]),
    )

    viewer.add_layer(native_layer)

    # Native adoption is deferred from the layer-inserted event with
    # `QTimer.singleShot(0, ...)`:
    # napari inserts layer -> layers.events.inserted emitted ->
    # `_on_viewer_layer_inserted(...)` -> `QTimer.singleShot(0, ...)` ->
    # later `_maybe_adopt_native_shapes_layer(...)`. The adopted layer should
    # also be the Spatiato replacement, not the original plain napari Shapes layer.
    qtbot.waitUntil(
        lambda: (
            widget._annotation_session is not None
            and widget._annotation_session.shapes_name == "native_shapes"
            and widget._annotation_layer is not None
            and widget._annotation_layer is not native_layer
        )
    )
    adopted_layer = widget._annotation_layer
    assert isinstance(adopted_layer, Shapes)
    assert native_layer not in viewer.layers
    assert adopted_layer in viewer.layers
    binding = widget.app_state.viewer_adapter.layer_bindings.get_binding(adopted_layer)
    assert isinstance(binding, ShapesLayerBinding)
    assert binding.element_name == "native_shapes"
    assert binding.coordinate_system == "global"
    assert binding.shapes_role == "primary"
    assert binding.shapes_rendering_mode == "shapes"
    assert binding.source_shapes_index_feature_name == "instance_id"
    assert widget._annotation_session is not None
    assert widget._annotation_session.mode == "create_new"
    assert widget._annotation_session.shapes_name == "native_shapes"
    assert widget._annotation_edit_guard.layer is adopted_layer
    assert "_drag_modes" in vars(adopted_layer)
    assert adopted_layer._drag_modes[Mode.DIRECT] is widget._annotation_edit_guard._wrapped_direct_callback
    assert (
        widget._annotation_context.shapes_target == shapes_annotation_widget_module._ShapesAnnotationTarget.create_new()
    )
    assert widget._test_parent.shapes_combo.currentText() == "Create shapes..."
    assert widget.name_edit.text() == "native_shapes"
    assert adopted_layer.name == "native_shapes"
    np.testing.assert_allclose(adopted_layer.affine.affine_matrix, np.eye(3))
    assert adopted_layer.current_edge_width == 1
    np.testing.assert_allclose(to_rgba(adopted_layer.current_edge_color), to_rgba("#00FFFF"))
    np.testing.assert_allclose(to_rgba(adopted_layer.current_face_color), to_rgba(PRIMARY_SHAPES_FACE_COLOR))
    assert adopted_layer.opacity == 0.8
    assert hasattr(adopted_layer, _SHAPES_EDGE_WIDTH_SYNC_CALLBACK_ATTR)
    assert hasattr(adopted_layer, _SHAPES_EDGE_COLOR_SYNC_CALLBACK_ATTR)
    assert widget.create_layer_button.isEnabled() is False
    assert widget.save_shapes_button.isEnabled() is False


def test_shapes_annotation_widget_saves_adopted_native_nonempty_shapes_layer(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer()
    app_state = get_or_create_app_state(viewer)
    app_state.set_sdata(sdata_blobs)
    widget = _create_embedded_shapes_annotation(qtbot, viewer)
    native_layer = _native_polygon_layer("native_import")

    viewer.add_layer(native_layer)
    qtbot.waitUntil(lambda: widget._annotation_layer is not None and widget._annotation_layer is not native_layer)
    adopted_layer = widget._annotation_layer
    assert isinstance(adopted_layer, Shapes)
    assert native_layer not in viewer.layers
    np.testing.assert_allclose(adopted_layer.edge_color[0], to_rgba("#00FFFF"))
    np.testing.assert_allclose(adopted_layer.face_color[0], to_rgba(PRIMARY_SHAPES_FACE_COLOR))
    np.testing.assert_allclose(to_rgba(adopted_layer.current_edge_color), to_rgba("#00FFFF"))
    np.testing.assert_allclose(to_rgba(adopted_layer.current_face_color), to_rgba(PRIMARY_SHAPES_FACE_COLOR))
    widget.save_shapes_button.click()

    assert "native_import" in sdata_blobs.shapes
    assert sdata_blobs.shapes["native_import"].index.tolist() == ["__annotation_0"]
    assert adopted_layer.features["instance_id"].tolist() == ["__annotation_0"]
    assert "instance_id: __annotation_0" in adopted_layer.get_status(position=(1, 1))["value"]
    assert widget._annotation_session is not None
    assert widget._annotation_session.mode == "edit_existing"
    assert widget._annotation_session.shapes_name == "native_import"
    assert widget._test_parent.shapes_combo.currentText() == "native_import"
    assert widget.name_edit.text() == ""


def test_shapes_annotation_widget_saves_native_csv_layer_with_polygon_hole(
    qtbot,
    tmp_path,
) -> None:
    """Save a napari-native CSV Shapes import with a hole through the widget."""
    polygon_1, polygon_2 = _polygon_hole_roundtrip_fixture()
    sdata = _make_polygon_hole_roundtrip_sdata(shapes_name="reference_regions")
    output_path = tmp_path / "native_hole_import.csv"
    napari_write_shapes(
        str(output_path),
        [
            shapely_polygon_to_napari_polygon_vertices(polygon_1),
            shapely_polygon_to_napari_polygon_vertices(polygon_2),
        ],
        {"shape_type": ["polygon", "polygon"]},
    )
    loaded = csv_to_layer_data(str(output_path), require_type="shapes")
    assert loaded is not None
    data, meta, _layer_type = loaded
    native_layer = Shapes(data, **meta, name="native_hole_import")
    viewer = DummyViewer()
    app_state = get_or_create_app_state(viewer)
    app_state.set_sdata(sdata)
    widget = _create_embedded_shapes_annotation(qtbot, viewer)

    viewer.add_layer(native_layer)
    qtbot.waitUntil(lambda: widget._annotation_layer is not None and widget._annotation_layer is not native_layer)
    adopted_layer = widget._annotation_layer
    assert isinstance(adopted_layer, Shapes)
    assert native_layer not in viewer.layers
    widget.save_shapes_button.click()

    assert "native_hole_import" in sdata.shapes
    saved = sdata.shapes["native_hole_import"]
    assert saved.index.name == "instance_id"
    assert saved.index.tolist() == ["__annotation_0", "__annotation_1"]
    assert adopted_layer.features["instance_id"].tolist() == ["__annotation_0", "__annotation_1"]
    _assert_polygon_hole_geometries_preserved(saved, polygon_1, polygon_2)
    assert widget._annotation_session is not None
    assert widget._annotation_session.mode == "edit_existing"
    assert widget._annotation_session.shapes_name == "native_hole_import"
    assert widget._test_parent.shapes_combo.currentText() == "native_hole_import"
    assert "Shapes Saved" in _status_text(widget)


def test_shapes_annotation_widget_saves_reloads_adopted_translated_native_shapes_layer(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer()
    app_state = get_or_create_app_state(viewer)
    app_state.set_sdata(sdata_blobs)
    widget = _create_embedded_shapes_annotation(qtbot, viewer)
    native_layer = _native_polygon_layer(
        "native_translated",
        affine=np.asarray([[1.0, 0.0, 5.0], [0.0, 1.0, 7.0], [0.0, 0.0, 1.0]]),
    )
    expected_layer_vertices = np.asarray(
        [
            [5.0, 7.0],
            [5.0, 9.0],
            [7.0, 9.0],
            [7.0, 7.0],
        ]
    )
    expected_geometry_coords = np.asarray(
        [
            [7.0, 5.0],
            [9.0, 5.0],
            [9.0, 7.0],
            [7.0, 7.0],
            [7.0, 5.0],
        ]
    )
    # Reloading goes through the saved Shapely polygon exterior, which is a
    # closed ring and therefore repeats the first vertex at the end.
    expected_reloaded_vertices = np.asarray(
        [
            [5.0, 7.0],
            [5.0, 9.0],
            [7.0, 9.0],
            [7.0, 7.0],
            [5.0, 7.0],
        ]
    )

    viewer.add_layer(native_layer)

    qtbot.waitUntil(lambda: widget._annotation_layer is not None and widget._annotation_layer is not native_layer)
    adopted_layer = widget._annotation_layer
    assert isinstance(adopted_layer, Shapes)
    assert native_layer not in viewer.layers
    np.testing.assert_allclose(adopted_layer.affine.affine_matrix, np.eye(3))
    np.testing.assert_allclose(adopted_layer.data[0], expected_layer_vertices)

    widget.save_shapes_button.click()

    assert "native_translated" in sdata_blobs.shapes
    np.testing.assert_allclose(
        np.asarray(sdata_blobs.shapes["native_translated"].geometry.iloc[0].exterior.coords),
        expected_geometry_coords,
    )

    viewer.layers.remove(adopted_layer)

    reloaded = widget.app_state.viewer_adapter.ensure_shapes_loaded(sdata_blobs, "native_translated", "global").layer

    assert isinstance(reloaded, Shapes)
    assert reloaded is not adopted_layer
    np.testing.assert_allclose(reloaded.data[0], expected_reloaded_vertices)


def test_shapes_annotation_widget_native_name_falls_back_and_suffixes_collision(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    sdata_blobs.shapes["new_shapes"] = sdata_blobs.shapes["blobs_polygons"].copy()
    sdata_blobs.shapes["New_Shapes_1"] = sdata_blobs.shapes["blobs_polygons"].copy()
    viewer = DummyViewer()
    app_state = get_or_create_app_state(viewer)
    app_state.set_sdata(sdata_blobs)
    widget = _create_embedded_shapes_annotation(qtbot, viewer)
    native_layer = Shapes([], shape_type="polygon", name="bad/name")

    viewer.add_layer(native_layer)

    qtbot.waitUntil(lambda: widget._annotation_layer is not None and widget._annotation_layer is not native_layer)
    adopted_layer = widget._annotation_layer
    assert isinstance(adopted_layer, Shapes)
    assert native_layer not in viewer.layers
    binding = widget.app_state.viewer_adapter.layer_bindings.get_binding(adopted_layer)
    assert isinstance(binding, ShapesLayerBinding)
    assert binding.element_name == "new_shapes_2"
    assert widget.name_edit.text() == "new_shapes_2"
    assert adopted_layer.name == "new_shapes_2"


def test_shapes_annotation_widget_deferred_native_adoption_ignores_spatiato_loaded_shapes(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer()
    app_state = get_or_create_app_state(viewer)
    app_state.set_sdata(sdata_blobs)
    widget = _create_embedded_shapes_annotation(qtbot, viewer)

    result = widget.app_state.viewer_adapter.ensure_shapes_loaded(sdata_blobs, "blobs_polygons", "global")
    qtbot.wait(10)

    assert result.created is True
    assert widget.app_state.viewer_adapter.layer_bindings.get_binding(result.layer) is not None
    assert widget._annotation_layer is None
    assert widget._annotation_session is None
    assert (
        widget._annotation_context.shapes_target == shapes_annotation_widget_module._ShapesAnnotationTarget.create_new()
    )


def test_shapes_annotation_widget_native_adoption_cancel_removes_pending_import_and_keeps_dirty_session(
    qtbot,
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    viewer = AutoActivatingDummyViewer()
    widget = _create_ready_annotation_widget(qtbot, viewer, sdata_blobs)
    widget.create_layer_button.click()
    annotation_layer = viewer.layers[0]
    _add_polygon(annotation_layer)
    native_layer = _native_polygon_layer("native_shapes")
    confirm_calls: list[str] = []

    def cancel_discard(*, reason: str) -> bool:
        confirm_calls.append(reason)
        return False

    monkeypatch.setattr(widget, "_confirm_discard_annotation_layer", cancel_discard)

    viewer.add_layer(native_layer)

    qtbot.waitUntil(lambda: native_layer not in viewer.layers and viewer.layers.selection.active is annotation_layer)
    assert confirm_calls == ["shapes_target"]
    assert widget._annotation_layer is annotation_layer
    assert widget._annotation_session is not None
    assert widget._annotation_session.shapes_name == "new_regions"
    assert widget.app_state.viewer_adapter.layer_bindings.get_binding(native_layer) is None


def test_shapes_annotation_widget_native_adoption_confirm_discards_dirty_session(
    qtbot,
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    viewer = AutoActivatingDummyViewer()
    widget = _create_ready_annotation_widget(qtbot, viewer, sdata_blobs)
    widget.create_layer_button.click()
    annotation_layer = viewer.layers[0]
    _add_polygon(annotation_layer)
    native_layer = _native_polygon_layer("native_shapes")
    confirm_calls: list[str] = []

    def confirm_discard(*, reason: str) -> bool:
        confirm_calls.append(reason)
        return True

    monkeypatch.setattr(widget, "_confirm_discard_annotation_layer", confirm_discard)

    viewer.add_layer(native_layer)

    qtbot.waitUntil(
        lambda: (
            widget._annotation_session is not None
            and widget._annotation_session.shapes_name == "native_shapes"
            and widget._annotation_layer is not None
            and widget._annotation_layer is not native_layer
        )
    )
    adopted_layer = widget._annotation_layer
    assert isinstance(adopted_layer, Shapes)
    assert confirm_calls == ["shapes_target"]
    assert annotation_layer not in viewer.layers
    assert native_layer not in viewer.layers
    assert adopted_layer in viewer.layers
    assert widget.app_state.viewer_adapter.layer_bindings.get_binding(annotation_layer) is None
    assert widget.app_state.viewer_adapter.layer_bindings.get_binding(native_layer) is None
    assert widget.app_state.viewer_adapter.layer_bindings.get_binding(adopted_layer) is not None
    assert widget._annotation_session is not None
    assert widget._annotation_session.mode == "create_new"
    assert widget._annotation_session.shapes_name == "native_shapes"
    assert widget.name_edit.text() == "native_shapes"
    np.testing.assert_allclose(adopted_layer.edge_color[0], to_rgba("#00FFFF"))
    np.testing.assert_allclose(adopted_layer.face_color[0], to_rgba(PRIMARY_SHAPES_FACE_COLOR))


def test_shapes_annotation_widget_coordinate_discard_guard_avoids_duplicate_cleanup(
    qtbot,
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    _patch_coordinate_system_names(monkeypatch, ["global", "local"])
    viewer = DummyViewer()
    widget = _create_ready_annotation_widget(qtbot, viewer, sdata_blobs)
    widget.create_layer_button.click()
    _add_polygon(viewer.layers[0])
    monkeypatch.setattr(widget, "_confirm_discard_annotation_layer", lambda *, reason: True)
    layer_transition_guard_values: list[bool] = []
    clear_call_count = 0
    original_remove_annotation_layer = widget._remove_annotation_layer
    original_clear_annotation_state = widget._clear_annotation_state

    def remove_annotation_layer() -> None:
        layer_transition_guard_values.append(widget._is_handling_widget_owned_layer_transition)
        original_remove_annotation_layer()

    def clear_annotation_state() -> None:
        nonlocal clear_call_count
        clear_call_count += 1
        original_clear_annotation_state()

    monkeypatch.setattr(widget, "_remove_annotation_layer", remove_annotation_layer)
    monkeypatch.setattr(widget, "_clear_annotation_state", clear_annotation_state)

    widget._test_parent.coordinate_system_combo.setCurrentIndex(1)

    assert layer_transition_guard_values == [True]
    assert clear_call_count == 1
    assert widget._is_handling_widget_owned_layer_transition is False
    assert widget.app_state.coordinate_system == "local"
    assert list(viewer.layers) == []


def test_shapes_annotation_widget_discard_saved_annotation_layer_reloads_clean_primary_layer(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer()
    widget = _create_ready_annotation_widget(qtbot, viewer, sdata_blobs)
    widget.create_layer_button.click()
    dirty_layer = viewer.layers[0]
    _add_polygon(dirty_layer)
    widget.save_shapes_button.click()
    assert widget._annotation_session is not None

    _add_polygon(dirty_layer, offset=10)
    assert len(dirty_layer.data) == 2

    widget._discard_annotation_layer()

    assert dirty_layer not in viewer.layers
    assert widget.app_state.viewer_adapter.layer_bindings.get_binding(dirty_layer) is None
    assert len(viewer.layers) == 1
    clean_layer = viewer.layers[0]
    assert clean_layer is not dirty_layer
    assert isinstance(clean_layer, Shapes)
    assert clean_layer.name == "new_regions"
    assert len(clean_layer.data) == 1
    binding = widget.app_state.viewer_adapter.layer_bindings.get_binding(clean_layer)
    assert isinstance(binding, ShapesLayerBinding)
    assert binding.element_name == "new_regions"
    assert binding.coordinate_system == "global"
    assert binding.shapes_role == "primary"
    assert widget._annotation_layer is None
    assert widget._annotation_shapes_name is None
    assert widget._annotation_coordinate_system is None
    assert widget._annotation_session is None
    assert widget.name_edit.isHidden() is True
    assert widget.name_edit.isEnabled() is False


def test_shapes_annotation_widget_backed_edit_existing_discard_reloads_clean_primary_layer(
    qtbot,
    backed_sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer()
    app_state = get_or_create_app_state(viewer)
    app_state.set_sdata(backed_sdata_blobs)
    widget = _create_embedded_shapes_annotation(qtbot, viewer)
    shapes_name = "blobs_polygons"

    widget._test_parent.shapes_combo.setCurrentIndex(
        _combo_index_for_text(widget._test_parent.shapes_combo, shapes_name)
    )
    dirty_layer = widget._annotation_layer
    assert isinstance(dirty_layer, Shapes)
    initial_row_count = len(dirty_layer.data)

    _add_polygon(dirty_layer, offset=100)
    assert len(dirty_layer.data) == initial_row_count + 1

    widget._discard_annotation_layer()

    assert dirty_layer not in viewer.layers
    assert app_state.viewer_adapter.layer_bindings.get_binding(dirty_layer) is None
    assert len(viewer.layers) == 1
    clean_layer = viewer.layers[0]
    assert clean_layer is not dirty_layer
    assert isinstance(clean_layer, Shapes)
    assert clean_layer.name == shapes_name
    assert len(clean_layer.data) == initial_row_count
    binding = app_state.viewer_adapter.layer_bindings.get_binding(clean_layer)
    assert isinstance(binding, ShapesLayerBinding)
    assert binding.element_name == shapes_name
    assert binding.coordinate_system == "global"
    assert binding.shapes_role == "primary"
    assert widget._annotation_layer is None
    assert widget._annotation_session is None

    reread = read_zarr(backed_sdata_blobs.path)
    assert len(reread.shapes[shapes_name]) == initial_row_count


def test_shapes_annotation_widget_cancelled_shapes_reload_is_a_strict_no_op(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    backed_sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer()
    app_state = get_or_create_app_state(viewer)
    app_state.set_sdata(backed_sdata_blobs)
    widget = _create_embedded_shapes_annotation(qtbot, viewer)
    shapes_name = "blobs_polygons"
    widget._test_parent.shapes_combo.setCurrentIndex(
        _combo_index_for_text(widget._test_parent.shapes_combo, shapes_name)
    )
    layer = widget._annotation_layer
    session = widget._annotation_session
    live_element = backed_sdata_blobs.shapes[shapes_name]
    assert isinstance(layer, Shapes)
    assert session is not None
    assert widget.reload_shapes_button.isEnabled()

    _add_polygon(layer, offset=100)
    assert widget._test_parent.annotation_context.has_unsaved_shapes_changes is True
    context = widget._test_parent.annotation_context
    status_html = widget.status_label.text()
    emitted_events: list[ShapesElementReloadedEvent] = []
    app_state.shapes_element_reloaded.connect(emitted_events.append)
    monkeypatch.setattr(widget, "_confirm_discard_annotation_layer", lambda *, reason: False)
    monkeypatch.setattr(
        shapes_annotation_widget_module,
        "load_shapes_element_from_store",
        lambda *args, **kwargs: pytest.fail("Cancel must not read the backing store."),
    )

    widget.reload_shapes_button.click()

    assert widget._annotation_layer is layer
    assert widget._annotation_session is session
    assert backed_sdata_blobs.shapes[shapes_name] is live_element
    assert widget._test_parent.annotation_context == context
    assert widget.status_label.text() == status_html
    assert emitted_events == []


@pytest.mark.parametrize("dirty_session", [False, True])
def test_shapes_annotation_widget_reload_replaces_live_layer_and_publishes_one_event(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    backed_sdata_blobs: SpatialData,
    dirty_session: bool,
) -> None:
    viewer = DummyViewer()
    app_state = get_or_create_app_state(viewer)
    app_state.set_sdata(backed_sdata_blobs)
    widget = _create_embedded_shapes_annotation(qtbot, viewer)
    shapes_name = "blobs_polygons"
    widget._test_parent.shapes_combo.setCurrentIndex(
        _combo_index_for_text(widget._test_parent.shapes_combo, shapes_name)
    )
    previous_layer = widget._annotation_layer
    assert isinstance(previous_layer, Shapes)

    persisted_sdata = read_zarr(backed_sdata_blobs.path)
    persisted_replacement = persisted_sdata.shapes[shapes_name].copy()
    persisted_replacement.geometry = persisted_replacement.geometry.translate(xoff=100)
    expected_bounds = persisted_replacement.total_bounds
    write_shapes_element(
        persisted_sdata,
        shapes_name,
        persisted_replacement,
        overwrite=True,
    )

    if dirty_session:
        _add_polygon(previous_layer, offset=200)
    assert widget._test_parent.annotation_context.has_unsaved_shapes_changes is dirty_session
    confirmation_reasons: list[str] = []

    def accept_discard(*, reason: str) -> bool:
        confirmation_reasons.append(reason)
        return True

    monkeypatch.setattr(widget, "_confirm_discard_annotation_layer", accept_discard)
    reloaded_events: list[ShapesElementReloadedEvent] = []
    written_events: list[ShapesElementWrittenEvent] = []
    app_state.shapes_element_reloaded.connect(reloaded_events.append)
    app_state.shapes_element_written.connect(written_events.append)

    widget.reload_shapes_button.click()

    reloaded_layer = widget._annotation_layer
    assert isinstance(reloaded_layer, Shapes)
    assert reloaded_layer is not previous_layer
    assert previous_layer not in viewer.layers
    assert reloaded_layer in viewer.layers
    assert viewer.layers.selection.active is reloaded_layer
    np.testing.assert_allclose(backed_sdata_blobs.shapes[shapes_name].total_bounds, expected_bounds)
    assert widget._annotation_session is not None
    assert widget._annotation_session.mode == "edit_existing"
    assert widget._annotation_session.source_geodataframe is not None
    np.testing.assert_allclose(widget._annotation_session.source_geodataframe.total_bounds, expected_bounds)
    assert widget._test_parent.annotation_context.has_unsaved_shapes_changes is False
    assert confirmation_reasons == (["reload"] if dirty_session else [])
    assert reloaded_events == [
        ShapesElementReloadedEvent(
            sdata=backed_sdata_blobs,
            shapes_name=shapes_name,
            coordinate_system="global",
            source="shapes_annotation_widget",
        )
    ]
    assert written_events == []
    assert "Shapes Reloaded" in _status_text(widget)


def test_shapes_annotation_widget_failed_shapes_reload_preserves_dirty_session(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    backed_sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer()
    app_state = get_or_create_app_state(viewer)
    app_state.set_sdata(backed_sdata_blobs)
    widget = _create_embedded_shapes_annotation(qtbot, viewer)
    shapes_name = "blobs_polygons"
    widget._test_parent.shapes_combo.setCurrentIndex(
        _combo_index_for_text(widget._test_parent.shapes_combo, shapes_name)
    )
    layer = widget._annotation_layer
    session = widget._annotation_session
    live_element = backed_sdata_blobs.shapes[shapes_name]
    assert isinstance(layer, Shapes)
    assert session is not None

    _add_polygon(layer, offset=100)
    context = widget._test_parent.annotation_context
    assert context.has_unsaved_shapes_changes is True
    monkeypatch.setattr(widget, "_confirm_discard_annotation_layer", lambda *, reason: True)
    monkeypatch.setattr(
        shapes_annotation_widget_module,
        "load_shapes_element_from_store",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("persisted Shapes validation failed")),
    )
    emitted_events: list[ShapesElementReloadedEvent] = []
    app_state.shapes_element_reloaded.connect(emitted_events.append)

    widget.reload_shapes_button.click()

    assert widget._annotation_layer is layer
    assert widget._annotation_session is session
    assert backed_sdata_blobs.shapes[shapes_name] is live_element
    assert widget._test_parent.annotation_context == context
    assert widget._test_parent.annotation_context.has_unsaved_shapes_changes is True
    assert emitted_events == []
    assert "Could Not Reload Shapes" in _status_text(widget)
    assert "persisted Shapes validation failed" in _status_text(widget)


def test_shapes_annotation_widget_save_calls_core_with_locked_request_and_reports_success(
    qtbot,
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer()
    widget = _create_ready_annotation_widget(qtbot, viewer, sdata_blobs)
    widget.create_layer_button.click()
    layer = viewer.layers[0]
    _add_polygon(layer)
    parent = widget._test_parent
    assert parent.annotation_context.has_unsaved_shapes_changes is True
    published_contexts: list[AnnotationContext] = []
    parent.annotation_context_changed.connect(published_contexts.append)
    captured_requests = []
    captured_layers = []
    emitted_events: list[object] = []
    widget.app_state.shapes_element_written.connect(emitted_events.append)

    def fake_create_shapes_element(request, napari_layer):
        captured_requests.append(request)
        captured_layers.append(napari_layer)
        request.sdata.shapes[request.shapes_name] = request.sdata.shapes["blobs_polygons"].copy()
        return AnnotateShapesElementResult(
            shapes_name=request.shapes_name,
            coordinate_system=request.coordinate_system,
            row_count=3,
        )

    monkeypatch.setattr(
        shapes_annotation_widget_module,
        "create_shapes_element_from_napari_shapes_layer",
        fake_create_shapes_element,
    )
    # Deliberately perturb the applied parent context to verify that Save uses
    # the active session's locked coordinate system rather than UI selection.
    widget._annotation_context = replace(widget._annotation_context, coordinate_system="local")

    widget.save_shapes_button.click()

    assert captured_layers == [layer]
    assert len(captured_requests) == 1
    request = captured_requests[0]
    assert request.sdata is sdata_blobs
    assert request.shapes_name == "new_regions"
    assert request.coordinate_system == "global"
    assert request.overwrite is False
    assert request.index_name == "instance_id"
    assert request.index_prefix == "__annotation"
    assert widget._annotation_session is not None
    assert widget._annotation_session.mode == "edit_existing"
    assert widget.save_shapes_button.isEnabled() is False
    assert parent.annotation_context.saved_shapes_name == "new_regions"
    assert parent.annotation_context.has_unsaved_shapes_changes is False
    assert published_contexts == [parent.annotation_context]
    assert emitted_events == [
        ShapesElementWrittenEvent(
            sdata=sdata_blobs,
            shapes_name="new_regions",
            coordinate_system="global",
            source="shapes_annotation_widget",
        )
    ]
    status = _status_text(widget)
    assert "Shapes Saved" in status
    assert 'Saved "new_regions" with 3 shape(s) in coordinate system "global".' in status


def test_shapes_annotation_widget_repeated_save_uses_edit_helper_after_create_success(
    qtbot,
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer()
    widget = _create_ready_annotation_widget(qtbot, viewer, sdata_blobs)
    widget.create_layer_button.click()
    _add_polygon(viewer.layers[0])
    original_create_shapes_element = shapes_annotation_widget_module.create_shapes_element_from_napari_shapes_layer
    original_edit_shapes_element = shapes_annotation_widget_module.edit_shapes_element_from_napari_shapes_layer
    create_overwrites: list[bool] = []
    edit_requests = []
    emitted_events: list[object] = []
    widget.app_state.shapes_element_written.connect(emitted_events.append)

    def fake_create_shapes_element(request, napari_layer):
        create_overwrites.append(request.overwrite)
        return original_create_shapes_element(request, napari_layer)

    def fake_edit_shapes_element(request, napari_layer):
        edit_requests.append(request)
        return original_edit_shapes_element(request, napari_layer)

    monkeypatch.setattr(
        shapes_annotation_widget_module,
        "create_shapes_element_from_napari_shapes_layer",
        fake_create_shapes_element,
    )
    monkeypatch.setattr(
        shapes_annotation_widget_module,
        "edit_shapes_element_from_napari_shapes_layer",
        fake_edit_shapes_element,
    )

    widget.save_shapes_button.click()
    _add_polygon(viewer.layers[0], offset=10)
    widget.save_shapes_button.click()

    assert create_overwrites == [False]
    assert len(edit_requests) == 1
    assert edit_requests[0].sdata is sdata_blobs
    assert edit_requests[0].shapes_name == "new_regions"
    assert edit_requests[0].coordinate_system == "global"
    assert edit_requests[0].source_shapes_index_feature_name == "instance_id"
    assert widget._annotation_session is not None
    assert widget._annotation_session.mode == "edit_existing"
    assert emitted_events == [
        ShapesElementWrittenEvent(
            sdata=sdata_blobs,
            shapes_name="new_regions",
            coordinate_system="global",
            source="shapes_annotation_widget",
        ),
        ShapesElementWrittenEvent(
            sdata=sdata_blobs,
            shapes_name="new_regions",
            coordinate_system="global",
            source="shapes_annotation_widget",
        ),
    ]


def test_shapes_annotation_widget_failed_first_save_keeps_later_overwrite_false(
    qtbot,
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer()
    widget = _create_ready_annotation_widget(qtbot, viewer, sdata_blobs)
    widget.create_layer_button.click()
    _add_polygon(viewer.layers[0])
    overwrites: list[bool] = []

    def fake_create_shapes_element(request, napari_layer):
        del napari_layer
        overwrites.append(request.overwrite)
        if len(overwrites) == 1:
            raise ValueError("same-name element appeared externally")
        request.sdata.shapes[request.shapes_name] = request.sdata.shapes["blobs_polygons"].copy()
        return AnnotateShapesElementResult(
            shapes_name=request.shapes_name,
            coordinate_system=request.coordinate_system,
            row_count=1,
        )

    monkeypatch.setattr(
        shapes_annotation_widget_module,
        "create_shapes_element_from_napari_shapes_layer",
        fake_create_shapes_element,
    )

    widget.save_shapes_button.click()

    assert overwrites == [False]
    assert widget._annotation_session is not None
    assert widget._annotation_session.mode == "create_new"
    assert widget.save_shapes_button.isEnabled() is True
    assert "same-name element appeared externally" in _status_text(widget)

    widget.save_shapes_button.click()

    assert overwrites == [False, False]
    assert widget._annotation_session is not None
    assert widget._annotation_session.mode == "edit_existing"


def test_shapes_annotation_widget_empty_layer_save_is_disabled_and_does_not_write(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer()
    widget = _create_ready_annotation_widget(qtbot, viewer, sdata_blobs)
    widget.create_layer_button.click()

    assert widget.save_shapes_button.isEnabled() is False
    widget.save_shapes_button.click()

    assert "new_regions" not in sdata_blobs.shapes
    assert "Annotation Layer Ready" in _status_text(widget)


def test_shapes_annotation_widget_save_writes_real_shapes_element(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer()
    widget = _create_ready_annotation_widget(qtbot, viewer, sdata_blobs)
    widget.create_layer_button.click()
    layer = viewer.layers[0]
    _add_polygon(layer)

    widget.save_shapes_button.click()

    assert "new_regions" in sdata_blobs.shapes
    assert sdata_blobs.shapes["new_regions"].index.name == "instance_id"
    assert sdata_blobs.shapes["new_regions"].index.tolist() == ["__annotation_0"]
    assert layer.features["instance_id"].tolist() == ["__annotation_0"]
    binding = widget.app_state.viewer_adapter.layer_bindings.get_binding(layer)
    assert isinstance(binding, ShapesLayerBinding)
    assert list(binding.source_row_id_by_rendered_row) == [0]
    assert binding.source_shapes_index_feature_name == "instance_id"
    assert list(viewer.layers) == [layer]
    assert widget._annotation_layer is layer
    assert widget._annotation_session is not None
    assert widget._annotation_session.mode == "edit_existing"
    assert widget._annotation_session.source_geodataframe is not sdata_blobs.shapes["new_regions"]
    assert widget._annotation_session.source_geodataframe is not None
    assert widget._annotation_session.source_geodataframe.index.tolist() == ["__annotation_0"]
    assert (
        widget._annotation_context.shapes_target
        == shapes_annotation_widget_module._ShapesAnnotationTarget.edit_existing("new_regions")
    )
    assert widget._test_parent.shapes_combo.currentText() == "new_regions"
    assert widget.name_edit.isHidden() is True
    assert widget.name_edit.text() == ""
    assert "Shapes Saved" in _status_text(widget)

    widget._test_parent.shapes_combo.setCurrentIndex(
        _combo_index_for_text(widget._test_parent.shapes_combo, "Create shapes...")
    )

    assert widget.name_edit.isHidden() is False
    assert widget.name_edit.text() == ""


def test_shapes_annotation_widget_save_syncs_binding_without_primary_registration_event(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer()
    widget = _create_ready_annotation_widget(qtbot, viewer, sdata_blobs)
    emitted_bindings: list[object] = []
    widget.app_state.viewer_adapter.primary_shapes_layer_registered.connect(emitted_bindings.append)
    widget.create_layer_button.click()
    layer = viewer.layers[0]
    emitted_bindings.clear()
    _add_polygon(layer)

    widget.save_shapes_button.click()

    assert emitted_bindings == []
    binding = widget.app_state.viewer_adapter.layer_bindings.get_binding(layer)
    assert isinstance(binding, ShapesLayerBinding)
    assert list(binding.source_row_id_by_rendered_row) == [0]


def test_shapes_annotation_widget_saved_create_new_layer_can_be_reopened_after_target_switch(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer()
    widget = _create_ready_annotation_widget(qtbot, viewer, sdata_blobs)
    widget.create_layer_button.click()
    layer = viewer.layers[0]
    _add_polygon(layer)
    widget.save_shapes_button.click()
    _add_polygon(layer, offset=10)
    widget.save_shapes_button.click()

    binding = widget.app_state.viewer_adapter.layer_bindings.get_binding(layer)
    assert isinstance(binding, ShapesLayerBinding)
    assert list(binding.source_row_id_by_rendered_row) == [0, 1]

    widget._test_parent.shapes_combo.setCurrentIndex(
        _combo_index_for_text(widget._test_parent.shapes_combo, "blobs_polygons")
    )
    assert widget._annotation_session is not None
    assert widget._annotation_session.shapes_name == "blobs_polygons"

    widget._test_parent.shapes_combo.setCurrentIndex(
        _combo_index_for_text(widget._test_parent.shapes_combo, "new_regions")
    )

    assert widget._annotation_layer is layer
    assert widget._annotation_session is not None
    assert widget._annotation_session.shapes_name == "new_regions"
    assert widget.save_shapes_button.isEnabled() is False
    assert "Could Not Open Shapes" not in _status_text(widget)


def test_shapes_annotation_widget_keeps_ownership_when_viewer_adds_saved_primary_shapes(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer()
    widget = _create_ready_annotation_widget(qtbot, viewer, sdata_blobs)
    widget.create_layer_button.click()
    layer = viewer.layers[0]
    _add_polygon(layer)
    widget.save_shapes_button.click()

    result = widget.app_state.viewer_adapter.ensure_shapes_loaded(sdata_blobs, "new_regions", "global")

    assert result.layer is layer
    assert result.created is False
    assert list(viewer.layers) == [layer]
    assert widget._annotation_layer is layer
    assert widget._annotation_shapes_name == "new_regions"
    assert widget._annotation_coordinate_system == "global"
    assert widget._annotation_layer_binding_matches()
    assert widget.save_shapes_button.isEnabled() is False


def test_shapes_annotation_widget_binding_mismatch_disables_save_without_calling_core(
    qtbot,
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer()
    widget = _create_ready_annotation_widget(qtbot, viewer, sdata_blobs)
    widget.create_layer_button.click()
    layer = viewer.layers[0]
    widget.app_state.viewer_adapter.unregister_layer(layer)

    def fail_if_called(request, napari_layer):
        del request, napari_layer
        raise AssertionError("binding mismatch should not call the core save helper")

    monkeypatch.setattr(
        shapes_annotation_widget_module,
        "create_shapes_element_from_napari_shapes_layer",
        fail_if_called,
    )

    # The button is already disabled for the clean empty layer. Invoke the slot
    # directly to verify its fresh readiness guard also catches a stale binding.
    widget._on_save_shapes_clicked()

    assert widget.save_shapes_button.isEnabled() is False
    assert "no longer registered as the widget-owned primary shapes layer" in _status_text(widget)
