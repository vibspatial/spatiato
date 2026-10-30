from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace

import anndata as ad
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_rgba
from napari.layers import Image, Labels, Points, Shapes
from napari.utils.colormaps import CyclicLabelColormap
from shapely.geometry import LineString, Point, Polygon
from spatialdata.models import PointsModel, ShapesModel, TableModel
from spatialdata.transformations import Affine, Identity
from xarray import DataArray

import spatiato.viewer._styling as styling_module
from spatiato._app_state import get_or_create_app_state
from spatiato._points_value_index import PointsValueSelection
from spatiato.core._color_source import ShapeColumnColorSourceSpec, TableColorSourceSpec
from spatiato.core.class_palette import default_categorical_colors
from spatiato.viewer.adapter import (
    ImageLayerBinding,
    LabelsLayerBinding,
    LayerBindingRegistry,
    PointsLayerBinding,
    PointsLayerIdentity,
    ShapesLayerBinding,
    ViewerAdapter,
    _prepare_napari_point_radius_shapes_layer_inputs,
    _prepare_napari_shapes_layer_inputs,
)
from spatiato.viewer.labels_colormap import CompactLabelColormap
from spatiato.viewer.points_styling import POINTS_SELECTION_SOLID_COLOR
from spatiato.viewer.shapes_styling import (
    _SHAPES_EDGE_COLOR_SYNC_CALLBACK_ATTR,
    _SHAPES_EDGE_WIDTH_SYNC_CALLBACK_ATTR,
    _SHAPES_FACE_COLOR_SYNC_CALLBACK_ATTR,
    PRIMARY_SHAPES_FACE_COLOR,
    SHAPES_FACE_ALPHA,
)


class DummyEventEmitter:
    def __init__(self) -> None:
        self._callbacks: list[Callable[[object], None]] = []

    def connect(self, callback: Callable[[object], None]) -> None:
        self._callbacks.append(callback)

    def emit(self, value: object | None = None) -> None:
        event = SimpleNamespace(value=value)
        for callback in list(self._callbacks):
            callback(event)


class DummyLayers(list):
    def __init__(self, layers: list[object] | None = None) -> None:
        super().__init__(layers or [])
        self.selection = SimpleNamespace(active=None, select_only=self._select_only)
        self.events = SimpleNamespace(
            inserted=DummyEventEmitter(),
            removed=DummyEventEmitter(),
            reordered=DummyEventEmitter(),
        )

    def _select_only(self, layer: object) -> None:
        self.selection.active = layer

    def remove(self, layer: object) -> None:
        super().remove(layer)
        self.events.removed.emit(layer)


class DummyViewer:
    def __init__(self, layers: list[object] | None = None, *, reset_camera_on_layer_change: bool = False) -> None:
        self.layers = DummyLayers(layers)
        self.camera = SimpleNamespace(center=(10.0, 20.0), zoom=3.5)
        self._reset_camera_on_layer_change = reset_camera_on_layer_change

    def add_layer(self, layer: object) -> object:
        self.layers.append(layer)
        self.layers.events.inserted.emit(layer)
        if self._reset_camera_on_layer_change:
            self.camera.center = (0.0, 0.0)
            self.camera.zoom = 1.0
        return layer


class SilentRemovalLayers(DummyLayers):
    def remove(self, layer: object) -> None:
        list.remove(self, layer)


class SilentRemovalViewer(DummyViewer):
    def __init__(self, layers: list[object] | None = None) -> None:
        super().__init__()
        self.layers = SilentRemovalLayers(layers)


class UniqueNameDummyViewer(DummyViewer):
    def add_layer(self, layer: object) -> object:
        name = getattr(layer, "name", None)
        if isinstance(name, str):
            existing_names = {getattr(existing_layer, "name", None) for existing_layer in self.layers}
            if name in existing_names:
                suffix = 1
                candidate_name = f"{name} [{suffix}]"
                while candidate_name in existing_names:
                    suffix += 1
                    candidate_name = f"{name} [{suffix}]"
                layer.name = candidate_name
        return super().add_layer(layer)


class UnsupportedViewer:
    def __init__(self) -> None:
        self.layers = None


class ShapesSpatialDataWithTables(SimpleNamespace):
    def __getitem__(self, key: str):
        return self.tables[key]


def make_labels_layer(*, sdata, labels_name: str = "blobs_labels", metadata: dict[str, object] | None = None) -> Labels:
    return Labels(
        sdata.labels[labels_name],
        name=labels_name,
        metadata={} if metadata is None else metadata,
    )


def make_image_layer(*, name: str, metadata: dict[str, object] | None = None) -> Image:
    return Image(
        np.zeros((8, 8), dtype=np.float32),
        name=name,
        metadata={} if metadata is None else metadata,
    )


def make_shapes_layer(*, name: str = "cell_boundaries", metadata: dict[str, object] | None = None) -> Shapes:
    return Shapes(
        [np.asarray([(0, 0), (0, 4), (4, 4), (4, 0)], dtype=float)],
        shape_type="polygon",
        name=name,
        metadata={} if metadata is None else metadata,
    )


def make_points_selection(
    row_values: list[str],
    *,
    selected_values: tuple[str, ...] | None = None,
    selection_mode: str = "values",
    total_count: int | None = None,
    render_point_budget: int = 100_000,
    is_sampled: bool = False,
    warning: str | None = None,
) -> PointsValueSelection:
    if selected_values is None:
        selected_values = tuple(dict.fromkeys(row_values))
    value_id_by_value = {value: index for index, value in enumerate(selected_values)}
    coordinates = np.asarray(
        [[float(index), float(index + 10)] for index in range(len(row_values))],
        dtype="float32",
    )
    features = pd.DataFrame(
        {
            "gene": pd.Categorical(row_values, categories=list(selected_values)),
            "value_id": pd.Series([value_id_by_value[value] for value in row_values], dtype="uint32"),
        }
    )
    return PointsValueSelection(
        coordinates=coordinates,
        features=features,
        index_column="gene",
        selected_values=selected_values,
        selected_value_ids=tuple(value_id_by_value[value] for value in selected_values),
        selection_mode=selection_mode,  # type: ignore[arg-type]
        total_count=len(row_values) if total_count is None else total_count,
        render_point_budget=render_point_budget,
        is_sampled=is_sampled,
        warning=warning,
    )


def make_points_identity(
    sdata: object,
    *,
    index_column: str = "gene",
    coordinate_system: str = "global",
) -> PointsLayerIdentity:
    return PointsLayerIdentity(
        sdata=sdata,
        points_name="transcripts",
        coordinate_system=coordinate_system,
        index_column=index_column,
    )


def make_shapes_sdata(geodataframe: gpd.GeoDataFrame, shapes_name: str = "cell_boundaries") -> SimpleNamespace:
    shapes = ShapesModel.parse(geodataframe, transformations={"global": Identity()})
    return SimpleNamespace(shapes={shapes_name: shapes})


def make_colorable_shapes_sdata(shapes_name: str = "cell_boundaries") -> SimpleNamespace:
    geodataframe = gpd.GeoDataFrame(
        {
            "cell_type": pd.Categorical(["T", "B"], categories=["T", "B"]),
            "cell_type_colors": ["#ff0000", "#00ff00"],
            "score": [0.0, 1.0],
        },
        geometry=[
            Polygon([(0, 0), (4, 0), (4, 4), (0, 4), (0, 0)]),
            Polygon([(5, 0), (9, 0), (9, 4), (5, 4), (5, 0)]),
        ],
        index=["cell_1", "cell_2"],
    )
    return make_shapes_sdata(geodataframe, shapes_name=shapes_name)


def make_colorable_point_radius_shapes_sdata(shapes_name: str = "cell_centroids") -> SimpleNamespace:
    geodataframe = gpd.GeoDataFrame(
        {
            "radius": [2.0, 3.0],
            "cell_type": pd.Categorical(["T", "B"], categories=["T", "B"]),
            "cell_type_colors": ["#ff0000", "#00ff00"],
            "score": [0.0, 1.0],
        },
        geometry=[Point(10, 20), Point(30, 40)],
        index=["cell_1", "cell_2"],
    )
    return make_shapes_sdata(geodataframe, shapes_name=shapes_name)


def make_table_backed_shapes_sdata() -> ShapesSpatialDataWithTables:
    geodataframe = gpd.GeoDataFrame(
        {"instance_id": ["cell_1", "cell_2", "cell_3"]},
        geometry=[
            Polygon([(0, 0), (4, 0), (4, 4), (0, 4), (0, 0)]),
            Polygon([(5, 0), (9, 0), (9, 4), (5, 4), (5, 0)]),
            Polygon([(10, 0), (14, 0), (14, 4), (10, 4), (10, 0)]),
        ],
        index=["cell_1", "cell_2", "cell_3"],
    )
    geodataframe.index.name = "instance_id"
    shapes = ShapesModel.parse(geodataframe, transformations={"global": Identity()})
    table = TableModel.parse(
        ad.AnnData(
            obs=pd.DataFrame(
                {
                    "region": ["cell_boundaries", "cell_boundaries"],
                    "instance_id": ["cell_1", "cell_2"],
                    "cell_type": pd.Categorical(["T", "B"], categories=["T", "B"]),
                },
                index=["obs_1", "obs_2"],
            )
        ),
        region="cell_boundaries",
        region_key="region",
        instance_key="instance_id",
    )
    table.uns["cell_type_colors"] = ["#ff0000", "#00ff00"]
    return ShapesSpatialDataWithTables(shapes={"cell_boundaries": shapes}, tables={"table": table})


def make_table_backed_point_radius_shapes_sdata() -> ShapesSpatialDataWithTables:
    geodataframe = gpd.GeoDataFrame(
        {
            "instance_id": ["cell_1", "cell_2", "cell_3"],
            "radius": [2.0, 3.0, 4.0],
        },
        geometry=[Point(10, 20), Point(30, 40), Point(50, 60)],
        index=["cell_1", "cell_2", "cell_3"],
    )
    geodataframe.index.name = "instance_id"
    shapes = ShapesModel.parse(geodataframe, transformations={"global": Identity()})
    table = TableModel.parse(
        ad.AnnData(
            X=np.asarray([[1.0], [2.0]]),
            obs=pd.DataFrame(
                {
                    "region": ["cell_centroids", "cell_centroids"],
                    "instance_id": ["cell_1", "cell_2"],
                    "cell_type": pd.Categorical(["T", "B"], categories=["T", "B"]),
                },
                index=["obs_1", "obs_2"],
            ),
            var=pd.DataFrame(index=["GeneA"]),
        ),
        region="cell_centroids",
        region_key="region",
        instance_key="instance_id",
    )
    table.uns["cell_type_colors"] = ["#ff0000", "#00ff00"]
    return ShapesSpatialDataWithTables(shapes={"cell_centroids": shapes}, tables={"table": table})


def expected_radius_reference_size(sizes: np.ndarray) -> float:
    sizes = np.asarray(sizes, dtype=float)
    return float(sizes[0]) if np.allclose(sizes, sizes[0]) else float(np.median(sizes))


def make_two_table_backed_shapes_sdata() -> ShapesSpatialDataWithTables:
    sdata = make_table_backed_shapes_sdata()
    table_b = TableModel.parse(
        ad.AnnData(
            obs=pd.DataFrame(
                {
                    "region": ["cell_boundaries", "cell_boundaries"],
                    "instance_id": ["cell_1", "cell_2"],
                    "cell_type": pd.Categorical(["T", "B"], categories=["T", "B"]),
                },
                index=["obs_1", "obs_2"],
            )
        ),
        region="cell_boundaries",
        region_key="region",
        instance_key="instance_id",
    )
    table_b.uns["cell_type_colors"] = ["#0000ff", "#ffff00"]
    sdata.tables["table_b"] = table_b
    return sdata


def get_shapes_binding(adapter: ViewerAdapter, layer: Shapes | Points) -> ShapesLayerBinding:
    binding = adapter.layer_bindings.get_binding(layer)
    assert isinstance(binding, ShapesLayerBinding)
    return binding


def test_app_state_initializes_viewer_services() -> None:
    viewer = DummyViewer()

    state = get_or_create_app_state(viewer)

    assert state.layer_bindings is not None
    assert state.viewer_adapter is not None
    assert state.viewer_adapter.layer_bindings is state.layer_bindings


def test_layer_binding_registry_registers_and_unregisters_layers() -> None:
    registry = LayerBindingRegistry()
    layer = make_image_layer(name="raw image")

    binding = registry.register_image_layer(
        layer,
        element_name="blobs_image",
        coordinate_system="global",
    )

    assert isinstance(binding, ImageLayerBinding)
    assert binding.layer is layer
    assert binding.element_name == "blobs_image"
    assert binding.element_type == "image"
    assert binding.coordinate_system == "global"
    assert registry.get_binding(layer) == binding
    assert registry.find_bindings(element_name="blobs_image", element_type="image") == [binding]
    assert "element_name" not in layer.metadata
    assert "element_type" not in layer.metadata
    assert "coordinate_system" not in layer.metadata

    removed = registry.unregister_layer(layer)

    assert removed == binding
    assert registry.get_binding(layer) is None


def test_layer_binding_registry_tracks_channel_overlay_identity() -> None:
    registry = LayerBindingRegistry()
    layer = make_image_layer(name="channel_0")

    binding = registry.register_image_layer(
        layer,
        element_name="blobs_image",
        coordinate_system="global",
        image_display_mode="overlay",
        channel_index=0,
        channel_name="DAPI",
    )

    assert isinstance(binding, ImageLayerBinding)
    assert binding.image_display_mode == "overlay"
    assert binding.channel_index == 0
    assert binding.channel_name == "DAPI"
    assert registry.find_bindings(
        element_name="blobs_image",
        element_type="image",
        image_display_mode="overlay",
        channel_index=0,
        channel_name="DAPI",
    ) == [binding]
    assert "image_display_mode" not in layer.metadata
    assert "channel_index" not in layer.metadata
    assert "channel_name" not in layer.metadata


def test_layer_binding_registry_tracks_points_identity() -> None:
    registry = LayerBindingRegistry()
    layer = Points(np.asarray([[0.0, 1.0]], dtype="float32"), name="transcripts: gene=AAMP")

    binding = registry.register_points_layer(
        layer,
        element_name="transcripts",
        coordinate_system="global",
        index_column="gene",
    )

    assert isinstance(binding, PointsLayerBinding)
    assert binding.element_name == "transcripts"
    assert binding.element_type == "points"
    assert binding.coordinate_system == "global"
    assert binding.index_column == "gene"
    assert registry.find_bindings(
        element_name="transcripts",
        element_type="points",
        points_index_column="gene",
    ) == [binding]
    assert "element_name" not in layer.metadata
    assert "element_type" not in layer.metadata
    assert "coordinate_system" not in layer.metadata
    assert "index_column" not in layer.metadata


def test_layer_binding_registry_tracks_shapes_identity() -> None:
    registry = LayerBindingRegistry()
    layer = make_shapes_layer()

    binding = registry.register_shapes_layer(
        layer,
        element_name="cell_boundaries",
        coordinate_system="global",
        source_row_id_by_rendered_row=(0,),
        source_shapes_index_feature_name="cell_id",
        skipped_geometry_count=2,
    )

    assert isinstance(binding, ShapesLayerBinding)
    assert binding.element_name == "cell_boundaries"
    assert binding.element_type == "shapes"
    assert binding.coordinate_system == "global"
    assert binding.shapes_role == "primary"
    assert binding.shapes_rendering_mode == "shapes"
    assert binding.style_spec is None
    assert binding.source_row_id_by_rendered_row == (0,)
    assert binding.source_shapes_index_feature_name == "cell_id"
    assert binding.skipped_geometry_count == 2
    assert registry.find_bindings(element_name="cell_boundaries", element_type="shapes") == [binding]
    assert "element_name" not in layer.metadata
    assert "element_type" not in layer.metadata
    assert "coordinate_system" not in layer.metadata
    assert "source_row_id_by_rendered_row" not in layer.metadata
    assert "source_shapes_index_feature_name" not in layer.metadata
    assert "skipped_geometry_count" not in layer.metadata


def test_layer_binding_registry_tracks_shapes_role_and_style_spec() -> None:
    registry = LayerBindingRegistry()
    layer = make_shapes_layer()
    style_spec = ShapeColumnColorSourceSpec(
        source_kind="shape_column",
        value_key="cell_type",
        value_kind="categorical",
    )

    binding = registry.register_shapes_layer(
        layer,
        element_name="cell_boundaries",
        coordinate_system="global",
        shapes_role="styled",
        style_spec=style_spec,
        source_row_id_by_rendered_row=(0,),
        source_shapes_index_feature_name="cell_id",
    )

    assert isinstance(binding, ShapesLayerBinding)
    assert binding.shapes_role == "styled"
    assert binding.shapes_rendering_mode == "shapes"
    assert binding.style_spec == style_spec
    assert binding.source_row_id_by_rendered_row == (0,)
    assert registry.find_bindings(
        element_name="cell_boundaries",
        element_type="shapes",
        shapes_role="styled",
        style_spec=style_spec,
    ) == [binding]
    assert "shapes_role" not in layer.metadata
    assert "style_spec" not in layer.metadata
    assert "style_source_kind" not in layer.metadata
    assert "style_value_key" not in layer.metadata
    assert "style_value_kind" not in layer.metadata


def test_layer_binding_registry_tracks_table_backed_shapes_style_spec() -> None:
    registry = LayerBindingRegistry()
    layer = make_shapes_layer()
    style_spec = TableColorSourceSpec(
        table_name="table",
        source_kind="obs_column",
        value_key="cell_type",
        value_kind="categorical",
    )

    binding = registry.register_shapes_layer(
        layer,
        element_name="cell_boundaries",
        coordinate_system="global",
        shapes_role="styled",
        style_spec=style_spec,
        source_row_id_by_rendered_row=(0,),
        source_shapes_index_feature_name="cell_id",
    )

    assert binding.style_spec == style_spec
    assert registry.find_bindings(
        element_name="cell_boundaries",
        element_type="shapes",
        shapes_role="styled",
        style_spec=style_spec,
    ) == [binding]


def test_shapes_layer_binding_rejects_invalid_role_style_spec_combinations() -> None:
    layer = make_shapes_layer()
    style_spec = ShapeColumnColorSourceSpec(
        source_kind="shape_column",
        value_key="cell_type",
        value_kind="categorical",
    )

    with pytest.raises(ValueError, match="Primary shapes bindings must not carry"):
        ShapesLayerBinding(
            layer=layer,
            element_name="cell_boundaries",
            coordinate_system="global",
            shapes_role="primary",
            style_spec=style_spec,
        )

    with pytest.raises(ValueError, match="Styled shapes bindings require"):
        ShapesLayerBinding(
            layer=layer,
            element_name="cell_boundaries",
            coordinate_system="global",
            shapes_role="styled",
        )


def test_labels_layer_binding_rejects_invalid_role_style_spec_combinations() -> None:
    layer = make_labels_layer(sdata=SimpleNamespace(labels={"blobs_labels": np.zeros((2, 2), dtype=np.int32)}))
    style_spec = TableColorSourceSpec(
        table_name="table",
        source_kind="obs_column",
        value_key="cell_type",
        value_kind="categorical",
    )

    with pytest.raises(ValueError, match="Primary labels bindings must not carry"):
        LabelsLayerBinding(
            layer=layer,
            element_name="blobs_labels",
            coordinate_system="global",
            labels_role="primary",
            style_spec=style_spec,
        )

    with pytest.raises(ValueError, match="Styled labels bindings require"):
        LabelsLayerBinding(
            layer=layer,
            element_name="blobs_labels",
            coordinate_system="global",
            labels_role="styled",
        )


def test_points_layer_binding_rejects_empty_index_column() -> None:
    layer = Points(np.asarray([[0.0, 1.0]], dtype="float32"), name="points")

    with pytest.raises(ValueError, match="index column"):
        PointsLayerBinding(
            layer=layer,
            element_name="transcripts",
            coordinate_system="global",
            index_column="",
        )


def test_layer_binding_registry_tracks_labels_role_and_style_spec() -> None:
    registry = LayerBindingRegistry()
    layer = make_labels_layer(sdata=SimpleNamespace(labels={"blobs_labels": np.zeros((2, 2), dtype=np.int32)}))
    style_spec = TableColorSourceSpec(
        table_name="table",
        source_kind="obs_column",
        value_key="cell_type",
        value_kind="categorical",
    )

    binding = registry.register_labels_layer(
        layer,
        element_name="blobs_labels",
        coordinate_system="global",
        labels_role="styled",
        style_spec=style_spec,
    )

    assert isinstance(binding, LabelsLayerBinding)
    assert binding.labels_role == "styled"
    assert binding.style_spec == style_spec
    assert registry.find_bindings(
        element_name="blobs_labels",
        element_type="labels",
        labels_role="styled",
        style_spec=style_spec,
    ) == [binding]
    assert "labels_role" not in layer.metadata
    assert "style_spec" not in layer.metadata
    assert "style_table_name" not in layer.metadata
    assert "style_source_kind" not in layer.metadata
    assert "style_value_key" not in layer.metadata
    assert "style_value_kind" not in layer.metadata


def test_viewer_adapter_activate_layer_selects_only_matching_layer() -> None:
    image_layer = make_image_layer(name="image")
    viewer = DummyViewer([image_layer])
    adapter = ViewerAdapter(viewer)

    activated = adapter.activate_layer(image_layer)

    assert activated is True
    assert viewer.layers.selection.active is image_layer


def test_viewer_adapter_emits_active_layer_changed_on_activation() -> None:
    image_layer = make_image_layer(name="image")
    viewer = DummyViewer([image_layer])
    adapter = ViewerAdapter(viewer)
    active_layers: list[object] = []

    adapter.active_layer_changed.connect(active_layers.append)

    adapter.activate_layer(image_layer)

    assert active_layers == [image_layer]


def test_viewer_adapter_ensure_labels_loaded_rejects_unsupported_viewer(sdata_blobs) -> None:
    adapter = ViewerAdapter(UnsupportedViewer())

    try:
        adapter.ensure_labels_loaded(sdata_blobs, "blobs_labels", "global")
    except ValueError as error:
        assert "does not expose napari-compatible layer APIs" in str(error)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected ensure_labels_loaded to reject an unsupported viewer.")


def test_viewer_adapter_finds_registered_labels_layer(sdata_blobs) -> None:
    labels_layer = make_labels_layer(sdata=sdata_blobs)
    viewer = DummyViewer([labels_layer])
    adapter = ViewerAdapter(viewer)

    adapter.register_labels_layer(
        labels_layer,
        sdata=sdata_blobs,
        labels_name="blobs_labels",
        coordinate_system="global",
    )

    assert adapter.get_loaded_primary_labels_layer(sdata_blobs, "blobs_labels") is labels_layer


def test_viewer_adapter_primary_labels_lookup_ignores_styled_variants(sdata_blobs) -> None:
    styled_layer = make_labels_layer(sdata=sdata_blobs)
    viewer = DummyViewer([styled_layer])
    adapter = ViewerAdapter(viewer)
    style_spec = TableColorSourceSpec(
        table_name="table",
        source_kind="obs_column",
        value_key="cell_type",
        value_kind="categorical",
    )

    adapter.register_labels_layer(
        styled_layer,
        sdata=sdata_blobs,
        labels_name="blobs_labels",
        coordinate_system="global",
        labels_role="styled",
        style_spec=style_spec,
    )

    assert adapter.get_loaded_primary_labels_layer(sdata_blobs, "blobs_labels", "global") is None
    assert adapter.get_loaded_styled_labels_layer(sdata_blobs, "blobs_labels", style_spec, "global") is styled_layer
    assert adapter.get_loaded_styled_labels_layers(sdata_blobs, "blobs_labels", "global") == [styled_layer]


def test_viewer_adapter_ignores_unregistered_labels_layer_even_with_legacy_metadata(sdata_blobs) -> None:
    labels_layer = make_labels_layer(
        sdata=sdata_blobs,
        metadata={"sdata": sdata_blobs, "name": "blobs_labels", "_current_cs": "global"},
    )
    viewer = DummyViewer([labels_layer])
    adapter = ViewerAdapter(viewer)

    loaded_layer = adapter.get_loaded_primary_labels_layer(sdata_blobs, "blobs_labels", "global")

    assert loaded_layer is None
    assert adapter.layer_bindings.get_binding(labels_layer) is None


def test_viewer_adapter_remove_labels_layer_removes_matching_binding(sdata_blobs) -> None:
    labels_layer = make_labels_layer(sdata=sdata_blobs)
    viewer = DummyViewer([labels_layer])
    adapter = ViewerAdapter(viewer)

    adapter.register_labels_layer(
        labels_layer,
        sdata=sdata_blobs,
        labels_name="blobs_labels",
        coordinate_system="global",
    )

    removed_layer = adapter.remove_labels_layer(sdata_blobs, "blobs_labels", "global")

    assert removed_layer is labels_layer
    assert list(viewer.layers) == []
    assert adapter.layer_bindings.get_binding(labels_layer) is None


def test_viewer_adapter_returns_registered_image_layers_in_viewer_order(sdata_blobs) -> None:
    first_layer = make_image_layer(name="channel_0")
    second_layer = make_image_layer(name="channel_1")
    viewer = DummyViewer([second_layer, first_layer])
    adapter = ViewerAdapter(viewer)

    adapter.register_image_layer(
        first_layer,
        sdata=sdata_blobs,
        image_name="blobs_image",
        coordinate_system="global",
    )
    adapter.register_image_layer(
        second_layer,
        sdata=sdata_blobs,
        image_name="blobs_image",
        coordinate_system="global",
    )

    assert adapter.get_loaded_image_layers(sdata_blobs, "blobs_image") == [second_layer, first_layer]


def test_viewer_adapter_remove_image_layers_removes_stack_and_overlay_bindings(sdata_blobs) -> None:
    stack_layer = make_image_layer(name="blobs_image")
    overlay_layer = make_image_layer(name="blobs_image[0]")
    viewer = DummyViewer([stack_layer, overlay_layer])
    adapter = ViewerAdapter(viewer)

    adapter.register_image_layer(
        stack_layer,
        sdata=sdata_blobs,
        image_name="blobs_image",
        coordinate_system="global",
        image_display_mode="stack",
    )
    adapter.register_image_layer(
        overlay_layer,
        sdata=sdata_blobs,
        image_name="blobs_image",
        coordinate_system="global",
        image_display_mode="overlay",
        channel_index=0,
        channel_name="0",
    )

    removed_layers = adapter.remove_image_layers(sdata_blobs, "blobs_image", "global")

    assert removed_layers == [stack_layer, overlay_layer]
    assert list(viewer.layers) == []
    assert adapter.layer_bindings.get_binding(stack_layer) is None
    assert adapter.layer_bindings.get_binding(overlay_layer) is None


def test_viewer_adapter_remove_image_stack_layer_preserves_nonmatching_layers(sdata_blobs) -> None:
    stack_layer = make_image_layer(name="blobs_image")
    overlay_layer = make_image_layer(name="blobs_image[0]")
    other_stack_layer = make_image_layer(name="other_image")
    viewer = DummyViewer([stack_layer, overlay_layer, other_stack_layer])
    adapter = ViewerAdapter(viewer)
    adapter.register_image_layer(
        stack_layer,
        sdata=sdata_blobs,
        image_name="blobs_image",
        coordinate_system="global",
        image_display_mode="stack",
    )
    adapter.register_image_layer(
        overlay_layer,
        sdata=sdata_blobs,
        image_name="blobs_image",
        coordinate_system="global",
        image_display_mode="overlay",
        channel_index=0,
        channel_name="0",
    )
    adapter.register_image_layer(
        other_stack_layer,
        sdata=sdata_blobs,
        image_name="other_image",
        coordinate_system="global",
        image_display_mode="stack",
    )
    image_events: list[str] = []
    adapter.image_layers_changed.connect(lambda: image_events.append("changed"))

    removed_layer = adapter.remove_image_stack_layer(
        sdata_blobs,
        "blobs_image",
        "global",
    )

    assert removed_layer is stack_layer
    assert list(viewer.layers) == [overlay_layer, other_stack_layer]
    assert adapter.layer_bindings.get_binding(stack_layer) is None
    assert adapter.layer_bindings.get_binding(overlay_layer) is not None
    assert adapter.layer_bindings.get_binding(other_stack_layer) is not None
    assert image_events == ["changed"]


def test_viewer_adapter_remove_image_stack_layer_rejects_duplicate_live_matches(
    sdata_blobs,
) -> None:
    first_layer = make_image_layer(name="blobs_image")
    second_layer = make_image_layer(name="blobs_image duplicate")
    viewer = DummyViewer([first_layer, second_layer])
    adapter = ViewerAdapter(viewer)
    for layer in (first_layer, second_layer):
        adapter.register_image_layer(
            layer,
            sdata=sdata_blobs,
            image_name="blobs_image",
            coordinate_system="global",
            image_display_mode="stack",
        )

    with pytest.raises(ValueError, match="multiple live stack layers"):
        adapter.remove_image_stack_layer(
            sdata_blobs,
            "blobs_image",
            "global",
        )

    assert list(viewer.layers) == [first_layer, second_layer]


def test_viewer_adapter_remove_image_overlay_channel_preserves_nonmatching_layers(sdata_blobs) -> None:
    target_layer = make_image_layer(name="blobs_image[1]")
    sibling_layer = make_image_layer(name="blobs_image[2]")
    stack_layer = make_image_layer(name="blobs_image")
    other_image_layer = make_image_layer(name="other_image[1]")
    other_coordinate_system_layer = make_image_layer(name="blobs_image_local[1]")
    other_sdata_layer = make_image_layer(name="other_sdata[1]")
    other_sdata = object()
    kept_layers = [
        sibling_layer,
        stack_layer,
        other_image_layer,
        other_coordinate_system_layer,
        other_sdata_layer,
    ]
    viewer = DummyViewer([target_layer, *kept_layers])
    adapter = ViewerAdapter(viewer)

    for layer, image_name, coordinate_system, sdata, mode, channel_index in (
        (target_layer, "blobs_image", "global", sdata_blobs, "overlay", 1),
        (sibling_layer, "blobs_image", "global", sdata_blobs, "overlay", 2),
        (stack_layer, "blobs_image", "global", sdata_blobs, "stack", None),
        (other_image_layer, "other_image", "global", sdata_blobs, "overlay", 1),
        (other_coordinate_system_layer, "blobs_image", "local", sdata_blobs, "overlay", 1),
        (other_sdata_layer, "blobs_image", "global", other_sdata, "overlay", 1),
    ):
        adapter.register_image_layer(
            layer,
            sdata=sdata,
            image_name=image_name,
            coordinate_system=coordinate_system,
            image_display_mode=mode,
            channel_index=channel_index,
            channel_name=None if channel_index is None else str(channel_index),
        )

    overlay_events: list[str] = []
    adapter.image_layers_changed.connect(lambda: overlay_events.append("changed"))

    removed_layer = adapter.remove_image_overlay_channel(
        sdata_blobs,
        "blobs_image",
        "global",
        channel_index=1,
    )

    assert removed_layer is target_layer
    assert target_layer not in viewer.layers
    assert adapter.layer_bindings.get_binding(target_layer) is None
    assert list(viewer.layers) == kept_layers
    assert all(adapter.layer_bindings.get_binding(layer) is not None for layer in kept_layers)
    assert overlay_events == ["changed"]


def test_viewer_adapter_remove_image_overlay_channel_missing_is_safe_noop(sdata_blobs) -> None:
    layer = make_image_layer(name="blobs_image[0]")
    viewer = DummyViewer([layer])
    adapter = ViewerAdapter(viewer)
    adapter.register_image_layer(
        layer,
        sdata=sdata_blobs,
        image_name="blobs_image",
        coordinate_system="global",
        image_display_mode="overlay",
        channel_index=0,
        channel_name="0",
    )
    overlay_events: list[str] = []
    adapter.image_layers_changed.connect(lambda: overlay_events.append("changed"))

    removed_layer = adapter.remove_image_overlay_channel(
        sdata_blobs,
        "blobs_image",
        "global",
        channel_index=2,
    )

    assert removed_layer is None
    assert list(viewer.layers) == [layer]
    assert adapter.layer_bindings.get_binding(layer) is not None
    assert overlay_events == []


def test_viewer_adapter_remove_image_overlay_channel_rejects_negative_index(sdata_blobs) -> None:
    adapter = ViewerAdapter(DummyViewer())

    with pytest.raises(ValueError, match="must be non-negative"):
        adapter.remove_image_overlay_channel(
            sdata_blobs,
            "blobs_image",
            "global",
            channel_index=-1,
        )


def test_viewer_adapter_remove_image_overlay_channel_rejects_duplicate_live_matches(sdata_blobs) -> None:
    first_layer = make_image_layer(name="blobs_image[0]")
    second_layer = make_image_layer(name="blobs_image[0]-duplicate")
    viewer = DummyViewer([first_layer, second_layer])
    adapter = ViewerAdapter(viewer)
    for layer in (first_layer, second_layer):
        adapter.register_image_layer(
            layer,
            sdata=sdata_blobs,
            image_name="blobs_image",
            coordinate_system="global",
            image_display_mode="overlay",
            channel_index=0,
            channel_name="0",
        )

    with pytest.raises(ValueError, match="multiple live overlay layers"):
        adapter.remove_image_overlay_channel(
            sdata_blobs,
            "blobs_image",
            "global",
            channel_index=0,
        )

    assert list(viewer.layers) == [first_layer, second_layer]


def test_viewer_adapter_remove_layers_outside_coordinate_system_removes_only_nonmatching_registered_layers(
    sdata_blobs,
) -> None:
    keep_layer = make_image_layer(name="keep")
    remove_image_layer = make_image_layer(name="remove_image")
    remove_labels_layer = make_labels_layer(sdata=sdata_blobs)
    remove_shapes_layer = make_shapes_layer(name="remove_shapes")
    external_layer = make_image_layer(name="external")
    viewer = DummyViewer([keep_layer, remove_image_layer, remove_labels_layer, remove_shapes_layer, external_layer])
    adapter = ViewerAdapter(viewer)

    adapter.register_image_layer(
        keep_layer,
        sdata=sdata_blobs,
        image_name="keep_image",
        coordinate_system="global",
    )
    adapter.register_image_layer(
        remove_image_layer,
        sdata=sdata_blobs,
        image_name="remove_image",
        coordinate_system="local",
    )
    adapter.register_labels_layer(
        remove_labels_layer,
        sdata=sdata_blobs,
        labels_name="blobs_labels",
        coordinate_system="local",
    )
    adapter.register_shapes_layer(
        remove_shapes_layer,
        sdata=sdata_blobs,
        shapes_name="blobs_polygons",
        coordinate_system="local",
    )

    removed_bindings = adapter.remove_layers_outside_coordinate_system(sdata=sdata_blobs, coordinate_system="global")

    assert [binding.element_name for binding in removed_bindings] == ["remove_image", "blobs_labels", "blobs_polygons"]
    assert list(viewer.layers) == [keep_layer, external_layer]
    assert adapter.layer_bindings.get_binding(keep_layer) is not None
    assert adapter.layer_bindings.get_binding(remove_image_layer) is None
    assert adapter.layer_bindings.get_binding(remove_labels_layer) is None
    assert adapter.layer_bindings.get_binding(remove_shapes_layer) is None
    assert adapter.layer_bindings.get_binding(external_layer) is None


def test_viewer_adapter_create_empty_primary_shapes_layer_registers_and_styles_layer(sdata_blobs) -> None:
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)

    layer = adapter.create_empty_primary_shapes_layer(sdata_blobs, "new_regions", "global")

    assert list(viewer.layers) == [layer]
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
    assert hasattr(layer, _SHAPES_FACE_COLOR_SYNC_CALLBACK_ATTR)

    binding = adapter.layer_bindings.get_binding(layer)
    assert isinstance(binding, ShapesLayerBinding)
    assert binding.element_name == "new_regions"
    assert binding.coordinate_system == "global"
    assert binding.shapes_role == "primary"
    assert binding.shapes_rendering_mode == "shapes"
    assert binding.style_spec is None


def test_viewer_adapter_create_empty_primary_shapes_layer_removes_layer_if_registration_fails(
    monkeypatch,
    sdata_blobs,
) -> None:
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)

    def fail_register(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("registration failed")

    monkeypatch.setattr(adapter, "register_shapes_layer", fail_register)

    with pytest.raises(RuntimeError, match="registration failed"):
        adapter.create_empty_primary_shapes_layer(sdata_blobs, "new_regions", "global")

    assert list(viewer.layers) == []
    assert adapter.layer_bindings.iter_bindings() == ()


def test_viewer_adapter_remove_layers_for_sdata_removes_only_matching_registered_layers(sdata_blobs) -> None:
    removed_image_layer = make_image_layer(name="remove_image")
    removed_labels_layer = make_labels_layer(sdata=sdata_blobs)
    removed_shapes_layer = make_shapes_layer(name="remove_shapes")
    kept_layer = make_image_layer(name="keep")
    external_layer = make_image_layer(name="external")
    other_sdata = SimpleNamespace()
    viewer = DummyViewer([removed_image_layer, removed_labels_layer, removed_shapes_layer, kept_layer, external_layer])
    adapter = ViewerAdapter(viewer)

    adapter.register_image_layer(
        removed_image_layer,
        sdata=sdata_blobs,
        image_name="remove_image",
        coordinate_system="global",
    )
    adapter.register_labels_layer(
        removed_labels_layer,
        sdata=sdata_blobs,
        labels_name="blobs_labels",
        coordinate_system="global",
    )
    adapter.register_shapes_layer(
        removed_shapes_layer,
        sdata=sdata_blobs,
        shapes_name="blobs_polygons",
        coordinate_system="global",
    )
    adapter.register_image_layer(
        kept_layer,
        sdata=other_sdata,
        image_name="keep_image",
        coordinate_system="global",
    )

    removed_bindings = adapter.remove_layers_for_sdata(sdata_blobs)

    assert [binding.element_name for binding in removed_bindings] == ["remove_image", "blobs_labels", "blobs_polygons"]
    assert list(viewer.layers) == [kept_layer, external_layer]
    assert adapter.layer_bindings.get_binding(removed_image_layer) is None
    assert adapter.layer_bindings.get_binding(removed_labels_layer) is None
    assert adapter.layer_bindings.get_binding(removed_shapes_layer) is None
    assert adapter.layer_bindings.get_binding(kept_layer) is not None
    assert adapter.layer_bindings.get_binding(external_layer) is None


def test_viewer_adapter_ignores_unregistered_image_layer_even_with_legacy_metadata(sdata_blobs) -> None:
    image_layer = make_image_layer(
        name="raw image",
        metadata={"sdata": sdata_blobs, "name": "blobs_image", "_current_cs": "global"},
    )
    viewer = DummyViewer([image_layer])
    adapter = ViewerAdapter(viewer)

    loaded_layers = adapter.get_loaded_image_layers(sdata_blobs, "blobs_image")

    assert loaded_layers == []
    assert adapter.layer_bindings.get_binding(image_layer) is None


def test_viewer_adapter_ensure_points_layer_from_selection_creates_registered_layer() -> None:
    sdata = SimpleNamespace()
    identity = make_points_identity(sdata)
    selection = make_points_selection(["AAMP", "AAMP"], selected_values=("AAMP",))
    viewer = DummyViewer(reset_camera_on_layer_change=True)
    adapter = ViewerAdapter(viewer)

    result = adapter._ensure_points_layer_from_selection(identity, selection=selection)

    assert result.created is True
    assert str(result.layer.face_color_mode) == "cycle"
    assert result.color_mode == "categorical"
    assert result.categorical_coloring_disabled is False
    assert result.selected_value_count == 1
    assert result.categorical_limit == 102
    assert result.layer in viewer.layers
    assert result.layer.name == "transcripts: gene=AAMP"
    np.testing.assert_array_equal(result.layer.data, selection.coordinates)
    assert result.layer.features.equals(selection.features)
    assert np.all(result.layer.size == 5.0)
    assert result.layer.current_size == 5.0
    assert result.layer.opacity == 0.8
    assert result.layer.current_symbol.value == "disc"
    assert all(symbol.value == "disc" for symbol in result.layer.symbol)
    assert np.all(result.layer.border_width == 0)
    expected_colors = np.asarray([to_rgba(default_categorical_colors(1)[0])] * selection.loaded_count)
    assert np.allclose(result.layer.face_color, expected_colors)
    assert np.allclose(result.layer.border_color, result.layer.face_color)
    binding = adapter.layer_bindings.get_binding(result.layer)
    assert isinstance(binding, PointsLayerBinding)
    assert binding.element_name == "transcripts"
    assert binding.element_type == "points"
    assert binding.coordinate_system == "global"
    assert binding.index_column == "gene"
    assert binding.sdata_id == id(sdata)
    assert viewer.camera.center == (0.0, 0.0)
    assert viewer.camera.zoom == 1.0


def test_viewer_adapter_ensure_points_layer_from_selection_uses_categorical_color_for_single_value() -> None:
    sdata = SimpleNamespace()
    identity = make_points_identity(sdata)
    selection = make_points_selection(["AAMP", "AAMP"], selected_values=("AAMP",))
    adapter = ViewerAdapter(DummyViewer())

    result = adapter._ensure_points_layer_from_selection(identity, selection=selection)

    assert str(result.layer.face_color_mode) == "cycle"
    assert result.color_mode == "categorical"
    assert np.allclose(
        result.layer.face_color,
        np.asarray([to_rgba(default_categorical_colors(1)[0])] * selection.loaded_count),
    )
    assert np.allclose(result.layer.border_color, result.layer.face_color)


def test_viewer_adapter_ensure_points_layer_from_selection_handles_empty_selection() -> None:
    sdata = SimpleNamespace()
    identity = make_points_identity(sdata)
    selection = PointsValueSelection(
        coordinates=np.empty((0, 2), dtype="float32"),
        features=pd.DataFrame(
            {
                "gene": pd.Categorical([], categories=[]),
                "value_id": pd.Series([], dtype="uint32"),
            }
        ),
        index_column="gene",
        selected_values=(),
        selected_value_ids=(),
        selection_mode="values",
        total_count=0,
        render_point_budget=100_000,
        is_sampled=False,
        warning=None,
    )
    adapter = ViewerAdapter(DummyViewer())

    result = adapter._ensure_points_layer_from_selection(identity, selection=selection)

    assert result.layer.name == "transcripts: no gene values"
    assert result.color_mode == "solid"
    assert result.categorical_coloring_disabled is False
    assert result.selected_value_count == 0


def test_viewer_adapter_ensure_points_layer_from_selection_applies_points_affine() -> None:
    points = PointsModel.parse(
        pd.DataFrame({"x": [10.0], "y": [5.0], "gene": ["AAMP"]}),
        transformations={
            "translated": Affine(
                [[1.0, 0.0, 10.0], [0.0, 1.0, 20.0], [0.0, 0.0, 1.0]],
                input_axes=("x", "y"),
                output_axes=("x", "y"),
            )
        },
    )
    sdata = SimpleNamespace(points={"transcripts": points})
    identity = make_points_identity(sdata, coordinate_system="translated")
    selection = make_points_selection(["AAMP"], selected_values=("AAMP",))
    adapter = ViewerAdapter(DummyViewer())

    result = adapter._ensure_points_layer_from_selection(identity, selection=selection)

    np.testing.assert_allclose(
        result.layer.affine.affine_matrix,
        np.asarray([[1.0, 0.0, 20.0], [0.0, 1.0, 10.0], [0.0, 0.0, 1.0]]),
    )


def test_viewer_adapter_ensure_points_layer_from_selection_applies_point_size_control_to_all_points() -> None:
    sdata = SimpleNamespace()
    identity = make_points_identity(sdata)
    selection = make_points_selection(["AAMP", "AAMP"], selected_values=("AAMP",))
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)

    result = adapter._ensure_points_layer_from_selection(identity, selection=selection)
    result.layer.current_size = 12.0

    assert np.all(result.layer.size == 12.0)


def test_viewer_adapter_ensure_points_layer_from_selection_applies_symbol_control_to_all_points() -> None:
    sdata = SimpleNamespace()
    identity = make_points_identity(sdata)
    selection = make_points_selection(["AAMP", "AAMP"], selected_values=("AAMP",))
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)

    result = adapter._ensure_points_layer_from_selection(identity, selection=selection)
    result.layer.current_symbol = "square"

    assert all(symbol.value == "square" for symbol in result.layer.symbol)


def test_viewer_adapter_ensure_points_layer_from_selection_applies_solid_face_color_control_to_all_points() -> None:
    sdata = SimpleNamespace()
    identity = make_points_identity(sdata)
    selected_values = tuple(f"gene_{index}" for index in range(103))
    selection = make_points_selection(list(selected_values), selected_values=selected_values)
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)

    result = adapter._ensure_points_layer_from_selection(identity, selection=selection)
    result.layer.current_face_color = "red"

    assert np.allclose(result.layer.face_color, np.asarray([to_rgba("red")] * selection.loaded_count))
    assert np.allclose(result.layer.border_color, result.layer.face_color)


def test_viewer_adapter_ensure_points_layer_from_selection_applies_single_categorical_face_color_control() -> None:
    sdata = SimpleNamespace()
    identity = make_points_identity(sdata)
    selection = make_points_selection(["AAMP", "AAMP"], selected_values=("AAMP",))
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)

    result = adapter._ensure_points_layer_from_selection(identity, selection=selection)
    result.layer.current_face_color = "red"

    assert np.allclose(result.layer.face_color, np.asarray([to_rgba("red")] * selection.loaded_count))
    assert np.allclose(result.layer.border_color, result.layer.face_color)


def test_viewer_adapter_ensure_points_layer_from_selection_keeps_multi_categorical_palette_owned_by_spatiato() -> None:
    sdata = SimpleNamespace()
    identity = make_points_identity(sdata)
    selection = make_points_selection(["AAMP", "AXL"], selected_values=("AAMP", "AXL"))
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)

    result = adapter._ensure_points_layer_from_selection(identity, selection=selection)
    face_color = result.layer.face_color.copy()
    border_color = result.layer.border_color.copy()
    result.layer.current_face_color = "red"

    np.testing.assert_array_equal(result.layer.face_color, face_color)
    np.testing.assert_array_equal(result.layer.border_color, border_color)


def test_viewer_adapter_ensure_points_layer_from_selection_replaces_existing_layer_preserving_visibility() -> None:
    sdata = SimpleNamespace()
    identity = make_points_identity(sdata)
    first_selection = make_points_selection(["AAMP"], selected_values=("AAMP",))
    second_selection = make_points_selection(["AXL"], selected_values=("AXL",))
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)
    first = adapter._ensure_points_layer_from_selection(identity, selection=first_selection)
    first.layer.visible = False

    second = adapter._ensure_points_layer_from_selection(identity, selection=second_selection)

    assert first.layer is not second.layer
    assert first.created is True
    assert second.created is False
    assert first.layer not in viewer.layers
    assert second.layer.visible is False
    assert len(viewer.layers) == 1
    assert second.layer.name == "transcripts: gene=AXL"
    np.testing.assert_array_equal(second.layer.data, second_selection.coordinates)
    assert second.layer.features.equals(second_selection.features)
    assert adapter.layer_bindings.get_binding(first.layer) is None
    assert isinstance(adapter.layer_bindings.get_binding(second.layer), PointsLayerBinding)


def test_viewer_adapter_ensure_points_layer_from_selection_replaces_stale_point_layer_object() -> None:
    sdata = SimpleNamespace()
    identity = make_points_identity(sdata)
    first_selection = make_points_selection(["A", "B", "C", "D", "E"], selected_values=("A", "B", "C", "D", "E"))
    second_selection = make_points_selection(["A", "A"], selected_values=("A",))
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)
    first = adapter._ensure_points_layer_from_selection(identity, selection=first_selection)
    first.layer._slicing_state._indices_view = np.arange(first_selection.loaded_count, dtype=int)
    first.layer.selected_data = {first_selection.loaded_count - 1}

    second = adapter._ensure_points_layer_from_selection(identity, selection=second_selection)

    assert second.layer is not first.layer
    assert first.layer not in viewer.layers
    assert adapter.layer_bindings.get_binding(first.layer) is None
    np.testing.assert_array_equal(second.layer._view_data, second_selection.coordinates)


def test_viewer_adapter_ensure_points_layer_from_selection_preserves_camera_when_replacing_layer() -> None:
    sdata = SimpleNamespace()
    identity = make_points_identity(sdata)
    first_selection = make_points_selection(["AAMP"], selected_values=("AAMP",))
    second_selection = make_points_selection(["AXL"], selected_values=("AXL",))
    viewer = DummyViewer(reset_camera_on_layer_change=True)
    adapter = ViewerAdapter(viewer)
    adapter._ensure_points_layer_from_selection(identity, selection=first_selection)
    viewer.camera.center = (123.0, 456.0)
    viewer.camera.zoom = 7.5

    adapter._ensure_points_layer_from_selection(identity, selection=second_selection)

    assert viewer.camera.center == (123.0, 456.0)
    assert viewer.camera.zoom == 7.5


def test_viewer_adapter_ensure_points_layer_from_selection_preserves_point_size_when_replacing_layer() -> None:
    sdata = SimpleNamespace()
    identity = make_points_identity(sdata)
    first_selection = make_points_selection(["AAMP"], selected_values=("AAMP",))
    second_selection = make_points_selection(["AXL", "AXL"], selected_values=("AXL",))
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)
    first = adapter._ensure_points_layer_from_selection(identity, selection=first_selection)
    first.layer.current_size = 12.0

    second = adapter._ensure_points_layer_from_selection(identity, selection=second_selection)

    assert second.layer.current_size == 12.0
    assert np.all(second.layer.size == 12.0)


def test_viewer_adapter_ensure_points_layer_from_selection_preserves_symbol_when_replacing_layer() -> None:
    sdata = SimpleNamespace()
    identity = make_points_identity(sdata)
    first_selection = make_points_selection(["AAMP"], selected_values=("AAMP",))
    second_selection = make_points_selection(["AXL", "AXL"], selected_values=("AXL",))
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)
    first = adapter._ensure_points_layer_from_selection(identity, selection=first_selection)
    first.layer.current_symbol = "square"

    second = adapter._ensure_points_layer_from_selection(identity, selection=second_selection)

    assert second.layer.current_symbol.value == "square"
    assert all(symbol.value == "square" for symbol in second.layer.symbol)


def test_viewer_adapter_ensure_points_layer_from_selection_preserves_solid_face_color_when_replacing_layer() -> None:
    sdata = SimpleNamespace()
    identity = make_points_identity(sdata)
    first_values = tuple(f"gene_{index}" for index in range(103))
    second_values = tuple(f"target_{index}" for index in range(103))
    first_selection = make_points_selection(list(first_values), selected_values=first_values)
    second_selection = make_points_selection(list(second_values), selected_values=second_values)
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)
    first = adapter._ensure_points_layer_from_selection(identity, selection=first_selection)
    first.layer.current_face_color = "red"

    second = adapter._ensure_points_layer_from_selection(identity, selection=second_selection)

    assert second.layer.current_face_color == "red"
    assert np.allclose(second.layer.face_color, np.asarray([to_rgba("red")] * second_selection.loaded_count))
    assert np.allclose(second.layer.border_color, second.layer.face_color)


def test_viewer_adapter_ensure_points_layer_from_selection_preserves_overridden_categorical_face_color() -> None:
    sdata = SimpleNamespace()
    identity = make_points_identity(sdata)
    first_selection = make_points_selection(["AAMP", "AAMP"], selected_values=("AAMP",))
    second_selection = make_points_selection(["AXL", "AXL"], selected_values=("AXL",))
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)
    first = adapter._ensure_points_layer_from_selection(identity, selection=first_selection)
    first.layer.current_face_color = "red"

    second = adapter._ensure_points_layer_from_selection(identity, selection=second_selection)

    assert second.layer.current_face_color == "red"
    assert np.allclose(second.layer.face_color, np.asarray([to_rgba("red")] * second_selection.loaded_count))
    assert np.allclose(second.layer.border_color, second.layer.face_color)


def test_viewer_adapter_ensure_points_layer_from_selection_does_not_preserve_single_color_into_multi_category() -> None:
    sdata = SimpleNamespace()
    identity = make_points_identity(sdata)
    first_selection = make_points_selection(["AAMP", "AAMP"], selected_values=("AAMP",))
    second_selection = make_points_selection(["AAMP", "AXL"], selected_values=("AAMP", "AXL"))
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)
    first = adapter._ensure_points_layer_from_selection(identity, selection=first_selection)
    first.layer.current_face_color = "red"

    second = adapter._ensure_points_layer_from_selection(identity, selection=second_selection)

    expected_colors = np.asarray([to_rgba(color) for color in default_categorical_colors(2)], dtype=np.float32)
    assert np.allclose(second.layer.face_color, expected_colors)
    assert np.allclose(second.layer.border_color, second.layer.face_color)


def test_viewer_adapter_ensure_points_layer_from_selection_ignores_unregistered_same_name_layer() -> None:
    sdata = SimpleNamespace()
    identity = make_points_identity(sdata)
    selection = make_points_selection(["AAMP"], selected_values=("AAMP",))
    external_layer = Points(selection.coordinates, name="transcripts: gene=AAMP")
    viewer = DummyViewer([external_layer])
    adapter = ViewerAdapter(viewer)

    result = adapter._ensure_points_layer_from_selection(identity, selection=selection)

    assert result.layer is not external_layer
    assert list(viewer.layers) == [external_layer, result.layer]
    assert adapter.layer_bindings.get_binding(external_layer) is None
    assert isinstance(adapter.layer_bindings.get_binding(result.layer), PointsLayerBinding)


def test_viewer_adapter_ensure_points_layer_from_selection_rejects_identity_selection_mismatch() -> None:
    sdata = SimpleNamespace()
    identity = make_points_identity(sdata, index_column="target")
    selection = make_points_selection(["AAMP"], selected_values=("AAMP",))
    adapter = ViewerAdapter(DummyViewer())

    with pytest.raises(ValueError, match="index_column"):
        adapter._ensure_points_layer_from_selection(identity, selection=selection)


def test_viewer_adapter_ensure_points_layer_from_selection_uses_all_values_name() -> None:
    sdata = SimpleNamespace()
    identity = make_points_identity(sdata)
    selection = make_points_selection(["AAMP", "AXL"], selected_values=("AAMP", "AXL"), selection_mode="all")
    adapter = ViewerAdapter(DummyViewer())

    result = adapter._ensure_points_layer_from_selection(identity, selection=selection)

    assert result.layer.name == "transcripts: all gene values"


def test_viewer_adapter_ensure_points_layer_from_selection_restores_name_after_duplicate_add() -> None:
    sdata = SimpleNamespace()
    identity = make_points_identity(sdata)
    selection = make_points_selection(["AAMP", "AXL"], selected_values=("AAMP", "AXL"), selection_mode="all")
    viewer = UniqueNameDummyViewer()
    adapter = ViewerAdapter(viewer)
    first = adapter._ensure_points_layer_from_selection(identity, selection=selection)

    second = adapter._ensure_points_layer_from_selection(identity, selection=selection)

    assert first.layer not in viewer.layers
    assert second.layer in viewer.layers
    assert second.layer.name == "transcripts: all gene values"
    assert len(viewer.layers) == 1


def test_viewer_adapter_ensure_points_layer_from_selection_uses_categorical_color_for_single_all_value() -> None:
    sdata = SimpleNamespace()
    identity = make_points_identity(sdata)
    selection = make_points_selection(["AAMP", "AAMP"], selected_values=("AAMP",), selection_mode="all")
    adapter = ViewerAdapter(DummyViewer())

    result = adapter._ensure_points_layer_from_selection(identity, selection=selection)

    assert str(result.layer.face_color_mode) == "cycle"
    assert result.color_mode == "categorical"
    assert np.allclose(
        result.layer.face_color,
        np.asarray([to_rgba(default_categorical_colors(1)[0])] * selection.loaded_count),
    )


def test_viewer_adapter_ensure_points_layer_from_selection_uses_categorical_colors_for_small_selections() -> None:
    sdata = SimpleNamespace()
    identity = make_points_identity(sdata)
    selection = make_points_selection(["AAMP", "AXL"], selected_values=("AAMP", "AXL"))
    adapter = ViewerAdapter(DummyViewer())

    result = adapter._ensure_points_layer_from_selection(identity, selection=selection)

    assert str(result.layer.face_color_mode) == "cycle"
    assert result.color_mode == "categorical"
    assert result.categorical_coloring_disabled is False
    assert result.selected_value_count == 2
    assert result.categorical_limit == 102
    expected_colors = np.asarray([to_rgba(color) for color in default_categorical_colors(2)], dtype=np.float32)
    assert np.allclose(result.layer.face_color, expected_colors)


def test_viewer_adapter_ensure_points_layer_from_selection_uses_explicit_categorical_colors() -> None:
    sdata = SimpleNamespace()
    identity = make_points_identity(sdata)
    selection = make_points_selection(["AXL", "AAMP"], selected_values=("AAMP", "AXL"))
    adapter = ViewerAdapter(DummyViewer())

    result = adapter._ensure_points_layer_from_selection(
        identity,
        selection=selection,
        categorical_colors=("#ff0000", "#00ff00"),
    )

    expected_colors = np.asarray([to_rgba("#00ff00"), to_rgba("#ff0000")], dtype=np.float32)
    assert np.allclose(result.layer.face_color, expected_colors)
    assert np.allclose(result.layer.border_color, result.layer.face_color)


def test_viewer_adapter_ensure_points_layer_from_selection_uses_solid_color_above_categorical_limit() -> None:
    sdata = SimpleNamespace()
    identity = make_points_identity(sdata)
    selected_values = tuple(f"gene_{index}" for index in range(103))
    selection = make_points_selection(list(selected_values), selected_values=selected_values)
    adapter = ViewerAdapter(DummyViewer())

    result = adapter._ensure_points_layer_from_selection(identity, selection=selection)

    assert str(result.layer.face_color_mode) == "direct"
    assert result.color_mode == "solid"
    assert result.categorical_coloring_disabled is True
    assert result.selected_value_count == 103
    assert result.categorical_limit == 102
    assert np.allclose(
        result.layer.face_color,
        np.asarray([to_rgba(POINTS_SELECTION_SOLID_COLOR)] * selection.loaded_count),
    )


def test_viewer_adapter_ensure_points_layer_from_selection_keeps_sampling_warning_on_selection() -> None:
    sdata = SimpleNamespace()
    identity = make_points_identity(sdata)
    selection = make_points_selection(
        ["AAMP", "AXL"],
        selected_values=("AAMP", "AXL"),
        total_count=10,
        render_point_budget=2,
        is_sampled=True,
        warning="Showing 2 of 10 selected points.",
    )
    adapter = ViewerAdapter(DummyViewer())

    result = adapter._ensure_points_layer_from_selection(identity, selection=selection)

    assert result.color_mode == "categorical"
    assert result.categorical_coloring_disabled is False
    assert selection.warning == "Showing 2 of 10 selected points."


def test_viewer_adapter_ensure_labels_loaded_adds_layer_and_registers_binding(sdata_blobs) -> None:
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)
    labels_events: list[str] = []

    adapter.primary_labels_layers_changed.connect(lambda: labels_events.append("changed"))

    result = adapter.ensure_labels_loaded(sdata_blobs, "blobs_labels", "global")
    layer = result.layer

    assert result.created is True
    assert result.value_kind is None
    assert result.palette_source is None
    assert result.coercion_applied is False
    assert layer in viewer.layers
    assert layer.name == "blobs_labels"
    assert layer.affine is not None
    binding = adapter.layer_bindings.get_binding(layer)
    assert binding is not None
    assert isinstance(binding, LabelsLayerBinding)
    assert binding.element_name == "blobs_labels"
    assert binding.element_type == "labels"
    assert binding.coordinate_system == "global"
    assert binding.labels_role == "primary"
    assert "element_name" not in layer.metadata
    assert "element_type" not in layer.metadata
    assert "coordinate_system" not in layer.metadata
    assert "labels_role" not in layer.metadata
    assert "sdata" not in layer.metadata
    assert "_current_cs" not in layer.metadata
    assert labels_events == ["changed"]


def test_viewer_adapter_primary_labels_signal_ignores_styled_bindings(sdata_blobs) -> None:
    primary_layer = make_labels_layer(sdata=sdata_blobs, labels_name="blobs_labels")
    styled_layer = make_labels_layer(sdata=sdata_blobs, labels_name="blobs_labels")
    styled_layer.name = "blobs_labels[cell_type]"
    viewer = DummyViewer([primary_layer, styled_layer])
    adapter = ViewerAdapter(viewer)
    labels_events: list[str] = []
    style_spec = TableColorSourceSpec(
        table_name="table",
        source_kind="obs_column",
        value_key="cell_type",
        value_kind="categorical",
    )

    adapter.primary_labels_layers_changed.connect(lambda: labels_events.append("changed"))

    adapter.register_labels_layer(
        styled_layer,
        sdata=sdata_blobs,
        labels_name="blobs_labels",
        coordinate_system="global",
        labels_role="styled",
        style_spec=style_spec,
    )
    assert labels_events == []

    adapter.register_labels_layer(
        primary_layer,
        sdata=sdata_blobs,
        labels_name="blobs_labels",
        coordinate_system="global",
    )
    assert labels_events == ["changed"]

    viewer.layers.remove(styled_layer)
    assert labels_events == ["changed"]

    viewer.layers.remove(primary_layer)
    assert labels_events == ["changed", "changed"]


def test_viewer_adapter_image_signal_emits_for_loaded_overlay_binding(sdata_blobs) -> None:
    layer = make_image_layer(name="blobs_image[0]")
    viewer = DummyViewer([layer])
    adapter = ViewerAdapter(viewer)
    emitted_channel_names: list[str | None] = []

    def record_overlay_change() -> None:
        binding = adapter.layer_bindings.get_binding(layer)
        emitted_channel_names.append(binding.channel_name if isinstance(binding, ImageLayerBinding) else None)

    adapter.image_layers_changed.connect(record_overlay_change)

    adapter.register_image_layer(
        layer,
        sdata=sdata_blobs,
        image_name="blobs_image",
        coordinate_system="global",
        image_display_mode="overlay",
        channel_index=0,
        channel_name="0",
    )

    assert emitted_channel_names == ["0"]


def test_viewer_adapter_image_signal_emits_for_stack_and_ignores_unusable_overlay(
    sdata_blobs,
) -> None:
    stack_layer = make_image_layer(name="blobs_image")
    overlay_without_channel = make_image_layer(name="blobs_image")
    viewer = DummyViewer([stack_layer, overlay_without_channel])
    adapter = ViewerAdapter(viewer)
    overlay_events: list[str] = []

    adapter.image_layers_changed.connect(lambda: overlay_events.append("changed"))

    adapter.register_image_layer(
        stack_layer,
        sdata=sdata_blobs,
        image_name="blobs_image",
        coordinate_system="global",
        image_display_mode="stack",
    )
    adapter.register_image_layer(
        overlay_without_channel,
        sdata=sdata_blobs,
        image_name="blobs_image",
        coordinate_system="global",
        image_display_mode="overlay",
    )

    assert overlay_events == ["changed"]


def test_viewer_adapter_image_signal_emits_when_prebound_layer_is_inserted(sdata_blobs) -> None:
    layer = make_image_layer(name="blobs_image[0]")
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)
    overlay_events: list[str] = []

    adapter.image_layers_changed.connect(lambda: overlay_events.append("changed"))

    adapter.register_image_layer(
        layer,
        sdata=sdata_blobs,
        image_name="blobs_image",
        coordinate_system="global",
        image_display_mode="overlay",
        channel_index=0,
        channel_name="0",
    )
    assert overlay_events == []

    viewer.add_layer(layer)

    assert overlay_events == ["changed"]


def test_viewer_adapter_image_signal_emits_after_registered_overlay_removal(sdata_blobs) -> None:
    layer = make_image_layer(name="blobs_image[0]")
    viewer = DummyViewer([layer])
    adapter = ViewerAdapter(viewer)
    bindings_seen_on_signal: list[ImageLayerBinding | None] = []

    def record_overlay_change() -> None:
        binding = adapter.layer_bindings.get_binding(layer)
        bindings_seen_on_signal.append(binding if isinstance(binding, ImageLayerBinding) else None)

    adapter.image_layers_changed.connect(record_overlay_change)

    adapter.register_image_layer(
        layer,
        sdata=sdata_blobs,
        image_name="blobs_image",
        coordinate_system="global",
        image_display_mode="overlay",
        channel_index=0,
        channel_name="0",
    )
    assert isinstance(bindings_seen_on_signal[-1], ImageLayerBinding)

    bindings_seen_on_signal.clear()
    viewer.layers.remove(layer)

    assert bindings_seen_on_signal == [None]
    assert adapter.layer_bindings.get_binding(layer) is None


def test_viewer_adapter_image_signal_ignores_external_layer_removal() -> None:
    layer = make_image_layer(name="external")
    viewer = DummyViewer([layer])
    adapter = ViewerAdapter(viewer)
    overlay_events: list[str] = []

    adapter.image_layers_changed.connect(lambda: overlay_events.append("changed"))

    viewer.layers.remove(layer)

    assert overlay_events == []


def test_viewer_adapter_image_signal_emits_when_image_layers_are_reordered(sdata_blobs) -> None:
    layer = make_image_layer(name="blobs_image")
    viewer = DummyViewer([layer])
    adapter = ViewerAdapter(viewer)
    adapter.register_image_layer(
        layer,
        sdata=sdata_blobs,
        image_name="blobs_image",
        coordinate_system="global",
        image_display_mode="stack",
    )
    image_events: list[str] = []
    adapter.image_layers_changed.connect(lambda: image_events.append("changed"))

    viewer.layers.events.reordered.emit()

    assert image_events == ["changed"]


def test_viewer_adapter_remove_image_overlay_channel_fallback_unregisters_and_emits_lifecycle_without_removal_event(
    sdata_blobs,
) -> None:
    layer = make_image_layer(name="blobs_image[0]")
    viewer = SilentRemovalViewer([layer])
    adapter = ViewerAdapter(viewer)
    adapter.register_image_layer(
        layer,
        sdata=sdata_blobs,
        image_name="blobs_image",
        coordinate_system="global",
        image_display_mode="overlay",
        channel_index=0,
        channel_name="0",
    )
    overlay_events: list[str] = []
    adapter.image_layers_changed.connect(lambda: overlay_events.append("changed"))

    removed_layer = adapter.remove_image_overlay_channel(
        sdata_blobs,
        "blobs_image",
        "global",
        channel_index=0,
    )

    assert removed_layer is layer
    assert layer not in viewer.layers
    assert overlay_events == ["changed"]
    assert adapter.layer_bindings.get_binding(layer) is None


def test_viewer_adapter_ensure_styled_labels_loaded_creates_registered_overlay_with_stored_palette(sdata_blobs) -> None:
    table = sdata_blobs["table"]
    region_rows = table.obs.loc[table.obs["region"] == "blobs_labels"].copy()
    table.obs["cell_type"] = pd.Categorical(
        np.where(region_rows["instance_id"].to_numpy() % 2 == 0, "even", "odd"),
        categories=["odd", "even"],
    )
    table.uns["cell_type_colors"] = ["#ff0000", "#00ff00"]

    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)
    style_spec = TableColorSourceSpec(
        table_name="table",
        source_kind="obs_column",
        value_key="cell_type",
        value_kind="categorical",
    )

    result = adapter.ensure_styled_labels_loaded(sdata_blobs, "blobs_labels", "global", style_spec)

    assert result.created is True
    assert result.value_kind == "categorical"
    assert result.palette_source == "stored"
    assert result.coercion_applied is False
    assert result.layer in viewer.layers
    assert result.layer.name == "blobs_labels[obs:cell_type]"
    assert isinstance(result.layer.colormap, CompactLabelColormap)
    binding = adapter.layer_bindings.get_binding(result.layer)
    assert isinstance(binding, LabelsLayerBinding)
    assert binding.labels_role == "styled"
    assert binding.style_spec == style_spec
    assert "style_table_name" not in result.layer.metadata
    assert "style_source_kind" not in result.layer.metadata
    assert "style_value_key" not in result.layer.metadata
    assert "style_value_kind" not in result.layer.metadata
    assert list(result.layer.features.columns) == ["index", "instance_id", "cell_type"]

    odd_instance = int(region_rows.loc[region_rows["instance_id"] % 2 == 1, "instance_id"].iloc[0])
    even_instance = int(region_rows.loc[region_rows["instance_id"] % 2 == 0, "instance_id"].iloc[0])
    odd_feature_row = result.layer.features.iloc[result.layer._label_index[odd_instance]]
    assert odd_feature_row["index"] == odd_instance
    assert odd_feature_row["instance_id"] == odd_instance
    assert np.allclose(result.layer.colormap.map(odd_instance), np.asarray(to_rgba("#ff0000"), dtype=np.float32))
    assert np.allclose(result.layer.colormap.map(even_instance), np.asarray(to_rgba("#00ff00"), dtype=np.float32))


def test_viewer_adapter_ensure_styled_labels_loaded_reuses_matching_variant(sdata_blobs) -> None:
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)
    style_spec = TableColorSourceSpec(
        table_name="table",
        source_kind="x_var",
        value_key="channel_0_sum",
        value_kind="continuous",
    )

    first = adapter.ensure_styled_labels_loaded(sdata_blobs, "blobs_labels", "global", style_spec)
    second = adapter.ensure_styled_labels_loaded(sdata_blobs, "blobs_labels", "global", style_spec)

    assert first.layer is second.layer
    assert first.created is True
    assert second.created is False
    assert second.value_kind == "continuous"
    assert len(viewer.layers) == 1


def test_viewer_adapter_ensure_styled_labels_loaded_force_syncs_after_recolor(
    monkeypatch: pytest.MonkeyPatch,
    sdata_blobs,
) -> None:
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)
    sync_calls: list[Labels] = []
    style_spec = TableColorSourceSpec(
        table_name="table",
        source_kind="x_var",
        value_key="channel_0_sum",
        value_kind="continuous",
    )

    monkeypatch.setattr(adapter, "sync_labels_display_after_colormap_change", sync_calls.append)

    first = adapter.ensure_styled_labels_loaded(sdata_blobs, "blobs_labels", "global", style_spec)
    second = adapter.ensure_styled_labels_loaded(sdata_blobs, "blobs_labels", "global", style_spec)

    assert first.layer is second.layer
    assert sync_calls == [first.layer, first.layer]


def test_viewer_adapter_ensure_styled_labels_loaded_updates_reused_variant_from_current_table(
    sdata_blobs,
) -> None:
    table = sdata_blobs["table"]
    region_rows = table.obs.loc[table.obs["region"] == "blobs_labels"]
    odd_instance = int(region_rows.loc[region_rows["instance_id"] % 2 == 1, "instance_id"].iloc[0])
    even_instance = int(region_rows.loc[region_rows["instance_id"] % 2 == 0, "instance_id"].iloc[0])
    table.obs["cell_type"] = pd.Categorical(
        np.where(table.obs["instance_id"].to_numpy() % 2 == 0, "even", "odd"),
        categories=["odd", "even"],
    )
    table.uns["cell_type_colors"] = ["#ff0000", "#00ff00"]

    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)
    style_spec = TableColorSourceSpec(
        table_name="table",
        source_kind="obs_column",
        value_key="cell_type",
        value_kind="categorical",
    )

    first = adapter.ensure_styled_labels_loaded(sdata_blobs, "blobs_labels", "global", style_spec)

    assert np.allclose(first.layer.colormap.map(odd_instance), np.asarray(to_rgba("#ff0000"), dtype=np.float32))
    # Change an instance that was originally "odd" to "even", then update
    # the palette so the second "ensure" call should recolor the same layer.
    table.obs.loc[table.obs["instance_id"] == odd_instance, "cell_type"] = "even"
    table.uns["cell_type_colors"] = ["#0000ff", "#ffff00"]

    second = adapter.ensure_styled_labels_loaded(sdata_blobs, "blobs_labels", "global", style_spec)

    assert first.layer is second.layer
    assert first.created is True
    assert second.created is False
    assert len(viewer.layers) == 1
    assert np.allclose(second.layer.colormap.map(odd_instance), np.asarray(to_rgba("#ffff00"), dtype=np.float32))
    assert np.allclose(second.layer.colormap.map(even_instance), np.asarray(to_rgba("#ffff00"), dtype=np.float32))
    features = second.layer.features.set_index("index")
    assert features.loc[odd_instance, "cell_type"] == "even"


def test_viewer_adapter_ensure_styled_labels_loaded_creates_distinct_variants_for_different_style_specs(
    sdata_blobs,
) -> None:
    table = sdata_blobs["table"]
    table.obs["cell_type"] = pd.Categorical(["odd"] * table.n_obs)
    style_a = TableColorSourceSpec(
        table_name="table",
        source_kind="obs_column",
        value_key="cell_type",
        value_kind="categorical",
    )
    style_b = TableColorSourceSpec(
        table_name="table",
        source_kind="x_var",
        value_key="channel_0_sum",
        value_kind="continuous",
    )
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)

    first = adapter.ensure_styled_labels_loaded(sdata_blobs, "blobs_labels", "global", style_a)
    second = adapter.ensure_styled_labels_loaded(sdata_blobs, "blobs_labels", "global", style_b)

    assert first.layer is not second.layer
    assert len(viewer.layers) == 2
    assert adapter.get_loaded_styled_labels_layer(sdata_blobs, "blobs_labels", style_a, "global") is first.layer
    assert adapter.get_loaded_styled_labels_layer(sdata_blobs, "blobs_labels", style_b, "global") is second.layer


def test_viewer_adapter_ensure_styled_labels_loaded_invalid_palette_falls_back(sdata_blobs) -> None:
    table = sdata_blobs["table"]
    table.obs["cell_type"] = pd.Categorical(["odd"] * table.n_obs + [])
    table.uns["cell_type_colors"] = ["#ff0000", "#00ff00"]
    table.obs["cell_type"] = pd.Categorical(["odd"] * table.n_obs, categories=["odd"])

    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)
    style_spec = TableColorSourceSpec(
        table_name="table",
        source_kind="obs_column",
        value_key="cell_type",
        value_kind="categorical",
    )

    result = adapter.ensure_styled_labels_loaded(sdata_blobs, "blobs_labels", "global", style_spec)

    assert result.palette_source == "default_invalid"
    assert result.coercion_applied is False


def test_viewer_adapter_ensure_styled_labels_loaded_colors_bool_obs_categorically(sdata_blobs) -> None:
    table = sdata_blobs["table"]
    table.obs["is_even"] = pd.Series(
        table.obs["instance_id"].to_numpy(dtype=np.int64) % 2 == 0,
        index=table.obs.index,
    )
    table.uns["is_even_colors"] = ["#ff0000", "#00ff00"]

    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)
    style_spec = TableColorSourceSpec(
        table_name="table",
        source_kind="obs_column",
        value_key="is_even",
        value_kind="categorical",
    )

    result = adapter.ensure_styled_labels_loaded(sdata_blobs, "blobs_labels", "global", style_spec)

    assert result.value_kind == "categorical"
    assert result.palette_source == "stored"
    assert result.coercion_applied is False
    features = result.layer.features.set_index("index")
    odd_instance = int(table.obs.loc[table.obs["instance_id"] % 2 == 1, "instance_id"].iloc[0])
    even_instance = int(table.obs.loc[table.obs["instance_id"] % 2 == 0, "instance_id"].iloc[0])
    assert features.loc[odd_instance, "is_even"] == np.False_
    assert features.loc[even_instance, "is_even"] == np.True_
    assert np.allclose(result.layer.colormap.map(odd_instance), np.asarray(to_rgba("#ff0000"), dtype=np.float32))
    assert np.allclose(result.layer.colormap.map(even_instance), np.asarray(to_rgba("#00ff00"), dtype=np.float32))


def test_viewer_adapter_ensure_styled_labels_loaded_colors_binary_int_obs_categorically(sdata_blobs) -> None:
    table = sdata_blobs["table"]
    table.obs["binary_state"] = pd.Series(
        table.obs["instance_id"].to_numpy(dtype=np.int64) % 2,
        index=table.obs.index,
        dtype="int64",
    )
    table.uns["binary_state_colors"] = ["#ff0000", "#00ff00"]

    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)
    style_spec = TableColorSourceSpec(
        table_name="table",
        source_kind="obs_column",
        value_key="binary_state",
        value_kind="categorical",
    )

    result = adapter.ensure_styled_labels_loaded(sdata_blobs, "blobs_labels", "global", style_spec)

    assert result.value_kind == "categorical"
    assert result.palette_source == "stored"
    assert result.coercion_applied is False
    features = result.layer.features.set_index("index")
    zero_instance = int(table.obs.loc[table.obs["binary_state"] == 0, "instance_id"].iloc[0])
    one_instance = int(table.obs.loc[table.obs["binary_state"] == 1, "instance_id"].iloc[0])
    assert int(features.loc[zero_instance, "binary_state"]) == 0
    assert int(features.loc[one_instance, "binary_state"]) == 1
    assert np.allclose(result.layer.colormap.map(zero_instance), np.asarray(to_rgba("#ff0000"), dtype=np.float32))
    assert np.allclose(result.layer.colormap.map(one_instance), np.asarray(to_rgba("#00ff00"), dtype=np.float32))


def test_viewer_adapter_ensure_styled_labels_loaded_colors_non_binary_int_obs_continuously(sdata_blobs) -> None:
    table = sdata_blobs["table"]
    table.obs["object_score"] = pd.Series(
        table.obs["instance_id"].to_numpy(dtype=np.int64),
        index=table.obs.index,
        dtype="int64",
    )

    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)
    style_spec = TableColorSourceSpec(
        table_name="table",
        source_kind="obs_column",
        value_key="object_score",
        value_kind="continuous",
    )

    result = adapter.ensure_styled_labels_loaded(sdata_blobs, "blobs_labels", "global", style_spec)

    assert result.value_kind == "continuous"
    assert result.palette_source is None
    assert result.coercion_applied is False
    features = result.layer.features.set_index("index")
    min_instance = int(table.obs["instance_id"].min())
    max_instance = int(table.obs["instance_id"].max())
    assert float(features.loc[min_instance, "object_score"]) == float(min_instance)
    assert float(features.loc[max_instance, "object_score"]) == float(max_instance)
    assert isinstance(result.layer.colormap, CompactLabelColormap)
    assert not np.allclose(result.layer.colormap.map(min_instance), result.layer.colormap.map(max_instance))


def test_viewer_adapter_ensure_styled_labels_loaded_colors_instance_key_as_labels(sdata_blobs) -> None:
    table = sdata_blobs["table"]
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)
    style_spec = TableColorSourceSpec(
        table_name="table",
        source_kind="obs_column",
        value_key="instance_id",
        value_kind="instance",
    )

    result = adapter.ensure_styled_labels_loaded(sdata_blobs, "blobs_labels", "global", style_spec)

    assert result.value_kind == "instance"
    assert result.palette_source is None
    assert result.coercion_applied is False
    assert isinstance(result.layer.colormap, CyclicLabelColormap)
    assert result.layer.name == "blobs_labels[obs:instance_id]"
    assert "style_value_kind" not in result.layer.metadata
    assert list(result.layer.features.columns) == ["index", "instance_id"]

    instance_ids = table.obs.loc[table.obs["region"] == "blobs_labels", "instance_id"].astype("int64").tolist()
    first_instance, second_instance = instance_ids[:2]
    mapped_colors = result.layer.colormap.map(np.asarray([first_instance, second_instance], dtype=np.int64))
    assert not np.allclose(mapped_colors[0], mapped_colors[1])
    assert np.allclose(result.layer.colormap.map(np.asarray([0], dtype=np.int64))[0], np.zeros(4))


def test_viewer_adapter_ensure_styled_labels_loaded_rejects_duplicate_instance_ids(sdata_blobs) -> None:
    table = sdata_blobs["table"]
    first_index, second_index = table.obs.index[:2]
    table.obs.loc[second_index, "instance_id"] = table.obs.loc[first_index, "instance_id"]
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)
    style_spec = TableColorSourceSpec(
        table_name="table",
        source_kind="obs_column",
        value_key="instance_id",
        value_kind="instance",
    )

    with pytest.raises(ValueError, match="contains duplicate values within that region"):
        adapter.ensure_styled_labels_loaded(sdata_blobs, "blobs_labels", "global", style_spec)


def test_viewer_adapter_ensure_styled_labels_loaded_rejects_duplicates_after_numeric_coercion(
    sdata_blobs,
) -> None:
    table = sdata_blobs["table"]
    table.obs["instance_id"] = table.obs["instance_id"].astype(str)
    first_index, second_index = table.obs.index[:2]
    table.obs.loc[first_index, "instance_id"] = "1"
    table.obs.loc[second_index, "instance_id"] = "01"
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)
    style_spec = TableColorSourceSpec(
        table_name="table",
        source_kind="obs_column",
        value_key="instance_id",
        value_kind="instance",
    )

    with pytest.raises(ValueError, match="after labels-specific numeric coercion"):
        adapter.ensure_styled_labels_loaded(sdata_blobs, "blobs_labels", "global", style_spec)


def test_viewer_adapter_ensure_styled_labels_loaded_coerces_string_obs_to_categorical(sdata_blobs) -> None:
    table = sdata_blobs["table"]
    table.obs["sample_type"] = pd.Series(
        np.where(table.obs["instance_id"].to_numpy() % 2 == 0, "even", "odd"),
        index=table.obs.index,
        dtype="object",
    )
    table.uns["sample_type_colors"] = ["#123456", "#654321"]

    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)
    style_spec = TableColorSourceSpec(
        table_name="table",
        source_kind="obs_column",
        value_key="sample_type",
        value_kind="categorical",
    )

    result = adapter.ensure_styled_labels_loaded(sdata_blobs, "blobs_labels", "global", style_spec)

    assert result.value_kind == "categorical"
    assert result.coercion_applied is True
    assert result.palette_source == "default_missing"
    assert isinstance(result.layer.colormap, CompactLabelColormap)


def test_viewer_adapter_ensure_styled_labels_loaded_warns_for_high_cardinality_string_obs(
    sdata_blobs,
    monkeypatch,
) -> None:
    table = sdata_blobs["table"]
    table.obs["cell_uuid"] = pd.Series(
        [f"cell-{index:04d}" for index in range(table.n_obs)],
        index=table.obs.index,
        dtype="object",
    )

    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)
    style_spec = TableColorSourceSpec(
        table_name="table",
        source_kind="obs_column",
        value_key="cell_uuid",
        value_kind="categorical",
    )

    warning_messages: list[str] = []

    class DummyLogger:
        def warning(self, message: str) -> None:
            warning_messages.append(message)

    monkeypatch.setattr(styling_module, "logger", DummyLogger())

    result = adapter.ensure_styled_labels_loaded(sdata_blobs, "blobs_labels", "global", style_spec)

    assert result.value_kind == "categorical"
    assert result.coercion_applied is True
    assert result.palette_source == "default_missing"
    assert isinstance(result.layer.colormap, CompactLabelColormap)
    assert len(warning_messages) == 1
    assert "exceeds the categorical viewer-coloring threshold" in warning_messages[0]
    assert "Spatiato will render it with the default categorical palette anyway" in warning_messages[0]


def test_viewer_adapter_ensure_styled_labels_loaded_names_x_var_layer_after_the_var(sdata_blobs) -> None:
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)
    style_spec = TableColorSourceSpec(
        table_name="table",
        source_kind="x_var",
        value_key="channel_1_sum",
        value_kind="continuous",
    )

    result = adapter.ensure_styled_labels_loaded(sdata_blobs, "blobs_labels", "global", style_spec)

    assert result.layer.name == "blobs_labels[X:channel_1_sum]"


def test_viewer_adapter_ensure_labels_loaded_reuses_matching_existing_layer(sdata_blobs) -> None:
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)

    first = adapter.ensure_labels_loaded(sdata_blobs, "blobs_labels", "global")
    second = adapter.ensure_labels_loaded(sdata_blobs, "blobs_labels", "global")

    assert first.layer is second.layer
    assert first.created is True
    assert second.created is False
    assert len(viewer.layers) == 1


def test_viewer_adapter_unregisters_binding_when_user_removes_labels_layer(sdata_blobs) -> None:
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)

    result = adapter.ensure_labels_loaded(sdata_blobs, "blobs_labels", "global")
    layer = result.layer

    viewer.layers.remove(layer)

    assert layer not in viewer.layers
    assert adapter.layer_bindings.get_binding(layer) is None


def test_viewer_adapter_ensure_labels_loaded_supports_multiscale_labels(sdata_blobs) -> None:
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)

    result = adapter.ensure_labels_loaded(sdata_blobs, "blobs_multiscale_labels", "global")
    layer = result.layer

    assert layer.multiscale is True
    assert len(layer.data) == 3
    assert layer in viewer.layers


def test_viewer_adapter_ensure_labels_loaded_rejects_unknown_coordinate_system(sdata_blobs) -> None:
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)

    try:
        adapter.ensure_labels_loaded(sdata_blobs, "blobs_labels", "not_a_coordinate_system")
    except ValueError as error:
        assert "Coordinate system `not_a_coordinate_system`" in str(error)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected ensure_labels_loaded to reject an unknown coordinate system.")


def test_viewer_adapter_ensure_shapes_loaded_adds_polygon_layer_and_registers_binding(sdata_blobs) -> None:
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)
    labels_events: list[str] = []

    adapter.primary_labels_layers_changed.connect(lambda: labels_events.append("changed"))

    result = adapter.ensure_shapes_loaded(sdata_blobs, "blobs_polygons", "global")
    layer = result.layer

    assert result.created is True
    assert result.value_kind is None
    assert result.palette_source is None
    assert result.coercion_applied is False
    assert result.skipped_geometry_count == 0
    assert result.shapes_rendering_mode == "shapes"
    assert layer in viewer.layers
    assert layer.name == "blobs_polygons"
    assert layer.shape_type == ["polygon"] * len(sdata_blobs.shapes["blobs_polygons"])
    expected_source_index_by_row = tuple(sdata_blobs.shapes["blobs_polygons"].index.to_list())
    expected_source_row_ids = tuple(range(len(expected_source_index_by_row)))
    assert list(layer.features.columns) == ["index"]
    assert tuple(layer.features["index"].to_list()) == expected_source_index_by_row
    binding = get_shapes_binding(adapter, layer)
    assert binding.element_name == "blobs_polygons"
    assert binding.element_type == "shapes"
    assert binding.coordinate_system == "global"
    assert binding.shapes_role == "primary"
    assert binding.style_spec is None
    assert binding.source_row_id_by_rendered_row == expected_source_row_ids
    assert binding.source_shapes_index_feature_name == "index"
    assert binding.skipped_geometry_count == 0
    assert "element_name" not in layer.metadata
    assert "element_type" not in layer.metadata
    assert "coordinate_system" not in layer.metadata
    assert "source_row_id_by_rendered_row" not in layer.metadata
    assert "source_shapes_index_feature_name" not in layer.metadata
    assert "skipped_geometry_count" not in layer.metadata
    assert labels_events == []


def test_viewer_adapter_shapes_replacement_restores_name_after_duplicate_add(sdata_blobs) -> None:
    viewer = UniqueNameDummyViewer()
    adapter = ViewerAdapter(viewer)
    previous = adapter.ensure_shapes_loaded(sdata_blobs, "blobs_polygons", "global")

    replacement = adapter.replace_primary_shapes_layer(
        sdata_blobs,
        "blobs_polygons",
        "global",
    )

    assert previous.layer not in viewer.layers
    assert replacement.layer in viewer.layers
    assert replacement.layer.name == "blobs_polygons"
    assert len(viewer.layers) == 1


def test_viewer_adapter_ensure_shapes_loaded_removes_layer_if_registration_fails(
    monkeypatch,
    sdata_blobs,
) -> None:
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)

    def fail_register(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("registration failed")

    monkeypatch.setattr(adapter, "register_shapes_layer", fail_register)

    with pytest.raises(RuntimeError, match="registration failed"):
        adapter.ensure_shapes_loaded(sdata_blobs, "blobs_polygons", "global")

    assert list(viewer.layers) == []
    assert adapter.layer_bindings.iter_bindings() == ()


def test_viewer_adapter_ensure_shapes_loaded_expands_multipolygons_with_source_mapping(sdata_blobs) -> None:
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)

    result = adapter.ensure_shapes_loaded(sdata_blobs, "blobs_multipolygons", "global")
    layer = result.layer

    expected_indices: list[object] = []
    expected_source_row_ids: list[int] = []
    for source_row_id, (source_index, geometry) in enumerate(
        sdata_blobs.shapes["blobs_multipolygons"].geometry.items()
    ):
        expected_indices.extend([source_index] * len(geometry.geoms))
        expected_source_row_ids.extend([source_row_id] * len(geometry.geoms))

    assert len(layer.data) == len(expected_indices)
    assert layer.shape_type == ["polygon"] * len(expected_indices)
    binding = get_shapes_binding(adapter, layer)
    assert binding.shapes_rendering_mode == "shapes"
    assert binding.source_row_id_by_rendered_row == tuple(expected_source_row_ids)
    assert tuple(layer.features["index"].to_list()) == tuple(expected_indices)


def test_viewer_adapter_ensure_shapes_loaded_renders_circles_as_points(sdata_blobs) -> None:
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)

    result = adapter.ensure_shapes_loaded(sdata_blobs, "blobs_circles", "global")
    layer = result.layer

    circles = sdata_blobs.shapes["blobs_circles"]
    expected_coordinates = np.column_stack(
        (
            circles.geometry.y.to_numpy(dtype=float),
            circles.geometry.x.to_numpy(dtype=float),
        )
    )

    assert isinstance(layer, Points)
    assert result.shapes_rendering_mode == "points"
    np.testing.assert_allclose(layer.data, expected_coordinates)
    expected_sizes = 2.0 * circles["radius"].to_numpy(dtype=float)
    np.testing.assert_allclose(layer.size, expected_sizes)
    assert layer.current_size == expected_radius_reference_size(expected_sizes)
    np.testing.assert_allclose(layer.face_color, np.asarray([to_rgba("#00FFFF")] * len(circles)))
    np.testing.assert_allclose(layer.border_color, np.asarray([to_rgba("#00FFFF")] * len(circles)))
    assert layer.opacity == 0.8
    binding = get_shapes_binding(adapter, layer)
    assert binding.element_type == "shapes"
    assert binding.shapes_role == "primary"
    assert binding.shapes_rendering_mode == "points"
    assert binding.source_row_id_by_rendered_row == range(len(circles))
    assert tuple(layer.features["index"].to_list()) == tuple(circles.index.to_list())
    status_value = layer.get_status(position=tuple(layer.data[0]))["value"]
    assert "index:" in status_value
    assert status_value.count("index:") == 1
    assert adapter.get_loaded_primary_shapes_layer(sdata_blobs, "blobs_circles", "global") is layer


def test_viewer_adapter_point_backed_primary_shapes_syncs_presentation_controls(sdata_blobs) -> None:
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)

    result = adapter.ensure_shapes_loaded(sdata_blobs, "blobs_circles", "global")
    layer = result.layer
    original_size = layer.size.copy()
    reference_size = expected_radius_reference_size(original_size)

    assert isinstance(layer, Points)
    layer.current_symbol = "square"
    layer.current_face_color = "red"
    layer.current_size = 99.0

    assert all(symbol.value == "square" for symbol in layer.symbol)
    np.testing.assert_allclose(layer.face_color, np.asarray([to_rgba("red")] * len(layer.data)))
    np.testing.assert_allclose(layer.border_color, layer.face_color)
    np.testing.assert_allclose(layer.border_width, np.zeros(len(layer.data)))
    np.testing.assert_allclose(layer.size, original_size * (99.0 / reference_size))

    layer.current_size = 0.0
    np.testing.assert_allclose(layer.size, original_size)


def test_viewer_adapter_point_backed_fixed_radius_shapes_scale_like_point_size_control() -> None:
    geodataframe = gpd.GeoDataFrame(
        {"radius": [5.0, 5.0]},
        geometry=[Point(10, 20), Point(30, 40)],
        index=["cell_1", "cell_2"],
    )
    sdata = make_shapes_sdata(geodataframe, shapes_name="cell_centroids")
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)

    result = adapter.ensure_shapes_loaded(sdata, "cell_centroids", "global")
    layer = result.layer

    assert isinstance(layer, Points)
    assert layer.current_size == 10.0
    np.testing.assert_allclose(layer.size, np.asarray([10.0, 10.0]))

    layer.current_size = 25.0

    np.testing.assert_allclose(layer.size, np.asarray([25.0, 25.0]))


def test_viewer_adapter_ensure_shapes_loaded_reuses_point_radius_layer(sdata_blobs) -> None:
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)

    first = adapter.ensure_shapes_loaded(sdata_blobs, "blobs_circles", "global")
    second = adapter.ensure_shapes_loaded(sdata_blobs, "blobs_circles", "global")

    assert isinstance(first.layer, Points)
    assert first.layer is second.layer
    assert first.created is True
    assert second.created is False
    assert second.shapes_rendering_mode == "points"
    assert len(viewer.layers) == 1
    assert get_shapes_binding(adapter, first.layer).shapes_rendering_mode == "points"


def test_prepare_napari_shapes_layer_inputs_does_not_use_iterrows(monkeypatch: pytest.MonkeyPatch) -> None:
    geodataframe = gpd.GeoDataFrame(
        {"radius": [1.0, 2.0]},
        geometry=[Point(1, 2), Point(3, 4)],
        index=["cell_1", "cell_2"],
    )

    def fail_iterrows(_self):
        raise AssertionError("_prepare_napari_shapes_layer_inputs should not call GeoDataFrame.iterrows().")

    monkeypatch.setattr(gpd.GeoDataFrame, "iterrows", fail_iterrows)

    result = _prepare_napari_shapes_layer_inputs(geodataframe)

    assert result.shape_types == ["ellipse", "ellipse"]
    assert result.source_row_id_by_rendered_row == (0, 1)
    assert result.features["index"].to_list() == ["cell_1", "cell_2"]
    assert result.skipped_geometry_count == 0


def test_prepare_napari_point_radius_shapes_layer_inputs_returns_points_arrays() -> None:
    geodataframe = gpd.GeoDataFrame(
        {"radius": [1.5, 2.0]},
        geometry=[Point(10, 20), Point(30, 40)],
        index=["cell_1", "cell_2"],
    )

    result = _prepare_napari_point_radius_shapes_layer_inputs(geodataframe)

    assert result is not None
    np.testing.assert_allclose(result.coordinates, np.asarray([[20.0, 10.0], [40.0, 30.0]]))
    np.testing.assert_allclose(result.sizes, np.asarray([3.0, 4.0]))
    assert result.features["index"].to_list() == ["cell_1", "cell_2"]
    assert result.source_shapes_index_feature_name == "index"
    assert result.source_row_id_by_rendered_row == range(2)
    assert result.skipped_geometry_count == 0


def test_prepare_napari_point_radius_shapes_layer_inputs_preserves_duplicate_indices() -> None:
    geodataframe = gpd.GeoDataFrame(
        {"radius": [1.0, 2.0]},
        geometry=[Point(1, 2), Point(3, 4)],
        index=["cell_1", "cell_1"],
    )

    result = _prepare_napari_point_radius_shapes_layer_inputs(geodataframe)

    assert result is not None
    assert result.features["index"].to_list() == ["cell_1", "cell_1"]
    assert result.source_row_id_by_rendered_row == range(2)


def test_prepare_napari_point_radius_shapes_layer_inputs_preserves_named_index() -> None:
    geodataframe = gpd.GeoDataFrame(
        {"radius": [1.0]},
        geometry=[Point(1, 2)],
        index=["cell_1"],
    )
    geodataframe.index.name = "cell_ID"

    result = _prepare_napari_point_radius_shapes_layer_inputs(geodataframe)

    assert result is not None
    assert result.source_shapes_index_feature_name == "cell_ID"
    assert list(result.features.columns) == ["cell_ID"]
    assert result.features["cell_ID"].to_list() == ["cell_1"]


def test_prepare_napari_point_radius_shapes_layer_inputs_returns_none_without_radius() -> None:
    geodataframe = gpd.GeoDataFrame(
        {"cell_type": ["T"]},
        geometry=[Point(1, 2)],
        index=["cell_1"],
    )

    assert _prepare_napari_point_radius_shapes_layer_inputs(geodataframe) is None


def test_prepare_napari_point_radius_shapes_layer_inputs_returns_none_for_invalid_radius() -> None:
    for radius in (0.0, -1.0, np.nan, np.inf, "not-a-radius"):
        geodataframe = gpd.GeoDataFrame(
            {"radius": [radius]},
            geometry=[Point(1, 2)],
            index=["cell_1"],
        )

        assert _prepare_napari_point_radius_shapes_layer_inputs(geodataframe) is None


def test_prepare_napari_point_radius_shapes_layer_inputs_returns_none_for_mixed_geometries() -> None:
    geodataframe = gpd.GeoDataFrame(
        {"radius": [1.0, 2.0]},
        geometry=[Point(1, 2), Polygon([(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)])],
        index=["cell_1", "cell_2"],
    )

    assert _prepare_napari_point_radius_shapes_layer_inputs(geodataframe) is None


def test_prepare_napari_point_radius_shapes_layer_inputs_returns_none_for_empty_geometry() -> None:
    for geometry in (Point(), None):
        geodataframe = gpd.GeoDataFrame(
            {"radius": [1.0]},
            geometry=[geometry],
            index=["cell_1"],
        )

        assert _prepare_napari_point_radius_shapes_layer_inputs(geodataframe) is None


def test_prepare_napari_point_radius_shapes_layer_inputs_does_not_use_iterrows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    geodataframe = gpd.GeoDataFrame(
        {"radius": [1.0, 2.0]},
        geometry=[Point(1, 2), Point(3, 4)],
        index=["cell_1", "cell_2"],
    )

    def fail_iterrows(_self):
        raise AssertionError("_prepare_napari_point_radius_shapes_layer_inputs should not call iterrows().")

    monkeypatch.setattr(gpd.GeoDataFrame, "iterrows", fail_iterrows)

    result = _prepare_napari_point_radius_shapes_layer_inputs(geodataframe)

    assert result is not None
    assert result.source_row_id_by_rendered_row == range(2)


def test_viewer_adapter_ensure_shapes_loaded_preserves_polygon_holes() -> None:
    polygon = Polygon(
        shell=[(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)],
        holes=[[(2, 2), (2, 5), (5, 5), (5, 2), (2, 2)]],
    )
    geodataframe = gpd.GeoDataFrame({"name": ["polygon_with_hole"], "geometry": [polygon]}, index=["donut"])
    sdata = make_shapes_sdata(geodataframe, shapes_name="donuts")
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)

    result = adapter.ensure_shapes_loaded(sdata, "donuts", "global")
    layer = result.layer
    labels = layer.to_labels(labels_shape=(12, 12))

    binding = get_shapes_binding(adapter, layer)
    assert binding.source_row_id_by_rendered_row == (0,)
    assert list(layer.features.columns) == ["index"]
    assert layer.features.iloc[0]["index"] == "donut"
    assert "index: donut" in layer.get_status(position=(1, 1))["value"]
    assert labels[1, 1] == 1
    assert labels[3, 3] == 0


def test_viewer_adapter_ensure_shapes_loaded_uses_named_geodataframe_index_in_features() -> None:
    polygon = Polygon([(0, 0), (4, 0), (4, 4), (0, 4), (0, 0)])
    geodataframe = gpd.GeoDataFrame(
        {"cell_id": ["column_value"], "name": ["boundary"]}, geometry=[polygon], index=["cell_1"]
    )
    geodataframe.index.name = "cell_id"
    sdata = make_shapes_sdata(geodataframe, shapes_name="cell_boundaries")
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)

    result = adapter.ensure_shapes_loaded(sdata, "cell_boundaries", "global")
    layer = result.layer

    binding = get_shapes_binding(adapter, layer)
    assert binding.source_row_id_by_rendered_row == (0,)
    assert binding.source_shapes_index_feature_name == "cell_id"
    assert list(layer.features.columns) == ["cell_id"]
    assert layer.features.iloc[0]["cell_id"] == "cell_1"
    assert "cell_id: cell_1" in layer.get_status(position=(1, 1))["value"]


def test_viewer_adapter_normalizes_native_shapes_layer_to_spatiato_status_layer() -> None:
    native_layer = Shapes(
        [
            np.asarray(
                [
                    [0.0, 0.0],
                    [0.0, 4.0],
                    [4.0, 4.0],
                    [4.0, 0.0],
                ],
                dtype=float,
            )
        ],
        shape_type="polygon",
        features=pd.DataFrame({"instance_id": ["cell_1"]}),
        affine=np.asarray([[1.0, 0.0, 5.0], [0.0, 1.0, 7.0], [0.0, 0.0, 1.0]]),
        name="native_shapes",
    )
    viewer = DummyViewer([native_layer])
    adapter = ViewerAdapter(viewer)

    replacement = adapter.normalize_native_shapes_layer_for_annotation(
        native_layer,
        source_shapes_index_feature_name="instance_id",
    )

    assert replacement is not native_layer
    assert native_layer not in viewer.layers
    assert viewer.layers == [replacement]
    np.testing.assert_allclose(
        replacement.data[0],
        np.asarray(
            [
                [5.0, 7.0],
                [5.0, 11.0],
                [9.0, 11.0],
                [9.0, 7.0],
            ]
        ),
    )
    np.testing.assert_allclose(replacement.affine.affine_matrix, np.eye(3))
    assert list(replacement.features.columns) == ["instance_id"]
    assert replacement.features["instance_id"].to_list() == ["cell_1"]
    assert "instance_id: cell_1" in replacement.get_status(position=(6, 8))["value"]


def test_viewer_adapter_ensure_shapes_loaded_uses_internal_row_ids_with_duplicate_geodataframe_index() -> None:
    polygon = Polygon([(0, 0), (4, 0), (4, 4), (0, 4), (0, 0)])
    geodataframe = gpd.GeoDataFrame(
        {"name": ["left", "right"]},
        geometry=[polygon, Polygon([(5, 0), (9, 0), (9, 4), (5, 4), (5, 0)])],
        index=["cell_1", "cell_1"],
    )
    sdata = make_shapes_sdata(geodataframe, shapes_name="cell_boundaries")
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)

    result = adapter.ensure_shapes_loaded(sdata, "cell_boundaries", "global")
    layer = result.layer

    binding = get_shapes_binding(adapter, layer)
    assert binding.source_row_id_by_rendered_row == (0, 1)
    assert layer.features["index"].to_list() == ["cell_1", "cell_1"]
    assert "index: cell_1" in layer.get_status(position=(1, 1))["value"]


def test_viewer_adapter_ensure_shapes_loaded_skips_empty_invalid_or_unsupported_geometries() -> None:
    valid_polygon = Polygon([(0, 0), (3, 0), (3, 3), (0, 3), (0, 0)])
    invalid_bowtie = Polygon([(4, 0), (7, 3), (7, 0), (4, 3), (4, 0)])
    empty_polygon = Polygon()
    unsupported_line = LineString([(0, 5), (5, 5)])
    geodataframe = gpd.GeoDataFrame(
        {"geometry": [valid_polygon, invalid_bowtie, empty_polygon, unsupported_line]},
        index=["valid", "bowtie", "empty", "line"],
    )
    sdata = make_shapes_sdata(geodataframe, shapes_name="mixed_shapes")
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)

    result = adapter.ensure_shapes_loaded(sdata, "mixed_shapes", "global")
    layer = result.layer

    assert result.skipped_geometry_count == 2
    assert layer.shape_type == ["polygon", "polygon", "polygon"]
    binding = get_shapes_binding(adapter, layer)
    assert binding.source_row_id_by_rendered_row == (0, 1, 1)
    assert binding.skipped_geometry_count == 2


def test_viewer_adapter_ensure_shapes_loaded_reuses_matching_existing_layer(sdata_blobs) -> None:
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)

    first = adapter.ensure_shapes_loaded(sdata_blobs, "blobs_polygons", "global")
    second = adapter.ensure_shapes_loaded(sdata_blobs, "blobs_polygons", "global")

    assert first.layer is second.layer
    assert first.created is True
    assert second.created is False
    assert len(viewer.layers) == 1


def test_viewer_adapter_primary_shapes_layer_applies_edge_width_control_to_all_shapes(sdata_blobs) -> None:
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)

    result = adapter.ensure_shapes_loaded(sdata_blobs, "blobs_polygons", "global")
    result.layer.current_edge_width = 6

    assert result.layer.edge_width == [6] * len(result.layer.data)


def test_viewer_adapter_primary_shapes_layer_applies_edge_color_control_to_all_shapes(sdata_blobs) -> None:
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)

    result = adapter.ensure_shapes_loaded(sdata_blobs, "blobs_polygons", "global")
    result.layer.current_edge_color = "red"

    expected_color = np.asarray([to_rgba("red")] * len(result.layer.data))
    np.testing.assert_allclose(result.layer.edge_color, expected_color)


def test_viewer_adapter_primary_shapes_layer_applies_face_color_control_to_all_shapes(sdata_blobs) -> None:
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)

    result = adapter.ensure_shapes_loaded(sdata_blobs, "blobs_polygons", "global")
    result.layer.current_face_color = "#ff000080"

    expected_color = np.asarray([to_rgba("#ff000080")] * len(result.layer.data))
    np.testing.assert_allclose(result.layer.face_color, expected_color)


def test_viewer_adapter_ensure_styled_shapes_loaded_creates_registered_variant_with_stored_palette() -> None:
    sdata = make_colorable_shapes_sdata()
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)
    style_spec = ShapeColumnColorSourceSpec(
        source_kind="shape_column",
        value_key="cell_type",
        value_kind="categorical",
    )

    result = adapter.ensure_styled_shapes_loaded(sdata, "cell_boundaries", "global", style_spec)

    assert result.created is True
    assert result.value_kind == "categorical"
    assert result.palette_source == "stored"
    assert result.coercion_applied is False
    assert result.layer in viewer.layers
    assert result.layer.name == "cell_boundaries[shapes_column:cell_type]"
    assert adapter.get_loaded_primary_shapes_layer(sdata, "cell_boundaries", "global") is None
    binding = get_shapes_binding(adapter, result.layer)
    assert binding.shapes_role == "styled"
    assert binding.style_spec == style_spec
    assert binding.source_row_id_by_rendered_row == (0, 1)
    assert binding.source_shapes_index_feature_name == "index"
    assert list(result.layer.features.columns) == ["index", "cell_type"]
    assert result.layer.features["cell_type"].to_list() == ["T", "B"]
    np.testing.assert_allclose(result.layer.face_color[0], (*to_rgba("#ff0000")[:3], 0.0))
    np.testing.assert_allclose(result.layer.edge_color[1], (*to_rgba("#00ff00")[:3], 1.0))
    assert "shapes_role" not in result.layer.metadata
    assert "style_source_kind" not in result.layer.metadata
    assert "style_value_key" not in result.layer.metadata
    assert "style_value_kind" not in result.layer.metadata


def test_viewer_adapter_ensure_styled_shapes_loaded_removes_layer_if_registration_fails(monkeypatch) -> None:
    sdata = make_colorable_shapes_sdata()
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)
    style_spec = ShapeColumnColorSourceSpec(
        source_kind="shape_column",
        value_key="cell_type",
        value_kind="categorical",
    )

    def fail_register(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("registration failed")

    monkeypatch.setattr(adapter, "register_shapes_layer", fail_register)

    with pytest.raises(RuntimeError, match="registration failed"):
        adapter.ensure_styled_shapes_loaded(sdata, "cell_boundaries", "global", style_spec)

    assert list(viewer.layers) == []
    assert adapter.layer_bindings.iter_bindings() == ()


def test_viewer_adapter_ensure_styled_shapes_loaded_creates_point_backed_categorical_variant() -> None:
    sdata = make_colorable_point_radius_shapes_sdata()
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)
    style_spec = ShapeColumnColorSourceSpec(
        source_kind="shape_column",
        value_key="cell_type",
        value_kind="categorical",
    )

    result = adapter.ensure_styled_shapes_loaded(sdata, "cell_centroids", "global", style_spec)

    assert result.created is True
    assert result.value_kind == "categorical"
    assert result.palette_source == "stored"
    assert result.coercion_applied is False
    assert result.shapes_rendering_mode == "points"
    assert isinstance(result.layer, Points)
    assert result.layer in viewer.layers
    assert result.layer.name == "cell_centroids[shapes_column:cell_type]"
    assert adapter.get_loaded_primary_shapes_layer(sdata, "cell_centroids", "global") is None
    assert adapter.get_loaded_styled_shapes_layer(sdata, "cell_centroids", style_spec, "global") is result.layer
    binding = get_shapes_binding(adapter, result.layer)
    assert binding.shapes_role == "styled"
    assert binding.shapes_rendering_mode == "points"
    assert binding.style_spec == style_spec
    assert binding.source_row_id_by_rendered_row == range(2)
    assert binding.source_shapes_index_feature_name == "index"
    assert list(result.layer.features.columns) == ["index", "cell_type"]
    assert result.layer.features["index"].to_list() == ["cell_1", "cell_2"]
    assert result.layer.features["cell_type"].to_list() == ["T", "B"]
    np.testing.assert_allclose(result.layer.size, np.asarray([4.0, 6.0]))
    assert result.layer.current_size == 5.0
    result.layer.current_size = 15.0
    np.testing.assert_allclose(result.layer.size, np.asarray([12.0, 18.0]))
    np.testing.assert_allclose(result.layer.face_color[0], to_rgba("#ff0000"))
    np.testing.assert_allclose(result.layer.border_color[1], to_rgba("#00ff00"))
    np.testing.assert_allclose(result.layer.border_width, np.zeros(2))
    status_value = result.layer.get_status(position=tuple(result.layer.data[0]))["value"]
    assert "index:" in status_value
    assert status_value.count("index:") == 1
    assert "cell_type:" in status_value
    assert status_value.count("cell_type:") == 1
    assert status_value.index("index:") < status_value.index("cell_type:")


def test_viewer_adapter_point_backed_shape_column_styling_keeps_data_driven_colors_after_ui_color_change() -> None:
    sdata = make_colorable_point_radius_shapes_sdata()
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)
    style_spec = ShapeColumnColorSourceSpec(
        source_kind="shape_column",
        value_key="cell_type",
        value_kind="categorical",
    )

    result = adapter.ensure_styled_shapes_loaded(sdata, "cell_centroids", "global", style_spec)
    face_color = result.layer.face_color.copy()
    border_color = result.layer.border_color.copy()

    result.layer.current_size = 15.0
    scaled_size = result.layer.size.copy()
    result.layer.current_face_color = "red"
    result.layer.current_symbol = "square"

    np.testing.assert_allclose(result.layer.face_color, face_color)
    np.testing.assert_allclose(result.layer.border_color, border_color)
    np.testing.assert_allclose(result.layer.size, scaled_size)
    assert all(symbol.value == "square" for symbol in result.layer.symbol)


def test_viewer_adapter_ensure_styled_shapes_loaded_creates_point_backed_continuous_variant() -> None:
    sdata = make_colorable_point_radius_shapes_sdata()
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)
    style_spec = ShapeColumnColorSourceSpec(
        source_kind="shape_column",
        value_key="score",
        value_kind="continuous",
    )

    first = adapter.ensure_styled_shapes_loaded(sdata, "cell_centroids", "global", style_spec)
    first.layer.current_size = 12.0
    scaled_size = first.layer.size.copy()
    second = adapter.ensure_styled_shapes_loaded(sdata, "cell_centroids", "global", style_spec, fill=True)

    assert isinstance(first.layer, Points)
    assert first.layer is second.layer
    assert first.created is True
    assert second.created is False
    assert first.value_kind == "continuous"
    assert first.palette_source is None
    assert first.coercion_applied is False
    assert get_shapes_binding(adapter, first.layer).shapes_rendering_mode == "points"
    assert list(first.layer.features.columns) == ["index", "score"]
    assert first.layer.features["score"].to_list() == [0.0, 1.0]
    np.testing.assert_allclose(first.layer.size, scaled_size)
    np.testing.assert_allclose(first.layer.size, np.asarray([9.6, 14.4]))
    np.testing.assert_allclose(first.layer.face_color, first.layer.border_color)
    assert len(viewer.layers) == 1


def test_viewer_adapter_point_backed_shape_column_styling_disambiguates_index_feature_collision() -> None:
    geodataframe = gpd.GeoDataFrame(
        {
            "radius": [2.0],
            "cell_type": pd.Categorical(["T"], categories=["T"]),
            "cell_type_colors": ["#ff0000"],
        },
        geometry=[Point(10, 20)],
        index=["cell_1"],
    )
    geodataframe.index.name = "cell_type"
    sdata = make_shapes_sdata(geodataframe, shapes_name="cell_centroids")
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)
    style_spec = ShapeColumnColorSourceSpec(
        source_kind="shape_column",
        value_key="cell_type",
        value_kind="categorical",
    )

    result = adapter.ensure_styled_shapes_loaded(sdata, "cell_centroids", "global", style_spec)

    assert isinstance(result.layer, Points)
    assert get_shapes_binding(adapter, result.layer).source_shapes_index_feature_name == "cell_type"
    assert list(result.layer.features.columns) == ["cell_type", "cell_type__value"]
    assert result.layer.features["cell_type"].to_list() == ["cell_1"]
    assert result.layer.features["cell_type__value"].to_list() == ["T"]


def test_viewer_adapter_ensure_styled_shapes_loaded_creates_table_backed_variant() -> None:
    sdata = make_table_backed_shapes_sdata()
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)
    style_spec = TableColorSourceSpec(
        table_name="table",
        source_kind="obs_column",
        value_key="cell_type",
        value_kind="categorical",
    )

    result = adapter.ensure_styled_shapes_loaded(sdata, "cell_boundaries", "global", style_spec)

    assert result.created is True
    assert result.value_kind == "categorical"
    assert result.palette_source == "stored"
    assert result.coercion_applied is False
    assert result.unannotated_source_shape_count == 1
    assert result.unannotated_rendered_shape_count == 1
    assert result.layer in viewer.layers
    assert result.layer.name == "cell_boundaries[obs:cell_type]"
    assert adapter.get_loaded_primary_shapes_layer(sdata, "cell_boundaries", "global") is None
    binding = get_shapes_binding(adapter, result.layer)
    assert binding.shapes_role == "styled"
    assert binding.style_spec == style_spec
    assert binding.source_row_id_by_rendered_row == (0, 1, 2)
    assert list(result.layer.features.columns) == ["instance_id", "cell_type"]
    assert result.layer.features["instance_id"].to_list() == ["cell_1", "cell_2", "cell_3"]
    assert result.layer.features["cell_type"].to_list()[:2] == ["T", "B"]
    assert pd.isna(result.layer.features["cell_type"].iloc[2])
    np.testing.assert_allclose(result.layer.edge_color[0], (*to_rgba("#ff0000")[:3], 1.0))
    np.testing.assert_allclose(result.layer.edge_color[1], (*to_rgba("#00ff00")[:3], 1.0))
    assert result.layer.edge_color[2, 3] == 0.0


def test_viewer_adapter_ensure_styled_shapes_loaded_creates_point_backed_table_variant() -> None:
    sdata = make_table_backed_point_radius_shapes_sdata()
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)
    style_spec = TableColorSourceSpec(
        table_name="table",
        source_kind="obs_column",
        value_key="cell_type",
        value_kind="categorical",
    )

    result = adapter.ensure_styled_shapes_loaded(sdata, "cell_centroids", "global", style_spec)
    result.layer.current_size = 12.0
    second = adapter.ensure_styled_shapes_loaded(sdata, "cell_centroids", "global", style_spec)

    assert result.created is True
    assert second.created is False
    assert second.layer is result.layer
    assert result.value_kind == "categorical"
    assert result.palette_source == "stored"
    assert result.coercion_applied is False
    assert result.unannotated_source_shape_count == 1
    assert result.unannotated_rendered_shape_count == 1
    assert result.shapes_rendering_mode == "points"
    assert isinstance(result.layer, Points)
    assert result.layer in viewer.layers
    assert result.layer.name == "cell_centroids[obs:cell_type]"
    binding = get_shapes_binding(adapter, result.layer)
    assert binding.shapes_role == "styled"
    assert binding.shapes_rendering_mode == "points"
    assert binding.style_spec == style_spec
    assert binding.source_row_id_by_rendered_row == range(3)
    assert list(result.layer.features.columns) == ["instance_id", "cell_type"]
    assert result.layer.features["instance_id"].to_list() == ["cell_1", "cell_2", "cell_3"]
    assert result.layer.features["cell_type"].to_list()[:2] == ["T", "B"]
    assert pd.isna(result.layer.features["cell_type"].iloc[2])
    np.testing.assert_allclose(result.layer.size, np.asarray([8.0, 12.0, 16.0]))
    np.testing.assert_allclose(result.layer.face_color[0], to_rgba("#ff0000"))
    np.testing.assert_allclose(result.layer.border_color[1], to_rgba("#00ff00"))
    assert result.layer.face_color[2, 3] == 0.0
    assert result.layer.border_color[2, 3] == 0.0
    np.testing.assert_allclose(result.layer.border_width, np.zeros(3))
    assert len(viewer.layers) == 1


def test_viewer_adapter_point_backed_table_styling_keeps_data_driven_colors_after_ui_color_change() -> None:
    sdata = make_table_backed_point_radius_shapes_sdata()
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)
    style_spec = TableColorSourceSpec(
        table_name="table",
        source_kind="obs_column",
        value_key="cell_type",
        value_kind="categorical",
    )

    result = adapter.ensure_styled_shapes_loaded(sdata, "cell_centroids", "global", style_spec)
    face_color = result.layer.face_color.copy()
    border_color = result.layer.border_color.copy()

    result.layer.current_size = 12.0
    scaled_size = result.layer.size.copy()
    result.layer.current_face_color = "red"
    result.layer.current_symbol = "square"

    np.testing.assert_allclose(result.layer.face_color, face_color)
    np.testing.assert_allclose(result.layer.border_color, border_color)
    np.testing.assert_allclose(result.layer.size, scaled_size)
    assert all(symbol.value == "square" for symbol in result.layer.symbol)


def test_viewer_adapter_point_backed_table_x_var_styling() -> None:
    sdata = make_table_backed_point_radius_shapes_sdata()
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)
    style_spec = TableColorSourceSpec(
        table_name="table",
        source_kind="x_var",
        value_key="GeneA",
        value_kind="continuous",
    )

    result = adapter.ensure_styled_shapes_loaded(sdata, "cell_centroids", "global", style_spec)

    assert isinstance(result.layer, Points)
    assert result.value_kind == "continuous"
    assert result.palette_source is None
    assert result.coercion_applied is False
    assert result.unannotated_source_shape_count == 1
    assert result.unannotated_rendered_shape_count == 1
    assert result.layer.name == "cell_centroids[X:GeneA]"
    assert get_shapes_binding(adapter, result.layer).shapes_rendering_mode == "points"
    assert result.layer.features["instance_id"].to_list() == ["cell_1", "cell_2", "cell_3"]
    assert result.layer.features["GeneA"].to_list()[:2] == [1.0, 2.0]
    assert pd.isna(result.layer.features["GeneA"].iloc[2])
    assert result.layer.face_color[0, 3] == 1.0
    assert result.layer.face_color[1, 3] == 1.0
    assert result.layer.face_color[2, 3] == 0.0


def test_viewer_adapter_point_backed_table_instance_styling() -> None:
    sdata = make_table_backed_point_radius_shapes_sdata()
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)
    style_spec = TableColorSourceSpec(
        table_name="table",
        source_kind="obs_column",
        value_key="instance_id",
        value_kind="instance",
    )

    result = adapter.ensure_styled_shapes_loaded(sdata, "cell_centroids", "global", style_spec)

    assert isinstance(result.layer, Points)
    assert result.value_kind == "instance"
    assert result.palette_source is None
    assert result.coercion_applied is False
    assert result.unannotated_source_shape_count == 1
    assert result.unannotated_rendered_shape_count == 1
    assert get_shapes_binding(adapter, result.layer).shapes_rendering_mode == "points"
    assert result.layer.features["instance_id"].to_list() == ["cell_1", "cell_2", "cell_3"]
    assert result.layer.features["instance_id__value"].to_list()[:2] == ["cell_1", "cell_2"]
    assert pd.isna(result.layer.features["instance_id__value"].iloc[2])
    assert result.layer.face_color[0, 3] == 1.0
    assert result.layer.face_color[1, 3] == 1.0
    assert result.layer.face_color[2, 3] == 0.0


def test_viewer_adapter_table_backed_shapes_uses_named_index_as_instance_key() -> None:
    geodataframe = gpd.GeoDataFrame(
        {"quality": [1.0, 0.8, 0.5]},
        geometry=[
            Polygon([(0, 0), (4, 0), (4, 4), (0, 4), (0, 0)]),
            Polygon([(5, 0), (9, 0), (9, 4), (5, 4), (5, 0)]),
            Polygon([(10, 0), (14, 0), (14, 4), (10, 4), (10, 0)]),
        ],
        index=["cell_1", "cell_2", "cell_3"],
    )
    geodataframe.index.name = "instance_id"
    shapes = ShapesModel.parse(geodataframe, transformations={"global": Identity()})
    table = TableModel.parse(
        ad.AnnData(
            obs=pd.DataFrame(
                {
                    "region": ["cell_boundaries", "cell_boundaries"],
                    "instance_id": ["cell_1", "cell_2"],
                    "cell_type": pd.Categorical(["T", "B"], categories=["T", "B"]),
                },
                index=["obs_1", "obs_2"],
            )
        ),
        region="cell_boundaries",
        region_key="region",
        instance_key="instance_id",
    )
    table.uns["cell_type_colors"] = ["#ff0000", "#00ff00"]
    sdata = ShapesSpatialDataWithTables(shapes={"cell_boundaries": shapes}, tables={"table": table})
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)
    style_spec = TableColorSourceSpec(
        table_name="table",
        source_kind="obs_column",
        value_key="cell_type",
        value_kind="categorical",
    )

    result = adapter.ensure_styled_shapes_loaded(sdata, "cell_boundaries", "global", style_spec)

    binding = get_shapes_binding(adapter, result.layer)
    assert binding.source_row_id_by_rendered_row == (0, 1, 2)
    assert binding.source_shapes_index_feature_name == "instance_id"
    assert list(result.layer.features.columns) == ["instance_id", "cell_type"]
    assert result.layer.features["instance_id"].to_list() == ["cell_1", "cell_2", "cell_3"]
    assert result.layer.features["cell_type"].to_list()[:2] == ["T", "B"]
    assert pd.isna(result.layer.features["cell_type"].iloc[2])
    np.testing.assert_allclose(result.layer.edge_color[0], (*to_rgba("#ff0000")[:3], 1.0))
    np.testing.assert_allclose(result.layer.edge_color[1], (*to_rgba("#00ff00")[:3], 1.0))
    assert result.layer.edge_color[2, 3] == 0.0


def test_viewer_adapter_table_backed_shapes_allows_duplicate_instance_index() -> None:
    geodataframe = gpd.GeoDataFrame(
        {},
        geometry=[
            Polygon([(0, 0), (4, 0), (4, 4), (0, 4), (0, 0)]),
            Polygon([(5, 0), (9, 0), (9, 4), (5, 4), (5, 0)]),
        ],
        index=["cell_1", "cell_1"],
    )
    geodataframe.index.name = "instance_id"
    shapes = ShapesModel.parse(geodataframe, transformations={"global": Identity()})
    table = TableModel.parse(
        ad.AnnData(
            obs=pd.DataFrame(
                {
                    "region": ["cell_boundaries"],
                    "instance_id": ["cell_1"],
                    "cell_type": pd.Categorical(["T"], categories=["T", "B"]),
                },
                index=["obs_1"],
            )
        ),
        region="cell_boundaries",
        region_key="region",
        instance_key="instance_id",
    )
    table.uns["cell_type_colors"] = ["#ff0000", "#00ff00"]
    sdata = ShapesSpatialDataWithTables(shapes={"cell_boundaries": shapes}, tables={"table": table})
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)
    style_spec = TableColorSourceSpec(
        table_name="table",
        source_kind="obs_column",
        value_key="cell_type",
        value_kind="categorical",
    )

    result = adapter.ensure_styled_shapes_loaded(sdata, "cell_boundaries", "global", style_spec)

    binding = get_shapes_binding(adapter, result.layer)
    assert binding.source_row_id_by_rendered_row == (0, 1)
    assert result.layer.features["instance_id"].to_list() == ["cell_1", "cell_1"]
    assert result.layer.features["cell_type"].to_list() == ["T", "T"]
    np.testing.assert_allclose(result.layer.edge_color[0], (*to_rgba("#ff0000")[:3], 1.0))
    np.testing.assert_allclose(result.layer.edge_color[1], (*to_rgba("#ff0000")[:3], 1.0))


def test_viewer_adapter_ensure_styled_shapes_loaded_reuses_table_backed_variant_and_updates_fill() -> None:
    sdata = make_table_backed_shapes_sdata()
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)
    style_spec = TableColorSourceSpec(
        table_name="table",
        source_kind="obs_column",
        value_key="cell_type",
        value_kind="categorical",
    )

    first = adapter.ensure_styled_shapes_loaded(sdata, "cell_boundaries", "global", style_spec)

    np.testing.assert_allclose(first.layer.face_color[:, 3], np.zeros(len(first.layer.data)))

    second = adapter.ensure_styled_shapes_loaded(sdata, "cell_boundaries", "global", style_spec, fill=True)

    assert second.layer is first.layer
    assert second.created is False
    assert len(viewer.layers) == 1
    np.testing.assert_allclose(first.layer.face_color[:2, 3], np.full(2, SHAPES_FACE_ALPHA))
    assert first.layer.face_color[2, 3] == 0.0
    np.testing.assert_allclose(first.layer.edge_color[:2, 3], np.ones(2))
    assert first.layer.edge_color[2, 3] == 0.0


def test_viewer_adapter_ensure_styled_shapes_loaded_updates_fill_alpha_on_existing_variant() -> None:
    sdata = make_colorable_shapes_sdata()
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)
    style_spec = ShapeColumnColorSourceSpec(
        source_kind="shape_column",
        value_key="cell_type",
        value_kind="categorical",
    )

    first = adapter.ensure_styled_shapes_loaded(sdata, "cell_boundaries", "global", style_spec)

    np.testing.assert_allclose(first.layer.face_color[:, 3], np.zeros(len(first.layer.data)))

    second = adapter.ensure_styled_shapes_loaded(sdata, "cell_boundaries", "global", style_spec, fill=True)

    assert second.layer is first.layer
    assert second.created is False
    assert len(viewer.layers) == 1
    np.testing.assert_allclose(first.layer.face_color[:, 3], np.full(len(first.layer.data), SHAPES_FACE_ALPHA))
    np.testing.assert_allclose(first.layer.edge_color[:, 3], np.ones(len(first.layer.data)))


def test_viewer_adapter_styled_shapes_layer_applies_edge_width_control_to_all_shapes() -> None:
    sdata = make_colorable_shapes_sdata()
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)
    style_spec = ShapeColumnColorSourceSpec(
        source_kind="shape_column",
        value_key="cell_type",
        value_kind="categorical",
    )

    result = adapter.ensure_styled_shapes_loaded(sdata, "cell_boundaries", "global", style_spec)
    result.layer.current_edge_width = 6

    assert result.layer.edge_width == [6] * len(result.layer.data)


def test_viewer_adapter_styled_shapes_layer_keeps_palette_when_current_edge_color_changes() -> None:
    sdata = make_colorable_shapes_sdata()
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)
    style_spec = ShapeColumnColorSourceSpec(
        source_kind="shape_column",
        value_key="cell_type",
        value_kind="categorical",
    )

    result = adapter.ensure_styled_shapes_loaded(sdata, "cell_boundaries", "global", style_spec)
    edge_color = result.layer.edge_color.copy()
    result.layer.current_edge_color = "red"

    np.testing.assert_array_equal(result.layer.edge_color, edge_color)


def test_viewer_adapter_styled_shapes_status_includes_selected_shape_column() -> None:
    sdata = make_colorable_shapes_sdata()
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)
    style_spec = ShapeColumnColorSourceSpec(
        source_kind="shape_column",
        value_key="cell_type",
        value_kind="categorical",
    )

    result = adapter.ensure_styled_shapes_loaded(sdata, "cell_boundaries", "global", style_spec)

    status = result.layer.get_status([1, 1], dims_displayed=[0, 1])

    assert "index: cell_1" in status["value"]
    assert "cell_type: T" in status["value"]


def test_viewer_adapter_ensure_styled_shapes_loaded_coexists_with_primary_shapes() -> None:
    sdata = make_colorable_shapes_sdata()
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)
    style_spec = ShapeColumnColorSourceSpec(
        source_kind="shape_column",
        value_key="cell_type",
        value_kind="categorical",
    )

    primary_layer = adapter.ensure_shapes_loaded(sdata, "cell_boundaries", "global").layer
    styled_result = adapter.ensure_styled_shapes_loaded(sdata, "cell_boundaries", "global", style_spec)

    assert primary_layer is not styled_result.layer
    assert adapter.ensure_shapes_loaded(sdata, "cell_boundaries", "global").layer is primary_layer
    assert adapter.get_loaded_primary_shapes_layer(sdata, "cell_boundaries", "global") is primary_layer
    assert adapter.get_loaded_styled_shapes_layer(sdata, "cell_boundaries", style_spec, "global") is styled_result.layer
    assert adapter.get_loaded_styled_shapes_layers(sdata, "cell_boundaries", "global") == [styled_result.layer]
    assert list(viewer.layers) == [primary_layer, styled_result.layer]
    primary_binding = get_shapes_binding(adapter, primary_layer)
    styled_binding = get_shapes_binding(adapter, styled_result.layer)
    assert primary_binding.shapes_role == "primary"
    assert primary_binding.style_spec is None
    assert styled_binding.shapes_role == "styled"
    assert styled_binding.style_spec == style_spec


def test_viewer_adapter_ensure_styled_shapes_loaded_reuses_matching_variant() -> None:
    sdata = make_colorable_shapes_sdata()
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)
    style_spec = ShapeColumnColorSourceSpec(
        source_kind="shape_column",
        value_key="score",
        value_kind="continuous",
    )

    first = adapter.ensure_styled_shapes_loaded(sdata, "cell_boundaries", "global", style_spec)
    second = adapter.ensure_styled_shapes_loaded(sdata, "cell_boundaries", "global", style_spec)

    assert first.layer is second.layer
    assert first.created is True
    assert second.created is False
    assert second.value_kind == "continuous"
    assert len(viewer.layers) == 1


def test_viewer_adapter_ensure_styled_shapes_loaded_creates_distinct_variants_for_different_style_specs() -> None:
    sdata = make_colorable_shapes_sdata()
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)
    categorical_style = ShapeColumnColorSourceSpec(
        source_kind="shape_column",
        value_key="cell_type",
        value_kind="categorical",
    )
    continuous_style = ShapeColumnColorSourceSpec(
        source_kind="shape_column",
        value_key="score",
        value_kind="continuous",
    )

    first = adapter.ensure_styled_shapes_loaded(sdata, "cell_boundaries", "global", categorical_style)
    second = adapter.ensure_styled_shapes_loaded(sdata, "cell_boundaries", "global", continuous_style)

    assert first.layer is not second.layer
    assert len(viewer.layers) == 2
    assert adapter.get_loaded_styled_shapes_layer(sdata, "cell_boundaries", categorical_style, "global") is first.layer
    assert adapter.get_loaded_styled_shapes_layer(sdata, "cell_boundaries", continuous_style, "global") is second.layer
    assert adapter.get_loaded_styled_shapes_layers(sdata, "cell_boundaries", "global") == [first.layer, second.layer]


def test_viewer_adapter_ensure_styled_shapes_loaded_creates_distinct_variants_for_different_tables() -> None:
    sdata = make_two_table_backed_shapes_sdata()
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)
    table_a_style = TableColorSourceSpec(
        table_name="table",
        source_kind="obs_column",
        value_key="cell_type",
        value_kind="categorical",
    )
    table_b_style = TableColorSourceSpec(
        table_name="table_b",
        source_kind="obs_column",
        value_key="cell_type",
        value_kind="categorical",
    )

    first = adapter.ensure_styled_shapes_loaded(sdata, "cell_boundaries", "global", table_a_style)
    second = adapter.ensure_styled_shapes_loaded(sdata, "cell_boundaries", "global", table_b_style)

    assert first.layer is not second.layer
    assert len(viewer.layers) == 2
    assert first.layer.name == "cell_boundaries[obs:cell_type]"
    assert second.layer.name == "cell_boundaries[obs:cell_type]"
    assert adapter.get_loaded_styled_shapes_layer(sdata, "cell_boundaries", table_a_style, "global") is first.layer
    assert adapter.get_loaded_styled_shapes_layer(sdata, "cell_boundaries", table_b_style, "global") is second.layer
    assert adapter.get_loaded_styled_shapes_layers(sdata, "cell_boundaries", "global") == [first.layer, second.layer]
    np.testing.assert_allclose(first.layer.edge_color[0], (*to_rgba("#ff0000")[:3], 1.0))
    np.testing.assert_allclose(second.layer.edge_color[0], (*to_rgba("#0000ff")[:3], 1.0))


def test_viewer_adapter_remove_shapes_layer_removes_only_primary_shapes_layer() -> None:
    sdata = make_colorable_shapes_sdata()
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)
    style_spec = ShapeColumnColorSourceSpec(
        source_kind="shape_column",
        value_key="cell_type",
        value_kind="categorical",
    )
    primary_layer = adapter.ensure_shapes_loaded(sdata, "cell_boundaries", "global").layer
    styled_layer = adapter.ensure_styled_shapes_loaded(sdata, "cell_boundaries", "global", style_spec).layer

    removed_layer = adapter.remove_shapes_layer(sdata, "cell_boundaries", "global")

    assert removed_layer is primary_layer
    assert list(viewer.layers) == [styled_layer]
    assert adapter.layer_bindings.get_binding(primary_layer) is None
    assert get_shapes_binding(adapter, styled_layer).shapes_role == "styled"


def test_viewer_adapter_remove_layers_outside_coordinate_system_removes_primary_and_styled_shapes() -> None:
    sdata = make_colorable_shapes_sdata()
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)
    style_spec = ShapeColumnColorSourceSpec(
        source_kind="shape_column",
        value_key="cell_type",
        value_kind="categorical",
    )
    primary_layer = adapter.ensure_shapes_loaded(sdata, "cell_boundaries", "global").layer
    styled_layer = adapter.ensure_styled_shapes_loaded(sdata, "cell_boundaries", "global", style_spec).layer

    removed_bindings = adapter.remove_layers_outside_coordinate_system(sdata=sdata, coordinate_system="local")

    assert [binding.layer for binding in removed_bindings] == [primary_layer, styled_layer]
    assert list(viewer.layers) == []
    assert adapter.layer_bindings.get_binding(primary_layer) is None
    assert adapter.layer_bindings.get_binding(styled_layer) is None


def test_viewer_adapter_remove_shapes_layer_removes_registered_layer(sdata_blobs) -> None:
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)
    layer = adapter.ensure_shapes_loaded(sdata_blobs, "blobs_polygons", "global").layer

    removed_layer = adapter.remove_shapes_layer(sdata_blobs, "blobs_polygons", "global")

    assert removed_layer is layer
    assert list(viewer.layers) == []
    assert adapter.layer_bindings.get_binding(layer) is None


def test_viewer_adapter_ensure_shapes_loaded_rejects_unknown_coordinate_system(sdata_blobs) -> None:
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)

    try:
        adapter.ensure_shapes_loaded(sdata_blobs, "blobs_polygons", "not_a_coordinate_system")
    except ValueError as error:
        assert "Coordinate system `not_a_coordinate_system`" in str(error)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected ensure_shapes_loaded to reject an unknown coordinate system.")


def test_viewer_adapter_ensure_image_loaded_adds_stack_layer_and_registers_binding(sdata_blobs) -> None:
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)

    result = adapter.ensure_image_loaded(sdata_blobs, "blobs_image", "global", mode="stack")
    layer = result.primary_layer

    assert result.layers == (layer,)
    assert result.mode == "stack"
    assert result.created is True
    assert result.channels == ()
    assert layer in viewer.layers
    assert layer.name == "blobs_image"
    assert layer.affine is not None
    assert layer.rgb is False
    binding = adapter.layer_bindings.get_binding(layer)
    assert binding is not None
    assert binding.element_name == "blobs_image"
    assert binding.element_type == "image"
    assert binding.coordinate_system == "global"
    assert binding.image_display_mode == "stack"
    assert "image_display_mode" not in layer.metadata


def test_viewer_adapter_ensure_image_loaded_reuses_matching_existing_stack_layer(sdata_blobs) -> None:
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)

    first = adapter.ensure_image_loaded(sdata_blobs, "blobs_image", "global", mode="stack")
    second = adapter.ensure_image_loaded(sdata_blobs, "blobs_image", "global", mode="stack")

    assert first.primary_layer is second.primary_layer
    assert first.created is True
    assert second.created is False
    assert len(viewer.layers) == 1


def test_viewer_adapter_ensure_image_loaded_supports_multiscale_images(sdata_blobs) -> None:
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)

    result = adapter.ensure_image_loaded(sdata_blobs, "blobs_multiscale_image", "global", mode="stack")
    layer = result.primary_layer

    assert layer.multiscale is True
    assert len(layer.data) == 3
    assert all(not isinstance(scale_data, DataArray) for scale_data in layer.data)
    assert layer in viewer.layers
    binding = adapter.layer_bindings.get_binding(layer)
    assert binding is not None
    assert binding.image_display_mode == "stack"


def test_viewer_adapter_ensure_image_loaded_rejects_unknown_coordinate_system(sdata_blobs) -> None:
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)

    try:
        adapter.ensure_image_loaded(sdata_blobs, "blobs_image", "not_a_coordinate_system", mode="stack")
    except ValueError as error:
        assert "Coordinate system `not_a_coordinate_system`" in str(error)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected ensure_image_loaded to reject an unknown coordinate system.")


def test_viewer_adapter_ensure_image_loaded_overlay_adds_one_layer_per_selected_channel(sdata_blobs) -> None:
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)

    result = adapter.ensure_image_loaded(
        sdata_blobs,
        "blobs_image",
        "global",
        mode="overlay",
        channels=[0, 2],
        channel_colors=["blue", "magenta"],
    )
    layers = result.layers

    assert result.mode == "overlay"
    assert result.created is True
    assert result.channels == (0, 2)
    assert result.channel_names == ("0", "2")
    assert len(layers) == 2
    assert [layer.name for layer in layers] == ["blobs_image[0]", "blobs_image[2]"]
    assert [layer.blending for layer in layers] == ["additive", "additive"]
    assert [binding.channel_index for binding in (adapter.layer_bindings.get_binding(layer) for layer in layers)] == [
        0,
        2,
    ]
    assert [binding.channel_name for binding in (adapter.layer_bindings.get_binding(layer) for layer in layers)] == [
        "0",
        "2",
    ]


def test_viewer_adapter_ensure_image_loaded_overlay_reuses_existing_channel_layers(sdata_blobs) -> None:
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)

    first = adapter.ensure_image_loaded(
        sdata_blobs,
        "blobs_image",
        "global",
        mode="overlay",
        channels=[0, 1],
        channel_colors=["blue", "red"],
    )
    second = adapter.ensure_image_loaded(
        sdata_blobs,
        "blobs_image",
        "global",
        mode="overlay",
        channels=[0, 1],
        channel_colors=["cyan", "yellow"],
    )

    assert first.layers == second.layers
    assert first.created is True
    assert second.created is False
    assert len(viewer.layers) == 2
    assert [str(layer.colormap).lower() for layer in second.layers] != []


def test_viewer_adapter_unregisters_binding_when_user_removes_overlay_layer(sdata_blobs) -> None:
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)

    result = adapter.ensure_image_loaded(
        sdata_blobs,
        "blobs_image",
        "global",
        mode="overlay",
        channels=[0, 1],
        channel_colors=["blue", "red"],
    )
    layers = result.layers

    viewer.layers.remove(layers[0])

    assert layers[0] not in viewer.layers
    assert adapter.layer_bindings.get_binding(layers[0]) is None
    assert adapter.layer_bindings.get_binding(layers[1]) is not None


def test_viewer_adapter_ensure_image_loaded_overlay_removes_existing_stack_layer(sdata_blobs) -> None:
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)

    stack_layer = adapter.ensure_image_loaded(sdata_blobs, "blobs_image", "global", mode="stack").primary_layer
    overlay_layers = adapter.ensure_image_loaded(
        sdata_blobs,
        "blobs_image",
        "global",
        mode="overlay",
        channels=[0, 1],
        channel_colors=["blue", "red"],
    ).layers

    assert stack_layer not in viewer.layers
    assert adapter.layer_bindings.get_binding(stack_layer) is None
    assert len(overlay_layers) == 2
    assert all(layer in viewer.layers for layer in overlay_layers)


def test_viewer_adapter_ensure_image_loaded_overlay_removes_stale_channel_layers(sdata_blobs) -> None:
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)

    adapter.ensure_image_loaded(
        sdata_blobs,
        "blobs_image",
        "global",
        mode="overlay",
        channels=[0, 1],
        channel_colors=["blue", "red"],
    )
    result = adapter.ensure_image_loaded(
        sdata_blobs,
        "blobs_image",
        "global",
        mode="overlay",
        channels=[1],
        channel_colors=["green"],
    )
    layers = result.layers

    assert result.created is False
    assert len(layers) == 1
    assert len(viewer.layers) == 1
    binding = adapter.layer_bindings.get_binding(layers[0])
    assert binding is not None
    assert binding.channel_index == 1

    mixed_result = adapter.ensure_image_loaded(
        sdata_blobs,
        "blobs_image",
        "global",
        mode="overlay",
        channels=[1, 2],
        channel_colors=["green", "magenta"],
    )

    assert mixed_result.created is True
    assert mixed_result.channels == (1, 2)
    assert mixed_result.channel_names == ("1", "2")
    assert mixed_result.layers[0] is layers[0]
    assert len(viewer.layers) == 2


def test_viewer_adapter_ensure_image_overlay_channel_loaded_preserves_sibling_channels(sdata_blobs) -> None:
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)

    existing_result = adapter.ensure_image_loaded(
        sdata_blobs,
        "blobs_image",
        "global",
        mode="overlay",
        channels=[0, 2],
        channel_colors=["blue", "magenta"],
    )

    result = adapter.ensure_image_overlay_channel_loaded(
        sdata_blobs,
        "blobs_image",
        "global",
        channel=1,
        channel_color="#123456",
    )

    assert result.created is True
    assert result.channels == (1, 0, 2)
    assert result.channel_names == ("1", "0", "2")
    assert result.primary_layer.name == "blobs_image[1]"
    assert len(viewer.layers) == 3
    assert existing_result.layers[0] in viewer.layers
    assert existing_result.layers[1] in viewer.layers
    assert result.primary_layer.colormap.name == "#123456"
    assert existing_result.layers[0].colormap.name == "blue"
    assert existing_result.layers[1].colormap.name == "magenta"


def test_viewer_adapter_ensure_image_overlay_channel_loaded_preserves_changed_sibling_color(sdata_blobs) -> None:
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)

    existing_result = adapter.ensure_image_loaded(
        sdata_blobs,
        "blobs_image",
        "global",
        mode="overlay",
        channels=[0],
        channel_colors=["blue"],
    )
    existing_result.primary_layer.colormap = "#ABCDEF"

    result = adapter.ensure_image_overlay_channel_loaded(
        sdata_blobs,
        "blobs_image",
        "global",
        channel=1,
        channel_color="#123456",
    )

    assert result.channels == (1, 0)
    assert result.primary_layer.colormap.name == "#123456"
    assert existing_result.primary_layer in viewer.layers
    assert existing_result.primary_layer.colormap.name.lower() == "#abcdef"


def test_viewer_adapter_ensure_image_overlay_channel_loaded_reuses_matching_channel(sdata_blobs) -> None:
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)

    existing_result = adapter.ensure_image_loaded(
        sdata_blobs,
        "blobs_image",
        "global",
        mode="overlay",
        channels=[0],
        channel_colors=["blue"],
    )

    result = adapter.ensure_image_overlay_channel_loaded(
        sdata_blobs,
        "blobs_image",
        "global",
        channel="0",
        channel_color="#123456",
    )

    assert result.created is False
    assert result.channels == (0,)
    assert result.primary_layer is existing_result.primary_layer
    assert len(viewer.layers) == 1
    assert result.primary_layer.colormap.name == "#123456"


def test_viewer_adapter_ensure_image_loaded_stack_removes_existing_overlay_layers(sdata_blobs) -> None:
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)

    overlay_layers = adapter.ensure_image_loaded(
        sdata_blobs,
        "blobs_image",
        "global",
        mode="overlay",
        channels=[0, 1],
        channel_colors=["blue", "red"],
    ).layers
    stack_layer = adapter.ensure_image_loaded(sdata_blobs, "blobs_image", "global", mode="stack").primary_layer

    assert all(layer not in viewer.layers for layer in overlay_layers)
    assert all(adapter.layer_bindings.get_binding(layer) is None for layer in overlay_layers)
    assert stack_layer in viewer.layers


def test_viewer_adapter_ensure_image_loaded_overlay_accepts_channel_names(sdata_blobs) -> None:
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)

    sdata_blobs.images["blobs_image"] = sdata_blobs.images["blobs_image"].assign_coords(c=["DAPI", "CD3", "CD8"])
    result = adapter.ensure_image_loaded(
        sdata_blobs,
        "blobs_image",
        "global",
        mode="overlay",
        channels=["CD3", "CD8"],
        channel_colors=["green", "magenta"],
    )
    layers = result.layers

    assert result.channels == (1, 2)
    assert result.channel_names == ("CD3", "CD8")
    assert [layer.name for layer in layers] == ["blobs_image[CD3]", "blobs_image[CD8]"]
    assert [adapter.layer_bindings.get_binding(layer).channel_name for layer in layers] == ["CD3", "CD8"]


def test_viewer_adapter_ensure_image_loaded_overlay_supports_multiscale_images(sdata_blobs) -> None:
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)

    result = adapter.ensure_image_loaded(
        sdata_blobs,
        "blobs_multiscale_image",
        "global",
        mode="overlay",
        channels=[0, 2],
    )
    layers = result.layers

    assert result.channels == (0, 2)
    assert len(layers) == 2
    assert all(layer.multiscale is True for layer in layers)
    assert all(len(layer.data) == 3 for layer in layers)
    assert all(not isinstance(scale_data, DataArray) for layer in layers for scale_data in layer.data)


def test_viewer_adapter_ensure_image_loaded_overlay_requires_selected_channels(sdata_blobs) -> None:
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)

    try:
        adapter.ensure_image_loaded(sdata_blobs, "blobs_image", "global", mode="overlay", channels=None)
    except ValueError as error:
        assert "requires at least one selected channel" in str(error)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected ensure_image_loaded to require selected channels for overlay mode.")


def test_viewer_adapter_ensure_image_loaded_overlay_rejects_duplicate_channels(sdata_blobs) -> None:
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)

    try:
        adapter.ensure_image_loaded(sdata_blobs, "blobs_image", "global", mode="overlay", channels=[0, 0])
    except ValueError as error:
        assert "does not accept duplicate channel selections" in str(error)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected ensure_image_loaded to reject duplicate overlay channels.")
