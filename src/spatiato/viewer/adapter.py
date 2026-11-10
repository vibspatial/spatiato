from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeGuard

import dask.array as da
import numpy as np
import pandas as pd
from loguru import logger
from napari.layers import Image, Labels, Layer, Points, Shapes
from qtpy.QtCore import QObject, Signal
from shapely import make_valid
from shapely.errors import GEOSException
from shapely.geometry import GeometryCollection, MultiPolygon, Point, Polygon
from shapely.geometry.base import BaseGeometry
from spatialdata import transform as transform_spatial_element
from spatialdata.models import get_axes_names
from spatialdata.transformations import get_transformation
from xarray import DataArray, DataTree

from spatiato._shapes_triangulation import ensure_shapes_triangulation_backend
from spatiato.core._color_source import (
    ShapeColumnColorSourceSpec,
    TableColorSourceSpec,
)
from spatiato.core.shapes_geometry import shapely_polygon_to_napari_polygon_vertices
from spatiato.viewer.image_styling import DEFAULT_OVERLAY_COLORS, ImageDisplayMode, ImageLoadResult
from spatiato.viewer.labels_contours import install_fast_labels_contours
from spatiato.viewer.labels_styling import (
    LabelsLoadResult,
    apply_table_color_source_to_labels_layer,
    build_styled_labels_layer_name,
)
from spatiato.viewer.points_styling import (
    _POINTS_FACE_COLOR_OVERRIDE_ATTR,
    POINTS_SELECTION_MAX_CATEGORICAL_COLORS,
    POINTS_SELECTION_SOLID_COLOR,
    PointsLoadResult,
    apply_points_selection_style,
    build_points_selection_layer_name,
    connect_current_face_color_to_global_point_face_color,
    connect_current_size_to_radius_scaled_point_size,
    connect_current_symbol_to_global_point_symbol,
)
from spatiato.viewer.shapes_styling import (
    ShapesLoadResult,
    apply_shape_column_color_source_to_shapes_layer,
    apply_table_color_source_to_shapes_layer,
    build_styled_shapes_layer_name,
)
from spatiato.viewer.shapes_styling import (
    apply_primary_shapes_layer_style as _apply_primary_shapes_layer_style,
)
from spatiato.viewer.tiled_points.application import (
    TiledPointsApplicationSettings,
    TiledPointsCacheDescriptor,
    canonical_value_palette,
)
from spatiato.viewer.tiled_points.napari import TiledPointsLayerModel, register_tiled_points_layer
from spatiato.viewer.tiled_points.runtime.layer_runtime import _TiledPointsLayerRuntime

if TYPE_CHECKING:
    from spatialdata import SpatialData

    from spatiato._points_value_index import PointsValueSelection

ElementType = Literal["labels", "image", "shapes", "points"]
ShapesLayerShapeType = Literal["polygon", "ellipse"]
ShapesRenderingMode = Literal["shapes", "points"]
# Generic shapes can repeat source row ids when one source row expands into
# multiple rendered polygons. Point-radius shapes are one-to-one, so `range`
# avoids materializing large `tuple(range(n))` mappings.
SourceRowIdByRenderedRow = tuple[int, ...] | range
# Name of the layer.features column that stores the source GeoDataFrame index.
DEFAULT_SHAPES_INDEX_FEATURE_NAME = "index"


class _SpatiatoShapes(Shapes):
    """Napari ``Shapes`` layer with Spatiato-specific status-bar text.

    This subclass only customizes display behavior: when the cursor is over a
    rendered shape, it reads the feature row from ``layer.features`` and
    appends non-missing values to napari's status bar.
    """

    def __init__(
        self,
        *args: Any,
        source_shapes_index_feature_name: str = DEFAULT_SHAPES_INDEX_FEATURE_NAME,
        **kwargs: Any,
    ) -> None:
        self._source_shapes_index_feature_name = source_shapes_index_feature_name
        super().__init__(*args, **kwargs)

    def get_status(
        self,
        position: Any | None = None,
        *,
        view_direction: Any | None = None,
        dims_displayed: list[int] | None = None,
        world: bool = False,
        value: Any | None = None,
    ) -> dict[str, str]:
        status = super().get_status(
            position,
            view_direction=view_direction,
            dims_displayed=dims_displayed,
            world=world,
            value=value,
        )
        if position is None:
            return status

        shape_value = self.get_value(
            position,
            view_direction=view_direction,
            dims_displayed=dims_displayed,
            world=world,
        )
        feature_status = self._get_feature_status(shape_value)
        if feature_status:
            status["coordinates"] += f"; {feature_status}"
            status["value"] += f"; {feature_status}"
        return status

    def _get_feature_status(self, value: Any) -> str | None:
        """Return status text for the picked rendered shape row.

        Spatiato keeps the source GeoDataFrame index in ``layer.features`` so a
        rendered napari row can be traced back to its original source row, even
        when one source geometry expands into multiple rendered rows. That
        source index is shown first, followed by any other non-missing feature
        values such as the selected styled shape column.
        """
        if not isinstance(value, tuple) or not value:
            return None

        shape_index = value[0]
        if shape_index is None:
            return None

        try:
            feature_row = self.features.iloc[int(shape_index)]
        except (IndexError, TypeError, ValueError):
            return None

        status_parts: list[str] = []
        index_feature_name = self._source_shapes_index_feature_name
        if index_feature_name in feature_row:
            source_index = feature_row[index_feature_name]
            if not _is_missing_feature_value(source_index):
                status_parts.append(f"{index_feature_name}: {source_index}")

        for feature_name, feature_value in feature_row.items():
            if feature_name == index_feature_name or _is_missing_feature_value(feature_value):
                continue
            status_parts.append(f"{feature_name}: {feature_value}")

        return "; ".join(status_parts) or None


class _SpatiatoPointRadiusShapes(Points):
    """Napari ``Points`` layer with shapes-style Spatiato status-bar text."""

    def __init__(
        self,
        *args: Any,
        source_shapes_index_feature_name: str = DEFAULT_SHAPES_INDEX_FEATURE_NAME,
        **kwargs: Any,
    ) -> None:
        self._source_shapes_index_feature_name = source_shapes_index_feature_name
        super().__init__(*args, **kwargs)

    def get_status(
        self,
        position: Any | None = None,
        *,
        view_direction: Any | None = None,
        dims_displayed: list[int] | None = None,
        world: bool = False,
        value: Any | None = None,
    ) -> dict[str, str]:
        status = super().get_status(
            position,
            view_direction=view_direction,
            dims_displayed=dims_displayed,
            world=world,
        )
        if position is None:
            return status

        # Napari Points already appends normal feature columns to the status
        # text. The fallback feature name "index" is special-cased by napari
        # and is not shown, so Spatiato only appends that source index manually.
        if self._source_shapes_index_feature_name != DEFAULT_SHAPES_INDEX_FEATURE_NAME:
            return status

        point_value = self.get_value(
            position,
            view_direction=view_direction,
            dims_displayed=dims_displayed,
            world=world,
        )
        index_status = self._get_source_index_status(point_value)
        if index_status and f"{DEFAULT_SHAPES_INDEX_FEATURE_NAME}:" not in status["value"]:
            status["coordinates"] = _insert_feature_status_first(status["coordinates"], index_status)
            status["value"] = _insert_feature_status_first(status["value"], index_status)
        return status

    def _get_source_index_status(self, value: Any) -> str | None:
        """Return status text for the source index when napari omits it."""
        point_index = value[0] if isinstance(value, tuple) and value else value
        if point_index is None:
            return None

        try:
            feature_row = self.features.iloc[int(point_index)]
        except (IndexError, TypeError, ValueError):
            return None

        if DEFAULT_SHAPES_INDEX_FEATURE_NAME not in feature_row:
            return None

        source_index = feature_row[DEFAULT_SHAPES_INDEX_FEATURE_NAME]
        if _is_missing_feature_value(source_index):
            return None
        return f"{DEFAULT_SHAPES_INDEX_FEATURE_NAME}: {source_index}"


def _is_missing_feature_value(value: Any) -> bool:
    if value is None:
        return True

    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _insert_feature_status_first(status_text: str, feature_status: str) -> str:
    """Insert Spatiato source-index status before napari Points feature text."""
    if not status_text:
        return feature_status
    if ";" not in status_text:
        return f"{status_text}; {feature_status}"

    base_status, feature_text = status_text.split(";", 1)
    return f"{base_status}; {feature_status};{feature_text}"


@dataclass(frozen=True, kw_only=True)
class BaseLayerBinding:
    """Runtime binding between a napari layer and a SpatialData element."""

    layer: Layer
    element_name: str
    element_type: ElementType
    coordinate_system: str | None
    sdata_id: int | None = None


@dataclass(frozen=True, kw_only=True)
class LabelsLayerBinding(BaseLayerBinding):
    """Binding metadata specific to labels layers."""

    element_type: Literal["labels"] = "labels"
    labels_role: Literal["primary", "styled"] = "primary"
    style_spec: TableColorSourceSpec | None = None

    def __post_init__(self) -> None:
        if self.labels_role == "primary" and self.style_spec is not None:
            raise ValueError("Primary labels bindings must not carry a style specification.")
        if self.labels_role == "styled" and self.style_spec is None:
            raise ValueError("Styled labels bindings require a style specification.")


@dataclass(frozen=True, kw_only=True)
class ImageLayerBinding(BaseLayerBinding):
    """Binding metadata specific to image layers."""

    element_type: Literal["image"] = "image"
    image_display_mode: ImageDisplayMode | None = None
    channel_index: int | None = None
    channel_name: str | None = None


@dataclass(frozen=True, kw_only=True)
class PointsLayerBinding(BaseLayerBinding):
    """Binding metadata specific to points value-selection layers."""

    element_type: Literal["points"] = "points"
    index_column: str

    def __post_init__(self) -> None:
        if not isinstance(self.index_column, str) or not self.index_column:
            raise ValueError("Points layer bindings require a non-empty index column.")


@dataclass(frozen=True, kw_only=True)
class TiledPointsLayerBinding(BaseLayerBinding):
    """Own one persistent tiled-points model and its terminal runtime lifecycle."""

    layer: TiledPointsLayerModel
    element_type: Literal["points"] = "points"
    value_column: str
    cache_root: Path
    cache_generation_id: str
    runtime: _TiledPointsLayerRuntime

    def __post_init__(self) -> None:
        if not self.value_column:
            raise ValueError("Tiled points bindings require a non-empty value column.")
        if not isinstance(self.cache_root, Path):
            raise ValueError("Tiled points bindings require a pathlib.Path cache root.")
        if self.cache_generation_id != self.layer.data.cache_generation_id:
            raise ValueError("Binding and tiled-points layer cache generations must match.")


@dataclass(frozen=True, kw_only=True)
class ShapesLayerBinding(BaseLayerBinding):
    """Binding metadata specific to shapes layers.

    Parameters
    ----------
    element_type
        Fixed layer binding discriminator for shapes elements.
    shapes_role
        Whether this layer is the primary geometry layer or a viewer-only
        styled variant for one shape-column color source.
    shapes_rendering_mode
        How the shapes element is represented in napari. Generic geometries
        use a napari ``Shapes`` layer, while point-radius shapes can use a
        napari ``Points`` layer for faster primary visualization.
    style_spec
        Shape color source used for styled shape variants. Primary shape
        bindings do not carry a style specification.
    source_row_id_by_rendered_row
        Internal integer source GeoDataFrame row id for each rendered napari
        shape row.
        This can be longer than the source GeoDataFrame row count when one
        source row, such as a ``MultiPolygon``, expands into multiple rendered
        napari shapes. For example, if source row position ``7`` expands into
        three polygons and source row position ``8`` expands into one polygon,
        this mapping is ``(7, 7, 7, 8)``. Styled shapes use it to repeat the
        source row's style value for every rendered part.
    source_shapes_index_feature_name
        Name of the ``layer.features`` column that stores the source
        GeoDataFrame index for napari-visible inspection and status-bar text.
        This follows the GeoDataFrame index name, falling back to ``"index"``
        for unnamed indexes.
    skipped_geometry_count
        Number of source geometry rows that could not be rendered, for example
        empty, unsupported, or invalid geometries that could not be converted
        into renderable polygons or ellipses.
    """

    element_type: Literal["shapes"] = "shapes"
    shapes_role: Literal["primary", "styled"] = "primary"
    shapes_rendering_mode: ShapesRenderingMode = "shapes"
    style_spec: ShapeColumnColorSourceSpec | TableColorSourceSpec | None = None
    source_row_id_by_rendered_row: SourceRowIdByRenderedRow = ()
    source_shapes_index_feature_name: str = DEFAULT_SHAPES_INDEX_FEATURE_NAME
    skipped_geometry_count: int = 0

    def __post_init__(self) -> None:
        if self.shapes_rendering_mode not in ("shapes", "points"):
            raise ValueError("Shapes bindings require `shapes_rendering_mode` to be 'shapes' or 'points'.")
        if self.shapes_role == "primary" and self.style_spec is not None:
            raise ValueError("Primary shapes bindings must not carry a style specification.")
        if self.shapes_role == "styled" and self.style_spec is None:
            raise ValueError("Styled shapes bindings require a style specification.")


LayerBinding = (
    LabelsLayerBinding | ImageLayerBinding | PointsLayerBinding | TiledPointsLayerBinding | ShapesLayerBinding
)


@dataclass(frozen=True)
class TiledPointsLoadResult:
    """Report whether a persistent tiled-points binding was created or reused."""

    binding: TiledPointsLayerBinding
    created: bool


@dataclass(frozen=True)
class PointsLayerIdentity:
    """Source identity for one points value-selection layer."""

    sdata: SpatialData
    points_name: str
    coordinate_system: str
    index_column: str

    def __post_init__(self) -> None:
        if self.sdata is None:
            raise ValueError("`sdata` must not be None.")
        if not isinstance(self.points_name, str) or not self.points_name:
            raise ValueError("`points_name` must be a non-empty string.")
        if not isinstance(self.coordinate_system, str) or not self.coordinate_system:
            raise ValueError("`coordinate_system` must be a non-empty string.")
        if not isinstance(self.index_column, str) or not self.index_column:
            raise ValueError("`index_column` must be a non-empty string.")


@dataclass(frozen=True)
class _ViewerCameraState:
    """Small snapshot of viewer camera state restored after layer replacement."""

    center: Any
    zoom: Any


@dataclass(frozen=True)
class _NapariShapesLayerInputs:
    """Prepared inputs for constructing one napari ``Shapes`` layer.

    Parameters
    ----------
    data
        One vertex array per rendered napari shape. This length can differ
        from the source GeoDataFrame row count because a ``MultiPolygon`` row
        expands into multiple rendered shapes, while empty, invalid, or
        unsupported geometries are skipped.
    shape_types
        Napari shape type for each row in ``data``.
    features
        DataFrame row-aligned to ``data``. It currently stores the source
        GeoDataFrame index for status-bar display.
    source_shapes_index_feature_name
        Name of the ``features`` column that stores the source GeoDataFrame
        index, using the GeoDataFrame index name or ``"index"`` fallback.
    source_row_id_by_rendered_row
        Internal integer source GeoDataFrame row id for each rendered napari
        row. This is later stored in the Spatiato layer binding so every row in
        ``data`` can be mapped back to its source row even when ``len(data)``
        differs from the source GeoDataFrame row count. For example, if source
        row position ``7`` is a ``MultiPolygon`` that expands into three
        rendered napari polygons and source row position ``8`` expands into one
        rendered napari polygon, this value is ``(7, 7, 7, 8)``.
    skipped_geometry_count
        Number of source rows that could not be rendered.
    """

    data: list[np.ndarray]
    shape_types: list[ShapesLayerShapeType]
    features: pd.DataFrame
    source_shapes_index_feature_name: str
    source_row_id_by_rendered_row: tuple[int, ...]
    skipped_geometry_count: int


@dataclass(frozen=True)
class _NapariPointRadiusShapesLayerInputs:
    """Prepared inputs for rendering point-radius shapes as napari ``Points``.

    Parameters
    ----------
    coordinates
        Point coordinates in napari `(y, x)` order, one row per source
        GeoDataFrame row.
    sizes
        Point diameters derived from the transformed source radius values.
    features
        DataFrame row-aligned to `coordinates`. It stores the source
        GeoDataFrame index for status-bar display.
    source_shapes_index_feature_name
        Name of the `features` column that stores the source GeoDataFrame
        index, using the GeoDataFrame index name or `"index"` fallback.
    source_row_id_by_rendered_row
        Internal integer source GeoDataFrame row id for each rendered napari
        point row. Point-radius rendering is one-to-one, so this is a `range`
        instead of a materialized tuple.
    skipped_geometry_count
        Number of source rows that could not be rendered. Qualifying
        point-radius inputs currently require all rows to render, so this is
        zero.
    """

    coordinates: np.ndarray
    sizes: np.ndarray
    features: pd.DataFrame
    source_shapes_index_feature_name: str
    source_row_id_by_rendered_row: range
    skipped_geometry_count: int


@dataclass(frozen=True)
class _BuiltShapesLayer:
    layer: Shapes | Points
    shapes_rendering_mode: ShapesRenderingMode
    source_shapes_index_feature_name: str
    source_row_id_by_rendered_row: SourceRowIdByRenderedRow
    skipped_geometry_count: int


class LayerBindingRegistry:
    """In-memory mapping from napari layers to Spatiato SpatialData element identity.

    Internally, this registry is keyed by the live napari layer object
    identity (``id(layer)``) and stores a ``LayerBinding`` value describing
    what that layer means from Spatiato's perspective.

    In other words, it maps:

    - a concrete napari layer object
    - to the corresponding ``SpatialData`` element identity and Spatiato-specific
      binding metadata for that layer

    That metadata includes the shared element identity:

    - ``sdata_id``
    - ``element_name``
    - ``element_type``
    - ``coordinate_system``

    and also the layer-type-specific semantics that Spatiato needs for lookup and
    viewer behavior, for example:

    - labels-layer role such as ``primary`` or ``styled``
    - styled-labels overlay metadata via ``style_spec``
    - image display mode such as ``stack`` or ``overlay``
    - overlay channel identity via ``channel_index`` / ``channel_name``
    - shapes-layer role such as ``primary`` or ``styled``
    - styled-shapes layer metadata via ``style_spec``

    The registry is Spatiato's source of truth for answering questions such as:

    - which napari layer represents a given labels element
    - what styling / role metadata is attached to a given labels layer
    - which napari layers belong to a given image element
    - whether an image is shown in stack mode or overlay mode
    - which overlay layer corresponds to which image channel

    This keeps Spatiato's layer lookup logic in one central place instead of
    relying on napari layer metadata as the primary contract.
    """

    def __init__(self) -> None:
        self._bindings: dict[int, LayerBinding] = {}

    def register_labels_layer(
        self,
        layer: Labels,
        *,
        element_name: str,
        coordinate_system: str | None = None,
        sdata: SpatialData | None = None,
        labels_role: Literal["primary", "styled"] = "primary",
        style_spec: TableColorSourceSpec | None = None,
    ) -> LabelsLayerBinding:
        """Register a labels layer binding."""
        binding = LabelsLayerBinding(
            layer=layer,
            element_name=element_name,
            coordinate_system=coordinate_system,
            sdata_id=_get_sdata_id(sdata),
            labels_role=labels_role,
            style_spec=style_spec,
        )
        self._register_binding(binding)
        return binding

    def register_image_layer(
        self,
        layer: Image,
        *,
        element_name: str,
        coordinate_system: str | None = None,
        sdata: SpatialData | None = None,
        image_display_mode: ImageDisplayMode | None = None,
        channel_index: int | None = None,
        channel_name: str | None = None,
    ) -> ImageLayerBinding:
        """Register an image layer binding."""
        binding = ImageLayerBinding(
            layer=layer,
            element_name=element_name,
            coordinate_system=coordinate_system,
            sdata_id=_get_sdata_id(sdata),
            image_display_mode=image_display_mode,
            channel_index=channel_index,
            channel_name=channel_name,
        )
        self._register_binding(binding)
        return binding

    def register_points_layer(
        self,
        layer: Points,
        *,
        element_name: str,
        coordinate_system: str | None = None,
        sdata: SpatialData | None = None,
        index_column: str,
    ) -> PointsLayerBinding:
        """Register a points layer binding."""
        binding = PointsLayerBinding(
            layer=layer,
            element_name=element_name,
            coordinate_system=coordinate_system,
            sdata_id=_get_sdata_id(sdata),
            index_column=index_column,
        )
        self._register_binding(binding)
        return binding

    def register_tiled_points_layer(
        self,
        layer: TiledPointsLayerModel,
        *,
        element_name: str,
        coordinate_system: str,
        sdata: SpatialData,
        value_column: str,
        cache_root: Path,
        cache_generation_id: str,
        runtime: _TiledPointsLayerRuntime,
    ) -> TiledPointsLayerBinding:
        """Register one cache-backed tiled-points layer binding."""
        binding = TiledPointsLayerBinding(
            layer=layer,
            element_name=element_name,
            coordinate_system=coordinate_system,
            sdata_id=id(sdata),
            value_column=value_column,
            cache_root=cache_root,
            cache_generation_id=cache_generation_id,
            runtime=runtime,
        )
        self._register_binding(binding)
        return binding

    def register_shapes_layer(
        self,
        layer: Shapes | Points,
        *,
        element_name: str,
        coordinate_system: str | None = None,
        sdata: SpatialData | None = None,
        shapes_role: Literal["primary", "styled"] = "primary",
        shapes_rendering_mode: ShapesRenderingMode = "shapes",
        style_spec: ShapeColumnColorSourceSpec | TableColorSourceSpec | None = None,
        source_row_id_by_rendered_row: SourceRowIdByRenderedRow = (),
        source_shapes_index_feature_name: str = DEFAULT_SHAPES_INDEX_FEATURE_NAME,
        skipped_geometry_count: int = 0,
    ) -> ShapesLayerBinding:
        """Register a shapes layer binding."""
        binding = ShapesLayerBinding(
            layer=layer,
            element_name=element_name,
            coordinate_system=coordinate_system,
            sdata_id=_get_sdata_id(sdata),
            shapes_role=shapes_role,
            shapes_rendering_mode=shapes_rendering_mode,
            style_spec=style_spec,
            source_row_id_by_rendered_row=source_row_id_by_rendered_row,
            source_shapes_index_feature_name=source_shapes_index_feature_name,
            skipped_geometry_count=skipped_geometry_count,
        )
        self._register_binding(binding)
        return binding

    def _register_binding(self, binding: LayerBinding) -> LayerBinding:
        """Store a binding in the registry."""
        self._bindings[id(binding.layer)] = binding
        return binding

    def unregister_layer(self, layer: Layer) -> LayerBinding | None:
        """Remove a layer binding if present."""
        return self._bindings.pop(id(layer), None)

    def get_binding(self, layer: Layer) -> LayerBinding | None:
        """Return the binding for a specific layer."""
        return self._bindings.get(id(layer))

    def iter_bindings(self) -> tuple[LayerBinding, ...]:
        """Return all registered bindings in insertion order."""
        return tuple(self._bindings.values())

    def find_bindings(
        self,
        *,
        sdata: SpatialData | None = None,
        element_name: str | None = None,
        element_type: ElementType | None = None,
        coordinate_system: str | None = None,
        labels_role: Literal["primary", "styled"] | None = None,
        shapes_role: Literal["primary", "styled"] | None = None,
        style_spec: ShapeColumnColorSourceSpec | TableColorSourceSpec | None = None,
        image_display_mode: ImageDisplayMode | None = None,
        channel_index: int | None = None,
        channel_name: str | None = None,
        points_index_column: str | None = None,
    ) -> list[LayerBinding]:
        """Find bindings matching the provided filters."""
        sdata_id = None if sdata is None else id(sdata)
        matches: list[LayerBinding] = []
        for binding in self._bindings.values():
            if sdata_id is not None and binding.sdata_id != sdata_id:
                continue
            if element_name is not None and binding.element_name != element_name:
                continue
            if element_type is not None and binding.element_type != element_type:
                continue
            if coordinate_system is not None and binding.coordinate_system != coordinate_system:
                continue
            if labels_role is not None:
                if not isinstance(binding, LabelsLayerBinding) or binding.labels_role != labels_role:
                    continue
            if shapes_role is not None:
                if not isinstance(binding, ShapesLayerBinding) or binding.shapes_role != shapes_role:
                    continue
            if style_spec is not None:
                if not isinstance(binding, (LabelsLayerBinding, ShapesLayerBinding)):
                    continue
                if binding.style_spec != style_spec:
                    continue
            if image_display_mode is not None:
                if not isinstance(binding, ImageLayerBinding) or binding.image_display_mode != image_display_mode:
                    continue
            if channel_index is not None:
                if not isinstance(binding, ImageLayerBinding) or binding.channel_index != channel_index:
                    continue
            if channel_name is not None:
                if not isinstance(binding, ImageLayerBinding) or binding.channel_name != channel_name:
                    continue
            if points_index_column is not None:
                if not isinstance(binding, PointsLayerBinding) or binding.index_column != points_index_column:
                    continue
            matches.append(binding)
        return matches


class ViewerAdapter(QObject):
    """Spatiato-owned service for viewer-facing layer lookup and activation.

    The adapter wraps a napari viewer and provides the Spatiato-level operations
    that care about what loaded layers *mean* in terms of `SpatialData`
    elements.

    It is responsible for tasks such as:

    - registering and unregistering Spatiato-managed layer bindings
    - resolving which loaded napari layer corresponds to a labels element
    - resolving which loaded napari layers correspond to an image element
    - activating a requested layer in the viewer

    The adapter does not use napari layer metadata as the primary contract.
    Instead, it relies on `LayerBindingRegistry` as the authoritative in-memory
    mapping from napari layers to `SpatialData` element identity.

    Primary Labels presentation
    ---------------------------
    A primary Labels layer is a shared presentation surface used by multiple
    spatiato workflows. It has no persistent styling owner or styling
    history. Each accepted styling operation updates the same live layer, and
    the most recently applied style is therefore the visible style.

    Loading or activating an existing primary layer is not a styling operation:
    its current colormap and features are preserved. A primary layer created
    because none exists starts with napari's default Labels presentation.
    Object Classification and Spatial Query can subsequently style that shared
    layer.

    This last-styling-wins contract also applies to asynchronous updates. For
    example, when classifier predictions finish, Object Classification may
    restyle the primary layer using its selected Color-by mode even when
    Spatial Query styled that layer earlier. Viewer styled overlays are
    separate layers with their own bindings and are unaffected.

    Image membership lifecycle
    --------------------------
    ``image_layers_changed`` is an invalidation signal for the set and order of
    live, usable image bindings. It is emitted after a stack or overlay binding
    is added to or removed from the viewer, and after loaded image layers are
    reordered. Consumers such as Viewer and Histogram re-query the binding
    registry and current viewer layers rather than relying on an event payload
    that may already be stale after removal.

    The signal represents membership only. It is not emitted for presentation
    changes such as visibility, colormap, or contrast limits; consumers that
    synchronize those properties subscribe directly to the corresponding
    napari layer events.
    """

    # Emitted when the set/order of live primary Labels layers changes.
    # Used by consumers that depend on the annotation-capable labels-layer
    # lifecycle, including Object Classification and Spatial Query.
    primary_labels_layers_changed = Signal()
    # Emitted after a primary shapes layer has a Spatiato binding while loaded in
    # the viewer. Consumers can rely on the binding registry being ready.
    primary_shapes_layer_registered = Signal(object)
    # Emitted when the set/order of live, usable image bindings changes.
    # Viewer and Histogram consumers should re-query current viewer membership;
    # the signal carries no payload to avoid stale binding data on removal.
    image_layers_changed = Signal()
    active_layer_changed = Signal(object)

    def __init__(self, viewer: Any | None = None, layer_bindings: LayerBindingRegistry | None = None) -> None:
        super().__init__()
        self._viewer = viewer
        self._layer_bindings = layer_bindings or LayerBindingRegistry()
        self._connect_layer_events()

    @property
    def layer_bindings(self) -> LayerBindingRegistry:
        """Return the shared layer-binding registry."""
        return self._layer_bindings

    def register_labels_layer(
        self,
        layer: Labels,
        *,
        labels_name: str,
        coordinate_system: str | None = None,
        sdata: SpatialData | None = None,
        labels_role: Literal["primary", "styled"] = "primary",
        style_spec: TableColorSourceSpec | None = None,
    ) -> LabelsLayerBinding:
        """Register a labels layer in the shared binding registry.

        For primary labels layers, registration itself may be the moment when a
        live napari layer becomes Spatiato-usable. This happens on the normal
        ``insert -> register`` path used by ``ensure_labels_loaded(...)`` and
        also when Spatiato discovers a pre-existing viewer layer and binds it
        later. Emit ``primary_labels_layers_changed`` here when the layer is
        already present in the viewer so those flows do not depend on the
        viewer's ``inserted`` event having seen a binding already.
        """
        binding = self._layer_bindings.register_labels_layer(
            layer,
            element_name=labels_name,
            coordinate_system=coordinate_system,
            sdata=sdata,
            labels_role=labels_role,
            style_spec=style_spec,
        )
        self._handle_registered_binding(binding)
        return binding

    def register_image_layer(
        self,
        layer: Image,
        *,
        image_name: str,
        coordinate_system: str | None = None,
        sdata: SpatialData | None = None,
        image_display_mode: ImageDisplayMode | None = None,
        channel_index: int | None = None,
        channel_name: str | None = None,
    ) -> ImageLayerBinding:
        """Register an image layer in the shared binding registry."""
        binding = self._layer_bindings.register_image_layer(
            layer,
            element_name=image_name,
            coordinate_system=coordinate_system,
            sdata=sdata,
            image_display_mode=image_display_mode,
            channel_index=channel_index,
            channel_name=channel_name,
        )
        self._handle_registered_binding(binding)
        return binding

    def register_points_layer(
        self,
        layer: Points,
        *,
        points_name: str,
        coordinate_system: str | None = None,
        sdata: SpatialData | None = None,
        index_column: str,
    ) -> PointsLayerBinding:
        """Register a points layer in the shared binding registry."""
        binding = self._layer_bindings.register_points_layer(
            layer,
            element_name=points_name,
            coordinate_system=coordinate_system,
            sdata=sdata,
            index_column=index_column,
        )
        self._handle_registered_binding(binding)
        return binding

    def register_tiled_points_binding(
        self,
        layer: TiledPointsLayerModel,
        *,
        points_name: str,
        coordinate_system: str,
        sdata: SpatialData,
        value_column: str,
        cache_root: Path,
        cache_generation_id: str,
        runtime: _TiledPointsLayerRuntime,
    ) -> TiledPointsLayerBinding:
        """Register a persistent cache-backed points layer."""
        return self._layer_bindings.register_tiled_points_layer(
            layer,
            element_name=points_name,
            coordinate_system=coordinate_system,
            sdata=sdata,
            value_column=value_column,
            cache_root=cache_root,
            cache_generation_id=cache_generation_id,
            runtime=runtime,
        )

    def register_shapes_layer(
        self,
        layer: Shapes | Points,
        *,
        shapes_name: str,
        coordinate_system: str | None = None,
        sdata: SpatialData | None = None,
        shapes_role: Literal["primary", "styled"] = "primary",
        shapes_rendering_mode: ShapesRenderingMode = "shapes",
        style_spec: ShapeColumnColorSourceSpec | TableColorSourceSpec | None = None,
        source_row_id_by_rendered_row: SourceRowIdByRenderedRow = (),
        source_shapes_index_feature_name: str = DEFAULT_SHAPES_INDEX_FEATURE_NAME,
        skipped_geometry_count: int = 0,
    ) -> ShapesLayerBinding:
        """Register a shapes layer in the shared binding registry."""
        binding = self._layer_bindings.register_shapes_layer(
            layer,
            element_name=shapes_name,
            coordinate_system=coordinate_system,
            sdata=sdata,
            shapes_role=shapes_role,
            shapes_rendering_mode=shapes_rendering_mode,
            style_spec=style_spec,
            source_row_id_by_rendered_row=source_row_id_by_rendered_row,
            source_shapes_index_feature_name=source_shapes_index_feature_name,
            skipped_geometry_count=skipped_geometry_count,
        )
        self._handle_registered_binding(binding)
        return binding

    def apply_primary_shapes_layer_style(self, layer: Shapes) -> None:
        """Apply Spatiato's editable primary shapes presentation to one layer."""
        _apply_primary_shapes_layer_style(layer)

    def normalize_native_shapes_layer_for_annotation(
        self,
        layer: Shapes,
        *,
        source_shapes_index_feature_name: str,
    ) -> Shapes:
        """Replace a native napari Shapes layer with Spatiato's status-aware layer."""
        if isinstance(layer, _SpatiatoShapes):
            layer._source_shapes_index_feature_name = source_shapes_index_feature_name
            return layer

        if not self._is_layer_loaded_in_viewer(layer):
            raise ValueError("Cannot normalize a Shapes layer that is not loaded in the viewer.")

        _normalize_native_shapes_layer_transform(layer)
        replacement = _build_spatiato_shapes_layer_from_native_layer(
            layer,
            source_shapes_index_feature_name=source_shapes_index_feature_name,
        )
        _remove_layer_from_viewer(self._viewer, layer)
        _add_layer_to_viewer(self._viewer, replacement)
        return replacement

    def sync_primary_shapes_layer_binding(
        self,
        layer: Shapes,
        *,
        sdata: SpatialData,
        shapes_name: str,
        coordinate_system: str,
        source_row_id_by_rendered_row: SourceRowIdByRenderedRow,
        source_shapes_index_feature_name: str,
    ) -> ShapesLayerBinding:
        """Refresh a loaded primary shapes layer binding without emitting a registration event."""
        return self._layer_bindings.register_shapes_layer(
            layer,
            element_name=shapes_name,
            coordinate_system=coordinate_system,
            sdata=sdata,
            shapes_role="primary",
            shapes_rendering_mode="shapes",
            source_row_id_by_rendered_row=source_row_id_by_rendered_row,
            source_shapes_index_feature_name=source_shapes_index_feature_name,
        )

    def _handle_registered_binding(self, binding: LayerBinding) -> None:
        if _is_primary_labels_binding(binding) and self._is_layer_loaded_in_viewer(binding.layer):
            self.primary_labels_layers_changed.emit()
        if _is_primary_shapes_binding(binding) and self._is_layer_loaded_in_viewer(binding.layer):
            self.primary_shapes_layer_registered.emit(binding)
        if _is_usable_image_binding(binding) and self._is_layer_loaded_in_viewer(binding.layer):
            self.image_layers_changed.emit()

    def unregister_layer(self, layer: Layer) -> LayerBinding | None:
        """Remove a layer from the shared binding registry."""
        binding = self._layer_bindings.unregister_layer(layer)
        if isinstance(binding, TiledPointsLayerBinding):
            binding.runtime.close()
        return binding

    def _connect_layer_events(self) -> None:
        """Keep the registry synchronized with the viewer's layer list when possible."""
        layers = getattr(self._viewer, "layers", None)
        events = getattr(layers, "events", None)
        if events is None:
            return

        for event_name, handler in (
            ("inserted", self._on_viewer_layer_inserted),
            ("removed", self._on_viewer_layer_removed),
            ("reordered", self._on_viewer_layers_reordered),
        ):
            event_emitter = getattr(events, event_name, None)
            connect = getattr(event_emitter, "connect", None)
            if callable(connect):
                connect(handler)

    def _on_viewer_layer_inserted(self, event: Any) -> None:
        """React to viewer-layer insertion when binding already exists.

        In the current built-in loading paths, Spatiato usually adds the napari
        layer to the viewer first and registers it second, so this handler is
        not the main signal path for primary labels or usable image
        availability. Keep it so the adapter still behaves correctly for
        external or future flows where a layer is registered before it is
        inserted into the viewer.
        """
        layer = getattr(event, "value", None)
        if not isinstance(layer, Layer):
            logger.warning("Ignoring viewer layer insertion event without a napari Layer payload.")
            return
        binding = self._layer_bindings.get_binding(layer)
        if _is_pickable_primary_labels_layer(layer, binding):
            self.primary_labels_layers_changed.emit()
        if _is_usable_image_binding(binding):
            self.image_layers_changed.emit()

    def _on_viewer_layer_removed(self, event: Any) -> None:
        """Unregister Spatiato-managed layers when they disappear from the viewer."""
        layer = getattr(event, "value", None)
        if not isinstance(layer, Layer):
            logger.warning("Ignoring viewer layer removal event without a napari Layer payload.")
            return
        binding = self._layer_bindings.get_binding(layer)
        had_primary_labels_semantics = _is_pickable_primary_labels_layer(layer, binding)
        had_usable_image_semantics = _is_usable_image_binding(binding)
        removed_binding = self.unregister_layer(layer)
        if removed_binding is None:
            logger.warning(
                "Removed napari layer `%s` had no matching Spatiato layer binding.", getattr(layer, "name", layer)
            )
        if had_primary_labels_semantics:
            self.primary_labels_layers_changed.emit()
        if had_usable_image_semantics:
            self.image_layers_changed.emit()

    def _on_viewer_layers_reordered(self, event: Any) -> None:
        del event
        if any(
            _is_pickable_primary_labels_layer(layer, self._layer_bindings.get_binding(layer))
            for layer in self._iter_candidate_layers()
        ):
            self.primary_labels_layers_changed.emit()
        if any(
            _is_usable_image_binding(self._layer_bindings.get_binding(layer)) for layer in self._iter_candidate_layers()
        ):
            self.image_layers_changed.emit()

    def activate_layer(self, layer: Layer | None) -> bool:
        """Make a layer active in the viewer when supported."""
        if layer is None:
            return False

        layers = getattr(self._viewer, "layers", None)
        selection = getattr(layers, "selection", None)
        if selection is None:
            return False

        select_only = getattr(selection, "select_only", None)
        if callable(select_only):
            select_only(layer)
            self.active_layer_changed.emit(layer)
            return True

        if hasattr(selection, "active"):
            selection.active = layer
            self.active_layer_changed.emit(layer)
            return True

        return False

    def get_loaded_primary_labels_layer(
        self,
        sdata: SpatialData,
        labels_name: str,
        coordinate_system: str | None = None,
    ) -> Labels | None:
        """Return the loaded primary labels layer for one labels element.

        This lookup is registry-backed only. A pre-existing viewer layer that
        happens to carry legacy ``metadata['sdata']`` / ``metadata['name']``
        fields is not considered loaded unless it has been explicitly
        registered in ``LayerBindingRegistry``.
        """
        for layer in self._iter_candidate_layers():
            if not _is_pickable_labels_layer(layer):
                continue

            binding = self._layer_bindings.get_binding(layer)
            if _matches_labels_binding(
                binding,
                sdata=sdata,
                element_name=labels_name,
                coordinate_system=coordinate_system,
                labels_role="primary",
            ):
                return layer

        return None

    def get_loaded_styled_labels_layer(
        self,
        sdata: SpatialData,
        labels_name: str,
        style_spec: TableColorSourceSpec,
        coordinate_system: str | None = None,
    ) -> Labels | None:
        """Return one loaded styled labels layer for a specific style variant."""
        for layer in self._iter_candidate_layers():
            if not _is_pickable_labels_layer(layer):
                continue

            binding = self._layer_bindings.get_binding(layer)
            if _matches_labels_binding(
                binding,
                sdata=sdata,
                element_name=labels_name,
                coordinate_system=coordinate_system,
                labels_role="styled",
                style_spec=style_spec,
            ):
                return layer

        return None

    def get_loaded_styled_labels_layers(
        self,
        sdata: SpatialData,
        labels_name: str,
        coordinate_system: str | None = None,
    ) -> list[Labels]:
        """Return all loaded styled labels layers for one labels element."""
        matches: list[Labels] = []
        for layer in self._iter_candidate_layers():
            if not _is_pickable_labels_layer(layer):
                continue

            binding = self._layer_bindings.get_binding(layer)
            if _matches_labels_binding(
                binding,
                sdata=sdata,
                element_name=labels_name,
                coordinate_system=coordinate_system,
                labels_role="styled",
            ):
                matches.append(layer)

        return matches

    def get_loaded_primary_shapes_layer(
        self,
        sdata: SpatialData,
        shapes_name: str,
        coordinate_system: str | None = None,
    ) -> Shapes | Points | None:
        """Return the loaded primary shapes layer for one shapes element."""
        for layer in self._iter_candidate_layers():
            if not _is_semantic_shapes_layer(layer):
                continue

            binding = self._layer_bindings.get_binding(layer)
            if _matches_shapes_binding(
                binding,
                sdata=sdata,
                element_name=shapes_name,
                coordinate_system=coordinate_system,
                shapes_role="primary",
            ):
                return layer

        return None

    def get_loaded_styled_shapes_layer(
        self,
        sdata: SpatialData,
        shapes_name: str,
        style_spec: ShapeColumnColorSourceSpec | TableColorSourceSpec,
        coordinate_system: str | None = None,
    ) -> Shapes | Points | None:
        """Return one loaded styled shapes layer for a specific style variant."""
        for layer in self._iter_candidate_layers():
            if not _is_semantic_shapes_layer(layer):
                continue

            binding = self._layer_bindings.get_binding(layer)
            if _matches_shapes_binding(
                binding,
                sdata=sdata,
                element_name=shapes_name,
                coordinate_system=coordinate_system,
                shapes_role="styled",
                style_spec=style_spec,
            ):
                return layer

        return None

    def get_loaded_styled_shapes_layers(
        self,
        sdata: SpatialData,
        shapes_name: str,
        coordinate_system: str | None = None,
    ) -> list[Shapes | Points]:
        """Return all loaded styled shapes layers for one shapes element."""
        matches: list[Shapes | Points] = []
        for layer in self._iter_candidate_layers():
            if not _is_semantic_shapes_layer(layer):
                continue

            binding = self._layer_bindings.get_binding(layer)
            if _matches_shapes_binding(
                binding,
                sdata=sdata,
                element_name=shapes_name,
                coordinate_system=coordinate_system,
                shapes_role="styled",
            ):
                matches.append(layer)

        return matches

    def get_loaded_image_layers(self, sdata: SpatialData, image_name: str) -> list[Image]:
        """Return loaded image layers for a SpatialData image element."""
        matches: list[Image] = []
        for layer in self._iter_candidate_layers():
            if not _is_image_layer(layer):
                continue

            binding = self._layer_bindings.get_binding(layer)
            if _matches_binding(binding, sdata=sdata, element_name=image_name, element_type="image"):
                matches.append(layer)

        return matches

    def _ensure_points_layer_from_selection(
        self,
        identity: PointsLayerIdentity,
        *,
        selection: PointsValueSelection,
        categorical_colors: Sequence[str] | None = None,
    ) -> PointsLoadResult:
        """Create or update the points value-selection layer for an already loaded selection."""
        if not isinstance(identity, PointsLayerIdentity):
            raise ValueError("`identity` must be a PointsLayerIdentity.")
        if identity.index_column != selection.index_column:
            raise ValueError(
                "`identity.index_column` and `selection.index_column` must match. "
                f"Got {identity.index_column!r} and {selection.index_column!r}."
            )

        old_layer = self._get_loaded_points_layer_for_identity(identity)
        created = old_layer is None
        visible = True if old_layer is None else old_layer.visible
        camera_state = _capture_viewer_camera_state(self._viewer) if old_layer is not None else None

        layer = _build_points_layer_from_selection(identity, selection)
        intended_layer_name = layer.name
        layer.visible = visible
        style_result = apply_points_selection_style(
            layer,
            selection,
            point_size=getattr(old_layer, "current_size", None),
            point_symbol=getattr(old_layer, "current_symbol", None),
            point_face_color=_get_preserved_points_face_color(old_layer, selection=selection),
            categorical_colors=categorical_colors,
        )

        try:
            _add_layer_to_viewer(self._viewer, layer)
            self.register_points_layer(
                layer,
                sdata=identity.sdata,
                points_name=identity.points_name,
                coordinate_system=identity.coordinate_system,
                index_column=identity.index_column,
            )
            if old_layer is not None:
                # Replacing the napari Points object avoids mutating a live
                # active layer while napari's hover/status thread may be
                # reading private view caches such as `_indices_view` and
                # `_view_size`. In-place mutation has produced stale-cache
                # errors when switching between point selections.
                self._remove_layer_from_viewer_and_registry(old_layer)
                layer.name = intended_layer_name
        finally:
            _restore_viewer_camera_state(self._viewer, camera_state)

        return PointsLoadResult(
            layer=layer,
            created=created,
            color_mode=style_result.color_mode,
            categorical_coloring_disabled=style_result.categorical_coloring_disabled,
            selected_value_count=style_result.selected_value_count,
            categorical_limit=style_result.categorical_limit,
        )

    def get_loaded_tiled_points_binding(
        self,
        sdata: SpatialData,
        points_name: str,
        coordinate_system: str,
        value_column: str,
    ) -> TiledPointsLayerBinding | None:
        """Return the live tiled binding for one logical points identity."""
        for layer in self._iter_candidate_layers():
            binding = self._layer_bindings.get_binding(layer)
            if (
                isinstance(binding, TiledPointsLayerBinding)
                and binding.sdata_id == id(sdata)
                and binding.element_name == points_name
                and binding.coordinate_system == coordinate_system
                and binding.value_column == value_column
            ):
                return binding
        return None

    def ensure_tiled_points_layer(
        self,
        *,
        sdata: SpatialData,
        points_name: str,
        coordinate_system: str,
        descriptor: TiledPointsCacheDescriptor,
        requested_value_ids: tuple[int, ...] | None,
        hard_render_point_budget: int,
        settings: TiledPointsApplicationSettings,
    ) -> TiledPointsLoadResult:
        """Create or update one persistent cache-backed points layer."""
        info = descriptor.dataset_info
        if info.points_name != points_name:
            raise ValueError(
                f"Cache {descriptor.cache_root} describes points element {info.points_name!r}, "
                f"not selected element {points_name!r}."
            )
        existing = self.get_loaded_tiled_points_binding(
            sdata,
            points_name,
            coordinate_system,
            info.value_column,
        )
        if existing is not None and existing.runtime.closed:
            self._remove_layer_from_viewer_and_registry(existing.layer)
            existing = None
        if existing is not None and existing.cache_generation_id == info.cache_generation_id:
            existing.layer.hard_render_point_budget = hard_render_point_budget
            existing.runtime.set_selected_value_ids(requested_value_ids)
            return TiledPointsLoadResult(binding=existing, created=False)
        if existing is not None:
            self._remove_layer_from_viewer_and_registry(existing.layer)

        affine = _get_points_affine_transform(sdata, points_name, coordinate_system)
        if affine is None:
            raise ValueError(
                f"Coordinate system {coordinate_system!r} is not available for points element {points_name!r}."
            )
        register_tiled_points_layer()
        layer = TiledPointsLayerModel(
            descriptor.dataset_reference,
            value_palette=canonical_value_palette(len(info.value_names)),
            max_vertex_payload_bytes=settings.max_vertex_payload_bytes,
            affine=affine,
            name=points_name,
            hard_render_point_budget=hard_render_point_budget,
        )
        runtime = _TiledPointsLayerRuntime(
            layer,
            descriptor.cache_root,
            settings.cache_session_settings,
            initial_requested_value_ids=requested_value_ids,
        )
        try:
            _add_layer_to_viewer(self._viewer, layer)
            binding = self.register_tiled_points_binding(
                layer,
                points_name=points_name,
                coordinate_system=coordinate_system,
                sdata=sdata,
                value_column=info.value_column,
                cache_root=descriptor.cache_root,
                cache_generation_id=info.cache_generation_id,
                runtime=runtime,
            )
        except Exception:
            runtime.close()
            if self._is_layer_loaded_in_viewer(layer):
                _remove_layer_from_viewer(self._viewer, layer)
            raise
        return TiledPointsLoadResult(binding=binding, created=True)

    def ensure_labels_loaded(self, sdata: SpatialData, labels_name: str, coordinate_system: str) -> LabelsLoadResult:
        """Load a primary Labels layer without restyling an existing one.

        An existing primary layer is returned with its current colormap and
        features unchanged. When no primary layer exists, a new layer is
        created with napari's default Labels presentation.
        """
        existing_layer = self._get_loaded_labels_layer_for_coordinate_system(sdata, labels_name, coordinate_system)
        if existing_layer is not None:
            return LabelsLoadResult(
                layer=existing_layer,
                created=False,
                value_kind=None,
                palette_source=None,
                coercion_applied=False,
            )

        layer = _build_labels_layer(sdata, labels_name, coordinate_system, name=labels_name)
        _add_layer_to_viewer(self._viewer, layer)
        self.register_labels_layer(
            layer,
            sdata=sdata,
            labels_name=labels_name,
            coordinate_system=coordinate_system,
        )
        return LabelsLoadResult(
            layer=layer,
            created=True,
            value_kind=None,
            palette_source=None,
            coercion_applied=False,
        )

    def ensure_styled_labels_loaded(
        self,
        sdata: SpatialData,
        labels_name: str,
        coordinate_system: str,
        style_spec: TableColorSourceSpec,
    ) -> LabelsLoadResult:
        """Load or update one styled labels overlay variant."""
        existing_layer = self.get_loaded_styled_labels_layer(
            sdata,
            labels_name,
            style_spec,
            coordinate_system,
        )
        created = existing_layer is None
        if existing_layer is None:
            layer = _build_labels_layer(
                sdata,
                labels_name,
                coordinate_system,
                name=build_styled_labels_layer_name(labels_name, style_spec),
            )
            _add_layer_to_viewer(self._viewer, layer)
            self.register_labels_layer(
                layer,
                sdata=sdata,
                labels_name=labels_name,
                coordinate_system=coordinate_system,
                labels_role="styled",
                style_spec=style_spec,
            )
        else:
            layer = existing_layer

        layer.name = build_styled_labels_layer_name(labels_name, style_spec)
        # This method backs the viewer's "Add / Update in viewer" action:
        # existing styled overlays are intentionally restyled from the current
        # table state rather than returned unchanged.
        style_result = apply_table_color_source_to_labels_layer(
            layer,
            sdata=sdata,
            labels_name=labels_name,
            style_spec=style_spec,
        )
        return LabelsLoadResult(
            layer=layer,
            created=created,
            value_kind=style_result.value_kind,
            palette_source=style_result.palette_source,
            coercion_applied=style_result.coercion_applied,
        )

    def ensure_image_loaded(
        self,
        sdata: SpatialData,
        image_name: str,
        coordinate_system: str,
        *,
        mode: ImageDisplayMode = "stack",
        channels: Sequence[int | str] | None = None,
        channel_colors: Sequence[str] | None = None,
    ) -> ImageLoadResult:
        """Load an image element into napari if it is not already present."""
        images = getattr(sdata, "images", {})
        if image_name not in images:
            raise ValueError(f"Image element `{image_name}` is not available in the selected SpatialData object.")

        image_element = images[image_name]
        available_coordinate_systems = set(get_transformation(image_element, get_all=True).keys())
        if coordinate_system not in available_coordinate_systems:
            raise ValueError(
                f"Coordinate system `{coordinate_system}` is not available for image element `{image_name}`."
            )

        if mode == "stack":
            existing_overlay_layers = self._get_loaded_image_layer_for_coordinate_system(
                sdata,
                image_name,
                coordinate_system,
                image_display_mode="overlay",
            )
            for layer in existing_overlay_layers:
                self._remove_layer_from_viewer_and_registry(layer)

            layers = self._get_loaded_image_layer_for_coordinate_system(
                sdata,
                image_name,
                coordinate_system,
                image_display_mode=mode,
            )
            existing_layer = layers[0] if layers else None
            if existing_layer is not None:
                return ImageLoadResult(
                    layers=(existing_layer,),
                    mode=mode,
                    created=False,
                )

            image_data, rgb = _get_stack_image_layer_data(image_element)
            layer = Image(
                _unwrap_xarray_data_for_napari(image_data),
                name=image_name,
                affine=_get_affine_transform(image_element, coordinate_system),
                rgb=rgb,
            )
            _add_layer_to_viewer(self._viewer, layer)
            self.register_image_layer(
                layer,
                sdata=sdata,
                image_name=image_name,
                coordinate_system=coordinate_system,
                image_display_mode=mode,
            )
            return ImageLoadResult(
                layers=(layer,),
                mode=mode,
                created=True,
            )

        if mode != "overlay":
            raise NotImplementedError(f"Image display mode `{mode}` is not implemented yet.")

        resolved_channels = _resolve_overlay_channels(image_element, channels)
        resolved_colors = _resolve_overlay_colors(len(resolved_channels), channel_colors)
        affine = _get_affine_transform(image_element, coordinate_system)

        loaded_overlay_layers: list[Image] = []
        created = False
        desired_channel_indices = {channel_index for channel_index, _ in resolved_channels}

        layers = self._get_loaded_image_layer_for_coordinate_system(
            sdata,
            image_name,
            coordinate_system,
            image_display_mode="stack",
        )
        existing_stack_layer = layers[0] if layers else None
        if existing_stack_layer is not None:
            self._remove_layer_from_viewer_and_registry(existing_stack_layer)

        existing_overlay_layers = self._get_loaded_image_layer_for_coordinate_system(
            sdata,
            image_name,
            coordinate_system,
            image_display_mode="overlay",
        )
        for layer in existing_overlay_layers:
            binding = self._layer_bindings.get_binding(layer)
            assert isinstance(binding, ImageLayerBinding)
            if binding.channel_index in desired_channel_indices:
                continue
            self._remove_layer_from_viewer_and_registry(layer)

        for (channel_index, channel_name), color in zip(resolved_channels, resolved_colors, strict=False):
            layers = self._get_loaded_image_layer_for_coordinate_system(
                sdata,
                image_name,
                coordinate_system,
                image_display_mode=mode,
                channel_index=channel_index,
            )
            existing_layer = layers[0] if layers else None
            if existing_layer is None:
                created = True
                channel_data = _get_overlay_channel_layer_data(image_element, channel_index)
                layer = Image(
                    _unwrap_xarray_data_for_napari(channel_data),
                    name=f"{image_name}[{channel_name}]",
                    affine=affine,
                    blending="additive",
                    colormap=color,
                )
                _add_layer_to_viewer(self._viewer, layer)
                self.register_image_layer(
                    layer,
                    sdata=sdata,
                    image_name=image_name,
                    coordinate_system=coordinate_system,
                    image_display_mode=mode,
                    channel_index=channel_index,
                    channel_name=channel_name,
                )
            else:
                layer = existing_layer
                layer.name = f"{image_name}[{channel_name}]"
                layer.blending = "additive"
                layer.colormap = color

            loaded_overlay_layers.append(layer)

        return ImageLoadResult(
            layers=tuple(loaded_overlay_layers),
            mode="overlay",
            created=created,
            channels=tuple(channel_index for channel_index, _ in resolved_channels),
            channel_names=tuple(channel_name for _, channel_name in resolved_channels),
        )

    def ensure_image_overlay_channel_loaded(
        self,
        sdata: SpatialData,
        image_name: str,
        coordinate_system: str,
        *,
        channel: int | str,
        channel_color: str,
    ) -> ImageLoadResult:
        """Load or update one overlay channel while preserving loaded sibling channels."""
        images = getattr(sdata, "images", {})
        if image_name not in images:
            raise ValueError(f"Image element `{image_name}` is not available in the selected SpatialData object.")

        image_element = images[image_name]
        available_coordinate_systems = set(get_transformation(image_element, get_all=True).keys())
        if coordinate_system not in available_coordinate_systems:
            raise ValueError(
                f"Coordinate system `{coordinate_system}` is not available for image element `{image_name}`."
            )

        requested_channel_index, _requested_channel_name = _resolve_overlay_channels(image_element, [channel])[0]
        channels: list[int] = [requested_channel_index]
        colors: list[str] = [channel_color]
        seen_channel_indices = {requested_channel_index}

        existing_overlay_layers = self._get_loaded_image_layer_for_coordinate_system(
            sdata,
            image_name,
            coordinate_system,
            image_display_mode="overlay",
        )
        for layer in existing_overlay_layers:
            binding = self._layer_bindings.get_binding(layer)
            assert isinstance(binding, ImageLayerBinding)
            if binding.channel_index is None or binding.channel_index in seen_channel_indices:
                continue

            channels.append(binding.channel_index)
            colors.append(
                _image_layer_colormap_name(
                    layer,
                    fallback=DEFAULT_OVERLAY_COLORS[binding.channel_index % len(DEFAULT_OVERLAY_COLORS)],
                )
            )
            seen_channel_indices.add(binding.channel_index)

        return self.ensure_image_loaded(
            sdata,
            image_name,
            coordinate_system,
            mode="overlay",
            channels=channels,
            channel_colors=colors,
        )

    def ensure_shapes_loaded(self, sdata: SpatialData, shapes_name: str, coordinate_system: str) -> ShapesLoadResult:
        """Load a shapes element into napari if it is not already present."""
        existing_layer = self._get_loaded_shapes_layer_for_coordinate_system(sdata, shapes_name, coordinate_system)
        if existing_layer is not None:
            binding = self._layer_bindings.get_binding(existing_layer)
            if not isinstance(binding, ShapesLayerBinding):
                raise ValueError("Loaded shapes layer is missing its Spatiato shapes binding.")
            return ShapesLoadResult(
                layer=existing_layer,
                created=False,
                value_kind=None,
                palette_source=None,
                coercion_applied=False,
                skipped_geometry_count=binding.skipped_geometry_count,
                shapes_rendering_mode=binding.shapes_rendering_mode,
            )

        built_layer = _build_shapes_layer(
            sdata,
            shapes_name,
            coordinate_system,
            name=shapes_name,
            sync_current_colors=True,
        )
        layer = built_layer.layer
        _add_layer_to_viewer(self._viewer, layer)
        try:
            self.register_shapes_layer(
                layer,
                sdata=sdata,
                shapes_name=shapes_name,
                coordinate_system=coordinate_system,
                shapes_rendering_mode=built_layer.shapes_rendering_mode,
                source_row_id_by_rendered_row=built_layer.source_row_id_by_rendered_row,
                source_shapes_index_feature_name=built_layer.source_shapes_index_feature_name,
                skipped_geometry_count=built_layer.skipped_geometry_count,
            )
        except Exception:
            # The layer is already visible in napari. Remove it so failed Spatiato
            # registration does not leave an unbound Spatiato-created layer for the
            # Annotation widget's native-layer adoption listener to react to.
            _remove_layer_after_failed_registration(self._viewer, layer)
            raise
        return ShapesLoadResult(
            layer=layer,
            created=True,
            value_kind=None,
            palette_source=None,
            coercion_applied=False,
            skipped_geometry_count=built_layer.skipped_geometry_count,
            shapes_rendering_mode=built_layer.shapes_rendering_mode,
        )

    def replace_primary_shapes_layer(
        self,
        sdata: SpatialData,
        shapes_name: str,
        coordinate_system: str,
    ) -> ShapesLoadResult:
        """Register a complete replacement before removing the loaded primary layer.

        Construction or registration failure removes only the incomplete
        replacement and leaves the previously loaded layer and binding intact.
        """
        previous_layer = self._get_loaded_shapes_layer_for_coordinate_system(
            sdata,
            shapes_name,
            coordinate_system,
        )
        built_layer = _build_shapes_layer(
            sdata,
            shapes_name,
            coordinate_system,
            name=shapes_name,
            sync_current_colors=True,
        )
        replacement_layer = built_layer.layer
        _add_layer_to_viewer(self._viewer, replacement_layer)
        try:
            self.register_shapes_layer(
                replacement_layer,
                sdata=sdata,
                shapes_name=shapes_name,
                coordinate_system=coordinate_system,
                shapes_rendering_mode=built_layer.shapes_rendering_mode,
                source_row_id_by_rendered_row=built_layer.source_row_id_by_rendered_row,
                source_shapes_index_feature_name=built_layer.source_shapes_index_feature_name,
                skipped_geometry_count=built_layer.skipped_geometry_count,
            )
        except Exception:
            _remove_layer_after_failed_registration(self._viewer, replacement_layer)
            raise

        if previous_layer is not None:
            try:
                self._remove_layer_from_viewer_and_registry(previous_layer)
            except Exception:
                self._remove_layer_from_viewer_and_registry(replacement_layer)
                raise
            # Napari adds a uniqueness suffix while the previous same-named
            # layer is still present (for example, ``shapes [1]``). Restore
            # the SpatialData element name after the safe replacement removes
            # that temporary name collision.
            replacement_layer.name = shapes_name

        return ShapesLoadResult(
            layer=replacement_layer,
            created=True,
            value_kind=None,
            palette_source=None,
            coercion_applied=False,
            skipped_geometry_count=built_layer.skipped_geometry_count,
            shapes_rendering_mode=built_layer.shapes_rendering_mode,
        )

    def create_empty_primary_shapes_layer(
        self,
        sdata: SpatialData,
        shapes_name: str,
        coordinate_system: str,
        *,
        source_shapes_index_feature_name: str = DEFAULT_SHAPES_INDEX_FEATURE_NAME,
    ) -> Shapes:
        """Create and register an empty primary polygon shapes layer."""
        existing_layer = self._get_loaded_shapes_layer_for_coordinate_system(sdata, shapes_name, coordinate_system)
        if existing_layer is not None:
            raise ValueError(
                f"Shapes layer `{shapes_name}` is already loaded in coordinate system `{coordinate_system}`."
            )

        layer = _build_empty_primary_shapes_layer(
            name=shapes_name,
            source_shapes_index_feature_name=source_shapes_index_feature_name,
        )
        _add_layer_to_viewer(self._viewer, layer)
        try:
            self.register_shapes_layer(
                layer,
                sdata=sdata,
                shapes_name=shapes_name,
                coordinate_system=coordinate_system,
                shapes_rendering_mode="shapes",
                source_shapes_index_feature_name=source_shapes_index_feature_name,
            )
        except Exception:
            # The layer is already visible in napari. Remove it so failed Spatiato
            # registration does not leave an unbound Spatiato-created layer for the
            # Annotation widget's native-layer adoption listener to react to.
            _remove_layer_after_failed_registration(self._viewer, layer)
            raise
        return layer

    def ensure_styled_shapes_loaded(
        self,
        sdata: SpatialData,
        shapes_name: str,
        coordinate_system: str,
        style_spec: ShapeColumnColorSourceSpec | TableColorSourceSpec,
        *,
        fill: bool = False,
    ) -> ShapesLoadResult:
        """Load or update one styled shapes layer variant."""
        existing_layer = self.get_loaded_styled_shapes_layer(
            sdata,
            shapes_name,
            style_spec,
            coordinate_system,
        )
        created = existing_layer is None
        if existing_layer is None:
            built_layer = _build_shapes_layer(
                sdata,
                shapes_name,
                coordinate_system,
                name=build_styled_shapes_layer_name(shapes_name, style_spec),
                sync_current_colors=False,
            )
            layer = built_layer.layer
            _add_layer_to_viewer(self._viewer, layer)
            try:
                binding = self.register_shapes_layer(
                    layer,
                    sdata=sdata,
                    shapes_name=shapes_name,
                    coordinate_system=coordinate_system,
                    shapes_role="styled",
                    shapes_rendering_mode=built_layer.shapes_rendering_mode,
                    style_spec=style_spec,
                    source_row_id_by_rendered_row=built_layer.source_row_id_by_rendered_row,
                    source_shapes_index_feature_name=built_layer.source_shapes_index_feature_name,
                    skipped_geometry_count=built_layer.skipped_geometry_count,
                )
            except Exception:
                # The layer is already visible in napari. Remove it so failed Spatiato
                # registration does not leave an unbound Spatiato-created layer for the
                # Annotation widget's native-layer adoption listener to react to.
                _remove_layer_after_failed_registration(self._viewer, layer)
                raise
        else:
            layer = existing_layer
            binding = self._layer_bindings.get_binding(layer)
            if not isinstance(binding, ShapesLayerBinding):
                raise ValueError("Styled shapes layer is missing its Spatiato shapes binding.")

        layer.name = build_styled_shapes_layer_name(shapes_name, style_spec)
        if isinstance(style_spec, ShapeColumnColorSourceSpec):
            shapes = getattr(sdata, "shapes", {})
            if shapes_name not in shapes:
                raise ValueError(f"Shapes element `{shapes_name}` is not available in the selected SpatialData object.")

            style_result = apply_shape_column_color_source_to_shapes_layer(
                layer,
                shapes_element=shapes[shapes_name],
                style_spec=style_spec,
                source_row_id_by_rendered_row=binding.source_row_id_by_rendered_row,
                source_shapes_index_feature_name=binding.source_shapes_index_feature_name,
                fill=fill,
            )
        elif isinstance(style_spec, TableColorSourceSpec):
            style_result = apply_table_color_source_to_shapes_layer(
                layer,
                sdata=sdata,
                shapes_name=shapes_name,
                style_spec=style_spec,
                source_row_id_by_rendered_row=binding.source_row_id_by_rendered_row,
                source_shapes_index_feature_name=binding.source_shapes_index_feature_name,
                fill=fill,
            )
        else:
            raise ValueError(f"Unsupported styled shapes color source spec: {style_spec!r}.")
        return ShapesLoadResult(
            layer=layer,
            created=created,
            value_kind=style_result.value_kind,
            palette_source=style_result.palette_source,
            coercion_applied=style_result.coercion_applied,
            unannotated_source_shape_count=style_result.unannotated_source_shape_count,
            unannotated_rendered_shape_count=style_result.unannotated_rendered_shape_count,
            skipped_geometry_count=binding.skipped_geometry_count,
            shapes_rendering_mode=binding.shapes_rendering_mode,
        )

    def remove_labels_layer(self, sdata: SpatialData, labels_name: str, coordinate_system: str) -> Labels | None:
        """Remove the loaded labels layer for one labels element in one coordinate system."""
        layer = self._get_loaded_labels_layer_for_coordinate_system(sdata, labels_name, coordinate_system)
        if layer is None:
            return None

        self._remove_layer_from_viewer_and_registry(layer)
        return layer

    def remove_image_layers(self, sdata: SpatialData, image_name: str, coordinate_system: str) -> list[Image]:
        """Remove loaded stack and overlay layers for one image in one coordinate system."""
        removed_layers: list[Image] = []
        for image_display_mode in ("stack", "overlay"):
            layers = self._get_loaded_image_layer_for_coordinate_system(
                sdata,
                image_name,
                coordinate_system,
                image_display_mode=image_display_mode,
            )
            for layer in layers:
                self._remove_layer_from_viewer_and_registry(layer)
                removed_layers.append(layer)

        return removed_layers

    def remove_image_stack_layer(
        self,
        sdata: SpatialData,
        image_name: str,
        coordinate_system: str,
    ) -> Image | None:
        """Remove one loaded stack while preserving any unrelated image layers."""
        layers = self._get_loaded_image_layer_for_coordinate_system(
            sdata,
            image_name,
            coordinate_system,
            image_display_mode="stack",
        )
        if not layers:
            return None
        if len(layers) > 1:
            raise ValueError(
                f"Found multiple live stack layers for image `{image_name}` "
                f"and coordinate system `{coordinate_system}`."
            )

        layer = layers[0]
        self._remove_layer_from_viewer_and_registry(layer)
        return layer

    def remove_image_overlay_channel(
        self,
        sdata: SpatialData,
        image_name: str,
        coordinate_system: str,
        *,
        channel_index: int,
    ) -> Image | None:
        """Remove one loaded overlay channel while preserving sibling layers."""
        if channel_index < 0:
            raise ValueError("`channel_index` must be non-negative.")

        layers = self._get_loaded_image_layer_for_coordinate_system(
            sdata,
            image_name,
            coordinate_system,
            image_display_mode="overlay",
            channel_index=channel_index,
        )
        if not layers:
            return None
        if len(layers) > 1:
            raise ValueError(
                f"Found multiple live overlay layers for image `{image_name}`, "
                f"coordinate system `{coordinate_system}`, and channel index {channel_index}."
            )

        layer = layers[0]
        self._remove_layer_from_viewer_and_registry(layer)
        return layer

    def remove_shapes_layer(
        self, sdata: SpatialData, shapes_name: str, coordinate_system: str
    ) -> Shapes | Points | None:
        """Remove the loaded shapes layer for one shapes element in one coordinate system."""
        layer = self._get_loaded_shapes_layer_for_coordinate_system(sdata, shapes_name, coordinate_system)
        if layer is None:
            return None

        self._remove_layer_from_viewer_and_registry(layer)
        return layer

    def remove_layers_outside_coordinate_system(
        self,
        *,
        sdata: SpatialData | None,
        coordinate_system: str | None,
    ) -> list[LayerBinding]:
        """Remove Spatiato-managed layers that do not belong to the active coordinate system."""
        removed_bindings: list[LayerBinding] = []
        for binding in self._layer_bindings.iter_bindings():
            if sdata is not None and binding.sdata_id != id(sdata):
                continue
            if coordinate_system is not None and binding.coordinate_system == coordinate_system:
                continue
            removed_bindings.append(binding)
            self._remove_layer_from_viewer_and_registry(binding.layer)

        return removed_bindings

    def remove_layers_for_sdata(self, sdata: SpatialData | None) -> list[LayerBinding]:
        """Remove all Spatiato-managed layers for one SpatialData object."""
        if sdata is None:
            return []

        removed_bindings: list[LayerBinding] = []
        for binding in self._layer_bindings.iter_bindings():
            if binding.sdata_id != id(sdata):
                continue
            removed_bindings.append(binding)
            self._remove_layer_from_viewer_and_registry(binding.layer)

        return removed_bindings

    def close_tiled_points_runtimes(self) -> None:
        """Terminally close every registered cache-backed points runtime."""
        for binding in self._layer_bindings.iter_bindings():
            if isinstance(binding, TiledPointsLayerBinding):
                binding.runtime.close()

    def _iter_candidate_layers(self) -> Iterable[Layer]:
        layers = getattr(self._viewer, "layers", None)
        if layers is not None:
            return tuple(layers)
        return ()

    def _is_layer_loaded_in_viewer(self, layer: Layer) -> bool:
        return any(candidate is layer for candidate in self._iter_candidate_layers())

    def _remove_layer_from_viewer_and_registry(self, layer: Layer) -> None:
        """Remove a layer from the viewer and keep the registry synchronized."""
        _remove_layer_from_viewer(self._viewer, layer)
        # On the normal path, viewer removal emits `layers.events.removed` and
        # `_on_viewer_layer_removed(...)` has already unregistered the binding.
        # Keep this fallback for viewer-like objects that remove layers without
        # exposing or emitting the napari removal event.
        # So in short: normal path is event-driven.
        # We do the manual cleanup as a safety net.
        if self._layer_bindings.get_binding(layer) is not None:
            binding = self.unregister_layer(layer)
            if _is_primary_labels_binding(binding):
                self.primary_labels_layers_changed.emit()
            if _is_usable_image_binding(binding):
                self.image_layers_changed.emit()

    def _get_loaded_labels_layer_for_coordinate_system(
        self,
        sdata: SpatialData,
        labels_name: str,
        coordinate_system: str,
    ) -> Labels | None:
        return self.get_loaded_primary_labels_layer(sdata, labels_name, coordinate_system)

    def _get_loaded_image_layer_for_coordinate_system(
        self,
        sdata: SpatialData,
        image_name: str,
        coordinate_system: str,
        *,
        image_display_mode: ImageDisplayMode,
        channel_index: int | None = None,
    ) -> list[Image]:
        matches: list[Image] = []
        for layer in self._iter_candidate_layers():
            if not _is_image_layer(layer):
                continue

            binding = self._layer_bindings.get_binding(layer)
            if not _matches_binding(binding, sdata=sdata, element_name=image_name, element_type="image"):
                continue
            if not isinstance(binding, ImageLayerBinding):
                continue
            if binding.coordinate_system != coordinate_system:
                continue
            if binding.image_display_mode != image_display_mode:
                continue
            if channel_index is not None and binding.channel_index != channel_index:
                continue
            matches.append(layer)

        return matches

    def _get_loaded_points_layer_for_identity(self, identity: PointsLayerIdentity) -> Points | None:
        for layer in self._iter_candidate_layers():
            if not _is_points_layer(layer):
                continue

            binding = self._layer_bindings.get_binding(layer)
            if _matches_points_binding(binding, identity=identity):
                return layer

        return None

    def _get_loaded_shapes_layer_for_coordinate_system(
        self,
        sdata: SpatialData,
        shapes_name: str,
        coordinate_system: str,
    ) -> Shapes | Points | None:
        return self.get_loaded_primary_shapes_layer(sdata, shapes_name, coordinate_system)


def _get_sdata_id(sdata: SpatialData | None) -> int | None:
    return None if sdata is None else id(sdata)


def _build_points_layer_from_selection(identity: PointsLayerIdentity, selection: PointsValueSelection) -> Points:
    layer = Points(
        selection.coordinates,
        ndim=2,
        name=build_points_selection_layer_name(
            identity.points_name,
            identity.index_column,
            selection,
        ),
        affine=_get_points_affine_transform(identity.sdata, identity.points_name, identity.coordinate_system),
        features=selection.features,
        size=1.0,
        opacity=0.8,
        symbol="disc",
        border_width=0,
        face_color=POINTS_SELECTION_SOLID_COLOR,
    )
    return layer


def _get_points_affine_transform(
    sdata: SpatialData,
    points_name: str,
    coordinate_system: str,
) -> np.ndarray | None:
    points = getattr(sdata, "points", {}).get(points_name)
    if points is None:
        return None

    transform = get_transformation(points, get_all=True).get(coordinate_system)
    if transform is None:
        return None

    return transform.to_affine_matrix(input_axes=("y", "x"), output_axes=("y", "x"))


def _get_preserved_points_face_color(layer: Points | None, *, selection: PointsValueSelection) -> Any | None:
    if layer is None:
        return None

    selected_value_count = len(selection.selected_values)
    selection_uses_solid_color = (
        selected_value_count == 0 or selected_value_count > POINTS_SELECTION_MAX_CATEGORICAL_COLORS
    )
    selection_accepts_single_face_color = selection_uses_solid_color or selected_value_count == 1
    # A user override is a single layer-wide color; only carry it into new
    # selections where that cannot flatten a meaningful multi-value palette.
    if not selection_accepts_single_face_color:
        return None

    if getattr(layer, _POINTS_FACE_COLOR_OVERRIDE_ATTR, False):
        return getattr(layer, "current_face_color", None)

    if not selection_uses_solid_color or getattr(layer, "face_color_mode", None) != "direct":
        return None
    return getattr(layer, "current_face_color", None)


def _capture_viewer_camera_state(viewer: Any | None) -> _ViewerCameraState | None:
    camera = getattr(viewer, "camera", None)
    if camera is None:
        return None

    missing = object()
    center = getattr(camera, "center", missing)
    zoom = getattr(camera, "zoom", missing)
    if center is missing or zoom is missing:
        return None
    return _ViewerCameraState(center=center, zoom=zoom)


def _restore_viewer_camera_state(viewer: Any | None, camera_state: _ViewerCameraState | None) -> None:
    if camera_state is None:
        return

    camera = getattr(viewer, "camera", None)
    if camera is None:
        return

    try:
        camera.center = camera_state.center
        camera.zoom = camera_state.zoom
    except (AttributeError, RuntimeError, TypeError, ValueError):  # pragma: no cover - defensive viewer fallback
        logger.debug("Could not restore viewer camera after replacing the points layer.", exc_info=True)


def _add_layer_to_viewer(viewer: Any | None, layer: Layer) -> None:
    if viewer is None:
        raise ValueError("A napari viewer is required to load layers into the viewer.")

    add_layer = getattr(viewer, "add_layer", None)
    if callable(add_layer):
        add_layer(layer)
        return

    layers = getattr(viewer, "layers", None)
    if layers is None:
        raise ValueError("The provided viewer does not expose napari-compatible layer APIs.")

    append = getattr(layers, "append", None)
    if callable(append):
        append(layer)
        events = getattr(layers, "events", None)
        inserted = getattr(events, "inserted", None)
        if inserted is not None and hasattr(inserted, "emit"):
            inserted.emit(layer)
        return

    raise ValueError("The provided viewer does not support adding layers.")


def _remove_layer_from_viewer(viewer: Any | None, layer: Layer) -> None:
    if viewer is None:
        raise ValueError("A napari viewer is required to remove layers from the viewer.")

    layers = getattr(viewer, "layers", None)
    if layers is None:
        raise ValueError("The provided viewer does not expose napari-compatible layer APIs.")

    remove = getattr(layers, "remove", None)
    if callable(remove):
        remove(layer)
        return

    raise ValueError("The provided viewer does not support removing layers.")


def _remove_layer_after_failed_registration(viewer: Any | None, layer: Layer) -> None:
    try:
        _remove_layer_from_viewer(viewer, layer)
    except (AttributeError, RuntimeError, TypeError, ValueError):  # pragma: no cover - defensive cleanup fallback
        logger.debug("Could not remove napari layer after Spatiato registration failed.", exc_info=True)


def _get_stack_image_layer_data(element: DataArray | DataTree) -> tuple[DataArray | list[DataArray], bool]:
    """Prepare image data for one napari stack-mode image layer.

    The helper keeps ordinary multiplex images in their existing raster layout,
    but detects true RGB(A) images by channel names and converts those to
    channel-last layout so napari can render them with ``rgb=True``.

    For multiscale rasters, the returned image data is flattened to the list
    of per-scale arrays expected by napari layers.
    """
    axes = get_axes_names(element)

    if "c" in axes:
        assert axes.index("c") == 0

        if isinstance(element, DataArray):
            channel_coords = element.coords.indexes["c"]
        else:
            channel_coords = element["scale0"].coords.indexes["c"]
    else:
        channel_coords = []

    if len(channel_coords) != 0 and set(channel_coords) - {"r", "g", "b"} <= {"a"}:
        rgb = True
        if isinstance(element, DataArray):
            image_data: DataArray | DataTree = element.transpose("y", "x", "c").reindex(
                c=["r", "g", "b", "a"][: len(channel_coords)]
            )
        else:
            image_data = element.msi.transpose("y", "x", "c")
            image_data = image_data.msi.reindex_data_arrays({"c": ["r", "g", "b", "a"][: len(channel_coords)]})
    else:
        rgb = False
        image_data = element

    if isinstance(image_data, DataTree):
        return _flatten_multiscale_element(image_data), rgb

    return image_data, rgb


def _get_overlay_channel_layer_data(element: DataArray | DataTree, channel_index: int) -> DataArray | list[DataArray]:
    if isinstance(element, DataTree):
        channel_arrays: list[DataArray] = []
        for key in element:
            scale_array = next(iter(element[key].values()))
            channel_array = scale_array.isel(c=channel_index)
            channel_arrays.append(channel_array)
        return channel_arrays

    return element.isel(c=channel_index)


def _unwrap_xarray_data_for_napari(
    data: DataArray | list[DataArray],
) -> da.Array | np.ndarray | list[da.Array] | list[np.ndarray]:
    if isinstance(data, DataArray):
        return data.data

    return [scale.data for scale in data]


def _resolve_overlay_channels(
    element: DataArray | DataTree,
    channels: Sequence[int | str] | None,
) -> list[tuple[int, str]]:
    axes = get_axes_names(element)
    if "c" not in axes:
        channel_names: list[Any] = []
    elif isinstance(element, DataArray):
        channel_names = list(element.coords.indexes["c"])
    else:
        scale0 = next(iter(element["scale0"].values()))
        channel_names = list(scale0.coords.indexes["c"])

    if not channel_names:
        raise ValueError("Overlay mode requires an image element with a channel axis.")
    if not channels:
        raise ValueError("Overlay mode requires at least one selected channel.")

    resolved: list[tuple[int, str]] = []
    seen_indices: set[int] = set()
    for channel in channels:
        if isinstance(channel, int):
            channel_index = channel
            if channel_index < 0 or channel_index >= len(channel_names):
                raise ValueError(f"Channel index `{channel_index}` is out of range for the selected image element.")
            channel_name = str(channel_names[channel_index])
        else:
            if channel in channel_names:
                channel_index = channel_names.index(channel)
                channel_name = str(channel)
            else:
                string_channel_names = [str(channel_name) for channel_name in channel_names]
                if channel not in string_channel_names:
                    raise ValueError(
                        f"Channel `{channel}` is not available in the selected image element. "
                        "If needed, update the channel names in the SpatialData object with "
                        "`sdata.set_channel_names(...)`."
                    )
                channel_index = string_channel_names.index(channel)
                channel_name = channel

        if channel_index in seen_indices:
            raise ValueError("Overlay mode does not accept duplicate channel selections.")

        resolved.append((channel_index, channel_name))
        seen_indices.add(channel_index)

    return resolved


def _resolve_overlay_colors(channel_count: int, channel_colors: Sequence[str] | None) -> list[str]:
    if channel_colors is None:
        return [DEFAULT_OVERLAY_COLORS[index % len(DEFAULT_OVERLAY_COLORS)] for index in range(channel_count)]

    if len(channel_colors) != channel_count:
        raise ValueError("The number of overlay channel colors must match the number of selected channels.")

    return list(channel_colors)


def _image_layer_colormap_name(layer: Image, *, fallback: str) -> str:
    colormap = getattr(layer, "colormap", None)
    name = getattr(colormap, "name", None)
    if isinstance(name, str) and name:
        return name
    if isinstance(colormap, str) and colormap:
        return colormap
    return fallback


def _flatten_multiscale_element(element: DataTree) -> list[DataArray]:
    multiscale_data: list[DataArray] = []
    for key in element:
        values = element[key].values()
        assert len(values) == 1
        multiscale_data.append(next(iter(values)))
    return multiscale_data


def _build_shapes_layer(
    sdata: SpatialData,
    shapes_name: str,
    coordinate_system: str,
    *,
    name: str,
    sync_current_colors: bool = True,
) -> _BuiltShapesLayer:
    shapes = getattr(sdata, "shapes", {})
    if shapes_name not in shapes:
        raise ValueError(f"Shapes element `{shapes_name}` is not available in the selected SpatialData object.")

    shapes_element = shapes[shapes_name]
    available_coordinate_systems = set(get_transformation(shapes_element, get_all=True).keys())
    if coordinate_system not in available_coordinate_systems:
        raise ValueError(
            f"Coordinate system `{coordinate_system}` is not available for shapes element `{shapes_name}`."
        )

    # Unlike raster images and labels, vector shapes can be transformed by
    # moving their coordinates directly without resampling array data. Doing
    # that first lets geometry repair, hole handling, and circle-radius
    # conversion operate in the final display coordinate system.
    transformed_shapes = transform_spatial_element(shapes_element, to_coordinate_system=coordinate_system)
    point_radius_inputs = _prepare_napari_point_radius_shapes_layer_inputs(transformed_shapes)
    if point_radius_inputs is not None:
        layer = _SpatiatoPointRadiusShapes(
            point_radius_inputs.coordinates,
            ndim=2,
            name=name,
            features=point_radius_inputs.features,
            source_shapes_index_feature_name=point_radius_inputs.source_shapes_index_feature_name,
            size=point_radius_inputs.sizes,
            opacity=0.8,
            symbol="disc",
            border_width=0,
            face_color="#00FFFF",
            border_color="#00FFFF",
        )
        connect_current_size_to_radius_scaled_point_size(layer, point_radius_inputs.sizes)
        connect_current_symbol_to_global_point_symbol(layer)
        # Styled variants pass `sync_current_colors=False`; keep color callbacks
        # primary-only so styled palettes stay data-driven.
        if sync_current_colors:
            connect_current_face_color_to_global_point_face_color(layer)
        return _BuiltShapesLayer(
            layer=layer,
            shapes_rendering_mode="points",
            source_shapes_index_feature_name=point_radius_inputs.source_shapes_index_feature_name,
            source_row_id_by_rendered_row=point_radius_inputs.source_row_id_by_rendered_row,
            skipped_geometry_count=point_radius_inputs.skipped_geometry_count,
        )

    napari_layer_inputs = _prepare_napari_shapes_layer_inputs(transformed_shapes)
    if not napari_layer_inputs.data:
        raise ValueError(
            f"Shapes element `{shapes_name}` has no renderable geometries in coordinate system `{coordinate_system}`."
        )

    ensure_shapes_triangulation_backend()
    layer = _SpatiatoShapes(
        napari_layer_inputs.data,
        name=name,
        shape_type=napari_layer_inputs.shape_types,
        features=napari_layer_inputs.features,
        source_shapes_index_feature_name=napari_layer_inputs.source_shapes_index_feature_name,
    )
    # Primary shapes own their current colors as presentation, while styled
    # shapes use data-driven palettes that should not be flattened by napari's
    # current color controls.
    _apply_primary_shapes_layer_style(layer, sync_current_colors=sync_current_colors)
    return _BuiltShapesLayer(
        layer=layer,
        shapes_rendering_mode="shapes",
        source_shapes_index_feature_name=napari_layer_inputs.source_shapes_index_feature_name,
        source_row_id_by_rendered_row=napari_layer_inputs.source_row_id_by_rendered_row,
        skipped_geometry_count=napari_layer_inputs.skipped_geometry_count,
    )


def _build_empty_primary_shapes_layer(
    *,
    name: str,
    source_shapes_index_feature_name: str = DEFAULT_SHAPES_INDEX_FEATURE_NAME,
) -> _SpatiatoShapes:
    ensure_shapes_triangulation_backend()
    layer = _SpatiatoShapes(
        [],
        ndim=2,
        name=name,
        source_shapes_index_feature_name=source_shapes_index_feature_name,
    )
    _apply_primary_shapes_layer_style(layer)
    return layer


def _normalize_native_shapes_layer_transform(layer: Shapes) -> None:
    """Bake one native napari Shapes-layer transform into its vertices."""
    baked_data = [
        # `Shapes.data_to_world(...)` is a single-position helper; use the
        # array-capable underlying transform so one call can transform the
        # whole `(n_vertices, ndim)` array at once.
        np.asarray(layer._data_to_world(vertices), dtype=float)
        for vertices in layer.data
    ]

    layer.scale = (1.0,) * layer.ndim
    layer.translate = (0.0,) * layer.ndim
    layer.rotate = 0.0
    layer.shear = (0.0,) * max(layer.ndim * (layer.ndim - 1) // 2, 0)
    layer.affine = np.eye(layer.ndim + 1, dtype=float)

    if baked_data:
        layer.data = baked_data


def _build_spatiato_shapes_layer_from_native_layer(
    layer: Shapes,
    *,
    source_shapes_index_feature_name: str,
) -> _SpatiatoShapes:
    data = [np.asarray(vertices, dtype=float).copy() for vertices in layer.data]
    shape_type = [str(shape_type) for shape_type in layer.shape_type]
    features = layer.features.copy(deep=True)
    metadata = dict(getattr(layer, "metadata", {}) or {})
    selected_data = {int(index) for index in getattr(layer, "selected_data", set()) if int(index) < len(data)}
    mode = getattr(layer, "mode", None)

    kwargs: dict[str, Any] = {
        "name": layer.name,
        "features": features,
        "metadata": metadata,
        "ndim": layer.ndim,
        "source_shapes_index_feature_name": source_shapes_index_feature_name,
    }
    if data:
        kwargs["shape_type"] = shape_type

    ensure_shapes_triangulation_backend()
    replacement = _SpatiatoShapes(data, **kwargs)
    _apply_primary_shapes_layer_style(replacement)
    replacement.opacity = layer.opacity
    replacement.blending = layer.blending
    replacement.visible = layer.visible
    replacement.affine = np.asarray(layer.affine.affine_matrix, dtype=float).copy()
    replacement.selected_data = selected_data
    if mode is not None:
        replacement.mode = mode
    return replacement


def _prepare_napari_shapes_layer_inputs(shapes_element: Any) -> _NapariShapesLayerInputs:
    data: list[np.ndarray] = []
    shape_types: list[ShapesLayerShapeType] = []
    feature_rows: list[dict[str, Any]] = []
    source_row_id_by_rendered_row: list[int] = []
    skipped_geometry_count = 0
    has_radius = "radius" in getattr(shapes_element, "columns", [])
    geometry_column_name = getattr(getattr(shapes_element, "geometry", None), "name", "geometry")
    index_feature_name = _get_shapes_index_feature_name(shapes_element)
    source_index_values = shapes_element.index.to_numpy(copy=False)
    geometry_values = shapes_element[geometry_column_name].to_numpy(copy=False)
    radius_values = shapes_element["radius"].to_numpy(copy=False) if has_radius else None

    for source_row_id, (source_index, geometry) in enumerate(
        zip(source_index_values, geometry_values, strict=True),
    ):
        row_shape_count = len(data)
        feature_row = {index_feature_name: source_index}

        if _is_empty_geometry(geometry):
            skipped_geometry_count += 1
            continue

        if isinstance(geometry, Point):
            if not has_radius:
                skipped_geometry_count += 1
                continue

            if radius_values is None:  # pragma: no cover - defensive for non-GeoDataFrame-like inputs
                skipped_geometry_count += 1
                continue

            ellipse = _circle_to_napari_ellipse(geometry, radius_values[source_row_id])
            if ellipse is None:
                skipped_geometry_count += 1
                continue

            data.append(ellipse)
            shape_types.append("ellipse")
            feature_rows.append(feature_row)
            source_row_id_by_rendered_row.append(source_row_id)
            continue

        for polygon in _iter_renderable_polygons(geometry):
            data.append(_shapely_polygon_to_napari_polygon_vertices(polygon))
            shape_types.append("polygon")
            feature_rows.append(feature_row)
            source_row_id_by_rendered_row.append(source_row_id)

        if len(data) == row_shape_count:
            skipped_geometry_count += 1

    return _NapariShapesLayerInputs(
        data=data,
        shape_types=shape_types,
        features=pd.DataFrame(feature_rows),
        source_shapes_index_feature_name=index_feature_name,
        source_row_id_by_rendered_row=tuple(source_row_id_by_rendered_row),
        skipped_geometry_count=skipped_geometry_count,
    )


def _prepare_napari_point_radius_shapes_layer_inputs(
    shapes_element: Any,
) -> _NapariPointRadiusShapesLayerInputs | None:
    """Return point-radius layer inputs when all source rows qualify."""
    if "radius" not in getattr(shapes_element, "columns", []):
        return None

    geometry_column_name = getattr(getattr(shapes_element, "geometry", None), "name", "geometry")
    index_feature_name = _get_shapes_index_feature_name(shapes_element)
    source_index_values = shapes_element.index.to_numpy(copy=False)
    geometry_values = shapes_element[geometry_column_name]

    if len(source_index_values) == 0:
        return None

    try:
        missing_or_empty_geometry = geometry_values.isna().to_numpy(dtype=bool) | geometry_values.is_empty.to_numpy(
            dtype=bool
        )
        if np.any(missing_or_empty_geometry):
            return None

        if not bool(geometry_values.geom_type.eq("Point").all()):
            return None

        radius_values = pd.to_numeric(shapes_element["radius"], errors="coerce").to_numpy(dtype=float, copy=False)
    except (AttributeError, TypeError, ValueError):
        return None

    if len(geometry_values) != len(source_index_values) or len(radius_values) != len(source_index_values):
        return None
    if not np.all(np.isfinite(radius_values)) or not np.all(radius_values > 0):
        return None

    coordinates = np.column_stack(
        (
            geometry_values.y.to_numpy(dtype=float),
            geometry_values.x.to_numpy(dtype=float),
        )
    )
    sizes = 2.0 * radius_values

    return _NapariPointRadiusShapesLayerInputs(
        coordinates=coordinates,
        sizes=sizes,
        features=pd.DataFrame({index_feature_name: source_index_values}),
        source_shapes_index_feature_name=index_feature_name,
        source_row_id_by_rendered_row=range(len(source_index_values)),
        skipped_geometry_count=0,
    )


def _get_shapes_index_feature_name(shapes_element: Any) -> str:
    index_name = getattr(getattr(shapes_element, "index", None), "name", None)
    if index_name is None:
        return DEFAULT_SHAPES_INDEX_FEATURE_NAME
    return str(index_name)


def _circle_to_napari_ellipse(point: Point, radius: Any) -> np.ndarray | None:
    radius_value = _coerce_positive_radius(radius)
    if radius_value is None:
        return None

    y = float(point.y)
    x = float(point.x)
    return np.asarray(
        [
            (y - radius_value, x - radius_value),
            (y + radius_value, x - radius_value),
            (y + radius_value, x + radius_value),
            (y - radius_value, x + radius_value),
        ],
        dtype=float,
    )


def _coerce_positive_radius(radius: Any) -> float | None:
    try:
        radius_value = float(radius)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(radius_value) or radius_value <= 0:
        return None

    return radius_value


def _iter_renderable_polygons(geometry: BaseGeometry) -> Iterable[Polygon]:
    repaired_geometry = _repair_geometry(geometry)
    if _is_empty_geometry(repaired_geometry):
        return

    if isinstance(repaired_geometry, Polygon):
        if _is_renderable_polygon(repaired_geometry):
            yield repaired_geometry
        return

    if isinstance(repaired_geometry, MultiPolygon):
        for polygon in repaired_geometry.geoms:
            if _is_renderable_polygon(polygon):
                yield polygon
        return

    if isinstance(repaired_geometry, GeometryCollection):
        for part in repaired_geometry.geoms:
            yield from _iter_renderable_polygons(part)


def _repair_geometry(geometry: BaseGeometry) -> BaseGeometry:
    if _is_empty_geometry(geometry) or getattr(geometry, "is_valid", True):
        return geometry

    try:
        return make_valid(geometry)
    except (GEOSException, TypeError, ValueError) as error:  # pragma: no cover - defensive around GEOS failures
        logger.warning("Could not repair invalid shapes geometry: {}", error)
        return geometry


def _is_empty_geometry(geometry: Any) -> bool:
    return geometry is None or bool(getattr(geometry, "is_empty", True))


def _is_renderable_polygon(polygon: Polygon) -> bool:
    return not polygon.is_empty and polygon.is_valid and len(polygon.exterior.coords) >= 4


def _shapely_polygon_to_napari_polygon_vertices(polygon: Polygon) -> np.ndarray:
    """Encode a Shapely polygon as one napari polygon vertex row.

    Napari can render polygon holes when the interior rings are embedded in the
    same vertex row as the exterior ring and wind in the opposite direction. The
    repeated shell anchor creates bridge edges that napari's
    triangulation removes because they are traversed twice.
    """
    return shapely_polygon_to_napari_polygon_vertices(polygon)


def _build_labels_layer(
    sdata: SpatialData,
    labels_name: str,
    coordinate_system: str,
    *,
    name: str,
) -> Labels:
    labels = getattr(sdata, "labels", {})
    if labels_name not in labels:
        raise ValueError(f"Labels element `{labels_name}` is not available in the selected SpatialData object.")

    label_element = labels[labels_name]
    available_coordinate_systems = set(get_transformation(label_element, get_all=True).keys())
    if coordinate_system not in available_coordinate_systems:
        raise ValueError(
            f"Coordinate system `{coordinate_system}` is not available for labels element `{labels_name}`."
        )

    labels_data = _flatten_multiscale_element(label_element) if isinstance(label_element, DataTree) else label_element
    layer = Labels(
        labels_data,
        name=name,
        affine=_get_affine_transform(label_element, coordinate_system),
    )
    install_fast_labels_contours(layer)
    return layer


def _get_affine_transform(element: DataArray | DataTree, coordinate_system: str) -> np.ndarray | None:
    transformations = get_transformation(element, get_all=True)
    transform = transformations.get(coordinate_system)
    if transform is None:
        return None

    axes_element = get_axes_names(element)
    if "z" in axes_element:
        axes = ("z", "y", "x")
    else:
        axes = ("y", "x")
    return transform.to_affine_matrix(input_axes=axes, output_axes=axes)


def _matches_binding(
    binding: LayerBinding | None,
    *,
    sdata: SpatialData,
    element_name: str,
    element_type: ElementType,
) -> bool:
    if binding is None:
        return False
    return (
        binding.sdata_id == id(sdata) and binding.element_name == element_name and binding.element_type == element_type
    )


def _matches_labels_binding(
    binding: LayerBinding | None,
    *,
    sdata: SpatialData,
    element_name: str,
    coordinate_system: str | None = None,
    labels_role: Literal["primary", "styled"] | None = None,
    style_spec: TableColorSourceSpec | None = None,
) -> bool:
    if not isinstance(binding, LabelsLayerBinding):
        return False
    if binding.sdata_id != id(sdata) or binding.element_name != element_name:
        return False
    if coordinate_system is not None and binding.coordinate_system != coordinate_system:
        return False
    if labels_role is not None and binding.labels_role != labels_role:
        return False
    if style_spec is not None and binding.style_spec != style_spec:
        return False
    return True


def _matches_shapes_binding(
    binding: LayerBinding | None,
    *,
    sdata: SpatialData,
    element_name: str,
    coordinate_system: str | None = None,
    shapes_role: Literal["primary", "styled"] | None = None,
    style_spec: ShapeColumnColorSourceSpec | TableColorSourceSpec | None = None,
) -> bool:
    if not isinstance(binding, ShapesLayerBinding):
        return False
    if binding.sdata_id != id(sdata) or binding.element_name != element_name:
        return False
    if coordinate_system is not None and binding.coordinate_system != coordinate_system:
        return False
    if shapes_role is not None and binding.shapes_role != shapes_role:
        return False
    if style_spec is not None and binding.style_spec != style_spec:
        return False
    return True


def _matches_points_binding(
    binding: LayerBinding | None,
    *,
    identity: PointsLayerIdentity,
) -> bool:
    if not isinstance(binding, PointsLayerBinding):
        return False
    return (
        binding.sdata_id == id(identity.sdata)
        and binding.element_name == identity.points_name
        and binding.coordinate_system == identity.coordinate_system
        and binding.index_column == identity.index_column
    )


def _is_labels_binding(binding: LayerBinding | None) -> TypeGuard[LabelsLayerBinding]:
    return isinstance(binding, LabelsLayerBinding)


def _is_image_binding(binding: LayerBinding | None) -> TypeGuard[ImageLayerBinding]:
    return isinstance(binding, ImageLayerBinding)


def _is_shapes_binding(binding: LayerBinding | None) -> TypeGuard[ShapesLayerBinding]:
    return isinstance(binding, ShapesLayerBinding)


def _is_points_binding(binding: LayerBinding | None) -> TypeGuard[PointsLayerBinding]:
    return isinstance(binding, PointsLayerBinding)


def _is_primary_labels_binding(binding: LayerBinding | None) -> TypeGuard[LabelsLayerBinding]:
    return isinstance(binding, LabelsLayerBinding) and binding.labels_role == "primary"


def _is_primary_shapes_binding(binding: LayerBinding | None) -> TypeGuard[ShapesLayerBinding]:
    return isinstance(binding, ShapesLayerBinding) and binding.shapes_role == "primary"


def _is_usable_image_binding(binding: LayerBinding | None) -> TypeGuard[ImageLayerBinding]:
    if (
        not isinstance(binding, ImageLayerBinding)
        or binding.sdata_id is None
        or not isinstance(binding.coordinate_system, str)
        or not binding.coordinate_system
    ):
        return False
    if binding.image_display_mode == "stack":
        return True
    return (
        binding.image_display_mode == "overlay"
        and isinstance(binding.channel_index, int)
        and binding.channel_index >= 0
        and isinstance(binding.channel_name, str)
        and bool(binding.channel_name)
    )


def _is_pickable_primary_labels_layer(layer: Layer, binding: LayerBinding | None) -> bool:
    return _is_pickable_labels_layer(layer) and _is_primary_labels_binding(binding)


def _is_pickable_labels_layer(layer: Layer) -> bool:
    events = getattr(layer, "events", None)
    return isinstance(layer, Labels) and getattr(events, "selected_label", None) is not None


def _is_image_layer(layer: Layer) -> bool:
    # napari `Labels` layers are scalar-field siblings of `Image`, not subclasses.
    return isinstance(layer, Image)


def _is_shapes_layer(layer: Layer) -> bool:
    return isinstance(layer, Shapes)


def _is_semantic_shapes_layer(layer: Layer) -> bool:
    return isinstance(layer, (Shapes, Points))


def _is_points_layer(layer: Layer) -> bool:
    return isinstance(layer, Points)
