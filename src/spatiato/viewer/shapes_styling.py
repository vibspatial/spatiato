from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
from anndata import AnnData
from loguru import logger
from matplotlib.colors import to_rgba
from napari.layers import Points, Shapes
from napari.utils.colormaps import label_colormap

from spatiato.core._color_source import (
    ShapeColorValueKind,
    ShapeColumnColorSourceSpec,
    TableColorSourceSpec,
    validate_shape_color_value_kind,
)
from spatiato.core.class_palette import (
    CategoricalPaletteSource,
    resolve_table_categorical_palette,
    validate_categorical_palette_source,
)
from spatiato.core.spatialdata import SpatialDataTableMetadata, get_table, get_table_metadata
from spatiato.core.table_matrix import read_matrix_column
from spatiato.viewer._styling import (
    build_string_categorical_values,
    categorical_rgba_for_values,
    continuous_rgba_for_values,
    default_categorical_palette_for_categories,
    is_string_like_series,
    normalize_category_value,
)

if TYPE_CHECKING:
    import geopandas as gpd
    from spatialdata import SpatialData

SHAPES_MISSING_BASE_COLOR = "#808080"
SHAPES_FACE_ALPHA = 0.35
SHAPES_EDGE_ALPHA = 1.0
PRIMARY_SHAPES_EDGE_COLOR = "#00FFFF"
PRIMARY_SHAPES_FACE_COLOR = "#00FFFF20"
PRIMARY_SHAPES_EDGE_WIDTH = 1
PRIMARY_SHAPES_OPACITY = 0.8
_SHAPES_EDGE_WIDTH_SYNC_CALLBACK_ATTR = "_spatiato_shapes_edge_width_sync_callback"
_SHAPES_EDGE_COLOR_SYNC_CALLBACK_ATTR = "_spatiato_shapes_edge_color_sync_callback"
_SHAPES_FACE_COLOR_SYNC_CALLBACK_ATTR = "_spatiato_shapes_face_color_sync_callback"
ShapesStyleValueKind = ShapeColorValueKind | Literal["instance"]
ShapesRenderingMode = Literal["shapes", "points"]


@dataclass(frozen=True, kw_only=True)
class ShapesStyleResult:
    """Describe shapes styling metadata, if styling was applied."""

    value_kind: ShapesStyleValueKind | None
    palette_source: CategoricalPaletteSource | None
    coercion_applied: bool
    unannotated_source_shape_count: int = 0
    unannotated_rendered_shape_count: int = 0

    def __post_init__(self) -> None:
        if self.value_kind is not None and self.value_kind != "instance":
            validate_shape_color_value_kind(self.value_kind)
        if self.palette_source is not None:
            validate_categorical_palette_source(self.palette_source)


@dataclass(frozen=True)
class ShapesLoadResult(ShapesStyleResult):
    """Describe a primary or styled shapes or points layer load/update result."""

    layer: Shapes | Points
    created: bool
    skipped_geometry_count: int = 0
    shapes_rendering_mode: ShapesRenderingMode = "shapes"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.shapes_rendering_mode not in ("shapes", "points"):
            raise ValueError("Shapes load results require `shapes_rendering_mode` to be 'shapes' or 'points'.")


@dataclass(frozen=True)
class _ShapeTableRowAlignment:
    """Table-backed values aligned to source shapes and rendered napari shapes.

    Parameters
    ----------
    source_row_values
        Selected table-backed value for each source GeoDataFrame row. Shapes
        with no matching table row have a missing value here.
    rendered_row_values
        Selected table-backed value for each rendered napari shape row. Values
        are repeated when one source row, such as a ``MultiPolygon``, expands
        into multiple rendered shapes.
    source_row_has_table_row
        Boolean mask with one value per source GeoDataFrame row. ``False`` means
        the shape instance is not annotated by the selected table.
    rendered_row_has_table_row
        Boolean mask with one value per rendered napari shape row. This
        distinguishes unannotated shapes from annotated shapes whose selected
        table value is missing.
    """

    source_row_values: pd.Series
    rendered_row_values: pd.Series
    source_row_has_table_row: np.ndarray
    rendered_row_has_table_row: np.ndarray


def apply_primary_shapes_layer_style(layer: Shapes, *, sync_current_colors: bool = True) -> None:
    """Apply Spatiato's primary polygon-shapes style to an existing napari layer."""
    layer.current_edge_color = PRIMARY_SHAPES_EDGE_COLOR
    layer.current_face_color = PRIMARY_SHAPES_FACE_COLOR
    layer.current_edge_width = PRIMARY_SHAPES_EDGE_WIDTH
    layer.edge_color = PRIMARY_SHAPES_EDGE_COLOR
    layer.face_color = PRIMARY_SHAPES_FACE_COLOR
    layer.edge_width = PRIMARY_SHAPES_EDGE_WIDTH
    layer.opacity = PRIMARY_SHAPES_OPACITY
    _connect_current_edge_width_to_global_edge_width(layer)
    if sync_current_colors:
        _connect_current_edge_color_to_global_edge_color(layer)
        _connect_current_face_color_to_global_face_color(layer)


def apply_shape_column_color_source_to_shapes_layer(
    layer: Shapes | Points,
    *,
    shapes_element: gpd.GeoDataFrame,
    style_spec: ShapeColumnColorSourceSpec,
    source_row_id_by_rendered_row: tuple[int, ...] | range,
    source_shapes_index_feature_name: str,
    fill: bool = False,
) -> ShapesStyleResult:
    """Apply one direct shapes-column color source to a semantic shapes layer.

    Parameters
    ----------
    source_row_id_by_rendered_row
        Integer source GeoDataFrame row id for each rendered napari shape row.
        This can be longer than the source GeoDataFrame row count when one
        source row, such as a ``MultiPolygon``, expands into multiple rendered
        napari shapes. For example, if source row position ``7`` expands into
        three polygons and source row position ``8`` expands into one polygon,
        this mapping is ``(7, 7, 7, 8)``. Point-backed shapes are one-to-one
        and can pass ``range(n)``. Styled shapes use this mapping to repeat the
        source row's style value for every rendered part.
    source_shapes_index_feature_name
        Name of the ``layer.features`` column that stores the source
        GeoDataFrame index for napari-visible inspection and status-bar text.
        This follows the GeoDataFrame index name, falling back to ``"index"``
        for unnamed indexes.
    """
    if style_spec.source_kind != "shape_column":
        raise ValueError(f"Shape color sources must use `source_kind='shape_column'`, got `{style_spec.source_kind}`.")
    if style_spec.value_key not in shapes_element.columns:
        raise ValueError(f"Shape column `{style_spec.value_key}` is not available in the selected shapes element.")
    source_row_id_by_rendered_row = _validate_source_row_id_by_rendered_row(
        source_row_id_by_rendered_row,
        source_row_count=len(shapes_element),
        rendered_row_count=len(layer.data),
    )

    # Start with one style value per source GeoDataFrame row.
    source_values = shapes_element[style_spec.value_key]
    # Use integer row ids rather than GeoDataFrame index labels: pandas indexes
    # can be duplicated, while rendered-row style lookup must identify exactly
    # one source row.
    rendered_row_values = source_values.iloc[source_row_id_by_rendered_row]
    rendered_row_values.index = pd.RangeIndex(len(rendered_row_values))

    if style_spec.value_kind == "categorical":
        style_result, rendered_row_colors, feature_values = _build_categorical_shape_style(
            shapes_element=shapes_element,
            column_name=style_spec.value_key,
            source_values=source_values,
            rendered_row_values=rendered_row_values,
        )
    else:
        style_result, rendered_row_colors, feature_values = _build_continuous_shape_style(rendered_row_values)

    if isinstance(layer, Shapes):
        _apply_rendered_row_colors_to_shapes_layer(layer, rendered_row_colors, fill=fill)
    elif isinstance(layer, Points):
        _apply_rendered_row_colors_to_points_layer(layer, rendered_row_colors)
    else:  # pragma: no cover - defensive for future layer subclasses.
        raise TypeError(f"Unsupported styled shapes layer type: {type(layer)!r}.")
    # Unlike styled labels, styled shapes already have a useful feature table
    # from the geometry-loading path: the source GeoDataFrame index column used
    # for status-bar display. Preserve it and add only the selected style source
    # column for inspection/coloring.
    _set_shape_style_feature(
        layer,
        feature_values,
        style_column_name=style_spec.value_key,
        source_shapes_index_feature_name=source_shapes_index_feature_name,
    )
    return style_result


def apply_table_color_source_to_shapes_layer(
    layer: Shapes | Points,
    *,
    sdata: SpatialData,
    shapes_name: str,
    style_spec: TableColorSourceSpec,
    source_row_id_by_rendered_row: tuple[int, ...] | range,
    source_shapes_index_feature_name: str,
    fill: bool = False,
) -> ShapesStyleResult:
    """Apply one table-backed color source to a semantic shapes layer.

    Parameters
    ----------
    layer
        Napari layer representing the SpatialData shapes element. Generic
        shapes render as ``Shapes``; point-radius shapes can render as
        ``Points`` while still using the same shapes/table alignment semantics.
    sdata
        SpatialData object containing the source shapes element and the linked
        AnnData table named by ``style_spec.table_name``.
    shapes_name
        Name of the shapes element to style.
    style_spec
        Table-backed color source describing the linked table, source kind, and
        selected value key.
    source_row_id_by_rendered_row
        Integer source GeoDataFrame row id for each rendered napari shape row.
        This can be longer than the source GeoDataFrame row count when one
        source row, such as a ``MultiPolygon``, expands into multiple rendered
        napari shapes. For example, if source row position ``7`` expands into
        three polygons and source row position ``8`` expands into one polygon,
        this mapping is ``(7, 7, 7, 8)``. Table-backed styling uses it to
        repeat each source row's aligned table value for every rendered part.
        Point-backed shapes are one-to-one and can pass ``range(n)``.
    source_shapes_index_feature_name
        Name of the ``layer.features`` column that stores the source
        GeoDataFrame index for napari-visible inspection and status-bar text.
        This follows the GeoDataFrame index name, falling back to ``"index"``
        for unnamed indexes.
    fill
        Whether to apply visible face alpha in addition to edge colors.
    """
    table = get_table(sdata, style_spec.table_name)
    table_metadata = get_table_metadata(sdata, style_spec.table_name)
    shapes = getattr(sdata, "shapes", {})
    if shapes_name not in shapes:
        raise ValueError(f"Shapes element `{shapes_name}` is not available in the selected SpatialData object.")
    if len(source_row_id_by_rendered_row) != len(layer.data):
        raise ValueError(
            "`source_row_id_by_rendered_row` must contain one source row id for each rendered napari shape row."
        )

    aligned_values = _align_table_color_source_to_shapes_rows(
        table=table,
        table_metadata=table_metadata,
        shapes_name=shapes_name,
        shapes_element=shapes[shapes_name],
        style_spec=style_spec,
        source_row_id_by_rendered_row=source_row_id_by_rendered_row,
    )

    if style_spec.value_kind == "instance" or (
        style_spec.source_kind == "obs_column" and style_spec.value_key == table_metadata.instance_key
    ):
        if style_spec.source_kind != "obs_column" or style_spec.value_key != table_metadata.instance_key:
            raise ValueError(
                "Instance ID coloring must use the table instance key "
                f"`{table_metadata.instance_key}` as an observation source."
            )
        style_result, rendered_row_colors, feature_values = _build_table_instance_shape_style(aligned_values)
    elif style_spec.source_kind == "obs_column":
        style_result, rendered_row_colors, feature_values = _build_table_obs_shape_style(
            table=table,
            aligned_values=aligned_values,
            column_name=style_spec.value_key,
        )
    elif style_spec.source_kind == "x_var":
        style_result, rendered_row_colors, feature_values = _build_continuous_shape_style(
            aligned_values.rendered_row_values
        )
    else:
        raise ValueError(f"Unsupported table color source kind `{style_spec.source_kind}`.")

    style_result = ShapesStyleResult(
        value_kind=style_result.value_kind,
        palette_source=style_result.palette_source,
        coercion_applied=style_result.coercion_applied,
        unannotated_source_shape_count=int((~aligned_values.source_row_has_table_row).sum()),
        unannotated_rendered_shape_count=int((~aligned_values.rendered_row_has_table_row).sum()),
    )
    if isinstance(layer, Shapes):
        _apply_table_rendered_row_colors_to_shapes_layer(
            layer,
            rendered_row_colors,
            rendered_row_has_table_row=aligned_values.rendered_row_has_table_row,
            fill=fill,
        )
    elif isinstance(layer, Points):
        _apply_table_rendered_row_colors_to_points_layer(
            layer,
            rendered_row_colors,
            rendered_row_has_table_row=aligned_values.rendered_row_has_table_row,
        )
    else:  # pragma: no cover - defensive for future layer subclasses.
        raise TypeError(f"Unsupported styled shapes layer type: {type(layer)!r}.")
    _set_shape_style_feature(
        layer,
        feature_values,
        style_column_name=style_spec.value_key,
        source_shapes_index_feature_name=source_shapes_index_feature_name,
    )
    return style_result


def build_styled_shapes_layer_name(
    shapes_name: str,
    style_spec: ShapeColumnColorSourceSpec | TableColorSourceSpec,
) -> str:
    """Return the user-facing layer name for one styled shapes variant."""
    if style_spec.source_kind == "obs_column":
        return f"{shapes_name}[obs:{style_spec.value_key}]"
    if style_spec.source_kind == "x_var":
        return f"{shapes_name}[X:{style_spec.value_key}]"
    return f"{shapes_name}[shapes_column:{style_spec.value_key}]"


def _validate_source_row_id_by_rendered_row(
    source_row_id_by_rendered_row: tuple[int, ...] | range,
    *,
    source_row_count: int,
    rendered_row_count: int | None = None,
) -> np.ndarray:
    """Validate rendered-row to source-row mapping and return it as an integer array.

    Shapes styling expands source GeoDataFrame rows to rendered napari rows
    with integer source row ids instead of GeoDataFrame index labels. This keeps
    lookup unambiguous when the GeoDataFrame index contains duplicate labels,
    and it lets MultiPolygon parts repeat the same source row id.
    """
    if rendered_row_count is not None and len(source_row_id_by_rendered_row) != rendered_row_count:
        raise ValueError(
            "`source_row_id_by_rendered_row` must contain one source row id for each rendered napari shape row."
        )

    invalid_row_ids = [
        row_id
        for row_id in source_row_id_by_rendered_row
        if (
            not isinstance(row_id, int | np.integer)
            or isinstance(row_id, bool | np.bool_)
            or int(row_id) < 0
            or int(row_id) >= source_row_count
        )
    ]
    if invalid_row_ids:
        preview = _format_value_preview(pd.unique(pd.Index(invalid_row_ids)).tolist())
        raise ValueError(f"Could not align rendered shapes back to source row id(s): {preview}.")

    return np.asarray(source_row_id_by_rendered_row, dtype=np.int64)


def _align_table_color_source_to_shapes_rows(
    *,
    table: AnnData,
    table_metadata: SpatialDataTableMetadata,
    shapes_name: str,
    shapes_element: gpd.GeoDataFrame,
    style_spec: TableColorSourceSpec,
    source_row_id_by_rendered_row: tuple[int, ...] | range,
) -> _ShapeTableRowAlignment:
    """Align one table-backed color source to source and rendered shapes rows."""
    if style_spec.table_name != table_metadata.table_name:
        raise ValueError(
            f"Table color source `{style_spec.table_name}` does not match table metadata `{table_metadata.table_name}`."
        )
    if not table_metadata.annotates(shapes_name):
        raise ValueError(f"Table `{table_metadata.table_name}` does not annotate shapes element `{shapes_name}`.")
    if table_metadata.region_key not in table.obs.columns:
        raise ValueError(
            f"Table `{table_metadata.table_name}` is missing required obs column `{table_metadata.region_key}`."
        )
    if table_metadata.instance_key not in table.obs.columns:
        raise ValueError(
            f"Table `{table_metadata.table_name}` is missing required obs column `{table_metadata.instance_key}`."
        )
    shape_instance_values = _get_shape_instance_values(
        shapes_element=shapes_element,
        shapes_name=shapes_name,
        instance_key=table_metadata.instance_key,
    )
    missing_shape_instances = shape_instance_values.loc[shape_instance_values.isna()]
    if not missing_shape_instances.empty:
        preview = _format_value_preview(missing_shape_instances.index.tolist())
        raise ValueError(
            f"Shapes element `{shapes_name}` cannot be aligned to table `{table_metadata.table_name}` because "
            f"`{table_metadata.instance_key}` contains missing values for source row(s): {preview}."
        )
    # Duplicate shape instance values are allowed: multiple geometries can
    # represent the same instance and receive the same table-backed value.

    region_rows = table.obs.loc[table.obs[table_metadata.region_key] == shapes_name].copy()
    if region_rows.empty:
        raise ValueError(
            f"Table `{table_metadata.table_name}` declares shapes element `{shapes_name}`, "
            "but no table rows annotate that shapes element."
        )

    region_instances = region_rows[table_metadata.instance_key]
    missing_table_instances = region_instances.loc[region_instances.isna()]
    if not missing_table_instances.empty:
        preview = _format_value_preview(missing_table_instances.index.tolist())
        raise ValueError(
            f"Table `{table_metadata.table_name}` cannot be aligned to shapes element `{shapes_name}` because "
            f"`{table_metadata.instance_key}` contains missing values for table row(s): {preview}."
        )

    # Duplicate table instance values are not allowed within the selected
    # region: one instance must map to exactly one selected table-backed value.
    duplicate_table_instances = region_instances.loc[region_instances.duplicated(keep=False)]
    if not duplicate_table_instances.empty:
        preview = _format_value_preview(duplicate_table_instances.drop_duplicates().tolist())
        raise ValueError(
            f"Table `{table_metadata.table_name}` cannot be aligned to shapes element `{shapes_name}` because "
            f"`{table_metadata.instance_key}` contains duplicate values within that region: {preview}."
        )

    # Series from selected-region table instance identity to the selected
    # table-backed value. Shape instances that are absent from the table are not
    # present here; they are tracked separately with `source_row_has_table_row`.
    # index      value
    # cell_1     3.2
    # cell_2     7.5
    table_value_by_instance = _get_table_color_values_by_instance(
        table=table,
        region_rows=region_rows,
        region_instances=region_instances,
        table_metadata=table_metadata,
        style_spec=style_spec,
    )
    missing_shape_instances_for_table = pd.Index(table_value_by_instance.index)[
        ~pd.Index(table_value_by_instance.index).isin(shape_instance_values)
    ]
    if len(missing_shape_instances_for_table) > 0:
        preview = _format_value_preview(missing_shape_instances_for_table.tolist())
        raise ValueError(
            f"Table `{table_metadata.table_name}` contains instance(s) for shapes element `{shapes_name}` that "
            f"are not present in the resolved shapes instance identities for `{table_metadata.instance_key}`: {preview}."
        )

    source_row_has_table_row = shape_instance_values.isin(table_value_by_instance.index).to_numpy(
        dtype=bool,
        copy=False,
    )
    # Avoid `shape_instance_values.map(table_value_by_instance)`: pandas may
    # upcast positive integer instance IDs to floats when unannotated shapes
    # introduce missing values, which would break labels-like identity coloring.
    table_value_lookup = table_value_by_instance.to_dict()
    source_row_values = pd.Series(
        [
            table_value_lookup[value] if has_table_row else pd.NA
            for value, has_table_row in zip(shape_instance_values, source_row_has_table_row, strict=True)
        ],
        index=pd.RangeIndex(len(shape_instance_values)),
        name=style_spec.value_key,
        dtype="object",
    )

    source_row_id_by_rendered_row = _validate_source_row_id_by_rendered_row(
        source_row_id_by_rendered_row,
        source_row_count=len(shapes_element),
    )

    rendered_row_values = source_row_values.iloc[source_row_id_by_rendered_row]
    rendered_row_values.index = pd.RangeIndex(len(rendered_row_values))
    rendered_row_has_table_row = source_row_has_table_row[source_row_id_by_rendered_row]

    return _ShapeTableRowAlignment(
        source_row_values=source_row_values,
        rendered_row_values=rendered_row_values,
        source_row_has_table_row=source_row_has_table_row,
        rendered_row_has_table_row=rendered_row_has_table_row,
    )


def _get_shape_instance_values(
    *,
    shapes_element: gpd.GeoDataFrame,
    shapes_name: str,
    instance_key: str,
) -> pd.Series:
    """Return source-row-aligned shape instance identities for table lookup.

    Table-backed shapes styling uses the GeoDataFrame index as the shape
    instance identity. A same-named GeoDataFrame column is allowed only when it
    duplicates the index values exactly.
    """
    if getattr(shapes_element.index, "name", None) != instance_key:
        raise ValueError(
            f"Shapes element `{shapes_name}` must use GeoDataFrame index `{instance_key}` for table-backed styling. "
            f"Set `sdata.shapes[{shapes_name!r}].index.name = {instance_key!r}` and store the shape instance "
            "identities in that index before styling from a linked table."
        )

    index_values = pd.Series(
        shapes_element.index.to_numpy(copy=False),
        index=pd.RangeIndex(len(shapes_element)),
        name=instance_key,
        dtype="object",
    )
    if instance_key not in shapes_element.columns:
        return index_values

    column_values = pd.Series(
        shapes_element[instance_key].to_numpy(copy=False),
        index=pd.RangeIndex(len(shapes_element)),
        name=instance_key,
        dtype="object",
    )
    both_missing = column_values.isna() & index_values.isna()
    equal_values = column_values.eq(index_values).fillna(False)
    disagreement = ~(both_missing | equal_values)
    disagreement_row_ids = disagreement[disagreement].index.to_list()
    if disagreement_row_ids:
        preview = _format_value_preview(disagreement_row_ids)
        raise ValueError(
            f"Shapes element `{shapes_name}` has instance key `{instance_key}` both as a GeoDataFrame column and "
            f"as the GeoDataFrame index name, but they disagree for source row(s): {preview}."
        )

    return index_values


def _get_table_color_values_by_instance(
    *,
    table: AnnData,
    region_rows: pd.DataFrame,
    region_instances: pd.Series,
    table_metadata: SpatialDataTableMetadata,
    style_spec: TableColorSourceSpec,
) -> pd.Series:
    if style_spec.value_kind == "instance" or (
        style_spec.source_kind == "obs_column" and style_spec.value_key == table_metadata.instance_key
    ):
        if style_spec.source_kind != "obs_column" or style_spec.value_key != table_metadata.instance_key:
            raise ValueError(
                "Instance ID coloring must use the table instance key "
                f"`{table_metadata.instance_key}` as an observation source."
            )
        values = region_instances.copy()
    elif style_spec.source_kind == "obs_column":
        if style_spec.value_key not in table.obs:
            raise ValueError(f"Observation column `{style_spec.value_key}` is not available in the selected table.")
        values = region_rows[style_spec.value_key]
    elif style_spec.source_kind == "x_var":
        var_name = style_spec.value_key
        if var_name not in table.var_names:
            raise ValueError(f"Var `{var_name}` is not available in the selected table.")

        var_index = table.var_names.get_loc(var_name)
        if not isinstance(var_index, (int, np.integer)):
            raise ValueError(f"Var `{var_name}` does not resolve to one unique column in `table.var_names`.")

        obs_positions = table.obs.index.get_indexer(region_rows.index)
        if np.any(obs_positions < 0):
            raise ValueError(f"Could not align linked table rows back to `X[:, {var_name!r}]` for viewer coloring.")

        x_values = read_matrix_column(table.X, int(var_index), rows=obs_positions)
        values = pd.Series(x_values, index=region_rows.index, name=var_name)
    else:
        raise ValueError(f"Unsupported table color source kind `{style_spec.source_kind}`.")

    return pd.Series(
        values.to_numpy(copy=False),
        index=pd.Index(region_instances.to_numpy(copy=False), name=table_metadata.instance_key),
        name=style_spec.value_key,
    )


def _connect_current_edge_width_to_global_edge_width(layer: Shapes) -> None:
    if getattr(layer, _SHAPES_EDGE_WIDTH_SYNC_CALLBACK_ATTR, None) is not None:
        return

    def _sync_current_edge_width_to_all_shapes(_event: Any | None = None) -> None:
        layer.edge_width = layer.current_edge_width

    # Napari exposes `current_edge_width`, but emits its changes through
    # the `edge_width` event.
    layer.events.edge_width.connect(_sync_current_edge_width_to_all_shapes)
    setattr(layer, _SHAPES_EDGE_WIDTH_SYNC_CALLBACK_ATTR, _sync_current_edge_width_to_all_shapes)


def _connect_current_edge_color_to_global_edge_color(layer: Shapes) -> None:
    if getattr(layer, _SHAPES_EDGE_COLOR_SYNC_CALLBACK_ATTR, None) is not None:
        return

    def _sync_current_edge_color_to_all_shapes(_event: Any | None = None) -> None:
        layer.edge_color = layer.current_edge_color

    layer.events.current_edge_color.connect(_sync_current_edge_color_to_all_shapes)
    setattr(layer, _SHAPES_EDGE_COLOR_SYNC_CALLBACK_ATTR, _sync_current_edge_color_to_all_shapes)


def _connect_current_face_color_to_global_face_color(layer: Shapes) -> None:
    if getattr(layer, _SHAPES_FACE_COLOR_SYNC_CALLBACK_ATTR, None) is not None:
        return

    def _sync_current_face_color_to_all_shapes(_event: Any | None = None) -> None:
        layer.face_color = layer.current_face_color

    layer.events.current_face_color.connect(_sync_current_face_color_to_all_shapes)
    setattr(layer, _SHAPES_FACE_COLOR_SYNC_CALLBACK_ATTR, _sync_current_face_color_to_all_shapes)


def disambiguate_shape_style_feature_name(style_column_name: str, source_shapes_index_feature_name: str) -> str:
    """Return the layer.features column name for a selected shape style value.

    GeoPandas and SpatialData allow a shapes GeoDataFrame to have both an
    index named, for example, ``cell_id`` and a normal column named
    ``cell_id``. Spatiato stores the source GeoDataFrame index in
    ``layer.features`` under the index name for status display, while styled
    shapes also store the selected style column in ``layer.features`` for
    inspection. Coloring by the normal ``cell_id`` column would otherwise
    collide with the source-index feature column, so the selected style value
    is stored as ``cell_id__value``.
    """
    if style_column_name == source_shapes_index_feature_name:
        return f"{style_column_name}__value"
    return style_column_name


def _build_categorical_shape_style(
    *,
    shapes_element: gpd.GeoDataFrame,
    column_name: str,
    source_values: pd.Series,
    rendered_row_values: pd.Series,
) -> tuple[ShapesStyleResult, np.ndarray, pd.Series]:
    """Build categorical colors from source-level values and rendered-row values.

    Parameters
    ----------
    shapes_element
        Source GeoDataFrame used to resolve optional companion palette columns.
    column_name
        Source column name used in warning messages and companion palette
        lookup.
    source_values
        Complete source column with one value per source GeoDataFrame row.
        Category discovery, companion palette validation, and string/object
        coercion warnings use this complete series so styling remains stable
        even when only a subset of source rows is rendered.
    rendered_row_values
        Values already aligned to rendered napari shape rows. This series can
        repeat one source value for multiple rendered polygon parts, and can be
        a subset of ``source_values`` when empty, invalid, or unsupported source
        geometries were skipped before rendering.

    Returns
    -------
    ShapesStyleResult
        Metadata describing the categorical styling decision, including palette
        source and whether string/object values were coerced.
    np.ndarray
        ``Nx4`` float RGBA colors aligned one-to-one with rendered napari shape
        rows. Alpha is left as the base color alpha here; layer-specific face,
        edge, or border alpha is applied later.
    pd.Series
        Rendered-row categorical values to store in ``layer.features`` for
        inspection and status display.
    """
    companion_palette_allowed = True
    coercion_applied = False

    if _is_categorical_dtype(source_values):
        normalized_source_values = _normalize_category_series(source_values)
        normalized_rendered_row_values = _normalize_category_series(rendered_row_values)
        # Keep the original categorical series for color generation so
        # `categorical_rgba_for_values(...)` can use pandas category codes.
        # The normalized object series is still used for layer.features.
        color_values = rendered_row_values
        categories = _present_categories(normalized_source_values, source_values.cat.categories)
    elif _is_bool_series(source_values):
        normalized_source_values = _normalize_category_series(source_values)
        normalized_rendered_row_values = _normalize_category_series(rendered_row_values)
        color_values = normalized_rendered_row_values
        categories = _present_categories(normalized_source_values, [False, True])
    elif _is_exact_binary_integer_series(source_values):
        numeric_source_values = pd.to_numeric(source_values, errors="coerce").astype("Int64")
        numeric_rendered_row_values = pd.to_numeric(rendered_row_values, errors="coerce").astype("Int64")
        normalized_source_values = _normalize_category_series(numeric_source_values)
        normalized_rendered_row_values = _normalize_category_series(numeric_rendered_row_values)
        color_values = numeric_rendered_row_values
        categories = _present_categories(normalized_source_values, [0, 1])
    elif is_string_like_series(source_values):
        normalized_rendered_row_values, categories = build_string_categorical_values(
            full_values=source_values,
            row_values=rendered_row_values,
            column_name=column_name,
        )
        normalized_source_values = pd.Series(
            [pd.NA if pd.isna(value) else str(value) for value in source_values],
            index=source_values.index,
            name=column_name,
            dtype="object",
        )
        color_values = normalized_rendered_row_values
        companion_palette_allowed = False
        coercion_applied = True
    else:
        normalized_source_values = _normalize_category_series(source_values)
        normalized_rendered_row_values = _normalize_category_series(rendered_row_values)
        color_values = normalized_rendered_row_values
        categories = list(pd.unique(normalized_source_values.dropna()))
        companion_palette_allowed = False

    if companion_palette_allowed:
        palette_source, palette = _resolve_shape_categorical_palette(
            shapes_element=shapes_element,
            column_name=column_name,
            source_values=normalized_source_values,
            categories=categories,
        )
    else:
        palette_source = "default_missing"
        palette = default_categorical_palette_for_categories(categories)

    rendered_row_colors = categorical_rgba_for_values(
        color_values,
        categories=categories,
        palette=palette,
        missing_color=SHAPES_MISSING_BASE_COLOR,
    )
    return (
        ShapesStyleResult(
            value_kind="categorical",
            palette_source=palette_source,
            coercion_applied=coercion_applied,
        ),
        rendered_row_colors,
        normalized_rendered_row_values,
    )


def _build_continuous_shape_style(
    rendered_row_values: pd.Series,
) -> tuple[ShapesStyleResult, np.ndarray, pd.Series]:
    numeric_rendered_row_values = pd.to_numeric(rendered_row_values, errors="coerce").astype("float64")
    rendered_row_colors = continuous_rgba_for_values(
        numeric_rendered_row_values,
        missing_color=SHAPES_MISSING_BASE_COLOR,
    )
    return (
        ShapesStyleResult(
            value_kind="continuous",
            palette_source=None,
            coercion_applied=False,
        ),
        rendered_row_colors,
        numeric_rendered_row_values,
    )


def _build_table_obs_shape_style(
    *,
    table: AnnData,
    aligned_values: _ShapeTableRowAlignment,
    column_name: str,
) -> tuple[ShapesStyleResult, np.ndarray, pd.Series]:
    if column_name not in table.obs:
        raise ValueError(f"Observation column `{column_name}` is not available in the selected table.")

    full_series = table.obs[column_name]
    rendered_row_values = aligned_values.rendered_row_values

    if _is_categorical_dtype(full_series):
        normalized_rendered_row_values = _normalize_category_series(rendered_row_values)
        # Table-to-shapes alignment currently stores aligned values as an
        # object Series, so rebuild a categorical Series with the table's
        # original categories to use the fast categorical-code RGBA path.
        color_values = pd.Series(
            pd.Categorical(rendered_row_values, categories=full_series.cat.categories),
            index=rendered_row_values.index,
            name=rendered_row_values.name,
        )
        categories = [normalize_category_value(value) for value in full_series.cat.categories]
        palette_source, palette = resolve_table_categorical_palette(
            table=table,
            column_name=column_name,
            categories=categories,
        )
        rendered_row_colors = categorical_rgba_for_values(
            color_values,
            categories=categories,
            palette=palette,
            missing_color=SHAPES_MISSING_BASE_COLOR,
        )
        style_result = ShapesStyleResult(
            value_kind="categorical",
            palette_source=palette_source,
            coercion_applied=False,
        )
        return style_result, rendered_row_colors, normalized_rendered_row_values

    if _is_bool_series(full_series):
        normalized_rendered_row_values = _normalize_category_series(rendered_row_values)
        present_values = set(full_series.dropna().tolist())
        categories = [value for value in (False, True) if value in present_values]
        palette_source, palette = resolve_table_categorical_palette(
            table=table,
            column_name=column_name,
            categories=categories,
        )
        rendered_row_colors = categorical_rgba_for_values(
            normalized_rendered_row_values,
            categories=categories,
            palette=palette,
            missing_color=SHAPES_MISSING_BASE_COLOR,
        )
        style_result = ShapesStyleResult(
            value_kind="categorical",
            palette_source=palette_source,
            coercion_applied=False,
        )
        return style_result, rendered_row_colors, normalized_rendered_row_values

    if _is_exact_binary_integer_series(full_series):
        numeric_rendered_row_values = pd.to_numeric(rendered_row_values, errors="coerce").astype("Int64")
        non_missing_source = pd.to_numeric(full_series.dropna(), errors="coerce").astype("int64")
        present_values = set(non_missing_source.tolist())
        categories = [value for value in (0, 1) if value in present_values]
        palette_source, palette = resolve_table_categorical_palette(
            table=table,
            column_name=column_name,
            categories=categories,
        )
        rendered_row_colors = categorical_rgba_for_values(
            numeric_rendered_row_values,
            categories=categories,
            palette=palette,
            missing_color=SHAPES_MISSING_BASE_COLOR,
        )
        style_result = ShapesStyleResult(
            value_kind="categorical",
            palette_source=palette_source,
            coercion_applied=False,
        )
        return style_result, rendered_row_colors, numeric_rendered_row_values

    if is_string_like_series(full_series):
        string_rendered_row_values, categories = build_string_categorical_values(
            full_values=full_series,
            row_values=rendered_row_values,
            column_name=column_name,
        )
        rendered_row_colors = categorical_rgba_for_values(
            string_rendered_row_values,
            categories=categories,
            palette=default_categorical_palette_for_categories(categories),
            missing_color=SHAPES_MISSING_BASE_COLOR,
        )
        style_result = ShapesStyleResult(
            value_kind="categorical",
            palette_source="default_missing",
            coercion_applied=True,
        )
        return style_result, rendered_row_colors, string_rendered_row_values

    return _build_continuous_shape_style(rendered_row_values)


def _build_table_instance_shape_style(
    aligned_values: _ShapeTableRowAlignment,
) -> tuple[ShapesStyleResult, np.ndarray, pd.Series]:
    rendered_row_values = aligned_values.rendered_row_values
    annotated_source_values = aligned_values.source_row_values.loc[aligned_values.source_row_has_table_row]
    rendered_row_codes = _instance_identity_codes(
        source_values=annotated_source_values,
        rendered_row_values=rendered_row_values,
    )
    rendered_row_colors = np.asarray(
        label_colormap(background_value=0).map(rendered_row_codes.to_numpy(dtype=np.int64, copy=False)),
        dtype="float64",
    )
    return (
        ShapesStyleResult(
            value_kind="instance",
            palette_source=None,
            coercion_applied=False,
        ),
        rendered_row_colors,
        rendered_row_values,
    )


def _instance_identity_codes(*, source_values: pd.Series, rendered_row_values: pd.Series) -> pd.Series:
    non_missing_source_values = source_values.dropna()
    if _is_positive_integer_instance_series(non_missing_source_values):
        numeric_codes = pd.to_numeric(rendered_row_values, errors="coerce").fillna(0).astype("int64")
        return pd.Series(
            numeric_codes.to_numpy(copy=False), index=rendered_row_values.index, name=rendered_row_values.name
        )

    unique_values = pd.unique(non_missing_source_values)
    code_by_value = {value: code for code, value in enumerate(unique_values, start=1)}
    codes = [0 if pd.isna(value) else int(code_by_value[value]) for value in rendered_row_values]
    return pd.Series(codes, index=rendered_row_values.index, name=rendered_row_values.name, dtype="int64")


def _is_positive_integer_instance_series(values: pd.Series) -> bool:
    if values.empty:
        return False
    return all(
        isinstance(value, int | np.integer) and not isinstance(value, bool | np.bool_) and int(value) > 0
        for value in values.tolist()
    )


def _resolve_shape_categorical_palette(
    *,
    shapes_element: gpd.GeoDataFrame,
    column_name: str,
    source_values: pd.Series,
    categories: Sequence[object],
) -> tuple[CategoricalPaletteSource, list[Any]]:
    colors_column_name = f"{column_name}_colors"
    if colors_column_name not in shapes_element.columns:
        logger.info(f"No `{colors_column_name}` companion color column found; using the default categorical palette.")
        return "default_missing", default_categorical_palette_for_categories(categories)

    companion_colors = shapes_element[colors_column_name]
    color_by_category: dict[object, Any] = {}
    for category in categories:
        category_mask = source_values == category
        category_colors = companion_colors.loc[category_mask.fillna(False)].dropna()
        if category_colors.empty:
            logger.warning(
                f"Companion color column `{colors_column_name}` is missing a color for category `{category}`; "
                "using the default categorical palette."
            )
            return "default_invalid", default_categorical_palette_for_categories(categories)

        normalized_colors: set[tuple[float, float, float, float]] = set()
        for color in category_colors:
            normalized_color = _normalize_color(color)
            if normalized_color is None:
                logger.warning(
                    f"Companion color column `{colors_column_name}` contains invalid color values; "
                    "using the default categorical palette."
                )
                return "default_invalid", default_categorical_palette_for_categories(categories)
            normalized_colors.add(normalized_color)

        if len(normalized_colors) != 1:
            logger.warning(
                f"Companion color column `{colors_column_name}` maps category `{category}` to multiple colors; "
                "using the default categorical palette."
            )
            return "default_invalid", default_categorical_palette_for_categories(categories)

        color_by_category[normalize_category_value(category)] = next(iter(normalized_colors))

    logger.info(f"Using companion color column `{colors_column_name}` for shapes viewer coloring.")
    return "stored", [color_by_category[normalize_category_value(category)] for category in categories]


def _apply_rendered_row_colors_to_shapes_layer(
    layer: Shapes,
    rendered_row_colors: np.ndarray,
    *,
    fill: bool = False,
) -> None:
    layer.face_color = _copy_rgba_with_alpha(
        rendered_row_colors,
        _styled_shapes_face_alpha(fill),
        row_count=len(layer.data),
        row_kind="shape",
    )
    layer.edge_color = _copy_rgba_with_alpha(
        rendered_row_colors,
        SHAPES_EDGE_ALPHA,
        row_count=len(layer.data),
        row_kind="shape",
    )
    refresh = getattr(layer, "refresh", None)
    if callable(refresh):
        refresh()


def _apply_rendered_row_colors_to_points_layer(
    layer: Points,
    rendered_row_colors: np.ndarray,
) -> None:
    colors = _copy_rgba_with_alpha(
        rendered_row_colors,
        SHAPES_EDGE_ALPHA,
        row_count=len(layer.data),
        row_kind="point",
    )
    layer.face_color = colors
    layer.border_color = colors
    layer.border_width = 0
    refresh = getattr(layer, "refresh", None)
    if callable(refresh):
        refresh()


def _apply_table_rendered_row_colors_to_shapes_layer(
    layer: Shapes,
    rendered_row_colors: np.ndarray,
    *,
    rendered_row_has_table_row: np.ndarray,
    fill: bool = False,
) -> None:
    if len(rendered_row_has_table_row) != len(layer.data):
        raise ValueError("Rendered-row table coverage must contain one value for each rendered napari shape row.")

    face_color = _copy_rgba_with_alpha(
        rendered_row_colors,
        _styled_shapes_face_alpha(fill),
        row_count=len(layer.data),
        row_kind="shape",
    )
    edge_color = _copy_rgba_with_alpha(
        rendered_row_colors,
        SHAPES_EDGE_ALPHA,
        row_count=len(layer.data),
        row_kind="shape",
    )
    # Unannotated shapes and annotated rows with missing selected values both
    # appear as missing in `rendered_row_values`, so color builders map both to
    # gray. Use the table-row coverage mask to keep missing table values gray
    # while making shapes with no table row fully transparent.
    unannotated_rows = ~rendered_row_has_table_row
    face_color[unannotated_rows, 3] = 0.0
    edge_color[unannotated_rows, 3] = 0.0
    layer.face_color = face_color
    layer.edge_color = edge_color
    refresh = getattr(layer, "refresh", None)
    if callable(refresh):
        refresh()


def _apply_table_rendered_row_colors_to_points_layer(
    layer: Points,
    rendered_row_colors: np.ndarray,
    *,
    rendered_row_has_table_row: np.ndarray,
) -> None:
    if len(rendered_row_has_table_row) != len(layer.data):
        raise ValueError("Rendered-row table coverage must contain one value for each rendered napari point row.")

    colors = _copy_rgba_with_alpha(
        rendered_row_colors,
        SHAPES_EDGE_ALPHA,
        row_count=len(layer.data),
        row_kind="point",
    )
    # Keep the same distinction as table-backed Shapes layers: annotated rows
    # with missing selected values stay gray, while points with no table row are
    # transparent.
    colors[~rendered_row_has_table_row, 3] = 0.0
    layer.face_color = colors
    layer.border_color = colors
    layer.border_width = 0
    refresh = getattr(layer, "refresh", None)
    if callable(refresh):
        refresh()


def _styled_shapes_face_alpha(fill: bool) -> float:
    return SHAPES_FACE_ALPHA if fill else 0.0


def _set_shape_style_feature(
    layer: Shapes | Points,
    values: pd.Series,
    *,
    style_column_name: str,
    source_shapes_index_feature_name: str,
) -> None:
    feature_name = disambiguate_shape_style_feature_name(style_column_name, source_shapes_index_feature_name)
    features = layer.features.copy()
    if len(features) == 0 and len(values) > 0:
        features = pd.DataFrame(index=pd.RangeIndex(len(values)))
    if len(features) != len(values):
        raise ValueError("Shapes layer features must contain one row for each rendered napari shape row.")

    features[feature_name] = values.to_numpy()
    layer.features = features


def _copy_rgba_with_alpha(colors: np.ndarray, alpha: float, *, row_count: int, row_kind: str) -> np.ndarray:
    rgba = np.asarray(colors, dtype="float64")
    if rgba.shape != (row_count, 4):
        raise ValueError(f"Rendered-row colors must contain one RGBA color for each rendered napari {row_kind} row.")
    rgba = rgba.copy()
    rgba[:, 3] = alpha
    return rgba


def _normalize_color(color: object) -> tuple[float, float, float, float] | None:
    try:
        return tuple(float(component) for component in to_rgba(color))
    except (TypeError, ValueError):
        return None


def _normalize_category_series(values: pd.Series) -> pd.Series:
    """Return object values suitable for layer.features/status display.

    This is intentionally row-wise: pandas categorical `to_numpy()` can coerce
    integer categories with missing values to floats, which makes hover/status
    values less faithful to the source data.
    """
    return pd.Series(
        [pd.NA if pd.isna(value) else normalize_category_value(value) for value in values],
        index=values.index,
        name=values.name,
        dtype="object",
    )


def _present_categories(values: pd.Series, candidate_categories: Sequence[object]) -> list[object]:
    present_values = {normalize_category_value(value) for value in values.dropna().tolist()}
    return [
        normalized_category
        for category in candidate_categories
        if (normalized_category := normalize_category_value(category)) in present_values
    ]


def _format_value_preview(values: Sequence[Any]) -> str:
    preview = ", ".join(repr(value) for value in values[:5])
    if len(values) > 5:
        preview += ", ..."
    return preview


def _is_categorical_dtype(values: pd.Series) -> bool:
    return isinstance(values.dtype, pd.CategoricalDtype)


def _is_bool_series(values: pd.Series) -> bool:
    non_null = values.dropna()
    if non_null.empty:
        return False
    if pd.api.types.is_bool_dtype(values.dtype):
        return True
    return all(isinstance(value, bool | np.bool_) for value in non_null.tolist())


def _is_exact_binary_integer_series(values: pd.Series) -> bool:
    non_null = values.dropna()
    if non_null.empty:
        return False
    if not pd.api.types.is_integer_dtype(values.dtype) and not all(
        isinstance(value, int | np.integer) and not isinstance(value, bool | np.bool_) for value in non_null.tolist()
    ):
        return False

    normalized_values = {int(value) for value in pd.to_numeric(non_null, errors="coerce").dropna().tolist()}
    return normalized_values == {0, 1}
