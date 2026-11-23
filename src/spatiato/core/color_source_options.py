"""Discovery of the values that can color a viewer layer: table `obs` columns and matrices, and shapes columns."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from spatiato.core._color_source import (
    ShapeColumnColorSourceSpec,
    TableColorSourceSpec,
    TableColorValueKind,
)
from spatiato.core.color_value_classification import classify_color_values
from spatiato.core.spatialdata import get_table, get_table_metadata

if TYPE_CHECKING:
    from spatialdata import SpatialData


def get_table_obs_color_source_options(sdata: SpatialData, table_name: str) -> list[TableColorSourceSpec]:
    """Return colorable `obs` columns for one linked table."""
    table = get_table(sdata, table_name)
    table_metadata = get_table_metadata(sdata, table_name)
    excluded_columns = {table_metadata.region_key}

    options: list[TableColorSourceSpec] = []
    for column_name in table.obs.columns:
        if column_name in excluded_columns:
            continue

        value_kind: TableColorValueKind
        if column_name == table_metadata.instance_key:
            value_kind = "instance"
        else:
            classification = classify_color_values(table.obs[column_name])
            if classification is None:
                continue
            value_kind = classification.value_kind

        options.append(
            TableColorSourceSpec(
                table_name=table_name,
                source_kind="obs_column",
                value_key=str(column_name),
                value_kind=value_kind,
            )
        )

    return options


def get_table_var_layers(sdata: SpatialData, table_name: str) -> tuple[str | None, ...]:
    """Return the matrices of one linked table whose vars can color: `None` for `X`, then the layer names.

    `X` comes first, when it is set; anndata also lists it among the layers,
    as `layers[None]`. The layer names follow, sorted: harpy's lazy reader
    returns them in no fixed order.
    """
    table = get_table(sdata, table_name)
    x_layer: tuple[str | None, ...] = () if table.X is None else (None,)
    return x_layer + tuple(sorted(key for key in table.layers.keys() if key is not None))


def get_shape_column_color_source_options(sdata: SpatialData, shapes_name: str) -> list[ShapeColumnColorSourceSpec]:
    """Return colorable columns stored directly on a shapes GeoDataFrame.

    This discovers values that can color `sdata.shapes[shapes_name]` from the
    shapes element itself. Linked AnnData table sources are discovered
    separately: `.obs` columns with `get_table_obs_color_source_options(...)`,
    and the matrices that vars can be read from with `get_table_var_layers(...)`.
    """
    shapes = sdata.shapes[shapes_name]
    geometry_column_name = _get_geometry_column_name(shapes)

    options: list[ShapeColumnColorSourceSpec] = []
    for column_name in shapes.columns:
        column_key = str(column_name)
        if column_name == geometry_column_name or _is_shape_color_helper_column(column_key):
            continue

        classification = classify_color_values(shapes[column_name])
        if classification is None:
            continue

        options.append(
            ShapeColumnColorSourceSpec(
                source_kind="shape_column",
                value_key=column_key,
                value_kind=classification.value_kind,
            )
        )

    return options


def _get_geometry_column_name(shapes_element: Any) -> Any:
    return getattr(getattr(shapes_element, "geometry", None), "name", "geometry")


def _is_shape_color_helper_column(column_name: str) -> bool:
    return column_name.endswith(("_colors", "_color", ".color"))
