"""Discovery of the values that can color a viewer layer: table `obs` columns and matrices, and shapes columns."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral, Real
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from spatiato.core._color_source import (
    ShapeColorValueKind,
    ShapeColumnColorSourceSpec,
    TableColorSourceSpec,
    TableColorValueKind,
)
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

        if column_name == table_metadata.instance_key:
            value_kind = "instance"
        else:
            value_kind = _classify_obs_color_source(table.obs[column_name])
        if value_kind is None:
            continue

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

        value_kind = _classify_shape_column_color_source(shapes[column_name])
        if value_kind is None:
            continue

        options.append(
            ShapeColumnColorSourceSpec(
                source_kind="shape_column",
                value_key=column_key,
                value_kind=value_kind,
            )
        )

    return options


def _classify_obs_color_source(column: pd.Series) -> TableColorValueKind | None:
    return _classify_scalar_column_color_source(column)


def _classify_shape_column_color_source(column: pd.Series) -> ShapeColorValueKind | None:
    return _classify_scalar_column_color_source(column)


def _classify_scalar_column_color_source(column: pd.Series) -> ShapeColorValueKind | None:
    if isinstance(column.dtype, pd.CategoricalDtype):
        return "categorical"

    non_null = column.dropna()
    if non_null.empty:
        return None

    if pd.api.types.is_bool_dtype(column.dtype):
        return "categorical"

    if pd.api.types.is_integer_dtype(column.dtype):
        return "categorical" if has_exact_binary_zero_one_values(non_null.tolist()) else "continuous"

    if pd.api.types.is_float_dtype(column.dtype):
        return "continuous"

    return _classify_scalar_values_as_color_source(non_null.tolist())


def _classify_scalar_values_as_color_source(
    values: Sequence[Any],
) -> ShapeColorValueKind | None:
    if all(_is_bool_scalar(value) for value in values):
        return "categorical"

    if all(_is_integer_scalar(value) for value in values):
        return "categorical" if has_exact_binary_zero_one_values(values) else "continuous"

    if all(_is_real_scalar(value) for value in values):
        return "continuous"

    if all(_is_string_scalar(value) for value in values):
        return "categorical"

    return None


def has_exact_binary_zero_one_values(values: Sequence[Any]) -> bool:
    """Return whether integer `values` are exactly the set {0, 1}, both present.

    Such a column is colored as categorical; integers with other values, or
    with only 0s or only 1s, are colored as continuous.
    """
    unique_values = {int(value) for value in values}
    return unique_values == {0, 1}


def _get_geometry_column_name(shapes_element: Any) -> Any:
    return getattr(getattr(shapes_element, "geometry", None), "name", "geometry")


def _is_shape_color_helper_column(column_name: str) -> bool:
    return column_name.endswith(("_colors", "_color", ".color"))


def _is_bool_scalar(value: Any) -> bool:
    return isinstance(value, bool | np.bool_)


def _is_integer_scalar(value: Any) -> bool:
    return isinstance(value, Integral) and not _is_bool_scalar(value)


def _is_real_scalar(value: Any) -> bool:
    return isinstance(value, Real) and not _is_bool_scalar(value)


def _is_string_scalar(value: Any) -> bool:
    return isinstance(value, str | bytes | np.str_ | np.bytes_)
