from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from matplotlib import colormaps
from matplotlib.colors import to_rgba

from spatiato.core.class_palette import (
    DEFAULT_NEUTRAL_COLOR,
    default_categorical_colors,
)

MISSING_CATEGORICAL_COLOR = DEFAULT_NEUTRAL_COLOR
MISSING_CONTINUOUS_COLOR = DEFAULT_NEUTRAL_COLOR
OVERLAY_CONTINUOUS_COLORMAP = "magma"
STRING_CATEGORICAL_WARNING_MIN_UNIQUE_COUNT = 20
STRING_CATEGORICAL_WARNING_ROW_COUNT_DIVISOR = 100


def build_string_categorical_values(
    *,
    full_values: pd.Series,
    row_values: pd.Series,
    column_name: str,
) -> tuple[pd.Series, list[object]]:
    """Return viewer-only categorical values for a plain string/object column.

    Parameters
    ----------
    full_values
        Complete source column used for category discovery and cardinality
        warnings. Deriving categories from this complete series keeps the
        temporary categorical palette stable even when the currently rendered
        rows contain only a subset of the source values.
    row_values
        Already-aligned subset or repetition used by the viewer layer, for
        example linked labels table rows for one region or rendered shape rows
        after a ``MultiPolygon`` expands into multiple napari rows.
    column_name
        Source column name used in viewer-facing warning messages.
    """
    normalized_full_values = pd.Series(
        [normalize_string_value(value) for value in full_values],
        index=full_values.index,
        name=column_name,
        dtype="object",
    )
    normalized_row_values = pd.Series(
        [normalize_string_value(value) for value in row_values],
        index=row_values.index,
        name=column_name,
        dtype="object",
    )

    non_missing_values = normalized_full_values.dropna().tolist()
    if has_high_cardinality_string_values(non_missing_values, row_count=len(full_values)):
        unique_count = len({str(value) for value in non_missing_values})
        threshold = string_categorical_warning_threshold(len(full_values))
        logger.warning(
            f"Column `{column_name}` has {unique_count} unique string values across "
            f"{len(full_values)} rows, which exceeds the categorical viewer-coloring threshold of {threshold}. "
            "Spatiato will render it with the default categorical palette anyway; "
            "convert the column to pandas categorical dtype to mark this as intentional."
        )
    else:
        logger.warning(
            f"Coercing plain string/object column `{column_name}` to temporary categorical values for viewer coloring."
        )

    categories = list(pd.unique(normalized_full_values.dropna()))
    return normalized_row_values, categories


def string_categorical_warning_threshold(row_count: int) -> int:
    return max(STRING_CATEGORICAL_WARNING_MIN_UNIQUE_COUNT, row_count // STRING_CATEGORICAL_WARNING_ROW_COUNT_DIVISOR)


def has_high_cardinality_string_values(values: Sequence[Any], *, row_count: int) -> bool:
    return len({str(value) for value in values}) > string_categorical_warning_threshold(row_count)


def is_string_like_series(values: pd.Series) -> bool:
    non_null = values.dropna()
    if non_null.empty:
        return False
    return all(_is_string_scalar(value) for value in non_null.tolist())


def normalize_string_value(value: object) -> object:
    if pd.isna(value):
        return pd.NA
    return str(value)


def normalize_category_value(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    return value


def default_categorical_palette_for_categories(categories: Sequence[object]) -> list[str]:
    return default_categorical_colors(len(categories))


def categorical_colors_for_values(
    values: pd.Series,
    *,
    categories: Sequence[object],
    palette: Sequence[str],
    missing_color: Any = MISSING_CATEGORICAL_COLOR,
) -> pd.Series:
    lookup = {normalize_category_value(category): color for category, color in zip(categories, palette, strict=False)}
    colors = {}
    for index, value in values.items():
        if pd.isna(value):
            colors[index] = missing_color
            continue
        colors[index] = lookup.get(normalize_category_value(value), missing_color)
    return pd.Series(colors, index=values.index, dtype="object")


def categorical_rgba_for_values(
    values: pd.Series,
    *,
    categories: Sequence[object],
    palette: Sequence[str],
    missing_color: Any = MISSING_CATEGORICAL_COLOR,
) -> np.ndarray:
    """Return one RGBA color row per categorical value."""
    rgba = np.empty((len(values), 4), dtype="float64")
    rgba[:] = to_rgba(missing_color)
    if len(values) == 0:
        return rgba

    normalized_categories = [normalize_category_value(category) for category in categories]
    palette_rgba = np.asarray(
        [to_rgba(color) for _, color in zip(normalized_categories, palette, strict=False)],
        dtype="float64",
    )
    if len(palette_rgba) == 0:
        return rgba

    category_code_by_value = {
        category: code for code, category in enumerate(normalized_categories[: len(palette_rgba)])
    }
    if isinstance(values.dtype, pd.CategoricalDtype):
        value_categories = [normalize_category_value(category) for category in values.cat.categories]
        palette_code_by_value_code = np.asarray(
            [category_code_by_value.get(category, -1) for category in value_categories],
            dtype=np.int64,
        )
        value_codes = values.cat.codes.to_numpy(copy=False)
        present_values = value_codes >= 0
        if np.any(present_values):
            palette_codes = palette_code_by_value_code[value_codes[present_values]]
            known_values = palette_codes >= 0
            present_indices = np.flatnonzero(present_values)
            rgba[present_indices[known_values]] = palette_rgba[palette_codes[known_values]]
        return rgba

    normalized_values = pd.Series(
        [pd.NA if pd.isna(value) else normalize_category_value(value) for value in values],
        index=values.index,
        dtype="object",
    )
    palette_codes = normalized_values.map(category_code_by_value)
    known_values = palette_codes.notna().to_numpy(dtype=bool, copy=False)
    if np.any(known_values):
        rgba[known_values] = palette_rgba[palette_codes.loc[known_values].to_numpy(dtype=np.int64, copy=False)]
    return rgba


def continuous_colors_for_values(
    values: pd.Series,
    *,
    missing_color: Any = MISSING_CONTINUOUS_COLOR,
    colormap_name: str = OVERLAY_CONTINUOUS_COLORMAP,
) -> pd.Series:
    colors = {}
    non_missing = values.dropna()
    if non_missing.empty:
        for index in values.index:
            colors[index] = missing_color
        return pd.Series(colors, index=values.index, dtype="object")

    cmap = colormaps[colormap_name]
    min_value = float(non_missing.min())
    max_value = float(non_missing.max())
    if max_value == min_value:
        normalized_values = dict.fromkeys(non_missing.index, 0.5)
    else:
        normalized_values = {
            index: float((value - min_value) / (max_value - min_value)) for index, value in non_missing.items()
        }

    for index in values.index:
        value = values.at[index]
        if pd.isna(value):
            colors[index] = missing_color
        else:
            colors[index] = cmap(float(np.clip(normalized_values[index], 0.0, 1.0)))
    return pd.Series(colors, index=values.index, dtype="object")


def continuous_rgba_for_values(
    values: pd.Series,
    *,
    missing_color: Any = MISSING_CONTINUOUS_COLOR,
    colormap_name: str = OVERLAY_CONTINUOUS_COLORMAP,
) -> np.ndarray:
    """Return one RGBA color row per continuous value."""
    value_array = pd.to_numeric(values, errors="coerce").to_numpy(dtype="float64", copy=False)
    rgba = np.empty((len(value_array), 4), dtype="float64")
    rgba[:] = to_rgba(missing_color)
    present_values = ~np.isnan(value_array)
    if not np.any(present_values):
        return rgba

    non_missing = value_array[present_values]
    cmap = colormaps[colormap_name]
    min_value = float(non_missing.min())
    max_value = float(non_missing.max())
    if max_value == min_value:
        normalized_values = np.full(len(non_missing), 0.5, dtype="float64")
    else:
        normalized_values = np.clip((non_missing - min_value) / (max_value - min_value), 0.0, 1.0)
    rgba[present_values] = cmap(normalized_values)
    return rgba


def zeros_mean_not_detected(values: pd.Series) -> bool:
    """Return whether exact zeros in an expression (`var`) column mean "not detected".

    True when every finite value is non-negative, as in counts, normalized or
    log1p data. Table styling then draws those zeros transparent: in the
    colormap's lowest color (near black for magma) they would hide the image
    underneath. Scaled data with negative values keeps 0 as an ordinary value.
    """
    numeric_values = pd.to_numeric(values, errors="coerce").to_numpy(dtype="float64", copy=False)
    finite_values = numeric_values[np.isfinite(numeric_values)]
    return finite_values.size > 0 and bool(np.all(finite_values >= 0))


def _is_string_scalar(value: object) -> bool:
    return isinstance(value, (str, bytes, np.str_, np.bytes_))
