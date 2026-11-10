from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_rgba

from spatiato.viewer._styling import (
    categorical_colors_for_values,
    categorical_rgba_for_values,
    continuous_colors_for_values,
    continuous_rgba_for_values,
    zeros_mean_not_detected,
)
from spatiato.viewer.labels_colormap import (
    CompactLabelColormap,
    compact_categorical_label_colormap_from_values,
    compact_continuous_label_colormap_from_values,
)


def _colors_series_to_rgba(colors: pd.Series) -> np.ndarray:
    return np.asarray([to_rgba(color) for color in colors], dtype="float64")


def test_continuous_rgba_for_values_matches_existing_colors_with_missing_values() -> None:
    values = pd.Series([0.0, 1.0, np.nan, 0.5], index=["a", "b", "c", "d"])

    expected = _colors_series_to_rgba(continuous_colors_for_values(values))
    actual = continuous_rgba_for_values(values)

    assert actual.dtype == np.dtype("float64")
    assert actual.shape == (len(values), 4)
    np.testing.assert_allclose(actual, expected)


def test_continuous_rgba_for_values_matches_existing_colors_for_constant_values() -> None:
    values = pd.Series([2.0, 2.0, np.nan], index=[10, 11, 12])

    expected = _colors_series_to_rgba(continuous_colors_for_values(values))
    actual = continuous_rgba_for_values(values)

    np.testing.assert_allclose(actual, expected)


def test_continuous_rgba_for_values_matches_existing_colors_for_all_missing_values() -> None:
    values = pd.Series([np.nan, pd.NA, None], dtype="object")

    expected = _colors_series_to_rgba(continuous_colors_for_values(values))
    actual = continuous_rgba_for_values(values)

    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (pd.Series([0.0, 3.0, np.nan]), True),  # counts with a missing value
        (pd.Series([0.0, 0.0]), True),
        (pd.Series([0.0, 1.5, 2.0]), True),  # normalized or log1p data
        (pd.Series([-0.5, 0.0, 1.2]), False),  # scaled data: 0 is an ordinary value
        (pd.Series([np.nan, np.nan]), False),
        (pd.Series([], dtype="float64"), False),
    ],
)
def test_zeros_mean_not_detected_only_for_non_negative_values(values: pd.Series, expected: bool) -> None:
    assert zeros_mean_not_detected(values) is expected


def test_categorical_rgba_for_values_matches_existing_colors_with_missing_values() -> None:
    categories = ["T", "B", "NK"]
    palette = ["#ff0000", "#00ff00", "#0000ff"]
    values = pd.Series(pd.Categorical(["T", "B", None, "NK"], categories=categories))

    expected = _colors_series_to_rgba(categorical_colors_for_values(values, categories=categories, palette=palette))
    actual = categorical_rgba_for_values(values, categories=categories, palette=palette)

    assert actual.dtype == np.dtype("float64")
    assert actual.shape == (len(values), 4)
    np.testing.assert_allclose(actual, expected)


def test_categorical_rgba_for_values_matches_existing_colors_for_unknown_values() -> None:
    categories = ["T", "B"]
    palette = ["#ff0000", "#00ff00"]
    values = pd.Series(["T", "unknown", None, "B"], dtype="object")

    expected = _colors_series_to_rgba(categorical_colors_for_values(values, categories=categories, palette=palette))
    actual = categorical_rgba_for_values(values, categories=categories, palette=palette)

    np.testing.assert_allclose(actual, expected)


def test_categorical_rgba_for_values_matches_existing_numpy_scalar_normalization() -> None:
    categories = [np.int64(1), np.int64(2)]
    palette = ["#ff0000", "#00ff00"]
    values = pd.Series([1, np.int64(2), 3, None], dtype="object")

    expected = _colors_series_to_rgba(categorical_colors_for_values(values, categories=categories, palette=palette))
    actual = categorical_rgba_for_values(values, categories=categories, palette=palette)

    np.testing.assert_allclose(actual, expected)


def test_build_continuous_label_colormap_uses_compact_mapping_and_preserves_background() -> None:
    values = pd.Series([0.0, 10.0, np.nan], index=pd.Index([1, 2, 3], name="index"))

    colormap = compact_continuous_label_colormap_from_values(values)
    expected = continuous_rgba_for_values(values)

    assert isinstance(colormap, CompactLabelColormap)
    np.testing.assert_allclose(colormap.map(0), np.zeros(4, dtype=np.float32))
    np.testing.assert_allclose(colormap.map(values.index.to_numpy(dtype=np.int64)), expected)


def test_build_categorical_label_colormap_uses_compact_mapping_and_preserves_colors() -> None:
    categories = ["T", "B"]
    palette = ["#ff0000", "#00ff00"]
    values = pd.Series(
        ["T", "unknown", None, "B"],
        index=pd.Index([1, 2, 3, 4], name="index"),
        dtype="object",
    )

    colormap = compact_categorical_label_colormap_from_values(values, categories=categories, palette=palette)
    expected = categorical_rgba_for_values(values, categories=categories, palette=palette)

    assert isinstance(colormap, CompactLabelColormap)
    np.testing.assert_allclose(colormap.map(0), np.zeros(4, dtype=np.float32))
    np.testing.assert_allclose(colormap.map(values.index.to_numpy(dtype=np.int64)), expected)
