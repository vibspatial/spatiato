from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spatiato.core.color_value_classification import ColorValueClassification, classify_color_values


@pytest.mark.parametrize(
    ("values", "expected_encoding"),
    [
        pytest.param(pd.Series(pd.Categorical(["a", "b", "a"])), "pandas_categorical", id="categorical"),
        pytest.param(
            pd.Series(pd.Categorical([None, None], categories=["a", "b"])),
            "pandas_categorical",
            id="categorical-all-missing",
        ),
        pytest.param(pd.Series([True, False, True], dtype="bool"), "boolean", id="bool"),
        pytest.param(pd.Series([True, None, False], dtype="boolean"), "boolean", id="nullable-bool"),
        pytest.param(pd.Series([True, None, False], dtype="object"), "boolean", id="object-bool-with-missing"),
        pytest.param(pd.Series([0, 1, 0], dtype="int64"), "binary_integer", id="binary-int"),
        pytest.param(pd.Series([0, None, 1], dtype="Int64"), "binary_integer", id="nullable-binary-int"),
        pytest.param(pd.Series([0, None, 1], dtype="object"), "binary_integer", id="object-binary-int-with-missing"),
        pytest.param(pd.Series([1, 1, 1], dtype="int64"), "continuous", id="int-only-ones"),
        pytest.param(pd.Series([0, 1, 2], dtype="int64"), "continuous", id="int"),
        pytest.param(pd.Series([3, None, 7], dtype="object"), "continuous", id="object-int-with-missing"),
        pytest.param(pd.Series([0.0, 1.0, 0.0], dtype="float64"), "continuous", id="float-zeros-and-ones"),
        pytest.param(pd.Series([0.5, np.nan, 2.5], dtype="float64"), "continuous", id="float-with-missing"),
        pytest.param(pd.Series([1, 2.5, None], dtype="object"), "continuous", id="object-real-numbers"),
        pytest.param(pd.Series(["alpha", None, "beta"], dtype="object"), "string", id="object-strings"),
        pytest.param(pd.Series(["alpha", None, "beta"], dtype="string"), "string", id="string-dtype"),
    ],
)
def test_classify_color_values_returns_the_encoding(values: pd.Series, expected_encoding: str) -> None:
    classification = classify_color_values(values)

    assert classification == ColorValueClassification(expected_encoding)


@pytest.mark.parametrize(
    "values",
    [
        pytest.param(pd.Series([np.nan, np.nan], dtype="float64"), id="float-all-missing"),
        pytest.param(pd.Series([None, None], dtype="object"), id="object-all-missing"),
        pytest.param(pd.Series([], dtype="float64"), id="empty"),
        pytest.param(pd.Series(["alpha", 1, "beta"], dtype="object"), id="mixed"),
        pytest.param(pd.Series([True, 1, 0], dtype="object"), id="booleans-and-integers"),
        pytest.param(pd.Series([{"a": 1}, {"b": 2}], dtype="object"), id="objects"),
        pytest.param(pd.Series(pd.date_range("2024-01-01", periods=3)), id="datetime"),
    ],
)
def test_classify_color_values_returns_none_for_values_that_cannot_color(values: pd.Series) -> None:
    assert classify_color_values(values) is None


@pytest.mark.parametrize(
    ("encoding", "expected_value_kind"),
    [
        ("pandas_categorical", "categorical"),
        ("boolean", "categorical"),
        ("binary_integer", "categorical"),
        ("string", "categorical"),
        ("continuous", "continuous"),
    ],
)
def test_color_value_classification_reports_the_value_kind(encoding: str, expected_value_kind: str) -> None:
    assert ColorValueClassification(encoding).value_kind == expected_value_kind


def test_color_value_classification_rejects_an_unknown_encoding() -> None:
    with pytest.raises(ValueError, match="Invalid color value encoding"):
        ColorValueClassification("instance")
