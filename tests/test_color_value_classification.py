from __future__ import annotations

import datetime
from decimal import Decimal
from fractions import Fraction

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
        pytest.param(
            pd.Series([np.int64(0), np.uint8(1), 0], dtype="object"),
            "binary_integer",
            id="object-numpy-and-python-zeros-and-ones",
        ),
        pytest.param(pd.Series([1, 1, 1], dtype="int64"), "continuous", id="int-only-ones"),
        pytest.param(pd.Series([0, 1, 2], dtype="int64"), "continuous", id="int"),
        pytest.param(pd.Series([3, None, 7], dtype="Int64"), "continuous", id="nullable-int"),
        pytest.param(pd.Series([0, 2**63 + 5], dtype="uint64"), "continuous", id="uint64-beyond-int64"),
        pytest.param(pd.Series([3, None, 7], dtype="object"), "continuous", id="object-int-with-missing"),
        pytest.param(pd.Series([2**70, 0], dtype="object"), "continuous", id="object-int-beyond-int64"),
        pytest.param(pd.Series([0.0, 1.0, 0.0], dtype="float64"), "continuous", id="float-zeros-and-ones"),
        pytest.param(pd.Series([0.5, np.nan, 2.5], dtype="float64"), "continuous", id="float-with-missing"),
        pytest.param(pd.Series([0.5, None, 2.5], dtype="object"), "continuous", id="object-floats"),
        pytest.param(pd.Series([1, 2.5, None], dtype="object"), "continuous", id="object-real-numbers"),
        pytest.param(pd.Series(["alpha", None, "beta"], dtype="object"), "string", id="object-strings"),
        pytest.param(pd.Series([b"alpha", None, b"beta"], dtype="object"), "string", id="object-bytes"),
        pytest.param(pd.Series(["alpha", None, "beta"], dtype="string"), "string", id="string-dtype"),
        # Mixes that pandas' type inference does not name exactly are checked value by value.
        pytest.param(pd.Series(["alpha", b"beta"], dtype="object"), "string", id="object-strings-and-bytes"),
        pytest.param(pd.Series([Fraction(1, 2), Fraction(3, 4)], dtype="object"), "continuous", id="object-fractions"),
        pytest.param(
            pd.Series([1, Fraction(1, 2)], dtype="object"),
            "continuous",
            id="object-integers-and-fractions",
        ),
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
        pytest.param(pd.Series(pd.to_timedelta([1, 2], unit="s")), id="timedelta"),
        pytest.param(pd.Series(pd.period_range("2024-01", periods=2, freq="M")), id="period"),
        pytest.param(pd.Series(pd.interval_range(0, 2)), id="interval"),
        pytest.param(pd.Series([1 + 2j, 3j], dtype="complex128"), id="complex"),
        pytest.param(
            pd.Series([datetime.datetime(2024, 1, 1), datetime.datetime(2024, 1, 2)], dtype="object"),
            id="object-datetimes",
        ),
        pytest.param(
            pd.Series([datetime.date(2024, 1, 1), datetime.date(2024, 1, 2)], dtype="object"), id="object-dates"
        ),
        pytest.param(pd.Series([datetime.time(1, 2), datetime.time(3, 4)], dtype="object"), id="object-times"),
        pytest.param(
            pd.Series([datetime.timedelta(days=1), datetime.timedelta(days=2)], dtype="object"),
            id="object-timedeltas",
        ),
        pytest.param(pd.Series([Decimal("1.5"), Decimal("2")], dtype="object"), id="object-decimals"),
    ],
)
def test_classify_color_values_returns_none_for_values_that_cannot_color(values: pd.Series) -> None:
    assert classify_color_values(values) is None


def test_classify_color_values_returns_none_for_an_object_column_of_numpy_timedeltas() -> None:
    """numpy makes `np.timedelta64` a subclass of its integer types, but a timedelta does not color a layer.

    Checking each value, the column would pass as integers and fail when the
    values are read as `int`.
    """
    values = pd.Series([np.timedelta64(0, "s"), np.timedelta64(1, "s")], dtype="object")

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
