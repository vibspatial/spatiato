"""Classification of the values of one column for viewer coloring.

Discovery decides with it which `obs` columns and shapes columns can color a
layer, and the labels and shapes coloring code decides with it how to color
them. Both read the same rule, so a column is colored the way discovery
described it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Literal, get_args

import numpy as np
import pandas as pd

ColorValueEncoding = Literal["pandas_categorical", "boolean", "binary_integer", "string", "continuous"]
COLOR_VALUE_ENCODINGS: tuple[ColorValueEncoding, ...] = get_args(ColorValueEncoding)


@dataclass(frozen=True)
class ColorValueClassification:
    """How the values of one column are colored.

    Parameters
    ----------
    encoding
        How the values are read:

        - `"pandas_categorical"`: a pandas categorical column, colored by its
          categories;
        - `"boolean"`: booleans, colored as the categories `False` and `True`;
        - `"binary_integer"`: integers that are exactly 0 and 1, both present,
          colored as the categories `0` and `1`;
        - `"string"`: strings, colored as temporary categories;
        - `"continuous"`: other numbers, colored with a continuous colormap.

        Coloring code reads it to know where the categories come from and
        whether a stored palette may be used: a `"string"` column has no
        stored palette.
    """

    encoding: ColorValueEncoding

    def __post_init__(self) -> None:
        if self.encoding not in COLOR_VALUE_ENCODINGS:
            allowed = ", ".join(repr(encoding) for encoding in COLOR_VALUE_ENCODINGS)
            raise ValueError(f"Invalid color value encoding {self.encoding!r}. Expected one of: {allowed}.")

    @property
    def value_kind(self) -> Literal["categorical", "continuous"]:
        """Return `"continuous"` for a continuous encoding, otherwise `"categorical"`."""
        return "continuous" if self.encoding == "continuous" else "categorical"


def classify_color_values(values: pd.Series) -> ColorValueClassification | None:
    """Return how the values of one column are colored, or `None` when they cannot color a layer.

    Parameters
    ----------
    values
        The full column, such as `table.obs[column]` or
        `shapes_element[column]`, never only the rows one element draws. A
        subset can classify differently: the rows of one region may hold only
        the 0s of a 0/1 column, and only 0s classify as continuous.

    Returns
    -------
    The classification, checked in this order:

    - a pandas categorical column is `"pandas_categorical"`, even when all its
      values are missing;
    - any other column without a non-missing value is `None`;
    - a boolean dtype is `"boolean"`;
    - an integer dtype is `"binary_integer"` when its values are exactly 0 and
      1, both present, and `"continuous"` otherwise;
    - a float dtype is `"continuous"`;
    - for any other dtype, such as `object`, the non-missing values decide:
      all booleans, all integers (as above), all real numbers, or all strings.
      Values of mixed or other types are `None`.

    A boolean or 0/1 column with missing values has `object` dtype in pandas,
    and is still `"boolean"` or `"binary_integer"` by its values.
    """
    if isinstance(values.dtype, pd.CategoricalDtype):
        return ColorValueClassification("pandas_categorical")

    non_missing = values.dropna()
    if non_missing.empty:
        return None

    if pd.api.types.is_bool_dtype(values.dtype):
        return ColorValueClassification("boolean")

    if pd.api.types.is_integer_dtype(values.dtype):
        return _classify_integers(non_missing.tolist())

    if pd.api.types.is_float_dtype(values.dtype):
        return ColorValueClassification("continuous")

    return _classify_scalar_values(non_missing.tolist())


def _classify_scalar_values(values: Sequence[Any]) -> ColorValueClassification | None:
    if all(_is_bool_scalar(value) for value in values):
        return ColorValueClassification("boolean")

    if all(_is_integer_scalar(value) for value in values):
        return _classify_integers(values)

    if all(_is_real_scalar(value) for value in values):
        return ColorValueClassification("continuous")

    if all(_is_string_scalar(value) for value in values):
        return ColorValueClassification("string")

    return None


def _classify_integers(values: Sequence[Any]) -> ColorValueClassification:
    """Return `"binary_integer"` when integer `values` are exactly 0 and 1, both present, else `"continuous"`."""
    unique_values = {int(value) for value in values}
    if unique_values == {0, 1}:
        return ColorValueClassification("binary_integer")
    return ColorValueClassification("continuous")


def _is_bool_scalar(value: Any) -> bool:
    return isinstance(value, bool | np.bool_)


def _is_integer_scalar(value: Any) -> bool:
    return isinstance(value, Integral) and not _is_bool_scalar(value)


def _is_real_scalar(value: Any) -> bool:
    return isinstance(value, Real) and not _is_bool_scalar(value)


def _is_string_scalar(value: Any) -> bool:
    return isinstance(value, str | bytes | np.str_ | np.bytes_)
