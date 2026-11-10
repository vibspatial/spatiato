from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

TableColorSourceKind = Literal["obs_column", "var"]
TableColorValueKind = Literal["categorical", "continuous", "instance"]
ShapeColorSourceKind = Literal["shape_column"]
ShapeColorValueKind = Literal["categorical", "continuous"]
TABLE_COLOR_SOURCE_KINDS: tuple[TableColorSourceKind, ...] = ("obs_column", "var")
TABLE_COLOR_VALUE_KINDS: tuple[TableColorValueKind, ...] = ("categorical", "continuous", "instance")
SHAPE_COLOR_SOURCE_KINDS: tuple[ShapeColorSourceKind, ...] = ("shape_column",)
SHAPE_COLOR_VALUE_KINDS: tuple[ShapeColorValueKind, ...] = ("categorical", "continuous")


def validate_table_color_source_kind(kind: str) -> TableColorSourceKind:
    """Return a validated table color source kind."""
    if kind not in TABLE_COLOR_SOURCE_KINDS:
        raise ValueError(_format_invalid_kind_error("table color source kind", kind, TABLE_COLOR_SOURCE_KINDS))

    return cast(TableColorSourceKind, kind)


def validate_table_color_value_kind(kind: str) -> TableColorValueKind:
    """Return a validated table color value kind."""
    if kind not in TABLE_COLOR_VALUE_KINDS:
        raise ValueError(_format_invalid_kind_error("table color value kind", kind, TABLE_COLOR_VALUE_KINDS))

    return cast(TableColorValueKind, kind)


def validate_shape_color_source_kind(kind: str) -> ShapeColorSourceKind:
    """Return a validated shape color source kind."""
    if kind not in SHAPE_COLOR_SOURCE_KINDS:
        raise ValueError(_format_invalid_kind_error("shape color source kind", kind, SHAPE_COLOR_SOURCE_KINDS))

    return cast(ShapeColorSourceKind, kind)


def validate_shape_color_value_kind(kind: str) -> ShapeColorValueKind:
    """Return a validated shape color value kind."""
    if kind not in SHAPE_COLOR_VALUE_KINDS:
        raise ValueError(_format_invalid_kind_error("shape color value kind", kind, SHAPE_COLOR_VALUE_KINDS))

    return cast(ShapeColorValueKind, kind)


def format_var_matrix(layer: str | None) -> str:
    """Return the anndata access of a var matrix: `X` for `layer=None`, else `layers["<layer>"]`."""
    return "X" if layer is None else f'layers["{layer}"]'


def format_var_matrix_label(layer: str | None) -> str:
    """Return the user-facing name of a var matrix: `X` for `layer=None`, else the layer name."""
    return "X" if layer is None else layer


@dataclass(frozen=True)
class TableColorSourceSpec:
    """Semantic description of one table-backed source used for viewer coloring.

    Parameters
    ----------
    table_name
        Name of the table in the SpatialData object.
    source_kind
        Where the values come from: `"obs_column"` reads the `obs` column
        `value_key`, and `"var"` reads the var `value_key` from the matrix that
        `var_layer` names.
    value_key
        Name of the `obs` column, or of the var.
    value_kind
        How the values are colored: `"categorical"`, `"continuous"`, or
        `"instance"` for the table's instance key column, where each instance
        gets its own color.
    var_layer
        Matrix of a `"var"` source: `None` for `X`, as in anndata's
        `layers[None]`, otherwise a key of `layers`. Every matrix of a table
        has the same vars by construction: anndata keeps one `var` index per
        table and rejects a layer whose shape is not `(n_obs, n_vars)`. It must
        be `None` for an `"obs_column"` source.
    """

    table_name: str
    source_kind: TableColorSourceKind
    value_key: str
    value_kind: TableColorValueKind
    var_layer: str | None = None

    def __post_init__(self) -> None:
        validate_table_color_source_kind(self.source_kind)
        validate_table_color_value_kind(self.value_kind)
        if self.var_layer is None:
            return
        if self.source_kind != "var":
            raise ValueError(
                f"Only a `var` color source has a `var_layer`; got {self.var_layer!r} for {self.source_kind!r}."
            )
        if not isinstance(self.var_layer, str) or not self.var_layer:
            raise ValueError(f"`var_layer` must be a non-empty layer name, or `None` for `X`; got {self.var_layer!r}.")

    @property
    def identity(self) -> tuple[str, TableColorSourceKind, str]:
        """Return a stable identity for preserving selection across refreshes.

        It leaves out `var_layer`, so a selected var stays selected when the user
        switches to another matrix of the table.
        """
        return (self.table_name, self.source_kind, self.value_key)

    @property
    def display_name(self) -> str:
        """Return the default user-facing name for this source."""
        return self.value_key

    @property
    def source_label(self) -> str:
        """Return the user-facing name of the values: `obs: <column>`, or `vars (<matrix>): <var>`.

        `<matrix>` is `X`, or the name of the layer.
        """
        if self.source_kind == "obs_column":
            return f"obs: {self.value_key}"
        return f"vars ({format_var_matrix_label(self.var_layer)}): {self.value_key}"


@dataclass(frozen=True)
class ShapeColumnColorSourceSpec:
    """Semantic description of one shapes column used for viewer coloring.

    The spec does not name the shapes element; callers pass its name next to
    the spec.

    Parameters
    ----------
    source_kind
        Where the values come from: always `"shape_column"`, a column of the
        shapes GeoDataFrame itself rather than of a linked table.
    value_key
        Name of the column.
    value_kind
        How the values are colored: `"categorical"` or `"continuous"`.
    """

    source_kind: ShapeColorSourceKind
    value_key: str
    value_kind: ShapeColorValueKind

    def __post_init__(self) -> None:
        validate_shape_color_source_kind(self.source_kind)
        validate_shape_color_value_kind(self.value_kind)

    @property
    def identity(self) -> tuple[ShapeColorSourceKind, str]:
        """Return a stable identity for preserving selection across refreshes."""
        return (self.source_kind, self.value_key)

    @property
    def display_name(self) -> str:
        """Return the default user-facing name for this source."""
        return self.value_key

    @property
    def source_label(self) -> str:
        """Return the user-facing name of the values: `column: <column>`."""
        return f"column: {self.value_key}"


def _format_invalid_kind_error(kind_label: str, kind: str, allowed_kinds: tuple[str, ...]) -> str:
    allowed = ", ".join(repr(allowed_kind) for allowed_kind in allowed_kinds)
    return f"Invalid {kind_label} {kind!r}. Expected one of: {allowed}."
