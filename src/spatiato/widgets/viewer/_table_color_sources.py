"""Color sources of the linked tables, shared by the labels and shapes cards of the Viewer widget."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from spatiato.core._color_source import (
    SHAPE_COLOR_SOURCE_KINDS,
    TABLE_COLOR_SOURCE_KINDS,
    ShapeColorSourceKind,
    TableColorSourceKind,
    TableColorSourceSpec,
    format_var_matrix_label,
)
from spatiato.core.color_source_options import get_table_obs_color_source_options, get_table_var_layers
from spatiato.core.spatialdata import get_table

if TYPE_CHECKING:
    from spatialdata import SpatialData

_COLOR_SOURCE_KINDS: tuple[ShapeColorSourceKind | TableColorSourceKind, ...] = (
    *SHAPE_COLOR_SOURCE_KINDS,
    *TABLE_COLOR_SOURCE_KINDS,
)


@dataclass(frozen=True)
class ColorSourceKindEntry:
    """One entry of a card's "Color source" combo.

    Parameters
    ----------
    label
        Text of the entry, such as `"Observations"` or `"Vars (counts)"`.
    source_kind
        Kind of the color sources the entry offers.
    var_layer
        Matrix of a `"var"` entry: `None` for `X`, otherwise a key of
        `layers`. It must be `None` for any other kind, as for
        `TableColorSourceSpec.var_layer`.
    """

    label: str
    source_kind: ShapeColorSourceKind | TableColorSourceKind
    var_layer: str | None = None

    def __post_init__(self) -> None:
        if self.source_kind not in _COLOR_SOURCE_KINDS:
            raise ValueError(f"Invalid color source kind {self.source_kind!r}.")
        if self.var_layer is not None and self.source_kind != "var":
            raise ValueError(f"Only a `var` entry has a `var_layer`; got {self.var_layer!r} for {self.source_kind!r}.")


@dataclass(frozen=True)
class LinkedTableColorSources:
    """The color sources one linked table offers to a card of the Viewer widget.

    Build it with `from_table(...)`. The `obs` side and the var side are held
    differently on purpose. An `obs` column is a complete source: its value
    kind is known from the column, and it reads no matrix. A var is only
    complete once a matrix is chosen, through the selected "Vars" entry, so
    the vars are held as names, and `sources_for(...)` builds their sources.

    Parameters
    ----------
    table_name
        Name of the table in the SpatialData object.
    obs_sources
        The colorable `obs` columns of the table, as complete sources.
    var_names
        The vars of the table. Every matrix of a table has the same vars, and
        every var is colored as `"continuous"`.
    var_layers
        The matrices the vars can be read from: `None` for `X` when it is set,
        then the layer names, sorted. Empty when the table has no matrix.
    """

    table_name: str
    obs_sources: tuple[TableColorSourceSpec, ...]
    var_names: tuple[str, ...]
    var_layers: tuple[str | None, ...]

    @classmethod
    def from_table(cls, sdata: SpatialData, table_name: str) -> LinkedTableColorSources:
        """Return the color sources of the linked table `table_name`.

        The `obs` columns are classified by `get_table_obs_color_source_options(...)`,
        and the matrices are listed by `get_table_var_layers(...)`.
        """
        table = get_table(sdata, table_name)
        return cls(
            table_name=table_name,
            obs_sources=tuple(get_table_obs_color_source_options(sdata, table_name)),
            var_names=tuple(str(var_name) for var_name in table.var_names),
            var_layers=get_table_var_layers(sdata, table_name),
        )

    def kind_entries(self) -> list[ColorSourceKindEntry]:
        """Return the "Color source" entries: "Observations", then one "Vars" entry per matrix."""
        entries: list[ColorSourceKindEntry] = []
        if self.obs_sources:
            entries.append(ColorSourceKindEntry(label="Observations", source_kind="obs_column"))
        if self.var_names:
            for var_layer in self.var_layers:
                entries.append(
                    ColorSourceKindEntry(
                        label=f"Vars ({format_var_matrix_label(var_layer)})",
                        source_kind="var",
                        var_layer=var_layer,
                    )
                )
        return entries

    def sources_for(self, entry: ColorSourceKindEntry) -> list[TableColorSourceSpec]:
        """Return the color sources `entry` offers: the `obs` columns, or the vars of the entry's matrix.

        An entry of another kind, such as a shapes column, offers no sources of
        the table.
        """
        if entry.source_kind == "obs_column":
            return list(self.obs_sources)
        if entry.source_kind != "var":
            return []
        return [
            TableColorSourceSpec(
                table_name=self.table_name,
                source_kind="var",
                value_key=var_name,
                value_kind="continuous",
                var_layer=entry.var_layer,
            )
            for var_name in self.var_names
        ]
