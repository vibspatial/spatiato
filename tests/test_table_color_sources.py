from __future__ import annotations

import pandas as pd
import pytest
from spatialdata import SpatialData

from spatiato.core._color_source import TableColorSourceSpec
from spatiato.widgets.viewer._table_color_sources import ColorSourceKindEntry, LinkedTableColorSources

_OBS_SOURCE = TableColorSourceSpec(
    table_name="table", source_kind="obs_column", value_key="cell_type", value_kind="categorical"
)
_OBSERVATIONS_ENTRY = ColorSourceKindEntry(label="Observations", source_kind="obs_column")


def test_linked_table_color_sources_from_table_collects_the_obs_sources_vars_and_matrices(
    sdata_blobs: SpatialData,
) -> None:
    table = sdata_blobs["table"]
    table.obs["cat_obs"] = pd.Categorical(["a"] * table.n_obs)
    table.layers["counts"] = table.X.copy()

    linked_table = LinkedTableColorSources.from_table(sdata_blobs, "table")

    assert linked_table.table_name == "table"
    assert [source.identity for source in linked_table.obs_sources] == [
        ("table", "obs_column", "instance_id"),
        ("table", "obs_column", "cat_obs"),
    ]
    assert linked_table.var_names == ("channel_0_sum", "channel_1_sum", "channel_2_sum")
    assert linked_table.var_layers == (None, "counts")
    assert linked_table.kind_entries() == [
        _OBSERVATIONS_ENTRY,
        ColorSourceKindEntry(label="Vars (X)", source_kind="var"),
        ColorSourceKindEntry(label="Vars (counts)", source_kind="var", var_layer="counts"),
    ]


def test_linked_table_color_sources_offer_the_vars_of_the_matrix_of_an_entry() -> None:
    linked_table = LinkedTableColorSources(
        table_name="table",
        obs_sources=(_OBS_SOURCE,),
        var_names=("GeneA", "GeneB"),
        var_layers=(None, "counts"),
    )
    obs_entry, x_entry, counts_entry = linked_table.kind_entries()

    assert linked_table.sources_for(obs_entry) == [_OBS_SOURCE]
    assert linked_table.sources_for(counts_entry) == [
        TableColorSourceSpec(
            table_name="table", source_kind="var", value_key=var_name, value_kind="continuous", var_layer="counts"
        )
        for var_name in ("GeneA", "GeneB")
    ]
    assert [source.var_layer for source in linked_table.sources_for(x_entry)] == [None, None]
    assert linked_table.sources_for(ColorSourceKindEntry(label="Shapes column", source_kind="shape_column")) == []


@pytest.mark.parametrize(
    ("var_names", "var_layers"),
    [(("GeneA",), ()), ((), (None,))],
    ids=["without_a_matrix", "without_vars"],
)
def test_linked_table_color_sources_offer_no_vars_entry(
    var_names: tuple[str, ...], var_layers: tuple[str | None, ...]
) -> None:
    linked_table = LinkedTableColorSources(
        table_name="table", obs_sources=(_OBS_SOURCE,), var_names=var_names, var_layers=var_layers
    )

    assert linked_table.kind_entries() == [_OBSERVATIONS_ENTRY]


def test_color_source_kind_entry_has_a_var_layer_only_for_a_var_entry() -> None:
    with pytest.raises(ValueError, match="Only a `var` entry has a `var_layer`"):
        ColorSourceKindEntry(label="Observations", source_kind="obs_column", var_layer="counts")
    with pytest.raises(ValueError, match="Invalid color source kind"):
        ColorSourceKindEntry(label="Layers", source_kind="layer")
