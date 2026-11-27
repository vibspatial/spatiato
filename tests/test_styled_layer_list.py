from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from napari.layers import Labels

from spatiato.core._color_source import TableColorSourceSpec
from spatiato.viewer.adapter import LabelsLayerBinding
from spatiato.widgets.viewer._table_color_sources import LinkedTableColorSources
from spatiato.widgets.viewer.labels_widget import _LabelsCardWidget
from spatiato.widgets.viewer.styled_layer_list import _StyledLayerList

CELL_TYPE = TableColorSourceSpec(table_name="table", source_kind="obs_column", value_key="cell_type")
GENE_A = TableColorSourceSpec(table_name="table", source_kind="var", value_key="GeneA")
OTHER_CELL_TYPE = TableColorSourceSpec(table_name="other_table", source_kind="obs_column", value_key="cell_type")


def _styled_binding(style_spec: TableColorSourceSpec, *, labels_name: str = "cells") -> LabelsLayerBinding:
    return LabelsLayerBinding(
        layer=Labels(np.zeros((4, 4), dtype=np.int32)),
        element_name=labels_name,
        coordinate_system="global",
        sdata_id=1,
        labels_role="styled",
        style_spec=style_spec,
    )


def _linked_table(table_name: str) -> LinkedTableColorSources:
    return LinkedTableColorSources(
        table_name=table_name,
        obs_sources=(TableColorSourceSpec(table_name=table_name, source_kind="obs_column", value_key="cell_type"),),
        var_names=(),
        var_layers=(),
    )


def _row_labels(styled_list: _StyledLayerList) -> list[str]:
    return [row.display_label for row in styled_list.rows]


def test_styled_layer_list_reconciles_rows_in_snapshot_order(qtbot) -> None:
    styled_list = _StyledLayerList(element_name="cells")
    qtbot.addWidget(styled_list)
    cell_type = _styled_binding(CELL_TYPE)
    gene_a = _styled_binding(GENE_A)

    styled_list.set_styled_bindings([cell_type, gene_a])
    cell_type_row, gene_a_row = styled_list.rows

    assert _row_labels(styled_list) == ["obs: cell_type", "vars (X): GeneA"]
    assert styled_list.loaded_specs == (CELL_TYPE, GENE_A)
    assert styled_list.count_label.text() == "2 colored layers"

    # A new order keeps the rows; a new binding object for a source rebuilds its row.
    replaced_gene_a = replace(gene_a)
    styled_list.set_styled_bindings([replaced_gene_a, cell_type])

    assert styled_list.rows[1] is cell_type_row
    assert styled_list.rows[0] is not gene_a_row
    assert styled_list.rows[0].binding is replaced_gene_a

    # The disposed row no longer follows its layer.
    gene_a.layer.visible = False
    assert gene_a_row.visibility_button.isChecked()


def test_styled_layer_list_rejects_a_snapshot_with_a_source_twice_before_changing_rows(qtbot) -> None:
    styled_list = _StyledLayerList(element_name="cells")
    qtbot.addWidget(styled_list)
    cell_type = _styled_binding(CELL_TYPE)
    styled_list.set_styled_bindings([cell_type])
    row = styled_list.rows[0]

    with pytest.raises(ValueError, match="multiple live styled layers for `cells` and source `obs: cell_type`"):
        styled_list.set_styled_bindings([_styled_binding(GENE_A), cell_type, _styled_binding(CELL_TYPE)])

    assert styled_list.rows == [row]


def test_styled_layer_list_forwards_intent_only_from_the_current_row(qtbot) -> None:
    styled_list = _StyledLayerList(element_name="cells")
    qtbot.addWidget(styled_list)
    removals: list[TableColorSourceSpec] = []
    visibility: list[tuple[TableColorSourceSpec, bool]] = []
    remove_all: list[str] = []
    styled_list.remove_requested.connect(removals.append)
    styled_list.visibility_requested.connect(lambda spec, visible: visibility.append((spec, visible)))
    styled_list.remove_all_requested.connect(lambda: remove_all.append("all"))
    binding = _styled_binding(CELL_TYPE)
    styled_list.set_styled_bindings([binding])
    stale_row = styled_list.rows[0]

    stale_row.visibility_button.click()
    stale_row.remove_button.click()
    styled_list.remove_all_button.click()

    assert visibility == [(CELL_TYPE, False)]
    assert removals == [CELL_TYPE]
    assert remove_all == ["all"]
    # The list never changes the layer itself.
    assert binding.layer.visible is True

    styled_list.set_styled_bindings([replace(binding)])
    stale_row.remove_button.click()

    assert removals == [CELL_TYPE]


def test_styled_layer_list_keeps_warnings_by_source_across_row_rebuilds(qtbot) -> None:
    styled_list = _StyledLayerList(element_name="cells")
    qtbot.addWidget(styled_list)
    binding = _styled_binding(CELL_TYPE)
    styled_list.set_styled_bindings([binding])
    message = "Source `obs: cell_type` is no longer available."

    styled_list.set_styled_source_warning(CELL_TYPE, message)

    assert styled_list.rows[0].warning_message == message

    # A new binding object, for example after a Fill change, rebuilds the row.
    replaced = replace(binding)
    styled_list.set_styled_bindings([replaced])

    assert styled_list.rows[0].binding is replaced
    assert styled_list.rows[0].warning_message == message

    styled_list.set_styled_source_warning(CELL_TYPE, None)

    assert styled_list.rows[0].warning_message is None
    assert styled_list.warning_for(CELL_TYPE) is None


def test_styled_layer_list_drops_the_warning_of_a_source_that_is_no_longer_loaded(qtbot) -> None:
    styled_list = _StyledLayerList(element_name="cells")
    qtbot.addWidget(styled_list)
    styled_list.set_styled_bindings([_styled_binding(CELL_TYPE)])
    styled_list.set_styled_source_warning(CELL_TYPE, "Gone.")
    # A warning for a source without a row is ignored.
    styled_list.set_styled_source_warning(GENE_A, "Gone too.")

    styled_list.set_styled_bindings([])
    styled_list.set_styled_bindings([_styled_binding(CELL_TYPE), _styled_binding(GENE_A)])

    assert [row.warning_message for row in styled_list.rows] == [None, None]


def test_styled_layer_list_prefixes_table_sources_with_their_table_name_on_request(qtbot) -> None:
    styled_list = _StyledLayerList(element_name="cells")
    qtbot.addWidget(styled_list)
    styled_list.set_styled_bindings([_styled_binding(CELL_TYPE), _styled_binding(OTHER_CELL_TYPE)])

    assert _row_labels(styled_list) == ["obs: cell_type", "obs: cell_type"]
    rows = styled_list.rows

    styled_list.set_show_table_names(True)

    assert styled_list.rows == rows
    assert _row_labels(styled_list) == ["table · obs: cell_type", "other_table · obs: cell_type"]
    assert styled_list.rows[0].remove_button.accessibleName() == (
        "Remove colored layer table · obs: cell_type from viewer"
    )


def test_labels_card_prefixes_rows_with_the_table_name_only_with_several_linked_tables(qtbot) -> None:
    card = _LabelsCardWidget(
        labels_name="cells",
        linked_tables={"table": _linked_table("table"), "other_table": _linked_table("other_table")},
    )
    qtbot.addWidget(card)

    card.set_loaded_labels_bindings(primary_binding=None, styled_bindings=[_styled_binding(CELL_TYPE)])

    assert [row.display_label for row in card.styled_rows] == ["table · obs: cell_type"]
    row = card.styled_rows[0]

    card.set_linked_tables({"table": _linked_table("table")})

    assert card.styled_rows == [row]
    assert [row.display_label for row in card.styled_rows] == ["obs: cell_type"]


def test_labels_card_rejects_bindings_of_another_element_before_changing_rows(qtbot) -> None:
    card = _LabelsCardWidget(labels_name="cells", linked_tables={"table": _linked_table("table")})
    qtbot.addWidget(card)
    card.set_loaded_labels_bindings(primary_binding=None, styled_bindings=[_styled_binding(CELL_TYPE)])
    row = card.styled_rows[0]

    with pytest.raises(ValueError, match="not for labels `cells`"):
        card.set_loaded_labels_bindings(
            primary_binding=None,
            styled_bindings=[_styled_binding(GENE_A, labels_name="nuclei")],
        )

    assert card.styled_rows == [row]
