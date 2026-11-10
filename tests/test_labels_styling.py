from __future__ import annotations

from dataclasses import replace

import harpy as hp
import numpy as np
import pandas as pd
import pytest
from _table_matrix_fixtures import FORMATS_AND_MODES, MATRIX_TYPE_BY_FORMAT_AND_MODE, matrix_in_format
from matplotlib.colors import to_rgba
from napari.layers import Labels
from scipy import sparse
from spatialdata import SpatialData

from spatiato.core._color_source import TableColorSourceSpec
from spatiato.core.class_palette import DEFAULT_NEUTRAL_COLOR
from spatiato.core.spatialdata import get_table_var_matrix
from spatiato.viewer.labels_colormap import CompactLabelColormap, NeutralLabelColormap
from spatiato.viewer.labels_styling import (
    LabelsStyleResult,
    apply_neutral_labels_style,
    apply_table_color_source_to_labels_layer,
)


def test_apply_neutral_labels_style_installs_neutral_label_colormap() -> None:
    layer = Labels(np.asarray([[0, 3], [5, 11]], dtype=np.int32))

    apply_neutral_labels_style(layer)

    assert isinstance(layer.colormap, NeutralLabelColormap)
    assert not isinstance(layer.colormap, CompactLabelColormap)
    assert str(layer._color_mode) == "direct"
    np.testing.assert_allclose(layer.colormap.map(5), to_rgba(DEFAULT_NEUTRAL_COLOR))
    np.testing.assert_allclose(layer.colormap.map(0), np.zeros(4))


def _style_blobs_labels(sdata: SpatialData, style_spec: TableColorSourceSpec) -> tuple[Labels, LabelsStyleResult]:
    layer = Labels(np.asarray(sdata.labels["blobs_labels"]))
    result = apply_table_color_source_to_labels_layer(
        layer,
        sdata=sdata,
        labels_name="blobs_labels",
        style_spec=style_spec,
    )
    return layer, result


@pytest.mark.parametrize("var_layer", [None, "counts"])
@pytest.mark.parametrize(("x_format", "mode"), FORMATS_AND_MODES)
def test_apply_table_color_source_to_labels_layer_colors_var_continuously(
    x_format: str,
    mode: str,
    var_layer: str | None,
    sdata_blobs: SpatialData,
    tmp_path,
) -> None:
    style_spec = TableColorSourceSpec(
        table_name="table",
        source_kind="var",
        value_key="channel_1_sum",
        value_kind="continuous",
    )
    # Reference: the same var from the fixture's in-memory `X`.
    expected_layer, _ = _style_blobs_labels(sdata_blobs, style_spec)
    table = sdata_blobs["table"]
    dense_x = table.X.toarray() if sparse.issparse(table.X) else np.asarray(table.X)
    if var_layer is None:
        table.X = matrix_in_format(dense_x, x_format)
    else:
        # The values are in the layer; `X` holds their negation, so a read from `X` would give other colors.
        table.layers[var_layer] = matrix_in_format(dense_x, x_format)
        table.X = matrix_in_format(-dense_x, x_format)
    path = tmp_path / "blobs.zarr"
    sdata_blobs.write(path)
    # Small sparse blocks give a lazy CSR matrix of several Dask blocks.
    stored_sdata = hp.io.read_zarr(path, table_mode=mode, sparse_chunks=7)
    # Test setup: harpy's reader attaches the matrix in the representation under test.
    stored_matrix = get_table_var_matrix(stored_sdata["table"], var_layer)
    assert isinstance(stored_matrix, MATRIX_TYPE_BY_FORMAT_AND_MODE[x_format, mode])

    layer, result = _style_blobs_labels(stored_sdata, replace(style_spec, var_layer=var_layer))

    assert result.value_kind == "continuous"
    assert result.palette_source is None
    assert result.coercion_applied is False
    assert isinstance(layer.colormap, CompactLabelColormap)
    pd.testing.assert_frame_equal(layer.features, expected_layer.features)
