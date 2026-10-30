from __future__ import annotations

import harpy as hp
import numpy as np
import pandas as pd
import pytest
from _table_matrix_fixtures import FORMATS_AND_MODES, MATRIX_TYPE_BY_FORMAT_AND_MODE, matrix_in_format
from napari.layers import Labels
from scipy import sparse
from spatialdata import SpatialData

from spatiato.core._color_source import TableColorSourceSpec
from spatiato.viewer.labels_colormap import CompactLabelColormap
from spatiato.viewer.labels_styling import LabelsStyleResult, apply_table_color_source_to_labels_layer


def _style_blobs_labels(sdata: SpatialData, style_spec: TableColorSourceSpec) -> tuple[Labels, LabelsStyleResult]:
    layer = Labels(np.asarray(sdata.labels["blobs_labels"]))
    result = apply_table_color_source_to_labels_layer(
        layer,
        sdata=sdata,
        labels_name="blobs_labels",
        style_spec=style_spec,
    )
    return layer, result


@pytest.mark.parametrize(("x_format", "mode"), FORMATS_AND_MODES)
def test_apply_table_color_source_to_labels_layer_colors_x_var_continuously(
    x_format: str,
    mode: str,
    sdata_blobs: SpatialData,
    tmp_path,
) -> None:
    style_spec = TableColorSourceSpec(
        table_name="table",
        source_kind="x_var",
        value_key="channel_1_sum",
        value_kind="continuous",
    )
    # Reference: the same var from the fixture's in-memory `X`.
    expected_layer, _ = _style_blobs_labels(sdata_blobs, style_spec)
    x_values = sdata_blobs["table"].X
    dense_x = x_values.toarray() if sparse.issparse(x_values) else np.asarray(x_values)
    sdata_blobs["table"].X = matrix_in_format(dense_x, x_format)
    path = tmp_path / "blobs.zarr"
    sdata_blobs.write(path)
    # Small sparse blocks give a lazy CSR `X` of several Dask blocks.
    stored_sdata = hp.io.read_zarr(path, table_mode=mode, sparse_chunks=7)
    # Test setup: harpy's reader attaches `X` in the representation under test.
    assert isinstance(stored_sdata["table"].X, MATRIX_TYPE_BY_FORMAT_AND_MODE[x_format, mode])

    layer, result = _style_blobs_labels(stored_sdata, style_spec)

    assert result.value_kind == "continuous"
    assert result.palette_source is None
    assert result.coercion_applied is False
    assert isinstance(layer.colormap, CompactLabelColormap)
    pd.testing.assert_frame_equal(layer.features, expected_layer.features)
