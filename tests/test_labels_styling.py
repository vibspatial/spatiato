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


def _obs_style_spec(column_name: str, value_kind: str, *, table_name: str = "table") -> TableColorSourceSpec:
    return TableColorSourceSpec(
        table_name=table_name,
        source_kind="obs_column",
        value_key=column_name,
        value_kind=value_kind,
    )


def test_apply_table_color_source_to_labels_layer_colors_float_obs_continuously(sdata_blobs: SpatialData) -> None:
    table = sdata_blobs["table"]
    table.obs["score"] = pd.Series(np.linspace(0.0, 1.0, table.n_obs), index=table.obs.index, dtype="float64")

    layer, result = _style_blobs_labels(sdata_blobs, _obs_style_spec("score", "continuous"))

    assert result.value_kind == "continuous"
    assert result.palette_source is None
    assert result.coercion_applied is False
    assert isinstance(layer.colormap, CompactLabelColormap)


def test_apply_table_color_source_to_labels_layer_classifies_the_full_column_not_the_region_rows(
    sdata_blobs_multi_region: SpatialData,
) -> None:
    # The rows of `blobs_labels` hold only 0s and those of `blobs_labels_2` only
    # 1s. The full column is a 0/1 column, while the rows of `blobs_labels`
    # alone would classify as continuous.
    table = sdata_blobs_multi_region["table_multi"]
    table.obs["state"] = pd.Series(
        (table.obs["region"].astype(str) == "blobs_labels_2").to_numpy(dtype=np.int64),
        index=table.obs.index,
        dtype="int64",
    )
    layer = Labels(np.asarray(sdata_blobs_multi_region.labels["blobs_labels"]))

    result = apply_table_color_source_to_labels_layer(
        layer,
        sdata=sdata_blobs_multi_region,
        labels_name="blobs_labels",
        style_spec=_obs_style_spec("state", "categorical", table_name="table_multi"),
    )

    assert result.value_kind == "categorical"


@pytest.mark.parametrize(("true_value", "false_value"), [(True, False), (1, 0)], ids=["bool", "binary"])
def test_apply_table_color_source_to_labels_layer_colors_object_bool_and_binary_obs_categorically(
    sdata_blobs: SpatialData,
    true_value: object,
    false_value: object,
) -> None:
    # A boolean or 0/1 column with missing values has `object` dtype in pandas.
    table = sdata_blobs["table"]
    instance_ids = table.obs["instance_id"].to_numpy(dtype=np.int64)
    values: list[object] = []
    for row, instance_id in enumerate(instance_ids):
        if row == 0:
            values.append(None)
        elif instance_id % 2 == 0:
            values.append(true_value)
        else:
            values.append(false_value)
    table.obs["state"] = pd.Series(values, index=table.obs.index, dtype="object")
    # The palette follows the categories `[False, True]` or `[0, 1]`.
    table.uns["state_colors"] = ["#ff0000", "#00ff00"]

    layer, result = _style_blobs_labels(sdata_blobs, _obs_style_spec("state", "categorical"))

    assert result.value_kind == "categorical"
    assert result.palette_source == "stored"
    true_instance = int(table.obs.loc[table.obs["state"] == true_value, "instance_id"].iloc[0])
    false_instance = int(table.obs.loc[table.obs["state"] == false_value, "instance_id"].iloc[0])
    assert np.allclose(layer.colormap.map(true_instance), np.asarray(to_rgba("#00ff00"), dtype=np.float32))
    assert np.allclose(layer.colormap.map(false_instance), np.asarray(to_rgba("#ff0000"), dtype=np.float32))


@pytest.mark.parametrize("column_kind", ["all_missing", "mixed"])
def test_apply_table_color_source_to_labels_layer_rejects_an_obs_column_that_cannot_color(
    sdata_blobs: SpatialData,
    column_kind: str,
) -> None:
    table = sdata_blobs["table"]
    if column_kind == "all_missing":
        values = pd.Series(np.nan, index=table.obs.index, dtype="float64")
    else:
        values = pd.Series(
            ["a" if row % 2 else row for row in range(table.n_obs)],
            index=table.obs.index,
            dtype="object",
        )
    table.obs["unusable"] = values

    with pytest.raises(ValueError, match="cannot color a layer"):
        _style_blobs_labels(sdata_blobs, _obs_style_spec("unusable", "continuous"))


@pytest.mark.parametrize(("other_value", "zero_is_transparent"), [(5.0, True), (-1.0, False)])
def test_apply_table_color_source_to_labels_layer_draws_not_detected_var_zeros_transparent(
    sdata_blobs: SpatialData,
    other_value: float,
    zero_is_transparent: bool,
) -> None:
    # Non-negative data: 0 means "not detected" and must not hide the image
    # underneath. A negative value marks scaled data, where 0 is an ordinary value.
    table = sdata_blobs["table"]
    x = table.X.toarray() if sparse.issparse(table.X) else np.asarray(table.X)
    x = x.astype(np.float64)
    column = table.var_names.get_loc("channel_1_sum")
    region_rows = np.flatnonzero(table.obs["region"].astype(str).to_numpy() == "blobs_labels")
    zero_row, other_row, positive_row = region_rows[:3]
    x[region_rows, column] = 5.0
    x[zero_row, column] = 0.0
    x[other_row, column] = other_value
    table.X = x
    style_spec = TableColorSourceSpec(
        table_name="table",
        source_kind="var",
        value_key="channel_1_sum",
        value_kind="continuous",
    )

    layer, _ = _style_blobs_labels(sdata_blobs, style_spec)

    zero_label = int(table.obs["instance_id"].iloc[zero_row])
    positive_label = int(table.obs["instance_id"].iloc[positive_row])
    assert bool(layer.colormap.map(zero_label)[3] == 0.0) is zero_is_transparent
    assert layer.colormap.map(positive_label)[3] > 0.0
    assert layer.features.set_index("index").loc[zero_label, "channel_1_sum"] == 0.0


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
