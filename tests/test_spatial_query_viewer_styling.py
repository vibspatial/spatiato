from __future__ import annotations

import copy
from collections.abc import Callable
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_rgba

from spatiato.viewer._styling import MISSING_CATEGORICAL_COLOR
from spatiato.viewer.adapter import LabelsLayerBinding, ViewerAdapter
from spatiato.widgets.spatial_query.viewer_styling import (
    load_and_style_spatial_annotation_labels,
    load_and_style_unannotated_spatial_annotation_labels,
)


class _EventEmitter:
    def __init__(self) -> None:
        self._callbacks: list[Callable[[object], None]] = []

    def connect(self, callback: Callable[[object], None]) -> None:
        self._callbacks.append(callback)

    def emit(self, value: object) -> None:
        event = SimpleNamespace(value=value)
        for callback in list(self._callbacks):
            callback(event)


class _Layers(list):
    def __init__(self) -> None:
        super().__init__()
        self.selection = SimpleNamespace(active=None, select_only=self._select_only)
        self.events = SimpleNamespace(
            inserted=_EventEmitter(),
            removed=_EventEmitter(),
            reordered=_EventEmitter(),
        )

    def _select_only(self, layer: object) -> None:
        self.selection.active = layer


class _Viewer:
    def __init__(self) -> None:
        self.layers = _Layers()

    def add_layer(self, layer: object) -> object:
        self.layers.append(layer)
        self.layers.events.inserted.emit(layer)
        return layer


def _add_annotation_column(sdata) -> None:
    table = sdata.tables["table"]
    values = np.where(table.obs["instance_id"].to_numpy() % 2 == 0, "even", "odd").astype(object)
    table.obs["annotation"] = pd.Categorical(values, categories=["odd", "even"])


def test_unannotated_spatial_annotation_styling_uses_one_neutral_foreground_without_table_mutation(
    sdata_blobs,
) -> None:
    table = sdata_blobs.tables["table"]
    obs_before = table.obs.copy(deep=True)
    uns_before = copy.deepcopy(table.uns)
    viewer = _Viewer()
    adapter = ViewerAdapter(viewer)

    first = load_and_style_unannotated_spatial_annotation_labels(
        adapter,
        sdata=sdata_blobs,
        coordinate_system="global",
        labels_name="blobs_labels",
    )
    second = load_and_style_unannotated_spatial_annotation_labels(
        adapter,
        sdata=sdata_blobs,
        coordinate_system="global",
        labels_name="blobs_labels",
    )

    assert first.created is True
    assert second.created is False
    assert second.layer is first.layer
    assert second.value_kind is None
    assert second.palette_source is None
    assert second.coercion_applied is False
    assert viewer.layers == [first.layer]
    assert viewer.layers.selection.active is first.layer
    binding = adapter.layer_bindings.get_binding(first.layer)
    assert isinstance(binding, LabelsLayerBinding)
    assert binding.labels_role == "primary"
    assert binding.style_spec is None

    neutral_rgba = np.asarray(to_rgba(MISSING_CATEGORICAL_COLOR), dtype=np.float32)
    assert np.allclose(first.layer.colormap.map(0), np.zeros(4, dtype=np.float32))
    assert np.allclose(first.layer.colormap.map(1), neutral_rgba)
    assert np.allclose(first.layer.colormap.map(400_000), neutral_rgba)
    pd.testing.assert_frame_equal(table.obs, obs_before)
    assert table.uns == uns_before


def test_spatial_annotation_styling_loads_reuses_and_activates_primary_layer_without_table_mutation(
    sdata_blobs,
) -> None:
    _add_annotation_column(sdata_blobs)
    table = sdata_blobs.tables["table"]
    table.uns["annotation_colors"] = ["#ff0000", "#00ff00"]
    obs_before = table.obs.copy(deep=True)
    uns_before = copy.deepcopy(table.uns)
    viewer = _Viewer()
    adapter = ViewerAdapter(viewer)

    first = load_and_style_spatial_annotation_labels(
        adapter,
        sdata=sdata_blobs,
        coordinate_system="global",
        labels_name="blobs_labels",
        table_name="table",
        column_name="annotation",
    )
    second = load_and_style_spatial_annotation_labels(
        adapter,
        sdata=sdata_blobs,
        coordinate_system="global",
        labels_name="blobs_labels",
        table_name="table",
        column_name="annotation",
    )

    assert first.created is True
    assert second.created is False
    assert second.layer is first.layer
    assert second.value_kind == "categorical"
    assert second.palette_source == "stored"
    assert second.coercion_applied is False
    assert viewer.layers == [first.layer]
    assert viewer.layers.selection.active is first.layer
    binding = adapter.layer_bindings.get_binding(first.layer)
    assert isinstance(binding, LabelsLayerBinding)
    assert binding.labels_role == "primary"
    assert binding.style_spec is None
    odd_instance = int(table.obs.loc[table.obs["annotation"] == "odd", "instance_id"].iloc[0])
    assert np.allclose(first.layer.colormap.map(odd_instance), np.asarray(to_rgba("#ff0000"), dtype=np.float32))
    pd.testing.assert_frame_equal(table.obs, obs_before)
    assert table.uns == uns_before


def test_spatial_annotation_styling_supports_positive_integer_categories_without_table_mutation(
    sdata_blobs,
) -> None:
    table = sdata_blobs.tables["table"]
    instance_ids = table.obs["instance_id"].to_numpy()
    values = np.where(instance_ids % 2 == 0, 3, 1).astype(object)
    values[0] = pd.NA
    table.obs["user_class"] = pd.Categorical(values, categories=[3, 1])
    table.uns["user_class_colors"] = ["#ff0000", "#00ff00"]
    obs_before = table.obs.copy(deep=True)
    uns_before = copy.deepcopy(table.uns)
    viewer = _Viewer()

    result = load_and_style_spatial_annotation_labels(
        ViewerAdapter(viewer),
        sdata=sdata_blobs,
        coordinate_system="global",
        labels_name="blobs_labels",
        table_name="table",
        column_name="user_class",
    )

    class_three_instance = int(table.obs.loc[table.obs["user_class"] == 3, "instance_id"].iloc[0])
    class_one_instance = int(table.obs.loc[table.obs["user_class"] == 1, "instance_id"].iloc[0])
    missing_instance = int(table.obs["instance_id"].iloc[0])
    assert result.palette_source == "stored"
    assert np.allclose(result.layer.colormap.map(class_three_instance), np.asarray(to_rgba("#ff0000")))
    assert np.allclose(result.layer.colormap.map(class_one_instance), np.asarray(to_rgba("#00ff00")))
    assert np.allclose(
        result.layer.colormap.map(missing_instance),
        np.asarray(to_rgba(MISSING_CATEGORICAL_COLOR)),
    )
    pd.testing.assert_frame_equal(table.obs, obs_before)
    assert table.uns == uns_before


@pytest.mark.parametrize(
    "values",
    [
        pd.Series(["odd", "even"], dtype="string"),
        pd.Series(pd.Categorical([0, 0], categories=[0])),
    ],
)
def test_spatial_annotation_styling_rejects_incompatible_column_before_loading(
    sdata_blobs,
    values: pd.Series,
) -> None:
    table = sdata_blobs.tables["table"]
    table.obs["annotation"] = pd.Series(
        np.resize(values.to_numpy(), table.n_obs),
        index=table.obs.index,
        dtype=values.dtype,
    )
    viewer = _Viewer()
    adapter = ViewerAdapter(viewer)

    with pytest.raises(ValueError, match="must be categorical|only string categories or positive integer categories"):
        load_and_style_spatial_annotation_labels(
            adapter,
            sdata=sdata_blobs,
            coordinate_system="global",
            labels_name="blobs_labels",
            table_name="table",
            column_name="annotation",
        )

    assert viewer.layers == []
