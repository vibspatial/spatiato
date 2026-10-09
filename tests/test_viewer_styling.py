from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from matplotlib import colormaps
from matplotlib.colors import to_rgba
from napari.utils.colormaps import DirectLabelColormap
from spatialdata import SpatialData

import spatiato.widgets.object_classification.viewer_styling as viewer_styling_module
from spatiato.core.class_palette import (
    DEFAULT_NEUTRAL_COLOR,
    default_categorical_colors,
    default_class_colors,
    default_labeled_class_color,
)
from spatiato.core.object_classification.annotation import (
    USER_CLASS_COLORS_KEY,
    USER_CLASS_COLUMN,
    UserClassStateChange,
)
from spatiato.viewer.labels_colormap import CompactLabelColormap
from spatiato.widgets.object_classification.annotation_controller import UserClassAnnotationChange
from spatiato.widgets.object_classification.controller import (
    PRED_CLASS_COLORS_KEY,
    PRED_CLASS_COLUMN,
    PRED_CONFIDENCE_COLUMN,
)
from spatiato.widgets.object_classification.viewer_styling import (
    COLOR_BY_PRED_CLASS,
    COLOR_BY_PRED_CONFIDENCE,
    COLOR_BY_USER_CLASS,
    ViewerStylingController,
)


def _user_class_annotation_change(
    instance_id: int,
    class_id: int | None,
) -> UserClassAnnotationChange:
    return UserClassAnnotationChange(
        instance_id=instance_id,
        class_id=class_id,
        state_change=UserClassStateChange(
            user_class_changed=True,
            palette_changed=False,
        ),
    )


class _FakeEventEmitter:
    def __init__(self) -> None:
        self.call_count = 0

    def __call__(self) -> None:
        self.call_count += 1


class _FakeLabelsEvents:
    def __init__(self) -> None:
        self.colormap = _FakeEventEmitter()


class _FakeLabelsLayer:
    def __init__(self) -> None:
        self.colormap: DirectLabelColormap | None = None
        self.features = pd.DataFrame()
        self.refresh_count = 0
        self.refresh_kwargs: list[dict[str, Any]] = []
        self.events = _FakeLabelsEvents()

    def refresh(self, **kwargs: Any) -> None:
        self.refresh_kwargs.append(kwargs)
        self.refresh_count += 1


class _FakeViewerAdapter:
    def __init__(self, layer: _FakeLabelsLayer) -> None:
        self._layer = layer
        self.sync_display_layers: list[_FakeLabelsLayer] = []

    def get_loaded_primary_labels_layer(self, *args: Any, **kwargs: Any) -> _FakeLabelsLayer:
        del args, kwargs
        return self._layer

    def sync_labels_display_after_colormap_change(self, layer: _FakeLabelsLayer) -> None:
        self.sync_display_layers.append(layer)


def _make_controller(sdata: SpatialData, layer: _FakeLabelsLayer) -> ViewerStylingController:
    controller = ViewerStylingController(_FakeViewerAdapter(layer))
    controller.bind(sdata, "blobs_labels", "table")
    return controller


def _make_controller_with_adapter(
    sdata: SpatialData,
    layer: _FakeLabelsLayer,
) -> tuple[ViewerStylingController, _FakeViewerAdapter]:
    adapter = _FakeViewerAdapter(layer)
    controller = ViewerStylingController(adapter)
    controller.bind(sdata, "blobs_labels", "table")
    return controller, adapter


def _feature_rows(
    user_class_by_instance: dict[int, int | None],
    *,
    pred_class_by_instance: dict[int, int | None] | None = None,
    pred_confidence_by_instance: dict[int, float] | None = None,
) -> pd.DataFrame:
    instance_ids = sorted(user_class_by_instance)
    pred_class_by_instance = pred_class_by_instance or {}
    pred_confidence_by_instance = pred_confidence_by_instance or {}
    index = pd.Index(instance_ids, name="index", dtype="int64")
    return pd.DataFrame(
        {
            USER_CLASS_COLUMN: pd.Series(
                [user_class_by_instance[instance_id] for instance_id in instance_ids],
                index=index,
                dtype="Int64",
            ),
            PRED_CLASS_COLUMN: pd.Series(
                [pred_class_by_instance.get(instance_id, pd.NA) for instance_id in instance_ids],
                index=index,
                dtype="Int64",
            ),
            PRED_CONFIDENCE_COLUMN: pd.Series(
                [pred_confidence_by_instance.get(instance_id, np.nan) for instance_id in instance_ids],
                index=index,
                dtype="float64",
            ),
        },
        index=index,
    )


def _set_user_classes(
    sdata: SpatialData,
    class_by_instance: dict[int, int],
    *,
    categories: list[int],
    colors: list[str] | None = None,
) -> None:
    table = sdata["table"]
    values = [
        class_by_instance.get(int(instance_id), pd.NA) if str(region) == "blobs_labels" else pd.NA
        for region, instance_id in zip(table.obs["region"], table.obs["instance_id"], strict=True)
    ]
    table.obs[USER_CLASS_COLUMN] = pd.Series(
        pd.Categorical(values, categories=categories),
        index=table.obs.index,
        name=USER_CLASS_COLUMN,
    )
    table.uns[USER_CLASS_COLORS_KEY] = colors or default_class_colors(categories)


def _set_pred_classes(
    sdata: SpatialData,
    class_by_instance: dict[int, int],
    *,
    categories: list[int],
    colors: list[str] | None = None,
) -> None:
    table = sdata["table"]
    values = [
        class_by_instance.get(int(instance_id), pd.NA) if str(region) == "blobs_labels" else pd.NA
        for region, instance_id in zip(table.obs["region"], table.obs["instance_id"], strict=True)
    ]
    table.obs[PRED_CLASS_COLUMN] = pd.Series(
        pd.Categorical(values, categories=categories),
        index=table.obs.index,
        name=PRED_CLASS_COLUMN,
    )
    table.uns[PRED_CLASS_COLORS_KEY] = colors or default_class_colors(categories)


def _expected_rgba(color: str) -> np.ndarray:
    return np.asarray(to_rgba(color), dtype=np.float32)


def _expected_class_rgba(class_id: int) -> np.ndarray:
    return _expected_rgba(default_class_colors([class_id])[0])


def test_user_class_annotation_change_rejects_negative_class_id() -> None:
    with pytest.raises(ValueError, match="positive"):
        _user_class_annotation_change(instance_id=5, class_id=-1)


def test_refresh_reuses_one_region_feature_snapshot(monkeypatch: pytest.MonkeyPatch, sdata_blobs: SpatialData) -> None:
    _set_user_classes(sdata_blobs, {5: 4}, categories=[4])
    layer = _FakeLabelsLayer()
    controller = _make_controller(sdata_blobs, layer)
    feature_rows = _feature_rows({1: None, 5: 4, 6: None})
    calls = 0

    def fake_feature_rows() -> pd.DataFrame:
        nonlocal calls
        calls += 1
        return feature_rows.copy()

    monkeypatch.setattr(controller, "_get_region_feature_rows", fake_feature_rows)

    controller.refresh()

    assert calls == 1
    assert isinstance(layer.colormap, CompactLabelColormap)
    assert len(layer.colormap.color_dict) <= 3
    np.testing.assert_allclose(layer.colormap.map(0), np.zeros(4, dtype=np.float32))
    np.testing.assert_allclose(layer.colormap.map(5), _expected_class_rgba(4))
    np.testing.assert_allclose(layer.colormap.map(6), _expected_rgba(DEFAULT_NEUTRAL_COLOR))
    np.testing.assert_allclose(layer.colormap.map(99), np.zeros(4, dtype=np.float32))
    assert layer.refresh_count == 0
    assert USER_CLASS_COLUMN in layer.features.columns


def test_user_class_color_lookup_uses_valid_categorical_without_full_normalization(
    monkeypatch: pytest.MonkeyPatch,
    sdata_blobs: SpatialData,
) -> None:
    _set_user_classes(sdata_blobs, {5: 4}, categories=[4])
    layer = _FakeLabelsLayer()
    controller = _make_controller(sdata_blobs, layer)
    feature_rows = _feature_rows({1: None, 5: 4, 6: None})

    def fail_normalization(*args: Any, **kwargs: Any) -> pd.Series:
        del args, kwargs
        raise AssertionError("valid categorical user_class should not be normalized")

    monkeypatch.setattr(viewer_styling_module, "normalize_class_values", fail_normalization)

    controller.refresh_layer_colors(feature_rows=feature_rows)

    assert isinstance(layer.colormap, CompactLabelColormap)
    np.testing.assert_allclose(layer.colormap.map(5), _expected_class_rgba(4))
    np.testing.assert_allclose(layer.colormap.map(6), _expected_rgba(DEFAULT_NEUTRAL_COLOR))


def test_user_class_coloring_uses_missing_palette_fallback_without_table_mutation(
    sdata_blobs: SpatialData,
) -> None:
    _set_user_classes(sdata_blobs, {5: 4}, categories=[4, 7])
    table = sdata_blobs["table"]
    table.uns.pop(USER_CLASS_COLORS_KEY)
    layer = _FakeLabelsLayer()
    controller = _make_controller(sdata_blobs, layer)

    controller.refresh_layer_colors(feature_rows=_feature_rows({5: 4}))

    assert USER_CLASS_COLORS_KEY not in table.uns
    np.testing.assert_allclose(layer.colormap.map(5), _expected_rgba(default_categorical_colors(2)[0]))


def test_all_unlabeled_user_class_coloring_force_syncs_after_full_repaint(sdata_blobs: SpatialData) -> None:
    _set_user_classes(sdata_blobs, {}, categories=[])
    layer = _FakeLabelsLayer()
    controller, adapter = _make_controller_with_adapter(sdata_blobs, layer)
    feature_rows = _feature_rows({1: None, 5: None, 6: None})

    controller.refresh_layer_colors(feature_rows=feature_rows)

    assert adapter.sync_display_layers == [layer]
    assert isinstance(layer.colormap, CompactLabelColormap)
    np.testing.assert_allclose(layer.colormap.map(0), np.zeros(4, dtype=np.float32))
    np.testing.assert_allclose(layer.colormap.map(1), _expected_rgba(DEFAULT_NEUTRAL_COLOR))


def test_absent_class_and_prediction_sources_remain_read_only_and_neutral(
    sdata_blobs: SpatialData,
) -> None:
    table = sdata_blobs["table"]
    layer = _FakeLabelsLayer()
    controller = _make_controller(sdata_blobs, layer)

    controller.refresh()
    controller.set_color_by(COLOR_BY_PRED_CLASS)
    controller.set_color_by(COLOR_BY_PRED_CONFIDENCE)

    assert USER_CLASS_COLUMN not in table.obs
    assert PRED_CLASS_COLUMN not in table.obs
    assert PRED_CONFIDENCE_COLUMN not in table.obs
    assert PRED_CLASS_COLORS_KEY not in table.uns
    assert isinstance(layer.colormap, CompactLabelColormap)


def test_pred_class_color_lookup_uses_valid_categorical_without_full_normalization(
    monkeypatch: pytest.MonkeyPatch,
    sdata_blobs: SpatialData,
) -> None:
    _set_pred_classes(sdata_blobs, {5: 2, 6: 2}, categories=[2])
    layer = _FakeLabelsLayer()
    controller = _make_controller(sdata_blobs, layer)
    controller.set_color_by(COLOR_BY_PRED_CLASS)
    feature_rows = _feature_rows(
        {1: None, 5: None, 6: None},
        pred_class_by_instance={1: None, 5: 2, 6: 2},
    )

    def fail_normalization(*args: Any, **kwargs: Any) -> pd.Series:
        del args, kwargs
        raise AssertionError("valid categorical pred_class should not be normalized")

    monkeypatch.setattr(viewer_styling_module, "normalize_class_values", fail_normalization)

    controller.refresh_layer_colors(feature_rows=feature_rows)

    assert isinstance(layer.colormap, CompactLabelColormap)
    np.testing.assert_allclose(layer.colormap.map(1), _expected_rgba(DEFAULT_NEUTRAL_COLOR))
    np.testing.assert_allclose(layer.colormap.map(5), _expected_class_rgba(2))


def test_pred_class_coloring_derives_shared_colors_from_user_palette(
    sdata_blobs: SpatialData,
) -> None:
    _set_user_classes(sdata_blobs, {5: 4}, categories=[4], colors=["#123456"])
    _set_pred_classes(
        sdata_blobs,
        {5: 4, 6: 7},
        categories=[4, 7],
        colors=["#000000", "#000000"],
    )
    table = sdata_blobs["table"]
    previous_pred_palette = list(table.uns[PRED_CLASS_COLORS_KEY])
    layer = _FakeLabelsLayer()
    controller = _make_controller(sdata_blobs, layer)
    controller.set_color_by(COLOR_BY_PRED_CLASS)

    controller.refresh_layer_colors(
        feature_rows=_feature_rows(
            {5: 4, 6: None},
            pred_class_by_instance={5: 4, 6: 7},
        )
    )

    np.testing.assert_allclose(layer.colormap.map(5), _expected_rgba("#123456"))
    np.testing.assert_allclose(layer.colormap.map(6), _expected_rgba(default_labeled_class_color(7)))
    assert table.uns[PRED_CLASS_COLORS_KEY] == previous_pred_palette


def test_class_value_reader_uses_vectorized_integer_fast_path(monkeypatch: pytest.MonkeyPatch) -> None:
    values = pd.Series([pd.NA, 4, 4, 2], dtype="Int64")

    def fail_scalar_missing_check(*args: Any, **kwargs: Any) -> bool:
        del args, kwargs
        raise AssertionError("integer class ids should use the vectorized path")

    monkeypatch.setattr(viewer_styling_module.pd, "isna", fail_scalar_missing_check)

    assert viewer_styling_module._read_class_values_without_normalizing(values) == {2, 4}


def test_user_class_color_lookup_rejects_invalid_table_state(sdata_blobs: SpatialData) -> None:
    table = sdata_blobs["table"]
    table.obs[USER_CLASS_COLUMN] = pd.Series(1, index=table.obs.index, name=USER_CLASS_COLUMN, dtype="int64")
    table.uns[USER_CLASS_COLORS_KEY] = default_class_colors([1])
    layer = _FakeLabelsLayer()
    controller = _make_controller(sdata_blobs, layer)
    feature_rows = _feature_rows({1: None, 5: 4})

    with pytest.raises(viewer_styling_module.ClassStateError, match="categorical dtype"):
        controller.refresh_layer_colors(feature_rows=feature_rows)

    assert layer.colormap is None


def test_refresh_layer_methods_still_work_without_precomputed_feature_rows(
    monkeypatch: pytest.MonkeyPatch,
    sdata_blobs: SpatialData,
) -> None:
    _set_user_classes(sdata_blobs, {5: 4}, categories=[4])
    layer = _FakeLabelsLayer()
    controller = _make_controller(sdata_blobs, layer)
    feature_rows = _feature_rows({1: None, 5: 4})
    calls = 0

    def fake_feature_rows() -> pd.DataFrame:
        nonlocal calls
        calls += 1
        return feature_rows.copy()

    monkeypatch.setattr(controller, "_get_region_feature_rows", fake_feature_rows)

    controller.refresh_layer_colors()
    controller.refresh_layer_features()

    assert calls == 2
    assert isinstance(layer.colormap, CompactLabelColormap)
    assert USER_CLASS_COLUMN in layer.features.columns


def test_pred_class_coloring_uses_neutral_fallback_for_missing_predictions(sdata_blobs: SpatialData) -> None:
    _set_user_classes(sdata_blobs, {}, categories=[])
    _set_pred_classes(sdata_blobs, {5: 2, 6: 2}, categories=[2])
    layer = _FakeLabelsLayer()
    controller = _make_controller(sdata_blobs, layer)
    controller.set_color_by(COLOR_BY_PRED_CLASS)
    feature_rows = _feature_rows(
        {1: None, 5: None, 6: None},
        pred_class_by_instance={1: None, 5: 2, 6: 2},
    )

    controller.refresh_layer_colors(feature_rows=feature_rows)

    assert isinstance(layer.colormap, CompactLabelColormap)
    assert len(layer.colormap.color_dict) <= 3
    np.testing.assert_allclose(layer.colormap.map(0), np.zeros(4, dtype=np.float32))
    np.testing.assert_allclose(layer.colormap.map(1), _expected_rgba(DEFAULT_NEUTRAL_COLOR))
    np.testing.assert_allclose(layer.colormap.map(5), layer.colormap.map(6))
    np.testing.assert_allclose(layer.colormap.map(99), np.zeros(4, dtype=np.float32))
    assert layer.refresh_count == 0
    assert not np.allclose(layer.colormap.map(1), layer.colormap.map(0))


def test_pred_confidence_coloring_uses_compact_continuous_colormap_without_explicit_refresh(
    sdata_blobs: SpatialData,
) -> None:
    _set_user_classes(sdata_blobs, {}, categories=[])
    layer = _FakeLabelsLayer()
    controller, adapter = _make_controller_with_adapter(sdata_blobs, layer)
    controller.set_color_by(COLOR_BY_PRED_CONFIDENCE)
    feature_rows = _feature_rows(
        {1: None, 5: None, 6: None},
        pred_confidence_by_instance={1: 0.0, 5: 1.0},
    )

    controller.refresh_layer_colors(feature_rows=feature_rows)

    assert isinstance(layer.colormap, CompactLabelColormap)
    expected_colors = np.asarray(
        colormaps[viewer_styling_module.PRED_CONFIDENCE_COLORMAP](np.asarray([0.0, 1.0])),
        dtype=np.float32,
    )
    np.testing.assert_allclose(layer.colormap.map(1), expected_colors[0])
    np.testing.assert_allclose(layer.colormap.map(5), expected_colors[1])
    np.testing.assert_allclose(layer.colormap.map(6), _expected_rgba(viewer_styling_module.MISSING_CONTINUOUS_COLOR))
    np.testing.assert_allclose(layer.colormap.map(0), np.zeros(4, dtype=np.float32))
    np.testing.assert_allclose(layer.colormap.map(99), np.zeros(4, dtype=np.float32))
    assert layer.refresh_count == 0
    assert adapter.sync_display_layers == [layer]


@pytest.mark.parametrize("color_by", [COLOR_BY_USER_CLASS, COLOR_BY_PRED_CLASS, COLOR_BY_PRED_CONFIDENCE])
def test_coloring_without_bound_table_keeps_all_labels_neutral(sdata_blobs: SpatialData, color_by: str) -> None:
    layer = _FakeLabelsLayer()
    controller = ViewerStylingController(_FakeViewerAdapter(layer))
    controller.bind(sdata_blobs, "blobs_labels", None)
    controller.set_color_by(color_by)

    controller.refresh()

    assert isinstance(layer.colormap, CompactLabelColormap)
    np.testing.assert_allclose(layer.colormap.map(0), np.zeros(4, dtype=np.float32))
    np.testing.assert_allclose(layer.colormap.map(1), _expected_rgba(DEFAULT_NEUTRAL_COLOR))
    np.testing.assert_allclose(layer.colormap.map(99), _expected_rgba(DEFAULT_NEUTRAL_COLOR))


def test_row_scoped_user_class_annotation_inserts_compact_label_and_refreshes_layer(
    monkeypatch: pytest.MonkeyPatch,
    sdata_blobs: SpatialData,
) -> None:
    _set_user_classes(sdata_blobs, {5: 4}, categories=[4])
    layer = _FakeLabelsLayer()
    controller, adapter = _make_controller_with_adapter(sdata_blobs, layer)
    feature_rows = _feature_rows({1: None, 5: 4, 6: None})
    controller.refresh_layer_colors(feature_rows=feature_rows)
    controller.refresh_layer_features(feature_rows=feature_rows)
    original_colormap = layer.colormap
    adapter.sync_display_layers.clear()

    def fail_full_feature_rows() -> pd.DataFrame:
        raise AssertionError("compact sparse annotation refresh must not rebuild all feature rows")

    monkeypatch.setattr(controller, "_get_region_feature_rows", fail_full_feature_rows)

    handled = controller.refresh_user_class_colormap_and_feature(
        _user_class_annotation_change(instance_id=6, class_id=4)
    )

    assert handled is True
    assert layer.colormap is original_colormap
    assert layer.refresh_count == 1
    assert layer.refresh_kwargs == [{"extent": False}]
    assert layer.events.colormap.call_count == 0
    assert adapter.sync_display_layers == []
    assert layer.features.set_index("index").loc[6, USER_CLASS_COLUMN] == 4
    assert isinstance(layer.colormap, CompactLabelColormap)
    np.testing.assert_allclose(layer.colormap.map(6), _expected_class_rgba(4))
    mapping = layer.colormap._compact_mapping
    assert 6 in mapping.label_ids
    assert bool(np.all(mapping.label_ids[1:] > mapping.label_ids[:-1]))


def test_row_scoped_user_class_annotation_clear_uses_missing_color_and_refreshes_layer(
    monkeypatch: pytest.MonkeyPatch,
    sdata_blobs: SpatialData,
) -> None:
    _set_user_classes(sdata_blobs, {5: 4}, categories=[4])
    layer = _FakeLabelsLayer()
    controller = _make_controller(sdata_blobs, layer)
    feature_rows = _feature_rows({1: None, 5: 4, 6: None})
    controller.refresh_layer_colors(feature_rows=feature_rows)
    controller.refresh_layer_features(feature_rows=feature_rows)
    original_colormap = layer.colormap

    def fail_full_feature_rows() -> pd.DataFrame:
        raise AssertionError("compact sparse annotation refresh must not rebuild all feature rows")

    monkeypatch.setattr(controller, "_get_region_feature_rows", fail_full_feature_rows)

    handled = controller.refresh_user_class_colormap_and_feature(
        _user_class_annotation_change(instance_id=5, class_id=None)
    )

    assert handled is True
    assert layer.colormap is original_colormap
    assert layer.refresh_count == 1
    assert layer.refresh_kwargs == [{"extent": False}]
    assert layer.events.colormap.call_count == 0
    assert pd.isna(layer.features.set_index("index").loc[5, USER_CLASS_COLUMN])
    assert isinstance(layer.colormap, CompactLabelColormap)
    mapping = layer.colormap._compact_mapping
    position = int(np.searchsorted(mapping.label_ids, 5))
    assert int(mapping.label_ids[position]) == 5
    assert int(mapping.texture_codes[position]) == mapping.missing_texture_code
    np.testing.assert_allclose(layer.colormap.map(5), _expected_rgba(DEFAULT_NEUTRAL_COLOR))
    np.testing.assert_allclose(layer.colormap.map(99), np.zeros(4, dtype=np.float32))


def test_row_scoped_user_class_annotation_clear_uses_reserved_missing_color_when_all_rows_were_labeled(
    sdata_blobs: SpatialData,
) -> None:
    _set_user_classes(sdata_blobs, {5: 4, 6: 4}, categories=[4])
    layer = _FakeLabelsLayer()
    controller = _make_controller(sdata_blobs, layer)
    feature_rows = _feature_rows({5: 4, 6: 4})
    controller.refresh_layer_colors(feature_rows=feature_rows)
    controller.refresh_layer_features(feature_rows=feature_rows)
    original_colormap = layer.colormap
    assert isinstance(original_colormap, CompactLabelColormap)
    original_texture_count = len(original_colormap._compact_mapping.texture_rgba)
    assert original_colormap._compact_mapping.missing_texture_code is not None

    handled = controller.refresh_user_class_colormap_and_feature(
        _user_class_annotation_change(instance_id=5, class_id=None)
    )

    assert handled is True
    assert layer.colormap is original_colormap
    assert layer.events.colormap.call_count == 0
    assert layer.refresh_kwargs == [{"extent": False}]
    assert len(layer.colormap._compact_mapping.texture_rgba) == original_texture_count
    np.testing.assert_allclose(layer.colormap.map(5), _expected_rgba(DEFAULT_NEUTRAL_COLOR))
    np.testing.assert_allclose(layer.colormap.map(6), _expected_class_rgba(4))
    np.testing.assert_allclose(layer.colormap.map(99), np.zeros(4, dtype=np.float32))


def test_row_scoped_user_class_annotation_updates_existing_compact_label(
    sdata_blobs: SpatialData,
) -> None:
    colors = default_class_colors([4, 7])
    _set_user_classes(sdata_blobs, {5: 4}, categories=[4, 7], colors=colors)
    layer = _FakeLabelsLayer()
    controller = _make_controller(sdata_blobs, layer)
    feature_rows = _feature_rows({5: 4})
    controller.refresh_layer_colors(feature_rows=feature_rows)
    controller.refresh_layer_features(feature_rows=feature_rows)
    original_colormap = layer.colormap
    _set_user_classes(sdata_blobs, {5: 7}, categories=[4, 7], colors=colors)

    handled = controller.refresh_user_class_colormap_and_feature(
        _user_class_annotation_change(instance_id=5, class_id=7)
    )

    assert handled is True
    assert layer.colormap is original_colormap
    assert layer.refresh_kwargs == [{"extent": False}]
    assert layer.events.colormap.call_count == 0
    assert isinstance(layer.colormap, CompactLabelColormap)
    assert 5 in layer.colormap._compact_mapping.label_ids
    np.testing.assert_allclose(layer.colormap.map(5), _expected_class_rgba(7))


def test_row_scoped_user_class_annotation_appends_new_class_texture(
    sdata_blobs: SpatialData,
) -> None:
    initial_colors = default_class_colors([4])
    updated_colors = default_class_colors([4, 9])
    _set_user_classes(sdata_blobs, {5: 4}, categories=[4], colors=initial_colors)
    layer = _FakeLabelsLayer()
    controller = _make_controller(sdata_blobs, layer)
    feature_rows = _feature_rows({5: 4, 6: None})
    controller.refresh_layer_colors(feature_rows=feature_rows)
    controller.refresh_layer_features(feature_rows=feature_rows)
    assert isinstance(layer.colormap, CompactLabelColormap)
    original_texture_count = len(layer.colormap._compact_mapping.texture_rgba)
    _set_user_classes(sdata_blobs, {5: 4, 6: 9}, categories=[4, 9], colors=updated_colors)

    handled = controller.refresh_user_class_colormap_and_feature(
        _user_class_annotation_change(instance_id=6, class_id=9)
    )

    assert handled is True
    assert layer.refresh_kwargs == [{"extent": False}]
    assert layer.events.colormap.call_count == 1
    assert isinstance(layer.colormap, CompactLabelColormap)
    mapping = layer.colormap._compact_mapping
    assert len(mapping.texture_rgba) == original_texture_count + 1
    assert mapping.value_texture_codes is not None
    assert mapping.value_texture_codes[9] == len(mapping.texture_rgba) - 1
    assert 6 in mapping.label_ids
    np.testing.assert_allclose(layer.colormap.map(6), _expected_class_rgba(9))


def test_row_scoped_user_class_annotation_requires_compact_colormap(
    sdata_blobs: SpatialData,
) -> None:
    _set_user_classes(sdata_blobs, {5: 4}, categories=[4])
    layer = _FakeLabelsLayer()
    controller = _make_controller(sdata_blobs, layer)
    feature_rows = _feature_rows({5: 4})
    controller.refresh_layer_features(feature_rows=feature_rows)
    layer.colormap = DirectLabelColormap(
        color_dict={
            None: np.zeros(4, dtype=np.float32),
            0: np.zeros(4, dtype=np.float32),
            5: _expected_rgba("#ff0000"),
        },
        background_value=0,
    )

    with pytest.raises(RuntimeError, match="CompactLabelColormap"):
        controller.refresh_user_class_colormap_and_feature(
            _user_class_annotation_change(instance_id=5, class_id=4)
        )

    assert layer.refresh_count == 0
    assert layer.events.colormap.call_count == 0
    assert isinstance(layer.colormap, DirectLabelColormap)


def test_row_scoped_user_class_feature_only_refresh_keeps_prediction_colormap(
    monkeypatch: pytest.MonkeyPatch,
    sdata_blobs: SpatialData,
) -> None:
    _set_user_classes(sdata_blobs, {5: 4}, categories=[4])
    _set_pred_classes(sdata_blobs, {5: 2, 6: 2}, categories=[2])
    layer = _FakeLabelsLayer()
    controller = _make_controller(sdata_blobs, layer)
    controller.set_color_by(COLOR_BY_PRED_CLASS)
    feature_rows = _feature_rows(
        {1: None, 5: 4, 6: None},
        pred_class_by_instance={1: None, 5: 2, 6: 2},
    )
    controller.refresh_layer_colors(feature_rows=feature_rows)
    controller.refresh_layer_features(feature_rows=feature_rows)
    original_colormap = layer.colormap
    original_refresh_count = layer.refresh_count

    def fail_full_feature_rows() -> pd.DataFrame:
        raise AssertionError("feature-only annotation refresh must not rebuild all feature rows")

    monkeypatch.setattr(controller, "_get_region_feature_rows", fail_full_feature_rows)

    handled = controller.refresh_user_class_feature_only(
        _user_class_annotation_change(instance_id=6, class_id=4)
    )

    assert handled is True
    assert layer.colormap is original_colormap
    assert layer.refresh_count == original_refresh_count
    assert layer.features.set_index("index").loc[6, USER_CLASS_COLUMN] == 4
    assert pd.isna(layer.features.set_index("index").loc[1, USER_CLASS_COLUMN])


def test_row_scoped_user_class_annotation_returns_false_for_missing_feature_row(sdata_blobs: SpatialData) -> None:
    _set_user_classes(sdata_blobs, {5: 4}, categories=[4])
    layer = _FakeLabelsLayer()
    controller = _make_controller(sdata_blobs, layer)
    feature_rows = _feature_rows({1: None, 5: 4, 6: None})
    controller.refresh_layer_colors(feature_rows=feature_rows)
    controller.refresh_layer_features(feature_rows=feature_rows)
    original_color_keys = set(layer.colormap.color_dict)
    original_features = layer.features.copy()

    handled = controller.refresh_user_class_colormap_and_feature(
        _user_class_annotation_change(instance_id=99, class_id=4)
    )

    assert handled is False
    assert set(layer.colormap.color_dict) == original_color_keys
    pd.testing.assert_frame_equal(layer.features, original_features)
