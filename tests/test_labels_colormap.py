from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_rgba
from napari.layers import Labels
from napari.utils.colormap_backend import get_backend, set_backend
from napari.utils.colormaps import DirectLabelColormap

import spatiato.viewer.labels_colormap as labels_colormap_module
from spatiato.core.class_palette import DEFAULT_NEUTRAL_COLOR
from spatiato.viewer._styling import (
    MISSING_CATEGORICAL_COLOR,
    MISSING_CONTINUOUS_COLOR,
    continuous_rgba_for_values,
)
from spatiato.viewer.labels_colormap import (
    CompactLabelColormap,
    CompactLabelsMapping,
    compact_categorical_label_colormap_from_values,
    compact_categorical_labels_mapping_from_values,
    compact_continuous_labels_mapping_from_values,
    direct_label_colormap_from_rgba,
)


def _rgba(red: float, green: float, blue: float, alpha: float) -> np.ndarray:
    return np.asarray([red, green, blue, alpha], dtype=np.float64)


def test_missing_states_share_the_neutral_color() -> None:
    assert DEFAULT_NEUTRAL_COLOR == "#DCE8F2CC"
    assert MISSING_CATEGORICAL_COLOR == DEFAULT_NEUTRAL_COLOR
    assert MISSING_CONTINUOUS_COLOR == DEFAULT_NEUTRAL_COLOR


@pytest.fixture
def restore_colormap_backend() -> Iterator[None]:
    previous_backend = get_backend()
    try:
        yield
    finally:
        set_backend(previous_backend)


def _expanded_color_dict(mapping: CompactLabelsMapping) -> dict[int | None, np.ndarray]:
    color_dict: dict[int | None, np.ndarray] = {
        None: mapping.texture_rgba[mapping.default_texture_code],
        mapping.background_value: mapping.texture_rgba[mapping.background_texture_code],
    }
    for label_id, texture_code in zip(mapping.label_ids, mapping.texture_codes, strict=True):
        color_dict[int(label_id)] = mapping.texture_rgba[int(texture_code)]
    return color_dict


def _mapped_rgba(mapping: CompactLabelsMapping, label_ids: pd.Index | list[int]) -> np.ndarray:
    texture_code_by_label = dict(zip(mapping.label_ids.tolist(), mapping.texture_codes.tolist(), strict=True))
    return np.asarray(
        [mapping.texture_rgba[int(texture_code_by_label[int(label_id)])] for label_id in label_ids],
        dtype=np.float64,
    )


def test_compact_categorical_labels_mapping_preserves_current_missing_semantics() -> None:
    values = pd.Series(
        pd.Categorical(["T", None, "unknown", "B"], categories=["T", "B", "unknown"]),
        index=pd.Index([9, 3, 7, 5], name="index"),
    )

    mapping = compact_categorical_labels_mapping_from_values(
        values,
        categories=["T", "B"],
        palette=["#ff0000", "#00ff00"],
    )

    np.testing.assert_array_equal(mapping.label_ids, np.asarray([3, 5, 7, 9], dtype=np.int64))
    assert mapping.default_texture_code == 0
    assert mapping.background_texture_code == 1
    np.testing.assert_allclose(mapping.texture_rgba[0], np.zeros(4, dtype=np.float32))
    np.testing.assert_allclose(mapping.texture_rgba[1], np.zeros(4, dtype=np.float32))

    code_by_label = dict(zip(mapping.label_ids.tolist(), mapping.texture_codes.tolist(), strict=True))
    assert code_by_label[9] != mapping.default_texture_code
    assert code_by_label[5] != mapping.default_texture_code
    assert mapping.missing_texture_code is not None
    assert code_by_label[3] == mapping.missing_texture_code
    assert code_by_label[7] == mapping.missing_texture_code
    np.testing.assert_allclose(mapping.texture_rgba[mapping.missing_texture_code], to_rgba(MISSING_CATEGORICAL_COLOR))


def test_compact_continuous_labels_mapping_matches_current_rgba_for_representative_values() -> None:
    values = pd.Series(
        [1.0, 0.0, np.nan, 0.5],
        index=pd.Index([30, 10, 40, 20], name="index"),
    )

    mapping = compact_continuous_labels_mapping_from_values(values)

    np.testing.assert_array_equal(mapping.label_ids, np.asarray([10, 20, 30, 40], dtype=np.int64))
    assert len(mapping.texture_rgba) == 259  # default, background, 256 bins, missing
    assert mapping.default_texture_code == 0
    assert mapping.background_texture_code == 1
    assert mapping.missing_texture_code == 258
    np.testing.assert_allclose(mapping.texture_rgba[mapping.background_texture_code], np.zeros(4, dtype=np.float32))
    np.testing.assert_allclose(_mapped_rgba(mapping, values.index), continuous_rgba_for_values(values), atol=1e-6)


def test_compact_continuous_labels_mapping_reuses_bins_for_repeated_values() -> None:
    values = pd.Series([0.25, 0.25, 1.0], index=pd.Index([5, 6, 7], name="index"))

    mapping = compact_continuous_labels_mapping_from_values(values)
    code_by_label = dict(zip(mapping.label_ids.tolist(), mapping.texture_codes.tolist(), strict=True))

    assert code_by_label[5] == code_by_label[6]
    assert code_by_label[7] != code_by_label[5]


def test_compact_continuous_labels_mapping_maps_all_missing_values_to_missing_color() -> None:
    values = pd.Series(
        [np.nan, pd.NA, None, np.inf, -np.inf],
        index=pd.Index([11, 12, 13, 14, 15], name="index"),
        dtype="object",
    )

    mapping = compact_continuous_labels_mapping_from_values(values)

    assert mapping.missing_texture_code is not None
    assert set(mapping.texture_codes.tolist()) == {mapping.missing_texture_code}
    expected = np.repeat(
        np.asarray(to_rgba(MISSING_CONTINUOUS_COLOR), dtype=np.float64).reshape(1, 4),
        len(values),
        axis=0,
    )
    np.testing.assert_allclose(_mapped_rgba(mapping, values.index), expected)


def test_compact_continuous_labels_mapping_matches_current_rgba_for_constant_values() -> None:
    values = pd.Series([2.0, 2.0, np.nan], index=pd.Index([10, 11, 12], name="index"))

    mapping = compact_continuous_labels_mapping_from_values(values)

    np.testing.assert_allclose(_mapped_rgba(mapping, values.index), continuous_rgba_for_values(values), atol=1e-6)


def test_compact_continuous_labels_mapping_uses_explicit_range_and_clamps_values() -> None:
    values = pd.Series(
        [-1.0, 0.0, 0.5, 1.0, 2.0, np.nan],
        index=pd.Index([5, 6, 7, 8, 9, 10], name="index"),
    )

    mapping = compact_continuous_labels_mapping_from_values(values, value_range=(0.0, 1.0))
    code_by_label = dict(zip(mapping.label_ids.tolist(), mapping.texture_codes.tolist(), strict=True))

    assert code_by_label[5] == 2
    assert code_by_label[6] == 2
    assert code_by_label[7] == 130
    assert code_by_label[8] == 257
    assert code_by_label[9] == 257
    assert code_by_label[10] == mapping.missing_texture_code


def test_compact_continuous_labels_mapping_accepts_large_label_ids() -> None:
    large_label_id = 2**40
    values = pd.Series([1.0, 0.0], index=pd.Index([large_label_id, 5], name="index"))

    mapping = compact_continuous_labels_mapping_from_values(values)
    code_by_label = dict(zip(mapping.label_ids.tolist(), mapping.texture_codes.tolist(), strict=True))

    np.testing.assert_array_equal(mapping.label_ids, np.asarray([5, large_label_id], dtype=np.int64))
    assert code_by_label[5] == 2
    assert code_by_label[large_label_id] == 257


def test_compact_continuous_labels_mapping_skips_unique_check_for_sorted_label_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = pd.Series([0.0, 0.5, 1.0], index=pd.Index([5, 6, 7], name="index"))

    def fail_unique(_values: object) -> object:
        raise AssertionError("strictly increasing label ids should not call np.unique")

    monkeypatch.setattr(labels_colormap_module.np, "unique", fail_unique)

    mapping = compact_continuous_labels_mapping_from_values(values)

    np.testing.assert_array_equal(mapping.label_ids, np.asarray([5, 6, 7], dtype=np.int64))


@pytest.mark.parametrize(
    ("values", "match"),
    [
        (pd.Series([1.0], index=pd.Index([0], name="index")), "positive label ids"),
        (pd.Series([1.0, 2.0], index=pd.Index([1, 1], name="index")), "unique label ids"),
        (pd.Series([1.0], index=pd.Index(["cell-1"], name="index")), "integer label ids"),
    ],
)
def test_compact_continuous_labels_mapping_rejects_invalid_label_ids(values: pd.Series, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        compact_continuous_labels_mapping_from_values(values)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"bins": 1}, "bins"),
        ({"value_range": (1.0, 1.0)}, "min < max"),
        ({"value_range": (0.0, np.inf)}, "finite"),
    ],
)
def test_compact_continuous_labels_mapping_rejects_invalid_binning_contract(
    kwargs: dict[str, object],
    match: str,
) -> None:
    values = pd.Series([1.0], index=pd.Index([5], name="index"))

    with pytest.raises(ValueError, match=match):
        compact_continuous_labels_mapping_from_values(values, **kwargs)


def test_compact_categorical_labels_mapping_accepts_nontransparent_default_color() -> None:
    values = pd.Series(["a"], index=pd.Index([5], name="index"), dtype="object")

    mapping = compact_categorical_labels_mapping_from_values(
        values,
        categories=["a"],
        palette=["#ff0000"],
        default_color=DEFAULT_NEUTRAL_COLOR,
    )

    np.testing.assert_allclose(mapping.texture_rgba[mapping.default_texture_code], to_rgba(DEFAULT_NEUTRAL_COLOR))
    np.testing.assert_allclose(mapping.texture_rgba[mapping.background_texture_code], np.zeros(4, dtype=np.float32))


def test_compact_categorical_labels_mapping_reuses_repeated_rgba_texture_codes() -> None:
    values = pd.Series(["a", "b", "c"], index=pd.Index([1, 2, 3], name="index"), dtype="object")

    mapping = compact_categorical_labels_mapping_from_values(
        values,
        categories=["a", "b", "c"],
        palette=["#ff0000", "#00ff00", "#ff0000"],
    )

    code_by_label = dict(zip(mapping.label_ids.tolist(), mapping.texture_codes.tolist(), strict=True))
    assert code_by_label[1] == code_by_label[3]
    assert code_by_label[2] != code_by_label[1]
    assert len(mapping.texture_rgba) == 5  # default, background, red, green, reserved missing
    assert set(mapping.texture_codes.tolist()) == {2, 3}


def test_compact_categorical_labels_mapping_accepts_unused_categories() -> None:
    """Unused categories have RGBA rows but are not used by texture codes."""
    values = pd.Series(
        pd.Categorical(["T", "B", "T"], categories=["T", "B", "unused"]),
        index=pd.Index([10, 20, 30], name="index"),
    )

    mapping = compact_categorical_labels_mapping_from_values(
        values,
        categories=["T", "B", "unused"],
        palette=["#ff0000", "#00ff00", "#0000ff"],
    )

    code_by_label = dict(zip(mapping.label_ids.tolist(), mapping.texture_codes.tolist(), strict=True))
    assert code_by_label[10] == code_by_label[30]
    assert code_by_label[20] != code_by_label[10]
    assert set(mapping.texture_codes.tolist()) == {2, 3}


def test_compact_categorical_labels_mapping_reserves_missing_texture_without_missing_values() -> None:
    values = pd.Series(["a", "b"], index=pd.Index([1, 2], name="index"), dtype="object")

    mapping = compact_categorical_labels_mapping_from_values(
        values,
        categories=["a", "b"],
        palette=["#ff0000", "#00ff00"],
        missing_color="#0000ff",
    )

    assert mapping.missing_texture_code == 4
    assert mapping.missing_texture_code not in mapping.texture_codes.tolist()
    np.testing.assert_allclose(mapping.texture_rgba[mapping.missing_texture_code], to_rgba("#0000ff"))


def test_compact_categorical_labels_mapping_uses_array_backed_texture_codes() -> None:
    values = pd.Series(["a", "b"], index=pd.Index([1001, 42], name="index"), dtype="object")

    mapping = compact_categorical_labels_mapping_from_values(
        values,
        categories=["a", "b"],
        palette=["#ff0000", "#00ff00"],
    )

    np.testing.assert_array_equal(mapping.label_ids, np.asarray([42, 1001], dtype=np.int64))
    assert isinstance(mapping.texture_codes, np.ndarray)
    assert mapping.texture_codes.dtype == np.dtype(np.uint8)
    assert isinstance(mapping.texture_rgba, np.ndarray)
    assert mapping.texture_rgba.dtype == np.dtype(np.float32)


def test_compact_categorical_labels_mapping_skips_unique_check_for_sorted_label_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = pd.Series(
        pd.Categorical(["a", "b", "a"], categories=["a", "b"]),
        index=pd.Index([5, 6, 7], name="index"),
    )

    def fail_unique(_values: object) -> object:
        raise AssertionError("strictly increasing label ids should not call np.unique")

    monkeypatch.setattr(labels_colormap_module.np, "unique", fail_unique)

    mapping = compact_categorical_labels_mapping_from_values(
        values,
        categories=["a", "b"],
        palette=["#ff0000", "#00ff00"],
    )

    np.testing.assert_array_equal(mapping.label_ids, np.asarray([5, 6, 7], dtype=np.int64))


@pytest.mark.parametrize(
    ("values", "match"),
    [
        (pd.Series(["a"], index=pd.Index([0], name="index")), "positive label ids"),
        (pd.Series(["a", "b"], index=pd.Index([1, 1], name="index")), "unique label ids"),
        (pd.Series(["a"], index=pd.Index(["cell-1"], name="index")), "integer label ids"),
    ],
)
def test_compact_categorical_labels_mapping_rejects_invalid_label_ids(values: pd.Series, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        compact_categorical_labels_mapping_from_values(
            values,
            categories=["a"],
            palette=["#ff0000"],
        )


def test_compact_categorical_label_colormap_from_values_returns_ready_colormap(
    restore_colormap_backend: None,
) -> None:
    values = pd.Series(["a", "b", "a"], index=pd.Index([10, 20, 30], name="index"), dtype="object")

    colormap = compact_categorical_label_colormap_from_values(
        values,
        categories=["a", "b"],
        palette=["#ff0000", "#00ff00"],
    )

    assert isinstance(colormap, CompactLabelColormap)
    labels = np.asarray([0, 10, 20, 30, 99], dtype=np.int64)
    mapping = compact_categorical_labels_mapping_from_values(
        values,
        categories=["a", "b"],
        palette=["#ff0000", "#00ff00"],
    )
    expanded_colormap = direct_label_colormap_from_rgba(_expanded_color_dict(mapping))
    np.testing.assert_allclose(colormap.map(labels), expanded_colormap.map(labels))


def test_compact_categorical_label_colormap_from_values_accepts_numeric_rgba_palette(
    restore_colormap_backend: None,
) -> None:
    values = pd.Series(["a", "b"], index=pd.Index([1, 2], name="index"), dtype="object")
    red = np.asarray([1.0, 0.0, 0.0, 1.0], dtype=np.float32)
    green = np.asarray([0.0, 1.0, 0.0, 1.0], dtype=np.float32)

    colormap = compact_categorical_label_colormap_from_values(
        values,
        categories=["a", "b"],
        palette=[red, green],
    )

    np.testing.assert_allclose(colormap.map(1), red)
    np.testing.assert_allclose(colormap.map(2), green)


def test_compact_categorical_label_colormap_from_values_uses_configured_default_color(
    restore_colormap_backend: None,
) -> None:
    values = pd.Series(["a"], index=pd.Index([5], name="index"), dtype="object")

    colormap = compact_categorical_label_colormap_from_values(
        values,
        categories=["a"],
        palette=["#ff0000"],
        default_color=DEFAULT_NEUTRAL_COLOR,
    )

    np.testing.assert_allclose(colormap.map(99), to_rgba(DEFAULT_NEUTRAL_COLOR))
    np.testing.assert_allclose(colormap.map(0), np.zeros(4, dtype=np.float32))


def test_compact_categorical_label_colormap_sparse_category_updates_existing_texture(
    restore_colormap_backend: None,
) -> None:
    values = pd.Series([1], index=pd.Index([5], name="index"), dtype="int64")
    colormap = compact_categorical_label_colormap_from_values(
        values,
        categories=[0, 1, 2],
        palette=[DEFAULT_NEUTRAL_COLOR, "#ff0000", "#0000ff"],
        default_color=DEFAULT_NEUTRAL_COLOR,
    )
    original_texture_count = len(colormap._compact_mapping.texture_rgba)

    result = colormap.set_label_value(6, 2)

    assert len(colormap._compact_mapping.texture_rgba) == original_texture_count
    assert result.texture_code == colormap._compact_mapping.value_texture_codes[2]
    assert result.texture_table_changed is False
    assert 6 in colormap._compact_mapping.label_ids
    np.testing.assert_allclose(colormap.map(6), to_rgba("#0000ff"))


def test_compact_categorical_label_colormap_sparse_missing_reuses_missing_texture(
    restore_colormap_backend: None,
) -> None:
    values = pd.Series([1, pd.NA], index=pd.Index([5, 6], name="index"), dtype="Int64")
    colormap = compact_categorical_label_colormap_from_values(
        values,
        categories=[1, 2],
        palette=["#ff0000", "#0000ff"],
        missing_color="#00ff00",
    )
    original_texture_count = len(colormap._compact_mapping.texture_rgba)
    missing_texture_code = colormap._compact_mapping.missing_texture_code
    assert missing_texture_code is not None

    result = colormap.set_label_missing(5)

    assert len(colormap._compact_mapping.texture_rgba) == original_texture_count
    assert result.texture_code == missing_texture_code
    assert result.texture_table_changed is False
    assert 5 in colormap._compact_mapping.label_ids
    np.testing.assert_allclose(colormap.map(5), to_rgba("#00ff00"))
    np.testing.assert_allclose(colormap.map(99), np.zeros(4, dtype=np.float32))


def test_compact_categorical_label_colormap_sparse_missing_uses_reserved_texture_when_all_rows_have_values(
    restore_colormap_backend: None,
) -> None:
    values = pd.Series([1, 2], index=pd.Index([5, 6], name="index"), dtype="int64")
    colormap = compact_categorical_label_colormap_from_values(
        values,
        categories=[1, 2],
        palette=["#ff0000", "#0000ff"],
        missing_color=DEFAULT_NEUTRAL_COLOR,
    )
    original_texture_count = len(colormap._compact_mapping.texture_rgba)
    missing_texture_code = colormap._compact_mapping.missing_texture_code
    assert missing_texture_code is not None

    result = colormap.set_label_missing(5)

    assert len(colormap._compact_mapping.texture_rgba) == original_texture_count
    assert result.texture_code == missing_texture_code
    assert result.texture_table_changed is False
    assert 5 in colormap._compact_mapping.label_ids
    np.testing.assert_allclose(colormap.map(5), to_rgba(DEFAULT_NEUTRAL_COLOR))
    np.testing.assert_allclose(colormap.map(6), to_rgba("#0000ff"))
    np.testing.assert_allclose(colormap.map(99), np.zeros(4, dtype=np.float32))


def test_compact_labels_mapping_rejects_out_of_range_missing_texture_code() -> None:
    with pytest.raises(ValueError, match="missing texture code is out of range"):
        CompactLabelsMapping(
            label_ids=np.asarray([5], dtype=np.int64),
            texture_codes=np.asarray([2], dtype=np.uint8),
            texture_rgba=np.asarray(
                [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 1.0]],
                dtype=np.float32,
            ),
            missing_texture_code=3,
        )


def test_compact_categorical_label_colormap_sparse_new_category_appends_texture(
    restore_colormap_backend: None,
) -> None:
    values = pd.Series([1], index=pd.Index([5], name="index"), dtype="int64")
    colormap = compact_categorical_label_colormap_from_values(
        values,
        categories=[0, 1],
        palette=[DEFAULT_NEUTRAL_COLOR, "#ff0000"],
        default_color=DEFAULT_NEUTRAL_COLOR,
    )
    original_texture_count = len(colormap._compact_mapping.texture_rgba)

    result = colormap.set_label_value(9, 7, value_color="#0000ff")

    assert len(colormap._compact_mapping.texture_rgba) == original_texture_count + 1
    assert result.texture_code == len(colormap._compact_mapping.texture_rgba) - 1
    assert result.texture_table_changed is True
    assert colormap._compact_mapping.value_texture_codes[7] == result.texture_code
    assert 9 in colormap._compact_mapping.label_ids
    assert bool(np.all(colormap._compact_mapping.label_ids[1:] > colormap._compact_mapping.label_ids[:-1]))
    np.testing.assert_allclose(colormap.map(9), to_rgba("#0000ff"))


def test_compact_continuous_label_colormap_rejects_sparse_value_update(
    restore_colormap_backend: None,
) -> None:
    values = pd.Series([0.0, 1.0], index=pd.Index([5, 6], name="index"), dtype="float64")
    colormap = CompactLabelColormap(compact_continuous_labels_mapping_from_values(values))
    original_mapping = colormap._compact_mapping
    original_rgba = colormap.map(5).copy()

    with pytest.raises(ValueError, match="require a categorical compact labels mapping"):
        colormap.set_label_value(5, 7, value_color="#0000ff")

    assert colormap._compact_mapping is original_mapping
    np.testing.assert_allclose(colormap.map(5), original_rgba)


def test_compact_categorical_label_colormap_sparse_update_widens_texture_code_dtype(
    restore_colormap_backend: None,
) -> None:
    values = pd.Series([1], index=pd.Index([5], name="index"), dtype="int64")
    categories = list(range(300))
    palette = [DEFAULT_NEUTRAL_COLOR] + [
        np.asarray([i / 300.0, 1.0 - (i / 300.0), (i % 11) / 10.0, 1.0]) for i in range(1, 300)
    ]
    colormap = compact_categorical_label_colormap_from_values(
        values,
        categories=categories,
        palette=palette,
        default_color=DEFAULT_NEUTRAL_COLOR,
    )
    assert colormap._compact_mapping.texture_codes.dtype == np.uint8

    result = colormap.set_label_value(5, 299)

    assert result.texture_code > np.iinfo(np.uint8).max
    assert result.texture_table_changed is False
    assert colormap._compact_mapping.texture_codes.dtype == np.uint16
    assert int(colormap._compact_mapping.texture_codes[0]) == result.texture_code
    np.testing.assert_allclose(colormap.map(5), palette[299])


def test_compact_categorical_label_colormap_maps_like_expanded_direct_colormap(
    restore_colormap_backend: None,
) -> None:
    values = pd.Series(
        pd.Categorical(["T", None, "unknown", "B"], categories=["T", "B", "unknown"]),
        index=pd.Index([9, 3, 7, 5], name="index"),
    )
    compact_mapping = compact_categorical_labels_mapping_from_values(
        values,
        categories=["T", "B"],
        palette=["#ff0000", "#00ff00"],
    )

    compact_colormap = CompactLabelColormap(compact_mapping)
    expanded_colormap = direct_label_colormap_from_rgba(_expanded_color_dict(compact_mapping))

    assert isinstance(compact_colormap, DirectLabelColormap)
    assert len(compact_colormap.color_dict) < len(_expanded_color_dict(compact_mapping))
    labels = np.asarray([0, 3, 5, 7, 9, 123], dtype=np.int64)
    np.testing.assert_allclose(compact_colormap.map(labels), expanded_colormap.map(labels))


def test_compact_categorical_label_colormap_values_mapping_stays_compact(
    restore_colormap_backend: None,
) -> None:
    values = pd.Series(["a", "b"], index=pd.Index([1001, 42], name="index"), dtype="object")
    compact_mapping = compact_categorical_labels_mapping_from_values(
        values,
        categories=["a", "b"],
        palette=["#ff0000", "#00ff00"],
    )

    colormap = CompactLabelColormap(compact_mapping)
    label_mapping, texture_color_dict = colormap._values_mapping_to_minimum_values_set()

    assert not isinstance(label_mapping, dict)
    assert len(label_mapping) == len(compact_mapping.label_ids) + 2
    assert label_mapping[None] == compact_mapping.default_texture_code
    assert label_mapping[compact_mapping.background_value] == compact_mapping.background_texture_code
    assert label_mapping[42] == compact_mapping.texture_codes[0]
    assert set(texture_color_dict) == set(range(len(compact_mapping.texture_rgba)))


def test_compact_categorical_label_colormap_high_bit_texture_mapping_uses_jitted_typed_dict(
    restore_colormap_backend: None,
) -> None:
    values = pd.Series(["a", "b"], index=pd.Index([1001, 42], name="index"), dtype="object")
    compact_mapping = compact_categorical_labels_mapping_from_values(
        values,
        categories=["a", "b"],
        palette=["#ff0000", "#00ff00"],
    )
    colormap = CompactLabelColormap(compact_mapping)
    code_by_label = dict(zip(compact_mapping.label_ids.tolist(), compact_mapping.texture_codes.tolist(), strict=True))

    texture_values = colormap._data_to_texture(np.asarray([0, 42, 1001, 9999], dtype=np.int64))

    np.testing.assert_array_equal(
        texture_values,
        np.asarray(
            [
                compact_mapping.background_texture_code,
                code_by_label[42],
                code_by_label[1001],
                compact_mapping.default_texture_code,
            ],
            dtype=texture_values.dtype,
        ),
    )
    assert "_compact_int64_typed_dict" in colormap._cache_other
    assert not any(key == "_label_mapping_and_color_dict" for key in colormap.__dict__)


def test_compact_categorical_label_colormap_small_labels_use_dense_lookup(
    restore_colormap_backend: None,
) -> None:
    values = pd.Series(["a", "b"], index=pd.Index([3, 5], name="index"), dtype="object")
    compact_mapping = compact_categorical_labels_mapping_from_values(
        values,
        categories=["a", "b"],
        palette=["#ff0000", "#00ff00"],
    )
    compact_colormap = CompactLabelColormap(compact_mapping)
    expanded_colormap = direct_label_colormap_from_rgba(_expanded_color_dict(compact_mapping))

    labels = np.asarray([0, 3, 5, 11], dtype=np.uint16)

    np.testing.assert_array_equal(compact_colormap._data_to_texture(labels), labels)
    np.testing.assert_allclose(compact_colormap.map(labels), expanded_colormap.map(labels))
    assert compact_colormap._get_mapping_from_cache(np.dtype(np.uint16)).shape == (65536, 4)


def test_compact_categorical_label_colormap_selection_mode_matches_direct_colormap(
    restore_colormap_backend: None,
) -> None:
    values = pd.Series(["a", "b"], index=pd.Index([3, 5], name="index"), dtype="object")
    compact_mapping = compact_categorical_labels_mapping_from_values(
        values,
        categories=["a", "b"],
        palette=["#ff0000", "#00ff00"],
    )
    compact_colormap = CompactLabelColormap(compact_mapping)
    expanded_colormap = direct_label_colormap_from_rgba(_expanded_color_dict(compact_mapping))
    compact_colormap.use_selection = True
    expanded_colormap.use_selection = True
    compact_colormap.selection = 5
    expanded_colormap.selection = 5

    labels = np.asarray([0, 3, 5, 11], dtype=np.int64)

    np.testing.assert_allclose(compact_colormap.map(labels), expanded_colormap.map(labels))
    label_mapping, color_dict = compact_colormap._values_mapping_to_minimum_values_set()
    assert label_mapping == {5: 1, None: 0}
    assert set(color_dict) == {0, 1}


def test_compact_categorical_label_colormap_assigns_to_labels_layer_in_direct_mode(
    restore_colormap_backend: None,
) -> None:
    values = pd.Series(["a", "b"], index=pd.Index([3, 5], name="index"), dtype="object")
    compact_mapping = compact_categorical_labels_mapping_from_values(
        values,
        categories=["a", "b"],
        palette=["#ff0000", "#00ff00"],
    )
    colormap = CompactLabelColormap(compact_mapping)
    layer = Labels(np.asarray([[0, 3], [5, 11]], dtype=np.int64))

    layer.colormap = colormap

    assert layer.colormap is colormap
    assert str(layer._color_mode) == "direct"
    np.testing.assert_allclose(layer.get_color(5), colormap.map(5))


def test_direct_label_colormap_from_rgba_matches_normal_constructor() -> None:
    color_dict = {
        None: _rgba(0.0, 0.0, 0.0, 0.0),
        0: _rgba(0.0, 0.0, 0.0, 0.0),
        5: _rgba(1.0, 0.0, 0.0, 1.0),
        9: _rgba(0.0, 0.0, 1.0, 0.5),
    }

    fast_colormap = direct_label_colormap_from_rgba(color_dict, background_value=0)
    normal_colormap = DirectLabelColormap(color_dict=color_dict, background_value=0)

    assert isinstance(fast_colormap, DirectLabelColormap)
    assert fast_colormap.background_value == normal_colormap.background_value
    values = np.asarray([0, 5, 9, 42], dtype=np.int64)
    np.testing.assert_allclose(fast_colormap.map(values), normal_colormap.map(values))


def test_direct_label_colormap_from_rgba_uses_configured_background_value() -> None:
    color_dict = {
        None: _rgba(0.0, 0.0, 0.0, 0.0),
        99: _rgba(0.0, 0.0, 0.0, 0.0),
        5: _rgba(1.0, 0.0, 0.0, 1.0),
    }

    fast_colormap = direct_label_colormap_from_rgba(color_dict, background_value=99)
    normal_colormap = DirectLabelColormap(color_dict=color_dict, background_value=99)

    assert fast_colormap.background_value == 99
    values = np.asarray([99, 5, 0], dtype=np.int64)
    np.testing.assert_allclose(fast_colormap.map(values), normal_colormap.map(values))


def test_direct_label_colormap_from_rgba_keeps_event_emitters_and_clear_cache() -> None:
    color_dict = {
        None: _rgba(0.0, 0.0, 0.0, 0.0),
        0: _rgba(0.0, 0.0, 0.0, 0.0),
        7: _rgba(0.25, 0.5, 0.75, 1.0),
    }

    colormap = direct_label_colormap_from_rgba(color_dict, background_value=0)
    _ = colormap._num_unique_colors
    _ = colormap._label_mapping_and_color_dict

    assert hasattr(colormap.events, "color_dict")
    colormap._clear_cache()
    assert "_num_unique_colors" not in colormap.__dict__
    assert "_label_mapping_and_color_dict" not in colormap.__dict__


def test_direct_label_colormap_from_rgba_rejects_string_default_colors() -> None:
    color_dict = {
        None: _rgba(0.0, 0.0, 0.0, 0.0),
        0: "transparent",
        1: _rgba(1.0, 0.0, 0.0, 1.0),
    }

    with pytest.raises(ValueError, match="default/background color"):
        direct_label_colormap_from_rgba(color_dict)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("color_dict", "match"),
    [
        (
            {0: _rgba(0.0, 0.0, 0.0, 0.0), 1: _rgba(1.0, 0.0, 0.0, 1.0)},
            "`None` default color",
        ),
        (
            {None: _rgba(0.0, 0.0, 0.0, 0.0), 1: _rgba(1.0, 0.0, 0.0, 1.0)},
            "background label",
        ),
        (
            {None: _rgba(0.0, 0.0, 0.0, 0.0), 0: np.asarray([0.0, 0.0, 0.0]), 1: _rgba(1.0, 0.0, 0.0, 1.0)},
            "shape",
        ),
    ],
)
def test_direct_label_colormap_from_rgba_rejects_invalid_default_input(
    color_dict: dict[object, object],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        direct_label_colormap_from_rgba(color_dict)  # type: ignore[arg-type]


def test_direct_label_colormap_from_rgba_trusts_per_label_values() -> None:
    color_dict = {
        None: _rgba(0.0, 0.0, 0.0, 0.0),
        0: _rgba(0.0, 0.0, 0.0, 0.0),
        1.5: _rgba(1.0, 0.0, 0.0, 1.0),
        2: "trusted-by-caller",
    }

    colormap = direct_label_colormap_from_rgba(color_dict)  # type: ignore[arg-type]

    assert 1.5 in colormap.color_dict
    assert colormap.color_dict[2] == "trusted-by-caller"


def test_direct_label_colormap_from_rgba_uses_trusted_mapping_without_copying() -> None:
    color_dict = {
        None: _rgba(0.0, 0.0, 0.0, 0.0),
        0: _rgba(0.0, 0.0, 0.0, 0.0),
        1: _rgba(1.0, 0.0, 0.0, 1.0),
    }

    colormap = direct_label_colormap_from_rgba(color_dict)
    color_dict[2] = _rgba(0.0, 1.0, 0.0, 1.0)

    assert colormap.color_dict is color_dict
    assert 2 in colormap.color_dict
