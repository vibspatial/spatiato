from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from napari.layers import Labels
from napari.layers.labels._labels_utils import get_contours
from spatialdata import SpatialData

from spatiato.core.class_palette import DEFAULT_NEUTRAL_COLOR
from spatiato.viewer.adapter import _build_labels_layer
from spatiato.viewer.labels_colormap import compact_categorical_label_colormap_from_values, neutral_label_colormap
from spatiato.viewer.labels_contours import (
    _calculate_labels_contour,
    _contours_2d,
    _contours_2d_simple,
    install_fast_labels_contours,
    labels_contours,
)

_INTEGER_DTYPES = [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64]


def _block_labels(shape: tuple[int, int], *, cell: int = 6, max_label: int = 500, seed: int = 0) -> np.ndarray:
    """Square objects of `cell` pixels with random ids, about 40% background."""
    rng = np.random.default_rng(seed)
    grid = (shape[0] // cell + 1, shape[1] // cell + 1)
    ids = rng.integers(1, max_label + 1, size=grid, dtype=np.int32)
    ids[rng.random(grid) < 0.4] = 0
    return np.kron(ids, np.ones((cell, cell), dtype=np.int32))[: shape[0], : shape[1]]


def _full_slice(data: np.ndarray) -> tuple[slice, ...]:
    return tuple(slice(0, size) for size in data.shape)


def _reference_contours(labels: np.ndarray, thickness: int, background: int) -> np.ndarray:
    """napari's contour rule written with numpy, without scipy.

    A pixel keeps its label when the maximum over its cross neighbourhood
    differs from the minimum over the diamond of radius `thickness`. Borders
    are mirrored as in scipy's "reflect" mode, which is numpy's "symmetric".
    """
    padded = np.pad(labels, thickness, mode="symmetric")
    height, width = labels.shape
    info = np.iinfo(labels.dtype)
    cross_max = np.full(labels.shape, info.min, dtype=labels.dtype)
    diamond_min = np.full(labels.shape, info.max, dtype=labels.dtype)
    for dy in range(-thickness, thickness + 1):
        for dx in range(-thickness, thickness + 1):
            distance = abs(dy) + abs(dx)
            if distance > thickness:
                continue
            shifted = padded[thickness + dy : thickness + dy + height, thickness + dx : thickness + dx + width]
            diamond_min = np.minimum(diamond_min, shifted)
            if distance <= 1:
                cross_max = np.maximum(cross_max, shifted)
    contours = labels.copy()
    contours[cross_max == diamond_min] = background
    return contours


def _irregular_labels(dtype: type[np.integer]) -> dict[str, np.ndarray]:
    """Small label images with holes, thin parts, touching labels and labels along the edges."""
    cases = {}

    holes = np.zeros((25, 31), dtype=dtype)
    holes[3:22, 4:27] = 5
    holes[9:16, 11:20] = 0
    holes[11:14, 14:17] = 9  # a cell inside the hole
    holes[12, 22] = 0  # a one-pixel hole
    cases["cell with holes and a nested cell"] = holes

    thin = np.zeros((20, 20), dtype=dtype)
    thin[5, 2:18] = 3
    thin[2:18, 10] = 4
    thin[15, 15] = 6
    cases["one-pixel-wide lines and a one-pixel cell"] = thin

    cases["checkerboard of touching labels"] = (np.indices((16, 17)).sum(axis=0) % 2).astype(dtype) + 1

    corner = np.zeros((12, 12), dtype=dtype)
    corner[:6, :6] = 1
    corner[6:, 6:] = 2
    cases["labels touching only at a corner"] = corner

    edges = np.zeros((14, 18), dtype=dtype)
    edges[:, :3] = 7
    edges[0, :] = 8
    edges[-2:, -2:] = 9
    cases["labels along the slice edges"] = edges

    pieces = np.zeros((15, 30), dtype=dtype)
    pieces[2:6, 2:6] = 4
    pieces[9:13, 20:27] = 4
    cases["one label in two pieces"] = pieces
    return cases


@pytest.mark.parametrize("dtype", _INTEGER_DTYPES)
def test_labels_contours_matches_napari(dtype: type[np.integer]) -> None:
    rng = np.random.default_rng(3)
    for shape in ((1, 1), (1, 7), (7, 1), (5, 3), (37, 53)):
        data = rng.integers(0, 6, size=shape).astype(dtype)
        data[rng.random(shape) < 0.3] = 0
        for thickness in (1, 2, 3, 5):
            for background in (0, 2):
                contours = labels_contours(data, thickness, background)
                assert contours.dtype == data.dtype
                np.testing.assert_array_equal(contours, get_contours(data, thickness, background))


def test_labels_contours_matches_napari_with_non_zero_background() -> None:
    # Large enough for pixels away from the edge, where the kernel skips the
    # index reflection, and with background pixels that are not label 0.
    data = _block_labels((61, 83))
    data[data == 0] = 7
    for thickness in (1, 2, 3):
        np.testing.assert_array_equal(labels_contours(data, thickness, 7), get_contours(data, thickness, 7))
        # A background value that no pixel has.
        np.testing.assert_array_equal(labels_contours(data, thickness, 0), get_contours(data, thickness, 0))


def test_labels_contours_matches_napari_on_non_contiguous_input() -> None:
    data = _block_labels((40, 61))
    for view in (data[:, ::2], data.T, data[::-1]):
        assert not view.flags.c_contiguous
        for thickness in (1, 2):
            np.testing.assert_array_equal(labels_contours(view, thickness, 0), get_contours(view, thickness, 0))


@pytest.mark.parametrize("dtype", _INTEGER_DTYPES)
def test_labels_contours_matches_napari_on_irregular_labels(dtype: type[np.integer]) -> None:
    info = np.iinfo(dtype)
    for name, data in _irregular_labels(dtype).items():
        for thickness in (1, 2, 3, 5):
            for background in (0, 1, int(info.min), int(info.max)):
                expected = get_contours(data, thickness, background)
                message = f"{name}, contour {thickness}, background {background}"
                np.testing.assert_array_equal(labels_contours(data, thickness, background), expected, err_msg=message)
                # The numpy reference used below for extreme values agrees with napari here.
                np.testing.assert_array_equal(
                    _reference_contours(data, thickness, background), expected, err_msg=message
                )


@pytest.mark.parametrize("dtype", _INTEGER_DTYPES)
def test_labels_contours_follows_napari_rule_at_extreme_label_values(dtype: type[np.integer]) -> None:
    # Compared with the numpy reference rather than napari: for 64-bit labels
    # with neighbouring values near the dtype limits, the scipy grey dilation
    # and erosion that napari's `get_contours` uses return wrong maxima and
    # minima, so napari draws those outlines wrong.
    info = np.iinfo(dtype)
    extremes = np.full((20, 23), info.max, dtype=dtype)
    extremes[4:12, 3:9] = info.min
    extremes[6:9, 5:7] = 0
    extremes[14:18, 10:20] = info.max - 1
    full_range = np.random.default_rng(7).integers(info.min, info.max, size=(41, 47), dtype=dtype, endpoint=True)
    for data in (extremes, full_range):
        for thickness in (1, 2, 3, 5):
            for background in (0, int(info.min), int(info.max)):
                np.testing.assert_array_equal(
                    labels_contours(data, thickness, background),
                    _reference_contours(data, thickness, background),
                )


def _run_contour_kernel(kernel, labels: np.ndarray, thickness: int, background: int) -> np.ndarray:
    out = np.empty(labels.shape, dtype=labels.dtype)
    return kernel(labels, thickness, labels.dtype.type(background), out)


@pytest.mark.parametrize("dtype", [np.uint16, np.int32, np.int64])
def test_contour_kernel_matches_simple_kernel(dtype: type[np.integer]) -> None:
    # The shortcuts in `_contours_2d` must not change its output.
    rng = np.random.default_rng(5)
    noise = rng.integers(0, 50, size=(301, 407)).astype(dtype)
    noise[rng.random(noise.shape) < 0.3] = 0
    images = [*_irregular_labels(dtype).values(), _block_labels((301, 407)).astype(dtype), noise]
    for data in images:
        for thickness in (1, 2, 3, 5):
            for background in (0, 1, 9):
                np.testing.assert_array_equal(
                    _run_contour_kernel(_contours_2d, data, thickness, background),
                    _run_contour_kernel(_contours_2d_simple, data, thickness, background),
                )


def test_labels_contours_matches_napari_on_blobs_segmentation(sdata_blobs: SpatialData) -> None:
    segmentation = np.asarray(sdata_blobs.labels["blobs_labels"])
    for thickness in (1, 2, 3, 5):
        np.testing.assert_array_equal(
            labels_contours(segmentation, thickness, 0),
            get_contours(segmentation, thickness, 0),
        )


@pytest.mark.parametrize("shape", [(0, 5), (5, 0), (0, 0)])
def test_labels_contours_handles_empty_slices(shape: tuple[int, int]) -> None:
    contours = labels_contours(np.zeros(shape, dtype=np.int32), 2, 0)

    assert contours.shape == shape


@pytest.mark.parametrize(
    ("labels", "thickness", "background", "match"),
    [
        (np.zeros((2, 3, 4), dtype=np.int32), 1, 0, "2D integer"),
        (np.zeros((3, 4), dtype=np.float32), 1, 0, "2D integer"),
        (np.zeros((3, 4), dtype=np.int32), 0, 0, "at least 1"),
        (np.zeros((3, 4), dtype=np.uint8), 1, -1, "does not fit"),
    ],
)
def test_labels_contours_rejects_unsupported_input(
    labels: np.ndarray,
    thickness: int,
    background: int,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        labels_contours(labels, thickness, background)


def _styled_colormap(data: np.ndarray):
    label_ids = np.unique(data[data > 0])
    values = pd.Series(
        np.where(label_ids % 2 == 0, "even", "odd").astype(object),
        index=pd.Index(label_ids, name="index"),
    )
    return compact_categorical_label_colormap_from_values(
        values,
        categories=["odd", "even"],
        palette=["#ff0000", "#00ff00"],
    )


@pytest.mark.parametrize("contour", [0, 1, 2])
@pytest.mark.parametrize("colormap_kind", ["napari_default", "styled", "neutral"])
def test_installed_layer_draws_like_napari(contour: int, colormap_kind: str) -> None:
    data = _block_labels((61, 83))
    napari_layer = Labels(data)
    fast_layer = Labels(data)
    install_fast_labels_contours(fast_layer)
    if colormap_kind == "napari_default":
        colormap = napari_layer.colormap
    elif colormap_kind == "styled":
        colormap = _styled_colormap(data)
    else:
        colormap = neutral_label_colormap(DEFAULT_NEUTRAL_COLOR, background_value=0)
    napari_layer.colormap = colormap
    fast_layer.colormap = colormap
    napari_layer.contour = contour
    fast_layer.contour = contour

    np.testing.assert_array_equal(fast_layer._raw_to_displayed(data), napari_layer._raw_to_displayed(data))
    # A partial slice, as when painting: the border around it must match napari's.
    partial_slice = (slice(7, 30), slice(11, 52))
    np.testing.assert_array_equal(
        fast_layer._raw_to_displayed(data, partial_slice),
        napari_layer._raw_to_displayed(data, partial_slice),
    )


@pytest.mark.parametrize("contour", [1, 2])
def test_installed_layer_never_uses_napari_contours(monkeypatch: pytest.MonkeyPatch, contour: int) -> None:
    # This also fails if napari renames `_calculate_contour`: the replacement
    # would then never be called, and napari's scipy path would run again.
    data = _block_labels((40, 50))
    reference_layer = Labels(data)
    reference_layer.contour = contour
    expected = reference_layer._raw_to_displayed(data)

    def fail_napari_contours(*args: object, **kwargs: object) -> np.ndarray:
        raise AssertionError("napari's scipy contour path was used")

    monkeypatch.setattr("napari.layers.labels.labels.get_contours", fail_napari_contours)
    napari_layer = Labels(data)
    with pytest.raises(AssertionError, match="scipy contour path"):
        # The patch is on napari's real path: setting `contour` refreshes the slice.
        napari_layer.contour = contour

    fast_layer = Labels(data)
    install_fast_labels_contours(fast_layer)
    fast_layer.contour = contour

    np.testing.assert_array_equal(fast_layer._raw_to_displayed(data), expected)


def test_installed_layer_keeps_napari_behavior_without_2d_contours() -> None:
    data = _block_labels((20, 30))
    layer = Labels(data)
    install_fast_labels_contours(layer)

    assert layer._calculate_contour(data, _full_slice(data)) is None  # contour 0: no contours

    layer.contour = 1
    volume = np.stack([data, data, data])
    with pytest.warns(UserWarning, match="3D rendering"):
        assert layer._calculate_contour(volume, _full_slice(volume)) is None


def test_build_labels_layer_installs_fast_contours(sdata_blobs: SpatialData) -> None:
    layer = _build_labels_layer(sdata_blobs, "blobs_labels", "global", name="blobs_labels")

    assert layer._calculate_contour.__func__ is _calculate_labels_contour
