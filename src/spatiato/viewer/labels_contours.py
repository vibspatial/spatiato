"""Fast contours for napari Labels layers.

With `contour > 0`, napari draws a Labels layer by keeping only the outline
pixels of every new 2D slice before recoding it: `Labels._calculate_contour`
calls napari's `get_contours`, a single-threaded scipy grey dilation and
erosion on the Qt main thread. That takes about 74 ms for a 5.2 Mpx slice with
contour 1 and 100 ms with contour 2. The parallel numba kernel here gives the
same output in about 1 ms and 6 ms. Spatiato installs it on every Labels layer
it builds.

The override can go once napari ships fast contours itself:
https://github.com/vibspatial/spatiato/issues/136.
"""

from __future__ import annotations

from types import MethodType

import numpy as np
from napari.layers import Labels
from numba import njit, prange


@njit(cache=True, inline="always")
def _reflect(index, size):
    """Map an index outside `[0, size)` as scipy.ndimage's "reflect" mode: -1 -> 0, size -> size - 1."""
    period = 2 * size
    index = index % period
    if index < 0:
        index += period
    if index >= size:
        index = period - 1 - index
    return index


# napari keeps a pixel when the maximum over its cross neighbourhood differs
# from the minimum over the diamond of radius `thickness`. The diamond contains
# the cross and the pixel itself, so a pixel is on an outline when one of its 4
# neighbours differs from it, or, for thicker contours, when any pixel in the
# diamond is smaller than it. The two functions below test that rule, with
# plain and with reflected indices.


@njit(inline="always")
def _is_boundary(labels, y, x, thickness, value):
    """Whether pixel (y, x) is on an outline; its whole diamond lies inside `labels`."""
    if labels[y - 1, x] != value or labels[y + 1, x] != value or labels[y, x - 1] != value or labels[y, x + 1] != value:
        return True
    if thickness > 1:
        for dy in range(-thickness, thickness + 1):
            radius = thickness - abs(dy)
            row = y + dy
            for dx in range(-radius, radius + 1):
                if labels[row, x + dx] < value:
                    return True
    return False


@njit(inline="always")
def _is_boundary_reflect(labels, y, x, thickness, value, height, width):
    """Whether pixel (y, x) is on an outline, mirroring indices outside `labels`."""
    if (
        labels[_reflect(y - 1, height), x] != value
        or labels[_reflect(y + 1, height), x] != value
        or labels[y, _reflect(x - 1, width)] != value
        or labels[y, _reflect(x + 1, width)] != value
    ):
        return True
    if thickness > 1:
        for dy in range(-thickness, thickness + 1):
            radius = thickness - abs(dy)
            row = _reflect(y + dy, height)
            for dx in range(-radius, radius + 1):
                if labels[row, _reflect(x + dx, width)] < value:
                    return True
    return False


@njit(parallel=True, cache=True)
def _contours_2d_simple(labels, thickness, background, out):
    """The straightforward version of `_contours_2d`: the same output, more slowly.

    It tests every pixel with mirrored indices. `_contours_2d` adds two
    shortcuts that keep the output identical: background pixels stay
    background without a check, and pixels at least `thickness` from the slice
    edge use plain indices. On a 5.2 Mpx slice (Apple M2 Max), this version
    takes about 4, 23 and 34 ms for contour 1, 2 and 3; `_contours_2d` takes
    about 1, 6 and 7 ms. Not used for drawing; kept as the readable reference
    and tested against `_contours_2d`.
    """
    height, width = labels.shape
    for y in prange(height):
        for x in range(width):
            value = labels[y, x]
            boundary = _is_boundary_reflect(labels, y, x, thickness, value, height, width)
            out[y, x] = value if boundary else background
    return out


@njit(parallel=True, cache=True)
def _contours_2d(labels, thickness, background, out):
    """Write the contours of `labels` into `out`; see `labels_contours`.

    Gives the same output as `_contours_2d_simple`, with two shortcuts that
    make it about four times faster.
    """
    height, width = labels.shape
    for y in prange(height):
        interior_row = thickness <= y < height - thickness
        for x in range(width):
            value = labels[y, x]
            if value == background:
                # A background pixel stays background, on an outline or not.
                out[y, x] = background
                continue
            # Only pixels within `thickness` of the slice edge reach outside
            # it; the index reflection is too slow to pay for every pixel.
            if interior_row and thickness <= x < width - thickness:
                boundary = _is_boundary(labels, y, x, thickness, value)
            else:
                boundary = _is_boundary_reflect(labels, y, x, thickness, value, height, width)
            out[y, x] = value if boundary else background
    return out


def labels_contours(labels: np.ndarray, thickness: int, background: int) -> np.ndarray:
    """Return the contours of a 2D integer labels array, as napari's `get_contours`.

    Outline pixels keep their label and every other pixel becomes
    `background`. `thickness` sets how far the outline reaches into each
    object, like napari's `contour` setting. Pixels outside the array are
    mirrored, as in scipy.ndimage's "reflect" mode that napari uses.

    Parameters
    ----------
    labels
        2D array of integer label ids. It may be non-contiguous.
    thickness
        Outline thickness, at least 1.
    background
        Label id for pixels that are not on an outline. It must fit
        `labels.dtype`.

    Returns
    -------
    np.ndarray
        A new array with the shape and dtype of `labels`.
    """
    if labels.ndim != 2 or labels.dtype.kind not in "iu":
        raise ValueError(f"Labels contours need a 2D integer array, got {labels.ndim}D {labels.dtype}.")
    if thickness < 1:
        raise ValueError(f"Labels contour thickness must be at least 1, got {thickness}.")
    if not _fits_dtype(background, labels.dtype):
        raise ValueError(f"Background label {background} does not fit labels of dtype {labels.dtype}.")
    out = np.empty(labels.shape, dtype=labels.dtype)
    return _contours_2d(labels, int(thickness), labels.dtype.type(background), out)


def install_fast_labels_contours(layer: Labels) -> None:
    """Draw `layer`'s contours with the numba kernel instead of napari's scipy step.

    `Labels._raw_to_displayed` calls `self._calculate_contour` for every new
    slice, so an instance attribute replaces napari's method on this layer
    only. Other Labels layers in the session keep napari's behavior.
    """
    layer._calculate_contour = MethodType(_calculate_labels_contour, layer)


def _calculate_labels_contour(
    layer: Labels,
    labels: np.ndarray,
    data_slice: tuple[slice, ...],
) -> np.ndarray | None:
    """Replace `Labels._calculate_contour` with the same result from the numba kernel.

    Contours off, 3D slices and input the kernel does not take go through
    napari's own method, which returns `None` or warns as before.
    """
    background = layer.colormap.background_value
    if (
        layer.contour < 1
        or not isinstance(labels, np.ndarray)
        or labels.ndim != 2
        or labels.dtype.kind not in "iu"
        or not _fits_dtype(background, labels.dtype)
    ):
        return Labels._calculate_contour(layer, labels, data_slice)

    # As napari does: compute the contours on the slice plus a border, so
    # pixels at the edge of a partial slice see their real neighbours, then
    # trim the border off again.
    border = max(1, int(layer.contour))
    expanded_slice = _expand_slice(data_slice, labels.shape, border)
    contours = labels_contours(labels[expanded_slice], int(layer.contour), background)
    return contours[
        tuple(
            slice(inner.start - outer.start, inner.stop - outer.start)
            for inner, outer in zip(data_slice, expanded_slice, strict=False)
        )
    ]


def _expand_slice(data_slice: tuple[slice, ...], shape: tuple[int, ...], offset: int) -> tuple[slice, ...]:
    """Grow each slice by `offset` on both sides, clipped to `shape`, as napari's `expand_slice`."""
    return tuple(
        slice(
            max(0, min(size, axis_slice.start - offset)),
            max(0, min(size, axis_slice.stop + offset)),
            axis_slice.step,
        )
        for axis_slice, size in zip(data_slice, shape, strict=False)
    )


def _fits_dtype(value: int, dtype: np.dtype) -> bool:
    info = np.iinfo(dtype)
    return bool(info.min <= value <= info.max)
