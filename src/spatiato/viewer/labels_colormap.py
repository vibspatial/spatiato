from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from matplotlib import colormaps
from matplotlib.colors import to_rgba
from napari.utils.colormap_backend import ColormapBackend, set_backend
from napari.utils.colormaps import DirectLabelColormap
from napari.utils.colormaps import _accelerated_cmap as _accel_cmap
from numba import njit, prange, typed, types

from spatiato.viewer._styling import (
    MISSING_CATEGORICAL_COLOR,
    MISSING_CONTINUOUS_COLOR,
    OVERLAY_CONTINUOUS_COLORMAP,
    normalize_category_value,
)

_TRANSPARENT_RGBA = np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

# A continuous mapping has `bins + 3` texture rows: default, background, the
# bins and missing, plus a zero row when zeros get their own color. Napari
# stores texture codes as uint8 only up to 255 colors
# (`minimum_dtype_for_labels`), so 252 bins, or 251 with a zero row, is the
# largest count that keeps the per-slice labels texture uint8 instead of uint16.
_CONTINUOUS_LABEL_BINS = 252

# Upper bound for one colormap's dense table (the `label_id -> texture_code`
# array of `CompactLabelColormap._get_dense_texture_lut`). Its size follows the
# largest table label id, so typical tables stay small; very large or hashed
# label ids fall back to napari's per-pixel typed-dict lookup.
_DENSE_LUT_MAX_BYTES = 128 * 1024**2
_DENSE_LUT_CACHE_KEY = "_compact_dense_texture_lut"


@njit(cache=True)
def _fill_typed_label_texture_mapping(
    typed_mapping,
    label_ids: np.ndarray,
    texture_codes: np.ndarray,
    background_value,
    background_texture_code,
):
    typed_mapping[background_value] = background_texture_code
    for i in range(label_ids.shape[0]):
        typed_mapping[label_ids[i]] = texture_codes[i]
    return typed_mapping


@njit(parallel=True, cache=True)
def _gather_signed_label_texture_codes(flat_values, lut, out):
    """Write `lut[value]` for every in-table value; leave other entries untouched."""
    n = lut.shape[0]
    for i in prange(flat_values.size):
        value = flat_values[i]
        if value >= 0 and value < n:
            out[i] = lut[value]
    return out


@njit(parallel=True, cache=True)
def _gather_unsigned_label_texture_codes(flat_values, lut, out):
    """Unsigned variant, so numba never compares int64 with uint64."""
    n = np.uint64(lut.shape[0])
    for i in prange(flat_values.size):
        value = np.uint64(flat_values[i])
        if value < n:
            out[i] = lut[value]
    return out


@dataclass(frozen=True)
class CompactLabelsMapping:
    """Compact labels color state.

    `label_ids -> texture_codes` is the logical per-object mapping. RGBA values
    live once in `texture_rgba`, where each row index is the texture code. The
    color for `label_ids[i]` is therefore `texture_rgba[texture_codes[i]]`. The
    state is intentionally independent of napari colormap internals; the dense
    table (`CompactLabelColormap._get_dense_texture_lut`), napari's numba typed
    dicts and napari's low-bit `label_id -> RGBA` lookups are derived from these
    arrays when napari asks for them.

    Parameters
    ----------
    label_ids
        Positive label ids / instance ids represented by this compact mapping.
    texture_codes
        Texture code for each entry in `label_ids`.
    texture_rgba
        RGBA lookup table where the row index is the texture code.
    missing_texture_code
        Texture code used for known table rows whose source value is missing or
        palette-unknown. The compact builders always reserve this row, also when
        no row is missing yet, so sparse updates can mark a label missing
        without growing `texture_rgba`.
    value_texture_codes
        Source value to texture-code lookup for categorical mappings. This is
        used by sparse updates that need to change one label's value without
        rebuilding the full compact mapping. `None` for continuous mappings,
        which have no categorical values and do not support
        `CompactLabelColormap.set_label_value(...)`.
    default_texture_code
        Texture code used for labels that are not present in `label_ids`.
        This corresponds to the current `DirectLabelColormap` `None` default
        color for unmapped labels, not to known table rows with missing
        categorical values.
    background_texture_code
        Texture code used for the background label.
    background_value
        Background label id.

    Examples
    --------
    If four labels are red and four labels are green:

    ```text
    label_ids:     [101, 102, 103, 104, 105, 106, 107, 108]
    texture_codes: [2,   2,   2,   2,   3,   3,   3,   3]

    texture_rgba: [transparent default, transparent background, red, green, neutral]

    texture_rgba:
      0 -> transparent default for unmapped labels
      1 -> transparent background label
      2 -> red
      3 -> green
      4 -> neutral missing-value color (reserved, currently unused)

    value_texture_codes:
      "tumor"  -> 2
      "stroma" -> 3

    missing_texture_code: 4
    ```

    If a later sparse edit annotates label `109` as `"tumor"`, Spatiato can add
    `109 -> 2` to `label_ids` / `texture_codes` without rebuilding the full
    colormap. If the edit introduces a brand-new value such as `"immune"`,
    Spatiato appends one new RGBA row, adds `"immune" -> 5` to
    `value_texture_codes`, and maps the edited label to texture code `5`.
    This is why value ownership of texture codes is stored explicitly
    instead of rediscovering it from repeated RGBA values.
    """

    label_ids: np.ndarray
    texture_codes: np.ndarray
    texture_rgba: np.ndarray
    missing_texture_code: int
    value_texture_codes: dict[object, int] | None = None
    default_texture_code: int = 0
    background_texture_code: int = 1
    background_value: int = 0

    def __post_init__(self) -> None:
        if self.label_ids.ndim != 1:
            raise ValueError("Compact labels mapping `label_ids` must be one-dimensional.")
        if self.texture_codes.ndim != 1:
            raise ValueError("Compact labels mapping `texture_codes` must be one-dimensional.")
        if len(self.label_ids) != len(self.texture_codes):
            raise ValueError("Compact labels mapping must contain one texture code per label id.")
        if self.texture_rgba.ndim != 2 or self.texture_rgba.shape[1] != 4:
            raise ValueError("Compact labels mapping `texture_rgba` must have shape `(n, 4)`.")
        if len(self.texture_rgba) == 0:
            raise ValueError("Compact labels mapping must contain at least one texture RGBA row.")
        if self.default_texture_code != 0:
            raise ValueError("Compact labels mapping default texture code must be 0.")
        if not 0 <= self.background_texture_code < len(self.texture_rgba):
            raise ValueError("Compact labels mapping background texture code is out of range.")
        if not 0 <= self.missing_texture_code < len(self.texture_rgba):
            raise ValueError("Compact labels mapping missing texture code is out of range.")
        if len(self.texture_codes) and int(np.max(self.texture_codes)) >= len(self.texture_rgba):
            raise ValueError("Compact labels mapping contains a texture code without an RGBA row.")
        if self.value_texture_codes is not None and any(
            not 0 <= int(texture_code) < len(self.texture_rgba) for texture_code in self.value_texture_codes.values()
        ):
            raise ValueError("Compact labels mapping contains a value texture code without an RGBA row.")
        if np.any(self.label_ids == self.background_value):
            raise ValueError("Compact labels mapping label ids must not include the background value.")


@dataclass(frozen=True)
class _CompactSparseLabelUpdateResult:
    """Result of one Spatiato sparse annotation update on compact label colors."""

    texture_code: int
    texture_table_changed: bool


class _CompactLabelToTextureMapping(Mapping[int | None, int]):
    """Array-backed `label_id -> texture_code` mapping view.

    This wraps the `CompactLabelsMapping` passed to `__init__` and
    exposes its `label_ids` / `texture_codes` arrays through the mapping API.
    Napari expects the first item of `_label_mapping_and_color_dict` to behave
    like a mapping. In scalar high-bit conversion, napari does roughly:

    ```python
    mapper = colormap._label_mapping_and_color_dict[0]
    texture_code = mapper.get(label_id, default)
    ```

    This view supports that without building a large Python dict. Full
    high-bit label arrays use the dense table from
    `CompactLabelColormap._get_dense_texture_lut()`, or
    `_get_typed_dict_mapping(...)` when that table is not used, so this scalar
    lookup path is not the hot per-pixel rendering path.
    """

    def __init__(self, compact_mapping: CompactLabelsMapping) -> None:
        self._compact_mapping = compact_mapping

    def __getitem__(self, key: int | None) -> int:
        compact = self._compact_mapping
        if key is None:
            return compact.default_texture_code
        label_id = int(key)
        if label_id == compact.background_value:
            return compact.background_texture_code
        position = np.searchsorted(compact.label_ids, label_id)
        if position < len(compact.label_ids) and compact.label_ids[position] == label_id:
            return int(compact.texture_codes[position])
        raise KeyError(key)

    def __iter__(self) -> Iterator[int | None]:
        yield None
        yield self._compact_mapping.background_value
        yield from (int(label_id) for label_id in self._compact_mapping.label_ids)

    def __len__(self) -> int:
        return len(self._compact_mapping.label_ids) + 2


class CompactLabelColormap(DirectLabelColormap):
    """Direct labels colormap backed by compact label-to-texture state.

    Methods that mirror names from `DirectLabelColormap` are napari-facing
    private hooks. Methods prefixed with `_compact_` are Spatiato-only helpers for
    looking up RGBA values from `CompactLabelsMapping`.

    Building: napari and vispy still expect direct-label internals shaped like
    `label_id -> texture_code` and `texture_code -> RGBA`. This subclass
    satisfies that contract with `_label_to_texture_mapping`
    (`label_id -> texture_code`) and `_texture_color_dict`
    (`texture_code -> RGBA`), avoiding the huge `label_id -> RGBA` dictionary
    used by ordinary `DirectLabelColormap` construction.

    Drawing: each new slice is recoded to texture codes on the Qt main thread.
    For high-bit labels this uses the dense table and a parallel gather instead
    of napari's per-pixel typed-dict lookup, see `_data_to_texture`.

    Dense table: a 1-D array indexed by label id that holds each label's
    texture code, for every id from 0 to the largest table label id, including
    ids without a table row (code 0). Built and cached by
    `_get_dense_texture_lut`; used only to recode high-bit arrays.

    See `docs/development/labels_coloring.md` ("The compact colormap").
    """

    def __init__(self, compact_mapping: CompactLabelsMapping) -> None:
        set_backend(ColormapBackend.numba)

        small_color_dict: dict[int | None, np.ndarray] = {
            None: compact_mapping.texture_rgba[compact_mapping.default_texture_code],
            compact_mapping.background_value: compact_mapping.texture_rgba[compact_mapping.background_texture_code],
        }
        if len(compact_mapping.label_ids):
            # Initialize `DirectLabelColormap` with one real label color:
            # passing the huge `label_id -> RGBA` mapping through
            # napari/pydantic is known to be slow for large labels layers,
            # while one real label still makes napari detect direct mode
            # instead of treating the colormap as default/background only.
            first_label_id = int(compact_mapping.label_ids[0])
            small_color_dict[first_label_id] = compact_mapping.texture_rgba[int(compact_mapping.texture_codes[0])]

        super().__init__(
            color_dict=small_color_dict,
            background_value=compact_mapping.background_value,
        )
        object.__setattr__(self, "_compact_mapping", compact_mapping)
        object.__setattr__(self, "_label_to_texture_mapping", _CompactLabelToTextureMapping(compact_mapping))
        object.__setattr__(
            self,
            "_texture_color_dict",
            dict(enumerate(compact_mapping.texture_rgba)),
        )
        # True once the dense-table fallback has been logged. Kept outside
        # `_cache_other`, which `_install_compact_mapping` clears after every
        # sparse annotation edit; otherwise each edit would log the fallback again.
        object.__setattr__(self, "_dense_lut_fallback_logged", False)

    def set_label_value(
        self,
        label_id: int,
        value: object,
        *,
        value_color: Any | None = None,
    ) -> _CompactSparseLabelUpdateResult:
        """Spatiato helper for sparse annotation: set one label's value.

        If the value was not present when the compact colormap was built,
        `value_color` is required and is appended as a new texture row. Only
        categorical mappings support this; continuous mappings raise.
        """
        label_id = self._validate_sparse_label_id(label_id)
        original_texture_count = len(self._compact_mapping.texture_rgba)
        texture_code = self._texture_code_for_value(value, value_color=value_color)
        texture_table_changed = len(self._compact_mapping.texture_rgba) != original_texture_count
        compact_mapping = _compact_mapping_with_label_texture_code(
            self._compact_mapping,
            label_id=label_id,
            texture_code=texture_code,
        )
        self._install_compact_mapping(compact_mapping)
        return _CompactSparseLabelUpdateResult(
            texture_code=texture_code,
            texture_table_changed=texture_table_changed,
        )

    def set_label_missing(self, label_id: int) -> _CompactSparseLabelUpdateResult:
        """Spatiato helper for sparse annotation: mark one label's value missing.

        The label stays explicitly mapped, to the missing-value texture code, so
        it keeps the color of known table rows without a value instead of
        falling through to the default texture code used for labels without a
        table row. The compact builders always reserve the missing-value row,
        so this never grows the texture table.
        """
        label_id = self._validate_sparse_label_id(label_id)
        # Missing values are not categorical values, so their texture code lives
        # in `missing_texture_code` instead of `value_texture_codes`.
        texture_code = int(self._compact_mapping.missing_texture_code)
        compact_mapping = _compact_mapping_with_label_texture_code(
            self._compact_mapping,
            label_id=label_id,
            texture_code=texture_code,
        )
        self._install_compact_mapping(compact_mapping)
        return _CompactSparseLabelUpdateResult(
            texture_code=texture_code,
            texture_table_changed=False,
        )

    def _texture_code_for_value(
        self,
        value: object,
        *,
        value_color: Any | None,
    ) -> int:
        """Spatiato helper for sparse annotation: resolve a value texture code.

        Sparse annotation updates receive a value/class id, but the compact
        labels colormap stores per-label colors as texture codes. This helper
        preserves the missing `value -> texture_code` relationship so Spatiato
        can update one label without rebuilding the full colormap. If the
        value is new, `value_color` is used to append one RGBA row and
        remember the new value-to-texture-code mapping for future edits.
        """
        compact = self._compact_mapping
        if compact.value_texture_codes is None:
            raise ValueError("Sparse value updates require a categorical compact labels mapping.")
        normalized_value = normalize_category_value(value)
        value_texture_codes = dict(compact.value_texture_codes)
        texture_code = value_texture_codes.get(normalized_value)
        if texture_code is not None:
            return int(texture_code)
        if value_color is None:
            raise ValueError(f"Compact labels colormap does not know value `{value}`.")

        rgba = np.asarray(to_rgba(value_color), dtype=np.float32)
        texture_code = len(compact.texture_rgba)
        value_texture_codes[normalized_value] = texture_code
        # Install the expanded compact mapping so napari-facing texture-code
        # lookup views and derived caches stay synchronized with Spatiato state.
        self._install_compact_mapping(
            replace(
                compact,
                # A brand-new value needs one new texture-code -> RGBA row;
                # existing labels/classes keep using their current rows.
                texture_rgba=np.vstack([compact.texture_rgba, rgba.reshape(1, 4)]),
                value_texture_codes=value_texture_codes,
            )
        )
        return texture_code

    def _install_compact_mapping(self, compact_mapping: CompactLabelsMapping) -> None:
        """Spatiato helper for sparse annotation: install updated compact state."""
        object.__setattr__(self, "_compact_mapping", compact_mapping)
        object.__setattr__(self, "_label_to_texture_mapping", _CompactLabelToTextureMapping(compact_mapping))
        object.__setattr__(self, "_texture_color_dict", dict(enumerate(compact_mapping.texture_rgba)))
        self._clear_cache()

    def _validate_sparse_label_id(self, label_id: int) -> int:
        """Spatiato helper for sparse annotation: validate an edited label id."""
        label_id = int(label_id)
        if label_id <= 0 or label_id == self._compact_mapping.background_value:
            raise ValueError("Compact sparse label updates require a positive non-background label id.")
        return label_id

    @property
    def _num_unique_colors(self) -> int:
        """Napari hook: report non-default/background texture rows."""
        return max(0, len(self._compact_mapping.texture_rgba) - 2)

    @property
    def _label_mapping_and_color_dict(self) -> tuple[Mapping[int | None, int], dict[int, np.ndarray]]:
        """Napari hook: return the direct-label compact mapping attribute.

        Napari reads `_label_mapping_and_color_dict` directly in scalar and
        high-bit conversion paths, so this cannot be only inlined into
        `_values_mapping_to_minimum_values_set(...)`. Keep it as a cheap view
        over compact arrays rather than the inherited cached property, which
        would rebuild mappings from `color_dict`.
        """
        return self._label_to_texture_mapping, self._texture_color_dict

    def _values_mapping_to_minimum_values_set(
        self,
        apply_selection: bool = True,
    ) -> tuple[Mapping[int | None, int], dict[int, np.ndarray]]:
        """Napari hook: return raw-label-to-texture mapping and color table.

        This mirrors `DirectLabelColormap` while keeping the label mapping
        backed by compact arrays. Napari may read the first tuple item as a
        `label_id -> texture_code` mapping, while vispy reads the second item
        as the `texture_code -> RGBA` table. Returning a mapping view here is
        what lets Spatiato avoid building a huge `label_id -> RGBA` dictionary.
        """
        if self.use_selection and apply_selection:
            # Mirror `DirectLabelColormap` selected-label rendering: only
            # `selection` maps to a visible texture code, and all other labels
            # fall through to transparent default color.
            return {self.selection: 1, None: 0}, {
                0: _TRANSPARENT_RGBA,
                1: self._compact_rgba_for_label(self.selection, apply_selection=False),
            }
        # When `use_selection` is false, return the cheap compact mapping view
        # plus the small texture-code-to-RGBA table.
        return self._label_mapping_and_color_dict

    def _data_to_texture(self, values: np.ndarray | np.integer) -> np.ndarray | np.integer:
        """Napari hook: recode a slice of label ids for the GPU data texture.

        This is the drawing step. Napari calls it on the Qt main thread for
        every new slice, so its speed decides how smooth panning and zooming
        are. With the same cached dense table, it also runs for:

        - the layer thumbnail, through `map`: napari first downsamples it to at
          most 32 x 32 pixels, so this takes well under a millisecond;
        - for multiscale data, the coarsest pyramid level, which napari
          converts with every new slice although its thumbnail only reads the
          raw label ids (https://github.com/vibspatial/spatiato/issues/136);
        - painted pixels, and an empty array to choose the view dtype.

        High-bit labels ((u)int32/64 arrays): each label id is recoded to its
        texture code with the dense table (`_get_dense_texture_lut`) and a
        parallel gather, instead of napari's per-pixel typed-dict lookup. Where
        the dense table does not apply (a negative background value, a table
        above `_DENSE_LUT_MAX_BYTES`, or a failed allocation), napari's
        typed-dict path gives the same result, more slowly.

        Low-bit labels ((u)int8/16 arrays): nothing is recoded. Napari passes
        the label ids on unchanged (a zero-copy unsigned view), and the GPU
        looks up each label's color by its label id.

        Scalars and selected-label mode also use napari's `DirectLabelColormap`
        path. See `docs/development/labels_coloring.md` ("The compact
        colormap").
        """
        if (
            not isinstance(values, np.ndarray)
            or values.dtype.kind not in "iu"
            or values.dtype.itemsize <= 2
            or self.use_selection
        ):
            return super()._data_to_texture(values)
        lut = self._get_dense_texture_lut()
        if lut is None:
            return super()._data_to_texture(values)

        flat_values = values.reshape(-1)  # copies only when not contiguous
        # `CompactLabelsMapping` enforces `default_texture_code == 0`, so zeros
        # already hold the code for labels outside the table.
        out = np.zeros(flat_values.shape, dtype=lut.dtype)
        if values.dtype.kind == "u":
            _gather_unsigned_label_texture_codes(flat_values, lut, out)
        else:
            _gather_signed_label_texture_codes(flat_values, lut, out)
        return out.reshape(values.shape)

    def _get_dense_texture_lut(self) -> np.ndarray | None:
        """Spatiato helper: return the dense table, built on first use.

        The dense table is a `label_id -> texture_code` array with one entry per
        label id from 0 to the largest table label id (or the background value),
        in the texture dtype napari expects for this colormap. Labels without a
        table row keep the default code.

        Returns `None` when napari's typed-dict path must be used instead: a
        negative background value, a table above `_DENSE_LUT_MAX_BYTES`, or a
        failed allocation. Both outcomes are cached in `_cache_other`, which
        `_install_compact_mapping(...)` clears after sparse edits, through
        napari's `_clear_cache()`.
        """
        if _DENSE_LUT_CACHE_KEY in self._cache_other:
            return self._cache_other[_DENSE_LUT_CACHE_KEY]

        compact = self._compact_mapping
        lut = None
        if compact.background_value < 0:
            self._log_dense_lut_fallback(
                "DEBUG",
                f"Labels colormap: negative background value {compact.background_value} is not supported "
                "by the dense table; using the slower per-pixel lookup.",
            )
        else:
            target_dtype = _accel_cmap.minimum_dtype_for_labels(len(compact.texture_rgba))
            max_label_id = int(compact.label_ids[-1]) if len(compact.label_ids) else 0
            size = max(max_label_id, compact.background_value) + 1
            table_bytes = size * target_dtype.itemsize
            if table_bytes > _DENSE_LUT_MAX_BYTES:
                self._log_dense_lut_fallback(
                    "INFO",
                    f"Labels colormap: largest label id {max_label_id:,} needs a "
                    f"{table_bytes / 1024**2:,.0f} MB lookup table, above the "
                    f"{_DENSE_LUT_MAX_BYTES / 1024**2:,.0f} MB cap; using the slower per-pixel lookup. "
                    "Colors are unaffected.",
                )
            else:
                try:
                    lut = _dense_texture_lut(compact, size=size, dtype=target_dtype)
                except MemoryError:
                    self._log_dense_lut_fallback(
                        "WARNING",
                        f"Labels colormap: could not allocate a {table_bytes / 1024**2:,.0f} MB lookup table "
                        f"for label ids up to {max_label_id:,}; using the slower per-pixel lookup. "
                        "Colors are unaffected.",
                    )

        self._cache_other[_DENSE_LUT_CACHE_KEY] = lut
        return lut

    def _log_dense_lut_fallback(self, level: str, message: str) -> None:
        """Spatiato helper: log the dense-table fallback once per colormap."""
        if self._dense_lut_fallback_logged:
            return
        object.__setattr__(self, "_dense_lut_fallback_logged", True)
        logger.log(level, message)

    def _get_typed_dict_mapping(self, data_dtype: np.dtype) -> typed.Dict:
        """Napari hook: return the `label_id -> texture_code` typed dict for one data dtype.

        Napari's numba backend looks up every pixel of a high-bit array in this
        dict; a label id that is not in it gets the default texture code 0. Here
        it is only used when `_data_to_texture(...)` cannot use the dense table
        (`_get_dense_texture_lut`), for slices and, through `map(...)`, for
        thumbnails: a negative background value, a table above
        `_DENSE_LUT_MAX_BYTES`, or a failed allocation. Low-bit data never uses
        it: napari passes those label ids on unchanged, and their colors come
        from napari's low-bit `label_id -> RGBA` lookup (`_map_without_cache`).

        Napari's `DirectLabelColormap` builds this dict from its per-label colors
        in a Python loop. This override fills it from the compact arrays in one
        numba call (`_fill_typed_label_texture_mapping`): the background first,
        then one entry per table label id that `data_dtype` can represent.
        Labels without a table row are left out, so they get the default code.
        The dict is cached per data dtype in `_cache_other`.
        """
        data_dtype = np.dtype(data_dtype)
        cache_key = f"_compact_{data_dtype.name}_typed_dict"
        if cache_key in self._cache_other:
            return self._cache_other[cache_key]

        compact = self._compact_mapping
        target_dtype = _accel_cmap.minimum_dtype_for_labels(len(compact.texture_rgba))
        iinfo = np.iinfo(data_dtype)
        representable = (compact.label_ids >= iinfo.min) & (compact.label_ids <= iinfo.max)
        label_ids = compact.label_ids[representable].astype(data_dtype, copy=False)
        texture_codes = compact.texture_codes[representable].astype(target_dtype, copy=False)

        typed_mapping = typed.Dict.empty(
            key_type=getattr(types, data_dtype.name),
            value_type=getattr(types, target_dtype.name),
        )
        typed_mapping = _fill_typed_label_texture_mapping(
            typed_mapping,
            label_ids,
            texture_codes,
            data_dtype.type(compact.background_value),
            target_dtype.type(compact.background_texture_code),
        )

        self._cache_other[cache_key] = typed_mapping
        return typed_mapping

    def map(self, values: np.ndarray | np.integer | int) -> np.ndarray:
        """Napari hook: map scalar or array label ids to RGBA colors.

        Napari uses this for thumbnails and single-label colors, not for
        rendering slices (see `_data_to_texture`). The thumbnail is colored on
        every layer refresh, including the refresh after each annotation edit.
        Napari downsamples it to at most 32 x 32 pixels first, so recoding it is
        cheap; the dense table is shared with the slices.

        - Scalars resolve through the compact arrays.
        - Low-bit arrays use napari's low-bit `label_id -> RGBA` lookup.
        - High-bit arrays are recoded by `_data_to_texture`, the same recoding
          as for slices, so they reuse the dense table (`_get_dense_texture_lut`);
          napari's typed dict is only built where the dense table cannot be
          used. The inherited `_map_precast` then turns the texture codes into
          RGBA through `_values_mapping_to_minimum_values_set(...)[1]`.
        """
        if isinstance(values, np.integer):
            values = int(values)
        if isinstance(values, int):
            return self._compact_rgba_for_label(values, apply_selection=True)
        if isinstance(values, list | tuple):
            values = np.asarray(values)
        if not isinstance(values, np.ndarray) or values.dtype.kind in "fU":
            raise TypeError("DirectLabelColormap can only be used with int")

        if values.dtype.itemsize <= 2:
            # Low-bit labels use napari's low-bit `label_id -> RGBA` lookup.
            mapper = self._get_mapping_from_cache(values.dtype)
            mapped = mapper[values]
        else:
            texture_codes = self._data_to_texture(values)
            mapped = self._map_precast(texture_codes, apply_selection=True)

        if self.use_selection:
            mapped[(values != self.selection)] = 0
        return mapped

    def _map_without_cache(self, values: np.ndarray) -> np.ndarray:
        """Napari hook: build small-label lookup textures without selection."""
        return self._compact_rgba_for_values(values, apply_selection=False)

    @property
    def _array_map(self) -> np.ndarray:
        """Napari fallback hook that should not be used by this colormap."""
        raise RuntimeError("Compact label colormaps require napari's numba colormap backend.")

    def _clear_cache(self) -> None:
        """Napari hook: clear derived lookup caches.

        Napari's base class resets `_cache_other` (dense table, typed dicts)
        and `_cache_mapping` (low-bit RGBA lookups).
        """
        super()._clear_cache()

    def _compact_rgba_for_label(self, label_id: int, *, apply_selection: bool) -> np.ndarray:
        """Spatiato helper: resolve one label id through compact texture codes."""
        if apply_selection and self.use_selection and label_id != self.selection:
            return _TRANSPARENT_RGBA
        compact = self._compact_mapping
        texture_code = compact.default_texture_code
        if label_id == compact.background_value:
            texture_code = compact.background_texture_code
        else:
            position = np.searchsorted(compact.label_ids, label_id)
            if position < len(compact.label_ids) and compact.label_ids[position] == label_id:
                texture_code = int(compact.texture_codes[position])
        return compact.texture_rgba[texture_code]

    def _compact_rgba_for_values(self, values: np.ndarray, *, apply_selection: bool) -> np.ndarray:
        """Spatiato helper: resolve an array of label ids through compact state."""
        compact = self._compact_mapping
        flat_values = values.ravel()
        texture_codes = np.full(flat_values.shape, compact.default_texture_code, dtype=np.int64)

        background = flat_values == compact.background_value
        texture_codes[background] = compact.background_texture_code

        positions = np.searchsorted(compact.label_ids, flat_values)
        in_bounds = positions < len(compact.label_ids)
        matched = np.zeros(flat_values.shape, dtype=bool)
        matched[in_bounds] = compact.label_ids[positions[in_bounds]] == flat_values[in_bounds]
        if np.any(matched):
            texture_codes[matched] = compact.texture_codes[positions[matched]]

        mapped = compact.texture_rgba[texture_codes].reshape(values.shape + (4,))
        if apply_selection and self.use_selection:
            mapped = mapped.copy()
            mapped[values != self.selection] = _TRANSPARENT_RGBA
        return mapped


def _dense_texture_lut(compact: CompactLabelsMapping, *, size: int, dtype: np.dtype) -> np.ndarray:
    """Build the dense table for compact state: the texture code of every label id below `size`."""
    # `CompactLabelsMapping` enforces `default_texture_code == 0`, so a zeroed
    # table already maps every label without a table row to the default code.
    lut = np.zeros(size, dtype=dtype)
    lut[compact.label_ids] = compact.texture_codes
    lut[compact.background_value] = compact.background_texture_code
    return lut


def _compact_mapping_with_label_texture_code(
    compact: CompactLabelsMapping,
    *,
    label_id: int,
    texture_code: int,
) -> CompactLabelsMapping:
    """Spatiato helper for sparse annotation: set one label texture code.

    This intentionally returns a replaced `CompactLabelsMapping`
    instead of mutating the existing arrays in place, so the caller can install
    one coherent compact state and clear all napari-derived caches together.
    We benchmarked this full-replace path on large synthetic mappings; the
    array copy/insert cost was small enough for row-scoped annotation compared
    with napari refresh/rendering work.
    """
    if not 0 <= int(texture_code) < len(compact.texture_rgba):
        raise ValueError("Compact sparse label update received an unknown texture code.")

    label_ids = compact.label_ids
    texture_codes = compact.texture_codes
    texture_code = int(texture_code)
    target_dtype = _minimum_unsigned_dtype(max(int(np.max(texture_codes, initial=0)), texture_code))
    position = int(np.searchsorted(label_ids, label_id))
    if position < len(label_ids) and int(label_ids[position]) == label_id:
        # Existing explicit label: update only its texture code.
        updated_texture_codes = texture_codes.astype(target_dtype, copy=True)
        updated_texture_codes[position] = texture_code
        return replace(
            compact,
            texture_codes=_minimum_texture_code_dtype(updated_texture_codes),
        )

    updated_label_ids = np.insert(label_ids, position, label_id).astype(np.int64, copy=False)
    updated_texture_codes = np.insert(texture_codes.astype(target_dtype, copy=False), position, texture_code)
    # Previously unmapped/default label: keep `label_ids` sorted and insert
    # the texture code at the same index, because `texture_codes[i]` belongs
    # to `label_ids[i]`.
    return replace(
        compact,
        label_ids=updated_label_ids,
        texture_codes=_minimum_texture_code_dtype(updated_texture_codes),
    )


def _minimum_texture_code_dtype(texture_codes: np.ndarray) -> np.ndarray:
    max_value = int(np.max(texture_codes, initial=0))
    return texture_codes.astype(_minimum_unsigned_dtype(max_value), copy=False)


def compact_categorical_labels_mapping_from_values(
    values: pd.Series,
    *,
    categories: Sequence[object],
    palette: Sequence[Any],
    default_color: Any = _TRANSPARENT_RGBA,
    missing_color: Any = MISSING_CATEGORICAL_COLOR,
    background_value: int = 0,
) -> CompactLabelsMapping:
    """Return compact labels coloring state for categorical table values.

    Parameters
    ----------
    values
        Table-aligned categorical column for one labels element. The index
        contains positive label ids / instance ids, and each row value is the
        categorical value used to color that label. This is not the labels image
        data itself.
    categories
        Ordered categorical values that correspond to `palette`.
    palette
        Color values for `categories`.
    default_color
        Color for labels that are not present in `values`, i.e. labels without
        a table row. The default is transparent, matching styled-labels
        behavior. Callers without a bound table can pass a neutral color while
        keeping the background label transparent.
    missing_color
        Color for known table rows with missing or palette-unknown categorical
        values.
    background_value
        Background label id.

    Known table rows with missing or palette-unknown categorical values receive
    the missing categorical color. Label ids not present in `values` are not
    stored here and later fall through to `default_color`.
    """
    if not isinstance(background_value, int) or isinstance(background_value, bool):
        raise ValueError("Compact labels mapping background value must be an integer label id.")

    label_ids, label_ids_sorted = _positive_label_ids_from_index(values.index)
    normalized_categories = [normalize_category_value(category) for category in categories]

    default_rgba = np.asarray(to_rgba(default_color), dtype=np.float32)
    texture_rgba_rows: list[np.ndarray] = [default_rgba, _TRANSPARENT_RGBA.copy()]
    color_to_texture_code: dict[tuple[float, float, float, float], int] = {
        _rgba_key(texture_rgba_rows[1]): 1,
    }

    def texture_code_for_rgba(color: Any) -> int:
        rgba = np.asarray(to_rgba(color), dtype=np.float32)
        key = _rgba_key(rgba)
        texture_code = color_to_texture_code.get(key)
        if texture_code is not None:
            return texture_code
        texture_code = len(texture_rgba_rows)
        color_to_texture_code[key] = texture_code
        texture_rgba_rows.append(rgba)
        return texture_code

    category_texture_code_by_value = {
        category: texture_code_for_rgba(color) for category, color in zip(normalized_categories, palette, strict=False)
    }

    # Reserve the missing-value row even when no row is missing yet, so a
    # sparse edit that clears a value never needs to grow the texture table.
    # It comes after the category rows to keep category texture codes stable.
    missing_texture_code = texture_code_for_rgba(missing_color)

    texture_codes = _categorical_texture_codes(
        values,
        category_texture_code_by_value=category_texture_code_by_value,
    )
    texture_codes[texture_codes < 0] = missing_texture_code
    if not label_ids_sorted:
        order = np.argsort(label_ids)
        label_ids = label_ids[order]
        texture_codes = texture_codes[order]

    texture_codes = texture_codes.astype(_minimum_unsigned_dtype(int(np.max(texture_codes, initial=0))), copy=False)
    texture_rgba = np.asarray(texture_rgba_rows, dtype=np.float32)

    return CompactLabelsMapping(
        label_ids=label_ids,
        texture_codes=texture_codes,
        texture_rgba=texture_rgba,
        value_texture_codes=dict(category_texture_code_by_value),
        default_texture_code=0,
        background_texture_code=1,
        background_value=background_value,
        missing_texture_code=missing_texture_code,
    )


def compact_categorical_label_colormap_from_values(
    values: pd.Series,
    *,
    categories: Sequence[object],
    palette: Sequence[Any],
    default_color: Any = _TRANSPARENT_RGBA,
    missing_color: Any = MISSING_CATEGORICAL_COLOR,
    background_value: int = 0,
) -> CompactLabelColormap:
    """Return a compact direct labels colormap for categorical values."""
    mapping = compact_categorical_labels_mapping_from_values(
        values,
        categories=categories,
        palette=palette,
        default_color=default_color,
        missing_color=missing_color,
        background_value=background_value,
    )
    return CompactLabelColormap(mapping)


def compact_continuous_labels_mapping_from_values(
    values: pd.Series,
    *,
    bins: int | None = None,
    colormap_name: str = OVERLAY_CONTINUOUS_COLORMAP,
    missing_color: Any = MISSING_CONTINUOUS_COLOR,
    default_color: Any = _TRANSPARENT_RGBA,
    background_value: int = 0,
    value_range: tuple[float, float] | None = None,
    zero_color: Any | None = None,
) -> CompactLabelsMapping:
    """Return compact labels coloring state for continuous table values.

    Parameters
    ----------
    values
        Table-aligned continuous column for one labels element. The index
        contains positive label ids / instance ids, and each row value is the
        continuous value used to color that label. This is not the labels image
        data itself.
    bins
        Number of color bins sampled from `colormap_name`. Finite source values
        map to these bins, while missing/non-finite values map to
        `missing_color`. The default, 252 or 251 with `zero_color`, keeps the
        texture codes, and so the per-slice labels texture, in uint8; more bins
        still work with uint16 textures.
    colormap_name
        Matplotlib colormap used for finite values.
    missing_color
        Color for known table rows with missing or non-finite values.
    default_color
        Color for labels that are not present in `values`.
    background_value
        Background label id.
    value_range
        Optional `(min, max)` range used for normalization. If omitted, the
        range is derived from finite values. Values outside the range are
        clamped before binning.
    zero_color
        Optional color for values that are exactly 0, in their own texture
        row. `None` colors them like any other value. Zeros still count for the
        value range either way, so the other colors do not shift. Var styling
        passes transparent when zeros mean "not detected".

    Source values such as `0.0` are table values attached to positive label
    ids. `background_value=0` is reserved for the labels image background.
    Constant finite values map to the midpoint bin, matching the current
    continuous RGBA helper's `0.5` normalization behavior.
    """
    if not isinstance(background_value, int) or isinstance(background_value, bool):
        raise ValueError("Compact labels mapping background value must be an integer label id.")
    if bins is None:
        bins = _CONTINUOUS_LABEL_BINS if zero_color is None else _CONTINUOUS_LABEL_BINS - 1
    if not isinstance(bins, int) or isinstance(bins, bool) or bins < 2:
        raise ValueError("Compact continuous labels mapping `bins` must be an integer >= 2.")
    if value_range is not None:
        min_value, max_value = value_range
        min_value = float(min_value)
        max_value = float(max_value)
        if not np.isfinite(min_value) or not np.isfinite(max_value):
            raise ValueError("Compact continuous labels mapping `value_range` must contain finite values.")
        if max_value <= min_value:
            raise ValueError("Compact continuous labels mapping `value_range` must satisfy min < max.")
        value_range = (min_value, max_value)

    label_ids, label_ids_sorted = _positive_label_ids_from_index(values.index)
    value_array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64, copy=False)
    finite_values = np.isfinite(value_array)

    # Cached as a small allocation polish; the hot path is label-id validation
    # and per-label texture-code construction, not this small binned RGBA table.
    texture_rgba = _continuous_texture_rgba(
        bins=bins,
        colormap_name=colormap_name,
        default_rgba=_rgba_cache_key(default_color),
        missing_rgba=_rgba_cache_key(missing_color),
        zero_rgba=None if zero_color is None else _rgba_cache_key(zero_color),
    )
    missing_texture_code = bins + 2
    zero_texture_code = bins + 3
    texture_codes = np.full(len(value_array), missing_texture_code, dtype=np.int64)

    if np.any(finite_values):
        finite_numeric_values = value_array[finite_values]
        if value_range is None:
            min_value = float(np.min(finite_numeric_values))
            max_value = float(np.max(finite_numeric_values))
        else:
            min_value, max_value = value_range

        if max_value == min_value:
            normalized_values = np.full(len(finite_numeric_values), 0.5, dtype=np.float64)
        else:
            normalized_values = np.clip(
                (finite_numeric_values - min_value) / (max_value - min_value),
                0.0,
                1.0,
            )
        color_bins = np.floor(normalized_values * bins).astype(np.int64)
        color_bins = np.clip(color_bins, 0, bins - 1)
        texture_codes[finite_values] = color_bins + 2
        if zero_color is not None:
            texture_codes[finite_values & (value_array == 0.0)] = zero_texture_code

    if not label_ids_sorted:
        order = np.argsort(label_ids)
        label_ids = label_ids[order]
        texture_codes = texture_codes[order]
    return CompactLabelsMapping(
        label_ids=label_ids,
        texture_codes=_minimum_texture_code_dtype(texture_codes),
        texture_rgba=texture_rgba,
        default_texture_code=0,
        background_texture_code=1,
        background_value=background_value,
        missing_texture_code=missing_texture_code,
    )


def compact_continuous_label_colormap_from_values(
    values: pd.Series,
    *,
    bins: int | None = None,
    colormap_name: str = OVERLAY_CONTINUOUS_COLORMAP,
    missing_color: Any = MISSING_CONTINUOUS_COLOR,
    default_color: Any = _TRANSPARENT_RGBA,
    background_value: int = 0,
    value_range: tuple[float, float] | None = None,
    zero_color: Any | None = None,
) -> CompactLabelColormap:
    """Return a compact direct labels colormap for continuous values."""
    mapping = compact_continuous_labels_mapping_from_values(
        values,
        bins=bins,
        colormap_name=colormap_name,
        missing_color=missing_color,
        default_color=default_color,
        background_value=background_value,
        value_range=value_range,
        zero_color=zero_color,
    )
    return CompactLabelColormap(mapping)


class NeutralLabelColormap(DirectLabelColormap):
    """Direct labels colormap that draws every foreground label in one color.

    Its color dict holds only the default (`None`) color and the transparent
    background. Napari maps labels without an explicit color to texture code 0
    and the background to texture code 1, so recoding a slice is one
    comparison with the background value instead of napari's per-pixel
    typed-dict lookup.

    Build it with `neutral_label_colormap(...)`. It keeps napari's constructor,
    because napari can rebuild a colormap with
    `self.__class__(**self.model_dump())` (`_cmap_without_selection`).

    It intentionally does not subclass `CompactLabelColormap`.
    `isinstance(colormap, CompactLabelColormap)` is how Object Classification
    recognizes a table-styled layer whose colors it may edit one label at a
    time (`set_label_value`); a neutral layer has no table values to edit, so
    it must fail that check. `CompactLabelColormap`'s constructor and hooks also
    expect a `CompactLabelsMapping`, which a neutral colormap does not have.
    """

    def _data_to_texture(self, values: np.ndarray | np.integer) -> np.ndarray | np.integer:
        """Napari hook: recode a slice of label ids for the GPU data texture.

        This is the drawing step, called for every new slice (see
        `CompactLabelColormap._data_to_texture`). For a high-bit array it
        returns `values == background_value` as uint8: the background gets
        code 1 and every other label code 0, the default code that draws the
        neutral color. That is correct only when napari codes this colormap
        exactly as `{None: 0, background: 1}`.

        Everything else uses napari's `DirectLabelColormap` path:

        - scalars and non-integer input, which the array shortcut does not
          cover;
        - low-bit labels ((u)int8/16), which are not recoded at all: napari
          passes the label ids on unchanged and the GPU looks up each label's
          color by its label id;
        - selected-label mode, where napari codes the selected label as 1 and
          every other label as 0, so the shortcut would draw the background in
          the selected color;
        - a color dict with explicit label colors, for example added by other
          code, which gives those labels their own codes
          (`_has_neutral_texture_codes`).
        """
        if (
            not isinstance(values, np.ndarray)
            or values.dtype.kind not in "iu"
            or values.dtype.itemsize <= 2
            or self.use_selection
            or not self._has_neutral_texture_codes()
        ):
            return super()._data_to_texture(values)
        # Background -> 1, every other label -> 0 (the default code).
        return (values == self.background_value).view(np.uint8)

    def _has_neutral_texture_codes(self) -> bool:
        """Spatiato helper: check napari's `None -> 0`, background `-> 1` coding."""
        label_mapping = self._label_mapping_and_color_dict[0]
        return (
            len(label_mapping) == 2 and label_mapping.get(None) == 0 and label_mapping.get(self.background_value) == 1
        )


def neutral_label_colormap(color: Any, *, background_value: int = 0) -> NeutralLabelColormap:
    """Return a labels colormap that draws every foreground label in `color`.

    The background label stays transparent.
    """
    return NeutralLabelColormap(
        color_dict={None: color, background_value: "transparent"},
        background_value=background_value,
    )


def direct_label_colormap_from_rgba(
    color_dict: dict[int | None, np.ndarray],
    *,
    background_value: int = 0,
) -> DirectLabelColormap:
    """Construct a direct labels colormap from Spatiato-generated RGBA arrays.

    Napari's public ``DirectLabelColormap(color_dict=...)`` constructor becomes
    expensive for large labels layers because it validates and color-normalizes
    every ``label_id -> RGBA`` entry. Spatiato's styled-labels paths already build
    numeric RGBA arrays, so the large constructor pass is redundant.

    To avoid that bottleneck, construct a tiny normal ``DirectLabelColormap``
    with only the default/background colors so napari/pydantic still initializes
    the model and event emitters correctly. Then install the trusted full RGBA
    mapping directly and clear napari's derived colormap caches.

    This is an internal fast path for RGBA dictionaries generated by Spatiato
    itself. It performs only cheap structural checks on the default/background
    entries and does not revalidate every per-label color in the mapping. The
    mapping container and per-label arrays are not copied; callers should treat
    the input mapping and RGBA arrays as immutable after construction.
    """
    if not isinstance(background_value, int) or isinstance(background_value, bool):
        raise ValueError("Labels colormap background value must be an integer label id.")
    if None not in color_dict:
        raise ValueError("Direct labels RGBA color dictionary must include a `None` default color.")
    if background_value not in color_dict:
        raise ValueError(f"Direct labels RGBA color dictionary must include background label `{background_value}`.")

    _validate_default_color(color_dict[None], label_id=None)
    _validate_default_color(color_dict[background_value], label_id=background_value)
    small_color_dict = {
        None: color_dict[None],
        background_value: color_dict[background_value],
    }
    colormap = DirectLabelColormap(color_dict=small_color_dict, background_value=background_value)
    object.__setattr__(colormap, "color_dict", color_dict)
    colormap._clear_cache()
    return colormap


@lru_cache(maxsize=32)
def _continuous_texture_rgba(
    bins: int,
    colormap_name: str,
    default_rgba: tuple[float, float, float, float],
    missing_rgba: tuple[float, float, float, float],
    zero_rgba: tuple[float, float, float, float] | None = None,
) -> np.ndarray:
    finite_colors = colormaps[colormap_name](np.linspace(0.0, 1.0, bins))
    rows = [
        np.asarray(default_rgba, dtype=np.float32).reshape(1, 4),
        _TRANSPARENT_RGBA.reshape(1, 4),
        np.asarray(finite_colors, dtype=np.float32),
        np.asarray(missing_rgba, dtype=np.float32).reshape(1, 4),
    ]
    if zero_rgba is not None:
        rows.append(np.asarray(zero_rgba, dtype=np.float32).reshape(1, 4))
    texture_rgba = np.vstack(rows)
    texture_rgba.setflags(write=False)
    return texture_rgba


def _rgba_cache_key(color: Any) -> tuple[float, float, float, float]:
    return tuple(float(channel) for channel in to_rgba(color))


def _validate_default_color(color: np.ndarray, *, label_id: int | None) -> None:
    if not isinstance(color, np.ndarray):
        raise ValueError(f"Direct labels RGBA default/background color for label `{label_id}` must be a numpy array.")
    if color.shape != (4,):
        raise ValueError(f"Direct labels RGBA default/background color for label `{label_id}` must have shape `(4,)`.")


def _positive_label_ids_from_index(index: pd.Index) -> tuple[np.ndarray, bool]:
    raw_label_ids = index.to_numpy()
    try:
        label_ids = raw_label_ids.astype(np.int64, copy=False)
    except (TypeError, ValueError) as error:
        raise ValueError("Compact labels mapping index must contain integer label ids.") from error
    if not np.array_equal(raw_label_ids, label_ids):
        raise ValueError("Compact labels mapping index must contain integer label ids.")
    if len(label_ids) == 0:
        return label_ids, True
    # Fast path for the common viewer-aligned case: strictly increasing positive
    # labels prove positivity, uniqueness, and sorted order without `np.unique`.
    if label_ids[0] > 0 and np.all(label_ids[1:] > label_ids[:-1]):
        return label_ids, True
    # Fallback for unsorted or malformed public inputs; this remains the main
    # validation cost for large tables that do not satisfy the sorted fast path.
    if len(label_ids) != len(np.unique(label_ids)):
        raise ValueError("Compact labels mapping index must contain unique label ids.")
    if np.any(label_ids <= 0):
        raise ValueError("Compact labels mapping index must contain positive label ids.")
    return label_ids, False


def _categorical_texture_codes(
    values: pd.Series,
    *,
    category_texture_code_by_value: dict[object, int],
) -> np.ndarray:
    """Return one texture code per row, using `-1` for missing/unknown values.

    Pandas categorical values use the fast `.cat.codes` path and map category
    codes through the palette texture-code table. Other dtypes are normalized
    value-by-value so NumPy scalar variants and missing values match Spatiato's
    existing categorical-coloring semantics.
    """
    if len(values) == 0:
        return np.asarray([], dtype=np.int64)

    if isinstance(values.dtype, pd.CategoricalDtype):
        value_categories = [normalize_category_value(category) for category in values.cat.categories]
        code_by_value_code = np.asarray(
            [category_texture_code_by_value.get(category, -1) for category in value_categories],
            dtype=np.int64,
        )
        value_codes = values.cat.codes.to_numpy(copy=False)
        texture_codes = np.full(len(values), -1, dtype=np.int64)
        present = value_codes >= 0
        if np.any(present):
            texture_codes[present] = code_by_value_code[value_codes[present]]
        return texture_codes

    normalized_values = pd.Series(
        [pd.NA if pd.isna(value) else normalize_category_value(value) for value in values],
        index=values.index,
        dtype="object",
    )
    mapped = normalized_values.map(category_texture_code_by_value)
    known = mapped.notna().to_numpy(dtype=bool, copy=False)
    texture_codes = np.empty(len(values), dtype=np.int64)
    if np.any(known):
        texture_codes[known] = mapped.loc[known].to_numpy(dtype=np.int64, copy=False)
    if np.any(~known):
        texture_codes[~known] = -1
    return texture_codes


def _rgba_key(rgba: np.ndarray) -> tuple[float, float, float, float]:
    return tuple(float(channel) for channel in rgba)


def _minimum_unsigned_dtype(max_value: int) -> np.dtype:
    if max_value <= np.iinfo(np.uint8).max:
        return np.dtype(np.uint8)
    if max_value <= np.iinfo(np.uint16).max:
        return np.dtype(np.uint16)
    if max_value <= np.iinfo(np.uint32).max:
        return np.dtype(np.uint32)
    return np.dtype(np.uint64)
