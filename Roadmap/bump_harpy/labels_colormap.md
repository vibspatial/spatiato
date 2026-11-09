# Labels Colormap: Fast Per-Slice Texture Recoding

Status: implemented on 2026-11-09 on `feature/lazy_feature_matrix`. Tracks
https://github.com/vibspatial/spatiato/issues/130. The fix below was checked
against the current code with a scratch prototype (results under "Prototype
Results"). Decisions are listed near the end. "Follow-up: Contour Mode" at the
end covers a related slowdown with `contour > 0`
(https://github.com/vibspatial/spatiato/issues/136); its fix is decided but
not implemented yet.

## Problem

Panning or zooming a Labels layer styled from a table column is less smooth
than the same element loaded as a primary layer. The neutral style has the same
problem. The cost is not building the colormap. It is converting every new
slice to texture codes on the Qt main thread:

```text
QtViewer._on_slice_ready                      (@ensure_main_thread)
  -> ScalarFieldSlicingState._update_slice_response
    -> Labels._raw_to_displayed
      -> colormap._data_to_texture(raw_slice)
```

- Primary layers use napari's `CyclicLabelColormap`: a parallel numba modulo
  per pixel.
- Styled layers use `CompactLabelColormap` (`viewer/labels_colormap.py`), which
  does not override `_data_to_texture`. For int32 labels, napari's
  `DirectLabelColormap` path ends in `_labels_raw_to_texture_direct_loop`
  (`napari/utils/colormaps/_accelerated_cmap.py`), which does two numba
  typed-dict lookups per pixel (`val in dkt`, then `dkt[val]`).
- The dict size barely matters. `apply_neutral_labels_style`
  (`viewer/labels_styling.py`) uses a plain two-entry `DirectLabelColormap`,
  so a one-key typed dict, and costs about the same per slice. The per-pixel
  hash lookup is the bottleneck.

Real CosMx data from the issue (22,560 × 51,136 int32 labels, 15,929 table
rows, max label id 75,115): 12–14 ms per 5.2 Mpx slice styled, against
0.5–0.7 ms for the primary layer; 33–49 ms against 1.2–1.8 ms at 15–21 Mpx.

## napari Versions

`pyproject.toml` requires `napari[all]>=0.7.1`; `.venv` has napari 0.7.1, and
the issue was measured on 0.9.1. The files on this path
(`utils/colormaps/colormap.py`, `utils/colormaps/_accelerated_cmap.py`,
`_vispy/layers/labels.py`) were diffed between the two. The hot path is the
same. The one relevant difference: in 0.9.1, `_cache_mapping` and
`_cache_other` are a `_RebuildableCache` that is emptied on deepcopy. Caching a
NumPy array in `_cache_other` is safe in both versions.

## What napari Reads From the Colormap

Every place napari turns high-bit labels into texture values goes through
`colormap._data_to_texture`, except thumbnails:

| Caller | What it does |
|---|---|
| `Labels._raw_to_displayed` | each new slice (the hot path) |
| `Labels._slice_dtype` | `_data_to_texture(np.zeros(0, dtype)).dtype` decides the view dtype |
| `Labels.data_setitem` | paint/fill: recodes only the changed pixels |
| `DirectLabelColormap._selection_as_minimum_dtype` | scalar, calls `_cast_labels_data_to_texture_dtype_direct` directly |
| `Labels._update_thumbnail` → `colormap.map(...)` | runs on a thumbnail-sized array, after `ndi.zoom` |

Vispy builds the color texture from
`_values_mapping_to_minimum_values_set()[1]` and derives the shader scale from
`_texture_dtype(layer._direct_colormap._num_unique_colors + 2, raw_dtype)`.
For `CompactLabelColormap`, `_num_unique_colors` is `len(texture_rgba) - 2`, so
the texture dtype must stay
`minimum_dtype_for_labels(len(texture_rgba))`: uint8 up to 255 rows, uint16 up
to 65,535 rows, float32 above. Continuous mappings have 259 rows today
(256 bins plus 3 reserved rows), so they use uint16; section 7 brings them to
255 rows and uint8.

## Proposed Fix

All changes are in `viewer/labels_colormap.py`, plus a small change in
`viewer/labels_styling.py` for the neutral style.

### 1. Dense lookup table

Derive a dense `label_id -> texture_code` array from `CompactLabelsMapping`:

- size `max(label_ids[-1], background_value) + 1` (`label_ids` is sorted and
  positive; sparse inserts keep it sorted);
- filled with `default_texture_code` (always 0, enforced by
  `CompactLabelsMapping.__post_init__`);
- `lut[label_ids] = texture_codes`, then
  `lut[background_value] = background_texture_code`;
- dtype `minimum_dtype_for_labels(len(texture_rgba))`, the same as
  `_get_typed_dict_mapping` uses today.

For the CosMx dataset this is 75 KB. The size follows the largest table label
id, not the number of rows or the data.

Store it in `self._cache_other` under one key (it does not depend on the data
dtype). Store `None` when the fallback applies, so the decision is not
recomputed per slice.

### 2. Override `_data_to_texture`

```python
def _data_to_texture(self, values):
    if not isinstance(values, np.ndarray) or values.dtype.itemsize <= 2 or self.use_selection:
        # Scalars, (u)int8/16 views and selected-label mode keep napari's path.
        return super()._data_to_texture(values)
    lut = self._get_dense_texture_lut()
    if lut is None:
        # Large or negative label ids: keep the typed-dict path.
        return super()._data_to_texture(values)
    flat_values = values.reshape(-1)  # copies only when not contiguous
    # Zeros are the default texture code for unmapped labels.
    out = np.zeros(flat_values.shape, dtype=lut.dtype)
    if values.dtype.kind == "u":
        _gather_unsigned_label_texture_codes(flat_values, lut, out)
    else:
        _gather_signed_label_texture_codes(flat_values, lut, out)
    return out.reshape(values.shape)
```

- `use_selection` keeps napari's `(data == selection).astype(np.uint8)`.
- Scalars keep the `_CompactLabelToTextureMapping.get(...)` lookup.
- (u)int8/16 data keeps napari's zero-copy unsigned view.
- Empty arrays (from `_slice_dtype`) return an empty array of the LUT dtype.

### 3. Gather kernels

Two parallel numba kernels, so numba never compares int64 with uint64:

```python
@njit(parallel=True, cache=True)
def _gather_signed_label_texture_codes(flat_values, lut, out):
    n = lut.shape[0]
    for i in prange(flat_values.size):
        value = flat_values[i]
        if value >= 0 and value < n:
            out[i] = lut[value]
    return out


@njit(parallel=True, cache=True)
def _gather_unsigned_label_texture_codes(flat_values, lut, out):
    n = np.uint64(lut.shape[0])
    for i in prange(flat_values.size):
        value = np.uint64(flat_values[i])
        if value < n:
            out[i] = lut[value]
    return out
```

Values outside the table (negative, larger than the largest table id) keep the
default code 0, which is what the typed dict does for unknown keys.

### 4. Cache invalidation

No new code is needed. `set_label_value` / `set_label_missing` /
`_texture_code_for_value` all end in `_install_compact_mapping`, which calls
`_clear_cache()`. napari's `LabelColormapBase._clear_cache` replaces
`_cache_other`, so the table is rebuilt lazily on the next slice. This also
covers a new value that grows the texture table past 255 rows: the rebuilt
table becomes uint16, and `object_classification/viewer_styling.py` already
emits `layer.events.colormap()` when `texture_table_changed`, so vispy updates
its scale.

Rebuild cost is O(table size) and runs on the main thread after every sparse
edit. Measured here: negligible at 75 KB, 1.7 ms at 32 MB, 4.9 ms at 128 MB,
6.9 ms at 256 MB. Patching the table in place on sparse edits is possible
later, but not needed.

### 5. Fallback

Return `None` from `_get_dense_texture_lut` (use the typed-dict path as today)
when:

- `background_value < 0`;
- the table would exceed `_DENSE_LUT_MAX_BYTES`
  (`size * target_dtype.itemsize > _DENSE_LUT_MAX_BYTES`, 128 MB, see
  "Decisions"); or
- creating the table raises `MemoryError`. Cache `None` in that case too, so a
  failing allocation is not retried on every slice; the next `_clear_cache()`
  (a sparse edit or a restyle) tries again.

The Spatiato builders always pass `background_value=0`, so in practice only
very large or hashed label ids reach the fallback.

Log the fallback with the package's loguru `logger`, at a level that depends on
the reason:

| Reason | Level | Why |
|---|---|---|
| Table would exceed `_DENSE_LUT_MAX_BYTES` | `info` | An expected, documented limit: colors are correct, only panning is slower. |
| `MemoryError` while creating the table | `warning` | Unexpected: the machine is short on memory. |
| `background_value < 0` | `debug` | Spatiato's builders never do this, so it can only come from direct API use. |

This follows `viewer/shapes_styling.py`, which logs `info` when it falls back
to the default palette and `warning` for real problems.

- Log once per colormap instance. The decision is cached, so it is never
  logged per slice, but every sparse edit clears `_cache_other`, so a layer on
  the fallback would log again after each annotation click. Keep a "fallback
  logged" flag on the instance (set with `object.__setattr__`, like
  `_compact_mapping`), not in `_cache_other`. A restyle creates a new colormap
  and may log once more.
- Use `logger`, not `warnings.warn`: napari turns Python warnings into
  notification pop-ups, which is too loud for a speed note the user cannot act
  on.
- The message gives the largest label id, the table size it would need and
  the cap, for example: "Labels colormap: largest label id 210,000,000 needs a
  210 MB lookup table, above the 128 MB cap; using the slower per-pixel
  lookup. Colors are unaffected." The colormap does not know its layer, so the
  message cannot name it.

### 6. Neutral style

The neutral style is the primary labels layer drawn in one neutral color
(`DEFAULT_NEUTRAL_COLOR`) when there are no table values to show:

- spatial annotation before an annotation column exists
  (`load_and_style_unannotated_spatial_annotation_labels` in
  `widgets/spatial_query/viewer_styling.py`);
- object classification when the table binding is rejected
  (`_refresh_layer_styling` in `widgets/object_classification/widget.py`).

It is not napari's default `CyclicLabelColormap` (a random color per label),
which is already fast. Today it is a plain two-entry `DirectLabelColormap`, so
it takes the same typed-dict path as styled layers, and the
`CompactLabelColormap` override does not reach it.

Give it its own `DirectLabelColormap` subclass, for example
`NeutralLabelColormap` in `viewer/labels_colormap.py`, and use it in
`apply_neutral_labels_style`. It must not subclass `CompactLabelColormap`, so
that `isinstance(colormap, CompactLabelColormap)` in
`object_classification/viewer_styling.py` keeps meaning "table-styled".

Its texture needs no lookup table and no numba. With only the default (`None`)
and background entries, napari maps every unknown label to texture code 0 and
the background to code 1 (`DirectLabelColormap._label_mapping_and_color_dict`).
The texture is one comparison:

```python
class NeutralLabelColormap(DirectLabelColormap):
    def _data_to_texture(self, values):
        if not isinstance(values, np.ndarray) or values.dtype.itemsize <= 2 or self.use_selection:
            return super()._data_to_texture(values)
        # Background -> 1, every other label -> 0 (the default code).
        return (values == self.background_value).view(np.uint8)
```

- Do not rely silently on napari numbering the default 0 and the background 1:
  check both codes (and that there are only two) in
  `_label_mapping_and_color_dict[0]` before using the comparison, and use
  napari's path otherwise. The mapping is cached by napari, so the check is
  cheap, and it also covers a colormap rebuilt from `model_dump()`.
- The output dtype is right: `_num_unique_colors + 2` is 4, so vispy expects
  uint8.
- Keep napari's constructor (no custom `__init__`).
  `_cmap_without_selection` rebuilds the colormap with
  `self.__class__(**self.model_dump())`, used by `map()` on (u)int8/16 data in
  selected-label mode. Build it through a small factory function that takes the
  neutral color and `background_value=0`.
- `map()` is inherited, so existing tests that check `colormap.map(5)` against
  `DEFAULT_NEUTRAL_COLOR` work unchanged.
- napari's `Labels._is_default_colors` stays false (the neutral color is not
  black), so the layer stays in DIRECT mode.

### 7. Continuous styles: 252 bins

Lower the default `bins` of `compact_continuous_labels_mapping_from_values`
and `compact_continuous_label_colormap_from_values` from 256 to 252, through
one module constant (for example `_CONTINUOUS_LABEL_BINS = 252`) with a comment
on why. No caller passes `bins` explicitly today.

Why 252: a continuous mapping has `bins + 3` texture rows (default, background,
the bins, missing). napari's `minimum_dtype_for_labels` uses uint8 only up to
255 colors (`num_colors <= 255`), so 256 bins (259 rows) need uint16, and 252
bins (255 rows) fit in uint8. This halves, for continuous layers:

- the dense table (1 byte per entry instead of 2);
- the per-slice texture uploaded to the GPU (uint8 instead of uint16);
- the `texture_codes` array.

`bins` stays a parameter. A caller that passes more than 252 bins still works,
with uint16 textures.

Trade-off: with 256 bins, labels colors are exactly matplotlib's `magma(x)`,
because `magma` is itself a 256-entry table. Shapes color continuous values
with `continuous_rgba_for_values` (`viewer/_styling.py`), which calls
`magma(x)` directly, so today the same value gets the same color in both. With
252 bins, about 31% of values land one `magma` step away (largest channel
difference 0.0096, about 2.5 of 255 levels; measured over 1M random values).
That is not visible, but labels and shapes colors for the same value are no
longer bit-identical.

### Unchanged

- `_get_typed_dict_mapping` stays: it serves the fallback and `map()`.
- `map()` keeps going through napari's typed-dict path. Thumbnails are
  downsampled to thumbnail size before `map` is called, so there is nothing
  measurable to gain.
- `set_backend(ColormapBackend.numba)` in `CompactLabelColormap.__init__`
  stays: the fallback and `map()` still use napari's backend.

### Docstrings to update

- `CompactLabelsMapping`: the derived structures now include the dense table.
- `_CompactLabelToTextureMapping`: says full high-bit arrays use
  `_get_typed_dict_mapping(...)`; that is now the fallback only.
- `CompactLabelColormap.map`: the call chain stays, but should not suggest it
  is the rendering path.
- `compact_continuous_labels_mapping_from_values`: the comment on the
  "256-bin RGBA table".

## Prototype Results

Scratch prototype (a subclass of `CompactLabelColormap` with the override
above), synthetic labels as in the issue's reproducer, napari 0.7.1, Apple
silicon. Median of 7 runs of `Labels._raw_to_displayed`:

| Slice | napari default | Styled now | Styled + table | Neutral now |
|---|---|---|---|---|
| 1800 × 2900 (5.2 Mpx) | 1.6 ms | 9.8 ms | 1.2 ms | 8.4 ms |
| 3600 × 5800 (20.9 Mpx) | 6.2 ms | 33.5 ms | 3.9 ms | 30.6 ms |

For the neutral subclass, the bare comparison `(values == 0).view(np.uint8)`
takes 0.7 ms at 5.2 Mpx and 1.7 ms at 20.9 Mpx (timed on its own, not through
the layer). A parallel numba version was slower (0.8 ms and 2.6 ms).

The issue reports 0.6 ms (5.2 Mpx) and 2.2 ms (21 Mpx) on the real dataset with
the same approach.

Output was identical to the current typed-dict path for:

- categorical (uint8) and continuous (uint16, with today's 256 bins) mappings,
  with missing values;
- int32, int64, uint32 and uint64 data;
- ids above the largest table id, negative values, the dtype maximum,
  background and unmapped labels;
- scalar input and `use_selection`;
- after `set_label_value` on an existing label, on a new label, with a new
  value that appends a texture row, and after `set_label_missing`.

## Things to Watch

- **Memory.** The table is one per styled layer and scales with the largest
  label id; the cap is an upper limit, not an allocation, so typical datasets
  use KB to a few MB. The 128 MB cap allows label ids up to about 134 million
  for 1-byte tables (continuous styles with 252 bins, categorical styles with
  up to 252 categories) and about 67 million for 2-byte tables. Above that, the
  layer keeps today's speed.
- **Table size and slice speed.** Measured on a 5.2 Mpx slice with ids spread
  over the whole table: 0.7 ms with a 75 KB table, 0.9 ms at 32 MB, 1.0 ms at
  128 MB, 1.1 ms at 256 MB. Pixels of one object share an id, so table entries
  are reused.
- **First compile.** Without numba's on-disk cache, the first styled slice pays
  the JIT compile on the main thread: about 365 ms for the first
  (data dtype, texture dtype) pair and about 100 ms per extra pair, measured
  here. With `cache=True` this is paid once per environment, like the existing
  `_fill_typed_label_texture_mapping`.
- **Threading.** The kernels run in parallel on the main thread, like napari's
  own label kernels, so no new threading layer is involved.

## Tests

In `tests/test_labels_colormap.py`. Use napari's current path as the
reference: `_accel_cmap.labels_raw_to_texture_direct(data, colormap)` still
calls `_get_typed_dict_mapping`, so it gives today's output.

- Equality with the reference, parametrized over categorical and continuous
  mappings and int32/int64/uint32/uint64 data, including ids above the table,
  negative values, the dtype maximum, background, unmapped and missing labels.
- Output dtype is `minimum_dtype_for_labels(len(texture_rgba))`: uint8 for a
  default continuous mapping, uint16 for a categorical mapping with 256+ rows.
- `use_selection`, scalar input and (u)int8/16 data give the same result as
  `DirectLabelColormap._data_to_texture`.
- Non-contiguous input (for example `data[:, ::2]`) and empty input.
- After `set_label_value` (existing label, new label, new value) and
  `set_label_missing`, output equals a fresh `CompactLabelColormap` built from
  the updated `_compact_mapping`; a new value that crosses 255 rows switches the
  output to uint16.
- Fallback: with `_DENSE_LUT_MAX_BYTES` monkeypatched small, with a negative
  background, and with table creation monkeypatched to raise `MemoryError`,
  output equals the reference.
- Fallback logging: capture loguru output with a temporary sink
  (`logger.add(...)`, as in `tests/test_feature_extraction_core.py`).
  - Over the cap: one `INFO` message naming the largest label id and the cap.
  - `MemoryError`: one `WARNING` message.
  - Several slices plus several `set_label_value` / `set_label_missing` calls
    on the same colormap still log only once.
  - A colormap that uses the dense table logs nothing.
- Fast path is used: with `CompactLabelColormap._get_typed_dict_mapping`
  monkeypatched to raise, `Labels._raw_to_displayed` on a styled int32 layer
  (categorical and continuous) still returns the expected texture. This catches
  a later change that sends styled layers back to the slow path while colors
  stay correct, without measuring time.
- Typed-dict path: one existing test pins it.
  `test_compact_categorical_label_colormap_high_bit_texture_mapping_uses_jitted_typed_dict`
  calls `_data_to_texture` on int64 data and asserts that
  `"_compact_int64_typed_dict"` is in `colormap._cache_other`. With the dense
  table that dict is no longer built, so the test would fail.
  - Keep its expected texture values, rename it (now
    `test_compact_categorical_label_colormap_high_bit_texture_mapping_uses_dense_table`),
    and assert that the dense table is cached instead.
  - Move the typed-dict assertion into the fallback tests, where
    `_get_typed_dict_mapping` is still used.
- 252 bins: three existing tests pin the 256-bin layout and need new numbers.
  - `test_compact_continuous_labels_mapping_matches_current_rgba_for_representative_values`:
    255 rows (was 259), missing code 254 (was 258);
  - `test_compact_continuous_labels_mapping_uses_explicit_range_and_clamps_values`:
    0.5 gets code 128 (was 130), 1.0 and 2.0 get 253 (was 257);
  - `test_compact_continuous_labels_mapping_accepts_large_label_ids`: the top
    bin gets code 253 (was 257).

  The tests that compare with `continuous_rgba_for_values` use the values 0,
  0.5 and 1, which give the same color with 252 bins, so they pass unchanged.
- Neutral style:
  - `NeutralLabelColormap._data_to_texture` equals today's two-entry
    `DirectLabelColormap` output for int32/int64/uint32/uint64 data;
  - scalars, (u)int8/16 data and `use_selection` give the same result as
    `DirectLabelColormap`;
  - `NeutralLabelColormap(**colormap.model_dump())` rebuilds an equal colormap;
  - `apply_neutral_labels_style` installs it, the layer is in DIRECT mode, and
    `colormap.map(5)` is the neutral color;
  - fast path is used: with `NeutralLabelColormap._get_typed_dict_mapping`
    monkeypatched to raise, `Labels._raw_to_displayed` on an int32 layer still
    returns the expected texture.

## Decisions

1. **Table cap.** Decided on 2026-11-09: a cap in bytes,
   `_DENSE_LUT_MAX_BYTES`, rather than the issue's `2**24` entries, because
   bytes are what the cap is meant to limit. Value: 128 MB. Larger tables
   barely slow slice recoding; the costs that grow are memory per styled layer
   and the rebuild after each sparse edit (about 5 ms at 128 MB). 256 MB was
   considered, but several styled views of a large-id dataset would then
   approach 1 GB or more.
2. **Neutral style.** Decided on 2026-11-09: a dedicated `DirectLabelColormap`
   subclass (section 6), not an empty compact mapping, so the
   `isinstance(colormap, CompactLabelColormap)` check keeps meaning
   "table-styled".
3. **Benchmark.** Decided on 2026-11-09: no benchmark script, only unit tests,
   with no timing assertions. A script would be more code to maintain, and
   timing in pytest is flaky. The issue's acceptance criterion (styled layers
   within ~1.5× of the cyclic colormap, "with a benchmark added") is met by
   measuring once with the issue's reproducer during implementation, putting
   the numbers in the PR description, and noting there that the speed was
   measured rather than added as a script. The fast-path tests (see "Tests")
   guard against a silent return to the slow path.
4. **Continuous bins.** Decided on 2026-11-09: 252 instead of 256, done
   together with this fix (section 7).

## Upstream

napari already has a dense-array mapper (`DirectLabelColormap._array_map`), but
only for its pure-NumPy backend and only up to `2**16`; the numba backend
always uses the typed dict. A napari change that uses a dense array in the
numba backend when the keys are small would fix this for every
`DirectLabelColormap`. That is a longer-term option; the override above does
not depend on it.

## Follow-up: Contour Mode

Status: investigated and decided on 2026-11-09 (see "Decision"), not
implemented yet. Tracks https://github.com/vibspatial/spatiato/issues/136.
Reported after the fix above: with `contour > 0`, a styled
labels layer becomes slow again, possibly slower than a primary layer with the
same contour.

### Where the time goes

With `contour >= 1`, napari adds a step before the recoding:

```text
Labels._raw_to_displayed
  -> Labels._calculate_contour                 (2D slices only)
    -> get_contours                            (napari/layers/labels/_labels_utils.py)
      -> scipy ndi.grey_dilation  (cross footprint)
      -> scipy ndi.grey_erosion   (diamond of radius `contour`)
  -> colormap._data_to_texture(contoured_slice)
```

`get_contours` keeps a pixel when the maximum over its cross neighbourhood
differs from the minimum over the diamond, and sets every other pixel to the
background value. It runs single-threaded on the Qt main thread for every new
slice of every labels layer, whatever its colormap. For multiscale data napari
also runs it on the (small) thumbnail level, because `to_displayed` in
`napari/layers/_scalar_field/_slice.py` applies `_raw_to_displayed` to both.

Measured per slice (median of `Labels._raw_to_displayed`, synthetic labels as
in "Prototype Results", napari 0.7.1, Apple silicon):

| Slice | napari default | Styled | Neutral |
|---|---|---|---|
| 5.2 Mpx, contour 0 | 1.6 ms | 1.4 ms | 0.6 ms |
| 5.2 Mpx, contour 1 | 74 ms | 74 ms | 75 ms |
| 5.2 Mpx, contour 2 | 102 ms | 103 ms | 102 ms |
| 20.9 Mpx, contour 1 | 293 ms | 293 ms | 294 ms |

- Recoding the contoured slice takes under 1 ms for all three colormaps, so
  nearly all the time is in `get_contours`.
- The thumbnail costs little: a 32 × 32 `colormap.map` takes about 0.3 ms for
  the styled colormap.
- napari 0.9.1 calls the same `get_contours`; only the border it adds around
  the slice before computing contours differs.

So contour mode costs about 50× plain recoding for any labels layer. After the
fix, the contrast is simply larger on styled layers.

### Styled versus primary

A styled layer being slower than a primary layer with the same contour was not
reproduced:

- styled and primary layers are built by the same `_build_labels_layer`
  (`viewer/adapter.py`) from the same element, so contours are computed on
  identical input;
- Spatiato does not listen to contour or slicing events on labels layers;
  `sync_labels_display_after_colormap_change` (`viewer/labels_async.py`) runs
  once per colormap change, not while panning.

Still to check on the real data: time
`layer._raw_to_displayed(layer._slice.image.raw)` for the styled and the
primary layer at the same position and `contour`. Two labels layers visible at
once each pay the contour cost, and a different zoom or field of view changes
the slice size.

### Prototype: numba contours

The same rule, rewritten: because the diamond contains the cross, a pixel is
kept when any 4-neighbour differs from it, or (for `contour > 1`) any pixel in
the diamond is smaller than it. Borders follow scipy's `'reflect'` mode.
Parallel over rows:

```python
@njit(cache=True, inline="always")
def _reflect(i, n):
    # scipy.ndimage 'reflect' (half-sample symmetric): -1 -> 0, n -> n - 1.
    period = 2 * n
    i = i % period
    if i < 0:
        i += period
    if i >= n:
        i = period - 1 - i
    return i


@njit(parallel=True, cache=True)
def _contours_2d(labels, thickness, background, out):
    h, w = labels.shape
    for y in prange(h):
        for x in range(w):
            v = labels[y, x]
            boundary = (
                labels[_reflect(y - 1, h), x] != v
                or labels[_reflect(y + 1, h), x] != v
                or labels[y, _reflect(x - 1, w)] != v
                or labels[y, _reflect(x + 1, w)] != v
            )
            if not boundary and thickness > 1:
                for dy in range(-thickness, thickness + 1):
                    r = thickness - abs(dy)
                    yy = _reflect(y + dy, h)
                    for dx in range(-r, r + 1):
                        if labels[yy, _reflect(x + dx, w)] < v:
                            boundary = True
                            break
                    if boundary:
                        break
            out[y, x] = v if boundary else background
    return out
```

Output was identical to `get_contours` for int32, int64, uint32 and uint16
data; shapes from 1 × 1 to 300 × 411; contour 1, 2, 3 and 5; background 0 and
2; and both full slice sizes below.

| Slice | napari (scipy) | numba prototype |
|---|---|---|
| 5.2 Mpx, contour 1 | 77 ms | 4 ms |
| 5.2 Mpx, contour 2 | 103 ms | 21 ms |
| 20.9 Mpx, contour 1 | 319 ms | 17 ms |
| 20.9 Mpx, contour 2 | 411 ms | 84 ms |

Thicker contours can get faster: the diamond loop reflects every index, which
only pixels near the slice edge need.

### Options

The contour step belongs to napari's `Labels` layer, not to the colormap, so
the colormap override above cannot reach it.

1. **Replace `_calculate_contour` on each layer Spatiato builds**, as an
   instance attribute set in `_build_labels_layer`. The smallest change, and
   the layer stays a plain `Labels`. It overrides a private napari method, and
   it is less visible than a class.
2. **A `Labels` subclass (`SpatiatoLabels`)** that overrides
   `_calculate_contour`. One visible class, and the natural place for later
   layer-specific behaviour (trimming the layer controls, blocking painting
   modes). It must override `_type_string`: napari derives the layer type from
   the class name (`napari/layers/base/base.py`), and `"spatiatolabels"` would
   break the layer-list icon, the labels entries in the layer-list context
   menu, Duplicate layer and napari's writers. Everything else in napari
   (vispy visual, layer controls, key bindings) already resolves a subclass
   through `isinstance` or the class MRO.
3. **Patch `napari.layers.labels.labels.get_contours` globally.** One line,
   same output, but it changes napari for every labels layer in the session,
   including other plugins' layers. Not done from a plugin.
4. **Outlines inside the Spatiato colormaps**: keep napari's `contour` at 0
   and detect edges in `_data_to_texture`. No change to `Labels`, but it needs
   its own control, does not cover napari's random colormap on primary layers,
   and is wrong on the small painted patches `_data_to_texture` also receives.
5. **Outlines as data in a separate layer**, computed lazily with dask (one
   chunk at a time, with overlap) by the numba kernel. The only option that
   moves the work off the Qt main thread: with async slicing, napari loads the
   slice data in a worker thread (`__call__` in
   `napari/layers/_scalar_field/_slice.py`), while `to_displayed`, which runs
   the contour step, stays on the main thread. A feature rather than a fix: an
   extra layer, dask work and caching, its own control, and the outline
   thickness is fixed per resolution level.
6. **Draw outlines on the GPU**, with a shader that compares neighbouring
   texels. Almost no CPU cost, but it replaces napari's labels visual. Better
   proposed upstream.
7. **Upstream in napari.** Speed up `get_contours` with numba, as napari
   already does for its label colormaps (`_accelerated_cmap.py`). Every napari
   user benefits and Spatiato needs no workaround, but it waits on a napari
   release.

### Decision

Decided on 2026-11-09:

- Option 1 now, on the current branch.
- Option 7 (upstream) as well. Once every napari version Spatiato supports
  ships fast contours, remove the override.
- Trimming the labels layer controls (color mode, shuffle, brush size, label,
  blending, contiguous, show selected, ...) is a separate feature and not part
  of this change. Option 2 is the likely vehicle for it.
- The investigation is kept in two Spatiato issues, so it is not lost:
  - https://github.com/vibspatial/spatiato/issues/136: contour rendering, with
    all seven options, the plan, the reproducer and the acceptance criteria.
    The change below only partly addresses it; the upstream fix (option 7)
    is still open there.
  - https://github.com/vibspatial/spatiato/issues/137: trimming the labels
    layer controls, with the findings on subclassing `Labels`, the options
    and the open questions.

### Proposed Fix: Per-Layer `_calculate_contour`

New module `viewer/labels_contours.py`:

- the numba kernel from "Prototype: numba contours" (`_reflect`,
  `_contours_2d`);
- `labels_contours(labels, thickness, background)`: the same contract as
  napari's `get_contours` for 2D integer arrays (a new array of the same dtype);
- `_calculate_labels_contour(layer, labels, data_slice)`: the replacement for
  `Labels._calculate_contour`;
- `install_fast_labels_contours(layer)`: sets
  `layer._calculate_contour = MethodType(_calculate_labels_contour, layer)`.

`_build_labels_layer` (`viewer/adapter.py`) calls
`install_fast_labels_contours` on the layer it builds. It builds every
Spatiato labels layer, primary and styled, so all of them get it, including a
primary layer in the neutral style.

```python
def _calculate_labels_contour(layer, labels, data_slice):
    background = layer.colormap.background_value
    if (
        layer.contour < 1
        or not isinstance(labels, np.ndarray)
        or labels.ndim != 2
        or labels.dtype.kind not in "iu"
        or not _fits_dtype(background, labels.dtype)
    ):
        # Contours off, 3D, or input the kernel does not handle: napari's path.
        return Labels._calculate_contour(layer, labels, data_slice)
    expanded_slice = _expand_slice(data_slice, labels.shape, layer.contour)
    contours = labels_contours(labels[expanded_slice], layer.contour, background)
    # Trim the border back off.
    return contours[
        tuple(
            slice(s.start - e.start, s.stop - e.start)
            for s, e in zip(data_slice, expanded_slice, strict=True)
        )
    ]
```

- Calling `Labels._calculate_contour(layer, ...)` (the class function, not the
  instance attribute) keeps napari's behaviour for `contour < 1` (returns
  `None`) and for 3D (napari's warning).
- `_expand_slice` is a small local copy of napari's `expand_slice`
  (`napari/layers/labels/_labels_utils.py`), to avoid one more private import.
- Full slices, which is every pan and zoom: the expansion is clipped to the
  slice, so the output is identical to napari's.
- Partial slices only come from painting (`_partial_labels_refresh`). The
  border is `max(1, contour)` pixels, the same as napari 0.9.1; napari 0.7.1
  uses 1. The result equals computing contours on the full slice and cropping.
  With `contour >= 2` this can differ from napari 0.7.1 near the edge of the
  painted region, where a 1-pixel border is too narrow for the diamond
  erosion.
- Multiscale thumbnails also go through `_raw_to_displayed`, so they use the
  kernel too.
- `_calculate_contour` has the same signature in napari 0.7.1 and 0.9.1.

### Tests

In a new `tests/test_labels_contours.py`, plus one adapter test:

- `labels_contours` equals napari's `get_contours` for int8 to uint64 data,
  shapes from 1 × 1 to 37 × 53, contour 1, 2, 3 and 5, background 0 and a
  non-zero background, and non-contiguous input (`data[:, ::2]`).
- Through the layer: after `install_fast_labels_contours`,
  `_raw_to_displayed` equals a plain napari `Labels` with the same colormap,
  for contour 0, 1 and 2, with a styled, a neutral and napari's default
  colormap.
- Partial `data_slice`: equals the full-slice result cropped (contour 1 and
  2), and equals napari's for contour 1.
- `contour = 0` and a 3D slice give napari's result.
- Fast path is used: with `napari.layers.labels.labels.get_contours`
  monkeypatched to raise, an installed layer still renders with contour 1 and
  2. This also catches napari renaming `_calculate_contour`, which would
  silently bring back the slow path.
- `_build_labels_layer` installs it: the built layer's `_calculate_contour`
  is `_calculate_labels_contour`.

### Things to Watch

- **Private API.** `_calculate_contour` is private. The fast-path test is the
  guard.
- **First compile.** The first contoured slice pays a numba compile on the
  main thread per data dtype. With `cache=True` this is paid once per
  environment, like the colormap kernels.
- **Duplicate layer.** napari's Duplicate layer rebuilds a plain `Labels`
  from `as_layer_data_tuple()`, so the copy has napari's contour speed.
- **Thick contours.** 21 ms for contour 2 on a 5.2 Mpx slice. Skipping the
  index reflection for pixels away from the slice edge should bring this down
  if needed.
