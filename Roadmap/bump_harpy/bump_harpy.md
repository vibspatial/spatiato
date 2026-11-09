# Bump Harpy to 0.5.3: Lazy Matrices and Table Component I/O

Status: investigation done. Problems are listed below and will be tackled one by
one, in the order of "Migration Plan"; each section's direction is a proposal to
settle when we pick it up. Target moved from harpy 0.5.0 to 0.5.2 on
2026-10-25, and to 0.5.3 on 2026-11-05 (canonical-center refactor and
reserved `.obsm` keys, see "Harpy 0.5.3: What Changed").

## Goal

Move Spatiato to `harpy-analysis>=0.5.3` and handle the two harpy 0.5.0 changes
that affect Spatiato:

- **lazy matrices**: harpy now attaches table matrices as Dask arrays or backed
  on-disk handles instead of in-memory arrays;
- **table component I/O**: harpy now has its own API to read, write, and delete
  individual table components, overlapping with Spatiato's own reader/writer.

harpy 0.5.1 also raises the `anndata` floor to `>=0.13`, so the move to anndata
0.13 is part of this bump. harpy 0.5.3 moves canonical centers fully to harpy
(P8) and reserves two `.obsm` keys (P9).

Development policy: develop against harpy releases from PyPI, not against
harpy's `main` branch. When Spatiato needs an unreleased harpy change, pin it
temporarily with a uv source on a commit, which only uv reads (the published
requirement stays `harpy-analysis>=...`):

```toml
[tool.uv.sources]
harpy-analysis = { git = "https://github.com/saeyslab/harpy", rev = "<sha>" }
```

Remove it once harpy releases the change, and before any Spatiato release.

## Environment

- `pyproject.toml`: `harpy-analysis>=0.5.3` (raised from `>=0.4.4` to `>=0.5.0`
  in `d895ea5`, then to `>=0.5.2`, and to `>=0.5.3` on 2026-11-05),
  `spatialdata>=0.8.0` (needed with anndata 0.13, see P0),
  `napari[all]>=0.7.1`. Spatiato does not declare `anndata`; harpy's floor
  applies.
- `.venv` has **harpy 0.5.3** (since 2026-11-05), `anndata 0.13.4`,
  `spatialdata 0.8.0`, `dask 2026.8.0`, `zarr 3.2.1`, `numpy 2.3.5`. See P0
  for the test runs on 0.5.0, 0.5.2 and 0.5.3.
- harpy 0.5.2 requires `anndata>=0.13`, `spatialdata>=0.7.3`, and `zarr>=3.2.1`.
  spatialdata 0.7.3 does not work with anndata 0.13, so in practice it needs
  `spatialdata>=0.8.0` (see P0).
- harpy releases:

  | Version | Tag commit | Content |
  |---|---|---|
  | 0.5.0 | `6299266` | lazy matrices and table component I/O ("Feature/table qc (#264)") |
  | 0.5.1 | `a65e793` | "Feature/qc metrics (#271)", the `feature/qc_metrics` branch; anndata `>=0.13` |
  | 0.5.2 | `5ff8c3e` | "Fix/cosmx reader fov coordinates (#275)", including the fix for https://github.com/saeyslab/harpy/issues/274 |
  | 0.5.3 | `4363428` | "Feature/concat tables (#283)": the canonical-center refactor (https://github.com/saeyslab/harpy/issues/279, https://github.com/saeyslab/harpy/issues/282) and reserved `.obsm` keys |

- Source reviewed at tag `v0.5.0`, on the `feature/qc_metrics` branch (before it
  became 0.5.1), and in #275. The storage design is documented in harpy's
  `docs/development/storage.md`.

## Harpy 0.5.0: What Changed

### Lazy matrices

harpy has its own table readers; they use
`anndata.experimental.read_elem_lazy`, not `read_lazy`, and they bypass
spatialdata's table reader.

- `hp.io.read_zarr(store, *, table_name=None, table_mode="lazy", ...)` reads
  images/labels/points/shapes with `spatialdata.read_zarr` and attaches tables
  with `hp.tb.io.read_table(...)`. Default table mode is **lazy**.
- `hp.tb.io.read_table(store, *, table_name, mode="lazy"|"backed"|"eager", ...)`
  and `hp.tb.io.read_table_components(...)`.

Representation per mode (`harpy/_storage/_anndata.py`):

| Mode | Matrices (`X`, `layers`, `obsm`, `varm`, `obsp`, `varp`, `raw`) |
|---|---|
| `lazy` | `dask.array.Array`; sparse data stays as CSR/CSC blocks inside Dask |
| `backed` | read-only `zarr.Array`, or AnnData `CSRDataset` / `CSCDataset` |
| `eager` | NumPy / SciPy |

`obs`, `var`, and `uns` are always in memory. The returned AnnData reports
`isbacked=False` even when its matrices are lazy or backed.

Functions Spatiato is exposed to:

- `hp.tb.add_feature_matrix(...)` (called by Spatiato's feature extraction):
  - new table: goes through `hp.tb.io.add_table`, which reopens the written table
    lazily for backed SpatialData;
  - existing table: goes through `add_table_components_by_region`
    (`harpy/table/io/_components_by_region.py`), which for backed SpatialData
    writes the component and reattaches it with `mode="lazy"` (verified in
    source);
  - so after feature extraction on backed SpatialData, `table.obsm[feature_key]`
    is a **Dask array**. For unbacked SpatialData, harpy 0.5.0 installs a "lazy
    merge" of the existing and new rows, also a Dask array (verified at runtime,
    see P0). From 0.5.2 it computes the merge when the existing matrix is
    absent or in memory, and attaches NumPy or SciPy (see "Harpy 0.5.1 and
    0.5.2: What Changed").
- `aggregate_points` / `allocate` and `add_canonical_centers` attach **backed**
  handles (`CSRDataset`, `zarr.Array`), not Dask arrays.

harpy's feature-matrix metadata (`uns["feature_matrices"][<key>]`) is unchanged
in shape; `backend` is hard-coded `"numpy"` and laziness is not recorded.

There is no public "materialize" helper; options are `mode="eager"` or
`adata.to_memory()`.

Checked against 0.5.2 on 2026-10-30 (for D1):

- `hp.io.read_zarr` leaves tables out of spatialdata's read
  (`selection=("images", "labels", "points", "shapes")`) and reads each table
  with one `table_mode`; there is no per-component mode.
- `uns` is always read eagerly, as are DataFrame-valued `obsm`/`varm` entries
  and `raw.var`.
- Lazy and backed reads raise on matrix encodings harpy does not support
  (only dense `array`/`string-array` 0.2.0 and CSR/CSC 0.1.0); they do not
  fall back to an eager read.
- `read_table_components(...)` returns a dict of components, not an AnnData.
  On backed SpatialData, harpy has no public way to install a value in memory
  without writing it; plain assignment (`table.obsm[key] = value`) does that.
- No harpy function lets the caller choose the reattachment mode; the chunk
  settings of `add_table_components_by_region(...)` shape only the merge, and
  the reattached array uses `"auto"` chunks.
- `add_canonical_centers` attaches only the canonical `obsm` matrix, as a
  backed `zarr.Array`. `allocate` is a deprecated alias of `aggregate_points`.
- anndata 0.13.4 `AnnData.to_memory()` returns a new AnnData (Dask is
  computed, `zarr.Array` and `CSRDataset`/`CSCDataset` are read); it does not
  change the original.
- Lazy sparse blocks are SciPy CSR (or CSC) matrices; `.compute()` returns a
  `csr_matrix`. Dense lazy blocks span whole rows, about 128 MiB each with
  `"auto"`; harpy stores dense matrices in about 4 MiB row chunks.

### Table component I/O

Store-only functions (`hp.tb.io`, return `None`):

- `write_table_components(store, *, table_name, components, delete=(), obs_identity=None, var_names=None, raw_var_names=None, overwrite=False)`
- `delete_table_components(store, *, table_name, components)`
- `read_table_components(store, *, table_name, components, missing="raise"|"omit", mode=..., ...)`
- `write_table_updates(store, *, table_name, adata, x_to=None, overwrite=False)`
  (writes components that differ from the store; never deletes)
- `write_table_components_by_region(...)` (complete-region `obsm` writes)

SpatialData adapters (update the attached table, and the store when
`sdata.path` is set; written components are reattached lazily):
`add_table_components`, `remove_table_components`, `add_table_updates`,
`add_table_components_by_region`, `add_table`.

Supported component paths (tuples):

| Path | Write | Delete | Read |
|---|---|---|---|
| `("X",)` | yes | yes | yes |
| `("obs",)`, `("var",)` | whole dataframe only | no | yes |
| `(layers\|obsm\|varm\|obsp\|varp, key)` | yes | yes | yes |
| `("uns", ...)` at any depth | yes | nested keys (not `spatialdata_attrs`) | yes |
| `("raw", ...)` | yes | yes | yes |

Guarantees and requirements:

- staged writes: data goes to a sibling `.<store>.harpy-table-staging-*`
  directory, old destinations are moved to a backup, staged paths are renamed
  into place, and backups are dropped only after consolidation succeeds; root
  metadata is restored on failure. This is rollback for caught exceptions, not
  crash atomicity;
- lazy replacements may read from components that the same call overwrites or
  deletes;
- zarr v2 and v3; `consolidate_metadata` after publication;
- obs-aligned writes need `obs_identity` (a `(region, instance)` dataframe with a
  categorical region column, or an `obs` dataframe);
- local stores only; no reader/writer locking;
- no special handling of categorical columns or their `uns["<col>_colors"]`
  palettes; callers include palette paths explicitly.

Deprecated in 0.5.0: `harpy.tb.add_table` (use `harpy.tb.io.add_table`) and the
scanpy wrappers (`kmeans`, `leiden`, `preprocess_*`, ...). Spatiato uses none of
them; it calls `hp.tb.add_feature_matrix`, and `datasets.py` uses
`hp.im.add_image` / `hp.im.add_labels`.

## Harpy 0.5.1 and 0.5.2: What Changed

0.5.1 is the `feature/qc_metrics` branch (first checked on 2026-10-22 at commit
`e9b9bb1`), merged as #271. 0.5.2 adds #275. Rechecked against `v0.5.2` on
2026-10-25.

Unchanged for P7:

- `table/io/_write_by_region.py` is identical to `v0.5.0`;
  `table/_add_feature_matrix.py` differs only in a docstring (0.5.2). The
  regional merge and its compatibility checks are the same.
- `_SOURCE_KIND = "harpy_add_feature_matrix"`; `_FEATURE_MATRIX_SCHEMA_VERSION`
  is still `1`, now defined in `harpy/table/_metadata.py`.
- `hp.tb.io.remove_table_components(sdata, *, table_name, components)` is
  unchanged and still exported.

Unbacked regional updates (0.5.2, #275, closes
https://github.com/saeyslab/harpy/issues/274):

- for unbacked SpatialData, `add_table_components_by_region(...)` now computes
  the merge when the supplied and the attached matrix are both in memory (NumPy,
  or SciPy CSR/CSC; no existing matrix also counts), and attaches the result in
  the same form. This also applies to `add_feature_matrix(...)` on an existing
  table, whose computed features are NumPy;
- Dask arrays, Zarr-backed and memory-mapped matrices keep the lazy merge, and
  the docstrings advise computing between repeated updates in that case. Once
  an unbacked table holds a Dask matrix, repeated updates still chain merges;
- backed SpatialData is unchanged.

Relevant elsewhere:

- **anndata floor `>=0.12.10` -> `>=0.13`** (0.5.1; `.venv` now has 0.13.4).
  In anndata 0.13 `X` lives in memory as `layers[None]`; harpy adapts
  internally. Spatiato only uses `table.X`, never `layers`, so the risk is low;
  the P0 rerun covers it.
- **zarr v3 sharding**: anndata 0.13 enables sharding by default
  (`ad.settings.auto_shard_zarr_v3` is `True` in 0.13.4). harpy still writes dense matrices with
  `shards=None` because of https://github.com/saeyslab/harpy/issues/270 (open).
  Spatiato's own `ad.io.write_elem` writes in `core/persistence.py` do not opt
  out; see P5.
- harpy removed its workaround for lazily reading empty sparse matrices,
  relying on anndata 0.13 (background for P2/P3).
- New public `hp.tb.validate_table(sdata, table_name, feature_matrices_key=...)`
  validates feature-matrix records against `obsm`. It does not replace P7's
  merge-compatibility check.
- harpy's `[tool.uv] override-dependencies` (`numpy>=2.4.6`, `numba>=0.64.0`)
  apply to harpy's own environment only; neither is in harpy 0.5.2's published
  requirements, so they do not reach Spatiato (`.venv` has numpy 2.3.5).

## Harpy 0.5.3: What Changed

Checked on 2026-11-05 against the installed 0.5.3 and harpy's
`roadmap/concat_tables/canonical_cache.md`. https://github.com/saeyslab/harpy/issues/279
and https://github.com/saeyslab/harpy/issues/282 are addressed by #283, though
both are still open on GitHub.

- **Canonical centers** (P8):
  - `hp.tb.add_canonical_centers(sdata, *, table_name, overwrite=False)` loses
    `labels_name` and always computes every labels element the table
    annotates. It installs through `hp.tb.io.add_table_components`: on
    unbacked SpatialData the matrix is NumPy in memory; on backed SpatialData
    the two canonical components are written to the store and the matrix is
    reattached lazily, as a Dask array, no longer as a read-only `zarr.Array`.
    `overwrite` only concerns the store (`FileExistsError` without it). It
    first checks the whole table as `hp.tb.validate_table` does, except the
    canonical centers it replaces.
  - `harpy.table.canonical_centers.validate_canonical_centers(sdata, *,
    table_name)` replaces `validate_canonical_payload` and reads the region
    and instance keys and the declared regions from the table's annotation.
    It returns `None` without canonical centers, the metadata when they are
    valid, and raises `ValueError` otherwise.
  - The record is schema version 2: no `coverage` (row count and instance
    digest), so row subsets and reorders keep the centers valid. A version-1
    record, such as every record Spatiato wrote, raises with a hint to
    rebuild with `overwrite=True`. Validation compares each declared region's
    labels source (dims, shape, dtype) and algorithm version, and requires
    finite values and `z = 0` for 2-D labels.
  - The cache API is deleted or private: `inspect_canonical_cache`,
    `read_canonical_centers_from_cache`, the reports, states and update
    payloads, `build_canonical_region_binding` and the instance digest. A
    reader finds a labels element's rows from the table annotation.
- **Reserved `.obsm` keys** (P9): `spatial_canonical` and `spatial`.
  `add_feature_matrix` rejects either as `feature_key`, before computing, and
  `validate_table` rejects a feature-matrix record under either.
- **Unchanged for Spatiato:** every private name it imports
  (`_resolve_channels`, `_get_spatial_element`, `_component_update_destination`,
  `_check_in_memory_versus_storage_axes`, `_validated_observation_pairs`,
  `_FEATURE_MATRICES_KEY`) and the public `get_dataarray` and
  `RasterAggregator` still exist; regional updates of an unbacked matrix still
  chain lazy merges unless both matrices are in memory (3c.3 relies on this).
  Spatiato's custom feature-matrix records (`source_kind="custom_obsm"`) pass
  harpy's record checks.

## Spatiato Today

### Matrix consumption

Every table-matrix read assumes an in-memory NumPy or SciPy-sparse array. There
is no Dask, `.compute()`, backed, or lazy handling for table matrices, and no
tests use them.

Main consumers:

- `normalize_feature_matrix(...)` (`core/feature_matrix_metadata.py`): sparse
  branch returns the matrix; dense branch calls
  `np.asarray(feature_matrix, dtype=np.float64)` (full materialization).
- `inspect_feature_matrix_metadata(...)` and
  `register_feature_matrix_metadata(...)`: normalize the full matrix to read
  `shape[1]` / dtype; `register_...` records `backend` as `"sparse"` or
  `"numpy"` only.
- Object Classification:
  - `ClassifierController._prepare_classifier_summary(...)` normalizes and
    builds the finite-row mask **on every widget refresh / status update**;
  - `_prepare_classifier_job(...)` normalizes with `copy=True` and row-indexes
    the matrix in the main thread before the worker starts;
  - `_validate_current_feature_matrix_matches_columns(...)` normalizes only to
    compare `shape[1]`;
  - `apply_classifier(...)` / `_get_finite_feature_row_mask(...)`
    (`core/object_classification/classifier.py`): the sparse branch uses
    SciPy-only `.data` / `.tocoo()`.
- Coloring by `X` var: `_build_x_var_colormap(...)` (`viewer/labels_styling.py`)
  and `_get_table_color_values_by_instance(...)` (`viewer/shapes_styling.py`)
  read `table.X[obs_positions, col]`, then `.toarray()` or `np.asarray(...)`.
- Spatial query canonical matrix (`obsm["spatial_canonical"]`):
  `_validate_canonical_matrix(...)` and other readers in `core/spatial_query/`
  use `np.asarray(...)` and reject sparse.

Inventory of 2026-10-30 (for D1), adding to the list above:

- Object Classification reads the selected matrix whole four times per refresh
  (`_update_selection_status()`: twice through the widget's metadata check,
  twice in `_prepare_classifier_summary(...)`), plus the finite-row mask; two
  more once a trained model exists (export readiness).
- Any `obsm` key can be the classifier's feature matrix: Object Classification
  lists every key, `spatial_canonical` included, and selects the first one.
- `X` is read only by the two coloring functions above. The var lists in the
  UI use `var_names` only. `layers`, `obsp`, `varm`, `varp` and `raw` have no
  consumers.
- Lazy matrices already occur without a lazy reader: on backed SpatialData,
  feature extraction leaves `obsm[feature_key]` as Dask, and creating a new
  table reopens the whole table lazily, `X` included.
- Reload reads every `obsm` key eagerly (`ad.io.read_elem`) and never reloads
  `X`, `var` or `layers`.

Observed with anndata 0.12.16 (scratch scripts, not in the test suite):

| Matrix | Behavior |
|---|---|
| dense Dask `obsm` | works, but each `normalize_feature_matrix` call materializes the full matrix |
| lazy sparse `obsm` (Dask of CSR blocks) | `normalize_feature_matrix` raises `ValueError: setting an array element with a sequence.` |
| lazy sparse `X`, coloring by var | `X[positions, col]` is a Dask array without `.toarray()`; `np.asarray` yields an object array of shape `(1,)` |
| backed `zarr.Array` / `CSRDataset` | `X`-var coloring works (see P2); other consumers not tested |

### Table component persistence

`core/persistence.py`:

- `TableComponentPath` supports `obs`, `obsm`, and `uns` only (no `X`, `var`,
  `layers`); `obs` and `obsm` paths take one key, `uns` paths may be nested.
- `write_table_components(...)`: opens the store with
  `zarr.open_group(sdata.path, mode="r+", use_consolidated=False)`; any `obs`
  path rewrites the whole `obs` dataframe with `ad.io.write_elem`; `obsm` keys
  and `uns` paths are written in place or deleted; ends with
  `zarr.consolidate_metadata(...)`. No staging, rollback, or locking.
- `reload_table_components(...)`: reads with `ad.io.read_elem` (always eager),
  validates the reload binding (row count/order, region and instance keys) and
  `obsm` row counts, installs `obs`/`obsm`/`uns`, runs `TableModel.validate`,
  and restores the previous in-memory state on failure.
- `write_table_prediction_state(...)` (used by headless) writes `obs` plus
  selected `uns` keys.

Dirty tracking lives in `SpatiatoAppState` (`_app_state.py`): widgets record
explicit `TableComponentPath`s with change tokens; writes snapshot, write, and
acknowledge only unchanged tokens.

Feature extraction relies on harpy persisting its own output: for backed
SpatialData, the widget calls `record_persisted_table_change(...)` for
`("obsm", key)` and `("uns", ("feature_matrices", key))` instead of marking them
dirty (comment "Contract: for backed SpatialData, `hp.tb.add_feature_matrix()`
has already persisted ...").

Tests: `tests/test_persistence.py`, `tests/test_persistence_controls.py`,
`tests/test_multi_widget_backed_zarr.py`, `tests/test_app_state.py`,
`tests/test_headless.py`, plus the feature extraction, object classification,
and spatial query widget tests.

Existing notes: `Roadmap/dask_backed_tables.md` (anndata 0.12.10 observations
on Dask-backed `X` / `obsm`), and
`Roadmap/feature_extraction_roadmap/feature_extraction.md`.

## Migration Plan

Agreed on 2026-10-27. Moving Spatiato's table I/O to harpy's API, and dropping
Spatiato's own table I/O code, is the core of this bump. Order:

1. **P7.** Its spec is complete. It lives in feature extraction, not
   persistence, and already goes through harpy's API. It makes the suite green,
   so the persistence refactor starts from a clean baseline.
2. **D1: how tables live in memory** (decided on 2026-10-30). Every table
   matrix stays lazy: `X`, `obsm`, `layers`, `obsp`, `varm`, `varp` and `raw`.
   Only `obs`, `var`, `uns` and DataFrame-valued `obsm` entries are in memory,
   as harpy always reads them. This is harpy's default reader,
   `hp.io.read_zarr(store)` (`table_mode="lazy"`), with nothing loaded
   afterwards. Today every table is fully loaded when a store is opened
   (`spatialdata.read_zarr`, which reads tables with `anndata.read_zarr`).

   Why (inventory of 2026-10-30, see "Spatiato Today"):
   - `X` is read in two places, both coloring by an `X` var on a user click:
     one column, for the rows of one region. `layers`, `obsp`, `varm`, `varp`
     and `raw` have no consumers.
   - `obsm` feature matrices are read whole only by Object Classification, and
     only because it derives the usable rows from the values (the finite-row
     mask) on every refresh. That derivation moves (step 3), so values are read
     only when a classifier job runs, and only the rows it uses.
   - harpy attaches lazy matrices after its own writes on backed SpatialData
     (feature extraction, `add_table`), so lazy is the representation Spatiato
     meets anyway. Loading at open would only hold until the next extraction.
   - Considered and rejected: loading `obsm` at open, at reload and after each
     extraction (the first proposal). Simpler for consumers, but it costs
     memory and open time for matrices that may never be used, and needs a load
     step after every harpy write.

   Consequences:
   - opening holds no matrix values; each classifier job reads its rows from
     the store (no cache for now);
   - lazy views depend on the store: data overwritten outside Spatiato (a
     notebook, another process) is seen on the next read, as harpy documents;
   - P1 mostly disappears with step 3; reload reads lazily too (step 4);
   - on unbacked SpatialData whose caller attached a Dask matrix, harpy 0.5.2
     keeps chaining regional merges onto it, because computing could read the
     whole matrix from a store (the Dask array may still read from one).
     Spatiato does not compute it either; feature extraction raises before
     such a regional update, and tells the user to load the matrix into memory
     or write to a store (decided on 2026-10-30, step 3, 3c.3; see P1,
     "Resolution").

   Related points:
   - **Chunking** (a later optimization, out of scope for steps 3-5). Dask
     reads every block that contains a requested row. With harpy's `"auto"`
     chunking a dense block is about 128 MiB (Dask's `array.chunk-size`), so
     smaller matrices are read whole even for a few rows, and scattered labeled
     training rows touch many blocks. `dense_chunks="storage"` keeps harpy's
     stored row chunks (about 4 MiB). The reader keeps harpy's defaults; measure
     chunking, and lazy against backed `X` for coloring by var, only if reads
     turn out noticeably slow.
   - **Reading shared blocks twice** (considered on 2026-10-30, not planned).
     A classifier job reads its training rows and its prediction rows
     separately, so blocks they share are read twice, for example when both
     scopes are the selected labels element. Reading the union of the
     positions once and splitting it would avoid that. Accepted for now;
     optimize only if it noticeably slows training down.
   - **Unsupported encodings** (decided on 2026-10-30). A lazy read raises on
     matrix encodings harpy does not support (`_decode_anndata_element(...)` in
     `harpy/_storage/_anndata.py`). Spatiato does not fall back to an eager
     read: the reader (step 5) and the reload (step 4) read lazily and raise
     harpy's error. Handling such encodings is harpy's responsibility. A store
     that opens today with spatialdata's eager reader can therefore fail to
     open after the switch; accepted, since anndata writes standard matrices in
     supported encodings.
   - **`Interactive(sdata)` and the headless API** take the caller's
     SpatialData and never open a store (the headless `*_from_path(...)`
     functions only load a classifier file). Its tables are used as they are,
     with no conversion on entry. Consumers handle any representation
     (step 3).
3. **Make consumers representation-agnostic** (P2, P1, P3), and read values
   only where they are used:
   - shared helpers: one turns any supported matrix, or selected rows of it,
     into NumPy or SciPy (NumPy, SciPy, dense and sparse Dask, `zarr.Array`,
     `CSRDataset`/`CSCDataset`); one reads one column for coloring by var,
     for the rows of one labels element only (P2);
   - shape and dtype from the matrix object, without loading:
     `inspect_feature_matrix_metadata(...)`,
     `register_feature_matrix_metadata(...)`,
     `_validate_current_feature_matrix_matches_columns(...)`;
   - Object Classification no longer scans the matrix (agreed on 2026-10-30).
     One contract for every feature matrix, created by
     `hp.tb.add_feature_matrix` or registered as custom: scope rows are the
     rows of the scope's labels elements that the table annotates, and the
     worker skips rows without finite features. The feature-matrix metadata
     (`source_label`) is not used to scope rows: it would only allow an earlier
     message, and it can be out of date when the matrix was changed outside
     harpy.
     - the summary, on every refresh, reads only `obs` and `uns`: scope rows as
       `_resolve_classifier_scopes(...)` already resolves them (the selected
       labels element, or every region of the table), the labeled count and
       classes from `obs`, `n_features` from `shape[1]`. It keeps every check
       that needs only `obs` and `uns` and blocks Train early: the table has
       rows for the labels element (`obs`, `uns["spatialdata_attrs"]`), the
       matrix exists with registered metadata, it has columns, and there are
       enough labeled rows in at least two classes. Every job reads `obsm` to
       train and predict. The one problem that only shows up there is a labels
       element whose rows exist but have no features (for example, not covered
       by a regional extraction): the job skips those rows and reports it;
     - the job carries the matrix object and the row positions instead of
       copies made on the main thread. The worker reads the labeled training
       rows and the prediction rows and skips every row with a NaN or ±inf in
       any column, as today's finite-row mask does; only the place of the
       check changes. Such rows come from labels elements without calculated
       features, table rows without an object in the labels element, and
       undefined features:
       - a skipped labeled row is not used for training and does not count
         towards the minimum labeled count or the two classes, so the worker
         checks both again before fitting;
       - a skipped prediction row gets no prediction, and its previous
         prediction is cleared, as today (the prediction write clears every row
         in the scope and fills only the predicted ones).

       The result carries the predicted row positions and the number of
       skipped rows; both are known only after the job. When every row of a
       labels element in the scope is skipped, the message names that labels
       element (for example, features were not calculated for it). Train is
       therefore no longer disabled in advance for a labels element without
       features;
     - headless `apply_classifier(...)` does the same;
     - a classifier job may read while a feature extraction rewrites the same
       key. Object Classification already drops in-flight results on an
       `"updated"` event for its key
       (`invalidate_for_feature_matrix_overwrite(...)`), and a failed read
       becomes an error status. Accepted;
   - spatial query's canonical centers (`obsm["spatial_canonical"]`,
     `(n_obs, 3)` in z, y, x order) are computed and validated by harpy, not by
     Spatiato (P8, decided on 2026-11-01). Spatiato only reads the selected
     labels element's rows, with `read_matrix_rows(...)`;
   - feature extraction on unbacked SpatialData raises before a regional
     update onto a matrix that is not in memory (see P1, "Resolution");
   - tests with lazy and backed matrices per consumer.

   Needed regardless of D1: harpy already attaches lazy matrices after a backed
   extraction, and P2 was hit in real use. This step comes before the reader
   switch, which would otherwise spread the P2 bug to every table.

   Readiness check (2026-10-30): every sub-step is ready to implement; the
   four design points for 3d below were settled the same day. Sub-steps, each
   committed separately:

   | Sub-step | Content | Status |
   |---|---|---|
   | 3a | Matrix helpers, with unit tests per representation | implemented on 2026-10-30 (`core/table_matrix.py`, `tests/test_table_matrix.py`) |
   | 3b | Coloring by `X` var through the column helper, for the rows of the labels element only (P2) | implemented on 2026-10-30; fixes P2 |
   | 3c.1 | Shape and dtype without loading: `inspect_feature_matrix_metadata(...)` and `register_feature_matrix_metadata(...)` (`core/feature_matrix_metadata.py`), and the column check `_validate_current_feature_matrix_matches_columns(...)` (`core/object_classification/classifier.py`). Same checks as today (2-D, row count, numeric dtype); registration records `backend` with `is_sparse_matrix(...)` | implemented on 2026-10-30 (`feature_matrix_n_features(...)`); an `object`-dtype matrix is now rejected as not numeric, and a DataFrame feature matrix is rejected (harpy never writes one), so `core/table_matrix.py` no longer supports DataFrames either |
   | 3c.2a | Spatial Query's canonical centers through harpy (P8), additive: a new `core/spatial_query/canonical_centers.py` with the centers status (`validate_canonical_centers(...)`), the calculation (`hp.tb.add_canonical_centers(..., overwrite=True)`), a labels element's centers from the table annotation and `read_matrix_rows(...)`, and the 2-D labels check; unit tests on unbacked and backed SpatialData, including a version-1 record and a table harpy cannot compute | implemented on 2026-11-05: `canonical_centers_status(...)` (`CanonicalCentersStatus`), `compute_canonical_centers(...)` (returns `"created"` or `"rebuilt"`), `read_canonical_centers_region(...)` (`CanonicalCentersRegion`) and `is_canonical_query_labels(...)`; not re-exported until 3c.2c; `tests/test_spatial_query_canonical_centers.py` |
   | 3c.2b | The switch: the query and its models, annotation, controller (design points 1-2, 8 and 9 below), widget and status card use 3c.2a; the tests that keep their behavior are re-expressed (about 35, in the query, annotation, controller, widget, status-card, multi-widget and annotation-widget modules) | implemented on 2026-11-06, after 3c.2a, 3c.5 and 3c.6: `CANONICAL_CENTERS_PATHS` in `canonical_centers.py`; the query and annotation take `CanonicalCentersRegion`, and the provenance check at Apply follows design point 4; the controller's centers job calls harpy in the worker, marks the table busy (`table_busy_marker=`), publishes its change also when detached (`on_canonical_centers_changed`), and is detached, not quit, on cancel and shutdown; `is_running` and `has_job_in_flight` (design point 9), with the waiting status; the widget records the change as persisted (backed) or as a mutation (unbacked); the status card has three states; the separate "inspection error" state is dropped, since `canonical_centers_status(...)` never raises; `docs/development/table_state.md` follows. The old modules stay until 3c.2c |
   | 3c.2c | Cleanup: delete `canonical.py`, `canonical_models.py`, `centroids.py`, their re-exports and their two test modules (28 tests of Spatiato's own cache logic), and the obsolete persistence tests (design point 5); docs and plan | implemented on 2026-11-06: the three modules (1,502 lines), their re-exports and their two test modules (28 tests) are gone. The package re-exports the `canonical_centers` API instead (`CANONICAL_CENTERS_PATHS`, the status, region, state and change types, `canonical_centers_status`, `compute_canonical_centers`, `read_canonical_centers_region`, `is_canonical_query_labels`), without `CANONICAL_OBSM_KEY` and `SPATIAL_COORDINATES_KEY`, which consumers import from harpy. Of the two persistence tests, the round trip is deleted (nested `uns` and `obsm` writes stay covered by other persistence and multi-widget tests); the failed second write is kept in a generic form, `test_failed_write_acknowledges_none_of_its_paths`, as the only test that a Write failing partway acknowledges none of its paths. The developer docs did not mention the old modules |
   | 3c.3 | Raise before a regional update onto a matrix that is not in memory, on unbacked SpatialData: `check_existing_feature_matrix(...)` (`core/feature_extraction.py`) raises `ValueError` when the SpatialData is unbacked, the existing entry is compatible, and `obsm[feature_key]` is not NumPy or SciPy CSR/CSC; the message says to load the matrix into memory or write to a store first. Tests: raises for a Dask array and a `zarr.Array`; does not raise for NumPy, backed SpatialData, or an incompatible entry. The developer doc's error table gets the row | implemented on 2026-11-06: `is_in_memory_matrix(...)` in `core/table_matrix.py` follows harpy's rule (`_in_memory_matrix` in `harpy/table/io/_components_by_region.py`), so a memory-mapped NumPy array counts as not in memory too; the check raises with the advice to load the matrix into memory or write to a store. Tests: the helper over every representation and a memmap (`test_table_matrix.py`); the check and the worker's re-check raise for Dask, Zarr and memmap without changing the matrix, and an incompatible Dask matrix is replaced (`test_feature_extraction_core.py`). `docs/development/feature_extraction.md`: the classification flowchart, a bullet, a row in the unbacked/backed table and in the error table, and "Temporary parts" no longer says "two parts" for three |
   | 3c.4 | Reserved `.obsm` keys (P9): Feature Extraction rejects `spatial_canonical` and `spatial` in its key check; Object Classification leaves them out of its feature-matrix choices and refuses to register them | implemented on 2026-11-05 (`RESERVED_OBSM_KEYS` and `validate_feature_key_not_reserved(...)` in `core/feature_matrix_metadata.py`); the feature-extraction core also checks first, so an existing `obsm["spatial"]` is never removed |
   | 3c.5 | Two-pass reload preparation (P10): the reload participant protocol gets `table_reload_blocker(request)`, which returns a reason or `None` without side effects; app state first asks every participant and raises one error with every reason before any participant acts, then calls `prepare_for_table_reload(request)`, which only invalidates. Feature Extraction's controller returns its "still writing" message as its blocker and no longer raises; the Object Classification and Spatial Query widgets return `None`. Tests in the app-state, feature-extraction, persistence-controls and Spatial Query widget modules. A developer document, `docs/development/table_state.md`, describes table state, Write, Reload and the participants. 3c.6 replaces the participants' blocker with app state's busy state | implemented on 2026-11-06, before 3c.6 and 3c.2b; 3c.6 then replaced the participants' blockers with the busy state, and the first pass became the busy check |
   | 3c.6 | A table is busy while a job writes it (P11), extending 3c.5: app state tracks busy marks per table (`mark_table_busy(...)` returning a token, `release_table_busy(token)`, `table_busy_reasons(...)`) and emits `table_busy_state_changed`. The persistence controls disable Write and Reload while the table is busy, with the reason as tooltip. `write_table_state()` and the first pass of `prepare_for_table_reload()` raise `TableBusyError` as guards, and the controls show it. The reload participant protocol goes back to `prepare_for_table_reload(request)` only: the blocker methods and `check_table_persistence(...)` go. Feature Extraction's controller marks the table busy in `calculate()` and releases it in `_clear_job()`; it stops being a reload participant, so its deferred release and the widget's pending unregister go. Details from the readiness check: the controller gets a required `table_busy_marker=` keyword typed as a narrow `TableBusyMarker` protocol (`mark_table_busy`, `release_table_busy`), which app state implements, so the controller does not depend on app state; required, unlike the callbacks, because a controller without it would silently drop the guard (the widget passes app state; the unit tests construct the controller through a helper that passes a throwaway app state); app state also offers `is_table_busy(...)` and `require_table_not_busy(...)`, which formats 'Table "T" is busy: …' in one place for both guards and the tooltip; busy marks survive loading another SpatialData (the job keeps the old object alive, so its release still finds the mark); Reload needs no check before its dialog, since the button is disabled while busy and the first pass guards a job that starts meanwhile. After the widget closes, superqt keeps the running worker alive, and its signal connections keep the controller alive until `finished`, so the release is reached. Tests: busy marks and the signal in app state; buttons disabled with the reason, and enabled again on release; the guards; Feature Extraction releases the mark when its job returns, errors, or finishes after the widget closed; the 3c.5 tests without the blocker. `docs/development/table_state.md` follows | implemented on 2026-11-06 (busy state instead of polled blockers), after 3c.5. Also in 3c.6: the Spatial Query widget invalidates its Run on the `reloaded` event too, so every participant's hook is a defense (`TableReloadParticipant`); `PersistenceController.write_table_state()` leaves a table without unsaved changes as it is instead of raising `ValueError` |
   | 3c.7 | No job starts on a busy table (P11, point 6): `TableBusyMarker` gains `table_busy_reasons(...)`; Feature Extraction's `calculate()` and Spatial Query's `start_spatial_query()` refuse while another job keeps the table busy; their widgets disable Calculate and Run with 'Table "T" is busy: …' and refresh on `table_busy_state_changed`; the Spatial Query widget checks the centers again when the table is no longer busy. Tests in the controller, widget and status-card modules; the multi-widget reload test accepts Spatial Query's Run before Feature Extraction's job starts. `docs/development/table_state.md` follows | implemented on 2026-11-06, after 3c.2b. Not covered: a query-only Run does not mark the table busy, and jobs on different tables of one store are not serialized (design point 6) |
   | 3d.1 | Object Classification without scanning the matrix (3d design points 1-8 below): the summary reads only `obs` and `uns`, with the column count from `feature_matrix_n_features(...)`; the job carries the matrix object and the row positions instead of copies; the worker reads its rows with `read_matrix_rows(...)`, skips rows without finite features, checks the labeled count and the classes again, and returns an "ineligible" result with its reason when they no longer suffice; the result carries the counts and the predicted row positions, which the prediction write and `uns["classifier_config"]` use (point 5); the status card and the messages report the rows in the scope before a job, and the rows used and skipped after it. Tests: the controller, widget and multi-widget tests listed with the design points change; new tests as listed with the readiness check | implemented on 2026-11-06. `ResolvedClassifierScope` keeps one position array, named `table_row_positions`, now every row of the scope (the old "raw" positions); `raw_table_row_positions` and the eligible counts go. The summary uses `feature_matrix_n_features(...)`; its two "no usable rows" blockers move to the worker. A leaner contract between the main thread and the worker (settled the same day, after a review of the first version): the worker gets only a `ClassifierWorkerInput` (the matrix, the labeled training rows with their labels, the prediction rows, and the region of every row it reads, so it can name a labels element whose rows it all skipped; its docstring states the accepted risk of point 7) and returns only what it computed, a `ClassifierTrainingResult` (predictions with `predicted_row_positions`, the labeled count and classes used, the skipped counts, the estimator) or a `ClassifierIneligibleResult` (reason, labeled count and classes used). `ClassifierJob` stays on the main thread (job ID, feature key, table, summary, worker input) and is bound to the worker's signals, so the handlers combine it with the result; the summary never reaches the worker, the results echo nothing, and `labels_name`, `model_params` and `_active_job` go. `_apply_ineligible_state(feature_key=, table_name=, summary=)` serves both kinds of ineligibility. `_build_classifier_config(...)` takes `n_predicted_rows`. Tests: 19 existing tests updated (a `_fake_training_result(...)` helper in `test_classifier.py`; two tests wrap a fake in a real thread worker to keep the job); new: the summary and the job read no values (a Dask matrix that fails when computed), the worker skips rows without finite features (dense, CSR, CSC), the labeled rows are checked again after skipping, a prediction scope without features is reported by the worker and names its labels element, and the success status names it too |
   | 3d.2 | Headless `apply_classifier(...)` the same way (3d design points 6 and 9-11 below): `_read_rows_with_finite_features(...)` moves to the core, and `apply_classifier(...)` reads only the prediction rows through it, skipping rows without finite features; the estimator width check uses the shape; `normalize_feature_matrix(...)`, then without callers, is deleted; `uns["classifier_apply_config"]` drops `applied`, `table_name`, `n_predicted_rows` and `n_skipped_feature_invalid_rows`. Tests: headless with a lazy matrix, reading only the prediction rows, and the apply-config assertions | implemented on 2026-11-07: `_read_rows_with_finite_features(...)` is in `core/object_classification/classifier.py`, and the widget's worker and headless `apply_classifier(...)` both read through it; `_validate_estimator_matches_feature_matrix(...)` takes the column count; `normalize_feature_matrix(...)` is gone, and with it `feature_matrix_metadata.py`'s now-unused `numpy` import and `issparse` fallback; `_build_classifier_apply_config(...)` writes the seven kept fields. Tests: the apply-config assertions check the exact field set; new: a lazy matrix predicts the same as a NumPy one, and only the prediction rows are read (a Dask matrix whose chunk for `blobs_labels_2` fails when computed) |
   | 3d.3 | One shared classifier config (3d design points 12 and 13 below): `uns["classifier_config"]` maps each prediction class column to one entry, written by the widget (`"pred_class"`) and by headless `apply_classifier(...)` (its `pred_class_column`): `source`, `pred_class_column`, `pred_confidence_column`, `feature_key`, `prediction_regions`, `prediction_timestamp`, `prediction_scope` (widget), `classifier_path` (headless), `reason`, and `model_provenance` (`None` when nothing was trained); whoever writes a column replaces its entry; `trained` and `uns["classifier_apply_config"]` go. The export bundle's `source_classifier_config` becomes `model_provenance`, the entry's provenance as is; the bundle schema version becomes 2, checked before the payload fields. The reload status reads the `"pred_class"` entry and shows the untrained state, asking the user to train, when the entry is missing or headless wrote it. The headless docs say that the widget does not see headless run on the viewer's own SpatialData object. From the readiness check (2026-11-07): a failed training writes no entry, since it wrote no column (its status shows the error; the metadata-only table event and its `"object_classification_metadata"` source go); one entry builder in the core, used by both writers, takes every field as a required keyword; the model snapshot keeps `model_provenance` instead of the whole config, and drops its unread `trained_at`; `prediction_timestamp` is the training result's `trained_at` (the time of writing when nothing was trained) for the widget, and `applied_at` for headless. Tests: the entry's field set for both writers; the bundle's `model_provenance`, and a version-1 file refused; each writer replaces the other's `"pred_class"` entry and leaves other entries alone; a backed headless apply persists its entry; a reload after a headless apply, and of a table never trained, shows the untrained state with export unavailable; training again after a headless apply keeps the headless predictions outside the widget's prediction scope | implemented on 2026-11-07: `build_classifier_config_entry(...)`, `store_classifier_config_entry(...)` and `get_classifier_config_entry(...)` in `core/object_classification/classifier.py`; the controller builds its entry with `_build_widget_config_entry(...)` and `_build_model_provenance(...)`, and `ClassifierTableStateChangeSource` keeps only `"object_classification_inference"`; `CLASSIFIER_APPLY_CONFIG_KEY` and `_build_classifier_apply_config(...)` are gone; README and the headless docstrings describe the entry and the Reload. Tests: the widget's entry field set; headless's entry, also through the store with its `None` fields; a custom-column apply leaves the widget's entry alone, and a default-column apply replaces it but not another column's; the widget replaces headless's `pred_class` entry, keeps a custom column's entry and the predictions outside its scope; a failed training writes no entry and no table event; a reload without a widget entry (never trained, headless wrote `pred_class`) asks to train with export unavailable; a version-1 bundle file is refused |
   | 3d.4 | The confidence column follows from the class column (3d design point 14 below): `f"{pred_class_column}_confidence"`, through one helper, `prediction_confidence_column(...)`; the default becomes `pred_class_confidence` (`PRED_CONFIDENCE_COLUMN`); the four headless functions and the core `apply_classifier(...)` lose `pred_confidence_column`, and the internal writers derive it; a table with the old `pred_confidence` is refused by the existing pair checks, without migration; the color-by tooltip, `viewer_styling.py`'s docstrings and the README follow. Tests: the tests that name the confidence column follow (8 literal names in the widget and Spatial Query modules, 16 parameter uses, mostly headless); new: headless writes `<class column>_confidence`, and a custom class column touches neither `pred_confidence` nor `pred_class_confidence` | implemented on 2026-11-08: `prediction_confidence_column(...)` in `core/object_classification/classifier.py`; `build_classifier_config_entry(...)` derives the entry's `pred_confidence_column` too, so the entry cannot record a mismatched pair either; the color-by tooltip uses `COLOR_BY_PRED_CONFIDENCE`. Tests: the custom-column headless test now starts from the widget's predictions and entry, and checks that the confidences go to `headless_class_confidence`, that `pred_class`, `pred_class_confidence` and the widget's entry are untouched, and that no `pred_confidence` appears; the backed and feature-extraction headless tests use the derived names; the literal names in the widget (including the status bar) and Spatial Query tests follow |

   3c.1-3c.6 are committed separately, 3c.2 in its three parts (a, b, c, in
   that order), with 3c.5 and then 3c.6 between 3c.2a and 3c.2b: design
   point 8 relies on their contract, and 3c.2b adds a second job that writes
   the table; 3c.7 follows 3c.2b. 3d is committed in four parts, 3d.1, 3d.2,
   3d.3 and 3d.4. 3c.1 comes first, because the classifier
   summary in 3d relies on its shape-only check. 3c.4 comes before 3c.2, because a feature-matrix
   record under a reserved key would make `add_canonical_centers` reject the
   table. 3c.2 comes next: until it lands, Spatial Query rewrites harpy 0.5.3's
   version-2 canonical centers as version-1 records, which harpy rejects. 3c.3
   and 3d are independent of 3c.2 and 3c.4. 3c.3 replaces a safeguard that computed the matrix after
   each extraction, dropped on 2026-10-30 (see P1, "Resolution"). 3c.2
   replaced, on 2026-11-01, a plan to route Spatiato's own canonical-center
   code through the matrix helpers; harpy 0.5.3 unblocked it on 2026-11-05.

   Facts for the helpers, checked at runtime (anndata 0.13.4, dask 2026.8.0,
   zarr 3.2.1) on matrices written with anndata and read back through harpy's
   reader in lazy and backed mode:
   - sparse without reading: `issparse(x._meta)` for Dask,
     `isinstance(x, anndata.abc.CSRDataset)` (or `CSCDataset`) for backed
     handles; `issparse(x)` is False for both;
   - rows: `x[positions]` with a sorted NumPy position array works for every
     representation (Dask then needs `.compute()`) and returns NumPy or SciPy,
     in the input's sparse format. Pass a NumPy array, not a list: a list makes
     `CSRDataset` read the whole matrix. `CSCDataset` always reads the whole
     matrix for a row selection (rare: harpy writes dense `obsm` and CSR `X`);
   - one column: `x[:, j:j+1]` is the only form that stays 2-D for every
     representation; `x[:, j]` differs per representation and fails on a
     multi-block Dask array of `csr_array` blocks. For the rows of one labels
     element, read the rows first for matrices stored by rows (dense, CSR)
     and the column first for CSC (see P2);
   - never `np.asarray(...)` on sparse representations: for SciPy sparse,
     sparse Dask and `CSRDataset` it silently returns a 0-d object array.
     Today's `normalize_feature_matrix(...)` relies on it;
   - finite rows: `np.isfinite(m).all(axis=1)` for dense; through `indptr`
     and `.data` for CSR; CSC needs `.tocsr()` first;
   - computing Dask arrays and reading backed handles in worker threads works,
     also concurrently;
   - shape and dtype without loading must keep the checks
     `normalize_feature_matrix(...)` makes today: 2-D (a 1-D matrix counts as
     one column), the row count, and a numeric dtype (otherwise "invalid
     matrix"). DataFrame entries have `.dtypes`, not `.dtype`; `CSRDataset`
     has no `ndim`.

   Design points for 3d (settled on 2026-10-30). Today the summary's
   "eligible rows" (scope rows with finite features) feed more than scoping:
   1. **What is reported before a job.** The status card ("Prediction rows:
      N"), the launch status ("… from N eligible rows …") and the "Need at
      least …" messages use eligible counts. Decision: before a job, report
      the rows in the scope; the rows used and skipped are reported after the
      job.
   2. **The counts in `uns["classifier_config"]`.** `n_training_rows` and
      `n_predicted_rows` come from the summary today
      (`_build_classifier_config(...)`); they are persisted, and "Loaded
      predictions for N objects" reads them back. Decision: the job result
      carries the counts and the predicted row positions, and both the config
      and the prediction write (`_write_classifier_predictions(...)`, whose
      fill positions are already a subset of the cleared ones) use them.
   3. **Ineligibility found in the worker**: too few labeled rows, or fewer
      than two classes, after skipping; or no usable rows in the prediction
      scope. Today ineligibility is handled on the main thread by
      `_apply_ineligible_state(...)` (clears the scope's predictions, writes
      the config with `trained=False`). Decision: the worker returns an
      "ineligible" result with its reason, not an exception, and the main
      thread passes it to `_apply_ineligible_state(...)`.
   4. **Data structures.** Decision: `ResolvedClassifierScope` drops
      `table_row_positions`, `n_eligible_rows` and
      `n_excluded_feature_invalid_rows`; `ClassifierJob` carries the matrix
      object and the positions instead of arrays copied on the main thread;
      `ClassifierJobResult` gains the counts and the predicted positions. The
      comments that state today's snapshot contract change with it
      (`_prepare_classifier_job(...)`, `normalize_feature_matrix(...)`, the
      `ClassifierJob` docstring). Safe: Spatiato never writes into `obsm`
      arrays in place, and the job keeps its own reference to the matrix.

   A behaviour change that follows: labeled counts and classes are counted
   over all labeled rows in the scope, not only those with finite features,
   which is why the worker checks them again. Tests: about 12 controller
   tests, 3 widget tests and 1 multi-widget test build fake results from
   `scope.table_row_positions` or assert eligible counts, and change with
   point 4. The headless tests keep their field names
   (`n_predicted_rows`, `n_skipped_feature_invalid_rows`). No test covers the
   sparse finite check or most of the count messages today.

   Readiness check for 3d (2026-11-06, against the code after 3c.7). The
   design above still matches the code: the summary reads the whole matrix on
   every refresh, through `normalize_feature_matrix(..., copy=False)` and the
   finite-row mask; the job copies its rows on the main thread; headless
   `apply_classifier(...)` reads the whole matrix. 3c.1's
   `feature_matrix_n_features(...)` gives the summary its shape-only checks,
   and 3a's `read_matrix_rows(...)` lets the worker and headless read only
   their rows; `_get_finite_feature_row_mask(...)` works on what it returns
   (NumPy, or SciPy CSR/CSC). Nothing in `src` writes into an `obsm` array in
   place, so point 4's premise holds. Settled the same day:
   5. **What the counts in `uns["classifier_config"]` mean**, which point 2
      left open. After 3d the worker reads only the labeled training rows and
      the prediction rows, so it cannot count the finite unlabeled training
      rows that `n_training_rows` counts today. Decision:
      - `n_training_rows`: the rows in the training scope, as reported before
        a job (point 1);
      - `n_labeled_objects`: the labeled rows used for training, after
        skipping; when the summary blocks Train before a job, the labeled rows
        in the scope;
      - `n_predicted_rows`: the rows that received a prediction, and 0 when
        nothing was trained (today an ineligible config stores the eligible
        prediction count there).

      Only `_update_status_from_reloaded_table(...)` reads a count back
      ("Loaded predictions for N objects"), and only when `trained` is true.

      Revised on 2026-11-07, after a review of every field: no field is kept
      only for compatibility with older stores or exported bundles. A field
      stays when a reader uses it, or when it records how the model behind
      the predictions was obtained (the table holds no estimator). The config
      keeps 13 fields:
      - for the reload status: `trained`; `reason`, `None` when trained;
        `feature_key`; `prediction_scope`; `prediction_regions`;
      - how the model was obtained: `model_type`, `rf_params`,
        `training_scope`, `training_regions`, `n_features`, and, describing
        the trained model only (`None`, 0 and `[]` when nothing was trained),
        `training_timestamp`, `n_training_labels` (renamed from
        `n_labeled_objects`: the labeled rows the model was trained on) and
        `class_labels_seen`.

      Dropped: `table_name` (the config lives in that table, so the reload
      status's table check goes too); `eligible` (always equal to `trained`,
      so its reload branch goes); `roi_mode` (a constant nobody reads);
      `n_training_rows` (misleading, read by nobody); `n_predicted_rows` (the
      reload status counts the predictions in `obs` instead). The export
      bundle loses its six properties that read config fields nobody uses;
      its `model_type` check stays. `ClassifierIneligibleResult` keeps only
      its reason. 3d.3 then reshapes this config into one entry per
      prediction column, with the provenance fields nested (design point 12).
   6. **`normalize_feature_matrix(...)` goes.** Its three callers (the summary,
      the job and headless `apply_classifier(...)`) stop using it. The test
      that wraps it ("the preparation summary should not construct worker
      feature arrays") becomes "the summary reads no matrix values", with a
      Dask array that fails when computed.
   7. **The busy state (3c.6, 3c.7) changes nothing here.** Classifier jobs
      still only calculate: they do not mark the table busy, and Train does not
      wait for a busy table. The risk accepted above, a job reading a key that
      Feature Extraction rewrites, stays accepted; it becomes a little more
      likely, since the read moves into the worker. The code says so: the
      `ClassifierJob` docstring states that this is accepted on purpose and how
      each outcome is caught (a running job's result is discarded, a finished
      job's predictions are flagged as stale, a failed read becomes an error
      status), and `invalidate_for_feature_matrix_overwrite(...)` points to it.
   8. **An ineligible result from the worker** (point 3) needs
      `_apply_ineligible_state(...)` to take the worker's reason and counts,
      not only the job's summary.

   3d is split in two parts, committed separately: 3d.1, the widget path;
   3d.2, headless `apply_classifier(...)` and the removal of
   `normalize_feature_matrix(...)`, which needs both. Tests: the about 16
   tests above change; new tests cover the summary reading no matrix values,
   the worker skipping non-finite rows (dense, CSR and CSC), the counts
   checked again after skipping and the ineligible result that follows, the
   message naming a labels element whose rows were all skipped, the config
   counts, and headless with a lazy matrix and skipped rows.

   Readiness check for 3d.2 (2026-11-07, against the code after 3d.1).
   Headless `apply_classifier(...)` still reads the whole matrix
   (`normalize_feature_matrix(..., copy=False)`, then the finite-row mask over
   every row of the table) and is the last caller of
   `normalize_feature_matrix(...)`; its estimator width check needs only the
   shape. Settled the same day:
   9. **One place reads feature values, in the core.**
      `_read_rows_with_finite_features(...)` moves from the widget controller
      to `core/object_classification/classifier.py`, next to
      `_get_finite_feature_row_mask(...)` and
      `_predict_classes_and_confidences(...)`; the controller imports it. The
      core and headless do not import from `spatiato.widgets`
      (`test_headless_module_avoids_direct_interactive_classifier_imports`).
   10. **Headless reads only the prediction rows.** `apply_classifier(...)`
       resolves the prediction scope's rows first and reads only those, with
       `_read_rows_with_finite_features(...)`; rows without finite features
       are skipped and cleared, as before. The estimator width check uses
       `feature_matrix_n_features(...)`. `normalize_feature_matrix(...)` is
       deleted, and the `feature_matrix_n_features(...)` docstring states its
       checks without referring to it.
   11. **The apply config follows design point 5's rule** (no field kept only
       for compatibility). `uns["classifier_apply_config"]` keeps
       `apply_timestamp`, `classifier_path`, `feature_key`,
       `prediction_regions`, `pred_class_column`, `pred_confidence_column`
       and `source_classifier_config` (how these predictions were made, and
       with which model). Dropped: `applied` (always `True`), `table_name`
       (the config lives in that table), `n_predicted_rows` and
       `n_skipped_feature_invalid_rows` (derivable from the prediction column
       within `prediction_regions`; `ClassifierApplyResult` still returns both
       to the caller).

   Tests for 3d.2: the existing headless tests for skipped rows and selected
   regions keep their assertions; the assertions on the dropped apply-config
   fields change; new tests: a lazy matrix predicts the same as a NumPy one,
   and only the prediction rows are read (a Dask matrix whose chunk for the
   other labels element fails when computed, if the fixture keeps each
   region's rows contiguous).

   Review of both classifier configs (2026-11-07, after 3d.2). They follow
   the same rule (design points 5 and 11), but they do not line up:
   - the bundle copies the whole training config, whose reload-status
     fields mean nothing there (`trained` is always true and `reason` always
     `None`; `prediction_scope` and `prediction_regions` describe what the
     training session predicted), so `prediction_regions` appears twice in
     the apply config, with two meanings;
   - when the widget and headless both write `pred_class`, the last writer
     owns the predictions in `obs`, but the other's config stays and
     describes predictions that are gone: after a headless apply, the
     widget's reload status would report the headless predictions as its
     own;
   - the one apply config describes one headless apply only, so a run on
     custom columns replaces the description of a run on the default
     columns.

   Settled the same day, as 3d.3 and 3d.4. A first version kept both
   configs: the bundle kept a subset of the training config, each writer
   removed the other's config for the default columns, and headless needed
   a transient fix to delete a removed config from the store. The shared
   config below replaced it: no writer removes anything, so nothing needs
   deleting.
   12. **One `classifier_config`, keyed by prediction column, shared by the
       widget and headless.** `uns["classifier_config"]` maps each
       prediction class column to one entry. The widget writes the
       `"pred_class"` entry; headless `apply_classifier(...)` writes the
       entry of its `pred_class_column`. Whoever writes a column replaces its
       entry, so one entry describes each column by construction, and
       `uns["classifier_apply_config"]` goes. An entry holds:
       - `source`: `"widget"` or `"apply_classifier"`;
       - `pred_class_column` and `pred_confidence_column`, so the entry
         describes itself (the key repeats the class column);
       - `feature_key`: the matrix the predictions were computed from;
       - `prediction_regions`: the regions this writer cleared and predicted
         in these columns;
       - `prediction_timestamp`;
       - `prediction_scope`: the widget's scope mode, `None` for headless;
       - `classifier_path`: the exported file headless applied, if given;
         `None` for the widget;
       - `reason`: why nothing was trained, `None` otherwise;
       - `model_provenance`: how the model was obtained, `None` when nothing
         was trained: `feature_key` (the key the model was trained on),
         `model_type`, `rf_params`, `training_scope`, `training_regions`,
         `n_features`, `training_timestamp`, `n_training_labels` and
         `class_labels_seen`.

       `trained` goes: `model_provenance` being `None` says the same. This
       reshapes the 13 fields of design point 5: the reload-status fields
       stay at the entry's top level, and the provenance fields move into
       `model_provenance`.

       The export bundle's `source_classifier_config` becomes
       `model_provenance`, the widget entry's provenance as is, so there is
       no subset to select; the bundle's readers (`feature_key`,
       `model_type`, `class_labels_seen`) read it. Not `model`: next to the
       bundle's `estimator`, the fitted model itself, `model` would read as
       the model rather than its description. The payload key changes too, so
       `CLASSIFIER_EXPORT_SCHEMA_VERSION` becomes 2, and
       `_payload_to_bundle(...)` checks the version before the payload
       fields: a file exported before fails with the version error. Headless
       copies the bundle's `model_provenance` into its entry.

       Nothing is deleted from the store. The widget's table event keeps the
       `("uns", ("classifier_config",))` path, and headless persistence
       writes the whole `classifier_config` key, which
       `write_table_prediction_state(...)` already does for a key in memory.

       Old stores are not migrated, and get no special error messages: a
       flat `classifier_config` has no `"pred_class"` entry, so the widget
       reads it as untrained, and new entries are added next to its old
       fields; an old `classifier_apply_config` stays, unread.
   13. **After a headless apply, the widget behaves as if nothing was
       trained.**
       - The reload status reads only `classifier_config["pred_class"]`.
         When that entry is missing, or headless wrote it, the reload status
         marks the classifier stale and asks the user to train ("no trained
         model for this table"), instead of reporting "model is up to date",
         which it does today when the config is missing, also for a table
         that was never trained. The reload already clears the model
         snapshot, so export is unavailable. Binding needs no change: a new
         selection already marks the classifier stale. The headless
         predictions stay in `obs` as data, because binding and reloading do
         not change prediction state.
       - The widget does not see headless run on the viewer's own SpatialData
         object: headless notifies no app state, and making it do so would
         couple the core to the UI. Accepted, and documented with the
         headless API: run headless on the store or on another SpatialData
         object, and Reload to see its result.
       - Training again after a headless apply rewrites only the widget's
         prediction scope and replaces the `"pred_class"` entry. Predictions
         headless wrote outside that scope stay in `obs`, described by no
         entry, and the reload status reports a labels element outside the
         scope as stale ("loaded predictions do not cover labels element
         ..."). The same holds the other way round, for widget predictions
         outside a later headless apply. Accepted: clearing the default
         columns on every row instead would delete output the user asked
         for.
   14. **The confidence column follows from the class column** (3d.4):
       `f"{pred_class_column}_confidence"`, as the palette is
       `f"{pred_class_column}_colors"`. Configuring it separately adds code
       for little gain, and lets two prediction columns write one confidence
       column. Today that happens by default: headless given only a custom
       `pred_class_column` writes its confidences to `pred_confidence`. It
       then either fails the pair check (`pred_confidence` exists, the
       custom class column does not), or leaves `pred_confidence` without
       `pred_class`, which the widget refuses to bind. The widget's default
       becomes `pred_class_confidence` (`PRED_CONFIDENCE_COLUMN`, which is
       also a "color by" choice, shown by its name, and one of Spatial
       Query's reserved columns), and the four headless functions lose
       `pred_confidence_column`. Separate from 3d.3, whose entries record
       whichever confidence column is in use.

       Readiness check (2026-11-08), settled the same day:
       - One helper, `prediction_confidence_column(pred_class_column)`, in
         `core/object_classification/classifier.py`;
         `PRED_CONFIDENCE_COLUMN` is
         `prediction_confidence_column(PRED_CLASS_COLUMN)`.
       - The internal writers `_write_classifier_predictions(...)`,
         `_clear_predictions_for_row_positions(...)` and
         `_update_table_prediction_state(...)` drop their
         `pred_confidence_column` parameter and derive it, so no caller can
         pass a mismatched pair.
       - The core `apply_classifier(...)` drops the parameter and its "must
         be different" check, which can no longer fail.
         `ClassifierApplyResult.pred_confidence_column` and the entry's field
         stay: they report the column.
       - Accepted: a class column named like `a_confidence` is also the
         confidence column of a class column `a`. No guard: it needs two
         runs with deliberately chosen names.
       - Accepted, without migration: a table with `pred_class` and the old
         `pred_confidence` has `pred_class` without `pred_class_confidence`,
         so the existing pair checks refuse it. The widget shows a binding
         error naming the missing column, and headless cannot write the
         default columns, until the user drops `pred_class` or renames
         `pred_confidence` by hand. Such a table's stale `pred_confidence`
         shows as an ordinary column in Spatial Query's choices.
       - The color-by tooltip uses the constant instead of the literal
         `"pred_confidence"`; `viewer_styling.py`'s docstrings and the
         README follow.

   Readiness check for 3c.2 (2026-11-05, against harpy 0.5.3). Today the
   centers worker computes without changing anything, and the main thread
   installs the result (`apply_canonical_cache_update(...)`). With harpy,
   `add_canonical_centers(...)` computes and installs in the worker, and, for
   backed SpatialData, writes to the store there. What uses Spatiato's
   canonical code, and what replaces it:

   | Where | Today | With harpy |
   |---|---|---|
   | widget: inspection (labels change, table change, reload, every Run) | `inspect_canonical_cache(...)`, a report with five states | `validate_canonical_centers(...)`: absent, valid, or invalid with harpy's message |
   | widget: labels choices | Spatiato's source signature, 2-D only | scale0 dims `(y, x)` and an integer dtype |
   | controller: valid centers | read the cache on the main thread | read the labels element's rows with `read_matrix_rows(...)` |
   | controller: otherwise | the worker computes without changing anything; the main thread applies the regional update | the worker runs `add_canonical_centers(...)` and changes the table |
   | query and its models | `CanonicalCentersResult` with a binding and a source signature; the 2-D check reads `source_signature.dims` | a small "region centers" type: table, labels element, rows, instance IDs, centers |
   | annotation: the live binding and the provenance check at Apply | harpy-style binding, digest, `n_obs`, per-region metadata | rows and instance IDs from the table annotation |
   | status card | texts for valid, absent, partial, stale and invalid | three states |

   Spatial Query is a reload participant but never blocks a reload: its
   `prepare_for_table_reload(...)` only cancels its own work, which was safe
   while its worker did not change the table. Cancelling calls `quit()`, which
   would now lose the result of a worker that did change the table.

   Design points for 3c.2 (settled on 2026-11-05):
   1. **The centers worker follows Feature Extraction's job contract**
      (decisions 9 and P7 step 2, points 7-9): one centers job at a time; a
      cancel detaches the job instead of calling `quit()`; a detached or late
      job still publishes its table change when the same SpatialData is still
      selected; the table is busy, so Write and Reload wait, until the job
      finishes, also after the widget closes (3c.6's busy mark, as in Feature
      Extraction).
   2. **Events and persistence:** `"created"` when the table had no canonical
      centers at launch, `"rebuilt"` otherwise; the regions are every labels
      element of the table, since harpy recomputes all of them; backed
      SpatialData records a persisted change with the dirty snapshot taken at
      launch, unbacked SpatialData records a mutation.
   3. **Status card:** "Centers for "X" will be reused." (valid); "Centers for
      every labels element of table "T" will be calculated before querying."
      (absent); "... will be recalculated before querying." (invalid), with
      harpy's message as the tooltip, also for Spatiato's version-1 records.
   4. **Provenance check at Apply:** the centers are still valid, the labels
      element's rows and instance IDs are unchanged (from the annotation), and
      its centers are unchanged. The digest, `n_obs` and source-signature
      comparisons go: harpy no longer has them.
   5. **Persistence tests of the canonical cache** (`test_persistence.py`,
      "round trips as one dirty unit" and "failed second write") are deleted:
      Spatiato no longer writes these components itself. Check first that the
      generic nested-`uns` persistence they exercised stays covered (Spatial
      Query's palette paths, Object Classification's config).
   6. **Concurrent harpy writes** of Feature Extraction and Spatial Query to
      the same store stay unprotected for now, as Feature Extraction already
      accepts across controllers: 3c.6's busy state makes Spatiato's Write and
      Reload wait for a job, but does not stop another job from launching.
      Revised on 2026-11-06 by 3c.7: neither starts a job on a busy table, so
      two jobs no longer write one table at the same time. Still unprotected:
      jobs on different tables of one store, which share the store's
      consolidated metadata, and a Run whose centers are already valid, which
      only queries and does not mark the table busy, so Feature Extraction can
      start meanwhile and the Run then applies its annotation on the main
      thread while that job writes.
   7. **In line with Feature Extraction** (agreed on 2026-11-05). Points 1, 2
      and 6 copy Feature Extraction's contract for writing a table through
      harpy: the harpy call installs (and writes the store) in the worker, one
      job per controller, detaching instead of `quit()`, publishing for
      detached jobs, the busy mark released when the worker finishes,
      persisted change against mutation with the launch snapshot. Four
      differences are intentional:
      - no confirmation dialog and no unsaved-changes block: Feature Extraction
        protects values that cannot be recomputed (other labels elements'
        features, unsaved edits harpy's store-based merge would drop), while
        canonical centers are derived data that harpy replaces entirely and
        that can be recomputed at any time;
      - `"rebuilt"` instead of Feature Extraction's `"updated"` for a full
        replacement: Feature Extraction uses `"updated"` because Object
        Classification only invalidates its classifier on it, and no consumer
        depends on `"updated"` for canonical centers;
      - the event's regions are every labels element of the table, since
        harpy recomputes all of them, not only the ones the job targeted;
      - detachment is explicit, on cancel, without reattachment (added on
        2026-11-06): Feature Extraction derives detachment from the selection
        and reattaches when the inputs come back, while a Spatial Query Run is
        a one-shot action that never comes back once invalidated.

      The job contract is mirrored in Spatial Query's controller, not shared:
      3c.2b is already the large part, and changing Feature Extraction's
      controller inside it adds risk. Extracting a shared "table-writing job"
      helper (job slot, busy mark, publishing for detached jobs; detachment
      itself differs, see above) is a follow-up, once both controllers exist
      and are known to stay identical.
   8. **Reload participants** (settled on 2026-11-05, from the readiness check
      for 3c.2b; revised on 2026-11-06 for 3c.6's busy state). The controller
      marks the table busy while its centers job can still write it, and
      releases the mark when the worker finishes, also when the widget closes
      first, as Feature Extraction's controller does. So Write and Reload of
      the table wait for the job, and the controller is not a reload
      participant. The widget stays a participant, to invalidate a run in its
      query phase. With 3c.5's two passes, whose first pass is the busy check,
      a blocked reload changes nothing, in Spatial Query or elsewhere.
   9. **An attached operation against a job in flight** (settled on
      2026-11-06, from the second readiness check for 3c.2b). Today the
      controller has one slot (`_active`), and `is_running` means "an
      operation the widget is attached to". The widget relies on that:
      `_on_controller_state_changed(...)` raises when the controller runs
      while no status card is set, and Run is enabled only when the controller
      is not running. A cancelled centers job is detached but keeps writing,
      and `_invalidate_run()` clears the status card, so both meanings are
      needed, as in Feature Extraction (`has_job_in_flight`):
      - `is_running` keeps meaning an attached operation, so the status-card
        invariants hold;
      - a new `has_job_in_flight` means a centers job is still writing,
        attached or detached;
      - Run is enabled only when neither holds (one centers job at a time,
        point 1);
      - without a Run, the configuration status says "Waiting for the previous
        center calculation to finish writing." until the job finishes, and the
        controller's state callback after `finished` refreshes it.

   Also part of 3c.2b (found in its readiness check on 2026-11-05):
   - the two canonical component paths (`obsm/spatial_canonical` and
     `uns/spatial_coordinates/spatial_canonical`), today `CANONICAL_CACHE_PATHS`
     in `canonical.py`, move to `canonical_centers.py`, for the event and the
     dirty snapshot;
   - the table change is published through a controller callback of its own,
     whenever the job's SpatialData is still selected (point 1), apart from the
     run: today `_on_centers_ready(...)` drops the event when the run no longer
     matches the selection. A cancelled centers job reports that the centers
     are still being written, and the Write description becomes "spatial
     annotations", since Spatiato no longer writes the centers;
   - three tests change on purpose: the controller test that expects `quit()`
     on cancel, and the two multi-widget tests that treat canonical centers on
     backed SpatialData as a dirty mutation that Spatiato writes;
   - `docs/development/table_state.md` follows: Spatial Query's controller
     marks the table busy while its centers job writes, and its canonical
     centers become a persisted change on backed SpatialData, installed by
     harpy in the worker;
   - accepted cost: the status check runs on the main thread when the labels
     element or table changes, after a reload, on every Run and at Apply. For a
     lazy backed matrix each check reads all centers (24 bytes per row).

   Also part of 3c.2b (found in its second readiness check on 2026-11-06,
   after 3c.5 and 3c.6):
   - the controller takes a required `table_busy_marker=` keyword
     (`TableBusyMarker`), as Feature Extraction's controller does; its seven
     unit tests construct it through a helper that passes a throwaway app
     state;
   - `shutdown()` splits by phase: a centers job detaches and keeps its busy
     mark until `finished`, without `quit()`; a query worker may still be
     quit, since it only calculates. Today `_cancel_active_worker()` quits
     both;
   - the widget's separate "inspection error" state may go:
     `canonical_centers_status(...)` already maps harpy's errors to
     `"invalid"` with the reason, and the other sources (a table that does not
     annotate the labels element, labels that are not 2-D) are filtered out
     by the choice lists. Decided during implementation: the state is dropped
     if nothing produces it;
   - design point 9 (an attached operation against a job in flight) and the
     fourth intentional difference of point 7 (explicit detachment).

   3c.2b is the large part: the types flow from the controller through the
   query to annotation, so the switch cannot be split further without
   temporary adapters.
4. **Swap the table I/O** (P5 writes, P4 reload):
   - writes through harpy's SpatialData adapter
     `hp.tb.io.add_table_components(...)`, or `remove_table_components(...)`
     for deletions only. Spatiato's dirty tracking already knows which paths
     changed, which maps onto explicit components and deletes. The adapter
     writes the store, then reopens every written component into the attached
     table, lazily for matrices. Revised on 2026-11-08 (step-4 design point
     2): the store-only `write_table_components(...)` was the first choice, to
     leave the in-memory values alone, but under D1 a written `obsm` key
     should become a lazy view of the store anyway;
   - not `write_table_updates(...)`: it compares the whole table with the store
     and never deletes, so it does not fit explicit dirty paths (non-goal);
   - reload through `hp.tb.io.read_table_components(...)`, lazily (D1);
   - removed from `core/persistence.py` (416 lines): the direct zarr and
     anndata reads and writes, zarr group-structure handling
     (`_resolve_backed_table_group(...)` and the parent-chain helpers), and
     metadata consolidation. Kept: `TableComponentPath`, dirty tracking in
     `SpatiatoAppState`, the reload checks (`_validate_reload_binding(...)`),
     and restoring the in-memory state when a reload fails.

   Readiness check for step 4 (2026-11-08, against harpy 0.5.3), settled the
   same day. Sub-steps, each committed separately:

   | Sub-step | Content | Status |
   |---|---|---|
   | 4a | Writes through harpy (P5, step-4 design points 1-4 below): `write_table_components(...)` keeps its signature and result, and maps the paths onto `hp.tb.io.add_table_components(...)`, or onto `remove_table_components(...)` when every path is absent; harpy reopens what it writes into the attached table, lazily for `obsm`; `write_table_prediction_state(...)` is deleted and headless calls `write_table_components(...)`; Write reports every error. Tests: a write and a delete in one call; a failed write leaves the store and the attached table unchanged and acknowledges nothing (the existing test patches the harpy call instead of `ad.io.write_elem`); a written `obsm` key is attached lazily afterwards and reads the written values; headless persistence as before | implemented on 2026-11-08: `write_table_components(...)` builds the components and deletions and calls `hp.tb.io.add_table_components(...)` (or `remove_table_components(...)`); a new `_require_backed_table(...)` refuses a table stored anywhere but `tables/<table_name>`; the disk-parent helpers, the consolidation and `write_table_prediction_state(...)` are gone; headless lists one `obs` path and the `uns` keys in memory; the controls catch `Exception` around Write, so their `TableBusyError` import goes; `docs/development/table_state.md` describes the write. Tests: the failed-write test injects a failure while harpy installs the published paths, and checks that the store and the attached table are unchanged and nothing is acknowledged; new: a write and a delete in one call, with the written `obsm` reopened as a Dask view holding the written values; Write reports an `OSError` as feedback. No existing test changed because of a dtype round trip. Renamed the same day: `write_table_components(...)` is now `persist_table_components(...)`, and its result `TableComponentPersistResult`, since it takes paths and makes the store follow memory, unlike either harpy function; it pairs with `reload_table_components(...)` |
   | 4b | Reload through harpy (P4, step-4 design point 5 below): `build_full_table_reload_paths(...)` and `reload_table_components(...)` read with `hp.tb.io.read_table_components(..., mode="lazy", missing="omit")`, and `build_full_table_reload_paths(...)` lists the `obs` columns from memory instead of reading `obs` (design point 5c); `_resolve_backed_table_group(...)`, `_read_disk_path(...)` and the `zarr` and `ad.io` imports go. Tests: a column that exists only on disk still arrives after a full reload (the existing snapshot test); after a full reload, `obsm` entries are Dask arrays holding the values on disk; a key deleted on disk is removed from memory; a request that includes `spatialdata_attrs` (design point 5a); new, since no test covered it: a failing `TableModel.validate` after the install restores the previous `obs`, `obsm` and `uns` objects; the existing reload binding tests unchanged | implemented on 2026-11-09 (decided on 2026-11-08; re-checked on 2026-11-09 against the code after 4a, design points 5a-5d below): `build_full_table_reload_paths(...)` reads `("obsm",)` and `("uns",)` through `hp.tb.io.read_table_components(...)` and lists the `obs` columns from memory; `reload_table_components(...)` reads `("obs",)`, `("uns", "spatialdata_attrs")` and the requested `obsm` and `uns` paths in one call, and takes a requested path under `spatialdata_attrs` from the attrs value; `_resolve_backed_table_group(...)`, `_read_disk_path(...)` and the `anndata`, `zarr` and `Any` imports are gone; `docs/development/table_state.md` describes the reload. Tests: the test of an `obsm` key written straight into the store now also checks that every reloaded `obsm` entry is a Dask array; new: a full reload removes what exists only in memory (an `obsm` key deleted from the store, dirty entries added in memory, an untracked `uns` key); a nested path under `spatialdata_attrs`; the rollback when validation fails after the install. Two tests compare a reloaded Dask array through `np.asarray(...)` instead of `np.array_equal(...)` |

   1. **Path mapping (4a).** Any `("obs", col)` becomes `("obs",)`, the
      whole dataframe, as today. A path present in memory is a replacement;
      an absent one goes to `delete=`, and harpy skips a target absent from
      both memory and store (logged at INFO), as Spatiato does today. The
      call passes `overwrite=True`, and `obs_identity=table.obs[[region_key,
      instance_key]]` only when an `obsm` entry is written. A request with
      only deletions goes to `remove_table_components(...)`, since
      `components` must not be empty. harpy creates missing nested `uns`
      parents as encoded mappings, consolidates the store's metadata, and
      refuses overlapping paths, so `_remove_redundant_uns_descendants(...)`
      stays. It also checks that the attached table's axes match the stored
      ones, and that the `obs` index and the region and instance linkage are
      unchanged, which Spatiato's flows never change. What goes: the three
      disk-parent helpers and the consolidation.
   2. **harpy reopens what it writes (4a).** harpy stages a write before
      replacing the stored key, so writing an `obsm` value that is a lazy
      view of that same key is safe. But the store-only writer does not
      update existing references ("reopen affected data after writing"): if
      the new chunk layout differs, a stale in-memory view could silently read
      the wrong chunks. `add_table_components(...)` reopens every written
      component from the store and installs it in the attached table, inside
      its rollback window, so Spatiato needs no re-attach code of its own;
      this is consistent with D1. A failure also restores the previous
      in-memory objects, not only the store. The first version of this
      decision used the store-only writer and re-attached written `obsm` keys
      in Spatiato; revised the same day. Rare today: on backed data, Feature
      Extraction and Spatial Query write `obsm` through harpy and record a
      persisted change, so `obsm` is dirty only for data changed in memory
      before it was backed. 4b makes lazy views of the store common.
   3. **Write reports every error (4a).** harpy can also raise `TypeError`,
      `KeyError`, `FileNotFoundError`, `OSError`, and `RuntimeError` when its
      own rollback fails. The persistence controls catch `Exception` around
      Write, as they already do around Reload, instead of only `ValueError`
      and `TableBusyError`.
   4. **Headless (4a).** `write_table_prediction_state(...)` is deleted;
      headless calls `write_table_components(...)` with the `obs` path and
      the `uns` keys present in memory.
   5. **Reload reads lazily (4b).** With `read_table_components(...,
      mode="lazy", missing="omit")`, `("obs",)` and `uns` are decoded into
      memory, and `obsm` entries come back as Dask views of the store. For a
      full reload, the disk's keys come from reading the `("obsm",)` root
      lazily (no values computed) and `("uns",)` eagerly. The request always
      includes `("obs",)` and `("uns", "spatialdata_attrs")` for the binding
      check, without overlapping a requested path. The reload binding check,
      `TableModel.validate` and the in-memory rollback stay.

      Re-checked on 2026-11-09, against the code after 4a and harpy 0.5.3:

      - **5a. `spatialdata_attrs` once.** harpy refuses duplicate and nested
        paths in one request ("Component paths must be unique and
        non-overlapping"), and a full reload lists `spatialdata_attrs` among
        the top-level `uns` keys. The request holds `("obs",)`, `("uns",
        "spatialdata_attrs")`, every `("obsm", key)` and every `uns` path not
        under `spatialdata_attrs`; a requested path under `spatialdata_attrs`
        is taken from the attrs value read, with `_read_mapping_path(...)`.
      - **5b. A non-mapping parent on disk fails the reload.** For a nested
        `uns` path whose parent on disk is not a mapping, harpy raises
        `ValueError`, also with `missing="omit"`, where `_read_disk_path(...)`
        treated the path as missing and removed it from memory. Accepted: only
        a store damaged outside Spatiato does this, and the reload fails with
        harpy's message before anything is installed.
      - **5c. Listing the keys for a full reload.** The reload installs only
        the paths it is given, so `build_full_table_reload_paths(...)` also
        lists the keys that exist only on disk. harpy has no call that lists
        keys only, so it reads `("obsm",)` (lazily) and `("uns",)` in one
        call and keeps their keys. It no longer reads `obs` (revised the same
        day): requesting any `obs` path reloads the whole `obs`, the reload
        reports the columns of memory and disk together, and the request's
        paths are used only by the reload (the participants check only the
        dataset and the table), so the `obs` columns in memory suffice. `obs`
        is then read once per reload instead of twice (0.13 s at 1M rows).
        Accepted: `uns` is decoded twice, here and in the reload, and a Dask
        array is built for every `obsm` entry only to list its key, which
        reads only metadata. Considered and not taken: a full reload that
        reads the `obs`, `obsm` and `uns` roots once, in the reload, and
        derives the reloaded paths from what it read. It removes every double
        read, but turns the request's exact paths into "full reload plus dirty
        paths", in `TableReloadRequest` and the persistence controller, for
        little gain.
      - **5d. Memory follows the store, as today.** A full reload lists the
        keys and `obs` columns in memory as well as on disk. One that exists
        only in memory is absent from what harpy returns (`missing="omit"`),
        or from the `obs` read, so it is removed from memory and its dirty
        state is cleared. Reload's Write / Discard / Cancel prompt protects
        what Spatiato recorded as dirty; changes made outside Spatiato's
        tracking, such as in the napari console or a notebook, are dropped
        without asking.

   Accepted: the adapter reopens every written component, not only `obsm`.
   After each Write, the attached `obs`, which Spatiato always writes whole,
   and each written `uns` entry are copies read back from the store (harpy
   always decodes dataframes and `uns` into memory). That costs a read of
   `obs` per Write; the objects are new, and their values come back as
   stored, so a list in `uns` becomes a numpy array (Spatiato has no
   identity-based caches on them); and what the
   encoding normalizes, such as a dtype, shows in memory right after the
   Write instead of after the next Reload, so memory matches the store.
   harpy stages a write in a temporary directory next to the store
   root, so the store's parent directory must be writable and have room for
   the staged copy (Spatiato wrote in place). Until step 5, a reloaded
   `obsm` is lazy while a freshly opened store still gives in-memory arrays;
   consumers handle both since step 3. harpy assumes the table lives at
   `tables/<table_name>`; `resolve_table_path(...)` stays, as a guard and for
   the path Write returns. `docs/development/table_state.md` gets a short
   note that Write and Reload go through harpy, with rollback.
5. **Switch the reader** to `hp.io.read_zarr` (lazy, D1), last: it changes
   every table at once. Both places that open a store call
   `spatialdata.read_zarr` today and both switch, so they give the same
   representation: the napari reader (`_reader.py`) and the Viewer's "Open"
   action (`widgets/viewer/widget.py`).

P6 (cosmetic) can be picked up along the way.

## Problems

### P0. Run the suite on harpy 0.5.2

- Maintainer installs harpy into `.venv` (the environment is managed manually,
  per `CLAUDE.md`): 0.5.0 first, then 0.5.2 with anndata 0.13.4 and
  spatialdata 0.8.0 on 2026-10-25.
- Raise Spatiato's `spatialdata` floor to match: `>=0.7.3` for harpy 0.5.0,
  then `>=0.8.0` for anndata 0.13 (see the second run). Done.
- Run the full suite and triage failures. Expect at least feature-extraction
  tests to observe a lazy `obsm[feature_key]`. Done on 0.5.0 and on 0.5.2
  (below).
- Recheck the anndata 0.12.16 observations under "Matrix consumption" and in
  P2 on anndata 0.13.4. Still to do; the suite does not cover them.
- Verify the unbacked `add_feature_matrix` result type at runtime (the adapter
  docstring describes a lazy merge). Done (below).

First run on harpy 0.5.0 with anndata 0.12.16 (`-n auto`):

- one harpy-related failure,
  `test_compute_features_for_classifier_returns_normalized_target`; see P7;
- two `tests/viewer/tiled_points/napari/test_viewport.py` failures, unrelated
  to harpy: the tests relied on the old tiled-points budget defaults. Fixed by
  passing the budgets explicitly;
- a hang in a parallel worker, unrelated to harpy: a deadlock when a tiled-points
  cache session is garbage-collected while its `QThread` is still finishing.
  Investigating it also showed a crash in real use: removing a tiled-points
  layer while its cache worker is busy lets garbage collection destroy the
  session and its still-running `QThread`, and Qt aborts the process
  ("QThread: Destroyed while thread is still running"). Both fixed in
  `viewer/tiled_points/runtime/cache_session.py`:
  - `start()` adds the session to the module-level keep-alive set
    `_SESSIONS_WITH_RUNNING_THREADS`, so owners may drop it at any time;
  - `_on_thread_finished()` waits for the thread (`thread.wait()`) before
    emitting `closed`, then releases the keep-alive on the next event-loop
    iteration (`QTimer.singleShot(0, ...)`);
  - covered by
    `tests/viewer/tiled_points/runtime/test_cache_session.py::test_started_session_outlives_dropped_owner_until_its_thread_stops`.

Second run on harpy 0.5.2 with anndata 0.13.4 (2026-10-25, `-n auto`):

- with spatialdata 0.7.3: 699 errors and 20 failures, all with the same cause,
  `AttributeError: 'NoneType' object has no attribute 'lower'` in
  `spatialdata/_core/validation.py:153`:
  - anndata 0.13 keeps `X` in memory as `layers[None]`;
    `TableModel.parse(...)` checks every layer key with `key.lower()`, which
    fails on `None`. Every table with an `X` is affected;
  - most of them error at setup, because the shared fixtures build their data
    with spatialdata's own `blobs()` dataset, which calls `TableModel.parse`
    (`tests/conftest.py`). The 20 failures hit the same line inside the test
    body. None of them fail in Spatiato code;
  - fixed in spatialdata 0.8.0 (released 2026-07-02, "Fix adata layers None
    after anndata X unification", scverse/spatialdata#1122 and #1123), which
    skips `None` keys. Spatiato's lock was still on 0.7.3 only because
    `uv lock` keeps versions it already pinned;
- fix: Spatiato's floor raised to `spatialdata>=0.8.0`; `uv.lock` and `.venv`
  moved to spatialdata 0.8.0, which also moves dask from 2026.1.1 to 2026.8.0
  (spatialdata 0.8.0 requires `dask>=2026.3.0`). harpy's own `uv.lock` for
  `v0.5.2` uses the same spatialdata, anndata, and dask versions;
- other spatialdata 0.8.0 changes look low-risk: the main ones are a refactor of
  relational queries and new table validation for them, and Spatiato calls none
  of those APIs (`join_spatialelement_table`, `match_table_to_element`,
  `get_values`, ...);
- with spatialdata 0.8.0: 1 failed, 2384 passed, 1 skipped. The failure is the
  known P7 test, `test_compute_features_for_classifier_returns_normalized_target`,
  with the same harpy error as on 0.5.0. The warnings are zarr, napari and
  spatialdata messages; none are anndata deprecations or come from Spatiato
  code;
- upstream gap, not reported: harpy 0.5.1 and 0.5.2 declare
  `spatialdata>=0.7.3` together with `anndata>=0.13`, a combination that cannot
  parse a table with `X`. harpy's own CI uses spatialdata 0.8.0, so it does not
  hit this.

A green suite does not cover the lazy and backed matrix paths (no tests use
them, see "Matrix consumption"): P1-P5 are still open, and P2 was seen in real
use.

Remaining P0 items, checked on 2026-10-23:

- **`spatialdata` floor**: raised from `>=0.7.2` to `>=0.7.3` (later to
  `>=0.8.0`, see the second run above). The old floor
  could never be installed, because harpy 0.5.0 requires `spatialdata>=0.7.3`;
  `uv.lock` already resolved 0.7.3. `uv.lock` records Spatiato's own specifier,
  so it needs a `uv lock` refresh (resolved versions do not change). Nothing
  else in the repository pins `spatialdata`.
- **Unbacked `add_feature_matrix` result type** (scratch script,
  `blobs_multi_region()` with `sdata.path` `None`, feature `area`):

  | Case | `obsm[feature_key]` |
  |---|---|
  | new key on an existing table | dense Dask array of NumPy blocks |
  | overwrite of the same key | dense Dask array; same values |
  | multi-region table, one region | dense Dask array (lazy merge); rows of the other region are NaN |
  | then the other region into the same key | dense Dask array; no NaN left, first region's rows unchanged |
  | new output table (`output_table_name`) | NumPy; `X` is `None` |

  So on harpy 0.5.0, feature extraction into an existing table leaves a Dask
  `obsm` for unbacked SpatialData as well, and repeated extractions grow the
  lazy graph; see P1.

  Rerun on 2026-10-24 against the #275 branch (the code released as 0.5.2,
  with anndata 0.12.16): every case above, including 5 more overwrites, gives
  NumPy.
- Run on 2026-11-05 against harpy 0.5.3 (anndata 0.13.4, spatialdata 0.8.0),
  in parallel, over the 19 test modules that call harpy or depend on what it
  attaches: feature extraction (core, controller, widget), multi-widget,
  headless, persistence, app state, the six Spatial Query modules, the
  classifier and Object Classification widget, the matrix helpers, feature
  matrix metadata, and labels and shapes styling. 681 passed. Spatiato does
  not call harpy's canonical-center API yet (P8), so 0.5.3's changes there do
  not reach these tests.

### P1. Dense Dask feature matrices are fully materialized on every use

After P0, feature extraction on backed SpatialData leaves `obsm[feature_key]` as
a Dask array (0.5.0 and 0.5.2). It works, but `normalize_feature_matrix(...)` reads the whole matrix from the store
on every call, including on every Object Classification refresh. On large tables
(e.g. ~407k cells) this makes widget refreshes slow.

Unbacked SpatialData: fixed upstream in harpy 0.5.2 for in-memory matrices (see
"Resolution" below). Findings on harpy 0.5.0 (2026-10-23, see P0): feature
extraction into an existing table also leaves `obsm[feature_key]` as a dense
Dask array. Computing it combines arrays already in memory, so each use is
cheaper than a store read, but the graph grows:

- every extraction builds a new lazy merge on top of the previous lazy matrix,
  in the unbacked branch of harpy's `add_table_components_by_region(...)`
  (`harpy/table/io/_components_by_region.py`);
- the task count grows by about 2 per extraction (3 after one, 26 after 12),
  whether the same region or alternating regions are replaced;
- values stay correct, but every compute replays the whole chain;
- the chain keeps the regional values of every earlier update in memory, even
  when later updates replaced those rows. Confirmed on 2026-10-24 with a
  harpy-only reproducer (two regions of 100k rows, 20 features, one region
  updated 10 times through `hp.tb.io.add_table_components_by_region(...)`):
  memory held grows by 16.0 MiB per update, one regional matrix, from 54.6 to
  198.9 MiB; one `.compute()` and reinstall brings it back to 49.1 MiB.

harpy 0.5.0's docstring describes the lazy merge as deliberate: it attaches the
result "without executing the merge", to avoid materializing the complete matrix
just to attach it. That makes sense when the existing matrix is backed by some
store, but not when every input is already in memory.

Resolution:

- **Upstream (harpy)**: reported as https://github.com/saeyslab/harpy/issues/274
  and fixed in 0.5.2 (#275). The unbacked branch now computes the merge when
  both matrices are in memory. Checked on 2026-10-24 against the #275 branch
  with the issue's reproducer (10 updates):

  | Attached matrix at the start | harpy 0.5.0 | harpy 0.5.2 |
  |---|---|---|
  | NumPy | Dask, 31 tasks, 198.9 MiB held | NumPy, flat at 48.4 MiB |
  | SciPy CSR | not measured | CSR, flat at 41.6 MiB |
  | Dask array | not measured | still chains: 104 tasks, 168 MiB held |

  Values are correct in every case.
- **Spatiato**: raise before such an update (decided on 2026-10-30; step 3,
  3c.3). harpy 0.5.2 computes the merge only when the supplied and the
  attached matrix are both in memory (`_components_by_region.py`: "Computing
  reads nothing from storage"). When the attached matrix is not in memory (a
  Dask array, or a `zarr.Array` or `CSRDataset` a caller attached), it keeps
  chaining on unbacked SpatialData, on purpose: the matrix may still read from
  a store, and computing would read all of it. Its docstring tells callers to
  compute between repeated updates. That case only arises when a caller passes
  such SpatialData (`Interactive(sdata)` or the headless API), for example a
  table read lazily and attached to in-memory SpatialData, or a custom Dask
  matrix. Spatiato's own reader gives backed SpatialData, where harpy writes to
  the store and reattaches a fresh lazy read, so nothing chains.
  - The check: `check_existing_feature_matrix(...)` raises `ValueError` when
    the SpatialData is unbacked, the existing entry is compatible (a regional
    update; on unbacked SpatialData, absent and incompatible entries give a
    fresh in-memory matrix, and on backed SpatialData every write ends with a
    fresh lazy read of the store, so nothing chains), and `obsm[feature_key]`
    is not NumPy or SciPy CSR/CSC; as in harpy, a memory-mapped NumPy array
    counts as not in memory (added at the 3c.3 readiness check). Whether a Dask array
    reads from a store cannot be detected reliably (harpy records it only in a
    private registry), and the chain grows either way.
  - The message says what to do: load the matrix into memory (for Dask,
    `table.obsm[key] = table.obsm[key].compute()`) or write the SpatialData to
    a store first. The widget shows it as a status and launches nothing; headless
    raises; the worker's re-check raises too.
  - Why raise, not warn: values stay correct, but memory and every later read
    grow with each regional update, which a user only notices when it hurts; a
    warning in napari's console is easily missed. Raising matches the contract
    for unsupported encodings and P7's "check before changing anything", and
    the case is rare.
  - Dropped: a safeguard that computed the matrix after each extraction. It
    would read the whole matrix, possibly from a store, against D1's "no
    conversion; consumers handle any representation".

Direction (settled with D1 on 2026-10-30, see "Migration Plan", step 3):

- read `shape` / `dtype` without materializing (summary, metadata inspection,
  column-count validation);
- no finite-row mask on refresh: the summary scopes rows by the labels
  elements the table annotates, and the worker drops non-finite rows among the
  rows it reads (this replaces the earlier idea of caching the mask);
- read values once per classifier job, inside the worker thread, and only the
  rows the job uses, instead of copying the matrix in
  `_prepare_classifier_job(...)` on the main thread.

### P2. Lazy sparse matrices break

Dask arrays of CSR blocks (e.g. `X` from `hp.io.read_zarr`, or sparse `obsm`)
fail in `normalize_feature_matrix(...)`, `_get_finite_feature_row_mask(...)`,
and `X`-var coloring.

Seen in real use (2026-10-23): coloring a labels layer by an `X` var fails with
"ValueError: setting an array element with a sequence."

How a table gets a lazy sparse `X`: Spatiato's reader still loads tables
eagerly (`spatialdata.read_zarr`), but harpy functions that write a table to
backed SpatialData go through `hp.tb.io.add_table`, which reopens the written
table with `mode="lazy"` (e.g. allocation, `allocation_intensity`,
preprocessing, clustering). A sparse `X` (e.g. transcript counts) then comes
back as a Dask array of CSR blocks. `hp.io.read_zarr` gives the same result,
because its default table mode is lazy.

Failure path in `_build_x_var_colormap(...)` (`viewer/labels_styling.py`):

1. `table.X[obs_positions, var_index]` returns a lazy Dask array whose metadata
   claims a 1-D NumPy array of shape `(n,)`; computing it actually gives a
   `csr_matrix` of shape `(n, 1)`;
2. the Dask array has no `.toarray()`, so the code takes the `np.asarray(...)`
   branch;
3. `np.asarray` computes the Dask array and wraps the resulting `csr_matrix` in
   a 0-d object array; `.reshape(-1)` gives one element holding the whole
   matrix;
4. `pd.to_numeric(..., errors="coerce")` leaves that element unchanged, and
   `.astype("float64")` calls `float(csr_matrix)`, which raises the error above.

Coloring shapes by an `X` var (`_get_table_color_values_by_instance(...)` in
`viewer/shapes_styling.py`) uses the same code; there the one-element array
reaches `pd.Series(..., index=region_rows.index)` and fails with a length
mismatch instead.

`X` types checked for `X`-var coloring (scratch scripts, anndata 0.12.16):

| `X` | Result |
|---|---|
| SciPy CSR | works |
| dense Dask | works (`np.asarray` computes it) |
| Dask array of CSR blocks | fails as above |
| backed `CSRDataset` | works (indexing returns a `csr_matrix`) |
| backed dense `zarr.Array` | works |

Direction: one shared helper that turns any supported matrix type into an
in-memory NumPy or SciPy array (NumPy, SciPy, dense Dask, sparse Dask, backed
`zarr.Array`, `CSRDataset` / `CSCDataset`), and one helper that extracts a single
column for `X`-var coloring without materializing the full matrix. Labels and
shapes styling both use the column helper,
`read_matrix_column(matrix, column, rows=obs_positions)` (step 3a), and cast
the result to float64.

Revised on 2026-10-30: the first direction was to select the column first
(`X[:, var_index]`) and the region's rows afterwards, in memory. That reads the
column for every row of the table, so for a CSR `X` (the usual case) every
block of a lazy matrix, or every row of a `CSRDataset`, although coloring needs
only the rows of one labels element. The helper now picks the order from how
the matrix is stored:

| Stored by | Read order | Reads |
|---|---|---|
| rows: dense, CSR (in memory, Dask, `CSRDataset`) | rows first, then the column in memory | only the blocks or rows of the labels element |
| columns: CSC (in memory, Dask, `CSCDataset`) | the column first, then the rows in memory | one column block, or one column |
| backed dense `zarr.Array` | rows and column in one orthogonal selection | only the chunks that overlap both |

Values come back in the order of the row positions. The helper does not sort
them: every representation handles unsorted positions, and for Dask sorting
only saves a reorder step (measured on 2026-10-30 with dask 2026.8.0: 38
against 76 tasks for 300 of 1,000 rows, no extra block reads).

### P3. Backed handles are untested

`zarr.Array` and `CSRDataset` / `CSCDataset` (from `aggregate_points`,
`add_canonical_centers`, or `read_table(mode="backed")`) have only been tried in
`X`-var coloring (works, see P2). Every other Spatiato path is untested,
including spatial query's `obsm["spatial_canonical"]`.

Direction: cover them with the P2 helper and add tests per consumer.

### P4. Reload drops laziness

Status: decided on 2026-11-08, as step 4b (see "Migration Plan", step 4);
re-checked on 2026-11-09 against the code after 4a, and implemented the same
day.

`reload_table_components(...)` uses `ad.io.read_elem`, so "Reload Table State"
replaces a lazy feature matrix with an in-memory copy.

Direction: read with `hp.tb.io.read_table_components(...)`, lazily (D1; see
"Migration Plan", step 4), and keep Spatiato's reload-binding validation,
`TableModel.validate`, and in-memory rollback around it.

### P5. Replace Spatiato's on-disk writes with harpy's component API

Status: decided on 2026-11-08, as step 4a (see "Migration Plan", step 4).

Spatiato writes in place without staging or rollback. Once `obsm` entries are
lazy views of the store, an in-place `write_elem` over such a key could write to
the location it is reading from. harpy's `write_table_components(...)` /
`delete_table_components(...)` stage writes, roll back on failure, handle lazy
replacements reading from overwritten components, and support zarr v2/v3.

Direction:

- keep Spatiato's dirty tracking (`SpatiatoAppState`), `TableComponentPath`,
  write snapshots/acknowledgement, and reload validation;
- replace only the store I/O in `core/persistence.py`
  (`write_table_components`, `write_table_prediction_state`,
  `reload_table_components`) with `hp.tb.io` calls: writes through the
  SpatialData adapter `add_table_components(...)`, which reopens what it
  writes, and reloads through `read_table_components(...)` (see "Migration
  Plan", step 4; the store-only writer was the first choice, revised on
  2026-11-08);
- map paths: `("obs", col)` -> `("obs",)` (whole dataframe, as today),
  `("obsm", key)` -> `("obsm", key)`, `("uns", *keys)` -> `("uns", *keys)`;
- pass `obs_identity` for obs-aligned writes; keep including `_colors` palette
  paths explicitly, as widgets already do;
- blocking table writes while a job writes the same table through harpy is
  done earlier, as 3c.6 (P11), with the blocker that already guards reloads.

Additional reason (harpy 0.5.1+, anndata 0.13): anndata 0.13 shards zarr v3
arrays by default (`ad.settings.auto_shard_zarr_v3` is `True` in anndata 0.13.4,
checked on 2026-10-27), so Spatiato's own `ad.io.write_elem` calls now write
sharded arrays. harpy deliberately writes dense matrices without shards
(https://github.com/saeyslab/harpy/issues/270). Writing through harpy keeps
Spatiato's writes consistent with harpy's storage choices.

### P6. Feature-matrix metadata `backend` label

`register_feature_matrix_metadata(...)` records `"numpy"` for a Dask matrix;
harpy also hard-codes `"numpy"`. Decide whether to record the real backend or
drop the field from Spatiato's interpretation. Cosmetic; lowest priority.

### P7. `overwrite_feature_key` changed meaning in harpy 0.5.0

Status: implemented on 2026-10-29. Decisions 1-9 agreed (4-8 added on
2026-10-26 and 9 on 2026-10-27, after checking the spec against the code).
Step 1 (core function, headless, core tests) and step 2 (controller and widget,
following the step 2 design points 1-9 below) are done; the flow is documented
in `docs/development/feature_extraction.md`. Open follow-up: the public
pre-launch check requested in https://github.com/saeyslab/harpy/issues/281,
which would replace the copied and private harpy checks.

Found by the P0 run:
`tests/test_headless.py::test_compute_features_for_classifier_returns_normalized_target`
fails with "Existing feature metadata must have a compatible schema and
identical feature columns in order." The test seeds `obsm["computed_area"]`
without a harpy metadata record.

What changed:

- harpy 0.4.4: `overwrite_feature_key=True` replaced `obsm[feature_key]`
  entirely.
- harpy 0.5.0: it updates by region. Rows of the selected labels elements are
  replaced; rows of other labels elements are kept. This requires a compatible
  existing entry (`add_feature_matrix` existing-matrix checks and
  `_merge_feature_matrix_metadata` in `harpy/table/_add_feature_matrix.py`):
  - dense matrix of shape `(n_obs, n_columns)`, with a safe dtype cast;
  - metadata record `uns["feature_matrices"][feature_key]` with the expected
    `schema_version` and `source_kind == "harpy_add_feature_matrix"`;
  - identical `feature_columns` (in order) and identical `source_channels`.

  Otherwise harpy raises `ValueError`.
- Affected cases that used to replace and now fail: an existing key with
  different features or channels, and an existing key holding a custom matrix
  (registered through Spatiato, or without harpy metadata).
- Spatiato still describes overwrite as full replacement: the Feature
  Extraction confirmation button "Replace feature matrix"
  (`_prompt_overwrite_feature_key_confirmation(...)` in
  `widgets/feature_extraction/widget.py`) and the headless docstrings
  "Whether to replace an existing `table.obsm[feature_key]`".

Decisions:

1. **Overwrite means "replace regions".** For a compatible existing entry, the
   widget dialog names the selected labels elements and states that values for
   other labels elements are kept, e.g.:

   > Feature matrix "computed_area" already exists. Replace its values for the
   > selected labels element(s) "blobs_labels"? Values for other labels
   > elements are kept.

   Button: "Replace selected regions".
2. **Incompatible or custom entries are still replaceable**, by removing them
   first:
   - `hp.tb.io.remove_table_components(sdata, table_name=..., components=[("obsm", key), ("uns", "feature_matrices", key)])`
     removes the entries in memory and, for backed SpatialData, in the store;
     entries absent from both are skipped;
   - then `add_feature_matrix(...)` creates a fresh matrix, so harpy's
     compatibility checks do not apply.

   Values for all other labels elements are lost (their rows become NaN). The
   widget shows a separate, stronger confirmation naming the affected labels
   elements, with a "Replace entire feature matrix" button.

   This is two steps and not atomic: if extraction fails after the removal, the
   old matrix is gone. Accepted for the first version, because the user has
   just confirmed discarding it. A safer follow-up is extracting into a
   temporary key and swapping it in with one rollback-protected
   `hp.tb.io.add_table_components(..., delete=...)` call.
3. **Widget and headless behave the same**, through one shared core function.
   - Headless keeps the single `overwrite_feature_key` flag; no extra flag.
     `True` means: replace the selected regions when the existing entry is
     compatible, otherwise remove and replace the whole matrix.
   - In the full-replacement case, log a warning naming the labels elements
     whose values are dropped.
   - The widget's two dialogs both lead to `overwrite_feature_key=True`, and
     pass the confirmed outcome along (decision 4).
   - The flag means the same in both storage modes: when an entry exists
     (compatible or incompatible) and `overwrite_feature_key=False`, Spatiato
     raises, also for unbacked SpatialData. harpy itself only requires the flag
     for the store, so without this, headless would merge silently when
     unbacked and raise when backed.
4. **The confirmed outcome travels with the job.** The widget checks the
   existing entry on the main thread, shows the matching dialog, and the
   request carries what the user confirmed:
   `confirmed_existing_entry: Literal["absent", "compatible", "incompatible"] | None`
   on `FeatureExtractionRequest`. The shared core function checks again in the
   worker before it removes or computes anything. If the result differs, it
   raises "Feature matrix "X" changed since you confirmed; calculate again", and
   nothing has been removed or written. Without this, a table change between
   dialog and worker could turn a confirmed regional replacement into an
   unconfirmed full replacement. The widget always passes the kind it found,
   also `"absent"` (no dialog): a matrix that appears between the click and the
   worker then stops the job with this message rather than the generic "already
   exists". Headless and table creation pass `None`: whatever the check finds
   is acted on. The re-check covers what is in memory and in the store,
   not dirty state: that lives in `SpatiatoAppState` on the main thread, so the
   unsaved-changes block of decision 7 runs only in the widget, at launch.
   If the check raises (rows harpy would reject, unknown channels, a backed
   table missing from the store), the widget shows the error in its status and
   launches nothing; the worker's re-check reports the same errors through
   `_on_worker_errored`.
5. **The compatibility check mirrors harpy's checks** (`add_feature_matrix`
   existing-matrix checks and `_merge_feature_matrix_metadata`). The entry is
   compatible only when all of these hold; anything else is incompatible, with
   a reason:
   - metadata is a mapping with `source_kind ==
     HARPY_ADD_FEATURE_MATRIX_SOURCE_KIND` and `schema_version ==
     FEATURE_MATRIX_METADATA_SCHEMA_VERSION` (Spatiato's constants in
     `core/feature_matrix_metadata.py`; a test pins them to harpy's values);
   - features are equal in harpy's column order: intensity features first, then
     morphology features, keeping the order within each group. harpy derives
     `feature_columns` from the features and channels, so equal groups and
     channels give equal columns. The widget already sends one fixed order
     (`selected_feature_names`), so this only matters for headless input;
   - channels: a small function resolves the request against the first
     triplet's image the way harpy's `_resolve_channels` does (`None` is all
     channels, an integer is an index, a string is a name, duplicates removed,
     an unnamed channel becomes `channel_{i}`). The result must equal
     `source_channels`, which is `None` for morphology-only features;
   - the matrix is dense (NumPy, Dask with NumPy blocks, or `zarr.Array`), with
     shape `(n_obs, len(feature_columns))`, and its dtype can hold float64
     safely.

   The check reads memory for unbacked SpatialData. For backed SpatialData it
   reads the store, without loading matrix values:
   `hp.tb.io.read_table_components(sdata.path, table_name=..., components=[("obsm", key), ("uns", "feature_matrices", key)], mode="backed", missing="omit")`.
   If harpy still raises, its error is shown; the check runs before extraction,
   so nothing has been removed by then.

   Spatiato keeps its own copy of these checks (`_incompatibility_reason(...)`
   and its helpers in `core/feature_extraction.py`) instead of reusing harpy's.
   harpy has no function that classifies an entry without computing features:
   its matrix checks run after the computation and raise, and the helpers
   (`_resolve_channels`, `_matrix_format`, `_merge_feature_matrix_metadata`) are
   private. A public check is requested in
   https://github.com/saeyslab/harpy/issues/281 (2026-10-28); once it is
   released, replace the copy with it.
6. **Events.** All events use the `obsm` and `uns` metadata paths of the key:
   - regional replacement: `change_kind="updated"`, `regions` = the selected
     labels elements, as today;
   - full replacement: `change_kind="updated"` (not `"rebuilt"`: Object
     Classification only invalidates its classifier on `"updated"`), `regions`
     = every region the table declares (`get_table_metadata(...).regions`), as
     the `TableStateChangedEvent` contract requires;
   - new key: `"created"` with the selected regions, as today;
   - the change kind and regions come from the `FeatureMatrixCalculation` that
     `calculate_feature_matrix(...)` returns, which the worker passes on in its
     result; not from what is in memory (today
     `_prepare_feature_extraction_job(...)` looks at `table.obsm`).
     `FeatureExtractionJob.change_kind` goes;
   - failure after the removal: the core function raises a dedicated error that
     wraps the original and records that the removal happened. The controller
     then publishes a `"removed"` event with every region, through
     `record_persisted_table_change(...)` for backed SpatialData (the removal
     is already on disk) and `record_table_mutation(...)` for unbacked. The
     status reads "Feature extraction failed; the previous feature matrix "X"
     was removed."
7. **Memory and store disagree (backed SpatialData).**
   - The entry is only in memory, not in the store (never saved): incompatible,
     with the reason "it is not saved". The
     full-replacement dialog is shown; then `remove_table_components(...)`
     removes it from memory and a fresh matrix is extracted. Without this,
     harpy replaces it silently.
   - The entry is in the store, and memory holds unsaved changes to it (its
     `obsm` or metadata path is dirty in `SpatiatoAppState`, see
     `snapshot_table_dirty_state(...)`): blocked with "Feature matrix "X" has
     unsaved changes; write or reload the table first". harpy would merge with
     the stored version and drop those changes. This is a real Spatiato flow:
     Object Classification's "register feature metadata" for a custom matrix
     changes the metadata in memory and marks it dirty until the table is
     written. The widget checks this on the main thread before launching; the
     worker's re-check (decision 4) does not repeat it, so a change made
     between launch and the worker's check is not caught. Accepted.
   - The entry is only in the store, not in memory (only possible through code
     outside Spatiato's widgets): the check follows the store, and the dialog
     describes the stored entry. A regional merge brings the stored values for
     other regions back into memory.
   - Changes outside Spatiato's dirty tracking (notebook edits, another process
     writing the store) cannot be detected: the store wins silently. Accepted
     limitation, the same as harpy's own behavior.
   - Rows that differ between memory and store (also only through changes
     outside Spatiato): harpy rejects every feature update of such a table,
     and `remove_table_components(...)` does not check rows. Without a check,
     an incompatible entry would be removed before harpy's rejection. The
     check therefore runs harpy's row validation first, for every kind, and
     raises harpy's error (also for unbacked SpatialData with invalid
     region/instance pairs). It calls harpy's private helpers
     (`_check_in_memory_versus_storage_axes`, `_validated_observation_pairs`):
     copying them would mean reading the stored row identities ourselves.
     Remove once the public check requested in
     https://github.com/saeyslab/harpy/issues/281 is released.
   - Otherwise the store is checked, as in decision 5.
   - Headless has no dirty tracking: it follows the store, like harpy, and its
     docstring says so.
8. **Partial entries.**
   - Metadata without a matrix: absent. There are no values to lose, so no dialog.
     `overwrite_feature_key=True` is still passed to harpy (backed SpatialData
     needs it for the stored metadata); harpy writes fresh metadata, and other
     regions' rows are NaN. The event is `"created"`; an info log notes that
     the orphaned metadata was replaced.
   - A matrix without metadata (custom or unregistered): incompatible, with the
     reason "no harpy feature metadata". The full-replacement dialog names the
     labels elements whose values are lost.
9. **Jobs that are no longer active still publish their table changes.** Any
   change to the widget's inputs (a feature checkbox, the key, the table)
   cancels the running job (`_bind_batch_state(...)` in
   `widgets/feature_extraction/controller.py`), but `quit()` cannot interrupt
   a running harpy call, so the job still writes. Its result is lost today:
   superqt's worker never emits `returned` after `quit()` (`WorkerBase.run()`
   in `superqt/utils/_qthreading.py`; `errored` and `finished` still fire), and
   `_on_worker_returned(...)` and `_on_worker_errored(...)` also ignore any job
   that is not the active one. No event is published, and Object
   Classification keeps a classifier trained on the old values. This already
   happens for regional writes; P7 adds full replacement and the removal after
   a failure, which lose more.

   What happens when a job finishes:

   | Job when it finishes | Table event | Table-choice refresh | Status |
   |---|---|---|---|
   | Active | yes | yes, as today | yes |
   | Cancelled, same SpatialData still selected | yes (new) | yes (added later, see below) | no |
   | Cancelled, other SpatialData selected, or widget closed | no | no | no |

   Added after step 2: a returned detached job also refreshes the table
   choices. Before, a table created by a detached create-table job appeared in
   the drop-down only at the next selection edit. The widget keeps the current
   selection and only promotes a created table that is still the create-table
   target (`_on_controller_table_selection_refresh_required(...)` in
   `widgets/feature_extraction/widget.py`).

   Also added after step 2: detachment is derived, not recorded. A job is
   detached while the current selection is not the one it was launched for
   (`_job_detached` and `_job_matches_selection(...)` in
   `widgets/feature_extraction/controller.py`); a create-table job also matches
   the existing-table selection of its table. Before, any rebind set a flag
   that stayed set, so promoting the created table detached the active job
   (its "wrote …" status was lost at `finished`), and changing an input back
   left its own calculation showing "waiting for the previous calculation".

   A table event is decision 6's event for a success, or the `"removed"` event
   after a failure that follows the removal.

   Implementation:

   1. **Cancelling no longer calls `quit()`.** For a function worker, `quit()`
      only sets the abort flag that suppresses `returned`; the work is a plain
      function call (`FunctionWorker.work()`). Cancelling becomes detaching:
      `_cancel_active_worker(...)` clears `_active_worker` and
      `_active_worker_job_id` as today, and the job stays in `_in_flight_jobs`,
      so reloads of its table stay blocked until `finished`.
   2. **`_on_worker_returned(...)` branches on the job.** `returned` fires
      before `finished`, so the job is still in `_in_flight_jobs`:

      ```python
      job = self._in_flight_jobs.get(job_id)
      if job is None or job.sdata is not self._selected_spatialdata:
          return
      if job_id == self._active_worker_job_id:
          ...  # as today: table-choice refresh, event, status
      else:
          self._notify_feature_matrix_changed(result)  # event only
      ```
   3. **`_on_worker_errored(...)` branches the same way.** On the dedicated
      failure-after-removal error of decision 6 it publishes the `"removed"`
      event, active job or not; the status message is shown for the active job
      only.
   4. **The widget is unchanged.** `_on_controller_feature_matrix_changed(...)`
      (`widgets/feature_extraction/widget.py`) builds the event from
      `selected_spatialdata`, which is the job's SpatialData under the
      condition of point 2, and already uses an empty dirty snapshot for a job
      that is not the active one.
   5. **Shutdown is unchanged.** `shutdown(...)` detaches the callbacks and
      keeps calling `quit()`, so a job still running when the widget closes
      loses its event. Accepted for now.
   6. **Tests.**
      - `_DeferredWorker` in `tests/test_feature_extraction.py` behaves like
        superqt: no `returned` after `quit()`. Today it emits it, which hides
        the lost result.
      - The two rebinding tests that assert `quit_called is True`
        (`test_feature_extraction_controller_drops_stale_results_after_rebinding`
        and
        `test_feature_extraction_controller_blocks_matching_reload_until_every_superseded_worker_finishes`)
        change to: `quit()` is not called. The first also asserts that the
        event for the old key is published, with no table-choice refresh or
        status change. The shutdown test keeps its assertion.
      - New: start an extraction, change the feature key while it runs, and
        assert that the event for the old key is still published; the same for
        a failure after the removal (`"removed"` event).

Implementation notes:

- Classify the existing entry into one of three outcomes before starting, each
  with a short reason for dialogs and warnings:
  - **absent**: no `obsm[key]` (metadata alone counts as absent, decision 8);
    plain extraction;
  - **compatible**: decision 5 holds; replace the selected regions;
  - **incompatible**: anything else, including custom matrices and backed
    memory-only entries (decision 7); remove and replace the whole matrix.
- **Check the same state harpy checks.** For backed SpatialData, harpy reads the
  existing matrix and metadata from the **store** (`_existing_feature_matrix`
  in `harpy/table/_add_feature_matrix.py`), not from memory; only unbacked
  SpatialData is checked in memory. harpy's `overwrite_feature_key` "only
  concerns the store": an entry present only in memory is replaced without it.
  A memory-only pre-check could therefore pick a different dialog than harpy's
  actual behavior whenever memory and store differ; decision 7 covers where
  they differ.
- The labels elements named in the full-replacement warning and dialog are the
  table's regions that are not selected for extraction.
- Shared core function (e.g. in `core/feature_extraction.py`) used by the
  widget's `_run_feature_extraction_job(...)`
  (`widgets/feature_extraction/controller.py`) and by
  `headless.compute_features_for_classifier(...)`. The removal runs in the same
  worker as the extraction.
- The widget uses the same compatibility check to choose which confirmation to
  show before launching the job, and passes the outcome as
  `confirmed_existing_entry` (decision 4).
- Backed SpatialData: keep calling `record_persisted_table_change(...)` for
  `("obsm", key)` and `("uns", ("feature_matrices", key))` after extraction,
  as today, with the regions and change kind of decision 6.
- Update the `overwrite_feature_key` docstrings in `headless.py` to describe
  both cases.
- Fix `test_compute_features_for_classifier_returns_normalized_target` by
  dropping the pre-seeded matrix; the test is about target normalization.

Implementation order:

1. Shared core function (classification with channel resolution, the
   confirmed-outcome check, regional or full replacement, the error after a
   removal, warning), headless wiring and docstrings, and the core tests. This
   makes the suite green.
2. Widget and controller: use the classification to pick the confirmation
   ("Replace selected regions" or "Replace entire feature matrix"), pass the
   confirmed outcome, block on unsaved changes (decision 7), publish the events
   of decision 6, also for jobs that are no longer active (decision 9), and the
   widget tests.

Step 2 design (agreed on 2026-10-29, after checking the code):

1. **The check lives in the controller.** A new
   `FeatureExtractionController.check_existing_feature_matrix(*, dirty_paths)
   -> ExistingFeatureMatrix | None` runs on the controller's bound SpatialData,
   triplets, table, features and key: the inputs the job will use.
   - For backed SpatialData it first applies the unsaved-changes block
     (decision 7) to `dirty_paths`, then runs the core check.
   - On a block or a check error (decision 4) it sets the error status itself
     and returns `None`. The widget's feedback card only shows the controller's
     status (`_update_feature_extraction_feedback(...)`), so the controller owns
     every status message, and both rules are testable without the widget.
   - The widget calls it in `_calculate_feature_matrix(...)` on the main thread
     with `snapshot_table_dirty_state(...).paths` (plain `TableComponentPath`s,
     so the controller needs no `SpatiatoAppState`), launches nothing on
     `None`, otherwise picks the dialog and passes the kind to
     `controller.calculate(..., confirmed_existing_entry=...)`.
   - Calculate is only enabled when the controller can calculate and is bound
     to the staged selection (`_get_calculate_button_blocking_reason(...)`), so
     the check always sees inputs the controller accepts. Table creation skips
     the check.
   - On `FeatureMatrixRemovedError`, the status is the error's own message,
     without the usual "Feature extraction: calculation failed: " prefix: the
     message already says "Feature extraction failed; the previous feature
     matrix "X" was removed: …".
2. **Results carry the change kind and regions.**
   - `FeatureExtractionRequest` gets `confirmed_existing_entry`.
   - `_run_feature_extraction_job(...)` calls `calculate_feature_matrix(...)`
     and copies the change kind and regions of its `FeatureMatrixCalculation`
     into `FeatureExtractionResult`, which gets a `regions` field.
     `FeatureMatrixWriteChangeKind` also allows `"removed"`.
   - `FeatureExtractionJob.change_kind` and the `table.obsm` lookup in
     `_prepare_feature_extraction_job(...)` go.
   - On `FeatureMatrixRemovedError`, the controller builds a result with
     `"removed"` and the error's regions, and calls the same
     `on_feature_matrix_changed` callback. The widget's
     `_on_controller_feature_matrix_changed(...)` uses `result.regions` instead
     of `result.labels_names`, so recording stays where it is:
     `record_persisted_table_change(...)` when backed,
     `record_table_mutation(...)` when unbacked.
3. **Unsaved changes are an overlap test.** For backed SpatialData, the widget
   blocks when a path in `snapshot_table_dirty_state(...).paths` overlaps
   `obsm/<key>` or `uns/feature_matrices/<key>`: equal, or one a prefix of the
   other. Today's flows mark the exact paths (Object Classification's metadata
   registration marks `uns/feature_matrices/<key>`); the overlap test also
   covers a dirty parent such as `uns/feature_matrices`.
4. **The main-thread cost is accepted.** The check reads the stored row
   identities (the index and the two identity columns) and the entry's
   metadata, and the worker reads them again. Not measured; revisit if it lags
   on tables with millions of rows.
5. **Dialog wording.**
   - Compatible: the text of decision 1, button "Replace selected regions".
   - Incompatible: "Feature matrix "X" exists but cannot be updated by region,
     because {reason}. Replacing it removes the values for labels element(s)
     "…"." The last sentence is left out when the table has no other labels
     elements. Button "Replace entire feature matrix".
   - Both replace `_prompt_overwrite_feature_key_confirmation(...)`.
6. **Cancelling and shutdown split.** `_cancel_active_worker(...)` serves both
   rebinding and `shutdown(...)` today. Rebinding detaches without `quit()`
   (decision 9); shutdown keeps calling `quit()`.
7. **One extraction at a time while a job is still writing** (found on
   2026-10-29; the gap predates P7, because `quit()` never stopped the work).
   - How two jobs overlap today: the widget does not lock its inputs during a
     run, any input change detaches the running job (its harpy call keeps
     running), and `can_calculate` only checks `is_running`, which looks at the
     active job. Calculate is therefore enabled again, and a second click runs
     a second extraction in parallel, usually on the same table.
   - harpy gives no protection against concurrent writers ("No locking
     coordinates concurrent access", `_publish_staged_paths(...)` in
     `harpy/_storage/_publication.py`). One backed extraction publishes only
     `obsm/<key>` and `uns/feature_matrices/<key>` (checked on 2026-10-29), so
     different keys do not overwrite each other's data paths, but they share:
     - the store's root metadata: each write saves its bytes before
       publishing and restores them on failure (`_publish_table_paths(...)` in
       `harpy/table/io/_write.py`), so a failing job can drop the other job's
       entry from the consolidated metadata;
     - created parents: the job that creates `uns/feature_matrices` removes it
       on failure, together with the other job's metadata in it;
     - the table in memory: harpy installs whole `obsm` and `uns` slots built
       from a copy taken just before (`_install_memory_updates(...)` in
       `harpy/table/io/_components.py`), so overlapping installs can drop the
       other job's entry from memory.

     With the same key it is worse: the second job checks, and harpy reads the
     existing entry, before the first job writes, so it acts on an outdated
     view (the confirmed-kind re-check does not help), and the later result
     silently replaces the earlier one.
   - Fix: `can_calculate` and `calculate()` check whether any job is in flight
     (`has_job_in_flight`, see point 8) instead of only the active job, with
     the status "Feature extraction: waiting for the previous calculation to
     finish writing.". `_on_worker_finished(...)` also
     notifies a state change when a detached job finishes, so Calculate is
     enabled again (today it only does so for the active job).
8. **One job slot** (agreed on 2026-10-29). With point 7 the controller has at
   most one harpy job, so `_in_flight_jobs` (a dict) and the separate
   active-worker pointer become one slot: `_job`, its `_worker`, and
   `_job_detached`, set when an input changes (later derived from the
   selection; see decision 9).
   - Public contract: `has_job_in_flight` (was `has_in_flight_jobs`): a harpy
     job has not finished, also when detached; it may still write the table,
     in memory and, for backed SpatialData, in the store, and blocks launches
     and reloads. With `can_calculate` and `status_message`, that is the whole
     contract: `is_running` and, after point 9, `active_job_id` were removed,
     and the status code reads the private `_job_detached` flag.
   - Signals still carry the job id, so a stray signal is never taken for the
     current job.
   - Launching into an occupied slot raises `RuntimeError`. `calculate()`
     returns first, with a status, so this only catches programming errors.
   - The guard is per controller: a widget opened while a closed one's job
     still writes gets its own controller and may start a second job.
     Accepted: rare, and a guard across widgets would need shared state in
     `SpatiatoAppState`.
9. **The launch snapshot travels with the job** (agreed on 2026-10-29). For
   backed SpatialData the widget records a result with
   `record_persisted_table_change(event, snapshot)`, whose
   `acknowledge_table_write(...)` (`_app_state.py`) clears the dirty paths
   that the snapshot captured and that an event path (`obsm/<key>`,
   `uns/feature_matrices/<key>`) covers; a change made after launch has a
   fresh token and stays dirty.
   - The widget takes one dirty snapshot at the click. Its paths go to the
     check (the unsaved-changes block of point 3), and the snapshot itself to
     `calculate(..., dirty_snapshot=...)`. The controller carries it without
     interpreting it: `FeatureExtractionJob.dirty_snapshot`, copied into the
     success result by the worker and into the `"removed"` result by
     `_on_worker_errored(...)`. The widget records each result with its own
     `result.dirty_snapshot`.
   - Recording is therefore correct by construction, independent of the
     block: the block keeps its own purpose (decision 7, unsaved changes that
     harpy would drop). A first version recorded without a snapshot, which
     was only correct because of the block, so the widget depended on a
     controller rule it could not see.
   - Gone from the widget: `_active_feature_extraction_snapshot`, its
     job-id matching and its clearing on detach; detached jobs use their own
     launch snapshot. Decision 9 point 4 (empty snapshot for a job that is no
     longer active) becomes moot.
   - `record_persisted_table_change(...)` takes an optional snapshot; a new
     table's result has none and only publishes the event.
   - The widget no longer uses `active_job_id`, so the controller no longer
     exposes it; the one test that read it takes the job id from the job
     passed to its worker factory.

Tests that change: `_DeferredWorker` in `tests/test_feature_extraction.py`
stops emitting `returned` after `quit()`; the two rebinding tests of decision
9; the `job.change_kind` assertion in `tests/test_feature_extraction.py`; the
monkeypatched `_prompt_overwrite_feature_key_confirmation` in
`tests/test_multi_widget_backed_zarr.py`; results built in tests need
`regions`.

Order within step 2: the controller first (request, result, the worker calling
the core, decision 9, controller tests), then the widget (check,
unsaved-changes block, dialogs, widget tests).

### P8. Spatial Query computes canonical centers itself

Status: decided on 2026-11-01. The harpy side shipped in 0.5.3 on 2026-11-05
(#283, addressing https://github.com/saeyslab/harpy/issues/279 and
https://github.com/saeyslab/harpy/issues/282; see "Harpy 0.5.3: What
Changed"). Implemented as 3c.2 of the "Migration Plan" (parts 3c.2a-c, with
the readiness check and design points of 2026-11-05 under step 3), after P9.
Fixed on 2026-11-06: 3c.2c removed the modules described below.

Before 3c.2, Spatiato computed canonical centers per labels element
(`core/spatial_query/centroids.py`, 162 lines) and kept a regional cache with
its own models, schema and validation (`canonical_models.py`, 448 lines;
`canonical.py`, 892 lines): states absent, partial, stale, invalid and valid;
actions create, extend, refresh and rebuild; other regions preserved; matrix
and metadata installed together; the cache marked dirty for Write.

harpy has the same design upstream: the public `harpy.table.canonical_centers`
package, documented as the integration contract for Spatiato
(`docs/canonical_centers.md` in harpy), and `hp.tb.add_canonical_centers`. Same
algorithm (`RasterAggregator.center_of_mass` on `scale0`, intrinsic pixel
frame, `(z, y, x)` float64) and record layout; only the instance-digest domain
differed (`spatiato/...` against `napari-harpy/...`). harpy 0.5.3 drops the
instance digest and the cache API, and moves the record to schema version 2,
so that difference no longer matters.

Decision:

- **Canonical centers are all or nothing**, as harpy defines them: a table
  holds canonical centers only when every labels element it annotates has
  them, every row is finite, and the metadata is current. No regional updates:
  considered on 2026-11-01 (`add_canonical_centers(..., region=...)`) and
  dropped, as not worth it for harpy or Spatiato.
- **Spatiato relies on harpy and keeps no canonical logic of its own:**
  - status, and every Run: `validate_canonical_centers(sdata,
    table_name=...)` (public, in `harpy.table.canonical_centers`;
    `hp.tb.validate_table` runs it too). It reads the annotation from the
    table, returns `None` when there are no canonical centers, and raises
    `ValueError` when they are stale (another labels source or algorithm
    version), not finite, malformed, or in schema version 1. It reads the
    `(n_obs, 3)` matrix and labels metadata, no pixels. Row subsets and
    reorders keep the centers valid;
  - when absent or invalid: `hp.tb.add_canonical_centers(sdata,
    table_name=..., overwrite=True)` in the worker, which recomputes every
    labels element of the table. `overwrite=True` is needed for backed
    SpatialData whose store already has canonical centers; it is ignored for
    unbacked SpatialData. Afterwards the matrix is NumPy (unbacked) or a lazy
    Dask array (backed); `read_matrix_rows(...)` reads either;
  - reading: the selected labels element's rows with `read_matrix_rows(...)`.
    Spatiato finds those rows and their instance IDs from the table's
    annotation (`get_table_keys(table)`), as Object Classification does with
    `_resolve_region_row_positions(...)`; harpy keeps its region binding
    private. The provenance check at Apply validates again and compares those
    rows;
  - deleted: `canonical_models.py`, `canonical.py`, `centroids.py`, the cache
    states and actions, and the regional update. The status card shows
    "valid", "absent", or "needs recalculation" with harpy's message. Spatial
    Query keeps offering only 2-D labels: harpy also accepts 3-D labels, but
    the query uses y and x.
- **Persistence:** for backed SpatialData, harpy writes the canonical
  components itself. Spatiato records a persisted change, as for feature
  extraction, not a dirty mutation, and blocks reloads of the table while the
  worker writes. For unbacked SpatialData the centers exist only in memory and
  are recorded as a mutation, as Feature Extraction does. The event is
  `"created"` when the table had no canonical centers, and `"rebuilt"` when
  they were replaced. Feature Extraction and Spatial Query can then both write
  through harpy at once, which harpy does not protect against, so the shared
  write guard noted under P5 must cover both widgets.
- **Accepted costs:**
  - a stale labels element makes every labels element of the table be
    recomputed;
  - Spatial Query is unavailable for a table whose centers cannot all be
    computed: a region that is not a labels element, a region without rows,
    or a table row without pixels. The same holds for a table that
    `hp.tb.validate_table` rejects for an unrelated reason, since
    `add_canonical_centers` checks the whole table first: for example a
    feature-matrix record under a reserved `.obsm` key (P9). The status shows
    harpy's reason;
  - canonical centers written by Spatiato so far are schema version 1, fail
    validation with harpy's rebuild hint, and are recomputed once. Not
    backward compatible; these are young features.

harpy issues: https://github.com/saeyslab/harpy/issues/282 (opened on
2026-11-01) asked `add_canonical_centers` to support SpatialData without a
backing store, and https://github.com/saeyslab/harpy/issues/279 to stop row
subsets invalidating canonical centers. Both are addressed in 0.5.3 (#283),
which also drops the instance digest whose domain #282 asked to rename.

### P9. Reserved `.obsm` keys

Status: found and implemented on 2026-11-05 (harpy 0.5.3), as 3c.4 of the
"Migration Plan", before 3c.2.

harpy 0.5.3 reserves `obsm["spatial_canonical"]` and `obsm["spatial"]`:
`add_feature_matrix` rejects either as `feature_key`, and `validate_table`
rejects a feature-matrix record under either. Spatiato blocks neither:

- Feature Extraction's key check (`_get_feature_key_validation_error(...)` in
  `widgets/feature_extraction/controller.py`) accepts both, so harpy raises in
  the worker instead of the widget refusing up front;
- Object Classification lists every `obsm` key (`get_table_obsm_keys(...)` in
  `core/spatialdata.py`), so a user can register `spatial` or
  `spatial_canonical` as a custom feature matrix. harpy's `validate_table`
  then rejects the table, and so does `add_canonical_centers`, which checks
  the whole table first: Spatial Query becomes unavailable because of an
  unrelated registration.

Direction: Feature Extraction rejects the reserved keys in its key check, with
the other key errors; Object Classification leaves them out of its
feature-matrix choices and refuses to register them. Take the reserved set
from harpy if it becomes public; until then, Spatiato names the two keys
(harpy's `_RESERVED_OBSM_KEYS` is private).

### P10. A blocked reload still invalidates other workflows

Status: found on 2026-11-05, in the readiness check for 3c.2b. Implemented as
3c.5 of the "Migration Plan", before 3c.2b.

App state calls `prepare_for_table_reload(request)` on every registered
participant, in registration order (`_app_state.py`), and each widget
registers when it opens. A participant refuses the reload by raising: Feature
Extraction's controller does so while its job writes the table. The loop then
stops and nothing is reloaded, but the participants visited before it have
already acted. For example, with Object Classification opened before Feature
Extraction, a reload while Feature Extraction writes the table goes like this:

1. Object Classification, visited first, freezes for the reload: it cancels
   its jobs and drops its classifier (the model snapshot), with the reason
   "the table is being reloaded";
2. Feature Extraction's controller, visited second, raises, and the reload
   stops.

The table is unchanged, but the user has to train the classifier again, since
no "reloaded" event follows to reset it. With the widgets opened in the other
order, Feature Extraction raises first and Object Classification keeps its
classifier: the same click has a different outcome, depending only on which
widget was opened first.

Decision: prepare a reload in two passes.

1. Every participant says, without side effects, whether the reload must wait:
   `table_reload_blocker(request)` returns a reason, or `None`. If any
   participant gives a reason, app state raises one error listing every
   reason, and nothing has changed.
2. Only then does app state call `prepare_for_table_reload(request)` on every
   participant, which invalidates its work and no longer refuses.

Both passes and the reload run synchronously on the main thread, so no job
starts or finishes in between.

Which participants block: only one whose worker writes the table itself,
which a reload can neither interrupt nor undo. Feature Extraction's
`hp.tb.add_feature_matrix(...)` installs its result in the table from the
worker and, for backed SpatialData, writes the store. A reload during that call
could read a half-written store, or harpy could install its result into the
reloaded table afterwards. Its controller returns its "still writing" message
as its blocker; Spatial Query's centers job joins it in 3c.2b (design point 8).
Object Classification's jobs and Spatial Query's query only compute: the main
thread applies their results, and only for the current job. A reload therefore
lets them go. Their late results are ignored, nothing is half-written, and a
result computed from the replaced table would be wrong to apply anyway;
blocking would only make the user wait for a result that has to be dropped.
These widgets return `None` and keep their invalidation.

Not included in 3c.5: running the check before the persistence controls ask
Write / Discard / Cancel, and blocking Write itself. Both follow in 3c.6
(P11), which also replaces the participants' blocker with a busy state that
app state tracks per table: the first pass then checks that state, and
`prepare_for_table_reload(request)` stays the participants' only method.

### P11. Write is not blocked while a job writes the table

Status: found on 2026-11-06, while reviewing the reload flow after 3c.5.
Implemented as 3c.6 of the "Migration Plan", after 3c.5 and before 3c.2b. It
replaces the write block planned under P5.

Since 3c.5, a reload waits for a job whose worker writes the table. Write does
not, which leaves two problems:

1. **Two writers on one store.** A Write while Feature Extraction's harpy call
   writes the same table runs Spatiato's writer and harpy's at the same time.
   They write different components, since Feature Extraction refuses to launch
   while its key's paths are dirty, but they share the table's group in the
   store, and both rewrite the store's consolidated metadata. harpy does not
   protect against a second writer.
2. **A question before a refusal.** A reload of a dirty table first asks
   Write / Discard / Cancel, and only then asks the blockers. A user who
   chooses Write writes, and is then told the reload is blocked.
3. **A refusal only after a click.** Write and Reload stay enabled while a job
   writes the table; the user learns that the table cannot be touched only
   from an error after clicking.

3c.2b adds a second job that writes the table (Spatial Query's centers job),
which needs the same guard.

Decision (revised on 2026-11-06): **a table is busy while a job writes it**,
and a busy table can be neither written nor reloaded. App state tracks the
busy state per table, as it tracks the dirty state, instead of polling the
participants.

1. **Marking.** The owner of a job whose worker writes the table marks it
   busy when the job launches, with a reason, and releases it when the worker
   finishes, on every path: returned, errored, or detached and finished later.
   App state keeps the marks per SpatialData object and table, so two jobs on
   one table are two marks, and the table stays busy until both are released.
   API: `mark_table_busy(sdata, table_name, *, reason)` returns a token;
   `release_table_busy(token)`; `table_busy_reasons(sdata, table_name)`.
2. **Announcing.** App state emits `table_busy_state_changed` when a table
   becomes busy or no longer busy, as it emits `table_dirty_state_changed`.
3. **Buttons.** The persistence controls disable Write and Reload while the
   table is busy, with the reason as tooltip, and refresh on that signal. The
   user sees why before clicking, and a busy table never shows Write /
   Discard / Cancel.
4. **Guards.** `PersistenceController.write_table_state()` and the first pass
   of `prepare_for_table_reload()` raise `TableBusyError` with every reason,
   for example 'Table "T" is busy: Feature Extraction is still writing its
   results. Try again once the calculation has finished.' They cover callers
   other than the buttons, and a job that starts while the reload dialog is
   open. The persistence controls show the error.
5. **Participants.** The reload participant protocol goes back to one method,
   `prepare_for_table_reload(request)`. The blocker method, the
   `check_table_persistence(...)` check and its `action` parameter go.
   Feature Extraction's controller stops being a reload participant, since its
   `prepare_for_table_reload` does nothing; with it go the deferred release
   (`release_reload_participant_when_safe(...)`,
   `_release_reload_participant_if_safe()`) and the pending unregister in the
   widget's teardown. The controller marks the table busy in `calculate()` and
   releases it in `_clear_job()`; a detached job still reaches
   `_clear_job()` through its worker's `finished` signal after the widget
   closes.
6. **Jobs** (added on 2026-11-06, after 3c.2b, as 3c.7). Feature Extraction
   and Spatial Query do not start on a busy table either. Their controllers
   read `table_busy_reasons(...)` through `TableBusyMarker` and refuse; their
   widgets disable Calculate and Run with the reason, and refresh on
   `table_busy_state_changed`. Without it, a Spatial Query centers job and a
   Feature Extraction job could write one table at the same time, and a
   widget closed mid-job and opened again gets a new controller, whose
   one-job limit does not see the old job.

This replaces a first, pull-based implementation of the same day, in which
app state asked every participant's `table_persistence_blocker(...)` before a
Write or Reload, and the user learned of the block only by clicking. The
two-pass reload of P10 stays: its first pass becomes the busy check.

Risk: a mark that is never released locks the table. The release sits on the
path that clears the job today (`_clear_job()`), which every end of a job
reaches; a test covers the errored and the detached job.

Out of scope: the headless API writes predictions directly, but has no widgets
and no running jobs; Shapes saves write a different element of the store.

## Tests to Add

- lazy (dense Dask) and backed `obsm` feature matrices through metadata
  inspection, the classifier summary, training and prediction (D1, step 3):
  - the summary reads no matrix values;
  - a job reads only its rows, in the worker, and skips non-finite rows;
  - the same for a matrix from `hp.tb.add_feature_matrix` and a custom one;
  - a labels element without calculated features (all rows NaN) in the scope:
    the job skips its rows, and the message names it;
- lazy sparse `X` coloring by var (labels and shapes);
- Spatial Query with harpy's canonical centers (P8): valid centers are reused
  without recomputation; absent or stale centers are recomputed with
  `add_canonical_centers`; only the selected labels element's rows are read;
  backed SpatialData records a persisted change and the table is busy while the
  worker writes; in-memory SpatialData records a mutation; a version-1 record
  is recomputed; a table whose centers cannot all be computed shows harpy's
  reason;
- reserved `.obsm` keys (P9): Feature Extraction rejects them before
  launching; Object Classification does not offer or register them;
- reload preparation in two passes (P10): a blocked reload leaves every
  participant's work untouched, whatever the registration order; the error
  lists every blocker's reason; an unblocked reload invalidates every
  affected participant once;
- a table is busy while a job writes it (P11): app state counts two marks on
  one table as busy until both are released, and emits the busy signal only
  on a change; Write and Reload are disabled with the reason while busy, and
  enabled again on release; the guards raise for a busy table; Feature
  Extraction releases its mark when its job returns, errors, or finishes after
  the widget closed; Feature Extraction and Spatial Query refuse to start on a
  busy table, and their widgets disable Calculate and Run with the reason
  until it is released;
- backed `zarr.Array` / `CSRDataset` consumers (P3);
- reload keeps `obsm` lazy (P4);
- persistence through harpy's API: write, delete, rollback on failure,
  overwriting a lazily attached `obsm` key (P5);
- feature-key overwrite (P7), on a multi-region dataset (e.g. the
  `sdata_blobs_multi_region` fixture), in memory and backed:
  - compatible entry: only the selected labels elements' rows change;
  - compatible with channels given as indices, names, or `None`, and with
    features in a different order across the intensity and morphology groups;
  - different features or channels: full replacement, other rows become NaN,
    warning logged;
  - custom matrix without harpy metadata: full replacement;
  - metadata without a matrix: no dialog, `"created"` event;
  - `overwrite_feature_key=False` with an existing entry raises, in memory and
    backed;
  - the check result changes between dialog and worker: the job aborts and the
    table is unchanged;
  - backed, entry only in memory: full-replacement dialog; entry with unsaved
    changes: blocked;
  - full replacement publishes `"updated"` with every table region; a failure
    after the removal publishes `"removed"`, and Object Classification drops
    the key;
  - widget shows the matching confirmation, and Cancel leaves the table
    unchanged.

## Non-Goals

- Changing Spatiato's dirty-tracking model to harpy's value-diff
  `write_table_updates(...)`: it compares the whole table with the store and
  never deletes, while Spatiato already knows which paths changed.
- Writing `X`, `var`, `layers`, or `raw` from Spatiato.
- Locking or crash-atomic writes (neither Spatiato nor harpy provides them).

## Open Questions

- The D1 open point (see "Migration Plan", step 2), as a later optimization
  only if reads turn out noticeably slow: chunking of lazy reads (and lazy
  against backed `X`).
- Follow-up after 3c.2: extract one "table-writing job" helper shared by the
  Feature Extraction and Spatial Query controllers (3c.2 design point 7).

Settled on 2026-10-30 (see "Migration Plan", step 2):

- D1: every table matrix stays lazy. Feature extraction keeps the lazy `obsm`
  that harpy attaches; the reader and the reload read lazily.
- Unsupported matrix encodings: no fallback in Spatiato; a lazy read raises
  harpy's error.
- Step 3d design points 1-4 (see "Migration Plan", step 3): counts before
  and after a job, config counts from the job result, ineligibility found in
  the worker, and the data structures.

Settled on 2026-11-01 (see P8):

- Spatial Query relies on harpy for canonical centers, all or nothing: no
  regional updates, no canonical logic left in Spatiato. harpy 0.5.3 supports
  SpatialData without a backing store (settled on 2026-11-05).

Settled on 2026-11-05:

- Target harpy 0.5.3 (`harpy-analysis>=0.5.3`), which ships the canonical-center
  refactor (P8) and reserves two `.obsm` keys (P9, implemented as 3c.4 before
  3c.2).
- 3c.2 design points 1-8 (see "Migration Plan", step 3), including the reload
  participants (point 8): the controller blocks reloads while its centers job
  writes, and the widget invalidates its run.
- P10: reload preparation in two passes (3c.5, before 3c.2b). First every
  participant may block the reload, without side effects; only then do they
  invalidate their work.

Settled on 2026-11-09:

- Step 4b readiness re-check (see "Migration Plan", step 4, design points
  5a-5d): the reload request reads `spatialdata_attrs` once and takes
  requested paths below it from that value; a non-mapping parent on disk
  fails the reload with harpy's error; a full reload lists the `obs` columns
  from memory and the `obsm` and `uns` keys by reading `obsm` (lazily) and
  `uns`, so `obs` is read once per reload. Reading the three roots once, in
  the reload, was considered and not taken: it would change
  `TableReloadRequest` from exact paths to "full reload plus dirty paths". A
  test for the in-memory rollback of a reload is added, since none existed.

Settled on 2026-11-08:

- Step 4 readiness check (see "Migration Plan", step 4): split into 4a,
  writes through harpy (P5), and 4b, reload through harpy (P4).
  `write_table_components(...)` keeps its signature and maps Spatiato's paths
  onto harpy's replacements and deletions, through the adapter
  `add_table_components(...)`, which reopens what it writes into the attached
  table (lazily for `obsm`) and restores memory on failure; this replaced,
  the same day, the store-only writer with re-attach code in Spatiato. Write
  reports every error, as Reload does; `write_table_prediction_state(...)` is deleted, and
  headless calls `write_table_components(...)`. Reload reads lazily with
  `read_table_components(..., missing="omit")`. Accepted: after a Write, the
  attached `obs` and written `uns` entries are copies read back from the
  store; staging needs a writable parent directory of the store; until step
  5, reloaded `obsm` is lazy while a freshly opened store is not.

Settled on 2026-11-07:

- Step 3d design point 5, revised: `uns["classifier_config"]` keeps only the
  fields a reader uses or that record how the model was obtained (13 fields);
  nothing is kept only for compatibility. `table_name`, `eligible`,
  `roi_mode`, `n_training_rows` and `n_predicted_rows` go;
  `n_labeled_objects` becomes `n_training_labels`.
- Step 3d.2 readiness check, design points 9-11 (see "Migration Plan", step
  3): `_read_rows_with_finite_features(...)` moves to the core, so headless
  `apply_classifier(...)` reads only its prediction rows through it;
  `normalize_feature_matrix(...)` is deleted; `uns["classifier_apply_config"]`
  drops `applied`, `table_name`, `n_predicted_rows` and
  `n_skipped_feature_invalid_rows`.
- Step 3d.3, design points 12 and 13 (see "Migration Plan", step 3): one
  `uns["classifier_config"]`, keyed by prediction class column, shared by
  the widget and headless; whoever writes a column replaces its entry, so
  `classifier_apply_config` goes and nothing is deleted from the store. The
  entry nests the model's provenance as `model_provenance` (`None` when
  nothing was trained), and the export bundle carries it under the same
  name (not `model`, which next to the bundle's `estimator` would read as
  the fitted model itself), with bundle schema version 2. After a headless
  apply, the widget behaves as if nothing was trained: a reload without a
  widget-written `"pred_class"` entry asks the user to train. The widget
  does not see headless run on the viewer's own SpatialData object
  (documented), and training again keeps the headless predictions outside
  the widget's prediction scope. No migration of old stores or bundles.
  This replaced, the same day, a first version with two configs that
  removed each other.
- Step 3d.4, design point 14: the confidence column is
  `f"{pred_class_column}_confidence"`, so the default becomes
  `pred_class_confidence` and headless loses `pred_confidence_column`.
  From the readiness check (2026-11-08): one helper derives the name, and
  the internal writers derive it too, so no pair can mismatch; a table with
  the old `pred_confidence` is refused by the existing pair checks until the
  user fixes it by hand (no migration); a class column named like
  `a_confidence` colliding with `a`'s confidence column is accepted.

Settled on 2026-11-06:

- P11: a table is busy while a job writes it, and a busy table can be neither
  written nor reloaded (3c.6, after 3c.5 and before 3c.2b). App state tracks
  the busy state per table, as it tracks the dirty state; the persistence
  controls disable Write and Reload with the reason; the job's owner marks and
  releases; reload participants keep only `prepare_for_table_reload`.
- P11, point 6 (3c.7, after 3c.2b): Feature Extraction and Spatial Query do
  not start on a busy table; their widgets disable Calculate and Run with the
  reason. Jobs on different tables of one store, and a Spatial Query Run that
  only queries, stay unprotected (3c.2 design point 6).
- 3c.2 design point 9 (see "Migration Plan", step 3): Spatial Query's
  controller separates an attached operation (`is_running`) from a centers job
  still writing (`has_job_in_flight`); Run waits for both, and a detached job
  is detached explicitly on cancel, without reattachment.
- Step 3d readiness check, design points 5-8 (see "Migration Plan", step 3):
  the counts in `uns["classifier_config"]` get defined meanings;
  `normalize_feature_matrix(...)` goes; classifier jobs still do not mark the
  table busy; an ineligible result from the worker reaches
  `_apply_ineligible_state(...)` with its reason and counts. 3d is split into
  3d.1 (the widget path) and 3d.2 (headless and the cleanup).

Settled on 2026-10-27 (see "Migration Plan"):

- Spatiato's napari reader (`_reader.py`, currently `spatialdata.read_zarr`,
  eager tables) switches to `hp.io.read_zarr`, as the last migration step.
- P5 uses harpy's store-only functions (Spatiato installs results in memory
  itself), not the SpatialData adapters, which would reattach lazily.
  Revised on 2026-11-08: P5 writes through the adapter
  `add_table_components(...)`, because under D1 a written `obsm` key should
  be reattached lazily anyway (see "Migration Plan", step 4).
