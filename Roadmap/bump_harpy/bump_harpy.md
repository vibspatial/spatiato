# Bump Harpy to 0.5.2: Lazy Matrices and Table Component I/O

Status: investigation done. Problems are listed below and will be tackled one by
one, in the order of "Migration Plan"; each section's direction is a proposal to
settle when we pick it up. Target moved from harpy 0.5.0 to 0.5.2 on
2026-10-25.

## Goal

Move Spatiato to `harpy-analysis>=0.5.2` and handle the two harpy 0.5.0 changes
that affect Spatiato:

- **lazy matrices**: harpy now attaches table matrices as Dask arrays or backed
  on-disk handles instead of in-memory arrays;
- **table component I/O**: harpy now has its own API to read, write, and delete
  individual table components, overlapping with Spatiato's own reader/writer.

harpy 0.5.1 also raises the `anndata` floor to `>=0.13`, so the move to anndata
0.13 is part of this bump.

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

- `pyproject.toml`: `harpy-analysis>=0.5.2` (raised from `>=0.4.4` to `>=0.5.0`
  in `d895ea5`, then to `>=0.5.2`), `spatialdata>=0.8.0` (needed with anndata
  0.13, see P0), `napari[all]>=0.7.1`. Spatiato does not declare `anndata`;
  harpy's floor applies.
- `.venv` has **harpy 0.5.2**, `anndata 0.13.4`, `spatialdata 0.8.0`,
  `dask 2026.8.0`, `zarr 3.2.1`, `numpy 2.3.5`; `uv.lock` is refreshed. See P0
  for the test runs on 0.5.0 and 0.5.2.
- harpy 0.5.2 requires `anndata>=0.13`, `spatialdata>=0.7.3`, and `zarr>=3.2.1`.
  spatialdata 0.7.3 does not work with anndata 0.13, so in practice it needs
  `spatialdata>=0.8.0` (see P0).
- harpy releases:

  | Version | Tag commit | Content |
  |---|---|---|
  | 0.5.0 | `6299266` | lazy matrices and table component I/O ("Feature/table qc (#264)") |
  | 0.5.1 | `a65e793` | "Feature/qc metrics (#271)", the `feature/qc_metrics` branch; anndata `>=0.13` |
  | 0.5.2 | `5ff8c3e` | "Fix/cosmx reader fov coordinates (#275)", including the fix for https://github.com/saeyslab/harpy/issues/274 |

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
   - D1 is about matrices read from a store. On unbacked SpatialData a Dask
     matrix is an in-memory graph with no store behind it, and harpy 0.5.2
     keeps chaining merges onto it. The P1 safeguard computes it once after a
     feature extraction (step 3); this does not contradict D1.

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
     `CSRDataset`/`CSCDataset`); one reads one column for coloring by var (P2);
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
   - spatial query's `obsm["spatial_canonical"]` (n × 2) stays as is:
     `np.asarray(...)` computes dense Dask and `zarr.Array`, and the helpers
     cover the rest;
   - feature extraction on unbacked SpatialData gets the P1 safeguard: if
     `obsm[feature_key]` is still a Dask array after extraction, compute it once
     and reinstall it (see P1, "Resolution");
   - tests with lazy and backed matrices per consumer.

   Needed regardless of D1: harpy already attaches lazy matrices after a backed
   extraction, and P2 was hit in real use. This step comes before the reader
   switch, which would otherwise spread the P2 bug to every table.

   Readiness check (2026-10-30): every sub-step is ready to implement; the
   four design points for 3d below were settled the same day. Sub-steps, each
   committed separately:

   | Sub-step | Content | Ready? |
   |---|---|---|
   | 3a | Matrix helpers, with unit tests per representation | yes |
   | 3b | Coloring by `X` var through the column helper (P2) | yes |
   | 3c | Shape and dtype without loading (metadata inspection and registration, export and column checks), spatial query through the helpers, the P1 safeguard | yes |
   | 3d | Object Classification refactor and headless `apply_classifier(...)`, with test updates | yes (design points 1-4 settled) |

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
     multi-block Dask array of `csr_array` blocks;
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

   Also from the check: `_validate_canonical_matrix(...)` (spatial query)
   rejects a sparse Dask or backed matrix only by accident (the 0-d result of
   `np.asarray(...)` fails its shape check); the helpers' sparse detection
   makes that explicit. The P1 safeguard belongs in
   `calculate_feature_matrix(...)` after `hp.tb.add_feature_matrix(...)`; it
   runs in the worker, where harpy already installs its results.

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
4. **Swap the table I/O** (P5 writes, P4 reload):
   - writes through harpy's store-only `hp.tb.io.write_table_components(...)`
     and `delete_table_components(...)`. Spatiato's dirty tracking already
     knows which paths changed, which maps onto explicit components and
     deletes. The store-only functions leave the in-memory values alone; the
     SpatialData adapters (`add_table_components(...)` and the like) would
     replace them with lazy views of the store on every save;
   - not `write_table_updates(...)`: it compares the whole table with the store
     and never deletes, so it does not fit explicit dirty paths (non-goal);
   - reload through `hp.tb.io.read_table_components(...)`, lazily (D1);
   - removed from `core/persistence.py` (416 lines): the direct zarr and
     anndata reads and writes, zarr group-structure handling
     (`_resolve_backed_table_group(...)` and the parent-chain helpers), and
     metadata consolidation. Kept: `TableComponentPath`, dirty tracking in
     `SpatiatoAppState`, the reload checks (`_validate_reload_binding(...)`),
     and restoring the in-memory state when a reload fails.
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
- **Spatiato**: only a safeguard is left. After extraction on unbacked
  SpatialData, if `obsm[feature_key]` is still a Dask array (the attached matrix
  was a Dask array before extraction, e.g. a table read lazily and attached to
  in-memory SpatialData, or a custom Dask matrix), compute it once and
  reinstall it. This stops the chain and gives interactive use an in-memory
  matrix; it is a no-op for in-memory matrices. Planned in step 3 of the
  "Migration Plan"; it does not contradict D1, which is about matrices read
  from a store.

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
shapes styling both use the column helper:

- select the column first (`X[:, var_index]`), and `.compute()` it if it is a
  Dask array;
- if the result is sparse, call `.toarray()`; then
  `np.asarray(..., dtype=np.float64).reshape(-1)`;
- then select the rows with `[obs_positions]` in memory.

Selecting the column first avoids having Dask reorder all rows by a long
position array on large tables. For Dask arrays of CSR blocks split by rows this
still reads every block, but never builds the dense matrix.

### P3. Backed handles are untested

`zarr.Array` and `CSRDataset` / `CSCDataset` (from `aggregate_points`,
`add_canonical_centers`, or `read_table(mode="backed")`) have only been tried in
`X`-var coloring (works, see P2). Every other Spatiato path is untested,
including spatial query's `obsm["spatial_canonical"]`.

Direction: cover them with the P2 helper and add tests per consumer.

### P4. Reload drops laziness

`reload_table_components(...)` uses `ad.io.read_elem`, so "Reload Table State"
replaces a lazy feature matrix with an in-memory copy.

Direction: read with `hp.tb.io.read_table_components(...)`, lazily (D1; see
"Migration Plan", step 4), and keep Spatiato's reload-binding validation,
`TableModel.validate`, and in-memory rollback around it.

### P5. Replace Spatiato's on-disk writes with harpy's component API

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
  `reload_table_components`) with `hp.tb.io` calls, using harpy's store-only
  functions rather than the SpatialData adapters (see "Migration Plan",
  step 4);
- map paths: `("obs", col)` -> `("obs",)` (whole dataframe, as today),
  `("obsm", key)` -> `("obsm", key)`, `("uns", *keys)` -> `("uns", *keys)`;
- pass `obs_identity` for obs-aligned writes; keep including `_colors` palette
  paths explicitly, as widgets already do;
- block table writes while a feature extraction on the same table is still
  writing, as reloads already are (`prepare_for_table_reload(...)` in
  `widgets/feature_extraction/controller.py`). Today Write is not blocked, so
  Spatiato's own writer and harpy's can run on the same store at once, which
  harpy does not protect against (see P7, step 2 design, point 7).

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

## Tests to Add

- lazy (dense Dask) and backed `obsm` feature matrices through metadata
  inspection, the classifier summary, training and prediction (D1, step 3):
  - the summary reads no matrix values;
  - a job reads only its rows, in the worker, and skips non-finite rows;
  - the same for a matrix from `hp.tb.add_feature_matrix` and a custom one;
  - a labels element without calculated features (all rows NaN) in the scope:
    the job skips its rows, and the message names it;
- lazy sparse `X` coloring by var (labels and shapes);
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

Settled on 2026-10-30 (see "Migration Plan", step 2):

- D1: every table matrix stays lazy. Feature extraction keeps the lazy `obsm`
  that harpy attaches; the reader and the reload read lazily.
- Unsupported matrix encodings: no fallback in Spatiato; a lazy read raises
  harpy's error.
- Step 3d design points 1-4 (see "Migration Plan", step 3): counts before
  and after a job, config counts from the job result, ineligibility found in
  the worker, and the data structures.

Settled on 2026-10-27 (see "Migration Plan"):

- Spatiato's napari reader (`_reader.py`, currently `spatialdata.read_zarr`,
  eager tables) switches to `hp.io.read_zarr`, as the last migration step.
- P5 uses harpy's store-only functions (Spatiato installs results in memory
  itself), not the SpatialData adapters, which would reattach lazily.
