# Bump Harpy to 0.5.0: Lazy Matrices and Table Component I/O

Status: investigation done. Problems are listed below and will be tackled one by
one; each section's direction is a proposal to settle when we pick it up.

## Goal

Move Spatiato to `harpy-analysis>=0.5.0` and handle the two harpy 0.5.0 changes
that affect Spatiato:

- **lazy matrices**: harpy now attaches table matrices as Dask arrays or backed
  on-disk handles instead of in-memory arrays;
- **table component I/O**: harpy now has its own API to read, write, and delete
  individual table components, overlapping with Spatiato's own reader/writer.

## Environment

- `pyproject.toml` raises `harpy-analysis>=0.4.4` to `>=0.5.0` (and
  `napari[all]>=0.4.18` to `>=0.7.1`); this change is not committed yet.
- `.venv` now has **harpy 0.5.0** (with `spatialdata 0.7.3`, `anndata 0.12.16`);
  see P0 for the first test run.
- harpy 0.5.0 requires `anndata>=0.12.10`, `spatialdata>=0.7.3`, and
  `zarr>=3.2.1`. Spatiato still declares `spatialdata>=0.7.2`.
- harpy 0.5.0 source reviewed at tag `v0.5.0` (`6299266`). Most of the new code
  landed in "Feature/table qc (#264)"; the storage design is documented in
  harpy's `docs/development/storage.md`.

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
    is a **Dask array**. For unbacked SpatialData the same adapter installs a
    "lazy merge" of the existing and new rows (per its docstring); not yet
    verified at runtime.
- `aggregate_points` / `allocate` and `add_canonical_centers` attach **backed**
  handles (`CSRDataset`, `zarr.Array`), not Dask arrays.

harpy's feature-matrix metadata (`uns["feature_matrices"][<key>]`) is unchanged
in shape; `backend` is hard-coded `"numpy"` and laziness is not recorded.

There is no public "materialize" helper; options are `mode="eager"` or
`adata.to_memory()`.

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

## Next harpy: `feature/qc_metrics` Branch

Checked on 2026-10-22 at commit `e9b9bb1`, 171 commits ahead of `v0.5.0` (not
released yet).

Unchanged for P7:

- `table/_add_feature_matrix.py` and `table/io/_components_by_region.py` are
  identical to `v0.5.0`: regional merge and its compatibility checks are the
  same.
- `_SOURCE_KIND = "harpy_add_feature_matrix"`; `_FEATURE_MATRIX_SCHEMA_VERSION`
  is still `1`, now defined in `harpy/table/_metadata.py`.
- `hp.tb.io.remove_table_components(sdata, *, table_name, components)` is
  unchanged and still exported.

Relevant elsewhere:

- **anndata floor `>=0.12.10` -> `>=0.13`** (Spatiato's `.venv` has 0.12.16).
  In anndata 0.13 `X` lives in memory as `layers[None]`; harpy adapts
  internally. Spatiato only uses `table.X`, never `layers`, so the risk is low,
  but run the suite on anndata 0.13 when this lands.
- **zarr v3 sharding**: per harpy's updated docstrings, anndata 0.13 enables
  sharding by default (`auto_shard_zarr_v3`). harpy writes dense matrices with
  `shards=None` because of https://github.com/saeyslab/harpy/issues/270.
  Spatiato's own `ad.io.write_elem` writes in `core/persistence.py` do not opt
  out; see P5.
- harpy removed its workaround for lazily reading empty sparse matrices,
  relying on anndata 0.13 (background for P2/P3).
- New public `hp.tb.validate_table(sdata, table_name, feature_matrices_key=...)`
  validates feature-matrix records against `obsm`. It does not replace P7's
  merge-compatibility check.
- The `numba<0.67` pin in harpy's working-tree `pyproject.toml` is a uv override
  for harpy's own environment, not a runtime requirement; it does not reach
  Spatiato.

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

## Problems

### P0. Run the suite on harpy 0.5.0

- Maintainer installs harpy 0.5.0 into `.venv` (the environment is managed
  manually, per `CLAUDE.md`).
- Raise Spatiato's `spatialdata` floor to `>=0.7.3` to match harpy 0.5.0.
- Run the full suite and triage failures. Expect at least feature-extraction
  tests to observe a lazy `obsm[feature_key]`.
- Verify the unbacked `add_feature_matrix` result type at runtime (the adapter
  docstring describes a lazy merge).

First run on harpy 0.5.0 (`-n auto`):

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

### P1. Dense Dask feature matrices are fully materialized on every use

After P0, feature extraction leaves `obsm[feature_key]` as a Dask array. It
works, but `normalize_feature_matrix(...)` reads the whole matrix from the store
on every call, including on every Object Classification refresh. On large tables
(e.g. ~407k cells) this makes widget refreshes slow.

Direction:

- read `shape` / `dtype` without materializing (summary, metadata inspection,
  column-count validation);
- compute the finite-row mask once and cache it per matrix (e.g. keyed by table,
  `obsm` key, and matrix object identity);
- materialize once per classifier job, preferably inside the worker thread
  instead of in `_prepare_classifier_job(...)` on the main thread.

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

Direction: read with `hp.tb.io.read_table_components(..., mode="lazy")` and keep
Spatiato's reload-binding validation, `TableModel.validate`, and in-memory
rollback around it.

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
  `reload_table_components`) with `hp.tb.io` calls;
- map paths: `("obs", col)` -> `("obs",)` (whole dataframe, as today),
  `("obsm", key)` -> `("obsm", key)`, `("uns", *keys)` -> `("uns", *keys)`;
- pass `obs_identity` for obs-aligned writes; keep including `_colors` palette
  paths explicitly, as widgets already do.

Additional reason (next harpy, anndata 0.13): anndata 0.13 shards zarr v3
arrays by default, and harpy deliberately writes dense matrices without shards
(https://github.com/saeyslab/harpy/issues/270). Writing through harpy keeps
Spatiato's writes consistent with harpy's storage choices.

### P6. Feature-matrix metadata `backend` label

`register_feature_matrix_metadata(...)` records `"numpy"` for a Dask matrix;
harpy also hard-codes `"numpy"`. Decide whether to record the real backend or
drop the field from Spatiato's interpretation. Cosmetic; lowest priority.

### P7. `overwrite_feature_key` changed meaning in harpy 0.5.0

Status: decisions agreed; not implemented yet.

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
   - The widget's two dialogs both lead to `overwrite_feature_key=True`.

Implementation notes:

- Classify the existing entry into one of three outcomes before starting:
  - **none**: no `obsm[key]` and no metadata record; plain extraction, the
    overwrite flag is not needed;
  - **compatible**: harpy's regional merge will succeed; replace the selected
    regions;
  - **incompatible**: anything else, including custom matrices; remove and
    replace the whole matrix.
- Decide compatibility without knowing harpy's column names: existing metadata
  has the expected `source_kind` and `schema_version`, its `features` and
  `source_channels` equal the request, and the matrix is dense with `n_obs`
  rows. If harpy still raises, surface its error.
- **Check the same state harpy checks.** For backed SpatialData, harpy reads the
  existing matrix and metadata from the **store** (`_existing_feature_matrix`
  in `harpy/table/_add_feature_matrix.py`), not from memory; only unbacked
  SpatialData is checked in memory. harpy's `overwrite_feature_key` "only
  concerns the store": an entry present only in memory is replaced without it.
  A memory-only pre-check could therefore pick a different dialog than harpy's
  actual behavior whenever memory and store differ.
- The labels elements named in the full-replacement warning and dialog are the
  table's regions that are not selected for extraction.
- Shared core function (e.g. in `core/feature_extraction.py`) used by the
  widget's `_run_feature_extraction_job(...)`
  (`widgets/feature_extraction/controller.py`) and by
  `headless.compute_features_for_classifier(...)`. The removal runs in the same
  worker as the extraction.
- The widget uses the same compatibility check to choose which confirmation to
  show before launching the job.
- Backed SpatialData: keep calling `record_persisted_table_change(...)` for
  `("obsm", key)` and `("uns", ("feature_matrices", key))` after extraction,
  as today.
- Update the `overwrite_feature_key` docstrings in `headless.py` to describe
  both cases.
- Fix `test_compute_features_for_classifier_returns_normalized_target` by
  dropping the pre-seeded matrix; the test is about target normalization.

Implementation order:

1. Shared core function (classification, regional or full replacement,
   warning), headless wiring and docstrings, and the core tests. This makes the
   suite green.
2. Widget: use the classification to pick the confirmation ("Replace selected
   regions" or "Replace entire feature matrix"), and the widget tests.

## Tests to Add

- dense Dask `obsm` feature matrix through metadata inspection, classifier
  summary, training, and prediction, asserting no repeated full materialization
  where P1 applies;
- lazy sparse `X` coloring by var (labels and shapes);
- backed `zarr.Array` / `CSRDataset` consumers (P3);
- reload keeps `obsm` lazy (P4);
- persistence through harpy's API: write, delete, rollback on failure,
  overwriting a lazily attached `obsm` key (P5);
- feature-key overwrite (P7), on a multi-region dataset (e.g. the
  `sdata_blobs_multi_region` fixture), in memory and backed:
  - compatible entry: only the selected labels elements' rows change;
  - different features or channels: full replacement, other rows become NaN,
    warning logged;
  - custom matrix without harpy metadata: full replacement;
  - widget shows the matching confirmation, and Cancel leaves the table
    unchanged.

## Non-Goals

- Changing Spatiato's dirty-tracking model to harpy's value-diff
  `write_table_updates(...)`.
- Writing `X`, `var`, `layers`, or `raw` from Spatiato.
- Locking or crash-atomic writes (neither Spatiato nor harpy provides them).

## Open Questions

- Should Spatiato's napari reader (`_reader.py`, currently
  `spatialdata.read_zarr`, eager tables) switch to `hp.io.read_zarr` with lazy
  tables?
- Should feature extraction keep the lazy `obsm` that harpy attaches, or
  materialize it once after extraction for interactive use?
- For P5: use harpy's store-only functions (Spatiato installs results in memory
  itself) or the SpatialData adapters (`add_table_components`, which also
  reattach lazily)?
