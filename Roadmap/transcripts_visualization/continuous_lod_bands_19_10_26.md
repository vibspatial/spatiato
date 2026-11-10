# Continuous level of detail with additive priority bands — 19 October 2026

## Summary

The current points cache stores a replacement pyramid: Exact, a same-grid Bridge, and a series of Spatial levels, each a sampled copy of its finer level with a fixed per-tile capacity. Measurements on the full Xenium cache show four problems with that design: per-tile quotas flatten density and produce tile-boundary artefacts, switching between levels changes the drawn density abruptly, the viewer under-fills its point budget at mid zoom, and the sampled levels duplicate 36% extra rows in each physical order.

This document proposes replacing the stored pyramid with **additive priority bands** and a **continuous detail threshold**:

- every transcript receives one deterministic pseudorandom priority, shared by both physical orders;
- the viewer draws the transcripts whose priority lies below a threshold `t`, chosen continuously from the point budget and the number of selected transcripts in view;
- every transcript is stored exactly once per physical order, in the band that contains its priority; low-priority bands use coarse tiles and high-priority bands use fine tiles;
- band boundaries and band tile sizes follow from the screen-density target and the data's transcript density, not from the render budget or the canvas size;
- inside every band tile, rows are sorted by priority, so a partially needed band is read as a prefix of each tile.

The tile-major and gene-major orders, Zarr storage, tile-relative coordinates, and catalog-first planning remain. A simulation from the real per-tile counts indicates that the design fills the budget in view at every zoom, decoding at most about 2.7 rows per row shown (1.5 with smaller band tiles); reads about as many rows as the current pyramid along a slow zoom from the whole section to a 0.25 mm view; and stores 27% fewer rows per order.

**Status:** design proposal for discussion. Nothing has been implemented or rebuilt; the numbers come from a read-only simulation over the published cache's catalog arrays.

This continues the direction recommended in [sampler_17_9_26.md](/Users/arne.defauw/VIB/napari_harpy/Roadmap/transcripts_visualization/sampler_17_9_26.md) section 5: a common sampling fraction applied consistently across tiles, implemented with deterministic point priorities and level-wide thresholds.

## 1. Problems with the current pyramid

All numbers below come from the published cache of the full Xenium section used in earlier roadmaps: 136,578,750 transcripts, 5,122 values, 7,294 full-resolution tiles of 512 units (108.8 µm), and nine serialized levels.

### Per-tile quotas flatten density

Full-resolution tiles hold a median of 5,013 transcripts, a p90 of 66,045, and at most 108,598. Bridge keeps `min(Exact points, 4,096)` per tile, so **56.5% of tiles (4,123)** are capped at exactly 4,096. Neighbouring capped tiles whose original counts differ up to 26× are drawn at the same density. At every coarser Spatial level, 74–91% of tiles sit at their capacity. The 16 × 16 microgrid preserves relative density inside one tile but not between tiles; see the earlier sampler note for the mechanism and a reproduced example.

### Level changes are abrupt

Going from Exact to Bridge reduces the whole section from 136.6M to 21.7M points, 6.3× in one step. Per tile the step is up to 26× for the densest tile (108,598 → 4,096) and about 16× for a p90 tile (66,045 → 4,096). Hysteresis changes when the switch happens; it does not soften it.

### Mid zoom under-fills the budget

The viewer's effective budget is the screen-density budget capped by the hard limit. With [DEFAULT_TARGET_PIXELS_PER_POINT](/Users/arne.defauw/VIB/napari_harpy/src/spatiato/viewer/tiled_points/contracts.py:19) = 0.25 (four points per screen pixel) and [DEFAULT_HARD_RENDER_POINT_BUDGET](/Users/arne.defauw/VIB/napari_harpy/src/spatiato/viewer/tiled_points/contracts.py:18) = 2,000,000, the effective budget is usually 2M.

Level selection counts every point of each intersecting tile, including points outside the view. For a 1 mm view in the densest area, no level finer than Bridge fits a 2M budget, so the viewer shows about 0.50M points in view while 2M would be allowed. With a 5M budget the same view still shows 0.50M. Zooming in further switches to Exact and multiplies the drawn points by the per-tile factors above.

### Duplication and level count

The nine levels store 186,056,149 rows per physical order for 136,578,750 transcripts: 36% extra, in both the tile-major and the gene-major order. Adding bridge levels to soften transitions would add more duplication and more discrete switches.

## 2. Proposal

### Priority

Each transcript receives one priority `p` in `[0, 1)`, a versioned hash of its stable point identity. Unlike the current sampler, `p` does not depend on level, tile, or microgrid cell. Both physical orders use the same `p`. Because `p` is uniform and independent of position and value, the set `p < t` is a uniform random sample at rate `t`, with the same expected fraction in every tile and for every gene.

### Continuous threshold

For a viewport and a selection, the viewer computes:

```text
N_view = selected transcripts in the full-resolution tiles intersecting the view
t      = min(1, budget / N_view)
```

and draws the selected transcripts in view with `p < t`. `t` changes continuously with zoom and budget. Increasing `t` only adds points, so zooming in never removes a point that was already drawn in the overlapping region. A narrow opacity ramp just below `t` can make admission of new points less visible.

### Additive bands

Band `k` holds the transcripts with priority in `[l_k, u_k)`; every transcript is stored once per order, in the band that contains its priority. The bands are a storage and loading structure, not displayed levels: what the user sees is determined by `t` alone, and any band boundaries produce the same image. Band boundaries only decide how efficiently a given `t` can be read. They are therefore a build-time choice and must not depend on runtime settings such as the hard render budget or the canvas size; a different budget or window size only changes `t`.

### Band boundaries from the density target

The screen-density target turns the bands into zoom ranges. At a screen pixel size of `s` µm, a target of `ρ` points per pixel, and a transcript density of `D` per µm², the viewer wants the rate:

```text
t = min(1, ρ / (s² · D))
```

Halving the pixel size multiplies `t` by four. Bands are therefore defined from the finest end, with upper bounds that shrink by a factor of four:

```text
upper bounds u_k:      100%, 25%, 6.25%, 1.5625%, ...     (lower bound of a band = upper bound of the next coarser band)
band complete at s_k:  s_finest = sqrt(ρ / D_ref), doubling per coarser band
```

The coarsest band holds every priority below its upper bound. Bands are added until the coarsest band is small enough to read as an overview.

A band tile is `P` screen pixels wide at the pixel size where its band is complete, rounded to a power-of-two multiple of the full-resolution tile so that band tiles align with the existing grid. At the reference density, each band tile then holds about the same number of rows, roughly `3/4 · ρ · P²`.

Build parameters:

- `ρ`: the screen-density target. spatiato's default is 4 points per pixel ([DEFAULT_TARGET_PIXELS_PER_POINT](/Users/arne.defauw/VIB/napari_harpy/src/spatiato/viewer/tiled_points/contracts.py:19) = 0.25);
- `D_ref`: a reference transcript density, derived from the data;
- `P`: the band tile width in screen pixels;
- a size limit for the coarsest band.

None of them depends on the render budget, the canvas size, or the dataset's total size.

For this section, with `ρ = 4`, `D_ref` = 1.58 transcripts per µm² (the mean over non-empty full-resolution tiles), `P = 128` pixels, and a coarsest band of at most 1M rows:

| Band | Priority range | Fills while zooming from → to | Band tile edge | Rows in band |
|---:|---|---|---|---:|
| 0 | 0–0.39% | coarser than 25 µm/px | 32 tiles (3.48 mm) | 0.53M |
| 1 | 0.39–1.56% | 25 → 12.7 µm/px | 16 tiles (1.74 mm) | 1.60M |
| 2 | 1.56–6.25% | 12.7 → 6.4 µm/px | 8 tiles (0.87 mm) | 6.40M |
| 3 | 6.25–25% | 6.4 → 3.2 µm/px | 4 tiles (0.44 mm) | 25.6M |
| 4 | 25–100% | 3.2 → 1.6 µm/px | 2 tiles (0.22 mm) | 102.4M |

At the reference density a band tile holds about 56k rows, 14 chunks of 4,096. Five bands replace nine levels.

Local density varies across this section from about 0.4 to 9.2 transcripts per µm² between full-resolution tiles, so dense tiles hold several times more rows, and "everything shown" is reached at about 3 µm/px in sparse tissue and 0.7 µm/px in the densest tiles. That affects read efficiency only, because `t` is computed from the transcripts actually in view. Denser or larger datasets, such as Atera-scale data, add bands at the coarse end.

Appendix A walks through the whole path, from the Parquet source to the points drawn, with this section's numbers.

## 3. Physical layout

### Tile-major order

Rows are ordered by `(band, band tile, p)`:

```text
| band 0 ............| band 1 ...............| band 2 ......... | band 3 ...  | band 4 |
| t0 | t1 | ...      | t0 | t1 | ...          |                  |             |        |
  each band-tile run: rows sorted by priority  [lowest p .......... highest p]
```

Each band is a contiguous section and each band tile is a contiguous run. Sorting by value inside a tile is no longer needed, because gene subsets are served by the gene-major order. The point-level `value_id` array remains for colouring all-values views.

### Gene-major order

Rows are ordered by `(value_id, band, band tile, p)`. A value whose transcripts all fit the budget is still one contiguous interval. Point-level value IDs remain implicit from the value interval, as in the current value-major sidecar.

### Index

Resident and small:

- per band: the tile grid and, per band tile, its row start and row count in the tile-major order;
- per value: its row interval in the gene-major order and, per band, its band-tile records with row start and count;
- the full-resolution per-tile totals and per-(value, tile) counts used to compute `N_view`.

Prefix lengths need no stored priorities. Within one band, priorities are uniform, so the first `ceil(f * count)` rows of every needed band tile, with the same fraction `f = (t - l_K) / (u_K - l_K)` for the band `K` containing `t`, form a uniform sample at the same rate across tiles. An alternative is a small per-tile table of counts at fixed sub-thresholds; this is an open question.

## 4. Read path

For threshold `t`, let `K` be the band containing `t`:

- bands `0 .. K-1`: read the complete run of every band tile intersecting the view;
- band `K`: read a prefix of every band tile intersecting the view;
- bands above `K`: read nothing.

Every read is a contiguous row range, touching consecutive chunks only.

### Zoomed out

For the whole section on a 1600 × 1000 canvas, the density target asks for about 6.4M points. The 2M hard limit binds, so `t = 2M / 136.6M ≈ 1.46%`. The reader reads band 0 completely and the first 92% of every band-1 tile. Bands 0 and 1 are adjacent sections of the array, so the overview is essentially one sequential read of about 2.1M rows. The priority filter is applied by the write order, not by scanning rows.

Without bands, with only full-resolution tiles sorted by priority, the same view needs a short prefix from each of the 7,294 tiles: 7,294 separate reads, each decoding at least one 4,096-row chunk to use about 270 rows.

### Zoomed in

At 0.25 mm, `t = 100%` and every band is needed, but only the band tiles intersecting the view: one or a few per band, nine runs in total in the simulation below.

### Why the number of reads stays bounded

A band starts to fill when `t` reaches its lower bound and is complete at its upper bound. When the density target sets `t`, this happens while the pixel size goes from `2 s_k` to `s_k`, during which a band tile spans `P/2` to `P` screen pixels. A view of `W × H` pixels then touches about `W·H / P²` to `4·W·H / P²` tiles of the band being filled, independent of the dataset's size. For a 1000 × 1000 canvas and `P = 128`, that is about 60–240 tiles; for `P = 256`, about 15–60. If band tiles are ordered spatially within a band (row-major or Morton order), the complete bands below coalesce into roughly one run per row of tiles; only the band being filled needs one prefix per tile. Finer bands are not read until the view is small enough.

On large canvases the hard limit binds instead: `t` is lower than the density target would give, so bands are used at finer pixel sizes than they were designed for, and their tiles look larger on screen. That means fewer runs but more over-read. The simulation in section 5 is in this regime, because its views are not tied to a canvas.

### Over-read of coarse bands

When zoomed in, the reader still reads whole tiles of the coarse bands, which extend far beyond the view. Three mitigations:

1. **Keep coarse bands resident.** Bands 0 and 1 are the overview at the 2M hard limit (about 2.1M rows, 17 MB of float32 xy) and are loaded first. When the user zooms in they are usually already in memory, so new reads come only from finer bands near the view.
2. **Use smaller tiles for coarse bands.** Less over-read when zoomed in, more runs when zoomed out.
3. **Split coarse band tiles into spatial sub-blocks,** each sorted by priority. Zoomed-in reads take only the sub-blocks in view; partial reads take one prefix per sub-block.

Option 1 is the simplest and matches how a viewer is used: overview first, then zoom.

### Choosing the band tile width `P`

While a band fills, its tiles span `q` screen pixels, between `P/2` and `P`. A view of `W × H` pixels draws about `ρ · W · H` points, and two costs depend on `q`:

```text
chunk waste    ≈ C / (ρ · q²)        at most one partly used chunk of C rows per partial-band tile in view; wants large tiles
edge over-read ≈ q / W + q / H       tiles extending past the view; wants small tiles
```

Their sum is smallest at:

```text
q* ≈ ( 2C / (ρ · (1/W + 1/H)) )^(1/3)        square view: q* ≈ (C · W / ρ)^(1/3)
```

With `C` = 4,096 rows, `ρ = 4`, and a 1000-pixel view, `q*` ≈ 100 pixels: about 10% chunk waste and 20% edge over-read. On a 1600 × 1000 canvas `q*` ≈ 108 pixels. Since a tile spans `P/2` to `P` while its band fills, `P` ≈ 128–160 keeps most of that range near the optimum.

Implications:

- **Canvas size matters little:** through the cube root, doubling the view width changes `q*` by only 26%, so one build-time `P` suits typical screens.
- **`P` and the chunk size belong together:** `P` scales with `C^(1/3)`; 1,024-row chunks give `q*` ≈ 63 pixels.
- **The build-time `ρ` matters:** when the hard limit binds on a large canvas, bands are used at finer zooms and their tiles appear about √(6.4M / 2M) ≈ 1.8× larger on screen, which is why 64-pixel tiles did better in the simulation of section 5. Building with the effective density (2M / 1.6M pixels ≈ 1.25 points per pixel) gives `q*` ≈ 150 pixels; see open question 2.
- **Remote access favours larger tiles:** a 1000 × 1000 view at `q` ≈ 100 touches about 100 partial-band tiles, so about 100 range requests unless they are coalesced. For HTTP, `P` ≈ 256–512 with more edge over-read is likely better.

A practical build-time rule:

```text
P ≈ 1.4 · (C · W_ref / ρ_build)^(1/3)
```

with `W_ref` a reference view width of about 1000–1600 pixels. Each band's tile edge is rounded to a power-of-two multiple of the full-resolution tile, with at least one full-resolution tile for the finest band, and `P` is recorded in the cache metadata. The model ignores codec cost, shards, and resident data, so a benchmark of `P` = 64, 128, and 256 on the real reader should make the final choice.

## 5. Simulation

### Method

A read-only script estimated read cost from the published cache's catalog arrays, without decoding point payloads. It used the real full-resolution tile counts, assumed uniform priorities (expected counts), rounded every tile read up to whole 4,096-row chunks, and placed zoomed views on the densest full-resolution tile.

The bands follow section 2: `ρ = 4`, `D_ref` = 1.58 per µm², a coarsest band of at most 1M rows, and band tiles of 128 pixels, with 64 pixels for comparison. The runtime budget is spatiato's 2M hard limit. The views are not tied to a canvas size, so the hard limit sets `t` throughout.

"Shown" means points inside the view: for the current pyramid, the in-view share of the selected level's tiles; for the threshold designs, `t * N_view`. The current pyramid selects the finest level whose intersecting tiles fit the budget, without hysteresis.

### Results at a 2M budget (all values)

| View | Current: shown / decoded (reads) | Bands, 128-px tiles: decoded (reads) | Bands, 64-px tiles: decoded (reads) | Flat, no bands: decoded (reads) |
|---|---|---|---|---|
| Whole section | 1.39M / 1.39M (12) | 2.09M (47) | 2.42M (173) | 29.9M (7,294) |
| 4 mm | 1.10M / 1.21M (78) | 2.24M (34) | 2.36M (110) | 4.59M (1,115) |
| 1 mm | 0.50M / 0.50M (121) | 4.34M (24) | 3.00M (58) | 2.25M (121) |
| 0.25 mm | 0.72M / 0.75M (9) | 1.97M (9) | 1.06M (17) | 0.75M (9) |

The threshold designs show 2.0M points in view for the first three views and all 0.72M at 0.25 mm.

Observations:

- The band design fills the budget in view at every zoom. The current pyramid shows 0.50–1.39M of a 2M budget outside full resolution.
- With 128-px band tiles, the rows decoded per row shown are about 1.05 for the whole section, 1.1 at 4 mm, 2.2 at 1 mm, and 2.7 at 0.25 mm. With 64-px tiles they are 1.2, 1.2, 1.5, and 1.5, at the cost of 2–4× more runs. The over-read at small views comes from the hard-limit regime described in section 4: band tiles look larger on screen than designed.
- With a budget in the millions, the flat layout is close to the band design at 1 mm and 0.25 mm, and fails only zoomed out (15× at the whole section). Without the coarse bands, the design does not work zoomed out.
- Stored rows per order fall from 186.1M to 136.6M (−27%).

At a 5M budget the current pyramid still shows 0.50M in the 1 mm view, where a threshold design shows 5.0M. At a 100k budget the current pyramid falls back to the whole-section overview for 1 mm and 4 mm views and shows about 6,300 and 37,000 points in view.

### Cumulative reads during a slow zoom

The large fine band raises an obvious concern: band 4 holds 102.4M rows, 75% of all transcripts, so does zooming in gradually end up reading most of it? A second simulation followed one zoom path into the densest area, from the whole section (11.5 mm wide) to a 0.25 mm view in 37 steps, each 10% narrower, at a 2M budget with 128-px band tiles. Nothing loaded was evicted, so the totals are the unique rows read over the whole path:

| | Rows read over the whole zoom | Share of the band |
|---|---:|---:|
| Band 0 | 0.56M | 104% |
| Band 1 | 1.66M | 104% |
| Band 2 | 3.21M | 50% |
| Band 3 | 3.58M | 14% |
| Band 4 | 2.86M | 2.8% |
| **Bands, total** | **11.9M** | 8.7% of all transcripts |
| Current pyramid, same path | 12.5M | 9.1% of all transcripts |

Shares above 100% come from rounding reads up to whole chunks. With 64-px band tiles the total is 11.5M (8.4%).

The band design reads about as much as the current pyramid along the same path, while every step shows the full budget in view.

Band 4's size does not matter, for two reasons:

1. **Only band tiles near the view are read.** Band 4 is first needed when `t` passes 25%, that is, once the view holds at most 2M / 0.25 = 8M transcripts: about a 1 mm view in the densest area. Only the band-4 tiles (0.22 mm) around that view are touched, tens out of thousands.
2. **Only a prefix of each band tile is read.** When band 4 starts, its prefixes are nearly empty; they grow with `t`, and a band-4 tile is read completely only at `t = 100%`, when the view holds at most 2M transcripts (around 0.5 mm here).

The general rule: within the view, the rows needed are `t * N_view`, which equals the budget; the extra comes only from band tiles extending past the view and from rounding to whole chunks. What a view reads is therefore about the budget plus a margin, independent of how large the bands are. Panning across the whole tissue at full resolution does eventually read most of bands 3 and 4, but that is inherent to viewing every transcript, and the current Exact level reads the same data.

### Limits

Expected counts, all-values reads in the tile-major order only, no latency, no codec or shard effects, no gene-major simulation, and one dataset. Reads are counted per tile run; a real reader would batch runs per shard, and spatially ordered complete bands would coalesce. The views are not tied to a canvas size, so the density-bound regime of section 4 is not simulated. The zoom-path simulation follows one path into the densest area and never evicts loaded data.

## 6. What changes and what stays

Stays:

- the dual layout (tile-major for all values, gene-major for selected values);
- Zarr v3 arrays with sharding, tile-relative float32 coordinates, deterministic construction, staged publication, and independent validation;
- catalog-first planning: the reader decides every read from resident index arrays before touching point payloads.

Changes:

- **Sampler:** the level-, tile-, and cell-dependent priorities and per-tile capacities are replaced by one global priority per transcript.
- **Build plan and writers:** Exact, Bridge, and Spatial writers are replaced by one band writer per physical order; levels become bands.
- **Catalog:** per-level manifests and value-tile records become per-band records.
- **Reader:** level selection is replaced by computing `t`, the bands below it, and prefix lengths for the partial band.
- **Viewer:** discrete LOD choice and hysteresis are replaced by a continuous threshold; refinement is additive, so resident data stays valid when zooming in and level switches no longer replace the whole render batch.
- **Renderer:** optionally an opacity ramp near the threshold.

## 7. Open questions

1. **Band tile width `P`** (partly resolved): bands now follow from the density target (section 2), and section 4 gives a starting value from a cost model (`P` ≈ 128–160 for local reads). Benchmark 64, 128, and 256 pixels with realistic canvas sizes and budgets, in both the density-bound and the hard-limit regime.
2. **Reference density and build-time `ρ`:** whether `D_ref` is the mean, the median, or a high percentile of tissue tile density; and whether the build should use the default `ρ` or the lower effective density that the hard limit produces on typical canvases.
3. **Coarse-band over-read:** residency only, smaller coarse tiles, or spatial sub-blocks.
4. **Prefix lengths:** uniform-fraction prefixes or a per-tile table of counts at sub-thresholds.
5. **Random versus stratified priority:** uniform random priorities preserve density but lose the even spread of the current microgrid at very low rates; a stratified priority could restore it if the overview looks worse.
6. **Budget accounting:** `N_view` at full-resolution tile granularity still counts points just outside the view; whether finer counts are worth their index size. This also determines whether full detail is reachable inside a single tile denser than the budget.
7. **Gene-major partial bands:** for the few values above the budget, partial bands mean per-tile prefixes scattered within the value's interval; measure the cost.
8. **Index size** of per-(value, band, tile) records compared with the current `value_tiles` catalog.
9. **Remote access:** reads per view (tens to hundreds of runs instead of one to ten) and shard layout over HTTP.
10. **Scale:** band count and storage at billions of transcripts (Atera-scale data).
11. **Compatibility:** this is a new cache schema; the current cache stays the performance reference and is not migrated.

## 8. Suggested next steps

1. Build a prototype converter that reads the published cache's Exact tile-major arrays (`location`, `point_id`, `value_id`), computes priorities from `point_id`, and writes both band orders to a scratch Zarr store. No source re-read is needed.
2. Benchmark real reads against the current cache at the same viewports and budgets: cold and warm latency, decoded rows and bytes, read operations, and peak memory, for all-values and selected-value views.
3. Compare images at matched viewports, budgets, point size, and opacity: tile seams should disappear, real density differences must remain, and zooming should change density smoothly.
4. Choose `P`, `D_ref`, coarse-band residency, and prefix-length policy from those results.
5. Only then plan schema, writer, reader, and viewer changes as a new format version.

## Appendix A. End-to-end walkthrough

This appendix follows one dataset, the Xenium section above, from the canonical Parquet source to the points drawn on screen.

### A.1 What `t = ρ / (s² · D)` says

Picture one screen pixel lying on the tissue:

- `s` is the zoom in µm per pixel. At `s = 7` µm/px, one pixel covers 7 × 7 = 49 µm² of tissue; in general a pixel covers `s²` µm².
- `D` is the tissue's transcript density in transcripts per µm²; here it averages 1.58.
- One pixel therefore covers `s² · D` transcripts: at `s = 7`, about 49 × 1.58 ≈ 77.
- `ρ` is the number of points wanted per pixel; spatiato's default is 4.
- To draw 4 points out of the 77 under each pixel, the viewer keeps 4 / 77 ≈ 5% of the transcripts. That fraction is `t`:

```text
t = (points wanted per pixel) / (transcripts under one pixel) = ρ / (s² · D)
```

This is the same as the count form the viewer uses:

```text
budget = ρ × number of pixels
N_view = D × area in view = D × number of pixels × s²
t      = budget / N_view = ρ / (s² · D)
```

The count form runs at runtime, with the real number of transcripts in view and the hard limit. The density form is used to reason about zoom when designing the bands.

### A.2 From the formula to the bands

Zooming in by a factor of two halves `s`, so a pixel covers four times less tissue and `t` rises fourfold. At this section's average density:

| Zoom (µm per pixel) | Transcripts under one pixel | `t` |
|---:|---:|---:|
| 25.4 | 1,024 | 0.39% |
| 12.7 | 256 | 1.56% |
| 6.4 | 64 | 6.25% |
| 3.2 | 16 | 25% |
| 1.6 | 4 | 100% (everything shown) |

The bands are the cuts between these rows on the priority axis:

```text
priority  0% |b0|   b1   |      b2      |         b3          |              b4               | 100%
              0  0.39%  1.56%          6.25%                  25%                             100%
                 (not to scale)
t moves right as you zoom in: each doubling of zoom moves t one band to the right.
```

Band 4 (25–100%) holds the points that appear while zooming from 3.2 to 1.6 µm/px, band 3 those that appear from 6.4 to 3.2 µm/px, and so on. A band is one doubling of zoom's worth of additional detail. The cuts sit at factors of four because the bands follow the zoom, not a budget.

Band tile size follows the same logic. Each band is used around its own zoom, so its tiles are made about 128 screen pixels wide at that zoom. Coarse bands are used when each pixel covers more tissue, so their tiles are larger:

| Band | Priority range | Used around | Tile edge |
|---|---|---:|---:|
| 0 | 0–0.39% | 25 µm/px and coarser | 3.48 mm |
| 1 | 0.39–1.56% | 25 → 12.7 µm/px | 1.74 mm |
| 2 | 1.56–6.25% | 12.7 → 6.4 µm/px | 0.87 mm |
| 3 | 6.25–25% | 6.4 → 3.2 µm/px | 0.44 mm |
| 4 | 25–100% | 3.2 → 1.6 µm/px | 0.22 mm |

### A.3 Build: from Parquet to the band layout

1. **Read the canonical points Parquet** (x, y, gene, and any other columns) and give each transcript its stable `point_id`, as now: its row number in the source.
2. **Priority.** `p = hash(point_id)`: a number between 0 and 1 that looks random but is always the same for the same transcript.
3. **Reference density.** Count transcripts per full-resolution tile and take the mean over tissue tiles: `D_ref` = 1.58 per µm².
4. **Band plan.** The finest zoom is `s = sqrt(ρ / D_ref)` = 1.59 µm/px, where `t = 100%`. Band cuts are 100%, 25%, 6.25%, and so on, dividing by four until the coarsest band would hold at most 1M rows: five bands here. Each band's tile edge is 128 pixels times its zoom, rounded to 2, 4, 8, 16, or 32 full-resolution tiles.
5. **Place every transcript.** Its band is the range containing `p`; its band tile is its xy divided by that band's tile edge. For example, a transcript with `p = 3%` falls in band 2 (1.56–6.25%) and is stored in the 0.87 mm band-2 tile that contains it.
6. **Write the tile-major copy.** Sort by `(band, band tile, p)` and write xy (relative to the band tile's origin) and the value ID. The index stores where each band tile's run starts and how many rows it has.
7. **Write the gene-major copy.** Sort by `(value, band, band tile, p)` and write xy. The index stores the same, per value.
8. **Counts for runtime.** Transcripts per full-resolution tile, and per value per full-resolution tile (what the current `value_tiles` already holds), so the viewer can compute `N_view`.

Every transcript is written once per copy and nothing is discarded. Sampling happens later, purely through how far into each run the reader reads.

### A.4 Runtime: from a viewport to the points drawn

1. **Budget.** `min(hard limit 2M, ρ × canvas pixels)`, as in [_screen_density_budget](/Users/arne.defauw/VIB/napari_harpy/src/spatiato/viewer/tiled_points/napari/viewport.py:130). On a 1600 × 1000 canvas: min(2M, 6.4M) = 2M.
2. **`N_view`.** From the counts: the selected transcripts in the full-resolution tiles intersecting the view.
3. **`t = min(1, budget / N_view)`.**
4. **Which bands.**
   - Bands entirely below `t`: read the complete run of each of their tiles intersecting the view.
   - The band containing `t`: read a prefix of each tile intersecting the view, with fraction `f = (t - lower) / (upper - lower)`.
   - Bands above `t`: read nothing.
5. **Read** those runs, from the tile-major copy for all values or the gene-major copy for a selection, then draw, culling points outside the view.

Two examples:

- **Whole section.** `N_view` = 136.6M, so `t` = 2M / 136.6M = 1.46%. Read band 0 completely, plus the first (1.46 − 0.39) / (1.56 − 0.39) = 92% of every band-1 tile: about 2M points.
- **1 mm view in the densest area.** `N_view` = 8.66M, so `t` = 23%. Read bands 0–2 completely for the tiles intersecting the view, plus the first (23 − 6.25) / (25 − 6.25) = 90% of each band-3 tile there.

### A.5 How the two forms of `t` fit together

- **At build time,** the density form decides where to cut the priority axis and how large each band's tiles are. It uses one reference density and `ρ`, so the bands line up with zoom levels.
- **At runtime,** the count form gives the exact `t`, using the real transcripts in view and the hard limit. If the hard limit lowers `t`, or the region is denser or sparser than `D_ref`, `t` falls in a different band than the table in A.2 predicts. The image stays correct; only read efficiency changes slightly.

## References

- [sampler_17_9_26.md](/Users/arne.defauw/VIB/napari_harpy/Roadmap/transcripts_visualization/sampler_17_9_26.md): tile-boundary sampling artefacts and the recommendation for common sampling fractions.
- [CACHE_FORMAT.md](/Users/arne.defauw/VIB/napari_harpy/src/spatiato/core/multi_scale_cache_points_zarr/CACHE_FORMAT.md): current cache contract.
- van Oosterom et al., [Organizing and visualizing point clouds with continuous levels of detail](https://www.gdmc.nl/publications/2022/cLoD_ISPRS_Journal.pdf) (2022): continuous detail values per point, used as an organizing dimension of storage.
- Schütz et al., [Real-Time Continuous Level of Detail Rendering of Point Clouds](https://www.cg.tuwien.ac.at/research/publications/2019/schuetz-2019-CLOD/) (2019): point-wise instead of chunk-wise density transitions during rendering.
- Geospatial point-cloud formats such as COPC, Potree, and Entwine store each point once in one node of a hierarchy; 3D Tiles calls this additive refinement.
