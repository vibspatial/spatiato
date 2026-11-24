# Auto-Load Table Coloring For Labels And Shapes

## Motivation

The Viewer widget can color labels and shapes elements from a linked AnnData table (`obs` columns and vars of `X` or a layer), and shapes elements also from their own GeoDataFrame columns. Today this works like a form: the user picks a linked table, a color source kind and a value, then clicks **Add / Update in viewer**. The card has no overview of which colored layers are already loaded for the element.

Image overlay channels already work the way we want here. Picking a channel in the search field loads it immediately, and the card shows a list of the channels currently loaded in the viewer, each with visibility, color and remove controls.

This feature gives labels and shapes cards the same behavior:

- picking a color source loads (or reuses) its styled layer immediately, without an **Add / Update** click;
- the card lists every styled layer of that element that is live in the viewer, with visibility and remove controls;
- the list is rebuilt from live viewer state, so it stays correct when layers are removed or reordered in napari.

## Goals

- Load a styled labels layer as soon as the user picks an `obs` column or var from the completer.
- Load a styled shapes layer as soon as the user picks an `obs` column, var or shapes column from the completer.
- Show a list of loaded styled layers per element card, in napari layer order, with eye toggle, label and remove button, plus a "Remove all" action.
- Keep the primary labels and primary shapes layers loadable through an explicit action, separate from the styled layers.
- Keep loaded styled layers in sync with the table when the table changes, replacing the current "click Add / Update again to restyle" behavior.
- Keep loaded styled shapes layers in sync with their shapes element when the Annotation widget saves or reloads it.
- Classify a column's values with one rule, for discovery and for coloring, whether labels or shapes are colored from a table or shapes from their own columns.
- Reconcile the card from live viewer membership, never from card-local bookkeeping, as the image card already does.

## Non-Goals

- Asynchronous or background loading of styled layers. Loading stays synchronous, as it is today for both images and styled layers.
- Changes to how values are mapped to colors (palettes, continuous ranges, colorbars). See `Roadmap/colormap/colormap.md`.
- Points elements.
- Remembering the loaded styled layers across sessions.
- Editing the color of a styled layer from its row. The image row's color button does not apply to label/shape colormaps.

## Current Code Shape

Relevant files:

- `src/spatiato/widgets/viewer/image_widget.py`: the image card, the reference behavior.
- `src/spatiato/widgets/viewer/labels_widget.py`: the labels card.
- `src/spatiato/widgets/viewer/shapes_widget.py`: the shapes card.
- `src/spatiato/widgets/viewer/_table_color_sources.py`: the color sources a linked table offers.
- `src/spatiato/widgets/viewer/widget.py`: `ViewerWidget`, which validates intent and calls the adapter.
- `src/spatiato/widgets/image_layer_row.py`: the row widget for one loaded image layer.
- `src/spatiato/viewer/adapter.py`: `ViewerAdapter` and the layer-binding registry.
- `src/spatiato/core/_color_source.py`: `TableColorSourceSpec` and `ShapeColumnColorSourceSpec`.
- `src/spatiato/core/color_source_options.py`: discovery of the color sources, and its value classifier.
- `src/spatiato/viewer/labels_styling.py` and `src/spatiato/viewer/shapes_styling.py`: coloring of labels and shapes layers.
- `src/spatiato/_app_state.py`: `TableStateChangedEvent`.

### Image overlay flow (reference)

1. Accepting a channel in the completer emits `overlay_channel_add_requested` from `_ImageCardWidget._request_channel_from_text(...)`.
2. `ViewerWidget._add_image_overlay_channel(...)` validates the context and calls `ViewerAdapter.ensure_image_overlay_channel_loaded(...)`.
3. Registering the binding emits `ViewerAdapter.image_layers_changed`, a payload-free invalidation signal. It is also emitted on removal from napari and on reorder.
4. `ViewerWidget._on_image_layers_changed()` re-queries live bindings through `_get_live_image_bindings(...)` and calls `_ImageCardWidget.set_loaded_image_bindings(...)`, which reconciles the rows.
5. Row intents (visibility, color, remove) carry the row's construction-time binding. The card checks that the row still owns that binding before forwarding the intent to `ViewerWidget`.

### Labels and shapes flow (today)

- The cards emit one `add_update_requested(LabelsLoadRequest | ShapesLoadRequest)` from the **Add / Update in viewer** button.
- "No color source" plus the button loads the primary layer (`ensure_labels_loaded` / `ensure_shapes_loaded`).
- A selected source plus the button calls `ensure_styled_labels_loaded(...)` / `ensure_styled_shapes_loaded(...)`. These create a styled layer or restyle the existing one from the current table state. The **Update** half of the button exists to make that restyle happen after the table changed.
- The shapes card's **Fill** checkbox is passed with the request.

### What already supports the new behavior

- Styled layers are already keyed per source: `LabelsLayerBinding` and `ShapesLayerBinding` carry `role="styled"` and a `style_spec`. Several styled layers per element can coexist, one per source, just as one overlay layer exists per channel.
- `get_loaded_styled_labels_layers(...)` and `get_loaded_styled_shapes_layers(...)` already list them in viewer order.
- `_ImageLayerRow` uses only `binding.layer` and `layer.events.visible` when `show_colormap=False`, so it is already a generic "eye / label / ×" row apart from its type annotation. It is also used by the Histogram widget.

### What is missing

- **Membership signals.** The adapter emits `image_layers_changed` for images and `primary_labels_layers_changed` for primary labels. Nothing fires when a styled labels or shapes layer is added, removed or reordered, and nothing fires when a primary shapes layer is removed (`primary_shapes_layer_registered` covers registration only).
- **Removing one styled layer.** `remove_labels_layer(...)` and `remove_shapes_layer(...)` only remove the primary layer.
- **A loaded-layer list in the labels and shapes cards**, and the matching `ViewerWidget` handlers.
- **Following shapes geometry writes.** A styled shapes layer holds the geometry and the rendered-row mapping from when it was built. When the Annotation widget saves or reloads the shapes element (`shapes_element_written`, `shapes_element_reloaded`), the Viewer only rebuilds its shapes cards (`_refresh_shapes_section_for_shared_event(...)`), and styled shapes layers keep the old geometry. Clicking **Add / Update** again does not help: it restyles the existing layer with its old row mapping.
- **One rule for classifying values.** Discovery and the three coloring paths each decide on their own whether a column is categorical or continuous, and they already disagree for some columns (see "Shared value classification").

## Proposed Design

### Card layout

```
▾ blobs_labels
  Primary layer   [ Load in viewer ]          ← or, once loaded:  👁  blobs_labels   ×
  Linked table    [ table              ▾ ]
  Color source    [ Observations       ▾ ]
  Observation     [ Search and add …     ]
  3 colored layers                     Remove all
    👁  obs: leiden                      ×
    👁  vars (X): CD3                    ×
    👁  vars (counts): CD3               ×
```

- The **Linked table** and **Color source** combos only filter the completer. They no longer describe a pending request.
- The "No color source" entry is removed from the **Color source** combo. The primary layer has its own row (see below). The combo defaults to its first entry.
- Accepting a value in the completer (completer activation or Enter on an exact match) sends one add intent. On success the input is cleared, as `finish_overlay_channel_add(...)` does for images.
- Sources that are already loaded are left out of the completer, as loaded channels are for images.
- The loaded list shows every styled layer of the element, across all linked tables and source kinds, in napari layer order. The row label is `style_spec.source_label` (`obs: leiden`, `vars (X): CD3`, `column: area`). When the element has more than one linked table, table-backed rows are prefixed with the table name.
- The list reuses the image card's presentation: count label, "Remove all", scroll area capped at 10 visible rows, and an empty state ("No colored layers in viewer").
- The **Add / Update in viewer** button and the action hint label are removed.

### Primary layer row

The primary labels layer is the shared surface that Object Classification and Spatial Query style; the primary shapes layer is the editable annotation surface. Neither should load as a side effect of browsing color sources.

Mirror the image card's Stack pattern:

- no primary layer loaded: show a **Load in viewer** button;
- primary layer loaded: show a row with eye toggle, the layer name and a remove button.

The row is the first form row of the card, labelled "Primary layer", above "Linked table". It is kept out of the loaded list of styled layers (see "Decisions"), and the card reconciles it from a separate `primary_binding`.

Loading keeps today's semantics: `ensure_labels_loaded(...)` / `ensure_shapes_loaded(...)` return an existing primary layer unchanged.

### Shared value classification (`src/spatiato/core/color_value_classification.py`)

Phase 1 implemented this section. Before it, four places decided how a column's values were colored, each with its own rules:

| Path | Function | Rule |
|---|---|---|
| Discovery, for `obs` columns and shapes columns | `_classify_scalar_column_color_source(...)` in `src/spatiato/core/color_source_options.py` | the dtype, then the values of an `object` column |
| Labels colored from a table | `_build_obs_column_colormap(...)` in `src/spatiato/viewer/labels_styling.py` | the dtype only, for booleans and 0/1 integers |
| Shapes colored from a table | `_build_table_obs_shape_style(...)` in `src/spatiato/viewer/shapes_styling.py` | the dtype or the values, through `_is_bool_series(...)` and `_is_exact_binary_integer_series(...)` |
| Shapes colored from a shapes column | `apply_shape_column_color_source_to_shapes_layer(...)`, then `_build_categorical_shape_style(...)` | `value_kind` from the spec, then its own dtype checks, with a fallback that treats any other values as categories |

They disagreed. A boolean column with missing values, or a 0/1 column with missing values, has `object` dtype in pandas. Discovery and both shapes paths classified it as categorical, but labels coloring fell through to continuous. The same `obs` column was therefore colored categorical on a shapes element and continuous on a labels element. The shapes-column path also colored with the kind stored on the spec, which was out of date when the column changed after discovery.

One classifier replaced the four. Coloring needs more than categorical or continuous: it needs to know which kind of categorical column it has, because that decides where the categories come from and whether a stored palette may be used. So the classifier returns an encoding:

```python
ColorValueEncoding = Literal["pandas_categorical", "boolean", "binary_integer", "string", "continuous"]

@dataclass(frozen=True)
class ColorValueClassification:
    encoding: ColorValueEncoding

    @property
    def value_kind(self) -> Literal["categorical", "continuous"]: ...

def classify_color_values(values: pd.Series) -> ColorValueClassification | None: ...
```

- **Rules.** Its rules are the discovery rules from before phase 1, which already looked at the values of `object` columns. It returns `None` where discovery does: for a column that is not pandas categorical and has no non-missing values, and for a column with mixed value types. A pandas categorical column is `"pandas_categorical"` even when all its values are missing, because discovery checks for it before it checks for missing values.
- **Input: always the full column.** Callers pass the whole column, never only the rows being drawn: `table.obs[column]` for discovery and for both table coloring paths, and `shapes_element[column]` for discovery and for the shapes-column path. The paths already did this before phase 1; labels coloring, for example, classified `full_series`, not `region_series`. A subset can classify differently: the rows of one region may hold only the 0s of a 0/1 column, and only 0s classify as continuous.
- **Discovery** uses the result to decide whether a column is offered: `None` means it is not. After phase 2a it no longer stores the kind on the spec (see "Matching a loaded source").
- **The three coloring paths** branch on `.encoding` instead of their own checks:
  - `"string"` keeps the default palette and `coercion_applied=True`;
  - `"pandas_categorical"`, `"boolean"` and `"binary_integer"` may use a stored palette.

  When the result is `None`, the coloring path raises `ValueError`, which the Viewer turns into the "source no longer available" row warning.
- **The shapes-column path no longer reads `value_kind` from the spec.** No coloring path reads a spec's `value_kind` to decide how to color, and coloring always reads the data again. Phase 2a removes the field from the specs (see "Matching a loaded source").
- **The classifier lives in `core`.** `core` must not import from `viewer`, so the classifier has its own value checks, and `is_string_like_series(...)` in `src/spatiato/viewer/_styling.py` was deleted. The viewer modules import the classifier from `core`. `_is_bool_series(...)`, `_is_exact_binary_integer_series(...)`, the separate labels dtype checks and the discovery helpers in `src/spatiato/core/color_source_options.py` were deleted.

What stays outside the classifier, because it is not about the values:

- instance coloring: the role of the table's instance-key column, decided by its name before classification;
- vars: always colored as continuous;
- palette lookup: the table's stored palette for `obs` columns, a companion `_colors` column for shapes columns;
- building colors: labels build a colormap and shapes build per-row RGBA colors; each keeps its own builder.

**Behavior changes.** Tests pinned down the earlier results per column type before the change, so that these two are the only differences:

- Labels coloring of `object`-dtype boolean and 0/1 columns moved from continuous to categorical, matching discovery and shapes.
- A column that discovery would not offer now raises `ValueError` when it is colored, instead of being colored anyway. Before, labels coloring fell through to continuous, which typically drew every object as missing, and the shapes-column path colored by the column's unique values through a fallback in `_build_categorical_shape_style(...)`. That fallback was removed. This happens when a layer loaded earlier is colored again after its column changed. The Viewer already catches the `ValueError` and shows an error card, and in the new flow the row warning.

Spatial Query also colors the primary labels layer through `apply_table_color_source_to_labels_layer(...)`, but it only accepts pandas categorical columns, which classify the same before and after.

### Discovery cost

Discovery classifies every `obs` column of each linked table, and every column of a shapes element, to decide which columns to offer. It keeps doing so (see "Decisions"):

- in the new card, accepting a completer value loads it immediately, so a column that cannot color would only produce an error card: a datetime column, a column of dicts or lists, an all-missing column, an ID column of mixed types;
- skipping the classification saves nothing for a column that is picked, because coloring classifies the column again;
- a filter on the dtype alone was considered and rejected: `object` columns of dicts or mixed types would still be offered and fail when picked.

Discovery runs when the cards are built and after table changes, except for Object Classification's `user_class` updates, which `_on_table_state_changed(...)` skips. It does not run while the user types or selects. Two things make it slower than it needs to be:

- **The classifier loops in Python.** Measured on columns of 1M rows: categorical, boolean and float columns take 0–2 ms; integer columns 23–37 ms, for a Python `set` of every value; `object` columns 110–230 ms, for `.tolist()` and an `all(isinstance(...))` pass per type. Eight such columns together took about 600 ms.
- **Each linked table is classified once per card.** `_rebuild_labels_cards(...)`, `_rebuild_shapes_cards(...)`, `_refresh_labels_card_linked_tables(...)` and `_refresh_shapes_card_linked_tables(...)` call `_get_linked_tables(...)` for every card, and it builds `LinkedTableColorSources.from_table(...)` again each time. A table that annotates ten labels elements has its `obs` columns classified ten times per refresh, and once more for each linked shapes card.

The changes:

- **A vectorized classifier with the same rules.** `classify_color_values(...)` keeps its rules, its order and its results:
  - an integer column reads its unique values with `pd.unique(...)` instead of a Python `set` (2–4 ms on 1M rows);
  - a column of any other dtype, such as `object`, is first classified with `pd.api.types.infer_dtype(values, skipna=True)` (5–10 ms on 1M rows). Answers that match one rule exactly decide directly: `"boolean"` is boolean; `"integer"` goes to the integer rule; `"floating"` and `"mixed-integer-float"` are continuous; `"string"` and `"bytes"` are string. Answers for values that no rule accepts are `None`: date and time types (`"datetime64"`, `"datetime"`, `"date"`, `"timedelta64"`, `"timedelta"`, `"time"`, `"period"`, `"interval"`), `"complex"` and `"decimal"`. Every other answer, such as `"mixed"`, `"mixed-integer"` or `"unknown-array"`, falls back to the current per-value check, so rare columns keep their current result: strings mixed with bytes (`"mixed"`, string), `Fraction` values (`"mixed"`, continuous), integers mixed with `Fraction` values (`"mixed-integer"`, continuous). Each of these answers was checked against the current classifier on sample columns.
- **Each linked table is classified once per pass.** A pass is one rebuild or refresh of the cards. It builds `LinkedTableColorSources` once per table, in a dictionary keyed by table name, and hands every card linked to that table the same object. `LinkedTableColorSources` is a frozen dataclass of tuples, so cards can share it.
  - One dictionary covers the labels cards and the shapes cards of the same pass, so a table that annotates both labels and shapes elements is classified once per event. The full rebuild calls `_rebuild_labels_cards(...)` and `_rebuild_shapes_cards(...)` in one pass, and a table change calls `_refresh_labels_card_linked_tables(...)` and `_refresh_shapes_card_linked_tables(...)` in one pass. The shapes-only rebuild after a shapes save or reload is a pass of its own.
  - The dictionary lives only for its pass; the next pass classifies again from the current data. It is not a cache kept between passes: that would need invalidation whenever a table changes, and the events do not cover every change. `_on_table_state_changed(...)` skips Object Classification's `user_class` updates on purpose, and edits made outside Spatiato, for example from the console, send no event. Within one pass the tables cannot change, so sharing needs no invalidation.
  - Shapes columns are not affected: each shapes card classifies the columns of its own shapes element, once per card build.

Together this brings discovery of a large table with several `object` columns from hundreds of milliseconds per card to tens of milliseconds per pass. Timing is checked by hand on a large table, not in the test suite.

### Adapter changes (`src/spatiato/viewer/adapter.py`)

1. **Membership signals.** Add two payload-free invalidation signals:

   ```python
   # Emitted when the set/order of live labels layers (primary or styled) changes.
   labels_layers_changed = Signal()
   # Emitted when the set/order of live shapes layers (primary or styled) changes.
   shapes_layers_changed = Signal()
   ```

   Emit them at the same five points as `image_layers_changed`:

   - `_handle_registered_binding(...)`, after a binding is registered for a layer already in the viewer;
   - `_on_viewer_layer_inserted(...)`, for a layer registered before it is added to the viewer;
   - `_on_viewer_layer_removed(...)`, decided before the binding is unregistered, as `had_usable_image_semantics` is today;
   - `_on_viewer_layers_reordered(...)`, when any labels or shapes binding is live;
   - `_remove_layer_from_viewer_and_registry(...)`, in its fallback for viewers that do not emit napari's removal event.

   Every `LabelsLayerBinding` counts for `labels_layers_changed`, and every `ShapesLayerBinding` counts for `shapes_layers_changed`, whatever their role. Unlike `_is_usable_image_binding(...)`, no SpatialData object or coordinate system is required: consumers filter on both when they re-query, so an extra emission costs nothing. The existing `_is_labels_binding(...)` and `_is_shapes_binding(...)` helpers express this.

   `sync_primary_shapes_layer_binding(...)` also emits `shapes_layers_changed`. It replaces the binding object of a live primary shapes layer, and the Annotation widget calls it after writes. Card rows hold binding objects and are kept only while the binding object is unchanged, so they must be told to re-read. It still does not emit `primary_shapes_layer_registered`: that is the registration event the Annotation widget reacts to, and avoiding it is why the method exists. `shapes_layers_changed` only asks consumers to re-query.

   Like `image_layers_changed`, the new signals carry no payload; consumers re-query the registry. The existing `primary_labels_layers_changed` and `primary_shapes_layer_registered` stay as they are for their current consumers.

2. **Remove one styled layer.**

   ```python
   def remove_styled_labels_layer(
       self, sdata, labels_name, coordinate_system, *, style_spec
   ) -> Labels | None: ...

   def remove_styled_shapes_layer(
       self, sdata, shapes_name, coordinate_system, *, style_spec
   ) -> Shapes | Points | None: ...
   ```

   They follow `remove_image_overlay_channel(...)`: return `None` when nothing matches, raise `ValueError` when more than one live layer matches.

   Detecting more than one match needs every live layer for a spec. Today `get_loaded_styled_labels_layer(...)` / `get_loaded_styled_shapes_layer(...)` return only the first match, and `get_loaded_styled_labels_layers(...)` / `get_loaded_styled_shapes_layers(...)` return every styled layer of the element, whatever its spec. The `get_loaded_styled_*_layers(...)` functions get an optional `style_spec` filter, passed on to `_matches_labels_binding(...)` / `_matches_shapes_binding(...)`, and the remove functions use it.

3. **Specs without `value_kind`.** See "Matching a loaded source" below. Remove `value_kind` from `TableColorSourceSpec` and `ShapeColumnColorSourceSpec`, from the specs that discovery, the var sources and Spatial Query build, and from the `"instance"` checks in the coloring code.

   Each of the three `"instance"` checks becomes one name test: a spec colors instances exactly when it is an `"obs_column"` source for the table's instance-key column. Each check today also has an inner guard that raises "Instance ID coloring must use the table instance key …" for a spec that claims `"instance"` for another column. Once the outer test is only the name test, that guard can never be reached, so it is deleted. The checks are in `apply_table_color_source_to_labels_layer(...)`, `apply_table_color_source_to_shapes_layer(...)` and `_get_table_color_values_by_instance(...)`.

   The lookups by spec need no change: `_matches_labels_binding(...)`, `_matches_shapes_binding(...)` and `LayerBindingRegistry.find_bindings(style_spec=...)` keep comparing whole specs, which now compares exactly the fields that name the data.

4. **Fill on the shapes binding.** Add `fill: bool = False` to `ShapesLayerBinding`, so the Fill state of a styled shapes layer is read from live state rather than kept by the card (see "Shapes Fill"):

   - `ShapesLayerBinding.__post_init__(...)` rejects `fill=True` unless the binding is styled and rendered as `Shapes` (`shapes_rendering_mode == "shapes"`);
   - `LayerBindingRegistry.register_shapes_layer(...)` and `ViewerAdapter.register_shapes_layer(...)` accept `fill`;
   - `ensure_styled_shapes_loaded(..., fill=...)` records `fill` on the binding of a layer rendered as `Shapes`. For a point-radius layer rendered as `Points` it records `False`, since fill has no effect there. Today's card passes its Fill checkbox for point-radius shapes too, so normalizing instead of raising keeps today's flow working;
   - when an existing styled layer is restyled with a different `fill`, its binding is replaced. Build the new binding with `dataclasses.replace(binding, fill=fill)`, which keeps every other field, including `source_row_id_by_rendered_row`, `source_shapes_index_feature_name` and `skipped_geometry_count`, which registering again through `register_shapes_layer(...)` would otherwise need passed again. Store it with a new `LayerBindingRegistry.replace_binding(binding)`, which replaces the binding of `binding.layer` and raises `ValueError` when that layer has no binding yet. Today the registry can only store a binding it builds itself from keyword arguments, and `_register_binding(...)`, which stores a ready-made one, is private. The adapter then calls `_handle_registered_binding(...)`, so the replacement emits `shapes_layers_changed`. The binding is replaced only after the restyle succeeds, so it always records the fill the layer shows. When `fill` is unchanged, the binding is left as it is, so a plain restyle emits no membership signal and the card keeps its rows.

5. **A failed first load leaves no layer.** Today `ensure_styled_labels_loaded(...)` and `ensure_styled_shapes_loaded(...)` add a new layer to the viewer and register it before they color it. Coloring can raise `ValueError`: when the table, column or var is gone, and since phase 1 when a column can no longer color a layer. The new layer then stays in the viewer, registered as a styled layer but never colored, and the Viewer only shows an error card. In the new flow, registering would give that layer a row in the loaded list, for a source that never loaded.

   The rule:

   - a layer the call creates is colored before it is added to the viewer and registered: build, color, add, register. A coloring failure then leaves nothing behind. Coloring uses only the layer and the SpatialData object, never the viewer or the binding registry:
     - labels coloring reads the table and sets the layer's colormap and features;
     - shapes coloring also needs `source_row_id_by_rendered_row` and `source_shapes_index_feature_name`. Today it reads them from the binding, which does not exist before registration. They are the same values that `_build_shapes_layer(...)` returns in `_BuiltShapesLayer` and that registration copies into the binding, so a newly built layer takes them from `_BuiltShapesLayer`. A restyle of an existing layer still reads them from its binding.

     This relies on napari keeping a layer's colormap, colors and features when the layer is added to the viewer; an adapter test checks it (see "Tests");
   - when registering fails, the new layer is removed with `_remove_layer_after_failed_registration(...)` and the error is raised again. The shapes path already does this, because the Annotation widget would otherwise adopt the unregistered `Shapes` layer as a new annotation layer. The labels path gets it too: nothing reacts to an unregistered `Labels` layer today, but the two paths stay the same;
   - neither failure emits `labels_layers_changed` or `shapes_layers_changed`: the layer is never registered, and removing an unregistered layer emits nothing. The card never shows a row for it;
   - an existing layer whose restyle fails stays as it is, registered and with its last colors, and its row gets the warning (see "Refresh after table changes" and "Decisions"). Only a layer the call created is removed.

   `rebuild_styled_shapes_layer(...)` in phase 8 follows the same order (see "Shapes geometry writes").

### Shared layer row

Split today's `_ImageLayerRow` into a neutral base class and an image subclass (see "Decisions").

New module `src/spatiato/widgets/layer_row.py` with `_LayerRow`:

- accepts any `BaseLayerBinding`;
- has the eye toggle, label and remove button, syncs visibility from `layer.events.visible`, and has an idempotent `dispose()`;
- emits identity-free intents (`remove_requested`, `visibility_change_requested`) and never mutates the layer. Its docstring states this contract generally; it is the contract `_ImageLayerRow` documents today;
- has a warning state, used for "source no longer available";
- lets an owner or subclass add extra controls between the label and the remove button, used for **Fill** on shapes rows.

`src/spatiato/widgets/image_layer_row.py` keeps `_ImageLayerRow(_LayerRow)`. It adds the color button, the `layer.events.colormap` subscription and the colormap sync. The image-specific helpers stay in that module: `_binding_channel_index`, `_binding_channel_name`, `_colormap_presentation_from_layer`, `_solid_color_from_layer` and `_normalized_color_or_none`. The imports in `widgets/viewer/widget.py`, `widgets/viewer/image_widget.py` and `widgets/histogram/widget.py` do not change.

Users of each row class:

- image Stack and Overlay rows, and Histogram rows: `_ImageLayerRow`, unchanged;
- primary labels and primary shapes rows, and styled labels rows: `_LayerRow`;
- styled shapes rows: `_LayerRow`, with a **Fill** checkbox when the layer is rendered as `Shapes` (see "Shapes Fill" below), as an extra control or a small subclass.

The color button must not move into `_LayerRow`. In the installed napari, `Shapes` and `Points` layers have no `colormap` event, so subscribing to it fails for shapes rows, and point-radius shapes are rendered as `Points`. `Labels` layers do have the event, but the colormap helpers detect solid colors in image colormaps and give no meaningful preview for a labels colormap.

### Labels card (`src/spatiato/widgets/viewer/labels_widget.py`)

Replace `add_update_requested` with focused intents, following `_ImageCardWidget`:

```python
primary_load_requested = Signal(str)                # labels_name
primary_remove_requested = Signal(str)              # labels_name
primary_visibility_requested = Signal(str, bool)    # labels_name, visible
styled_source_add_requested = Signal(str, object)   # labels_name, TableColorSourceSpec
styled_source_remove_requested = Signal(str, object)
styled_sources_remove_all_requested = Signal(str)
styled_source_visibility_requested = Signal(str, object, bool)
```

and a reconcile method:

```python
def set_loaded_labels_bindings(
    self,
    *,
    primary_binding: LabelsLayerBinding | None,
    styled_bindings: Sequence[LabelsLayerBinding],
) -> None: ...
```

It follows `set_loaded_image_bindings(...)`: validate the whole snapshot first, keep rows whose binding is unchanged, build rows for new bindings, dispose rows for absent ones, never mutate napari layers. Row intents use `partial(..., expected_binding)` and are checked against the current row before they are forwarded.

The loaded list (count label, "Remove all", scroll area capped at 10 rows, empty state, and the reconcile of styled rows) is built as a widget of its own, so the shapes card reuses it. The two cards already duplicate their color-source control logic; the loaded list should not be duplicated as well.

The loaded list also holds the "source no longer available" warnings, as a mapping from spec to message, and applies them every time it reconciles its styled rows. They are not stored on the rows: a row is rebuilt whenever its binding object changes, for example on a Fill replacement or a geometry rebuild, and a flag on the row would be lost. `ViewerWidget` sets and clears a warning through the card:

```python
def set_styled_source_warning(
    self, style_spec: TableColorSourceSpec | ShapeColumnColorSourceSpec, message: str | None
) -> None: ...
```

A warning is cleared after a successful restyle of its source, and dropped when its layer is no longer live at reconcile time or when the card is rebuilt, for example on a coordinate-system change.

The card no longer emits `LabelsLoadRequest`, and the type is deleted. The status-card builders in `src/spatiato/widgets/viewer/status_card.py` read only the labels name and `selected_color_source` from it today, so they take those directly:

```python
def build_primary_labels_loaded_card_spec(labels_name: str, result: LabelsLoadResult) -> _ViewerStatusCardSpec: ...
def build_styled_labels_card_spec(
    labels_name: str, style_spec: TableColorSourceSpec, result: LabelsLoadResult
) -> _ViewerStatusCardSpec: ...
```

### Shapes card (`src/spatiato/widgets/viewer/shapes_widget.py`)

Same structure as the labels card, and it reuses the labels card's loaded-list widget. The differences are:

- sources can also be `ShapeColumnColorSourceSpec`;
- a styled row has a **Fill** checkbox only when its layer is rendered as `Shapes` (see "Shapes Fill").

**Shapes-column sources are classified from the current column when they are colored**, through the shared classifier (see "Shared value classification"). This matters for the shapes card: a shapes-column layer is restyled with its stored spec when its Fill is toggled or its geometry is rebuilt (see "Shapes geometry writes"), and the column's values may have changed since the source was listed, for example after the shapes element was saved or reloaded.

The card no longer emits `ShapesLoadRequest`, and the type is deleted. As for labels, the builders take the shapes name and the spec directly: `build_primary_shapes_loaded_card_spec(shapes_name, result)` and `build_styled_shapes_card_spec(shapes_name, style_spec, result)`, where `style_spec` is a `ShapeColumnColorSourceSpec | TableColorSourceSpec`.

### Viewer widget (`src/spatiato/widgets/viewer/widget.py`)

Mirror the image handlers:

- `_get_live_labels_bindings(...)` / `_get_live_shapes_bindings(...)`: partition the live bindings of one element in the active `sdata` and coordinate system into the primary binding and the styled bindings, in viewer order. Raise `ValueError` for duplicates.
- `_refresh_labels_card_membership(card)` / `_refresh_shapes_card_membership(card)`: call the reconcile method, or report a card-scoped membership error.
- Connect `labels_layers_changed` / `shapes_layers_changed` to refresh every labels / shapes card. Refresh each card once when it is built in `_rebuild_labels_cards(...)` / `_rebuild_shapes_cards(...)`.
- Add handlers: when the source is already live, reconcile and activate its layer. Otherwise call `ensure_styled_labels_loaded(...)` / `ensure_styled_shapes_loaded(...)`, activate the new layer and show the existing status card (`build_styled_labels_card_spec(...)` / `build_styled_shapes_card_spec(...)`). When the load raises `ValueError`, show the error card as today. The adapter leaves no layer behind (see "Adapter changes", item 5), so no row appears and the source stays in the completer.
- Add remove, remove-all and visibility handlers. They resolve the live binding before mutating, as `_remove_image_overlay_channel(...)` and `_change_image_overlay_channel_visibility(...)` do.
- Add primary load, remove and visibility handlers, reusing `_add_or_update_primary_labels_layer(...)` / `_add_or_update_primary_shapes_layer(...)` for loading.

### Refresh after table changes

Today a user restyles a loaded layer after a table change by clicking **Add / Update** again. Once a loaded source is no longer offered in the completer, that route is gone, so the Viewer restyles loaded layers itself. This is decided (see "Decisions"): the restyle is automatic and scoped, and rows get no refresh button.

The rule, in `ViewerWidget._on_table_state_changed(...)`:

- `change_kind` `"updated"` or `"created"`: restyle each loaded styled layer whose spec is an `obs_column` source of `event.table_name` whose column has a matching `TableComponentPath("obs", (column,))` in `event.paths`. Var sources are not restyled for these events: `TableComponentPath` only covers `obs`, `obsm` and `uns`, so the app never reports `X` or `layers` changes this way.
- `change_kind` `"rebuilt"` or `"reloaded"`: restyle every loaded styled layer that reads `event.table_name`.
- `change_kind` `"removed"`, or a restyle that raises `ValueError` because the table or column is gone, or because the column can no longer color a layer: keep the layer as it is and show a warning on its row ("source no longer available") through `set_styled_source_warning(...)`. The Viewer never removes the layer automatically; the user removes it with the row's remove button or "Remove all".

Restyling calls `ensure_styled_*_loaded(...)` with the spec stored on the binding. For shapes it passes the binding's `fill`. No freshly built source has to be looked up:

- **Colors follow the current data.** The coloring code reads the data every time it runs: `obs` columns and shapes columns are classified from their current values by `classify_color_values(...)` (see "Shared value classification"), vars are always colored as continuous, and instance coloring is decided by the table's instance-key column name. A column whose kind changed is therefore colored correctly.
- **A source that is gone or no longer colorable raises.** A removed table, column or var raises `ValueError` in the coloring code, and since phase 1 so does a column that can no longer color a layer, for example one that holds only missing values. The restyle fails, and the row gets the warning.
- **The stored spec cannot be out of date.** After phase 2a a spec holds only the fields that name the data (see "Matching a loaded source"), and those do not change when the data changes.

`_on_table_state_changed(...)` still refreshes the linked-table sources of the cards (`set_linked_tables(...)`) as it does today, so the completer offers the current sources. Shapes-column sources are not table sources, so table events never restyle them; saves and reloads of the shapes element do (see "Shapes geometry writes").

The early return for `object_classification_annotation` / `"updated"` events in `_on_table_state_changed(...)` exists so that rebuilding the linked-table controls does not slow down annotation. It should keep skipping that rebuild, but the scoped restyle above still has to run, so that a loaded layer colored by `user_class` follows the annotations. Because the restyle is scoped to matching paths, it costs nothing unless such a layer is loaded.

### Matching a loaded source

The card (to filter the completer and find rows) and the adapter (to find an existing layer) must agree on when two specs mean the same styled layer. Neither comparison that exists today fits:

- `TableColorSourceSpec.identity` leaves out `var_layer`, but `vars (X): CD3` and `vars (counts): CD3` are separate layers that can both be loaded.
- Full dataclass equality includes `value_kind`. If a table update changes how a column is classified, for example from continuous to categorical, the new spec no longer equals the loaded one, and loading it creates a second layer for the same column.

**`value_kind` is removed from both spec classes** in `src/spatiato/core/_color_source.py`. After phase 1 no coloring path reads it to decide how to color. It is still read in two places, and neither needs it:

- the `"instance"` checks in `apply_table_color_source_to_labels_layer(...)`, `apply_table_color_source_to_shapes_layer(...)` and `_get_table_color_values_by_instance(...)`. Each check also matches when the column is the table's instance-key column; the `value_kind` part only raises for a spec that claims `"instance"` for another column, which discovery never produces. The checks keep only the column-name test, and their inner guard, which can then never be reached, is deleted (see "Adapter changes", item 3);
- the adapter's whole-spec comparison, which this section is about.

A spec then holds only the fields that name the data, and plain equality of the frozen dataclasses identifies the layer:

- `TableColorSourceSpec`: `table_name`, `source_kind`, `value_key`, `var_layer`;
- `ShapeColumnColorSourceSpec`: `source_kind`, `value_key`.

No separate key is needed: the card and the adapter compare specs with `==`, and a spec can be a dictionary key. `identity` stays as it is: it leaves out `var_layer` on purpose, so that a selected var stays selected when the user switches matrix. Specs of the two classes are never equal to each other.

**Why these fields.** A layer's identity is made only of the fields that name the data the user picked: the table, `obs` column or var, the key, and the matrix. They do not change when the data changes. `value_kind` describes the data instead: discovery classified it from the column's values at one moment, and it can be out of date by the time the layer is looked up again. That is why it is removed from the spec rather than only left out of a comparison. A field added to a spec later must also name the data, not describe it; how the data was colored belongs in the style result.

The style result (`LabelsStyleResult`, `ShapesStyleResult`) does not identify the layer either, for three reasons:

- **The lookup happens before coloring.** `ensure_styled_*_loaded(...)` first looks for an existing layer for the requested spec, and only then colors it. In the new card, leaving loaded sources out of the completer and matching rows to layers happen without any coloring at all.
- **The binding stores the spec, not the style result.** The style result goes to the status card and is then dropped.
- **It describes how the layer was colored last, and changes between restyles of the same layer.** `palette_source` changes from `"default_missing"` to `"stored"` once a palette for the column is added to the table's `uns`, and `value_kind` changes with the column. Keyed on the style result, such a restyle would look like a different layer.

What else follows from the removal:

- **Discovery** still classifies each column, to decide whether to offer it, but no longer stores the result on the spec.
- **The style results keep their `value_kind`.** They record what coloring did, and the status cards read it. `TableColorValueKind`, `ShapeColorValueKind` and their validation functions stay for them.
- **A binding is never replaced because of a changed kind.** The binding replacement in "Adapter changes" is only needed when a shapes binding's `fill` changes.
- **The planned colorbar** in `Roadmap/colormap/colormap.md` decides from the loaded layer's style result whether a source is continuous, not from the spec.

### Shapes Fill

**Fill** moves from the request to the loaded row:

- the checkbox on a styled row toggles fill on that layer only;
- the first implementation calls `ensure_styled_shapes_loaded(..., fill=...)`, which already restyles an existing layer in place;
- new rows start unfilled, as today.

**Only layers rendered as `Shapes` have Fill.** Point-radius shapes are rendered as a `Points` layer, and the `Points` color functions in `src/spatiato/viewer/shapes_styling.py` (`_apply_rendered_row_colors_to_points_layer(...)`, `_apply_table_rendered_row_colors_to_points_layer(...)`) take no `fill` and always fill. A row shows the checkbox only when its binding has `shapes_rendering_mode == "shapes"`.

**The Fill state lives on the binding** (`ShapesLayerBinding.fill`, see "Adapter changes"). The row renders its checkbox from `binding.fill`. Reading it back from the layer is not reliable: fill only changes the face alpha (`_styled_shapes_face_alpha(...)`), and hidden rows (shapes with no table row, and not-detected values) also get face alpha 0, so a layer whose rows are all hidden looks unfilled either way. Keeping the state on the binding also follows the goal of reconciling the card from live state rather than card-local bookkeeping. Toggling Fill replaces the binding, which emits `shapes_layers_changed` and rebuilds that row from the new binding.

### Shapes geometry writes

A styled shapes layer is built from the shapes element as it was at load time. `_build_shapes_layer(...)` copies the geometry into the napari layer and records which source row each rendered row comes from (`source_row_id_by_rendered_row`). When the Annotation widget saves the element (`shapes_element_written`) or reloads it from disk (`shapes_element_reloaded`), that copy is out of date:

- deleted shapes leave row ids that no longer exist, and a restyle raises in `_validate_source_row_id_by_rendered_row(...)`;
- added shapes are missing from the styled layer;
- edited shapes keep their old geometry.

A restyle cannot fix this, so the layer is rebuilt. The rendering mode can change too, for example when only point-radius shapes are left, which turns a `Shapes` layer into a `Points` layer. The rebuilt layer is therefore a new layer object, not the old one with new data.

**Adapter.** Add

```python
def rebuild_styled_shapes_layer(
    self, sdata, shapes_name, coordinate_system, *, style_spec, fill=False
) -> ShapesLoadResult: ...
```

following `replace_primary_shapes_layer(...)`:

- build a replacement with `_build_shapes_layer(..., sync_current_colors=False)` and color it with the shapes-column or table coloring function;
- add it to the viewer and register it as a styled binding with `style_spec` and `fill` before the old layer is removed. `fill` is normalized to `False` when the replacement is rendered as `Points`. If building, coloring or registering fails, only the replacement is removed, and the old layer and binding stay as they are;
- keep the old layer's place in the layer list, its `visible` and `opacity`, and its name. `_add_layer_to_viewer(...)` appends, so the replacement is moved to the old layer's index (`LayerList.move(...)`) before the old layer is removed. Napari adds a suffix to the name while both layers are present, so the name is set again after the removal, as `replace_primary_shapes_layer(...)` does.

Registering the replacement and removing the old layer emit `shapes_layers_changed`, so the card swaps the row. The row's Fill state comes from the new binding, and its warning, if any, is keyed by the spec, so neither is lost.

**Viewer.** `_on_shapes_element_written(...)` and `_on_shapes_element_reloaded(...)` already rebuild the shapes cards for an event in the active SpatialData object and coordinate system. After that, they rebuild every live styled shapes layer of `event.shapes_name`:

- the event does not say what changed, so every write or reload rebuilds, whether or not the geometry changed;
- the spec is the one stored on the binding, as for table refresh. A rebuild that raises `ValueError`, for example because the colored column was removed or can no longer color a layer, keeps the old layer and gets a row warning with the error message;
- `fill` is the binding's `fill`;
- the primary shapes layer is left alone. The Annotation widget already keeps it in sync: `sync_primary_shapes_layer_binding(...)` after a save, and `replace_primary_shapes_layer(...)` after a reload.

The cards are rebuilt before the layers, so warnings set during the layer rebuild are not dropped by the card rebuild.

## Load Cost

Each accepted selection runs one synchronous load. That is the same work a single **Add / Update** click does today; it is just triggered sooner.

- Labels: the layer itself is lazy. Styling reads one `obs` column or one matrix column (`read_matrix_column(...)`) and builds a compact colormap. This is cheap for in-memory tables. A lazily backed feature matrix may hit disk.
- Shapes: each new styled layer builds a full napari layer through `_build_shapes_layer(...)`. Point-radius shapes become a `Points` layer and load fast. Large polygon sets can take seconds per selection.
- Discovery: the cards classify the columns of their linked tables and shapes elements when they are built and after table changes, not on each selection. Phase 3 makes this cheaper without changing which columns are offered (see "Discovery cost").
- Shapes geometry writes: each save or reload of a shapes element rebuilds all of its styled layers (see "Shapes geometry writes"). With several polygon layers loaded, a save can take several seconds. The alternative is a styled layer that no longer matches the geometry, so this cost is accepted.

If shapes loading feels too slow with immediate loading, a busy indicator on the card is the first mitigation. Background loading is out of scope here.

## Decisions

- **Refresh after table changes**: the Viewer restyles affected loaded layers automatically, scoped to the table changes that touch them, as described in "Refresh after table changes". Rows get no refresh button.
- **Missing source**: when the table or column of a loaded layer disappears, the layer stays in the viewer as it is and its row shows a warning. The Viewer never removes it automatically.
- **Failed first load**: a styled layer is colored before it is added to the viewer and registered. When coloring or registering a newly created layer fails, no layer is left, no row appears, and the Viewer shows the error card. A failed restyle of a loaded layer keeps the layer, as under "Missing source". This is part of phase 2b. See "Adapter changes", item 5.
- **Row labels with several linked tables**: when an element has more than one linked table, table-backed rows are prefixed with the table name. Rows stay in one flat list in napari layer order; they are not grouped under table headings.
- **Shared row**: the generic part of `_ImageLayerRow` moves to `_LayerRow` in a new neutral module, `src/spatiato/widgets/layer_row.py`. `_ImageLayerRow` stays in `image_layer_row.py` as a subclass that adds the color button and colormap sync, together with the image channel and colormap helpers. It is not generalized in place, because labels and shapes rows would then sit in an image-named class with a color option that has to be off for them. It is not moved and renamed as a whole either, because the neutral module would then hold image colormap and channel code. See "Shared layer row".
- **Primary row placement**: the primary layer has its own row at the top of the card, above "Linked table" and the other color-source controls. It is not part of the loaded list, because:
  - the list follows napari layer order, so the primary layer could not be pinned first without an exception;
  - "Remove all" and the "N colored layers" count cover only styled layers;
  - while the primary layer is absent, its row shows a **Load in viewer** button, and the list shows only live layers;
  - the image card keeps its Stack row in a container separate from the overlay list in the same way;
  - other widgets restyle (Object Classification, Spatial Query) or edit (Annotation) the primary layer, so it is a different kind of layer from the colored ones;
  - the list's scroll area could scroll it out of view.
- **Primary shapes binding sync**: `sync_primary_shapes_layer_binding(...)` emits `shapes_layers_changed`, so cards re-read the replaced binding object. It still does not emit `primary_shapes_layer_registered`. See "Adapter changes".
- **Specs name the data**: `value_kind` is removed from `TableColorSourceSpec` and `ShapeColumnColorSourceSpec`, so a spec holds only the fields that name the data the user picked, and plain spec equality identifies a loaded layer, for the adapter and the cards alike. No coloring path reads `value_kind` after phase 1, and it can go out of date when the data changes. The style result does not identify a layer either: it describes how the layer was last colored, and it keeps its `value_kind`. `identity` keeps its role of preserving a selection across matrix switches. This is phase 2a. See "Matching a loaded source".
- **Restyle spec**: a restyle after a table change, or a rebuild after a shapes write, uses the spec stored on the binding. Coloring reads the current data, and a source that is gone or can no longer color a layer raises and gets the row warning, so no freshly built source is looked up. See "Refresh after table changes".
- **Fill for point-radius shapes**: only rows of layers rendered as `Shapes` show a Fill checkbox. Layers rendered as `Points` are always filled. See "Shapes Fill".
- **Fill state**: stored as `fill` on `ShapesLayerBinding`, not in the card and not read back from the layer. It is added in phase 2b with the other adapter features. See "Adapter changes" and "Shapes Fill".
- **Source warnings**: the card's loaded list keeps the "source no longer available" warnings per spec and reapplies them when it reconciles its rows; `ViewerWidget` sets them through `set_styled_source_warning(...)`. See "Labels card".
- **Status-card builders**: they take the element name and the spec directly, and `LabelsLoadRequest` / `ShapesLoadRequest` are deleted. See "Labels card" and "Shapes card".
- **Shared value classification**: one classifier, `classify_color_values(...)` in `src/spatiato/core/color_value_classification.py`, decides how a column's values are colored, for discovery and for every coloring path: labels and shapes colored from a table, and shapes colored from their own columns. It returns an encoding, not only a kind, so the coloring paths know where categories and palettes come from. Coloring no longer reads `value_kind` from the spec, and a column that is no longer colorable raises and gets the row warning. Labels coloring of `object`-dtype boolean and 0/1 columns changed from continuous to categorical. This was phase 1, which is done. See "Shared value classification".
- **Geometry writes**: styled shapes layers follow saves and reloads of their shapes element. They are rebuilt as new layers in the same place, with the same visibility, opacity, name, spec and Fill; a failed rebuild keeps the old layer and shows a row warning. This is phase 8, in the same PR as the rest of this feature. See "Shapes geometry writes".
- **Discovery keeps classifying**: discovery offers only the columns the classifier can classify, because in the new card a picked value loads immediately and an unusable column would only produce an error card. Its cost is reduced instead, without changing any result: a vectorized classifier with the same rules, and one discovery per linked table per pass of the labels and shapes cards rather than per card, without a cache kept between passes. This is phase 3. See "Discovery cost".

## Implementation Phases

Labels go before shapes:

- labels have one source type (`TableColorSourceSpec`), no Fill and no shapes-column sources;
- the primary labels layer has a simpler lifecycle. The Annotation widget also removes and re-adds the primary shapes layer (`normalize_native_shapes_layer_for_annotation(...)`), swaps it (`replace_primary_shapes_layer(...)`) and replaces its binding object (`sync_primary_shapes_layer_binding(...)`, which today emits no signal and after phase 2b emits only `shapes_layers_changed`). Those reconcile edge cases are easier to handle once the pattern works for labels;
- styled labels layers load cheaply, so the flow is quick to try by hand; polygon shapes can take seconds per layer;
- the main table-refresh case, Object Classification annotating `user_class`, is a labels case;
- labels have fewer existing Add / Update tests to rewrite (about 10, against about 16 for shapes).

Phases 1 to 4 change no card. Phases 2 and 4 still cover the parts shapes need, so the shared design is tested before the labels card depends on it.

1. **Shared value classification** (done): `classify_color_values(...)` in `src/spatiato/core/color_value_classification.py`, used by discovery and by the three coloring paths, including the shapes-column path, which no longer reads `value_kind` from the spec. Tests pinned down the earlier results per column type first. It touched only `core` and the two styling modules and has no UI, and phases 6 to 8 rely on discovery and coloring agreeing.
2. **Adapter, for labels and shapes together**, in two steps. Both element types are handled in the same adapter functions (`_handle_registered_binding(...)`, `_on_viewer_layer_inserted(...)`, `_on_viewer_layer_removed(...)`, `_on_viewer_layers_reordered(...)`, `_remove_layer_from_viewer_and_registry(...)`), so the work is split by kind of change, not by element type, which would edit those functions twice:
   - **2a. Specs without `value_kind`** (done; "Adapter changes", item 3): remove the field from both spec classes in `src/spatiato/core/_color_source.py` and from their docstrings, from the specs that discovery, the var sources and Spatial Query build, and from the tests. Reduce the three `"instance"` checks to the name test. The lookups by spec need no change. Apart from spec equality, behavior stays the same; most of the diff is removing `value_kind=` arguments from the tests (see "Tests"). Run the affected test modules before starting 2b.
   - **2b. Adapter features** (done; "Adapter changes", items 1, 2, 4 and 5): the `labels_layers_changed` / `shapes_layers_changed` signals, including the emission from `sync_primary_shapes_layer_binding(...)`; `remove_styled_labels_layer(...)` / `remove_styled_shapes_layer(...)`, with the `style_spec` filter on `get_loaded_styled_*_layers(...)`; the `fill` field on `ShapesLayerBinding`, and the binding replacement through `LayerBindingRegistry.replace_binding(...)` when `fill` changes; and the build, color, add, register order in `ensure_styled_labels_loaded(...)` / `ensure_styled_shapes_loaded(...)`, so a failed first load leaves no layer. Write the tests for each part first.

   2a comes first because it is mechanical and touches many test files, while 2b adds behavior. Kept apart, the 2b diff holds only new behavior, and its tests are written against specs without `value_kind`.
3. **Discovery cost**: the vectorized `classify_color_values(...)`, with the same rules, and one `LinkedTableColorSources` per linked table per pass, shared by the labels and shapes cards of that pass, instead of one per card (see "Discovery cost"). It changes no card and no result: the existing classification and discovery tests pass unchanged. It is independent of the other phases, and comes before the cards so that the refreshes they add stay cheap.
4. **Shared row**: add `_LayerRow` in `src/spatiato/widgets/layer_row.py` and make `_ImageLayerRow` a subclass of it. Build and test the extra-control slot here, with a placeholder widget, although only shapes rows use it, so the row API does not change in phase 7.
5. **Labels card**: primary row, completer-driven add, loaded list, reconcile method, and the `ViewerWidget` handlers and membership refresh. Build the loaded list as a reusable widget so the shapes card can use it in phase 7, including its warning mapping and `set_styled_source_warning(...)`. Change the labels status-card builders to take the labels name and spec, and delete `LabelsLoadRequest`.
6. **Table refresh**: scoped restyle on `table_state_changed` with the spec stored on the binding, and stale-source warnings.
7. **Shapes card**: reuse the loaded list from phase 5, and add shapes-column sources and per-row Fill, shown only for layers rendered as `Shapes`. Change the shapes status-card builders to take the shapes name and spec, and delete `ShapesLoadRequest`.
8. **Shapes geometry writes**: `rebuild_styled_shapes_layer(...)` in the adapter, and the rebuild of live styled shapes layers in `_on_shapes_element_written(...)` / `_on_shapes_element_reloaded(...)`. It is a separate slice because it fixes a gap that exists today, independent of the new card, but it ships in the same PR. It comes after phase 7, which provides the per-row Fill it relies on; the coloring-time classification comes from phase 1.

`docs/development/styled_layers.md` describes the adapter contract as of phase 2b. Each of phases 5 to 8 updates it with what it adds:

- phase 5: the labels card as a listener of `labels_layers_changed`, loading on pick instead of **Add / Update in viewer**, and the new `ViewerWidget` handlers under "Where to find it". Also update the definition of a styled layer in `docs/development/labels_coloring.md`, which says it is created by **Add / Update in viewer**;
- phase 6: the restyle on `table_state_changed` with the spec stored on the binding, and the stale-source warnings;
- phase 7: the shapes card, and the per-row Fill checkbox as the source of the `fill` passed to a restyle;
- phase 8: `rebuild_styled_shapes_layer(...)`, the rebuild after a geometry write, added to "Loading and restyling".

Release phases 5 to 8 together. Between them, the labels card loads immediately while the shapes card still uses **Add / Update in viewer**. That is fine on the feature branch, but confusing in a release.

## Tests

Value classification (new `tests/test_color_value_classification.py`):

- one case per encoding: a pandas categorical column, a boolean column, an `object` column of booleans with missing values, 0/1 integers with and without missing values, other integers, floats, and strings;
- `None` for an all-missing non-categorical column and for a column with mixed value types, and `"pandas_categorical"` for an all-missing categorical column;
- discovery (`tests/test_color_source_options.py`) offers the same sources as before.

Discovery cost (phase 3):

- the existing tests in `tests/test_color_value_classification.py` and `tests/test_color_source_options.py` pass unchanged;
- each `infer_dtype` answer that decides directly gets a case with the same result as before: `object` columns of booleans, integers, 0/1, floats, integers mixed with floats, strings and bytes; `None` for datetime64, timedelta64, period, interval and complex dtypes, and for `object` columns of datetimes, dates, times, timedeltas and `Decimal` values;
- the fallback keeps the current result: strings mixed with bytes are string, `Fraction` values and integers mixed with `Fraction` values are continuous, booleans mixed with integers, integers mixed with strings, and dicts are `None`;
- a nullable integer column (`Int64`) and a pandas `string` column classify as before;
- in the Viewer widget, building the cards and refreshing them after a table change build `LinkedTableColorSources` once per linked table, even when the table annotates several labels and shapes elements (for example `table_multi` in `sdata_blobs_multi_region`), and the cards linked to one table hold the same object.

Labels styling (`tests/test_labels_styling.py`):

- an `object`-dtype boolean column with missing values, and an `object`-dtype 0/1 column, are colored as categorical (before phase 1: continuous);
- a 0/1 column is colored as categorical on a labels element whose own rows hold only 0s, because the full column is classified;
- a column that discovery would not offer, for example one holding only missing values, raises `ValueError` (before phase 1: colored as continuous);
- every other column type is colored as before.

Color source specs (`tests/test_color_source.py`):

- `TableColorSourceSpec` and `ShapeColumnColorSourceSpec` no longer take `value_kind`;
- two table specs are equal exactly when their table, source kind, key and matrix are equal, so `vars (X)` and `vars (counts)` of the same var are different specs;
- a table spec never equals a shapes-column spec;
- `identity` is unchanged.

Adapter (`tests/test_viewer_adapter.py`):

- `labels_layers_changed` / `shapes_layers_changed` fire on register, on insertion of an already registered layer, on napari removal, on reorder and on `_remove_layer_from_viewer_and_registry(...)`, for primary and styled layers;
- they fire for bindings registered without a coordinate system;
- `sync_primary_shapes_layer_binding(...)` emits `shapes_layers_changed` but not `primary_shapes_layer_registered`;
- `remove_styled_*_layer(...)` removes only the matching variant and leaves siblings and the primary layer alone;
- `remove_styled_*_layer(...)` returns `None` when no layer matches, and raises `ValueError` when two live layers match the same spec;
- `get_loaded_styled_*_layers(..., style_spec=...)` returns only the layers for that spec, and every styled layer of the element without it;
- loading the same source again after its column changed kind resolves to the existing styled layer and restyles it, without a second layer;
- `X` and a layer for the same var stay separate layers;
- the instance-key column is still colored as instances, decided by its name alone;
- `ShapesLayerBinding` rejects `fill=True` for a primary binding and for a binding rendered as `Points`;
- `ensure_styled_shapes_loaded(..., fill=True)` records `fill=True` for a layer rendered as `Shapes`, and `False` for a point-radius layer rendered as `Points`;
- restyling with a different `fill` replaces the binding and emits `shapes_layers_changed`; restyling with the same spec and `fill` keeps the binding object and emits nothing;
- `LayerBindingRegistry.replace_binding(...)` replaces the binding of a registered layer, and raises `ValueError` for a layer without a binding;
- when coloring a newly created styled labels or shapes layer raises, the call raises, no layer or binding is left, and no membership signal fires;
- when registering a newly created styled labels or shapes layer fails, the layer is removed from the viewer;
- when restyling an existing styled layer raises, the layer and its binding stay;
- a layer colored before it is added to the viewer keeps its colors after insertion: the colormap and features of a labels layer, and the colors of a shapes layer rendered as `Shapes` or as `Points`;
- `rebuild_styled_shapes_layer(...)` puts a new layer in the old layer's place in the layer list, keeps `visible`, `opacity`, name, spec and `fill`, and emits `shapes_layers_changed`;
- it handles a change of rendering mode from `Shapes` to `Points`, with `fill` normalized to `False`;
- when building, coloring or registering the replacement fails, the old layer and binding stay, and no replacement is left in the viewer.

Shapes styling (`tests/test_shapes_styling.py`):

- a shapes-column layer whose column changed kind after loading is colored with the column's current kind;
- a shapes column that is no longer colorable raises `ValueError`;
- table-backed and shapes-column coloring color every column type as before.

Layer rows:

- new `tests/test_layer_row.py` for `_LayerRow`: intents are emitted without mutating the layer, visibility follows native events, `dispose()` is idempotent, the warning state renders, an extra control can be added and its signals reach the owner, and rows can be built for `Labels`, `Shapes` and `Points` layers;
- the existing `tests/test_image_layer_row.py` keeps passing unchanged.

Viewer widget (`tests/test_viewer_widget.py`):

- accepting a completer value loads the styled layer without any button click, and the row appears;
- a first load whose coloring fails shows the error card, adds no row and leaves no layer in the viewer;
- a loaded source is left out of the completer and comes back after removal;
- deleting a styled layer in napari removes its row;
- reordering layers in napari reorders the rows;
- "Remove all" removes only styled layers, not the primary layer;
- the primary row switches between **Load in viewer** and a live row when the primary layer is added or removed, including from napari;
- an `obs` table update restyles a loaded layer colored by that column and leaves other layers alone;
- a table rebuild that changes how a column is classified restyles the loaded layer in place: no second layer is created, and the colors follow the new kind;
- a column that still exists but can no longer color a layer, for example one holding only missing values, gets the row warning: the restyle raises;
- a removed source keeps the layer and shows a row warning;
- a warning survives a row rebuild (for example a Fill toggle) and is cleared by a successful restyle;
- the status-card builders produce the same cards from the element name and spec as they did from the request;
- shapes: toggling Fill on a row changes only that layer, and the checkbox follows `binding.fill`;
- shapes: rows of point-radius layers rendered as `Points` have no Fill checkbox;
- shapes: after a save that deletes and adds shapes, a styled shapes layer shows the new geometry, its row stays with the same Fill, and the Viewer does not touch the primary layer;
- shapes: after a save that removes the colored shapes column, the old layer stays and its row shows a warning;
- shapes: a reload rebuilds styled shapes layers in the same way as a save.

About 30 existing references to `add_update_button` in `tests/test_viewer_widget.py` cover the labels and shapes Add / Update flow and need to be rewritten for the new intents.

Removing `value_kind` from the specs in phase 2a removed about 130 `value_kind=` arguments from spec constructors and test helpers, across nine test files: `tests/test_viewer_adapter.py`, `tests/test_shapes_styling.py`, `tests/test_viewer_widget.py`, `tests/test_viewer_status_card.py`, `tests/test_color_source.py`, `tests/test_labels_styling.py`, `tests/test_table_color_sources.py`, `tests/test_shapes_annotation_widget.py` and `tests/test_spatialdata.py`. The style results keep their `value_kind`, so assertions on `result.value_kind` stay, and so do the `value_kind` fields of load-result stubs in `tests/test_viewer_widget.py`.
