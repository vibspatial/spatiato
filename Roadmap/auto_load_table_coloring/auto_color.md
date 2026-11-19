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

Today four places decide how a column's values are colored, each with its own rules:

| Path | Function | Rule |
|---|---|---|
| Discovery, for `obs` columns and shapes columns | `_classify_scalar_column_color_source(...)` in `src/spatiato/core/color_source_options.py` | the dtype, then the values of an `object` column |
| Labels colored from a table | `_build_obs_column_colormap(...)` in `src/spatiato/viewer/labels_styling.py` | the dtype only, for booleans and 0/1 integers |
| Shapes colored from a table | `_build_table_obs_shape_style(...)` in `src/spatiato/viewer/shapes_styling.py` | the dtype or the values, through `_is_bool_series(...)` and `_is_exact_binary_integer_series(...)` |
| Shapes colored from a shapes column | `apply_shape_column_color_source_to_shapes_layer(...)`, then `_build_categorical_shape_style(...)` | `value_kind` from the spec, then its own dtype checks, with a fallback that treats any other values as categories |

They already disagree. A boolean column with missing values, or a 0/1 column with missing values, has `object` dtype in pandas. Discovery and both shapes paths classify it as categorical, but labels coloring falls through to continuous. The same `obs` column is therefore colored categorical on a shapes element and continuous on a labels element. The shapes-column path also colors with the kind stored on the spec, which can be stale (see "Shapes card").

One classifier replaces the four. Coloring needs more than categorical or continuous: it needs to know which kind of categorical column it has, because that decides where the categories come from and whether a stored palette may be used. So the classifier returns an encoding:

```python
ColorValueEncoding = Literal["pandas_categorical", "boolean", "binary_integer", "string", "continuous"]

@dataclass(frozen=True)
class ColorValueClassification:
    encoding: ColorValueEncoding

    @property
    def value_kind(self) -> Literal["categorical", "continuous"]: ...

def classify_color_values(values: pd.Series) -> ColorValueClassification | None: ...
```

- **Rules.** Its rules are today's discovery rules, which already look at the values of `object` columns. It returns `None` where discovery does: for a column with no non-missing values, or with mixed value types.
- **Discovery** uses `.value_kind`, and `None` still means the column is not offered.
- **The three coloring paths** branch on `.encoding` instead of their own checks:
  - `"string"` keeps the default palette and `coercion_applied=True`;
  - `"pandas_categorical"`, `"boolean"` and `"binary_integer"` may use a stored palette.

  When the result is `None`, the coloring path raises `ValueError`, which the Viewer turns into the "source no longer available" row warning.
- **The shapes-column path stops reading `value_kind` from the spec.** For every source type, `value_kind` on a spec then records what discovery saw, and coloring always reads the data again.
- **Helpers move to `core`.** `is_string_like_series(...)` lives in `src/spatiato/viewer/_styling.py` today, and `core` must not import from `viewer`. The viewer modules import the classifier from `core`, and `_is_bool_series(...)`, `_is_exact_binary_integer_series(...)` and the separate labels dtype checks are deleted.

What stays outside the classifier, because it is not about the values:

- instance coloring: the role of the table's instance-key column, decided by its name before classification;
- vars: always colored as continuous;
- palette lookup: the table's stored palette for `obs` columns, a companion `_colors` column for shapes columns;
- building colors: labels build a colormap and shapes build per-row RGBA colors; each keeps its own builder.

**Behavior change.** Labels coloring of `object`-dtype boolean and 0/1 columns moves from continuous to categorical, matching discovery and shapes. The fallback in `_build_categorical_shape_style(...)` that treats other values as categories becomes unreachable and is removed. Start with tests that pin down today's results per column type, so this is the only difference.

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

3. **Matching key for a styled source.** See "Matching a loaded source" below. Every lookup by spec compares `style_spec.layer_key` instead of full spec equality:

   - `get_loaded_styled_labels_layer(...)` and `get_loaded_styled_shapes_layer(...)`, through `_matches_labels_binding(...)` and `_matches_shapes_binding(...)`;
   - `LayerBindingRegistry.find_bindings(style_spec=...)`. No caller passes `style_spec` today (the Histogram widget filters image bindings only), so this keeps one rule at no cost.

4. **Fill on the shapes binding.** Add `fill: bool = False` to `ShapesLayerBinding`, so the Fill state of a styled shapes layer is read from live state rather than kept by the card (see "Shapes Fill"):

   - `ShapesLayerBinding.__post_init__(...)` rejects `fill=True` unless the binding is styled and rendered as `Shapes` (`shapes_rendering_mode == "shapes"`);
   - `LayerBindingRegistry.register_shapes_layer(...)` and `ViewerAdapter.register_shapes_layer(...)` accept `fill`;
   - `ensure_styled_shapes_loaded(..., fill=...)` records `fill` on the binding of a layer rendered as `Shapes`. For a point-radius layer rendered as `Points` it records `False`, since fill has no effect there. Today's card passes its Fill checkbox for point-radius shapes too, so normalizing instead of raising keeps today's flow working;
   - when an existing styled layer is restyled with a different `fill` or a different `style_spec`, its binding is replaced through the same path as a `value_kind` change (see "Matching a loaded source"). When neither differs, the binding is left as it is, so a plain restyle emits no membership signal and the card keeps its rows.

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

The loaded list also holds the "source no longer available" warnings, as a mapping from `layer_key` to message, and applies them every time it reconciles its styled rows. They are not stored on the rows: a row is rebuilt whenever its binding object changes, for example on a `value_kind` or Fill replacement, and a flag on the row would be lost. `ViewerWidget` sets and clears a warning through the card:

```python
def set_styled_source_warning(self, layer_key: tuple[object, ...], message: str | None) -> None: ...
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

**Shapes-column sources are classified from the current column when they are colored**, through the shared classifier (see "Shared value classification"), not from `value_kind` on the spec. This matters for the shapes card: a shapes-column layer is restyled with its stored spec when its Fill is toggled or its geometry is rebuilt (see "Shapes geometry writes"), and the stored kind can be stale after the shapes element was saved or reloaded. A column that became strings but is stored as continuous would otherwise be colored entirely as missing values.

The card no longer emits `ShapesLoadRequest`, and the type is deleted. As for labels, the builders take the shapes name and the spec directly: `build_primary_shapes_loaded_card_spec(shapes_name, result)` and `build_styled_shapes_card_spec(shapes_name, style_spec, result)`, where `style_spec` is a `ShapeColumnColorSourceSpec | TableColorSourceSpec`.

### Viewer widget (`src/spatiato/widgets/viewer/widget.py`)

Mirror the image handlers:

- `_get_live_labels_bindings(...)` / `_get_live_shapes_bindings(...)`: partition the live bindings of one element in the active `sdata` and coordinate system into the primary binding and the styled bindings, in viewer order. Raise `ValueError` for duplicates.
- `_refresh_labels_card_membership(card)` / `_refresh_shapes_card_membership(card)`: call the reconcile method, or report a card-scoped membership error.
- Connect `labels_layers_changed` / `shapes_layers_changed` to refresh every labels / shapes card. Refresh each card once when it is built in `_rebuild_labels_cards(...)` / `_rebuild_shapes_cards(...)`.
- Add handlers: when the source is already live, reconcile and activate its layer. Otherwise call `ensure_styled_labels_loaded(...)` / `ensure_styled_shapes_loaded(...)`, activate the new layer and show the existing status card (`build_styled_labels_card_spec(...)` / `build_styled_shapes_card_spec(...)`).
- Add remove, remove-all and visibility handlers. They resolve the live binding before mutating, as `_remove_image_overlay_channel(...)` and `_change_image_overlay_channel_visibility(...)` do.
- Add primary load, remove and visibility handlers, reusing `_add_or_update_primary_labels_layer(...)` / `_add_or_update_primary_shapes_layer(...)` for loading.

### Refresh after table changes

Today a user restyles a loaded layer after a table change by clicking **Add / Update** again. Once a loaded source is no longer offered in the completer, that route is gone, so the Viewer restyles loaded layers itself. This is decided (see "Decisions"): the restyle is automatic and scoped, and rows get no refresh button.

The rule, in `ViewerWidget._on_table_state_changed(...)`:

- `change_kind` `"updated"` or `"created"`: restyle each loaded styled layer whose spec is an `obs_column` source of `event.table_name` whose column has a matching `TableComponentPath("obs", (column,))` in `event.paths`. Var sources are not restyled for these events: `TableComponentPath` only covers `obs`, `obsm` and `uns`, so the app never reports `X` or `layers` changes this way.
- `change_kind` `"rebuilt"` or `"reloaded"`: restyle every loaded styled layer that reads `event.table_name`.
- `change_kind` `"removed"`, a restyle that raises `ValueError` because the table or column is gone, or a source with no fresh counterpart (see below): keep the layer as it is and show a warning on its row ("source no longer available") through `set_styled_source_warning(...)`. The Viewer never removes the layer automatically; the user removes it with the row's remove button or "Remove all".

Restyling uses the source as it is classified now, not the spec stored on the binding:

1. `_on_table_state_changed(...)` first refreshes the linked-table sources of the cards (`set_linked_tables(...)`), as it does today.
2. For each loaded styled layer to restyle, it finds the freshly built source with the same `layer_key` in the refreshed `LinkedTableColorSources`.
3. It calls `ensure_styled_*_loaded(...)` with that fresh spec. For shapes it passes the binding's `fill`.

The fresh spec is not needed for correct colors, because the coloring code reads the data every time it runs:

- `obs` columns and shapes columns are classified from their current values by `classify_color_values(...)` (see "Shared value classification");
- vars are always colored as continuous;
- `value_kind` is read from the spec only for `"instance"`, which is tied to the table's instance-key column and also matched by column name.

A restyle with the stored spec would therefore also color a column whose kind changed correctly. Shapes-column sources are not table sources, so table events never restyle them; saves and reloads of the shapes element do (see "Shapes geometry writes").

The fresh lookup exists for two other reasons:

- **It catches a column that still exists but is no longer colorable.** `_classify_scalar_column_color_source(...)` drops such a column, for example one that holds only missing values, so the completer no longer offers it. A restyle with the stored spec would not fail; it would color the layer from what is left, typically as missing values, without telling the user. With the fresh lookup there is no matching source, and the row gets the warning.
- **It keeps the binding's spec equal to what the completer offers.** When the fresh spec has a different `value_kind`, the adapter replaces the binding (see "Matching a loaded source"). Table refresh is the only route by which this happens, because a loaded source is left out of the completer and cannot be picked again.

The lookup costs only a search in the sources that step 1 already rebuilt.

When no fresh source has the same `layer_key`, the column or var is gone or no longer colorable, and the row gets the warning instead of a restyle.

The stored spec is used only where the sources are not rebuilt: the Object Classification `"updated"` early return described below. Annotation columns keep their categorical kind there, so the stored spec is still correct.

The early return for `object_classification_annotation` / `"updated"` events in `_on_table_state_changed(...)` exists so that rebuilding the linked-table controls does not slow down annotation. It should keep skipping that rebuild, but the scoped restyle above still has to run, so that a loaded layer colored by `user_class` follows the annotations. Because the restyle is scoped to matching paths, it costs nothing unless such a layer is loaded.

### Matching a loaded source

The card (to filter the completer and find rows) and the adapter (to find an existing layer) must agree on when two specs mean the same styled layer. Neither existing comparison fits:

- `TableColorSourceSpec.identity` leaves out `var_layer`, but `vars (X): CD3` and `vars (counts): CD3` are separate layers that can both be loaded.
- Full dataclass equality includes `value_kind`. If a table update changes how a column is classified, for example from continuous to categorical, the new spec no longer equals the loaded one, and loading it creates a second layer for the same column.

The key is a new `layer_key` property on both spec classes in `src/spatiato/core/_color_source.py`:

- `TableColorSourceSpec.layer_key`: `(table_name, source_kind, value_key, var_layer)`;
- `ShapeColumnColorSourceSpec.layer_key`: `(source_kind, value_key)`.

It lives on the spec classes, not in the adapter, because the card needs the same key in phase 4. It is a new property, not a change to `identity`: `identity` leaves out `var_layer` on purpose, so that a selected var stays selected when the user switches matrix, and it keeps that role. The two keys cannot collide, because a shapes-column key has a different `source_kind` and length.

A source whose `value_kind` changed then restyles the existing layer, and the binding's `style_spec` is replaced with the new spec. The same replacement is used when a shapes binding's `fill` changes. Build the new binding with `dataclasses.replace(binding, style_spec=new_spec)`, which keeps every other field. For shapes these include `source_row_id_by_rendered_row`, `source_shapes_index_feature_name` and `skipped_geometry_count`, which registering again through `register_shapes_layer(...)` would otherwise need passed again. Store it through a path that also calls `_handle_registered_binding(...)`, so the replacement emits `labels_layers_changed` / `shapes_layers_changed`. The registry stores one binding per layer object, so the new binding replaces the old one.

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

Registering the replacement and removing the old layer emit `shapes_layers_changed`, so the card swaps the row. The row's Fill state comes from the new binding, and its warning, if any, is keyed by `layer_key`, so neither is lost.

**Viewer.** `_on_shapes_element_written(...)` and `_on_shapes_element_reloaded(...)` already rebuild the shapes cards for an event in the active SpatialData object and coordinate system. After that, they rebuild every live styled shapes layer of `event.shapes_name`:

- the event does not say what changed, so every write or reload rebuilds, whether or not the geometry changed;
- the spec is the fresh one with the same `layer_key`, as for table refresh, taken from the rebuilt card's shapes-column sources or linked-table sources. A source with no fresh counterpart, or a rebuild that raises `ValueError`, keeps the old layer and gets a row warning with the error message;
- `fill` is the binding's `fill`;
- the primary shapes layer is left alone. The Annotation widget already keeps it in sync: `sync_primary_shapes_layer_binding(...)` after a save, and `replace_primary_shapes_layer(...)` after a reload.

The cards are rebuilt before the layers, so warnings set during the layer rebuild are not dropped by the card rebuild.

## Load Cost

Each accepted selection runs one synchronous load. That is the same work a single **Add / Update** click does today; it is just triggered sooner.

- Labels: the layer itself is lazy. Styling reads one `obs` column or one matrix column (`read_matrix_column(...)`) and builds a compact colormap. This is cheap for in-memory tables. A lazily backed feature matrix may hit disk.
- Shapes: each new styled layer builds a full napari layer through `_build_shapes_layer(...)`. Point-radius shapes become a `Points` layer and load fast. Large polygon sets can take seconds per selection.
- Shapes geometry writes: each save or reload of a shapes element rebuilds all of its styled layers (see "Shapes geometry writes"). With several polygon layers loaded, a save can take several seconds. The alternative is a styled layer that no longer matches the geometry, so this cost is accepted.

If shapes loading feels too slow with immediate loading, a busy indicator on the card is the first mitigation. Background loading is out of scope here.

## Decisions

- **Refresh after table changes**: the Viewer restyles affected loaded layers automatically, scoped to the table changes that touch them, as described in "Refresh after table changes". Rows get no refresh button.
- **Missing source**: when the table or column of a loaded layer disappears, the layer stays in the viewer as it is and its row shows a warning. The Viewer never removes it automatically.
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
- **Matching key**: styled layers are matched on a new `layer_key` property of the spec classes in `src/spatiato/core/_color_source.py`, shared by the adapter and the cards. `identity` keeps its role of preserving a selection across matrix switches. See "Matching a loaded source".
- **Restyle spec**: a restyle after a table change uses the freshly classified source with the same `layer_key`, not the spec stored on the binding. Colors would be correct either way, because the coloring code classifies table columns itself. The lookup turns a column that is no longer colorable into the "source no longer available" warning instead of a silent restyle, and keeps the binding's spec equal to what the completer offers. The stored spec is used only on the Object Classification `"updated"` early return. See "Refresh after table changes".
- **Fill for point-radius shapes**: only rows of layers rendered as `Shapes` show a Fill checkbox. Layers rendered as `Points` are always filled. See "Shapes Fill".
- **Fill state**: stored as `fill` on `ShapesLayerBinding`, not in the card and not read back from the layer. It is added in phase 2 with the other adapter changes. See "Adapter changes" and "Shapes Fill".
- **Source warnings**: the card's loaded list keeps the "source no longer available" warnings per `layer_key` and reapplies them when it reconciles its rows; `ViewerWidget` sets them through `set_styled_source_warning(...)`. See "Labels card".
- **Status-card builders**: they take the element name and the spec directly, and `LabelsLoadRequest` / `ShapesLoadRequest` are deleted. See "Labels card" and "Shapes card".
- **Shared value classification**: one classifier, `classify_color_values(...)` in `src/spatiato/core/color_value_classification.py`, decides how a column's values are colored, for discovery and for every coloring path: labels and shapes colored from a table, and shapes colored from their own columns. It returns an encoding, not only a kind, so the coloring paths know where categories and palettes come from. Coloring no longer reads `value_kind` from the spec, and a column that is no longer colorable raises and gets the row warning. Labels coloring of `object`-dtype boolean and 0/1 columns changes from continuous to categorical. This is phase 1. See "Shared value classification".
- **Geometry writes**: styled shapes layers follow saves and reloads of their shapes element. They are rebuilt as new layers in the same place, with the same visibility, opacity, name, spec and Fill; a failed rebuild keeps the old layer and shows a row warning. This is phase 7, in the same PR as the rest of this feature. See "Shapes geometry writes".

## Implementation Phases

Labels go before shapes:

- labels have one source type (`TableColorSourceSpec`), no Fill and no shapes-column sources;
- the primary labels layer has a simpler lifecycle. The Annotation widget also removes and re-adds the primary shapes layer (`normalize_native_shapes_layer_for_annotation(...)`), swaps it (`replace_primary_shapes_layer(...)`) and replaces its binding object (`sync_primary_shapes_layer_binding(...)`, which today emits no signal and after phase 2 emits only `shapes_layers_changed`). Those reconcile edge cases are easier to handle once the pattern works for labels;
- styled labels layers load cheaply, so the flow is quick to try by hand; polygon shapes can take seconds per layer;
- the main table-refresh case, Object Classification annotating `user_class`, is a labels case;
- labels have fewer existing Add / Update tests to rewrite (about 10, against about 16 for shapes).

Phases 1 to 3 change no card. Phases 2 and 3 still cover the parts shapes need, so the shared design is tested before the labels card depends on it.

1. **Shared value classification**: `classify_color_values(...)` in `src/spatiato/core/color_value_classification.py`, used by discovery and by the three coloring paths, including the shapes-column path, which stops reading `value_kind` from the spec. Start with tests that pin down today's results per column type. It touches only `core` and the two styling modules and has no UI, and phases 5 to 7 rely on discovery and coloring agreeing.
2. **Adapter, for labels and shapes together**: the `layer_key` property on both spec classes in `src/spatiato/core/_color_source.py`; the `labels_layers_changed` / `shapes_layers_changed` signals, including the emission from `sync_primary_shapes_layer_binding(...)`; `remove_styled_labels_layer(...)` / `remove_styled_shapes_layer(...)`; key-based matching in the styled-layer lookups and `find_bindings(...)`; the `fill` field on `ShapesLayerBinding`; and the binding replacement when `value_kind` or `fill` changes. Both element types are handled in the same adapter functions (`_handle_registered_binding(...)`, `_on_viewer_layer_inserted(...)`, `_on_viewer_layer_removed(...)`, `_on_viewer_layers_reordered(...)`, `_remove_layer_from_viewer_and_registry(...)`), so splitting the work would edit them twice.
3. **Shared row**: add `_LayerRow` in `src/spatiato/widgets/layer_row.py` and make `_ImageLayerRow` a subclass of it. Build and test the extra-control slot here, with a placeholder widget, although only shapes rows use it, so the row API does not change in phase 6.
4. **Labels card**: primary row, completer-driven add, loaded list, reconcile method, and the `ViewerWidget` handlers and membership refresh. Build the loaded list as a reusable widget so the shapes card can use it in phase 6, including its warning mapping and `set_styled_source_warning(...)`. Change the labels status-card builders to take the labels name and spec, and delete `LabelsLoadRequest`.
5. **Table refresh**: scoped restyle on `table_state_changed` with the freshly classified spec, and stale-source warnings.
6. **Shapes card**: reuse the loaded list from phase 4, and add shapes-column sources and per-row Fill, shown only for layers rendered as `Shapes`. Change the shapes status-card builders to take the shapes name and spec, and delete `ShapesLoadRequest`.
7. **Shapes geometry writes**: `rebuild_styled_shapes_layer(...)` in the adapter, and the rebuild of live styled shapes layers in `_on_shapes_element_written(...)` / `_on_shapes_element_reloaded(...)`. It is a separate slice because it fixes a gap that exists today, independent of the new card, but it ships in the same PR. It comes after phase 6, which provides the fresh shapes-column sources and the per-row Fill it relies on; the coloring-time classification comes from phase 1.

Release phases 4 to 7 together. Between them, the labels card loads immediately while the shapes card still uses **Add / Update in viewer**. That is fine on the feature branch, but confusing in a release.

## Tests

Value classification (new `tests/test_color_value_classification.py`):

- one case per encoding: a pandas categorical column, a boolean column, an `object` column of booleans with missing values, 0/1 integers with and without missing values, other integers, floats, and strings;
- `None` for an all-missing column and for a column with mixed value types;
- discovery (`tests/test_color_source_options.py`) offers the same sources with the same `value_kind` as before.

Labels styling (`tests/test_labels_styling.py`):

- an `object`-dtype boolean column with missing values, and an `object`-dtype 0/1 column, are colored as categorical (today: continuous);
- every other column type is colored as before.

Color source specs (`tests/test_color_source.py`):

- `layer_key` includes `var_layer` and leaves out `value_kind`, for `TableColorSourceSpec`;
- `layer_key` of `ShapeColumnColorSourceSpec` is `(source_kind, value_key)`;
- `identity` is unchanged.

Adapter (`tests/test_viewer_adapter.py`):

- `labels_layers_changed` / `shapes_layers_changed` fire on register, on insertion of an already registered layer, on napari removal, on reorder and on `_remove_layer_from_viewer_and_registry(...)`, for primary and styled layers;
- they fire for bindings registered without a coordinate system;
- `sync_primary_shapes_layer_binding(...)` emits `shapes_layers_changed` but not `primary_shapes_layer_registered`;
- `remove_styled_*_layer(...)` removes only the matching variant and leaves siblings and the primary layer alone;
- a spec that differs only in `value_kind` resolves to the existing styled layer, the binding then carries the new spec, keeps its other fields (for shapes, the rendered-row mapping), and the replacement emits the membership signal;
- `find_bindings(style_spec=...)` matches on `layer_key`;
- `X` and a layer for the same var stay separate layers;
- `ShapesLayerBinding` rejects `fill=True` for a primary binding and for a binding rendered as `Points`;
- `ensure_styled_shapes_loaded(..., fill=True)` records `fill=True` for a layer rendered as `Shapes`, and `False` for a point-radius layer rendered as `Points`;
- restyling with a different `fill` replaces the binding and emits `shapes_layers_changed`; restyling with the same spec and `fill` keeps the binding object and emits nothing;
- `rebuild_styled_shapes_layer(...)` puts a new layer in the old layer's place in the layer list, keeps `visible`, `opacity`, name, spec and `fill`, and emits `shapes_layers_changed`;
- it handles a change of rendering mode from `Shapes` to `Points`, with `fill` normalized to `False`;
- when building, coloring or registering the replacement fails, the old layer and binding stay, and no replacement is left in the viewer.

Shapes styling (`tests/test_shapes_styling.py`):

- a shapes-column layer whose column changed kind after loading is colored with the column's current kind, whatever `value_kind` the spec carries;
- a shapes column that is no longer colorable raises `ValueError`;
- table-backed and shapes-column coloring color every column type as before.

Layer rows:

- new `tests/test_layer_row.py` for `_LayerRow`: intents are emitted without mutating the layer, visibility follows native events, `dispose()` is idempotent, the warning state renders, an extra control can be added and its signals reach the owner, and rows can be built for `Labels`, `Shapes` and `Points` layers;
- the existing `tests/test_image_layer_row.py` keeps passing unchanged.

Viewer widget (`tests/test_viewer_widget.py`):

- accepting a completer value loads the styled layer without any button click, and the row appears;
- a loaded source is left out of the completer and comes back after removal;
- deleting a styled layer in napari removes its row;
- reordering layers in napari reorders the rows;
- "Remove all" removes only styled layers, not the primary layer;
- the primary row switches between **Load in viewer** and a live row when the primary layer is added or removed, including from napari;
- an `obs` table update restyles a loaded layer colored by that column and leaves other layers alone;
- a table rebuild that changes a column's `value_kind` restyles the loaded layer in place: no second layer is created, the binding carries the new spec, and the colors follow the new kind;
- a column that still exists but is no longer colorable, for example one holding only missing values, gets the row warning instead of a restyle;
- a removed source keeps the layer and shows a row warning;
- a warning survives a row rebuild (for example a Fill toggle) and is cleared by a successful restyle;
- the status-card builders produce the same cards from the element name and spec as they did from the request;
- shapes: toggling Fill on a row changes only that layer, and the checkbox follows `binding.fill`;
- shapes: rows of point-radius layers rendered as `Points` have no Fill checkbox;
- shapes: after a save that deletes and adds shapes, a styled shapes layer shows the new geometry, its row stays with the same Fill, and the Viewer does not touch the primary layer;
- shapes: after a save that removes the colored shapes column, the old layer stays and its row shows a warning;
- shapes: a reload rebuilds styled shapes layers in the same way as a save.

About 30 existing references to `add_update_button` in `tests/test_viewer_widget.py` cover the labels and shapes Add / Update flow and need to be rewritten for the new intents.
