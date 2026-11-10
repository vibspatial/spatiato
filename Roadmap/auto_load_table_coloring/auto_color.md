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

Loading keeps today's semantics: `ensure_labels_loaded(...)` / `ensure_shapes_loaded(...)` return an existing primary layer unchanged.

### Adapter changes (`src/spatiato/viewer/adapter.py`)

1. **Membership signals.** Add two payload-free invalidation signals:

   ```python
   # Emitted when the set/order of live labels layers (primary or styled) changes.
   labels_layers_changed = Signal()
   # Emitted when the set/order of live shapes layers (primary or styled) changes.
   shapes_layers_changed = Signal()
   ```

   Emit them at the same points as `image_layers_changed`: `_handle_registered_binding(...)`, `_on_viewer_layer_removed(...)`, `_on_viewer_layers_reordered(...)` and `_remove_layer_from_viewer_and_registry(...)`. Like `image_layers_changed`, they carry no payload; consumers re-query the registry. The existing `primary_labels_layers_changed` and `primary_shapes_layer_registered` stay as they are for their current consumers.

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

3. **Matching key for a styled source.** See "Matching a loaded source" below. `get_loaded_styled_labels_layer(...)` and `get_loaded_styled_shapes_layer(...)` should match on that key instead of full spec equality.

### Shared layer row (`src/spatiato/widgets/image_layer_row.py`)

Generalize `_ImageLayerRow` so its binding may be any `BaseLayerBinding`. The colormap button stays image-only through `show_colormap`. Renaming it to `_LayerRow` is optional; the Histogram widget is the other user.

Shapes rows get a per-row **Fill** checkbox (see "Shapes Fill" below). This can be an optional extra control on the shared row or a small subclass.

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

The card no longer emits `LabelsLoadRequest`. The status-card builders in `src/spatiato/widgets/viewer/status_card.py` (`build_primary_labels_loaded_card_spec(...)`, `build_styled_labels_card_spec(...)`) still take one; either build it in `ViewerWidget` from the intent, or change the builders to take the labels name and spec directly.

### Shapes card (`src/spatiato/widgets/viewer/shapes_widget.py`)

Same structure as the labels card. The differences are:

- sources can also be `ShapeColumnColorSourceSpec`;
- each styled row has a **Fill** checkbox (see below).

The card no longer emits `ShapesLoadRequest`. As for labels, `build_primary_shapes_loaded_card_spec(...)` and `build_styled_shapes_card_spec(...)` still take one, so build it in `ViewerWidget` or change the builders' arguments.

### Viewer widget (`src/spatiato/widgets/viewer/widget.py`)

Mirror the image handlers:

- `_get_live_labels_bindings(...)` / `_get_live_shapes_bindings(...)`: partition the live bindings of one element in the active `sdata` and coordinate system into the primary binding and the styled bindings, in viewer order. Raise `ValueError` for duplicates.
- `_refresh_labels_card_membership(card)` / `_refresh_shapes_card_membership(card)`: call the reconcile method, or report a card-scoped membership error.
- Connect `labels_layers_changed` / `shapes_layers_changed` to refresh every labels / shapes card. Refresh each card once when it is built in `_rebuild_labels_cards(...)` / `_rebuild_shapes_cards(...)`.
- Add handlers: when the source is already live, reconcile and activate its layer. Otherwise call `ensure_styled_labels_loaded(...)` / `ensure_styled_shapes_loaded(...)`, activate the new layer and show the existing status card (`build_styled_labels_card_spec(...)` / `build_styled_shapes_card_spec(...)`).
- Add remove, remove-all and visibility handlers. They resolve the live binding before mutating, as `_remove_image_overlay_channel(...)` and `_change_image_overlay_channel_visibility(...)` do.
- Add primary load, remove and visibility handlers, reusing `_add_or_update_primary_labels_layer(...)` / `_add_or_update_primary_shapes_layer(...)` for loading.

### Refresh after table changes

Today a user restyles a loaded layer after a table change by clicking **Add / Update** again. Once a loaded source is no longer offered in the completer, that route is gone, so the Viewer restyles loaded layers itself.

Proposed rule, in `ViewerWidget._on_table_state_changed(...)`:

- `change_kind` `"updated"` or `"created"`: restyle each loaded styled layer whose spec is an `obs_column` source of `event.table_name` whose column has a matching `TableComponentPath("obs", (column,))` in `event.paths`. Var sources are not restyled for these events: `TableComponentPath` only covers `obs`, `obsm` and `uns`, so the app never reports `X` or `layers` changes this way.
- `change_kind` `"rebuilt"` or `"reloaded"`: restyle every loaded styled layer that reads `event.table_name`.
- `change_kind` `"removed"`, or a restyle that raises `ValueError` because the table or column is gone: keep the layer as it is and show a warning on its row ("source no longer available"). The user removes it explicitly.

Restyling calls `ensure_styled_*_loaded(...)` with the loaded spec. For shapes it passes the row's current Fill state.

The early return for `object_classification_annotation` / `"updated"` events in `_on_table_state_changed(...)` exists so that rebuilding the linked-table controls does not slow down annotation. It should keep skipping that rebuild, but the scoped restyle above still has to run, so that a loaded layer colored by `user_class` follows the annotations. Because the restyle is scoped to matching paths, it costs nothing unless such a layer is loaded.

### Matching a loaded source

The card (to filter the completer and find rows) and the adapter (to find an existing layer) must agree on when two specs mean the same styled layer. Neither existing comparison fits:

- `TableColorSourceSpec.identity` leaves out `var_layer`, but `vars (X): CD3` and `vars (counts): CD3` are separate layers that can both be loaded.
- Full dataclass equality includes `value_kind`. If a table update changes how a column is classified, for example from continuous to categorical, the new spec no longer equals the loaded one, and loading it creates a second layer for the same column.

Proposed key:

- `TableColorSourceSpec`: `(table_name, source_kind, value_key, var_layer)`;
- `ShapeColumnColorSourceSpec`: `(source_kind, value_key)`.

A source whose `value_kind` changed then restyles the existing layer. The binding's `style_spec` is replaced with the new spec when that happens.

### Shapes Fill

**Fill** moves from the request to the loaded row:

- the checkbox on a styled row toggles fill on that layer only;
- the first implementation calls `ensure_styled_shapes_loaded(..., fill=...)`, which already restyles an existing layer in place;
- new rows start unfilled, as today.

The row needs the current fill state to render its checkbox. Store the requested fill state per loaded source in the card, unless a reliable way to read it back from the layer exists.

## Load Cost

Each accepted selection runs one synchronous load. That is the same work a single **Add / Update** click does today; it is just triggered sooner.

- Labels: the layer itself is lazy. Styling reads one `obs` column or one matrix column (`read_matrix_column(...)`) and builds a compact colormap. This is cheap for in-memory tables. A lazily backed feature matrix may hit disk.
- Shapes: each new styled layer builds a full napari layer through `_build_shapes_layer(...)`. Point-radius shapes become a `Points` layer and load fast. Large polygon sets can take seconds per selection.

If shapes loading feels too slow with immediate loading, a busy indicator on the card is the first mitigation. Background loading is out of scope here.

## Open Questions

1. **Refresh after table changes**: automatic scoped restyle (proposed above), or a refresh button on each row?
2. **Missing source**: keep the layer and warn (proposed), or remove the layer automatically when its table or column disappears?
3. **Row labels with several linked tables**: prefix the table name (proposed), or group rows under a table heading?
4. **Shared row**: generalize `_ImageLayerRow` in place, or rename it to `_LayerRow` and move it to a neutral module?
5. **Primary row placement**: above the color-source controls (proposed), or as the first row of the loaded list?

## Implementation Phases

1. **Adapter**: the `labels_layers_changed` / `shapes_layers_changed` signals, `remove_styled_labels_layer(...)` / `remove_styled_shapes_layer(...)`, and the styled-source matching key.
2. **Shared row**: generalize `_ImageLayerRow` to any binding.
3. **Labels card**: primary row, completer-driven add, loaded list, reconcile method, and the `ViewerWidget` handlers and membership refresh.
4. **Table refresh**: scoped restyle on `table_state_changed`, and stale-source warnings.
5. **Shapes card**: same as labels, plus shapes-column sources and per-row Fill.

## Tests

Adapter (`tests/test_viewer_adapter.py`):

- `labels_layers_changed` / `shapes_layers_changed` fire on register, on napari removal, on reorder and on `_remove_layer_from_viewer_and_registry(...)`, for primary and styled layers;
- `remove_styled_*_layer(...)` removes only the matching variant and leaves siblings and the primary layer alone;
- a spec that differs only in `value_kind` resolves to the existing styled layer;
- `X` and a layer for the same var stay separate layers.

Viewer widget (`tests/test_viewer_widget.py`):

- accepting a completer value loads the styled layer without any button click, and the row appears;
- a loaded source is left out of the completer and comes back after removal;
- deleting a styled layer in napari removes its row;
- reordering layers in napari reorders the rows;
- "Remove all" removes only styled layers, not the primary layer;
- the primary row switches between **Load in viewer** and a live row when the primary layer is added or removed, including from napari;
- an `obs` table update restyles a loaded layer colored by that column and leaves other layers alone;
- a removed source keeps the layer and shows a row warning;
- shapes: toggling Fill on a row changes only that layer.

About 30 existing references to `add_update_button` in `tests/test_viewer_widget.py` cover the labels and shapes Add / Update flow and need to be rewritten for the new intents.
