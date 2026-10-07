# Distinguish Labels Instances Missing From the Annotation Table

Status: spec settled; ready for implementation.

## Goal

An AnnData table can annotate only a subset of the instances in a labels
element. In the Object Classification widget, the instances that have no table
row currently look exactly like table rows that are still waiting for a
`user_class`. The user only finds out that an object cannot be annotated after
clicking it.

Instances that are absent from the bound table should be visually distinct from
instances that are present but unlabeled, so the annotatable set is visible at a
glance.

## Current Behavior

### Coloring

`ViewerStylingController.refresh_layer_colors(...)`
(`src/spatiato/widgets/object_classification/viewer_styling.py`) builds the
primary labels colormap from `_get_region_feature_rows()`. Those rows are the
table rows for the selected labels element, indexed by instance id. Labels ids
that have no table row are not in that frame at all.

The compact colormap builders (`src/spatiato/viewer/labels_colormap.py`) already
separate two fallback colors:

- `missing_color`: known table rows whose value is missing or not in the
  palette;
- `default_color`: label ids that are not in the passed values, i.e. instances
  without a table row. Its builder default is transparent.

Object Classification overrides `default_color` with the same neutral color it
uses for `missing_color`, so all states below collapse into one:

| Instance state | Colormap slot | Color |
|---|---|---|
| In table, `user_class` / `pred_class` missing | `missing_color` (default `MISSING_CATEGORICAL_COLOR`) | `DEFAULT_NEUTRAL_COLOR` |
| In table, `pred_confidence` missing | `missing_color=MISSING_CONTINUOUS_COLOR` | `DEFAULT_NEUTRAL_COLOR` |
| **Not in table** (all `Color by` modes) | `default_color=DEFAULT_NEUTRAL_COLOR` / `MISSING_CONTINUOUS_COLOR` | `DEFAULT_NEUTRAL_COLOR` |
| No table selected | every label falls through to `default_color` | `DEFAULT_NEUTRAL_COLOR` |
| Table binding error | `apply_neutral_labels_style(...)` | `DEFAULT_NEUTRAL_COLOR` |

`DEFAULT_NEUTRAL_COLOR` is `#DCE8F2CC` (`src/spatiato/core/class_palette.py`),
and both `MISSING_CATEGORICAL_COLOR` and `MISSING_CONTINUOUS_COLOR` alias it
(`src/spatiato/viewer/_styling.py`).

The generic table-backed styling path (`apply_table_color_source_to_labels_layer`
in `src/spatiato/viewer/labels_styling.py`, used by the viewer widget's styled
labels layers and by the spatial query widget on the same primary labels layer)
keeps the builder's transparent `default_color`. There, instances without a
table row are already invisible.

### Picking and annotation

Picking an instance that has no table row is allowed and is handled safely:

- `AnnotationController._on_layer_mouse_pick(...)` selects the instance as
  usual;
- the selection status card switches to "Selection Warning" with
  `missing_table_row_message` ("... is not present in annotation table ... and
  cannot receive a user class.");
- `AnnotationController.can_annotate` is `False`, so `Add (A)` is disabled;
- if `_set_current_class(...)` is reached anyway (e.g. via the shortcut), the
  table is not mutated and the warning message is returned as feedback.

No table state can be corrupted. The problem is only discoverability: the
annotatable set is not visible until objects are clicked one by one.

## Proposed Behavior

Render three visual states in the Object Classification primary labels layer:

| Instance state | Color |
|---|---|
| In table, with a class / finite confidence | class palette / `plasma` (unchanged) |
| In table, value missing | `DEFAULT_NEUTRAL_COLOR` (unchanged) |
| Not in table | fully transparent |

Rules:

- render not-in-table instances fully transparent in all three `Color by` modes
  (`user_class`, `pred_class`, `pred_confidence`); an instance without a table
  row can have neither a user class nor a prediction;
- this matches the generic styled-labels path and the spatial query widget, so
  "no table row" means "not drawn" consistently across widgets;
- keep the "no table selected" and "table binding error" states fully neutral;
  without a bound table there is no meaningful "in table" set to contrast
  against, and a fully transparent layer would look like a loading failure;
- keep picking, when the selection warning card is shown, and the disabled
  `Add (A)` button as they are; only the warning wording is extended (see
  Implementation Notes, step 3).

Transparent instances remain pickable: napari resolves the picked label id from
the labels data, independent of the colormap. Clicking an apparently empty area
can therefore select an invisible not-in-table instance and show the selection
warning. This is intended: the warning explains why nothing can be annotated
there, and its extended wording links it to the empty spot.

## Implementation Notes

### 1. Full refresh path

In `ViewerStylingController.refresh_layer_colors(...)`:

- categorical branch: stop passing `default_color=DEFAULT_NEUTRAL_COLOR`, so the
  builder's transparent default applies, and pass
  `missing_color=DEFAULT_NEUTRAL_COLOR` explicitly instead of relying on the
  builder default;
- continuous (`pred_confidence`) branch: stop passing
  `default_color=MISSING_CONTINUOUS_COLOR`, keep
  `missing_color=MISSING_CONTINUOUS_COLOR`;
- when no table is bound (`_get_bound_table()` is `None` or there is no table
  metadata), keep every foreground label neutral, e.g. by passing
  `default_color=DEFAULT_NEUTRAL_COLOR` in that case. Today
  `_refresh_layer_styling()` still calls `refresh()` in that state, with an
  empty feature frame, so without this guard the whole layer would become
  invisible.

No new color constant is needed.

### 2. Row-scoped clear path (main catch)

Clearing a class (`Remove (R)`) goes through
`_refresh_compact_user_class_colormap_and_feature(...)`, which calls
`CompactLabelColormap.remove_label(instance_id)`. That deletes the instance from
the compact mapping so it falls through to the **default** texture code. The
docstring states this explicitly: "Removed labels fall through to the
default/unmapped texture code, which is how compact user-class coloring
represents unlabeled class `0`."

This only looks right today because the default and missing colors are equal.
After the change, clearing a class would make the object disappear until the
next full refresh, even though it still has a table row.

Required change:

- add a sparse operation on `CompactLabelColormap` that maps one label to the
  **missing** texture code instead of removing it, e.g.
  `set_label_missing(label_id, *, missing_color)`;
- if `CompactLabelsMapping.missing_texture_code` is `None` (every table row had
  a class when the colormap was built), append one RGBA row for
  `missing_color`, record it as `missing_texture_code`, and report
  `texture_table_changed=True` so the caller emits `layer.events.colormap()`;
- call it from `_refresh_compact_user_class_colormap_and_feature(...)` when
  `change.class_id is None`;
- remove `CompactLabelColormap.remove_label(...)` and
  `_compact_mapping_without_label(...)`; after this change they have no
  callers.

Adding a class to a not-in-table instance is not reachable, because
`_set_current_class(...)` refuses before any styling update.

### 3. Selection warning wording

Extend `_SelectionTableState.missing_table_row_message`
(`src/spatiato/widgets/object_classification/annotation_controller.py`) with a
short sentence such as "It is not shown in the viewer." The full message then
reads: "Selected `<instance_key>` `<id>` is not present in annotation table
"`<table>`" for labels element "`<labels>`" and cannot receive a user class. It
is not shown in the viewer."

This changes only the wording. The message is still shown in the same cases:
in the selection card, and as annotation feedback when `_set_current_class(...)`
refuses.

## Tests

`tests/test_viewer_styling.py`:

- a labels id with no table row maps to transparent in `user_class`,
  `pred_class`, and `pred_confidence` modes;
- in-table rows with a missing value still map to `DEFAULT_NEUTRAL_COLOR`
  (existing assertions, e.g. `test_refresh_reuses_one_region_feature_snapshot`,
  already cover this);
- with no table bound, every foreground label maps to `DEFAULT_NEUTRAL_COLOR`;
- clearing a class through the row-scoped path maps the instance to
  `DEFAULT_NEUTRAL_COLOR`, not transparent;
- the same clear works when every row was labeled at build time
  (`missing_texture_code is None`), and triggers `layer.events.colormap()`.

`tests/test_labels_colormap.py`:

- unit tests for the new set-missing sparse operation (existing missing texture
  code reused; missing row appended when absent);
- remove
  `test_compact_categorical_label_colormap_sparse_remove_uses_default_texture`
  together with `remove_label(...)`.

`tests/test_object_classification_widget.py`:

- with a table covering a subset of instances, annotate and then clear an
  in-table instance; it ends neutral, while a not-in-table instance stays
  transparent;
- picking a not-in-table instance still shows the selection warning and keeps
  `Add (A)` disabled;
- the existing missing-row test (around the `"Selected cell_id 5 is not present
  in annotation table"` assertions) also asserts the new "It is not shown in the
  viewer." sentence in both `selection_status` and `annotation_feedback`. The
  existing assertions check substrings, so they keep passing unchanged.

Useful verification commands:

- `.venv/bin/pytest tests/test_labels_colormap.py tests/test_viewer_styling.py tests/test_object_classification_widget.py`
- `.venv/bin/pre-commit run ruff --all-files`

## Non-Goals

- Do not add table rows for instances that are missing from the table.
- Do not change picking, when the selection warning card is shown, or
  annotation write rules.
- Do not ignore picks on not-in-table instances; the warning is the feedback
  that explains them.
- Do not change generic styled-labels or spatial query coloring.
- Do not scan or materialize the labels array to find which ids exist; the
  colormap default slot already covers every id without a table row.
- Do not change the meaning of `user_class`, `pred_class`, or `pred_confidence`
  colors.
- Do not add a permanent hint about hidden instances to the widget in this
  slice.

## Decisions

- **Not-in-table instances are fully transparent**, not a faint color. This
  matches the generic styled-labels path and the spatial query widget; a sparse
  or near-empty layer already signals partial table coverage.
- **No table linked, or the selected table is rejected: every instance stays
  neutral, and the widget panel is unchanged.** "No table linked" means no
  table in the SpatialData annotates the labels element; the table combo
  otherwise auto-selects the first annotating table. The transparent rule
  contrasts instances against a bound table's rows, so it does not apply here,
  and an all-transparent layer right after selecting a labels element would
  look like a loading failure. This matches the existing table-binding-error
  path (`apply_neutral_labels_style(...)`) and the spatial query widget's
  unannotated primary layer. The panel keeps its current no-table behavior:
  the "no annotation table is linked" selection warning, and the disabled
  table combo, `Color by`, class spinbox, `Add (A)`, and `Remove (R)`.
- **`remove_label(...)` is removed.** It only repaints the colormap; clearing
  the class in the table (`clear_current_class()` ->
  `set_user_class_for_rows(table, rows, None)`) is unaffected. Once the clear
  path uses `set_label_missing(...)`, it has no callers, and its remaining
  meaning (make an instance look like it has no table row) only fits deleting
  table rows, which Object Classification never does.
- **`missing_table_row_message` gets "It is not shown in the viewer."**, so a
  warning after clicking an apparently empty spot explains itself.
- **No widget hint about hidden instances for now.** It could not be shown only
  when the table covers a subset of the labels, because detecting that requires
  scanning the labels data; it would be permanent UI text. Revisit if users
  report confusion.
