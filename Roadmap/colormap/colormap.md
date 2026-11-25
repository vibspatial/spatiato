# Continuous Colorbar And Range Controls

## Motivation

Harpy can already create styled labels and shapes layers by coloring:

- labels elements from a linked table observation column or `X[:, var]`
- shapes elements from a linked table observation column or `X[:, var]`
- shapes elements from a direct shapes GeoDataFrame column

Categorical and instance coloring are discrete and do not need a scalar range control. Continuous coloring currently maps the selected numeric values to RGBA colors immediately and writes those colors directly to the napari layer. Users cannot adjust the effective minimum or maximum after creating the styled layer.

This feature adds an interactive colorbar for continuous styled layers. The user should be able to move the lower or upper range handle and immediately update the existing styled labels or shapes layer.

## Goals

- Show a colorbar/range control when the selected color source is continuous.
- Support continuous labels coloring from linked table `obs` columns.
- Support continuous labels coloring from linked table `X[:, var]`.
- Support continuous shapes coloring from direct shapes columns.
- Support continuous shapes coloring from linked table `obs` columns.
- Support continuous shapes coloring from linked table `X[:, var]`.
- Update the existing styled layer when min/max changes.
- Preserve current behavior for categorical and instance coloring.
- Preserve table-backed shapes transparency rules:
  - annotated rows with missing selected values stay gray
  - source shapes with no linked table row stay transparent
- Keep range state out of layer identity so changing min/max does not create a new styled layer variant.

## Non-Goals

- Full legend system for categorical palettes.
- Palette editor for categorical values.
- Persistent storage of user-selected contrast limits.
- Multiple colormap choices in the first implementation.
- Replacing labels styling with napari-spatialdata coloring.
- Using napari native shape/point colormap mode only for shapes while labels use a separate implementation.

## Current Code Shape

Whether a source is colored continuously is decided when it is colored, not stored on the source:

- `TableColorSourceSpec` and `ShapeColumnColorSourceSpec` only name the data: the table, source kind, key and matrix, or the shapes column. They carry no value kind (see `Roadmap/auto_load_table_coloring/auto_color.md`, "Matching a loaded source").
- The coloring code classifies a column from its current values with `classify_color_values(...)` in `src/spatiato/core/color_value_classification.py`. Vars are always colored as continuous.
- The style result records the kind the coloring applied: `LabelsStyleResult.value_kind` and `ShapesStyleResult.value_kind` are `"continuous"` for a continuous layer. It is returned by `ensure_styled_labels_loaded(...)` / `ensure_styled_shapes_loaded(...)` and is not stored on the layer binding.

A colorbar therefore decides from a loaded layer's style result whether the layer is continuous, not from its spec.

Relevant files:

- `src/spatiato/core/_color_source.py`
- `src/spatiato/core/color_value_classification.py`
- `src/spatiato/core/spatialdata.py`
- `src/spatiato/viewer/_styling.py`
- `src/spatiato/viewer/labels_styling.py`
- `src/spatiato/viewer/shapes_styling.py`
- `src/spatiato/viewer/adapter.py`
- `src/spatiato/widgets/viewer/labels_widget.py`
- `src/spatiato/widgets/viewer/shapes_widget.py`
- `src/spatiato/widgets/viewer/widget.py`

Important current behavior:

- Labels use `DirectLabelColormap`, so there is no native labels contrast-limit control to expose.
- Shapes and point-backed shapes are also currently styled with direct RGBA arrays.
- Continuous RGBA values are generated in `continuous_rgba_for_values(...)`.
- Styled layer identity is keyed by the selected color source spec, not by display settings.

## Proposed Design

### Core Model

Add a small immutable model for continuous range metadata:

```python
@dataclass(frozen=True)
class ContinuousColorScale:
    data_min: float
    data_max: float
    contrast_min: float
    contrast_max: float
    colormap_name: str = OVERLAY_CONTINUOUS_COLORMAP
```

Rules:

- `data_min` and `data_max` describe the non-missing source values used to initialize the control.
- `contrast_min` and `contrast_max` are the currently applied range.
- If all values are missing, no scale is produced and the colorbar remains hidden/disabled.
- If `data_min == data_max`, colors should continue to map present values to the colormap midpoint, matching current constant-value behavior.
- If a user-supplied range is invalid or reversed, clamp/normalize it before applying:
  - lower bound cannot exceed upper bound
  - equal bounds are allowed for constant data and handled as midpoint
  - for non-constant data, equal bounds should be widened minimally or rejected at the widget layer

### Styling Helpers

Extend `continuous_rgba_for_values(...)` and `continuous_colors_for_values(...)` to accept optional contrast limits:

```python
def continuous_rgba_for_values(
    values: pd.Series,
    *,
    missing_color: Any = MISSING_CONTINUOUS_COLOR,
    colormap_name: str = OVERLAY_CONTINUOUS_COLORMAP,
    contrast_limits: tuple[float, float] | None = None,
) -> np.ndarray:
    ...
```

Mapping rules:

- Without `contrast_limits`, preserve current behavior:
  - min/max are computed from non-missing values
  - constant values map to `0.5`
- With `contrast_limits=(low, high)`:
  - values below `low` clip to `0.0`
  - values above `high` clip to `1.0`
  - values between map linearly
  - missing values use the existing missing color

Also add a helper such as:

```python
def continuous_color_scale_for_values(
    values: pd.Series,
    *,
    contrast_limits: tuple[float, float] | None = None,
    colormap_name: str = OVERLAY_CONTINUOUS_COLORMAP,
) -> ContinuousColorScale | None:
    ...
```

This avoids recomputing range metadata independently in labels and shapes code.

### Labels Styling

Thread optional contrast limits through labels styling:

```python
def apply_table_color_source_to_labels_layer(
    layer: Labels,
    *,
    sdata: SpatialData,
    labels_name: str,
    style_spec: TableColorSourceSpec,
    contrast_limits: tuple[float, float] | None = None,
) -> LabelsStyleResult:
    ...
```

Continuous paths:

- `_build_obs_column_colormap(...)`
- `_build_x_var_colormap(...)`
- `_build_continuous_color_dict(...)`

`LabelsStyleResult` should include optional continuous scale metadata:

```python
continuous_color_scale: ContinuousColorScale | None = None
```

For categorical/instance results, this remains `None`.

### Shapes Styling

Thread optional contrast limits through shapes styling:

```python
def apply_shape_column_color_source_to_shapes_layer(..., contrast_limits: tuple[float, float] | None = None)
def apply_table_color_source_to_shapes_layer(..., contrast_limits: tuple[float, float] | None = None)
```

Continuous paths:

- `_build_continuous_shape_style(...)`
- table `x_var` calls to `_build_continuous_shape_style(...)`
- table `obs_column` fallback to `_build_continuous_shape_style(...)`

`ShapesStyleResult` should include optional continuous scale metadata:

```python
continuous_color_scale: ContinuousColorScale | None = None
```

For table-backed shapes, range metadata should be based on rendered/aligned values that participate in coloring. The existing unannotated-row alpha masking must remain after colors are generated.

### Viewer Adapter

Extend adapter methods with optional contrast limits:

```python
def ensure_styled_labels_loaded(..., contrast_limits: tuple[float, float] | None = None) -> LabelsLoadResult

def ensure_styled_shapes_loaded(
    ...,
    fill: bool = False,
    contrast_limits: tuple[float, float] | None = None,
) -> ShapesLoadResult
```

Important:

- Do not include contrast limits in `TableColorSourceSpec` or `ShapeColumnColorSourceSpec`.
- Do not include contrast limits in `LayerBinding` identity matching.
- Calling `ensure_styled_*_loaded(...)` with a new range should find the existing styled layer and repaint it.

### Widget Requests

Extend request dataclasses:

```python
@dataclass(frozen=True)
class LabelsLoadRequest:
    ...
    contrast_limits: tuple[float, float] | None = None

@dataclass(frozen=True)
class ShapesLoadRequest:
    ...
    contrast_limits: tuple[float, float] | None = None
```

The card emits the selected contrast limits only when:

- a continuous source is selected
- the range control is initialized/enabled

Categorical, instance, and primary layer requests should pass `None`.

## UI Specification

### Placement

Add the colorbar inside each labels/shapes detail card, near the color source controls and above the "Add / Update in viewer" button.

The control is hidden unless the selected source is continuous.

### Initial State

Before a continuous styled layer is created:

- The card can show the control disabled with no numeric range, or keep it hidden until the first successful load.
- Preferred first implementation: show a compact disabled "range unavailable" state only after a continuous source is selected, then initialize it after successful layer creation.

After a successful continuous styled layer load:

- Initialize the colorbar from `result.continuous_color_scale`.
- Set handles to `contrast_min` and `contrast_max`.
- Use `data_min` and `data_max` as the slider bounds.

### Interaction

When the user moves the min or max handle:

1. The card updates its current `contrast_limits`.
2. The existing styled layer is repainted through `ensure_styled_*_loaded(...)`.
3. The active layer remains the styled layer.
4. The status card may show that the colored layer was updated.

To avoid excessive repainting:

- Repaint continuously while dragging only if performance is acceptable on typical data.
- Otherwise, debounce range changes or repaint on slider release.
- A first implementation can use `sliderReleased`/editing finished to minimize churn.

### Visual Shape

The widget should include:

- horizontal sampled gradient for the active continuous colormap
- min and max draggable handles or equivalent range slider
- compact numeric readouts for current min and max
- optional reset button to return to full data range

Avoid adding a broad legend/card system. This is a local control for the currently selected continuous source.

### Dependencies

`superqt` is available through napari in the current environment, but it is not a direct dependency of `spatiato`.

Preferred options:

1. Add `superqt` as an explicit dependency and use `QDoubleRangeSlider` or `QLabeledDoubleRangeSlider`.
2. Implement a small local Qt widget using `QWidget`, `QPainter`, and two numeric controls/sliders.

Recommendation: use a small local widget or explicitly add `superqt` if adopting it. Do not rely on transitive availability without declaring it.

## Status Card Behavior

The existing status cards can remain mostly unchanged.

Optional enhancement:

- For continuous results, append a line such as:
  - `Applied continuous range [0.12, 0.98].`

This should only appear when `continuous_color_scale` is present.

## Edge Cases

- All values missing:
  - styled layer uses missing color/transparent behavior as today
  - no active colorbar range is shown
- Constant values:
  - present values map to colormap midpoint
  - range control may show a disabled single-value state
- Partial table-backed shapes:
  - unannotated source rows remain transparent
  - missing table values remain gray
- Sparse `X` values:
  - preserve current sparse extraction behavior
- Source changes after widget refresh:
  - range state should reset when selected color source identity changes
- Repeated layer updates:
  - updating min/max should not create extra layers

## Testing Plan

### Unit Tests

Add tests in `tests/test_styling.py`:

- `continuous_rgba_for_values` preserves current output when `contrast_limits is None`.
- explicit contrast limits clip below/above range.
- missing values remain missing color.
- constant values remain midpoint-colored.
- `continuous_color_scale_for_values` returns expected data and contrast limits.

Add/update tests in `tests/test_shapes_styling.py`:

- direct shapes continuous column applies explicit contrast limits.
- table-backed shapes `obs_column` continuous applies explicit contrast limits.
- table-backed shapes `x_var` continuous applies explicit contrast limits.
- unannotated table-backed shapes remain transparent after range changes.

Add/update tests in `tests/test_viewer_adapter.py`:

- styled labels continuous layer reuses the same variant when contrast limits change.
- styled shapes continuous layer reuses the same variant when contrast limits change.
- result objects expose `continuous_color_scale`.

Add/update tests in `tests/test_viewer_widget.py`:

- labels card exposes/enables colorbar for continuous table source.
- labels card hides/disables colorbar for categorical/instance sources.
- shapes card exposes/enables colorbar for continuous shape/table source.
- changing range emits a styled load request with contrast limits.

### Manual QA

Use a sample SpatialData object with:

- labels annotated by numeric table obs
- labels annotated by `X[:, var]`
- shapes colored by direct numeric column
- shapes colored by numeric table obs
- shapes colored by `X[:, var]`

Verify:

- colorbar appears only for continuous sources
- min/max changes visibly recolor the layer
- no duplicate styled layers are created
- categorical and instance coloring still behave as before
- point-backed shapes preserve size/symbol behavior

## Implementation Phases

### Phase 1: Range-Aware Styling Core

- Add `ContinuousColorScale`.
- Add contrast-limit support to continuous RGBA helpers.
- Thread contrast limits through labels and shapes styling functions.
- Add result metadata.
- Add unit tests for color mapping and style outputs.

### Phase 2: Adapter Plumbing

- Add optional contrast limits to styled labels/shapes adapter methods.
- Ensure existing layer lookup remains source-spec based.
- Add adapter tests for same-layer updates.

### Phase 3: Widget UI

- Add reusable continuous colorbar/range widget.
- Add colorbar to labels and shapes cards.
- Extend load request dataclasses.
- Initialize/update colorbar from successful styled load results.
- Add widget tests.

### Phase 4: Polish

- Add optional status-card range line.
- Tune drag/debounce behavior.
- Validate rendering on realistic labels and shapes data.
