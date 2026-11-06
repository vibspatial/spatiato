"""Pure status-card specifications for the Spatial Query child."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from spatiato.core.spatial_query import SpatialAnnotationSummary
from spatiato.core.spatial_query.canonical_centers import CanonicalCentersStatus
from spatiato.widgets.shared_styles import StatusCardKind, validate_status_card_kind

_CENTER_CALCULATION_TOOLTIP = (
    "Canonical centers are calculated for every labels element the table annotates, "
    "and the table is busy until the calculation has finished."
)

type _StatusCardSource = Literal["configuration", "controller", "annotation_outcome"]


@dataclass(frozen=True)
class _SpatialQueryStatusCardSpec:
    title: str
    lines: tuple[str, ...]
    kind: StatusCardKind
    source: _StatusCardSource = "configuration"
    tooltip_message: str | None = None

    def __post_init__(self) -> None:
        if self.source not in ("configuration", "controller", "annotation_outcome"):
            raise ValueError(f"Unsupported Spatial Query status-card source: {self.source!r}.")
        validate_status_card_kind(self.kind)


def build_spatial_query_controller_status_card_spec(
    *,
    message: str,
    kind: StatusCardKind,
    is_running: bool,
) -> _SpatialQueryStatusCardSpec:
    """Build the current or final controller-owned status card."""
    if not isinstance(message, str) or not message:
        raise ValueError("Spatial Query execution status requires a non-empty message.")
    if is_running:
        title = "Spatial Query Running"
    elif kind == "error":
        title = "Spatial Query Failed"
    else:
        title = "Spatial Query Complete"
    return _SpatialQueryStatusCardSpec(
        title=title,
        lines=(message,),
        kind=kind,
        source="controller",
    )


def build_spatial_annotation_outcome_status_card_spec(
    summary: SpatialAnnotationSummary,
    *,
    layer_styling_error: str | None = None,
) -> _SpatialQueryStatusCardSpec:
    """Build the final status for one accepted annotation Apply."""
    if not isinstance(summary, SpatialAnnotationSummary):
        raise TypeError("Spatial annotation outcome status requires a SpatialAnnotationSummary.")

    if summary.changed_count == 0:
        if summary.is_removal:
            lines = (
                f"No annotations needed removal across {summary.matched_count} matched labeled objects.",
                f"Already missing: {summary.unchanged_count}.",
            )
        else:
            lines = (
                f"All {summary.matched_count} matched labeled objects already have "
                f"{_format_annotation_value(summary.annotation_value)}.",
                f"Already equal: {summary.unchanged_count}.",
            )
        return _SpatialQueryStatusCardSpec(
            title="No Annotation Changes",
            lines=lines,
            kind="info",
            source="annotation_outcome",
        )

    if summary.is_removal:
        lines = (
            f"Removed annotations from {summary.removal_count} matched labeled objects.",
            f"Already missing: {summary.unchanged_count}.",
        )
    else:
        lines = (
            f"Applied {_format_annotation_value(summary.annotation_value)} "
            f"to {summary.changed_count} matched labeled objects.",
            f"Overwritten: {summary.overwrite_count}. Already equal: {summary.unchanged_count}.",
        )

    if layer_styling_error is not None:
        lines = (*lines, f"Labels styling could not be refreshed: {layer_styling_error}")
    return _SpatialQueryStatusCardSpec(
        title="Annotation Applied",
        lines=lines,
        kind="warning" if layer_styling_error is not None else "success",
        source="annotation_outcome",
    )


def build_spatial_annotation_failure_status_card_spec(
    error: str,
) -> _SpatialQueryStatusCardSpec:
    """Build the final status for a rejected annotation Apply."""
    if not isinstance(error, str) or not error:
        raise ValueError("Spatial annotation failure status requires a non-empty error.")
    return _SpatialQueryStatusCardSpec(
        title="Annotation Failed",
        lines=(error, "Review the current annotation inputs and apply again."),
        kind="error",
        source="annotation_outcome",
    )


def build_spatial_query_status_card_spec(
    *,
    has_spatialdata: bool,
    coordinate_system: str | None,
    saved_shapes_name: str | None,
    has_unsaved_shapes_changes: bool,
    labels_name: str | None,
    table_name: str | None,
    centers_status: CanonicalCentersStatus | None,
    annotation_column_error: str | None,
    annotation_column_description: str | None,
    annotation_mutation_error: str | None,
    annotation_mutation_description: str | None,
    layer_styling_error: str | None,
    waiting_for_centers_job: bool = False,
    table_busy_message: str | None = None,
) -> _SpatialQueryStatusCardSpec:
    """Build the unified Spatial Query status from already-derived child state.

    `waiting_for_centers_job` is set while a detached centers job still
    writes, without a Run: a new Run waits for it. `table_busy_message` is set
    while another job writes the selected table: a new Run waits for it too.
    """
    if not has_spatialdata:
        return _SpatialQueryStatusCardSpec(
            title="No SpatialData Loaded",
            lines=("Load a SpatialData object before configuring Spatial Query.",),
            kind="warning",
        )
    if coordinate_system is None:
        return _SpatialQueryStatusCardSpec(
            title="Coordinate System Required",
            lines=("Choose a coordinate system in the Annotation widget.",),
            kind="warning",
        )
    if saved_shapes_name is None:
        return _SpatialQueryStatusCardSpec(
            title="Saved Shapes Required",
            lines=("Select an existing Shapes element or save the new Shapes annotation first.",),
            kind="warning",
        )
    if has_unsaved_shapes_changes:
        return _SpatialQueryStatusCardSpec(
            title="Save or Discard Shapes Changes",
            lines=("Spatial Query uses saved in-memory geometry and cannot run while the selected Shapes is dirty.",),
            kind="warning",
        )
    if labels_name is None:
        return _SpatialQueryStatusCardSpec(
            title="Labels Required",
            lines=("Choose a supported 2D labels element.",),
            kind="warning",
        )
    if table_name is None:
        return _SpatialQueryStatusCardSpec(
            title="Linked Table Required",
            lines=(f'No linked table is selected for "{labels_name}".',),
            kind="warning",
        )
    if centers_status is None:
        raise ValueError("A complete labels and table selection requires the canonical centers status.")
    if annotation_column_error is not None:
        return _SpatialQueryStatusCardSpec(
            title="Annotation Column Not Ready",
            lines=(annotation_column_error,),
            kind="warning",
        )
    if annotation_mutation_error is not None:
        return _SpatialQueryStatusCardSpec(
            title="Annotation Value Required",
            lines=(annotation_mutation_error,),
            kind="warning",
        )
    if waiting_for_centers_job:
        return _SpatialQueryStatusCardSpec(
            title="Center Calculation Still Writing",
            lines=("Waiting for the previous center calculation to finish writing.",),
            kind="info",
        )
    if table_busy_message is not None:
        return _SpatialQueryStatusCardSpec(
            title="Table Busy",
            lines=(table_busy_message,),
            kind="info",
        )
    if layer_styling_error is not None:
        return _SpatialQueryStatusCardSpec(
            title="Layer Styling Warning",
            lines=(layer_styling_error, "Spatial Query can still run."),
            kind="warning",
        )

    centers_line, kind, tooltip_message = _build_ready_centers_status(
        centers_status,
        labels_name,
        table_name,
    )
    return _SpatialQueryStatusCardSpec(
        title="Spatial Query Ready",
        lines=(
            f'Shapes "{saved_shapes_name}" will query labels "{labels_name}".',
            f"Target: {annotation_column_description or 'unknown annotation column'}.",
            annotation_mutation_description or "Annotation action is not configured.",
            centers_line,
        ),
        kind=kind,
        tooltip_message=tooltip_message,
    )


def _build_ready_centers_status(
    status: CanonicalCentersStatus,
    labels_name: str,
    table_name: str,
) -> tuple[str, StatusCardKind, str | None]:
    if status.state == "valid":
        return f'Centers for "{labels_name}" will be reused.', "success", None
    if status.state == "absent":
        return (
            f'Centers for every labels element of table "{table_name}" will be calculated before querying.',
            "info",
            _CENTER_CALCULATION_TOOLTIP,
        )
    # Invalid: harpy's reason, also for an older record, explains why.
    return (
        f'Centers for every labels element of table "{table_name}" will be recalculated before querying.',
        "info",
        status.reason,
    )


def _format_annotation_value(value: object) -> str:
    if isinstance(value, str):
        return f'"{value}"'
    return str(value)
