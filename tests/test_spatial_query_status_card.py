import pytest

from spatiato.core.spatial_query import SpatialAnnotationSummary
from spatiato.core.spatial_query.canonical_centers import CanonicalCentersStatus
from spatiato.widgets.spatial_query.status_card import (
    build_spatial_annotation_failure_status_card_spec,
    build_spatial_annotation_outcome_status_card_spec,
    build_spatial_query_controller_status_card_spec,
    build_spatial_query_status_card_spec,
)


def _configuration_spec(
    centers_status: CanonicalCentersStatus,
    *,
    waiting_for_centers_job: bool = False,
    table_busy_message: str | None = None,
):
    return build_spatial_query_status_card_spec(
        has_spatialdata=True,
        coordinate_system="global",
        saved_shapes_name="regions",
        has_unsaved_shapes_changes=False,
        labels_name="cells",
        table_name="table",
        centers_status=centers_status,
        annotation_column_error=None,
        annotation_column_description='New column "spatial_annotation"',
        annotation_mutation_error=None,
        annotation_mutation_description='Action: Set annotation to "tumor".',
        layer_styling_error=None,
        waiting_for_centers_job=waiting_for_centers_job,
        table_busy_message=table_busy_message,
    )


@pytest.mark.parametrize(
    ("centers_status", "expected_kind", "expected_centers_line"),
    [
        (CanonicalCentersStatus("table", "valid"), "success", 'Centers for "cells" will be reused.'),
        (
            CanonicalCentersStatus("table", "absent"),
            "info",
            'Centers for every labels element of table "table" will be calculated before querying.',
        ),
        (
            CanonicalCentersStatus("table", "invalid", "harpy's technical reason"),
            "info",
            'Centers for every labels element of table "table" will be recalculated before querying.',
        ),
    ],
    ids=["valid", "absent", "invalid"],
)
def test_spatial_query_status_describes_the_three_canonical_centers_states(
    centers_status: CanonicalCentersStatus,
    expected_kind: str,
    expected_centers_line: str,
) -> None:
    spec = _configuration_spec(centers_status)

    visible_text = "\n".join(spec.lines)
    assert spec.source == "configuration"
    assert spec.title == "Spatial Query Ready"
    assert spec.kind == expected_kind
    assert 'Shapes "regions" will query labels "cells".' in visible_text
    assert 'Target: New column "spatial_annotation".' in visible_text
    assert 'Action: Set annotation to "tumor".' in visible_text
    assert expected_centers_line in visible_text
    # harpy's reason is the tooltip of invalid centers, never the visible text.
    assert "harpy's technical reason" not in visible_text
    if centers_status.state == "invalid":
        assert spec.tooltip_message == "harpy's technical reason"


def test_spatial_query_status_says_a_detached_centers_job_is_still_writing() -> None:
    spec = _configuration_spec(CanonicalCentersStatus("table", "absent"), waiting_for_centers_job=True)

    assert spec.title == "Center Calculation Still Writing"
    assert spec.lines == ("Waiting for the previous center calculation to finish writing.",)


def test_spatial_query_status_says_another_job_keeps_the_table_busy() -> None:
    message = 'Table "table" is busy: Feature Extraction is still writing its results.'

    spec = _configuration_spec(CanonicalCentersStatus("table", "valid"), table_busy_message=message)

    assert spec.title == "Table Busy"
    assert spec.lines == (message,)


@pytest.mark.parametrize(
    ("is_running", "kind", "expected_title"),
    [
        (True, "info", "Spatial Query Running"),
        (False, "success", "Spatial Query Complete"),
        (False, "error", "Spatial Query Failed"),
    ],
)
def test_spatial_query_controller_status_uses_the_unified_card(
    is_running: bool,
    kind: str,
    expected_title: str,
) -> None:
    spec = build_spatial_query_controller_status_card_spec(
        message="current operation state",
        kind=kind,
        is_running=is_running,
    )

    assert spec.source == "controller"
    assert spec.title == expected_title
    assert spec.lines == ("current operation state",)
    assert spec.kind == kind


@pytest.mark.parametrize(
    ("summary", "expected_title", "expected_text"),
    [
        (
            SpatialAnnotationSummary(
                annotation_value="tumor",
                matched_count=4,
                current_missing_count=1,
                current_equal_count=1,
                current_other_count=2,
            ),
            "Annotation Applied",
            'Applied "tumor" to 3 matched labeled objects.',
        ),
        (
            SpatialAnnotationSummary(
                annotation_value=None,
                matched_count=4,
                current_missing_count=4,
                current_equal_count=0,
                current_other_count=0,
            ),
            "No Annotation Changes",
            "Already missing: 4.",
        ),
    ],
)
def test_spatial_annotation_outcome_status_reports_exact_summary_counts(
    summary: SpatialAnnotationSummary,
    expected_title: str,
    expected_text: str,
) -> None:
    spec = build_spatial_annotation_outcome_status_card_spec(summary)

    assert spec.source == "annotation_outcome"
    assert spec.title == expected_title
    assert expected_text in "\n".join(spec.lines)


def test_spatial_annotation_failure_status_is_actionable() -> None:
    spec = build_spatial_annotation_failure_status_card_spec("The prepared values changed.")

    assert spec.source == "annotation_outcome"
    assert spec.title == "Annotation Failed"
    assert spec.kind == "error"
    assert "apply again" in spec.lines[1]
