from __future__ import annotations

import harpy as hp
import numpy as np
import pytest
from qtpy.QtCore import QObject, Signal
from spatialdata import SpatialData

from spatiato._app_state import TableDirtySnapshot, TableReloadRequest
from spatiato.core.feature_extraction import (
    ExistingFeatureMatrix,
    FeatureMatrixCalculation,
    FeatureMatrixRemovedError,
)
from spatiato.core.persistence import TableComponentPath
from spatiato.core.spatialdata import get_table_metadata
from spatiato.widgets.feature_extraction import controller as controller_module
from spatiato.widgets.feature_extraction.controller import (
    FEATURE_EXTRACTION_IDLE_STATUS,
    FeatureExtractionBindingState,
    FeatureExtractionController,
    FeatureExtractionJob,
    FeatureExtractionRequest,
    FeatureExtractionResult,
    FeatureExtractionTriplet,
    _run_feature_extraction_job,
)


class _DeferredWorker(QObject):
    returned = Signal(object)
    errored = Signal(object)
    finished = Signal()

    def __init__(self, result: FeatureExtractionResult | None = None) -> None:
        super().__init__()
        self._result = result
        self.started = False
        self.quit_called = False

    def start(self) -> None:
        self.started = True

    def quit(self) -> None:
        self.quit_called = True

    def emit_returned(self) -> None:
        assert self._result is not None
        # Like superqt's `WorkerBase.run()`: no `returned` after `quit()`.
        if not self.quit_called:
            self.returned.emit(self._result)
        self.finished.emit()

    def emit_errored(self, error: Exception) -> None:
        self.errored.emit(error)
        self.finished.emit()

    def emit_finished(self) -> None:
        self.finished.emit()


def test_feature_extraction_controller_starts_idle() -> None:
    controller = FeatureExtractionController()

    assert controller.status_message == FEATURE_EXTRACTION_IDLE_STATUS
    assert controller.status_kind == "warning"
    assert controller.has_job_in_flight is False
    assert controller.can_calculate is False


def test_feature_extraction_request_maps_create_table_target_to_harpy_parameters() -> None:
    request = FeatureExtractionRequest(
        triplets=(),
        table_name="features_table",
        create_table=True,
        feature_names=("area",),
        feature_key="features",
    )

    assert request.harpy_table_name is None
    assert request.harpy_output_table_name == "features_table"


def test_feature_extraction_request_maps_existing_table_target_to_harpy_parameters() -> None:
    request = FeatureExtractionRequest(
        triplets=(),
        table_name="table",
        create_table=False,
        feature_names=("area",),
        feature_key="features",
    )

    assert request.harpy_table_name == "table"
    assert request.harpy_output_table_name is None


def test_feature_extraction_request_rejects_invalid_table_target_state() -> None:
    with pytest.raises(ValueError, match="requires a table name"):
        FeatureExtractionRequest(
            triplets=(),
            table_name="",
            create_table=True,
            feature_names=("area",),
            feature_key="features",
        )

    with pytest.raises(ValueError, match="Cannot overwrite a feature key while creating a new table"):
        FeatureExtractionRequest(
            triplets=(),
            table_name="features_table",
            create_table=True,
            feature_names=("area",),
            feature_key="features",
            overwrite_feature_key=True,
        )


def test_feature_extraction_binding_state_allows_incomplete_table_target() -> None:
    binding_state = FeatureExtractionBindingState(
        sdata=None,
        triplets=(),
        table_name=None,
        create_table=True,
        feature_names=(),
        feature_key=None,
    )

    assert binding_state.harpy_table_name is None
    assert binding_state.harpy_output_table_name is None


def test_feature_extraction_binding_state_rejects_empty_table_name() -> None:
    with pytest.raises(ValueError, match="binding table name cannot be empty"):
        FeatureExtractionBindingState(
            sdata=None,
            triplets=(),
            table_name=" ",
            create_table=False,
            feature_names=(),
            feature_key=None,
        )


def test_feature_extraction_controller_bind_is_passive_and_ready_for_existing_table(
    sdata_blobs: SpatialData,
) -> None:
    controller = FeatureExtractionController()

    context_changed = controller.bind(
        sdata_blobs,
        "blobs_labels",
        None,
        "table",
        "global",
        ["area", "perimeter"],
        "feature_matrix_1",
    )

    assert context_changed is True
    assert controller.status_message == "Feature extraction: ready to calculate."
    assert controller.status_kind == "success"
    assert controller.has_job_in_flight is False
    assert controller.can_calculate is True


def test_feature_extraction_controller_rejects_spatialdata_invalid_feature_key(
    sdata_blobs: SpatialData,
) -> None:
    controller = FeatureExtractionController()

    controller.bind(
        sdata_blobs,
        "blobs_labels",
        None,
        "table",
        "global",
        ["area"],
        "features tst",
    )

    assert controller.can_calculate is False
    assert controller.status_kind == "warning"
    assert "choose a valid feature matrix key" in controller.status_message
    assert "alphanumeric characters, underscores, dots and hyphens" in controller.status_message
    assert controller._prepare_feature_extraction_job(8, overwrite_feature_key=False) is None


def test_feature_extraction_controller_blocks_when_no_table_is_selected(sdata_blobs: SpatialData) -> None:
    controller = FeatureExtractionController()

    controller.bind(
        sdata_blobs,
        "blobs_labels",
        None,
        None,
        "global",
        ["area"],
        "feature_matrix_1",
    )

    assert controller.can_calculate is False
    assert controller.status_kind == "warning"
    assert (
        controller.status_message
        == "Feature extraction: choose an annotation table linked to the selected labels element."
    )


def test_feature_extraction_controller_bind_is_ready_for_create_table(
    sdata_blobs: SpatialData,
) -> None:
    controller = FeatureExtractionController()

    context_changed = controller.bind(
        sdata_blobs,
        "blobs_labels",
        None,
        "features_table",
        "global",
        ["area", "perimeter"],
        "feature_matrix_1",
        create_table=True,
    )

    job = controller._prepare_feature_extraction_job(8, overwrite_feature_key=False)

    assert context_changed is True
    assert controller.status_message == 'Feature extraction: ready to create table "features_table" and calculate.'
    assert controller.status_kind == "success"
    assert controller.can_calculate is True
    assert isinstance(job, FeatureExtractionJob)
    assert job.request.confirmed_existing_entry is None
    assert job.request.create_table is True
    assert job.request.table_name == "features_table"
    assert job.request.harpy_table_name is None
    assert job.request.harpy_output_table_name == "features_table"


def test_feature_extraction_controller_blocks_create_table_name_collision(
    sdata_blobs: SpatialData,
) -> None:
    controller = FeatureExtractionController()

    controller.bind(
        sdata_blobs,
        "blobs_labels",
        None,
        "table",
        "global",
        ["area"],
        "feature_matrix_1",
        create_table=True,
    )

    assert controller.can_calculate is False
    assert controller.status_kind == "warning"
    assert controller.status_message == (
        'Feature extraction: table "table" already exists. Choose a different table name.'
    )
    assert controller._prepare_feature_extraction_job(8, overwrite_feature_key=False) is None

    controller.bind(
        sdata_blobs,
        "blobs_labels",
        None,
        "TABLE",
        "global",
        ["area"],
        "feature_matrix_1",
        create_table=True,
    )

    assert controller.can_calculate is False
    assert controller.status_kind == "warning"
    assert controller.status_message == (
        'Feature extraction: table "TABLE" already exists. Choose a different table name.'
    )
    assert controller._prepare_feature_extraction_job(8, overwrite_feature_key=False) is None


def test_feature_extraction_controller_blocks_invalid_create_table_name(
    sdata_blobs: SpatialData,
) -> None:
    controller = FeatureExtractionController()

    controller.bind(
        sdata_blobs,
        "blobs_labels",
        None,
        "features table",
        "global",
        ["area"],
        "feature_matrix_1",
        create_table=True,
    )

    assert controller.can_calculate is False
    assert controller.status_kind == "warning"
    assert "choose a valid table name" in controller.status_message
    assert "alphanumeric characters, underscores, dots and hyphens" in controller.status_message
    assert controller._prepare_feature_extraction_job(8, overwrite_feature_key=False) is None


def test_feature_extraction_controller_blocks_create_table_feature_key_overwrite_override(
    sdata_blobs: SpatialData,
) -> None:
    controller = FeatureExtractionController()
    controller.bind(
        sdata_blobs,
        "blobs_labels",
        None,
        "features_table",
        "global",
        ["area"],
        "feature_matrix_1",
        create_table=True,
    )

    launched = controller.calculate(overwrite_feature_key=True)

    assert launched is False
    assert controller.status_kind == "warning"
    assert controller.status_message == (
        "Feature extraction: cannot overwrite a feature key while creating a new table."
    )


def test_feature_extraction_controller_requires_image_for_intensity_features(sdata_blobs: SpatialData) -> None:
    controller = FeatureExtractionController()

    controller.bind(
        sdata_blobs,
        "blobs_labels",
        None,
        "table",
        "global",
        ["mean", "area"],
        "feature_matrix_1",
    )

    assert controller.can_calculate is False
    assert controller.status_kind == "warning"
    assert controller.status_message == "Feature extraction: choose an image before calculating intensity features."


def test_feature_extraction_controller_requires_selected_channels_for_intensity_features(
    sdata_blobs: SpatialData,
) -> None:
    controller = FeatureExtractionController()

    controller.bind(
        sdata_blobs,
        "blobs_labels",
        "blobs_image",
        "table",
        "global",
        ["mean", "area"],
        "feature_matrix_1",
        channels=(),
    )

    assert controller.can_calculate is False
    assert controller.status_kind == "warning"
    assert (
        controller.status_message
        == "Feature extraction: select at least one channel before calculating intensity features."
    )
    assert controller._prepare_feature_extraction_job(8, overwrite_feature_key=False) is None


def test_feature_extraction_controller_requires_coordinate_system(sdata_blobs: SpatialData) -> None:
    controller = FeatureExtractionController()

    controller.bind(
        sdata_blobs,
        "blobs_labels",
        None,
        "table",
        None,
        ["area"],
        "feature_matrix_1",
    )

    assert controller.can_calculate is False
    assert controller.status_kind == "warning"
    assert controller.status_message == "Feature extraction: choose a coordinate system."
    assert controller._prepare_feature_extraction_job(8, overwrite_feature_key=False) is None


def test_feature_extraction_controller_requires_segmentation_mask(sdata_blobs: SpatialData) -> None:
    controller = FeatureExtractionController()

    controller.bind(
        sdata_blobs,
        None,
        None,
        "table",
        "global",
        ["area"],
        "feature_matrix_1",
    )

    assert controller.can_calculate is False
    assert controller.status_kind == "warning"
    assert controller.status_message == "Feature extraction: choose a labels element."
    assert controller._prepare_feature_extraction_job(8, overwrite_feature_key=False) is None


def test_feature_extraction_controller_bind_returns_false_for_unchanged_context(sdata_blobs: SpatialData) -> None:
    controller = FeatureExtractionController()

    first_changed = controller.bind(
        sdata_blobs,
        "blobs_labels",
        None,
        "table",
        "global",
        ["area"],
        "feature_matrix_1",
    )
    second_changed = controller.bind(
        sdata_blobs,
        "blobs_labels",
        None,
        "table",
        "global",
        ["area"],
        "feature_matrix_1",
    )

    assert first_changed is True
    assert second_changed is False


def test_feature_extraction_controller_prepares_immutable_job_payload(sdata_blobs: SpatialData) -> None:
    controller = FeatureExtractionController()
    controller.bind(
        sdata_blobs,
        "blobs_labels",
        "blobs_image",
        "table",
        "global",
        ["mean", "area", "mean"],
        "feature_matrix_1",
    )

    job = controller._prepare_feature_extraction_job(7, overwrite_feature_key=True)

    assert isinstance(job, FeatureExtractionJob)
    assert job.job_id == 7
    assert job.sdata is sdata_blobs
    assert job.labels_names == ("blobs_labels",)
    assert job.image_name == "blobs_image"
    assert job.channels is None
    assert job.table_name == "table"
    assert job.coordinate_system == "global"
    assert job.feature_names == ("mean", "area")
    assert job.feature_key == "feature_matrix_1"
    assert job.overwrite_feature_key is True
    assert job.request.triplets == (
        FeatureExtractionTriplet(
            coordinate_system="global",
            labels_name="blobs_labels",
            image_name="blobs_image",
            channels=None,
        ),
    )


def test_feature_extraction_controller_bind_stores_selected_channels_in_job(sdata_blobs: SpatialData) -> None:
    controller = FeatureExtractionController()
    controller.bind(
        sdata_blobs,
        "blobs_labels",
        "blobs_image",
        "table",
        "global",
        ["mean", "area"],
        "feature_matrix_1",
        channels=("1", "2"),
    )

    job = controller._prepare_feature_extraction_job(8, overwrite_feature_key=False)

    assert isinstance(job, FeatureExtractionJob)
    assert job.channels == ("1", "2")


def test_feature_extraction_controller_bind_rejects_duplicate_selected_channels(sdata_blobs: SpatialData) -> None:
    controller = FeatureExtractionController()

    with pytest.raises(ValueError, match="Duplicate channel selection is not allowed"):
        controller.bind(
            sdata_blobs,
            "blobs_labels",
            "blobs_image",
            "table",
            "global",
            ["mean", "area"],
            "feature_matrix_1",
            channels=("1", "1"),
        )


def test_feature_extraction_controller_morphology_only_job_keeps_channels_none(sdata_blobs: SpatialData) -> None:
    controller = FeatureExtractionController()
    controller.bind(
        sdata_blobs,
        "blobs_labels",
        None,
        "table",
        "global",
        ["area"],
        "feature_matrix_1",
        channels=None,
    )

    job = controller._prepare_feature_extraction_job(9, overwrite_feature_key=False)

    assert isinstance(job, FeatureExtractionJob)
    assert job.image_name is None
    assert job.channels is None


def test_feature_extraction_controller_bind_batch_is_ready_for_multi_target_request(
    sdata_blobs: SpatialData,
) -> None:
    controller = FeatureExtractionController()

    context_changed = controller.bind_batch(
        sdata_blobs,
        (
            FeatureExtractionTriplet(
                coordinate_system="global",
                labels_name="blobs_labels",
                image_name=None,
            ),
            FeatureExtractionTriplet(
                coordinate_system="global",
                labels_name="blobs_multiscale_labels",
                image_name=None,
            ),
        ),
        "table",
        ["area", "perimeter"],
        "feature_matrix_batch",
    )

    assert context_changed is True
    assert controller.status_message == "Feature extraction: ready to calculate."
    assert controller.status_kind == "success"
    assert controller.can_calculate is True


def test_feature_extraction_controller_exposes_binding_state_snapshot(
    sdata_blobs: SpatialData,
) -> None:
    controller = FeatureExtractionController()
    triplets = (
        FeatureExtractionTriplet(
            coordinate_system="global",
            labels_name="blobs_labels",
            image_name="blobs_image",
            channels=("0", "2"),
        ),
    )

    controller.bind_batch(
        sdata_blobs,
        triplets,
        "table",
        ["mean"],
        "feature_matrix_batch",
    )

    binding_state = controller.binding_state

    assert binding_state.sdata is sdata_blobs
    assert binding_state.triplets == triplets
    assert binding_state.table_name == "table"
    assert binding_state.feature_names == ("mean",)
    assert binding_state.feature_key == "feature_matrix_batch"


def test_feature_extraction_controller_bind_batch_rejects_mixed_channel_selections_for_intensity_features(
    sdata_blobs: SpatialData,
) -> None:
    controller = FeatureExtractionController()

    controller.bind_batch(
        sdata_blobs,
        (
            FeatureExtractionTriplet(
                coordinate_system="global",
                labels_name="blobs_labels",
                image_name="blobs_image",
                channels=("0", "1"),
            ),
            FeatureExtractionTriplet(
                coordinate_system="global",
                labels_name="blobs_multiscale_labels",
                image_name="blobs_multiscale_image",
                channels=("2",),
            ),
        ),
        "table",
        ["mean"],
        "feature_matrix_batch",
    )

    assert controller.can_calculate is False
    assert controller.status_kind == "warning"
    assert (
        controller.status_message
        == "Feature extraction: all selected extraction targets must currently use the same channel selection."
    )


def test_feature_extraction_controller_requests_table_selection_refresh_on_success(
    sdata_blobs: SpatialData,
) -> None:
    refresh_requests: list[FeatureExtractionResult] = []
    result = FeatureExtractionResult(
        job_id=1,
        labels_names=("blobs_labels",),
        table_name="table",
        feature_key="feature_matrix_1",
        regions=("blobs_labels",),
    )
    deferred_worker = _DeferredWorker(result)

    controller = FeatureExtractionController(
        on_table_selection_refresh_required=refresh_requests.append,
    )
    controller.bind(
        sdata_blobs,
        "blobs_labels",
        "blobs_image",
        "table",
        "global",
        ["mean", "area"],
        "feature_matrix_1",
    )
    controller._create_feature_extraction_worker = lambda job: deferred_worker  # type: ignore[method-assign]

    launched = controller.calculate()

    assert launched is True
    assert controller.status_kind == "info"
    assert deferred_worker.started is True

    deferred_worker.emit_returned()

    assert refresh_requests == [result]
    assert controller.status_kind == "success"
    assert (
        controller.status_message
        == 'Feature extraction: wrote "feature_matrix_1" into table "table" as ".obsm[\'feature_matrix_1\']" '
        "with metadata in \".uns['feature_matrices']['feature_matrix_1']\"."
    )
    assert controller.has_job_in_flight is False


def test_feature_extraction_controller_notifies_feature_matrix_change_on_success(sdata_blobs: SpatialData) -> None:
    changed_results: list[FeatureExtractionResult] = []
    result = FeatureExtractionResult(
        job_id=1,
        labels_names=("blobs_labels",),
        table_name="table",
        feature_key="feature_matrix_1",
        regions=("blobs_labels",),
        change_kind="created",
    )
    deferred_worker = _DeferredWorker(result)

    controller = FeatureExtractionController(
        on_feature_matrix_changed=changed_results.append,
    )
    controller.bind(
        sdata_blobs,
        "blobs_labels",
        "blobs_image",
        "table",
        "global",
        ["mean", "area"],
        "feature_matrix_1",
    )
    controller._create_feature_extraction_worker = lambda job: deferred_worker  # type: ignore[method-assign]

    launched = controller.calculate()

    assert launched is True

    deferred_worker.emit_returned()

    assert changed_results == [result]


def test_feature_extraction_controller_calculate_accepts_one_shot_overwrite_override(
    sdata_blobs: SpatialData,
) -> None:
    captured_overwrite_flags: list[bool] = []
    deferred_worker = _DeferredWorker(
        FeatureExtractionResult(
            job_id=1,
            labels_names=("blobs_labels",),
            table_name="table",
            feature_key="feature_matrix_1",
            regions=("blobs_labels",),
        )
    )

    controller = FeatureExtractionController()
    controller.bind(
        sdata_blobs,
        "blobs_labels",
        "blobs_image",
        "table",
        "global",
        ["mean", "area"],
        "feature_matrix_1",
    )

    def capture_worker(job: FeatureExtractionJob) -> _DeferredWorker:
        captured_overwrite_flags.append(job.overwrite_feature_key)
        return deferred_worker

    controller._create_feature_extraction_worker = capture_worker  # type: ignore[method-assign]

    launched = controller.calculate(overwrite_feature_key=True)

    assert launched is True
    assert captured_overwrite_flags == [True]


def test_feature_extraction_controller_calculate_launches_job_with_selected_channels(
    sdata_blobs: SpatialData,
) -> None:
    captured_channels: list[tuple[int | str, ...] | None] = []
    deferred_worker = _DeferredWorker(
        FeatureExtractionResult(
            job_id=1,
            labels_names=("blobs_labels",),
            table_name="table",
            feature_key="feature_matrix_1",
            regions=("blobs_labels",),
        )
    )

    controller = FeatureExtractionController()
    controller.bind(
        sdata_blobs,
        "blobs_labels",
        "blobs_image",
        "table",
        "global",
        ["mean", "area"],
        "feature_matrix_1",
        channels=("0", "2"),
    )

    def capture_worker(job: FeatureExtractionJob) -> _DeferredWorker:
        captured_channels.append(job.channels)
        return deferred_worker

    controller._create_feature_extraction_worker = capture_worker  # type: ignore[method-assign]

    launched = controller.calculate()

    assert launched is True
    assert captured_channels == [("0", "2")]


def test_feature_extraction_controller_calculate_passes_confirmed_entry_and_dirty_snapshot(
    sdata_blobs: SpatialData,
) -> None:
    captured_jobs: list[FeatureExtractionJob] = []
    controller = FeatureExtractionController()
    controller.bind(sdata_blobs, "blobs_labels", None, "table", "global", ["area"], "feature_matrix_1")
    dirty_snapshot = TableDirtySnapshot(sdata=sdata_blobs, table_name="table", captured_path_tokens=())

    def capture_worker(job: FeatureExtractionJob) -> _DeferredWorker:
        captured_jobs.append(job)
        return _DeferredWorker()

    controller._create_feature_extraction_worker = capture_worker  # type: ignore[method-assign]

    assert (
        controller.calculate(
            overwrite_feature_key=True,
            confirmed_existing_entry="compatible",
            dirty_snapshot=dirty_snapshot,
        )
        is True
    )
    (job,) = captured_jobs
    assert job.request.confirmed_existing_entry == "compatible"
    assert job.dirty_snapshot is dirty_snapshot


def test_feature_extraction_controller_checks_existing_feature_matrix(sdata_blobs: SpatialData) -> None:
    controller = FeatureExtractionController()
    controller.bind(sdata_blobs, "blobs_labels", None, "table", "global", ["area"], "feature_matrix_1")

    existing = controller.check_existing_feature_matrix(dirty_paths=())

    assert existing is not None
    assert existing.kind == "absent"


def test_feature_extraction_controller_reports_check_errors_in_its_status(sdata_blobs: SpatialData) -> None:
    controller = FeatureExtractionController()
    controller.bind(sdata_blobs, "blobs_labels", None, "table", "global", ["area"], "feature_matrix_1")
    table = sdata_blobs["table"]
    instance_key = get_table_metadata(sdata_blobs, "table").instance_key
    instance_ids = table.obs[instance_key].to_numpy().copy()
    instance_ids[1] = instance_ids[0]
    table.obs[instance_key] = instance_ids

    assert controller.check_existing_feature_matrix(dirty_paths=()) is None
    assert controller.status_kind == "error"
    assert "pairs must be non-null and unique" in controller.status_message


@pytest.mark.parametrize(
    "dirty_path",
    [
        TableComponentPath("obsm", ("feature_matrix_1",)),
        TableComponentPath("uns", ("feature_matrices", "feature_matrix_1")),
        TableComponentPath("uns", ("feature_matrices",)),
    ],
)
def test_feature_extraction_controller_blocks_unsaved_changes_to_the_key_when_backed(
    backed_sdata_blobs: SpatialData,
    dirty_path: TableComponentPath,
) -> None:
    controller = FeatureExtractionController()
    controller.bind(backed_sdata_blobs, "blobs_labels", None, "table", "global", ["area"], "feature_matrix_1")

    assert controller.check_existing_feature_matrix(dirty_paths={dirty_path}) is None
    assert controller.status_kind == "error"
    assert controller.status_message == (
        'Feature extraction: feature matrix "feature_matrix_1" has unsaved changes; write or reload the table first.'
    )


@pytest.mark.parametrize(
    "dirty_path",
    [
        TableComponentPath("uns", ("feature_matrices", "feature_matrix_2")),
        TableComponentPath("obs", ("annotation",)),
    ],
)
def test_feature_extraction_controller_allows_unsaved_changes_elsewhere_when_backed(
    backed_sdata_blobs: SpatialData,
    dirty_path: TableComponentPath,
) -> None:
    controller = FeatureExtractionController()
    controller.bind(backed_sdata_blobs, "blobs_labels", None, "table", "global", ["area"], "feature_matrix_1")

    existing = controller.check_existing_feature_matrix(dirty_paths={dirty_path})

    assert existing is not None
    assert existing.kind == "absent"


def test_feature_extraction_controller_ignores_unsaved_changes_when_unbacked(sdata_blobs: SpatialData) -> None:
    controller = FeatureExtractionController()
    controller.bind(sdata_blobs, "blobs_labels", None, "table", "global", ["area"], "feature_matrix_1")

    # Unbacked, harpy merges with the matrix in memory, so nothing is dropped; and
    # since unbacked data can't be written or reloaded, a block here would stop
    # every recalculation of the key.
    existing = controller.check_existing_feature_matrix(dirty_paths={TableComponentPath("obsm", ("feature_matrix_1",))})

    assert existing is not None
    assert existing.kind == "absent"


def test_run_feature_extraction_job_reports_change_kind_and_regions_of_the_core(
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    captured_kwargs: dict[str, object] = {}

    def fake_calculate_feature_matrix(sdata: SpatialData, **kwargs: object) -> FeatureMatrixCalculation:
        captured_kwargs.update(kwargs)
        return FeatureMatrixCalculation(
            ExistingFeatureMatrix("incompatible", "it is not saved"),
            "updated",
            ("blobs_labels", "blobs_labels_2"),
        )

    monkeypatch.setattr(controller_module, "calculate_feature_matrix", fake_calculate_feature_matrix)
    dirty_snapshot = TableDirtySnapshot(sdata=sdata_blobs, table_name="table", captured_path_tokens=())

    result = _run_feature_extraction_job.__wrapped__(
        FeatureExtractionJob(
            job_id=3,
            sdata=sdata_blobs,
            request=FeatureExtractionRequest(
                triplets=(FeatureExtractionTriplet("global", "blobs_labels", None),),
                table_name="table",
                create_table=False,
                feature_names=("area",),
                feature_key="feature_matrix_1",
                overwrite_feature_key=True,
                confirmed_existing_entry="incompatible",
            ),
            dirty_snapshot=dirty_snapshot,
        )
    )

    assert captured_kwargs["confirmed_existing_entry"] == "incompatible"
    assert captured_kwargs["overwrite_feature_key"] is True
    assert captured_kwargs["create_table"] is False
    assert result.change_kind == "updated"
    assert result.regions == ("blobs_labels", "blobs_labels_2")
    assert result.labels_names == ("blobs_labels",)
    assert result.dirty_snapshot is dirty_snapshot


def test_run_feature_extraction_job_passes_channels_to_harpy(monkeypatch, sdata_blobs: SpatialData) -> None:
    captured_kwargs: dict[str, object] = {}

    def fake_add_feature_matrix(**kwargs):
        captured_kwargs.update(kwargs)

    monkeypatch.setattr(hp.tb, "add_feature_matrix", fake_add_feature_matrix)

    result = _run_feature_extraction_job.__wrapped__(
        FeatureExtractionJob(
            job_id=4,
            sdata=sdata_blobs,
            request=FeatureExtractionRequest(
                triplets=(
                    FeatureExtractionTriplet(
                        coordinate_system="global",
                        labels_name="blobs_labels",
                        image_name="blobs_image",
                        channels=("0", "2"),
                    ),
                ),
                table_name="table",
                create_table=False,
                feature_names=("mean", "area"),
                feature_key="feature_matrix_1",
                overwrite_feature_key=False,
            ),
        )
    )

    assert captured_kwargs["channels"] == ["0", "2"]
    assert captured_kwargs["feature_matrices_key"] == "feature_matrices"
    assert captured_kwargs["output_table_name"] is None
    assert captured_kwargs["overwrite_output_table"] is False
    assert result == FeatureExtractionResult(
        job_id=4,
        labels_names=("blobs_labels",),
        table_name="table",
        feature_key="feature_matrix_1",
        regions=("blobs_labels",),
        change_kind="created",
    )


def test_run_feature_extraction_job_submits_multi_target_request_to_harpy(
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    captured_kwargs: dict[str, object] = {}

    def fake_add_feature_matrix(**kwargs):
        captured_kwargs.update(kwargs)

    monkeypatch.setattr(hp.tb, "add_feature_matrix", fake_add_feature_matrix)

    result = _run_feature_extraction_job.__wrapped__(
        FeatureExtractionJob(
            job_id=5,
            sdata=sdata_blobs,
            request=FeatureExtractionRequest(
                triplets=(
                    FeatureExtractionTriplet(
                        coordinate_system="global",
                        labels_name="blobs_labels",
                        image_name="blobs_image",
                        channels=("0", "2"),
                    ),
                    FeatureExtractionTriplet(
                        coordinate_system="aligned",
                        labels_name="blobs_multiscale_labels",
                        image_name="blobs_multiscale_image",
                        channels=("0", "2"),
                    ),
                ),
                table_name="table",
                create_table=False,
                feature_names=("mean", "area"),
                feature_key="feature_matrix_batch",
                overwrite_feature_key=True,
            ),
        )
    )

    assert captured_kwargs["labels_name"] == ["blobs_labels", "blobs_multiscale_labels"]
    assert captured_kwargs["image_name"] == ["blobs_image", "blobs_multiscale_image"]
    assert captured_kwargs["table_name"] == "table"
    assert captured_kwargs["output_table_name"] is None
    assert captured_kwargs["to_coordinate_system"] == ["global", "aligned"]
    assert captured_kwargs["channels"] == ["0", "2"]
    assert captured_kwargs["overwrite_output_table"] is False
    assert captured_kwargs["overwrite_feature_key"] is True
    assert result == FeatureExtractionResult(
        job_id=5,
        labels_names=("blobs_labels", "blobs_multiscale_labels"),
        table_name="table",
        feature_key="feature_matrix_batch",
        regions=("blobs_labels", "blobs_multiscale_labels"),
        change_kind="created",
    )
    assert result.triplet_count == 2


def test_run_feature_extraction_job_fills_one_feature_matrix_for_multiple_regions(
    monkeypatch,
    sdata_blobs_multi_region: SpatialData,
) -> None:
    table = sdata_blobs_multi_region["table_multi"]
    feature_key = "feature_matrix_multi_region"

    def fake_add_feature_matrix(**kwargs):
        assert kwargs["labels_name"] == ["blobs_labels", "blobs_labels_2"]
        assert kwargs["table_name"] == "table_multi"
        assert kwargs["feature_key"] == feature_key

        region_values = table.obs["region"].astype("string").to_numpy()
        features = np.full((table.n_obs, 1), np.nan, dtype=np.float64)
        features[region_values == "blobs_labels", 0] = 1.0
        features[region_values == "blobs_labels_2", 0] = 2.0
        table.obsm[feature_key] = features

    monkeypatch.setattr(hp.tb, "add_feature_matrix", fake_add_feature_matrix)

    result = _run_feature_extraction_job.__wrapped__(
        FeatureExtractionJob(
            job_id=6,
            sdata=sdata_blobs_multi_region,
            request=FeatureExtractionRequest(
                triplets=(
                    FeatureExtractionTriplet(
                        coordinate_system="global",
                        labels_name="blobs_labels",
                        image_name=None,
                    ),
                    FeatureExtractionTriplet(
                        coordinate_system="global_1",
                        labels_name="blobs_labels_2",
                        image_name=None,
                    ),
                ),
                table_name="table_multi",
                create_table=False,
                feature_names=("area",),
                feature_key=feature_key,
                overwrite_feature_key=True,
            ),
        )
    )

    features = np.asarray(table.obsm[feature_key], dtype=np.float64)
    region_values = table.obs["region"].astype("string").to_numpy()

    assert result == FeatureExtractionResult(
        job_id=6,
        labels_names=("blobs_labels", "blobs_labels_2"),
        table_name="table_multi",
        feature_key=feature_key,
        regions=("blobs_labels", "blobs_labels_2"),
        change_kind="created",
    )
    assert result.triplet_count == 2
    assert features.shape == (table.n_obs, 1)
    assert np.all(features[region_values == "blobs_labels", 0] == 1.0)
    assert np.all(features[region_values == "blobs_labels_2", 0] == 2.0)


def test_run_feature_extraction_job_submits_create_table_request_to_harpy(
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    captured_kwargs: dict[str, object] = {}

    def fake_add_feature_matrix(**kwargs):
        captured_kwargs.update(kwargs)

    monkeypatch.setattr(hp.tb, "add_feature_matrix", fake_add_feature_matrix)

    result = _run_feature_extraction_job.__wrapped__(
        FeatureExtractionJob(
            job_id=7,
            sdata=sdata_blobs,
            request=FeatureExtractionRequest(
                triplets=(
                    FeatureExtractionTriplet(
                        coordinate_system="global",
                        labels_name="blobs_labels",
                        image_name=None,
                    ),
                ),
                table_name="features_table",
                create_table=True,
                feature_names=("area",),
                feature_key="feature_matrix_1",
                overwrite_feature_key=False,
            ),
        )
    )

    assert captured_kwargs["table_name"] is None
    assert captured_kwargs["output_table_name"] == "features_table"
    assert captured_kwargs["overwrite_output_table"] is False
    assert captured_kwargs["overwrite_feature_key"] is False
    assert result == FeatureExtractionResult(
        job_id=7,
        labels_names=("blobs_labels",),
        table_name="features_table",
        feature_key="feature_matrix_1",
        regions=("blobs_labels",),
        change_kind="created",
    )


def test_feature_extraction_controller_propagates_worker_errors(sdata_blobs: SpatialData) -> None:
    refresh_requests: list[str] = []
    deferred_worker = _DeferredWorker()

    controller = FeatureExtractionController(
        on_table_selection_refresh_required=lambda result: refresh_requests.append(result.table_name),
    )
    controller.bind(
        sdata_blobs,
        "blobs_labels",
        "blobs_image",
        "table",
        "global",
        ["mean", "area"],
        "feature_matrix_1",
    )
    controller._create_feature_extraction_worker = lambda job: deferred_worker  # type: ignore[method-assign]

    launched = controller.calculate()

    assert launched is True
    deferred_worker.emit_errored(RuntimeError("boom"))

    assert refresh_requests == []
    assert controller.status_kind == "error"
    assert controller.status_message == "Feature extraction: calculation failed: boom"
    assert controller.has_job_in_flight is False


def test_feature_extraction_controller_requests_selection_refresh_before_feature_matrix_change(
    sdata_blobs: SpatialData,
) -> None:
    notification_order: list[str] = []
    result = FeatureExtractionResult(
        job_id=1,
        labels_names=("blobs_labels",),
        table_name="table",
        feature_key="feature_matrix_1",
        regions=("blobs_labels",),
        change_kind="created",
    )
    deferred_worker = _DeferredWorker(result)

    controller = FeatureExtractionController(
        on_table_selection_refresh_required=lambda table_result: notification_order.append(
            f"table:{table_result.table_name}"
        ),
        on_feature_matrix_changed=lambda changed_result: notification_order.append(
            f"event:{changed_result.table_name}"
        ),
    )
    controller.bind(
        sdata_blobs,
        "blobs_labels",
        "blobs_image",
        "table",
        "global",
        ["mean", "area"],
        "feature_matrix_1",
    )
    controller._create_feature_extraction_worker = lambda job: deferred_worker  # type: ignore[method-assign]

    assert controller.calculate() is True
    deferred_worker.emit_returned()

    assert notification_order == ["table:table", "event:table"]


def test_feature_extraction_controller_publishes_detached_job_result_without_status(sdata_blobs: SpatialData) -> None:
    """After a detached job returns, its table change is published, but not its status.

    The table was still written, so the table choices are refreshed and
    `on_feature_matrix_changed` fires (the widget turns it into a
    `TableStateChangedEvent`). The status follows the current selection instead
    of reporting the write.
    """
    refresh_requests: list[str] = []
    changed_results: list[FeatureExtractionResult] = []
    result = FeatureExtractionResult(
        job_id=1,
        labels_names=("blobs_labels",),
        table_name="table",
        feature_key="feature_matrix_1",
        regions=("blobs_labels",),
    )
    deferred_worker = _DeferredWorker(result)

    controller = FeatureExtractionController(
        on_table_selection_refresh_required=lambda result: refresh_requests.append(result.table_name),
        on_feature_matrix_changed=changed_results.append,
    )
    controller.bind(
        sdata_blobs,
        "blobs_labels",
        "blobs_image",
        "table",
        "global",
        ["mean", "area"],
        "feature_matrix_1",
    )
    controller._create_feature_extraction_worker = lambda job: deferred_worker  # type: ignore[method-assign]

    launched = controller.calculate()

    assert launched is True
    controller.bind(
        sdata_blobs,
        "blobs_labels",
        "blobs_image",
        "table",
        "global",
        ["mean", "area"],
        "feature_matrix_2",
    )

    assert controller.has_job_in_flight is True
    assert controller.status_message == "Feature extraction: waiting for the previous calculation to finish writing."

    deferred_worker.emit_returned()

    assert changed_results == [result]
    assert refresh_requests == ["table"]
    assert controller.status_message == "Feature extraction: ready to calculate."
    assert controller.status_kind == "success"


def test_feature_extraction_controller_reattaches_job_when_the_inputs_change_back(sdata_blobs: SpatialData) -> None:
    """A job is detached only while the selection differs from the one it was launched for."""
    deferred_worker = _DeferredWorker(
        FeatureExtractionResult(
            job_id=1,
            labels_names=("blobs_labels",),
            table_name="table",
            feature_key="feature_matrix_1",
            regions=("blobs_labels",),
        )
    )
    controller = FeatureExtractionController()
    controller.bind(sdata_blobs, "blobs_labels", None, "table", "global", ["area"], "feature_matrix_1")
    controller._create_feature_extraction_worker = lambda job: deferred_worker  # type: ignore[method-assign]

    assert controller.calculate() is True
    calculating_status = controller.status_message

    controller.bind(sdata_blobs, "blobs_labels", None, "table", "global", ["area"], "feature_matrix_2")

    assert controller.status_message == "Feature extraction: waiting for the previous calculation to finish writing."

    controller.bind(sdata_blobs, "blobs_labels", None, "table", "global", ["area"], "feature_matrix_1")

    assert controller.status_message == calculating_status

    deferred_worker.emit_returned()

    assert controller.status_kind == "success"
    assert controller.status_message.startswith('Feature extraction: wrote "feature_matrix_1" into table "table"')


def test_feature_extraction_controller_ignores_detached_job_of_another_spatialdata(
    sdata_blobs: SpatialData,
    sdata_blobs_multi_region: SpatialData,
) -> None:
    changed_results: list[FeatureExtractionResult] = []
    deferred_worker = _DeferredWorker(
        FeatureExtractionResult(
            job_id=1,
            labels_names=("blobs_labels",),
            table_name="table",
            feature_key="feature_matrix_1",
            regions=("blobs_labels",),
        )
    )
    controller = FeatureExtractionController(on_feature_matrix_changed=changed_results.append)
    controller.bind(sdata_blobs, "blobs_labels", None, "table", "global", ["area"], "feature_matrix_1")
    controller._create_feature_extraction_worker = lambda job: deferred_worker  # type: ignore[method-assign]

    assert controller.calculate() is True
    controller.bind(
        sdata_blobs_multi_region, "blobs_labels", None, "table_multi", "global", ["area"], "feature_matrix_1"
    )
    deferred_worker.emit_returned()

    # The event would be built for the selected SpatialData, which is not the job's.
    assert changed_results == []


def test_feature_extraction_controller_publishes_removal_when_extraction_fails_after_it(
    sdata_blobs: SpatialData,
) -> None:
    refresh_requests: list[str] = []
    changed_results: list[FeatureExtractionResult] = []
    deferred_worker = _DeferredWorker()
    controller = FeatureExtractionController(
        on_table_selection_refresh_required=lambda result: refresh_requests.append(result.table_name),
        on_feature_matrix_changed=changed_results.append,
    )
    controller.bind(sdata_blobs, "blobs_labels", None, "table", "global", ["area"], "feature_matrix_1")
    controller._create_feature_extraction_worker = lambda job: deferred_worker  # type: ignore[method-assign]
    dirty_snapshot = TableDirtySnapshot(sdata=sdata_blobs, table_name="table", captured_path_tokens=())

    assert (
        controller.calculate(
            overwrite_feature_key=True,
            confirmed_existing_entry="incompatible",
            dirty_snapshot=dirty_snapshot,
        )
        is True
    )
    deferred_worker.emit_errored(
        FeatureMatrixRemovedError(
            table_name="table",
            feature_key="feature_matrix_1",
            regions=("blobs_labels", "blobs_labels_2"),
            error=RuntimeError("boom"),
        )
    )

    assert changed_results == [
        FeatureExtractionResult(
            job_id=1,
            labels_names=("blobs_labels",),
            table_name="table",
            feature_key="feature_matrix_1",
            regions=("blobs_labels", "blobs_labels_2"),
            change_kind="removed",
            dirty_snapshot=dirty_snapshot,
        )
    ]
    assert refresh_requests == []
    assert controller.status_kind == "error"
    assert controller.status_message == (
        'Feature extraction failed; the previous feature matrix "feature_matrix_1" was removed: boom'
    )


@pytest.mark.parametrize(
    ("error", "expected_change_kinds"),
    [
        (
            FeatureMatrixRemovedError(
                table_name="table",
                feature_key="feature_matrix_1",
                regions=("blobs_labels",),
                error=RuntimeError("boom"),
            ),
            ["removed"],
        ),
        (RuntimeError("boom"), []),
    ],
)
def test_feature_extraction_controller_reports_detached_job_failure_as_event_only(
    sdata_blobs: SpatialData,
    error: Exception,
    expected_change_kinds: list[str],
) -> None:
    changed_results: list[FeatureExtractionResult] = []
    deferred_worker = _DeferredWorker()
    controller = FeatureExtractionController(on_feature_matrix_changed=changed_results.append)
    controller.bind(sdata_blobs, "blobs_labels", None, "table", "global", ["area"], "feature_matrix_1")
    controller._create_feature_extraction_worker = lambda job: deferred_worker  # type: ignore[method-assign]

    assert controller.calculate() is True
    controller.bind(sdata_blobs, "blobs_labels", None, "table", "global", ["area"], "feature_matrix_2")
    deferred_worker.emit_errored(error)

    assert [result.change_kind for result in changed_results] == expected_change_kinds
    assert controller.status_message == "Feature extraction: ready to calculate."


def test_feature_extraction_controller_blocks_reload_and_new_launches_until_the_detached_worker_finishes(
    sdata_blobs: SpatialData,
) -> None:
    state_changes: list[str] = []
    controller = FeatureExtractionController(on_state_changed=lambda: state_changes.append("changed"))
    controller.bind(
        sdata_blobs,
        "blobs_labels",
        None,
        "table",
        "global",
        ["area"],
        "feature_matrix_1",
    )
    workers: list[_DeferredWorker] = []

    def create_worker(job: FeatureExtractionJob) -> _DeferredWorker:
        worker = _DeferredWorker(
            FeatureExtractionResult(
                job_id=job.job_id,
                labels_names=job.labels_names,
                table_name=job.table_name,
                feature_key=job.feature_key,
                regions=job.labels_names,
            )
        )
        workers.append(worker)
        return worker

    controller._create_feature_extraction_worker = create_worker  # type: ignore[method-assign]
    matching_request = TableReloadRequest(
        sdata=sdata_blobs,
        table_name="table",
        paths=frozenset({TableComponentPath("obsm", ("feature_matrix_1",))}),
        region_name="blobs_labels",
        source="test",
    )
    unrelated_request = TableReloadRequest(
        sdata=sdata_blobs,
        table_name="other_table",
        paths=frozenset({TableComponentPath("obs", ("annotation",))}),
        region_name=None,
        source="test",
    )

    assert controller.calculate() is True
    controller.prepare_for_table_reload(unrelated_request)
    with pytest.raises(RuntimeError, match="Feature Extraction is still writing it"):
        controller.prepare_for_table_reload(matching_request)

    controller.bind(
        sdata_blobs,
        "blobs_labels",
        None,
        "table",
        "global",
        ["area"],
        "feature_matrix_2",
    )
    assert controller.has_job_in_flight is True
    with pytest.raises(RuntimeError, match="Feature Extraction is still writing it"):
        controller.prepare_for_table_reload(matching_request)

    # harpy does not protect concurrent writers, so the next launch waits too.
    waiting_status = "Feature extraction: waiting for the previous calculation to finish writing."
    assert controller.can_calculate is False
    assert controller.status_message == waiting_status
    assert controller.calculate() is False
    assert len(workers) == 1
    assert controller.status_message == waiting_status

    state_changes.clear()
    workers[0].emit_finished()

    assert controller.has_job_in_flight is False
    assert state_changes, "the widget must learn that Calculate is available again"
    assert controller.can_calculate is True
    assert controller.status_message == "Feature extraction: ready to calculate."
    controller.prepare_for_table_reload(matching_request)

    assert controller.calculate() is True
    assert len(workers) == 2


def test_feature_extraction_controller_defers_reload_participant_release_until_workers_finish(
    sdata_blobs: SpatialData,
) -> None:
    controller = FeatureExtractionController()
    controller.bind(
        sdata_blobs,
        "blobs_labels",
        None,
        "table",
        "global",
        ["area"],
        "feature_matrix_1",
    )
    worker = _DeferredWorker()
    controller._create_feature_extraction_worker = lambda _job: worker  # type: ignore[method-assign]

    assert controller.calculate() is True

    released: list[str] = []
    controller.shutdown()
    controller.release_reload_participant_when_safe(lambda: released.append("released"))

    assert worker.quit_called is True
    assert controller.has_job_in_flight is True
    assert released == []

    worker.emit_finished()

    assert controller.has_job_in_flight is False
    assert released == ["released"]
