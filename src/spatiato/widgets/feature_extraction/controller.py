from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Literal

from harpy.utils._keys import _FEATURE_MATRICES_KEY

from spatiato._app_state import TableDirtySnapshot, TableReloadRequest
from spatiato.core.feature_extraction import (
    ExistingFeatureMatrix,
    ExistingFeatureMatrixKind,
    FeatureExtractionChannel,
    FeatureExtractionTriplet,
    FeatureMatrixRemovedError,
    _get_triplet_channel_selection_error,
    _has_empty_intensity_channel_selection,
    _normalize_channels,
    _normalize_triplets,
    _requires_image,
    calculate_feature_matrix,
    check_existing_feature_matrix,
)
from spatiato.core.persistence import TableComponentPath
from spatiato.core.spatialdata import get_table
from spatiato.core.validation import normalize_spatialdata_name, validate_new_spatialdata_element_name

if TYPE_CHECKING:
    from anndata import AnnData
    from spatialdata import SpatialData


def _resolve_thread_worker() -> Any:
    try:
        from napari.qt.threading import thread_worker
    except Exception:  # pragma: no cover - fallback for sandboxed test imports  # noqa: BLE001
        from superqt.utils import thread_worker

    return thread_worker


FEATURE_EXTRACTION_IDLE_STATUS = "Feature extraction: choose a labels element, table, and output key."
# harpy does not protect concurrent writers to one store, so a detached job that
# is still writing holds back the next launch.
_WAITING_FOR_DETACHED_JOB_STATUS = "Feature extraction: waiting for the previous calculation to finish writing."
FeatureMatrixWriteChangeKind = Literal["created", "updated", "removed"]

thread_worker = _resolve_thread_worker()


@dataclass(frozen=True)
class FeatureExtractionRequest:
    """A validated feature-extraction request covering one or more triplets.

    `table_name` is always the spatiato target table name. `create_table`
    decides whether that name refers to an existing table or to a table Harpy
    should create while writing the feature matrix.

    `confirmed_existing_entry` is the kind of existing feature matrix found
    before launching, which the worker checks again; `None` acts on whatever is
    found (see `calculate_feature_matrix(...)`).
    """

    triplets: tuple[FeatureExtractionTriplet, ...]
    table_name: str
    create_table: bool
    feature_names: tuple[str, ...]
    feature_key: str
    overwrite_feature_key: bool = False
    confirmed_existing_entry: ExistingFeatureMatrixKind | None = None

    def __post_init__(self) -> None:
        if not self.table_name.strip():
            raise ValueError("Feature extraction request requires a table name.")
        if self.create_table and self.overwrite_feature_key:
            raise ValueError("Cannot overwrite a feature key while creating a new table.")

    @property
    def harpy_table_name(self) -> str | None:
        """Return Harpy's existing-table target parameter for this request."""
        return None if self.create_table else self.table_name

    @property
    def harpy_output_table_name(self) -> str | None:
        """Return Harpy's create-table output parameter for this request."""
        return self.table_name if self.create_table else None


@dataclass(frozen=True)
class FeatureExtractionBindingState:
    """Read-only snapshot of the controller's currently bound widget state.

    The Feature Extraction widget keeps its own staged UI state while the
    controller stores the state that would be used for the next calculation.
    This value object lets the widget compare those two views directly before it
    enables calculation. If the expected widget state differs from
    `controller.binding_state`, the UI treats the request as still
    synchronizing and keeps Calculate disabled.

    Unlike `FeatureExtractionRequest`, this binding state may represent an
    incomplete form. For example, `table_name` can be `None` while the user has
    not selected a valid annotation table or entered a new output table name
    yet. When present, `table_name` is the target name for both existing-table
    and create-table modes; `create_table` disambiguates which mode applies.
    """

    sdata: SpatialData | None
    triplets: tuple[FeatureExtractionTriplet, ...]
    table_name: str | None
    create_table: bool
    feature_names: tuple[str, ...]
    feature_key: str | None

    def __post_init__(self) -> None:
        if self.table_name is not None and not self.table_name.strip():
            raise ValueError("Feature extraction binding table name cannot be empty.")

    @property
    def harpy_table_name(self) -> str | None:
        """Return Harpy's existing-table target parameter for this binding."""
        if self.table_name is None:
            return None
        return None if self.create_table else self.table_name

    @property
    def harpy_output_table_name(self) -> str | None:
        """Return Harpy's create-table output parameter for this binding."""
        if self.table_name is None:
            return None
        return self.table_name if self.create_table else None


@dataclass(frozen=True)
class FeatureExtractionJob:
    """Immutable payload copied on the main thread and consumed by a worker.

    `dirty_snapshot` is the target table's dirty state captured at launch. The
    controller does not interpret it: every result of the job carries it back,
    and the widget records the result with it, so only paths that were dirty at
    launch and that the job wrote can become clean. In practice there are none,
    because launching is refused while the key's paths have unsaved changes.
    `None` when the caller tracks no dirty state, for example for a new table:
    the result is then only published.
    """

    job_id: int
    sdata: SpatialData
    request: FeatureExtractionRequest
    dirty_snapshot: TableDirtySnapshot | None = None

    @property
    def labels_names(self) -> tuple[str, ...]:
        return tuple(triplet.labels_name for triplet in self.request.triplets)

    @property
    def image_name(self) -> str | None:
        if len(self.request.triplets) != 1:
            return None
        return self.request.triplets[0].image_name

    @property
    def channels(self) -> tuple[FeatureExtractionChannel, ...] | None:
        if len(self.request.triplets) != 1:
            return None
        return self.request.triplets[0].channels

    @property
    def table_name(self) -> str:
        return self.request.table_name

    @property
    def coordinate_system(self) -> str | None:
        if len(self.request.triplets) != 1:
            return None
        return self.request.triplets[0].coordinate_system

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self.request.feature_names

    @property
    def feature_key(self) -> str:
        return self.request.feature_key

    @property
    def overwrite_feature_key(self) -> bool:
        return self.request.overwrite_feature_key

    @property
    def triplet_count(self) -> int:
        return len(self.request.triplets)


@dataclass(frozen=True)
class FeatureExtractionResult:
    """Summary of a feature-matrix change made by a feature-extraction worker.

    `regions` are the table regions whose rows changed, as table events report
    them: the selected labels elements, or every region of the table after a
    full replacement or a removal (`change_kind="removed"`, when extraction
    failed after the previous matrix was removed). `dirty_snapshot` is the
    job's launch snapshot (`FeatureExtractionJob.dirty_snapshot`).
    """

    job_id: int
    # Tuple because a single feature-extraction request can cover a batch of labels elements.
    labels_names: tuple[str, ...]
    table_name: str
    feature_key: str
    regions: tuple[str, ...]
    change_kind: FeatureMatrixWriteChangeKind = "created"
    dirty_snapshot: TableDirtySnapshot | None = None

    @property
    def triplet_count(self) -> int:
        return len(self.labels_names)


@thread_worker(start_thread=False, ignore_errors=True)
def _run_feature_extraction_job(job: FeatureExtractionJob) -> FeatureExtractionResult:
    request = job.request
    calculation = calculate_feature_matrix(
        job.sdata,
        table_name=request.table_name,
        feature_key=request.feature_key,
        triplets=request.triplets,
        feature_names=request.feature_names,
        create_table=request.create_table,
        overwrite_feature_key=request.overwrite_feature_key,
        confirmed_existing_entry=request.confirmed_existing_entry,
    )

    return FeatureExtractionResult(
        job_id=job.job_id,
        labels_names=job.labels_names,
        table_name=request.table_name,
        feature_key=request.feature_key,
        regions=calculation.regions,
        change_kind=calculation.change_kind,
        dirty_snapshot=job.dirty_snapshot,
    )


class FeatureExtractionController:
    """Manage widget-facing state for background feature extraction."""

    def __init__(
        self,
        *,
        on_state_changed: Callable[[], None] | None = None,
        on_table_selection_refresh_required: Callable[[FeatureExtractionResult], None] | None = None,
        on_feature_matrix_changed: Callable[[FeatureExtractionResult], None] | None = None,
    ) -> None:
        self._on_state_changed = on_state_changed
        self._on_table_selection_refresh_required = on_table_selection_refresh_required
        self._on_feature_matrix_changed = on_feature_matrix_changed

        self._selected_spatialdata: SpatialData | None = None
        self._selected_triplets: tuple[FeatureExtractionTriplet, ...] = ()
        self._selected_table_name: str | None = None
        self._create_table = False
        self._selected_feature_names: tuple[str, ...] = ()
        self._selected_feature_key: str | None = None
        self._selected_labels_name_hint: str | None = None
        self._selected_coordinate_system_hint: str | None = None

        self._last_job_id = 0
        # The feature-extraction job whose harpy call has not finished, and its worker:
        # - `calculate()` sets them; `_on_worker_finished()` clears them when the
        #   worker emits `finished`.
        # - At most one job at a time: harpy does not protect concurrent writers
        #   to one store, so `calculate()` waits while `_job` is set.
        # - The job is detached while the current selection is not the one it was
        #   launched for (`_job_detached`); changing the inputs back reattaches
        #   it. Changing an input does not call `quit()`, which cannot stop a
        #   running harpy call and would only suppress `returned`. A detached job
        #   keeps writing; its table change is still published and the table
        #   choices are refreshed, but the status and selection follow the
        #   current inputs.
        # - Until `finished`, reloads of its table stay blocked, also when the
        #   job is detached or the widget closed.
        # The limit holds per controller: closing the widget mid-run and opening
        # a new one gives a second controller, which may start a second job.
        # Accepted: rare, and a guard across widgets would need shared state in
        # `SpatiatoAppState`.
        self._job: FeatureExtractionJob | None = None
        self._worker: Any | None = None
        self._release_reload_participant: Callable[[], None] | None = None
        self._is_shutdown = False

        self._status_message = FEATURE_EXTRACTION_IDLE_STATUS
        self._status_kind = "warning"

    @property
    def status_message(self) -> str:
        """Return the latest user-facing feature-extraction status message."""
        return self._status_message

    @property
    def status_kind(self) -> str:
        """Return the latest status level: info, warning, success, or error."""
        return self._status_kind

    @property
    def has_job_in_flight(self) -> bool:
        """Return whether a harpy job has not finished, also when it was detached.

        Such a job may still write its table: in memory and, for backed
        SpatialData, in the store. It blocks new launches and reloads of its
        table.
        """
        return self._job is not None

    @property
    def binding_state(self) -> FeatureExtractionBindingState:
        """Return the current bound request snapshot for widget-side comparisons."""
        return FeatureExtractionBindingState(
            sdata=self._selected_spatialdata,
            triplets=self._selected_triplets,
            table_name=self._selected_table_name,
            create_table=self._create_table,
            feature_names=self._selected_feature_names,
            feature_key=self._selected_feature_key,
        )

    @property
    def can_calculate(self) -> bool:
        """Return whether the current selection has the minimum data to run."""
        return (
            self._get_table_target_validation_error() is None
            and bool(self._selected_triplets)
            and bool(self._selected_feature_names)
            and self._selected_feature_key is not None
            and bool(self._selected_feature_key.strip())
            and _get_feature_key_validation_error(self._selected_feature_key) is None
            and _get_triplet_channel_selection_error(self._selected_triplets, self._selected_feature_names) is None
            and not _has_empty_intensity_channel_selection(self._selected_triplets, self._selected_feature_names)
            and (
                not _requires_image(self._selected_feature_names)
                or all(triplet.image_name is not None for triplet in self._selected_triplets)
            )
            # Also a detached job: its harpy call keeps writing until `finished`.
            and not self.has_job_in_flight
        )

    def bind(
        self,
        sdata: SpatialData | None,
        labels_name: str | None,
        image_name: str | None,
        table_name: str | None,
        coordinate_system: str | None,
        feature_names: Sequence[str] | str | None,
        feature_key: str | None,
        *,
        channels: Sequence[FeatureExtractionChannel] | FeatureExtractionChannel | None = None,
        create_table: bool = False,
    ) -> bool:
        """Bind the controller to one visible single-triplet selection."""
        normalized_coordinate_system = None if coordinate_system is None else coordinate_system.strip() or None
        normalized_feature_names = _normalize_feature_names(feature_names)
        normalized_feature_key = None if feature_key is None else feature_key.strip()
        normalized_channels = _normalize_channels(channels)
        triplets: tuple[FeatureExtractionTriplet, ...] = ()
        if labels_name is not None and normalized_coordinate_system is not None:
            triplets = (
                FeatureExtractionTriplet(
                    coordinate_system=normalized_coordinate_system,
                    labels_name=labels_name,
                    image_name=image_name,
                    channels=normalized_channels,
                ),
            )

        return self._bind_batch_state(
            sdata=sdata,
            triplets=triplets,
            table_name=table_name,
            create_table=create_table,
            feature_names=normalized_feature_names,
            feature_key=normalized_feature_key,
            labels_name_hint=labels_name,
            coordinate_system_hint=normalized_coordinate_system,
        )

    def bind_batch(
        self,
        sdata: SpatialData | None,
        triplets: Sequence[FeatureExtractionTriplet] | FeatureExtractionTriplet | None,
        table_name: str | None,
        feature_names: Sequence[str] | str | None,
        feature_key: str | None,
        *,
        create_table: bool = False,
    ) -> bool:
        """Bind the controller to one or more explicit extraction triplets."""
        normalized_triplets = _normalize_triplets(triplets)
        normalized_feature_names = _normalize_feature_names(feature_names)
        normalized_feature_key = None if feature_key is None else feature_key.strip()
        labels_name_hint = normalized_triplets[0].labels_name if len(normalized_triplets) == 1 else None
        coordinate_system_hint = normalized_triplets[0].coordinate_system if len(normalized_triplets) == 1 else None

        return self._bind_batch_state(
            sdata=sdata,
            triplets=normalized_triplets,
            table_name=table_name,
            create_table=create_table,
            feature_names=normalized_feature_names,
            feature_key=normalized_feature_key,
            labels_name_hint=labels_name_hint,
            coordinate_system_hint=coordinate_system_hint,
        )

    def _bind_batch_state(
        self,
        *,
        sdata: SpatialData | None,
        triplets: tuple[FeatureExtractionTriplet, ...],
        table_name: str | None,
        create_table: bool,
        feature_names: tuple[str, ...],
        feature_key: str | None,
        labels_name_hint: str | None,
        coordinate_system_hint: str | None,
    ) -> bool:
        normalized_table_name = _normalize_optional_table_name(table_name)
        normalized_create_table = bool(create_table)
        context_changed = (
            sdata is not self._selected_spatialdata
            or triplets != self._selected_triplets
            or normalized_table_name != self._selected_table_name
            or normalized_create_table is not self._create_table
            or feature_names != self._selected_feature_names
            or feature_key != self._selected_feature_key
            or labels_name_hint != self._selected_labels_name_hint
            or coordinate_system_hint != self._selected_coordinate_system_hint
        )
        was_detached = self._job_detached

        self._selected_spatialdata = sdata
        self._selected_triplets = triplets
        self._selected_table_name = normalized_table_name
        self._create_table = normalized_create_table
        self._selected_feature_names = feature_names
        self._selected_feature_key = feature_key
        self._selected_labels_name_hint = labels_name_hint
        self._selected_coordinate_system_hint = coordinate_system_hint

        if was_detached and self._job is not None and not self._job_detached:
            # The inputs changed back to the job's: it is active again.
            self._set_calculating_status(self._job)
        else:
            self._update_idle_status()
        return context_changed

    def check_existing_feature_matrix(
        self,
        *,
        dirty_paths: Iterable[TableComponentPath],
    ) -> ExistingFeatureMatrix | None:
        """Classify the bound key's existing feature matrix before a launch.

        One extraction checks the existing entry twice:

        1. here, on the main thread when Calculate is clicked, so the widget
           can pick its confirmation dialog;
        2. again in the worker, by `calculate_feature_matrix(...)`, against the
           `ExistingFeatureMatrix.kind` passed to
           `calculate(..., confirmed_existing_entry=...)`. A different kind
           stops the job before anything is removed or calculated, in case the
           table changed in between.

        Runs with the inputs the job will use. `dirty_paths` are the table's
        component paths with unsaved changes. Returns `None`, after setting an
        error status, when nothing may be launched: the key's matrix or
        metadata has unsaved changes in backed SpatialData, which harpy would
        drop, or the check raises, for example for rows that harpy would
        reject. See `docs/development/feature_extraction.md` for the whole flow.
        """
        sdata = self._selected_spatialdata
        table_name = self._selected_table_name
        feature_key = self._selected_feature_key
        if sdata is None or table_name is None or feature_key is None or self._create_table:
            raise RuntimeError("Checking an existing feature matrix requires a bound existing table and feature key.")

        # Only backed data is checked. harpy merges with the existing entry it
        # reads from the store, so unsaved changes to the key in memory would be
        # dropped. Unbacked, harpy merges with the matrix in memory, so nothing
        # is lost; and since unbacked data can't be written or reloaded, a block
        # here would stop every recalculation of the key.
        if sdata.is_backed():
            key_paths = (
                TableComponentPath("obsm", (feature_key,)),
                TableComponentPath("uns", (_FEATURE_MATRICES_KEY, feature_key)),
            )
            if any(_component_paths_overlap(path, key_path) for path in dirty_paths for key_path in key_paths):
                self._set_status(
                    f'Feature extraction: feature matrix "{feature_key}" has unsaved changes; '
                    "write or reload the table first.",
                    kind="error",
                )
                return None

        try:
            return check_existing_feature_matrix(
                sdata,
                table_name=table_name,
                feature_key=feature_key,
                triplets=self._selected_triplets,
                feature_names=self._selected_feature_names,
            )
        except (ValueError, TypeError, FileNotFoundError) as error:
            self._set_status(f"Feature extraction: {error}", kind="error")
            return None

    def calculate(
        self,
        *,
        overwrite_feature_key: bool = False,
        confirmed_existing_entry: ExistingFeatureMatrixKind | None = None,
        dirty_snapshot: TableDirtySnapshot | None = None,
    ) -> bool:
        """Launch feature extraction for the current bound inputs.

        `confirmed_existing_entry` is the kind returned by
        `check_existing_feature_matrix(...)`, which the worker checks again
        before it removes or calculates anything. `dirty_snapshot`, the target
        table's dirty state at launch, is carried back unchanged in the job's
        results (`FeatureExtractionJob.dirty_snapshot`).
        """
        if self._is_shutdown:
            return False
        if self.has_job_in_flight:
            if self._job_detached:
                self._set_status(_WAITING_FOR_DETACHED_JOB_STATUS, kind="info")
            else:
                self._set_status("Feature extraction: calculation is already running.", kind="info")
            return False

        job = self._prepare_feature_extraction_job(
            self._last_job_id + 1,
            overwrite_feature_key=bool(overwrite_feature_key),
            confirmed_existing_entry=confirmed_existing_entry,
            dirty_snapshot=dirty_snapshot,
        )
        if job is None:
            return False

        if self._job is not None:  # pragma: no cover - the checks above return first
            raise RuntimeError("Feature extraction can only run one job at a time.")
        self._last_job_id = job.job_id
        worker = self._create_feature_extraction_worker(job)
        self._job = job
        self._worker = worker
        worker.returned.connect(partial(self._on_worker_returned, job.job_id))
        worker.errored.connect(partial(self._on_worker_errored, job.job_id))
        worker.finished.connect(partial(self._on_worker_finished, job.job_id))
        self._set_calculating_status(job)
        try:
            worker.start()
        except Exception:
            self._clear_job()
            raise
        return True

    def prepare_for_table_reload(self, request: TableReloadRequest) -> None:
        """Reject reload while the feature-extraction job can still write its table.

        `calculate()` stores the launched job in `_job`, and `_on_worker_finished()`
        clears it when its worker emits `finished`. Until then, a reload of its
        table is blocked, also when the job is detached because an input
        changed or the widget closed: its Harpy call cannot be stopped and may
        still be writing.

        The controller performs no AnnData mutation here. Raising stops the
        shared reload before live table components are replaced; reusable
        persistence controls present the message to the user.
        """
        job = self._job
        if job is not None and job.sdata is request.sdata and job.table_name == request.table_name:
            raise RuntimeError(
                f'Cannot reload table "{request.table_name}" while Feature Extraction is still writing it. '
                "Wait for Feature Extraction to finish and try again."
            )

    def shutdown(self, *args: object) -> None:
        """Stop accepting results and detach callbacks owned by the Qt widget."""
        del args
        if self._is_shutdown:
            return

        self._is_shutdown = True
        # A job still running when the widget closes loses its table event.
        if self._worker is not None:
            self._worker.quit()
        self._on_state_changed = None
        self._on_table_selection_refresh_required = None
        self._on_feature_matrix_changed = None

    def release_reload_participant_when_safe(self, release: Callable[[], None]) -> None:
        """Release the detached reload participant after all Harpy calls finish."""
        if self._release_reload_participant is not None:
            raise RuntimeError("Feature Extraction reload-participant release is already pending.")
        self._release_reload_participant = release
        self._release_reload_participant_if_safe()

    def _release_reload_participant_if_safe(self) -> None:
        """Run the deferred app-state unregister after all Harpy jobs finish.

        Widget destruction installs an unregister callback through
        ``release_reload_participant_when_safe()``. Keep that callback pending
        while the Feature Extraction job remains physically in flight so the
        controller continues protecting its table from reload. Its worker
        consumes the callback exactly once when it finishes.

        This helper is called at both lifecycle boundaries because either one
        may happen first:

        worker already finished, then widget teardown
            → the teardown call unregisters immediately

        widget teardown while worker is still busy
            → the teardown call leaves the unregister pending
            → the final ``_on_worker_finished()`` call unregisters
        """
        if self._job is not None:
            return

        release = self._release_reload_participant
        if release is None:
            return

        self._release_reload_participant = None
        release()

    def _prepare_feature_extraction_job(
        self,
        job_id: int,
        *,
        overwrite_feature_key: bool,
        confirmed_existing_entry: ExistingFeatureMatrixKind | None = None,
        dirty_snapshot: TableDirtySnapshot | None = None,
    ) -> FeatureExtractionJob | None:
        if self._selected_spatialdata is None:
            self._set_status(FEATURE_EXTRACTION_IDLE_STATUS, kind="warning")
            return None

        if not self._selected_triplets:
            if self._selected_coordinate_system_hint is None and self._selected_labels_name_hint is not None:
                self._set_status("Feature extraction: choose a coordinate system.", kind="warning")
            elif self._selected_coordinate_system_hint is not None and self._selected_labels_name_hint is None:
                self._set_status("Feature extraction: choose a labels element.", kind="warning")
            else:
                self._set_status(FEATURE_EXTRACTION_IDLE_STATUS, kind="warning")
            return None

        table_target_validation_error = self._get_table_target_validation_error()
        if table_target_validation_error is not None:
            self._set_status(table_target_validation_error, kind="warning")
            return None

        if not self._selected_feature_names:
            self._set_status("Feature extraction: choose at least one feature to calculate.", kind="warning")
            return None

        if self._selected_feature_key is None or not self._selected_feature_key.strip():
            self._set_status("Feature extraction: choose an output feature key.", kind="warning")
            return None

        feature_key_validation_error = _get_feature_key_validation_error(self._selected_feature_key)
        if feature_key_validation_error is not None:
            self._set_status(feature_key_validation_error, kind="warning")
            return None

        channel_selection_error = _get_triplet_channel_selection_error(
            self._selected_triplets, self._selected_feature_names
        )
        if channel_selection_error is not None:
            self._set_status(channel_selection_error, kind="warning")
            return None

        if _requires_image(self._selected_feature_names) and any(
            triplet.image_name is None for triplet in self._selected_triplets
        ):
            self._set_status(
                "Feature extraction: choose an image before calculating intensity features.",
                kind="warning",
            )
            return None

        if _has_empty_intensity_channel_selection(self._selected_triplets, self._selected_feature_names):
            self._set_status(
                "Feature extraction: select at least one channel before calculating intensity features.",
                kind="warning",
            )
            return None

        if self._create_table and overwrite_feature_key:
            self._set_status(
                "Feature extraction: cannot overwrite a feature key while creating a new table.",
                kind="warning",
            )
            return None

        selected_table_name = self._selected_table_name
        if selected_table_name is None:
            self._set_status(
                (
                    "Feature extraction: enter a new table name."
                    if self._create_table
                    else "Feature extraction: choose an annotation table linked to the selected labels element."
                ),
                kind="warning",
            )
            return None

        request = FeatureExtractionRequest(
            triplets=self._selected_triplets,
            table_name=selected_table_name,
            create_table=self._create_table,
            feature_names=self._selected_feature_names,
            feature_key=self._selected_feature_key,
            overwrite_feature_key=overwrite_feature_key,
            confirmed_existing_entry=confirmed_existing_entry,
        )
        return FeatureExtractionJob(
            job_id=job_id,
            sdata=self._selected_spatialdata,
            request=request,
            dirty_snapshot=dirty_snapshot,
        )

    def _set_status(self, message: str, *, kind: str) -> None:
        self._status_message = message
        self._status_kind = kind
        if self._on_state_changed is not None:
            self._on_state_changed()

    def _notify_table_selection_refresh_required(self, result: FeatureExtractionResult) -> None:
        # Keep this local widget refresh hook separate from the shared
        # table-state event. The owning widget refreshes its table choices and
        # keeps the current selection, except that a created table that is still
        # the create-table target is promoted to existing-table mode. This runs
        # before downstream widgets consume the shared semantic event.
        if self._on_table_selection_refresh_required is not None:
            self._on_table_selection_refresh_required(result)

    def _notify_feature_matrix_changed(self, result: FeatureExtractionResult) -> None:
        if self._on_feature_matrix_changed is None or self._selected_spatialdata is None:
            return

        self._on_feature_matrix_changed(result)

    def _update_idle_status(self) -> None:
        if self._selected_spatialdata is None:
            self._set_status(FEATURE_EXTRACTION_IDLE_STATUS, kind="warning")
            return

        if not self._selected_triplets and self._selected_coordinate_system_hint is not None:
            self._set_status("Feature extraction: choose a labels element.", kind="warning")
            return

        if not self._selected_triplets and self._selected_coordinate_system_hint is None:
            if self._selected_labels_name_hint is not None:
                self._set_status("Feature extraction: choose a coordinate system.", kind="warning")
            else:
                self._set_status(FEATURE_EXTRACTION_IDLE_STATUS, kind="warning")
            return

        table_target_validation_error = self._get_table_target_validation_error()
        if table_target_validation_error is not None:
            self._set_status(table_target_validation_error, kind="warning")
            return

        if not self._selected_feature_names:
            self._set_status("Feature extraction: choose at least one feature to calculate.", kind="warning")
            return

        if self._selected_feature_key is None or not self._selected_feature_key.strip():
            self._set_status("Feature extraction: choose an output feature key.", kind="warning")
            return

        feature_key_validation_error = _get_feature_key_validation_error(self._selected_feature_key)
        if feature_key_validation_error is not None:
            self._set_status(feature_key_validation_error, kind="warning")
            return

        channel_selection_error = _get_triplet_channel_selection_error(
            self._selected_triplets, self._selected_feature_names
        )
        if channel_selection_error is not None:
            self._set_status(channel_selection_error, kind="warning")
            return

        if _requires_image(self._selected_feature_names) and any(
            triplet.image_name is None for triplet in self._selected_triplets
        ):
            self._set_status(
                "Feature extraction: choose an image before calculating intensity features.",
                kind="warning",
            )
            return

        if _has_empty_intensity_channel_selection(self._selected_triplets, self._selected_feature_names):
            self._set_status(
                "Feature extraction: select at least one channel before calculating intensity features.",
                kind="warning",
            )
            return

        if self._job is not None:
            # Keep the job's own status ("calculating …", or its result) while it
            # matches the selection; explain the wait for a detached one.
            if self._job_detached:
                self._set_status(_WAITING_FOR_DETACHED_JOB_STATUS, kind="info")
            return

        if self._create_table and self._selected_table_name is not None:
            self._set_status(
                f'Feature extraction: ready to create table "{self._selected_table_name}" and calculate.',
                kind="success",
            )
            return

        self._set_status("Feature extraction: ready to calculate.", kind="success")

    def _set_calculating_status(self, job: FeatureExtractionJob) -> None:
        if job.triplet_count == 1:
            message = f'Feature extraction: calculating "{job.feature_key}" for labels element "{job.labels_names[0]}".'
        else:
            message = f'Feature extraction: calculating "{job.feature_key}" for {job.triplet_count} extraction targets.'
        self._set_status(message, kind="info")

    @property
    def _job_detached(self) -> bool:
        """Return whether a job is in flight for another selection than the current one."""
        job = self._job
        return job is not None and not self._job_matches_selection(job)

    def _job_matches_selection(self, job: FeatureExtractionJob) -> bool:
        """Return whether the current selection is the one `job` was launched for."""
        request = job.request
        same_mode = request.create_table == self._create_table
        # Once a create-table job created its table, the widget promotes the
        # selection from "create <table>" to the existing <table>: still this job.
        promoted_after_creation = request.create_table and not self._create_table
        return (
            job.sdata is self._selected_spatialdata
            and request.triplets == self._selected_triplets
            and request.table_name == self._selected_table_name
            and (same_mode or promoted_after_creation)
            and request.feature_names == self._selected_feature_names
            and request.feature_key == self._selected_feature_key
        )

    def _clear_job(self) -> None:
        self._job = None
        self._worker = None

    def _job_for(self, job_id: int) -> FeatureExtractionJob | None:
        """Return the job in flight if it has `job_id`; signals of other workers are ignored."""
        job = self._job
        return job if job is not None and job.job_id == job_id else None

    def _create_feature_extraction_worker(self, job: FeatureExtractionJob) -> Any:
        return _run_feature_extraction_job(job)

    def _on_worker_returned(self, job_id: int, result: FeatureExtractionResult) -> None:
        # `returned` fires before `finished`, so the job is still in flight.
        job = self._job_for(job_id)
        if self._is_shutdown or job is None or job.sdata is not self._selected_spatialdata:
            return

        # Any job may have created its table, also a detached one, so the widget
        # refreshes its table choices (see `_notify_table_selection_refresh_required`).
        self._notify_table_selection_refresh_required(result)
        self._notify_feature_matrix_changed(result)
        if self._job_detached:
            # The status follows the current selection, not this job.
            return

        self._set_status(
            "Feature extraction: "
            f'wrote "{result.feature_key}" into table "{result.table_name}" as '
            f'".obsm[{result.feature_key!r}]" with metadata in '
            f'".uns[{_FEATURE_MATRICES_KEY!r}][{result.feature_key!r}]".',
            kind="success",
        )

    def _on_worker_errored(self, job_id: int, error: Exception) -> None:
        job = self._job_for(job_id)
        if self._is_shutdown or job is None or job.sdata is not self._selected_spatialdata:
            return

        if isinstance(error, FeatureMatrixRemovedError):
            # The previous matrix is gone, also for a detached job: publish it.
            self._notify_feature_matrix_changed(
                FeatureExtractionResult(
                    job_id=job_id,
                    labels_names=job.labels_names,
                    table_name=error.table_name,
                    feature_key=error.feature_key,
                    regions=error.regions,
                    change_kind="removed",
                    dirty_snapshot=job.dirty_snapshot,
                )
            )
            if not self._job_detached:
                # The message already starts with "Feature extraction failed".
                self._set_status(str(error), kind="error")
            return

        if not self._job_detached:
            self._set_status(f"Feature extraction: calculation failed: {error}", kind="error")

    def _on_worker_finished(self, job_id: int) -> None:
        # Remove reload protection when the job finishes, also when it was
        # detached and is no longer the active job.
        if self._job_for(job_id) is not None:
            was_detached = self._job_detached
            self._clear_job()
            if was_detached and not self._is_shutdown:
                # The status still says "waiting for the previous calculation"; show
                # the current selection's status instead, now that Calculate is
                # available again.
                self._update_idle_status()
            elif self._on_state_changed is not None:
                self._on_state_changed()

        self._release_reload_participant_if_safe()

    def _get_bound_table(self) -> AnnData | None:
        if self._selected_spatialdata is None or self._selected_table_name is None or self._create_table:
            return None

        return get_table(self._selected_spatialdata, self._selected_table_name)

    def _get_table_target_validation_error(self) -> str | None:
        if self._selected_spatialdata is None:
            return None

        if self._selected_table_name is None:
            if self._create_table:
                return "Feature extraction: enter a new table name."
            return "Feature extraction: choose an annotation table linked to the selected labels element."

        if not self._create_table:
            self._get_bound_table()
            return None

        try:
            normalize_spatialdata_name(self._selected_table_name, "table name")
        except ValueError as error:
            return f"Feature extraction: choose a valid table name. {error}"

        try:
            validate_new_spatialdata_element_name(self._selected_spatialdata, self._selected_table_name, "table")
        except ValueError as error:
            return f"Feature extraction: {error}"

        return None


def _normalize_feature_names(feature_names: Sequence[str] | str | None) -> tuple[str, ...]:
    if feature_names is None:
        return ()
    if isinstance(feature_names, str):
        names = [feature_names]
    else:
        names = list(feature_names)

    normalized: list[str] = []
    seen: set[str] = set()
    for feature_name in names:
        normalized_name = str(feature_name).strip()
        if not normalized_name or normalized_name in seen:
            continue
        normalized.append(normalized_name)
        seen.add(normalized_name)
    return tuple(normalized)


def _normalize_optional_table_name(table_name: str | None) -> str | None:
    if table_name is None:
        return None

    normalized_table_name = str(table_name).strip()
    return normalized_table_name or None


def _get_feature_key_validation_error(feature_key: str | None) -> str | None:
    if feature_key is None or not feature_key.strip():
        return "Feature extraction: choose an output feature key."

    try:
        normalize_spatialdata_name(feature_key, "feature matrix key")
    except ValueError as error:
        return f"Feature extraction: choose a valid feature matrix key. {error}"
    return None


def _component_paths_overlap(first: TableComponentPath, second: TableComponentPath) -> bool:
    """Return whether the paths are equal, or one contains the other."""
    shared = min(len(first.keys), len(second.keys))
    return first.component == second.component and first.keys[:shared] == second.keys[:shared]
