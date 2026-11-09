from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Literal

from spatiato._app_state import TableBusyMark, TableBusyMarker, TableDirtySnapshot
from spatiato.core.spatial_query.canonical_centers import (
    CanonicalCentersChange,
    CanonicalCentersRegion,
    CanonicalCentersStatus,
    compute_canonical_centers,
    read_canonical_centers_region,
)
from spatiato.core.spatial_query.query import (
    build_canonical_center_query_request,
    evaluate_canonical_center_query,
)
from spatiato.core.spatial_query.query_models import (
    CanonicalCenterQueryRequest,
    CanonicalCenterQueryResult,
)
from spatiato.core.spatialdata import get_table_metadata

if TYPE_CHECKING:
    from spatialdata import SpatialData


def _resolve_thread_worker() -> Any:
    try:
        from napari.qt.threading import thread_worker
    except Exception:  # pragma: no cover - fallback for sandboxed test imports  # noqa: BLE001
        from superqt.utils import thread_worker

    return thread_worker


thread_worker = _resolve_thread_worker()

type SpatialQueryStatusKind = Literal["info", "warning", "success", "error"]
type SpatialQueryWorkerPhase = Literal["canonical_centers", "containment_query"]

SPATIAL_QUERY_IDLE_STATUS = "Spatial query: idle."
_CENTERS_JOB_BUSY_REASON = (
    "Spatial Query is still writing canonical centers. Try again once the calculation has finished."
)


@dataclass(frozen=True)
class _ActiveWorkerPhase:
    """The Run the widget is attached to, with the worker of its current phase."""

    operation_id: int
    phase: SpatialQueryWorkerPhase
    worker: Any
    sdata: SpatialData
    table_name: str
    labels_name: str
    shapes_name: str
    coordinate_system: str


@dataclass(frozen=True)
class _CanonicalCentersJob:
    """One calculation of a table's canonical centers through harpy, which changes the table."""

    operation_id: int
    sdata: SpatialData
    table_name: str
    dirty_snapshot: TableDirtySnapshot | None


@dataclass(frozen=True)
class CanonicalCentersChangeResult:
    """Canonical centers that harpy calculated and installed, for the widget to record.

    `regions` is every labels element of the table, since harpy calculates all
    of them. `dirty_snapshot` is the table's dirty state taken at launch.
    """

    sdata: SpatialData
    table_name: str
    change_kind: CanonicalCentersChange
    regions: tuple[str, ...]
    dirty_snapshot: TableDirtySnapshot | None


@thread_worker(start_thread=False, ignore_errors=True)
def _run_canonical_centers_calculation(sdata: SpatialData, table_name: str) -> CanonicalCentersChange:
    """Have harpy calculate and install the table's canonical centers; this changes the table."""
    return compute_canonical_centers(sdata, table_name=table_name)


@thread_worker(start_thread=False, ignore_errors=True)
def _run_canonical_center_query(
    request: CanonicalCenterQueryRequest,
) -> CanonicalCenterQueryResult:
    """Evaluate one immutable containment request without mutation."""
    return evaluate_canonical_center_query(request)


class SpatialQueryController:
    """Manage background Spatial Query work and main-thread result acceptance.

    With valid canonical centers, a Run reads them and queries:

        main thread
            read_canonical_centers_region()
            → the selected labels element's rows, instance IDs and centers
            → no SpatialData or AnnData mutation

        main thread
            build_canonical_center_query_request()
            → snapshot validated polygons and their labels-frame affine

        worker thread
            evaluate_canonical_center_query()
            → return matching instance IDs
            → no SpatialData or AnnData mutation

    With absent or invalid canonical centers, a Run first calculates them:

        worker thread
            compute_canonical_centers()
            → harpy calculates the centers of every labels element of the
              table and installs them, in memory and, for backed SpatialData,
              in the store

        main thread
            → publish the table change (`on_canonical_centers_changed`)
            → read the selected labels element's centers and query, as above

    The centers job changes the table and cannot be stopped. While it runs,
    its table is busy, so Write and Reload wait for it. A cancel detaches the
    job instead of quitting it: it keeps writing, still publishes its table
    change, and releases the busy mark when it finishes
    (`_on_canonical_centers_finished()`, through `_clear_centers_job()`); it
    never continues with a query. One centers job runs at a time, attached or detached
    (`has_job_in_flight`).

    This job contract mirrors Feature Extraction's controller: one job at a
    time, the busy mark, and publishing the change of a detached job. Sharing
    it in one helper: https://github.com/vibspatial/spatiato/issues/135

    `is_running` means a Run the widget is attached to. Both phases of a Run
    share one operation ID, and every late signal must match the attached
    operation ID and phase before it can affect that Run.
    """

    def __init__(
        self,
        *,
        table_busy_marker: TableBusyMarker,
        on_state_changed: Callable[[], None] | None = None,
        on_canonical_centers_changed: Callable[[CanonicalCentersChangeResult], None] | None = None,
        on_query_ready: Callable[[CanonicalCenterQueryResult], None] | None = None,
    ) -> None:
        # Marks the centers job's table busy while the job writes it. The widget
        # passes app state, which implements `TableBusyMarker`. Required, unlike
        # the callbacks: without it, Write and Reload would not wait for the
        # job. It is also kept after `shutdown()`, so a job that outlives the
        # widget still releases its mark.
        self._table_busy_marker = table_busy_marker
        self._on_state_changed = on_state_changed
        self._on_canonical_centers_changed = on_canonical_centers_changed
        self._on_query_ready = on_query_ready
        self._is_shutdown = False

        self._last_operation_id = 0
        self._active: _ActiveWorkerPhase | None = None
        # The centers job whose harpy call has not finished, attached or
        # detached, and the busy mark of its table. After the widget closes,
        # superqt keeps the running worker alive, and its signal connections
        # keep this controller alive, so `finished` still releases the mark.
        self._centers_job: _CanonicalCentersJob | None = None
        self._busy_mark: TableBusyMark | None = None

        self._status_message = SPATIAL_QUERY_IDLE_STATUS
        self._status_kind: SpatialQueryStatusKind = "info"

    @property
    def operation_id(self) -> int:
        """Return the most recently allocated operation ID."""
        return self._last_operation_id

    @property
    def active_phase(self) -> SpatialQueryWorkerPhase | None:
        """Return the phase of the attached Run, if any."""
        return None if self._active is None else self._active.phase

    @property
    def is_running(self) -> bool:
        """Return whether a Run the widget is attached to is in progress."""
        return self._active is not None

    @property
    def has_job_in_flight(self) -> bool:
        """Return whether a centers job is still writing its table, also when detached.

        A new Run waits for it: one centers job runs at a time.
        """
        return self._centers_job is not None

    @property
    def status_message(self) -> str:
        """Return the current user-facing status message."""
        return self._status_message

    @property
    def status_kind(self) -> SpatialQueryStatusKind:
        """Return the current status level."""
        return self._status_kind

    def start_spatial_query(
        self,
        sdata: SpatialData,
        centers_status: CanonicalCentersStatus,
        *,
        labels_name: str,
        shapes_name: str,
        coordinate_system: str,
        dirty_snapshot: TableDirtySnapshot | None = None,
    ) -> bool:
        """Start one Run: calculate the canonical centers when they are not valid, then query.

        Return `False` without starting while a Run or centers job of this
        controller is in flight, or while another job keeps the table busy.
        `centers_status` is the table's status checked for this Run.
        `dirty_snapshot` is the table's dirty state at launch, returned with a
        centers change.
        """
        if not isinstance(centers_status, CanonicalCentersStatus):
            raise TypeError("Spatial Query requires a CanonicalCentersStatus.")
        if not isinstance(labels_name, str) or not labels_name:
            raise ValueError("Spatial Query labels name must be a non-empty string.")
        if not isinstance(shapes_name, str) or not shapes_name:
            raise ValueError("Spatial Query Shapes name must be a non-empty string.")
        if not isinstance(coordinate_system, str) or not coordinate_system:
            raise ValueError("Spatial Query coordinate system must be a non-empty string.")
        if self._is_shutdown or self.is_running or self.has_job_in_flight:
            return False
        # Another job writes the table, for example Feature Extraction or a
        # centers job of a closed widget: a Run would read and write it
        # meanwhile. The widget disables Run and says why.
        if self._table_busy_marker.table_busy_reasons(sdata, centers_status.table_name):
            return False

        self._last_operation_id += 1
        operation_id = self._last_operation_id
        table_name = centers_status.table_name

        if centers_status.state == "valid":
            try:
                region = read_canonical_centers_region(sdata, table_name=table_name, labels_name=labels_name)
                query_phase = self._build_containment_query_phase(
                    operation_id,
                    sdata,
                    region,
                    shapes_name=shapes_name,
                    coordinate_system=coordinate_system,
                )
            except Exception as exc:  # noqa: BLE001
                self._set_status(f"Spatial query preflight failed: {exc}", kind="error")
                return True

            self._active = query_phase
            self._start_containment_query_worker(query_phase)
            return True

        job = _CanonicalCentersJob(
            operation_id=operation_id,
            sdata=sdata,
            table_name=table_name,
            dirty_snapshot=dirty_snapshot,
        )
        worker = self._create_canonical_centers_worker(job)
        self._centers_job = job
        self._busy_mark = self._table_busy_marker.mark_table_busy(sdata, table_name, reason=_CENTERS_JOB_BUSY_REASON)
        self._active = _ActiveWorkerPhase(
            operation_id=operation_id,
            phase="canonical_centers",
            worker=worker,
            sdata=sdata,
            table_name=table_name,
            labels_name=labels_name,
            shapes_name=shapes_name,
            coordinate_system=coordinate_system,
        )

        worker.returned.connect(partial(self._on_canonical_centers_returned, operation_id))
        worker.errored.connect(partial(self._on_canonical_centers_errored, operation_id))
        worker.finished.connect(partial(self._on_canonical_centers_finished, operation_id))
        self._set_status(f'Calculating centers for every labels element of table "{table_name}".', kind="info")
        try:
            worker.start()
        except Exception as exc:
            self._active = None
            self._clear_centers_job()
            self._set_status(f"Canonical-center calculation could not start: {exc}", kind="error")
            raise
        return True

    def cancel_active_operation(self) -> bool:
        """Detach the widget from its Run; every later signal of that Run is ignored for it.

        A query worker is quit. A centers job cannot be stopped: it keeps
        writing, detached, and still publishes its table change. Its table
        stays busy until the job finishes; then `_on_canonical_centers_finished()`
        releases the busy mark (`_clear_centers_job()`).
        """
        active = self._active
        if active is None:
            return False

        self._active = None
        if active.phase == "containment_query":
            self._quit_worker(active.worker)
            self._set_status("Spatial query containment query cancelled.", kind="info")
        else:
            self._set_status(
                "Spatial query cancelled; the canonical centers are still being written.",
                kind="info",
            )
        return True

    def shutdown(self, *args: object) -> None:
        """Detach the widget's callbacks.

        A centers job keeps writing and still releases its busy mark when it
        finishes (`_clear_centers_job()`).
        """
        del args
        if self._is_shutdown:
            return

        self._is_shutdown = True
        active = self._active
        self._active = None
        if active is not None and active.phase == "containment_query":
            self._quit_worker(active.worker)
        # A centers job cannot be stopped, so it keeps writing the table after
        # shutdown. Its table change is no longer published (the callbacks are
        # cleared below), so app state does not record it; the job still
        # releases its busy mark when it finishes.
        self._on_state_changed = None
        self._on_canonical_centers_changed = None
        self._on_query_ready = None

    def _create_canonical_centers_worker(self, job: _CanonicalCentersJob) -> Any:
        return _run_canonical_centers_calculation(job.sdata, job.table_name)

    def _create_containment_query_worker(
        self,
        request: CanonicalCenterQueryRequest,
    ) -> Any:
        return _run_canonical_center_query(request)

    def _on_canonical_centers_returned(
        self,
        operation_id: int,
        change_kind: CanonicalCentersChange,
    ) -> None:
        job = self._centers_job
        if job is None or job.operation_id != operation_id:
            return

        # Publish the table change, also for a detached job: harpy has already
        # installed the centers. The widget records it while the job's
        # SpatialData is still selected, also when another of its tables is
        # selected now.
        if not self._is_shutdown and self._on_canonical_centers_changed is not None:
            self._on_canonical_centers_changed(
                CanonicalCentersChangeResult(
                    sdata=job.sdata,
                    table_name=job.table_name,
                    change_kind=change_kind,
                    regions=get_table_metadata(job.sdata, job.table_name).regions,
                    dirty_snapshot=job.dirty_snapshot,
                )
            )

        active = self._get_matching_active_phase(operation_id, "canonical_centers")
        if active is None:
            # Detached: the Run was cancelled, so no query follows.
            return

        try:
            region = read_canonical_centers_region(
                active.sdata,
                table_name=active.table_name,
                labels_name=active.labels_name,
            )
            query_phase = self._build_containment_query_phase(
                operation_id,
                active.sdata,
                region,
                shapes_name=active.shapes_name,
                coordinate_system=active.coordinate_system,
            )
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"Spatial query preflight failed: {exc}", kind="error")
            return

        # Replace the active phase before starting the query worker. The
        # centers worker emits `finished` afterward; its handler must then
        # leave this query phase alone.
        self._active = query_phase
        self._start_containment_query_worker(query_phase)

    def _on_canonical_centers_errored(self, operation_id: int, error: Exception) -> None:
        if self._get_matching_active_phase(operation_id, "canonical_centers") is None:
            return
        self._set_status(f"Canonical-center calculation failed: {error}", kind="error")

    def _on_canonical_centers_finished(self, operation_id: int) -> None:
        job = self._centers_job
        if job is not None and job.operation_id == operation_id:
            # Every end of the job comes here, also when it was detached or the
            # widget has closed, so the table is never left busy.
            self._clear_centers_job()
        if self._get_matching_active_phase(operation_id, "canonical_centers") is not None:
            self._active = None
        # Also after a detached job: a new Run may start now.
        self._notify_state_changed()

    def _clear_centers_job(self) -> None:
        self._centers_job = None
        if self._busy_mark is not None:
            self._table_busy_marker.release_table_busy(self._busy_mark)
        self._busy_mark = None

    def _build_containment_query_phase(
        self,
        operation_id: int,
        sdata: SpatialData,
        region: CanonicalCentersRegion,
        *,
        shapes_name: str,
        coordinate_system: str,
    ) -> _ActiveWorkerPhase:
        request = build_canonical_center_query_request(
            sdata,
            shapes_name=shapes_name,
            coordinate_system=coordinate_system,
            canonical_centers=region,
        )
        worker = self._create_containment_query_worker(request)
        worker.returned.connect(partial(self._on_containment_query_returned, operation_id))
        worker.errored.connect(partial(self._on_containment_query_errored, operation_id))
        worker.finished.connect(partial(self._on_containment_query_finished, operation_id))
        return _ActiveWorkerPhase(
            operation_id=operation_id,
            phase="containment_query",
            worker=worker,
            sdata=sdata,
            table_name=region.table_name,
            labels_name=region.labels_name,
            shapes_name=shapes_name,
            coordinate_system=coordinate_system,
        )

    def _start_containment_query_worker(self, phase: _ActiveWorkerPhase) -> None:
        self._set_status(f'Querying centers inside "{phase.shapes_name}".', kind="info")
        try:
            phase.worker.start()
        except Exception as exc:
            self._active = None
            self._set_status(f"Containment query could not start: {exc}", kind="error")
            raise

    def _on_containment_query_returned(
        self,
        operation_id: int,
        result: CanonicalCenterQueryResult,
    ) -> None:
        if self._get_matching_active_phase(operation_id, "containment_query") is None:
            return
        if not isinstance(result, CanonicalCenterQueryResult):
            self._set_status("Spatial query returned an invalid containment result.", kind="error")
            return

        count = result.matched_instance_count
        if count == 0:
            self._set_status(
                "No instances from the selected Labels element have their center inside the selected Shapes.",
                kind="info",
            )
            return

        noun = "centroid" if count == 1 else "centroids"
        self._set_status(f"{count} instance {noun} found.", kind="success")
        if self._on_query_ready is not None:
            self._on_query_ready(result)

    def _on_containment_query_errored(self, operation_id: int, error: Exception) -> None:
        if self._get_matching_active_phase(operation_id, "containment_query") is None:
            return
        self._set_status(f"Containment query failed: {error}", kind="error")

    def _on_containment_query_finished(self, operation_id: int) -> None:
        if self._get_matching_active_phase(operation_id, "containment_query") is None:
            return

        self._active = None
        self._notify_state_changed()

    def _get_matching_active_phase(
        self,
        operation_id: int,
        phase: SpatialQueryWorkerPhase,
    ) -> _ActiveWorkerPhase | None:
        """Return the attached Run when both operation ID and phase match."""
        active = self._active
        if self._is_shutdown or active is None or active.operation_id != operation_id or active.phase != phase:
            return None
        return active

    @staticmethod
    def _quit_worker(worker: Any) -> None:
        quit_worker = getattr(worker, "quit", None)
        if callable(quit_worker):
            quit_worker()

    def _set_status(self, message: str, *, kind: SpatialQueryStatusKind) -> None:
        self._status_message = message
        self._status_kind = kind
        self._notify_state_changed()

    def _notify_state_changed(self) -> None:
        if self._on_state_changed is not None:
            self._on_state_changed()
