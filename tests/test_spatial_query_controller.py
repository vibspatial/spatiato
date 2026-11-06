from __future__ import annotations

import threading
from typing import Any

import pytest
from harpy.table.canonical_centers import CANONICAL_OBSM_KEY
from qtpy.QtCore import QObject, Signal
from spatialdata import SpatialData

import spatiato.widgets.spatial_query.controller as controller_module
from spatiato._app_state import SpatiatoAppState
from spatiato.core.spatial_query import CanonicalCenterQueryResult
from spatiato.core.spatial_query.canonical_centers import (
    CanonicalCentersStatus,
    canonical_centers_status,
    compute_canonical_centers,
    read_canonical_centers_region,
)
from spatiato.widgets.spatial_query.controller import (
    CanonicalCentersChangeResult,
    SpatialQueryController,
    _run_canonical_center_query,
    _run_canonical_centers_calculation,
)


class _DeferredWorker(QObject):
    returned = Signal(object)
    errored = Signal(object)
    finished = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.started = False
        self.quit_called = False

    def start(self) -> None:
        self.started = True

    def quit(self) -> None:
        self.quit_called = True

    def emit_returned(self, value: object) -> None:
        self.returned.emit(value)
        self.finished.emit()

    def emit_errored(self, error: Exception) -> None:
        self.errored.emit(error)
        self.finished.emit()


class _FailingStartWorker(_DeferredWorker):
    def start(self) -> None:
        raise RuntimeError("cannot start")


def _controller(**kwargs: Any) -> SpatialQueryController:
    """Return a controller whose busy marks go to a throwaway app state, unless the test passes its own."""
    kwargs.setdefault("table_busy_marker", SpatiatoAppState())
    return SpatialQueryController(**kwargs)


def _status(sdata: SpatialData) -> CanonicalCentersStatus:
    return canonical_centers_status(sdata, table_name="table")


def _query_result(sdata: SpatialData, *, count: int) -> CanonicalCenterQueryResult:
    region = read_canonical_centers_region(sdata, table_name="table", labels_name="blobs_labels")
    return CanonicalCenterQueryResult(
        canonical_centers=region,
        matched_instance_ids=region.instance_ids[:count],
    )


def _start(controller: SpatialQueryController, sdata: SpatialData) -> bool:
    return controller.start_spatial_query(
        sdata,
        _status(sdata),
        labels_name="blobs_labels",
        shapes_name="blobs_circles",
        coordinate_system="global",
    )


def test_worker_wrappers_calculate_centers_through_harpy_and_evaluate_the_query(
    sdata_blobs: SpatialData,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = object()
    query_result = object()
    monkeypatch.setattr(controller_module, "evaluate_canonical_center_query", lambda current: query_result)

    calculated = _run_canonical_centers_calculation.__wrapped__(sdata_blobs, "table")
    queried = _run_canonical_center_query.__wrapped__(request)

    # The centers job changes the table: harpy installs the centers.
    assert calculated == "created"
    assert CANONICAL_OBSM_KEY in sdata_blobs.tables["table"].obsm
    assert queried is query_result


def test_controller_calculates_centers_in_a_worker_publishes_the_change_and_queries(
    qtbot,
    sdata_blobs: SpatialData,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app_state = SpatiatoAppState()
    main_thread = threading.get_ident()
    calculation_threads: list[int] = []
    query_threads: list[int] = []
    changes: list[CanonicalCentersChangeResult] = []
    change_threads: list[int] = []
    accepted_queries: list[CanonicalCenterQueryResult] = []
    real_compute = compute_canonical_centers

    def compute(sdata: SpatialData, *, table_name: str) -> str:
        calculation_threads.append(threading.get_ident())
        return real_compute(sdata, table_name=table_name)

    def evaluate(request: object) -> CanonicalCenterQueryResult:
        del request
        query_threads.append(threading.get_ident())
        return _query_result(sdata_blobs, count=2)

    def record_change(change: CanonicalCentersChangeResult) -> None:
        change_threads.append(threading.get_ident())
        changes.append(change)

    monkeypatch.setattr(controller_module, "compute_canonical_centers", compute)
    monkeypatch.setattr(controller_module, "build_canonical_center_query_request", lambda *args, **kwargs: object())
    monkeypatch.setattr(controller_module, "evaluate_canonical_center_query", evaluate)
    controller = _controller(
        table_busy_marker=app_state,
        on_canonical_centers_changed=record_change,
        on_query_ready=accepted_queries.append,
    )

    assert _start(controller, sdata_blobs)
    assert controller.active_phase == "canonical_centers"
    assert controller.operation_id == 1
    assert app_state.is_table_busy(sdata_blobs, "table") is True
    # One Run at a time.
    assert not _start(controller, sdata_blobs)
    assert controller.operation_id == 1

    qtbot.waitUntil(lambda: not controller.is_running and not controller.has_job_in_flight, timeout=5000)

    assert calculation_threads and calculation_threads[0] != main_thread
    assert query_threads and query_threads[0] != main_thread
    assert change_threads == [main_thread]
    assert [(change.table_name, change.change_kind, change.regions) for change in changes] == [
        ("table", "created", ("blobs_labels",))
    ]
    assert len(accepted_queries) == 1
    assert accepted_queries[0].matched_instance_count == 2
    assert controller.status_kind == "success"
    assert controller.status_message == "2 instance centroids found."
    assert app_state.is_table_busy(sdata_blobs, "table") is False


def test_valid_centers_skip_the_centers_job_and_start_the_query_phase(
    sdata_blobs: SpatialData,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compute_canonical_centers(sdata_blobs, table_name="table")
    app_state = SpatiatoAppState()
    query_worker = _DeferredWorker()
    changes: list[CanonicalCentersChangeResult] = []
    accepted_queries: list[CanonicalCenterQueryResult] = []
    controller = _controller(
        table_busy_marker=app_state,
        on_canonical_centers_changed=changes.append,
        on_query_ready=accepted_queries.append,
    )
    controller._create_canonical_centers_worker = (  # type: ignore[method-assign]
        lambda job: pytest.fail("Valid canonical centers must not start a centers job.")
    )
    controller._create_containment_query_worker = (  # type: ignore[method-assign]
        lambda request: query_worker
    )
    monkeypatch.setattr(controller_module, "build_canonical_center_query_request", lambda *args, **kwargs: object())

    assert _start(controller, sdata_blobs)

    assert query_worker.started
    assert controller.active_phase == "containment_query"
    assert controller.has_job_in_flight is False
    assert app_state.is_table_busy(sdata_blobs, "table") is False
    query_worker.emit_returned(_query_result(sdata_blobs, count=1))

    assert not controller.is_running
    assert changes == []
    assert len(accepted_queries) == 1
    assert accepted_queries[0].matched_instance_count == 1
    assert controller.status_message == "1 instance centroid found."


def test_query_worker_that_fails_to_start_leaves_no_run_attached(
    sdata_blobs: SpatialData,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compute_canonical_centers(sdata_blobs, table_name="table")
    controller = _controller()
    controller._create_containment_query_worker = (  # type: ignore[method-assign]
        lambda request: _FailingStartWorker()
    )
    monkeypatch.setattr(controller_module, "build_canonical_center_query_request", lambda *args, **kwargs: object())

    with pytest.raises(RuntimeError, match="cannot start"):
        _start(controller, sdata_blobs)

    assert not controller.is_running
    assert controller.active_phase is None
    assert controller.status_kind == "error"
    assert controller.status_message == "Containment query could not start: cannot start"
    # The controller is not stuck: a new Run can start.
    query_worker = _DeferredWorker()
    controller._create_containment_query_worker = (  # type: ignore[method-assign]
        lambda request: query_worker
    )
    assert _start(controller, sdata_blobs)
    assert query_worker.started


def test_centers_worker_that_fails_to_start_leaves_no_job_and_releases_its_mark(
    sdata_blobs: SpatialData,
) -> None:
    app_state = SpatiatoAppState()
    controller = _controller(table_busy_marker=app_state)
    controller._create_canonical_centers_worker = (  # type: ignore[method-assign]
        lambda job: _FailingStartWorker()
    )

    with pytest.raises(RuntimeError, match="cannot start"):
        _start(controller, sdata_blobs)

    assert not controller.is_running
    assert controller.has_job_in_flight is False
    assert app_state.is_table_busy(sdata_blobs, "table") is False
    assert controller.status_kind == "error"
    assert controller.status_message == "Canonical-center calculation could not start: cannot start"


def test_start_refuses_while_another_job_keeps_the_table_busy(
    sdata_blobs: SpatialData,
) -> None:
    app_state = SpatiatoAppState()
    controller = _controller(table_busy_marker=app_state)
    controller._create_canonical_centers_worker = (  # type: ignore[method-assign]
        lambda job: pytest.fail("A busy table must not start a centers job.")
    )
    controller._create_containment_query_worker = (  # type: ignore[method-assign]
        lambda request: pytest.fail("A busy table must not start a query.")
    )
    other_job = app_state.mark_table_busy(sdata_blobs, "table", reason="Feature Extraction is still writing.")

    assert not _start(controller, sdata_blobs)
    assert controller.operation_id == 0
    assert not controller.is_running
    assert controller.has_job_in_flight is False

    app_state.release_table_busy(other_job)
    centers_worker = _DeferredWorker()
    controller._create_canonical_centers_worker = (  # type: ignore[method-assign]
        lambda job: centers_worker
    )
    assert _start(controller, sdata_blobs)
    assert centers_worker.started


def test_centers_phase_finished_signal_cannot_clear_the_installed_query_phase(
    sdata_blobs: SpatialData,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    centers_worker = _DeferredWorker()
    query_worker = _DeferredWorker()
    controller = _controller()
    controller._create_canonical_centers_worker = (  # type: ignore[method-assign]
        lambda job: centers_worker
    )
    controller._create_containment_query_worker = (  # type: ignore[method-assign]
        lambda request: query_worker
    )
    monkeypatch.setattr(controller_module, "build_canonical_center_query_request", lambda *args, **kwargs: object())

    assert _start(controller, sdata_blobs)
    # Stand in for the harpy call of the deferred worker.
    compute_canonical_centers(sdata_blobs, table_name="table")

    # emit_returned() emits the centers worker's `finished` signal after its
    # returned callback has installed the query phase.
    centers_worker.emit_returned("created")

    assert query_worker.started
    assert controller.is_running
    assert controller.active_phase == "containment_query"
    assert controller.has_job_in_flight is False


def test_cancel_detaches_the_centers_job_which_still_publishes_its_change_and_releases_its_mark(
    sdata_blobs: SpatialData,
) -> None:
    app_state = SpatiatoAppState()
    first_worker = _DeferredWorker()
    second_worker = _DeferredWorker()
    workers = [first_worker, second_worker]
    changes: list[CanonicalCentersChangeResult] = []
    controller = _controller(table_busy_marker=app_state, on_canonical_centers_changed=changes.append)
    controller._create_canonical_centers_worker = (  # type: ignore[method-assign]
        lambda job: workers.pop(0)
    )
    controller._create_containment_query_worker = (  # type: ignore[method-assign]
        lambda request: pytest.fail("A detached centers job must not start a query.")
    )

    assert _start(controller, sdata_blobs)
    assert controller.cancel_active_operation()

    # The harpy call cannot be stopped: the job is detached, not quit.
    assert first_worker.quit_called is False
    assert controller.is_running is False
    assert controller.has_job_in_flight is True
    assert app_state.is_table_busy(sdata_blobs, "table") is True
    assert controller.status_message == "Spatial query cancelled; the canonical centers are still being written."
    # One centers job at a time, attached or detached.
    assert not _start(controller, sdata_blobs)

    compute_canonical_centers(sdata_blobs, table_name="table")
    first_worker.emit_returned("created")

    assert [change.change_kind for change in changes] == ["created"]
    assert controller.has_job_in_flight is False
    assert app_state.is_table_busy(sdata_blobs, "table") is False

    # Valid centers now: the next Run goes straight to the query, which this
    # test does not start, so let it calculate again.
    sdata_blobs.tables["table"].obsm.pop(CANONICAL_OBSM_KEY)
    sdata_blobs.tables["table"].uns.pop("spatial_coordinates")
    assert _start(controller, sdata_blobs)
    assert controller.operation_id == 2
    assert second_worker.started


def test_centers_job_releases_its_mark_when_it_finishes_after_shutdown(
    sdata_blobs: SpatialData,
) -> None:
    app_state = SpatiatoAppState()
    centers_worker = _DeferredWorker()
    changes: list[CanonicalCentersChangeResult] = []
    controller = _controller(table_busy_marker=app_state, on_canonical_centers_changed=changes.append)
    controller._create_canonical_centers_worker = (  # type: ignore[method-assign]
        lambda job: centers_worker
    )

    assert _start(controller, sdata_blobs)
    controller.shutdown()

    assert centers_worker.quit_called is False
    assert app_state.is_table_busy(sdata_blobs, "table") is True

    compute_canonical_centers(sdata_blobs, table_name="table")
    centers_worker.emit_returned("created")

    # The widget is gone, so the change is not published; the mark is released.
    assert changes == []
    assert app_state.is_table_busy(sdata_blobs, "table") is False


def test_centers_job_error_stops_the_run_and_releases_its_mark(
    sdata_blobs: SpatialData,
) -> None:
    app_state = SpatiatoAppState()
    centers_worker = _DeferredWorker()
    changes: list[CanonicalCentersChangeResult] = []
    controller = _controller(table_busy_marker=app_state, on_canonical_centers_changed=changes.append)
    controller._create_canonical_centers_worker = (  # type: ignore[method-assign]
        lambda job: centers_worker
    )

    assert _start(controller, sdata_blobs)
    centers_worker.emit_errored(ValueError("has no finite center"))

    assert not controller.is_running
    assert controller.has_job_in_flight is False
    assert app_state.is_table_busy(sdata_blobs, "table") is False
    assert changes == []
    assert controller.status_kind == "error"
    assert "Canonical-center calculation failed: has no finite center" in controller.status_message


def test_cancel_during_the_query_phase_quits_the_query_worker(
    sdata_blobs: SpatialData,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compute_canonical_centers(sdata_blobs, table_name="table")
    query_worker = _DeferredWorker()
    accepted_queries: list[CanonicalCenterQueryResult] = []
    controller = _controller(on_query_ready=accepted_queries.append)
    controller._create_containment_query_worker = (  # type: ignore[method-assign]
        lambda request: query_worker
    )
    monkeypatch.setattr(controller_module, "build_canonical_center_query_request", lambda *args, **kwargs: object())

    assert _start(controller, sdata_blobs)
    assert controller.cancel_active_operation()

    assert query_worker.quit_called is True
    assert controller.status_message == "Spatial query containment query cancelled."
    query_worker.emit_returned(_query_result(sdata_blobs, count=1))
    assert accepted_queries == []


def test_zero_match_query_does_not_publish_query_ready(
    sdata_blobs: SpatialData,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compute_canonical_centers(sdata_blobs, table_name="table")
    query_worker = _DeferredWorker()
    accepted_queries: list[CanonicalCenterQueryResult] = []
    controller = _controller(on_query_ready=accepted_queries.append)
    controller._create_containment_query_worker = (  # type: ignore[method-assign]
        lambda request: query_worker
    )
    monkeypatch.setattr(controller_module, "build_canonical_center_query_request", lambda *args, **kwargs: object())
    _start(controller, sdata_blobs)

    query_worker.emit_returned(_query_result(sdata_blobs, count=0))

    assert accepted_queries == []
    assert controller.status_kind == "info"
    assert (
        controller.status_message
        == "No instances from the selected Labels element have their center inside the selected Shapes."
    )


def test_active_query_worker_error_stops_operation_without_delivering_result(
    sdata_blobs: SpatialData,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compute_canonical_centers(sdata_blobs, table_name="table")
    query_worker = _DeferredWorker()
    accepted_queries: list[CanonicalCenterQueryResult] = []
    controller = _controller(on_query_ready=accepted_queries.append)
    controller._create_containment_query_worker = (  # type: ignore[method-assign]
        lambda request: query_worker
    )
    monkeypatch.setattr(controller_module, "build_canonical_center_query_request", lambda *args, **kwargs: object())
    _start(controller, sdata_blobs)

    query_worker.emit_errored(RuntimeError("predicate failed"))

    assert not controller.is_running
    assert accepted_queries == []
    assert controller.status_kind == "error"
    assert "predicate failed" in controller.status_message
