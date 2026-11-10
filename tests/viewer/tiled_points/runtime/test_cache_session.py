from __future__ import annotations

import gc
import threading
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pytest

import spatiato.viewer.tiled_points.runtime.cache_session as cache_session_module
from spatiato.core.multi_scale_cache_points_zarr.reader import (
    _LevelSelection,
    _PlannedTileRead,
    _SelectedValueLevelIndex,
    _TileReadResult,
    _ViewportReadPlan,
    _ViewportReadResult,
)
from spatiato.core.multi_scale_cache_points_zarr.storage.bucket_reader import _BucketReader
from spatiato.viewer.tiled_points.contracts import (
    TILED_POINTS_VERTEX_DTYPE,
    TiledPointsRenderResult,
    TiledPointsViewportState,
    _ViewportRequest,
)
from spatiato.viewer.tiled_points.runtime.cache_session import (
    _CacheSessionFailure,
    _CacheSessionSettings,
    _CacheSessionState,
    _TiledPointsCacheSession,
    _TiledPointsCacheWorker,
)


@pytest.fixture(autouse=True)
def forbid_viewer_sparse_range_reads(monkeypatch: pytest.MonkeyPatch) -> None:
    # Guard physical range access whenever the worker uses a real bucket reader.
    # This remains active through startup, selection changes, and viewport reads.
    original_array = _BucketReader._array

    def guarded_array(self: _BucketReader, name: str):
        if name.startswith("ranges/"):
            raise AssertionError("The viewer cache session accessed bucket sparse ranges.")
        return original_array(self, name)

    monkeypatch.setattr(_BucketReader, "_array", guarded_array)


@dataclass
class _ReaderProbe:
    resident_bytes: int = 64
    operations: list[tuple[str, int]] = field(default_factory=list)
    selection_calls: list[tuple[int, ...]] = field(default_factory=list)
    selected_value_limits: list[int | None] = field(default_factory=list)
    fail_selection: bool = False
    pause_entry: bool = False
    entry_paused: threading.Event = field(default_factory=threading.Event)
    resume_entry: threading.Event = field(default_factory=threading.Event)
    fail_entry: bool = False
    pause_selection: bool = False
    selection_paused: threading.Event = field(default_factory=threading.Event)
    resume_selection: threading.Event = field(default_factory=threading.Event)
    pause_construction: bool = False
    construction_paused: threading.Event = field(default_factory=threading.Event)
    resume_construction: threading.Event = field(default_factory=threading.Event)
    planned_tile_x: tuple[int, ...] = (0, 1)
    viewport_reads: list[tuple[tuple[int, int, int], ...]] = field(default_factory=list)
    last_location_batch: np.ndarray | None = None
    last_value_id_batch: np.ndarray | None = None

    def record(self, operation: str) -> None:
        self.operations.append((operation, threading.get_ident()))


@dataclass(frozen=True)
class _FakeSelectedValueIndex:
    resident_bytes: int
    value_ids: np.ndarray
    levels: tuple[_SelectedValueLevelIndex, ...]


class _ControllableReader:
    value_names = ("A", "B", "C")

    def __init__(self, cache_root: Path, probe: _ReaderProbe) -> None:
        del cache_root
        self._probe = probe
        self.dataset_info = _FakeDatasetInfo()
        probe.record("construct")
        if probe.pause_construction:
            probe.construction_paused.set()
            assert probe.resume_construction.wait(timeout=5)

    def __enter__(self) -> _ControllableReader:
        self._probe.record("enter")
        if self._probe.pause_entry:
            self._probe.entry_paused.set()
            assert self._probe.resume_entry.wait(timeout=5)
        if self._probe.fail_entry:
            raise ValueError("invalid compact metadata")
        return self

    @property
    def resident_index_bytes(self) -> int:
        return self._probe.resident_bytes

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> bool:
        del exc_type, exc_value, traceback
        self._probe.record("exit")
        return False

    def load_selected_value_index(
        self,
        value_ids: np.ndarray,
        *,
        max_resident_bytes: int | None,
    ) -> _FakeSelectedValueIndex:
        self._probe.record("load_selection")
        self._probe.selected_value_limits.append(max_resident_bytes)
        selection = tuple(int(value_id) for value_id in value_ids)
        self._probe.selection_calls.append(selection)
        if self._probe.pause_selection:
            self._probe.selection_paused.set()
            assert self._probe.resume_selection.wait(timeout=5)
        if self._probe.fail_selection:
            raise ValueError("selection does not fit")
        tile_count = len(self._probe.planned_tile_x)
        value_indptr = np.arange(0, (len(value_ids) + 1) * tile_count, tile_count, dtype=np.uint64)
        manifest_index = np.tile(np.asarray(self._probe.planned_tile_x, dtype=np.uint64), len(value_ids))
        n_points = np.ones(len(manifest_index), dtype=np.uint64)
        return _FakeSelectedValueIndex(
            24 if max_resident_bytes is None else min(max_resident_bytes, 24),
            value_ids.copy(),
            (_SelectedValueLevelIndex(value_indptr, manifest_index, n_points),),
        )

    def select_level(self, viewport: object, point_budget: int, *, value_index: object) -> _LevelSelection:
        del viewport, value_index
        point_count = len(self._probe.planned_tile_x)
        return _LevelSelection(
            level=0,
            estimated_point_count=point_count,
            positive_visible_tile_count=point_count,
            fits_point_budget=point_count <= point_budget,
            omitted_value_ids=None,
        )

    def iter_level_candidates(self, viewport, point_budget, *, value_index):
        # The worker requests LOD candidates through this iterator.
        # These tests exercise lifecycle and batch reuse, not LOD-selection
        # rules, so yield one controlled candidate. Delegating to this fake's
        # select_level() lets individual tests override which level it supplies.
        yield self.select_level(viewport, point_budget, value_index=value_index)

    def plan_viewport(self, level: int, viewport: object, *, value_index: object) -> _ViewportReadPlan:
        del viewport
        requested_value_ids = (
            None if value_index is None else tuple(int(value_id) for value_id in value_index.value_ids)
        )
        return _ViewportReadPlan(
            cache_generation_id=_GENERATION_ID,
            requested_value_ids=requested_value_ids,
            level=level,
            requests=tuple(_PlannedTileRead(level, tile_x, 0, tile_x, 0) for tile_x in self._probe.planned_tile_x),
            route="tile_major_all_values" if value_index is None else "value_major_subset",
            selected_value_level_index=None if value_index is None else value_index.levels[level],
        )

    def read_planned_tiles(
        self,
        plan: _ViewportReadPlan,
        tile_keys_to_read: tuple[tuple[int, int, int], ...],
        *,
        raise_if_cancelled: Callable[[], None] | None = None,
    ) -> _ViewportReadResult:
        if raise_if_cancelled is not None:
            raise_if_cancelled()
        self._probe.record("read_viewport")
        self._probe.viewport_reads.append(tile_keys_to_read)
        location_batch = np.asarray(
            [[float(tile_x), float(tile_y)] for _, tile_x, tile_y in tile_keys_to_read],
            dtype=np.float32,
        )
        value_id_batch = np.zeros(len(tile_keys_to_read), dtype=np.uint32)
        self._probe.last_location_batch = location_batch
        self._probe.last_value_id_batch = value_id_batch
        return _ViewportReadResult(
            level=plan.level,
            tiles=tuple(
                _TileReadResult(
                    level=level,
                    tile_x=tile_x,
                    tile_y=tile_y,
                    tile_size=10,
                    location=location_batch[index : index + 1, :],
                    value_id=value_id_batch[index : index + 1],
                )
                for index, (level, tile_x, tile_y) in enumerate(tile_keys_to_read)
            ),
        )


_GENERATION_ID = "12345678-1234-5678-9234-567812345678"


@dataclass(frozen=True)
class _FakeLevelInfo:
    kind: str = "exact"


@dataclass(frozen=True)
class _FakeDatasetInfo:
    cache_generation_id: str = _GENERATION_ID
    value_names: tuple[str, ...] = ("A", "B", "C")
    levels: tuple[_FakeLevelInfo, ...] = (_FakeLevelInfo(),)


def _session(
    probe: _ReaderProbe,
    *,
    max_selected_value_index_bytes: int | None = 1_000,
    max_cpu_tile_bytes: int = 1_000,
    max_vertex_payload_bytes: int = 1_000_000,
) -> _TiledPointsCacheSession:
    return _TiledPointsCacheSession(
        Path("unused.zarr"),
        _CacheSessionSettings(
            max_selected_value_index_bytes=max_selected_value_index_bytes,
            max_cpu_tile_bytes=max_cpu_tile_bytes,
            max_vertex_payload_bytes=max_vertex_payload_bytes,
        ),
        reader_factory=lambda cache_root: _ControllableReader(cache_root, probe),
    )


def _start_ready(session: _TiledPointsCacheSession, qtbot) -> None:
    with qtbot.waitSignal(session.ready, timeout=5_000):
        session.start()
    assert session.state is _CacheSessionState.READY


def _close(session: _TiledPointsCacheSession, qtbot) -> None:
    if session.state is _CacheSessionState.CLOSED:
        return
    with qtbot.waitSignal(session.closed, timeout=5_000):
        session.close()
    assert session.state is _CacheSessionState.CLOSED


def _viewport_request(request_generation: int) -> _ViewportRequest:
    return _ViewportRequest(
        request_generation=request_generation,
        selection_generation=0,
        requested_value_ids=None,
        viewport=TiledPointsViewportState(
            displayed_axes=(0, 1),
            x_min=0.0,
            y_min=0.0,
            x_max=30.0,
            y_max=10.0,
            canvas_width=100,
            canvas_height=100,
            hard_render_point_budget=100,
            screen_density_budget=100,
        ),
    )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("max_cpu_tile_bytes", None),
        ("max_cpu_tile_bytes", True),
        ("max_cpu_tile_bytes", 0),
        ("max_cpu_tile_bytes", -1),
        ("max_vertex_payload_bytes", None),
        ("max_vertex_payload_bytes", True),
        ("max_vertex_payload_bytes", 0),
        ("max_vertex_payload_bytes", -1),
        ("max_vertex_payload_bytes", 1),
        ("max_vertex_payload_bytes", TILED_POINTS_VERTEX_DTYPE.itemsize - 1),
    ],
)
def test_session_settings_reject_invalid_worker_allocation_limits(name: str, value: object) -> None:
    values = {"max_cpu_tile_bytes": 1_000, "max_vertex_payload_bytes": 1_000}
    values[name] = value
    with pytest.raises(ValueError, match=name):
        _CacheSessionSettings(
            max_selected_value_index_bytes=None,
            **values,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("level", [0, 1, 2], ids=["exact", "bridge", "spatial"])
@pytest.mark.parametrize("selection", [None, (0,)], ids=["all-values", "subset"])
def test_retained_batch_skips_tile_work_after_current_lod_selection(level, selection, monkeypatch) -> None:
    """All physical routes/levels use one contained-view reuse contract."""
    probe = _ReaderProbe()
    worker = _TiledPointsCacheWorker(
        Path("unused"),
        _CacheSessionSettings(None, 1000, 1000),
        threading.Event(),
        lambda path: _ControllableReader(path, probe),
    )
    snapshots = []
    worker.viewport_ready.connect(snapshots.append)
    worker.start()
    worker.update_selected_value_index(selection)
    reader = worker._reader
    reader.dataset_info = replace(
        reader.dataset_info, levels=tuple(_FakeLevelInfo(kind) for kind in ("exact", "bridge", "spatial"))
    )
    if selection is not None:
        worker._selected_value_index = replace(
            worker._selected_value_index, levels=worker._selected_value_index.levels * 3
        )
    original_select = reader.select_level
    lod_calls = []

    def select(viewport, point_budget, **kwargs):
        lod_calls.append(viewport)
        return replace(original_select(viewport, point_budget, **kwargs), level=level)

    monkeypatch.setattr(reader, "select_level", select)
    request = replace(_viewport_request(1), requested_value_ids=selection)
    try:
        worker.read_viewport_snapshot(request)
        first = snapshots[-1]
        assert first.level == level
        assert worker.retained_render_batch_bytes == 0
        assert worker.pending_render_batch_bytes == 24
        worker.acknowledge_render_result(TiledPointsRenderResult(1, 0, True))
        assert worker.retained_render_batch_bytes == 24
        assert worker.pending_render_batch_bytes == 0

        def forbidden(*args, **kwargs):
            raise AssertionError("Contained reuse performed tile planning, lookup, IO, or packing")

        monkeypatch.setattr(cache_session_module, "_read_viewport_snapshot", forbidden)
        monkeypatch.setattr(reader, "plan_viewport", forbidden)
        monkeypatch.setattr(reader, "read_planned_tiles", forbidden)
        monkeypatch.setattr(worker._cpu_tile_residency, "get", forbidden)
        monkeypatch.setattr(cache_session_module, "pack_render_tiles", forbidden)
        # Include an empty visible estimate: empty space still belongs to the
        # original rectangle and does not discard its off-screen vertices.
        probe.planned_tile_x = ()
        for generation, bounds in enumerate(((5.0, 10.0), (15.0, 20.0), (0.0, 30.0)), start=2):
            if generation == 4:
                probe.planned_tile_x = (0, 1)
            current = replace(
                request,
                request_generation=generation,
                viewport=replace(request.viewport, x_min=bounds[0], x_max=bounds[1]),
            )
            worker.read_viewport_snapshot(current)
            snapshot = snapshots[-1]
            assert snapshot.request_generation == generation
            assert snapshot.render_batch is first.render_batch
            assert snapshot.estimated_point_count == len(probe.planned_tile_x)
            worker.acknowledge_render_result(TiledPointsRenderResult(generation, 0, True))
            assert worker._retained_viewport.bounds.x_min == 0.0
            assert worker._retained_viewport.bounds.x_max == 30.0
        assert len(lod_calls) == 4
    finally:
        worker.close()
    assert worker.retained_render_batch_bytes == worker.pending_render_batch_bytes == 0


def test_rejected_replacement_preserves_accepted_bounds_and_has_no_viewport_history() -> None:
    """Retain the last accepted viewport, not a history of packed batches.

    1. Rejecting B preserves A: after accepting viewport A, preparing and
       rejecting a disjoint viewport B must leave A available for reuse,
       with its original bounds and render batch.

    2. Accepting B replaces A: once B is accepted, returning to A requires
       a newly packed batch. Decoded CPU tiles may still be reused, but A's
       previous packed batch is no longer retained.
    """
    probe = _ReaderProbe()
    worker = _TiledPointsCacheWorker(
        Path("unused"),
        _CacheSessionSettings(None, 1000, 1000),
        threading.Event(),
        lambda path: _ControllableReader(path, probe),
    )
    snapshots = []
    worker.viewport_ready.connect(snapshots.append)
    worker.start()
    first_request = _viewport_request(1)
    outside = replace(
        first_request, request_generation=2, viewport=replace(first_request.viewport, x_min=40.0, x_max=70.0)
    )
    try:
        worker.read_viewport_snapshot(first_request)
        first = snapshots[-1]
        worker.acknowledge_render_result(TiledPointsRenderResult(1, 0, True))
        worker.read_viewport_snapshot(outside)
        assert snapshots[-1].render_batch is not first.render_batch
        assert worker.pending_render_batch_bytes == worker.retained_render_batch_bytes == 24
        worker.acknowledge_render_result(TiledPointsRenderResult(2, 0, False))
        assert worker.pending_render_batch_bytes == 0
        worker.read_viewport_snapshot(replace(first_request, request_generation=3))
        assert snapshots[-1].render_batch is first.render_batch
        worker.acknowledge_render_result(TiledPointsRenderResult(3, 0, True))
        worker.read_viewport_snapshot(replace(outside, request_generation=4))
        worker.acknowledge_render_result(TiledPointsRenderResult(4, 0, True))
        worker.read_viewport_snapshot(replace(first_request, request_generation=5))
        assert snapshots[-1].render_batch is not first.render_batch
        # Decoded tiles can avoid IO on a real replacement; packed batches are
        # not a history cache. This fake uses identical tiles for both rectangles.
        assert len(probe.viewport_reads) == 1
        worker.acknowledge_render_result(TiledPointsRenderResult(5, 0, True))
        worker.read_viewport_snapshot(
            replace(
                first_request,
                request_generation=6,
                viewport=replace(first_request.viewport, hard_render_point_budget=1),
            )
        )
        assert not snapshots[-1].within_hard_limits
        assert snapshots[-1].rendered_point_count == 0
        worker.acknowledge_render_result(TiledPointsRenderResult(6, 0, True))
        assert worker._retained_viewport.snapshot.request_generation == 5
        worker.update_selected_value_index((0,))
        assert worker.retained_render_batch_bytes == worker.pending_render_batch_bytes == 0
    finally:
        worker.close()


@pytest.mark.parametrize(
    "invalidates",
    ["lod", "point_capacity", "vertex_capacity", "cache_generation", "selection", "outside_original_viewport"],
)
def test_retained_viewport_reports_reuse_rejection_reason(invalidates) -> None:
    probe = _ReaderProbe()
    reader = _ControllableReader(Path("unused"), probe)
    request = _viewport_request(1)
    snapshot = cache_session_module._read_viewport_snapshot(
        reader,
        None,
        cache_session_module._CpuTileResidency(1000),
        request,
        viewport=cache_session_module._IntrinsicViewport(0, 0, 30, 10),
        level_selection=reader.select_level(request.viewport, 100, value_index=None),
        budget_message=None,
        max_vertex_payload_bytes=1000,
        raise_if_cancelled=lambda: None,
    )
    retained = cache_session_module._RetainedViewport(cache_session_module._IntrinsicViewport(0, 0, 30, 10), snapshot)
    kwargs = {"cache_generation_id": snapshot.cache_generation_id, "level": 0, "max_vertex_payload_bytes": 1000}
    inner = replace(request, request_generation=2, viewport=replace(request.viewport, x_max=10.0))
    if invalidates == "lod":
        kwargs["level"] = 1
    elif invalidates == "point_capacity":
        inner = replace(inner, viewport=replace(inner.viewport, hard_render_point_budget=1))
    elif invalidates == "vertex_capacity":
        kwargs["max_vertex_payload_bytes"] = 12
    elif invalidates == "cache_generation":
        kwargs["cache_generation_id"] = "87654321-4321-6789-a234-678943216789"
    elif invalidates == "selection":
        inner = replace(inner, selection_generation=1, requested_value_ids=(0,))
    else:
        inner = replace(inner, viewport=replace(inner.viewport, x_min=-0.01))
    assert retained.rejection_reason(inner, **kwargs) == invalidates


@pytest.mark.parametrize("phase", ["packing", "selection"])
def test_failed_preparation_does_not_discard_the_accepted_batch(phase, monkeypatch) -> None:
    probe = _ReaderProbe()
    worker = _TiledPointsCacheWorker(
        Path("unused"),
        _CacheSessionSettings(None, 1000, 1000),
        threading.Event(),
        lambda path: _ControllableReader(path, probe),
    )
    snapshots = []
    failures = []
    worker.viewport_ready.connect(snapshots.append)
    worker.viewport_failed.connect(lambda _generation, failure: failures.append(failure))
    worker.failed.connect(failures.append)
    worker.start()
    first_request = _viewport_request(1)
    try:
        worker.read_viewport_snapshot(first_request)
        first = snapshots[-1]
        worker.acknowledge_render_result(TiledPointsRenderResult(1, 0, True))
        if phase == "packing":

            def fail_pack(*args, **kwargs):
                raise RuntimeError("packing failed")

            monkeypatch.setattr(cache_session_module, "pack_render_tiles", fail_pack)
            worker.read_viewport_snapshot(
                replace(first_request, request_generation=2, viewport=replace(first_request.viewport, x_max=40.0))
            )
        else:
            probe.fail_selection = True
            worker.update_selected_value_index((0,))
        assert len(failures) == 1
        assert worker.pending_render_batch_bytes == 0
        worker.read_viewport_snapshot(replace(first_request, request_generation=3))
        assert snapshots[-1].request_generation == 3
        assert snapshots[-1].render_batch is first.render_batch
        assert worker.retained_render_batch_bytes == 24
    finally:
        worker.close()


def test_smaller_contained_payload_is_repacked_after_point_budget_reduction() -> None:
    probe = _ReaderProbe()
    worker = _TiledPointsCacheWorker(
        Path("unused"),
        _CacheSessionSettings(None, 1000, 1000),
        threading.Event(),
        lambda path: _ControllableReader(path, probe),
    )
    snapshots = []
    worker.viewport_ready.connect(snapshots.append)
    worker.start()
    request = _viewport_request(1)
    try:
        worker.read_viewport_snapshot(request)
        first = snapshots[-1]
        worker.acknowledge_render_result(TiledPointsRenderResult(1, 0, True))
        probe.planned_tile_x = (0,)
        inner = replace(
            request,
            request_generation=2,
            viewport=replace(
                request.viewport,
                x_max=10.0,
                hard_render_point_budget=1,
            ),
        )
        worker.read_viewport_snapshot(inner)
        result = snapshots[-1]
        assert result.request_generation == 2
        assert result.render_batch is not first.render_batch
        assert result.rendered_point_count == result.estimated_point_count == 1
        assert len(probe.viewport_reads) == 1  # replacement can still reuse decoded tiles
        worker.acknowledge_render_result(TiledPointsRenderResult(2, 0, True))
        assert worker._retained_viewport.bounds.x_max == inner.viewport.x_max
    finally:
        worker.close()


def test_contained_viewport_rebuilds_when_selected_lod_changes(monkeypatch) -> None:
    """Rebuild a contained viewport when its selected LOD changes.

    Accept viewport A at Bridge level, then request a smaller viewport B
    inside A at Exact level. B must receive a new batch: containment alone
    does not permit reuse of points from a different LOD.

    LOD selection is controlled by the test; this checks the worker's
    response to a level change, not the LOD-selection algorithm.
    """
    probe = _ReaderProbe()
    worker = _TiledPointsCacheWorker(
        Path("unused"),
        _CacheSessionSettings(None, 1000, 1000),
        threading.Event(),
        lambda path: _ControllableReader(path, probe),
    )
    snapshots = []
    worker.viewport_ready.connect(snapshots.append)
    worker.start()
    reader = worker._reader
    reader.dataset_info = replace(reader.dataset_info, levels=(_FakeLevelInfo(), _FakeLevelInfo("bridge")))
    original_select = reader.select_level
    levels = iter((1, 0))
    lod_calls = []

    def select(viewport, point_budget, **kwargs):
        lod_calls.append(viewport)
        return replace(original_select(viewport, point_budget, **kwargs), level=next(levels))

    monkeypatch.setattr(reader, "select_level", select)
    request = _viewport_request(1)
    try:
        worker.read_viewport_snapshot(request)
        first = snapshots[-1]
        assert first.level == 1
        worker.acknowledge_render_result(TiledPointsRenderResult(1, 0, True))
        inner = replace(request, request_generation=2, viewport=replace(request.viewport, x_max=10.0))
        worker.read_viewport_snapshot(inner)
        replacement = snapshots[-1]
        assert replacement.request_generation == 2
        assert replacement.level == 0
        assert replacement.render_batch is not first.render_batch
        assert len(lod_calls) == 2
        assert len(probe.viewport_reads) == 2
        worker.acknowledge_render_result(TiledPointsRenderResult(2, 0, True))
        assert worker._retained_viewport.bounds.x_max == inner.viewport.x_max
    finally:
        worker.close()


def test_session_owns_reader_on_one_worker_thread_and_reuses_selection(qtbot) -> None:
    probe = _ReaderProbe()
    session = _session(probe)
    gui_thread_id = threading.get_ident()
    callback_thread_ids: list[int] = []
    states: list[_CacheSessionState] = []
    session.ready.connect(lambda: callback_thread_ids.append(threading.get_ident()))
    session.value_selection_ready.connect(lambda _selection, _bytes: callback_thread_ids.append(threading.get_ident()))
    session.state_changed.connect(states.append)

    try:
        _start_ready(session, qtbot)
        assert session.resident_index_bytes == 64

        with qtbot.waitSignal(session.value_selection_ready, timeout=5_000):
            assert session.set_selected_value_ids((0,))
        qtbot.waitUntil(lambda: session.state is _CacheSessionState.READY)
        assert session.selected_value_ids == (0,)
        assert not session.set_selected_value_ids((0,))
        assert probe.selection_calls == [(0,)]

        # Production normalizes the complete vocabulary to the all-values
        # state and drops the selected index without another catalog load.
        with qtbot.waitSignal(session.value_selection_ready, timeout=5_000):
            assert session.set_selected_value_ids((0, 1, 2))
        qtbot.waitUntil(lambda: session.state is _CacheSessionState.READY)
        assert session.selected_value_ids is None
        assert probe.selection_calls == [(0,)]
    finally:
        _close(session, qtbot)

    worker_ids = {thread_id for _, thread_id in probe.operations}
    assert len(worker_ids) == 1
    assert worker_ids != {gui_thread_id}
    assert callback_thread_ids and set(callback_thread_ids) == {gui_thread_id}
    assert [operation for operation, _ in probe.operations] == [
        "construct",
        "enter",
        "load_selection",
        "exit",
    ]
    assert states == [
        _CacheSessionState.STARTING,
        _CacheSessionState.READY,
        _CacheSessionState.UPDATING_SELECTED_VALUE_INDEX,
        _CacheSessionState.READY,
        _CacheSessionState.UPDATING_SELECTED_VALUE_INDEX,
        _CacheSessionState.READY,
        _CacheSessionState.CLOSING,
        _CacheSessionState.CLOSED,
    ]


def test_session_reports_compact_metadata_startup_failure(qtbot) -> None:
    probe = _ReaderProbe(fail_entry=True)
    session = _session(probe)
    failures: list[_CacheSessionFailure] = []
    session.failed.connect(failures.append)

    with qtbot.waitSignal(session.closed, timeout=5_000):
        session.start()

    assert session.state is _CacheSessionState.CLOSED
    assert len(failures) == 1
    assert failures[0].phase == "startup"
    assert failures[0].message == "invalid compact metadata"
    assert [operation for operation, _ in probe.operations] == ["construct", "enter"]


def test_session_propagates_absent_selection_limit(qtbot) -> None:
    probe = _ReaderProbe(resident_bytes=2_000)
    session = _session(
        probe,
        max_selected_value_index_bytes=None,
    )

    try:
        _start_ready(session, qtbot)
        with qtbot.waitSignal(session.value_selection_ready, timeout=5_000):
            session.set_selected_value_ids((0,))
        qtbot.waitUntil(lambda: session.state is _CacheSessionState.READY)
        assert probe.selected_value_limits == [None]
    finally:
        _close(session, qtbot)


def test_selection_failure_retains_previous_ready_selection(qtbot) -> None:
    probe = _ReaderProbe()
    session = _session(probe)
    failures: list[_CacheSessionFailure] = []
    session.failed.connect(failures.append)

    try:
        _start_ready(session, qtbot)
        with qtbot.waitSignal(session.value_selection_ready, timeout=5_000):
            session.set_selected_value_ids((0,))
        qtbot.waitUntil(lambda: session.state is _CacheSessionState.READY)

        probe.fail_selection = True
        with qtbot.waitSignal(session.failed, timeout=5_000):
            session.set_selected_value_ids((1,))
        qtbot.waitUntil(lambda: session.state is _CacheSessionState.READY)

        assert failures[-1].phase == "selection"
        assert session.selected_value_ids == (0,)
        assert probe.selection_calls == [(0,), (1,)]
    finally:
        _close(session, qtbot)


def test_worker_treats_selection_without_reader_as_fatal() -> None:
    worker = _TiledPointsCacheWorker(
        Path("unused.zarr"),
        _CacheSessionSettings(
            max_selected_value_index_bytes=None,
            max_cpu_tile_bytes=1_000,
            max_vertex_payload_bytes=1_000_000,
        ),
        threading.Event(),
        lambda cache_root: _ControllableReader(cache_root, _ReaderProbe()),
    )
    states: list[_CacheSessionState] = []
    failures: list[_CacheSessionFailure] = []
    finished: list[None] = []
    worker.state_changed.connect(states.append)
    worker.failed.connect(failures.append)
    worker.finished.connect(lambda: finished.append(None))

    worker.update_selected_value_index((0,))

    assert states == [_CacheSessionState.FAILED]
    assert len(failures) == 1
    assert failures[0].phase == "selection"
    assert failures[0].message == "Cache reader is not ready."
    assert finished == [None]


def test_close_during_compact_metadata_loading_rolls_into_owner_thread_shutdown(qtbot) -> None:
    probe = _ReaderProbe(pause_entry=True)
    session = _session(probe)
    ready_events: list[None] = []

    session.ready.connect(lambda: ready_events.append(None))
    session.start()
    qtbot.waitUntil(probe.entry_paused.is_set, timeout=5_000)
    session.close()
    with qtbot.waitSignal(session.closed, timeout=5_000):
        probe.resume_entry.set()

    assert probe.entry_paused.is_set()
    assert ready_events == []
    assert session.state is _CacheSessionState.CLOSED
    assert [operation for operation, _ in probe.operations][-1] == "exit"


def test_close_during_selected_index_load_does_not_publish_late_selection(qtbot) -> None:
    probe = _ReaderProbe(pause_selection=True)
    session = _session(probe)
    published: list[object] = []
    session.value_selection_ready.connect(lambda selection, _bytes: published.append(selection))

    _start_ready(session, qtbot)
    session.set_selected_value_ids((0,))
    qtbot.waitUntil(probe.selection_paused.is_set, timeout=5_000)
    assert session.close()
    probe.resume_selection.set()
    qtbot.waitUntil(lambda: session.state is _CacheSessionState.CLOSED, timeout=5_000)

    assert published == []
    assert [operation for operation, _ in probe.operations][-1] == "exit"


def test_close_during_reader_construction_prevents_reader_entry(qtbot) -> None:
    probe = _ReaderProbe(pause_construction=True)
    session = _session(probe)

    session.start()
    qtbot.waitUntil(probe.construction_paused.is_set, timeout=5_000)
    assert session.close()
    probe.resume_construction.set()
    qtbot.waitUntil(lambda: session.state is _CacheSessionState.CLOSED, timeout=5_000)

    assert [operation for operation, _ in probe.operations] == ["construct"]


def test_started_session_outlives_dropped_owner_until_its_thread_stops(qtbot) -> None:
    """Guard the session keep-alive (`_SESSIONS_WITH_RUNNING_THREADS`).

    `start()` adds the session to that set; `_on_thread_finished()` removes it
    (deferred with `QTimer.singleShot`) once the worker thread has fully stopped.
    Owners may therefore drop a closed session while its worker is still busy,
    as layer removal does. Without the keep-alive, garbage collection would
    destroy the session and its still-running QThread, and Qt aborts the process.
    """
    probe = _ReaderProbe(pause_entry=True)
    session = _session(probe)
    destroyed: list[bool] = []
    session.destroyed.connect(lambda *_: destroyed.append(True))

    # 1. A started session is kept alive. Checked first: if the keep-alive
    #    regressed, dropping the session below would abort the whole test
    #    process instead of failing this test.
    session.start()
    qtbot.waitUntil(probe.entry_paused.is_set, timeout=5_000)
    assert session in cache_session_module._SESSIONS_WITH_RUNNING_THREADS

    # 2. The owner closes and drops the session while the worker is busy:
    #    the keep-alive is now the only reference, so the session survives.
    session.close()
    del session
    gc.collect()
    assert destroyed == []

    # 3. Once the worker thread stops, the keep-alive is released and the
    #    session is freed, so nothing leaks.
    def session_released() -> bool:
        gc.collect()  # the keep-alive was the last reference
        return bool(destroyed)

    probe.resume_entry.set()
    qtbot.waitUntil(session_released, timeout=5_000)


def test_passive_session_closes_idempotently_without_starting_reader(qtbot) -> None:
    probe = _ReaderProbe()
    session = _session(probe)

    assert session.state is _CacheSessionState.NEW
    assert probe.operations == []
    with qtbot.waitSignal(session.closed, timeout=1_000):
        assert session.close()
    assert not session.close()
    assert session.state is _CacheSessionState.CLOSED
    assert probe.operations == []
    with pytest.raises(RuntimeError, match="started only from NEW"):
        session.start()


def test_session_reads_only_nonresident_tiles_and_returns_complete_snapshots(qtbot) -> None:
    probe = _ReaderProbe(planned_tile_x=(0, 1))
    session = _session(probe, max_cpu_tile_bytes=24)
    snapshots: list[object] = []
    session.viewport_ready.connect(snapshots.append)

    try:
        _start_ready(session, qtbot)
        with qtbot.waitSignal(session.viewport_ready, timeout=5_000):
            session.request_viewport(_viewport_request(1))

        probe.planned_tile_x = (1, 2)
        with qtbot.waitSignal(session.viewport_ready, timeout=5_000):
            session.request_viewport(_viewport_request(2))

        assert probe.viewport_reads == [((0, 0, 0), (0, 1, 0)), ((0, 2, 0),)]
        assert snapshots[-1].rendered_tile_count == 2
        assert snapshots[-1].rendered_point_count == 2
        np.testing.assert_array_equal(
            snapshots[-1].render_batch.vertices["a_position"],
            np.asarray(((11.0, 0.0), (22.0, 0.0)), dtype=np.float32),
        )
    finally:
        _close(session, qtbot)


def test_session_residency_detaches_tiles_from_shared_reader_batches(qtbot) -> None:
    probe = _ReaderProbe(planned_tile_x=(0, 1))
    session = _session(probe, max_cpu_tile_bytes=1_000)
    snapshots: list[object] = []
    session.viewport_ready.connect(snapshots.append)

    try:
        _start_ready(session, qtbot)
        with qtbot.waitSignal(session.viewport_ready, timeout=5_000):
            session.request_viewport(_viewport_request(1))

        location_batch = probe.last_location_batch
        value_id_batch = probe.last_value_id_batch
        assert location_batch is not None
        assert value_id_batch is not None
        first_vertices = snapshots[-1].render_batch.vertices.copy()
        location_batch[:] = -100
        value_id_batch[:] = 2

        with qtbot.waitSignal(session.viewport_ready, timeout=5_000):
            session.request_viewport(_viewport_request(2))

        assert probe.viewport_reads == [((0, 0, 0), (0, 1, 0))]
        np.testing.assert_array_equal(snapshots[-1].render_batch.vertices, first_vertices)
    finally:
        _close(session, qtbot)


def test_session_rejects_over_budget_viewport_before_point_io(qtbot) -> None:
    probe = _ReaderProbe()
    session = _session(probe)
    snapshots: list[object] = []
    session.viewport_ready.connect(snapshots.append)

    try:
        _start_ready(session, qtbot)
        with qtbot.waitSignal(session.viewport_ready, timeout=5_000):
            request = _viewport_request(1)
            session.request_viewport(replace(request, viewport=replace(request.viewport, hard_render_point_budget=1)))

        assert not snapshots[-1].within_hard_limits
        assert snapshots[-1].rendered_tile_count == 0
        assert snapshots[-1].render_batch.point_count == 0
        assert probe.viewport_reads == []
    finally:
        _close(session, qtbot)


def test_oversized_tile_is_returned_transiently_but_not_retained(qtbot) -> None:
    probe = _ReaderProbe(planned_tile_x=(0,))
    session = _session(probe, max_cpu_tile_bytes=1)
    snapshots: list[object] = []
    session.viewport_ready.connect(snapshots.append)

    try:
        _start_ready(session, qtbot)
        for generation in (1, 2):
            with qtbot.waitSignal(session.viewport_ready, timeout=5_000):
                session.request_viewport(_viewport_request(generation))

        assert len(snapshots) == 2
        assert all(snapshot.rendered_point_count == 1 for snapshot in snapshots)
        assert probe.viewport_reads == [((0, 0, 0),), ((0, 0, 0),)]
    finally:
        _close(session, qtbot)


def test_worker_reports_vertex_payload_limit_before_point_io(qtbot) -> None:
    probe = _ReaderProbe(planned_tile_x=(0, 1))
    session = _session(probe, max_vertex_payload_bytes=12)
    failures: list[_CacheSessionFailure] = []
    snapshots: list[object] = []
    session.viewport_failed.connect(lambda _generation, failure: failures.append(failure))
    session.viewport_ready.connect(snapshots.append)

    try:
        _start_ready(session, qtbot)
        with qtbot.waitSignal(session.viewport_ready, timeout=5_000):
            session.request_viewport(_viewport_request(1))

        assert failures == []
        assert len(snapshots) == 1
        assert not snapshots[0].within_hard_limits
        assert snapshots[0].rendered_point_count == 0
        assert "24 vertex bytes required, limit 12 bytes" in snapshots[0].budget_message
        assert probe.viewport_reads == []
    finally:
        _close(session, qtbot)


def test_worker_rejects_nonspatial_tile_order_before_packing(qtbot, monkeypatch: pytest.MonkeyPatch) -> None:
    probe = _ReaderProbe(planned_tile_x=(1, 0))
    session = _session(probe)
    failures: list[_CacheSessionFailure] = []
    pack_called = False

    def _unexpected_pack(*args, **kwargs):
        nonlocal pack_called
        del args, kwargs
        pack_called = True
        raise AssertionError("packing should not be attempted")

    monkeypatch.setattr(cache_session_module, "pack_render_tiles", _unexpected_pack)
    session.viewport_failed.connect(lambda _generation, failure: failures.append(failure))
    try:
        _start_ready(session, qtbot)
        with qtbot.waitSignal(session.viewport_failed, timeout=5_000):
            session.request_viewport(_viewport_request(1))

        assert not pack_called
        assert len(failures) == 1
        assert failures[0].phase == "viewport"
        assert "spatial" in failures[0].message
    finally:
        _close(session, qtbot)


def test_queued_viewport_delivery_preserves_maximum_budget_vertex_allocation_identity(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe = _ReaderProbe(planned_tile_x=tuple(range(100)))
    session = _session(probe, max_cpu_tile_bytes=10_000, max_vertex_payload_bytes=1_200)
    worker_vertex_ids: list[int] = []
    original_pack = cache_session_module.pack_render_tiles

    def _record_pack(*args, **kwargs):
        batch = original_pack(*args, **kwargs)
        worker_vertex_ids.append(id(batch.vertices))
        return batch

    monkeypatch.setattr(cache_session_module, "pack_render_tiles", _record_pack)
    snapshots: list[object] = []
    session.viewport_ready.connect(snapshots.append)
    try:
        _start_ready(session, qtbot)
        with qtbot.waitSignal(session.viewport_ready, timeout=5_000):
            session.request_viewport(_viewport_request(1))

        snapshot = snapshots[-1]
        assert snapshot.render_batch.point_count == 100
        assert snapshot.render_batch.nbytes == 1_200
        assert id(snapshot.render_batch.vertices) == worker_vertex_ids[-1]
    finally:
        _close(session, qtbot)


@pytest.mark.parametrize("pause_check", [1, 2], ids=["before-allocation", "during-fragmented-pack"])
def test_terminal_close_cancels_worker_render_batch_packing(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    pause_check: int,
) -> None:
    probe = _ReaderProbe(planned_tile_x=tuple(range(129)))
    session = _session(probe, max_cpu_tile_bytes=10_000, max_vertex_payload_bytes=10_000)
    pack_paused = threading.Event()
    resume_pack = threading.Event()
    original_pack = cache_session_module.pack_render_tiles

    def _pause_pack(*args, raise_if_cancelled, **kwargs):
        checks = 0

        def _check() -> None:
            nonlocal checks
            checks += 1
            if checks == pause_check:
                pack_paused.set()
                assert resume_pack.wait(timeout=5)
            raise_if_cancelled()

        return original_pack(*args, raise_if_cancelled=_check, **kwargs)

    monkeypatch.setattr(cache_session_module, "pack_render_tiles", _pause_pack)
    snapshots: list[object] = []
    session.viewport_ready.connect(snapshots.append)
    _start_ready(session, qtbot)
    request = _viewport_request(1)
    session.request_viewport(replace(request, viewport=replace(request.viewport, hard_render_point_budget=129)))
    qtbot.waitUntil(pack_paused.is_set, timeout=5_000)
    assert session.close()
    with qtbot.waitSignal(session.closed, timeout=5_000):
        resume_pack.set()

    assert snapshots == []
    assert session.state is _CacheSessionState.CLOSED


def test_real_cache_session_opens_compact_metadata_and_loads_selection(real_cache_root: Path, qtbot) -> None:
    session = _TiledPointsCacheSession(
        real_cache_root,
        _CacheSessionSettings(
            max_selected_value_index_bytes=None,
            max_cpu_tile_bytes=1_000,
            max_vertex_payload_bytes=1_000_000,
        ),
    )
    try:
        _start_ready(session, qtbot)
        assert session.dataset_info is not None
        assert session.dataset_info.value_names == ("A", "B")
        assert session.resident_index_bytes is not None and session.resident_index_bytes > 0

        with qtbot.waitSignal(session.value_selection_ready, timeout=5_000):
            session.set_selected_value_ids((0,))
        qtbot.waitUntil(lambda: session.state is _CacheSessionState.READY)
        assert session.selected_value_ids == (0,)
    finally:
        _close(session, qtbot)


def test_real_cache_session_builds_generation_bound_viewport_snapshot(real_cache_root: Path, qtbot) -> None:
    session = _TiledPointsCacheSession(
        real_cache_root,
        _CacheSessionSettings(
            max_selected_value_index_bytes=None,
            max_cpu_tile_bytes=1_000_000,
            max_vertex_payload_bytes=1_000_000,
        ),
    )
    snapshots: list[object] = []
    session.viewport_ready.connect(snapshots.append)
    try:
        _start_ready(session, qtbot)
        request = _ViewportRequest(
            request_generation=1,
            selection_generation=0,
            requested_value_ids=None,
            viewport=TiledPointsViewportState(
                displayed_axes=(0, 1),
                x_min=0.0,
                y_min=0.0,
                x_max=20.0,
                y_max=10.0,
                canvas_width=100,
                canvas_height=100,
                hard_render_point_budget=100,
                screen_density_budget=100,
            ),
        )
        with qtbot.waitSignal(session.viewport_ready, timeout=5_000):
            session.request_viewport(request)

        snapshot = snapshots[-1]
        assert snapshot.request_generation == 1
        assert snapshot.within_hard_limits
        assert snapshot.rendered_point_count == 4
        assert snapshot.rendered_tile_count == 2
    finally:
        _close(session, qtbot)
