"""Own one long-lived points-cache reader on a dedicated Qt thread."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from enum import StrEnum
from functools import partial
from pathlib import Path
from typing import Literal

import numpy as np
from loguru import logger
from qtpy.QtCore import QObject, QThread, QTimer, Signal, Slot

from spatiato.core.multi_scale_cache_points_zarr.models import _expected_level_kind
from spatiato.core.multi_scale_cache_points_zarr.reader import (
    _CacheDatasetInfo,
    _IntrinsicViewport,
    _LevelSelection,
    _PointsCacheReader,
    _SelectedValueIndex,
)
from spatiato.viewer.tiled_points.contracts import (
    TILED_POINTS_VERTEX_DTYPE,
    TiledPointsRenderBatch,
    TiledPointsRenderResult,
    TiledPointsRenderSnapshot,
    TiledPointsRenderTile,
    TileResidencyKey,
    _ViewportRequest,
)
from spatiato.viewer.tiled_points.render_batch import pack_render_tiles
from spatiato.viewer.tiled_points.runtime.lod import _LodDecisionReason, _select_lod
from spatiato.viewer.tiled_points.runtime.residency import _CpuTileResidency

_UINT32_MAX = np.iinfo(np.uint32).max
_FailurePhase = Literal["startup", "selection", "viewport", "shutdown"]
_ReaderFactory = Callable[[Path], _PointsCacheReader]


class _CacheSessionState(StrEnum):
    """Identify one GUI-visible cache-session lifecycle state."""

    NEW = "new"
    STARTING = "starting"
    READY = "ready"
    UPDATING_SELECTED_VALUE_INDEX = "updating_selected_value_index"
    FAILED = "failed"
    CLOSING = "closing"
    CLOSED = "closed"


@dataclass(frozen=True)
class _CacheSessionSettings:
    """Bound metadata, decoded tiles, and packed batches in one cache session.

    Parameters
    ----------
    max_selected_value_index_bytes
        Maximum resident bytes for the current selected-value catalog index.
        ``None`` disables this configured preflight limit.
    max_cpu_tile_bytes
        Positive byte limit for the evicting decoded point-payload LRU.
    max_vertex_payload_bytes
        Byte limit for each worker-prepared renderer vertex payload, large
        enough to hold at least one vertex, including the payload retained after
        activation. A pending replacement may temporarily coexist with the
        accepted batch. These allocations are accounted separately from
        ``max_cpu_tile_bytes`` and the single VBO.
    """

    max_selected_value_index_bytes: int | None
    max_cpu_tile_bytes: int
    max_vertex_payload_bytes: int

    def __post_init__(self) -> None:
        value = self.max_selected_value_index_bytes
        if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value <= 0):
            raise ValueError("`max_selected_value_index_bytes` must be a positive integer or None.")
        value = self.max_cpu_tile_bytes
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError("`max_cpu_tile_bytes` must be a positive integer.")
        _vertex_point_capacity(self.max_vertex_payload_bytes)


def _vertex_point_capacity(max_vertex_payload_bytes: int) -> int:
    """Validate the byte limit and return a capacity of at least one vertex."""
    vertex_bytes = TILED_POINTS_VERTEX_DTYPE.itemsize
    if (
        not isinstance(max_vertex_payload_bytes, int)
        or isinstance(max_vertex_payload_bytes, bool)
        or max_vertex_payload_bytes < vertex_bytes
    ):
        raise ValueError(
            f"`max_vertex_payload_bytes` must be an integer of at least {vertex_bytes} bytes to hold one vertex."
        )
    return max_vertex_payload_bytes // vertex_bytes


@dataclass(frozen=True)
class _CacheSessionFailure:
    """Describe one worker failure without transporting a live traceback."""

    phase: _FailurePhase
    exception_type: str
    message: str

    def __post_init__(self) -> None:
        if self.phase not in (
            "startup",
            "selection",
            "viewport",
            "shutdown",
        ):
            raise ValueError("Unsupported cache-session failure phase.")
        for name in ("exception_type", "message"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"`{name}` must be a nonempty string.")


class _SessionCancelled(RuntimeError):
    """Stop active worker work without reporting cancellation as a failure."""


@dataclass(frozen=True)
class _RetainedViewport:
    """Pair a render snapshot with the original viewport requested for its batch.

    Parameters
    ----------
    bounds
        Rectangle in intrinsic coordinates taken from ``_ViewportRequest.viewport``
        when the render batch was first prepared. It includes empty space within
        that requested viewport; it is not a bounding box calculated from tiles
        or point positions. Reusing the batch for a contained viewport preserves
        this original rectangle rather than replacing it with the smaller view.
    snapshot
        Render snapshot containing the packed ``render_batch``, cache and selection
        identity, LOD, and request/status metadata. On contained-view reuse, a new
        snapshot carries the incoming request's generations, visible estimate and
        diagnostics while sharing the original batch. Its request metadata may
        therefore describe a smaller viewport than ``bounds``.

    Notes
    -----
    The worker stores one accepted entry in ``_retained_viewport``. A candidate
    in ``_pending_viewport`` may temporarily coexist with it until the GUI accepts
    or rejects that candidate. These entries are not a viewport-history cache.
    """

    bounds: _IntrinsicViewport
    snapshot: TiledPointsRenderSnapshot

    def rejection_reason(
        self,
        request: _ViewportRequest,
        *,
        cache_generation_id: str,
        level: int,
        max_vertex_payload_bytes: int,
    ) -> str | None:
        """Check identity, full allocation budgets, and rectangle containment."""
        snapshot = self.snapshot
        if snapshot.cache_generation_id != cache_generation_id:
            return "cache_generation"
        if (
            snapshot.selection_generation != request.selection_generation
            or snapshot.requested_value_ids != request.requested_value_ids
        ):
            return "selection"
        if snapshot.level != level:
            return "lod"
        if snapshot.rendered_point_count > request.viewport.hard_render_point_budget:
            return "point_capacity"
        if snapshot.render_batch.nbytes > max_vertex_payload_bytes:
            return "vertex_capacity"
        view = request.viewport
        bounds = self.bounds
        if not (
            bounds.x_min <= view.x_min <= view.x_max <= bounds.x_max
            and bounds.y_min <= view.y_min <= view.y_max <= bounds.y_max
        ):
            return "outside_original_viewport"
        return None


class _TiledPointsCacheWorker(QObject):
    """Own cache IO and reader state on one dedicated Qt worker thread.

    The GUI-thread ``_TiledPointsCacheSession`` constructs this object, moves it
    to a ``QThread``, and then starts that thread. Reader construction happens in
    :meth:`start`, after the move, so the reader and all opened Zarr resources
    are created, used, and closed on the worker thread. Neither the live reader
    nor the resident selected-value index crosses the GUI facade.

    Communication in both directions uses Qt signals and slots::

        GUI thread                              worker thread
        ----------                              -------------
        session.start()       --------------->  start()
        selection requested  --------------->  update_selected_value_index()
        viewport requested   --------------->  read_viewport_snapshot()
        renderer result      --------------->  acknowledge_render_result()
        close requested      --------------->  close()

        session handlers     <---------------  state/progress/result signals
        thread.quit()        <---------------  finished

    These connections are queued according to QObject thread affinity. They
    keep cache operations off the GUI thread while returning only immutable
    descriptions and snapshots, selected value IDs, byte counts, and structured
    failures.

    Notes
    -----
    The session and worker share the thread-safe ``cancellation`` event supplied
    to ``__init__``. Closing needs both this event and a queued ``close()`` slot::

        GUI thread                         worker thread
        ----------                         -------------
        session.close()
          cancellation.set()  ---------->  _require_not_cancelled()
          emit close signal                raises _SessionCancelled
                                                    |
                                                    v
                                            close reader and finish

    A busy worker cannot execute the queued ``close()`` slot until its current
    slot returns. Cancellation checkpoints let that active operation observe the
    event, raise ``_SessionCancelled``, and enter cleanup without publishing a
    late result. This is cooperative cancellation: it stops between operations
    or bucket reads, but it cannot interrupt a Zarr call already in progress.

    Conversely, an idle worker has no active checkpoint at which to observe the
    passive event. The queued ``close()`` slot therefore remains necessary to
    execute cleanup on the reader's owning thread. Active cancellation and idle
    closure both converge on idempotent :meth:`_shutdown`, which closes the
    reader once, clears worker-resident state, and emits ``finished`` so the
    session can terminate the thread and publish ``CLOSED``.
    """

    state_changed = Signal(object)
    dataset_available = Signal(object)
    ready = Signal(int)
    value_selection_ready = Signal(object, int)
    viewport_ready = Signal(object)
    viewport_failed = Signal(int, object)
    failed = Signal(object)
    finished = Signal()

    def __init__(
        self,
        cache_root: Path,
        settings: _CacheSessionSettings,
        cancellation: threading.Event,
        reader_factory: _ReaderFactory,
    ) -> None:
        super().__init__()
        self._cache_root = cache_root
        self._settings = settings
        self._cancellation = cancellation
        self._reader_factory = reader_factory
        self._reader: _PointsCacheReader | None = None
        self._selected_value_ids: tuple[int, ...] | None = None
        self._selected_value_index: _SelectedValueIndex | None = None
        self._cpu_tile_residency = _CpuTileResidency(settings.max_cpu_tile_bytes)
        self._retained_viewport: _RetainedViewport | None = None
        self._pending_viewport: _RetainedViewport | None = None
        self._finished = False

    @property
    def retained_render_batch_bytes(self) -> int:
        """Account the accepted packed allocation separately from CPU tile LRU bytes."""
        retained = self._retained_viewport
        return 0 if retained is None else retained.snapshot.render_batch.nbytes

    @property
    def pending_render_batch_bytes(self) -> int:
        """Account a transient candidate allocation, excluding an alias of the retained batch."""
        pending = self._pending_viewport
        retained = self._retained_viewport
        if pending is None or (
            retained is not None and pending.snapshot.render_batch is retained.snapshot.render_batch
        ):
            return 0
        return pending.snapshot.render_batch.nbytes

    @Slot()
    def start(self) -> None:
        """Load compact planning metadata before announcing readiness.

        Bucket stores open lazily on complete-tile reads. Neither startup nor
        selected-value viewport reads project or load bucket sparse ranges.
        """
        try:
            self._require_not_cancelled()
            # Construct the reader here so it and all opened Zarr resources are
            # owned by the worker thread.
            reader = self._reader_factory(self._cache_root)
            self._require_not_cancelled()
            reader.__enter__()
            self._reader = reader
            self.dataset_available.emit(reader.dataset_info)

            self._require_not_cancelled()
            self.state_changed.emit(_CacheSessionState.READY)
            self.ready.emit(reader.resident_index_bytes)
        except _SessionCancelled:
            self._shutdown(emit_closing=True)
        except Exception as error:  # noqa: BLE001
            logger.exception("Tiled-points cache session failed during startup.")
            self.state_changed.emit(_CacheSessionState.FAILED)
            self.failed.emit(_failure_from_exception("startup", error))
            self._shutdown(emit_closing=False)

    @Slot(object)
    def update_selected_value_index(self, requested_value_ids: tuple[int, ...] | None) -> None:
        """Update the worker-resident selected-value index for a new selection.

        This operation does not necessarily load data. An unchanged normalized
        selection reuses the current index, while the all-values selection
        (``None``) clears that index. Only a changed proper subset loads a new
        selected-value index from the catalog. The replacement is committed
        only after loading succeeds, so a recoverable failure leaves the
        previous selection active.

        The committed worker state has exactly two canonical forms::

            all values:
                _selected_value_ids = None
                _selected_value_index = None

            proper subset:
                _selected_value_ids = tuple[int, ...]
                _selected_value_index = _SelectedValueIndex

        A complete-vocabulary tuple is normalized to the all-values form before
        either field is updated.
        """
        if self._finished:
            return
        reader = self._reader
        if reader is None:
            error = RuntimeError("Cache reader is not ready.")
            logger.error("Tiled-points worker received a selection request without a live cache reader.")
            self.state_changed.emit(_CacheSessionState.FAILED)
            self.failed.emit(_failure_from_exception("selection", error))
            self._shutdown(emit_closing=False)
            return
        try:
            self._require_not_cancelled()
            requested_value_ids = _normalize_all_values(
                requested_value_ids,
                value_count=len(reader.value_names),
            )
            if requested_value_ids == self._selected_value_ids:
                resident_bytes = 0 if self._selected_value_index is None else self._selected_value_index.resident_bytes
                self.state_changed.emit(_CacheSessionState.READY)
                self.value_selection_ready.emit(requested_value_ids, resident_bytes)
                return

            if requested_value_ids is None:
                value_index = None
            else:
                value_index = reader.load_selected_value_index(
                    np.asarray(requested_value_ids, dtype=np.uint32),
                    max_resident_bytes=self._settings.max_selected_value_index_bytes,
                )

            self._require_not_cancelled()
            # Commit the normalized selection and its index as one canonical
            # pair: None/None for all values, or tuple/_SelectedValueIndex for
            # a proper subset.
            self._selected_value_ids = requested_value_ids
            self._selected_value_index = value_index
            self._retained_viewport = None
            self._pending_viewport = None
            resident_bytes = 0 if value_index is None else value_index.resident_bytes
            self.state_changed.emit(_CacheSessionState.READY)
            self.value_selection_ready.emit(requested_value_ids, resident_bytes)
        except _SessionCancelled:
            self._shutdown(emit_closing=True)
        except Exception as error:  # noqa: BLE001
            logger.exception("Tiled-points cache session failed while loading a selected-value index.")
            self._report_recoverable_selection_failure(error)

    @Slot()
    def close(self) -> None:
        """Close the reader and finish this worker exactly once."""
        self._shutdown(emit_closing=True)

    @Slot(object)
    def read_viewport_snapshot(self, request: _ViewportRequest) -> None:
        """Evaluate one viewport, reuse or prepare its batch, and publish a candidate.

        Budget policy and LOD selection
        -------------------------------
        The three limits have different roles::

            hard_render_point_budget   hard limit on the number of points
            max_vertex_payload_bytes   hard byte limit per packed vertex payload
            screen_density_budget      soft preferred point count for the canvas

        ``hard_point_capacity`` is the smaller of the hard point limit and the
        number of vertices that fit the byte limit. ``preferred_point_budget``
        additionally caps that capacity by the screen-density target. Estimates
        count selected points in complete intersecting tiles, not individually
        clipped points.

        Without compatible accepted history, select the finest LOD meeting the
        preferred target. Subsequent requests use hysteresis: a finer level can
        remain above that target within a bounded tolerance, while refinement
        requires crossing a lower threshold. All estimates concern the incoming
        viewport, not the old packed allocation. If no preferred fit or hysteresis
        choice is available, use the coarsest level only within the hard limits.
        For example, with a preferred target of 50,000 points and a hard capacity of
        100,000, a coarsest level containing 80,000 points is allowed, with an
        informational density message. A hard-limit rejection instead publishes a
        metadata-only snapshot explaining the limit, without reading point payloads.

        Accordingly, ``level_selection.fits_point_budget`` describes the reader's fit
        against the preferred target, whereas the published snapshot's
        ``within_hard_limits`` describes whether the hard limits permit rendering.

        Retained render-batch reuse
        --------------------------
        LOD selection and the hard-limit decision happen before considering reuse::

            current LOD and hard-limit checks
                            |
                 compatible accepted batch?
                            |
                            +-- yes: reuse packed batch with fresh metadata
                            |
                            +-- no: plan tiles, reuse CPU-resident tiles,
                                    read misses, pack a new batch

        Reuse requires the same cache generation, value selection and selected LOD,
        with the requested rectangle contained in the original accepted viewport.
        The entire retained batch must still fit both hard limits, including its
        off-screen points. Reuse skips tile planning, CPU tile lookups, payload IO
        and packing; request generations, visible estimates and diagnostics refresh.

        For example, with the same cache, selection and LOD, and valid hard limits::

            A: original retained viewport          C: disjoint viewport
            +---------------------------+          +------------------+
            |                           |          |                  |
            |     +---------------+     |          |                  |
            |     | B: zoomed-in   |     |          |                  |
            |     | viewport      |     |          |                  |
            |     +---------------+     |          |                  |
            |                           |          |                  |
            +---------------------------+          +------------------+

            A -> B -> A: reuse A's packed batch throughout;
                         its original bounds stay unchanged.

            A -> C -> A: accepting C replaces A's retained entry;
                         returning to A requires a new packed batch.

        This is one retained batch, not a viewport-history cache. Returning to A
        may still reuse decoded CPU tiles even though its packed batch was replaced.

        This worker method owns the reuse decision and pending entry. Only the
        replacement branch calls ``_read_viewport_snapshot()``, which receives the
        already selected LOD and knows nothing about retained viewports.
        ``acknowledge_render_result()`` promotes a candidate only after acceptance.
        """
        if self._finished:
            return
        reader = self._reader
        if reader is None:
            self._report_viewport_failure(request, RuntimeError("Cache reader is not ready."))
            return
        try:
            self._require_not_cancelled()
            # Production acknowledges every completion before dispatching the
            # next request. Never promote an unacknowledged candidate implicitly.
            self._pending_viewport = None
            if not isinstance(request, _ViewportRequest):
                raise ValueError("`request` must be _ViewportRequest.")
            if request.requested_value_ids != self._selected_value_ids:
                raise ValueError("Viewport request value IDs do not match the worker's committed selection.")
            max_vertex_payload_bytes = self._settings.max_vertex_payload_bytes
            # 1. Evaluate the current viewport's LOD and rendering limits once.
            viewport = _IntrinsicViewport(
                request.viewport.x_min,
                request.viewport.y_min,
                request.viewport.x_max,
                request.viewport.y_max,
            )
            hard_point_capacity = min(
                request.viewport.hard_render_point_budget,
                _vertex_point_capacity(max_vertex_payload_bytes),
            )
            preferred_point_budget = min(request.viewport.effective_point_budget, hard_point_capacity)
            level_selection, decision_reason = self._select_viewport_level(
                reader,
                request,
                viewport=viewport,
                preferred_point_budget=preferred_point_budget,
                hard_point_capacity=hard_point_capacity,
            )
            self._require_not_cancelled()
            # The policy returns evidence for the chosen level, including any
            # tolerated density overrun or coarsest fallback. Check its estimate
            # against hard capacity without another level scan or payload IO.
            point_count = level_selection.estimated_point_count
            within_hard_limits = point_count <= hard_point_capacity
            budget_message = None
            # A hard-limit violation produces a metadata-only TiledPointsRenderSnapshot
            # below, with within_hard_limits=False and an empty render batch. No point
            # payloads are read; the GUI retains the previous view rather than clearing it.
            if not within_hard_limits:
                limits = []
                if point_count > request.viewport.hard_render_point_budget:
                    limits.append(
                        f"{point_count:,} points required, render limit {request.viewport.hard_render_point_budget:,} points"
                    )
                required_bytes = point_count * TILED_POINTS_VERTEX_DTYPE.itemsize
                if required_bytes > max_vertex_payload_bytes:
                    limits.append(f"{required_bytes:,} vertex bytes required, limit {max_vertex_payload_bytes:,} bytes")
                budget_message = "View exceeds hard rendering limits: " + "; ".join(limits)
            elif point_count > request.viewport.screen_density_budget:
                explanation = (
                    "Coarsest level" if decision_reason == "coarsest_density_fallback" else "LOD hysteresis tolerance"
                )
                budget_message = (
                    f"{explanation}; above preferred screen density "
                    f"({point_count:,} points; target {request.viewport.screen_density_budget:,})"
                )
            dataset_info = reader.dataset_info
            level_kind = _expected_level_kind(level_selection.level)
            if dataset_info.levels[level_selection.level].kind != level_kind:
                raise RuntimeError("Serialized cache level kind is inconsistent with its level index.")
            omitted_value_ids = (
                ()
                if level_selection.omitted_value_ids is None
                else tuple(int(value_id) for value_id in level_selection.omitted_value_ids)
            )

            # 2. Reject, reuse the accepted batch, or prepare a replacement.
            if not within_hard_limits:
                # i) Reject: the requested view exceeds the hard rendering limits.
                snapshot = TiledPointsRenderSnapshot(
                    cache_generation_id=dataset_info.cache_generation_id,
                    request_generation=request.request_generation,
                    selection_generation=request.selection_generation,
                    requested_value_ids=request.requested_value_ids,
                    level=level_selection.level,
                    level_kind=level_kind,
                    within_hard_limits=False,
                    estimated_point_count=level_selection.estimated_point_count,
                    omitted_value_ids=omitted_value_ids,
                    rendered_tile_count=0,
                    render_batch=TiledPointsRenderBatch.empty(),
                    budget_message=budget_message,
                )
                candidate = None
            else:
                retained = self._retained_viewport
                if (
                    retained is not None
                    and retained.rejection_reason(
                        request,
                        cache_generation_id=dataset_info.cache_generation_id,
                        level=level_selection.level,
                        max_vertex_payload_bytes=max_vertex_payload_bytes,
                    )
                    is None
                ):
                    # ii) Reuse: keep the compatible accepted batch and its original bounds.
                    # Refresh request metadata, but preserve the original rectangle
                    # associated with this batch, even after repeated inner views.
                    snapshot = replace(
                        retained.snapshot,
                        request_generation=request.request_generation,
                        selection_generation=request.selection_generation,
                        estimated_point_count=level_selection.estimated_point_count,
                        omitted_value_ids=omitted_value_ids,
                        budget_message=budget_message,
                    )
                    candidate = _RetainedViewport(retained.bounds, snapshot)
                else:
                    # iii) Prepare a replacement: build a batch for the requested viewport.
                    snapshot = _read_viewport_snapshot(
                        reader,
                        self._selected_value_index,
                        self._cpu_tile_residency,
                        request,
                        viewport=viewport,
                        level_selection=level_selection,
                        budget_message=budget_message,
                        max_vertex_payload_bytes=max_vertex_payload_bytes,
                        raise_if_cancelled=self._require_not_cancelled,
                    )
                    candidate = _RetainedViewport(viewport, snapshot)
            self._require_not_cancelled()
            # Await GUI acceptance: acknowledge_render_result() promotes this
            # candidate only when result.applied is True. Rejection preserves
            # the previous retained viewport; over-budget results have no candidate.
            self._pending_viewport = candidate
            self.viewport_ready.emit(snapshot)
        except _SessionCancelled:
            self._shutdown(emit_closing=True)
        except Exception as error:  # noqa: BLE001
            logger.exception("Tiled-points cache session failed while reading a viewport snapshot.")
            self._pending_viewport = None
            self._report_viewport_failure(request, error)

    def _select_viewport_level(
        self,
        reader: _PointsCacheReader,
        request: _ViewportRequest,
        *,
        viewport: _IntrinsicViewport,
        preferred_point_budget: int,
        hard_point_capacity: int,
    ) -> tuple[_LevelSelection, _LodDecisionReason]:
        """Evaluate fresh candidates against compatible accepted LOD history.

        A disjoint pan or an oversized old allocation can still use the previous
        level as history. Only cache/selection identity matters here. Whether
        the complete old batch is reusable is checked separately, after the
        new viewport's level and hard rendering permission have been determined.
        This method does not commit history or perform point-payload IO.
        """
        previous_level = None
        retained = self._retained_viewport
        if retained is not None:
            snapshot = retained.snapshot
            if (
                snapshot.cache_generation_id == reader.dataset_info.cache_generation_id
                and snapshot.selection_generation == request.selection_generation
                and snapshot.requested_value_ids == request.requested_value_ids
            ):
                previous_level = snapshot.level

        def candidates() -> Iterator[_LevelSelection]:
            for candidate in reader.iter_level_candidates(
                viewport, preferred_point_budget, value_index=self._selected_value_index
            ):
                self._require_not_cancelled()
                yield candidate

        return _select_lod(
            candidates(),
            preferred_point_budget=preferred_point_budget,
            hard_point_capacity=hard_point_capacity,
            previous_level=previous_level,
        )

    @Slot(object)
    def acknowledge_render_result(self, result: TiledPointsRenderResult) -> None:
        """Resolve the pending viewport candidate using GUI activation feedback.

        Preparing a snapshot does not establish that the GUI accepted it, so
        ``read_viewport_snapshot()`` keeps it in ``_pending_viewport`` until this reply.

        For a matching request and selection generation, clear the pending entry.
        If ``result.applied`` is True, promote it to ``_retained_viewport``; otherwise
        preserve the previous retained entry. Unmatched replies are ignored.

        This retains the last accepted CPU batch, not necessarily drawable GPU
        contents. After a staging failure, the renderer may need to upload the
        retained batch again.
        """
        if self._finished or self._cancellation.is_set():
            return
        pending = self._pending_viewport
        if pending is None or (
            result.request_generation != pending.snapshot.request_generation
            or result.selection_generation != pending.snapshot.selection_generation
        ):
            return
        self._pending_viewport = None
        if result.applied:
            self._retained_viewport = pending

    def _require_not_cancelled(self) -> None:
        """Stop active work when the GUI has requested session closure.

        The thread-safe event can be observed while the worker's queued
        ``close()`` slot is still waiting for the current slot to return. This
        lets long-running work stop at its next checkpoint; reader cleanup
        remains on the worker thread.
        """
        if self._cancellation.is_set():
            raise _SessionCancelled

    def _report_recoverable_selection_failure(self, error: Exception) -> None:
        self.failed.emit(_failure_from_exception("selection", error))
        self.state_changed.emit(_CacheSessionState.READY)

    def _report_viewport_failure(self, request: object, error: Exception) -> None:
        request_generation = request.request_generation if isinstance(request, _ViewportRequest) else 0
        self.viewport_failed.emit(request_generation, _failure_from_exception("viewport", error))

    def _shutdown(self, *, emit_closing: bool) -> None:
        if self._finished:
            return
        self._finished = True
        if emit_closing:
            self.state_changed.emit(_CacheSessionState.CLOSING)

        reader = self._reader
        self._reader = None
        self._selected_value_ids = None
        self._selected_value_index = None
        self._cpu_tile_residency.clear()
        self._retained_viewport = None
        self._pending_viewport = None
        try:
            if reader is not None:
                reader.__exit__(None, None, None)
        except Exception as error:  # noqa: BLE001
            logger.exception("Tiled-points cache session failed while closing the reader.")
            self.state_changed.emit(_CacheSessionState.FAILED)
            self.failed.emit(_failure_from_exception("shutdown", error))
        finally:
            self.finished.emit()


# Sessions whose worker QThread may still be running. A started session keeps
# itself alive here until that thread has fully stopped, so owners may drop
# their references at any time (e.g. right after `close()` on layer removal).
# Otherwise garbage collection could destroy the session, and with it its child
# QThread, while the worker is still running: Qt aborts the process
# ("QThread: Destroyed while thread is still running"), or deadlocks on the GIL
# while the thread finishes its teardown.
# Nothing reads this set; holding the reference is its purpose. napari's
# `thread_worker` uses the same keep-alive pattern (superqt `WorkerBase._worker_set`).
_SESSIONS_WITH_RUNNING_THREADS: set[_TiledPointsCacheSession] = set()


class _TiledPointsCacheSession(QObject):
    """Expose one worker-owned cache reader to the GUI thread.

    Construction is passive. :meth:`start` creates the dedicated reader thread;
    :meth:`close` is terminal and idempotent. The reader and selected-value
    index never cross this facade.

    A started session stays alive until its worker thread has fully stopped,
    even if its owner drops it earlier (see `_SESSIONS_WITH_RUNNING_THREADS`).
    """

    state_changed = Signal(object)
    dataset_available = Signal(object)
    ready = Signal()
    value_selection_ready = Signal(object, int)
    viewport_ready = Signal(object)
    viewport_failed = Signal(int, object)
    failed = Signal(object)
    closed = Signal()

    # Queue a value-ID selection change on the worker; `value_selection_ready`
    # announces the committed result back to the GUI thread.
    _value_selection_change_requested = Signal(object)
    _viewport_requested = Signal(object)
    _render_result_received = Signal(object)
    _close_requested = Signal()

    def __init__(
        self,
        cache_root: str | Path,
        settings: _CacheSessionSettings,
        *,
        reader_factory: _ReaderFactory = _PointsCacheReader,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        if not isinstance(settings, _CacheSessionSettings):
            raise ValueError("`settings` must be _CacheSessionSettings.")
        if not callable(reader_factory):
            raise ValueError("`reader_factory` must be callable.")
        self._cache_root = Path(cache_root)
        self._settings = settings
        self._reader_factory = reader_factory
        self._state = _CacheSessionState.NEW
        self._dataset_info: _CacheDatasetInfo | None = None
        self._selected_value_ids: tuple[int, ...] | None = None
        self._resident_index_bytes: int | None = None
        self._cancellation = threading.Event()
        self._thread: QThread | None = None
        self._worker: _TiledPointsCacheWorker | None = None

    @property
    def state(self) -> _CacheSessionState:
        """Return the current GUI-side session state."""
        return self._state

    @property
    def dataset_info(self) -> _CacheDatasetInfo | None:
        """Return immutable opened-cache information when available."""
        return self._dataset_info

    @property
    def selected_value_ids(self) -> tuple[int, ...] | None:
        """Return the successfully applied value IDs; ``None`` means all values."""
        return self._selected_value_ids

    @property
    def resident_index_bytes(self) -> int | None:
        """Return compact NumPy-index bytes after startup, excluding Python objects and payloads."""
        return self._resident_index_bytes

    def start(self) -> None:
        """Create the worker thread and begin guarded cache startup."""
        if self._state is not _CacheSessionState.NEW:
            raise RuntimeError("A cache session can be started only from NEW.")

        thread = QThread(self)
        worker = _TiledPointsCacheWorker(
            self._cache_root,
            self._settings,
            self._cancellation,
            self._reader_factory,
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.start)
        self._value_selection_change_requested.connect(worker.update_selected_value_index)
        self._viewport_requested.connect(worker.read_viewport_snapshot)
        self._render_result_received.connect(worker.acknowledge_render_result)
        self._close_requested.connect(worker.close)
        worker.state_changed.connect(self._on_worker_state_changed)
        worker.dataset_available.connect(self._on_dataset_available)
        worker.ready.connect(self._on_ready)
        worker.value_selection_ready.connect(self._on_value_selection_ready)
        worker.viewport_ready.connect(self._on_viewport_ready)
        worker.viewport_failed.connect(self._on_viewport_failed)
        worker.failed.connect(self._on_failed)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(self._on_thread_finished)
        self._thread = thread
        self._worker = worker
        self._set_state(_CacheSessionState.STARTING)
        # Released in `_on_thread_finished()` once the thread has fully stopped.
        _SESSIONS_WITH_RUNNING_THREADS.add(self)
        thread.start()

    def set_selected_value_ids(self, requested_value_ids: tuple[int, ...] | None) -> bool:
        """Queue one selected-value index replacement from the ready state.

        ``None`` selects all canonical values. A tuple represents one sorted,
        unique, nonempty subset of canonical value IDs.
        """
        if self._state is not _CacheSessionState.READY:
            raise RuntimeError("Value selection can change only while the cache session is READY.")
        requested_value_ids = _require_requested_value_ids(requested_value_ids)
        if requested_value_ids == self._selected_value_ids:
            return False
        self._set_state(_CacheSessionState.UPDATING_SELECTED_VALUE_INDEX)
        self._value_selection_change_requested.emit(requested_value_ids)
        return True

    def request_viewport(self, request: _ViewportRequest) -> None:
        """Queue one scheduler-stamped viewport request on the reader worker."""
        if self._state is not _CacheSessionState.READY:
            raise RuntimeError("A viewport can be requested only while the cache session is READY.")
        if not isinstance(request, _ViewportRequest):
            raise ValueError("`request` must be _ViewportRequest.")
        if request.requested_value_ids != self._selected_value_ids:
            raise ValueError("Viewport request value IDs do not match the committed session selection.")
        self._viewport_requested.emit(request)

    def close(self) -> bool:
        """Request terminal owner-thread closure exactly once."""
        if self._state in (_CacheSessionState.CLOSING, _CacheSessionState.CLOSED):
            return False
        if self._state is _CacheSessionState.NEW:
            self._set_state(_CacheSessionState.CLOSING)
            self._set_state(_CacheSessionState.CLOSED)
            self.closed.emit()
            return True

        # The event stops active work; the queued slot closes an idle worker on
        # its owning thread. Both paths converge on idempotent worker shutdown.
        self._cancellation.set()
        self._set_state(_CacheSessionState.CLOSING)
        self._close_requested.emit()
        return True

    def acknowledge_render_result(self, result: TiledPointsRenderResult) -> None:
        """Queue a viewport candidate's activation outcome for the worker.

        The scheduler reports whether the candidate was accepted, including
        rejection of stale snapshots that never reached the renderer. This method
        queues feedback; it does not modify the worker's retained viewport directly.

        Feedback must be queued before the next viewport request so the worker
        resolves its pending candidate before evaluating subsequent batch reuse.
        """
        if self._state in (_CacheSessionState.CLOSING, _CacheSessionState.CLOSED):
            return
        self._render_result_received.emit(result)

    @Slot(object)
    def _on_worker_state_changed(self, state: _CacheSessionState) -> None:
        if self._state is _CacheSessionState.CLOSED:
            return
        if self._state is _CacheSessionState.CLOSING and state is not _CacheSessionState.FAILED:
            return
        self._set_state(state)

    @Slot(object)
    def _on_dataset_available(self, dataset_info: _CacheDatasetInfo) -> None:
        if self._state in (_CacheSessionState.CLOSING, _CacheSessionState.CLOSED):
            return
        self._dataset_info = dataset_info
        self.dataset_available.emit(dataset_info)

    @Slot(int)
    def _on_ready(self, resident_index_bytes: int) -> None:
        if self._state is not _CacheSessionState.READY:
            return
        self._resident_index_bytes = resident_index_bytes
        self.ready.emit()

    @Slot(object, int)
    def _on_value_selection_ready(self, selected_value_ids: tuple[int, ...] | None, resident_bytes: int) -> None:
        if self._state in (_CacheSessionState.CLOSING, _CacheSessionState.CLOSED):
            return
        self._selected_value_ids = selected_value_ids
        self.value_selection_ready.emit(selected_value_ids, resident_bytes)

    @Slot(object)
    def _on_viewport_ready(self, snapshot: TiledPointsRenderSnapshot) -> None:
        if self._state in (_CacheSessionState.CLOSING, _CacheSessionState.CLOSED):
            return
        self.viewport_ready.emit(snapshot)

    @Slot(int, object)
    def _on_viewport_failed(self, request_generation: int, failure: _CacheSessionFailure) -> None:
        if self._state in (_CacheSessionState.CLOSING, _CacheSessionState.CLOSED):
            return
        self.failed.emit(failure)
        self.viewport_failed.emit(request_generation, failure)

    @Slot(object)
    def _on_failed(self, failure: _CacheSessionFailure) -> None:
        if self._state is _CacheSessionState.CLOSED:
            return
        self.failed.emit(failure)

    @Slot()
    def _on_thread_finished(self) -> None:
        thread = self._thread
        self._thread = None
        self._worker = None
        if thread is not None:
            # `finished` fires before the thread runs its pending deleteLater()s,
            # such as the worker's. Wait (PyQt releases the GIL) so the thread has
            # fully exited before this session may be destroyed.
            thread.wait()
            thread.deleteLater()
        self._set_state(_CacheSessionState.CLOSED)
        self.closed.emit()
        # Release the keep-alive from `start()` only after this slot returns:
        # dropping the last reference here would delete this QObject while Qt
        # is still delivering the queued `finished` call to it.
        QTimer.singleShot(0, partial(_SESSIONS_WITH_RUNNING_THREADS.discard, self))

    def _set_state(self, state: _CacheSessionState) -> None:
        if state is self._state:
            return
        self._state = state
        self.state_changed.emit(state)


def _read_viewport_snapshot(
    reader: _PointsCacheReader,
    selected_value_index: _SelectedValueIndex | None,
    residency: _CpuTileResidency,
    request: _ViewportRequest,
    *,
    viewport: _IntrinsicViewport,
    level_selection: _LevelSelection,
    budget_message: str | None,
    max_vertex_payload_bytes: int,
    raise_if_cancelled: Callable[[], None],
) -> TiledPointsRenderSnapshot:
    """Prepare a replacement snapshot at the worker's already selected LOD.

    The caller has checked the hard rendering limits and decided that a
    replacement is needed. This helper plans logical tiles,
    reuses decoded CPU tiles, reads missing payloads and packs a new batch.
    It neither selects a level nor inspects or updates retained viewport state.

    ``level_selection`` supplies the selected level and its visible count and
    omission metadata. Its ``fits_point_budget`` flag concerns the preferred density
    target, not hard eligibility: hysteresis tolerance or a permitted coarsest
    fallback may leave it False. ``budget_message`` carries the worker's diagnostic.
    """
    dataset_info = reader.dataset_info
    level_kind = _expected_level_kind(level_selection.level)
    if dataset_info.levels[level_selection.level].kind != level_kind:
        raise RuntimeError("Serialized cache level kind is inconsistent with its level index.")
    omitted_value_ids = (
        ()
        if level_selection.omitted_value_ids is None
        else tuple(int(value_id) for value_id in level_selection.omitted_value_ids)
    )
    plan = reader.plan_viewport(level_selection.level, viewport, value_index=selected_value_index)
    raise_if_cancelled()
    if plan.requested_value_ids != request.requested_value_ids:
        raise RuntimeError("Viewport plan selection differs from its generation-bound request.")
    keys = tuple(
        TileResidencyKey(
            cache_generation_id=dataset_info.cache_generation_id,
            requested_value_ids=request.requested_value_ids,
            level=level,
            tile_x=tile_x,
            tile_y=tile_y,
        )
        for level, tile_x, tile_y in plan.tile_keys
    )
    payloads_by_key: dict[TileResidencyKey, TiledPointsRenderTile] = {}
    missing_keys: list[TileResidencyKey] = []
    for key in keys:
        tile = residency.get(key)
        if tile is None:
            missing_keys.append(key)
        else:
            payloads_by_key[key] = tile
    resident_keys = tuple(payloads_by_key)

    new_tiles: tuple[TiledPointsRenderTile, ...] = ()
    if missing_keys:
        result = reader.read_planned_tiles(
            plan,
            tuple(key.logical_tile_key for key in missing_keys),
            raise_if_cancelled=raise_if_cancelled,
        )
        raise_if_cancelled()
        key_by_logical_tile = {key.logical_tile_key: key for key in missing_keys}
        # Bucket reads expose per-tile views into shared batch allocations.
        # Copy once at the viewer-residency boundary so every render tile owns
        # exactly the point-array bytes accounted by `_CpuTileResidency`.
        new_tiles = tuple(
            TiledPointsRenderTile(
                key=key_by_logical_tile[(tile.level, tile.tile_x, tile.tile_y)],
                tile_size=tile.tile_size,
                location=tile.location.copy(order="C"),
                value_id=tile.value_id.copy(order="C"),
            )
            for tile in result.tiles
        )
        if {tile.key for tile in new_tiles} != set(missing_keys):
            raise RuntimeError("Viewport subset read did not return every requested nonresident tile.")
        payloads_by_key.update((tile.key, tile) for tile in new_tiles)

    ordered_tiles = tuple(payloads_by_key[key] for key in keys)
    _require_ordered_render_tiles(
        ordered_tiles,
        cache_generation_id=dataset_info.cache_generation_id,
        requested_value_ids=request.requested_value_ids,
        level=level_selection.level,
    )
    if new_tiles:
        residency.retain(new_tiles, protected_keys=resident_keys)
    render_batch = pack_render_tiles(
        ordered_tiles,
        point_count=level_selection.estimated_point_count,
        value_count=len(dataset_info.value_names),
        max_vertex_payload_bytes=max_vertex_payload_bytes,
        raise_if_cancelled=raise_if_cancelled,
    )
    raise_if_cancelled()
    return TiledPointsRenderSnapshot(
        cache_generation_id=dataset_info.cache_generation_id,
        request_generation=request.request_generation,
        selection_generation=request.selection_generation,
        requested_value_ids=request.requested_value_ids,
        level=level_selection.level,
        level_kind=level_kind,
        within_hard_limits=True,
        estimated_point_count=level_selection.estimated_point_count,
        omitted_value_ids=omitted_value_ids,
        rendered_tile_count=len(ordered_tiles),
        render_batch=render_batch,
        budget_message=budget_message,
    )


def _require_ordered_render_tiles(
    tiles: tuple[TiledPointsRenderTile, ...],
    *,
    cache_generation_id: str,
    requested_value_ids: tuple[int, ...] | None,
    level: int,
) -> None:
    """Validate the complete worker-local tile order before packing it."""
    keys = tuple(tile.key for tile in tiles)
    if len(set(keys)) != len(keys):
        raise RuntimeError("Snapshot tile residency keys must be unique.")
    coordinates = tuple((key.tile_y, key.tile_x) for key in keys)
    if coordinates != tuple(sorted(coordinates)):
        raise RuntimeError("Snapshot tiles must follow spatial (tile_y, tile_x) order.")
    if any(
        key.cache_generation_id != cache_generation_id
        or key.requested_value_ids != requested_value_ids
        or key.level != level
        for key in keys
    ):
        raise RuntimeError("Every snapshot tile must match its cache, selection, and level.")


def _require_requested_value_ids(requested_value_ids: tuple[int, ...] | None) -> tuple[int, ...] | None:
    if requested_value_ids is None:
        return None
    if (
        not isinstance(requested_value_ids, tuple)
        or not requested_value_ids
        or any(
            not isinstance(value_id, int) or isinstance(value_id, bool) or not 0 <= value_id <= _UINT32_MAX
            for value_id in requested_value_ids
        )
        or tuple(sorted(set(requested_value_ids))) != requested_value_ids
    ):
        raise ValueError("`requested_value_ids` must be None or sorted unique nonnegative uint32 integers.")
    return requested_value_ids


def _normalize_all_values(value_ids: tuple[int, ...] | None, *, value_count: int) -> tuple[int, ...] | None:
    """Normalize a complete vocabulary tuple to the all-values ``None`` state."""
    if value_ids == tuple(range(value_count)):
        return None
    return value_ids


def _failure_from_exception(phase: _FailurePhase, error: Exception) -> _CacheSessionFailure:
    exception_type = f"{type(error).__module__}.{type(error).__qualname__}"
    return _CacheSessionFailure(
        phase=phase,
        exception_type=exception_type,
        message=str(error) or repr(error),
    )
