"""Shared Spatiato app state bound to one napari viewer.

This module defines the per-viewer state object used across Spatiato widgets.

At the top level, ``_VIEWER_APP_STATES`` is a registry that maps one napari
viewer to one ``SpatiatoAppState`` instance. That is how multiple widgets opened
on the same viewer end up sharing the same Spatiato state.

Each ``SpatiatoAppState`` then holds the shared state and services for that
viewer:

- ``viewer``: the napari viewer this state belongs to
- ``sdata``: the currently loaded ``SpatialData`` object
- ``layer_bindings``: the in-memory registry that records which napari layers
  correspond to which ``SpatialData`` elements
- ``viewer_adapter``: the viewer-facing service that uses the shared registry
  to look up, activate, and later load Spatiato-managed layers

So the relationship is:

- global registry: viewer -> ``SpatiatoAppState``
- per-viewer state: ``viewer``, ``sdata``, ``layer_bindings``,
  ``viewer_adapter``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol
from weakref import WeakKeyDictionary

from qtpy.QtCore import QObject, Signal

from spatiato.core.persistence import TableComponentPath
from spatiato.core.spatialdata import get_coordinate_system_names_from_sdata
from spatiato.viewer.adapter import LayerBindingRegistry, ViewerAdapter

if TYPE_CHECKING:
    from spatialdata import SpatialData


_VIEWER_APP_STATES: WeakKeyDictionary[object, SpatiatoAppState] = WeakKeyDictionary()
TableChangeKind = Literal["created", "updated", "removed", "rebuilt", "reloaded"]


@dataclass(frozen=True)
class TableStateChangedEvent:
    """Describe one accepted change to explicit components of an AnnData table.

    ``regions`` is the producer-declared semantic scope of any row-scoped
    change. An empty tuple means the event is genuinely metadata-only and never
    means that the affected regions are unknown. A producer that cannot prove a
    narrower row scope must report every region declared by the table.
    """

    sdata: SpatialData
    table_name: str
    paths: frozenset[TableComponentPath]
    regions: tuple[str, ...]
    change_kind: TableChangeKind
    source: str

    def __post_init__(self) -> None:
        if not isinstance(self.table_name, str) or not self.table_name:
            raise ValueError("Table-state events require a non-empty table name.")
        paths = frozenset(self.paths)
        if not paths:
            raise ValueError("Table-state events require at least one component path.")
        object.__setattr__(self, "paths", paths)
        if self.change_kind not in ("created", "updated", "removed", "rebuilt", "reloaded"):
            raise ValueError(f"Unsupported table change kind: {self.change_kind!r}.")
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("Table-state events require a non-empty source.")
        regions = tuple(self.regions)
        if any(not isinstance(region, str) or not region for region in regions):
            raise ValueError("Table-state event regions must be non-empty strings.")
        if len(set(regions)) != len(regions):
            raise ValueError("Table-state event regions must be unique.")
        object.__setattr__(self, "regions", regions)


@dataclass(frozen=True)
class TableDirtyStateChangedEvent:
    """Report a table-wide dirty transition derived from the shared manifest.

    This event keeps every bound ``TablePersistenceControls`` UI synchronized
    when any workflow makes the selected table dirty or clean. In particular,
    controls use it to update Write-button readiness and the Reload tooltip
    warning about unsynced changes. It does not describe an AnnData mutation
    and is not a second source of dirty truth; consumers that bind later must
    read the current state from ``SpatiatoAppState``.
    """

    sdata: SpatialData
    table_name: str
    is_dirty: bool

    def __post_init__(self) -> None:
        if not isinstance(self.table_name, str) or not self.table_name:
            raise ValueError("Table dirty-state events require a non-empty table name.")
        if not isinstance(self.is_dirty, bool):
            raise ValueError("Table dirty-state events require a boolean dirty state.")


@dataclass(frozen=True)
class TableReloadRequest:
    """Identify one accepted table reload before live state is replaced.

    ``region_name`` is optional validation context for the spatial element
    currently using the table. It does not restrict the reload to that region's
    rows.

    Parameters
    ----------
    sdata
        Exact SpatialData object whose live table will be restored.
    table_name
        Name of the table to restore.
    paths
        Fully expanded component paths that the accepted reload will replace.
    region_name
        Optional Labels or Shapes element against which the restored table
        binding must remain valid.
    source
        Workflow or reusable control that initiated the reload request.
    """

    sdata: SpatialData
    table_name: str
    paths: frozenset[TableComponentPath]
    region_name: str | None
    source: str

    def __post_init__(self) -> None:
        if not isinstance(self.table_name, str) or not self.table_name:
            raise ValueError("Table reload requests require a non-empty table name.")
        paths = frozenset(self.paths)
        if not paths:
            raise ValueError("Table reload requests require at least one component path.")
        object.__setattr__(self, "paths", paths)
        if self.region_name is not None and (not isinstance(self.region_name, str) or not self.region_name):
            raise ValueError("Table reload request region names must be non-empty strings.")
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("Table reload requests require a non-empty source.")


class TableReloadParticipant(Protocol):
    """Prepare one table-consuming workflow before an accepted reload.

    A class satisfies this structural protocol by implementing
    ``prepare_for_table_reload()``; inheritance is not required. Participants
    compare the immutable request with the table they consume and ignore
    unrelated requests.
    """

    def prepare_for_table_reload(self, request: TableReloadRequest) -> None:
        """Invalidate work that must not survive the requested table reload."""


class _TableMutationToken:
    """Opaque identity token for one accepted in-memory table mutation."""

    __slots__ = ()


@dataclass(frozen=True)
class TableDirtySnapshot:
    """Capture one table's dirty component state before an operation starts.

    The snapshot preserves the mutation token associated with every component
    path that was dirty at capture time. Persistence acknowledgement later
    compares these captured tokens with the current dirty manifest. A path is
    eligible to become clean only when it was successfully persisted and its
    current token is still the captured token.

    The snapshot is session-only state. It does not copy the table and is never
    stored in AnnData or zarr.

    Parameters
    ----------
    sdata
        SpatialData object containing the affected table. Object identity is
        used to ensure that acknowledgement targets the same dataset.
    table_name
        Name of the affected AnnData table.
    captured_path_tokens
        Immutable pairs of dirty component paths and the mutation tokens
        captured for them. The token objects are retained by identity rather
        than copied.
    """

    sdata: SpatialData
    table_name: str
    captured_path_tokens: tuple[tuple[TableComponentPath, _TableMutationToken], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.table_name, str) or not self.table_name:
            raise ValueError("Dirty snapshots require a non-empty table name.")
        normalized = tuple(sorted(self.captured_path_tokens, key=lambda item: item[0]))
        paths = tuple(path for path, _token in normalized)
        if len(set(paths)) != len(paths):
            raise ValueError("Dirty snapshots cannot contain duplicate component paths.")
        if any(not isinstance(token, _TableMutationToken) for _, token in normalized):
            raise ValueError("Dirty snapshots require a mutation token for every component path.")
        object.__setattr__(self, "captured_path_tokens", normalized)

    @property
    def paths(self) -> frozenset[TableComponentPath]:
        """Return the component paths captured by this snapshot."""
        return frozenset(path for path, _token in self.captured_path_tokens)


@dataclass(frozen=True)
class ShapesElementWrittenEvent:
    """Describe an in-place write to a shared `SpatialData.shapes` element."""

    sdata: SpatialData
    shapes_name: str
    coordinate_system: str
    source: str = "shapes_annotation_widget"


@dataclass(frozen=True)
class ShapesElementReloadedEvent:
    """Describe successful adoption of one persisted Shapes element."""

    sdata: SpatialData
    shapes_name: str
    coordinate_system: str
    source: str = "shapes_annotation_widget"


@dataclass(frozen=True)
class CoordinateSystemChangedEvent:
    """Describe a shared active coordinate-system change for one viewer session."""

    sdata: SpatialData | None
    previous_coordinate_system: str | None
    coordinate_system: str | None
    source: str | None = None


@dataclass(frozen=True)
class CoordinateSystemChangeRequest:
    """Describe a requested coordinate-system change before it is committed.

    The coordinate-system change participant receives this immutable request
    while the old app-state value and viewer layers are still intact. Returning
    ``False`` rejects the request without emitting
    ``coordinate_system_changed`` or removing layers.
    """

    sdata: SpatialData | None
    previous_coordinate_system: str | None
    coordinate_system: str | None
    source: str | None = None


class CoordinateSystemChangeParticipant(Protocol):
    """Let Annotation participate in a shared coordinate-system transition.

    ``SpatiatoAppState`` cannot safely commit every coordinate-system change on
    its own. The Annotation widget may own an editable Shapes layer whose
    unsaved geometry would be lost when layers from the old coordinate system
    are removed. Annotation therefore registers itself as the one optional
    participant and implements ``prepare_coordinate_system_change()`` to
    delegate that preflight to
    ``ShapesAnnotation.try_close_edit_session()``.

    This is a structural ``Protocol`` rather than a required base class. A
    class does not need to inherit from ``CoordinateSystemChangeParticipant``;
    it satisfies the contract by providing the method declared below. This
    lets ``SpatiatoAppState`` call the participant without importing or depending
    directly on ``AnnotationWidget``.

    The runtime flow is::

        AnnotationWidget registers itself with SpatiatoAppState
            ↓
        a widget requests a shared coordinate-system change
            ↓
        SpatiatoAppState calls prepare_coordinate_system_change(request)
            ↓
        AnnotationWidget asks its Shapes child to close the edit session
            ├── True  → SpatiatoAppState commits the change
            └── False → SpatiatoAppState preserves the old state and layers
    """

    def prepare_coordinate_system_change(self, request: CoordinateSystemChangeRequest) -> bool:
        """Return whether the requested transition may be committed.

        Parameters
        ----------
        request
            Immutable description of the proposed change, captured while the
            previous app state and viewer layers are still intact.

        Returns
        -------
        bool
            ``True`` after any participant-owned state is ready for the
            transition. ``False`` rejects the request without changing app
            state, emitting ``coordinate_system_changed``, or removing layers.
        """
        ...


def _path_covers(parent: TableComponentPath, child: TableComponentPath) -> bool:
    """Return whether one synchronized path also restores a dirty path.

    ``parent`` is a path accepted as persisted or reloaded; ``child`` is a
    dirty path being considered for removal from the dirty manifest. Coverage
    follows the persistence granularity of each AnnData component: any
    ``obs`` path covers the complete dataframe, an ``obsm`` path covers only
    the same named entry, and an ``uns`` path covers itself and its nested
    descendants. Paths belonging to different components never overlap.
    """
    if parent.component != child.component:
        return False
    if parent.component == "obs":
        return True
    if parent.component == "obsm":
        return parent == child
    if parent.component == "uns":
        return child.keys[: len(parent.keys)] == parent.keys
    raise AssertionError(f"Unsupported table component: {parent.component!r}.")


class SpatiatoAppState(QObject):
    """Shared Spatiato state bound to a napari viewer.

    This object is the per-viewer event and state hub that Spatiato widgets use
    to stay synchronized without depending on each other directly.

    Cross-widget table updates are published through ``table_state_changed``
    with explicit AnnData component paths. Shapes writes retain their separate
    event because they do not mutate an AnnData table. Producing widgets do not
    need to know which other widgets consume either update.

    ``SpatiatoAppState`` also owns shared session-level dirty-table tracking, so
    in-memory table divergence from disk is modeled as shared viewer state
    rather than as widget-local state. ``table_dirty_state_changed`` reports
    only table-wide dirty-boolean transitions so already-bound persistence
    controls remain synchronized; it is derived from the manifest and does not
    replace component-level ``table_state_changed`` events.

    A dirty table component can become clean only through one of three
    accepted transitions:

    1. ``acknowledge_table_write()`` clears a component after the generic
       persistence service successfully writes it and its captured mutation
       token is still current.
    2. ``record_persisted_table_change()`` clears a component that an operation
       already persisted itself, when given the snapshot captured before the
       operation, again only when its captured token is still current.
    3. ``record_table_reload()`` clears a component whose in-memory state was
       deliberately replaced with the persisted state.

    Publishing a table-state event alone never clears dirty state, and an
    unrelated or outdated operation cannot clear a component.
    """

    sdata_changed = Signal(object)
    table_state_changed = Signal(object)
    table_dirty_state_changed = Signal(object)
    shapes_element_written = Signal(object)
    shapes_element_reloaded = Signal(object)
    coordinate_system_changed = Signal(object)

    def __init__(self, viewer: object | None = None) -> None:
        super().__init__()
        self.viewer = viewer
        self.sdata: SpatialData | None = None
        self.coordinate_system: str | None = None
        self.layer_bindings = LayerBindingRegistry()
        self.viewer_adapter = ViewerAdapter(viewer=viewer, layer_bindings=self.layer_bindings)
        self._dirty_table_tokens: dict[tuple[int, str], dict[TableComponentPath, _TableMutationToken]] = {}
        self._coordinate_system_change_participant: CoordinateSystemChangeParticipant | None = None
        self._table_reload_participants: list[TableReloadParticipant] = []

    def set_sdata(
        self,
        sdata: SpatialData | None,
        *,
        discard_current: bool = False,
    ) -> None:
        """Set the shared SpatialData after authorizing destructive replacement.

        An initial assignment and reassigning the exact active object are
        non-destructive. Replacing or clearing an active object requires
        ``discard_current=True`` so low-level callers cannot silently bypass
        the user-facing Proceed / Cancel boundary.
        """
        if not isinstance(discard_current, bool):
            raise TypeError("SpatialData discard authorization must be a boolean.")

        old_sdata = self.sdata
        old_coordinate_system = self.coordinate_system
        replaces_current = old_sdata is not None and old_sdata is not sdata
        if replaces_current and not discard_current:
            raise RuntimeError("Replacing or clearing the active SpatialData requires explicit discard authorization.")

        if replaces_current:
            self.viewer_adapter.remove_layers_for_sdata(old_sdata)
            self._discard_dirty_state_for_sdata(old_sdata)

        self.sdata = sdata
        next_coordinate_system = self._resolve_coordinate_system_for_sdata(sdata, previous=old_coordinate_system)
        self._update_coordinate_system_state(next_coordinate_system, source="set_sdata")
        # Notify connected widgets/controllers that the loaded SpatialData changed
        # (controllers/widgets that listen via e.g. self._app_state.sdata_changed.connect(self._on_sdata_changed))
        self.sdata_changed.emit(sdata)

    def clear_sdata(self, *, discard_current: bool = False) -> None:
        """Clear the loaded SpatialData object after explicit discard authorization."""
        self.set_sdata(None, discard_current=discard_current)

    def set_coordinate_system(
        self,
        coordinate_system: str | None,
        *,
        source: str | None = None,
    ) -> bool:
        """Request a change to the shared coordinate system.

        When the requested coordinate system is valid and differs from the current
        one, the registered participant is called immediately before the change
        is applied. If the participant rejects the request, the coordinate
        system remains unchanged, no ``coordinate_system_changed`` event is
        emitted, and no viewer layers are removed.

        Example flow
        ------------
        set_coordinate_system("local")
            ↓
        normalize and validate "local"
            ↓
        check that "local" differs from the current "global"
            ↓
        call the participant immediately
            ├── False
            │      → stop
            │      → coordinate system remains "global"
            │      → emit no coordinate_system_changed event
            │      → remove no viewer layers
            │
            └── True
                   → change app state to "local"
                   → emit coordinate_system_changed
                   → remove layers outside "local"
        """
        normalized_coordinate_system = self._normalize_coordinate_system(coordinate_system)
        self._validate_coordinate_system(normalized_coordinate_system)
        if normalized_coordinate_system == self.coordinate_system:
            return False

        participant = self._coordinate_system_change_participant
        if participant is not None:
            request = CoordinateSystemChangeRequest(
                sdata=self.sdata,
                previous_coordinate_system=self.coordinate_system,
                coordinate_system=normalized_coordinate_system,
                source=source,
            )
            # This must run before state mutation or layer removal: otherwise
            # an external coordinate switch could silently delete the only
            # copy of unsaved Shapes Annotation edits.
            # Today the participant is AnnotationWidget. Its implementation
            # delegates to ShapesAnnotation.try_close_edit_session(), which
            # preserves the current edit session when the user cancels.
            accepted = participant.prepare_coordinate_system_change(request)
            if accepted is False:
                return False

        self._update_coordinate_system_state(normalized_coordinate_system, source=source)

        self.viewer_adapter.remove_layers_outside_coordinate_system(
            sdata=self.sdata,
            coordinate_system=normalized_coordinate_system,
        )
        return True

    def register_coordinate_system_change_participant(
        self,
        participant: CoordinateSystemChangeParticipant,
    ) -> None:
        """Register the single owner of coordinate-system change preflight.

        The Annotation widget owns this optional per-viewer role and must
        unregister itself during teardown. Replacing another active participant
        is rejected so one widget cannot silently disable another widget's
        data-loss protection.
        """
        if (
            self._coordinate_system_change_participant is not None
            and participant is not self._coordinate_system_change_participant
        ):
            raise RuntimeError("A coordinate-system change participant is already registered for this viewer session.")
        self._coordinate_system_change_participant = participant

    def unregister_coordinate_system_change_participant(
        self,
        participant: CoordinateSystemChangeParticipant,
    ) -> bool:
        """Unregister ``participant`` only while it still owns preflight.

        Identity-safe removal prevents teardown of an older widget from
        removing a participant registered by a newer owner.
        """
        if self._coordinate_system_change_participant is not participant:
            return False
        self._coordinate_system_change_participant = None
        return True

    def register_table_reload_participant(
        self,
        participant: TableReloadParticipant,
    ) -> None:
        """Register one workflow for shared table-reload preparation."""
        if any(current is participant for current in self._table_reload_participants):
            return
        self._table_reload_participants.append(participant)

    def unregister_table_reload_participant(
        self,
        participant: TableReloadParticipant,
    ) -> bool:
        """Unregister ``participant`` by identity without affecting other workflows."""
        for index, current in enumerate(self._table_reload_participants):
            if current is participant:
                del self._table_reload_participants[index]
                return True
        return False

    def prepare_for_table_reload(self, request: TableReloadRequest) -> None:
        """Notify table-consuming workflows before live components are reloaded.

        ``PersistenceController`` calls this shared app-state boundary after a
        reload request has been accepted and immediately before it replaces
        the requested in-memory table components from disk. Each registered
        ``TableReloadParticipant`` receives the same immutable request and
        decides whether it consumes that SpatialData table. Affected workflows
        can then invalidate pending or active work that captured the old table
        state.

        This method does not reload the table or publish the post-reload event.
        The persistence controller performs those steps after every participant
        has prepared successfully.

        Participants are visited through a stable snapshot so registration or
        teardown during a callback cannot skip or duplicate another workflow
        in the accepted transition.

        Parameters
        ----------
        request
            Accepted reload request identifying the table and components that
            are about to be restored from disk.
        """
        for participant in tuple(self._table_reload_participants):
            participant.prepare_for_table_reload(request)

    def clear_coordinate_system(self, *, source: str | None = None) -> bool:
        """Clear the shared active coordinate system."""
        return self.set_coordinate_system(None, source=source)

    def emit_shapes_element_written(self, event: ShapesElementWrittenEvent) -> None:
        """Broadcast that a shapes element was written into the shared SpatialData."""
        self.shapes_element_written.emit(event)

    def emit_shapes_element_reloaded(self, event: ShapesElementReloadedEvent) -> None:
        """Broadcast that a persisted Shapes element replaced its live state."""
        self.shapes_element_reloaded.emit(event)

    def record_table_mutation(self, event: TableStateChangedEvent) -> None:
        """Record an accepted in-memory mutation and publish its table event."""
        selection_key = self._selection_key(event.sdata, event.table_name)
        if selection_key is None:  # pragma: no cover - event validation makes this unreachable.
            return

        was_dirty = bool(self._dirty_table_tokens.get(selection_key))
        mutation_token = _TableMutationToken()
        manifest = self._dirty_table_tokens.setdefault(selection_key, {})
        for path in event.paths:
            manifest[path] = mutation_token
        self._emit_table_dirty_transition(
            event.sdata,
            event.table_name,
            was_dirty=was_dirty,
        )
        self.table_state_changed.emit(event)

    def record_persisted_table_change(
        self,
        event: TableStateChangedEvent,
        snapshot: TableDirtySnapshot | None = None,
    ) -> None:
        """Publish an already-persisted change without clearing newer mutations.

        Pass the dirty snapshot captured before the operation to clear the dirty
        paths it persisted. Without one, nothing is cleared and the change is
        only published.
        """
        if snapshot is not None:
            self._validate_snapshot_identity(snapshot, event.sdata, event.table_name)
            self.acknowledge_table_write(snapshot, persisted_paths=event.paths)
        self.table_state_changed.emit(event)

    def record_table_reload(self, event: TableStateChangedEvent) -> None:
        """Publish a component reload and clear only dirty paths it restored."""
        if event.change_kind != "reloaded":
            raise ValueError("Table reload events must use change_kind='reloaded'.")
        selection_key = self._selection_key(event.sdata, event.table_name)
        if selection_key is None:  # pragma: no cover - event validation makes this unreachable.
            return
        was_dirty = bool(self._dirty_table_tokens.get(selection_key))
        manifest = self._dirty_table_tokens.get(selection_key)
        if manifest is not None:
            for dirty_path in tuple(manifest):
                if any(_path_covers(reloaded_path, dirty_path) for reloaded_path in event.paths):
                    del manifest[dirty_path]
            self._drop_empty_manifest(selection_key)
        self._emit_table_dirty_transition(
            event.sdata,
            event.table_name,
            was_dirty=was_dirty,
        )
        self.table_state_changed.emit(event)

    def is_table_dirty(self, sdata: SpatialData | None, table_name: str | None) -> bool:
        """Return whether a selected in-memory table has unsynced local changes."""
        selection_key = self._selection_key(sdata, table_name)
        return selection_key is not None and bool(self._dirty_table_tokens.get(selection_key))

    def snapshot_table_dirty_state(
        self,
        sdata: SpatialData,
        table_name: str,
    ) -> TableDirtySnapshot:
        """Capture the current component mutation tokens for a table."""
        selection_key = self._selection_key(sdata, table_name)
        if selection_key is None:  # pragma: no cover - concrete arguments make this unreachable.
            raise ValueError("Dirty snapshots require a SpatialData object and table name.")
        manifest = self._dirty_table_tokens.get(selection_key, {})
        return TableDirtySnapshot(
            sdata=sdata,
            table_name=table_name,
            captured_path_tokens=tuple(manifest.items()),
        )

    def acknowledge_table_write(
        self,
        snapshot: TableDirtySnapshot,
        *,
        persisted_paths: frozenset[TableComponentPath],
    ) -> None:
        """Mark successfully persisted component paths clean when still current.

        Each dirty component path carries the identity token of its latest
        accepted mutation. The snapshot captures those tokens before
        persistence starts.

        A persisted path is cleared only when its current token is the same
        object as the captured token. If the path changed while persistence was
        in progress, it has received a fresh token and remains dirty. This
        prevents completion of an older write from marking a newer in-memory
        mutation as persisted.
        """
        selection_key = self._selection_key(snapshot.sdata, snapshot.table_name)
        if selection_key is None:  # pragma: no cover - snapshot validation makes this unreachable.
            return
        was_dirty = bool(self._dirty_table_tokens.get(selection_key))
        manifest = self._dirty_table_tokens.get(selection_key)
        if manifest is None:
            return

        for path, captured_token in snapshot.captured_path_tokens:
            if not any(_path_covers(persisted_path, path) for persisted_path in persisted_paths):
                continue
            if manifest.get(path) is captured_token:
                del manifest[path]
        self._drop_empty_manifest(selection_key)
        self._emit_table_dirty_transition(
            snapshot.sdata,
            snapshot.table_name,
            was_dirty=was_dirty,
        )

    def _emit_table_dirty_transition(
        self,
        sdata: SpatialData,
        table_name: str,
        *,
        was_dirty: bool,
    ) -> None:
        """Notify bound controls only when the table-wide dirty state changed."""
        is_dirty = self.is_table_dirty(sdata, table_name)
        if is_dirty == was_dirty:
            return
        self.table_dirty_state_changed.emit(
            TableDirtyStateChangedEvent(
                sdata=sdata,
                table_name=table_name,
                is_dirty=is_dirty,
            )
        )

    def _drop_empty_manifest(self, selection_key: tuple[int, str]) -> None:
        if not self._dirty_table_tokens.get(selection_key):
            self._dirty_table_tokens.pop(selection_key, None)

    def _discard_dirty_state_for_sdata(self, sdata: SpatialData) -> None:
        """Remove obsolete dirty-table tracking after accepted SpatialData replacement.

        Proceed means the old SpatialData session has been deliberately
        discarded, so its tables can no longer be written through the active
        persistence controls. Retaining their mutation tokens would leave
        obsolete tables marked dirty. This removes only session bookkeeping;
        it does not mutate AnnData, write to disk, or publish table events.
        """
        sdata_id = id(sdata)
        for selection_key in tuple(self._dirty_table_tokens):
            if selection_key[0] == sdata_id:
                del self._dirty_table_tokens[selection_key]

    @staticmethod
    def _validate_snapshot_identity(
        snapshot: TableDirtySnapshot,
        sdata: SpatialData,
        table_name: str,
    ) -> None:
        if snapshot.sdata is not sdata or snapshot.table_name != table_name:
            raise ValueError("Dirty snapshot does not belong to the changed table.")

    @staticmethod
    def _selection_key(sdata: SpatialData | None, table_name: str | None) -> tuple[int, str] | None:
        if sdata is None or table_name is None:
            return None

        return (id(sdata), table_name)

    @staticmethod
    def _normalize_coordinate_system(coordinate_system: str | None) -> str | None:
        if coordinate_system is None:
            return None

        normalized_coordinate_system = coordinate_system.strip()
        return normalized_coordinate_system or None

    def _validate_coordinate_system(self, coordinate_system: str | None) -> None:
        if coordinate_system is None:
            return

        if self.sdata is None:
            raise ValueError("Cannot set an active coordinate system when no SpatialData is loaded.")

        available_coordinate_systems = get_coordinate_system_names_from_sdata(self.sdata)
        if coordinate_system not in available_coordinate_systems:
            raise ValueError(
                f"Coordinate system `{coordinate_system}` is not available in the selected SpatialData object."
            )

    def _update_coordinate_system_state(
        self,
        coordinate_system: str | None,
        *,
        source: str | None = None,
    ) -> bool:
        previous_coordinate_system = self.coordinate_system
        if coordinate_system == previous_coordinate_system:
            return False

        self.coordinate_system = coordinate_system
        self.coordinate_system_changed.emit(
            CoordinateSystemChangedEvent(
                sdata=self.sdata,
                previous_coordinate_system=previous_coordinate_system,
                coordinate_system=coordinate_system,
                source=source,
            )
        )
        return True

    @staticmethod
    def _resolve_coordinate_system_for_sdata(
        sdata: SpatialData | None,
        *,
        previous: str | None,
    ) -> str | None:
        if sdata is None:
            return None

        available_coordinate_systems = get_coordinate_system_names_from_sdata(sdata)
        if previous is not None and previous in available_coordinate_systems:
            return previous
        if available_coordinate_systems:
            return available_coordinate_systems[0]
        return None


def get_or_create_app_state(napari_viewer: object | None) -> SpatiatoAppState:
    """Return the shared Spatiato state for a napari viewer.

    When a real viewer is provided, the same ``SpatiatoAppState`` instance is
    returned for repeated calls with that viewer. When ``napari_viewer`` is
    ``None``, a fresh standalone state object is returned.
    """
    if napari_viewer is None:
        return SpatiatoAppState()

    try:
        state = _VIEWER_APP_STATES.get(napari_viewer)
    except TypeError:
        state = getattr(napari_viewer, "_spatiato_app_state", None)

    if state is not None:
        return state

    state = SpatiatoAppState(napari_viewer)

    try:
        _VIEWER_APP_STATES[napari_viewer] = state
    except TypeError:
        napari_viewer._spatiato_app_state = state

    return state
