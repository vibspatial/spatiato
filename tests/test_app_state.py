from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace

import pytest

import spatiato._app_state as app_state_module
import spatiato._interactive as interactive_module
import spatiato.widgets.object_classification.widget as object_widget_module
import spatiato.widgets.viewer.widget as viewer_widget_module
from spatiato._app_state import (
    CoordinateSystemChangedEvent,
    CoordinateSystemChangeRequest,
    ShapesElementWrittenEvent,
    SpatiatoAppState,
    TableDirtyStateChangedEvent,
    TableReloadRequest,
    TableStateChangedEvent,
    get_or_create_app_state,
)
from spatiato.core.persistence import TableComponentPath
from spatiato.widgets.feature_extraction.widget import FeatureExtractionWidget
from spatiato.widgets.object_classification.widget import ObjectClassificationWidget
from spatiato.widgets.viewer.widget import ViewerWidget


class DummyEventEmitter:
    def __init__(self) -> None:
        self._callbacks: list[Callable[[object], None]] = []

    def connect(self, callback: Callable[[object], None]) -> None:
        self._callbacks.append(callback)

    def emit(self, value: object | None = None) -> None:
        event = SimpleNamespace(value=value)
        for callback in list(self._callbacks):
            callback(event)


class DummyLayers(list):
    def __init__(self) -> None:
        super().__init__()
        self.selection = SimpleNamespace(active=None, select_only=self._select_only)
        self.events = SimpleNamespace(
            inserted=DummyEventEmitter(),
            removed=DummyEventEmitter(),
            reordered=DummyEventEmitter(),
        )

    def _select_only(self, layer: object) -> None:
        self.selection.active = layer


class DummyWindow:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None, bool]] = []
        self._dock_widgets: dict[tuple[str, str | None], tuple[object, object]] = {}

    def add_plugin_dock_widget(
        self,
        plugin_name: str,
        widget_name: str | None = None,
        tabify: bool = False,
    ) -> tuple[object, object]:
        key = (plugin_name, widget_name)
        if key in self._dock_widgets:
            return self._dock_widgets[key]

        dock_widget = SimpleNamespace(plugin_name=plugin_name, widget_name=widget_name, tabify=tabify)
        inner_widget = SimpleNamespace(plugin_name=plugin_name, widget_name=widget_name)
        self._dock_widgets[key] = (dock_widget, inner_widget)
        self.calls.append((plugin_name, widget_name, tabify))
        return self._dock_widgets[key]


class DummyViewer:
    def __init__(self) -> None:
        self.layers = DummyLayers()
        self.window = DummyWindow()


class DummyCoordinateSystemChangeParticipant:
    def __init__(self, callback: Callable[[CoordinateSystemChangeRequest], bool]) -> None:
        self._callback = callback

    def prepare_coordinate_system_change(self, request: CoordinateSystemChangeRequest) -> bool:
        return self._callback(request)


def _patch_shared_coordinate_system_names(monkeypatch, coordinate_systems_by_sdata_id: dict[int, list[str]]) -> None:
    def _get_coordinate_system_names(sdata: object) -> list[str]:
        return list(coordinate_systems_by_sdata_id.get(id(sdata), []))

    monkeypatch.setattr(app_state_module, "get_coordinate_system_names_from_sdata", _get_coordinate_system_names)
    monkeypatch.setattr(viewer_widget_module, "get_coordinate_system_names_from_sdata", _get_coordinate_system_names)
    monkeypatch.setattr(object_widget_module, "get_coordinate_system_names_from_sdata", _get_coordinate_system_names)


def _patch_empty_shared_widget_content(monkeypatch) -> None:
    monkeypatch.setattr(viewer_widget_module, "_get_labels_in_coordinate_system", lambda sdata, coordinate_system: [])
    monkeypatch.setattr(viewer_widget_module, "_get_images_in_coordinate_system", lambda sdata, coordinate_system: [])
    monkeypatch.setattr(
        object_widget_module,
        "get_spatialdata_labels_options_for_coordinate_system_from_sdata",
        lambda *, sdata, coordinate_system: [],
    )


def test_get_or_create_app_state_returns_same_state_for_same_viewer() -> None:
    viewer = DummyViewer()
    first = get_or_create_app_state(viewer)
    second = get_or_create_app_state(viewer)
    other = get_or_create_app_state(DummyViewer())

    assert first is second
    assert first.viewer is viewer
    assert other is not first


def test_spatiato_app_state_emits_sdata_changed(qtbot, sdata_blobs) -> None:
    state = SpatiatoAppState()

    with qtbot.waitSignal(state.sdata_changed) as blocker:
        state.set_sdata(sdata_blobs)

    assert blocker.args == [sdata_blobs]
    assert state.sdata is sdata_blobs

    with qtbot.waitSignal(state.sdata_changed) as blocker:
        state.clear_sdata(discard_current=True)

    assert blocker.args == [None]
    assert state.sdata is None


def test_spatiato_app_state_set_same_sdata_preserves_layers(monkeypatch) -> None:
    sdata = object()
    state = SpatiatoAppState()
    removed_sdata_calls: list[object] = []

    monkeypatch.setattr(app_state_module, "get_coordinate_system_names_from_sdata", lambda sdata: ["global"])
    monkeypatch.setattr(
        state.viewer_adapter,
        "remove_layers_for_sdata",
        lambda sdata: removed_sdata_calls.append(sdata) or [],
    )

    state.set_sdata(sdata)
    removed_sdata_calls.clear()

    state.set_sdata(sdata)

    assert state.sdata is sdata
    assert state.coordinate_system == "global"
    assert removed_sdata_calls == []


def test_spatiato_app_state_replacement_requires_explicit_discard(monkeypatch) -> None:
    first_sdata = object()
    second_sdata = object()
    state = SpatiatoAppState()
    monkeypatch.setattr(app_state_module, "get_coordinate_system_names_from_sdata", lambda _sdata: [])

    state.set_sdata(first_sdata)

    with pytest.raises(RuntimeError, match="requires explicit discard authorization"):
        state.set_sdata(second_sdata)

    assert state.sdata is first_sdata


def test_spatiato_app_state_authorized_replacement_discards_old_dirty_manifests(monkeypatch) -> None:
    """Accepted replacement removes all obsolete dirty-table tracking for the old SpatialData."""
    first_sdata = object()
    second_sdata = object()
    state = SpatiatoAppState()
    monkeypatch.setattr(app_state_module, "get_coordinate_system_names_from_sdata", lambda _sdata: [])
    state.set_sdata(first_sdata)
    state.record_table_mutation(
        TableStateChangedEvent(
            sdata=first_sdata,
            table_name="first_table",
            paths=frozenset({TableComponentPath("obs", ("annotation",))}),
            regions=("cells",),
            change_kind="updated",
            source="test",
        )
    )
    state.record_table_mutation(
        TableStateChangedEvent(
            sdata=first_sdata,
            table_name="second_table",
            paths=frozenset({TableComponentPath("obsm", ("features",))}),
            regions=("cells",),
            change_kind="updated",
            source="test",
        )
    )

    state.set_sdata(second_sdata, discard_current=True)

    assert state.sdata is second_sdata
    assert state.is_table_dirty(first_sdata, "first_table") is False
    assert state.is_table_dirty(first_sdata, "second_table") is False


def test_spatiato_app_state_records_component_tokens_and_emits_one_event(qtbot, sdata_blobs) -> None:
    state = SpatiatoAppState()
    paths = frozenset(
        {
            TableComponentPath("obsm", ("features_new",)),
            TableComponentPath("uns", ("feature_matrices", "features_new")),
        }
    )
    event = TableStateChangedEvent(
        sdata=sdata_blobs,
        table_name="table",
        paths=paths,
        regions=("blobs_labels",),
        change_kind="created",
        source="test",
    )

    assert state.is_table_dirty(sdata_blobs, "table") is False

    with qtbot.waitSignal(state.table_state_changed) as blocker:
        state.record_table_mutation(event)

    assert blocker.args == [event]
    assert state.is_table_dirty(sdata_blobs, "table") is True
    snapshot = state.snapshot_table_dirty_state(sdata_blobs, "table")
    assert snapshot.paths == paths
    tokens = tuple(token for _path, token in snapshot.captured_path_tokens)
    assert all(token is tokens[0] for token in tokens)

    state.acknowledge_table_write(snapshot, persisted_paths=paths)

    assert state.is_table_dirty(sdata_blobs, "table") is False


def test_spatiato_app_state_emits_table_wide_dirty_transitions(sdata_blobs) -> None:
    state = SpatiatoAppState()
    path = TableComponentPath("obs", ("user_class",))
    event = TableStateChangedEvent(
        sdata=sdata_blobs,
        table_name="table",
        paths=frozenset({path}),
        regions=("blobs_labels",),
        change_kind="updated",
        source="test",
    )
    dirty_events: list[object] = []
    state.table_dirty_state_changed.connect(dirty_events.append)

    state.record_table_mutation(event)
    snapshot = state.snapshot_table_dirty_state(sdata_blobs, "table")
    # Replacing the path token leaves the table-wide dirty boolean unchanged.
    state.record_table_mutation(event)
    state.acknowledge_table_write(snapshot, persisted_paths=frozenset({path}))

    assert dirty_events == [
        TableDirtyStateChangedEvent(
            sdata=sdata_blobs,
            table_name="table",
            is_dirty=True,
        )
    ]

    current_snapshot = state.snapshot_table_dirty_state(sdata_blobs, "table")
    state.acknowledge_table_write(current_snapshot, persisted_paths=frozenset({path}))

    assert dirty_events == [
        TableDirtyStateChangedEvent(
            sdata=sdata_blobs,
            table_name="table",
            is_dirty=True,
        ),
        TableDirtyStateChangedEvent(
            sdata=sdata_blobs,
            table_name="table",
            is_dirty=False,
        ),
    ]


def test_spatiato_app_state_emits_shapes_element_written(qtbot, sdata_blobs) -> None:
    state = SpatiatoAppState()
    event = ShapesElementWrittenEvent(
        sdata=sdata_blobs,
        shapes_name="new_regions",
        coordinate_system="global",
    )

    with qtbot.waitSignal(state.shapes_element_written) as blocker:
        state.emit_shapes_element_written(event)

    assert blocker.args == [event]


def test_spatiato_app_state_does_not_clear_a_newer_same_path_token(sdata_blobs) -> None:
    state = SpatiatoAppState()
    event = TableStateChangedEvent(
        sdata=sdata_blobs,
        table_name="table",
        paths=frozenset({TableComponentPath("obs", ("user_class",))}),
        regions=("blobs_labels",),
        change_kind="updated",
        source="test",
    )
    state.record_table_mutation(event)
    snapshot = state.snapshot_table_dirty_state(sdata_blobs, "table")
    state.record_table_mutation(event)

    state.record_persisted_table_change(event, snapshot)

    assert state.is_table_dirty(sdata_blobs, "table") is True


def test_spatiato_app_state_publishes_persisted_change_without_snapshot(qtbot, sdata_blobs) -> None:
    state = SpatiatoAppState()
    dirty_event = TableStateChangedEvent(
        sdata=sdata_blobs,
        table_name="table",
        paths=frozenset({TableComponentPath("obs", ("user_class",))}),
        regions=("blobs_labels",),
        change_kind="updated",
        source="test",
    )
    state.record_table_mutation(dirty_event)
    persisted_event = TableStateChangedEvent(
        sdata=sdata_blobs,
        table_name="table",
        paths=frozenset({TableComponentPath("obsm", ("features",))}),
        regions=("blobs_labels",),
        change_kind="created",
        source="test",
    )

    with qtbot.waitSignal(state.table_state_changed) as blocker:
        state.record_persisted_table_change(persisted_event)

    assert blocker.args == [persisted_event]
    # Without a snapshot nothing is cleared, and the persisted paths are not marked dirty.
    assert state.snapshot_table_dirty_state(sdata_blobs, "table").paths == {TableComponentPath("obs", ("user_class",))}


def test_spatiato_app_state_reload_clears_only_covered_paths(sdata_blobs) -> None:
    state = SpatiatoAppState()
    obs_path = TableComponentPath("obs", ("user_class",))
    feature_metadata_path = TableComponentPath("uns", ("feature_matrices", "features_1"))
    state.record_table_mutation(
        TableStateChangedEvent(
            sdata=sdata_blobs,
            table_name="table",
            paths=frozenset({obs_path, feature_metadata_path}),
            regions=("blobs_labels",),
            change_kind="updated",
            source="test",
        )
    )

    state.record_table_reload(
        TableStateChangedEvent(
            sdata=sdata_blobs,
            table_name="table",
            paths=frozenset({TableComponentPath("uns", ("feature_matrices",))}),
            regions=(),
            change_kind="reloaded",
            source="test",
        )
    )

    assert state.is_table_dirty(sdata_blobs, "table") is True
    assert state.snapshot_table_dirty_state(sdata_blobs, "table").paths == frozenset({obs_path})


def test_spatiato_app_state_prepares_registered_table_reload_participants_by_identity(
    sdata_blobs,
) -> None:
    state = SpatiatoAppState()
    calls: list[tuple[str, TableReloadRequest]] = []

    class Participant:
        def __init__(self, name: str) -> None:
            self.name = name

        def prepare_for_table_reload(self, request: TableReloadRequest) -> None:
            calls.append((self.name, request))

    first = Participant("first")
    second = Participant("second")
    request = TableReloadRequest(
        sdata=sdata_blobs,
        table_name="table",
        paths=frozenset({TableComponentPath("obs", ("user_class",))}),
        region_name="blobs_labels",
        source="test",
    )

    state.register_table_reload_participant(first)
    state.register_table_reload_participant(first)
    state.register_table_reload_participant(second)
    state.prepare_for_table_reload(request)

    assert calls == [("first", request), ("second", request)]
    assert state.unregister_table_reload_participant(first) is True
    assert state.unregister_table_reload_participant(first) is False

    calls.clear()
    state.prepare_for_table_reload(request)

    assert calls == [("second", request)]


def test_spatiato_app_state_set_coordinate_system_emits_event_and_prunes_layers(
    qtbot, monkeypatch, sdata_blobs
) -> None:
    state = SpatiatoAppState()
    state.sdata = sdata_blobs
    removed_calls: list[dict[str, object | None]] = []

    monkeypatch.setattr(
        state.viewer_adapter,
        "remove_layers_outside_coordinate_system",
        lambda *, sdata, coordinate_system: (
            removed_calls.append({"sdata": sdata, "coordinate_system": coordinate_system}) or []
        ),
    )

    with qtbot.waitSignal(state.coordinate_system_changed) as blocker:
        changed = state.set_coordinate_system("global", source="test")

    assert changed is True
    assert state.coordinate_system == "global"
    assert removed_calls == [{"sdata": sdata_blobs, "coordinate_system": "global"}]
    assert isinstance(blocker.args[0], CoordinateSystemChangedEvent)
    assert blocker.args[0] == CoordinateSystemChangedEvent(
        sdata=sdata_blobs,
        previous_coordinate_system=None,
        coordinate_system="global",
        source="test",
    )

    changed = state.set_coordinate_system("global", source="test")

    assert changed is False
    assert removed_calls == [{"sdata": sdata_blobs, "coordinate_system": "global"}]


def test_spatiato_app_state_coordinate_participant_rejects_before_event_or_layer_removal(
    monkeypatch,
    sdata_blobs,
) -> None:
    state = SpatiatoAppState()
    state.sdata = sdata_blobs
    state.coordinate_system = "global"
    requests: list[CoordinateSystemChangeRequest] = []
    coordinate_events: list[CoordinateSystemChangedEvent] = []
    removed_calls: list[object] = []
    monkeypatch.setattr(app_state_module, "get_coordinate_system_names_from_sdata", lambda _sdata: ["global", "local"])
    monkeypatch.setattr(
        state.viewer_adapter,
        "remove_layers_outside_coordinate_system",
        lambda **_kwargs: removed_calls.append(object()),
    )
    state.coordinate_system_changed.connect(coordinate_events.append)
    participant = DummyCoordinateSystemChangeParticipant(lambda request: requests.append(request) or False)
    state.register_coordinate_system_change_participant(participant)

    changed = state.set_coordinate_system("local", source="object_classification_widget")

    assert changed is False
    assert requests == [
        CoordinateSystemChangeRequest(
            sdata=sdata_blobs,
            previous_coordinate_system="global",
            coordinate_system="local",
            source="object_classification_widget",
        )
    ]
    assert state.coordinate_system == "global"
    assert coordinate_events == []
    assert removed_calls == []


def test_spatiato_app_state_coordinate_participant_runs_once_before_commit_and_supports_identity_safe_teardown(
    monkeypatch,
    sdata_blobs,
) -> None:
    state = SpatiatoAppState()
    state.sdata = sdata_blobs
    state.coordinate_system = "global"
    timeline: list[str] = []
    monkeypatch.setattr(app_state_module, "get_coordinate_system_names_from_sdata", lambda _sdata: ["global", "local"])
    monkeypatch.setattr(
        state.viewer_adapter,
        "remove_layers_outside_coordinate_system",
        lambda **_kwargs: timeline.append("layers_removed"),
    )
    state.coordinate_system_changed.connect(lambda _event: timeline.append("event_emitted"))

    def prepare(_request: CoordinateSystemChangeRequest) -> bool:
        assert state.coordinate_system == "global"
        assert timeline == []
        timeline.append("participant")
        return True

    participant = DummyCoordinateSystemChangeParticipant(prepare)
    state.register_coordinate_system_change_participant(participant)

    assert state.set_coordinate_system("local", source="viewer_widget") is True
    assert timeline == ["participant", "event_emitted", "layers_removed"]

    # A no-op does not ask the participant again.
    assert state.set_coordinate_system("local", source="viewer_widget") is False
    assert timeline == ["participant", "event_emitted", "layers_removed"]

    other_participant = DummyCoordinateSystemChangeParticipant(lambda _request: True)
    assert state.unregister_coordinate_system_change_participant(other_participant) is False
    with pytest.raises(RuntimeError, match="already registered"):
        state.register_coordinate_system_change_participant(other_participant)
    assert state.unregister_coordinate_system_change_participant(participant) is True
    state.register_coordinate_system_change_participant(other_participant)


def test_spatiato_app_state_set_sdata_keeps_previous_coordinate_system_when_still_valid(monkeypatch) -> None:
    first_sdata = object()
    second_sdata = object()
    state = SpatiatoAppState()
    removed_sdata_calls: list[object] = []
    coordinate_events: list[CoordinateSystemChangedEvent] = []

    coordinate_systems_by_sdata_id = {
        id(first_sdata): ["global", "local"],
        id(second_sdata): ["local", "zeta"],
    }
    monkeypatch.setattr(
        app_state_module,
        "get_coordinate_system_names_from_sdata",
        lambda sdata: coordinate_systems_by_sdata_id[id(sdata)],
    )
    monkeypatch.setattr(
        state.viewer_adapter,
        "remove_layers_for_sdata",
        lambda sdata: removed_sdata_calls.append(sdata) or [],
    )
    state.coordinate_system_changed.connect(coordinate_events.append)

    state.set_sdata(first_sdata)
    assert state.coordinate_system == "global"

    coordinate_events.clear()
    state.set_coordinate_system("local", source="test")
    assert state.coordinate_system == "local"

    coordinate_events.clear()
    removed_sdata_calls.clear()
    state.set_sdata(second_sdata, discard_current=True)

    assert state.sdata is second_sdata
    assert state.coordinate_system == "local"
    assert removed_sdata_calls == [first_sdata]
    assert coordinate_events == []


def test_spatiato_app_state_set_sdata_selects_first_sorted_coordinate_system_when_previous_is_invalid(
    monkeypatch,
) -> None:
    first_sdata = object()
    second_sdata = object()
    state = SpatiatoAppState()
    removed_sdata_calls: list[object] = []
    coordinate_events: list[CoordinateSystemChangedEvent] = []

    coordinate_systems_by_sdata_id = {
        id(first_sdata): ["local"],
        id(second_sdata): ["alpha", "beta"],
    }
    monkeypatch.setattr(
        app_state_module,
        "get_coordinate_system_names_from_sdata",
        lambda sdata: coordinate_systems_by_sdata_id[id(sdata)],
    )
    monkeypatch.setattr(
        state.viewer_adapter,
        "remove_layers_for_sdata",
        lambda sdata: removed_sdata_calls.append(sdata) or [],
    )
    state.coordinate_system_changed.connect(coordinate_events.append)

    state.set_sdata(first_sdata)
    assert state.coordinate_system == "local"

    coordinate_events.clear()
    removed_sdata_calls.clear()
    state.set_sdata(second_sdata, discard_current=True)

    assert state.sdata is second_sdata
    assert state.coordinate_system == "alpha"
    assert removed_sdata_calls == [first_sdata]
    assert coordinate_events == [
        CoordinateSystemChangedEvent(
            sdata=second_sdata,
            previous_coordinate_system="local",
            coordinate_system="alpha",
            source="set_sdata",
        )
    ]

    coordinate_events.clear()
    removed_sdata_calls.clear()
    state.clear_sdata(discard_current=True)

    assert state.sdata is None
    assert state.coordinate_system is None
    assert removed_sdata_calls == [second_sdata]
    assert coordinate_events == [
        CoordinateSystemChangedEvent(
            sdata=None,
            previous_coordinate_system="alpha",
            coordinate_system=None,
            source="set_sdata",
        )
    ]


def test_widgets_share_app_state_for_same_viewer(qtbot) -> None:
    viewer = DummyViewer()
    feature_widget = FeatureExtractionWidget(viewer)
    object_widget = ObjectClassificationWidget(viewer)

    qtbot.addWidget(feature_widget)
    qtbot.addWidget(object_widget)

    assert feature_widget.app_state is object_widget.app_state
    assert feature_widget.app_state is get_or_create_app_state(viewer)


def test_shared_viewer_and_object_widgets_keep_previous_coordinate_system_when_replacing_sdata(
    qtbot, monkeypatch
) -> None:
    first_sdata = object()
    second_sdata = object()
    _patch_shared_coordinate_system_names(
        monkeypatch,
        {
            id(first_sdata): ["global", "local"],
            id(second_sdata): ["local", "zeta"],
        },
    )
    _patch_empty_shared_widget_content(monkeypatch)

    viewer = DummyViewer()
    app_state = get_or_create_app_state(viewer)
    viewer_widget = ViewerWidget(viewer)
    object_widget = ObjectClassificationWidget(viewer)

    qtbot.addWidget(viewer_widget)
    qtbot.addWidget(object_widget)

    with qtbot.waitSignal(app_state.sdata_changed):
        app_state.set_sdata(first_sdata)

    assert app_state.coordinate_system == "global"
    assert viewer_widget.coordinate_system_combo.currentText() == "global"
    assert object_widget.coordinate_system_combo.currentText() == "global"

    with qtbot.waitSignal(app_state.coordinate_system_changed):
        viewer_widget.coordinate_system_combo.setCurrentIndex(1)

    assert app_state.coordinate_system == "local"
    assert viewer_widget.coordinate_system_combo.currentText() == "local"
    assert object_widget.coordinate_system_combo.currentText() == "local"

    with qtbot.waitSignal(app_state.sdata_changed):
        app_state.set_sdata(second_sdata, discard_current=True)

    assert app_state.sdata is second_sdata
    assert app_state.coordinate_system == "local"
    assert viewer_widget.coordinate_system_combo.currentText() == "local"
    assert object_widget.coordinate_system_combo.currentText() == "local"


def test_shared_coordinate_participant_rejection_restores_initiating_widget_selectors(qtbot, monkeypatch) -> None:
    sdata = object()
    _patch_shared_coordinate_system_names(monkeypatch, {id(sdata): ["global", "local"]})
    _patch_empty_shared_widget_content(monkeypatch)

    viewer = DummyViewer()
    app_state = get_or_create_app_state(viewer)
    viewer_widget = ViewerWidget(viewer)
    object_widget = ObjectClassificationWidget(viewer)
    qtbot.addWidget(viewer_widget)
    qtbot.addWidget(object_widget)
    app_state.set_sdata(sdata)
    requests: list[CoordinateSystemChangeRequest] = []
    participant = DummyCoordinateSystemChangeParticipant(lambda request: requests.append(request) or False)
    app_state.register_coordinate_system_change_participant(participant)

    viewer_widget.coordinate_system_combo.setCurrentIndex(1)

    assert app_state.coordinate_system == "global"
    assert viewer_widget.coordinate_system_combo.currentText() == "global"
    assert object_widget.coordinate_system_combo.currentText() == "global"

    object_widget.coordinate_system_combo.setCurrentIndex(1)

    assert app_state.coordinate_system == "global"
    assert viewer_widget.coordinate_system_combo.currentText() == "global"
    assert object_widget.coordinate_system_combo.currentText() == "global"
    assert [request.source for request in requests] == ["viewer_widget", "object_classification_widget"]


def test_shared_viewer_and_object_widgets_select_first_coordinate_system_when_previous_is_invalid_and_clear_on_sdata_clear(
    qtbot, monkeypatch
) -> None:
    first_sdata = object()
    second_sdata = object()
    _patch_shared_coordinate_system_names(
        monkeypatch,
        {
            id(first_sdata): ["local"],
            id(second_sdata): ["alpha", "beta"],
        },
    )
    _patch_empty_shared_widget_content(monkeypatch)

    viewer = DummyViewer()
    app_state = get_or_create_app_state(viewer)
    viewer_widget = ViewerWidget(viewer)
    object_widget = ObjectClassificationWidget(viewer)

    qtbot.addWidget(viewer_widget)
    qtbot.addWidget(object_widget)

    with qtbot.waitSignal(app_state.sdata_changed):
        app_state.set_sdata(first_sdata)

    assert app_state.coordinate_system == "local"
    assert viewer_widget.coordinate_system_combo.currentText() == "local"
    assert object_widget.coordinate_system_combo.currentText() == "local"

    with qtbot.waitSignal(app_state.sdata_changed):
        app_state.set_sdata(second_sdata, discard_current=True)

    assert app_state.sdata is second_sdata
    assert app_state.coordinate_system == "alpha"
    assert viewer_widget.coordinate_system_combo.currentText() == "alpha"
    assert object_widget.coordinate_system_combo.currentText() == "alpha"

    with qtbot.waitSignal(app_state.sdata_changed):
        app_state.clear_sdata(discard_current=True)

    assert app_state.sdata is None
    assert app_state.coordinate_system is None
    assert viewer_widget.coordinate_system_combo.count() == 0
    assert object_widget.coordinate_system_combo.count() == 0
    assert viewer_widget.coordinate_system_combo.currentIndex() == -1
    assert object_widget.coordinate_system_combo.currentIndex() == -1


def test_interactive_headless_sets_sdata_without_running_event_loop(monkeypatch, sdata_blobs) -> None:
    viewer = DummyViewer()
    run_calls: list[str] = []

    monkeypatch.setattr(interactive_module.napari, "run", lambda: run_calls.append("run"))

    interactive = interactive_module.Interactive(sdata_blobs, viewer=viewer, headless=True)

    assert interactive.viewer is viewer
    assert interactive.app_state is get_or_create_app_state(viewer)
    assert interactive.app_state.sdata is sdata_blobs
    assert run_calls == []
    assert viewer.window.calls == [
        ("spatiato", "Viewer", True),
        ("spatiato", "Feature Extraction", True),
        ("spatiato", "Image Histogram", True),
        ("spatiato", "Object Classification", True),
        ("spatiato", "Annotation", True),
    ]


def test_interactive_explicitly_replaces_existing_spatialdata_without_dialog(monkeypatch, sdata_blobs) -> None:
    viewer = DummyViewer()
    app_state = get_or_create_app_state(viewer)
    old_sdata = object()
    monkeypatch.setattr(
        app_state_module,
        "get_coordinate_system_names_from_sdata",
        lambda sdata: [] if sdata is old_sdata else ["global"],
    )
    app_state.set_sdata(old_sdata)
    monkeypatch.setattr(interactive_module.napari, "run", lambda: None)

    interactive = interactive_module.Interactive(
        sdata_blobs,
        viewer=viewer,
        headless=True,
        widgets=(),
    )

    assert interactive.app_state.sdata is sdata_blobs


def test_interactive_configures_default_and_explicit_triangulation_backends(monkeypatch, sdata_blobs) -> None:
    viewer = DummyViewer()
    configured_backends: list[str] = []

    monkeypatch.setattr(
        interactive_module,
        "configure_shapes_triangulation_backend",
        configured_backends.append,
    )

    interactive_module.Interactive(sdata_blobs, viewer=viewer, headless=True, widgets=())
    interactive_module.Interactive(
        sdata_blobs,
        viewer=viewer,
        headless=True,
        widgets=(),
        triangulation_backend="numba",
    )

    assert configured_backends == ["bermuda", "numba"]


def test_interactive_sets_async_slicing_when_requested(monkeypatch, sdata_blobs) -> None:
    viewer = DummyViewer()
    async_slicing_calls: list[bool] = []

    monkeypatch.setattr(interactive_module.napari, "run", lambda: None)
    monkeypatch.setattr(
        interactive_module,
        "_set_napari_async_slicing",
        lambda enabled: async_slicing_calls.append(enabled),
    )

    interactive_module.Interactive(sdata_blobs, viewer=viewer, headless=True, widgets=(), async_slicing=True)
    interactive_module.Interactive(sdata_blobs, viewer=viewer, headless=True, widgets=(), async_slicing=False)

    assert async_slicing_calls == [True, False]


def test_interactive_can_dock_a_single_widget(monkeypatch, sdata_blobs) -> None:
    viewer = DummyViewer()
    run_calls: list[str] = []

    monkeypatch.setattr(interactive_module.napari, "run", lambda: run_calls.append("run"))

    interactive = interactive_module.Interactive(
        sdata_blobs,
        viewer=viewer,
        headless=True,
        widgets="viewer",
    )

    assert interactive.app_state.sdata is sdata_blobs
    assert run_calls == []
    assert viewer.window.calls == [("spatiato", "Viewer", True)]


def test_interactive_can_dock_shapes_annotation_widget(monkeypatch, sdata_blobs) -> None:
    viewer = DummyViewer()

    monkeypatch.setattr(interactive_module.napari, "run", lambda: None)

    interactive_module.Interactive(
        sdata_blobs,
        viewer=viewer,
        headless=True,
        widgets="shapes_annotation",
    )

    assert viewer.window.calls == [("spatiato", "Annotation", True)]


def test_interactive_can_dock_a_widget_subset(monkeypatch, sdata_blobs) -> None:
    viewer = DummyViewer()

    monkeypatch.setattr(interactive_module.napari, "run", lambda: None)

    interactive_module.Interactive(
        sdata_blobs,
        viewer=viewer,
        headless=True,
        widgets=("viewer", "shapes_annotation", "viewer"),
    )

    assert viewer.window.calls == [
        ("spatiato", "Viewer", True),
        ("spatiato", "Annotation", True),
    ]


def test_interactive_all_docks_every_spatiato_widget(monkeypatch, sdata_blobs) -> None:
    viewer = DummyViewer()

    monkeypatch.setattr(interactive_module.napari, "run", lambda: None)

    interactive_module.Interactive(
        sdata_blobs,
        viewer=viewer,
        headless=True,
        widgets="all",
    )

    assert viewer.window.calls == [
        ("spatiato", "Viewer", True),
        ("spatiato", "Feature Extraction", True),
        ("spatiato", "Image Histogram", True),
        ("spatiato", "Object Classification", True),
        ("spatiato", "Annotation", True),
    ]


def test_interactive_can_load_sdata_without_docking_widgets(monkeypatch, sdata_blobs) -> None:
    viewer = DummyViewer()

    monkeypatch.setattr(interactive_module.napari, "run", lambda: None)

    interactive = interactive_module.Interactive(
        sdata_blobs,
        viewer=viewer,
        headless=True,
        widgets=(),
    )

    assert interactive.app_state.sdata is sdata_blobs
    assert viewer.window.calls == []


def test_interactive_rejects_unknown_widget_selection(monkeypatch, sdata_blobs) -> None:
    viewer = DummyViewer()

    monkeypatch.setattr(interactive_module.napari, "run", lambda: None)

    with pytest.raises(ValueError, match=r"Unknown Spatiato widget selection 'features'.*shapes_annotation"):
        interactive_module.Interactive(
            sdata_blobs,
            viewer=viewer,
            headless=True,
            widgets="features",  # type: ignore[arg-type]
        )

    assert get_or_create_app_state(viewer).sdata is None
    assert viewer.window.calls == []


def test_interactive_auto_runs_and_reuses_existing_plugin_widgets(monkeypatch, sdata_blobs) -> None:
    viewer = DummyViewer()
    run_calls: list[str] = []

    monkeypatch.setattr(interactive_module.napari, "run", lambda: run_calls.append("run"))

    interactive_module.Interactive(sdata_blobs, viewer=viewer, headless=False)
    interactive_module.Interactive(sdata_blobs, viewer=viewer, headless=False)

    assert run_calls == ["run", "run"]
    assert viewer.window.calls == [
        ("spatiato", "Viewer", True),
        ("spatiato", "Feature Extraction", True),
        ("spatiato", "Image Histogram", True),
        ("spatiato", "Object Classification", True),
        ("spatiato", "Annotation", True),
    ]
