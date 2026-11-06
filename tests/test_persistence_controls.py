from __future__ import annotations

from html import unescape

from spatialdata import SpatialData

from spatiato._app_state import (
    SpatiatoAppState,
    TableReloadRequest,
    TableStateChangedEvent,
)
from spatiato.core.persistence import TableComponentPath
from spatiato.widgets.persistence import controls as persistence_controls_module
from spatiato.widgets.persistence.controls import TablePersistenceControls


def _dirty_user_class(
    app_state: SpatiatoAppState,
    sdata: SpatialData,
    *,
    table_name: str = "table",
) -> None:
    app_state.record_table_mutation(
        TableStateChangedEvent(
            sdata=sdata,
            table_name=table_name,
            paths=frozenset({TableComponentPath("obs", ("user_class",))}),
            regions=("blobs_labels",),
            change_kind="updated",
            source="test",
        )
    )


def test_bound_controls_synchronize_after_one_control_writes(qtbot, backed_sdata_blobs: SpatialData) -> None:
    app_state = SpatiatoAppState()
    first = TablePersistenceControls(app_state)
    second = TablePersistenceControls(app_state)
    qtbot.addWidget(first)
    qtbot.addWidget(second)
    first.bind(backed_sdata_blobs, "table", "blobs_labels")
    second.bind(backed_sdata_blobs, "table", "blobs_labels")

    _dirty_user_class(app_state, backed_sdata_blobs)

    assert first.write_button.isEnabled()
    assert second.write_button.isEnabled()

    first.write_button.click()

    assert app_state.is_table_dirty(backed_sdata_blobs, "table") is False
    assert not first.write_button.isEnabled()
    assert not second.write_button.isEnabled()


def test_dirty_event_refreshes_only_controls_bound_to_affected_table(
    qtbot,
    monkeypatch,
    sdata_blobs: SpatialData,
    sdata_blobs_multi_region: SpatialData,
) -> None:
    app_state = SpatiatoAppState()
    affected = TablePersistenceControls(app_state)
    unrelated = TablePersistenceControls(app_state)
    qtbot.addWidget(affected)
    qtbot.addWidget(unrelated)
    affected.bind(sdata_blobs, "table", "blobs_labels")
    unrelated.bind(sdata_blobs_multi_region, "table_multi", "blobs_labels")
    refresh_calls: list[str] = []
    monkeypatch.setattr(affected, "refresh", lambda: refresh_calls.append("affected"))
    monkeypatch.setattr(unrelated, "refresh", lambda: refresh_calls.append("unrelated"))

    _dirty_user_class(app_state, sdata_blobs)

    assert refresh_calls == ["affected"]


def test_reload_request_prepares_participants_before_execution(
    qtbot,
    monkeypatch,
    backed_sdata_blobs: SpatialData,
) -> None:
    app_state = SpatiatoAppState()
    controls = TablePersistenceControls(
        app_state,
        reload_source="object_classification",
    )
    qtbot.addWidget(controls)
    controls.bind(backed_sdata_blobs, "table", "blobs_labels")
    request = TableReloadRequest(
        sdata=backed_sdata_blobs,
        table_name="table",
        paths=frozenset({TableComponentPath("obs", ("user_class",))}),
        region_name="blobs_labels",
        source="object_classification",
    )
    calls: list[str] = []

    class Participant:
        def prepare_for_table_reload(self, current_request: TableReloadRequest) -> None:
            assert current_request is request
            calls.append("prepare")

    app_state.register_table_reload_participant(Participant())
    monkeypatch.setattr(
        controls.controller,
        "capture_table_reload_request",
        lambda *, source: request,
    )
    monkeypatch.setattr(
        controls.controller,
        "_apply_table_reload_request",
        lambda current_request: calls.append("reload"),
    )

    controls.reload_button.click()

    assert calls == ["prepare", "reload"]


def test_participant_failure_stops_reload_and_reports_feedback(
    qtbot,
    monkeypatch,
    backed_sdata_blobs: SpatialData,
) -> None:
    app_state = SpatiatoAppState()
    controls = TablePersistenceControls(app_state)
    qtbot.addWidget(controls)
    controls.bind(backed_sdata_blobs, "table", "blobs_labels")
    request = TableReloadRequest(
        sdata=backed_sdata_blobs,
        table_name="table",
        paths=frozenset({TableComponentPath("obs", ("user_class",))}),
        region_name="blobs_labels",
        source="table_persistence_controls",
    )
    reload_calls: list[TableReloadRequest] = []

    class FailingParticipant:
        def prepare_for_table_reload(self, current_request: TableReloadRequest) -> None:
            del current_request
            raise RuntimeError("injected preparation failure")

    app_state.register_table_reload_participant(FailingParticipant())
    monkeypatch.setattr(
        controls.controller,
        "capture_table_reload_request",
        lambda *, source: request,
    )
    monkeypatch.setattr(
        controls.controller,
        "_apply_table_reload_request",
        reload_calls.append,
    )

    controls.reload_button.click()

    assert reload_calls == []
    assert "injected preparation failure" in controls.feedback_label.text()


def test_controls_disable_write_and_reload_while_the_table_is_busy(qtbot, backed_sdata_blobs: SpatialData) -> None:
    app_state = SpatiatoAppState()
    controls = TablePersistenceControls(app_state)
    qtbot.addWidget(controls)
    controls.bind(backed_sdata_blobs, "table", "blobs_labels")
    _dirty_user_class(app_state, backed_sdata_blobs)
    assert controls.write_button.isEnabled()
    assert controls.reload_button.isEnabled()

    mark = app_state.mark_table_busy(backed_sdata_blobs, "table", reason="Still writing.")

    assert not controls.write_button.isEnabled()
    assert not controls.reload_button.isEnabled()
    assert 'Table "table" is busy: Still writing.' in unescape(controls.write_button.toolTip())
    assert 'Table "table" is busy: Still writing.' in unescape(controls.reload_button.toolTip())

    app_state.release_table_busy(mark)

    assert controls.write_button.isEnabled()
    assert controls.reload_button.isEnabled()


def test_write_of_a_busy_table_is_refused_with_the_reason(qtbot, backed_sdata_blobs: SpatialData) -> None:
    app_state = SpatiatoAppState()
    controls = TablePersistenceControls(app_state)
    qtbot.addWidget(controls)
    controls.bind(backed_sdata_blobs, "table", "blobs_labels")
    _dirty_user_class(app_state, backed_sdata_blobs)
    app_state.mark_table_busy(backed_sdata_blobs, "table", reason="Still writing.")

    # The button is disabled; the guard covers every other caller.
    assert controls.write_table_state() is False

    assert 'Table "table" is busy: Still writing.' in unescape(controls.feedback_label.text())
    assert app_state.is_table_dirty(backed_sdata_blobs, "table") is True


def test_reload_is_refused_when_the_table_becomes_busy_while_its_dialog_is_open(
    qtbot,
    monkeypatch,
    backed_sdata_blobs: SpatialData,
) -> None:
    app_state = SpatiatoAppState()
    controls = TablePersistenceControls(app_state)
    qtbot.addWidget(controls)
    controls.bind(backed_sdata_blobs, "table", "blobs_labels")
    _dirty_user_class(app_state, backed_sdata_blobs)
    prepare_calls: list[TableReloadRequest] = []

    class Participant:
        def prepare_for_table_reload(self, request: TableReloadRequest) -> None:
            prepare_calls.append(request)

    app_state.register_table_reload_participant(Participant())

    def start_a_job_then_discard() -> persistence_controls_module._DirtyReloadDecision:
        app_state.mark_table_busy(backed_sdata_blobs, "table", reason="Still writing.")
        return persistence_controls_module._DirtyReloadDecision.RELOAD_DISCARD

    monkeypatch.setattr(controls, "_prompt_dirty_reload_decision", start_a_job_then_discard)

    assert controls.reload_table_state() is False

    assert 'Table "table" is busy: Still writing.' in unescape(controls.feedback_label.text())
    assert prepare_calls == []
    assert app_state.is_table_dirty(backed_sdata_blobs, "table") is True
