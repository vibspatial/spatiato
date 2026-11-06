from __future__ import annotations

import copy
from collections.abc import Callable
from html import unescape
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from harpy.table.canonical_centers import CANONICAL_OBSM_KEY
from matplotlib.colors import to_rgba
from qtpy.QtCore import QObject, Qt, Signal
from spatialdata import SpatialData
from spatialdata.transformations import Identity, set_transformation

import spatiato.widgets.persistence.controls as persistence_controls_module
import spatiato.widgets.spatial_query.widget as widget_module
from spatiato._app_state import ShapesElementReloadedEvent, TableReloadRequest, TableStateChangedEvent
from spatiato.core.persistence import TableComponentPath
from spatiato.core.spatial_query import CanonicalCenterQueryResult
from spatiato.core.spatial_query.canonical_centers import (
    CANONICAL_CENTERS_PATHS,
    canonical_centers_status,
    compute_canonical_centers,
    read_canonical_centers_region,
)
from spatiato.viewer._styling import MISSING_CATEGORICAL_COLOR
from spatiato.widgets.annotation.models import AnnotationContext, ShapesAnnotationTarget
from spatiato.widgets.spatial_query.controller import CanonicalCentersChangeResult
from spatiato.widgets.spatial_query.widget import (
    CANONICAL_CENTERS_SOURCE,
    SPATIAL_QUERY_ANNOTATION_SOURCE,
    SpatialQuery,
)


class _EventEmitter:
    def __init__(self) -> None:
        self._callbacks: list[Callable[[object], None]] = []

    def connect(self, callback: Callable[[object], None]) -> None:
        self._callbacks.append(callback)

    def emit(self, value: object) -> None:
        event = SimpleNamespace(value=value)
        for callback in list(self._callbacks):
            callback(event)


class _Selection:
    def __init__(self) -> None:
        self.active: object | None = None

    def select_only(self, layer: object) -> None:
        self.active = layer


class _Layers(list):
    def __init__(self) -> None:
        super().__init__()
        self.selection = _Selection()
        self.events = SimpleNamespace(
            inserted=_EventEmitter(),
            removed=_EventEmitter(),
            reordered=_EventEmitter(),
        )


class _Viewer:
    def __init__(self) -> None:
        self.layers = _Layers()

    def add_layer(self, layer: object) -> object:
        self.layers.append(layer)
        self.layers.events.inserted.emit(layer)
        return layer


def _context(
    sdata: SpatialData,
    *,
    coordinate_system: str = "global",
    dirty: bool = False,
    create_new: bool = False,
) -> AnnotationContext:
    target = (
        ShapesAnnotationTarget.create_new() if create_new else ShapesAnnotationTarget.edit_existing("blobs_circles")
    )
    return AnnotationContext(
        sdata=sdata,
        coordinate_system=coordinate_system,
        shapes_target=target,
        has_unsaved_shapes_changes=dirty,
    )


def _select_labels(widget: SpatialQuery, labels_name: str = "blobs_labels") -> None:
    index = widget.labels_combo.findData(labels_name)
    assert index >= 0
    widget.labels_combo.setCurrentIndex(index)


def _add_default_annotation_column(sdata: SpatialData) -> None:
    table = sdata.tables["table"]
    table.obs["spatial_annotation"] = pd.Categorical(
        ["A"] * table.n_obs,
        categories=["A"],
    )


def _query_result(sdata: SpatialData, *, count: int = 2) -> CanonicalCenterQueryResult:
    if canonical_centers_status(sdata, table_name="table").state != "valid":
        compute_canonical_centers(sdata, table_name="table")
    region = read_canonical_centers_region(sdata, table_name="table", labels_name="blobs_labels")
    return CanonicalCenterQueryResult(
        canonical_centers=region,
        matched_instance_ids=np.sort(region.instance_ids)[:count],
    )


def _status_text(label) -> str:
    return unescape(label.text())


def test_spatial_query_shell_starts_inactive_without_parent_context(qtbot) -> None:
    widget = SpatialQuery()
    qtbot.addWidget(widget)

    assert widget.selected_spatialdata is None
    assert widget.labels_combo.placeholderText() == "Choose a labels element"
    assert widget.labels_combo.isEnabled() is False
    assert widget.table_combo.isEnabled() is False
    assert widget.run_button.isEnabled() is False
    assert widget.status_label.objectName() == "spatial_query_status_label"
    assert widget.run_button.text() == "Apply Annotation"
    assert not hasattr(widget, "cache_status_label")
    assert not hasattr(widget, "readiness_status_label")
    assert "No SpatialData Loaded" in _status_text(widget.status_label)


def test_spatial_query_persistence_controls_bind_to_selected_labels_table(
    qtbot,
    backed_sdata_blobs: SpatialData,
) -> None:
    widget = SpatialQuery(_Viewer())
    qtbot.addWidget(widget)

    widget.apply_annotation_context(_context(backed_sdata_blobs))
    _select_labels(widget)

    request = widget.persistence_controls.controller.capture_table_reload_request(source="test")

    assert request.sdata is backed_sdata_blobs
    assert request.table_name == "table"
    assert request.region_name == "blobs_labels"
    assert widget.persistence_controls.reload_button.isEnabled()


def test_spatial_query_destruction_unregisters_table_reload_participant(qtbot) -> None:
    viewer = _Viewer()
    widget = SpatialQuery(viewer)
    app_state = widget.app_state

    assert any(participant is widget for participant in app_state._table_reload_participants)

    widget.deleteLater()
    qtbot.waitUntil(lambda: not any(participant is widget for participant in app_state._table_reload_participants))


def test_spatial_query_prepares_only_for_selected_table_and_ignores_late_query_result(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    sdata_blobs: SpatialData,
) -> None:
    """The participant hook alone invalidates a Run of the selected table, and a late query result is rejected.

    A reload of another table preserves the Run. This is one of two paths that
    each invalidate a Run on their own; the other is the `reloaded` event
    (`test_spatial_query_reloaded_event_alone_invalidates_a_run_of_the_selected_table`).
    Together they show that the hook is a defense (see `TableReloadParticipant`).
    """
    widget = SpatialQuery(_Viewer())
    qtbot.addWidget(widget)
    widget.apply_annotation_context(_context(sdata_blobs))
    _select_labels(widget)
    widget.new_column_edit.setText("reviewed_annotation")
    widget.annotation_value_edit.setText("tumor")
    monkeypatch.setattr(widget._controller, "start_spatial_query", lambda *args, **kwargs: True)
    qtbot.mouseClick(widget.run_button, Qt.MouseButton.LeftButton)
    assert widget._active_run_intent is not None

    unrelated_request = TableReloadRequest(
        sdata=sdata_blobs,
        table_name="other_table",
        paths=frozenset({TableComponentPath("obs", ("reviewed_annotation",))}),
        region_name="blobs_labels",
        source="test",
    )
    widget.prepare_for_table_reload(unrelated_request)
    assert widget._active_run_intent is not None

    matching_request = TableReloadRequest(
        sdata=sdata_blobs,
        table_name="table",
        paths=frozenset({TableComponentPath("uns", ("unrelated_metadata",))}),
        region_name="another_region",
        source="test",
    )
    widget.prepare_for_table_reload(matching_request)
    assert widget._active_run_intent is None

    widget._on_query_ready(_query_result(sdata_blobs))

    assert "reviewed_annotation" not in sdata_blobs.tables["table"].obs


def test_spatial_query_reloaded_event_alone_invalidates_a_run_of_the_selected_table(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    sdata_blobs: SpatialData,
) -> None:
    """The `reloaded` event alone invalidates a Run of the selected table, and a late query result is rejected.

    No reload happens and `prepare_for_table_reload()` is never called. This is
    one of two paths that each invalidate a Run on their own; the other is the
    participant hook
    (`test_spatial_query_prepares_only_for_selected_table_and_ignores_late_query_result`).
    Together they show that the hook is a defense (see `TableReloadParticipant`).
    """
    widget = SpatialQuery(_Viewer())
    qtbot.addWidget(widget)
    widget.apply_annotation_context(_context(sdata_blobs))
    _select_labels(widget)
    widget.new_column_edit.setText("reviewed_annotation")
    widget.annotation_value_edit.setText("tumor")
    monkeypatch.setattr(widget._controller, "start_spatial_query", lambda *args, **kwargs: True)
    qtbot.mouseClick(widget.run_button, Qt.MouseButton.LeftButton)
    assert widget._active_run_intent is not None

    def reloaded_event(table_name: str) -> TableStateChangedEvent:
        return TableStateChangedEvent(
            sdata=sdata_blobs,
            table_name=table_name,
            paths=frozenset({TableComponentPath("uns", ("unrelated_metadata",))}),
            regions=(),
            change_kind="reloaded",
            source="test",
        )

    widget.app_state.record_table_reload(reloaded_event("other_table"))
    assert widget._active_run_intent is not None

    widget.app_state.record_table_reload(reloaded_event("table"))
    assert widget._active_run_intent is None

    widget._on_query_ready(_query_result(sdata_blobs))

    assert "reviewed_annotation" not in sdata_blobs.tables["table"].obs


def test_spatial_query_cancelled_dirty_reload_preserves_accepted_run(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    backed_sdata_blobs: SpatialData,
) -> None:
    widget = SpatialQuery(_Viewer())
    qtbot.addWidget(widget)
    widget.apply_annotation_context(_context(backed_sdata_blobs))
    _select_labels(widget)
    widget.new_column_edit.setText("reviewed_annotation")
    widget.annotation_value_edit.setText("tumor")
    monkeypatch.setattr(widget._controller, "start_spatial_query", lambda *args, **kwargs: True)
    qtbot.mouseClick(widget.run_button, Qt.MouseButton.LeftButton)
    accepted_intent = widget._active_run_intent
    assert accepted_intent is not None

    backed_sdata_blobs.tables["table"].uns["reload_test"] = {"dirty": True}
    widget.app_state.record_table_mutation(
        TableStateChangedEvent(
            sdata=backed_sdata_blobs,
            table_name="table",
            paths=frozenset({TableComponentPath("uns", ("reload_test",))}),
            regions=("blobs_labels",),
            change_kind="created",
            source="test",
        )
    )
    monkeypatch.setattr(
        widget.persistence_controls,
        "_prompt_dirty_reload_decision",
        lambda: persistence_controls_module._DirtyReloadDecision.CANCEL,
    )

    widget.persistence_controls.reload_button.click()

    assert widget._active_run_intent is accepted_intent
    assert backed_sdata_blobs.tables["table"].uns["reload_test"] == {"dirty": True}
    assert widget.app_state.is_table_dirty(backed_sdata_blobs, "table")


def test_spatial_query_shapes_reload_invalidates_only_a_matching_accepted_run(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    sdata_blobs: SpatialData,
) -> None:
    """Reject late query results only when the reloaded Shapes source belongs to that Run."""
    widget = SpatialQuery(_Viewer())
    qtbot.addWidget(widget)
    widget.apply_annotation_context(_context(sdata_blobs))
    _select_labels(widget)
    widget.new_column_edit.setText("reviewed_annotation")
    widget.annotation_value_edit.setText("tumor")
    monkeypatch.setattr(widget._controller, "start_spatial_query", lambda *args, **kwargs: True)
    qtbot.mouseClick(widget.run_button, Qt.MouseButton.LeftButton)
    accepted_intent = widget._active_run_intent
    assert accepted_intent is not None

    widget.app_state.emit_shapes_element_reloaded(
        ShapesElementReloadedEvent(
            sdata=sdata_blobs,
            shapes_name="other_shapes",
            coordinate_system="global",
        )
    )
    assert widget._active_run_intent is accepted_intent

    widget.app_state.emit_shapes_element_reloaded(
        ShapesElementReloadedEvent(
            sdata=sdata_blobs,
            shapes_name="blobs_circles",
            coordinate_system="global",
        )
    )
    assert widget._active_run_intent is None

    widget._on_query_ready(_query_result(sdata_blobs))

    assert "reviewed_annotation" not in sdata_blobs.tables["table"].obs


def test_spatial_query_adopts_reloaded_columns_centers_status_and_neutral_styling(
    qtbot,
    backed_sdata_blobs: SpatialData,
) -> None:
    table = backed_sdata_blobs.tables["table"]
    table.obs["old_annotation"] = pd.Categorical(
        ["old"] * table.n_obs,
        categories=["old"],
    )
    table.uns["old_annotation_colors"] = ["#ff0000"]
    viewer = _Viewer()
    widget = SpatialQuery(viewer)
    qtbot.addWidget(widget)
    widget.apply_annotation_context(_context(backed_sdata_blobs))
    _select_labels(widget)
    widget.column_mode_combo.setCurrentIndex(widget.column_mode_combo.findData("existing"))
    widget.existing_column_combo.setCurrentIndex(widget.existing_column_combo.findData("old_annotation"))
    widget.new_column_edit.setText("draft_annotation")

    assert widget.selected_column_name == "old_annotation"
    assert np.allclose(viewer.layers[0].colormap.map(1), np.asarray(to_rgba("#ff0000"), dtype=np.float32))

    widget.persistence_controls.controller.reload_table_state()

    assert "old_annotation" not in table.obs
    assert "old_annotation_colors" not in table.uns
    assert widget.selected_column_mode == "new"
    assert widget.new_column_edit.text() == "draft_annotation"
    assert widget.selected_column_name == "draft_annotation"
    assert widget.centers_status is not None
    neutral_rgba = np.asarray(to_rgba(MISSING_CATEGORICAL_COLOR), dtype=np.float32)
    assert np.allclose(viewer.layers[0].colormap.map(1), neutral_rgba)


def test_spatial_query_shell_requires_an_explicit_new_column_name_and_captures_run_inputs(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    sdata_blobs: SpatialData,
) -> None:
    viewer = _Viewer()
    widget = SpatialQuery(viewer)
    qtbot.addWidget(widget)
    inspection_calls = 0
    real_status = widget_module.canonical_centers_status

    def record_inspection(*args, **kwargs):
        nonlocal inspection_calls
        inspection_calls += 1
        return real_status(*args, **kwargs)

    monkeypatch.setattr(widget_module, "canonical_centers_status", record_inspection)
    obs_before = sdata_blobs.tables["table"].obs.copy(deep=True)
    uns_before = copy.deepcopy(sdata_blobs.tables["table"].uns)

    widget.apply_annotation_context(_context(sdata_blobs))

    assert widget.selected_labels_name is None
    assert widget.selected_table_name is None
    assert widget.centers_status is None
    assert inspection_calls == 0
    assert viewer.layers == []

    _select_labels(widget)

    assert widget.selected_labels_name == "blobs_labels"
    assert widget.selected_table_name == "table"
    assert widget.selected_column_mode == "new"
    assert widget.new_column_edit.text() == ""
    assert widget.new_column_edit.placeholderText() == "spatial_annotation"
    assert widget.selected_column_name is None
    assert widget.centers_status is not None
    assert widget.run_button.isEnabled() is False
    assert "Enter a new annotation column name" in _status_text(widget.status_label)
    assert inspection_calls == 1
    assert len(viewer.layers) == 1  # Explicit labels selection may claim primary-label styling.
    neutral_rgba = np.asarray(to_rgba(MISSING_CATEGORICAL_COLOR), dtype=np.float32)
    assert np.allclose(viewer.layers[0].colormap.map(1), neutral_rgba)
    assert np.allclose(viewer.layers[0].colormap.map(10_000), neutral_rgba)

    widget.new_column_edit.setText("reviewed_annotation")
    assert widget.selected_column_name == "reviewed_annotation"
    assert widget.run_button.isEnabled() is False
    assert "Enter a non-empty annotation value" in _status_text(widget.status_label)
    widget.annotation_value_edit.setText("tumor")
    assert widget.run_button.isEnabled() is True
    assert inspection_calls == 1  # Status rendering reuses the checked status.

    starts: list[tuple[object, object, str, str, str]] = []

    def start_spatial_query(sdata, centers_status, *, labels_name, shapes_name, coordinate_system, dirty_snapshot):
        del dirty_snapshot
        starts.append((sdata, centers_status, labels_name, shapes_name, coordinate_system))
        return True

    monkeypatch.setattr(widget._controller, "start_spatial_query", start_spatial_query)
    qtbot.mouseClick(widget.run_button, Qt.MouseButton.LeftButton)

    assert len(starts) == 1
    assert starts[0][0] is sdata_blobs
    assert starts[0][1] is widget.centers_status
    assert starts[0][2:] == ("blobs_labels", "blobs_circles", "global")
    assert widget._active_run_intent is not None
    assert widget._active_run_intent.annotation_action == "set"
    assert widget._active_run_intent.annotation_value == "tumor"
    assert inspection_calls == 2  # Run checks the canonical centers status once more.
    pd.testing.assert_frame_equal(sdata_blobs.tables["table"].obs, obs_before)
    assert sdata_blobs.tables["table"].uns == uns_before


def test_spatial_query_shell_uses_compatible_default_and_styles_only_after_explicit_selection(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    _add_default_annotation_column(sdata_blobs)
    table = sdata_blobs.tables["table"]
    table.uns["spatial_annotation_colors"] = ["#ff0000"]
    obs_before = table.obs.copy(deep=True)
    uns_before = copy.deepcopy(table.uns)
    viewer = _Viewer()
    widget = SpatialQuery(viewer)
    qtbot.addWidget(widget)

    widget.apply_annotation_context(_context(sdata_blobs))

    assert widget.selected_labels_name is None
    assert viewer.layers == []

    _select_labels(widget)

    assert widget.selected_column_mode == "existing"
    assert widget.selected_column_name == "spatial_annotation"
    assert len(viewer.layers) == 1
    assert viewer.layers.selection.active is viewer.layers[0]
    assert viewer.layers[0].name == "blobs_labels"
    assert np.allclose(viewer.layers[0].colormap.map(1), np.asarray(to_rgba("#ff0000"), dtype=np.float32))
    assert widget.new_column_edit.text() == ""
    assert widget.new_column_edit.placeholderText() == "spatial_annotation"

    widget.column_mode_combo.setCurrentIndex(widget.column_mode_combo.findData("new"))

    assert widget.new_column_edit.text() == ""
    assert widget.selected_column_name is None
    assert widget.run_button.isEnabled() is False
    assert "Enter a new annotation column name" in _status_text(widget.status_label)
    assert np.allclose(
        viewer.layers[0].colormap.map(1),
        np.asarray(to_rgba(MISSING_CATEGORICAL_COLOR), dtype=np.float32),
    )
    pd.testing.assert_frame_equal(table.obs, obs_before)
    assert table.uns == uns_before


def test_spatial_query_shell_lists_user_class_but_excludes_classifier_outputs(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    table = sdata_blobs.tables["table"]
    table.obs["user_class"] = pd.Categorical([1] * table.n_obs, categories=[1])
    table.uns["user_class_colors"] = ["#ff0000"]
    table.obs["pred_class"] = pd.Categorical([1] * table.n_obs, categories=[1])
    table.obs["pred_confidence"] = np.ones(table.n_obs, dtype=np.float64)
    obs_before = table.obs.copy(deep=True)
    uns_before = copy.deepcopy(table.uns)
    viewer = _Viewer()
    widget = SpatialQuery(viewer)
    qtbot.addWidget(widget)

    widget.apply_annotation_context(_context(sdata_blobs))
    _select_labels(widget)

    available_columns = [
        widget.existing_column_combo.itemData(index) for index in range(widget.existing_column_combo.count())
    ]
    assert "user_class" in available_columns
    assert "pred_class" not in available_columns
    assert "pred_confidence" not in available_columns

    widget.column_mode_combo.setCurrentIndex(widget.column_mode_combo.findData("existing"))
    widget.existing_column_combo.setCurrentIndex(widget.existing_column_combo.findData("user_class"))

    assert widget.selected_column_name == "user_class"
    assert np.allclose(viewer.layers[0].colormap.map(1), np.asarray(to_rgba("#ff0000"), dtype=np.float32))
    pd.testing.assert_frame_equal(table.obs, obs_before)
    assert table.uns == uns_before


def test_spatial_query_shell_uses_typed_set_editor_and_explicit_remove_action(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    table = sdata_blobs.tables["table"]
    table.obs["user_class"] = pd.Categorical([pd.NA] * table.n_obs, categories=[])
    widget = SpatialQuery(_Viewer())
    qtbot.addWidget(widget)
    widget.apply_annotation_context(_context(sdata_blobs))
    _select_labels(widget)

    widget.column_mode_combo.setCurrentIndex(widget.column_mode_combo.findData("existing"))
    widget.existing_column_combo.setCurrentIndex(widget.existing_column_combo.findData("user_class"))

    assert widget.annotation_value_stack.currentWidget() is widget.annotation_class_spinbox
    assert widget.selected_annotation_action == "set"
    assert widget.selected_annotation_value == 1
    assert widget.run_button.isEnabled()

    widget.annotation_class_spinbox.setValue(4)
    assert widget.selected_annotation_value == 4

    widget.annotation_action_combo.setCurrentIndex(widget.annotation_action_combo.findData("remove"))

    assert widget.selected_annotation_action == "remove"
    assert widget.selected_annotation_value is None
    assert widget.annotation_value_stack.isHidden()
    assert widget.run_button.isEnabled()

    widget.column_mode_combo.setCurrentIndex(widget.column_mode_combo.findData("new"))

    assert widget.annotation_action_combo.findData("remove") == -1
    assert widget.selected_annotation_action == "set"
    assert widget.annotation_value_stack.currentWidget() is widget.annotation_value_edit


def test_spatial_query_shell_rejects_reserved_object_classification_new_columns(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    widget = SpatialQuery(_Viewer())
    qtbot.addWidget(widget)
    widget.apply_annotation_context(_context(sdata_blobs))
    _select_labels(widget)

    for column_name in ("user_class", "pred_class", "pred_confidence"):
        widget.new_column_edit.setText(column_name)
        assert widget.selected_column_name is None
        assert widget.run_button.isEnabled() is False
        status_text = _status_text(widget.status_label)
        assert f'New annotation column "{column_name}" is reserved for Object Classification' in status_text


def test_spatial_query_shell_uses_named_default_when_preferred_existing_column_disappears(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    _add_default_annotation_column(sdata_blobs)
    table = sdata_blobs.tables["table"]
    table.uns["spatial_annotation_colors"] = ["#ff0000"]
    table.obs["old_annotation"] = pd.Categorical(
        ["old"] * table.n_obs,
        categories=["old"],
    )
    viewer = _Viewer()
    widget = SpatialQuery(viewer)
    qtbot.addWidget(widget)
    widget.apply_annotation_context(_context(sdata_blobs))
    _select_labels(widget)

    old_index = widget.existing_column_combo.findData("old_annotation")
    assert old_index >= 0
    widget.existing_column_combo.setCurrentIndex(old_index)
    assert widget.selected_column_name == "old_annotation"
    widget.new_column_edit.setText("draft_annotation")

    del table.obs["old_annotation"]
    widget._refresh_columns(
        preferred_mode="existing",
        preferred_existing_column="old_annotation",
        preferred_new_column=widget.new_column_edit.text(),
    )

    assert widget.selected_column_mode == "existing"
    assert widget.selected_column_name == "spatial_annotation"
    assert widget.new_column_edit.text() == "draft_annotation"

    widget.column_mode_combo.setCurrentIndex(widget.column_mode_combo.findData("new"))

    assert widget.existing_column_combo.currentIndex() == -1
    assert widget.existing_column_combo.placeholderText() == "Choose an existing column"
    assert widget.new_column_edit.text() == "draft_annotation"
    assert widget.selected_column_name == "draft_annotation"
    neutral_rgba = np.asarray(to_rgba(MISSING_CATEGORICAL_COLOR), dtype=np.float32)
    assert np.allclose(viewer.layers[0].colormap.map(1), neutral_rgba)

    widget.column_mode_combo.setCurrentIndex(widget.column_mode_combo.findData("existing"))

    assert widget.existing_column_combo.currentIndex() == -1
    assert widget.selected_column_name is None
    assert widget.new_column_edit.text() == "draft_annotation"
    assert np.allclose(viewer.layers[0].colormap.map(1), neutral_rgba)

    widget.existing_column_combo.setCurrentIndex(widget.existing_column_combo.findData("spatial_annotation"))

    assert widget.selected_column_name == "spatial_annotation"
    assert np.allclose(viewer.layers[0].colormap.map(1), np.asarray(to_rgba("#ff0000"), dtype=np.float32))


def test_spatial_query_shell_explains_incompatible_preferred_column_with_empty_new_draft(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    table = sdata_blobs.tables["table"]
    table.obs["spatial_annotation"] = pd.Series(
        ["existing"] * table.n_obs,
        index=table.obs.index,
        dtype="string",
    )
    viewer = _Viewer()
    widget = SpatialQuery(viewer)
    qtbot.addWidget(widget)

    widget.apply_annotation_context(_context(sdata_blobs))
    _select_labels(widget)

    assert widget.selected_column_mode == "new"
    assert widget.new_column_edit.text() == ""
    assert widget.new_column_edit.placeholderText() == "spatial_annotation"
    assert widget.selected_column_name is None
    assert widget.run_button.isEnabled() is False
    status_text = _status_text(widget.status_label)
    assert 'Existing annotation column "spatial_annotation" cannot be used' in status_text
    assert "categorical column containing only strings or positive integers" in status_text
    assert "different New-column name" in status_text
    assert len(viewer.layers) == 1
    assert np.allclose(
        viewer.layers[0].colormap.map(1),
        np.asarray(to_rgba(MISSING_CATEGORICAL_COLOR), dtype=np.float32),
    )

    widget.new_column_edit.setText("reviewed_annotation")

    assert widget.selected_column_name == "reviewed_annotation"
    widget.annotation_value_edit.setText("tumor")
    assert widget.run_button.isEnabled() is True


def test_spatial_query_shell_clears_new_column_draft_when_table_changes(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    sdata_blobs.tables["second_table"] = sdata_blobs.tables["table"].copy()
    widget = SpatialQuery(_Viewer())
    qtbot.addWidget(widget)
    widget.apply_annotation_context(_context(sdata_blobs))
    _select_labels(widget)

    widget.new_column_edit.setText("draft_annotation")
    widget.annotation_value_edit.setText("tumor")
    assert widget.selected_column_name == "draft_annotation"
    assert widget.run_button.isEnabled() is True

    next_table = "table" if widget.selected_table_name == "second_table" else "second_table"
    next_table_index = widget.table_combo.findData(next_table)
    assert next_table_index >= 0
    widget.table_combo.setCurrentIndex(next_table_index)

    assert widget.selected_table_name == next_table
    assert widget.selected_column_mode == "new"
    assert widget.new_column_edit.text() == ""
    assert widget.new_column_edit.placeholderText() == "spatial_annotation"
    assert widget.selected_column_name is None
    assert widget.run_button.isEnabled() is False


def test_spatial_query_shell_keeps_invalid_centers_ready_for_recalculation(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    table = sdata_blobs.tables["table"]
    # A matrix without its record is invalid.
    table.obsm[CANONICAL_OBSM_KEY] = np.zeros((table.n_obs, 3), dtype=np.float64)
    widget = SpatialQuery(_Viewer())
    qtbot.addWidget(widget)

    widget.apply_annotation_context(_context(sdata_blobs))
    _select_labels(widget)
    widget.new_column_edit.setText("reviewed_annotation")
    widget.annotation_value_edit.setText("tumor")

    centers_status = widget.centers_status
    assert centers_status is not None
    assert centers_status.state == "invalid"
    assert widget.run_button.isEnabled() is True
    status_text = _status_text(widget.status_label)
    assert "Spatial Query Ready" in status_text
    assert 'Centers for every labels element of table "table" will be recalculated' in status_text
    # harpy's reason is the tooltip, not the visible text.
    assert centers_status.reason not in status_text


@pytest.mark.parametrize(
    ("dirty", "create_new", "expected_status"),
    [
        (True, False, "Save or Discard Shapes Changes"),
        (False, True, "Saved Shapes Required"),
    ],
)
def test_spatial_query_shell_shapes_context_blocks_run(
    qtbot,
    sdata_blobs: SpatialData,
    dirty: bool,
    create_new: bool,
    expected_status: str,
) -> None:
    widget = SpatialQuery()
    qtbot.addWidget(widget)

    widget.apply_annotation_context(_context(sdata_blobs, dirty=dirty, create_new=create_new))
    _select_labels(widget)

    assert widget.centers_status is not None
    assert widget.run_button.isEnabled() is False
    assert expected_status in _status_text(widget.status_label)


def test_spatial_query_shell_coordinate_change_clears_valid_labels_without_reinspection(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    sdata_blobs: SpatialData,
) -> None:
    inspection_calls = 0
    real_status = widget_module.canonical_centers_status

    def record_inspection(*args, **kwargs):
        nonlocal inspection_calls
        inspection_calls += 1
        return real_status(*args, **kwargs)

    monkeypatch.setattr(widget_module, "canonical_centers_status", record_inspection)
    viewer = _Viewer()
    widget = SpatialQuery(viewer)
    qtbot.addWidget(widget)
    widget.apply_annotation_context(_context(sdata_blobs))
    _select_labels(widget)
    assert inspection_calls == 1
    assert len(viewer.layers) == 1

    set_transformation(
        sdata_blobs.labels["blobs_labels"],
        Identity(),
        to_coordinate_system="shared",
    )
    widget.apply_annotation_context(_context(sdata_blobs, coordinate_system="shared"))

    assert widget.labels_combo.isEnabled() is True
    assert widget.labels_combo.currentIndex() == -1
    assert widget.labels_combo.currentText() == ""
    assert widget.labels_combo.placeholderText() == "Choose a labels element"
    assert widget.selected_labels_name is None
    assert widget.selected_table_name is None
    assert widget.centers_status is None
    assert widget.run_button.isEnabled() is False
    assert inspection_calls == 1


def test_spatial_query_shell_tracks_only_its_selected_primary_labels_layer(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = _Viewer()
    widget = SpatialQuery(viewer)
    qtbot.addWidget(widget)
    widget.apply_annotation_context(_context(sdata_blobs))

    unrelated_result = widget.app_state.viewer_adapter.ensure_labels_loaded(
        sdata_blobs,
        "blobs_multiscale_labels",
        "global",
    )
    assert widget.selected_labels_name is None  # Layer insertion never selects it.

    _select_labels(widget)
    widget.new_column_edit.setText("reviewed_annotation")
    widget.annotation_value_edit.setText("tumor")
    selected_layer = widget.app_state.viewer_adapter.get_loaded_primary_labels_layer(
        sdata_blobs,
        "blobs_labels",
        "global",
    )
    assert selected_layer is not None
    centers_status = widget.centers_status
    assert centers_status is not None

    viewer.layers.remove(unrelated_result.layer)
    viewer.layers.events.removed.emit(unrelated_result.layer)

    assert widget.selected_labels_name == "blobs_labels"
    assert widget.centers_status is centers_status
    assert widget.run_button.isEnabled() is True

    viewer.layers.remove(selected_layer)
    viewer.layers.events.removed.emit(selected_layer)

    assert widget.labels_combo.currentIndex() == -1
    assert widget.labels_combo.placeholderText() == "Choose a labels element"
    assert widget.selected_labels_name is None
    assert widget.selected_table_name is None
    assert widget.centers_status is None
    assert widget.run_button.isEnabled() is False


def _centers_change(sdata: SpatialData) -> CanonicalCentersChangeResult:
    """Calculate the table's canonical centers, as the centers job would, and describe the change."""
    change_kind = compute_canonical_centers(sdata, table_name="table")
    return CanonicalCentersChangeResult(
        sdata=sdata,
        table_name="table",
        change_kind=change_kind,
        regions=("blobs_labels",),
        dirty_snapshot=None,
    )


def test_spatial_query_child_records_a_canonical_centers_change_once_as_a_mutation_when_unbacked(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    widget = SpatialQuery(_Viewer())
    qtbot.addWidget(widget)
    widget.apply_annotation_context(_context(sdata_blobs))
    _select_labels(widget)
    assert widget.centers_status is not None
    assert widget.centers_status.state == "absent"
    emitted_events: list[TableStateChangedEvent] = []
    widget.app_state.table_state_changed.connect(emitted_events.append)

    widget._on_canonical_centers_changed(_centers_change(sdata_blobs))

    assert len(emitted_events) == 1
    event = emitted_events[0]
    assert event.sdata is sdata_blobs
    assert event.table_name == "table"
    assert event.paths == CANONICAL_CENTERS_PATHS
    assert event.regions == ("blobs_labels",)
    assert event.change_kind == "created"
    assert event.source == CANONICAL_CENTERS_SOURCE
    # Unbacked: nothing is on disk, so the change is an unsaved mutation.
    assert widget.app_state.snapshot_table_dirty_state(sdata_blobs, "table").paths == CANONICAL_CENTERS_PATHS
    assert widget.centers_status.state == "valid"


def test_spatial_query_child_records_a_canonical_centers_change_as_persisted_when_backed(
    qtbot,
    backed_sdata_blobs: SpatialData,
) -> None:
    widget = SpatialQuery(_Viewer())
    qtbot.addWidget(widget)
    widget.apply_annotation_context(_context(backed_sdata_blobs))
    _select_labels(widget)
    emitted_events: list[TableStateChangedEvent] = []
    widget.app_state.table_state_changed.connect(emitted_events.append)

    widget._on_canonical_centers_changed(_centers_change(backed_sdata_blobs))

    assert [event.paths for event in emitted_events] == [CANONICAL_CENTERS_PATHS]
    # harpy wrote the store, so nothing becomes dirty.
    assert widget.app_state.is_table_dirty(backed_sdata_blobs, "table") is False


def test_spatial_query_child_ignores_a_canonical_centers_change_of_another_spatialdata(
    qtbot,
    sdata_blobs: SpatialData,
    sdata_blobs_multi_region: SpatialData,
) -> None:
    widget = SpatialQuery(_Viewer())
    qtbot.addWidget(widget)
    widget.apply_annotation_context(_context(sdata_blobs))
    _select_labels(widget)
    emitted_events: list[TableStateChangedEvent] = []
    widget.app_state.table_state_changed.connect(emitted_events.append)

    widget._on_canonical_centers_changed(
        CanonicalCentersChangeResult(
            sdata=sdata_blobs_multi_region,
            table_name="table_multi",
            change_kind="created",
            regions=("blobs_labels", "blobs_labels_2"),
            dirty_snapshot=None,
        )
    )

    assert emitted_events == []


class _DeferredCentersWorker(QObject):
    returned = Signal(object)
    errored = Signal(object)
    finished = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.quit_called = False

    def start(self) -> None:
        pass

    def quit(self) -> None:
        self.quit_called = True


def test_spatial_query_waits_for_a_detached_centers_job_before_a_new_run(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    widget = SpatialQuery(_Viewer())
    qtbot.addWidget(widget)
    centers_worker = _DeferredCentersWorker()
    widget._controller._create_canonical_centers_worker = (  # type: ignore[method-assign]
        lambda job: centers_worker
    )
    widget.apply_annotation_context(_context(sdata_blobs))
    _select_labels(widget)
    widget.new_column_edit.setText("reviewed_annotation")
    widget.annotation_value_edit.setText("tumor")

    qtbot.mouseClick(widget.run_button, Qt.MouseButton.LeftButton)
    assert widget._controller.active_phase == "canonical_centers"
    assert widget.app_state.is_table_busy(sdata_blobs, "table") is True

    # Changing an input invalidates the Run; the centers job keeps writing.
    widget.annotation_value_edit.setText("stroma")

    assert widget._active_run_intent is None
    assert centers_worker.quit_called is False
    assert widget._controller.has_job_in_flight is True
    assert widget.run_button.isEnabled() is False
    assert "Waiting for the previous center calculation to finish writing." in _status_text(widget.status_label)

    # The job ends: its change is recorded, and a new Run may start.
    compute_canonical_centers(sdata_blobs, table_name="table")
    centers_worker.returned.emit("created")
    centers_worker.finished.emit()

    assert widget.app_state.is_table_busy(sdata_blobs, "table") is False
    assert widget.centers_status is not None
    assert widget.centers_status.state == "valid"
    assert widget.run_button.isEnabled() is True
    assert "Spatial Query Ready" in _status_text(widget.status_label)


def test_spatial_query_waits_while_another_job_keeps_the_selected_table_busy(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    widget = SpatialQuery(_Viewer())
    qtbot.addWidget(widget)
    widget.apply_annotation_context(_context(sdata_blobs))
    _select_labels(widget)
    widget.new_column_edit.setText("reviewed_annotation")
    widget.annotation_value_edit.setText("tumor")
    assert widget.run_button.isEnabled() is True
    assert widget.centers_status is not None
    assert widget.centers_status.state == "absent"

    other_job = widget.app_state.mark_table_busy(sdata_blobs, "table", reason="Another job is still writing.")

    assert widget.run_button.isEnabled() is False
    assert "Table Busy" in _status_text(widget.status_label)
    assert 'Table "table" is busy: Another job is still writing.' in _status_text(widget.status_label)

    # The other job wrote the centers; they are checked again when it ends.
    compute_canonical_centers(sdata_blobs, table_name="table")
    widget.app_state.release_table_busy(other_job)

    assert widget.run_button.isEnabled() is True
    assert widget.centers_status.state == "valid"
    assert "Spatial Query Ready" in _status_text(widget.status_label)


def test_spatial_query_child_applies_new_annotation_and_publishes_exact_paths(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    sdata_blobs: SpatialData,
) -> None:
    query_result = _query_result(sdata_blobs, count=2)
    table = sdata_blobs.tables["table"]
    widget = SpatialQuery(_Viewer())
    qtbot.addWidget(widget)
    widget.apply_annotation_context(_context(sdata_blobs))
    _select_labels(widget)
    widget.new_column_edit.setText("spatial_annotation")
    widget.annotation_value_edit.setText("tumor")
    monkeypatch.setattr(widget._controller, "start_spatial_query", lambda *args, **kwargs: True)
    emitted_events: list[TableStateChangedEvent] = []
    widget.app_state.table_state_changed.connect(emitted_events.append)

    qtbot.mouseClick(widget.run_button, Qt.MouseButton.LeftButton)
    widget._on_query_ready(query_result)

    matched_rows = query_result.canonical_centers.row_positions[
        np.isin(query_result.canonical_centers.instance_ids, query_result.matched_instance_ids)
    ]
    assert table.obs["spatial_annotation"].iloc[matched_rows].tolist() == ["tumor", "tumor"]
    assert table.obs["spatial_annotation"].drop(table.obs.index[matched_rows]).isna().all()
    assert table.uns["spatial_annotation_colors"]
    assert len(emitted_events) == 1
    event = emitted_events[0]
    assert event.source == SPATIAL_QUERY_ANNOTATION_SOURCE
    assert event.change_kind == "created"
    assert event.regions == ("blobs_labels",)
    assert event.paths == frozenset(
        {
            TableComponentPath("obs", ("spatial_annotation",)),
            TableComponentPath("uns", ("spatial_annotation_colors",)),
        }
    )
    assert widget.selected_column_mode == "existing"
    assert widget.selected_column_name == "spatial_annotation"
    assert widget.new_column_edit.text() == ""
    assert "Annotation Applied" in _status_text(widget.status_label)
    assert "Overwritten: 0" in _status_text(widget.status_label)
    assert widget.app_state.snapshot_table_dirty_state(sdata_blobs, "table").paths == event.paths


def test_spatial_query_child_removes_annotation_without_palette_event(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    sdata_blobs: SpatialData,
) -> None:
    query_result = _query_result(sdata_blobs, count=2)
    table = sdata_blobs.tables["table"]
    table.obs["spatial_annotation"] = pd.Categorical(["tumor"] * table.n_obs, categories=["tumor"])
    table.uns["spatial_annotation_colors"] = ["#ff0000"]
    widget = SpatialQuery(_Viewer())
    qtbot.addWidget(widget)
    widget.apply_annotation_context(_context(sdata_blobs))
    _select_labels(widget)
    widget.annotation_action_combo.setCurrentIndex(widget.annotation_action_combo.findData("remove"))
    monkeypatch.setattr(widget._controller, "start_spatial_query", lambda *args, **kwargs: True)
    emitted_events: list[TableStateChangedEvent] = []
    widget.app_state.table_state_changed.connect(emitted_events.append)

    qtbot.mouseClick(widget.run_button, Qt.MouseButton.LeftButton)
    widget._on_query_ready(query_result)

    matched_rows = query_result.canonical_centers.row_positions[
        np.isin(query_result.canonical_centers.instance_ids, query_result.matched_instance_ids)
    ]
    assert table.obs["spatial_annotation"].iloc[matched_rows].isna().all()
    assert table.uns["spatial_annotation_colors"] == ["#ff0000"]
    assert len(emitted_events) == 1
    assert emitted_events[0].change_kind == "updated"
    assert emitted_events[0].paths == frozenset({TableComponentPath("obs", ("spatial_annotation",))})
    assert "Removed annotations from 2 matched labeled objects" in _status_text(widget.status_label)


def test_spatial_query_child_invalidates_captured_run_when_annotation_value_changes(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    sdata_blobs: SpatialData,
) -> None:
    widget = SpatialQuery(_Viewer())
    qtbot.addWidget(widget)
    widget.apply_annotation_context(_context(sdata_blobs))
    _select_labels(widget)
    widget.new_column_edit.setText("first_annotation")
    widget.annotation_value_edit.setText("tumor")
    monkeypatch.setattr(widget._controller, "start_spatial_query", lambda *args, **kwargs: True)
    cancellation_calls = 0

    def cancel() -> bool:
        nonlocal cancellation_calls
        cancellation_calls += 1
        return True

    monkeypatch.setattr(widget._controller, "cancel_active_operation", cancel)
    qtbot.mouseClick(widget.run_button, Qt.MouseButton.LeftButton)
    assert widget._active_run_intent is not None

    widget.annotation_value_edit.setText("stroma")

    assert cancellation_calls == 1
    assert widget._active_run_intent is None
    assert widget._status_card_spec is None
