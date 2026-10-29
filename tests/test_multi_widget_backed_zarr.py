from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_rgba
from napari.layers import Labels
from qtpy.QtCore import QObject, Qt, Signal
from qtpy.QtWidgets import QCheckBox
from spatialdata import SpatialData, read_zarr

import spatiato.widgets.object_classification.controller as classifier_module
import spatiato.widgets.persistence.controls as persistence_controls_module
from spatiato._app_state import TableStateChangedEvent, get_or_create_app_state
from spatiato.core.feature_matrix_metadata import register_feature_matrix_metadata
from spatiato.core.object_classification.annotation import (
    USER_CLASS_COLORS_KEY,
    USER_CLASS_COLUMN,
)
from spatiato.core.persistence import TableComponentPath, write_table_components
from spatiato.core.spatial_query import (
    CANONICAL_CACHE_PATHS,
    CANONICAL_OBSM_KEY,
    CanonicalCacheState,
    CanonicalCenterQueryResult,
    apply_canonical_cache_update,
    build_canonical_cache_update_payload,
    inspect_canonical_cache,
    read_canonical_centers_from_cache,
)
from spatiato.viewer._styling import MISSING_CATEGORICAL_COLOR
from spatiato.widgets.annotation.models import AnnotationContext, ShapesAnnotationTarget
from spatiato.widgets.feature_extraction.controller import FeatureExtractionResult
from spatiato.widgets.feature_extraction.widget import FeatureExtractionWidget
from spatiato.widgets.object_classification.controller import CLASSIFIER_CONFIG_KEY
from spatiato.widgets.object_classification.widget import ObjectClassificationWidget
from spatiato.widgets.spatial_query.widget import SpatialQuery
from spatiato.widgets.viewer.widget import ViewerWidget


class _EventEmitter:
    def __init__(self) -> None:
        self._callbacks: list[Callable[[object], None]] = []

    def connect(self, callback: Callable[[object], None]) -> None:
        self._callbacks.append(callback)

    def disconnect(self, callback: Callable[[object], None]) -> None:
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def emit(self, value: object | None = None) -> None:
        event = SimpleNamespace(value=value)
        for callback in tuple(self._callbacks):
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

    def remove(self, layer: object) -> None:
        super().remove(layer)
        self.events.removed.emit(layer)


class _Viewer:
    def __init__(self) -> None:
        self.layers = _Layers()

    def add_layer(self, layer: object) -> object:
        self.layers.append(layer)
        self.layers.events.inserted.emit(layer)
        return layer


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


def _annotation_context(sdata: SpatialData) -> AnnotationContext:
    return AnnotationContext(
        sdata=sdata,
        coordinate_system="global",
        shapes_target=ShapesAnnotationTarget.edit_existing("blobs_circles"),
        has_unsaved_shapes_changes=False,
    )


def _select_spatial_query_labels(widget: SpatialQuery) -> None:
    index = widget.labels_combo.findData("blobs_labels")
    assert index >= 0
    widget.labels_combo.setCurrentIndex(index)
    assert widget.selected_table_name == "table"


def _select_object_classification_labels(widget: ObjectClassificationWidget) -> None:
    assert widget.segmentation_combo.count() > 0
    widget.segmentation_combo.setCurrentIndex(0)
    assert widget.selected_segmentation_name == "blobs_labels"
    assert widget.selected_table_name == "table"


def _configure_feature_extraction(widget: FeatureExtractionWidget, *, feature_key: str = "features") -> None:
    widget._coordinate_system_checkboxes["global"].setChecked(True)
    card = widget._triplet_card_widgets_by_coordinate_system["global"]
    card.segmentation_combo.setCurrentIndex(0)
    feature_checkbox = widget.findChild(QCheckBox, "feature_checkbox_area")
    assert feature_checkbox is not None
    feature_checkbox.setChecked(True)
    widget.output_key_line_edit.setText(feature_key)
    assert widget.selected_table_name == "table"
    assert widget.calculate_button.isEnabled()


def _install_canonical_cache(sdata: SpatialData) -> None:
    report = inspect_canonical_cache(
        sdata,
        table_name="table",
        labels_name="blobs_labels",
    )
    if report.state is CanonicalCacheState.VALID:
        return
    payload = build_canonical_cache_update_payload(
        binding=report.binding,
        centers=np.zeros((report.binding.n_obs, 3), dtype=np.float64),
        source_signature=report.source_signature,
    )
    apply_canonical_cache_update(sdata, payload)


def _canonical_query_result(sdata: SpatialData) -> CanonicalCenterQueryResult:
    _install_canonical_cache(sdata)
    report = inspect_canonical_cache(
        sdata,
        table_name="table",
        labels_name="blobs_labels",
    )
    assert report.state is CanonicalCacheState.VALID
    centers = read_canonical_centers_from_cache(sdata, report)
    return CanonicalCenterQueryResult(
        canonical_centers=centers,
        matched_instance_ids=np.sort(centers.binding.instance_ids)[:2],
    )


def _table_event(
    sdata: SpatialData,
    *,
    paths: frozenset[TableComponentPath],
    source: str,
    change_kind: str = "updated",
) -> TableStateChangedEvent:
    return TableStateChangedEvent(
        sdata=sdata,
        table_name="table",
        paths=paths,
        regions=("blobs_labels",),
        change_kind=change_kind,
        source=source,
    )


def _assert_color(layer: Labels, value: int, expected: str) -> None:
    assert np.allclose(
        layer.colormap.map(value),
        np.asarray(to_rgba(expected), dtype=np.float32),
    )


def _dense_array(value: object) -> np.ndarray:
    toarray = getattr(value, "toarray", None)
    return np.asarray(toarray() if callable(toarray) else value)


def test_mixed_table_mutations_share_dirty_controls_and_round_trip_declared_components(
    qtbot,
    backed_sdata_blobs: SpatialData,
) -> None:
    """One shared write persists mixed producer state and cleans every bound control."""
    viewer = _Viewer()
    app_state = get_or_create_app_state(viewer)
    app_state.set_sdata(backed_sdata_blobs)
    spatial_query = SpatialQuery(viewer)
    object_classification = ObjectClassificationWidget(viewer)
    qtbot.addWidget(spatial_query)
    qtbot.addWidget(object_classification)
    spatial_query.apply_annotation_context(_annotation_context(backed_sdata_blobs))
    _select_spatial_query_labels(spatial_query)
    _select_object_classification_labels(object_classification)

    table = backed_sdata_blobs.tables["table"]
    original_x = _dense_array(table.X).copy()
    original_var = table.var.copy(deep=True)
    original_features = np.asarray(table.obsm["features_1"]).copy()

    # These untracked local values prove component persistence does not fall
    # back to rewriting unrelated obsm/uns state.
    table.obsm["untracked_local"] = np.full((table.n_obs, 1), 99.0)
    table.uns["untracked_local"] = {"value": 99}

    _install_canonical_cache(backed_sdata_blobs)
    app_state.record_table_mutation(
        _table_event(
            backed_sdata_blobs,
            paths=CANONICAL_CACHE_PATHS,
            source="spatial_query_canonical_cache",
            change_kind="created",
        )
    )

    table.obs[USER_CLASS_COLUMN] = pd.Categorical(
        [1 if index % 2 == 0 else pd.NA for index in range(table.n_obs)],
        categories=[1],
    )
    table.uns[USER_CLASS_COLORS_KEY] = ["#ff0000"]
    app_state.record_table_mutation(
        _table_event(
            backed_sdata_blobs,
            paths=frozenset(
                {
                    TableComponentPath("obs", (USER_CLASS_COLUMN,)),
                    TableComponentPath("uns", (USER_CLASS_COLORS_KEY,)),
                }
            ),
            source="spatial_query_annotation",
            change_kind="created",
        )
    )

    table.uns[CLASSIFIER_CONFIG_KEY] = {"status": "stale", "reason": "integration test"}
    app_state.record_table_mutation(
        _table_event(
            backed_sdata_blobs,
            paths=frozenset({TableComponentPath("uns", (CLASSIFIER_CONFIG_KEY,))}),
            source="object_classification_classifier",
            change_kind="created",
        )
    )

    assert spatial_query.persistence_controls.write_button.isEnabled()
    assert object_classification.persistence_controls.write_button.isEnabled()

    assert spatial_query.persistence_controls.write_table_state()

    assert not spatial_query.persistence_controls.write_button.isEnabled()
    assert not object_classification.persistence_controls.write_button.isEnabled()
    assert not app_state.is_table_dirty(backed_sdata_blobs, "table")

    reopened = read_zarr(backed_sdata_blobs.path)
    reopened_table = reopened.tables["table"]
    assert CANONICAL_OBSM_KEY in reopened_table.obsm
    assert USER_CLASS_COLUMN in reopened_table.obs
    assert reopened_table.uns[USER_CLASS_COLORS_KEY] == ["#ff0000"]
    assert reopened_table.uns[CLASSIFIER_CONFIG_KEY] == {
        "status": "stale",
        "reason": "integration test",
    }
    assert "untracked_local" not in reopened_table.obsm
    assert "untracked_local" not in reopened_table.uns
    np.testing.assert_array_equal(_dense_array(reopened_table.X), original_x)
    pd.testing.assert_frame_equal(reopened_table.var, original_var)
    np.testing.assert_array_equal(np.asarray(reopened_table.obsm["features_1"]), original_features)


def test_feature_acknowledgement_preserves_other_and_newer_dirty_components(
    qtbot,
    backed_sdata_blobs: SpatialData,
) -> None:
    """A persisted Feature Extraction acknowledgement cannot clean another producer or a newer token."""
    viewer = _Viewer()
    app_state = get_or_create_app_state(viewer)
    app_state.set_sdata(backed_sdata_blobs)
    spatial_query = SpatialQuery(viewer)
    object_classification = ObjectClassificationWidget(viewer)
    qtbot.addWidget(spatial_query)
    qtbot.addWidget(object_classification)
    spatial_query.apply_annotation_context(_annotation_context(backed_sdata_blobs))
    _select_spatial_query_labels(spatial_query)
    _select_object_classification_labels(object_classification)

    table = backed_sdata_blobs.tables["table"]
    annotation_event = _table_event(
        backed_sdata_blobs,
        paths=frozenset({TableComponentPath("obs", ("review_state",))}),
        source="spatial_query_annotation",
        change_kind="created",
    )
    table.obs["review_state"] = pd.Categorical(["kept"] * table.n_obs)
    app_state.record_table_mutation(annotation_event)

    feature_paths = frozenset(
        {
            TableComponentPath("obsm", ("features_1",)),
            TableComponentPath("uns", ("feature_matrices", "features_1")),
        }
    )
    feature_event = _table_event(
        backed_sdata_blobs,
        paths=feature_paths,
        source="feature_extraction",
    )
    table.uns.setdefault("feature_matrices", {})["features_1"] = {
        "feature_columns": ["a", "b", "c", "d"],
    }
    app_state.record_table_mutation(feature_event)
    captured = app_state.snapshot_table_dirty_state(backed_sdata_blobs, "table")
    write_table_components(
        backed_sdata_blobs,
        table_name="table",
        paths=feature_paths,
    )
    app_state.record_persisted_table_change(feature_event, captured)

    assert app_state.snapshot_table_dirty_state(backed_sdata_blobs, "table").paths == annotation_event.paths
    assert spatial_query.persistence_controls.write_button.isEnabled()
    assert object_classification.persistence_controls.write_button.isEnabled()

    app_state.record_table_mutation(feature_event)
    stale_snapshot = app_state.snapshot_table_dirty_state(backed_sdata_blobs, "table")
    table.obsm["features_1"] = np.full_like(np.asarray(table.obsm["features_1"]), 7.0)
    app_state.record_table_mutation(feature_event)
    write_table_components(
        backed_sdata_blobs,
        table_name="table",
        paths=feature_paths,
    )
    app_state.record_persisted_table_change(feature_event, stale_snapshot)

    remaining_paths = app_state.snapshot_table_dirty_state(backed_sdata_blobs, "table").paths
    assert annotation_event.paths <= remaining_paths
    assert feature_paths <= remaining_paths

    assert object_classification.persistence_controls.write_table_state()
    assert not spatial_query.persistence_controls.write_button.isEnabled()
    assert not object_classification.persistence_controls.write_button.isEnabled()


def test_shared_reload_is_guarded_then_adopted_once_by_all_bound_workflows(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    backed_sdata_blobs: SpatialData,
) -> None:
    """One reload request prepares all workflows, respects Feature Extraction, and publishes once."""
    viewer = _Viewer()
    app_state = get_or_create_app_state(viewer)
    app_state.set_sdata(backed_sdata_blobs)
    spatial_query = SpatialQuery(viewer)
    object_classification = ObjectClassificationWidget(viewer)
    feature_extraction = FeatureExtractionWidget(viewer)
    qtbot.addWidget(spatial_query)
    qtbot.addWidget(object_classification)
    qtbot.addWidget(feature_extraction)
    spatial_query.apply_annotation_context(_annotation_context(backed_sdata_blobs))
    _select_spatial_query_labels(spatial_query)
    _select_object_classification_labels(object_classification)
    _configure_feature_extraction(feature_extraction, feature_key="features_1")

    worker = _DeferredWorker()
    monkeypatch.setattr(
        feature_extraction._feature_extraction_controller,
        "_create_feature_extraction_worker",
        lambda _job: worker,
    )
    monkeypatch.setattr(feature_extraction, "_confirm_regional_update", lambda *_args: True)
    monkeypatch.setattr(feature_extraction, "_confirm_full_replacement", lambda *_args: True)
    feature_extraction.calculate_button.click()
    assert worker.started

    spatial_query.new_column_edit.setText("late_annotation")
    spatial_query.annotation_value_edit.setText("tumor")
    monkeypatch.setattr(
        spatial_query._controller,
        "start_spatial_query",
        lambda *args, **kwargs: True,
    )
    qtbot.mouseClick(spatial_query.run_button, Qt.MouseButton.LeftButton)
    assert spatial_query._active_run_intent is not None

    freeze_calls: list[str] = []
    real_freeze = object_classification._classifier_controller.freeze_for_reload

    def record_freeze() -> None:
        freeze_calls.append("freeze")
        real_freeze()

    monkeypatch.setattr(
        object_classification._classifier_controller,
        "freeze_for_reload",
        record_freeze,
    )

    table = backed_sdata_blobs.tables["table"]
    _install_canonical_cache(backed_sdata_blobs)
    table.obs["temporary_annotation"] = pd.Categorical(["temporary"] * table.n_obs)
    del table.obsm["features_1"]
    dirty_event = _table_event(
        backed_sdata_blobs,
        paths=frozenset(
            {
                *CANONICAL_CACHE_PATHS,
                TableComponentPath("obs", ("temporary_annotation",)),
                TableComponentPath("obsm", ("features_1",)),
            }
        ),
        source="integration_test",
    )
    app_state.record_table_mutation(dirty_event)
    monkeypatch.setattr(
        spatial_query.persistence_controls,
        "_prompt_dirty_reload_decision",
        lambda: persistence_controls_module._DirtyReloadDecision.RELOAD_DISCARD,
    )
    reload_events: list[TableStateChangedEvent] = []
    app_state.table_state_changed.connect(
        lambda event: reload_events.append(event) if event.change_kind == "reloaded" else None
    )

    spatial_query.persistence_controls.reload_button.click()

    assert reload_events == []
    assert spatial_query._active_run_intent is None
    assert freeze_calls == ["freeze"]
    assert CANONICAL_OBSM_KEY in table.obsm
    assert "temporary_annotation" in table.obs
    assert "features_1" not in table.obsm

    worker.finished.emit()
    spatial_query.persistence_controls.reload_button.click()

    assert len(reload_events) == 1
    assert freeze_calls == ["freeze", "freeze"]
    assert CANONICAL_OBSM_KEY not in table.obsm
    assert "temporary_annotation" not in table.obs
    assert "features_1" in table.obsm
    assert spatial_query.cache_report is not None
    assert spatial_query.cache_report.state is CanonicalCacheState.ABSENT
    assert object_classification.selected_table_name == "table"
    assert feature_extraction.selected_table_name == "table"
    assert feature_extraction.selected_feature_key == "features_1"
    assert not app_state.is_table_dirty(backed_sdata_blobs, "table")
    labels_layer = app_state.viewer_adapter.get_loaded_primary_labels_layer(
        backed_sdata_blobs,
        "blobs_labels",
        "global",
    )
    assert labels_layer is not None
    _assert_color(labels_layer, 1, MISSING_CATEGORICAL_COLOR)


def test_later_object_classification_styling_wins_on_shared_primary_labels_layer(
    qtbot,
    backed_sdata_blobs: SpatialData,
) -> None:
    """The latest explicit workflow styling action owns the shared primary Labels presentation."""
    table = backed_sdata_blobs.tables["table"]
    table.obs["spatial_annotation"] = pd.Categorical(
        ["region"] * table.n_obs,
        categories=["region"],
    )
    table.uns["spatial_annotation_colors"] = ["#0000ff"]
    table.obs[USER_CLASS_COLUMN] = pd.Categorical(
        [1] * table.n_obs,
        categories=[1],
    )
    table.uns[USER_CLASS_COLORS_KEY] = ["#ff0000"]

    viewer = _Viewer()
    app_state = get_or_create_app_state(viewer)
    app_state.set_sdata(backed_sdata_blobs)
    object_classification = ObjectClassificationWidget(viewer)
    spatial_query = SpatialQuery(viewer)
    qtbot.addWidget(object_classification)
    qtbot.addWidget(spatial_query)
    _select_object_classification_labels(object_classification)
    spatial_query.apply_annotation_context(_annotation_context(backed_sdata_blobs))
    _select_spatial_query_labels(spatial_query)
    spatial_query.column_mode_combo.setCurrentIndex(spatial_query.column_mode_combo.findData("existing"))
    spatial_query.existing_column_combo.setCurrentIndex(
        spatial_query.existing_column_combo.findData("spatial_annotation")
    )

    labels_layer = app_state.viewer_adapter.get_loaded_primary_labels_layer(
        backed_sdata_blobs,
        "blobs_labels",
        "global",
    )
    assert labels_layer is not None
    assert len([layer for layer in viewer.layers if isinstance(layer, Labels)]) == 1
    _assert_color(labels_layer, 1, "#0000ff")

    # This is the widget callback used after accepted classifier prediction
    # state changes; its selected Color-by mode deliberately restyles the same
    # primary layer and therefore becomes the visible owner.
    object_classification._on_classifier_prediction_state_changed()

    _assert_color(labels_layer, 1, "#ff0000")


def test_spatialdata_replacement_ignores_late_query_classifier_and_feature_callbacks(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    backed_sdata_blobs: SpatialData,
    sdata_blobs: SpatialData,
) -> None:
    """Old-session work may finish only against its captured store and cannot publish into the replacement."""
    replacement_path = tmp_path / "replacement.zarr"
    sdata_blobs.write(replacement_path)
    replacement = read_zarr(replacement_path)
    old_table = backed_sdata_blobs.tables["table"]
    register_feature_matrix_metadata(
        old_table,
        "features_1",
        feature_columns=["a", "b", "c", "d"],
    )
    instance_ids = old_table.obs["instance_id"].to_numpy(dtype=np.int64)
    old_table.obs[USER_CLASS_COLUMN] = pd.Categorical(
        [
            1 if int(instance_id) in {1, 2} else 2 if int(instance_id) in {24, 25} else pd.NA
            for instance_id in instance_ids
        ],
        categories=[1, 2],
    )

    viewer = _Viewer()
    app_state = get_or_create_app_state(viewer)
    app_state.set_sdata(backed_sdata_blobs)
    viewer_widget = ViewerWidget(viewer)
    spatial_query = SpatialQuery(viewer)
    object_classification = ObjectClassificationWidget(viewer)
    feature_extraction = FeatureExtractionWidget(viewer)
    qtbot.addWidget(viewer_widget)
    qtbot.addWidget(spatial_query)
    qtbot.addWidget(object_classification)
    qtbot.addWidget(feature_extraction)
    spatial_query.apply_annotation_context(_annotation_context(backed_sdata_blobs))
    _select_spatial_query_labels(spatial_query)
    _select_object_classification_labels(object_classification)
    old_query_result = _canonical_query_result(backed_sdata_blobs)
    spatial_query.new_column_edit.setText("late_annotation")
    spatial_query.annotation_value_edit.setText("late")
    monkeypatch.setattr(
        spatial_query._controller,
        "start_spatial_query",
        lambda *args, **kwargs: True,
    )
    qtbot.mouseClick(spatial_query.run_button, Qt.MouseButton.LeftButton)
    assert spatial_query._active_run_intent is not None

    classifier_worker = _DeferredWorker()
    classifier_jobs: list[object] = []

    def create_classifier_worker(job: object) -> _DeferredWorker:
        classifier_jobs.append(job)
        return classifier_worker

    monkeypatch.setattr(
        object_classification._classifier_controller,
        "_create_training_worker",
        create_classifier_worker,
    )
    object_classification.retrain_button.click()
    assert len(classifier_jobs) == 1
    assert classifier_worker.started

    _configure_feature_extraction(feature_extraction, feature_key="late_features")
    worker = _DeferredWorker()
    feature_jobs: list[object] = []

    def create_feature_worker(job: object) -> _DeferredWorker:
        feature_jobs.append(job)
        return worker

    monkeypatch.setattr(
        feature_extraction._feature_extraction_controller,
        "_create_feature_extraction_worker",
        create_feature_worker,
    )
    feature_extraction.calculate_button.click()
    assert len(feature_jobs) == 1
    job_id = feature_jobs[0].job_id

    app_state.set_sdata(replacement, discard_current=True)
    table_events: list[TableStateChangedEvent] = []
    app_state.table_state_changed.connect(table_events.append)

    old_table.obsm["late_features"] = np.ones((old_table.n_obs, 1), dtype=np.float64)
    write_table_components(
        backed_sdata_blobs,
        table_name="table",
        paths=frozenset({TableComponentPath("obsm", ("late_features",))}),
    )
    worker.returned.emit(
        FeatureExtractionResult(
            job_id=job_id,
            labels_names=("blobs_labels",),
            table_name="table",
            feature_key="late_features",
            regions=("blobs_labels",),
            change_kind="created",
        )
    )
    worker.finished.emit()
    spatial_query._on_query_ready(old_query_result)
    classifier_job = classifier_jobs[0]
    classifier_worker.returned.emit(
        classifier_module.ClassifierJobResult(
            job_id=classifier_job.job_id,
            feature_key=classifier_job.feature_key,
            labels_name=classifier_job.labels_name,
            table_name=classifier_job.table_name,
            pred_classes=np.ones(
                classifier_job.prediction_scope.table_row_positions.shape,
                dtype=np.int64,
            ),
            pred_confidences=np.full(
                classifier_job.prediction_scope.table_row_positions.shape,
                0.9,
                dtype=np.float64,
            ),
            trained_at="2026-07-28T12:00:00+00:00",
            model_params=dict(classifier_module.RANDOM_FOREST_PARAMS),
            summary=classifier_job.summary,
        )
    )
    classifier_worker.finished.emit()

    assert table_events == []
    assert app_state.sdata is replacement
    assert "late_features" not in replacement.tables["table"].obsm
    assert "late_annotation" not in replacement.tables["table"].obs
    assert classifier_module.PRED_CLASS_COLUMN not in replacement.tables["table"].obs
    assert classifier_module.PRED_CONFIDENCE_COLUMN not in replacement.tables["table"].obs
    assert "late_features" in read_zarr(backed_sdata_blobs.path).tables["table"].obsm
