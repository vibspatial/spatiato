from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest
from qtpy.QtCore import QObject, Signal
from spatialdata import SpatialData

import spatiato.core.object_classification.classifier as classifier_core
import spatiato.widgets.object_classification.controller as classifier_module
from spatiato.core.class_palette import default_class_colors
from spatiato.core.feature_matrix_metadata import HARPY_ADD_FEATURE_MATRIX_SOURCE_KIND, FeatureMatrixMetadataState
from spatiato.core.object_classification.annotation import USER_CLASS_COLORS_KEY, USER_CLASS_COLUMN
from spatiato.core.object_classification.classifier import (
    ObjectClassificationStateError,
    validate_object_classification_table_state,
)
from spatiato.core.object_classification.classifier_export import read_classifier_export_bundle
from spatiato.core.persistence import TableComponentPath
from spatiato.widgets.object_classification.controller import (
    CLASSIFIER_CONFIG_KEY,
    PRED_CLASS_COLORS_KEY,
    PRED_CLASS_COLUMN,
    PRED_CONFIDENCE_COLUMN,
    ClassifierController,
)

_FEATURE_MATRICES_KEY = "feature_matrices"


def test_update_table_prediction_state_rejects_rows_outside_clear_scope(
    sdata_blobs: SpatialData,
) -> None:
    table = sdata_blobs["table"]

    with pytest.raises(ValueError, match="subset of clear row positions"):
        classifier_core._update_table_prediction_state(
            table,
            clear_row_positions=np.array([0], dtype=np.int64),
            prediction_row_positions=np.array([1], dtype=np.int64),
            pred_classes=np.array([1], dtype=np.int64),
            pred_confidences=np.array([0.9], dtype=np.float64),
            pred_class_column=PRED_CLASS_COLUMN,
            pred_confidence_column=PRED_CONFIDENCE_COLUMN,
        )

    assert PRED_CLASS_COLUMN not in table.obs
    assert PRED_CONFIDENCE_COLUMN not in table.obs
    assert PRED_CLASS_COLORS_KEY not in table.uns


def test_object_classification_state_validation_is_read_only_for_valid_columns(
    sdata_blobs: SpatialData,
) -> None:
    table = sdata_blobs["table"]
    user_values = [1, pd.NA, 2, *([pd.NA] * (table.n_obs - 3))]
    pred_values = [2, pd.NA, 1, *([pd.NA] * (table.n_obs - 3))]
    confidence_values = [0.8, np.nan, 0.7, *([np.nan] * (table.n_obs - 3))]
    table.obs[USER_CLASS_COLUMN] = pd.Categorical(user_values, categories=[2, 1, 7])
    table.obs[PRED_CLASS_COLUMN] = pd.Categorical(pred_values, categories=[2, 1, 9])
    table.obs[PRED_CONFIDENCE_COLUMN] = pd.Series(
        confidence_values,
        index=table.obs.index,
        dtype="float64",
    )
    previous_obs = table.obs.copy(deep=True)
    previous_uns = table.uns.copy()

    validate_object_classification_table_state(table)

    pd.testing.assert_frame_equal(table.obs, previous_obs)
    assert table.uns == previous_uns


@pytest.mark.parametrize(
    ("values", "expected_message"),
    [
        (pd.Series([1, 2], dtype="int64"), "categorical dtype"),
        (pd.Series(pd.Categorical([0, 1], categories=[0, 1])), "invalid category"),
        (pd.Series(pd.Categorical(["1", "2"])), "invalid category"),
    ],
)
def test_object_classification_state_validation_rejects_invalid_user_class(
    sdata_blobs: SpatialData,
    values: pd.Series,
    expected_message: str,
) -> None:
    table = sdata_blobs["table"]
    table.obs[USER_CLASS_COLUMN] = pd.Series(
        np.resize(values.to_numpy(), table.n_obs),
        index=table.obs.index,
        dtype=values.dtype,
    )

    with pytest.raises(ObjectClassificationStateError, match=expected_message):
        validate_object_classification_table_state(table)


@pytest.mark.parametrize("invalid_state", ["incomplete", "integer_confidence", "mismatched", "out_of_range"])
def test_object_classification_state_validation_rejects_invalid_prediction_state(
    sdata_blobs: SpatialData,
    invalid_state: str,
) -> None:
    table = sdata_blobs["table"]
    pred_values = pd.Categorical([1, pd.NA, *([pd.NA] * (table.n_obs - 2))], categories=[1])
    table.obs[PRED_CLASS_COLUMN] = pred_values
    if invalid_state == "incomplete":
        expected_message = "must either both exist or both be absent"
    elif invalid_state == "integer_confidence":
        table.obs[PRED_CONFIDENCE_COLUMN] = pd.Series(1, index=table.obs.index, dtype="int64")
        expected_message = "floating-point dtype"
    elif invalid_state == "mismatched":
        table.obs[PRED_CONFIDENCE_COLUMN] = pd.Series(0.5, index=table.obs.index, dtype="float64")
        expected_message = "missing exactly when"
    else:
        table.obs[PRED_CONFIDENCE_COLUMN] = pd.Series(
            [1.5, np.nan, *([np.nan] * (table.n_obs - 2))],
            index=table.obs.index,
            dtype="float64",
        )
        expected_message = "between 0 and 1"

    with pytest.raises(ObjectClassificationStateError, match=expected_message):
        validate_object_classification_table_state(table)


class _DeferredWorker(QObject):
    returned = Signal(object)
    errored = Signal(object)
    finished = Signal()

    def __init__(self, result: classifier_module.ClassifierJobResult) -> None:
        super().__init__()
        self._result = result
        self.started = False
        self.quit_called = False

    def start(self) -> None:
        self.started = True

    def quit(self) -> None:
        self.quit_called = True

    def emit_returned(self) -> None:
        self.returned.emit(self._result)
        self.finished.emit()


def _set_deterministic_features(sdata: SpatialData, *, table_name: str = "table") -> None:
    table = sdata[table_name]
    instance_ids = table.obs["instance_id"].to_numpy(dtype=np.int64)
    first_feature = (instance_ids > 13).astype(np.float64)
    second_feature = instance_ids.astype(np.float64) / instance_ids.max()
    table.obsm["features_1"] = np.column_stack([first_feature, second_feature])


def _set_feature_metadata(
    sdata: SpatialData,
    *,
    table_name: str = "table",
    feature_key: str = "features_1",
    feature_columns: tuple[str, ...] = ("is_large", "instance_fraction"),
    features: tuple[str, ...] = ("synthetic_size", "synthetic_position"),
) -> None:
    table = sdata[table_name]
    table.uns.setdefault(_FEATURE_MATRICES_KEY, {})[feature_key] = {
        "feature_columns": list(feature_columns),
        "schema_version": 1,
        "backend": "numpy",
        "dtype": str(np.asarray(table.obsm[feature_key]).dtype),
        "source_label": "blobs_labels",
        "source_image": None,
        "coordinate_system": "global",
        "features": list(features),
        "source_kind": HARPY_ADD_FEATURE_MATRIX_SOURCE_KIND,
    }


def _set_multi_region_feature_metadata(sdata: SpatialData) -> None:
    _set_feature_metadata(
        sdata,
        table_name="table_multi",
        feature_columns=("feature_0", "feature_1", "feature_2", "feature_3"),
        features=("synthetic_feature_0", "synthetic_feature_1", "synthetic_feature_2", "synthetic_feature_3"),
    )


def _set_user_classes(sdata: SpatialData, class_by_instance: dict[int, int], *, table_name: str = "table") -> None:
    table = sdata[table_name]
    values = [class_by_instance.get(int(instance_id), pd.NA) for instance_id in table.obs["instance_id"]]
    categories = sorted(set(class_by_instance.values()))
    table.obs[USER_CLASS_COLUMN] = pd.Categorical(values, categories=categories)


def _set_user_classes_by_region(
    sdata: SpatialData,
    class_by_region_instance: dict[tuple[str, int], int],
    *,
    table_name: str = "table_multi",
) -> None:
    table = sdata[table_name]
    region_values = table.obs["region"].astype("string").to_numpy()
    instance_values = table.obs["instance_id"].to_numpy(dtype=np.int64)
    values = [
        class_by_region_instance.get((str(region), int(instance_id)), pd.NA)
        for region, instance_id in zip(region_values, instance_values, strict=True)
    ]
    categories = sorted(set(class_by_region_instance.values()))
    table.obs[USER_CLASS_COLUMN] = pd.Categorical(values, categories=categories)


def _set_invalid_feature_rows_for_region(
    sdata: SpatialData,
    *,
    region_name: str,
    table_name: str = "table_multi",
    feature_key: str = "features_1",
) -> None:
    table = sdata[table_name]
    feature_matrix = np.asarray(table.obsm[feature_key], dtype=np.float64).copy()
    region_mask = (table.obs["region"].astype("string") == region_name).to_numpy(dtype=bool, copy=False)
    feature_matrix[region_mask, :] = np.nan
    table.obsm[feature_key] = feature_matrix


def _resolved_scope(
    positions: np.ndarray | list[int] | tuple[int, ...],
    *,
    labels_name: str = "blobs_labels",
    mode: classifier_module.ClassifierScopeMode = "selected_segmentation_only",
    regions: tuple[str, ...] | None = None,
    n_rows_in_regions: int | None = None,
) -> classifier_module.ResolvedClassifierScope:
    table_row_positions = np.asarray(positions, dtype=np.int64)
    raw_table_row_positions = table_row_positions
    if n_rows_in_regions is not None and n_rows_in_regions != table_row_positions.size:
        raw_table_row_positions = np.arange(n_rows_in_regions, dtype=np.int64)
    return classifier_module.ResolvedClassifierScope(
        mode=mode,
        regions=(labels_name,) if regions is None else regions,
        raw_table_row_positions=raw_table_row_positions,
        table_row_positions=table_row_positions,
    )


def test_classifier_controller_trains_on_labeled_rows_and_predicts_active_objects(
    qtbot, sdata_blobs: SpatialData
) -> None:
    _set_deterministic_features(sdata_blobs)
    _set_feature_metadata(sdata_blobs)
    _set_user_classes(sdata_blobs, {1: 1, 2: 1, 24: 2, 25: 2})
    user_colors = ["#123456", "#abcdef"]
    sdata_blobs["table"].uns[USER_CLASS_COLORS_KEY] = user_colors
    table_state_changes: list[str] = []
    prediction_state_changes: list[str] = []

    controller = ClassifierController(
        debounce_interval_ms=0,
        on_table_state_changed=lambda _change: table_state_changes.append("changed"),
        on_prediction_state_changed=lambda: prediction_state_changes.append("changed"),
    )
    controller.bind(sdata_blobs, "blobs_labels", "table", "features_1")
    controller.schedule_retrain(immediate=True)

    table = sdata_blobs["table"]
    qtbot.waitUntil(
        lambda: PRED_CLASS_COLUMN in table.obs and table.obs[PRED_CLASS_COLUMN].notna().any(),
        timeout=5000,
    )

    pred_class = table.obs.set_index("instance_id")[PRED_CLASS_COLUMN]
    pred_confidence = table.obs.set_index("instance_id")[PRED_CONFIDENCE_COLUMN]

    assert isinstance(table.obs[PRED_CLASS_COLUMN].dtype, pd.CategoricalDtype)
    assert list(table.obs[PRED_CLASS_COLUMN].cat.categories) == [1, 2]
    assert table.uns[PRED_CLASS_COLORS_KEY] == user_colors
    assert pred_class.loc[1] == 1
    assert pred_class.loc[5] == 1
    assert pred_class.loc[24] == 2
    assert pred_class.loc[26] == 2
    assert pd.to_numeric(pred_class.astype("string"), errors="coerce").min() >= 1
    assert pred_confidence.between(0.0, 1.0).all()
    assert table.uns[CLASSIFIER_CONFIG_KEY]["eligible"] is True
    assert table.uns[CLASSIFIER_CONFIG_KEY]["trained"] is True
    assert table.uns[CLASSIFIER_CONFIG_KEY]["class_labels_seen"] == [1, 2]
    assert controller.status_kind == "success"
    assert table_state_changes == ["changed"]
    assert prediction_state_changes == ["changed"]


def test_classifier_controller_exports_trained_classifier_bundle(
    qtbot,
    tmp_path,
    sdata_blobs: SpatialData,
) -> None:
    _set_deterministic_features(sdata_blobs)
    _set_feature_metadata(sdata_blobs)
    _set_user_classes(sdata_blobs, {1: 1, 2: 1, 24: 2, 25: 2})

    controller = ClassifierController(debounce_interval_ms=0)
    controller.bind(sdata_blobs, "blobs_labels", "table", "features_1")
    controller.retrain_now()

    table = sdata_blobs["table"]
    qtbot.waitUntil(lambda: controller.can_export_classifier, timeout=5000)

    export_path = tmp_path / "trained.spatiato-classifier.joblib"
    bundle = controller.export_classifier(export_path)
    loaded = read_classifier_export_bundle(export_path)

    feature_matrix = np.asarray(table.obsm["features_1"], dtype=np.float64)
    assert export_path.exists()
    assert bundle.feature_columns == ("is_large", "instance_fraction")
    assert loaded.feature_columns == ("is_large", "instance_fraction")
    assert loaded.feature_names == ("synthetic_size", "synthetic_position")
    assert loaded.n_features == 2
    assert loaded.class_labels_seen == (1, 2)
    assert loaded.source_classifier_config == table.uns[CLASSIFIER_CONFIG_KEY]
    assert loaded.source_feature_metadata == table.uns[_FEATURE_MATRICES_KEY]["features_1"]
    assert np.array_equal(loaded.estimator.predict(feature_matrix), bundle.estimator.predict(feature_matrix))


def test_classifier_controller_refuses_export_when_model_is_dirty(
    qtbot,
    tmp_path,
    sdata_blobs: SpatialData,
) -> None:
    _set_deterministic_features(sdata_blobs)
    _set_feature_metadata(sdata_blobs)
    _set_user_classes(sdata_blobs, {1: 1, 2: 1, 24: 2, 25: 2})

    controller = ClassifierController(debounce_interval_ms=0)
    controller.bind(sdata_blobs, "blobs_labels", "table", "features_1")
    controller.retrain_now()
    qtbot.waitUntil(lambda: controller.can_export_classifier, timeout=5000)

    controller.mark_dirty(reason="the annotations changed")

    assert controller.can_export_classifier is False
    with pytest.raises(ValueError, match="stale"):
        controller.export_classifier(tmp_path / "dirty.spatiato-classifier.joblib")


@pytest.mark.parametrize("schedule_replacement", [False, True])
def test_classifier_controller_mark_dirty_invalidates_active_worker_and_ignores_late_result(
    sdata_blobs: SpatialData,
    schedule_replacement: bool,
) -> None:
    _set_deterministic_features(sdata_blobs)
    _set_feature_metadata(sdata_blobs)
    _set_user_classes(sdata_blobs, {1: 1, 2: 1, 24: 2, 25: 2})
    workers: list[_DeferredWorker] = []
    table_state_changes: list[classifier_module.ClassifierTableStateChange] = []

    def fake_create_training_worker(job):
        worker = _DeferredWorker(
            classifier_module.ClassifierJobResult(
                job_id=job.job_id,
                feature_key=job.feature_key,
                labels_name=job.labels_name,
                table_name=job.table_name,
                pred_classes=np.full(job.prediction_scope.table_row_positions.shape, 1, dtype=np.int64),
                pred_confidences=np.full(job.prediction_scope.table_row_positions.shape, 0.9, dtype=np.float64),
                trained_at="2026-07-27T09:00:00+00:00",
                model_params=dict(classifier_module.RANDOM_FOREST_PARAMS),
                summary=job.summary,
            )
        )
        workers.append(worker)
        return worker

    controller = ClassifierController(
        debounce_interval_ms=0,
        on_table_state_changed=table_state_changes.append,
    )
    controller.bind(sdata_blobs, "blobs_labels", "table", "features_1")
    controller._create_training_worker = fake_create_training_worker  # type: ignore[method-assign]

    assert controller.retrain_now() is True
    worker = workers[0]
    assert controller.is_training is True

    controller.mark_dirty(reason="the annotations changed")

    assert worker.quit_called is True
    assert controller.is_training is False
    assert controller.is_dirty is True

    replacement_worker = None
    if schedule_replacement:
        assert controller.schedule_retrain(immediate=True) is True
        replacement_worker = workers[1]
        assert replacement_worker.started is True

    worker.emit_returned()

    table = sdata_blobs.tables["table"]
    assert PRED_CLASS_COLUMN not in table.obs
    assert PRED_CONFIDENCE_COLUMN not in table.obs
    assert table_state_changes == []
    assert controller.is_dirty is True
    assert controller.is_training is schedule_replacement

    if replacement_worker is not None:
        replacement_worker.emit_returned()
        assert PRED_CLASS_COLUMN in table.obs
        assert PRED_CONFIDENCE_COLUMN in table.obs
        assert len(table_state_changes) == 1
        assert controller.is_dirty is False


def test_classifier_controller_refuses_export_while_training(tmp_path, sdata_blobs: SpatialData) -> None:
    _set_deterministic_features(sdata_blobs)
    _set_feature_metadata(sdata_blobs)
    _set_user_classes(sdata_blobs, {1: 1, 2: 1, 24: 2, 25: 2})

    workers: dict[int, _DeferredWorker] = {}

    def fake_create_training_worker(job):
        result = classifier_module.ClassifierJobResult(
            job_id=job.job_id,
            feature_key=job.feature_key,
            labels_name=job.labels_name,
            table_name=job.table_name,
            pred_classes=np.full(job.prediction_scope.table_row_positions.shape, 1, dtype=np.int64),
            pred_confidences=np.full(job.prediction_scope.table_row_positions.shape, 0.9, dtype=np.float64),
            trained_at="2026-04-13T09:00:00+00:00",
            model_params=dict(classifier_module.RANDOM_FOREST_PARAMS),
            summary=job.summary,
        )
        worker = _DeferredWorker(result)
        workers[job.job_id] = worker
        return worker

    controller = ClassifierController(debounce_interval_ms=0)
    controller.bind(sdata_blobs, "blobs_labels", "table", "features_1")
    controller._create_training_worker = fake_create_training_worker  # type: ignore[method-assign]

    assert controller.retrain_now() is True

    assert workers[1].started is True
    assert controller.can_export_classifier is False
    with pytest.raises(ValueError, match="currently running"):
        controller.export_classifier(tmp_path / "running.spatiato-classifier.joblib")


def test_classifier_controller_clears_export_snapshot_after_reload(
    qtbot,
    tmp_path,
    sdata_blobs: SpatialData,
) -> None:
    _set_deterministic_features(sdata_blobs)
    _set_feature_metadata(sdata_blobs)
    _set_user_classes(sdata_blobs, {1: 1, 2: 1, 24: 2, 25: 2})

    controller = ClassifierController(debounce_interval_ms=0)
    controller.bind(sdata_blobs, "blobs_labels", "table", "features_1")
    controller.retrain_now()
    qtbot.waitUntil(lambda: controller.can_export_classifier, timeout=5000)

    controller.reset_after_reload()

    assert controller.can_export_classifier is False
    with pytest.raises(ValueError, match="reloaded from disk"):
        controller.export_classifier(tmp_path / "reloaded.spatiato-classifier.joblib")


def test_classifier_controller_refuses_export_after_feature_metadata_drift(
    qtbot,
    tmp_path,
    sdata_blobs: SpatialData,
) -> None:
    _set_deterministic_features(sdata_blobs)
    _set_feature_metadata(sdata_blobs)
    _set_user_classes(sdata_blobs, {1: 1, 2: 1, 24: 2, 25: 2})

    controller = ClassifierController(debounce_interval_ms=0)
    controller.bind(sdata_blobs, "blobs_labels", "table", "features_1")
    controller.retrain_now()
    qtbot.waitUntil(lambda: controller.can_export_classifier, timeout=5000)

    sdata_blobs["table"].uns[_FEATURE_MATRICES_KEY]["features_1"]["feature_columns"] = [
        "instance_fraction",
        "is_large",
    ]

    assert controller.can_export_classifier is False
    with pytest.raises(ValueError, match="metadata no longer matches"):
        controller.export_classifier(tmp_path / "drift.spatiato-classifier.joblib")


def test_classifier_controller_blocks_training_without_feature_metadata(sdata_blobs: SpatialData) -> None:
    _set_deterministic_features(sdata_blobs)
    _set_user_classes(sdata_blobs, {1: 1, 2: 1, 24: 2, 25: 2})

    controller = ClassifierController(debounce_interval_ms=0)
    controller.bind(sdata_blobs, "blobs_labels", "table", "features_1")
    table = sdata_blobs["table"]
    table.obs[PRED_CLASS_COLUMN] = pd.Categorical(np.full(table.n_obs, 9, dtype=np.int64), categories=[9])
    table.obs[PRED_CONFIDENCE_COLUMN] = pd.Series(np.full(table.n_obs, 0.55), index=table.obs.index, dtype="float64")
    summary = controller.describe_current_preparation()

    assert summary is not None
    assert summary.eligible is False
    assert summary.training_blocker_kind == "metadata"
    assert "Register feature metadata" in summary.reason
    assert controller.can_retrain is False
    assert controller.schedule_retrain(immediate=True) is False
    assert CLASSIFIER_CONFIG_KEY not in table.uns
    assert table.obs[PRED_CLASS_COLUMN].astype("string").eq("9").all()
    assert table.obs[PRED_CONFIDENCE_COLUMN].eq(0.55).all()
    assert controller.can_export_classifier is False
    assert "Register feature metadata" in controller.status_message


@pytest.mark.parametrize("status", ["missing_matrix", "invalid_matrix", "registered_mismatched"])
def test_feature_matrix_metadata_training_blocker_reason_requires_state_error(status: str) -> None:
    state = object.__new__(FeatureMatrixMetadataState)
    object.__setattr__(state, "feature_key", "features_1")
    object.__setattr__(state, "status", status)
    object.__setattr__(state, "n_features", None)
    object.__setattr__(state, "metadata_n_features", None)
    object.__setattr__(state, "source_kind", None)
    object.__setattr__(state, "error", None)

    with pytest.raises(ValueError, match="must include an error message"):
        classifier_module._feature_matrix_metadata_training_blocker_reason(state)


def test_multi_region_classifier_fixture_duplicates_instance_ids_only_across_regions(
    sdata_blobs_multi_region: SpatialData,
) -> None:
    table = sdata_blobs_multi_region["table_multi"]
    first_region_rows = table.obs.loc[table.obs["region"].astype("string") == "blobs_labels"]
    second_region_rows = table.obs.loc[table.obs["region"].astype("string") == "blobs_labels_2"]

    assert first_region_rows["instance_id"].is_unique
    assert second_region_rows["instance_id"].is_unique
    assert set(first_region_rows["instance_id"].tolist()) == set(second_region_rows["instance_id"].tolist())


def test_classifier_controller_selected_region_only_training_scope_ignores_labels_in_other_regions(
    sdata_blobs_multi_region: SpatialData,
) -> None:
    _set_multi_region_feature_metadata(sdata_blobs_multi_region)
    _set_user_classes_by_region(
        sdata_blobs_multi_region,
        {
            ("blobs_labels_2", 1): 1,
            ("blobs_labels_2", 2): 1,
            ("blobs_labels_2", 24): 2,
            ("blobs_labels_2", 25): 2,
        },
    )
    table = sdata_blobs_multi_region["table_multi"]

    controller = ClassifierController(debounce_interval_ms=0)
    controller.bind(
        sdata_blobs_multi_region,
        "blobs_labels",
        "table_multi",
        "features_1",
        training_scope="selected_segmentation_only",
    )

    job = controller._prepare_classifier_job(1)

    assert job is not None
    assert job.training_scope.regions == ("blobs_labels",)
    assert job.training_scope.n_rows_in_regions == table.n_obs // 2
    assert job.training_scope.n_eligible_rows == table.n_obs // 2
    assert job.summary.resolved_training_row_count == table.n_obs // 2
    assert job.summary.resolved_prediction_row_count == table.n_obs // 2
    assert job.summary.training_region_count == 1
    assert job.summary.labeled_count == 0
    assert job.summary.eligible is False
    assert "Need at least 2 labeled samples" in job.summary.reason


def test_classifier_controller_defaults_training_scope_to_all_and_can_use_labels_from_other_regions(
    sdata_blobs_multi_region: SpatialData,
) -> None:
    _set_multi_region_feature_metadata(sdata_blobs_multi_region)
    _set_user_classes_by_region(
        sdata_blobs_multi_region,
        {
            ("blobs_labels_2", 1): 1,
            ("blobs_labels_2", 2): 1,
            ("blobs_labels_2", 24): 2,
            ("blobs_labels_2", 25): 2,
        },
    )
    table = sdata_blobs_multi_region["table_multi"]

    controller = ClassifierController(debounce_interval_ms=0)
    controller.bind(
        sdata_blobs_multi_region,
        "blobs_labels",
        "table_multi",
        "features_1",
    )

    job = controller._prepare_classifier_job(1)

    assert job is not None
    assert job.training_scope.regions == ("blobs_labels", "blobs_labels_2")
    assert job.training_scope.n_rows_in_regions == table.n_obs
    assert job.training_scope.n_eligible_rows == table.n_obs
    assert job.prediction_scope.regions == ("blobs_labels",)
    assert job.prediction_scope.n_rows_in_regions == table.n_obs // 2
    assert job.prediction_scope.n_eligible_rows == table.n_obs // 2
    assert job.summary.resolved_training_row_count == table.n_obs
    assert job.summary.resolved_prediction_row_count == table.n_obs // 2
    assert job.summary.training_region_count == 2
    assert job.summary.labeled_count == 4
    assert job.summary.eligible is True


def test_classifier_controller_default_table_wide_training_excludes_invalid_rows_from_other_regions(
    sdata_blobs_multi_region: SpatialData,
) -> None:
    _set_multi_region_feature_metadata(sdata_blobs_multi_region)
    _set_user_classes_by_region(
        sdata_blobs_multi_region,
        {
            ("blobs_labels", 1): 1,
            ("blobs_labels", 2): 1,
            ("blobs_labels", 24): 2,
            ("blobs_labels", 25): 2,
        },
    )
    _set_invalid_feature_rows_for_region(sdata_blobs_multi_region, region_name="blobs_labels_2")
    table = sdata_blobs_multi_region["table_multi"]

    controller = ClassifierController(debounce_interval_ms=0)
    controller.bind(
        sdata_blobs_multi_region,
        "blobs_labels",
        "table_multi",
        "features_1",
    )

    job = controller._prepare_classifier_job(1)

    assert job is not None
    assert job.training_scope.regions == ("blobs_labels", "blobs_labels_2")
    assert job.training_scope.n_rows_in_regions == table.n_obs
    assert job.training_scope.n_eligible_rows == table.n_obs // 2
    assert job.training_scope.n_excluded_feature_invalid_rows == table.n_obs // 2
    assert job.prediction_scope.regions == ("blobs_labels",)
    assert job.prediction_scope.n_eligible_rows == table.n_obs // 2
    assert job.summary.resolved_training_row_count == table.n_obs // 2
    assert job.summary.resolved_prediction_row_count == table.n_obs // 2
    assert job.summary.training_region_count == 2
    assert job.summary.eligible is True


def test_classifier_controller_prediction_scope_all_clears_invalid_rows_in_scope(
    qtbot, sdata_blobs_multi_region: SpatialData
) -> None:
    _set_multi_region_feature_metadata(sdata_blobs_multi_region)
    _set_user_classes_by_region(
        sdata_blobs_multi_region,
        {
            ("blobs_labels", 1): 1,
            ("blobs_labels", 2): 1,
            ("blobs_labels", 24): 2,
            ("blobs_labels", 25): 2,
        },
    )
    _set_invalid_feature_rows_for_region(sdata_blobs_multi_region, region_name="blobs_labels_2")
    table = sdata_blobs_multi_region["table_multi"]
    table.obs[PRED_CLASS_COLUMN] = pd.Categorical(np.full(table.n_obs, 9, dtype=np.int64), categories=[9])
    table.obs[PRED_CONFIDENCE_COLUMN] = pd.Series(np.full(table.n_obs, 0.55), index=table.obs.index, dtype="float64")
    table_state_changes: list[classifier_module.ClassifierTableStateChange] = []

    controller = ClassifierController(
        debounce_interval_ms=0,
        on_table_state_changed=table_state_changes.append,
    )
    controller.bind(
        sdata_blobs_multi_region,
        "blobs_labels",
        "table_multi",
        "features_1",
        prediction_scope="all",
    )
    controller.schedule_retrain(immediate=True)

    region_values = table.obs["region"].astype("string")
    valid_rows = region_values == "blobs_labels"
    invalid_rows = region_values == "blobs_labels_2"

    qtbot.waitUntil(
        lambda: (
            CLASSIFIER_CONFIG_KEY in table.uns
            and table.uns[CLASSIFIER_CONFIG_KEY].get("trained") is True
            and controller.status_kind == "success"
        ),
        timeout=5000,
    )

    assert table.obs.loc[valid_rows, PRED_CLASS_COLUMN].notna().all()
    assert table.obs.loc[valid_rows, PRED_CONFIDENCE_COLUMN].between(0.0, 1.0).all()
    assert table.obs.loc[invalid_rows, PRED_CLASS_COLUMN].isna().all()
    assert table.obs.loc[invalid_rows, PRED_CONFIDENCE_COLUMN].isna().all()
    assert table.uns[CLASSIFIER_CONFIG_KEY]["prediction_scope"] == "all"
    assert table.uns[CLASSIFIER_CONFIG_KEY]["prediction_regions"] == ["blobs_labels", "blobs_labels_2"]
    assert table.uns[CLASSIFIER_CONFIG_KEY]["n_predicted_rows"] == int(valid_rows.sum())
    assert table.uns[CLASSIFIER_CONFIG_KEY]["training_scope"] == "all"
    assert table.uns[CLASSIFIER_CONFIG_KEY]["n_training_rows"] == int(valid_rows.sum())
    assert len(table_state_changes) == 1
    assert table_state_changes[0].regions == ("blobs_labels", "blobs_labels_2")
    assert table_state_changes[0].source == "object_classification_inference"


def test_classifier_controller_default_prediction_scope_leaves_hidden_regions_unchanged(
    monkeypatch, sdata_blobs_multi_region: SpatialData
) -> None:
    _set_multi_region_feature_metadata(sdata_blobs_multi_region)
    _set_user_classes_by_region(
        sdata_blobs_multi_region,
        {
            ("blobs_labels", 1): 1,
            ("blobs_labels", 2): 1,
            ("blobs_labels", 24): 2,
            ("blobs_labels", 25): 2,
        },
    )
    table = sdata_blobs_multi_region["table_multi"]
    table.obs[PRED_CLASS_COLUMN] = pd.Categorical(np.full(table.n_obs, 9, dtype=np.int64), categories=[3, 9])
    table.obs[PRED_CONFIDENCE_COLUMN] = pd.Series(np.full(table.n_obs, 0.55), index=table.obs.index, dtype="float64")
    workers: dict[int, _DeferredWorker] = {}

    def fake_create_training_worker(self, job):
        del self
        result = classifier_module.ClassifierJobResult(
            job_id=job.job_id,
            feature_key=job.feature_key,
            labels_name=job.labels_name,
            table_name=job.table_name,
            pred_classes=np.full(job.prediction_scope.table_row_positions.shape, 3, dtype=np.int64),
            pred_confidences=np.full(job.prediction_scope.table_row_positions.shape, 0.88, dtype=np.float64),
            trained_at="2026-04-13T09:00:00+00:00",
            model_params=dict(classifier_module.RANDOM_FOREST_PARAMS),
            summary=job.summary,
        )
        worker = _DeferredWorker(result)
        workers[job.job_id] = worker
        return worker

    monkeypatch.setattr(ClassifierController, "_create_training_worker", fake_create_training_worker)

    controller = ClassifierController(debounce_interval_ms=0)
    controller.bind(sdata_blobs_multi_region, "blobs_labels", "table_multi", "features_1")
    assert controller.retrain_now() is True

    workers[1].emit_returned()

    region_values = table.obs["region"].astype("string")
    selected_rows = region_values == "blobs_labels"
    hidden_rows = region_values == "blobs_labels_2"

    assert table.obs.loc[selected_rows, PRED_CLASS_COLUMN].astype("string").eq("3").all()
    assert table.obs.loc[selected_rows, PRED_CONFIDENCE_COLUMN].eq(0.88).all()
    assert table.obs.loc[hidden_rows, PRED_CLASS_COLUMN].astype("string").eq("9").all()
    assert table.obs.loc[hidden_rows, PRED_CONFIDENCE_COLUMN].eq(0.55).all()
    assert list(table.obs[PRED_CLASS_COLUMN].cat.categories) == [3, 9]
    assert table.uns[PRED_CLASS_COLORS_KEY] == default_class_colors([3, 9])
    assert table.uns[CLASSIFIER_CONFIG_KEY]["prediction_scope"] == "selected_segmentation_only"
    assert table.uns[CLASSIFIER_CONFIG_KEY]["prediction_regions"] == ["blobs_labels"]
    assert table.uns[CLASSIFIER_CONFIG_KEY]["n_predicted_rows"] == int(selected_rows.sum())


def test_classifier_controller_describes_current_preparation_without_building_worker_arrays(
    monkeypatch, sdata_blobs_multi_region: SpatialData
) -> None:
    _set_multi_region_feature_metadata(sdata_blobs_multi_region)
    _set_user_classes_by_region(
        sdata_blobs_multi_region,
        {
            ("blobs_labels", 1): 1,
            ("blobs_labels", 2): 1,
            ("blobs_labels_2", 24): 2,
            ("blobs_labels_2", 25): 2,
        },
    )

    original_normalize_feature_matrix = classifier_module.normalize_feature_matrix

    def fail_worker_feature_snapshot(feature_matrix, n_obs, *, copy):
        if copy:
            raise AssertionError("preparation summary should not construct worker feature arrays")
        return original_normalize_feature_matrix(feature_matrix, n_obs, copy=copy)

    monkeypatch.setattr(classifier_module, "normalize_feature_matrix", fail_worker_feature_snapshot)

    def fail_training_worker(*args, **kwargs):
        del args, kwargs
        raise AssertionError("preparation summary should not construct worker feature arrays")

    monkeypatch.setattr(classifier_module.ClassifierController, "_create_training_worker", fail_training_worker)

    controller = ClassifierController(debounce_interval_ms=0)
    controller.bind(
        sdata_blobs_multi_region,
        "blobs_labels",
        "table_multi",
        "features_1",
        prediction_scope="all",
    )

    summary = controller.describe_current_preparation()

    assert summary is not None
    assert summary.training_scope.regions == ("blobs_labels", "blobs_labels_2")
    assert summary.prediction_scope.regions == ("blobs_labels", "blobs_labels_2")
    assert summary.resolved_prediction_row_count == sdata_blobs_multi_region["table_multi"].n_obs
    assert summary.labeled_count == 4
    assert summary.eligible is True


def test_classifier_controller_can_retrain_requires_trainable_preparation(sdata_blobs: SpatialData) -> None:
    _set_deterministic_features(sdata_blobs)
    _set_feature_metadata(sdata_blobs)
    controller = ClassifierController(debounce_interval_ms=0)
    controller.bind(sdata_blobs, "blobs_labels", "table", "features_1")

    assert controller.can_retrain is False

    _set_user_classes(sdata_blobs, {1: 1, 2: 1})

    assert controller.can_retrain is False

    _set_user_classes(sdata_blobs, {1: 1, 24: 2})

    assert controller.can_retrain is True


def test_classifier_controller_does_not_create_absent_predictions_when_only_one_class_is_labeled(
    qtbot, sdata_blobs: SpatialData
) -> None:
    _set_deterministic_features(sdata_blobs)
    _set_feature_metadata(sdata_blobs)
    _set_user_classes(sdata_blobs, {1: 1, 2: 1})
    table_state_changes: list[classifier_module.ClassifierTableStateChange] = []
    prediction_state_changes: list[str] = []

    controller = ClassifierController(
        debounce_interval_ms=0,
        on_table_state_changed=table_state_changes.append,
        on_prediction_state_changed=lambda: prediction_state_changes.append("changed"),
    )
    controller.bind(sdata_blobs, "blobs_labels", "table", "features_1")
    controller.schedule_retrain(immediate=True)

    table = sdata_blobs["table"]

    assert PRED_CLASS_COLUMN not in table.obs
    assert PRED_CONFIDENCE_COLUMN not in table.obs
    assert PRED_CLASS_COLORS_KEY not in table.uns
    assert table.uns[CLASSIFIER_CONFIG_KEY]["eligible"] is False
    assert "two labeled classes" in table.uns[CLASSIFIER_CONFIG_KEY]["reason"]
    assert controller.status_kind == "warning"
    assert len(table_state_changes) == 1
    assert table_state_changes[0].regions == ("blobs_labels",)
    assert table_state_changes[0].source == "object_classification_inference"
    assert table_state_changes[0].paths == frozenset(
        {
            TableComponentPath("uns", (CLASSIFIER_CONFIG_KEY,)),
        }
    )
    assert prediction_state_changes == []


def test_classifier_controller_clears_existing_predictions_without_removing_vocabulary(
    sdata_blobs: SpatialData,
) -> None:
    _set_deterministic_features(sdata_blobs)
    _set_feature_metadata(sdata_blobs)
    _set_user_classes(sdata_blobs, {1: 1, 2: 1})
    table = sdata_blobs["table"]
    table.obs[PRED_CLASS_COLUMN] = pd.Categorical(
        np.full(table.n_obs, 7, dtype=np.int64),
        categories=[7, 1],
    )
    table.obs[PRED_CONFIDENCE_COLUMN] = pd.Series(
        np.full(table.n_obs, 0.5, dtype=np.float64),
        index=table.obs.index,
    )
    table.uns[PRED_CLASS_COLORS_KEY] = ["#000000", "#000000"]
    table_state_changes: list[classifier_module.ClassifierTableStateChange] = []

    controller = ClassifierController(
        debounce_interval_ms=0,
        on_table_state_changed=table_state_changes.append,
    )
    controller.bind(sdata_blobs, "blobs_labels", "table", "features_1")
    controller.schedule_retrain(immediate=True)

    assert table.obs[PRED_CLASS_COLUMN].isna().all()
    assert table.obs[PRED_CONFIDENCE_COLUMN].isna().all()
    assert list(table.obs[PRED_CLASS_COLUMN].cat.categories) == [7, 1]
    assert table.uns[PRED_CLASS_COLORS_KEY] == [
        default_class_colors([7])[0],
        default_class_colors([1])[0],
    ]
    assert table_state_changes[0].paths == frozenset(
        {
            TableComponentPath("obs", (PRED_CLASS_COLUMN,)),
            TableComponentPath("obs", (PRED_CONFIDENCE_COLUMN,)),
            TableComponentPath("uns", (PRED_CLASS_COLORS_KEY,)),
            TableComponentPath("uns", (CLASSIFIER_CONFIG_KEY,)),
        }
    )


def test_classifier_controller_validates_feature_matrix_shape(qtbot, monkeypatch, sdata_blobs: SpatialData) -> None:
    _set_deterministic_features(sdata_blobs)
    _set_feature_metadata(sdata_blobs)
    _set_user_classes(sdata_blobs, {1: 1, 24: 2})
    table = sdata_blobs["table"]

    def raise_shape_error(feature_matrix, n_obs, *, copy=True):
        del feature_matrix, n_obs, copy
        raise ValueError("Feature matrix has 25 rows but the table has 26 observations.")

    monkeypatch.setattr(classifier_module, "normalize_feature_matrix", raise_shape_error)

    controller = ClassifierController(
        debounce_interval_ms=0,
    )
    controller.bind(sdata_blobs, "blobs_labels", "table", "features_1")
    controller.schedule_retrain(immediate=True)

    assert PRED_CLASS_COLUMN not in table.obs
    assert PRED_CONFIDENCE_COLUMN not in table.obs
    assert PRED_CLASS_COLORS_KEY not in table.uns
    assert "rows but the table has" in table.uns[CLASSIFIER_CONFIG_KEY]["reason"]
    assert controller.status_kind == "warning"


def test_classifier_controller_drops_stale_results(qtbot, monkeypatch, sdata_blobs: SpatialData) -> None:
    _set_deterministic_features(sdata_blobs)
    _set_feature_metadata(sdata_blobs)
    _set_user_classes(sdata_blobs, {1: 1, 2: 1, 24: 2, 25: 2})

    call_log: list[int] = []

    def fake_fit(job):
        call_log.append(job.job_id)
        if job.job_id == 1:
            time.sleep(0.2)
            pred_class = np.full(job.prediction_scope.table_row_positions.shape, 1, dtype=np.int64)
        else:
            pred_class = np.full(job.prediction_scope.table_row_positions.shape, 2, dtype=np.int64)

        return classifier_module.ClassifierJobResult(
            job_id=job.job_id,
            feature_key=job.feature_key,
            labels_name=job.labels_name,
            table_name=job.table_name,
            pred_classes=pred_class,
            pred_confidences=np.full(job.prediction_scope.table_row_positions.shape, 0.9, dtype=np.float64),
            trained_at="2026-04-08T12:00:00+00:00",
            model_params=dict(classifier_module.RANDOM_FOREST_PARAMS),
            summary=job.summary,
        )

    monkeypatch.setattr(classifier_module, "_fit_classifier_job", fake_fit)

    controller = ClassifierController(
        debounce_interval_ms=0,
    )
    controller.bind(sdata_blobs, "blobs_labels", "table", "features_1")
    controller.schedule_retrain(immediate=True)
    controller.schedule_retrain(immediate=True)

    table = sdata_blobs["table"]
    qtbot.waitUntil(
        lambda: (
            PRED_CLASS_COLUMN in table.obs
            and table.obs[PRED_CLASS_COLUMN].nunique() == 1
            and table.obs[PRED_CLASS_COLUMN].iloc[0] == 2
        ),
        timeout=5000,
    )

    # Both jobs ran; their worker threads may start in either order.
    assert sorted(call_log) == [1, 2]
    assert table.obs[PRED_CLASS_COLUMN].eq(2).all()
    assert controller.status_kind == "success"


def test_classifier_controller_bind_is_passive_until_marked_dirty(qtbot, monkeypatch, sdata_blobs: SpatialData) -> None:
    _set_deterministic_features(sdata_blobs)
    _set_feature_metadata(sdata_blobs)
    _set_user_classes(sdata_blobs, {1: 1, 2: 1, 24: 2, 25: 2})

    call_log: list[int] = []
    table_state_changes: list[classifier_module.ClassifierTableStateChange] = []
    prediction_state_changes: list[str] = []

    def fake_fit(job):
        call_log.append(job.job_id)
        return classifier_module.ClassifierJobResult(
            job_id=job.job_id,
            feature_key=job.feature_key,
            labels_name=job.labels_name,
            table_name=job.table_name,
            pred_classes=np.full(job.prediction_scope.table_row_positions.shape, 1, dtype=np.int64),
            pred_confidences=np.full(job.prediction_scope.table_row_positions.shape, 0.9, dtype=np.float64),
            trained_at="2026-04-08T12:00:00+00:00",
            model_params=dict(classifier_module.RANDOM_FOREST_PARAMS),
            summary=job.summary,
        )

    monkeypatch.setattr(classifier_module, "_fit_classifier_job", fake_fit)

    controller = ClassifierController(
        debounce_interval_ms=0,
        on_table_state_changed=table_state_changes.append,
        on_prediction_state_changed=lambda: prediction_state_changes.append("changed"),
    )
    context_changed = controller.bind(sdata_blobs, "blobs_labels", "table", "features_1")

    assert context_changed is True
    table = sdata_blobs["table"]
    assert PRED_CLASS_COLUMN not in table.obs
    assert PRED_CONFIDENCE_COLUMN not in table.obs
    assert PRED_CLASS_COLORS_KEY not in table.uns
    assert controller.is_dirty is False
    assert controller.status_message == "Classifier: model is up to date."
    assert call_log == []
    assert table_state_changes == []
    assert prediction_state_changes == []

    controller.mark_dirty(reason="the feature matrix changed")

    assert controller.is_dirty is True
    assert "Classifier: model is stale because the feature matrix changed" == controller.status_message

    controller.retrain_now()
    qtbot.waitUntil(lambda: len(call_log) == 1, timeout=5000)
    assert controller.is_dirty is False
    assert len(table_state_changes) == 1
    assert table_state_changes[0].regions == ("blobs_labels",)
    assert table_state_changes[0].source == "object_classification_inference"
    assert prediction_state_changes == ["changed"]


def test_classifier_controller_notifies_table_state_change_when_training_errors(
    qtbot, monkeypatch, sdata_blobs: SpatialData
) -> None:
    _set_deterministic_features(sdata_blobs)
    _set_feature_metadata(sdata_blobs)
    _set_user_classes(sdata_blobs, {1: 1, 2: 1, 24: 2, 25: 2})
    table_state_changes: list[classifier_module.ClassifierTableStateChange] = []
    prediction_state_changes: list[str] = []

    def raise_fit_error(job):
        del job
        raise RuntimeError("boom")

    monkeypatch.setattr(classifier_module, "_fit_classifier_job", raise_fit_error)

    controller = ClassifierController(
        debounce_interval_ms=0,
        on_table_state_changed=table_state_changes.append,
        on_prediction_state_changed=lambda: prediction_state_changes.append("changed"),
    )
    controller.bind(sdata_blobs, "blobs_labels", "table", "features_1")
    controller.schedule_retrain(immediate=True)

    qtbot.waitUntil(lambda: controller.status_kind == "error", timeout=5000)

    assert len(table_state_changes) == 1
    assert table_state_changes[0].regions == ()
    assert table_state_changes[0].source == "object_classification_metadata"
    assert prediction_state_changes == []
    assert sdata_blobs["table"].uns[CLASSIFIER_CONFIG_KEY]["trained"] is False
    assert "boom" in sdata_blobs["table"].uns[CLASSIFIER_CONFIG_KEY]["reason"]
    assert sdata_blobs["table"].uns[CLASSIFIER_CONFIG_KEY]["training_scope"] == "all"
    assert sdata_blobs["table"].uns[CLASSIFIER_CONFIG_KEY]["prediction_scope"] == "selected_segmentation_only"


def test_classifier_controller_freeze_for_reload_cancels_pending_debounce(
    qtbot, monkeypatch, sdata_blobs: SpatialData
) -> None:
    _set_deterministic_features(sdata_blobs)
    _set_feature_metadata(sdata_blobs)
    _set_user_classes(sdata_blobs, {1: 1, 2: 1, 24: 2, 25: 2})
    created_job_ids: list[int] = []

    def fake_create_training_worker(self, job):
        del self
        created_job_ids.append(job.job_id)
        raise AssertionError("reload freeze should cancel the debounce before a worker is created")

    monkeypatch.setattr(ClassifierController, "_create_training_worker", fake_create_training_worker)

    controller = ClassifierController(
        debounce_interval_ms=50,
    )
    controller.bind(sdata_blobs, "blobs_labels", "table", "features_1")

    controller.schedule_retrain()
    controller.freeze_for_reload()
    qtbot.wait(150)

    assert created_job_ids == []
    assert controller.is_training is False
    assert controller.is_dirty is True
    assert controller.status_message == "Classifier: model is stale. Click Train Classifier to refresh predictions."


def test_classifier_controller_shutdown_cancels_pending_debounce(qtbot, monkeypatch, sdata_blobs: SpatialData) -> None:
    _set_deterministic_features(sdata_blobs)
    _set_feature_metadata(sdata_blobs)
    _set_user_classes(sdata_blobs, {1: 1, 2: 1, 24: 2, 25: 2})
    created_job_ids: list[int] = []
    state_changes: list[str] = []

    def fake_create_training_worker(self, job):
        del self
        created_job_ids.append(job.job_id)
        raise AssertionError("shutdown should cancel the debounce before a worker is created")

    monkeypatch.setattr(ClassifierController, "_create_training_worker", fake_create_training_worker)

    controller = ClassifierController(
        debounce_interval_ms=50,
        on_state_changed=lambda: state_changes.append("changed"),
    )
    controller.bind(sdata_blobs, "blobs_labels", "table", "features_1")

    assert controller.schedule_retrain() is True
    assert controller._debounce_timer.isActive() is True

    state_changes.clear()
    controller.shutdown()
    controller.shutdown()
    qtbot.wait(150)

    assert created_job_ids == []
    assert state_changes == []
    assert controller.is_training is False
    assert controller.can_retrain is False
    assert controller.schedule_retrain() is False


def test_classifier_controller_shutdown_quits_worker_and_ignores_late_signals(
    sdata_blobs: SpatialData,
) -> None:
    _set_deterministic_features(sdata_blobs)
    _set_feature_metadata(sdata_blobs)
    _set_user_classes(sdata_blobs, {1: 1, 2: 1, 24: 2, 25: 2})
    state_changes: list[str] = []
    table_state_changes: list[str] = []
    prediction_state_changes: list[str] = []
    workers: list[_DeferredWorker] = []

    def fake_create_training_worker(job):
        result = classifier_module.ClassifierJobResult(
            job_id=job.job_id,
            feature_key=job.feature_key,
            labels_name=job.labels_name,
            table_name=job.table_name,
            pred_classes=np.full(job.prediction_scope.table_row_positions.shape, 1, dtype=np.int64),
            pred_confidences=np.full(job.prediction_scope.table_row_positions.shape, 0.9, dtype=np.float64),
            trained_at="2026-04-13T09:00:00+00:00",
            model_params=dict(classifier_module.RANDOM_FOREST_PARAMS),
            summary=job.summary,
        )
        worker = _DeferredWorker(result)
        workers.append(worker)
        return worker

    controller = ClassifierController(
        debounce_interval_ms=0,
        on_state_changed=lambda: state_changes.append("changed"),
        on_table_state_changed=lambda _change: table_state_changes.append("changed"),
        on_prediction_state_changed=lambda: prediction_state_changes.append("changed"),
    )
    controller.bind(sdata_blobs, "blobs_labels", "table", "features_1")
    controller._create_training_worker = fake_create_training_worker  # type: ignore[method-assign]

    assert controller.retrain_now() is True
    assert len(workers) == 1
    worker = workers[0]
    assert worker.started is True
    assert controller.is_training is True

    state_changes.clear()
    table_state_changes.clear()
    prediction_state_changes.clear()

    controller.shutdown()
    controller.shutdown()

    assert worker.quit_called is True
    assert controller.is_training is False
    assert controller._on_state_changed is None
    assert controller._on_table_state_changed is None
    assert controller._on_prediction_state_changed is None

    worker.emit_returned()
    worker.errored.emit(RuntimeError("late error"))
    worker.finished.emit()

    assert state_changes == []
    assert table_state_changes == []
    assert prediction_state_changes == []


def test_classifier_controller_invalidates_pending_work_for_selected_feature_matrix_overwrite(
    sdata_blobs: SpatialData,
) -> None:
    _set_deterministic_features(sdata_blobs)
    _set_feature_metadata(sdata_blobs)
    _set_user_classes(sdata_blobs, {1: 1, 2: 1, 24: 2, 25: 2})
    result = classifier_module.ClassifierJobResult(
        job_id=1,
        feature_key="features_1",
        labels_name="blobs_labels",
        table_name="table",
        pred_classes=np.array([1, 2], dtype=np.int64),
        pred_confidences=np.array([0.9, 0.8], dtype=np.float64),
        trained_at="2026-04-13T09:00:00+00:00",
        model_params=dict(classifier_module.RANDOM_FOREST_PARAMS),
        summary=classifier_module.ClassifierPreparationSummary(
            training_scope=_resolved_scope([0, 1]),
            prediction_scope=_resolved_scope([0, 1]),
            reason="Ready to train.",
            labeled_count=2,
            class_labels=(1, 2),
            n_features=2,
        ),
    )
    worker = _DeferredWorker(result)

    controller = ClassifierController(debounce_interval_ms=0)
    controller.bind(sdata_blobs, "blobs_labels", "table", "features_1")
    controller._create_training_worker = lambda job: worker  # type: ignore[method-assign]

    assert controller.schedule_retrain(immediate=True) is True
    assert controller.is_training is True

    invalidated = controller.invalidate_for_feature_matrix_overwrite("features_1")

    assert invalidated is True
    assert worker.quit_called is True
    assert controller.is_training is False
    assert controller.is_dirty is True
    assert "overwritten" in controller.status_message


def test_classifier_controller_reset_after_reload_ignores_late_worker_results(
    monkeypatch, sdata_blobs: SpatialData
) -> None:
    _set_deterministic_features(sdata_blobs)
    _set_feature_metadata(sdata_blobs)
    _set_user_classes(sdata_blobs, {1: 1, 2: 1, 24: 2, 25: 2})
    workers: dict[int, _DeferredWorker] = {}
    prediction_state_changes: list[str] = []

    def fake_create_training_worker(self, job):
        del self
        result = classifier_module.ClassifierJobResult(
            job_id=job.job_id,
            feature_key=job.feature_key,
            labels_name=job.labels_name,
            table_name=job.table_name,
            pred_classes=np.full(job.prediction_scope.table_row_positions.shape, 1, dtype=np.int64),
            pred_confidences=np.full(job.prediction_scope.table_row_positions.shape, 0.91, dtype=np.float64),
            trained_at="2026-04-13T09:00:00+00:00",
            model_params=dict(classifier_module.RANDOM_FOREST_PARAMS),
            summary=job.summary,
        )
        worker = _DeferredWorker(result)
        workers[job.job_id] = worker
        return worker

    monkeypatch.setattr(ClassifierController, "_create_training_worker", fake_create_training_worker)

    controller = ClassifierController(
        debounce_interval_ms=0,
        on_prediction_state_changed=lambda: prediction_state_changes.append("changed"),
    )
    controller.bind(sdata_blobs, "blobs_labels", "table", "features_1")
    controller.retrain_now()

    worker = workers[1]
    assert worker.started is True
    assert controller.is_training is True

    controller.freeze_for_reload()

    table = sdata_blobs["table"]
    disk_predictions = np.full(table.n_obs, 2, dtype=np.int64)
    disk_confidences = np.full(table.n_obs, 0.77, dtype=np.float64)
    table.obs[PRED_CLASS_COLUMN] = pd.Categorical(disk_predictions, categories=[2])
    table.obs[PRED_CONFIDENCE_COLUMN] = pd.Series(
        disk_confidences,
        index=table.obs.index,
        dtype="float64",
    )
    table.uns[CLASSIFIER_CONFIG_KEY] = {
        "model_type": "RandomForestClassifier",
        "feature_key": "features_1",
        "table_name": "table",
        "roi_mode": "none",
        "trained": True,
        "eligible": True,
        "reason": "Ready to train.",
        "training_timestamp": "2026-04-13T09:00:00+00:00",
        "n_labeled_objects": 4,
        "n_features": 2,
        "class_labels_seen": [1, 2],
        "rf_params": dict(classifier_module.RANDOM_FOREST_PARAMS),
        "training_scope": "all",
        "training_regions": ["blobs_labels"],
        "n_training_rows": int(table.n_obs),
        "prediction_scope": "selected_segmentation_only",
        "prediction_regions": ["blobs_labels"],
        "n_predicted_rows": int(table.n_obs),
    }

    controller.reset_after_reload()

    assert worker.quit_called is True
    assert controller.is_training is False
    assert controller.is_dirty is False
    assert (
        controller.status_message
        == f"Classifier: model is up to date. Loaded predictions for {table.n_obs} objects from disk."
    )
    assert prediction_state_changes == []


def test_classifier_controller_reset_after_reload_marks_stale_when_prediction_scope_differs(
    sdata_blobs: SpatialData,
) -> None:
    table = sdata_blobs["table"]
    table.uns[CLASSIFIER_CONFIG_KEY] = {
        "model_type": "RandomForestClassifier",
        "feature_key": "features_1",
        "table_name": "table",
        "roi_mode": "none",
        "trained": True,
        "eligible": True,
        "reason": "Ready to train.",
        "training_timestamp": "2026-04-13T09:00:00+00:00",
        "n_labeled_objects": 4,
        "n_features": 2,
        "class_labels_seen": [1, 2],
        "rf_params": dict(classifier_module.RANDOM_FOREST_PARAMS),
        "training_scope": "all",
        "training_regions": ["blobs_labels"],
        "n_training_rows": int(table.n_obs),
        "prediction_scope": "all",
        "prediction_regions": ["blobs_labels"],
        "n_predicted_rows": int(table.n_obs),
    }

    controller = ClassifierController(debounce_interval_ms=0)
    controller.bind(sdata_blobs, "blobs_labels", "table", "features_1")
    controller.reset_after_reload()

    assert controller.is_dirty is True
    assert 'prediction scope "all"' in controller.status_message


def test_classifier_controller_reset_after_reload_accepts_multi_region_prediction_scope(
    sdata_blobs_multi_region: SpatialData,
) -> None:
    table = sdata_blobs_multi_region["table_multi"]
    table.uns[CLASSIFIER_CONFIG_KEY] = {
        "model_type": "RandomForestClassifier",
        "feature_key": "features_1",
        "table_name": "table_multi",
        "roi_mode": "none",
        "trained": True,
        "eligible": True,
        "reason": "Ready to train.",
        "training_timestamp": "2026-04-13T09:00:00+00:00",
        "n_labeled_objects": 4,
        "n_features": 4,
        "class_labels_seen": [1, 2],
        "rf_params": dict(classifier_module.RANDOM_FOREST_PARAMS),
        "training_scope": "all",
        "training_regions": ["blobs_labels", "blobs_labels_2"],
        "n_training_rows": int(table.n_obs),
        "prediction_scope": "all",
        "prediction_regions": ["blobs_labels", "blobs_labels_2"],
        "n_predicted_rows": int(table.n_obs),
    }

    controller = ClassifierController(debounce_interval_ms=0)
    controller.bind(
        sdata_blobs_multi_region,
        "blobs_labels",
        "table_multi",
        "features_1",
        prediction_scope="all",
    )
    controller.reset_after_reload()

    assert controller.is_dirty is False
    assert (
        controller.status_message
        == f"Classifier: model is up to date. Loaded predictions for {table.n_obs} objects from disk."
    )
