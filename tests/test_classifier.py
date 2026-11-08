from __future__ import annotations

import time

import dask.array as da
import numpy as np
import pandas as pd
import pytest
from qtpy.QtCore import QObject, Signal
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
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


def test_predict_classes_and_confidences_matches_predict_and_predict_proba() -> None:
    rng = np.random.default_rng(0)
    features = rng.normal(size=(60, 3))
    labels = np.where(features[:, 0] > 0, 4, 7)
    labels[:5] = 9
    # Few trees make ties between classes likely; argmax resolves them as `predict` does.
    estimator = RandomForestClassifier(n_estimators=4, random_state=0).fit(features, labels)
    rows = rng.normal(size=(40, 3))

    pred_classes, pred_confidences = classifier_core._predict_classes_and_confidences(estimator, rows)

    np.testing.assert_array_equal(pred_classes, estimator.predict(rows))
    np.testing.assert_array_equal(pred_confidences, estimator.predict_proba(rows).max(axis=1))
    assert pred_classes.dtype == np.int64


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

    def __init__(self, result: classifier_module.ClassifierWorkerResult) -> None:
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
) -> classifier_module.ResolvedClassifierScope:
    return classifier_module.ResolvedClassifierScope(
        mode=mode,
        regions=(labels_name,) if regions is None else regions,
        table_row_positions=np.asarray(positions, dtype=np.int64),
    )


def _fake_training_result(
    job: classifier_module.ClassifierJob,
    *,
    pred_class: int = 1,
    confidence: float = 0.9,
    training_timestamp: str = "2026-04-13T09:00:00+00:00",
) -> classifier_module.ClassifierTrainingResult:
    """Return a result that predicts `pred_class` for every row of the job's prediction scope."""
    positions = job.prediction_scope.table_row_positions
    return classifier_module.ClassifierTrainingResult(
        training_timestamp=training_timestamp,
        predicted_row_positions=positions,
        pred_classes=np.full(positions.shape, pred_class, dtype=np.int64),
        pred_confidences=np.full(positions.shape, confidence, dtype=np.float64),
        n_training_labels=job.summary.labeled_count,
        class_labels_seen=job.summary.class_labels,
    )


@pytest.mark.parametrize(
    ("blocker", "kind"),
    [("Choose a feature matrix.", None), (None, "data")],
)
def test_preparation_summary_requires_a_blocker_reason_with_its_kind(blocker: str | None, kind: str | None) -> None:
    with pytest.raises(ValueError, match="needs both a reason and a kind"):
        classifier_module.ClassifierPreparationSummary(
            training_scope=_resolved_scope([0, 1]),
            prediction_scope=_resolved_scope([0, 1]),
            labeled_count=0,
            class_labels=(),
            n_features=2,
            training_blocker_reason=blocker,
            training_blocker_kind=kind,
        )


_ENTRY_FIELDS = {
    "source",
    "pred_class_column",
    "pred_confidence_column",
    "feature_key",
    "prediction_regions",
    "prediction_timestamp",
    "prediction_scope",
    "classifier_path",
    "reason",
    "model_provenance",
}
_MODEL_PROVENANCE_FIELDS = {
    "feature_key",
    "model_type",
    "rf_params",
    "training_scope",
    "training_regions",
    "n_features",
    "training_timestamp",
    "n_training_labels",
    "class_labels_seen",
}


def _widget_entry(table) -> dict[str, object]:
    """Return the `uns["classifier_config"]` entry of `pred_class`, or `{}` before one is written."""
    return table.uns.get(CLASSIFIER_CONFIG_KEY, {}).get(PRED_CLASS_COLUMN, {})


def _trained_config(
    *,
    prediction_scope: str,
    prediction_regions: list[str],
    training_regions: list[str],
    n_features: int,
) -> dict[str, object]:
    """Return `uns["classifier_config"]` as the controller writes it after training."""
    return {
        PRED_CLASS_COLUMN: {
            "source": "widget",
            "pred_class_column": PRED_CLASS_COLUMN,
            "pred_confidence_column": PRED_CONFIDENCE_COLUMN,
            "feature_key": "features_1",
            "prediction_regions": prediction_regions,
            "prediction_timestamp": "2026-04-13T09:00:00+00:00",
            "prediction_scope": prediction_scope,
            "classifier_path": None,
            "reason": None,
            "model_provenance": {
                "feature_key": "features_1",
                "model_type": "RandomForestClassifier",
                "rf_params": dict(classifier_module.RANDOM_FOREST_PARAMS),
                "training_scope": "all",
                "training_regions": training_regions,
                "n_features": n_features,
                "training_timestamp": "2026-04-13T09:00:00+00:00",
                "n_training_labels": 4,
                "class_labels_seen": [1, 2],
            },
        }
    }


def _headless_entry(*, pred_class_column: str, prediction_regions: list[str]) -> dict[str, object]:
    """Return a `uns["classifier_config"]` entry as headless `apply_classifier(...)` writes it."""
    entry = _trained_config(
        prediction_scope="all",
        prediction_regions=prediction_regions,
        training_regions=["blobs_labels"],
        n_features=2,
    )[PRED_CLASS_COLUMN]
    return {
        **entry,
        "source": "apply_classifier",
        "pred_class_column": pred_class_column,
        "pred_confidence_column": f"{pred_class_column}_confidence",
        "prediction_scope": None,
        "classifier_path": "classifier.spatiato-classifier.joblib",
    }


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
    # One entry, for the column the widget wrote.
    assert set(table.uns[CLASSIFIER_CONFIG_KEY]) == {PRED_CLASS_COLUMN}
    entry = _widget_entry(table)
    assert set(entry) == _ENTRY_FIELDS
    assert set(entry["model_provenance"]) == _MODEL_PROVENANCE_FIELDS
    assert entry["source"] == "widget"
    assert entry["pred_class_column"] == PRED_CLASS_COLUMN
    assert entry["pred_confidence_column"] == PRED_CONFIDENCE_COLUMN
    assert entry["feature_key"] == "features_1"
    assert entry["prediction_timestamp"] == entry["model_provenance"]["training_timestamp"]
    assert entry["classifier_path"] is None
    assert entry["reason"] is None
    assert entry["model_provenance"]["feature_key"] == "features_1"
    assert entry["model_provenance"]["class_labels_seen"] == [1, 2]
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
    assert loaded.model_provenance == _widget_entry(table)["model_provenance"]
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
        worker = _DeferredWorker(_fake_training_result(job, training_timestamp="2026-07-27T09:00:00+00:00"))
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
        worker = _DeferredWorker(_fake_training_result(job))
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
    assert "Register feature metadata" in summary.training_blocker_reason
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
    assert job.summary.training_row_count == table.n_obs // 2
    assert job.summary.prediction_row_count == table.n_obs // 2
    assert job.summary.training_region_count == 1
    assert job.summary.labeled_count == 0
    assert job.summary.eligible is False
    assert "Need at least 2 labeled samples" in job.summary.training_blocker_reason


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
    assert job.prediction_scope.regions == ("blobs_labels",)
    assert job.prediction_scope.n_rows_in_regions == table.n_obs // 2
    assert job.summary.training_row_count == table.n_obs
    assert job.summary.prediction_row_count == table.n_obs // 2
    assert job.summary.training_region_count == 2
    assert job.summary.labeled_count == 4
    assert job.summary.eligible is True


def test_classifier_controller_table_wide_training_leaves_rows_without_features_to_the_worker(
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

    # The summary reads no feature values: it counts every row of the scope,
    # also those of "blobs_labels_2", whose features are NaN. The worker skips them.
    assert job is not None
    assert job.training_scope.regions == ("blobs_labels", "blobs_labels_2")
    assert job.training_scope.n_rows_in_regions == table.n_obs
    assert job.prediction_scope.regions == ("blobs_labels",)
    assert job.summary.training_row_count == table.n_obs
    assert job.summary.prediction_row_count == table.n_obs // 2
    assert job.summary.training_region_count == 2
    assert job.summary.eligible is True
    # The job carries the matrix itself and the labeled rows, not copies.
    assert job.worker_input is not None
    assert job.worker_input.feature_matrix is table.obsm["features_1"]
    assert job.worker_input.train_row_positions.size == 4
    assert set(job.worker_input.train_row_regions.tolist()) == {"blobs_labels"}


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
        lambda: _widget_entry(table).get("model_provenance") is not None and controller.status_kind == "success",
        timeout=5000,
    )

    assert table.obs.loc[valid_rows, PRED_CLASS_COLUMN].notna().all()
    assert table.obs.loc[valid_rows, PRED_CONFIDENCE_COLUMN].between(0.0, 1.0).all()
    assert table.obs.loc[invalid_rows, PRED_CLASS_COLUMN].isna().all()
    assert table.obs.loc[invalid_rows, PRED_CONFIDENCE_COLUMN].isna().all()
    entry = _widget_entry(table)
    assert entry["prediction_scope"] == "all"
    assert entry["prediction_regions"] == ["blobs_labels", "blobs_labels_2"]
    assert entry["model_provenance"]["training_scope"] == "all"
    # The worker skipped the rows of "blobs_labels_2", and the status names it.
    assert f"Updated predictions for {int(valid_rows.sum())} objects" in controller.status_message
    assert f"Skipped {int(invalid_rows.sum())} row(s) without finite features" in controller.status_message
    assert '"blobs_labels_2"' in controller.status_message
    assert entry["model_provenance"]["n_training_labels"] == 4
    assert len(table_state_changes) == 1
    assert table_state_changes[0].regions == ("blobs_labels", "blobs_labels_2")
    assert table_state_changes[0].source == "object_classification_inference"


def _set_nan_features_for_instances(sdata: SpatialData, instance_ids: set[int]) -> np.ndarray:
    """Make one feature of the given instances NaN, and return the mask of their rows."""
    table = sdata["table"]
    values = np.asarray(table.obsm["features_1"], dtype=np.float64).copy()
    rows = np.isin(table.obs["instance_id"].to_numpy(dtype=np.int64), list(instance_ids))
    values[rows, 1] = np.nan
    table.obsm["features_1"] = values
    return rows


@pytest.mark.parametrize("representation", ["dense", "csr", "csc"])
def test_classifier_worker_skips_rows_without_finite_features(sdata_blobs: SpatialData, representation: str) -> None:
    _set_deterministic_features(sdata_blobs)
    _set_feature_metadata(sdata_blobs)
    table = sdata_blobs["table"]
    instance_ids = sorted(int(instance_id) for instance_id in table.obs["instance_id"])
    # One labeled and one unlabeled row lose their features.
    labeled_skipped, unlabeled_skipped = instance_ids[2], instance_ids[10]
    _set_user_classes(sdata_blobs, {instance_ids[0]: 1, instance_ids[1]: 1, labeled_skipped: 1, 24: 2, 25: 2})
    skipped_rows = _set_nan_features_for_instances(sdata_blobs, {labeled_skipped, unlabeled_skipped})
    assert int(skipped_rows.sum()) == 2
    if representation == "csr":
        table.obsm["features_1"] = sparse.csr_matrix(table.obsm["features_1"])
    elif representation == "csc":
        table.obsm["features_1"] = sparse.csc_matrix(table.obsm["features_1"])
    controller = ClassifierController(debounce_interval_ms=0)
    controller.bind(sdata_blobs, "blobs_labels", "table", "features_1")
    job = controller._prepare_classifier_job(1)
    assert job is not None
    assert job.summary.labeled_count == 5

    assert job.worker_input is not None
    result = classifier_module._fit_classifier_job(job.worker_input)

    assert isinstance(result, classifier_module.ClassifierTrainingResult)
    np.testing.assert_array_equal(result.predicted_row_positions, np.flatnonzero(~skipped_rows))
    assert result.pred_classes.shape == (table.n_obs - 2,)
    assert result.n_skipped_prediction_rows == 2
    assert result.n_skipped_labeled_rows == 1
    # The labeled count and classes of the rows used for training.
    assert result.n_training_labels == 4
    assert result.class_labels_seen == (1, 2)
    assert result.prediction_regions_without_features == ()


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("train_labels", "must align with `train_row_positions`"),
        ("train_row_regions", "must align with `train_row_positions`"),
        ("prediction_row_regions", "must align with `prediction_row_positions`"),
    ],
)
def test_classifier_worker_input_requires_aligned_rows(field: str, message: str) -> None:
    aligned = {
        "feature_key": "features_1",
        "feature_matrix": np.zeros((4, 2)),
        "train_row_positions": np.array([0, 1], dtype=np.int64),
        "train_labels": np.array([1, 2], dtype=np.int64),
        "train_row_regions": np.array(["cells", "cells"], dtype=object),
        "prediction_row_positions": np.array([0, 1, 2], dtype=np.int64),
        "prediction_row_regions": np.array(["cells", "cells", "cells"], dtype=object),
    }
    classifier_module.ClassifierWorkerInput(**aligned)

    # One element short.
    with pytest.raises(ValueError, match=message):
        classifier_module.ClassifierWorkerInput(**{**aligned, field: aligned[field][:-1]})


def test_classifier_worker_input_is_derived_from_the_summary_scopes(sdata_blobs_multi_region: SpatialData) -> None:
    _set_multi_region_feature_metadata(sdata_blobs_multi_region)
    _set_user_classes_by_region(
        sdata_blobs_multi_region,
        {
            ("blobs_labels", 1): 1,
            ("blobs_labels_2", 24): 2,
        },
    )
    table = sdata_blobs_multi_region["table_multi"]
    controller = ClassifierController(debounce_interval_ms=0)
    controller.bind(sdata_blobs_multi_region, "blobs_labels", "table_multi", "features_1")
    summary = controller.describe_current_preparation()
    assert summary is not None

    worker_input = classifier_module.ClassifierWorkerInput.from_summary(
        summary,
        table,
        feature_key="features_1",
        region_key="region",
    )

    # The prediction rows are the prediction scope's own array.
    assert worker_input.prediction_row_positions is summary.prediction_scope.table_row_positions
    assert set(worker_input.prediction_row_regions.tolist()) == {"blobs_labels"}
    # The training rows are the labeled rows of the training scope, with their labels and regions.
    labeled = np.flatnonzero(table.obs[USER_CLASS_COLUMN].notna().to_numpy())
    np.testing.assert_array_equal(worker_input.train_row_positions, labeled)
    assert sorted(worker_input.train_labels.tolist()) == [1, 2]
    assert sorted(worker_input.train_row_regions.tolist()) == ["blobs_labels", "blobs_labels_2"]
    assert worker_input.feature_matrix is table.obsm["features_1"]


@pytest.mark.parametrize("field", ["pred_classes", "pred_confidences"])
def test_classifier_training_result_requires_aligned_predictions(field: str) -> None:
    aligned = {
        "training_timestamp": "2026-11-06T09:00:00+00:00",
        "predicted_row_positions": np.array([0, 1], dtype=np.int64),
        "pred_classes": np.array([1, 2], dtype=np.int64),
        "pred_confidences": np.array([0.9, 0.8], dtype=np.float64),
        "n_training_labels": 2,
        "class_labels_seen": (1, 2),
    }
    classifier_module.ClassifierTrainingResult(**aligned)

    # One element short.
    with pytest.raises(ValueError, match="must align with `predicted_row_positions`"):
        classifier_module.ClassifierTrainingResult(**{**aligned, field: aligned[field][:-1]})


def _set_existing_predictions(table) -> None:
    """Give `table` prediction columns, as an earlier run would have written them."""
    table.obs[PRED_CLASS_COLUMN] = pd.Categorical(np.full(table.n_obs, 1, dtype=np.int64), categories=[1, 2])
    table.obs[PRED_CONFIDENCE_COLUMN] = pd.Series(np.full(table.n_obs, 0.5), index=table.obs.index, dtype="float64")


def test_classifier_worker_checks_labeled_rows_again_after_skipping(qtbot, sdata_blobs: SpatialData) -> None:
    _set_deterministic_features(sdata_blobs)
    _set_feature_metadata(sdata_blobs)
    _set_user_classes(sdata_blobs, {1: 1, 2: 1, 24: 2, 25: 2})
    # Every labeled row of class 2.
    _set_nan_features_for_instances(sdata_blobs, {24, 25})
    table = sdata_blobs["table"]
    _set_existing_predictions(table)
    controller = ClassifierController(debounce_interval_ms=0)
    controller.bind(sdata_blobs, "blobs_labels", "table", "features_1")

    # The summary counts every labeled row, so Train is allowed.
    assert controller.can_retrain is True
    controller.retrain_now()
    qtbot.waitUntil(lambda: bool(_widget_entry(table).get("reason")), timeout=5000)

    entry = _widget_entry(table)
    assert "Need at least two labeled classes with finite features" in entry["reason"]
    # Nothing was trained, so there is no model to describe.
    assert entry["model_provenance"] is None
    assert set(entry) == _ENTRY_FIELDS
    assert table.obs[PRED_CLASS_COLUMN].isna().all()
    assert controller.status_kind == "warning"
    assert controller.status_message == f"Classifier: {entry['reason']}"


def test_classifier_worker_reports_a_prediction_scope_without_features(
    qtbot, sdata_blobs_multi_region: SpatialData
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
    # Features were not calculated for the labels element to predict.
    _set_invalid_feature_rows_for_region(sdata_blobs_multi_region, region_name="blobs_labels")
    table = sdata_blobs_multi_region["table_multi"]
    controller = ClassifierController(debounce_interval_ms=0)
    controller.bind(sdata_blobs_multi_region, "blobs_labels", "table_multi", "features_1")

    # No longer disabled in advance: only the worker reads the feature values.
    assert controller.can_retrain is True
    controller.retrain_now()
    qtbot.waitUntil(lambda: "has no usable rows in the prediction scope" in controller.status_message, timeout=5000)

    assert 'Labels element(s) "blobs_labels" have no rows with finite features' in controller.status_message
    assert controller.status_kind == "warning"
    # No prediction column was ever written, so there is no entry to describe one.
    assert CLASSIFIER_CONFIG_KEY not in table.uns


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
        worker = _DeferredWorker(_fake_training_result(job, pred_class=3, confidence=0.88))
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
    assert _widget_entry(table)["prediction_scope"] == "selected_segmentation_only"
    assert _widget_entry(table)["prediction_regions"] == ["blobs_labels"]


def test_classifier_controller_training_replaces_the_pred_class_entry_and_keeps_other_entries(
    monkeypatch, sdata_blobs_multi_region: SpatialData
) -> None:
    """Training replaces headless's `classifier_config["pred_class"]` entry with the widget's own.

    The entry of a custom column stays, and so do headless's predictions
    outside the widget's prediction scope.
    """
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
    # As headless leaves the table: `pred_class` predicted for both labels
    # elements, and entries for `pred_class` and for a custom column.
    table.obs[PRED_CLASS_COLUMN] = pd.Categorical(np.full(table.n_obs, 9, dtype=np.int64), categories=[3, 9])
    table.obs[PRED_CONFIDENCE_COLUMN] = pd.Series(np.full(table.n_obs, 0.55), index=table.obs.index, dtype="float64")
    both_regions = ["blobs_labels", "blobs_labels_2"]
    custom_entry = _headless_entry(pred_class_column="my_pred", prediction_regions=both_regions)
    table.uns[CLASSIFIER_CONFIG_KEY] = {
        PRED_CLASS_COLUMN: _headless_entry(pred_class_column=PRED_CLASS_COLUMN, prediction_regions=both_regions),
        "my_pred": custom_entry,
    }
    workers: dict[int, _DeferredWorker] = {}

    def fake_create_training_worker(self, job):
        del self
        worker = _DeferredWorker(_fake_training_result(job, pred_class=3, confidence=0.88))
        workers[job.job_id] = worker
        return worker

    monkeypatch.setattr(ClassifierController, "_create_training_worker", fake_create_training_worker)

    controller = ClassifierController(debounce_interval_ms=0)
    controller.bind(sdata_blobs_multi_region, "blobs_labels", "table_multi", "features_1")
    assert controller.retrain_now() is True
    workers[1].emit_returned()

    # The widget's entry replaces headless's for `pred_class`; the custom column's entry stays.
    assert _widget_entry(table)["source"] == "widget"
    assert _widget_entry(table)["prediction_regions"] == ["blobs_labels"]
    assert table.uns[CLASSIFIER_CONFIG_KEY]["my_pred"] == custom_entry
    # Headless's predictions outside the widget's prediction scope stay.
    hidden_rows = table.obs["region"].astype("string") == "blobs_labels_2"
    assert table.obs.loc[hidden_rows, PRED_CLASS_COLUMN].astype("string").eq("9").all()


def _fail_when_read(block: np.ndarray) -> np.ndarray:
    raise AssertionError("feature values must not be read on the main thread")


def test_classifier_controller_describes_current_preparation_without_reading_feature_values(
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
    # A lazy matrix that fails as soon as any value is read: the summary needs
    # only its shape and dtype.
    table = sdata_blobs_multi_region["table_multi"]
    values = np.asarray(table.obsm["features_1"])
    table.obsm["features_1"] = da.from_array(values).map_blocks(_fail_when_read, dtype=values.dtype)

    def fail_training_worker(*args, **kwargs):
        del args, kwargs
        raise AssertionError("preparation summary should not start a training worker")

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
    assert summary.prediction_row_count == sdata_blobs_multi_region["table_multi"].n_obs
    assert summary.labeled_count == 4
    assert summary.n_features == values.shape[1]
    assert summary.eligible is True
    # Preparing the job reads no values either: it carries the matrix itself.
    job = controller._prepare_classifier_job(1)
    assert job is not None
    assert job.worker_input is not None
    assert job.worker_input.feature_matrix is table.obsm["features_1"]


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
    # Clearing absent columns wrote nothing: no entry, and no table change.
    assert CLASSIFIER_CONFIG_KEY not in table.uns
    assert "two labeled classes" in controller.status_message
    assert controller.status_kind == "warning"
    assert table_state_changes == []
    assert prediction_state_changes == []


def test_classifier_controller_records_the_latest_reason_when_the_scope_is_already_cleared(
    qtbot, sdata_blobs: SpatialData
) -> None:
    _set_deterministic_features(sdata_blobs)
    _set_feature_metadata(sdata_blobs)
    table = sdata_blobs["table"]
    _set_existing_predictions(table)
    controller = ClassifierController(debounce_interval_ms=0)
    controller.bind(sdata_blobs, "blobs_labels", "table", "features_1")

    # The summary blocks Train, and the scope's predictions are cleared.
    _set_user_classes(sdata_blobs, {1: 1, 2: 1})
    controller.schedule_retrain(immediate=True)
    assert "two labeled classes" in _widget_entry(table)["reason"]
    assert table.obs[PRED_CLASS_COLUMN].isna().all()

    # The worker finds a new reason; clearing the scope again changes nothing.
    _set_user_classes(sdata_blobs, {1: 1, 2: 1, 24: 2, 25: 2})
    _set_nan_features_for_instances(sdata_blobs, {24, 25})
    controller.retrain_now()
    qtbot.waitUntil(lambda: "with finite features" in _widget_entry(table)["reason"], timeout=5000)

    assert _widget_entry(table)["model_provenance"] is None


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

    def raise_shape_error(feature_matrix, n_obs):
        del feature_matrix, n_obs
        raise ValueError("Feature matrix has 25 rows but the table has 26 observations.")

    monkeypatch.setattr(classifier_module, "feature_matrix_n_features", raise_shape_error)

    controller = ClassifierController(
        debounce_interval_ms=0,
    )
    controller.bind(sdata_blobs, "blobs_labels", "table", "features_1")
    controller.schedule_retrain(immediate=True)

    assert PRED_CLASS_COLUMN not in table.obs
    assert PRED_CONFIDENCE_COLUMN not in table.obs
    assert PRED_CLASS_COLORS_KEY not in table.uns
    assert "rows but the table has" in controller.status_message
    assert CLASSIFIER_CONFIG_KEY not in table.uns
    assert controller.status_kind == "warning"


def test_classifier_controller_drops_stale_results(qtbot, monkeypatch, sdata_blobs: SpatialData) -> None:
    _set_deterministic_features(sdata_blobs)
    _set_feature_metadata(sdata_blobs)
    _set_user_classes(sdata_blobs, {1: 1, 2: 1, 24: 2, 25: 2})

    call_log: list[int] = []

    # A real thread worker that knows its job, so the two jobs can differ.
    @classifier_module.thread_worker(start_thread=False, ignore_errors=True)
    def fake_fit(job):
        call_log.append(job.job_id)
        if job.job_id == 1:
            time.sleep(0.2)
            return _fake_training_result(job, pred_class=1, training_timestamp="2026-04-08T12:00:00+00:00")
        return _fake_training_result(job, pred_class=2, training_timestamp="2026-04-08T12:00:00+00:00")

    monkeypatch.setattr(ClassifierController, "_create_training_worker", lambda self, job: fake_fit(job))

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

    @classifier_module.thread_worker(start_thread=False, ignore_errors=True)
    def fake_fit(job):
        call_log.append(job.job_id)
        return _fake_training_result(job, training_timestamp="2026-04-08T12:00:00+00:00")

    monkeypatch.setattr(ClassifierController, "_create_training_worker", lambda self, job: fake_fit(job))

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


def test_classifier_controller_leaves_the_table_alone_when_training_errors(
    qtbot, monkeypatch, sdata_blobs: SpatialData
) -> None:
    _set_deterministic_features(sdata_blobs)
    _set_feature_metadata(sdata_blobs)
    _set_user_classes(sdata_blobs, {1: 1, 2: 1, 24: 2, 25: 2})
    table_state_changes: list[classifier_module.ClassifierTableStateChange] = []
    prediction_state_changes: list[str] = []

    def raise_fit_error(worker_input):
        del worker_input
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

    # A failed training wrote no column, so it writes no entry either.
    assert table_state_changes == []
    assert prediction_state_changes == []
    assert CLASSIFIER_CONFIG_KEY not in sdata_blobs["table"].uns
    assert "boom" in controller.status_message
    assert controller.is_dirty is True
    assert controller.can_export_classifier is False


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
        worker = _DeferredWorker(_fake_training_result(job))
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
    result = classifier_module.ClassifierTrainingResult(
        training_timestamp="2026-04-13T09:00:00+00:00",
        predicted_row_positions=np.array([0, 1], dtype=np.int64),
        pred_classes=np.array([1, 2], dtype=np.int64),
        pred_confidences=np.array([0.9, 0.8], dtype=np.float64),
        n_training_labels=2,
        class_labels_seen=(1, 2),
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
        worker = _DeferredWorker(_fake_training_result(job, confidence=0.91))
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
    table.uns[CLASSIFIER_CONFIG_KEY] = _trained_config(
        prediction_scope="selected_segmentation_only",
        prediction_regions=["blobs_labels"],
        training_regions=["blobs_labels"],
        n_features=2,
    )

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
    table.uns[CLASSIFIER_CONFIG_KEY] = _trained_config(
        prediction_scope="all",
        prediction_regions=["blobs_labels"],
        training_regions=["blobs_labels"],
        n_features=2,
    )

    controller = ClassifierController(debounce_interval_ms=0)
    controller.bind(sdata_blobs, "blobs_labels", "table", "features_1")
    controller.reset_after_reload()

    assert controller.is_dirty is True
    assert 'prediction scope "all"' in controller.status_message


def test_classifier_controller_reset_after_reload_accepts_multi_region_prediction_scope(
    sdata_blobs_multi_region: SpatialData,
) -> None:
    table = sdata_blobs_multi_region["table_multi"]
    # The loaded count comes from the predictions in the table.
    table.obs[PRED_CLASS_COLUMN] = pd.Categorical(np.full(table.n_obs, 2, dtype=np.int64), categories=[2])
    table.obs[PRED_CONFIDENCE_COLUMN] = pd.Series(np.full(table.n_obs, 0.77), index=table.obs.index, dtype="float64")
    table.uns[CLASSIFIER_CONFIG_KEY] = _trained_config(
        prediction_scope="all",
        prediction_regions=["blobs_labels", "blobs_labels_2"],
        training_regions=["blobs_labels", "blobs_labels_2"],
        n_features=4,
    )

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


@pytest.mark.parametrize("config", ["never_trained", "headless_wrote_pred_class", "old_flat_layout"])
def test_classifier_controller_reset_after_reload_asks_to_train_without_a_widget_entry(
    qtbot, sdata_blobs: SpatialData, config: str
) -> None:
    _set_deterministic_features(sdata_blobs)
    _set_feature_metadata(sdata_blobs)
    _set_user_classes(sdata_blobs, {1: 1, 2: 1, 24: 2, 25: 2})
    table = sdata_blobs["table"]
    controller = ClassifierController(debounce_interval_ms=0)
    controller.bind(sdata_blobs, "blobs_labels", "table", "features_1")
    controller.retrain_now()
    qtbot.waitUntil(lambda: controller.can_export_classifier, timeout=5000)

    # The table as reloaded from disk.
    if config == "never_trained":
        del table.uns[CLASSIFIER_CONFIG_KEY]
    elif config == "headless_wrote_pred_class":
        table.uns[CLASSIFIER_CONFIG_KEY] = {
            PRED_CLASS_COLUMN: _headless_entry(pred_class_column=PRED_CLASS_COLUMN, prediction_regions=["blobs_labels"])
        }
    else:
        table.uns[CLASSIFIER_CONFIG_KEY] = {"trained": True, "feature_key": "features_1"}
    controller.reset_after_reload()

    assert controller.is_dirty is True
    assert (
        controller.status_message == "Classifier: no trained model for this table. Click Train Classifier to train one."
    )
    assert controller.can_export_classifier is False
