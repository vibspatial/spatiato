from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
from qtpy.QtCore import QObject, QTimer
from sklearn.ensemble import RandomForestClassifier

import spatiato.core.object_classification.classifier as _classifier_core
from spatiato.core.class_palette import normalize_class_values
from spatiato.core.feature_matrix_metadata import (
    CUSTOM_OBSM_SOURCE_KIND,
    FeatureMatrixMetadataState,
    feature_matrix_n_features,
    inspect_feature_matrix_metadata,
    normalize_feature_columns,
    normalize_feature_matrix_source_kind,
)
from spatiato.core.object_classification.annotation import USER_CLASS_COLUMN
from spatiato.core.object_classification.classifier import (
    PredictionStateChange,
    _get_feature_metadata,
    _normalize_prediction_regions,
    _predict_classes_and_confidences,
    _read_rows_with_finite_features,
    _resolve_region_row_positions,
)
from spatiato.core.object_classification.classifier_export import (
    ClassifierExportBundle,
    ClassifierModelSnapshot,
    build_classifier_export_bundle,
    write_classifier_export_bundle,
)
from spatiato.core.persistence import TableComponentPath
from spatiato.core.spatialdata import SpatialDataTableMetadata, get_table, get_table_metadata


def _resolve_thread_worker() -> Any:
    try:
        from napari.qt.threading import thread_worker
    except Exception:  # pragma: no cover - fallback for sandboxed test imports  # noqa: BLE001
        from superqt.utils import thread_worker

    return thread_worker


if TYPE_CHECKING:
    from anndata import AnnData
    from spatialdata import SpatialData

thread_worker = _resolve_thread_worker()

BoolArray = _classifier_core.BoolArray
TableRowPositions = _classifier_core.TableRowPositions
PRED_CLASS_COLUMN = _classifier_core.PRED_CLASS_COLUMN
PRED_CLASS_COLORS_KEY = _classifier_core.PRED_CLASS_COLORS_KEY
PRED_CONFIDENCE_COLUMN = _classifier_core.PRED_CONFIDENCE_COLUMN
CLASSIFIER_CONFIG_KEY = _classifier_core.CLASSIFIER_CONFIG_KEY

DEFAULT_RETRAIN_DEBOUNCE_MS = 300
MIN_LABELED_SAMPLES = 2
ClassifierTrainingBlockerKind = Literal["metadata", "data"]
_CLASSIFIER_TRAINING_BLOCKER_KINDS: tuple[ClassifierTrainingBlockerKind, ...] = ("metadata", "data")
RANDOM_FOREST_PARAMS = {
    "n_estimators": 100,
    "random_state": 0,
    "n_jobs": -2,
}

ClassifierScopeMode = Literal["selected_segmentation_only", "all"]
ClassifierTableStateChangeSource = Literal[
    "object_classification_inference",
    "object_classification_metadata",
]
DEFAULT_TRAINING_SCOPE: ClassifierScopeMode = "all"
DEFAULT_PREDICTION_SCOPE: ClassifierScopeMode = "selected_segmentation_only"


@dataclass(frozen=True)
class ClassifierTableStateChange:
    """Carry an accepted classifier table change to the widget.

    The classifier controller knows the authoritative paths, affected regions,
    and source of a mutation. In particular, inference may affect every region
    in the resolved prediction scope rather than only the segmentation
    currently selected in the widget. This callback payload preserves that
    context so the widget can publish an accurate ``TableStateChangedEvent``
    without reconstructing or guessing the classifier's mutation scope.
    """

    paths: frozenset[TableComponentPath]
    regions: tuple[str, ...]
    source: ClassifierTableStateChangeSource


@dataclass(frozen=True)
class ResolvedClassifierScope:
    """Resolved classifier scope for one training or prediction selection.

    ``regions`` is the resolved semantic scope requested by the user, expressed
    as table region names. ``table_row_positions`` contains every table row
    position in those regions, ascending. Whether a row has finite features is
    known only in the worker, which reads the feature values and skips the
    rows without.
    """

    mode: ClassifierScopeMode
    regions: tuple[str, ...]
    table_row_positions: TableRowPositions

    @property
    def n_rows_in_regions(self) -> int:
        return int(self.table_row_positions.size)


@dataclass(frozen=True)
class ResolvedClassifierScopes:
    """Resolved training and prediction scopes for the current controller state."""

    training: ResolvedClassifierScope
    prediction: ResolvedClassifierScope


@dataclass(frozen=True)
class ClassifierPreparationSummary:
    """Describe the classifier inputs before a job, from `obs` and `uns` only.

    It reads no feature values: the column count comes from the matrix's shape.
    A job's outcome, such as the rows it used after skipping those without
    finite features, is not stored here: it goes into the classifier config
    directly (`_build_classifier_config(...)`).

    Parameters
    ----------
    training_scope
        The training scope: its labels elements and their table rows.
    prediction_scope
        The prediction scope: its labels elements and their table rows.
    labeled_count
        Labeled rows in the training scope, also those without finite features.
    class_labels
        The classes of those labeled rows.
    n_features
        Column count of the feature matrix, from its shape; `None` when the
        matrix is missing or cannot be used.
    training_blocker_reason
        Why Train is not allowed now, shown to the user; `None` when it is.
    training_blocker_kind
        What blocks Train: `"metadata"`, the feature metadata is missing or
        does not match, and nothing changes; or `"data"`, the selection or
        the labels do not allow training, and Train clears the scope's
        predictions and records the reason. `None` when Train is allowed.
    """

    training_scope: ResolvedClassifierScope
    prediction_scope: ResolvedClassifierScope
    labeled_count: int
    class_labels: tuple[int, ...]
    n_features: int | None
    training_blocker_reason: str | None = None
    training_blocker_kind: ClassifierTrainingBlockerKind | None = None

    def __post_init__(self) -> None:
        if (
            self.training_blocker_kind is not None
            and self.training_blocker_kind not in _CLASSIFIER_TRAINING_BLOCKER_KINDS
        ):
            raise ValueError(f"Unsupported classifier training blocker kind: {self.training_blocker_kind!r}.")
        if (self.training_blocker_reason is None) != (self.training_blocker_kind is None):
            raise ValueError("A training blocker needs both a reason and a kind, or neither.")

    @property
    def eligible(self) -> bool:
        return self.training_blocker_kind is None

    @property
    def training_row_count(self) -> int:
        return self.training_scope.n_rows_in_regions

    @property
    def prediction_row_count(self) -> int:
        return self.prediction_scope.n_rows_in_regions

    @property
    def training_region_count(self) -> int:
        return len(self.training_scope.regions)


@dataclass(frozen=True)
class ClassifierWorkerInput:
    """What a classifier worker reads: the feature matrix, the rows to read, and the training labels.

    The input carries the feature matrix object and the row positions to read,
    not copies. The worker reads only the labeled rows of the training scope
    and the rows of the prediction scope, with `read_matrix_rows(...)`. For a
    lazy or backed matrix, only these rows reach memory: an unlabeled row is
    read only to be predicted, and a row outside both scopes is never read.

    Parameters
    ----------
    feature_key
        Key of the feature matrix in `table.obsm`, for messages.
    feature_matrix
        The matrix object from `table.obsm[feature_key]`, as attached: NumPy,
        SciPy, Dask, or a backed handle.
    train_row_positions
        Table row positions of every labeled row in the training scope,
        including rows without finite features: the worker reads them and
        skips those.
    train_labels
        `user_class` of each row, aligned with `train_row_positions`.
    train_row_regions
        Region of each row, aligned with `train_row_positions` (rows without
        finite features included), so the worker can name a labels element
        whose labeled rows it all skipped.
    prediction_row_positions
        Table row positions of every row in the prediction scope, including
        rows without finite features: the worker reads them and skips those.
    prediction_row_regions
        Region of each row, aligned with `prediction_row_positions` (rows
        without finite features included), so the worker can name a labels
        element whose rows it all skipped.

    Notes
    -----
    Accepted on purpose: the worker may read the matrix while Feature
    Extraction rewrites the same key.

    - Unbacked, harpy installs a new matrix, and the job still reads the old
      one.
    - Backed, the job reads from the store that harpy is rewriting, so it may
      read a mix of old and new values, or fail.

    This needs the user to recalculate the very key the classifier uses while
    training on it, which is rare. The outcome is always visible, never
    silently wrong:

    - the job is still running when Feature Extraction ends: the "updated"
      event reaches `invalidate_for_feature_matrix_overwrite(...)`, which
      discards the job's result, so nothing wrong reaches the table;
    - the job finished first: its predictions, possibly made from mixed
      values, stay in the table, and the same call marks the model stale;
      the next Train replaces the predictions (it does not retrain by itself);
    - the read failed: `_on_worker_errored(...)` shows "training failed: …"
      and records the classifier config as not trained.

    It is not prevented: that would need the job to keep the table busy, so
    that Write and Reload wait for every training run, or Train to wait for a
    busy table, which does not cover Feature Extraction starting during a job.
    """

    feature_key: str
    feature_matrix: Any
    train_row_positions: TableRowPositions
    train_labels: np.ndarray
    train_row_regions: np.ndarray
    prediction_row_positions: TableRowPositions
    prediction_row_regions: np.ndarray

    def __post_init__(self) -> None:
        # The worker applies one mask per scope to these arrays, so they must
        # align. The positions are not checked against the matrix's shape:
        # `read_matrix_rows(...)` checks that they lie within it.
        n_train_rows = len(self.train_row_positions)
        if len(self.train_labels) != n_train_rows or len(self.train_row_regions) != n_train_rows:
            raise ValueError("`train_labels` and `train_row_regions` must align with `train_row_positions`.")
        if len(self.prediction_row_regions) != len(self.prediction_row_positions):
            raise ValueError("`prediction_row_regions` must align with `prediction_row_positions`.")

    @classmethod
    def from_summary(
        cls,
        summary: ClassifierPreparationSummary,
        table: AnnData,
        *,
        feature_key: str,
        region_key: str,
    ) -> ClassifierWorkerInput:
        """Derive what the worker reads from the summary's scopes, on the main thread.

        Only `obs` is read here, for the labels and the regions; the feature
        matrix is passed on as the object attached at `table.obsm[feature_key]`.
        `prediction_row_positions` is the prediction scope's positions array
        itself, and `train_row_positions` are the labeled rows of the training
        scope, with their labels.
        """
        user_class_values = _get_user_class_values(table.obs)
        training_user_class_values = user_class_values.iloc[summary.training_scope.table_row_positions]
        labeled_mask = training_user_class_values.notna().to_numpy()
        train_row_positions = summary.training_scope.table_row_positions[labeled_mask]
        prediction_row_positions = summary.prediction_scope.table_row_positions
        row_regions = np.asarray(table.obs[region_key].to_numpy(), dtype=object)
        return cls(
            feature_key=feature_key,
            feature_matrix=table.obsm[feature_key],
            train_row_positions=train_row_positions,
            train_labels=training_user_class_values.iloc[labeled_mask].to_numpy(dtype=np.int64),
            train_row_regions=row_regions[train_row_positions],
            prediction_row_positions=prediction_row_positions,
            prediction_row_regions=row_regions[prediction_row_positions],
        )


@dataclass(frozen=True)
class ClassifierJob:
    """One launched classifier job, kept on the main thread.

    The worker gets only `worker_input`; the main thread keeps the rest, and
    combines it with the worker's result.

    Parameters
    ----------
    job_id
        Identity of the job; a result of an older job is ignored.
    feature_key
        Key of the feature matrix in `table.obsm`.
    table_name
        Table the job reads, and writes its predictions to.
    summary
        The summary the job was launched from: the scopes, and the labeled
        count and classes before any row is skipped.
    worker_input
        What the worker reads, derived from `summary` by
        `ClassifierWorkerInput.from_summary(...)`; `None` when the summary is
        not eligible, for a job that never reaches a worker.
    """

    job_id: int
    feature_key: str
    table_name: str
    summary: ClassifierPreparationSummary
    worker_input: ClassifierWorkerInput | None = None

    @property
    def training_scope(self) -> ResolvedClassifierScope:
        return self.summary.training_scope

    @property
    def prediction_scope(self) -> ResolvedClassifierScope:
        return self.summary.prediction_scope


@dataclass(frozen=True)
class ClassifierTrainingResult:
    """What a classifier worker computed when it trained.

    Parameters
    ----------
    trained_at
        When the model was trained, as an ISO 8601 timestamp in UTC.
    predicted_row_positions
        Table row positions of the rows that got a prediction: the rows of the
        prediction scope with finite features. The other rows of the scope
        were skipped, and get no prediction.
    pred_classes
        Predicted class of each row, aligned with `predicted_row_positions`.
    pred_confidences
        Probability of that class for each row, aligned with
        `predicted_row_positions`. `_update_table_prediction_state(...)`
        checks both alignments again before it changes the table.
    labeled_count
        Labeled rows used for training, after skipping those without finite
        features.
    class_labels
        The classes of those labeled rows.
    n_skipped_labeled_rows
        Labeled rows not used for training, because they lack finite features.
    n_skipped_prediction_rows
        Rows of the prediction scope that got no prediction, for the same
        reason.
    prediction_regions_without_features
        Labels elements of the prediction scope whose rows were all skipped.
    estimator
        The fitted classifier, kept for export. `None` in a result built
        without one, as in tests; the model then cannot be exported.
    """

    trained_at: str
    predicted_row_positions: TableRowPositions
    pred_classes: np.ndarray
    pred_confidences: np.ndarray
    labeled_count: int
    class_labels: tuple[int, ...]
    n_skipped_labeled_rows: int = 0
    n_skipped_prediction_rows: int = 0
    prediction_regions_without_features: tuple[str, ...] = ()
    estimator: RandomForestClassifier | None = None

    def __post_init__(self) -> None:
        # As for `ClassifierWorkerInput`: the arrays must align. The prediction
        # write checks this again, with the positions against the scope.
        n_predicted_rows = len(self.predicted_row_positions)
        if len(self.pred_classes) != n_predicted_rows or len(self.pred_confidences) != n_predicted_rows:
            raise ValueError("`pred_classes` and `pred_confidences` must align with `predicted_row_positions`.")


@dataclass(frozen=True)
class ClassifierIneligibleResult:
    """What a classifier worker found when too few rows were usable after skipping those without finite features.

    Nothing was trained; `reason` says why, with the counts of the usable rows.
    """

    reason: str


ClassifierWorkerResult = ClassifierTrainingResult | ClassifierIneligibleResult


@thread_worker(start_thread=False, ignore_errors=True)
def _run_classifier_job(worker_input: ClassifierWorkerInput) -> ClassifierWorkerResult:
    return _fit_classifier_job(worker_input)


# Keep the synchronous fit/predict logic separate from the `@thread_worker`
# wrapper above: `_run_classifier_job(...)` returns a worker object, while this
# function directly computes and returns the eventual result.
def _fit_classifier_job(worker_input: ClassifierWorkerInput) -> ClassifierWorkerResult:
    # Rows without finite features (a NaN or ±inf in any column) are skipped:
    # labels elements without calculated features, table rows without an
    # object in the labels element, or undefined features.
    train_features, train_finite = _read_rows_with_finite_features(
        worker_input.feature_matrix, worker_input.train_row_positions
    )
    train_labels = worker_input.train_labels[train_finite]
    labeled_count = int(train_finite.sum())
    class_labels = tuple(sorted(int(value) for value in np.unique(train_labels)))

    # The summary counted every labeled row; check again with the rows used.
    training_reason = _training_rows_ineligible_reason(worker_input, labeled_count, class_labels, train_finite)
    if training_reason is not None:
        return ClassifierIneligibleResult(reason=training_reason)

    prediction_positions = worker_input.prediction_row_positions
    predict_features, predict_finite = _read_rows_with_finite_features(
        worker_input.feature_matrix, prediction_positions
    )
    prediction_regions_without_features = _regions_without_finite_rows(
        worker_input.prediction_row_regions, predict_finite
    )
    if not predict_finite.any():
        reason = (
            f'Feature matrix "{worker_input.feature_key}" has no usable rows in the prediction scope: '
            f"all {prediction_positions.size} row(s) have non-finite or missing values."
            f"{_regions_without_features_note(prediction_regions_without_features)}"
        )
        return ClassifierIneligibleResult(reason=reason)

    classifier = RandomForestClassifier(**RANDOM_FOREST_PARAMS)
    classifier.fit(train_features, train_labels)
    pred_classes, pred_confidences = _predict_classes_and_confidences(classifier, predict_features)

    return ClassifierTrainingResult(
        trained_at=datetime.now(UTC).isoformat(),
        predicted_row_positions=np.asarray(prediction_positions[predict_finite], dtype=np.int64),
        pred_classes=pred_classes,
        pred_confidences=pred_confidences,
        labeled_count=labeled_count,
        class_labels=class_labels,
        n_skipped_labeled_rows=int((~train_finite).sum()),
        n_skipped_prediction_rows=int((~predict_finite).sum()),
        prediction_regions_without_features=prediction_regions_without_features,
        estimator=classifier,
    )


def _regions_without_finite_rows(row_regions: np.ndarray, finite: BoolArray) -> tuple[str, ...]:
    """Return the regions, in order of appearance, whose rows all lack finite features."""
    regions_with_finite_rows = set(row_regions[finite].tolist())
    return tuple(str(region) for region in pd.unique(row_regions) if region not in regions_with_finite_rows)


def _regions_without_features_note(regions: tuple[str, ...]) -> str:
    if not regions:
        return ""
    names = ", ".join(f'"{region}"' for region in regions)
    return f" Labels element(s) {names} have no rows with finite features; were features calculated for them?"


def _training_rows_ineligible_reason(
    worker_input: ClassifierWorkerInput,
    labeled_count: int,
    class_labels: tuple[int, ...],
    train_finite: BoolArray,
) -> str | None:
    """Return why the labeled rows with finite features cannot train a classifier, or `None`."""
    note = _regions_without_features_note(_regions_without_finite_rows(worker_input.train_row_regions, train_finite))
    if labeled_count < MIN_LABELED_SAMPLES:
        return (
            f"Need at least {MIN_LABELED_SAMPLES} labeled samples with finite features before training the "
            f"classifier: {labeled_count} of {train_finite.size} labeled row(s) have finite features.{note}"
        )
    if len(class_labels) < 2:
        return (
            "Need at least two labeled classes with finite features before training the classifier: "
            f"the {labeled_count} labeled row(s) with finite features cover {len(class_labels)} class(es).{note}"
        )
    return None


def _feature_matrix_metadata_training_blocker_reason(
    metadata_state: FeatureMatrixMetadataState,
) -> str | None:
    if metadata_state.status == "registered_valid":
        return None
    if metadata_state.status == "unregistered":
        return f'Register feature metadata for "{metadata_state.feature_key}" before training the classifier.'
    if metadata_state.status == "registered_mismatched":
        if not metadata_state.error:
            raise ValueError("Feature matrix metadata state 'registered_mismatched' must include an error message.")
        detail = metadata_state.error
        return f'Feature matrix "{metadata_state.feature_key}" has mismatched metadata. {detail}'
    if metadata_state.status == "invalid_matrix":
        if not metadata_state.error:
            raise ValueError("Feature matrix metadata state 'invalid_matrix' must include an error message.")
        detail = metadata_state.error
        return f'Feature matrix "{metadata_state.feature_key}" cannot be used for training. {detail}'
    if metadata_state.status == "missing_matrix":
        if not metadata_state.error:
            raise ValueError("Feature matrix metadata state 'missing_matrix' must include an error message.")
        return metadata_state.error
    raise ValueError(f"Unsupported feature matrix metadata status: {metadata_state.status!r}.")


class ClassifierController:
    """Manage debounced background classifier retraining for the active table."""

    def __init__(
        self,
        *,
        debounce_interval_ms: int = DEFAULT_RETRAIN_DEBOUNCE_MS,
        on_state_changed: Callable[[], None] | None = None,
        on_table_state_changed: Callable[[ClassifierTableStateChange], None] | None = None,
        on_prediction_state_changed: Callable[[], None] | None = None,
        # In the widget, pass the widget as parent so Qt destroys the debounce
        # timer with the UI. Tests and non-widget callers can leave this as None.
        timer_parent: QObject | None = None,
    ) -> None:
        self._debounce_interval_ms = max(0, int(debounce_interval_ms))
        self._on_state_changed = on_state_changed
        self._on_table_state_changed = on_table_state_changed
        self._on_prediction_state_changed = on_prediction_state_changed
        self._is_shutdown = False

        self._selected_spatialdata: SpatialData | None = None
        self._selected_labels_name: str | None = None
        self._selected_table_name: str | None = None
        self._selected_feature_key: str | None = None
        self._selected_training_scope: ClassifierScopeMode = DEFAULT_TRAINING_SCOPE
        self._selected_prediction_scope: ClassifierScopeMode = DEFAULT_PREDICTION_SCOPE
        # TODO: clean up. There is overlap between the above class attributes and spatialdatatablemetadata.
        self._selected_table_metadata: SpatialDataTableMetadata | None = None

        self._latest_requested_job_id = 0
        self._active_worker_job_id: int | None = None
        self._active_worker: Any | None = None
        # A snapshot exists only when there is an exportable fitted model.
        # Reasons for missing export state stay separate so the snapshot type
        # never has to represent both success and failure.
        self._model_snapshot: ClassifierModelSnapshot | None = None
        self._model_snapshot_missing_reason: str | None = None
        self._is_dirty = False

        self._status_message = "Classifier: choose an annotation table and feature matrix."
        self._status_kind = "warning"

        self._debounce_timer = QTimer(timer_parent)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(self._debounce_interval_ms)
        self._debounce_timer.timeout.connect(self._launch_scheduled_retrain)

    @property
    def status_message(self) -> str:
        """Return the latest user-facing classifier status message."""
        return self._status_message

    @property
    def status_kind(self) -> str:
        """Return the latest status level: info, warning, success, or error."""
        return self._status_kind

    @property
    def is_training(self) -> bool:
        """Return whether a classifier worker is currently active."""
        return self._active_worker is not None

    @property
    def is_dirty(self) -> bool:
        """Return whether the current classifier outputs are stale."""
        return self._is_dirty

    @property
    def can_retrain(self) -> bool:
        """Return whether the current classifier inputs support a retrain request."""
        if self._is_shutdown or self.is_training:
            return False

        summary = self._prepare_classifier_summary()
        return summary is not None and summary.eligible

    @property
    def can_export_classifier(self) -> bool:
        """Return whether the current fitted classifier snapshot can be exported."""
        try:
            self._validate_export_ready()
        except ValueError:
            return False
        return True

    @property
    def classifier_export_unavailable_reason(self) -> str | None:
        """Return the current reason classifier export is unavailable, if any."""
        try:
            self._validate_export_ready()
        except ValueError as error:
            return str(error)
        return None

    def bind(
        self,
        sdata: SpatialData | None,
        labels_name: str | None,
        table_name: str | None,
        feature_key: str | None,
        training_scope: ClassifierScopeMode = DEFAULT_TRAINING_SCOPE,
        prediction_scope: ClassifierScopeMode = DEFAULT_PREDICTION_SCOPE,
    ) -> bool:
        """Bind the controller to the currently selected SpatialData inputs.

        Binding is read-only with respect to prediction state. Existing
        prediction columns are validated by the widget's table-state boundary,
        while absent prediction columns remain absent until a successful
        prediction write creates them.
        """
        if self._is_shutdown:
            return False

        next_table_metadata = None
        if sdata is not None and table_name is not None:
            next_table_metadata = get_table_metadata(sdata, table_name)

        normalized_training_scope = _normalize_scope_mode(training_scope)
        normalized_prediction_scope = _normalize_scope_mode(prediction_scope)
        context_changed = (
            sdata is not self._selected_spatialdata
            or labels_name != self._selected_labels_name
            or table_name != self._selected_table_name
            or feature_key != self._selected_feature_key
            or normalized_training_scope != self._selected_training_scope
            or normalized_prediction_scope != self._selected_prediction_scope
        )

        self._selected_spatialdata = sdata
        self._selected_labels_name = labels_name
        self._selected_table_name = table_name
        self._selected_feature_key = feature_key
        self._selected_training_scope = normalized_training_scope
        self._selected_prediction_scope = normalized_prediction_scope
        self._selected_table_metadata = next_table_metadata

        if context_changed:
            self._cancel_pending_and_active_jobs()
            self._clear_model_snapshot("the classifier selection changed")

        table = self._get_bound_table()
        if table is None:
            self._is_dirty = False
            self._set_status("Classifier: choose an annotation table and feature matrix.", kind="warning")
            return context_changed

        self._update_idle_status()
        return context_changed

    def mark_dirty(self, reason: str | None = None) -> None:
        """Invalidate older asynchronous work and mark classifier outputs stale."""
        if self._is_shutdown:
            return

        # mark_dirty() is also the asynchronous job-invalidation boundary:
        #
        # a pending or active classifier job captures the current inputs
        #     ↓
        # an annotation, feature matrix, or other classifier input changes
        #     ↓
        # invalidate the captured job identity
        #     ↓
        # any late result fails the identity guard in _on_worker_returned()
        #
        # Auto-train being disabled prevents a replacement job from starting;
        # it does not guarantee that no older job is still pending or running.
        # That older work must still be invalidated.
        self._invalidate_async_jobs()

        if self._get_bound_table() is None:
            self._is_dirty = False
            self._clear_model_snapshot()
            self._set_status("Classifier: choose an annotation table and feature matrix.", kind="warning")
            return

        self._is_dirty = True
        self._clear_model_snapshot("the classifier was marked stale")
        self._update_idle_status(reason=reason)

    def retrain_now(self) -> bool:
        """Immediately retrain the classifier for the current inputs."""
        return self.schedule_retrain(immediate=True)

    def schedule_retrain(self, *, immediate: bool = False) -> bool:
        """Schedule a background retraining job for the current classifier inputs."""
        if self._is_shutdown:
            return False

        if self._get_bound_table() is None:
            self._set_status("Classifier: choose an annotation table and feature matrix.", kind="warning")
            return False
        if self._selected_feature_key is None:
            self._set_status("Classifier: choose a feature matrix before training.", kind="warning")
            return False

        summary = self._prepare_classifier_summary()
        if summary is None:
            self._set_status("Classifier: choose an annotation table and feature matrix.", kind="warning")
            return False
        if summary.training_blocker_kind == "metadata":
            self._set_status(f"Classifier: {summary.training_blocker_reason}", kind="warning")
            return False

        self._latest_requested_job_id += 1
        self._debounce_timer.stop()
        self._cancel_active_worker()
        self._is_dirty = True
        self._clear_model_snapshot("classifier training was started")

        if immediate or self._debounce_interval_ms == 0:
            self._launch_retrain_job(self._latest_requested_job_id)
        else:
            self._debounce_timer.start()
            self._set_status("Classifier: model is stale. Classifier training is scheduled.", kind="info")

        return True

    def shutdown(self, *args: object) -> None:
        """Stop async classifier work and detach widget callbacks.

        Shutdown is terminal and idempotent. It is used when the owning widget is
        being destroyed, so pending timer or worker signals must not call back
        into Qt objects that may already be gone.

        Extra positional arguments are ignored so this method can be connected
        directly to Qt signals such as ``QObject.destroyed``.
        """
        del args

        if self._is_shutdown:
            return

        self._is_shutdown = True
        self._latest_requested_job_id += 1
        self._debounce_timer.stop()
        self._cancel_active_worker()
        self._on_state_changed = None
        self._on_table_state_changed = None
        self._on_prediction_state_changed = None

    def freeze_for_reload(self) -> None:
        """Cancel pending async work so reload cannot apply stale classifier results."""
        if self._is_shutdown:
            return

        self._invalidate_async_jobs()
        self._clear_model_snapshot("the table is being reloaded")
        self._update_idle_status()

    def invalidate_for_feature_matrix_overwrite(self, feature_key: str) -> bool:
        """Invalidate pending work when the selected feature matrix was overwritten in place.

        This also catches a job that read the matrix while Feature Extraction
        rewrote it, which is accepted on purpose (see `ClassifierJob`).
        """
        if self._is_shutdown:
            return False

        if self._get_bound_table() is None or self._selected_feature_key != feature_key:
            return False

        self._invalidate_async_jobs()
        self._is_dirty = True
        self._clear_model_snapshot(f'feature matrix "{feature_key}" was overwritten')
        self._update_idle_status(reason=f'feature matrix "{feature_key}" was overwritten')
        return True

    def reset_after_reload(self) -> None:
        """Recompute classifier state from the reloaded table without retraining."""
        if self._is_shutdown:
            return

        self._invalidate_async_jobs()
        self._clear_model_snapshot("the table was reloaded from disk")
        table = self._get_bound_table()
        if table is None:
            self._is_dirty = False
            self._set_status("Classifier: choose an annotation table and feature matrix.", kind="warning")
            return

        self._is_dirty = False
        self._update_status_from_reloaded_table(table)

    def describe_current_preparation(self) -> ClassifierPreparationSummary | None:
        """Return a side-effect-free summary of the currently bound classifier state."""
        if self._is_shutdown:
            return None

        return self._prepare_classifier_summary()

    def export_classifier(self, path: str | Path) -> ClassifierExportBundle:
        """Export the current fitted classifier snapshot to a joblib artifact."""
        self._validate_export_ready()
        assert self._model_snapshot is not None
        bundle = build_classifier_export_bundle(self._model_snapshot)
        write_classifier_export_bundle(path, bundle)
        return bundle

    def _launch_scheduled_retrain(self) -> None:
        if self._is_shutdown:
            return

        self._launch_retrain_job(self._latest_requested_job_id)

    def _launch_retrain_job(self, job_id: int) -> None:
        if self._is_shutdown:
            return

        if job_id != self._latest_requested_job_id:
            return

        job = self._prepare_classifier_job(job_id)
        if job is None:
            return

        summary = job.summary
        if summary.training_blocker_reason is not None:
            if summary.training_blocker_kind == "metadata":
                self._set_status(f"Classifier: {summary.training_blocker_reason}", kind="warning")
                return
            self._apply_ineligible_state(
                feature_key=job.feature_key,
                summary=summary,
                reason=summary.training_blocker_reason,
            )
            return

        worker = self._create_training_worker(job)
        self._active_worker = worker
        self._active_worker_job_id = job_id
        # Only `job.worker_input` goes to the worker, which trains in its own
        # thread. The job itself stays with these handlers, which Qt runs on
        # the main thread when the worker's signals arrive; they combine the
        # job with the worker's result.
        worker.returned.connect(partial(self._on_worker_returned, job))
        worker.errored.connect(partial(self._on_worker_errored, job))
        worker.finished.connect(partial(self._on_worker_finished, job))
        self._set_status(
            (
                f"Classifier: training RandomForest on {summary.labeled_count} labeled objects "
                f"across {len(summary.class_labels)} classes from {summary.training_row_count} "
                f"rows in {summary.training_region_count} region(s). "
                f"Prediction scope contains {summary.prediction_row_count} row(s)."
            ),
            kind="info",
        )
        worker.start()

    def _prepare_classifier_summary(self) -> ClassifierPreparationSummary | None:
        table = self._get_bound_table()
        metadata = self._selected_table_metadata
        if table is None or metadata is None or self._selected_labels_name is None or self._selected_table_name is None:
            return None

        scopes = self._resolve_classifier_scopes(table, metadata)
        prediction_scope = scopes.prediction
        training_scope = scopes.training
        if self._selected_feature_key is None:
            return ClassifierPreparationSummary(
                training_scope=training_scope,
                prediction_scope=prediction_scope,
                training_blocker_reason="Choose a feature matrix before training the classifier.",
                labeled_count=0,
                class_labels=(),
                n_features=None,
                training_blocker_kind="data",
            )

        feature_matrix_metadata_state = inspect_feature_matrix_metadata(table, self._selected_feature_key)
        feature_matrix_metadata_blocker_reason = _feature_matrix_metadata_training_blocker_reason(
            feature_matrix_metadata_state
        )
        if feature_matrix_metadata_blocker_reason is not None:
            return ClassifierPreparationSummary(
                training_scope=training_scope,
                prediction_scope=prediction_scope,
                training_blocker_reason=feature_matrix_metadata_blocker_reason,
                labeled_count=0,
                class_labels=(),
                n_features=feature_matrix_metadata_state.n_features,
                training_blocker_kind="metadata",
            )

        try:
            # The summary runs on every refresh, so it reads only the matrix's
            # shape and dtype, never its values: a lazy or backed matrix stays
            # unread. Rows without finite features are skipped by the worker.
            n_features = feature_matrix_n_features(table.obsm[self._selected_feature_key], table.n_obs)
        except KeyError:
            return ClassifierPreparationSummary(
                training_scope=training_scope,
                prediction_scope=prediction_scope,
                training_blocker_reason=f'Feature matrix "{self._selected_feature_key}" is not available in ".obsm".',
                labeled_count=0,
                class_labels=(),
                n_features=None,
                training_blocker_kind="data",
            )
        except ValueError as error:
            return ClassifierPreparationSummary(
                training_scope=training_scope,
                prediction_scope=prediction_scope,
                training_blocker_reason=str(error),
                labeled_count=0,
                class_labels=(),
                n_features=None,
                training_blocker_kind="data",
            )

        if prediction_scope.n_rows_in_regions == 0:
            return ClassifierPreparationSummary(
                training_scope=training_scope,
                prediction_scope=prediction_scope,
                training_blocker_reason=(
                    f'No table rows for labels element "{self._selected_labels_name}" were found in '
                    f'"{self._selected_table_name}".'
                ),
                labeled_count=0,
                class_labels=(),
                n_features=n_features,
                training_blocker_kind="data",
            )

        if n_features == 0:
            return ClassifierPreparationSummary(
                training_scope=training_scope,
                prediction_scope=prediction_scope,
                training_blocker_reason=f'Feature matrix "{self._selected_feature_key}" does not contain any columns.',
                labeled_count=0,
                class_labels=(),
                n_features=0,
                training_blocker_kind="data",
            )

        user_class_values = _get_user_class_values(table.obs)
        training_user_class_values = user_class_values.iloc[training_scope.table_row_positions]
        labeled_mask = training_user_class_values.notna().to_numpy()
        class_labels = tuple(sorted(int(value) for value in training_user_class_values.iloc[labeled_mask].unique()))
        summary = ClassifierPreparationSummary(
            training_scope=training_scope,
            prediction_scope=prediction_scope,
            labeled_count=int(labeled_mask.sum()),
            class_labels=class_labels,
            n_features=n_features,
        )

        if summary.labeled_count < MIN_LABELED_SAMPLES:
            summary = ClassifierPreparationSummary(
                training_blocker_reason=(
                    f"Need at least {MIN_LABELED_SAMPLES} labeled samples before training the classifier. "
                    f"Resolved {summary.training_row_count} training row(s) across "
                    f"{summary.training_region_count} region(s); {summary.labeled_count} row(s) are labeled. "
                    f"Prediction scope contains {summary.prediction_row_count} row(s)."
                ),
                training_scope=summary.training_scope,
                prediction_scope=summary.prediction_scope,
                labeled_count=summary.labeled_count,
                class_labels=summary.class_labels,
                n_features=summary.n_features,
                training_blocker_kind="data",
            )
        elif len(summary.class_labels) < 2:
            summary = ClassifierPreparationSummary(
                training_blocker_reason=(
                    "Need at least two labeled classes before training the classifier. "
                    f"Resolved {summary.training_row_count} training row(s) across "
                    f"{summary.training_region_count} region(s); {summary.labeled_count} row(s) are labeled "
                    f"across {len(summary.class_labels)} class(es). "
                    f"Prediction scope contains {summary.prediction_row_count} row(s)."
                ),
                training_scope=summary.training_scope,
                prediction_scope=summary.prediction_scope,
                labeled_count=summary.labeled_count,
                class_labels=summary.class_labels,
                n_features=summary.n_features,
                training_blocker_kind="data",
            )

        return summary

    def _prepare_classifier_job(self, job_id: int) -> ClassifierJob | None:
        table = self._get_bound_table()
        if table is None or self._selected_labels_name is None or self._selected_table_name is None:
            self._set_status("Classifier: choose an annotation table and feature matrix.", kind="warning")
            return None

        summary = self._prepare_classifier_summary()
        if summary is None:
            self._set_status("Classifier: choose an annotation table and feature matrix.", kind="warning")
            return None

        feature_key = "" if self._selected_feature_key is None else self._selected_feature_key
        job = ClassifierJob(
            job_id=job_id,
            feature_key=feature_key,
            table_name=self._selected_table_name,
            summary=summary,
        )
        if not summary.eligible or self._selected_feature_key is None or self._selected_table_metadata is None:
            return job

        # Only the worker reads feature values. Here, on the main thread, only
        # `obs` is read: the labels and the region of each row.
        worker_input = ClassifierWorkerInput.from_summary(
            summary,
            table,
            feature_key=feature_key,
            region_key=self._selected_table_metadata.region_key,
        )
        return replace(job, worker_input=worker_input)

    def _apply_ineligible_state(
        self,
        *,
        feature_key: str,
        summary: ClassifierPreparationSummary,
        reason: str,
    ) -> None:
        """Clear the scope's predictions and record why the classifier is not trained.

        `summary` gives the scopes. `reason` comes from the summary when it
        blocks Train before a job, or from a worker that found too few usable
        rows (`ClassifierIneligibleResult`).
        """
        table = self._get_bound_table()
        if table is None:
            self._set_status(f"Classifier: {reason}", kind="warning")
            return

        prediction_state_change = self._clear_predictions_for_prediction_regions(
            table,
            summary.prediction_scope.regions,
        )

        table.uns[CLASSIFIER_CONFIG_KEY] = self._build_classifier_config(
            feature_key=feature_key,
            summary=summary,
            training=None,
            reason=reason,
        )
        self._clear_model_snapshot("the latest classifier inputs are not trainable")
        self._notify_prediction_table_state_changed(
            summary.prediction_scope.regions,
            prediction_state_change,
        )
        self._is_dirty = True
        self._set_status(f"Classifier: {reason}", kind="warning")

    def _is_current_job(self, job: ClassifierJob) -> bool:
        """Return whether `job` is the latest requested job and its worker is still the active one."""
        return job.job_id == self._latest_requested_job_id and job.job_id == self._active_worker_job_id

    def _on_worker_returned(self, job: ClassifierJob, result: ClassifierWorkerResult) -> None:
        if self._is_shutdown or not self._is_current_job(job):
            return

        table = self._get_bound_table()
        if table is None:
            return

        if isinstance(result, ClassifierIneligibleResult):
            self._apply_ineligible_state(feature_key=job.feature_key, summary=job.summary, reason=result.reason)
            return

        # Every row of the scope is cleared; only the rows with finite features
        # get a prediction.
        apply_result = _classifier_core._write_classifier_predictions(
            table,
            table_name=job.table_name,
            feature_key=job.feature_key,
            prediction_regions=job.prediction_scope.regions,
            raw_prediction_table_row_positions=job.prediction_scope.table_row_positions,
            prediction_table_row_positions=result.predicted_row_positions,
            pred_classes=result.pred_classes,
            pred_confidences=result.pred_confidences,
        )
        classifier_config = self._build_classifier_config(
            feature_key=job.feature_key,
            summary=job.summary,
            training=result,
        )
        table.uns[CLASSIFIER_CONFIG_KEY] = classifier_config
        self._store_model_snapshot(table, job.feature_key, result, classifier_config)
        self._notify_prediction_table_state_changed(
            job.prediction_scope.regions,
            apply_result.prediction_state_change,
        )
        self._is_dirty = False
        self._set_status(_trained_status_message(result, apply_result.n_predicted_rows), kind="success")

    def _on_worker_errored(self, job: ClassifierJob, error: Exception) -> None:
        if self._is_shutdown or not self._is_current_job(job):
            return

        table = self._get_bound_table()
        if table is not None:
            table.uns[CLASSIFIER_CONFIG_KEY] = self._build_classifier_config(
                feature_key=job.feature_key,
                summary=job.summary,
                training=None,
                reason=str(error),
            )
            self._notify_table_state_changed(
                frozenset({TableComponentPath("uns", (CLASSIFIER_CONFIG_KEY,))}),
                regions=(),
                source="object_classification_metadata",
            )
        self._is_dirty = True
        self._clear_model_snapshot("classifier training failed")
        self._set_status(f"Classifier: training failed: {error}", kind="error")

    def _on_worker_finished(self, job: ClassifierJob) -> None:
        if self._is_shutdown:
            return

        if job.job_id != self._active_worker_job_id:
            return

        self._active_worker = None
        self._active_worker_job_id = None
        if self._on_state_changed is not None:
            self._on_state_changed()

    def _cancel_pending_and_active_jobs(self) -> None:
        self._debounce_timer.stop()
        self._cancel_active_worker()

    def _invalidate_async_jobs(self) -> None:
        self._latest_requested_job_id += 1
        self._cancel_pending_and_active_jobs()

    def _cancel_active_worker(self) -> None:
        if self._active_worker is None:
            return

        quit_worker = getattr(self._active_worker, "quit", None)
        if callable(quit_worker):
            quit_worker()
        self._active_worker = None
        self._active_worker_job_id = None

    def _create_training_worker(self, job: ClassifierJob) -> Any:
        return _run_classifier_job(job.worker_input)

    def _store_model_snapshot(
        self,
        table: AnnData,
        feature_key: str,
        result: ClassifierTrainingResult,
        classifier_config: dict[str, object],
    ) -> None:
        if result.estimator is None:
            self._clear_model_snapshot("the latest training result did not include a fitted estimator")
            return

        try:
            feature_metadata = _get_feature_metadata(table, feature_key)
            feature_columns = normalize_feature_columns(feature_metadata)
            source_kind = normalize_feature_matrix_source_kind(feature_metadata)
            # Export relies on `.uns["feature_matrices"]` for column order, so
            # ensure that metadata still matches the live `.obsm` matrix shape.
            _classifier_core._validate_current_feature_matrix_matches_columns(
                table,
                feature_key,
                feature_columns,
                custom_obsm=source_kind == CUSTOM_OBSM_SOURCE_KIND,
            )
        except ValueError as error:
            self._clear_model_snapshot(str(error))
            return

        self._model_snapshot = ClassifierModelSnapshot(
            estimator=result.estimator,
            classifier_config=deepcopy(classifier_config),
            feature_metadata=deepcopy(feature_metadata),
            feature_key=feature_key,
            trained_at=result.trained_at,
        )
        self._model_snapshot_missing_reason = None

    def _clear_model_snapshot(self, reason: str | None = None) -> None:
        self._model_snapshot = None
        self._model_snapshot_missing_reason = reason

    def _validate_export_ready(self) -> None:
        table = self._get_bound_table()
        if table is None:
            raise ValueError("Choose an annotation table before exporting a classifier.")
        if self._selected_feature_key is None:
            raise ValueError("Choose a feature matrix before exporting a classifier.")
        if self.is_training:
            raise ValueError("A classifier training job is currently running.")
        if self._is_dirty:
            raise ValueError("The classifier model is stale. Train it again before exporting.")
        if self._model_snapshot is None:
            if self._model_snapshot_missing_reason:
                raise ValueError(f"Classifier export is unavailable because {self._model_snapshot_missing_reason}")
            raise ValueError("No fitted classifier model is available to export. Train the classifier first.")

        snapshot = self._model_snapshot
        if self._selected_feature_key != snapshot.feature_key:
            raise ValueError(
                f'The selected feature matrix "{self._selected_feature_key}" does not match the fitted model '
                f'feature matrix "{snapshot.feature_key}".'
            )
        current_feature_metadata = _get_feature_metadata(table, snapshot.feature_key)
        current_feature_columns = normalize_feature_columns(current_feature_metadata)
        current_source_kind = normalize_feature_matrix_source_kind(current_feature_metadata)
        if current_feature_columns != snapshot.feature_columns:
            raise ValueError(
                "Current feature metadata no longer matches the fitted model snapshot. "
                "Train the classifier again before exporting."
            )
        # Export relies on `.uns["feature_matrices"]` for column order, so
        # ensure that metadata still matches the live `.obsm` matrix shape.
        _classifier_core._validate_current_feature_matrix_matches_columns(
            table,
            snapshot.feature_key,
            snapshot.feature_columns,
            custom_obsm=current_source_kind == CUSTOM_OBSM_SOURCE_KIND,
        )

    def _get_bound_table(self) -> AnnData | None:
        if self._selected_spatialdata is None or self._selected_table_name is None:
            return None

        return get_table(self._selected_spatialdata, self._selected_table_name)

    def _clear_predictions_for_prediction_regions(
        self,
        table: AnnData,
        prediction_regions: tuple[str, ...],
    ) -> PredictionStateChange:
        if self._selected_table_metadata is None:
            return PredictionStateChange(
                pred_class_changed=False,
                pred_confidence_changed=False,
                palette_changed=False,
            )

        prediction_table_row_positions = _resolve_region_row_positions(
            table.obs,
            self._selected_table_metadata.region_key,
            prediction_regions,
        )
        return _classifier_core._clear_predictions_for_row_positions(
            table,
            prediction_table_row_positions,
        )

    def _count_predicted_rows(self, table: AnnData, prediction_regions: tuple[str, ...]) -> int:
        """Return how many rows of `prediction_regions` hold a prediction."""
        if self._selected_table_metadata is None or PRED_CLASS_COLUMN not in table.obs:
            return 0
        positions = _resolve_region_row_positions(
            table.obs, self._selected_table_metadata.region_key, prediction_regions
        )
        return int(table.obs[PRED_CLASS_COLUMN].iloc[positions].notna().sum())

    def _build_classifier_config(
        self,
        *,
        feature_key: str | None,
        summary: ClassifierPreparationSummary,
        training: ClassifierTrainingResult | None,
        reason: str | None = None,
    ) -> dict[str, object]:
        """Return `uns["classifier_config"]`.

        It holds what the reload status reads, and how the model behind the
        predictions was obtained: the table holds no estimator. `training` is
        the trained model's result, or `None` when nothing was trained, with
        `reason` saying why. `summary` gives the scopes and the column count.
        `training_timestamp`, `n_training_labels` (the labeled rows the model
        was trained on) and `class_labels_seen` describe the trained model:
        `None`, 0 and `[]` when nothing was trained.
        """
        return {
            "trained": training is not None,
            "reason": None if training is not None else reason,
            "feature_key": feature_key,
            "prediction_scope": summary.prediction_scope.mode,
            "prediction_regions": list(summary.prediction_scope.regions),
            "model_type": "RandomForestClassifier",
            "rf_params": dict(RANDOM_FOREST_PARAMS),
            "training_scope": summary.training_scope.mode,
            "training_regions": list(summary.training_scope.regions),
            "n_features": summary.n_features,
            "training_timestamp": None if training is None else training.trained_at,
            "n_training_labels": 0 if training is None else training.labeled_count,
            "class_labels_seen": [] if training is None else list(training.class_labels),
        }

    def _set_status(self, message: str, *, kind: str) -> None:
        self._status_message = message
        self._status_kind = kind
        if self._is_shutdown:
            return

        if self._on_state_changed is not None:
            self._on_state_changed()

    def _notify_table_state_changed(
        self,
        paths: frozenset[TableComponentPath],
        *,
        regions: tuple[str, ...],
        source: ClassifierTableStateChangeSource,
    ) -> None:
        if self._is_shutdown:
            return

        if self._on_table_state_changed is not None:
            self._on_table_state_changed(
                ClassifierTableStateChange(
                    paths=paths,
                    regions=regions,
                    source=source,
                )
            )

    def _notify_prediction_table_state_changed(
        self,
        regions: tuple[str, ...],
        prediction_state_change: PredictionStateChange,
    ) -> None:
        if self._is_shutdown:
            return

        changed_paths = {
            TableComponentPath("uns", (CLASSIFIER_CONFIG_KEY,)),
        }
        if prediction_state_change.pred_class_changed:
            changed_paths.add(TableComponentPath("obs", (PRED_CLASS_COLUMN,)))
        if prediction_state_change.pred_confidence_changed:
            changed_paths.add(TableComponentPath("obs", (PRED_CONFIDENCE_COLUMN,)))
        if prediction_state_change.palette_changed:
            changed_paths.add(TableComponentPath("uns", (PRED_CLASS_COLORS_KEY,)))

        self._notify_table_state_changed(
            frozenset(changed_paths),
            regions=regions,
            source="object_classification_inference",
        )
        if prediction_state_change.changed and self._on_prediction_state_changed is not None:
            self._on_prediction_state_changed()

    def _update_idle_status(self, *, reason: str | None = None) -> None:
        if self._get_bound_table() is None:
            self._set_status("Classifier: choose an annotation table and feature matrix.", kind="warning")
            return

        if self._selected_feature_key is None:
            self._set_status("Classifier: choose a feature matrix to enable training.", kind="warning")
            return

        if self.is_training:
            return

        if self._is_dirty:
            if reason is None:
                self._set_status(
                    "Classifier: model is stale. Click Train Classifier to refresh predictions.",
                    kind="warning",
                )
            else:
                self._set_status(f"Classifier: model is stale because {reason}", kind="warning")
            return

        self._set_status("Classifier: model is up to date.", kind="success")

    def _update_status_from_reloaded_table(self, table: AnnData) -> None:
        if self._selected_feature_key is None:
            self._set_status("Classifier: choose a feature matrix to enable training.", kind="warning")
            return

        config = table.uns.get(CLASSIFIER_CONFIG_KEY)
        if not isinstance(config, Mapping):
            self._update_idle_status()
            return

        mismatches: list[str] = []
        config_feature_key = config.get("feature_key")
        if isinstance(config_feature_key, str) and config_feature_key != self._selected_feature_key:
            mismatches.append(
                f'loaded predictions use feature matrix "{config_feature_key}" but "{self._selected_feature_key}" is selected'
            )

        config_prediction_scope = config.get("prediction_scope")
        if not isinstance(config_prediction_scope, str):
            mismatches.append("loaded predictions do not describe their prediction scope")
        elif config_prediction_scope != self._selected_prediction_scope:
            mismatches.append(
                f'loaded predictions use prediction scope "{config_prediction_scope}" but "{self._selected_prediction_scope}" is selected'
            )

        config_prediction_regions = _normalize_prediction_regions(config.get("prediction_regions"))
        if self._selected_labels_name is not None and self._selected_labels_name not in config_prediction_regions:
            mismatches.append(f'loaded predictions do not cover labels element "{self._selected_labels_name}"')

        if mismatches:
            self._is_dirty = True
            self._set_status(f"Classifier: model is stale because {'; '.join(mismatches)}.", kind="warning")
            return

        trained = config.get("trained")
        if trained is True:
            # Counted from the table, which holds the predictions themselves.
            predicted_count = self._count_predicted_rows(table, config_prediction_regions)
            self._set_status(
                f"Classifier: model is up to date. Loaded predictions for {predicted_count} objects from disk.",
                kind="success",
            )
            return

        reason = config.get("reason")
        if trained is False and isinstance(reason, str) and reason:
            self._set_status(f"Classifier: {reason}", kind="warning")
            return

        self._update_idle_status()

    def _resolve_classifier_scopes(
        self,
        table: AnnData,
        metadata: SpatialDataTableMetadata,
    ) -> ResolvedClassifierScopes:
        def resolve_one_scope(scope_mode: ClassifierScopeMode) -> ResolvedClassifierScope:
            if scope_mode == "selected_segmentation_only":
                regions: tuple[str, ...] = () if self._selected_labels_name is None else (self._selected_labels_name,)
            elif scope_mode == "all":
                regions = metadata.regions
            else:  # pragma: no cover - guarded by _normalize_scope_mode
                raise ValueError(f"Unsupported classifier scope mode: {scope_mode!r}")

            return ResolvedClassifierScope(
                mode=scope_mode,
                regions=regions,
                table_row_positions=_resolve_region_row_positions(table.obs, metadata.region_key, regions),
            )

        return ResolvedClassifierScopes(
            training=resolve_one_scope(self._selected_training_scope),
            prediction=resolve_one_scope(self._selected_prediction_scope),
        )


def _normalize_scope_mode(scope_mode: ClassifierScopeMode | str) -> ClassifierScopeMode:
    if scope_mode in ("selected_segmentation_only", "all"):
        return scope_mode
    raise ValueError(f"Unsupported classifier scope mode: {scope_mode!r}")


def _trained_status_message(result: ClassifierTrainingResult, n_predicted_rows: int) -> str:
    message = f"Classifier: model is up to date. Updated predictions for {n_predicted_rows} objects."
    if result.n_skipped_prediction_rows:
        message += (
            f" Skipped {result.n_skipped_prediction_rows} row(s) without finite features, "
            "which get no prediction."
            f"{_regions_without_features_note(result.prediction_regions_without_features)}"
        )
    if result.n_skipped_labeled_rows:
        message += (
            f" {result.n_skipped_labeled_rows} labeled row(s) without finite features were not used for training."
        )
    return message


def _get_user_class_values(obs: pd.DataFrame) -> pd.Series:
    if USER_CLASS_COLUMN not in obs:
        return pd.Series(pd.NA, index=obs.index, dtype="Int64", name=USER_CLASS_COLUMN)

    return normalize_class_values(obs[USER_CLASS_COLUMN], column_name=USER_CLASS_COLUMN)
