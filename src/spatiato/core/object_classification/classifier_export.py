from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import joblib
from sklearn import __version__ as sklearn_version
from sklearn.ensemble import RandomForestClassifier

from spatiato import __version__ as spatiato_version
from spatiato.core.feature_matrix_metadata import normalize_feature_columns

CLASSIFIER_EXPORT_BUNDLE_TYPE = "spatiato_classifier"
CLASSIFIER_EXPORT_SCHEMA_VERSION = 1
DEFAULT_CLASSIFIER_EXPORT_SUFFIX = ".spatiato-classifier.joblib"


@dataclass(frozen=True)
class ClassifierExportBundle:
    """A fitted classifier plus the source feature schema needed to reuse it."""

    schema_version: int
    created_at: str
    spatiato_version: str | None
    sklearn_version: str | None
    estimator: RandomForestClassifier
    source_classifier_config: dict[str, object]
    source_feature_metadata: dict[str, object]

    @property
    def model_type(self) -> str:
        """Return the classifier model type recorded at training time."""
        return normalize_required_str(self.source_classifier_config, "model_type")

    @property
    def feature_key(self) -> str:
        """Return the source feature matrix key used for training."""
        return normalize_required_str(self.source_classifier_config, "feature_key")

    @property
    def class_labels_seen(self) -> tuple[int, ...]:
        """Return the class labels observed during training."""
        return normalize_int_tuple(self.source_classifier_config, "class_labels_seen")

    @property
    def feature_columns(self) -> tuple[str, ...]:
        """Return the exact feature column schema expected by the estimator."""
        return normalize_feature_columns(self.source_feature_metadata)

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Return the requested Harpy feature names used to create the matrix."""
        return normalize_feature_names(self.source_feature_metadata)

    @property
    def source_channels(self) -> tuple[str, ...] | None:
        """Return the channel selection captured during source feature extraction."""
        return normalize_source_channels(self.source_feature_metadata)

    @property
    def source_kind(self) -> str:
        """Return the source kind captured in the feature metadata."""
        from spatiato.core.feature_matrix_metadata import normalize_feature_matrix_source_kind

        return normalize_feature_matrix_source_kind(self.source_feature_metadata)

    @property
    def n_features(self) -> int:
        """Return the number of feature columns expected by the estimator."""
        return len(self.feature_columns)


@dataclass(frozen=True)
class ClassifierModelSnapshot:
    """In-memory fitted model snapshot owned by the widget controller."""

    estimator: RandomForestClassifier
    classifier_config: dict[str, object]
    feature_metadata: dict[str, object]
    feature_key: str
    trained_at: str

    @property
    def feature_columns(self) -> tuple[str, ...]:
        """Return the exact feature column schema captured at training time."""
        return normalize_feature_columns(self.feature_metadata)


def build_classifier_export_bundle(
    snapshot: ClassifierModelSnapshot,
    *,
    created_at: str | None = None,
) -> ClassifierExportBundle:
    """Build a serializable export bundle from an in-memory classifier snapshot."""
    bundle = ClassifierExportBundle(
        schema_version=CLASSIFIER_EXPORT_SCHEMA_VERSION,
        created_at=datetime.now(UTC).isoformat() if created_at is None else created_at,
        spatiato_version=spatiato_version,
        sklearn_version=sklearn_version,
        estimator=snapshot.estimator,
        source_classifier_config=deepcopy(snapshot.classifier_config),
        source_feature_metadata=deepcopy(snapshot.feature_metadata),
    )
    _validate_export_bundle(bundle)
    return bundle


def write_classifier_export_bundle(path: str | Path, bundle: ClassifierExportBundle) -> None:
    """Write a trusted joblib-backed classifier artifact to ``path``."""
    _validate_export_bundle(bundle)
    joblib.dump(_bundle_to_payload(bundle), Path(path))


def read_classifier_export_bundle(path: str | Path) -> ClassifierExportBundle:
    """Read and validate a trusted joblib-backed classifier artifact.

    Joblib uses pickle semantics. Only load artifacts from trusted sources.
    """
    payload = joblib.load(Path(path))
    if not isinstance(payload, Mapping):
        raise ValueError("Classifier artifact payload must be a mapping.")
    return _payload_to_bundle(payload)


def normalize_required_str(mapping: Mapping[str, object], key: str) -> str:
    """Return a required non-empty string value from a metadata mapping."""
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Classifier export metadata is missing required string `{key}`.")
    return value


def normalize_int_tuple(mapping: Mapping[str, object], key: str) -> tuple[int, ...]:
    """Return a tuple of integer values from a metadata mapping."""
    return tuple(int(value) for value in _normalize_sequence(mapping, key))


def normalize_feature_names(feature_metadata: Mapping[str, object]) -> tuple[str, ...]:
    """Return the requested feature names from Harpy feature metadata."""
    names = tuple(str(value) for value in _normalize_sequence(feature_metadata, "features"))
    if not names:
        raise ValueError("Feature metadata must contain at least one `features` entry.")
    if any(not name for name in names):
        raise ValueError("Feature metadata `features` must not contain empty strings.")
    return names


def normalize_source_channels(feature_metadata: Mapping[str, object]) -> tuple[str, ...] | None:
    """Return the optional source channel selection from Harpy feature metadata.

    Harpy stores `source_channels` as a flat sequence of channel names shared
    across all labels/image pairs, or `None` for morphology-only matrices. The
    sequence is cast to a tuple of strings as-is; Harpy owns channel validation.
    """
    channels = _normalize_sequence(feature_metadata, "source_channels", optional=True)
    if channels is None:
        return None
    return tuple(str(channel) for channel in channels)


def _normalize_sequence(
    mapping: Mapping[str, object], key: str, *, optional: bool = False
) -> tuple[object, ...] | None:
    """Coerce a metadata value into a tuple of items.

    Array-likes (e.g. numpy arrays) are coerced to lists first, and must be
    1-dimensional. When `optional` is set, a missing value yields `None`;
    otherwise a missing or non-sequence value is an error.
    """
    raw_value = mapping.get(key)
    ndim = getattr(raw_value, "ndim", None)
    if ndim is not None and ndim != 1:
        raise ValueError(f"Classifier export metadata `{key}` must be a 1-dimensional sequence.")
    value = _coerce_metadata_sequence(raw_value)
    if optional and value is None:
        return None
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError(f"Classifier export metadata `{key}` must be a sequence.")
    return tuple(value)


def _coerce_metadata_sequence(value: object) -> object:
    if hasattr(value, "tolist") and not isinstance(value, str):
        return value.tolist()
    return value


def _validate_export_bundle(bundle: ClassifierExportBundle) -> None:
    if bundle.schema_version != CLASSIFIER_EXPORT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported classifier export schema version: {bundle.schema_version!r}. "
            f"Expected {CLASSIFIER_EXPORT_SCHEMA_VERSION}."
        )
    if not isinstance(bundle.estimator, RandomForestClassifier):
        raise ValueError("Classifier export bundle estimator must be a RandomForestClassifier.")
    if bundle.model_type != "RandomForestClassifier":
        raise ValueError(f"Unsupported classifier model type: {bundle.model_type!r}.")
    _ = bundle.source_kind
    if bundle.n_features <= 0:
        raise ValueError("Classifier export bundle must describe at least one feature column.")
    estimator_feature_count = getattr(bundle.estimator, "n_features_in_", None)
    if estimator_feature_count is not None and int(estimator_feature_count) != bundle.n_features:
        raise ValueError(
            "Classifier export bundle feature schema does not match the fitted estimator: "
            f"metadata has {bundle.n_features} column(s), estimator expects {int(estimator_feature_count)}."
        )


def _bundle_to_payload(bundle: ClassifierExportBundle) -> dict[str, object]:
    return {
        "bundle_type": CLASSIFIER_EXPORT_BUNDLE_TYPE,
        "schema_version": bundle.schema_version,
        "created_at": bundle.created_at,
        "spatiato_version": bundle.spatiato_version,
        "sklearn_version": bundle.sklearn_version,
        "source_classifier_config": deepcopy(bundle.source_classifier_config),
        "source_feature_metadata": deepcopy(bundle.source_feature_metadata),
        "estimator": bundle.estimator,
    }


def _payload_to_bundle(payload: Mapping[str, object]) -> ClassifierExportBundle:
    bundle_type = payload.get("bundle_type")
    if bundle_type != CLASSIFIER_EXPORT_BUNDLE_TYPE:
        raise ValueError(f"Unsupported classifier artifact type: {bundle_type!r}.")

    source_classifier_config = payload.get("source_classifier_config")
    if not isinstance(source_classifier_config, Mapping):
        raise ValueError("Classifier artifact is missing `source_classifier_config` metadata.")
    source_feature_metadata = payload.get("source_feature_metadata")
    if not isinstance(source_feature_metadata, Mapping):
        raise ValueError("Classifier artifact is missing `source_feature_metadata` metadata.")

    schema_version = payload.get("schema_version")
    if not isinstance(schema_version, int):
        raise ValueError("Classifier artifact is missing integer `schema_version`.")
    created_at = payload.get("created_at")
    if not isinstance(created_at, str) or not created_at:
        raise ValueError("Classifier artifact is missing string `created_at`.")
    estimator = payload.get("estimator")
    if not isinstance(estimator, RandomForestClassifier):
        raise ValueError("Classifier artifact estimator must be a RandomForestClassifier.")

    napari_version = payload.get("spatiato_version")
    if napari_version is not None and not isinstance(napari_version, str):
        raise ValueError("Classifier artifact `spatiato_version` must be a string or None.")
    sklearn_artifact_version = payload.get("sklearn_version")
    if sklearn_artifact_version is not None and not isinstance(sklearn_artifact_version, str):
        raise ValueError("Classifier artifact `sklearn_version` must be a string or None.")

    bundle = ClassifierExportBundle(
        schema_version=schema_version,
        created_at=created_at,
        spatiato_version=napari_version,
        sklearn_version=sklearn_artifact_version,
        estimator=estimator,
        source_classifier_config=dict(source_classifier_config),
        source_feature_metadata=dict(source_feature_metadata),
    )
    _validate_export_bundle(bundle)
    return bundle
