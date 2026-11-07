from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from spatiato.core.object_classification.classifier_export import (
    CLASSIFIER_EXPORT_BUNDLE_TYPE,
    CLASSIFIER_EXPORT_SCHEMA_VERSION,
    ClassifierExportBundle,
    read_classifier_export_bundle,
)

_MISSING = object()


def _make_bundle(*, source_channels: object = _MISSING) -> ClassifierExportBundle:
    feature_columns = ["mean__CD3", "mean__CD8a"]
    estimator = RandomForestClassifier(n_estimators=5, random_state=0, n_jobs=1)
    estimator.fit(
        np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64),
        np.array([1, 2], dtype=np.int64),
    )
    source_feature_metadata: dict[str, object] = {
        "feature_columns": feature_columns,
        "features": ["mean"],
    }
    if source_channels is not _MISSING:
        source_feature_metadata["source_channels"] = source_channels

    return ClassifierExportBundle(
        schema_version=CLASSIFIER_EXPORT_SCHEMA_VERSION,
        created_at="2026-05-05T09:05:00+00:00",
        spatiato_version="0.0.0-test",
        sklearn_version=None,
        estimator=estimator,
        model_provenance={
            "feature_key": "features_cd3_cd8a",
            "model_type": "RandomForestClassifier",
            "rf_params": {"n_estimators": 5, "random_state": 0, "n_jobs": 1},
            "training_scope": "all",
            "training_regions": ["labels"],
            "n_features": 2,
            "training_timestamp": "2026-05-05T09:00:00+00:00",
            "n_training_labels": 2,
            "class_labels_seen": [1, 2],
        },
        source_feature_metadata=source_feature_metadata,
    )


def test_read_classifier_export_bundle_refuses_an_older_schema_version(tmp_path: Path) -> None:
    path = tmp_path / "old.spatiato-classifier.joblib"
    # Version 1 named the provenance `source_classifier_config`; the version is checked first.
    joblib.dump(
        {
            "bundle_type": CLASSIFIER_EXPORT_BUNDLE_TYPE,
            "schema_version": 1,
            "source_classifier_config": {"model_type": "RandomForestClassifier"},
        },
        path,
    )

    with pytest.raises(ValueError, match="Unsupported classifier export schema version: 1. Expected 2."):
        read_classifier_export_bundle(path)


def test_classifier_export_bundle_source_channels_missing_is_none() -> None:
    assert _make_bundle().source_channels is None
    assert _make_bundle(source_channels=None).source_channels is None


def test_classifier_export_bundle_source_channels_casts_sequence_to_tuple() -> None:
    bundle = _make_bundle(source_channels=["CD3", "CD8a"])

    assert bundle.source_channels == ("CD3", "CD8a")


def test_classifier_export_bundle_source_channels_casts_numpy_array() -> None:
    bundle = _make_bundle(source_channels=np.array(["CD3", "CD8a"]))

    assert bundle.source_channels == ("CD3", "CD8a")


def test_classifier_export_bundle_source_channels_rejects_non_sequence() -> None:
    with pytest.raises(ValueError, match="sequence"):
        _ = _make_bundle(source_channels="CD3").source_channels


def test_classifier_export_bundle_source_channels_rejects_multidimensional_array() -> None:
    with pytest.raises(ValueError, match="1-dimensional"):
        _ = _make_bundle(source_channels=np.array([["CD3", "CD8a"]])).source_channels
