from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from spatialdata import SpatialData, read_zarr
from spatialdata.transformations import Identity, set_transformation

from spatiato import headless
from spatiato.core.feature_matrix_metadata import (
    HARPY_ADD_FEATURE_MATRIX_SOURCE_KIND,
    register_feature_matrix_metadata,
)
from spatiato.core.object_classification.classifier import (
    CLASSIFIER_APPLY_CONFIG_KEY,
    PRED_CLASS_COLUMN,
    PRED_CONFIDENCE_COLUMN,
)
from spatiato.core.object_classification.classifier_export import (
    CLASSIFIER_EXPORT_SCHEMA_VERSION,
    ClassifierExportBundle,
    write_classifier_export_bundle,
)

_FEATURE_MATRICES_KEY = "feature_matrices"
_MISSING = object()


def _set_deterministic_features(
    sdata: SpatialData,
    *,
    table_name: str = "table",
    feature_key: str = "features_1",
) -> None:
    table = sdata[table_name]
    instance_ids = table.obs["instance_id"].to_numpy(dtype=np.int64)
    first_feature = (instance_ids > 13).astype(np.float64)
    second_feature = instance_ids.astype(np.float64) / instance_ids.max()
    table.obsm[feature_key] = np.column_stack([first_feature, second_feature])


def _set_feature_metadata(
    sdata: SpatialData,
    *,
    table_name: str = "table",
    feature_key: str = "features_1",
    feature_columns: tuple[str, ...] = ("is_large", "instance_fraction"),
    features: tuple[str, ...] = ("synthetic_size", "synthetic_position"),
    source_image: object = None,
    source_channels: object = _MISSING,
) -> None:
    table = sdata[table_name]
    metadata = {
        "feature_columns": list(feature_columns),
        "schema_version": 1,
        "backend": "numpy",
        "dtype": str(np.asarray(table.obsm[feature_key]).dtype),
        "source_label": "blobs_labels",
        "source_image": source_image,
        "coordinate_system": "global",
        "features": list(features),
        "source_kind": HARPY_ADD_FEATURE_MATRIX_SOURCE_KIND,
    }
    if source_channels is not _MISSING:
        metadata["source_channels"] = source_channels
    table.uns.setdefault(_FEATURE_MATRICES_KEY, {})[feature_key] = metadata


def _make_classifier_bundle(
    sdata: SpatialData,
    *,
    table_name: str = "table",
    feature_key: str = "features_1",
) -> ClassifierExportBundle:
    table = sdata[table_name]
    feature_matrix = np.asarray(table.obsm[feature_key], dtype=np.float64)
    train_labels = np.where(feature_matrix[:, 0] > 0.5, 2, 1).astype(np.int64)
    classifier = RandomForestClassifier(n_estimators=25, random_state=0, n_jobs=1)
    classifier.fit(feature_matrix, train_labels)
    regions = _table_regions(table)
    classifier_config = {
        "model_type": "RandomForestClassifier",
        "feature_key": feature_key,
        "table_name": table_name,
        "roi_mode": "none",
        "trained": True,
        "eligible": True,
        "reason": "Ready to train.",
        "training_timestamp": "2026-05-05T09:00:00+00:00",
        "n_labeled_objects": int(table.n_obs),
        "n_features": int(feature_matrix.shape[1]),
        "class_labels_seen": [1, 2],
        "rf_params": {"n_estimators": 25, "random_state": 0, "n_jobs": 1},
        "training_scope": "all",
        "training_regions": list(regions),
        "n_training_rows": int(table.n_obs),
        "prediction_scope": "all",
        "prediction_regions": list(regions),
        "n_predicted_rows": int(table.n_obs),
    }
    return ClassifierExportBundle(
        schema_version=CLASSIFIER_EXPORT_SCHEMA_VERSION,
        created_at="2026-05-05T09:05:00+00:00",
        spatiato_version="0.0.0-test",
        sklearn_version=None,
        estimator=classifier,
        source_classifier_config=classifier_config,
        source_feature_metadata=dict(table.uns[_FEATURE_MATRICES_KEY][feature_key]),
    )


def _make_custom_obsm_classifier_bundle(sdata: SpatialData) -> ClassifierExportBundle:
    _set_deterministic_features(sdata)
    register_feature_matrix_metadata(
        sdata["table"],
        "features_1",
        feature_columns=("is_large", "instance_fraction"),
    )
    return _make_classifier_bundle(sdata)


def _make_area_classifier_bundle() -> ClassifierExportBundle:
    classifier = RandomForestClassifier(n_estimators=10, random_state=0, n_jobs=1)
    classifier.fit(np.array([[0.0], [100.0]], dtype=np.float64), np.array([1, 2], dtype=np.int64))
    return ClassifierExportBundle(
        schema_version=CLASSIFIER_EXPORT_SCHEMA_VERSION,
        created_at="2026-05-05T09:05:00+00:00",
        spatiato_version="0.0.0-test",
        sklearn_version=None,
        estimator=classifier,
        source_classifier_config={
            "model_type": "RandomForestClassifier",
            "feature_key": "area_features",
            "table_name": "source_table",
            "roi_mode": "none",
            "trained": True,
            "eligible": True,
            "reason": "Ready to train.",
            "training_timestamp": "2026-05-05T09:00:00+00:00",
            "n_labeled_objects": 2,
            "n_features": 1,
            "class_labels_seen": [1, 2],
            "rf_params": {"n_estimators": 10, "random_state": 0, "n_jobs": 1},
            "training_scope": "all",
            "training_regions": ["source_labels"],
            "n_training_rows": 2,
            "prediction_scope": "all",
            "prediction_regions": ["source_labels"],
            "n_predicted_rows": 2,
        },
        source_feature_metadata={
            "feature_columns": ["area"],
            "schema_version": 1,
            "backend": "numpy",
            "dtype": "float64",
            "source_label": "source_labels",
            "source_image": None,
            "source_channels": None,
            "coordinate_system": "global",
            "features": ["area"],
            "source_kind": HARPY_ADD_FEATURE_MATRIX_SOURCE_KIND,
        },
    )


def _table_regions(table) -> tuple[str, ...]:
    regions = table.obs["region"]
    if isinstance(regions.dtype, pd.CategoricalDtype):
        return tuple(str(region) for region in regions.cat.categories)
    return tuple(str(region) for region in pd.unique(regions))


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


def test_apply_classifier_from_path_writes_predictions_and_apply_config(
    tmp_path: Path,
    sdata_blobs: SpatialData,
) -> None:
    _set_deterministic_features(sdata_blobs)
    _set_feature_metadata(sdata_blobs)
    bundle = _make_classifier_bundle(sdata_blobs)
    classifier_path = tmp_path / "classifier.spatiato-classifier.joblib"
    write_classifier_export_bundle(classifier_path, bundle)

    result = headless.apply_classifier_from_path(sdata_blobs, classifier_path, table_name="table")

    table = sdata_blobs["table"]
    feature_matrix = np.asarray(table.obsm["features_1"], dtype=np.float64)
    expected_classes = bundle.estimator.predict(feature_matrix)
    expected_confidences = bundle.estimator.predict_proba(feature_matrix).max(axis=1)
    config = table.uns[CLASSIFIER_APPLY_CONFIG_KEY]

    assert result.table_name == "table"
    assert result.feature_key == "features_1"
    assert result.prediction_regions == ("blobs_labels",)
    assert result.n_predicted_rows == table.n_obs
    assert result.n_skipped_feature_invalid_rows == 0
    assert table.obs[PRED_CLASS_COLUMN].astype("string").astype(int).to_numpy().tolist() == expected_classes.tolist()
    assert np.allclose(table.obs[PRED_CONFIDENCE_COLUMN].to_numpy(dtype=np.float64), expected_confidences)
    assert config["applied"] is True
    assert config["classifier_path"] == str(classifier_path)
    assert config["source_classifier_config"] == bundle.source_classifier_config
    assert config["table_name"] == "table"
    assert config["feature_key"] == "features_1"
    assert config["prediction_regions"] == ["blobs_labels"]
    assert config["pred_class_column"] == PRED_CLASS_COLUMN
    assert config["pred_confidence_column"] == PRED_CONFIDENCE_COLUMN


def test_write_classifier_export_bundle_rejects_missing_source_kind(tmp_path: Path) -> None:
    bundle = _make_area_classifier_bundle()
    source_feature_metadata = dict(bundle.source_feature_metadata)
    del source_feature_metadata["source_kind"]
    invalid_bundle = replace(bundle, source_feature_metadata=source_feature_metadata)

    with pytest.raises(ValueError, match="source_kind"):
        write_classifier_export_bundle(tmp_path / "missing-source-kind.spatiato-classifier.joblib", invalid_bundle)


def test_apply_classifier_can_write_custom_prediction_columns(sdata_blobs: SpatialData) -> None:
    _set_deterministic_features(sdata_blobs)
    _set_feature_metadata(sdata_blobs)
    classifier = _make_classifier_bundle(sdata_blobs)

    result = headless.apply_classifier(
        sdata_blobs,
        classifier=classifier,
        table_name="table",
        pred_class_column="headless_class",
        pred_confidence_column="headless_confidence",
    )

    table = sdata_blobs["table"]
    assert result.pred_class_column == "headless_class"
    assert result.pred_confidence_column == "headless_confidence"
    assert "headless_class" in table.obs
    assert "headless_confidence" in table.obs
    assert table.uns[CLASSIFIER_APPLY_CONFIG_KEY]["classifier_path"] is None
    assert table.uns[CLASSIFIER_APPLY_CONFIG_KEY]["pred_class_column"] == "headless_class"


def test_apply_classifier_rejects_spatialdata_invalid_prediction_column_names(sdata_blobs: SpatialData) -> None:
    _set_deterministic_features(sdata_blobs)
    _set_feature_metadata(sdata_blobs)
    classifier = _make_classifier_bundle(sdata_blobs)

    with pytest.raises(ValueError, match="pred_class_column must be a valid SpatialData dataframe column name"):
        headless.apply_classifier(
            sdata_blobs,
            classifier=classifier,
            table_name="table",
            pred_class_column="headless class",
        )

    with pytest.raises(ValueError, match="Name cannot be '_index'"):
        headless.apply_classifier(
            sdata_blobs,
            classifier=classifier,
            table_name="table",
            pred_class_column="_index",
        )


def test_compute_features_for_classifier_uses_target_mapping(sdata_blobs: SpatialData) -> None:
    classifier = _make_area_classifier_bundle()

    resolved_target = headless.compute_features_for_classifier(
        sdata_blobs,
        classifier=classifier,
        target=headless.HeadlessFeatureTarget(
            table_name="table",
            feature_key="computed_area",
            triplets=(
                headless.FeatureExtractionTriplet(
                    coordinate_system="global",
                    labels_name="blobs_labels",
                    image_name=None,
                ),
            ),
        ),
    )

    assert resolved_target.table_name == "table"
    assert resolved_target.feature_key == "computed_area"
    assert len(resolved_target.triplets) == 1
    assert "computed_area" in sdata_blobs["table"].obsm
    metadata = sdata_blobs["table"].uns[_FEATURE_MATRICES_KEY]["computed_area"]
    assert list(metadata["feature_columns"]) == ["area"]
    assert list(metadata["source_label"]) == ["blobs_labels"]
    assert list(metadata["source_image"]) == [None]
    assert metadata["source_channels"] is None


def test_compute_features_for_classifier_rejects_explicit_triplet_channels(sdata_blobs: SpatialData) -> None:
    _set_deterministic_features(sdata_blobs)
    _set_feature_metadata(
        sdata_blobs,
        feature_columns=("mean__1", "mean__0"),
        features=("mean",),
        source_image="blobs_image",
        source_channels=["1", "0"],
    )
    classifier = _make_classifier_bundle(sdata_blobs)

    with pytest.raises(ValueError, match="does not accept explicit channels"):
        headless.compute_features_for_classifier(
            sdata_blobs,
            classifier=classifier,
            target=headless.HeadlessFeatureTarget(
                table_name="table",
                feature_key="computed_features",
                triplets=(
                    headless.FeatureExtractionTriplet(
                        coordinate_system="global",
                        labels_name="blobs_labels",
                        image_name="blobs_image",
                        channels=("0",),
                    ),
                ),
            ),
        )

    assert "computed_features" not in sdata_blobs["table"].obsm


def test_compute_features_for_classifier_rejects_custom_obsm_bundle(sdata_blobs: SpatialData) -> None:
    classifier = _make_custom_obsm_classifier_bundle(sdata_blobs)

    with pytest.raises(ValueError, match="headless.apply_classifier"):
        headless.compute_features_for_classifier(
            sdata_blobs,
            classifier=classifier,
            target=headless.HeadlessFeatureTarget(
                table_name="table",
                feature_key="computed_features",
                triplets=(
                    headless.FeatureExtractionTriplet(
                        coordinate_system="global",
                        labels_name="blobs_labels",
                        image_name=None,
                    ),
                ),
            ),
        )

    assert "computed_features" not in sdata_blobs["table"].obsm


def test_apply_classifier_with_feature_extraction_uses_classifier_source_channels(sdata_blobs: SpatialData) -> None:
    _set_deterministic_features(sdata_blobs)
    _set_feature_metadata(
        sdata_blobs,
        feature_columns=("mean__1", "mean__0"),
        features=("mean",),
        source_image="blobs_image",
        source_channels=["1", "0"],
    )
    classifier = _make_classifier_bundle(sdata_blobs)

    result = headless.apply_classifier_with_feature_extraction(
        sdata_blobs,
        classifier=classifier,
        table_name="table",
        labels_name="blobs_labels",
        feature_key="computed_features",
        coordinate_system="global",
        image_name="blobs_image",
    )

    assert result.feature_key == "computed_features"
    assert PRED_CLASS_COLUMN in sdata_blobs["table"].obs
    assert PRED_CONFIDENCE_COLUMN in sdata_blobs["table"].obs
    metadata = sdata_blobs["table"].uns[_FEATURE_MATRICES_KEY]["computed_features"]
    assert list(metadata["feature_columns"]) == ["mean__1", "mean__0"]
    assert list(metadata["source_channels"]) == ["1", "0"]


def test_apply_classifier_with_feature_extraction_rejects_custom_obsm_bundle(sdata_blobs: SpatialData) -> None:
    classifier = _make_custom_obsm_classifier_bundle(sdata_blobs)

    with pytest.raises(ValueError, match="register_feature_matrix_metadata"):
        headless.apply_classifier_with_feature_extraction(
            sdata_blobs,
            classifier=classifier,
            table_name="table",
            labels_name="blobs_labels",
            feature_key="computed_features",
            coordinate_system="global",
        )

    assert "computed_features" not in sdata_blobs["table"].obsm


def test_apply_classifier_accepts_custom_obsm_bundle_with_existing_compatible_matrix(
    sdata_blobs: SpatialData,
) -> None:
    classifier = _make_custom_obsm_classifier_bundle(sdata_blobs)

    result = headless.apply_classifier(sdata_blobs, classifier=classifier, table_name="table")

    assert result.feature_key == "features_1"
    assert result.n_predicted_rows == sdata_blobs["table"].n_obs
    assert PRED_CLASS_COLUMN in sdata_blobs["table"].obs


def test_apply_classifier_with_feature_extraction_rejects_missing_source_channels(sdata_blobs: SpatialData) -> None:
    _set_deterministic_features(sdata_blobs)
    _set_feature_metadata(
        sdata_blobs,
        feature_columns=("mean__1", "mean__0"),
        features=("mean",),
        source_image="blobs_image",
    )
    classifier = _make_classifier_bundle(sdata_blobs)

    with pytest.raises(ValueError, match="does not contain `source_channels`"):
        headless.apply_classifier_with_feature_extraction(
            sdata_blobs,
            classifier=classifier,
            table_name="table",
            labels_name="blobs_labels",
            feature_key="computed_features",
            coordinate_system="global",
            image_name="blobs_image",
        )

    assert "computed_features" not in sdata_blobs["table"].obsm


def test_compute_features_for_classifier_rejects_spatialdata_invalid_feature_key(sdata_blobs: SpatialData) -> None:
    classifier = _make_area_classifier_bundle()

    with pytest.raises(ValueError, match="target\\.feature_key must be a valid SpatialData name"):
        headless.compute_features_for_classifier(
            sdata_blobs,
            classifier=classifier,
            target=headless.HeadlessFeatureTarget(
                table_name="table",
                feature_key="features tst",
                triplets=(headless.FeatureExtractionTriplet("global", "blobs_labels", None),),
            ),
        )

    assert "features tst" not in sdata_blobs["table"].obsm


def test_compute_features_for_classifier_returns_normalized_target(sdata_blobs: SpatialData) -> None:
    bundle = _make_area_classifier_bundle()
    table = sdata_blobs["table"]

    resolved_target = headless.compute_features_for_classifier(
        sdata_blobs,
        bundle,
        target=headless.HeadlessFeatureTarget(
            table_name=" table ",
            feature_key=" computed_area ",
            triplets=(headless.FeatureExtractionTriplet("global", "blobs_labels", None),),
            overwrite_feature_key=True,
        ),
    )

    assert resolved_target.table_name == "table"
    assert resolved_target.feature_key == "computed_area"
    assert resolved_target.overwrite_feature_key is True
    metadata = table.uns[_FEATURE_MATRICES_KEY]["computed_area"]
    assert list(metadata["feature_columns"]) == ["area"]


def test_compute_features_for_classifier_requires_overwrite_for_existing_matrix(sdata_blobs: SpatialData) -> None:
    bundle = _make_area_classifier_bundle()
    target = headless.HeadlessFeatureTarget(
        table_name="table",
        feature_key="computed_area",
        triplets=(headless.FeatureExtractionTriplet("global", "blobs_labels", None),),
    )
    headless.compute_features_for_classifier(sdata_blobs, bundle, target=target)

    with pytest.raises(ValueError, match="already exists"):
        headless.compute_features_for_classifier(sdata_blobs, bundle, target=target)


def test_compute_features_for_classifier_replaces_custom_matrix_when_overwriting(sdata_blobs: SpatialData) -> None:
    bundle = _make_area_classifier_bundle()
    table = sdata_blobs["table"]
    table.obsm["computed_area"] = np.zeros((table.n_obs, 3), dtype=np.float64)

    headless.compute_features_for_classifier(
        sdata_blobs,
        bundle,
        target=headless.HeadlessFeatureTarget(
            table_name="table",
            feature_key="computed_area",
            triplets=(headless.FeatureExtractionTriplet("global", "blobs_labels", None),),
            overwrite_feature_key=True,
        ),
    )

    assert table.obsm["computed_area"].shape == (table.n_obs, 1)
    assert list(table.uns[_FEATURE_MATRICES_KEY]["computed_area"]["feature_columns"]) == ["area"]


def test_apply_classifier_with_feature_extraction_writes_predictions(sdata_blobs: SpatialData) -> None:
    classifier = _make_area_classifier_bundle()

    result = headless.apply_classifier_with_feature_extraction(
        sdata_blobs,
        classifier=classifier,
        table_name="table",
        labels_name="blobs_labels",
        feature_key="computed_area",
        coordinate_system="global",
        pred_class_column="with_features_class",
        pred_confidence_column="with_features_confidence",
    )

    table = sdata_blobs["table"]
    assert result.feature_key == "computed_area"
    assert result.pred_class_column == "with_features_class"
    assert "computed_area" in table.obsm
    assert "with_features_class" in table.obs
    assert "with_features_confidence" in table.obs
    assert table.uns[CLASSIFIER_APPLY_CONFIG_KEY]["feature_key"] == "computed_area"


def test_apply_classifier_with_feature_extraction_rejects_incompatible_feature_columns(
    sdata_blobs: SpatialData,
) -> None:
    bundle = _make_area_classifier_bundle()
    bundle.source_feature_metadata["feature_columns"] = ["not_area"]

    with pytest.raises(ValueError, match="do not match"):
        headless.apply_classifier_with_feature_extraction(
            sdata_blobs,
            bundle,
            table_name="table",
            labels_name="blobs_labels",
            feature_key="computed_area",
            coordinate_system="global",
        )

    assert PRED_CLASS_COLUMN not in sdata_blobs["table"].obs
    assert CLASSIFIER_APPLY_CONFIG_KEY not in sdata_blobs["table"].uns


def test_apply_classifier_with_feature_extraction_requires_explicit_ambiguous_coordinate_system(
    sdata_blobs: SpatialData,
) -> None:
    bundle = _make_area_classifier_bundle()
    set_transformation(sdata_blobs.labels["blobs_labels"], Identity(), to_coordinate_system="also_global")

    with pytest.raises(ValueError, match="Pass `coordinate_system` explicitly"):
        headless.apply_classifier_with_feature_extraction(
            sdata_blobs,
            bundle,
            table_name="table",
            labels_name="blobs_labels",
            feature_key="computed_features",
        )


def test_apply_classifier_from_path_persists_backed_prediction_state(
    tmp_path: Path,
    backed_sdata_blobs: SpatialData,
) -> None:
    _set_deterministic_features(backed_sdata_blobs)
    _set_feature_metadata(backed_sdata_blobs)
    bundle = _make_classifier_bundle(backed_sdata_blobs)
    classifier_path = tmp_path / "classifier.spatiato-classifier.joblib"
    write_classifier_export_bundle(classifier_path, bundle)

    result = headless.apply_classifier_from_path(
        backed_sdata_blobs,
        classifier_path,
        table_name="table",
        pred_class_column="headless_class",
        pred_confidence_column="headless_confidence",
    )

    table = backed_sdata_blobs["table"]
    reread = read_zarr(backed_sdata_blobs.path)
    disk_table = reread["table"]
    config = disk_table.uns[CLASSIFIER_APPLY_CONFIG_KEY]

    assert result.table_name == "table"
    assert "headless_class" in disk_table.obs
    assert "headless_confidence" in disk_table.obs
    assert "headless_class_colors" in disk_table.uns
    assert (
        disk_table.obs["headless_class"].astype("string").tolist()
        == table.obs["headless_class"].astype("string").tolist()
    )
    assert np.allclose(
        disk_table.obs["headless_confidence"].to_numpy(dtype=np.float64),
        table.obs["headless_confidence"].to_numpy(dtype=np.float64),
    )
    assert config["classifier_path"] == str(classifier_path)
    assert config["table_name"] == "table"
    assert config["feature_key"] == "features_1"
    assert config["pred_class_column"] == "headless_class"
    assert config["pred_confidence_column"] == "headless_confidence"


def test_apply_classifier_with_feature_extraction_from_path_persists_backed_feature_and_prediction_state(
    tmp_path: Path,
    backed_sdata_blobs: SpatialData,
) -> None:
    bundle = _make_area_classifier_bundle()
    classifier_path = tmp_path / "classifier.spatiato-classifier.joblib"
    write_classifier_export_bundle(classifier_path, bundle)

    result = headless.apply_classifier_with_feature_extraction_from_path(
        backed_sdata_blobs,
        classifier_path,
        table_name="table",
        labels_name="blobs_labels",
        pred_class_column="with_features_class",
        pred_confidence_column="with_features_confidence",
    )

    reread = read_zarr(backed_sdata_blobs.path)
    disk_table = reread["table"]

    assert result.feature_key == "area_features"
    assert "area_features" in disk_table.obsm
    assert disk_table.obsm["area_features"].shape == backed_sdata_blobs["table"].obsm["area_features"].shape
    assert list(disk_table.uns[_FEATURE_MATRICES_KEY]["area_features"]["feature_columns"]) == list(
        bundle.feature_columns
    )
    assert "with_features_class" in disk_table.obs
    assert "with_features_confidence" in disk_table.obs
    assert disk_table.uns[CLASSIFIER_APPLY_CONFIG_KEY]["classifier_path"] == str(classifier_path)
    assert disk_table.uns[CLASSIFIER_APPLY_CONFIG_KEY]["feature_key"] == "area_features"


def test_apply_classifier_rejects_missing_feature_key(sdata_blobs: SpatialData) -> None:
    _set_deterministic_features(sdata_blobs)
    _set_feature_metadata(sdata_blobs)
    bundle = _make_classifier_bundle(sdata_blobs)

    with pytest.raises(ValueError, match="not available in `.obsm`"):
        headless.apply_classifier(sdata_blobs, bundle, table_name="table", feature_key="missing_features")


def test_apply_classifier_rejects_missing_feature_metadata(sdata_blobs: SpatialData) -> None:
    _set_deterministic_features(sdata_blobs)
    _set_feature_metadata(sdata_blobs)
    bundle = _make_classifier_bundle(sdata_blobs)
    sdata_blobs["table"].uns.pop(_FEATURE_MATRICES_KEY)

    with pytest.raises(ValueError, match="missing Harpy metadata"):
        headless.apply_classifier(sdata_blobs, bundle, table_name="table")


def test_apply_classifier_rejects_feature_matrix_shape_drift(sdata_blobs: SpatialData) -> None:
    _set_deterministic_features(sdata_blobs)
    _set_feature_metadata(sdata_blobs)
    bundle = _make_classifier_bundle(sdata_blobs)
    table = sdata_blobs["table"]
    table.obsm["features_1"] = np.column_stack(
        [np.asarray(table.obsm["features_1"], dtype=np.float64), np.zeros(table.n_obs)]
    )

    with pytest.raises(ValueError, match="metadata"):
        headless.apply_classifier(sdata_blobs, bundle, table_name="table")


def test_apply_classifier_rejects_custom_feature_matrix_shape_drift_with_custom_message(
    sdata_blobs: SpatialData,
) -> None:
    _set_deterministic_features(sdata_blobs)
    table = sdata_blobs["table"]
    register_feature_matrix_metadata(
        table,
        "features_1",
        feature_columns=("is_large", "instance_fraction"),
    )
    bundle = _make_classifier_bundle(sdata_blobs)
    # Simulate stale custom metadata: the registered schema still has two
    # feature columns, but `np.column_stack(...)` adds a third live `.obsm`
    # column below.
    table.obsm["features_1"] = np.column_stack(
        [np.asarray(table.obsm["features_1"], dtype=np.float64), np.zeros(table.n_obs)]
    )

    with pytest.raises(ValueError, match="live `.obsm` width"):
        headless.apply_classifier(sdata_blobs, bundle, table_name="table")


def test_apply_classifier_rejects_feature_column_order_mismatch(sdata_blobs: SpatialData) -> None:
    _set_deterministic_features(sdata_blobs)
    _set_feature_metadata(sdata_blobs)
    bundle = _make_classifier_bundle(sdata_blobs)
    sdata_blobs["table"].uns[_FEATURE_MATRICES_KEY]["features_1"]["feature_columns"] = [
        "instance_fraction",
        "is_large",
    ]

    with pytest.raises(ValueError, match="do not match"):
        headless.apply_classifier(sdata_blobs, bundle, table_name="table")


def test_apply_classifier_rejects_custom_feature_column_order_mismatch_with_custom_message(
    sdata_blobs: SpatialData,
) -> None:
    _set_deterministic_features(sdata_blobs)
    table = sdata_blobs["table"]
    register_feature_matrix_metadata(
        table,
        "features_1",
        feature_columns=("is_large", "instance_fraction"),
    )
    bundle = _make_classifier_bundle(sdata_blobs)
    table.uns[_FEATURE_MATRICES_KEY]["features_1"]["feature_columns"] = [
        "instance_fraction",
        "is_large",
    ]

    with pytest.raises(ValueError, match="same `feature_columns` in the same order"):
        headless.apply_classifier(sdata_blobs, bundle, table_name="table")


def test_apply_classifier_rejects_custom_target_feature_order_mismatch_with_custom_message(
    sdata_blobs: SpatialData,
) -> None:
    _set_deterministic_features(sdata_blobs)
    # Build the bundle from Harpy-style metadata with feature columns
    # `is_large`, `instance_fraction`.
    _set_feature_metadata(sdata_blobs)
    bundle = _make_classifier_bundle(sdata_blobs)
    register_feature_matrix_metadata(
        sdata_blobs["table"],
        "features_1",
        feature_columns=("instance_fraction", "is_large"),
        overwrite=True,
    )

    with pytest.raises(ValueError, match="same `feature_columns` in the same order"):
        headless.apply_classifier(sdata_blobs, bundle, table_name="table")


def test_apply_classifier_selected_region_leaves_other_regions_unchanged(
    sdata_blobs_multi_region: SpatialData,
) -> None:
    _set_deterministic_features(sdata_blobs_multi_region, table_name="table_multi")
    _set_feature_metadata(sdata_blobs_multi_region, table_name="table_multi")
    bundle = _make_classifier_bundle(sdata_blobs_multi_region, table_name="table_multi")
    table = sdata_blobs_multi_region["table_multi"]
    table.obs[PRED_CLASS_COLUMN] = pd.Categorical(np.full(table.n_obs, 9, dtype=np.int64), categories=[9])
    table.obs[PRED_CONFIDENCE_COLUMN] = pd.Series(np.full(table.n_obs, 0.55), index=table.obs.index, dtype="float64")

    result = headless.apply_classifier(
        sdata_blobs_multi_region,
        bundle,
        table_name="table_multi",
        prediction_regions=["blobs_labels"],
    )

    region_values = table.obs["region"].astype("string")
    selected_rows = region_values == "blobs_labels"
    hidden_rows = region_values == "blobs_labels_2"

    assert result.prediction_regions == ("blobs_labels",)
    assert result.n_predicted_rows == int(selected_rows.sum())
    assert table.obs.loc[selected_rows, PRED_CLASS_COLUMN].astype("string").ne("9").all()
    assert table.obs.loc[selected_rows, PRED_CONFIDENCE_COLUMN].between(0.0, 1.0).all()
    assert table.obs.loc[hidden_rows, PRED_CLASS_COLUMN].astype("string").eq("9").all()
    assert table.obs.loc[hidden_rows, PRED_CONFIDENCE_COLUMN].eq(0.55).all()


def test_apply_classifier_clears_invalid_rows_in_prediction_scope(
    sdata_blobs_multi_region: SpatialData,
) -> None:
    _set_deterministic_features(sdata_blobs_multi_region, table_name="table_multi")
    _set_feature_metadata(sdata_blobs_multi_region, table_name="table_multi")
    bundle = _make_classifier_bundle(sdata_blobs_multi_region, table_name="table_multi")
    _set_invalid_feature_rows_for_region(sdata_blobs_multi_region, region_name="blobs_labels_2")
    table = sdata_blobs_multi_region["table_multi"]
    table.obs[PRED_CLASS_COLUMN] = pd.Categorical(np.full(table.n_obs, 9, dtype=np.int64), categories=[9])
    table.obs[PRED_CONFIDENCE_COLUMN] = pd.Series(np.full(table.n_obs, 0.55), index=table.obs.index, dtype="float64")

    result = headless.apply_classifier(sdata_blobs_multi_region, bundle, table_name="table_multi")

    region_values = table.obs["region"].astype("string")
    valid_rows = region_values == "blobs_labels"
    invalid_rows = region_values == "blobs_labels_2"

    assert result.prediction_regions == ("blobs_labels", "blobs_labels_2")
    assert result.n_predicted_rows == int(valid_rows.sum())
    assert result.n_skipped_feature_invalid_rows == int(invalid_rows.sum())
    assert table.obs.loc[valid_rows, PRED_CLASS_COLUMN].notna().all()
    assert table.obs.loc[valid_rows, PRED_CONFIDENCE_COLUMN].between(0.0, 1.0).all()
    assert table.obs.loc[invalid_rows, PRED_CLASS_COLUMN].isna().all()
    assert table.obs.loc[invalid_rows, PRED_CONFIDENCE_COLUMN].isna().all()
    assert table.uns[CLASSIFIER_APPLY_CONFIG_KEY]["n_skipped_feature_invalid_rows"] == int(invalid_rows.sum())


def test_headless_module_avoids_direct_interactive_classifier_imports() -> None:
    source = (Path(__file__).resolve().parents[1] / "src" / "spatiato" / "headless.py").read_text()
    import_modules: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            import_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            import_modules.add(node.module)

    assert "spatiato.widgets.object_classification.controller" not in import_modules
    assert not any(module.startswith("spatiato.widgets") for module in import_modules)
    assert "napari" not in import_modules
    assert "qtpy" not in import_modules
    assert "thread_worker" not in source
    assert "spatiato.widgets.feature_extraction.controller" not in import_modules
