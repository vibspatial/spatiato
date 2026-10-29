from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

from spatiato.core.feature_extraction import (
    FeatureExtractionTriplet,
    _normalize_triplets,
    _requires_image,
    calculate_feature_matrix,
)
from spatiato.core.feature_matrix_metadata import CUSTOM_OBSM_SOURCE_KIND
from spatiato.core.object_classification.annotation import USER_CLASS_COLORS_KEY
from spatiato.core.object_classification.classifier import (
    CLASSIFIER_APPLY_CONFIG_KEY,
    CLASSIFIER_CONFIG_KEY,
    ClassifierApplyResult,
    _validate_feature_matrix_compatible_with_bundle,
)
from spatiato.core.object_classification.classifier import apply_classifier as _apply_classifier
from spatiato.core.object_classification.classifier_export import (
    ClassifierExportBundle,
    read_classifier_export_bundle,
)
from spatiato.core.persistence import write_table_prediction_state
from spatiato.core.spatialdata import _get_element_coordinate_systems, get_table
from spatiato.core.validation import normalize_spatialdata_name

if TYPE_CHECKING:
    from spatialdata import SpatialData


@dataclass(frozen=True)
class HeadlessFeatureTarget:
    """Target table and element mapping for headless feature calculation."""

    table_name: str
    feature_key: str
    triplets: tuple[FeatureExtractionTriplet, ...]
    overwrite_feature_key: bool = False


def load_classifier(path: str | Path) -> ClassifierExportBundle:
    """Load a trusted classifier artifact from disk."""
    return read_classifier_export_bundle(path)


def compute_features_for_classifier(
    sdata: SpatialData,
    classifier: ClassifierExportBundle,
    *,
    target: HeadlessFeatureTarget,
) -> HeadlessFeatureTarget:
    """Compute the feature matrix required by an exported classifier.

    Intensity-channel selection is read from the classifier's exported
    `source_channels` feature metadata so the computed matrix preserves the
    training schema.

    An existing `table.obsm[target.feature_key]` raises unless
    `target.overwrite_feature_key` is set. Then a matrix harpy calculated with
    the same features and channels is updated for the target labels elements
    only, keeping the values of other labels elements. Any other matrix is
    removed with its metadata and calculated again, so the values of other
    labels elements are lost, and a warning names them. If `sdata` is backed
    by zarr, the stored matrix decides, not unsaved changes in memory.
    """
    if classifier.source_kind == CUSTOM_OBSM_SOURCE_KIND:
        raise ValueError(
            "Custom `.obsm` classifier bundles cannot recompute features with "
            "`headless.compute_features_for_classifier(...)`. Use `headless.apply_classifier(...)` "
            "on a table that already contains a compatible feature matrix registered with "
            "`spatiato.core.feature_matrix_metadata.register_feature_matrix_metadata(...)`."
        )

    resolved_target = _normalize_headless_feature_target(target)
    feature_names = classifier.feature_names
    source_channels = _resolve_classifier_source_channels(classifier)

    calculate_feature_matrix(
        sdata,
        table_name=resolved_target.table_name,
        feature_key=resolved_target.feature_key,
        triplets=tuple(replace(triplet, channels=source_channels) for triplet in resolved_target.triplets),
        feature_names=feature_names,
        overwrite_feature_key=resolved_target.overwrite_feature_key,
    )

    table = get_table(sdata, resolved_target.table_name)
    _validate_feature_matrix_compatible_with_bundle(table, resolved_target.feature_key, classifier)
    return resolved_target


def apply_classifier(
    sdata: SpatialData,
    classifier: ClassifierExportBundle,
    *,
    table_name: str,
    feature_key: str | None = None,
    prediction_regions: Sequence[str] | None = None,
    pred_class_column: str = "pred_class",
    pred_confidence_column: str = "pred_confidence",
    classifier_path: str | Path | None = None,
) -> ClassifierApplyResult:
    """Apply an exported classifier to an existing feature matrix.

    The target table must already contain a compatible feature matrix in
    `.obsm`. When `feature_key` is omitted, the source feature key stored in the
    classifier is used. Predictions are written in place to `table.obs` using
    `pred_class_column` and `pred_confidence_column`, and apply provenance is
    recorded in `table.uns["classifier_apply_config"]`. If `sdata` is backed by
    zarr, the updated prediction state is written back to disk automatically.

    Parameters
    ----------
    sdata
        SpatialData object containing the target table.
    classifier
        Exported classifier returned by `load_classifier(...)` or
        `ClassifierController.export_classifier(...)`.
    table_name
        Name of the target annotation table in `sdata`.
    feature_key
        Key in `table.obsm` for the target feature matrix. If omitted, the
        classifier's source feature key is used.
    prediction_regions
        Optional table regions to classify. If omitted, all regions declared in
        the table metadata are classified.
    pred_class_column
        Column in `table.obs` that will receive predicted class labels.
    pred_confidence_column
        Column in `table.obs` that will receive prediction confidences.
    classifier_path
        Optional artifact path to record in `classifier_apply_config`; this
        does not affect prediction.

    The returned `ClassifierApplyResult` summarizes the rows that were written
    and any in-scope rows skipped because of invalid feature values.
    """
    result = _apply_classifier(
        sdata,
        classifier,
        table_name=table_name,
        feature_key=feature_key,
        prediction_regions=prediction_regions,
        pred_class_column=pred_class_column,
        pred_confidence_column=pred_confidence_column,
        classifier_path=classifier_path,
    )
    _write_backed_prediction_state_if_needed(sdata, result)
    return result


def apply_classifier_from_path(
    sdata: SpatialData,
    path: str | Path,
    *,
    table_name: str,
    feature_key: str | None = None,
    prediction_regions: Sequence[str] | None = None,
    pred_class_column: str = "pred_class",
    pred_confidence_column: str = "pred_confidence",
) -> ClassifierApplyResult:
    """Load a classifier artifact from disk and apply it to an existing feature matrix.

    This is a convenience wrapper around `load_classifier(...)` and
    `apply_classifier(...)`. The artifact path is passed through as
    `classifier_path`, so it is recorded in
    `table.uns["classifier_apply_config"]`.
    """
    classifier = load_classifier(path)
    return apply_classifier(
        sdata,
        classifier,
        table_name=table_name,
        feature_key=feature_key,
        prediction_regions=prediction_regions,
        pred_class_column=pred_class_column,
        pred_confidence_column=pred_confidence_column,
        classifier_path=path,
    )


def apply_classifier_with_feature_extraction(
    sdata: SpatialData,
    classifier: ClassifierExportBundle,
    *,
    table_name: str,
    labels_name: str | Sequence[str],
    feature_key: str | None = None,
    coordinate_system: str | Sequence[str] | None = None,
    image_name: str | Sequence[str] | None = None,
    overwrite_feature_key: bool = False,
    pred_class_column: str = "pred_class",
    pred_confidence_column: str = "pred_confidence",
    classifier_path: str | Path | None = None,
) -> ClassifierApplyResult:
    """Compute the classifier feature matrix on a target dataset, then apply it.

    This is the headless equivalent of first running Harpy feature extraction
    for `labels_name` and then calling `apply_classifier(...)` with the
    computed feature key. The generated feature matrix is written to
    `table.obsm[feature_key]`, and Harpy feature metadata is written to
    `table.uns["feature_matrices"][feature_key]`. The computed feature columns
    must exactly match the feature schema stored in `classifier`. For
    intensity-derived features, channel selection is read from the classifier's
    exported `source_channels` metadata.

    If `sdata` is backed by zarr, Harpy persists the feature matrix and feature
    metadata during feature extraction, and `apply_classifier(...)` persists
    the prediction columns and classifier apply metadata.

    Parameters
    ----------
    sdata
        SpatialData object containing the target table, labels, and optional
        image elements.
    classifier
        Exported classifier returned by `load_classifier(...)` or
        `ClassifierController.export_classifier(...)`.
    table_name
        Name of the target annotation table in `sdata`.
    labels_name
        Labels element or elements to use for feature extraction and prediction.
        These labels become the prediction regions after feature extraction.
    feature_key
        Key in `table.obsm` for the computed feature matrix. If omitted, the
        source feature key stored in the classifier is used.
    coordinate_system
        Coordinate system or systems to use for the selected labels. If omitted,
        the coordinate system is inferred only when unambiguous.
    image_name
        Image element or elements to use for intensity-derived features. This is
        required when the classifier was trained on intensity features
        and ignored for morphology-only classifiers.
    overwrite_feature_key
        Whether an existing `table.obsm[feature_key]` may be replaced. A matrix
        harpy calculated with the same features and channels is updated for
        `labels_name` only, keeping the values of other labels elements. Any
        other matrix is removed with its metadata and calculated again, so the
        values of other labels elements are lost, and a warning names them. If
        `sdata` is backed by zarr, the stored matrix decides, not unsaved
        changes in memory.
    pred_class_column
        Column in `table.obs` that will receive predicted class labels.
    pred_confidence_column
        Column in `table.obs` that will receive prediction confidences.
    classifier_path
        Optional artifact path to record in `classifier_apply_config`; this
        does not affect feature extraction or prediction.

    The returned `ClassifierApplyResult` summarizes the rows that were written
    and any in-scope rows skipped because of invalid feature values.
    """
    target = _build_headless_feature_target(
        sdata,
        classifier,
        table_name=table_name,
        labels_name=labels_name,
        feature_key=feature_key,
        coordinate_system=coordinate_system,
        image_name=image_name,
        overwrite_feature_key=overwrite_feature_key,
    )
    resolved_target = compute_features_for_classifier(sdata, classifier, target=target)
    return apply_classifier(
        sdata,
        classifier,
        table_name=resolved_target.table_name,
        feature_key=resolved_target.feature_key,
        prediction_regions=tuple(triplet.labels_name for triplet in resolved_target.triplets),
        pred_class_column=pred_class_column,
        pred_confidence_column=pred_confidence_column,
        classifier_path=classifier_path,
    )


def apply_classifier_with_feature_extraction_from_path(
    sdata: SpatialData,
    path: str | Path,
    *,
    table_name: str,
    labels_name: str | Sequence[str],
    feature_key: str | None = None,
    coordinate_system: str | Sequence[str] | None = None,
    image_name: str | Sequence[str] | None = None,
    overwrite_feature_key: bool = False,
    pred_class_column: str = "pred_class",
    pred_confidence_column: str = "pred_confidence",
) -> ClassifierApplyResult:
    """Load a classifier artifact, compute target features, and apply it.

    This is a convenience wrapper around `load_classifier(...)` and
    `apply_classifier_with_feature_extraction(...)`. The artifact path is
    passed through as `classifier_path`, so it is recorded in
    `table.uns["classifier_apply_config"]`.

    Parameters
    ----------
    sdata
        SpatialData object containing the target table, labels, and optional
        image elements.
    path
        Path to a trusted `.spatiato-classifier.joblib` classifier artifact.
    table_name
        Name of the target annotation table in `sdata`.
    labels_name
        Labels element or elements to use for feature extraction and prediction.
        These labels become the prediction regions after feature extraction.
    feature_key
        Key in `table.obsm` for the computed feature matrix. If omitted, the
        source feature key stored in the classifier is used.
    coordinate_system
        Coordinate system or systems to use for the selected labels. If omitted,
        the coordinate system is inferred only when unambiguous.
    image_name
        Image element or elements to use for intensity-derived features. This is
        required when the classifier was trained on intensity features
        and ignored for morphology-only classifiers.
    overwrite_feature_key
        Whether an existing `table.obsm[feature_key]` may be replaced. A matrix
        harpy calculated with the same features and channels is updated for
        `labels_name` only, keeping the values of other labels elements. Any
        other matrix is removed with its metadata and calculated again, so the
        values of other labels elements are lost, and a warning names them. If
        `sdata` is backed by zarr, the stored matrix decides, not unsaved
        changes in memory.
    pred_class_column
        Column in `table.obs` that will receive predicted class labels.
    pred_confidence_column
        Column in `table.obs` that will receive prediction confidences.

    The returned `ClassifierApplyResult` summarizes the rows that were written
    and any in-scope rows skipped because of invalid feature values.
    """
    classifier = load_classifier(path)
    return apply_classifier_with_feature_extraction(
        sdata,
        classifier,
        table_name=table_name,
        labels_name=labels_name,
        feature_key=feature_key,
        coordinate_system=coordinate_system,
        image_name=image_name,
        overwrite_feature_key=overwrite_feature_key,
        pred_class_column=pred_class_column,
        pred_confidence_column=pred_confidence_column,
        classifier_path=path,
    )


def _build_headless_feature_target(
    sdata: SpatialData,
    classifier: ClassifierExportBundle,
    *,
    table_name: str,
    labels_name: str | Sequence[str],
    feature_key: str | None,
    coordinate_system: str | Sequence[str] | None,
    image_name: str | Sequence[str] | None,
    overwrite_feature_key: bool,
) -> HeadlessFeatureTarget:
    normalized_labels = _normalize_name_sequence(labels_name, "labels_name")
    coordinate_systems = _resolve_coordinate_systems_for_labels(sdata, normalized_labels, coordinate_system)
    image_names = _normalize_optional_parallel_names(image_name, len(normalized_labels), "image_name")
    resolved_feature_key = normalize_spatialdata_name(
        classifier.feature_key if feature_key is None else feature_key,
        "feature_key",
    )

    triplets = tuple(
        FeatureExtractionTriplet(
            coordinate_system=resolved_coordinate_system,
            labels_name=resolved_labels_name,
            image_name=resolved_image_name,
        )
        for resolved_labels_name, resolved_coordinate_system, resolved_image_name in zip(
            normalized_labels,
            coordinate_systems,
            image_names,
            strict=True,
        )
    )

    return HeadlessFeatureTarget(
        table_name=_normalize_nonempty_str(table_name, "table_name"),
        feature_key=resolved_feature_key,
        triplets=triplets,
        overwrite_feature_key=overwrite_feature_key,
    )


def _normalize_headless_feature_target(target: HeadlessFeatureTarget) -> HeadlessFeatureTarget:
    normalized_table_name = _normalize_nonempty_str(target.table_name, "target.table_name")
    normalized_feature_key = normalize_spatialdata_name(target.feature_key, "target.feature_key")
    normalized_triplets = _normalize_triplets(target.triplets)
    if not normalized_triplets:
        raise ValueError("target.triplets must contain at least one feature-extraction triplet.")
    if any(triplet.channels is not None for triplet in normalized_triplets):
        raise ValueError(
            "Headless classifier feature extraction does not accept explicit channels. "
            "Channels are read from the exported classifier metadata `source_channels`."
        )

    return HeadlessFeatureTarget(
        table_name=normalized_table_name,
        feature_key=normalized_feature_key,
        triplets=normalized_triplets,
        overwrite_feature_key=bool(target.overwrite_feature_key),
    )


def _resolve_classifier_source_channels(classifier: ClassifierExportBundle) -> tuple[str, ...] | None:
    if not _requires_image(classifier.feature_names):
        return None

    source_channels = classifier.source_channels
    if source_channels is None:
        raise ValueError(
            "Classifier uses intensity-derived features but its feature metadata does not contain "
            "`source_channels`. Re-export the classifier with a newer spatiato version."
        )
    return source_channels


def _normalize_nonempty_str(value: str, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must not be empty.")
    return normalized


def _normalize_name_sequence(value: str | Sequence[str], name: str) -> tuple[str, ...]:
    values = (value,) if isinstance(value, str) else tuple(value)
    if not values:
        raise ValueError(f"{name} must contain at least one name.")

    normalized = tuple(_normalize_nonempty_str(item, name) for item in values)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must not contain duplicate names.")
    return normalized


def _normalize_parallel_names(value: str | Sequence[str], expected_count: int, name: str) -> tuple[str, ...]:
    values = (value,) if isinstance(value, str) else tuple(value)
    if not values:
        raise ValueError(f"{name} must contain at least one name.")
    normalized = tuple(_normalize_nonempty_str(item, name) for item in values)
    return _broadcast_parallel_values(normalized, expected_count, name)


def _normalize_optional_parallel_names(
    value: str | Sequence[str] | None,
    expected_count: int,
    name: str,
) -> tuple[str | None, ...]:
    if value is None:
        return (None,) * expected_count

    values = (value,) if isinstance(value, str) else tuple(value)
    if not values:
        raise ValueError(f"{name} must contain at least one name when provided.")

    normalized = tuple(str(item).strip() or None for item in values)
    return _broadcast_parallel_values(normalized, expected_count, name)


def _broadcast_parallel_values[T](values: tuple[T, ...], expected_count: int, name: str) -> tuple[T, ...]:
    if len(values) == 1:
        return values * expected_count
    if len(values) != expected_count:
        raise ValueError(f"{name} must be a single value or contain exactly {expected_count} entries.")
    return values


def _resolve_coordinate_systems_for_labels(
    sdata: SpatialData,
    labels_names: tuple[str, ...],
    coordinate_system: str | Sequence[str] | None,
) -> tuple[str, ...]:
    """Return one validated coordinate system per labels element.

    Explicit coordinate-system input is broadcast or matched one-to-one against
    `labels_name`, then checked against each labels element. If no coordinate
    system is provided, inference succeeds only when every labels element
    exposes exactly one coordinate system.
    """
    if coordinate_system is not None:
        coordinate_systems = _normalize_parallel_names(coordinate_system, len(labels_names), "coordinate_system")
        for labels_name, resolved_coordinate_system in zip(labels_names, coordinate_systems, strict=True):
            available_coordinate_systems = _get_label_coordinate_systems(sdata, labels_name)
            if resolved_coordinate_system not in available_coordinate_systems:
                available = ", ".join(f"`{name}`" for name in available_coordinate_systems)
                raise ValueError(
                    f"Labels element `{labels_name}` is not available in coordinate system "
                    f"`{resolved_coordinate_system}`. Available coordinate systems: {available}."
                )
        return coordinate_systems

    return _infer_coordinate_systems_for_labels(sdata, labels_names)


def _infer_coordinate_systems_for_labels(sdata: SpatialData, labels_names: tuple[str, ...]) -> tuple[str, ...]:
    coordinate_systems_by_label = {
        labels_name: _get_label_coordinate_systems(sdata, labels_name) for labels_name in labels_names
    }
    if all(len(coordinate_systems) == 1 for coordinate_systems in coordinate_systems_by_label.values()):
        return tuple(coordinate_systems[0] for coordinate_systems in coordinate_systems_by_label.values())

    options = "; ".join(
        f"{labels_name}: {', '.join(coordinate_systems)}"
        for labels_name, coordinate_systems in coordinate_systems_by_label.items()
    )
    raise ValueError(
        "Could not infer a unique coordinate system for feature extraction. "
        f"Pass `coordinate_system` explicitly. Available coordinate systems by labels element: {options}."
    )


def _get_label_coordinate_systems(sdata: SpatialData, labels_name: str) -> tuple[str, ...]:
    if labels_name not in sdata.labels:
        raise ValueError(f"Labels element `{labels_name}` is not available in the target SpatialData object.")

    coordinate_systems = _get_element_coordinate_systems(sdata.labels[labels_name])
    if not coordinate_systems:
        raise ValueError(f"Labels element `{labels_name}` does not expose any coordinate systems.")
    return coordinate_systems


def _write_backed_prediction_state_if_needed(sdata: SpatialData, result: ClassifierApplyResult) -> None:
    if not sdata.is_backed() or sdata.path is None:
        return

    write_table_prediction_state(
        sdata,
        table_name=result.table_name,
        uns_keys=(
            USER_CLASS_COLORS_KEY,
            f"{result.pred_class_column}_colors",
            CLASSIFIER_CONFIG_KEY,
            CLASSIFIER_APPLY_CONFIG_KEY,
        ),
    )
