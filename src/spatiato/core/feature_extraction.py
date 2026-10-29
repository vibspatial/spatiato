from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, get_args

import dask.array as da
import numpy as np
import zarr
from harpy.table.io._components import _check_in_memory_versus_storage_axes, _component_update_destination
from harpy.table.io._write_validation import _validated_observation_pairs
from harpy.utils._keys import _FEATURE_MATRICES_KEY
from loguru import logger
from spatialdata.models import TableModel
from xarray import DataArray

from spatiato.core.feature_matrix_metadata import (
    CUSTOM_OBSM_SOURCE_KIND,
    FEATURE_MATRIX_METADATA_SCHEMA_VERSION,
    HARPY_ADD_FEATURE_MATRIX_SOURCE_KIND,
)
from spatiato.core.spatialdata import get_table, get_table_metadata

if TYPE_CHECKING:
    from spatialdata import SpatialData

FeatureExtractionChannel = int | str
ExistingFeatureMatrixKind = Literal["absent", "compatible", "incompatible"]
FeatureMatrixChangeKind = Literal["created", "updated"]

_INTENSITY_FEATURES = frozenset({"sum", "mean", "var", "min", "max", "kurtosis", "skew"})


@dataclass(frozen=True)
class FeatureExtractionTriplet:
    """One explicit `coordinate_system -> labels element -> image` selection.

    Channel selection uses a small but important internal distinction:
    `channels=None` means no explicit channel selection was supplied, so Harpy
    may use its default behavior; `channels=()` means an explicit empty
    selection, which is invalid for intensity-derived features; a non-empty
    tuple carries the selected channels.
    """

    coordinate_system: str
    labels_name: str
    image_name: str | None
    channels: tuple[FeatureExtractionChannel, ...] | None = None


def _normalize_channels(
    channels: Sequence[FeatureExtractionChannel] | FeatureExtractionChannel | None,
) -> tuple[FeatureExtractionChannel, ...] | None:
    """Normalize optional channel input while preserving explicit empty selections.

    `None` is kept as "no explicit selection"; empty sequences normalize to
    `()`, which represents "a selector exists, but no channels are selected".
    """
    if channels is None:
        return None
    if isinstance(channels, (str, int)):
        values = [channels]
    else:
        values = list(channels)

    normalized: list[FeatureExtractionChannel] = []
    seen: set[FeatureExtractionChannel] = set()
    for channel in values:
        if isinstance(channel, str):
            normalized_channel: FeatureExtractionChannel = channel.strip()
            if not normalized_channel:
                continue
        else:
            normalized_channel = channel

        if normalized_channel in seen:
            raise ValueError(f"Duplicate channel selection is not allowed: `{normalized_channel}`.")
        normalized.append(normalized_channel)
        seen.add(normalized_channel)
    return tuple(normalized)


def _normalize_triplets(
    triplets: Sequence[FeatureExtractionTriplet] | FeatureExtractionTriplet | None,
) -> tuple[FeatureExtractionTriplet, ...]:
    if triplets is None:
        return ()
    if isinstance(triplets, FeatureExtractionTriplet):
        values = [triplets]
    else:
        values = list(triplets)

    normalized: list[FeatureExtractionTriplet] = []
    seen_labels_names: set[str] = set()
    for triplet in values:
        normalized_coordinate_system = str(triplet.coordinate_system).strip()
        normalized_labels_name = str(triplet.labels_name).strip()
        normalized_image_name = None if triplet.image_name is None else str(triplet.image_name).strip() or None
        normalized_channels = _normalize_channels(triplet.channels)

        if not normalized_coordinate_system:
            raise ValueError("Feature extraction triplets require a coordinate system.")
        if not normalized_labels_name:
            raise ValueError("Feature extraction triplets require a labels element name.")
        if normalized_labels_name in seen_labels_names:
            raise ValueError(
                f"Duplicate labels element selections are not allowed in a single feature-extraction request: "
                f"`{normalized_labels_name}`."
            )

        normalized.append(
            FeatureExtractionTriplet(
                coordinate_system=normalized_coordinate_system,
                labels_name=normalized_labels_name,
                image_name=normalized_image_name,
                channels=normalized_channels,
            )
        )
        seen_labels_names.add(normalized_labels_name)

    return tuple(normalized)


def _requires_image(feature_names: Sequence[str]) -> bool:
    return any(feature_name in _INTENSITY_FEATURES for feature_name in feature_names)


def _has_empty_intensity_channel_selection(
    triplets: Sequence[FeatureExtractionTriplet],
    feature_names: Sequence[str],
) -> bool:
    """Return whether intensity features are requested with an explicitly empty channel selection.

    A `channels` value of `None` means "use all channels" (the harpy default), so only an empty
    sequence counts as the user having selected no channels.
    """
    if not _requires_image(feature_names):
        return False

    return any(triplet.channels is not None and len(triplet.channels) == 0 for triplet in triplets)


def _get_triplet_channel_selection_error(
    triplets: Sequence[FeatureExtractionTriplet],
    feature_names: Sequence[str],
) -> str | None:
    if not _requires_image(feature_names) or len(triplets) <= 1:
        return None

    first_channels = triplets[0].channels
    for triplet in triplets[1:]:
        if triplet.channels != first_channels:
            return "Feature extraction: all selected extraction targets must currently use the same channel selection."

    return None


def _resolve_harpy_labels_name_parameter(
    triplets: Sequence[FeatureExtractionTriplet],
) -> str | list[str]:
    labels_names = [triplet.labels_name for triplet in triplets]
    return labels_names[0] if len(labels_names) == 1 else labels_names


def _resolve_harpy_coordinate_system_parameter(
    triplets: Sequence[FeatureExtractionTriplet],
) -> str | list[str]:
    coordinate_systems = [triplet.coordinate_system for triplet in triplets]
    return coordinate_systems[0] if len(coordinate_systems) == 1 else coordinate_systems


def _resolve_harpy_image_name_parameter(
    triplets: Sequence[FeatureExtractionTriplet],
    feature_names: Sequence[str],
) -> str | list[str] | None:
    if not _requires_image(feature_names):
        return None

    image_names = [triplet.image_name for triplet in triplets]
    if any(image_name is None for image_name in image_names):
        raise ValueError(
            "An image is required for every extraction target when intensity-derived features are selected."
        )

    resolved_image_names = [str(image_name) for image_name in image_names if image_name is not None]
    return resolved_image_names[0] if len(resolved_image_names) == 1 else resolved_image_names


def _resolve_harpy_channel_parameter(
    triplets: Sequence[FeatureExtractionTriplet],
    feature_names: Sequence[str],
) -> list[FeatureExtractionChannel] | None:
    if not _requires_image(feature_names):
        return None

    channel_selection_error = _get_triplet_channel_selection_error(triplets, feature_names)
    if channel_selection_error is not None:
        raise ValueError(channel_selection_error)

    channels = triplets[0].channels
    if channels is None:
        return None
    if len(channels) == 0:
        raise ValueError("Intensity-derived feature extraction requires at least one selected channel.")

    return list(channels)


@dataclass(frozen=True)
class ExistingFeatureMatrix:
    """How an existing `table.obsm[feature_key]` relates to a requested extraction.

    `kind` decides what an overwrite does:

    - `"absent"`: there is no matrix, and extraction creates it. Metadata without
      a matrix (`orphaned_metadata`) has no values to lose and is replaced.
    - `"compatible"`: harpy accepts the matrix for a regional update. Rows of
      the selected labels elements are replaced; rows of `unselected_regions`
      are kept.
    - `"incompatible"`: anything else, explained by `reason`. The matrix and
      its metadata are removed first, so the values of `unselected_regions`
      are lost.

    `reason` is set only for `"incompatible"`, and `orphaned_metadata` only for
    `"absent"`.
    """

    kind: ExistingFeatureMatrixKind
    reason: str | None = None
    unselected_regions: tuple[str, ...] = ()
    orphaned_metadata: bool = False

    def __post_init__(self) -> None:
        if self.kind not in get_args(ExistingFeatureMatrixKind):
            raise ValueError(f"Unsupported existing feature matrix kind: {self.kind!r}.")
        if self.kind == "incompatible" and self.reason is None:
            raise ValueError('Kind "incompatible" requires a `reason`.')
        if self.kind != "incompatible" and self.reason is not None:
            raise ValueError(f'`reason` only applies to kind "incompatible", not {self.kind!r}.')
        if self.orphaned_metadata and self.kind != "absent":
            raise ValueError(f'`orphaned_metadata` only applies to kind "absent", not {self.kind!r}.')


@dataclass(frozen=True)
class FeatureMatrixCalculation:
    """What `calculate_feature_matrix(...)` changed, for table-change events.

    `regions` are the selected labels elements, or every region of the table
    when an incompatible matrix was replaced entirely.
    """

    existing: ExistingFeatureMatrix
    change_kind: FeatureMatrixChangeKind
    regions: tuple[str, ...]


class FeatureMatrixChangedError(ValueError):
    """The existing feature matrix no longer matches what the user confirmed."""


class FeatureMatrixRemovedError(RuntimeError):
    """Feature extraction failed after the previous feature matrix was removed.

    The removal is not rolled back. `regions` are every region of the table,
    whose values in `feature_key` are gone.
    """

    def __init__(self, *, table_name: str, feature_key: str, regions: tuple[str, ...], error: Exception) -> None:
        super().__init__(f'Feature extraction failed; the previous feature matrix "{feature_key}" was removed: {error}')
        self.table_name = table_name
        self.feature_key = feature_key
        self.regions = regions


def check_existing_feature_matrix(
    sdata: SpatialData,
    *,
    table_name: str,
    feature_key: str,
    triplets: Sequence[FeatureExtractionTriplet],
    feature_names: Sequence[str],
) -> ExistingFeatureMatrix:
    """Classify `table.obsm[feature_key]` against a requested extraction, without changing anything.

    The checks mirror those `hp.tb.add_feature_matrix(...)` applies to an
    existing entry, so `"compatible"` means harpy accepts a regional update.
    Like harpy, backed SpatialData is checked against the store, without
    loading matrix values: an entry present only in memory was never saved and
    is incompatible, and unsaved changes to a stored entry are not seen.

    Raises `ValueError` when harpy would reject any feature update of the table
    because of its rows, so nothing is removed for an update that cannot succeed.
    """
    import harpy as hp

    table = get_table(sdata, table_name)
    selected_regions = {triplet.labels_name for triplet in triplets}
    unselected_regions = tuple(
        region for region in get_table_metadata(sdata, table_name).regions if region not in selected_regions
    )
    _validate_table_rows(sdata, table_name)
    memory_matrix = table.obsm[feature_key] if feature_key in table.obsm else None
    memory_metadata = _feature_metadata(table.uns, feature_key)

    if sdata.path is None:
        matrix, metadata = memory_matrix, memory_metadata
    else:
        matrix_path = ("obsm", feature_key)
        metadata_path = ("uns", _FEATURE_MATRICES_KEY, feature_key)
        # Backed mode returns stored matrices as handles, without loading values;
        # a DataFrame-valued entry is the exception and is read into memory.
        stored = hp.tb.io.read_table_components(
            sdata.path,
            table_name=table_name,
            components=[matrix_path, metadata_path],
            mode="backed",
            missing="omit",
        )
        matrix, metadata = stored.get(matrix_path), stored.get(metadata_path)
        if matrix is None and memory_matrix is not None:
            return ExistingFeatureMatrix("incompatible", "it is not saved", unselected_regions)

    if matrix is None:
        return ExistingFeatureMatrix(
            "absent",
            unselected_regions=unselected_regions,
            orphaned_metadata=metadata is not None or memory_metadata is not None,
        )

    reason = _incompatibility_reason(
        sdata,
        matrix,
        metadata,
        n_obs=table.n_obs,
        triplets=triplets,
        feature_names=feature_names,
    )
    return ExistingFeatureMatrix("compatible" if reason is None else "incompatible", reason, unselected_regions)


def calculate_feature_matrix(
    sdata: SpatialData,
    *,
    table_name: str,
    feature_key: str,
    triplets: Sequence[FeatureExtractionTriplet],
    feature_names: Sequence[str],
    create_table: bool = False,
    overwrite_feature_key: bool = False,
    confirmed_existing_entry: ExistingFeatureMatrixKind | None = None,
) -> FeatureMatrixCalculation:
    """Calculate `table.obsm[feature_key]` with harpy, replacing an existing entry when allowed.

    With `overwrite_feature_key=True`, a compatible existing matrix is updated
    for the selected labels elements only, and the values of other labels
    elements are kept. Any other existing matrix is removed with its metadata
    and calculated again, so the values of other labels elements are lost; a
    warning names them. Without it, an existing matrix raises, also for
    unbacked SpatialData, where harpy itself would merge silently.

    Parameters
    ----------
    sdata
        SpatialData object containing the labels, images and table.
    table_name
        Table to update, or to create when `create_table` is set.
    feature_key
        Key of the feature matrix in `table.obsm` and `table.uns["feature_matrices"]`.
    triplets
        Labels elements to calculate features for, with their coordinate
        systems, images and channels.
    feature_names
        Harpy feature names to calculate.
    create_table
        Whether harpy creates `table_name` as a new table.
    overwrite_feature_key
        Whether an existing feature matrix may be replaced.
    confirmed_existing_entry
        The kind of existing entry found before launching: for the widget, the
        kind the user confirmed in the overwrite dialog, or `"absent"` when there
        was no dialog. Checked again here before anything is removed or
        calculated. `None` acts on whatever is found.

    Raises
    ------
    FeatureMatrixChangedError
        The existing entry differs from `confirmed_existing_entry`. Nothing has
        been changed.
    FeatureMatrixRemovedError
        Extraction failed after an incompatible matrix was removed.
    """
    import harpy as hp

    if create_table and overwrite_feature_key:
        raise ValueError("Cannot overwrite a feature key while creating a new table.")

    labels_names = tuple(triplet.labels_name for triplet in triplets)
    if create_table:
        existing = ExistingFeatureMatrix("absent")
    else:
        existing = check_existing_feature_matrix(
            sdata,
            table_name=table_name,
            feature_key=feature_key,
            triplets=triplets,
            feature_names=feature_names,
        )
    if confirmed_existing_entry is not None and existing.kind != confirmed_existing_entry:
        raise FeatureMatrixChangedError(f'Feature matrix "{feature_key}" changed since you confirmed; calculate again.')
    if existing.kind != "absent" and not overwrite_feature_key:
        raise ValueError(
            f'Feature matrix "{feature_key}" already exists in table "{table_name}". '
            "Set `overwrite_feature_key=True` to replace it."
        )

    change_kind: FeatureMatrixChangeKind
    regions: tuple[str, ...]
    match existing.kind:
        case "absent":
            change_kind, regions = "created", labels_names
            if existing.orphaned_metadata:
                logger.info(
                    f'Replacing feature metadata "{feature_key}" in table "{table_name}", which has no feature matrix.'
                )
        case "compatible":
            change_kind, regions = "updated", labels_names
        case "incompatible":
            # The matrix is removed and calculated again, so every region changes.
            change_kind, regions = "updated", get_table_metadata(sdata, table_name).regions
            # Removing first is not atomic: validate what harpy would otherwise only
            # reject after the removal. `check_existing_feature_matrix(...)` above
            # already validated the table rows.
            _validate_labels_have_observations(sdata, table_name, labels_names)
            lost = (
                f" Values for labels element(s) {_quoted_names(existing.unselected_regions)} are lost."
                if existing.unselected_regions
                else ""
            )
            logger.warning(
                f'Replacing the entire feature matrix "{feature_key}" in table "{table_name}", '
                f"because {existing.reason}.{lost}"
            )
            hp.tb.io.remove_table_components(
                sdata,
                table_name=table_name,
                components=[("obsm", feature_key), ("uns", _FEATURE_MATRICES_KEY, feature_key)],
            )

    try:
        hp.tb.add_feature_matrix(
            sdata=sdata,
            labels_name=_resolve_harpy_labels_name_parameter(triplets),
            image_name=_resolve_harpy_image_name_parameter(triplets, feature_names),
            table_name=None if create_table else table_name,
            output_table_name=table_name if create_table else None,
            feature_key=feature_key,
            features=list(feature_names),
            channels=_resolve_harpy_channel_parameter(triplets, feature_names),
            feature_matrices_key=_FEATURE_MATRICES_KEY,
            # Table-name collisions are blocked in the UI/controller; replacing an
            # existing table would need a broader table and .obsm replacement policy.
            overwrite_output_table=False,
            # The existing entry was checked above. harpy's flag only concerns the
            # store, and is still needed there for orphaned stored metadata.
            overwrite_feature_key=not create_table,
            to_coordinate_system=_resolve_harpy_coordinate_system_parameter(triplets),
        )
    except Exception as error:
        if existing.kind == "incompatible":
            raise FeatureMatrixRemovedError(
                table_name=table_name,
                feature_key=feature_key,
                regions=regions,
                error=error,
            ) from error
        raise

    return FeatureMatrixCalculation(existing, change_kind, regions)


def _feature_metadata(uns: Mapping[str, Any], feature_key: str) -> object:
    feature_matrices = uns.get(_FEATURE_MATRICES_KEY)
    return feature_matrices.get(feature_key) if isinstance(feature_matrices, Mapping) else None


def _incompatibility_reason(
    sdata: SpatialData,
    matrix: Any,
    metadata: object,
    *,
    n_obs: int,
    triplets: Sequence[FeatureExtractionTriplet],
    feature_names: Sequence[str],
) -> str | None:
    """Return why harpy would reject a regional update of `matrix`, or `None` when it accepts one.

    Reads only the type, shape and dtype of `matrix` and loads no values.

    Mirrors the existing-matrix checks of `add_feature_matrix(...)` and
    `_merge_feature_matrix_metadata(...)` in `harpy/table/_add_feature_matrix.py`,
    because harpy has no public check that runs before feature computation.
    Replace this function and its helpers once
    https://github.com/saeyslab/harpy/issues/281 is released.
    """
    if metadata is None:
        return "it has no harpy feature metadata"
    if not isinstance(metadata, Mapping) or metadata.get("source_kind") != HARPY_ADD_FEATURE_MATRIX_SOURCE_KIND:
        if isinstance(metadata, Mapping) and metadata.get("source_kind") == CUSTOM_OBSM_SOURCE_KIND:
            return "it is registered as a custom matrix"
        return "its metadata does not come from harpy feature extraction"
    if metadata.get("schema_version") != FEATURE_MATRIX_METADATA_SCHEMA_VERSION:
        return "its metadata has an unsupported schema version"

    feature_columns, source_channels = _expected_feature_columns(sdata, triplets, feature_names)
    if not np.array_equal(metadata.get("feature_columns"), feature_columns):
        return "it was calculated with different features or channels"
    if not np.array_equal(metadata.get("source_channels"), source_channels):
        return "it was calculated from different channels"

    if not _is_dense_matrix(matrix):
        return "it is not a dense matrix"
    if tuple(matrix.shape) != (n_obs, len(feature_columns)):
        return (
            f"its shape {tuple(matrix.shape)} does not match {n_obs} observations "
            f"and {len(feature_columns)} feature columns"
        )
    if not np.can_cast(np.float64, matrix.dtype, casting="safe"):
        return f"its dtype {matrix.dtype} cannot hold float64 values"
    return None


def _expected_feature_columns(
    sdata: SpatialData,
    triplets: Sequence[FeatureExtractionTriplet],
    feature_names: Sequence[str],
) -> tuple[list[str], list[str] | None]:
    """Return the `feature_columns` and `source_channels` harpy writes for a request.

    harpy puts intensity columns (`{feature}__{channel}`) before morphology
    columns and keeps the requested order within each group.
    """
    requested = list(dict.fromkeys(feature_names))
    intensity_features = [name for name in requested if name in _INTENSITY_FEATURES]
    morphology_features = [name for name in requested if name not in _INTENSITY_FEATURES]
    if not intensity_features:
        return morphology_features, None

    # Both raise for requests harpy rejects: a missing image, or no channels.
    _resolve_harpy_image_name_parameter(triplets, feature_names)
    channels = _resolve_harpy_channel_parameter(triplets, feature_names)
    source_channels = _resolve_source_channels(sdata, str(triplets[0].image_name), channels)
    intensity_columns = [f"{feature}__{channel}" for feature in intensity_features for channel in source_channels]
    return [*intensity_columns, *morphology_features], source_channels


def _resolve_source_channels(
    sdata: SpatialData,
    image_name: str,
    channels: Sequence[FeatureExtractionChannel] | None,
) -> list[str]:
    """Resolve a channel selection to channel names, as harpy's `_resolve_channels(...)` does.

    `None` selects all channels, an integer is an index and a string is a name.
    Duplicates are dropped, and an unnamed channel becomes `channel_{index}`.
    """
    if image_name not in sdata.images:
        raise ValueError(f"Image element `{image_name}` is not available in the selected SpatialData object.")
    image_element = sdata.images[image_name]
    image = image_element if isinstance(image_element, DataArray) else next(iter(image_element["scale0"].values()))
    available_channels = list(image.c.data)

    if channels is None:
        indices = list(range(len(available_channels)))
    else:
        index_by_name = {str(name): index for index, name in enumerate(available_channels)}
        indices = []
        for channel in channels:
            if isinstance(channel, (int, np.integer)) and not isinstance(channel, bool):
                if channel < 0 or channel >= len(available_channels):
                    raise ValueError(
                        f"Channel index `{channel}` is out of range for image element `{image_name}`. "
                        f"Available indices are 0 through {len(available_channels) - 1}."
                    )
                index = int(channel)
            else:
                if str(channel) not in index_by_name:
                    raise ValueError(
                        f"Channel `{channel}` was not found in image element `{image_name}`. "
                        f"Available channels are {[str(name) for name in available_channels]}."
                    )
                index = index_by_name[str(channel)]
            if index not in indices:
                indices.append(index)

    return [
        f"channel_{index}" if available_channels[index] is None else str(available_channels[index]) for index in indices
    ]


def _is_dense_matrix(matrix: Any) -> bool:
    """Return whether harpy treats `matrix` as dense: NumPy, Zarr, or Dask with NumPy blocks."""
    meta = matrix._meta if isinstance(matrix, da.Array) else matrix
    return isinstance(meta, (np.ndarray, zarr.Array))


def _validate_labels_have_observations(sdata: SpatialData, table_name: str, labels_names: Sequence[str]) -> None:
    table = get_table(sdata, table_name)
    region_key = get_table_metadata(sdata, table_name).region_key
    observed_regions = set(table.obs[region_key].unique())
    for labels_name in labels_names:
        if labels_name not in sdata.labels:
            raise ValueError(f"Labels element `{labels_name}` does not exist in the selected SpatialData object.")
        if labels_name not in observed_regions:
            raise ValueError(f"Labels element `{labels_name}` has no observations in table `{table_name}`.")


def _validate_table_rows(sdata: SpatialData, table_name: str) -> None:
    """Raise when harpy would reject every feature update of the table because of its rows.

    Runs harpy's own row validation from `_existing_feature_matrix(...)`: the
    rows in memory need valid, unique region/instance pairs and, for backed
    SpatialData, must match the store in value and order. Only the stored index
    and identity columns are read. Remove once
    https://github.com/saeyslab/harpy/issues/281 is released: harpy's public
    check runs this validation.
    """
    table, group = _component_update_destination(sdata, table_name=table_name)
    if group is None:
        _validated_observation_pairs(table.obs, table.uns[TableModel.ATTRS_KEY], label="In-memory observation")
    else:
        _check_in_memory_versus_storage_axes(
            table=table, group=group, indices={"obs": table.obs.index}, stored_new_raw_var=None
        )


def _quoted_names(names: Sequence[str]) -> str:
    return ", ".join(f'"{name}"' for name in names)
