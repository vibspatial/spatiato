from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

import pandas as pd
from harpy.table.canonical_centers import CANONICAL_OBSM_KEY
from harpy.utils._keys import _FEATURE_MATRICES_KEY

from spatiato.core.table_matrix import is_sparse_matrix

if TYPE_CHECKING:
    from anndata import AnnData


CUSTOM_OBSM_FEATURE_NAME = "custom_obsm"
CUSTOM_OBSM_SOURCE_KIND = "custom_obsm"
HARPY_ADD_FEATURE_MATRIX_SOURCE_KIND = "harpy_add_feature_matrix"
FEATURE_MATRIX_METADATA_SCHEMA_VERSION = 1
FeatureMatrixSourceKind = Literal["harpy_add_feature_matrix", "custom_obsm"]
FEATURE_MATRIX_SOURCE_KINDS: tuple[FeatureMatrixSourceKind, ...] = (
    HARPY_ADD_FEATURE_MATRIX_SOURCE_KIND,
    CUSTOM_OBSM_SOURCE_KIND,
)
FeatureMatrixMetadataStatus = Literal[
    "missing_matrix",
    "invalid_matrix",
    "unregistered",
    "registered_valid",
    "registered_mismatched",
]
FEATURE_MATRIX_METADATA_STATUSES: tuple[FeatureMatrixMetadataStatus, ...] = (
    "missing_matrix",
    "invalid_matrix",
    "unregistered",
    "registered_valid",
    "registered_mismatched",
)
_FEATURE_MATRIX_METADATA_STATUSES_REQUIRING_ERROR: tuple[FeatureMatrixMetadataStatus, ...] = (
    "missing_matrix",
    "invalid_matrix",
    "registered_mismatched",
)
# NumPy dtype kinds of a numeric feature matrix: bool, signed and unsigned integer, float.
_NUMERIC_DTYPE_KINDS = "biuf"
# `.obsm` keys harpy reserves for spatial coordinates: harpy rejects a feature
# matrix under either, and a table registering one fails `hp.tb.validate_table`.
RESERVED_OBSM_KEYS = frozenset({CANONICAL_OBSM_KEY, "spatial"})


def validate_feature_key_not_reserved(feature_key: str) -> None:
    """Raise `ValueError` when `feature_key` is reserved for spatial coordinates."""
    if feature_key in RESERVED_OBSM_KEYS:
        raise ValueError(
            f'`.obsm` key "{feature_key}" is reserved for spatial coordinates and cannot hold a feature matrix.'
        )


@dataclass(frozen=True)
class FeatureMatrixMetadataState:
    """Read-only registration state for one `.obsm` feature matrix key.

    Parameters
    ----------
    feature_key
        Key inspected in `table.obsm`.
    status
        Registration state derived from the live `.obsm` matrix and any
        matching metadata in `table.uns["feature_matrices"]`.
    n_features
        Number of feature columns in the live `.obsm` matrix, when that matrix
        exists and can be normalized as a feature matrix.
    metadata_n_features
        Number of feature columns described by metadata, when metadata exists
        and `feature_columns` can be parsed.
    source_kind
        Validated `source_kind` metadata value, when registered metadata exists
        and declares an allowed source kind.
    error
        Human-readable reason for missing, invalid, or mismatched states.
    """

    feature_key: str
    status: FeatureMatrixMetadataStatus
    n_features: int | None = None
    metadata_n_features: int | None = None
    source_kind: FeatureMatrixSourceKind | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if self.status not in FEATURE_MATRIX_METADATA_STATUSES:
            raise ValueError(f"Unsupported feature matrix metadata status: {self.status!r}.")
        if self.source_kind is not None and self.source_kind not in FEATURE_MATRIX_SOURCE_KINDS:
            raise ValueError(f"Unsupported feature matrix metadata source kind: {self.source_kind!r}.")
        if self.status in _FEATURE_MATRIX_METADATA_STATUSES_REQUIRING_ERROR and not self.error:
            raise ValueError(f"Feature matrix metadata status {self.status!r} must include an error message.")

    @property
    def is_custom_obsm(self) -> bool:
        """Return whether metadata declares custom `.obsm` registration."""
        return self.source_kind == CUSTOM_OBSM_SOURCE_KIND


def normalize_feature_columns(feature_metadata: Mapping[str, object]) -> tuple[str, ...]:
    """Return the required feature column schema from Harpy feature metadata."""
    raw_columns = feature_metadata.get("feature_columns")
    ndim = getattr(raw_columns, "ndim", None)
    if ndim is not None and ndim != 1:
        raise ValueError("Feature metadata `feature_columns` must be a 1-dimensional sequence.")
    if hasattr(raw_columns, "tolist") and not isinstance(raw_columns, str):
        raw_columns = raw_columns.tolist()
    if isinstance(raw_columns, str) or not isinstance(raw_columns, Sequence):
        raise ValueError("Feature metadata `feature_columns` must be a sequence.")

    columns = tuple(str(value) for value in raw_columns)
    if not columns:
        raise ValueError("Feature metadata must contain at least one `feature_columns` entry.")
    if any(not column for column in columns):
        raise ValueError("Feature metadata `feature_columns` must not contain empty strings.")
    return columns


def feature_matrix_n_features(feature_matrix: Any, n_obs: int) -> int:
    """Return the number of feature columns, checking the matrix from its shape and dtype only.

    Checks that the matrix is 2-D, has one row per observation and holds
    numeric values, without reading values, so a lazy or backed matrix stays
    unread. A 1-D matrix counts as one column. DataFrames are rejected: harpy
    never writes or updates a feature matrix as one. Raises `ValueError`.
    """
    if isinstance(feature_matrix, pd.DataFrame):
        raise ValueError(
            "Feature matrices must be arrays, not DataFrames; store the values as an array, "
            "for example with `.to_numpy()`."
        )
    shape = getattr(feature_matrix, "shape", None)
    dtype = getattr(feature_matrix, "dtype", None)
    if shape is None or dtype is None:
        raise ValueError(f"Unsupported feature matrix type {type(feature_matrix).__name__!r}.")
    # `len(shape)`, not `ndim`: `CSRDataset`/`CSCDataset` have no `ndim`.
    if len(shape) not in (1, 2):
        raise ValueError("Feature matrices stored in `.obsm` must be 2-dimensional.")
    if shape[0] != n_obs:
        raise ValueError(f"Feature matrix has {shape[0]} rows but the table has {n_obs} observations.")
    # Bool, signed and unsigned integer, float. Without reading values, an
    # `object` matrix cannot be known to hold numbers.
    if dtype.kind not in _NUMERIC_DTYPE_KINDS:
        raise ValueError(f"Feature matrix dtype `{dtype}` is not numeric.")
    return 1 if len(shape) == 1 else int(shape[1])


def inspect_feature_matrix_metadata(table: AnnData, feature_key: str) -> FeatureMatrixMetadataState:
    """Return the non-mutating metadata registration state for one `.obsm` key."""
    resolved_feature_key = _normalize_nonempty_str(feature_key, "feature_key")
    if resolved_feature_key not in table.obsm:
        return FeatureMatrixMetadataState(
            feature_key=resolved_feature_key,
            status="missing_matrix",
            error=f'Feature matrix "{resolved_feature_key}" is not available in ".obsm".',
        )

    try:
        n_features = feature_matrix_n_features(table.obsm[resolved_feature_key], table.n_obs)
    except ValueError as error:
        return FeatureMatrixMetadataState(
            feature_key=resolved_feature_key,
            status="invalid_matrix",
            error=str(error),
        )

    feature_matrices_value = table.uns.get(_FEATURE_MATRICES_KEY)
    if feature_matrices_value is None:
        return FeatureMatrixMetadataState(
            feature_key=resolved_feature_key,
            status="unregistered",
            n_features=n_features,
        )
    if not isinstance(feature_matrices_value, Mapping):
        return FeatureMatrixMetadataState(
            feature_key=resolved_feature_key,
            status="registered_mismatched",
            n_features=n_features,
            error=f"Feature matrix metadata `.uns[{_FEATURE_MATRICES_KEY!r}]` must be a mapping.",
        )

    feature_metadata = feature_matrices_value.get(resolved_feature_key)
    if feature_metadata is None:
        return FeatureMatrixMetadataState(
            feature_key=resolved_feature_key,
            status="unregistered",
            n_features=n_features,
        )
    if not isinstance(feature_metadata, Mapping):
        return FeatureMatrixMetadataState(
            feature_key=resolved_feature_key,
            status="registered_mismatched",
            n_features=n_features,
            error=(
                f"Feature matrix metadata `.uns[{_FEATURE_MATRICES_KEY!r}]"
                f"[{resolved_feature_key!r}]` must be a mapping."
            ),
        )

    try:
        source_kind = normalize_feature_matrix_source_kind(feature_metadata)
    except ValueError as error:
        return FeatureMatrixMetadataState(
            feature_key=resolved_feature_key,
            status="registered_mismatched",
            n_features=n_features,
            error=str(error),
        )

    try:
        metadata_n_features = len(normalize_feature_columns(feature_metadata))
    except (TypeError, ValueError) as error:
        return FeatureMatrixMetadataState(
            feature_key=resolved_feature_key,
            status="registered_mismatched",
            n_features=n_features,
            source_kind=source_kind,
            error=str(error),
        )

    if metadata_n_features != n_features:
        return FeatureMatrixMetadataState(
            feature_key=resolved_feature_key,
            status="registered_mismatched",
            n_features=n_features,
            metadata_n_features=metadata_n_features,
            source_kind=source_kind,
            error=(
                f'Feature matrix "{resolved_feature_key}" has {n_features} column(s), '
                f"but its metadata describes {metadata_n_features} feature column(s)."
            ),
        )

    return FeatureMatrixMetadataState(
        feature_key=resolved_feature_key,
        status="registered_valid",
        n_features=n_features,
        metadata_n_features=metadata_n_features,
        source_kind=source_kind,
    )


def register_feature_matrix_metadata(
    table: AnnData,
    feature_key: str,
    *,
    feature_columns: Sequence[str] | str | None = None,
    features: Sequence[str] | str | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    """Register classifier schema metadata for an existing `.obsm` matrix."""
    resolved_feature_key = _normalize_nonempty_str(feature_key, "feature_key")
    validate_feature_key_not_reserved(resolved_feature_key)
    if resolved_feature_key not in table.obsm:
        raise ValueError(f'Feature matrix "{resolved_feature_key}" is not available in ".obsm".')

    raw_feature_matrix = table.obsm[resolved_feature_key]
    n_features = feature_matrix_n_features(raw_feature_matrix, table.n_obs)
    resolved_feature_columns = _normalize_feature_columns(
        feature_columns,
        feature_key=resolved_feature_key,
        n_features=n_features,
    )
    resolved_features = _normalize_features(features)

    feature_matrices_value = table.uns.get(_FEATURE_MATRICES_KEY)
    if feature_matrices_value is None:
        feature_matrices = {}
    elif isinstance(feature_matrices_value, Mapping):
        feature_matrices = dict(feature_matrices_value)
    else:
        raise ValueError(f"Feature matrix metadata `.uns[{_FEATURE_MATRICES_KEY!r}]` must be a mapping.")

    if resolved_feature_key in feature_matrices and not overwrite:
        raise ValueError(
            f'Feature matrix "{resolved_feature_key}" already has metadata in '
            f"`.uns[{_FEATURE_MATRICES_KEY!r}][{resolved_feature_key!r}]`. Pass `overwrite=True` to replace it."
        )

    metadata: dict[str, object] = {
        "feature_columns": list(resolved_feature_columns),
        "schema_version": FEATURE_MATRIX_METADATA_SCHEMA_VERSION,
        "backend": "sparse" if is_sparse_matrix(raw_feature_matrix) else "numpy",
        # `feature_matrix_n_features(...)` above guarantees a dtype.
        "dtype": str(raw_feature_matrix.dtype),
        "features": list(resolved_features),
        "source_kind": CUSTOM_OBSM_SOURCE_KIND,
    }
    feature_matrices[resolved_feature_key] = metadata
    table.uns[_FEATURE_MATRICES_KEY] = feature_matrices
    return dict(metadata)


def is_custom_obsm_feature_metadata(feature_metadata: Mapping[str, object]) -> bool:
    """Return whether feature metadata came from custom `.obsm` registration."""
    return normalize_feature_matrix_source_kind(feature_metadata) == CUSTOM_OBSM_SOURCE_KIND


def normalize_feature_matrix_source_kind(feature_metadata: Mapping[str, object]) -> FeatureMatrixSourceKind:
    """Return a valid feature-matrix metadata `source_kind` value."""
    source_kind = feature_metadata.get("source_kind")
    if not isinstance(source_kind, str) or source_kind not in FEATURE_MATRIX_SOURCE_KINDS:
        allowed = ", ".join(f"`{kind}`" for kind in FEATURE_MATRIX_SOURCE_KINDS)
        raise ValueError(f"Feature metadata `source_kind` must be one of {allowed}; got {source_kind!r}.")
    return cast(FeatureMatrixSourceKind, source_kind)


def _normalize_feature_columns(
    feature_columns: Sequence[str] | str | None,
    *,
    feature_key: str,
    n_features: int,
) -> tuple[str, ...]:
    if feature_columns is None:
        return tuple(f"{feature_key}_{index}" for index in range(n_features))

    columns = _normalize_str_sequence(feature_columns, name="feature_columns")
    if len(columns) != n_features:
        raise ValueError(
            f"`feature_columns` describes {len(columns)} feature column(s), "
            f"but `.obsm[{feature_key!r}]` has {n_features} column(s)."
        )
    if len(set(columns)) != len(columns):
        raise ValueError("`feature_columns` must not contain duplicate names.")
    return columns


def _normalize_features(features: Sequence[str] | str | None) -> tuple[str, ...]:
    if features is None:
        return (CUSTOM_OBSM_FEATURE_NAME,)

    values = _normalize_str_sequence(features, name="features")
    if len(set(values)) != len(values):
        raise ValueError("`features` must not contain duplicate names.")
    return values


def _normalize_str_sequence(value: Sequence[str] | str, *, name: str) -> tuple[str, ...]:
    values = [value] if isinstance(value, str) else list(value)
    if not values:
        raise ValueError(f"`{name}` must contain at least one value.")

    normalized = tuple(str(item).strip() for item in values)
    if any(not item for item in normalized):
        raise ValueError(f"`{name}` must not contain empty values.")
    return normalized


def _normalize_nonempty_str(value: str, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must not be empty.")
    return normalized
