"""Canonical centers for Spatial Query, computed and validated by harpy.

harpy owns the canonical centers of a table: `obsm["spatial_canonical"]`, one
`(z, y, x)` center per row in the intrinsic `scale0` frame of its labels
element, and the record in `uns["spatial_coordinates"]["spatial_canonical"]`.
They are all or nothing: present and valid for every labels element the table
annotates, or absent. Spatial Query checks them, has harpy recompute them all
when they are absent or invalid, and reads the rows of one labels element.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, get_args

import harpy as hp
import numpy as np
from harpy.image import get_dataarray
from harpy.table.canonical_centers import CANONICAL_OBSM_KEY, SPATIAL_COORDINATES_KEY, validate_canonical_centers

from spatiato.core.spatialdata import get_table, get_table_metadata
from spatiato.core.table_matrix import read_matrix_rows

if TYPE_CHECKING:
    from spatialdata import SpatialData

CanonicalCentersState = Literal["absent", "valid", "invalid"]
CanonicalCentersChange = Literal["created", "rebuilt"]


@dataclass(frozen=True)
class CanonicalCentersStatus:
    """Whether a table has valid canonical centers.

    - `"absent"`: the table has no canonical centers;
    - `"valid"`: harpy accepts them;
    - `"invalid"`: harpy rejects them, for example stale, from another labels
      element, or an older record version. `reason` is harpy's message.

    `reason` is set only for `"invalid"`.
    """

    table_name: str
    state: CanonicalCentersState
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.state not in get_args(CanonicalCentersState):
            raise ValueError(f"Unsupported canonical centers state: {self.state!r}.")
        if (self.state == "invalid") != (self.reason is not None):
            raise ValueError('`reason` is set exactly for state "invalid".')


@dataclass(frozen=True)
class CanonicalRegionCenters:
    """The canonical centers of one labels element, with the table rows they belong to.

    `row_positions`, `instance_ids` and `centers` are aligned and read-only:
    `centers[i]` is the `(z, y, x)` center of instance `instance_ids[i]`, at
    table row `row_positions[i]`.
    """

    table_name: str
    labels_name: str
    row_positions: np.ndarray
    instance_ids: np.ndarray
    centers: np.ndarray

    def __post_init__(self) -> None:
        n_rows = len(self.row_positions)
        if (
            self.row_positions.shape != (n_rows,)
            or self.instance_ids.shape != (n_rows,)
            or self.centers.shape != (n_rows, 3)
        ):
            raise ValueError(
                "Row positions, instance IDs and centers must describe the same rows: shapes (n,), (n,) and (n, 3)."
            )


def canonical_centers_status(sdata: SpatialData, *, table_name: str) -> CanonicalCentersStatus:
    """Return whether the table has valid canonical centers, without changing anything.

    Reads the canonical matrix and the labels elements' metadata, no labels
    pixels.
    """
    # harpy raises for invalid centers, including a matrix or a record without the
    # other. It returns None only when the table has neither, and the metadata of
    # valid centers otherwise.
    try:
        metadata = validate_canonical_centers(sdata, table_name=table_name)
    except ValueError as error:
        return CanonicalCentersStatus(table_name, "invalid", reason=str(error))
    if metadata is None:
        return CanonicalCentersStatus(table_name, "absent")
    return CanonicalCentersStatus(table_name, "valid")


def compute_canonical_centers(sdata: SpatialData, *, table_name: str) -> CanonicalCentersChange:
    """Have harpy compute the canonical centers of every labels element the table annotates.

    Replaces existing canonical centers, valid or not. For backed SpatialData,
    harpy also writes them to the store and reattaches the matrix lazily.
    Returns `"created"` when the table in memory had neither the matrix nor its
    record (status `"absent"`), and `"rebuilt"` otherwise. The store is not
    checked: harpy replaces what is stored either way. harpy's errors pass
    through, with the table unchanged.
    """
    table = get_table(sdata, table_name)
    registry = table.uns.get(SPATIAL_COORDINATES_KEY)
    # "created" only when `canonical_centers_status` reports "absent": any matrix or
    # record, even one without the other, makes this a rebuild.
    had_centers = CANONICAL_OBSM_KEY in table.obsm or (isinstance(registry, Mapping) and CANONICAL_OBSM_KEY in registry)
    # `overwrite` only concerns the store: it lets harpy replace centers stored before.
    hp.tb.add_canonical_centers(sdata, table_name=table_name, overwrite=True)
    return "rebuilt" if had_centers else "created"


def read_canonical_region_centers(
    sdata: SpatialData,
    *,
    table_name: str,
    labels_name: str,
) -> CanonicalRegionCenters:
    """Read the canonical centers of one labels element, and only its rows.

    Its rows are those whose region is `labels_name`, in table order, and their
    instance IDs come from the table's instance column. The caller checks first
    that the centers are valid (`canonical_centers_status(...)`).
    """
    table = get_table(sdata, table_name)
    table_metadata = get_table_metadata(sdata, table_name)
    if not table_metadata.annotates(labels_name):
        raise ValueError(f"Table `{table_name}` does not annotate labels element `{labels_name}`.")
    row_positions = np.flatnonzero(np.asarray(table.obs[table_metadata.region_key] == labels_name, dtype=bool))
    if row_positions.size == 0:
        raise ValueError(f"Table `{table_name}` has no rows for labels element `{labels_name}`.")

    instance_ids = table.obs[table_metadata.instance_key].to_numpy()[row_positions]
    if instance_ids.dtype.kind not in "iu" or np.any(instance_ids <= 0):
        raise ValueError(f"Instance IDs of labels element `{labels_name}` must be positive integers (0 is background).")
    if np.unique(instance_ids).size != instance_ids.size:
        raise ValueError(f"Instance IDs of labels element `{labels_name}` must be unique.")
    if CANONICAL_OBSM_KEY not in table.obsm:
        raise ValueError(f"Table `{table_name}` has no canonical centers.")
    centers = np.asarray(read_matrix_rows(table.obsm[CANONICAL_OBSM_KEY], row_positions), dtype=np.float64)

    return CanonicalRegionCenters(
        table_name=table_name,
        labels_name=labels_name,
        row_positions=_read_only(row_positions.astype(np.int64)),
        instance_ids=_read_only(instance_ids),
        centers=_read_only(centers),
    )


def is_canonical_query_labels(sdata: SpatialData, labels_name: str) -> bool:
    """Return whether Spatial Query supports a labels element: 2-D, with integer labels.

    Reads the `scale0` metadata only. harpy also computes canonical centers for
    3-D labels, but the query compares their y and x with 2-D shapes.
    """
    # `get_dataarray` also looks in `sdata.images`.
    if labels_name not in sdata.labels:
        return False
    try:
        scale0 = get_dataarray(sdata, labels_name, scale="scale0")
    except ValueError:
        return False
    return tuple(scale0.dims) == ("y", "x") and np.dtype(scale0.dtype).kind in "iu"


def _read_only(array: np.ndarray) -> np.ndarray:
    result = np.array(array, copy=True)
    result.flags.writeable = False
    return result
