from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from shapely.geometry import Polygon

from spatiato.core.spatial_query.canonical_centers import CanonicalCentersRegion


@dataclass(frozen=True)
class CanonicalCenterQueryRequest:
    """Immutable computational inputs for one canonical-center query.

    Parameters
    ----------
    canonical_centers
        Rows, instance IDs and canonical centers of the selected labels
        element, in its intrinsic coordinate frame.
    polygons
        Validated Polygon snapshots in the source Shapes element's intrinsic
        x/y coordinate frame.
    polygons_to_labels_affine
        Homogeneous 3 x 3 affine mapping polygon x/y coordinates into the
        selected labels element's intrinsic x/y coordinate frame.
    """

    canonical_centers: CanonicalCentersRegion
    polygons: tuple[Polygon, ...] = field(repr=False, compare=False)
    polygons_to_labels_affine: NDArray[np.float64] = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.canonical_centers, CanonicalCentersRegion):
            raise TypeError("Canonical-center query requires a CanonicalCentersRegion.")

        polygons = tuple(self.polygons)
        if not polygons:
            raise ValueError("Canonical-center query requires at least one Polygon geometry.")
        if any(not isinstance(polygon, Polygon) for polygon in polygons):
            raise TypeError("Canonical-center query geometries must be Shapely Polygon objects.")

        affine = _read_only(np.array(self.polygons_to_labels_affine, dtype=np.float64))
        if affine.shape != (3, 3):
            raise ValueError("Polygons-to-labels affine must be a 3 x 3 homogeneous matrix.")
        if not np.isfinite(affine).all():
            raise ValueError("Polygons-to-labels affine must contain only finite values.")

        object.__setattr__(self, "polygons", polygons)
        object.__setattr__(self, "polygons_to_labels_affine", affine)

    @property
    def table_name(self) -> str:
        """Return the table name of the queried canonical centers."""
        return self.canonical_centers.table_name

    @property
    def labels_name(self) -> str:
        """Return the labels element of the queried canonical centers."""
        return self.canonical_centers.labels_name


@dataclass(frozen=True)
class CanonicalCenterQueryResult:
    """The queried canonical centers and the sorted instance IDs inside the annotation.

    Parameters
    ----------
    canonical_centers
        Rows, instance IDs and canonical centers of the labels element the
        query evaluated.
    matched_instance_ids
        Unique, ascending instance IDs whose canonical centers intersect the
        annotation polygons: a subset of ``canonical_centers.instance_ids``,
        which holds every instance the query evaluated.

    Notes
    -----
    ``canonical_centers`` keeps the rows, instance IDs and centers the query
    used, not only the matches. The matched instance IDs are a geometric
    decision made from exactly these centers. Before the annotation is applied,
    ``_require_current_query_provenance()`` in
    ``spatiato.core.spatial_query.annotation`` compares them with the table's
    current rows and centers, so an old query cannot annotate rows after the
    centers or the table rows changed.
    """

    canonical_centers: CanonicalCentersRegion
    matched_instance_ids: NDArray[np.integer] = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.canonical_centers, CanonicalCentersRegion):
            raise TypeError("Canonical-center query result requires a CanonicalCentersRegion.")
        matched_instance_ids = np.asarray(self.matched_instance_ids)
        if matched_instance_ids.ndim != 1 or matched_instance_ids.dtype.kind not in "iu":
            raise ValueError("Canonical-center query result instance IDs must be a one-dimensional integer array.")
        if len(matched_instance_ids) > 1 and np.any(matched_instance_ids[1:] <= matched_instance_ids[:-1]):
            raise ValueError("Canonical-center query result instance IDs must be unique and sorted.")
        if (
            len(matched_instance_ids)
            and not np.isin(matched_instance_ids, self.canonical_centers.instance_ids, assume_unique=True).all()
        ):
            raise ValueError("Canonical-center query result instance IDs must be a subset of the selected region.")
        object.__setattr__(self, "matched_instance_ids", _read_only(matched_instance_ids))

    @property
    def eligible_instance_count(self) -> int:
        """Return the number of instances evaluated by the query."""
        return len(self.canonical_centers.instance_ids)

    @property
    def matched_instance_count(self) -> int:
        """Return the number of matching instances."""
        return len(self.matched_instance_ids)


def _read_only(array: np.ndarray) -> np.ndarray:
    result = np.array(array, copy=True)
    result.flags.writeable = False
    return result
