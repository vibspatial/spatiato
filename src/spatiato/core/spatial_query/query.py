from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import shapely
from shapely import affinity
from spatialdata.transformations import get_transformation_between_coordinate_systems

from spatiato.core.shapes_annotation import validate_existing_shapes_source_geodataframe
from spatiato.core.spatial_query.canonical_centers import CanonicalCentersRegion, is_canonical_query_labels
from spatiato.core.spatial_query.query_models import CanonicalCenterQueryRequest, CanonicalCenterQueryResult

if TYPE_CHECKING:
    from spatialdata import SpatialData


def build_canonical_center_query_request(
    sdata: SpatialData,
    *,
    shapes_name: str,
    coordinate_system: str,
    canonical_centers: CanonicalCentersRegion,
) -> CanonicalCenterQueryRequest:
    """Validate and snapshot the current inputs for one containment query."""
    if not isinstance(canonical_centers, CanonicalCentersRegion):
        raise TypeError("Canonical-center query construction requires a CanonicalCentersRegion.")
    if not isinstance(shapes_name, str) or not shapes_name:
        raise ValueError("Query Shapes name must be a non-empty string.")
    if not isinstance(coordinate_system, str) or not coordinate_system:
        raise ValueError("Query coordinate system must be a non-empty string.")
    if shapes_name not in sdata.shapes:
        raise ValueError(f"Shapes element `{shapes_name}` is not available in the selected SpatialData object.")

    labels_name = canonical_centers.labels_name
    # The query compares the centers' y and x with 2-D shapes.
    if not is_canonical_query_labels(sdata, labels_name):
        raise ValueError(
            f"Labels element `{labels_name}` is not a 2-D integer labels element of the selected SpatialData object."
        )

    shapes_element = validate_existing_shapes_source_geodataframe(sdata.shapes[shapes_name])
    labels_element = sdata.labels[labels_name]
    shapes_to_labels = get_transformation_between_coordinate_systems(
        sdata,
        source_coordinate_system=shapes_element,
        target_coordinate_system=labels_element,
        intermediate_coordinate_systems=coordinate_system,
    )
    polygons_to_labels_affine = shapes_to_labels.to_affine_matrix(
        input_axes=("x", "y"),
        output_axes=("x", "y"),
    )

    return CanonicalCenterQueryRequest(
        canonical_centers=canonical_centers,
        polygons=tuple(shapes_element.geometry),
        polygons_to_labels_affine=polygons_to_labels_affine,
    )


def evaluate_canonical_center_query(
    request: CanonicalCenterQueryRequest,
) -> CanonicalCenterQueryResult:
    """Return instances whose canonical centers intersect the annotation union."""
    if not isinstance(request, CanonicalCenterQueryRequest):
        raise TypeError("Canonical-center query evaluation requires a CanonicalCenterQueryRequest.")

    region_in_polygon_coordinates = shapely.union_all(request.polygons)
    region_in_labels = affinity.affine_transform(
        region_in_polygon_coordinates,
        _to_shapely_affine(request.polygons_to_labels_affine),
    )

    centers = request.canonical_centers.centers
    center_y = centers[:, 1]
    center_x = centers[:, 2]
    min_x, min_y, max_x, max_y = region_in_labels.bounds
    candidates = (center_x >= min_x) & (center_x <= max_x) & (center_y >= min_y) & (center_y <= max_y)
    candidate_positions = np.flatnonzero(candidates)

    if len(candidate_positions) == 0:
        matching_ids = request.canonical_centers.instance_ids[:0]
    else:
        shapely.prepare(region_in_labels)
        inside = np.asarray(
            shapely.intersects_xy(
                region_in_labels,
                center_x[candidate_positions],
                center_y[candidate_positions],
            ),
            dtype=bool,
        )
        matching_ids = np.sort(request.canonical_centers.instance_ids[candidate_positions[inside]])

    return CanonicalCenterQueryResult(
        canonical_centers=request.canonical_centers,
        matched_instance_ids=matching_ids,
    )


def _to_shapely_affine(matrix: np.ndarray) -> tuple[float, float, float, float, float, float]:
    """Convert a homogeneous x/y affine matrix to Shapely coefficient order."""
    return (
        float(matrix[0, 0]),
        float(matrix[0, 1]),
        float(matrix[1, 0]),
        float(matrix[1, 1]),
        float(matrix[0, 2]),
        float(matrix[1, 2]),
    )
