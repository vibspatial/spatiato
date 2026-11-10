from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from dask.callbacks import Callback
from shapely.geometry import Polygon
from spatialdata import SpatialData
from spatialdata.models import Labels3DModel, ShapesModel
from spatialdata.transformations import Affine, Identity, set_transformation

import spatiato.core.spatial_query.query as query_module
from spatiato.core.spatial_query import (
    CanonicalCenterQueryRequest,
    build_canonical_center_query_request,
    evaluate_canonical_center_query,
)
from spatiato.core.spatial_query.canonical_centers import CanonicalCentersRegion


def test_query_includes_boundaries_excludes_holes_and_reads_no_labels_tasks(
    sdata_blobs: SpatialData,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    polygon_with_hole = Polygon(
        [(0, 0), (10, 0), (10, 10), (0, 10)],
        holes=[[(3, 3), (7, 3), (7, 7), (3, 7)]],
    )
    disjoint_polygon = Polygon([(20, 20), (22, 20), (22, 22), (20, 22)])
    _add_shapes(sdata_blobs, "query", [polygon_with_hole, disjoint_polygon])
    canonical_centers = _canonical_centers(
        instance_ids=[5, 2, 9, 4, 7, 6, 11],
        centers_xy=[(1, 1), (0, 5), (3, 5), (5, 5), (15, 15), (21, 21), (100, 100)],
    )
    executed_tasks: list[object] = []
    predicate_input_sizes: list[int] = []
    intersects_xy = query_module.shapely.intersects_xy

    def record_predicate_input(geometry, x, y):
        predicate_input_sizes.append(len(x))
        return intersects_xy(geometry, x, y)

    # Wrap the real predicate to verify that the bounding-box prefilter removes distant centers first.
    monkeypatch.setattr(query_module.shapely, "intersects_xy", record_predicate_input)

    with Callback(pretask=lambda key, _dask, _state: executed_tasks.append(key)):
        request = build_canonical_center_query_request(
            sdata_blobs,
            shapes_name="query",
            coordinate_system="global",
            canonical_centers=canonical_centers,
        )
        result = evaluate_canonical_center_query(request)

    assert request.table_name == "table"
    assert request.labels_name == "blobs_labels"
    assert request.polygons == (polygon_with_hole, disjoint_polygon)
    assert not request.polygons_to_labels_affine.flags.writeable
    assert result.canonical_centers is canonical_centers
    assert result.eligible_instance_count == 7
    assert result.matched_instance_count == 4
    # Instance 4 has center (5, 5), strictly inside the polygon hole, and is therefore excluded.
    assert result.matched_instance_ids.tolist() == [2, 5, 6, 9]
    assert not result.matched_instance_ids.flags.writeable
    assert predicate_input_sizes == [6]
    assert executed_tasks == []


def test_request_uses_spatialdata_element_to_element_transformation(
    sdata_blobs: SpatialData,
) -> None:
    _add_shapes(sdata_blobs, "query", [Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])])
    shapes_element = sdata_blobs.shapes["query"]
    labels_element = sdata_blobs.labels["blobs_labels"]
    shapes_to_aligned = np.array(
        [
            [2.0, 0.0, 100.0],
            [0.0, 3.0, 50.0],
            [0.0, 0.0, 1.0],
        ]
    )
    labels_to_aligned = np.array(
        [
            [1.0, 0.0, 10.0],
            [0.0, 2.0, 10.0],
            [0.0, 0.0, 1.0],
        ]
    )
    set_transformation(
        shapes_element,
        Affine(shapes_to_aligned, input_axes=("x", "y"), output_axes=("x", "y")),
        to_coordinate_system="aligned",
    )
    set_transformation(
        labels_element,
        Affine(labels_to_aligned, input_axes=("x", "y"), output_axes=("x", "y")),
        to_coordinate_system="aligned",
    )
    expected_affine = np.linalg.inv(labels_to_aligned) @ shapes_to_aligned
    transformed_center = expected_affine @ np.array([1.0, 1.0, 1.0])
    canonical_centers = _canonical_centers(
        instance_ids=[8],
        centers_xy=[tuple(transformed_center[:2])],
    )

    request = build_canonical_center_query_request(
        sdata_blobs,
        shapes_name="query",
        coordinate_system="aligned",
        canonical_centers=canonical_centers,
    )
    result = evaluate_canonical_center_query(request)

    np.testing.assert_allclose(request.polygons_to_labels_affine, expected_affine)
    assert result.matched_instance_ids.tolist() == [8]


def test_request_uses_the_selected_region_coordinate_frame(
    sdata_blobs_multi_region: SpatialData,
) -> None:
    _add_shapes(sdata_blobs_multi_region, "query", [Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])])
    shapes_element = sdata_blobs_multi_region.shapes["query"]
    labels_element = sdata_blobs_multi_region.labels["blobs_labels_2"]
    labels_to_global_1 = np.array(
        [
            [1.0, 0.0, 100.0],
            [0.0, 1.0, 25.0],
            [0.0, 0.0, 1.0],
        ]
    )
    set_transformation(shapes_element, Identity(), to_coordinate_system="global_1")
    set_transformation(
        labels_element,
        Affine(labels_to_global_1, input_axes=("x", "y"), output_axes=("x", "y")),
        to_coordinate_system="global_1",
    )
    canonical_centers = _canonical_centers(
        labels_name="blobs_labels_2",
        instance_ids=[1],
        centers_xy=[(-99, -24)],
    )

    request = build_canonical_center_query_request(
        sdata_blobs_multi_region,
        shapes_name="query",
        coordinate_system="global_1",
        canonical_centers=canonical_centers,
    )

    assert request.labels_name == "blobs_labels_2"
    np.testing.assert_allclose(request.polygons_to_labels_affine, np.linalg.inv(labels_to_global_1))


@pytest.mark.parametrize(
    "affine",
    [
        np.array([[1.0, 0.0, 7.0], [0.0, 1.0, -3.0], [0.0, 0.0, 1.0]]),
        np.array([[2.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]]),
        np.array([[0.0, -1.0, 5.0], [1.0, 0.0, 2.0], [0.0, 0.0, 1.0]]),
        np.array([[-1.0, 0.0, 4.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    ],
    ids=["translation", "anisotropic_scale", "rotation", "reflection"],
)
def test_query_applies_supported_xy_affines(
    sdata_blobs: SpatialData,
    affine: np.ndarray,
) -> None:
    polygon = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
    transformed_center = affine @ np.array([1.0, 1.0, 1.0])
    canonical_centers = _canonical_centers(
        instance_ids=[3],
        centers_xy=[tuple(transformed_center[:2])],
    )
    request = CanonicalCenterQueryRequest(
        canonical_centers=canonical_centers,
        polygons=(polygon,),
        polygons_to_labels_affine=affine,
    )

    result = evaluate_canonical_center_query(request)

    assert result.matched_instance_ids.tolist() == [3]


def test_request_rejects_shapes_that_are_not_editable(
    sdata_blobs: SpatialData,
) -> None:
    canonical_centers = _canonical_centers(
        instance_ids=[1],
        centers_xy=[(0, 0)],
    )

    with pytest.raises(ValueError, match="unsupported geometry `MultiPolygon`"):
        build_canonical_center_query_request(
            sdata_blobs,
            shapes_name="blobs_multipolygons",
            coordinate_system="global",
            canonical_centers=canonical_centers,
        )


def test_request_rejects_a_labels_element_that_is_not_2d(
    sdata_blobs: SpatialData,
) -> None:
    _add_shapes(sdata_blobs, "query", [Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])])
    sdata_blobs.labels["labels_3d"] = Labels3DModel.parse(np.zeros((2, 4, 4), dtype=np.int32), dims=("z", "y", "x"))
    canonical_centers = _canonical_centers(
        labels_name="labels_3d",
        instance_ids=[1],
        centers_xy=[(0, 0)],
    )

    with pytest.raises(ValueError, match="not a 2-D integer labels element"):
        build_canonical_center_query_request(
            sdata_blobs,
            shapes_name="query",
            coordinate_system="global",
            canonical_centers=canonical_centers,
        )


def _add_shapes(
    sdata: SpatialData,
    shapes_name: str,
    polygons: list[Polygon],
) -> None:
    geodataframe = gpd.GeoDataFrame(
        geometry=polygons,
        index=pd.Index([f"polygon_{index}" for index in range(len(polygons))], name="instance_id"),
    )
    sdata.shapes[shapes_name] = ShapesModel.parse(
        geodataframe,
        transformations={"global": Identity()},
    )


def _canonical_centers(
    *,
    labels_name: str = "blobs_labels",
    instance_ids: list[int],
    centers_xy: list[tuple[float, float]],
) -> CanonicalCentersRegion:
    if len(instance_ids) != len(centers_xy):
        raise ValueError("Test instance IDs and centers must have equal lengths.")
    centers = np.zeros((len(instance_ids), 3), dtype=np.float64)
    centers[:, 1] = [center_y for _, center_y in centers_xy]
    centers[:, 2] = [center_x for center_x, _ in centers_xy]
    return CanonicalCentersRegion(
        table_name="table",
        labels_name=labels_name,
        row_positions=np.arange(len(instance_ids), dtype=np.int64),
        instance_ids=np.asarray(instance_ids, dtype=np.int64),
        centers=centers,
    )
