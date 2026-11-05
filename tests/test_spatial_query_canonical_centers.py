from __future__ import annotations

import dask.array as da
import numpy as np
import pytest
from harpy.table.canonical_centers import CANONICAL_OBSM_KEY, SPATIAL_COORDINATES_KEY
from spatialdata import SpatialData, read_zarr
from spatialdata.models import Labels3DModel

from spatiato.core.spatial_query.canonical_centers import (
    CanonicalCentersStatus,
    CanonicalRegionCenters,
    canonical_centers_status,
    compute_canonical_centers,
    is_canonical_query_labels,
    read_canonical_region_centers,
)
from spatiato.core.spatialdata import get_table_metadata

_MULTI = "table_multi"


def _state(sdata: SpatialData, table_name: str = "table") -> str:
    return canonical_centers_status(sdata, table_name=table_name).state


@pytest.mark.parametrize(
    ("state", "reason"),
    [("valid", "stale"), ("absent", "stale"), ("invalid", None), ("other", None)],
)
def test_canonical_centers_status_sets_a_reason_exactly_for_invalid(state: str, reason: str | None) -> None:
    with pytest.raises(ValueError):
        CanonicalCentersStatus("table", state, reason)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("instance_ids", "centers"),
    [
        (np.array([1, 2]), np.zeros((3, 3))),
        (np.array([1, 2, 3, 4]), np.zeros((3, 3))),
        (np.array([1, 2, 3]), np.zeros((3, 2))),
    ],
    ids=["fewer_instance_ids", "more_instance_ids", "two_columns"],
)
def test_canonical_region_centers_require_aligned_arrays(instance_ids: np.ndarray, centers: np.ndarray) -> None:
    with pytest.raises(ValueError, match="must describe the same rows"):
        CanonicalRegionCenters("table", "blobs_labels", np.array([0, 1, 2]), instance_ids, centers)


def test_canonical_centers_are_absent_then_created_then_rebuilt_in_memory(sdata_blobs: SpatialData) -> None:
    assert canonical_centers_status(sdata_blobs, table_name="table") == CanonicalCentersStatus("table", "absent")

    assert compute_canonical_centers(sdata_blobs, table_name="table") == "created"
    matrix = sdata_blobs["table"].obsm[CANONICAL_OBSM_KEY]
    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (sdata_blobs["table"].n_obs, 3)
    assert _state(sdata_blobs) == "valid"

    assert compute_canonical_centers(sdata_blobs, table_name="table") == "rebuilt"
    assert _state(sdata_blobs) == "valid"


def test_compute_canonical_centers_writes_backed_spatialdata_to_its_store(backed_sdata_blobs: SpatialData) -> None:
    assert compute_canonical_centers(backed_sdata_blobs, table_name="table") == "created"

    assert isinstance(backed_sdata_blobs["table"].obsm[CANONICAL_OBSM_KEY], da.Array)
    assert _state(backed_sdata_blobs) == "valid"
    assert _state(read_zarr(backed_sdata_blobs.path)) == "valid"


def test_canonical_centers_status_reports_harpys_reason_for_an_older_record(sdata_blobs: SpatialData) -> None:
    compute_canonical_centers(sdata_blobs, table_name="table")
    sdata_blobs["table"].uns[SPATIAL_COORDINATES_KEY][CANONICAL_OBSM_KEY]["schema_version"] = 1

    status = canonical_centers_status(sdata_blobs, table_name="table")

    assert status.state == "invalid"
    assert status.reason is not None
    assert "overwrite=True" in status.reason


def test_canonical_centers_status_reports_a_matrix_without_metadata_as_invalid(sdata_blobs: SpatialData) -> None:
    table = sdata_blobs["table"]
    table.obsm[CANONICAL_OBSM_KEY] = np.zeros((table.n_obs, 3))

    status = canonical_centers_status(sdata_blobs, table_name="table")

    assert status.state == "invalid"
    assert status.reason is not None
    assert "metadata" in status.reason


def test_compute_canonical_centers_rebuilds_a_record_without_its_matrix(sdata_blobs: SpatialData) -> None:
    compute_canonical_centers(sdata_blobs, table_name="table")
    del sdata_blobs["table"].obsm[CANONICAL_OBSM_KEY]
    assert _state(sdata_blobs) == "invalid"

    assert compute_canonical_centers(sdata_blobs, table_name="table") == "rebuilt"
    assert _state(sdata_blobs) == "valid"


def test_compute_canonical_centers_passes_harpys_error_through_without_changing_the_table(
    sdata_blobs: SpatialData,
) -> None:
    table = sdata_blobs["table"]
    instance_key = get_table_metadata(sdata_blobs, "table").instance_key
    instance_ids = table.obs[instance_key].to_numpy().copy()
    # An instance without pixels in the labels element has no center.
    instance_ids[0] = instance_ids.max() + 1000
    table.obs[instance_key] = instance_ids

    with pytest.raises(ValueError, match="has no finite center"):
        compute_canonical_centers(sdata_blobs, table_name="table")

    assert CANONICAL_OBSM_KEY not in table.obsm
    assert _state(sdata_blobs) == "absent"


def test_read_canonical_region_centers_reads_the_rows_of_one_labels_element(
    sdata_blobs_multi_region: SpatialData,
) -> None:
    compute_canonical_centers(sdata_blobs_multi_region, table_name=_MULTI)
    table = sdata_blobs_multi_region[_MULTI]
    rows = np.flatnonzero((table.obs["region"] == "blobs_labels_2").to_numpy())

    region = read_canonical_region_centers(sdata_blobs_multi_region, table_name=_MULTI, labels_name="blobs_labels_2")

    assert region.table_name == _MULTI
    assert region.labels_name == "blobs_labels_2"
    np.testing.assert_array_equal(region.row_positions, rows)
    np.testing.assert_array_equal(region.instance_ids, table.obs["instance_id"].to_numpy()[rows])
    np.testing.assert_array_equal(region.centers, np.asarray(table.obsm[CANONICAL_OBSM_KEY])[rows])
    assert region.centers.dtype == np.float64
    for array in (region.row_positions, region.instance_ids, region.centers):
        assert array.flags.writeable is False


def test_read_canonical_region_centers_reads_a_lazy_backed_matrix(backed_sdata_blobs: SpatialData) -> None:
    compute_canonical_centers(backed_sdata_blobs, table_name="table")
    matrix = backed_sdata_blobs["table"].obsm[CANONICAL_OBSM_KEY]

    region = read_canonical_region_centers(backed_sdata_blobs, table_name="table", labels_name="blobs_labels")

    np.testing.assert_array_equal(region.centers, matrix.compute()[region.row_positions])


def test_read_canonical_region_centers_rejects_an_unannotated_labels_element(sdata_blobs: SpatialData) -> None:
    compute_canonical_centers(sdata_blobs, table_name="table")

    with pytest.raises(ValueError, match="does not annotate labels element `blobs_multiscale_labels`"):
        read_canonical_region_centers(sdata_blobs, table_name="table", labels_name="blobs_multiscale_labels")


def test_read_canonical_region_centers_rejects_a_table_without_centers(sdata_blobs: SpatialData) -> None:
    with pytest.raises(ValueError, match="has no canonical centers"):
        read_canonical_region_centers(sdata_blobs, table_name="table", labels_name="blobs_labels")


def test_read_canonical_region_centers_rejects_a_background_instance_id(sdata_blobs: SpatialData) -> None:
    compute_canonical_centers(sdata_blobs, table_name="table")
    table = sdata_blobs["table"]
    instance_key = get_table_metadata(sdata_blobs, "table").instance_key
    instance_ids = table.obs[instance_key].to_numpy().copy()
    instance_ids[0] = 0
    table.obs[instance_key] = instance_ids

    with pytest.raises(ValueError, match="must be positive integers"):
        read_canonical_region_centers(sdata_blobs, table_name="table", labels_name="blobs_labels")


def test_is_canonical_query_labels_accepts_only_2d_labels_elements(sdata_blobs: SpatialData) -> None:
    sdata_blobs.labels["labels_3d"] = Labels3DModel.parse(np.zeros((2, 4, 4), dtype=np.int32), dims=("z", "y", "x"))

    assert is_canonical_query_labels(sdata_blobs, "blobs_labels") is True
    assert is_canonical_query_labels(sdata_blobs, "blobs_multiscale_labels") is True
    assert is_canonical_query_labels(sdata_blobs, "labels_3d") is False
    # An image, and a name that does not exist.
    assert is_canonical_query_labels(sdata_blobs, "blobs_image") is False
    assert is_canonical_query_labels(sdata_blobs, "missing") is False
