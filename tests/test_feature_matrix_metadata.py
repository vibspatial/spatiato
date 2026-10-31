from __future__ import annotations

import anndata as ad
import dask.array as da
import harpy as hp
import numpy as np
import pandas as pd
import pytest
import zarr
from anndata.abc import CSRDataset
from scipy import sparse
from spatialdata import SpatialData
from spatialdata.models import TableModel

from spatiato.core.feature_matrix_metadata import (
    CUSTOM_OBSM_FEATURE_NAME,
    CUSTOM_OBSM_SOURCE_KIND,
    HARPY_ADD_FEATURE_MATRIX_SOURCE_KIND,
    FeatureMatrixMetadataState,
    feature_matrix_n_features,
    inspect_feature_matrix_metadata,
    is_custom_obsm_feature_metadata,
    normalize_feature_columns,
    normalize_feature_matrix_source_kind,
    register_feature_matrix_metadata,
)
from spatiato.core.object_classification.classifier import _validate_current_feature_matrix_matches_columns

_FEATURE_MATRICES_KEY = "feature_matrices"


def _make_table(n_obs: int = 3) -> ad.AnnData:
    return ad.AnnData(np.empty((n_obs, 0), dtype=np.float64))


class _FeatureMatrixTable:
    def __init__(self, *, obsm: dict[str, object], n_obs: int = 3) -> None:
        self.obsm = obsm
        self.uns: dict[str, object] = {}
        self.n_obs = n_obs


def test_register_feature_matrix_metadata_rejects_missing_obsm_key() -> None:
    table = _make_table()

    with pytest.raises(ValueError, match='Feature matrix "missing" is not available in ".obsm"'):
        register_feature_matrix_metadata(table, "missing")


def test_register_feature_matrix_metadata_registers_one_dimensional_matrix() -> None:
    table = _make_table()
    table.obsm["scores"] = np.array([0.1, 0.2, 0.3], dtype=np.float32)

    metadata = register_feature_matrix_metadata(table, "scores")

    assert metadata["feature_columns"] == ["scores_0"]
    assert metadata["features"] == [CUSTOM_OBSM_FEATURE_NAME]
    assert metadata["source_kind"] == CUSTOM_OBSM_SOURCE_KIND
    assert metadata["schema_version"] == 1
    assert metadata["backend"] == "numpy"
    assert metadata["dtype"] == "float32"
    assert table.uns[_FEATURE_MATRICES_KEY]["scores"] == metadata
    assert normalize_feature_columns(metadata) == ("scores_0",)


def test_register_feature_matrix_metadata_registers_dense_matrix_with_default_columns() -> None:
    table = _make_table()
    table.obsm["custom"] = np.arange(6, dtype=np.float64).reshape(3, 2)

    metadata = register_feature_matrix_metadata(table, "custom")

    assert metadata["feature_columns"] == ["custom_0", "custom_1"]
    assert metadata["features"] == [CUSTOM_OBSM_FEATURE_NAME]
    assert metadata["source_kind"] == CUSTOM_OBSM_SOURCE_KIND
    assert set(metadata).isdisjoint({"source_image", "source_channels", "source_label", "coordinate_system"})


def test_register_feature_matrix_metadata_registers_sparse_matrix_backend_and_dtype() -> None:
    table = _make_table()
    table.obsm["sparse_features"] = sparse.csr_matrix(
        np.arange(6, dtype=np.float32).reshape(3, 2),
    )

    metadata = register_feature_matrix_metadata(table, "sparse_features")

    assert metadata["feature_columns"] == ["sparse_features_0", "sparse_features_1"]
    assert metadata["backend"] == "sparse"
    assert metadata["dtype"] == "float32"


def test_register_feature_matrix_metadata_preserves_supplied_columns_and_features() -> None:
    table = _make_table()
    table.obsm["custom"] = np.arange(6, dtype=np.float64).reshape(3, 2)

    metadata = register_feature_matrix_metadata(
        table,
        "custom",
        feature_columns=("area", "texture"),
        features=("custom_features",),
    )

    assert metadata["feature_columns"] == ["area", "texture"]
    assert metadata["features"] == ["custom_features"]


@pytest.mark.parametrize(
    ("feature_columns", "match"),
    [
        (("area",), "describes 1 feature column"),
        (("area", "area"), "duplicate"),
        (("area", ""), "empty"),
    ],
)
def test_register_feature_matrix_metadata_rejects_invalid_feature_columns(
    feature_columns: tuple[str, ...],
    match: str,
) -> None:
    table = _make_table()
    table.obsm["custom"] = np.arange(6, dtype=np.float64).reshape(3, 2)

    with pytest.raises(ValueError, match=match):
        register_feature_matrix_metadata(table, "custom", feature_columns=feature_columns)


@pytest.mark.parametrize(
    ("features", "match"),
    [
        ((), "at least one"),
        (("custom", ""), "empty"),
        (("custom", "custom"), "duplicate"),
    ],
)
def test_register_feature_matrix_metadata_rejects_invalid_features(
    features: tuple[str, ...],
    match: str,
) -> None:
    table = _make_table()
    table.obsm["custom"] = np.arange(6, dtype=np.float64).reshape(3, 2)

    with pytest.raises(ValueError, match=match):
        register_feature_matrix_metadata(table, "custom", features=features)


def test_register_feature_matrix_metadata_does_not_overwrite_existing_metadata_by_default() -> None:
    table = _make_table()
    table.obsm["custom"] = np.arange(6, dtype=np.float64).reshape(3, 2)
    table.uns[_FEATURE_MATRICES_KEY] = {
        "custom": {"feature_columns": ["old_0", "old_1"], "features": ["old"]},
        "other": {"feature_columns": ["other_0"], "features": ["other"]},
    }

    with pytest.raises(ValueError, match="already has metadata"):
        register_feature_matrix_metadata(table, "custom")

    metadata = register_feature_matrix_metadata(table, "custom", overwrite=True)

    assert metadata["source_kind"] == CUSTOM_OBSM_SOURCE_KIND
    assert table.uns[_FEATURE_MATRICES_KEY]["custom"] == metadata
    assert table.uns[_FEATURE_MATRICES_KEY]["other"] == {"feature_columns": ["other_0"], "features": ["other"]}


def test_register_feature_matrix_metadata_rejects_non_mapping_metadata_namespace() -> None:
    table = _make_table()
    table.obsm["custom"] = np.arange(6, dtype=np.float64).reshape(3, 2)
    table.uns[_FEATURE_MATRICES_KEY] = "not-a-mapping"

    with pytest.raises(ValueError, match="must be a mapping"):
        register_feature_matrix_metadata(table, "custom")


def test_is_custom_obsm_feature_metadata_requires_explicit_source_kind() -> None:
    custom_metadata = {"feature_columns": ["a"], "features": [CUSTOM_OBSM_FEATURE_NAME], "source_kind": "custom_obsm"}
    harpy_metadata = {
        "feature_columns": ["area"],
        "features": ["area"],
        "source_kind": HARPY_ADD_FEATURE_MATRIX_SOURCE_KIND,
    }

    assert is_custom_obsm_feature_metadata(custom_metadata)
    assert not is_custom_obsm_feature_metadata(harpy_metadata)


def test_normalize_feature_matrix_source_kind_rejects_missing_or_unknown_source_kind() -> None:
    with pytest.raises(ValueError, match="source_kind"):
        normalize_feature_matrix_source_kind({"feature_columns": ["area"], "features": ["area"]})

    with pytest.raises(ValueError, match="unknown"):
        normalize_feature_matrix_source_kind(
            {"feature_columns": ["area"], "features": ["area"], "source_kind": "unknown"},
        )


def test_feature_matrix_metadata_state_rejects_unknown_status() -> None:
    with pytest.raises(ValueError, match="Unsupported feature matrix metadata status"):
        FeatureMatrixMetadataState(feature_key="custom", status="registered")


def test_feature_matrix_metadata_state_rejects_unknown_source_kind() -> None:
    with pytest.raises(ValueError, match="Unsupported feature matrix metadata source kind"):
        FeatureMatrixMetadataState(feature_key="custom", status="registered_valid", source_kind="unknown")


@pytest.mark.parametrize("status", ["missing_matrix", "invalid_matrix", "registered_mismatched"])
def test_feature_matrix_metadata_state_requires_error_for_problem_statuses(status: str) -> None:
    with pytest.raises(ValueError, match="must include an error message"):
        FeatureMatrixMetadataState(feature_key="custom", status=status)


def test_inspect_feature_matrix_metadata_reports_missing_matrix() -> None:
    table = _make_table()

    state = inspect_feature_matrix_metadata(table, "missing")

    assert state.status == "missing_matrix"
    assert state.n_features is None
    assert state.metadata_n_features is None
    assert state.source_kind is None
    assert not state.is_custom_obsm
    assert state.error is not None


def test_inspect_feature_matrix_metadata_reports_invalid_matrix() -> None:
    table = _make_table()
    table.obsm["cube"] = np.ones((3, 2, 2), dtype=np.float64)

    state = inspect_feature_matrix_metadata(table, "cube")

    assert state.status == "invalid_matrix"
    assert state.n_features is None
    assert state.metadata_n_features is None
    assert state.source_kind is None
    assert state.error == "Feature matrices stored in `.obsm` must be 2-dimensional."


def test_inspect_feature_matrix_metadata_reports_wrong_row_count_as_invalid_matrix() -> None:
    table = _FeatureMatrixTable(obsm={"wrong_rows": np.ones((2, 1), dtype=np.float64)})

    state = inspect_feature_matrix_metadata(table, "wrong_rows")

    assert state.status == "invalid_matrix"
    assert state.source_kind is None
    assert state.error == "Feature matrix has 2 rows but the table has 3 observations."


def _unreadable_dask_matrix(shape: tuple[int, int], meta: object) -> da.Array:
    """Return a Dask array whose shape and dtype are known but whose values fail to compute."""

    def fail_to_read(block_info: object = None) -> None:
        raise AssertionError("the feature matrix values were read")

    return da.map_blocks(fail_to_read, chunks=((shape[0],), (shape[1],)), dtype=np.float64, meta=meta)


@pytest.mark.parametrize(
    ("meta", "backend"),
    [(np.empty((0, 0)), "numpy"), (sparse.csr_matrix((0, 0)), "sparse")],
    ids=["dense", "sparse"],
)
def test_feature_matrix_metadata_reads_only_shape_and_dtype_of_a_lazy_matrix(meta: object, backend: str) -> None:
    matrix = _unreadable_dask_matrix((3, 4), meta)
    # Test setup: reading the values fails.
    with pytest.raises(AssertionError, match="values were read"):
        matrix.compute()
    table = _FeatureMatrixTable(obsm={"lazy": matrix})

    unregistered = inspect_feature_matrix_metadata(table, "lazy")
    metadata = register_feature_matrix_metadata(table, "lazy")
    registered = inspect_feature_matrix_metadata(table, "lazy")

    assert unregistered.status == "unregistered"
    assert unregistered.n_features == 4
    assert metadata["backend"] == backend
    assert metadata["dtype"] == "float64"
    assert registered.status == "registered_valid"
    assert registered.n_features == 4


def _fail_on_value_reads(monkeypatch: pytest.MonkeyPatch, handle: object) -> None:
    """Make every way of reading the values of a backed handle fail."""

    def fail_to_read(*args: object, **kwargs: object) -> None:
        raise AssertionError("the feature matrix values were read")

    # `__array__` serves `np.asarray(...)`; `to_memory` loads a `CSRDataset`.
    for name in ("__getitem__", "__array__", "to_memory"):
        if hasattr(type(handle), name):
            monkeypatch.setattr(type(handle), name, fail_to_read)


@pytest.mark.parametrize("component", [("obsm", "dense"), ("X",)], ids=["zarr_array", "csr_dataset"])
def test_feature_matrix_n_features_reads_a_backed_matrix_without_loading_it(
    component: tuple[str, ...],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    adata = ad.AnnData(X=sparse.csr_matrix(np.eye(3, 5)))
    adata.obsm["dense"] = np.zeros((3, 5))
    path = tmp_path / "table.zarr"
    SpatialData(tables={"table": TableModel.parse(adata)}).write(path)
    backed = hp.tb.io.read_table_components(path, table_name="table", components=[component], mode="backed")[component]
    # Test setup: harpy's reader attaches a backed handle, and reading its values fails.
    assert isinstance(backed, (zarr.Array, CSRDataset))
    _fail_on_value_reads(monkeypatch, backed)
    with pytest.raises(AssertionError, match="values were read"):
        backed[:]

    assert feature_matrix_n_features(backed, 3) == 5


def test_classifier_column_check_reads_only_the_shape_of_a_lazy_matrix() -> None:
    table = _FeatureMatrixTable(obsm={"lazy": _unreadable_dask_matrix((3, 4), np.empty((0, 0)))})

    _validate_current_feature_matrix_matches_columns(table, "lazy", ("a", "b", "c", "d"))
    with pytest.raises(ValueError, match="has 4 column"):
        _validate_current_feature_matrix_matches_columns(table, "lazy", ("a", "b"))


def test_feature_matrix_n_features_counts_a_one_dimensional_matrix_as_one_column() -> None:
    assert feature_matrix_n_features(np.zeros(3), 3) == 1


def test_inspect_feature_matrix_metadata_reports_object_matrix_as_invalid() -> None:
    table = _FeatureMatrixTable(obsm={"values": np.array([[1, 2]] * 3, dtype=object)})

    state = inspect_feature_matrix_metadata(table, "values")

    assert state.status == "invalid_matrix"
    assert state.error == "Feature matrix dtype `object` is not numeric."


def test_inspect_feature_matrix_metadata_reports_dataframe_as_invalid() -> None:
    table = _FeatureMatrixTable(obsm={"values": pd.DataFrame({"area": [1.0, 2.0, 3.0]})})

    state = inspect_feature_matrix_metadata(table, "values")

    assert state.status == "invalid_matrix"
    assert state.error == (
        "Feature matrices must be arrays, not DataFrames; store the values as an array, for example with `.to_numpy()`."
    )


def test_inspect_feature_matrix_metadata_reports_unregistered_matrix() -> None:
    table = _make_table()
    table.obsm["custom"] = np.arange(6, dtype=np.float64).reshape(3, 2)

    state = inspect_feature_matrix_metadata(table, "custom")

    assert state.status == "unregistered"
    assert state.n_features == 2
    assert state.metadata_n_features is None
    assert state.source_kind is None
    assert not state.is_custom_obsm
    assert state.error is None
    assert _FEATURE_MATRICES_KEY not in table.uns


def test_inspect_feature_matrix_metadata_reports_valid_custom_metadata() -> None:
    table = _make_table()
    table.obsm["custom"] = np.arange(6, dtype=np.float64).reshape(3, 2)
    register_feature_matrix_metadata(table, "custom", feature_columns=("area", "texture"))

    state = inspect_feature_matrix_metadata(table, "custom")

    assert state.status == "registered_valid"
    assert state.n_features == 2
    assert state.metadata_n_features == 2
    assert state.source_kind == CUSTOM_OBSM_SOURCE_KIND
    assert state.is_custom_obsm
    assert state.error is None


def test_inspect_feature_matrix_metadata_reports_valid_harpy_metadata() -> None:
    table = _make_table()
    table.obsm["harpy"] = np.arange(6, dtype=np.float64).reshape(3, 2)
    table.uns[_FEATURE_MATRICES_KEY] = {
        "harpy": {
            "feature_columns": ["area", "perimeter"],
            "features": ["area", "perimeter"],
            "source_kind": HARPY_ADD_FEATURE_MATRIX_SOURCE_KIND,
        },
    }

    state = inspect_feature_matrix_metadata(table, "harpy")

    assert state.status == "registered_valid"
    assert state.n_features == 2
    assert state.metadata_n_features == 2
    assert state.source_kind == HARPY_ADD_FEATURE_MATRIX_SOURCE_KIND
    assert not state.is_custom_obsm


def test_inspect_feature_matrix_metadata_reports_missing_source_kind_as_mismatched() -> None:
    table = _make_table()
    table.obsm["harpy"] = np.arange(6, dtype=np.float64).reshape(3, 2)
    table.uns[_FEATURE_MATRICES_KEY] = {
        "harpy": {"feature_columns": ["area", "perimeter"], "features": ["area", "perimeter"]},
    }

    state = inspect_feature_matrix_metadata(table, "harpy")

    assert state.status == "registered_mismatched"
    assert state.n_features == 2
    assert state.metadata_n_features is None
    assert state.source_kind is None
    assert state.error is not None
    assert "source_kind" in state.error


def test_inspect_feature_matrix_metadata_reports_mismatched_metadata_column_count() -> None:
    table = _make_table()
    table.obsm["custom"] = np.arange(6, dtype=np.float64).reshape(3, 2)
    table.uns[_FEATURE_MATRICES_KEY] = {
        "custom": {
            "feature_columns": ["area"],
            "features": ["area"],
            "source_kind": HARPY_ADD_FEATURE_MATRIX_SOURCE_KIND,
        },
    }

    state = inspect_feature_matrix_metadata(table, "custom")

    assert state.status == "registered_mismatched"
    assert state.n_features == 2
    assert state.metadata_n_features == 1
    assert state.source_kind == HARPY_ADD_FEATURE_MATRIX_SOURCE_KIND
    assert not state.is_custom_obsm
    assert state.error == 'Feature matrix "custom" has 2 column(s), but its metadata describes 1 feature column(s).'
