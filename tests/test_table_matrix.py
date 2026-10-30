from __future__ import annotations

from pathlib import Path
from typing import Any

import anndata as ad
import dask.array as da
import harpy as hp
import numpy as np
import pandas as pd
import pytest
import zarr
from anndata.abc import CSCDataset, CSRDataset
from scipy import sparse
from spatialdata import SpatialData
from spatialdata.models import TableModel

from spatiato.core.table_matrix import is_sparse_matrix, read_matrix, read_matrix_column, read_matrix_rows

_N_OBS = 30
_POSITIONS = np.array([0, 3, 5, 17, 29])
_DENSE_REPRESENTATIONS = ("numpy", "dataframe", "dask_dense", "lazy_dense", "backed_dense")
_SPARSE_REPRESENTATIONS = (
    "csr",
    "csc",
    "csr_array",
    "coo",
    "lazy_csr",
    "lazy_csc",
    "backed_csr",
    "backed_csc",
)
_REPRESENTATIONS = _DENSE_REPRESENTATIONS + _SPARSE_REPRESENTATIONS


def _dense_values() -> np.ndarray:
    values = np.arange(_N_OBS * 4, dtype=np.float64).reshape(_N_OBS, 4)
    values[3, 1] = np.nan
    values[5, 2] = np.inf
    return values


def _sparse_values() -> sparse.csr_matrix:
    values = np.zeros((_N_OBS, 6), dtype=np.float64)
    values[::3, 2] = np.arange(10, dtype=np.float64) + 1
    values[1::4, 4] = -2.5
    # Row 5 and row 29 stay empty.
    values[5] = 0
    values[29] = 0
    return sparse.csr_matrix(values)


@pytest.fixture(scope="module")
def table_store(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Write a table with a dense `obsm` matrix, a CSR `X` and a CSC layer."""
    adata = ad.AnnData(X=_sparse_values())
    adata.obsm["dense"] = _dense_values()
    adata.layers["csc"] = _sparse_values().tocsc()
    path = tmp_path_factory.mktemp("table_matrix") / "table.zarr"
    SpatialData(tables={"table": TableModel.parse(adata)}).write(path)
    return path


def _read_component(store: Path, component: tuple[str, ...], mode: str) -> Any:
    # Small sparse blocks give a lazy matrix of several Dask blocks.
    components = hp.tb.io.read_table_components(
        store, table_name="table", components=[component], mode=mode, sparse_chunks=7
    )
    return components[component]


def _representation(name: str, store: Path) -> tuple[Any, np.ndarray]:
    """Return the matrix in representation `name` and its dense reference values."""
    dense = _dense_values()
    csr = _sparse_values()
    factories = {
        "numpy": lambda: dense,
        "dataframe": lambda: pd.DataFrame(dense),
        "dask_dense": lambda: da.from_array(dense, chunks=(7, -1)),
        "lazy_dense": lambda: _read_component(store, ("obsm", "dense"), "lazy"),
        "backed_dense": lambda: _read_component(store, ("obsm", "dense"), "backed"),
        "csr": lambda: csr,
        "csc": lambda: csr.tocsc(),
        "csr_array": lambda: sparse.csr_array(csr),
        "coo": lambda: csr.tocoo(),
        "lazy_csr": lambda: _read_component(store, ("X",), "lazy"),
        "lazy_csc": lambda: _read_component(store, ("layers", "csc"), "lazy"),
        "backed_csr": lambda: _read_component(store, ("X",), "backed"),
        "backed_csc": lambda: _read_component(store, ("layers", "csc"), "backed"),
    }
    reference = dense if name in _DENSE_REPRESENTATIONS else csr.toarray()
    return factories[name](), reference


def test_representations_have_the_type_their_name_says(table_store: Path) -> None:
    lazy_csr, _ = _representation("lazy_csr", table_store)
    assert isinstance(lazy_csr, da.Array)
    assert lazy_csr.numblocks[0] > 1
    assert isinstance(_representation("lazy_dense", table_store)[0], da.Array)
    assert isinstance(_representation("lazy_csc", table_store)[0], da.Array)
    assert isinstance(_representation("backed_dense", table_store)[0], zarr.Array)
    assert isinstance(_representation("backed_csr", table_store)[0], CSRDataset)
    assert isinstance(_representation("backed_csc", table_store)[0], CSCDataset)


def _assert_in_memory_values(result: Any, expected: np.ndarray) -> None:
    if sparse.issparse(result):
        assert result.format in ("csr", "csc")
        result = result.toarray()
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("name", _REPRESENTATIONS)
def test_is_sparse_matrix_recognizes_every_representation_without_reading(name: str, table_store: Path) -> None:
    matrix, _ = _representation(name, table_store)

    assert is_sparse_matrix(matrix) is (name in _SPARSE_REPRESENTATIONS)


@pytest.mark.parametrize("name", _REPRESENTATIONS)
def test_read_matrix_returns_the_whole_matrix_in_memory(name: str, table_store: Path) -> None:
    matrix, reference = _representation(name, table_store)

    _assert_in_memory_values(read_matrix(matrix), reference)


@pytest.mark.parametrize("name", _REPRESENTATIONS)
@pytest.mark.parametrize("positions", [_POSITIONS, np.array([17, 0, 5]), np.array([], dtype=np.int64)])
def test_read_matrix_rows_returns_the_requested_rows_in_order(
    name: str,
    positions: np.ndarray,
    table_store: Path,
) -> None:
    matrix, reference = _representation(name, table_store)

    result = read_matrix_rows(matrix, positions)

    assert result.shape == (positions.size, reference.shape[1])
    _assert_in_memory_values(result, reference[positions])


@pytest.mark.parametrize("name", _REPRESENTATIONS)
def test_read_matrix_column_returns_one_column_as_a_1d_array(name: str, table_store: Path) -> None:
    matrix, reference = _representation(name, table_store)

    result = read_matrix_column(matrix, 2)

    assert isinstance(result, np.ndarray)
    assert result.shape == (_N_OBS,)
    np.testing.assert_array_equal(result, reference[:, 2])


@pytest.mark.parametrize("name", _REPRESENTATIONS)
@pytest.mark.parametrize("rows", [_POSITIONS, np.array([17, 0, 5, 0]), np.array([], dtype=np.int64)])
def test_read_matrix_column_returns_the_requested_rows_in_order(
    name: str,
    rows: np.ndarray,
    table_store: Path,
) -> None:
    matrix, reference = _representation(name, table_store)

    result = read_matrix_column(matrix, 2, rows=rows)

    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, reference[rows, 2])


def _record_item_access(monkeypatch: pytest.MonkeyPatch, matrix: Any) -> list[Any]:
    """Record the keys `matrix` is indexed with, and fail on a read of the whole matrix."""
    keys: list[Any] = []
    get_item = type(matrix).__getitem__

    def recording_get_item(self: Any, key: Any) -> Any:
        keys.append(key)
        return get_item(self, key)

    def fail_whole_read(*args: object, **kwargs: object) -> None:
        raise AssertionError("the whole matrix was read")

    monkeypatch.setattr(type(matrix), "__getitem__", recording_get_item)
    monkeypatch.setattr(type(matrix), "to_memory", fail_whole_read)
    return keys


def test_read_matrix_column_reads_the_rows_first_from_a_backed_csr_dataset(
    monkeypatch: pytest.MonkeyPatch,
    table_store: Path,
) -> None:
    matrix, reference = _representation("backed_csr", table_store)
    keys = _record_item_access(monkeypatch, matrix)

    result = read_matrix_column(matrix, 2, rows=[17, 3])

    np.testing.assert_array_equal(result, reference[[17, 3], 2])
    # One read of the requested rows; never a column over all rows.
    assert len(keys) == 1
    np.testing.assert_array_equal(keys[0], [17, 3])


def test_read_matrix_column_reads_the_column_first_from_a_backed_csc_dataset(
    monkeypatch: pytest.MonkeyPatch,
    table_store: Path,
) -> None:
    matrix, reference = _representation("backed_csc", table_store)
    keys = _record_item_access(monkeypatch, matrix)

    result = read_matrix_column(matrix, 2, rows=[17, 3])

    np.testing.assert_array_equal(result, reference[[17, 3], 2])
    # Selecting rows of a CSC dataset would read the whole matrix.
    assert keys == [(slice(None), slice(2, 3))]


def test_read_matrix_rows_keeps_the_stored_dtype() -> None:
    matrix = np.arange(12, dtype=np.int32).reshape(6, 2)

    assert read_matrix_rows(da.from_array(matrix, chunks=(2, -1)), [1, 4]).dtype == np.int32
    assert read_matrix_rows(sparse.csr_matrix(matrix), [1, 4]).dtype == np.int32


def test_read_matrix_rows_reads_only_the_selected_rows_of_a_backed_csr_dataset(
    monkeypatch: pytest.MonkeyPatch,
    table_store: Path,
) -> None:
    matrix, reference = _representation("backed_csr", table_store)

    def fail_whole_read(*args: object, **kwargs: object) -> None:
        raise AssertionError("the whole matrix was read")

    monkeypatch.setattr(type(matrix), "to_memory", fail_whole_read)
    # Indexed with a list, anndata reads the whole matrix; the helper must not.
    with pytest.raises(AssertionError, match="the whole matrix was read"):
        matrix[[0, 3, 17]]

    result = read_matrix_rows(matrix, [0, 3, 17])

    _assert_in_memory_values(result, reference[[0, 3, 17]])


def test_table_matrix_helpers_reject_unsupported_types() -> None:
    with pytest.raises(TypeError, match="Unsupported table matrix type 'list'"):
        read_matrix([[1.0, 2.0]])


def test_table_matrix_helpers_reject_matrices_that_are_not_2d() -> None:
    with pytest.raises(ValueError, match="must be 2-dimensional"):
        read_matrix_rows(np.arange(5.0), [0])


@pytest.mark.parametrize(
    ("positions", "error", "message"),
    [
        ([0, 6], IndexError, r"must lie in \[0, 6\)"),
        ([-1], IndexError, r"must lie in \[0, 6\)"),
        ([0.0, 1.0], TypeError, "must be integers"),
        ([[0, 1]], ValueError, "must be 1-dimensional"),
    ],
)
def test_read_matrix_rows_rejects_invalid_positions(positions: list[Any], error: type[Exception], message: str) -> None:
    with pytest.raises(error, match=message):
        read_matrix_rows(np.zeros((6, 2)), positions)


@pytest.mark.parametrize("column", [-1, 2])
def test_read_matrix_column_rejects_columns_out_of_range(column: int) -> None:
    with pytest.raises(IndexError, match="out of range"):
        read_matrix_column(np.zeros((6, 2)), column)
