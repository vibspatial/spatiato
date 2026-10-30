"""Read table matrices into memory, whatever representation harpy attached.

harpy attaches table matrices (`X`, `obsm`, `layers`, ...) in several forms:
NumPy or SciPy arrays, Dask arrays (lazy; their blocks are NumPy or SciPy
CSR/CSC), and backed handles (`zarr.Array`, anndata `CSRDataset`/`CSCDataset`).
DataFrame-valued `obsm` entries are always in memory. These helpers read only
what is asked for and return NumPy arrays or SciPy CSR/CSC matrices in the
stored dtype; callers cast. A result may share memory with an in-memory input,
so do not modify it in place.

`np.asarray(...)` does not densify a sparse matrix: for a SciPy sparse matrix,
a sparse Dask array or a backed sparse handle, NumPy wraps the object itself in
a 0-d array of dtype `object`, without an error. Use the helpers below instead.
"""

from __future__ import annotations

from typing import Any

import dask.array as da
import numpy as np
import pandas as pd
import zarr
from anndata.abc import CSCDataset, CSRDataset
from numpy.typing import ArrayLike
from scipy.sparse import issparse

_BACKED_SPARSE_TYPES = (CSRDataset, CSCDataset)
# SciPy sparse matrices are supported too; `issparse(...)` recognizes them.
_SUPPORTED_DENSE_OR_HANDLE_TYPES = (np.ndarray, da.Array, zarr.Array, pd.DataFrame, *_BACKED_SPARSE_TYPES)
_IN_MEMORY_SPARSE_FORMATS = ("csr", "csc")


def is_sparse_matrix(matrix: Any) -> bool:
    """Return whether `matrix` holds sparse values, without reading them.

    `scipy.sparse.issparse(...)` alone recognizes only SciPy matrices; a Dask
    array of SciPy blocks and a backed `CSRDataset`/`CSCDataset` are sparse
    too.
    """
    if isinstance(matrix, da.Array):
        return issparse(matrix._meta)
    return issparse(matrix) or isinstance(matrix, _BACKED_SPARSE_TYPES)


def read_matrix(matrix: Any) -> Any:
    """Return the whole matrix in memory, as a NumPy array or a SciPy CSR/CSC matrix."""
    _validate_matrix(matrix)
    if isinstance(matrix, da.Array):
        return _in_memory(matrix.compute())
    if isinstance(matrix, _BACKED_SPARSE_TYPES):
        return matrix.to_memory()
    if isinstance(matrix, zarr.Array):
        return matrix[...]
    if isinstance(matrix, pd.DataFrame):
        return matrix.to_numpy()
    return _in_memory(matrix)


def read_matrix_rows(matrix: Any, positions: ArrayLike) -> Any:
    """Return the rows at `positions` in memory, as a NumPy array or a SciPy CSR/CSC matrix.

    Rows come back in the order of `positions`. Only the stored blocks or
    chunks that contain a requested row are read; a backed `CSCDataset` is read
    whole, because its rows are not stored contiguously.
    """
    _validate_matrix(matrix)
    # A NumPy array, not a list: anndata reads a whole `CSRDataset` when indexed with a list.
    row_positions = _validated_positions(positions, n_rows=matrix.shape[0])
    if isinstance(matrix, pd.DataFrame):
        return matrix.iloc[row_positions].to_numpy()
    if isinstance(matrix, da.Array):
        return _in_memory(matrix[row_positions].compute())
    if isinstance(matrix, zarr.Array):
        return matrix.oindex[row_positions]
    if isinstance(matrix, _BACKED_SPARSE_TYPES):
        return matrix[row_positions]
    return _in_memory(matrix)[row_positions]


def read_matrix_column(matrix: Any, column: int, rows: ArrayLike | None = None) -> np.ndarray:
    """Return one column in memory, as a 1-D NumPy array; only the rows at `rows` when given.

    With `rows`, values come back in the order of `rows`, and the matrix is read
    in the cheaper order for how it is stored: the rows first for matrices
    stored by rows (dense, CSR), the column first for CSC, and both at once for
    a backed dense `zarr.Array`. Without `rows`, a CSR column visits every
    stored row.
    """
    _validate_matrix(matrix)
    n_columns = matrix.shape[1]
    if not 0 <= column < n_columns:
        raise IndexError(f"Column {column} is out of range for a matrix with {n_columns} column(s).")
    if rows is None:
        return _column_values(matrix, column)

    row_positions = _validated_positions(rows, n_rows=matrix.shape[0])
    if _is_stored_by_columns(matrix):
        return _column_values(matrix, column)[row_positions]
    if isinstance(matrix, zarr.Array):
        return matrix.oindex[row_positions, column]
    return _column_values(read_matrix_rows(matrix, row_positions), column)


def _column_values(matrix: Any, column: int) -> np.ndarray:
    if isinstance(matrix, pd.DataFrame):
        return matrix.iloc[:, column].to_numpy()
    # `[:, j:j+1]` stays 2-D for every representation; `[:, j]` does not (a 1-D
    # sparse result, or an error for Dask arrays of several `csr_array` blocks).
    values = _in_memory(matrix)[:, column : column + 1]
    if isinstance(values, da.Array):
        values = values.compute()
    if issparse(values):
        values = values.toarray()
    return np.asarray(values).reshape(-1)


def _is_stored_by_columns(matrix: Any) -> bool:
    """Return whether `matrix` is CSC, so that one column is cheap and rows are not."""
    stored = matrix._meta if isinstance(matrix, da.Array) else matrix
    return isinstance(stored, CSCDataset) or (issparse(stored) and stored.format == "csc")


def _validate_matrix(matrix: Any) -> None:
    if not (isinstance(matrix, _SUPPORTED_DENSE_OR_HANDLE_TYPES) or issparse(matrix)):
        raise TypeError(f"Unsupported table matrix type {type(matrix).__name__!r}.")
    # `CSRDataset`/`CSCDataset` have no `ndim`.
    if len(matrix.shape) != 2:
        raise ValueError(f"Table matrices must be 2-dimensional; got shape {tuple(matrix.shape)}.")


def _validated_positions(positions: ArrayLike, *, n_rows: int) -> np.ndarray:
    # Positions are not sorted: every representation returns rows in the
    # requested order, callers pass table order (ascending) anyway, and for
    # Dask sorting would only save a reorder step, not block reads.
    row_positions = np.asarray(positions)
    if row_positions.ndim != 1:
        raise ValueError("Row positions must be 1-dimensional.")
    if row_positions.size == 0:
        return np.empty(0, dtype=np.int64)
    if not np.issubdtype(row_positions.dtype, np.integer):
        raise TypeError(f"Row positions must be integers; got dtype {row_positions.dtype}.")
    # Negative positions would silently count from the end.
    if row_positions.min() < 0 or row_positions.max() >= n_rows:
        raise IndexError(f"Row positions must lie in [0, {n_rows}).")
    return row_positions.astype(np.int64, copy=False)


def _in_memory(values: Any) -> Any:
    # Indexing needs CSR or CSC; other SciPy formats (e.g. COO) are converted.
    if issparse(values) and values.format not in _IN_MEMORY_SPARSE_FORMATS:
        return values.tocsr()
    return values
