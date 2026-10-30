from __future__ import annotations

import dask.array as da
import numpy as np
import zarr
from anndata.abc import CSCDataset, CSRDataset
from scipy import sparse

# What harpy's table reader attaches as a matrix, per stored format and read mode.
MATRIX_TYPE_BY_FORMAT_AND_MODE: dict[tuple[str, str], type] = {
    ("dense", "eager"): np.ndarray,
    ("dense", "lazy"): da.Array,
    ("dense", "backed"): zarr.Array,
    ("csr", "eager"): sparse.csr_matrix,
    ("csr", "lazy"): da.Array,
    ("csr", "backed"): CSRDataset,
    ("csc", "eager"): sparse.csc_matrix,
    ("csc", "lazy"): da.Array,
    ("csc", "backed"): CSCDataset,
}
FORMATS_AND_MODES = tuple(MATRIX_TYPE_BY_FORMAT_AND_MODE)


def matrix_in_format(values: np.ndarray, matrix_format: str) -> np.ndarray | sparse.spmatrix:
    """Return dense `values` stored as `matrix_format`: "dense", "csr" or "csc"."""
    if matrix_format == "csr":
        return sparse.csr_matrix(values)
    if matrix_format == "csc":
        return sparse.csc_matrix(values)
    return values
