from __future__ import annotations

import atexit
import os
import shutil
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from harpy.io import read_zarr
from spatialdata import SpatialData
from spatialdata.datasets import blobs

from spatiato.datasets import blobs_multi_region, blobs_points_repartitioned

TEST_HOME = Path(tempfile.mkdtemp(prefix="spatiato-test-home-"))
TEST_CACHE = TEST_HOME / ".cache"
TEST_CONFIG = TEST_HOME / ".config"
TEST_NAPARI_CONFIG = TEST_CONFIG / "napari" / "settings.yaml"

TEST_CACHE.mkdir(parents=True, exist_ok=True)
TEST_CONFIG.mkdir(parents=True, exist_ok=True)
TEST_NAPARI_CONFIG.parent.mkdir(parents=True, exist_ok=True)

os.environ["HOME"] = str(TEST_HOME)
os.environ["XDG_CACHE_HOME"] = str(TEST_CACHE)
os.environ["XDG_CONFIG_HOME"] = str(TEST_CONFIG)
os.environ["NAPARI_CONFIG"] = str(TEST_NAPARI_CONFIG)
os.environ["NAPARI_CACHE_DIR"] = str(TEST_CACHE / "napari")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

atexit.register(shutil.rmtree, TEST_HOME, ignore_errors=True)

TESTS = Path(__file__).resolve().parent
SRC = TESTS.parent / "src"

for import_path in (SRC, TESTS):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))


def _make_sdata_blobs() -> SpatialData:
    """Return a small SpatialData test object with dummy feature matrices in `table.obsm`."""
    sdata = blobs()
    table = sdata["table"]
    n_obs = table.n_obs
    rng = np.random.default_rng(seed=0)

    table.obsm["features_1"] = rng.normal(size=(n_obs, 4))
    table.obsm["features_2"] = rng.normal(size=(n_obs, 2))

    return sdata


@pytest.fixture
def sdata_blobs() -> SpatialData:
    """Return an in-memory SpatialData test object with dummy feature matrices."""
    return _make_sdata_blobs()


@pytest.fixture
def sdata_blobs_multi_region() -> SpatialData:
    """Return an in-memory SpatialData test object with a multi-region classifier table."""
    return blobs_multi_region()


@pytest.fixture
def sdata_blobs_points_repartitioned() -> SpatialData:
    """Return an in-memory SpatialData test object with a repartitioned points element."""
    return blobs_points_repartitioned()


@pytest.fixture
def backed_sdata_blobs(tmp_path) -> SpatialData:
    """Return a backed SpatialData test object stored in a temporary zarr directory.

    The backed fixtures open their store with harpy's reader, as Spatiato
    does, so their table matrices are lazy Dask arrays.
    """
    path = tmp_path / "blobs.zarr"
    sdata = _make_sdata_blobs()
    sdata.write(path)
    return read_zarr(path)


@pytest.fixture
def backed_sdata_blobs_multi_region(tmp_path) -> SpatialData:
    """Return a backed multi-region SpatialData test object stored in a temporary zarr directory."""
    path = tmp_path / "blobs_multi_region.zarr"
    sdata = blobs_multi_region()
    sdata.write(path)
    return read_zarr(path)


@pytest.fixture
def backed_sdata_blobs_points_repartitioned(tmp_path) -> SpatialData:
    """Return a backed SpatialData test object with a repartitioned points element."""
    path = tmp_path / "blobs_points_repartitioned.zarr"
    sdata = blobs_points_repartitioned()
    sdata.write(path)
    return read_zarr(path)


@pytest.fixture
def restore_triangulation_backend() -> Iterator[None]:
    """Restore Spatiato's configured backend and napari's backend state."""
    from napari.settings import get_settings
    from napari.utils.triangulation_backend import get_backend, set_backend

    import spatiato._shapes_triangulation as shapes_triangulation_module

    settings = get_settings()
    previous_configured_backend = shapes_triangulation_module._CONFIGURED_SHAPES_TRIANGULATION_BACKEND
    previous_settings_backend = settings.experimental.triangulation_backend
    previous_runtime_backend = get_backend()

    try:
        yield
    finally:
        shapes_triangulation_module._CONFIGURED_SHAPES_TRIANGULATION_BACKEND = previous_configured_backend
        settings.experimental.triangulation_backend = previous_settings_backend
        if get_backend() != previous_runtime_backend:
            set_backend(previous_runtime_backend)


@pytest.fixture
def numba_triangulation_backend(restore_triangulation_backend: None) -> None:
    """Run a test with Spatiato's configured backend set to Numba."""
    from spatiato._shapes_triangulation import configure_shapes_triangulation_backend

    configure_shapes_triangulation_backend("numba")
