from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import napari
from harpy.io import read_zarr

from spatiato._app_state import get_or_create_app_state
from spatiato.widgets.spatialdata_replacement_dialog import confirm_spatialdata_replacement

if TYPE_CHECKING:
    from napari.types import LayerData

PathLike = str | Path
PathOrPaths = PathLike | Sequence[PathLike]
_PLUGIN_NAME = "spatiato"
_VIEWER_WIDGET_NAME = "Viewer"
_READER_WIDGET_ORDER = ("Viewer", "Feature Extraction", "Object Classification")


def get_reader(path: PathOrPaths):
    """Return a reader function when the path looks like a SpatialData zarr store."""
    candidate = _normalize_single_path(path)
    if candidate is None:
        return None
    if not _is_spatialdata_zarr_store(candidate):
        return None
    return _read_spatialdata_store


def _read_spatialdata_store(path: PathOrPaths) -> list[LayerData]:
    """Load a SpatialData store into Spatiato app state and show the Spatiato viewer widget."""
    candidate = _normalize_single_path(path)
    if candidate is None:
        raise ValueError("spatiato reader expects exactly one SpatialData zarr store path.")

    viewer = napari.current_viewer()
    if viewer is None:
        raise RuntimeError("spatiato reader requires an active napari viewer.")

    app_state = get_or_create_app_state(viewer)
    if app_state.sdata is not None and not confirm_spatialdata_replacement():
        return [(None,)]

    # harpy's reader opens every table matrix lazily, as Dask arrays that read
    # from the store; `obs`, `var` and `uns` are read into memory.
    sdata = read_zarr(candidate)
    app_state.set_sdata(sdata, discard_current=True)
    _ensure_spatiato_widgets(viewer)
    return [(None,)]


def _normalize_single_path(path: PathOrPaths) -> str | None:
    if isinstance(path, (str, Path)):
        return str(path)

    if len(path) != 1:
        return None

    candidate = path[0]
    if isinstance(candidate, (str, Path)):
        return str(candidate)
    return None


def _is_spatialdata_zarr_store(path: str) -> bool:
    store_path = Path(path)
    if store_path.suffix != ".zarr" or not store_path.is_dir():
        return False

    root_attributes = _read_root_attributes(store_path)
    return "spatialdata_attrs" in root_attributes


def _read_root_attributes(store_path: Path) -> dict[str, object]:
    zarr_json_path = store_path / "zarr.json"
    if zarr_json_path.is_file():
        try:
            root_metadata = json.loads(zarr_json_path.read_text())
        except (OSError, json.JSONDecodeError):
            return {}

        attributes = root_metadata.get("attributes", {})
        return attributes if isinstance(attributes, dict) else {}

    zattrs_path = store_path / ".zattrs"
    if zattrs_path.is_file():
        try:
            root_attributes = json.loads(zattrs_path.read_text())
        except (OSError, json.JSONDecodeError):
            return {}

        return root_attributes if isinstance(root_attributes, dict) else {}

    return {}


def _ensure_spatiato_widgets(viewer: object) -> None:
    window = getattr(viewer, "window", None)
    add_plugin_dock_widget = getattr(window, "add_plugin_dock_widget", None)
    if not callable(add_plugin_dock_widget):
        return

    viewer_dock_widget: object | None = None
    for widget_name in _READER_WIDGET_ORDER:
        dock_widget, _inner_widget = add_plugin_dock_widget(_PLUGIN_NAME, widget_name, tabify=True)
        if widget_name == _VIEWER_WIDGET_NAME:
            viewer_dock_widget = dock_widget

    raise_dock_widget = getattr(viewer_dock_widget, "raise_", None)
    if callable(raise_dock_widget):
        raise_dock_widget()
