from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import harpy as hp
import pandas as pd
from spatialdata.models import TableModel

from spatiato.core.spatialdata import get_table, get_table_metadata, normalize_table_metadata

if TYPE_CHECKING:
    from anndata import AnnData
    from spatialdata import SpatialData


TableComponent = Literal["obs", "obsm", "uns"]
_MISSING = object()


@dataclass(frozen=True, order=True)
class TableComponentPath:
    """Identify one part of an AnnData table that is tracked, written and reloaded on its own.

    Only three components are supported:

    - `obs`: one column, for example `TableComponentPath("obs", ("pred_class",))`.
    - `obsm`: one entry, for example `TableComponentPath("obsm", ("features",))`.
    - `uns`: one entry at any depth, for example
      `TableComponentPath("uns", ("classifier_config", "pred_class"))`.

    Other components, such as `X`, `layers`, `var` or `obsp`, raise `ValueError`.
    """

    component: TableComponent
    keys: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.component not in ("obs", "obsm", "uns"):
            raise ValueError(f"Unsupported table component: {self.component!r}.")
        keys = tuple(self.keys)
        if not keys or any(not isinstance(key, str) or not key for key in keys):
            raise ValueError("Table component path keys must be non-empty strings.")
        if self.component in ("obs", "obsm") and len(keys) != 1:
            raise ValueError("obs and obsm paths must identify exactly one key.")
        object.__setattr__(self, "keys", keys)


@dataclass(frozen=True)
class TableComponentPersistResult:
    """Report the table store and logical paths persisted by one write."""

    table_path: str
    persisted_paths: frozenset[TableComponentPath]


@dataclass(frozen=True)
class TableComponentReloadResult:
    """Report the table store and logical path coverage restored from disk."""

    table_path: str
    reloaded_paths: frozenset[TableComponentPath]


def persist_table_components(
    sdata: SpatialData,
    *,
    table_name: str,
    paths: frozenset[TableComponentPath],
) -> TableComponentPersistResult:
    """Write the given `obs`, `obsm` and `uns` paths of a backed table to its zarr store.

    Only these three components are written, since `TableComponentPath`
    supports no others; the rest of the table in the store is left as it is.
    The store is made to follow memory, in one update through harpy:

    - `obs`: any `obs` path writes the whole `obs` dataframe, because harpy
      writes `obs` only whole. A column dropped in memory is gone from the store.
    - `obsm` and `uns`: a path with a value in memory is written; a path
      without one is deleted from the store.

    The attached table stays the same AnnData object in `sdata.tables`, but
    harpy's `add_table_components(...)` replaces what it wrote with what it
    reads back from the store: `obs` becomes a new dataframe, `uns` a new dict,
    and, when an `obsm` entry is written, `obsm` a new mapping whose written
    entries are lazy views of the store. Values come back as stored, so a list
    in `uns` becomes a numpy array, as after a reload. Code that kept a
    reference to the old `table.obs` or `table.uns` keeps the old object.

    If the write fails, both the store and the attached table are restored.

    Raises `ValueError` when `paths` is empty, when the table is not backed,
    or when it is stored anywhere other than `tables/<table_name>`.
    """
    table, table_path = _require_backed_table(sdata, table_name)
    normalized_paths = _remove_redundant_uns_descendants(frozenset(paths))
    if not normalized_paths:
        raise ValueError("At least one table component path is required for persistence.")

    components: dict[tuple[str, ...], object] = {}
    deletions: list[tuple[str, ...]] = []
    if any(path.component == "obs" for path in normalized_paths):
        components[("obs",)] = table.obs
    for path in sorted(path for path in normalized_paths if path.component != "obs"):
        if path.component == "obsm":
            exists = path.keys[0] in table.obsm
            value = table.obsm[path.keys[0]] if exists else _MISSING
        else:
            exists, value = _read_mapping_path(table.uns, path.keys)
        if exists:
            components[(path.component, *path.keys)] = value
        else:
            deletions.append((path.component, *path.keys))

    if components:
        obs_identity = None
        if any(component_path[0] == "obsm" for component_path in components):
            metadata = get_table_metadata(sdata, table_name)
            obs_identity = table.obs[[metadata.region_key, metadata.instance_key]]
        hp.tb.io.add_table_components(
            sdata,
            table_name=table_name,
            components=components,
            delete=tuple(deletions),
            obs_identity=obs_identity,
            overwrite=True,
        )
    else:
        hp.tb.io.remove_table_components(sdata, table_name=table_name, components=tuple(deletions))
    return TableComponentPersistResult(table_path=table_path, persisted_paths=normalized_paths)


def build_full_table_reload_paths(
    sdata: SpatialData,
    *,
    table_name: str,
    extra_paths: frozenset[TableComponentPath] = frozenset(),
) -> frozenset[TableComponentPath]:
    """List every path that a full reload of a backed table covers.

    `reload_table_components(...)` installs only the paths it is given, so
    the `obsm` and top-level `uns` keys are listed from memory and from the
    store: a key only in the store is added to memory, and a key only in
    memory is removed from it. harpy cannot list keys without reading, so
    `obsm` is read lazily, which reads only metadata, and `uns` into memory.

    `obs` is not read: any `obs` path reloads the whole `obs`, and the reload
    reports the columns of memory and store together, so the columns in
    memory suffice.

    `extra_paths` are the table's dirty paths. They matter only for a dirty
    path whose key exists neither in memory nor in the store, which
    Spatiato's own flows do not produce; listing it lets the reload still
    make it clean.
    """
    table, _table_path = _require_backed_table(sdata, table_name)
    disk = hp.tb.io.read_table_components(
        sdata.path,
        table_name=table_name,
        components=[("obsm",), ("uns",)],
        mode="lazy",
        missing="omit",
    )
    disk_obsm = disk.get(("obsm",))
    disk_uns = disk.get(("uns",))
    disk_obsm_keys = disk_obsm.keys() if isinstance(disk_obsm, Mapping) else ()
    disk_uns_keys = disk_uns.keys() if isinstance(disk_uns, Mapping) else ()

    paths = set(extra_paths)
    paths.update(TableComponentPath("obs", (str(column),)) for column in table.obs.columns)
    paths.update(TableComponentPath("obsm", (str(key),)) for key in {*table.obsm.keys(), *disk_obsm_keys})
    paths.update(TableComponentPath("uns", (str(key),)) for key in {*table.uns.keys(), *disk_uns_keys})
    if not paths:
        raise ValueError(f"Table `{table_name}` has no supported components to reload.")
    return _remove_redundant_uns_descendants(frozenset(paths))


def reload_table_components(
    sdata: SpatialData,
    *,
    table_name: str,
    paths: frozenset[TableComponentPath],
    region_name: str | None = None,
) -> TableComponentReloadResult:
    """Reload the given `obs`, `obsm` and `uns` paths of a backed table from its zarr store.

    The paths are read through harpy's `read_table_components(...)` and
    installed in the attached table, which stays the same AnnData object.
    Memory follows the store:

    - `obs`: any `obs` path reloads the whole `obs` dataframe, so its columns
      and row index stay consistent.
    - `obsm` and `uns`: each path is reloaded on its own, and other entries
      are left as they are. A path in the store replaces the value in memory;
      a path absent from the store is removed from memory.

    `obs` and `uns` are read into memory, and `obsm` entries as lazy views of
    the store. Nothing is installed unless the stored `obs` has the same
    rows, with the same region and instance values, as memory, and, when
    `region_name` is given, the stored table still annotates it. If
    installing or validating the table fails, the previous `obs`, `obsm` and
    `uns` are restored.

    harpy installs read components in an attached table only after its own
    writes, and checks the stored rows against memory only before them.
    Replace the install, its rollback, `_replace_mapping_path(...)` and
    `_validate_reload_binding(...)` with harpy's public function once
    https://github.com/saeyslab/harpy/issues/285 is released, with its check
    of the stored rows made fast (https://github.com/saeyslab/harpy/issues/284);
    the reported paths stay.
    """
    table, table_path = _require_backed_table(sdata, table_name)
    normalized_paths = _remove_redundant_uns_descendants(frozenset(paths))
    if not normalized_paths:
        raise ValueError("At least one table component path is required for reload.")

    # `obs` and `spatialdata_attrs` are always read, for the binding check.
    # harpy refuses overlapping paths in one request, so a requested path
    # under `spatialdata_attrs` is taken from that value below.
    attrs_path = ("uns", TableModel.ATTRS_KEY)
    components: list[tuple[str, ...]] = [("obs",), attrs_path]
    for path in sorted(normalized_paths):
        if path.component == "obsm" or (path.component == "uns" and path.keys[0] != TableModel.ATTRS_KEY):
            components.append((path.component, *path.keys))
    disk = hp.tb.io.read_table_components(
        sdata.path,
        table_name=table_name,
        components=components,
        mode="lazy",
        missing="omit",
    )
    disk_obs = disk.get(("obs",), _MISSING)
    disk_attrs = disk.get(attrs_path, _MISSING)
    _validate_reload_binding(
        table,
        table_name=table_name,
        disk_obs=disk_obs,
        disk_attrs=disk_attrs,
        region_name=region_name,
    )

    reload_obs = any(path.component == "obs" for path in normalized_paths)
    obsm_values: dict[str, object] = {}
    for path in sorted(path for path in normalized_paths if path.component == "obsm"):
        key = path.keys[0]
        value = disk.get(("obsm", key), _MISSING)
        if value is not _MISSING:
            shape = getattr(value, "shape", None)
            if shape is None or len(shape) == 0 or int(shape[0]) != table.n_obs:
                raise ValueError(f"Reloaded obsm entry `{key}` for table `{table_name}` must have {table.n_obs} rows.")
        obsm_values[key] = value

    uns_values: dict[tuple[str, ...], object] = {}
    for path in sorted(path for path in normalized_paths if path.component == "uns"):
        if path.keys[0] == TableModel.ATTRS_KEY:
            # `_validate_reload_binding(...)` has checked that it is a mapping.
            _exists, value = _read_mapping_path({TableModel.ATTRS_KEY: disk_attrs}, path.keys)
        else:
            value = disk.get(("uns", *path.keys), _MISSING)
        uns_values[path.keys] = value

    previous_obs = table.obs
    previous_obsm = dict(table.obsm)
    previous_uns = dict(table.uns)
    try:
        if reload_obs:
            table.obs = disk_obs
        if obsm_values:
            next_obsm = dict(table.obsm)
            for key, value in obsm_values.items():
                if value is _MISSING:
                    next_obsm.pop(key, None)
                else:
                    next_obsm[key] = value
            table.obsm = next_obsm
        if uns_values:
            next_uns = dict(table.uns)
            for keys, value in uns_values.items():
                next_uns = _replace_mapping_path(next_uns, keys, value)
            table.uns = next_uns
            if any(keys[0] == TableModel.ATTRS_KEY for keys in uns_values):
                normalize_table_metadata(table)
        TableModel.validate(table)
    except Exception:
        table.obs = previous_obs
        table.obsm = previous_obsm
        table.uns = previous_uns
        # Propagate the original reload failure after rollback so the calling
        # widget can turn expected errors into user-facing feedback.
        raise

    reloaded_paths = set(normalized_paths)
    if reload_obs:
        reloaded_paths.difference_update(path for path in tuple(reloaded_paths) if path.component == "obs")
        reloaded_paths.update(
            TableComponentPath("obs", (str(column),)) for column in {*previous_obs.columns, *disk_obs.columns}
        )
    return TableComponentReloadResult(
        table_path=table_path,
        reloaded_paths=frozenset(reloaded_paths),
    )


def resolve_table_path(sdata: SpatialData, table: AnnData, table_name: str) -> str:
    """Return the unique zarr path for `table` inside a backed SpatialData object."""
    table_paths = sdata.locate_element(table)
    if not table_paths:
        raise ValueError(f"Could not locate table `{table_name}` inside the backed SpatialData store.")

    if len(table_paths) > 1:
        raise ValueError(
            f"Table `{table_name}` resolved to multiple zarr paths: {table_paths}. A unique table path is required."
        )

    return table_paths[0]


def _require_backed_table(sdata: SpatialData, table_name: str) -> tuple[AnnData, str]:
    """Return the attached table and its path in the store.

    harpy reads and writes a table at `tables/<table_name>`, so a table stored
    anywhere else is refused here, before harpy is called.
    """
    if not sdata.is_backed() or sdata.path is None:
        raise ValueError("SpatialData must be backed by a zarr store for table persistence.")
    table = get_table(sdata, table_name)
    table_path = resolve_table_path(sdata, table, table_name)
    if table_path != f"tables/{table_name}":
        raise ValueError(
            f"Table `{table_name}` is stored at `{table_path}`, but tables are written and reloaded "
            f"at `tables/{table_name}`."
        )
    return table, table_path


def _read_mapping_path(mapping: Mapping[str, object], keys: tuple[str, ...]) -> tuple[bool, object]:
    current: object = mapping
    for index, key in enumerate(keys):
        if not isinstance(current, Mapping):
            parent = ".".join(keys[:index])
            raise ValueError(f"Table uns path `{parent}` must contain a mapping.")
        if key not in current:
            return False, _MISSING
        current = current[key]
    return True, current


def _replace_mapping_path(
    mapping: dict[str, object],
    keys: tuple[str, ...],
    value: object,
) -> dict[str, object]:
    result = dict(mapping)
    key = keys[0]
    if len(keys) == 1:
        if value is _MISSING:
            result.pop(key, None)
        else:
            result[key] = value
        return result

    child = result.get(key)
    if child is None:
        if value is _MISSING:
            # Nothing to remove under a parent that does not exist. A full
            # reload lists every top-level key and drops the nested paths below
            # them, so only a nested path whose parent exists neither in memory
            # nor in the store gets here, such as a dirty path left by an edit
            # outside Spatiato.
            return result
        child_mapping: dict[str, object] = {}
    elif isinstance(child, Mapping):
        child_mapping = dict(child)
    else:
        raise ValueError(f"Table uns path `{key}` must contain a mapping.")
    result[key] = _replace_mapping_path(child_mapping, keys[1:], value)
    return result


def _remove_redundant_uns_descendants(
    paths: frozenset[TableComponentPath],
) -> frozenset[TableComponentPath]:
    result = set(paths)
    uns_paths = tuple(path for path in paths if path.component == "uns")
    for path in uns_paths:
        if any(
            other != path and len(other.keys) < len(path.keys) and path.keys[: len(other.keys)] == other.keys
            for other in uns_paths
        ):
            result.discard(path)
    return frozenset(result)


def _validate_reload_binding(
    table: AnnData,
    *,
    table_name: str,
    disk_obs: object,
    disk_attrs: object,
    region_name: str | None,
) -> None:
    if not isinstance(disk_obs, pd.DataFrame):
        raise ValueError(f"Reloaded obs for table `{table_name}` must be a DataFrame.")
    if not isinstance(disk_attrs, Mapping):
        raise ValueError(f"Reloaded table `{table_name}` is missing `{TableModel.ATTRS_KEY}` metadata in uns.")
    current_attrs = table.uns.get(TableModel.ATTRS_KEY)
    if not isinstance(current_attrs, Mapping):
        raise ValueError(
            f"The current in-memory table `{table_name}` is missing `{TableModel.ATTRS_KEY}` metadata in uns."
        )

    disk_region_key = disk_attrs.get(TableModel.REGION_KEY_KEY)
    disk_instance_key = disk_attrs.get(TableModel.INSTANCE_KEY)
    current_region_key = current_attrs.get(TableModel.REGION_KEY_KEY)
    current_instance_key = current_attrs.get(TableModel.INSTANCE_KEY)
    if not disk_region_key or not disk_instance_key:
        raise ValueError(f"Reloaded table `{table_name}` is missing required SpatialData table linkage metadata.")
    if disk_region_key != current_region_key:
        raise ValueError(
            f"Cannot reload table `{table_name}`: disk uses region key `{disk_region_key}` but memory uses "
            f"`{current_region_key}`."
        )
    if disk_instance_key != current_instance_key:
        raise ValueError(
            f"Cannot reload table `{table_name}`: disk uses instance key `{disk_instance_key}` but memory uses "
            f"`{current_instance_key}`."
        )
    if disk_region_key not in disk_obs or disk_instance_key not in disk_obs:
        raise ValueError(f"Reloaded table `{table_name}` is missing required linkage columns in obs.")
    if len(disk_obs) != table.n_obs:
        raise ValueError(
            f"Cannot reload table `{table_name}`: disk has {len(disk_obs)} rows but memory has {table.n_obs}. "
            "Reload requires unchanged row identity and order."
        )

    current_regions = table.obs[current_region_key].astype("string").reset_index(drop=True)
    disk_regions = disk_obs[disk_region_key].astype("string").reset_index(drop=True)
    if not disk_regions.equals(current_regions):
        raise ValueError(
            f"Cannot reload table `{table_name}`: `{disk_region_key}` values do not exactly match row by row."
        )
    current_instances = table.obs[current_instance_key].astype("string").reset_index(drop=True)
    disk_instances = disk_obs[disk_instance_key].astype("string").reset_index(drop=True)
    if not disk_instances.equals(current_instances):
        raise ValueError(
            f"Cannot reload table `{table_name}`: `{disk_instance_key}` values do not exactly match row by row."
        )

    if region_name is not None:
        regions = _normalize_regions(disk_attrs.get(TableModel.REGION_KEY))
        if region_name not in regions:
            raise ValueError(
                f"Cannot reload table `{table_name}` for spatial element `{region_name}`: "
                "the disk table no longer annotates the selected spatial element."
            )


def _normalize_regions(region: str | Sequence[str] | None) -> tuple[str, ...]:
    if region is None:
        return ()
    if isinstance(region, str):
        return (region,)
    return tuple(str(region_name) for region_name in region)
