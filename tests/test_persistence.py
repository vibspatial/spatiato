from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import zarr
from harpy.utils._keys import _FEATURE_MATRICES_KEY
from spatialdata import SpatialData, read_zarr
from spatialdata.models import TableModel

import spatiato.core.persistence as persistence_module
import spatiato.widgets.persistence.controller as persistence_controller_module
from spatiato._app_state import SpatiatoAppState, TableStateChangedEvent
from spatiato.core.feature_matrix_metadata import CUSTOM_OBSM_SOURCE_KIND, register_feature_matrix_metadata
from spatiato.core.object_classification.annotation import USER_CLASS_COLORS_KEY, USER_CLASS_COLUMN
from spatiato.core.persistence import TableComponentPath
from spatiato.core.spatial_query import (
    CANONICAL_CACHE_PATHS,
    CANONICAL_OBSM_KEY,
    CanonicalCacheState,
    CanonicalCentersResult,
    apply_canonical_cache_update,
    build_canonical_cache_update_payload,
    inspect_canonical_cache,
)
from spatiato.core.spatialdata import get_table, get_table_metadata
from spatiato.widgets.object_classification.controller import (
    CLASSIFIER_CONFIG_KEY,
    PRED_CLASS_COLORS_KEY,
    PRED_CLASS_COLUMN,
    PRED_CONFIDENCE_COLUMN,
)
from spatiato.widgets.persistence.controller import PersistenceController


def _record_mutation(
    app_state: SpatiatoAppState,
    sdata: SpatialData,
    *paths: TableComponentPath,
    table_name: str = "table",
) -> None:
    regions = (
        get_table_metadata(sdata, table_name).regions
        if any(path.component in ("obs", "obsm") for path in paths)
        else ()
    )
    app_state.record_table_mutation(
        TableStateChangedEvent(
            sdata=sdata,
            table_name=table_name,
            paths=frozenset(paths),
            regions=regions,
            change_kind="updated",
            source="test",
        )
    )


def _apply_canonical_cache(sdata: SpatialData) -> CanonicalCentersResult:
    report = inspect_canonical_cache(sdata, table_name="table", labels_name="blobs_labels")
    centers = np.zeros((report.binding.n_obs, 3), dtype=np.float64)
    centers[:, 1:] = np.arange(report.binding.n_obs * 2, dtype=np.float64).reshape(-1, 2)
    payload = build_canonical_cache_update_payload(
        binding=report.binding,
        centers=centers,
        source_signature=report.source_signature,
    )
    cache_update = apply_canonical_cache_update(sdata, payload)
    return CanonicalCentersResult(
        source_signature=payload.source_signature,
        binding=payload.binding,
        centers=payload.centers,
        cache_update=cache_update,
    )


def test_persistence_controller_requires_backed_spatialdata(sdata_blobs: SpatialData) -> None:
    controller = PersistenceController()
    controller.bind(sdata_blobs, "table")

    with pytest.raises(ValueError, match="not backed by zarr"):
        controller.write_table_state()


def test_persistence_controller_write_of_a_clean_table_writes_nothing(
    backed_sdata_blobs: SpatialData,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app_state = SpatiatoAppState()
    controller = PersistenceController(app_state)
    controller.bind(backed_sdata_blobs, "table", "blobs_labels")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("A table without unsaved changes must not be written.")

    monkeypatch.setattr(persistence_controller_module, "write_table_components", fail_if_called)

    assert controller.write_table_state() == "tables/table"
    assert app_state.is_table_dirty(backed_sdata_blobs, "table") is False


def test_canonical_cache_update_round_trips_as_one_dirty_consistency_unit(
    backed_sdata_blobs: SpatialData,
) -> None:
    app_state = SpatiatoAppState()
    controller = PersistenceController(app_state)
    controller.bind(backed_sdata_blobs, "table", "blobs_labels")
    _apply_canonical_cache(backed_sdata_blobs)

    _record_mutation(app_state, backed_sdata_blobs, *CANONICAL_CACHE_PATHS)

    assert app_state.snapshot_table_dirty_state(backed_sdata_blobs, "table").paths == CANONICAL_CACHE_PATHS

    controller.write_table_state()
    reopened = read_zarr(backed_sdata_blobs.path)

    assert controller.has_unsynced_table_changes is False
    assert (
        inspect_canonical_cache(reopened, table_name="table", labels_name="blobs_labels").state
        is CanonicalCacheState.VALID
    )


def test_failed_second_canonical_write_acknowledges_neither_path(
    backed_sdata_blobs: SpatialData,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app_state = SpatiatoAppState()
    controller = PersistenceController(app_state)
    controller.bind(backed_sdata_blobs, "table", "blobs_labels")
    _apply_canonical_cache(backed_sdata_blobs)
    _record_mutation(app_state, backed_sdata_blobs, *CANONICAL_CACHE_PATHS)
    real_write_elem = persistence_module.ad.io.write_elem
    canonical_write_count = 0

    def fail_second_canonical_write(group, key, value, *args, **kwargs):
        nonlocal canonical_write_count
        if key == CANONICAL_OBSM_KEY:
            canonical_write_count += 1
            if canonical_write_count == 2:
                raise RuntimeError("injected canonical metadata write failure")
        return real_write_elem(group, key, value, *args, **kwargs)

    monkeypatch.setattr(persistence_module.ad.io, "write_elem", fail_second_canonical_write)

    with pytest.raises(RuntimeError, match="injected canonical metadata write failure"):
        controller.write_table_state()

    assert app_state.snapshot_table_dirty_state(backed_sdata_blobs, "table").paths == CANONICAL_CACHE_PATHS

    controller.reload_table_state()

    assert controller.has_unsynced_table_changes is False
    assert (
        inspect_canonical_cache(
            backed_sdata_blobs,
            table_name="table",
            labels_name="blobs_labels",
        ).state
        is CanonicalCacheState.INVALID
    )


def test_persistence_controller_tracks_dirty_state_per_selected_table(
    sdata_blobs: SpatialData,
    backed_sdata_blobs: SpatialData,
) -> None:
    app_state = SpatiatoAppState()
    controller = PersistenceController(app_state)
    controller.bind(sdata_blobs, "table")

    assert controller.has_unsynced_table_changes is False

    _record_mutation(app_state, sdata_blobs, TableComponentPath("obs", (USER_CLASS_COLUMN,)))

    assert controller.has_unsynced_table_changes is True

    controller.bind(backed_sdata_blobs, "table")

    assert controller.has_unsynced_table_changes is False

    controller.bind(sdata_blobs, "table")

    assert controller.has_unsynced_table_changes is True


def test_persistence_controller_reads_dirty_state_from_shared_app_state(sdata_blobs: SpatialData) -> None:
    app_state = SpatiatoAppState()
    controller = PersistenceController(app_state)
    controller.bind(sdata_blobs, "table")

    assert controller.has_unsynced_table_changes is False

    _record_mutation(app_state, sdata_blobs, TableComponentPath("obs", (USER_CLASS_COLUMN,)))

    assert controller.has_unsynced_table_changes is True


def test_persistence_controller_can_write_table_state_requires_backed_dirty_table(
    sdata_blobs: SpatialData,
    backed_sdata_blobs: SpatialData,
) -> None:
    app_state = SpatiatoAppState()
    controller = PersistenceController(app_state)
    controller.bind(sdata_blobs, "table")

    assert controller.can_sync is False
    assert controller.can_write_table_state is False

    controller.bind(backed_sdata_blobs, "table")

    assert controller.can_sync is True
    assert controller.has_unsynced_table_changes is False
    assert controller.can_write_table_state is False

    _record_mutation(app_state, backed_sdata_blobs, TableComponentPath("obs", (USER_CLASS_COLUMN,)))

    assert controller.can_sync is True
    assert controller.has_unsynced_table_changes is True
    assert controller.can_write_table_state is True

    snapshot = app_state.snapshot_table_dirty_state(backed_sdata_blobs, "table")
    app_state.acknowledge_table_write(snapshot, persisted_paths=snapshot.paths)

    assert controller.can_write_table_state is False


def test_persistence_controller_syncs_table_obs_and_colors_to_backed_store(backed_sdata_blobs: SpatialData) -> None:
    app_state = SpatiatoAppState()
    controller = PersistenceController(app_state)
    controller.bind(backed_sdata_blobs, "table")
    index = backed_sdata_blobs["table"].obs.index

    backed_sdata_blobs["table"].obs[USER_CLASS_COLUMN] = pd.Categorical(
        [pd.NA] * (backed_sdata_blobs["table"].n_obs - 1) + [2],
        categories=[2],
    )
    backed_sdata_blobs["table"].obs[PRED_CLASS_COLUMN] = pd.Categorical(
        [pd.NA] * (backed_sdata_blobs["table"].n_obs - 1) + [2],
        categories=[2],
    )
    backed_sdata_blobs["table"].obs[PRED_CONFIDENCE_COLUMN] = pd.Series(
        [np.nan] * (backed_sdata_blobs["table"].n_obs - 1) + [0.95],
        index=index,
        dtype="float64",
    )
    backed_sdata_blobs["table"].uns[USER_CLASS_COLORS_KEY] = ["#222222"]
    backed_sdata_blobs["table"].uns[PRED_CLASS_COLORS_KEY] = ["#222222"]
    backed_sdata_blobs["table"].uns[CLASSIFIER_CONFIG_KEY] = {
        "model_type": "RandomForestClassifier",
        "feature_key": "features_1",
        "table_name": "table",
        "roi_mode": "none",
        "trained": True,
        "eligible": True,
        "reason": "Ready to train.",
        "training_timestamp": "2026-04-13T09:00:00+00:00",
        "n_labeled_objects": 1,
        "n_features": 4,
        "class_labels_seen": [2],
        "rf_params": {"n_estimators": 100, "random_state": 0, "n_jobs": 1},
        "training_scope": "all",
        "training_regions": ["blobs_labels"],
        "n_training_rows": backed_sdata_blobs["table"].n_obs,
        "prediction_scope": "selected_segmentation_only",
        "prediction_regions": ["blobs_labels"],
        "n_predicted_rows": backed_sdata_blobs["table"].n_obs,
    }
    _record_mutation(
        app_state,
        backed_sdata_blobs,
        TableComponentPath("obs", (USER_CLASS_COLUMN,)),
        TableComponentPath("obs", (PRED_CLASS_COLUMN,)),
        TableComponentPath("obs", (PRED_CONFIDENCE_COLUMN,)),
        TableComponentPath("uns", (USER_CLASS_COLORS_KEY,)),
        TableComponentPath("uns", (PRED_CLASS_COLORS_KEY,)),
        TableComponentPath("uns", (CLASSIFIER_CONFIG_KEY,)),
    )

    table_path = controller.write_table_state()
    reread = read_zarr(backed_sdata_blobs.path)

    assert table_path == "tables/table"
    assert USER_CLASS_COLUMN in reread["table"].obs
    assert isinstance(reread["table"].obs[USER_CLASS_COLUMN].dtype, pd.CategoricalDtype)
    assert list(reread["table"].obs[USER_CLASS_COLUMN].cat.categories) == [2]
    assert int(reread["table"].obs[USER_CLASS_COLUMN].isna().sum()) == reread["table"].n_obs - 1
    assert reread["table"].obs[USER_CLASS_COLUMN].tolist().count(2) == 1
    assert isinstance(reread["table"].obs[PRED_CLASS_COLUMN].dtype, pd.CategoricalDtype)
    assert list(reread["table"].obs[PRED_CLASS_COLUMN].cat.categories) == [2]
    assert int(reread["table"].obs[PRED_CLASS_COLUMN].isna().sum()) == reread["table"].n_obs - 1
    assert reread["table"].obs[PRED_CLASS_COLUMN].tolist().count(2) == 1
    assert reread["table"].obs[PRED_CONFIDENCE_COLUMN].max() == 0.95
    assert list(reread["table"].uns[USER_CLASS_COLORS_KEY]) == ["#222222"]
    assert list(reread["table"].uns[PRED_CLASS_COLORS_KEY]) == ["#222222"]
    assert reread["table"].uns[CLASSIFIER_CONFIG_KEY]["feature_key"] == "features_1"
    assert reread["table"].uns[CLASSIFIER_CONFIG_KEY]["prediction_scope"] == "selected_segmentation_only"
    assert sorted(reread["table"].obsm.keys()) == ["features_1", "features_2"]


def test_persistence_controller_syncs_feature_matrix_metadata_to_backed_store(
    backed_sdata_blobs: SpatialData,
) -> None:
    app_state = SpatiatoAppState()
    controller = PersistenceController(app_state)
    controller.bind(backed_sdata_blobs, "table")
    table = backed_sdata_blobs["table"]
    feature_key = "features_1"
    feature_columns = [f"custom_feature_{index}" for index in range(table.obsm[feature_key].shape[1])]
    register_feature_matrix_metadata(
        table,
        feature_key,
        feature_columns=feature_columns,
        overwrite=True,
    )
    _record_mutation(
        app_state,
        backed_sdata_blobs,
        TableComponentPath("uns", (_FEATURE_MATRICES_KEY, feature_key)),
    )

    assert controller.has_unsynced_table_changes is True

    table_path = controller.write_table_state()
    reread = read_zarr(backed_sdata_blobs.path)
    metadata = reread["table"].uns[_FEATURE_MATRICES_KEY][feature_key]

    assert table_path == "tables/table"
    assert metadata["source_kind"] == CUSTOM_OBSM_SOURCE_KIND
    assert list(metadata["feature_columns"]) == feature_columns
    assert controller.has_unsynced_table_changes is False


def test_persistence_controller_creates_and_removes_nested_uns_entry(
    backed_sdata_blobs: SpatialData,
) -> None:
    app_state = SpatiatoAppState()
    controller = PersistenceController(app_state)
    controller.bind(backed_sdata_blobs, "table")
    table = backed_sdata_blobs["table"]
    path = TableComponentPath("uns", ("extension_metadata", "result"))
    table.uns["extension_metadata"] = {"result": {"version": 1}}
    _record_mutation(app_state, backed_sdata_blobs, path)

    controller.write_table_state()

    assert read_zarr(backed_sdata_blobs.path)["table"].uns["extension_metadata"]["result"]["version"] == 1

    del table.uns["extension_metadata"]["result"]
    _record_mutation(app_state, backed_sdata_blobs, path)
    controller.write_table_state()

    assert "result" not in read_zarr(backed_sdata_blobs.path)["table"].uns["extension_metadata"]


def test_persistence_controller_reloads_registered_custom_feature_matrix_metadata(
    backed_sdata_blobs: SpatialData,
) -> None:
    app_state = SpatiatoAppState()
    controller = PersistenceController(app_state)
    controller.bind(backed_sdata_blobs, "table")
    table = backed_sdata_blobs["table"]
    feature_key = "features_1"
    feature_columns = [f"custom_feature_{index}" for index in range(table.obsm[feature_key].shape[1])]
    register_feature_matrix_metadata(
        table,
        feature_key,
        feature_columns=feature_columns,
        overwrite=True,
    )
    _record_mutation(
        app_state,
        backed_sdata_blobs,
        TableComponentPath("uns", (_FEATURE_MATRICES_KEY, feature_key)),
    )
    controller.write_table_state()

    disk_features = np.asarray(table.obsm[feature_key]).copy()
    table.obsm[feature_key] = np.zeros_like(disk_features)
    table.uns[_FEATURE_MATRICES_KEY][feature_key] = {"source_kind": "stale_in_memory"}
    _record_mutation(
        app_state,
        backed_sdata_blobs,
        TableComponentPath("obsm", (feature_key,)),
        TableComponentPath("uns", (_FEATURE_MATRICES_KEY, feature_key)),
        TableComponentPath("obs", (USER_CLASS_COLUMN,)),
    )

    emitted_events: list[TableStateChangedEvent] = []
    app_state.table_state_changed.connect(emitted_events.append)
    result = controller.reload_table_components(
        frozenset(
            {
                TableComponentPath("obsm", (feature_key,)),
                TableComponentPath("uns", (_FEATURE_MATRICES_KEY, feature_key)),
            }
        )
    )
    metadata = table.uns[_FEATURE_MATRICES_KEY][feature_key]

    assert result.table_path == "tables/table"
    assert np.array_equal(table.obsm[feature_key], disk_features)
    assert metadata["source_kind"] == CUSTOM_OBSM_SOURCE_KIND
    assert list(metadata["feature_columns"]) == feature_columns
    assert len(emitted_events) == 1
    assert emitted_events[0].regions == get_table_metadata(backed_sdata_blobs, "table").regions
    assert app_state.snapshot_table_dirty_state(backed_sdata_blobs, "table").paths == frozenset(
        {TableComponentPath("obs", (USER_CLASS_COLUMN,))}
    )


def test_persistence_controller_syncs_multi_region_classifier_config_fields(
    backed_sdata_blobs_multi_region: SpatialData,
) -> None:
    app_state = SpatiatoAppState()
    controller = PersistenceController(app_state)
    controller.bind(backed_sdata_blobs_multi_region, "table_multi")
    table = backed_sdata_blobs_multi_region["table_multi"]
    index = table.obs.index
    regions = ["blobs_labels", "blobs_labels_2"]
    region_values = table.obs["region"].astype("string")
    pred_values = np.where(region_values == "blobs_labels", 1, 2)

    table.obs[USER_CLASS_COLUMN] = pd.Categorical(pred_values, categories=[1, 2])
    table.obs[PRED_CLASS_COLUMN] = pd.Categorical(pred_values, categories=[1, 2])
    table.obs[PRED_CONFIDENCE_COLUMN] = pd.Series(np.full(table.n_obs, 0.87), index=index, dtype="float64")
    table.uns[USER_CLASS_COLORS_KEY] = ["#111111", "#222222"]
    table.uns[PRED_CLASS_COLORS_KEY] = ["#111111", "#222222"]
    table.uns[CLASSIFIER_CONFIG_KEY] = {
        "model_type": "RandomForestClassifier",
        "feature_key": "features_1",
        "table_name": "table_multi",
        "roi_mode": "none",
        "trained": True,
        "eligible": True,
        "reason": "Ready to train.",
        "training_timestamp": "2026-04-13T09:00:00+00:00",
        "n_labeled_objects": int(table.n_obs),
        "n_features": int(table.obsm["features_1"].shape[1]),
        "class_labels_seen": [1, 2],
        "rf_params": {"n_estimators": 100, "random_state": 0, "n_jobs": 1},
        "training_scope": "all",
        "training_regions": regions,
        "n_training_rows": int(table.n_obs),
        "prediction_scope": "all",
        "prediction_regions": regions,
        "n_predicted_rows": int(table.n_obs),
    }
    _record_mutation(
        app_state,
        backed_sdata_blobs_multi_region,
        TableComponentPath("obs", (USER_CLASS_COLUMN,)),
        TableComponentPath("obs", (PRED_CLASS_COLUMN,)),
        TableComponentPath("obs", (PRED_CONFIDENCE_COLUMN,)),
        TableComponentPath("uns", (USER_CLASS_COLORS_KEY,)),
        TableComponentPath("uns", (PRED_CLASS_COLORS_KEY,)),
        TableComponentPath("uns", (CLASSIFIER_CONFIG_KEY,)),
        table_name="table_multi",
    )

    table_path = controller.write_table_state()
    reread = read_zarr(backed_sdata_blobs_multi_region.path)
    config = reread["table_multi"].uns[CLASSIFIER_CONFIG_KEY]

    assert table_path == "tables/table_multi"
    assert config["table_name"] == "table_multi"
    assert config["feature_key"] == "features_1"
    assert config["training_scope"] == "all"
    assert list(config["training_regions"]) == regions
    assert config["n_training_rows"] == table.n_obs
    assert config["prediction_scope"] == "all"
    assert list(config["prediction_regions"]) == regions
    assert config["n_predicted_rows"] == table.n_obs
    assert reread["table_multi"].obs[PRED_CLASS_COLUMN].tolist().count(1) == int(
        (region_values == "blobs_labels").sum()
    )
    assert reread["table_multi"].obs[PRED_CLASS_COLUMN].tolist().count(2) == int(
        (region_values == "blobs_labels_2").sum()
    )


def _write_disk_snapshot_payload(
    backed_sdata_blobs: SpatialData,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, object]]:
    table = backed_sdata_blobs["table"]
    obs = table.obs.copy()
    obs["disk_user_class"] = pd.Categorical([pd.NA] * (table.n_obs - 1) + [7], categories=[7])

    obsm = dict(table.obsm)
    obsm["disk_features"] = np.arange(table.n_obs, dtype=np.float64).reshape(table.n_obs, 1)

    uns = dict(table.uns)
    uns["disk_flag"] = {"source": "disk"}

    _write_disk_snapshot_state(backed_sdata_blobs, obs=obs, obsm=obsm, uns=uns)
    return obs, obsm, uns


def _write_disk_snapshot_state(
    backed_sdata_blobs: SpatialData,
    *,
    obs: pd.DataFrame,
    obsm: dict[str, object],
    uns: dict[str, object],
) -> None:
    root = zarr.open_group(backed_sdata_blobs.path, mode="a", use_consolidated=False)
    table_group = root["tables/table"]
    ad.io.write_elem(table_group, "obs", obs)
    ad.io.write_elem(table_group, "obsm", obsm)
    ad.io.write_elem(table_group, "uns", uns)


def test_persistence_controller_replaces_selected_table_with_reloaded_snapshot(
    backed_sdata_blobs: SpatialData,
) -> None:
    controller = PersistenceController()
    controller.bind(backed_sdata_blobs, "table")
    _write_disk_snapshot_payload(backed_sdata_blobs)
    original_table = backed_sdata_blobs["table"]

    table_path = controller.reload_table_state()
    reloaded_table = backed_sdata_blobs["table"]

    assert table_path == "tables/table"
    assert reloaded_table is original_table
    assert "disk_user_class" in reloaded_table.obs
    assert "disk_features" in reloaded_table.obsm
    assert reloaded_table.uns["disk_flag"] == {"source": "disk"}
    assert list(reloaded_table.var_names) == list(original_table.var_names)
    assert backed_sdata_blobs.locate_element(reloaded_table) == ["tables/table"]


def test_persistence_controller_captures_exact_reload_request(
    backed_sdata_blobs: SpatialData,
) -> None:
    controller = PersistenceController()
    controller.bind(backed_sdata_blobs, "table", "blobs_labels")

    request = controller.capture_table_reload_request(source="object_classification")

    assert request.sdata is backed_sdata_blobs
    assert request.table_name == "table"
    assert request.region_name == "blobs_labels"
    assert request.source == "object_classification"
    assert request.paths


def test_persistence_controller_clears_dirty_state_after_sync(backed_sdata_blobs: SpatialData) -> None:
    app_state = SpatiatoAppState()
    controller = PersistenceController(app_state)
    controller.bind(backed_sdata_blobs, "table")
    _record_mutation(app_state, backed_sdata_blobs, TableComponentPath("obs", (USER_CLASS_COLUMN,)))

    assert controller.has_unsynced_table_changes is True

    controller.write_table_state()

    assert controller.has_unsynced_table_changes is False


def test_persistence_controller_clears_dirty_state_after_reload(backed_sdata_blobs: SpatialData) -> None:
    app_state = SpatiatoAppState()
    controller = PersistenceController(app_state)
    controller.bind(backed_sdata_blobs, "table")
    _write_disk_snapshot_payload(backed_sdata_blobs)
    _record_mutation(app_state, backed_sdata_blobs, TableComponentPath("obs", (USER_CLASS_COLUMN,)))

    assert controller.has_unsynced_table_changes is True

    controller.reload_table_state()

    assert controller.has_unsynced_table_changes is False


def test_persistence_controller_reloads_obsm_key_written_directly_to_disk_group(
    backed_sdata_blobs: SpatialData,
) -> None:
    """Cover direct `tables/<table>/obsm/<key>` writes that future extensions will rely on.

    This is an important regression test because plugin extensions may add new
    feature matrices straight to the backed zarr group without rewriting the
    full `obsm` mapping. Reload must continue to discover those incrementally
    written keys and surface them on the in-memory table.
    """
    controller = PersistenceController()
    controller.bind(backed_sdata_blobs, "table")
    table = backed_sdata_blobs["table"]
    features_3 = np.arange(table.n_obs * 3, dtype=np.float32).reshape(table.n_obs, 3)

    root = zarr.open_group(backed_sdata_blobs.path, mode="a", use_consolidated=False)
    table_group = root["tables/table"]
    ad.io.write_elem(table_group["obsm"], "features_3", features_3)

    assert "features_3" not in table.obsm

    table_path = controller.reload_table_state()

    assert table_path == "tables/table"
    assert sorted(table.obsm.keys()) == ["features_1", "features_2", "features_3"]
    assert np.array_equal(table.obsm["features_3"], features_3)


def test_persistence_controller_normalizes_numpy_array_region_attrs_when_reloading(
    backed_sdata_blobs: SpatialData,
) -> None:
    controller = PersistenceController()
    controller.bind(backed_sdata_blobs, "table", "blobs_labels")
    table = backed_sdata_blobs["table"]

    obs = table.obs.copy()
    obsm = dict(table.obsm)
    uns = dict(table.uns)
    uns[TableModel.ATTRS_KEY] = {
        **uns[TableModel.ATTRS_KEY],
        TableModel.REGION_KEY: np.array(["blobs_labels"]),
    }
    _write_disk_snapshot_state(backed_sdata_blobs, obs=obs, obsm=obsm, uns=uns)

    table_path = controller.reload_table_state()

    assert table_path == "tables/table"
    assert backed_sdata_blobs["table"].uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY] == ["blobs_labels"]
    assert get_table(backed_sdata_blobs, "table") is backed_sdata_blobs["table"]


def test_persistence_controller_rejects_reload_when_row_count_changed(
    backed_sdata_blobs: SpatialData,
) -> None:
    controller = PersistenceController()
    controller.bind(backed_sdata_blobs, "table", "blobs_labels")
    table = backed_sdata_blobs["table"]

    obs = table.obs.iloc[:-1].copy()
    obsm = {key: value[:-1] for key, value in table.obsm.items()}
    uns = dict(table.uns)
    _write_disk_snapshot_state(backed_sdata_blobs, obs=obs, obsm=obsm, uns=uns)

    with pytest.raises(ValueError, match="requires unchanged row identity and order"):
        controller.reload_table_state()

    assert backed_sdata_blobs["table"].n_obs == table.n_obs
    assert backed_sdata_blobs["table"].obs.index.equals(table.obs.index)


def test_persistence_controller_allows_reload_when_obs_names_change_but_rowwise_identity_matches(
    backed_sdata_blobs: SpatialData,
) -> None:
    controller = PersistenceController()
    controller.bind(backed_sdata_blobs, "table", "blobs_labels")
    table = backed_sdata_blobs["table"]

    obs = table.obs.copy()
    obs.index = pd.Index([f"disk_row_{index}" for index in range(table.n_obs)], name=obs.index.name)
    obsm = dict(table.obsm)
    uns = dict(table.uns)
    _write_disk_snapshot_state(backed_sdata_blobs, obs=obs, obsm=obsm, uns=uns)

    table_path = controller.reload_table_state()

    assert table_path == "tables/table"
    assert backed_sdata_blobs["table"].obs.index.equals(obs.index)
    assert backed_sdata_blobs["table"].obs["instance_id"].equals(obs["instance_id"])


def test_persistence_controller_rejects_reload_when_region_key_values_changed_rowwise(
    backed_sdata_blobs: SpatialData,
) -> None:
    controller = PersistenceController()
    controller.bind(backed_sdata_blobs, "table", "blobs_labels")
    table = backed_sdata_blobs["table"]

    obs = table.obs.copy()
    obs["region"] = obs["region"].astype(object)
    obs.loc[obs.index[0], "region"] = "different_labels"
    obsm = dict(table.obsm)
    uns = dict(table.uns)
    _write_disk_snapshot_state(backed_sdata_blobs, obs=obs, obsm=obsm, uns=uns)

    with pytest.raises(ValueError, match="`region` values do not exactly match"):
        controller.reload_table_state()

    assert backed_sdata_blobs["table"].obs["region"].equals(table.obs["region"])


def test_persistence_controller_rejects_reload_when_row_order_changed(
    backed_sdata_blobs: SpatialData,
) -> None:
    controller = PersistenceController()
    controller.bind(backed_sdata_blobs, "table", "blobs_labels")
    table = backed_sdata_blobs["table"]
    reversed_positions = np.arange(table.n_obs - 1, -1, -1)

    obs = table.obs.iloc[reversed_positions].copy()
    obsm = {key: value[reversed_positions] for key, value in table.obsm.items()}
    uns = dict(table.uns)
    _write_disk_snapshot_state(backed_sdata_blobs, obs=obs, obsm=obsm, uns=uns)

    with pytest.raises(ValueError, match="instance_id` values do not exactly match"):
        controller.reload_table_state()

    assert backed_sdata_blobs["table"].obs.index.equals(table.obs.index)


def test_persistence_controller_rejects_reload_when_instance_key_values_changed_rowwise(
    backed_sdata_blobs: SpatialData,
) -> None:
    controller = PersistenceController()
    controller.bind(backed_sdata_blobs, "table", "blobs_labels")
    table = backed_sdata_blobs["table"]

    obs = table.obs.copy()
    swapped = obs["instance_id"].to_numpy(copy=True)
    swapped[[0, 1]] = swapped[[1, 0]]
    obs["instance_id"] = swapped
    obsm = dict(table.obsm)
    uns = dict(table.uns)
    _write_disk_snapshot_state(backed_sdata_blobs, obs=obs, obsm=obsm, uns=uns)

    with pytest.raises(ValueError, match="instance_id` values do not exactly match"):
        controller.reload_table_state()

    assert backed_sdata_blobs["table"].obs["instance_id"].equals(table.obs["instance_id"])


def test_persistence_controller_rejects_reload_when_spatialdata_attrs_missing(
    backed_sdata_blobs: SpatialData,
) -> None:
    controller = PersistenceController()
    controller.bind(backed_sdata_blobs, "table", "blobs_labels")
    table = backed_sdata_blobs["table"]

    obs = table.obs.copy()
    obsm = dict(table.obsm)
    uns = {"disk_flag": {"source": "disk"}}
    _write_disk_snapshot_state(backed_sdata_blobs, obs=obs, obsm=obsm, uns=uns)

    with pytest.raises(ValueError, match="missing `spatialdata_attrs` metadata"):
        controller.reload_table_state()


def test_persistence_controller_rejects_reload_when_selected_region_is_no_longer_annotated(
    backed_sdata_blobs: SpatialData,
) -> None:
    controller = PersistenceController()
    controller.bind(backed_sdata_blobs, "table", "blobs_labels")
    table = backed_sdata_blobs["table"]

    obs = table.obs.copy()
    obsm = dict(table.obsm)
    uns = dict(table.uns)
    uns["spatialdata_attrs"] = {
        **uns["spatialdata_attrs"],
        "region": ["different_labels"],
    }
    _write_disk_snapshot_state(backed_sdata_blobs, obs=obs, obsm=obsm, uns=uns)

    with pytest.raises(ValueError, match="no longer annotates the selected spatial element"):
        controller.reload_table_state()
