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


def test_failed_write_acknowledges_none_of_its_paths(
    backed_sdata_blobs: SpatialData,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app_state = SpatiatoAppState()
    controller = PersistenceController(app_state)
    controller.bind(backed_sdata_blobs, "table", "blobs_labels")
    table = backed_sdata_blobs["table"]
    obsm_path = TableComponentPath("obsm", ("features_1",))
    uns_path = TableComponentPath("uns", ("extension_metadata", "result"))
    table.obsm["features_1"] = np.zeros_like(np.asarray(table.obsm["features_1"]))
    table.uns["extension_metadata"] = {"result": {"version": 1}}
    _record_mutation(app_state, backed_sdata_blobs, obsm_path, uns_path)
    real_write_elem = persistence_module.ad.io.write_elem

    def fail_uns_write(group, key, value, *args, **kwargs):
        if key == "result":
            raise RuntimeError("injected uns write failure")
        return real_write_elem(group, key, value, *args, **kwargs)

    # The obsm entry is written first; the uns entry then fails.
    monkeypatch.setattr(persistence_module.ad.io, "write_elem", fail_uns_write)
    with pytest.raises(RuntimeError, match="injected uns write failure"):
        controller.write_table_state()

    # Neither path is acknowledged, also the obsm entry that was written.
    assert app_state.snapshot_table_dirty_state(backed_sdata_blobs, "table").paths == {obsm_path, uns_path}

    # So Write can be retried.
    monkeypatch.setattr(persistence_module.ad.io, "write_elem", real_write_elem)
    controller.write_table_state()

    reread = read_zarr(backed_sdata_blobs.path)["table"]
    assert controller.has_unsynced_table_changes is False
    np.testing.assert_array_equal(np.asarray(reread.obsm["features_1"]), 0)
    assert reread.uns["extension_metadata"]["result"]["version"] == 1


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
        PRED_CLASS_COLUMN: {
            "source": "widget",
            "pred_class_column": PRED_CLASS_COLUMN,
            "pred_confidence_column": PRED_CONFIDENCE_COLUMN,
            "feature_key": "features_1",
            "prediction_regions": ["blobs_labels"],
            "prediction_timestamp": "2026-04-13T09:00:00+00:00",
            "prediction_scope": "selected_segmentation_only",
            "classifier_path": None,
            "reason": None,
            "model_provenance": {
                "feature_key": "features_1",
                "model_type": "RandomForestClassifier",
                "rf_params": {"n_estimators": 100, "random_state": 0, "n_jobs": 1},
                "training_scope": "all",
                "training_regions": ["blobs_labels"],
                "n_features": 4,
                "training_timestamp": "2026-04-13T09:00:00+00:00",
                "n_training_labels": 1,
                "class_labels_seen": [2],
            },
        }
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
    assert reread["table"].uns[CLASSIFIER_CONFIG_KEY][PRED_CLASS_COLUMN]["feature_key"] == "features_1"
    assert (
        reread["table"].uns[CLASSIFIER_CONFIG_KEY][PRED_CLASS_COLUMN]["prediction_scope"]
        == "selected_segmentation_only"
    )
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
        PRED_CLASS_COLUMN: {
            "source": "widget",
            "pred_class_column": PRED_CLASS_COLUMN,
            "pred_confidence_column": PRED_CONFIDENCE_COLUMN,
            "feature_key": "features_1",
            "prediction_regions": regions,
            "prediction_timestamp": "2026-04-13T09:00:00+00:00",
            "prediction_scope": "all",
            "classifier_path": None,
            "reason": None,
            "model_provenance": {
                "feature_key": "features_1",
                "model_type": "RandomForestClassifier",
                "rf_params": {"n_estimators": 100, "random_state": 0, "n_jobs": 1},
                "training_scope": "all",
                "training_regions": regions,
                "n_features": int(table.obsm["features_1"].shape[1]),
                "training_timestamp": "2026-04-13T09:00:00+00:00",
                "n_training_labels": int(table.n_obs),
                "class_labels_seen": [1, 2],
            },
        }
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
    entry = reread["table_multi"].uns[CLASSIFIER_CONFIG_KEY][PRED_CLASS_COLUMN]

    assert table_path == "tables/table_multi"
    assert entry["feature_key"] == "features_1"
    assert entry["model_provenance"]["training_scope"] == "all"
    assert list(entry["model_provenance"]["training_regions"]) == regions
    assert entry["prediction_scope"] == "all"
    assert list(entry["prediction_regions"]) == regions
    assert entry["model_provenance"]["n_training_labels"] == table.n_obs
    # A trained entry has no reason; `None` survives the round trip.
    assert "reason" in entry
    assert entry["reason"] is None
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
