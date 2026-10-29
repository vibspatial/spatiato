from __future__ import annotations

import harpy as hp
import numpy as np
import pandas as pd
import pytest
from harpy.table._add_feature_matrix import _SOURCE_KIND
from harpy.table._metadata import _FEATURE_MATRIX_SCHEMA_VERSION
from loguru import logger
from spatialdata import SpatialData

from spatiato.core.feature_extraction import (
    ExistingFeatureMatrix,
    FeatureExtractionTriplet,
    FeatureMatrixChangedError,
    FeatureMatrixRemovedError,
    calculate_feature_matrix,
    check_existing_feature_matrix,
)
from spatiato.core.feature_matrix_metadata import (
    FEATURE_MATRIX_METADATA_SCHEMA_VERSION,
    HARPY_ADD_FEATURE_MATRIX_SOURCE_KIND,
    register_feature_matrix_metadata,
)
from spatiato.core.spatialdata import get_table_metadata

_TABLE = "table_multi"
_KEY = "computed"
_TRIPLETS = {
    "blobs_labels": FeatureExtractionTriplet("global", "blobs_labels", "blobs_image"),
    "blobs_labels_2": FeatureExtractionTriplet("global_1", "blobs_labels_2", "blobs_image_2"),
}


def _triplets(
    *labels_names: str, channels: tuple[int | str, ...] | None = None
) -> tuple[FeatureExtractionTriplet, ...]:
    return tuple(
        FeatureExtractionTriplet(
            _TRIPLETS[name].coordinate_system,
            name,
            _TRIPLETS[name].image_name,
            channels,
        )
        for name in labels_names
    )


def _calculate(
    sdata: SpatialData,
    *labels_names: str,
    feature_names: tuple[str, ...] = ("area",),
    channels: tuple[int | str, ...] | None = None,
    **kwargs: object,
):
    return calculate_feature_matrix(
        sdata,
        table_name=_TABLE,
        feature_key=_KEY,
        triplets=_triplets(*labels_names, channels=channels),
        feature_names=feature_names,
        **kwargs,
    )


def _check(
    sdata: SpatialData,
    *labels_names: str,
    feature_names: tuple[str, ...] = ("area",),
    channels: tuple[int | str, ...] | None = None,
):
    return check_existing_feature_matrix(
        sdata,
        table_name=_TABLE,
        feature_key=_KEY,
        triplets=_triplets(*labels_names, channels=channels),
        feature_names=feature_names,
    )


def _rows(sdata: SpatialData, region: str) -> np.ndarray:
    table = sdata[_TABLE]
    mask = (table.obs["region"] == region).to_numpy()
    return np.asarray(table.obsm[_KEY])[mask]


def test_metadata_constants_match_harpy() -> None:
    assert HARPY_ADD_FEATURE_MATRIX_SOURCE_KIND == _SOURCE_KIND
    assert FEATURE_MATRIX_METADATA_SCHEMA_VERSION == _FEATURE_MATRIX_SCHEMA_VERSION


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"kind": "unknown"}, "Unsupported existing feature matrix kind"),
        ({"kind": "incompatible"}, "requires a `reason`"),
        ({"kind": "compatible", "reason": "stale"}, '`reason` only applies to kind "incompatible"'),
        ({"kind": "compatible", "orphaned_metadata": True}, '`orphaned_metadata` only applies to kind "absent"'),
    ],
)
def test_existing_feature_matrix_rejects_fields_of_other_kinds(fields: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ExistingFeatureMatrix(**fields)


def test_calculate_feature_matrix_creates_new_key(sdata_blobs_multi_region: SpatialData) -> None:
    result = _calculate(sdata_blobs_multi_region, "blobs_labels")

    assert result.existing.kind == "absent"
    assert result.change_kind == "created"
    assert result.regions == ("blobs_labels",)
    assert not np.isnan(_rows(sdata_blobs_multi_region, "blobs_labels")).any()
    assert np.isnan(_rows(sdata_blobs_multi_region, "blobs_labels_2")).all()


def test_calculate_feature_matrix_updates_compatible_matrix_by_region(sdata_blobs_multi_region: SpatialData) -> None:
    _calculate(sdata_blobs_multi_region, "blobs_labels_2")
    kept = _rows(sdata_blobs_multi_region, "blobs_labels_2").copy()

    result = _calculate(sdata_blobs_multi_region, "blobs_labels", overwrite_feature_key=True)

    assert result.existing.kind == "compatible"
    assert result.existing.unselected_regions == ("blobs_labels_2",)
    assert result.change_kind == "updated"
    assert result.regions == ("blobs_labels",)
    assert not np.isnan(_rows(sdata_blobs_multi_region, "blobs_labels")).any()
    np.testing.assert_array_equal(_rows(sdata_blobs_multi_region, "blobs_labels_2"), kept)


def test_calculate_feature_matrix_requires_overwrite_for_existing_matrix(
    sdata_blobs_multi_region: SpatialData,
) -> None:
    _calculate(sdata_blobs_multi_region, "blobs_labels_2")

    with pytest.raises(ValueError, match="already exists"):
        _calculate(sdata_blobs_multi_region, "blobs_labels")


def test_calculate_feature_matrix_replaces_custom_matrix_entirely(sdata_blobs_multi_region: SpatialData) -> None:
    table = sdata_blobs_multi_region[_TABLE]
    table.obsm[_KEY] = np.ones((table.n_obs, 3))
    messages: list[str] = []
    sink_id = logger.add(lambda message: messages.append(str(message)), format="{message}", level="WARNING")
    try:
        result = _calculate(sdata_blobs_multi_region, "blobs_labels", overwrite_feature_key=True)
    finally:
        logger.remove(sink_id)

    assert result.existing.kind == "incompatible"
    assert result.existing.reason == "it has no harpy feature metadata"
    assert result.change_kind == "updated"
    assert result.regions == ("blobs_labels", "blobs_labels_2")
    assert table.obsm[_KEY].shape == (table.n_obs, 1)
    assert np.isnan(_rows(sdata_blobs_multi_region, "blobs_labels_2")).all()
    assert list(table.uns["feature_matrices"][_KEY]["feature_columns"]) == ["area"]
    assert any('"blobs_labels_2"' in message and "lost" in message for message in messages)


def test_check_existing_feature_matrix_reports_custom_registration(sdata_blobs_multi_region: SpatialData) -> None:
    table = sdata_blobs_multi_region[_TABLE]
    table.obsm[_KEY] = np.ones((table.n_obs, 1))
    register_feature_matrix_metadata(table, _KEY, feature_columns=("area",))

    existing = _check(sdata_blobs_multi_region, "blobs_labels")

    assert existing.kind == "incompatible"
    assert existing.reason == "it is registered as a custom matrix"


def test_check_existing_feature_matrix_compares_features_and_channels(sdata_blobs_multi_region: SpatialData) -> None:
    _calculate(sdata_blobs_multi_region, "blobs_labels_2", feature_names=("area", "mean"), channels=(0,))

    # harpy orders intensity columns before morphology columns, and channel
    # names and indices resolve to the same channel.
    assert _check(sdata_blobs_multi_region, "blobs_labels", feature_names=("mean", "area"), channels=("0",)).kind == (
        "compatible"
    )
    assert _check(sdata_blobs_multi_region, "blobs_labels", feature_names=("area",), channels=(0,)).kind == (
        "incompatible"
    )
    changed_channels = _check(sdata_blobs_multi_region, "blobs_labels", feature_names=("area", "mean"), channels=(1,))
    assert changed_channels.kind == "incompatible"
    assert changed_channels.reason == "it was calculated with different features or channels"


def test_check_existing_feature_matrix_treats_metadata_without_matrix_as_none(
    sdata_blobs_multi_region: SpatialData,
) -> None:
    _calculate(sdata_blobs_multi_region, "blobs_labels")
    del sdata_blobs_multi_region[_TABLE].obsm[_KEY]

    existing = _check(sdata_blobs_multi_region, "blobs_labels")
    result = _calculate(sdata_blobs_multi_region, "blobs_labels")

    assert existing.kind == "absent"
    assert existing.orphaned_metadata is True
    assert result.change_kind == "created"


def test_calculate_feature_matrix_rejects_changed_confirmation_before_changes(
    sdata_blobs_multi_region: SpatialData,
) -> None:
    table = sdata_blobs_multi_region[_TABLE]
    custom_matrix = np.ones((table.n_obs, 3))
    table.obsm[_KEY] = custom_matrix

    with pytest.raises(FeatureMatrixChangedError, match="changed since you confirmed"):
        _calculate(
            sdata_blobs_multi_region,
            "blobs_labels",
            overwrite_feature_key=True,
            confirmed_existing_entry="compatible",
        )

    assert table.obsm[_KEY] is custom_matrix


def test_calculate_feature_matrix_reports_removal_when_extraction_fails(
    sdata_blobs_multi_region: SpatialData,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = sdata_blobs_multi_region[_TABLE]
    table.obsm[_KEY] = np.ones((table.n_obs, 3))

    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("extraction failed")

    monkeypatch.setattr(hp.tb, "add_feature_matrix", fail)

    with pytest.raises(FeatureMatrixRemovedError, match="was removed: extraction failed") as error_info:
        _calculate(sdata_blobs_multi_region, "blobs_labels", overwrite_feature_key=True)

    assert error_info.value.regions == ("blobs_labels", "blobs_labels_2")
    assert _KEY not in table.obsm


def test_calculate_feature_matrix_validates_labels_before_removal(sdata_blobs_multi_region: SpatialData) -> None:
    table = sdata_blobs_multi_region[_TABLE]
    custom_matrix = np.ones((table.n_obs, 3))
    table.obsm[_KEY] = custom_matrix

    with pytest.raises(ValueError, match="has no observations"):
        calculate_feature_matrix(
            sdata_blobs_multi_region,
            table_name=_TABLE,
            feature_key=_KEY,
            triplets=(FeatureExtractionTriplet("global", "blobs_multiscale_labels", None),),
            feature_names=("area",),
            overwrite_feature_key=True,
        )

    assert table.obsm[_KEY] is custom_matrix


def test_check_existing_feature_matrix_rejects_duplicate_rows(sdata_blobs_multi_region: SpatialData) -> None:
    table = sdata_blobs_multi_region[_TABLE]
    instance_key = get_table_metadata(sdata_blobs_multi_region, _TABLE).instance_key
    instance_ids = table.obs[instance_key].to_numpy().copy()
    instance_ids[1] = instance_ids[0]
    table.obs[instance_key] = instance_ids

    with pytest.raises(ValueError, match="pairs must be non-null and unique"):
        _check(sdata_blobs_multi_region, "blobs_labels")


@pytest.mark.parametrize("existing_kind", ["compatible", "incompatible"])
def test_backed_rows_differing_from_store_raise_before_removal(
    backed_sdata_blobs_multi_region: SpatialData,
    existing_kind: str,
) -> None:
    sdata = backed_sdata_blobs_multi_region
    table = sdata[_TABLE]
    table_metadata = get_table_metadata(sdata, _TABLE)
    if existing_kind == "compatible":
        _calculate(sdata, "blobs_labels_2")
    else:
        hp.tb.io.add_table_components(
            sdata,
            table_name=_TABLE,
            components={("obsm", _KEY): np.ones((table.n_obs, 3))},
            obs_identity=table.obs[[table_metadata.region_key, table_metadata.instance_key]],
        )
    assert _check(sdata, "blobs_labels").kind == existing_kind
    # Swap two rows in memory only.
    instance_ids = table.obs[table_metadata.instance_key].to_numpy().copy()
    instance_ids[[0, 1]] = instance_ids[[1, 0]]
    table.obs[table_metadata.instance_key] = instance_ids

    with pytest.raises(ValueError, match="differ in order"):
        _calculate(sdata, "blobs_labels", overwrite_feature_key=True)

    stored = hp.tb.io.read_table_components(sdata.path, table_name=_TABLE, components=[("obsm", _KEY)], mode="backed")
    assert ("obsm", _KEY) in stored
    assert _KEY in table.obsm


def test_backed_compatible_update_keeps_stored_values_of_other_regions(
    backed_sdata_blobs_multi_region: SpatialData,
) -> None:
    sdata = backed_sdata_blobs_multi_region
    _calculate(sdata, "blobs_labels_2")

    result = _calculate(sdata, "blobs_labels", overwrite_feature_key=True)

    stored = hp.tb.io.read_table_components(sdata.path, table_name=_TABLE, components=[("obsm", _KEY)], mode="eager")[
        ("obsm", _KEY)
    ]
    assert result.existing.kind == "compatible"
    assert not np.isnan(stored).any()


def test_backed_matrix_only_in_memory_is_replaced_entirely(backed_sdata_blobs_multi_region: SpatialData) -> None:
    sdata = backed_sdata_blobs_multi_region
    table = sdata[_TABLE]
    table.obsm[_KEY] = np.ones((table.n_obs, 1))

    result = _calculate(sdata, "blobs_labels", overwrite_feature_key=True)

    assert result.existing.kind == "incompatible"
    assert result.existing.reason == "it is not saved"
    assert np.isnan(_rows(sdata, "blobs_labels_2")).all()


def test_backed_stored_dataframe_is_replaced_entirely(backed_sdata_blobs_multi_region: SpatialData) -> None:
    sdata = backed_sdata_blobs_multi_region
    table = sdata[_TABLE]
    table_metadata = get_table_metadata(sdata, _TABLE)
    _calculate(sdata, "blobs_labels_2")
    # Keep harpy's metadata, so the check reaches the matrix format.
    hp.tb.io.add_table_components(
        sdata,
        table_name=_TABLE,
        components={("obsm", _KEY): pd.DataFrame({"area": np.ones(table.n_obs)}, index=table.obs_names)},
        obs_identity=table.obs[[table_metadata.region_key, table_metadata.instance_key]],
        overwrite=True,
    )

    result = _calculate(sdata, "blobs_labels", overwrite_feature_key=True)

    stored = hp.tb.io.read_table_components(sdata.path, table_name=_TABLE, components=[("obsm", _KEY)], mode="eager")[
        ("obsm", _KEY)
    ]
    selected = (table.obs[table_metadata.region_key] == "blobs_labels").to_numpy()
    assert result.existing.kind == "incompatible"
    assert result.existing.reason == "it is not a dense matrix"
    assert isinstance(stored, np.ndarray)
    assert not np.isnan(stored[selected]).any()
    assert np.isnan(stored[~selected]).all()


def test_backed_check_follows_the_store(backed_sdata_blobs_multi_region: SpatialData) -> None:
    sdata = backed_sdata_blobs_multi_region
    _calculate(sdata, "blobs_labels")
    # Changes outside the store are not seen: harpy merges with the stored entry.
    sdata[_TABLE].uns["feature_matrices"][_KEY] = {"source_kind": "custom_obsm"}

    assert _check(sdata, "blobs_labels").kind == "compatible"


def test_backed_stored_metadata_without_matrix_is_replaced(backed_sdata_blobs_multi_region: SpatialData) -> None:
    sdata = backed_sdata_blobs_multi_region
    _calculate(sdata, "blobs_labels")
    hp.tb.io.remove_table_components(sdata, table_name=_TABLE, components=[("obsm", _KEY)])

    existing = _check(sdata, "blobs_labels")
    result = _calculate(sdata, "blobs_labels")

    assert existing.kind == "absent"
    assert existing.orphaned_metadata is True
    assert result.change_kind == "created"
