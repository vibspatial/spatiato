from __future__ import annotations

from types import SimpleNamespace

import anndata as ad
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Polygon
from spatialdata import SpatialData
from spatialdata.models import TableModel
from xarray import DataArray

import spatiato.core.spatialdata as spatialdata_module
from spatiato.core._color_source import ShapeColumnColorSourceSpec
from spatiato.core.spatialdata import (
    SpatialDataTableMetadata,
    get_annotating_table_names,
    get_coordinate_system_names_from_sdata,
    get_image_channel_names_from_sdata,
    get_spatialdata_feature_extraction_image_discovery_for_coordinate_system_and_label_from_sdata,
    get_spatialdata_feature_extraction_label_discovery_for_coordinate_system_from_sdata,
    get_spatialdata_image_options_for_coordinate_system_from_sdata,
    get_spatialdata_labels_options_for_coordinate_system_from_sdata,
    get_spatialdata_labels_options_from_sdata,
    get_spatialdata_points_options_for_coordinate_system_from_sdata,
    get_spatialdata_points_options_from_sdata,
    get_spatialdata_shapes_options_for_coordinate_system_from_sdata,
    get_spatialdata_shapes_options_from_sdata,
    get_table,
    get_table_annotated_labels_names,
    get_table_metadata,
    get_table_obsm_keys,
    get_table_var_matrix,
    normalize_table_metadata,
    validate_table_annotation_coverage,
    validate_table_binding,
    validate_table_region_instance_ids,
)


class DummySpatialData:
    def __init__(self, *, labels=None, images=None, shapes=None, points=None, tables=None) -> None:
        self.labels = {} if labels is None else labels
        self.images = {} if images is None else images
        self.shapes = {} if shapes is None else shapes
        self.points = {} if points is None else points
        self._tables = {} if tables is None else tables

    def __getitem__(self, key: str):
        return self._tables[key]

    @property
    def tables(self):
        return self._tables


class FakeTransform:
    def __init__(self, matrix: np.ndarray | list[list[float]]) -> None:
        self._matrix = np.asarray(matrix, dtype=float)

    def to_affine_matrix(self, *, input_axes, output_axes):
        assert input_axes == output_axes
        return self._matrix


def test_get_annotating_table_names_returns_tables_for_annotated_label(sdata_blobs: SpatialData) -> None:
    table_names = get_annotating_table_names(sdata_blobs, "blobs_labels")

    assert table_names == ["table"]


def test_get_annotating_table_names_returns_empty_list_for_unannotated_label(sdata_blobs: SpatialData) -> None:
    table_names = get_annotating_table_names(sdata_blobs, "blobs_multiscale_labels")

    assert table_names == []


def test_get_annotating_table_names_is_element_generic_for_shapes() -> None:
    shapes = gpd.GeoDataFrame(
        {"cell_id": ["cell_1"]},
        geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
    )
    shape_table = TableModel.parse(
        ad.AnnData(
            obs=pd.DataFrame(
                {
                    "region": ["cell_boundaries"],
                    "instance_id": ["cell_1"],
                }
            )
        ),
        region="cell_boundaries",
        region_key="region",
        instance_key="instance_id",
    )
    labels_table = TableModel.parse(
        ad.AnnData(
            obs=pd.DataFrame(
                {
                    "region": ["nuclei"],
                    "instance_id": [1],
                }
            )
        ),
        region="nuclei",
        region_key="region",
        instance_key="instance_id",
    )
    fake_sdata = DummySpatialData(
        shapes={"cell_boundaries": shapes},
        labels={"nuclei": object()},
        tables={"shape_table": shape_table, "labels_table": labels_table},
    )

    assert get_annotating_table_names(fake_sdata, "cell_boundaries") == ["shape_table"]


def test_sdata_blobs_fixture_adds_dummy_feature_matrices(sdata_blobs: SpatialData) -> None:
    table = sdata_blobs["table"]

    assert "features_1" in table.obsm
    assert "features_2" in table.obsm
    assert table.obsm["features_1"].shape == (table.n_obs, 4)
    assert table.obsm["features_2"].shape == (table.n_obs, 2)


def test_get_table_obsm_keys_returns_sorted_feature_matrix_keys(sdata_blobs: SpatialData) -> None:
    obsm_keys = get_table_obsm_keys(sdata_blobs, "table")

    assert obsm_keys == ["features_1", "features_2"]


def test_get_table_obsm_keys_leaves_out_the_keys_reserved_for_spatial_coordinates(sdata_blobs: SpatialData) -> None:
    table = sdata_blobs["table"]
    table.obsm["spatial"] = np.zeros((table.n_obs, 2))
    table.obsm["spatial_canonical"] = np.zeros((table.n_obs, 3))

    assert get_table_obsm_keys(sdata_blobs, "table") == ["features_1", "features_2"]


def test_get_table_var_matrix_returns_x_or_the_layer_as_attached(sdata_blobs: SpatialData) -> None:
    table = sdata_blobs["table"]
    counts = table.X.copy()
    table.layers["counts"] = counts

    assert get_table_var_matrix(table, None) is table.X
    assert get_table_var_matrix(table, "counts") is table.layers["counts"]
    with pytest.raises(ValueError, match='Layer "raw" is not available in the selected table'):
        get_table_var_matrix(table, "raw")

    table.X = None
    with pytest.raises(ValueError, match="The selected table has no `X`"):
        get_table_var_matrix(table, None)


def test_shape_color_source_spec_identity_and_display_name() -> None:
    source = ShapeColumnColorSourceSpec(
        source_kind="shape_column",
        value_key="cell_type",
        value_kind="categorical",
    )

    assert source.identity == ("shape_column", "cell_type")
    assert source.display_name == "cell_type"


def test_get_table_metadata_resolves_table_metadata(sdata_blobs: SpatialData) -> None:
    metadata = get_table_metadata(sdata_blobs, "table")

    assert metadata.table_name == "table"
    assert metadata.region_key == "region"
    assert metadata.instance_key == "instance_id"
    assert metadata.regions == ("blobs_labels",)
    assert metadata.annotates("blobs_labels")
    assert not metadata.annotates("blobs_multiscale_labels")


def test_get_table_metadata_returns_table_linkage(sdata_blobs: SpatialData) -> None:
    metadata = get_table_metadata(sdata_blobs, "table")

    assert metadata.region_key == "region"
    assert metadata.instance_key == "instance_id"


@pytest.mark.parametrize("regions", [(), ("",), ("cells", "cells")])
def test_spatialdata_table_metadata_rejects_invalid_regions(regions: tuple[str, ...]) -> None:
    with pytest.raises(ValueError, match="region"):
        SpatialDataTableMetadata(
            table_name="table",
            region_key="region",
            instance_key="instance_id",
            regions=regions,
        )


def test_get_table_metadata_normalizes_returned_values_without_mutating_attrs(sdata_blobs: SpatialData) -> None:
    table = sdata_blobs["table"]
    attrs = table.uns[TableModel.ATTRS_KEY]
    attrs[TableModel.REGION_KEY_KEY] = np.array(["region"])
    attrs[TableModel.INSTANCE_KEY] = np.array(["instance_id"])
    attrs[TableModel.REGION_KEY] = np.array(["blobs_labels"])
    original_values = dict(attrs)

    metadata = get_table_metadata(sdata_blobs, "table")

    assert metadata.region_key == "region"
    assert metadata.instance_key == "instance_id"
    assert metadata.regions == ("blobs_labels",)
    assert all(attrs[key] is value for key, value in original_values.items())


def test_validate_table_binding_rejects_duplicate_instance_ids_within_selected_region(sdata_blobs: SpatialData) -> None:
    table = sdata_blobs["table"]
    first_index, second_index = table.obs.index[:2]
    table.obs.loc[second_index, "instance_id"] = table.obs.loc[first_index, "instance_id"]

    with pytest.raises(ValueError, match="contains duplicate values within that region"):
        validate_table_binding(sdata_blobs, "blobs_labels", "table")


def test_get_table_annotated_labels_names_returns_sorted_regions_for_multi_region_table(
    sdata_blobs: SpatialData,
) -> None:
    table = sdata_blobs["table"].copy()
    table.obs["region"] = table.obs["region"].cat.add_categories(["blobs_multiscale_labels"])
    table.obs.loc[table.obs.index[0], "region"] = "blobs_multiscale_labels"
    table.uns[TableModel.ATTRS_KEY] = {
        **table.uns[TableModel.ATTRS_KEY],
        TableModel.REGION_KEY: ["blobs_multiscale_labels", "blobs_labels"],
    }
    fake_sdata = DummySpatialData(
        labels={
            "blobs_labels": sdata_blobs.labels["blobs_labels"],
            "blobs_multiscale_labels": sdata_blobs.labels["blobs_multiscale_labels"],
        },
        tables={"table": table},
    )

    assert get_table_annotated_labels_names(fake_sdata, "table") == ["blobs_labels", "blobs_multiscale_labels"]


def test_get_table_annotated_labels_names_rejects_invalid_regions(sdata_blobs: SpatialData) -> None:
    table = sdata_blobs["table"].copy()
    table.obs["region"] = table.obs["region"].cat.add_categories(["missing_labels"])
    table.obs.loc[table.obs.index[0], "region"] = "missing_labels"
    table.uns[TableModel.ATTRS_KEY] = {
        **table.uns[TableModel.ATTRS_KEY],
        TableModel.REGION_KEY: ["blobs_labels", "missing_labels"],
    }
    fake_sdata = DummySpatialData(
        labels={"blobs_labels": sdata_blobs.labels["blobs_labels"]},
        tables={"table": table},
    )

    with pytest.raises(ValueError, match="missing_labels"):
        get_table_annotated_labels_names(fake_sdata, "table")


def test_validate_table_annotation_coverage_rejects_unannotated_regions(sdata_blobs: SpatialData) -> None:
    table = sdata_blobs["table"].copy()
    fake_sdata = DummySpatialData(
        labels={
            "blobs_labels": sdata_blobs.labels["blobs_labels"],
            "blobs_multiscale_labels": sdata_blobs.labels["blobs_multiscale_labels"],
        },
        tables={"table": table},
    )

    with pytest.raises(ValueError, match="does not annotate labels element\\(s\\) `blobs_multiscale_labels`"):
        validate_table_annotation_coverage(fake_sdata, "table", ["blobs_multiscale_labels"])


def test_validate_table_region_instance_ids_allows_duplicate_instance_ids_across_regions(
    sdata_blobs: SpatialData,
) -> None:
    table = sdata_blobs["table"].copy()
    table.obs["region"] = table.obs["region"].cat.add_categories(["blobs_multiscale_labels"])
    table.uns[TableModel.ATTRS_KEY] = {
        **table.uns[TableModel.ATTRS_KEY],
        TableModel.REGION_KEY: ["blobs_labels", "blobs_multiscale_labels"],
    }
    first_index, second_index = table.obs.index[:2]
    table.obs.loc[second_index, "region"] = "blobs_multiscale_labels"
    table.obs.loc[second_index, "instance_id"] = table.obs.loc[first_index, "instance_id"]
    fake_sdata = DummySpatialData(
        labels={
            "blobs_labels": sdata_blobs.labels["blobs_labels"],
            "blobs_multiscale_labels": sdata_blobs.labels["blobs_multiscale_labels"],
        },
        tables={"table": table},
    )

    metadata = validate_table_region_instance_ids(fake_sdata, "table")

    assert metadata.regions == ("blobs_labels", "blobs_multiscale_labels")


def test_validate_table_region_instance_ids_rejects_duplicates_with_region_specific_message(
    sdata_blobs: SpatialData,
) -> None:
    table = sdata_blobs["table"].copy()
    table.obs["region"] = table.obs["region"].cat.add_categories(["blobs_multiscale_labels"])
    table.uns[TableModel.ATTRS_KEY] = {
        **table.uns[TableModel.ATTRS_KEY],
        TableModel.REGION_KEY: ["blobs_labels", "blobs_multiscale_labels"],
    }
    first_index, second_index = table.obs.index[:2]
    table.obs.loc[[first_index, second_index], "region"] = "blobs_multiscale_labels"
    table.obs.loc[second_index, "instance_id"] = table.obs.loc[first_index, "instance_id"]
    fake_sdata = DummySpatialData(
        labels={
            "blobs_labels": sdata_blobs.labels["blobs_labels"],
            "blobs_multiscale_labels": sdata_blobs.labels["blobs_multiscale_labels"],
        },
        tables={"table": table},
    )

    with pytest.raises(ValueError, match="labels element `blobs_multiscale_labels`"):
        validate_table_region_instance_ids(fake_sdata, "table")


def test_normalize_table_metadata_normalizes_numpy_array_region_attrs_in_place(sdata_blobs: SpatialData) -> None:
    table = sdata_blobs["table"]
    table.uns[TableModel.ATTRS_KEY] = {
        **table.uns[TableModel.ATTRS_KEY],
        TableModel.REGION_KEY: np.array(["blobs_labels"]),
    }

    normalize_table_metadata(table)
    validated = get_table(sdata_blobs, "table")

    assert validated is table
    assert table.uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY] == ["blobs_labels"]


def test_get_spatialdata_labels_options_from_sdata_returns_all_labels(sdata_blobs: SpatialData) -> None:
    options = get_spatialdata_labels_options_from_sdata(sdata_blobs)

    assert [option.labels_name for option in options] == [
        "blobs_labels",
        "blobs_multiscale_labels",
    ]
    assert [option.display_name for option in options] == [
        "blobs_labels",
        "blobs_multiscale_labels",
    ]
    assert [option.coordinate_systems for option in options] == [
        ("global",),
        ("global",),
    ]
    assert all(option.sdata is sdata_blobs for option in options)


def test_get_spatialdata_shapes_options_from_sdata_returns_all_shapes(sdata_blobs: SpatialData) -> None:
    options = get_spatialdata_shapes_options_from_sdata(sdata_blobs)

    assert [option.shapes_name for option in options] == [
        "blobs_circles",
        "blobs_multipolygons",
        "blobs_polygons",
    ]
    assert [option.display_name for option in options] == [
        "blobs_circles",
        "blobs_multipolygons",
        "blobs_polygons",
    ]
    assert [option.coordinate_systems for option in options] == [
        ("global",),
        ("global",),
        ("global",),
    ]
    assert all(option.sdata is sdata_blobs for option in options)


def test_get_spatialdata_points_options_from_sdata_returns_all_points(monkeypatch) -> None:
    points_a = object()
    points_b = object()
    fake_sdata = DummySpatialData(points={"points_a": points_a, "points_b": points_b})
    transformation_by_id = {
        id(points_a): {"global": object()},
        id(points_b): {"local": object(), "global": object()},
    }

    def _fake_get_transformation(element, get_all: bool = False):
        del get_all
        return transformation_by_id[id(element)]

    monkeypatch.setattr(spatialdata_module, "get_transformation", _fake_get_transformation)

    options = get_spatialdata_points_options_from_sdata(fake_sdata)

    assert [option.points_name for option in options] == ["points_a", "points_b"]
    assert [option.display_name for option in options] == ["points_a", "points_b"]
    assert [option.coordinate_systems for option in options] == [("global",), ("global", "local")]
    assert all(option.sdata is fake_sdata for option in options)


def test_get_coordinate_system_names_from_sdata_returns_sorted_union(
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    shapes_names = sorted(sdata_blobs.shapes.keys())
    transformation_by_id = {
        id(sdata_blobs.labels["blobs_labels"]): {"global": object(), "aligned": object()},
        id(sdata_blobs.labels["blobs_multiscale_labels"]): {"global": object()},
        id(sdata_blobs.images["blobs_image"]): {"global": object()},
        id(sdata_blobs.images["blobs_multiscale_image"]): {"local": object()},
        id(sdata_blobs.shapes[shapes_names[0]]): {"shape_space": object()},
        id(sdata_blobs.shapes[shapes_names[1]]): {"global": object()},
        id(sdata_blobs.shapes[shapes_names[2]]): {"global": object()},
        id(sdata_blobs.points["blobs_points"]): {"global": object()},
    }

    def _fake_get_transformation(element, get_all: bool = False):
        del get_all
        return transformation_by_id[id(element)]

    monkeypatch.setattr(spatialdata_module, "get_transformation", _fake_get_transformation)

    assert get_coordinate_system_names_from_sdata(sdata_blobs) == ["aligned", "global", "local", "shape_space"]


def test_get_coordinate_system_names_from_sdata_includes_shapes_only_data(monkeypatch) -> None:
    shape_element = object()
    fake_sdata = DummySpatialData(shapes={"cell_boundaries": shape_element})

    def _fake_get_transformation(element, get_all: bool = False):
        del get_all
        assert element is shape_element
        return {"shape_space": object()}

    monkeypatch.setattr(spatialdata_module, "get_transformation", _fake_get_transformation)

    assert get_coordinate_system_names_from_sdata(fake_sdata) == ["shape_space"]


def test_get_coordinate_system_names_from_sdata_includes_points_only_data(monkeypatch) -> None:
    points_element = object()
    fake_sdata = DummySpatialData(points={"transcripts": points_element})

    def _fake_get_transformation(element, get_all: bool = False):
        del get_all
        assert element is points_element
        return {"points_space": object()}

    monkeypatch.setattr(spatialdata_module, "get_transformation", _fake_get_transformation)

    assert get_coordinate_system_names_from_sdata(fake_sdata) == ["points_space"]


def test_get_image_channel_names_from_sdata_returns_channel_axis_names(sdata_blobs: SpatialData) -> None:
    assert get_image_channel_names_from_sdata(sdata_blobs, "blobs_image") == ["0", "1", "2"]


def test_get_image_channel_names_from_sdata_rejects_duplicate_channel_names(monkeypatch) -> None:
    fake_sdata = SimpleNamespace(
        images={
            "image_with_duplicates": DataArray(
                np.zeros((3, 2, 2)),
                dims=("c", "y", "x"),
                coords={"c": ["dup", "dup", "other"]},
            )
        }
    )
    monkeypatch.setattr(spatialdata_module, "_get_image_names", lambda sdata: ["image_with_duplicates"])

    with pytest.raises(ValueError, match="duplicate channel names"):
        get_image_channel_names_from_sdata(fake_sdata, "image_with_duplicates")


def test_get_spatialdata_labels_options_for_coordinate_system_from_sdata_filters_labels(
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    transformation_by_id = {
        id(sdata_blobs.labels["blobs_labels"]): {"global": object(), "aligned": object()},
        id(sdata_blobs.labels["blobs_multiscale_labels"]): {"global": object()},
        id(sdata_blobs.images["blobs_image"]): {"global": object()},
        id(sdata_blobs.images["blobs_multiscale_image"]): {"global": object()},
    }

    def _fake_get_transformation(element, get_all: bool = False):
        del get_all
        return transformation_by_id[id(element)]

    monkeypatch.setattr(spatialdata_module, "get_transformation", _fake_get_transformation)

    options = get_spatialdata_labels_options_for_coordinate_system_from_sdata(
        sdata=sdata_blobs,
        coordinate_system="aligned",
    )

    assert [option.labels_name for option in options] == ["blobs_labels"]
    assert options[0].coordinate_systems == ("aligned", "global")


def test_get_spatialdata_shapes_options_for_coordinate_system_from_sdata_filters_shapes(monkeypatch) -> None:
    global_shape = object()
    local_shape = object()
    fake_sdata = DummySpatialData(shapes={"global_shape": global_shape, "local_shape": local_shape})
    transformation_by_id = {
        id(global_shape): {"global": object(), "aligned": object()},
        id(local_shape): {"local": object()},
    }

    def _fake_get_transformation(element, get_all: bool = False):
        del get_all
        return transformation_by_id[id(element)]

    monkeypatch.setattr(spatialdata_module, "get_transformation", _fake_get_transformation)

    options = get_spatialdata_shapes_options_for_coordinate_system_from_sdata(
        sdata=fake_sdata,
        coordinate_system="aligned",
    )

    assert [option.shapes_name for option in options] == ["global_shape"]
    assert options[0].coordinate_systems == ("aligned", "global")


def test_get_spatialdata_points_options_for_coordinate_system_from_sdata_filters_points(monkeypatch) -> None:
    global_points = object()
    local_points = object()
    fake_sdata = DummySpatialData(points={"global_points": global_points, "local_points": local_points})
    transformation_by_id = {
        id(global_points): {"global": object(), "aligned": object()},
        id(local_points): {"local": object()},
    }

    def _fake_get_transformation(element, get_all: bool = False):
        del get_all
        return transformation_by_id[id(element)]

    monkeypatch.setattr(spatialdata_module, "get_transformation", _fake_get_transformation)

    options = get_spatialdata_points_options_for_coordinate_system_from_sdata(
        sdata=fake_sdata,
        coordinate_system="aligned",
    )

    assert [option.points_name for option in options] == ["global_points"]
    assert options[0].coordinate_systems == ("aligned", "global")


def test_get_spatialdata_image_options_for_coordinate_system_from_sdata_filters_images(
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    transformation_by_id = {
        id(sdata_blobs.labels["blobs_labels"]): {"global": object()},
        id(sdata_blobs.labels["blobs_multiscale_labels"]): {"global": object()},
        id(sdata_blobs.images["blobs_image"]): {"global": object(), "aligned": object()},
        id(sdata_blobs.images["blobs_multiscale_image"]): {"local": object()},
    }

    def _fake_get_transformation(element, get_all: bool = False):
        del get_all
        return transformation_by_id[id(element)]

    monkeypatch.setattr(spatialdata_module, "get_transformation", _fake_get_transformation)

    options = get_spatialdata_image_options_for_coordinate_system_from_sdata(
        sdata=sdata_blobs,
        coordinate_system="aligned",
    )

    assert [option.image_name for option in options] == ["blobs_image"]
    assert options[0].coordinate_systems == ("aligned", "global")


def test_get_spatialdata_feature_extraction_label_options_for_coordinate_system_filters_to_translation_compatible_labels(
    monkeypatch,
) -> None:
    eligible_alpha = DataArray(np.zeros((4, 5), dtype=np.int32), dims=("y", "x"))
    eligible_beta = DataArray(np.zeros((4, 5), dtype=np.int32), dims=("y", "x"))
    rotated = DataArray(np.zeros((4, 5), dtype=np.int32), dims=("y", "x"))
    scaled = DataArray(np.zeros((4, 5), dtype=np.int32), dims=("y", "x"))
    fake_sdata = DummySpatialData(
        labels={
            "scaled": scaled,
            "eligible_beta": eligible_beta,
            "rotated": rotated,
            "eligible_alpha": eligible_alpha,
        }
    )
    transformation_by_id = {
        id(eligible_alpha): {"global": FakeTransform([[1, 0, 3], [0, 1, -2], [0, 0, 1]])},
        id(eligible_beta): {"global": FakeTransform([[1, 0, 0], [0, 1, 0], [0, 0, 1]])},
        id(rotated): {"global": FakeTransform([[0, -1, 0], [1, 0, 0], [0, 0, 1]])},
        id(scaled): {"global": FakeTransform([[2, 0, 0], [0, 1, 0], [0, 0, 1]])},
    }

    def _fake_get_transformation(element, get_all: bool = False):
        del get_all
        return transformation_by_id[id(element)]

    monkeypatch.setattr(spatialdata_module, "get_transformation", _fake_get_transformation)

    discovery = get_spatialdata_feature_extraction_label_discovery_for_coordinate_system_from_sdata(
        sdata=fake_sdata,
        coordinate_system="global",
    )

    assert discovery.coordinate_system == "global"
    assert [option.labels_name for option in discovery.eligible_label_options] == ["eligible_alpha", "eligible_beta"]
    assert discovery.coordinate_system_labels_count == 4
    assert discovery.unavailable_label_count == 2


def test_get_spatialdata_matching_image_options_for_coordinate_system_and_label_filters_by_shape_and_transform(
    monkeypatch,
) -> None:
    segmentation = DataArray(np.zeros((4, 5), dtype=np.int32), dims=("y", "x"))
    matching_a = DataArray(np.zeros((4, 5), dtype=np.float32), dims=("y", "x"))
    matching_b = DataArray(np.zeros((3, 4, 5), dtype=np.float32), dims=("c", "y", "x"))
    wrong_shape = DataArray(np.zeros((3, 4, 6), dtype=np.float32), dims=("c", "y", "x"))
    wrong_transform = DataArray(np.zeros((4, 5), dtype=np.float32), dims=("y", "x"))
    scaled = DataArray(np.zeros((4, 5), dtype=np.float32), dims=("y", "x"))
    fake_sdata = DummySpatialData(
        labels={"segmentation": segmentation},
        images={
            "scaled": scaled,
            "matching_b": matching_b,
            "wrong_shape": wrong_shape,
            "matching_a": matching_a,
            "wrong_transform": wrong_transform,
        },
    )
    transformation_by_id = {
        id(segmentation): {"global": FakeTransform([[1, 0, 7], [0, 1, -4], [0, 0, 1]])},
        id(matching_a): {"global": FakeTransform([[1, 0, 7], [0, 1, -4], [0, 0, 1]])},
        id(matching_b): {"global": FakeTransform([[1, 0, 7], [0, 1, -4], [0, 0, 1]])},
        id(wrong_shape): {"global": FakeTransform([[1, 0, 7], [0, 1, -4], [0, 0, 1]])},
        id(wrong_transform): {"global": FakeTransform([[1, 0, 1], [0, 1, 2], [0, 0, 1]])},
        id(scaled): {"global": FakeTransform([[2, 0, 0], [0, 1, 0], [0, 0, 1]])},
    }

    def _fake_get_transformation(element, get_all: bool = False):
        del get_all
        return transformation_by_id[id(element)]

    monkeypatch.setattr(spatialdata_module, "get_transformation", _fake_get_transformation)

    discovery = get_spatialdata_feature_extraction_image_discovery_for_coordinate_system_and_label_from_sdata(
        sdata=fake_sdata,
        coordinate_system="global",
        labels_name="segmentation",
    )

    assert discovery.coordinate_system == "global"
    assert discovery.labels_name == "segmentation"
    assert [option.image_name for option in discovery.eligible_image_options] == ["matching_a", "matching_b"]
    assert discovery.coordinate_system_image_count == 5
    assert discovery.unavailable_image_count == 3
    assert [option.image_name for option in discovery.eligible_image_options] == ["matching_a", "matching_b"]
    assert all(option.coordinate_systems == ("global",) for option in discovery.eligible_image_options)
