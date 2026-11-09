from __future__ import annotations

from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from spatialdata import SpatialData

from spatiato.core.color_source_options import (
    get_shape_column_color_source_options,
    get_table_obs_color_source_options,
    get_table_var_layers,
)


def test_get_table_obs_color_source_options_classifies_supported_columns(sdata_blobs: SpatialData) -> None:
    table = sdata_blobs["table"]
    n_obs = table.n_obs
    repeated_values = ["a", "b"] * (n_obs // 2) + (["a"] if n_obs % 2 else [])

    table.obs["cat_obs"] = pd.Categorical(repeated_values)
    table.obs["bool_obs"] = pd.Series([index % 2 == 0 for index in range(n_obs)], index=table.obs.index)
    table.obs["binary_int_obs"] = pd.Series([index % 2 for index in range(n_obs)], index=table.obs.index, dtype="int64")
    table.obs["int_obs"] = pd.Series(np.arange(n_obs), index=table.obs.index, dtype="int64")
    table.obs["float_obs"] = pd.Series(np.linspace(0.0, 1.0, n_obs), index=table.obs.index, dtype="float64")
    table.obs["string_obs"] = pd.Series(repeated_values, index=table.obs.index, dtype="object")
    table.obs["string_id_obs"] = pd.Series(
        [f"cell-{index:04d}" for index in range(n_obs)],
        index=table.obs.index,
        dtype="object",
    )
    table.obs["cat_id_obs"] = pd.Categorical([f"cell-{index:04d}" for index in range(n_obs)])
    table.obs["datetime_obs"] = pd.date_range("2024-01-01", periods=n_obs)
    table.obs["object_obs"] = pd.Series(
        [{"index": index} for index in range(n_obs)], index=table.obs.index, dtype="object"
    )

    options = get_table_obs_color_source_options(sdata_blobs, "table")

    option_by_key = {option.value_key: option for option in options}

    assert "region" not in option_by_key
    assert option_by_key["instance_id"].value_kind == "instance"
    assert option_by_key["cat_obs"].value_kind == "categorical"
    assert option_by_key["bool_obs"].value_kind == "categorical"
    assert option_by_key["binary_int_obs"].value_kind == "categorical"
    assert option_by_key["int_obs"].value_kind == "continuous"
    assert option_by_key["float_obs"].value_kind == "continuous"
    assert option_by_key["string_obs"].value_kind == "categorical"
    assert option_by_key["string_id_obs"].value_kind == "categorical"
    assert option_by_key["cat_id_obs"].value_kind == "categorical"
    assert "datetime_obs" not in option_by_key
    assert "object_obs" not in option_by_key
    assert option_by_key["cat_obs"].source_kind == "obs_column"
    assert option_by_key["instance_id"].source_kind == "obs_column"
    assert option_by_key["cat_obs"].display_name == "cat_obs"
    assert option_by_key["instance_id"].display_name == "instance_id"
    assert option_by_key["cat_obs"].identity == ("table", "obs_column", "cat_obs")
    assert option_by_key["instance_id"].identity == ("table", "obs_column", "instance_id")


def test_get_table_var_layers_lists_x_first_then_the_layers_sorted(sdata_blobs: SpatialData) -> None:
    table = sdata_blobs["table"]
    table.layers["raw"] = table.X.copy()
    table.layers["counts"] = table.X.copy()

    assert get_table_var_layers(sdata_blobs, "table") == (None, "counts", "raw")


def test_get_table_var_layers_lists_only_the_layers_of_a_table_without_x(sdata_blobs: SpatialData) -> None:
    table = sdata_blobs["table"]
    table.layers["raw"] = table.X.copy()
    table.layers["counts"] = table.X.copy()
    table.X = None

    assert get_table_var_layers(sdata_blobs, "table") == ("counts", "raw")


def test_get_table_var_layers_is_empty_for_a_table_without_any_matrix(sdata_blobs: SpatialData) -> None:
    sdata_blobs["table"].X = None

    assert get_table_var_layers(sdata_blobs, "table") == ()


def test_get_shape_column_color_source_options_classifies_supported_columns() -> None:
    geodataframe = gpd.GeoDataFrame(
        {
            "cat_shape": pd.Categorical(["a", "b", "a"]),
            "bool_shape": pd.Series([True, False, True], dtype="bool"),
            "binary_int_shape": pd.Series([0, 1, 0], dtype="int64"),
            "int_shape": pd.Series([1, 2, 3], dtype="int64"),
            "float_shape": pd.Series([0.1, 0.2, 0.3], dtype="float64"),
            "string_shape": pd.Series(["alpha", "beta", "alpha"], dtype="object"),
            "radius": pd.Series([5.0, 6.0, 7.0], dtype="float64"),
            "cat_shape_colors": ["red", "blue", "red"],
            "manual_color": ["red", "blue", "red"],
            "style.color": ["red", "blue", "red"],
            "all_missing": pd.Series([None, None, None], dtype="object"),
            "mixed_shape": pd.Series(["alpha", 1, "beta"], dtype="object"),
            "object_shape": pd.Series([{"a": 1}, {"b": 2}, {"c": 3}], dtype="object"),
        },
        geometry=[
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(2, 0), (3, 0), (3, 1), (2, 1)]),
            Polygon([(4, 0), (5, 0), (5, 1), (4, 1)]),
        ],
    )
    fake_sdata = SimpleNamespace(shapes={"cell_boundaries": geodataframe})

    options = get_shape_column_color_source_options(fake_sdata, "cell_boundaries")

    option_by_key = {option.value_key: option for option in options}

    assert "geometry" not in option_by_key
    assert "cat_shape_colors" not in option_by_key
    assert "manual_color" not in option_by_key
    assert "style.color" not in option_by_key
    assert "all_missing" not in option_by_key
    assert "mixed_shape" not in option_by_key
    assert "object_shape" not in option_by_key
    assert option_by_key["cat_shape"].value_kind == "categorical"
    assert option_by_key["bool_shape"].value_kind == "categorical"
    assert option_by_key["binary_int_shape"].value_kind == "categorical"
    assert option_by_key["int_shape"].value_kind == "continuous"
    assert option_by_key["float_shape"].value_kind == "continuous"
    assert option_by_key["string_shape"].value_kind == "categorical"
    assert option_by_key["radius"].value_kind == "continuous"
    assert option_by_key["cat_shape"].source_kind == "shape_column"
    assert option_by_key["cat_shape"].display_name == "cat_shape"
    assert option_by_key["cat_shape"].identity == ("shape_column", "cat_shape")
