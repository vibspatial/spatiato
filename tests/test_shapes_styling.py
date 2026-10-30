from __future__ import annotations

import anndata as ad
import geopandas as gpd
import harpy as hp
import numpy as np
import pandas as pd
import pytest
from _table_matrix_fixtures import FORMATS_AND_MODES, MATRIX_TYPE_BY_FORMAT_AND_MODE, matrix_in_format
from matplotlib.colors import to_rgba
from napari.layers import Points, Shapes
from napari.utils.colormaps import label_colormap
from shapely.geometry import Point, Polygon
from spatialdata import SpatialData
from spatialdata.models import TableModel

from spatiato.core._color_source import ShapeColumnColorSourceSpec, TableColorSourceSpec
from spatiato.core.spatialdata import SpatialDataTableMetadata
from spatiato.viewer._styling import continuous_colors_for_values
from spatiato.viewer.shapes_styling import (
    SHAPES_EDGE_ALPHA,
    SHAPES_FACE_ALPHA,
    SHAPES_MISSING_BASE_COLOR,
    _align_table_color_source_to_shapes_rows,
    _apply_rendered_row_colors_to_shapes_layer,
    apply_shape_column_color_source_to_shapes_layer,
    apply_table_color_source_to_shapes_layer,
    build_styled_shapes_layer_name,
    disambiguate_shape_style_feature_name,
)


def _polygon(offset: float = 0.0) -> Polygon:
    return Polygon([(offset, 0), (offset + 1, 0), (offset + 1, 1), (offset, 1)])


def _shape_vertices(offset: float) -> np.ndarray:
    return np.asarray([(0, offset), (0, offset + 1), (1, offset + 1), (1, offset)], dtype=float)


def _make_shapes_layer(
    source_indices: tuple[object, ...],
    *,
    source_shapes_index_feature_name: str = "index",
) -> Shapes:
    return Shapes(
        [_shape_vertices(float(index)) for index in range(len(source_indices))],
        shape_type=["polygon"] * len(source_indices),
        features=pd.DataFrame({source_shapes_index_feature_name: list(source_indices)}),
    )


def _make_points_layer(
    source_indices: tuple[object, ...],
    *,
    source_shapes_index_feature_name: str = "index",
) -> Points:
    return Points(
        np.asarray([[float(index), float(index + 10)] for index in range(len(source_indices))], dtype=float),
        features=pd.DataFrame({source_shapes_index_feature_name: list(source_indices)}),
        size=np.asarray([4.0 + index for index in range(len(source_indices))], dtype=float),
    )


def _table_metadata(regions: tuple[str, ...] = ("cells",)) -> SpatialDataTableMetadata:
    return SpatialDataTableMetadata(
        table_name="table",
        region_key="region",
        instance_key="instance_id",
        regions=regions,
    )


class _DummySpatialData:
    def __init__(self, *, shapes: dict[str, gpd.GeoDataFrame], tables: dict[str, ad.AnnData]) -> None:
        self.shapes = shapes
        self.tables = tables

    def __getitem__(self, key: str) -> ad.AnnData:
        return self.tables[key]


def _make_shapes_sdata(shapes_element: gpd.GeoDataFrame, table: ad.AnnData) -> _DummySpatialData:
    parsed_table = TableModel.parse(
        table,
        region="cells",
        region_key="region",
        instance_key="instance_id",
    )
    return _DummySpatialData(shapes={"cells": shapes_element}, tables={"table": parsed_table})


def _rgba(color: object, alpha: float) -> tuple[float, float, float, float]:
    red, green, blue, _alpha = to_rgba(color)
    return (red, green, blue, alpha)


def test_align_table_color_source_to_shapes_rows_allows_partial_coverage_and_duplicate_shape_instances() -> None:
    geodataframe = gpd.GeoDataFrame(
        {},
        geometry=[_polygon(0), _polygon(2), _polygon(4), _polygon(6)],
        index=pd.Index(["cell_1", "cell_1", "cell_2", "cell_3"], name="instance_id"),
    )
    table = ad.AnnData(
        obs=pd.DataFrame(
            {
                "region": ["cells", "cells"],
                "instance_id": ["cell_1", "cell_2"],
                "cell_type": ["T", "B"],
            },
            index=["obs_1", "obs_2"],
        )
    )
    style_spec = TableColorSourceSpec(
        table_name="table",
        source_kind="obs_column",
        value_key="cell_type",
        value_kind="categorical",
    )

    aligned = _align_table_color_source_to_shapes_rows(
        table=table,
        table_metadata=_table_metadata(),
        shapes_name="cells",
        shapes_element=geodataframe,
        style_spec=style_spec,
        source_row_id_by_rendered_row=(0, 1, 1, 2, 3),
    )

    assert aligned.source_row_values.index.to_list() == [0, 1, 2, 3]
    assert aligned.source_row_values.iloc[:3].to_list() == ["T", "T", "B"]
    assert pd.isna(aligned.source_row_values.iloc[3])
    assert aligned.source_row_has_table_row.tolist() == [True, True, True, False]
    assert aligned.rendered_row_values.index.to_list() == [0, 1, 2, 3, 4]
    assert aligned.rendered_row_values.iloc[:4].to_list() == ["T", "T", "T", "B"]
    assert pd.isna(aligned.rendered_row_values.iloc[4])
    assert aligned.rendered_row_has_table_row.tolist() == [True, True, True, True, False]


@pytest.mark.parametrize(("x_format", "mode"), FORMATS_AND_MODES)
def test_align_table_color_source_to_shapes_rows_extracts_x_var_values_from_region_rows(
    x_format: str,
    mode: str,
    tmp_path,
) -> None:
    geodataframe = gpd.GeoDataFrame(
        {},
        geometry=[_polygon(0), _polygon(2)],
        index=pd.Index(["cell_1", "cell_2"], name="instance_id"),
    )
    table = ad.AnnData(
        X=matrix_in_format(np.asarray([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]), x_format),
        obs=pd.DataFrame(
            {
                "region": ["cells", "other", "cells"],
                "instance_id": ["cell_1", "other_1", "cell_2"],
            },
            index=["obs_1", "obs_2", "obs_3"],
        ),
        var=pd.DataFrame(index=["GeneA", "GeneB"]),
    )
    path = tmp_path / "table.zarr"
    parsed_table = TableModel.parse(table, region=["cells", "other"], region_key="region", instance_key="instance_id")
    SpatialData(tables={"table": parsed_table}).write(path)
    # Small sparse blocks put the two "cells" rows of a lazy CSR `X` in different blocks.
    stored_table = hp.tb.io.read_table(path, table_name="table", mode=mode, sparse_chunks=2)
    # Test setup: harpy's reader attaches `X` in the representation under test.
    assert isinstance(stored_table.X, MATRIX_TYPE_BY_FORMAT_AND_MODE[x_format, mode])
    style_spec = TableColorSourceSpec(
        table_name="table",
        source_kind="x_var",
        value_key="GeneB",
        value_kind="continuous",
    )

    aligned = _align_table_color_source_to_shapes_rows(
        table=stored_table,
        table_metadata=_table_metadata(regions=("cells", "other")),
        shapes_name="cells",
        shapes_element=geodataframe,
        style_spec=style_spec,
        source_row_id_by_rendered_row=(1, 0),
    )

    assert aligned.source_row_values.to_list() == [10.0, 30.0]
    assert aligned.rendered_row_values.to_list() == [30.0, 10.0]
    assert aligned.rendered_row_has_table_row.tolist() == [True, True]


def test_align_table_color_source_to_shapes_rows_preserves_instance_key_values_for_identity_coloring() -> None:
    geodataframe = gpd.GeoDataFrame(
        {},
        geometry=[_polygon(0), _polygon(2)],
        index=pd.Index(["cell_1", "cell_2"], name="instance_id"),
    )
    table = ad.AnnData(
        obs=pd.DataFrame(
            {
                "region": ["cells"],
                "instance_id": ["cell_1"],
            },
            index=["obs_1"],
        )
    )
    style_spec = TableColorSourceSpec(
        table_name="table",
        source_kind="obs_column",
        value_key="instance_id",
        value_kind="instance",
    )

    aligned = _align_table_color_source_to_shapes_rows(
        table=table,
        table_metadata=_table_metadata(),
        shapes_name="cells",
        shapes_element=geodataframe,
        style_spec=style_spec,
        source_row_id_by_rendered_row=(0, 1),
    )

    assert aligned.source_row_values.iloc[0] == "cell_1"
    assert pd.isna(aligned.source_row_values.iloc[1])
    assert aligned.source_row_has_table_row.tolist() == [True, False]


def test_align_table_color_source_to_shapes_rows_requires_table_to_annotate_shapes_element() -> None:
    geodataframe = gpd.GeoDataFrame(
        {},
        geometry=[_polygon(0)],
        index=pd.Index(["cell_1"], name="instance_id"),
    )
    table = ad.AnnData(obs=pd.DataFrame({"region": ["cells"], "instance_id": ["cell_1"]}, index=["obs_1"]))
    style_spec = TableColorSourceSpec(
        table_name="table",
        source_kind="obs_column",
        value_key="instance_id",
        value_kind="instance",
    )

    with pytest.raises(ValueError, match="does not annotate shapes element"):
        _align_table_color_source_to_shapes_rows(
            table=table,
            table_metadata=_table_metadata(regions=("other",)),
            shapes_name="cells",
            shapes_element=geodataframe,
            style_spec=style_spec,
            source_row_id_by_rendered_row=(0,),
        )


def test_align_table_color_source_to_shapes_rows_uses_named_index_as_shapes_instance_key() -> None:
    geodataframe = gpd.GeoDataFrame(
        {"cell_type": ["T", "B"]},
        geometry=[_polygon(0), _polygon(2)],
        index=["cell_1", "cell_2"],
    )
    geodataframe.index.name = "instance_id"
    table = ad.AnnData(
        obs=pd.DataFrame(
            {
                "region": ["cells", "cells"],
                "instance_id": ["cell_1", "cell_2"],
                "cell_type": ["T", "B"],
            },
            index=["obs_1", "obs_2"],
        )
    )
    style_spec = TableColorSourceSpec(
        table_name="table",
        source_kind="obs_column",
        value_key="cell_type",
        value_kind="categorical",
    )

    aligned = _align_table_color_source_to_shapes_rows(
        table=table,
        table_metadata=_table_metadata(),
        shapes_name="cells",
        shapes_element=geodataframe,
        style_spec=style_spec,
        source_row_id_by_rendered_row=(0, 1),
    )

    assert aligned.source_row_values.to_list() == ["T", "B"]
    assert aligned.source_row_has_table_row.tolist() == [True, True]


def test_align_table_color_source_to_shapes_rows_allows_duplicate_named_index_instances() -> None:
    geodataframe = gpd.GeoDataFrame(
        {"cell_type": ["T boundary", "T nucleus", "B"]},
        geometry=[_polygon(0), _polygon(2), _polygon(4)],
        index=["cell_1", "cell_1", "cell_2"],
    )
    geodataframe.index.name = "instance_id"
    table = ad.AnnData(
        obs=pd.DataFrame(
            {
                "region": ["cells", "cells"],
                "instance_id": ["cell_1", "cell_2"],
                "cell_type": ["T", "B"],
            },
            index=["obs_1", "obs_2"],
        )
    )
    style_spec = TableColorSourceSpec(
        table_name="table",
        source_kind="obs_column",
        value_key="cell_type",
        value_kind="categorical",
    )

    aligned = _align_table_color_source_to_shapes_rows(
        table=table,
        table_metadata=_table_metadata(),
        shapes_name="cells",
        shapes_element=geodataframe,
        style_spec=style_spec,
        source_row_id_by_rendered_row=(0, 1, 2),
    )

    assert aligned.source_row_values.to_list() == ["T", "T", "B"]
    assert aligned.source_row_has_table_row.tolist() == [True, True, True]


def test_align_table_color_source_to_shapes_rows_accepts_matching_instance_key_column_and_index() -> None:
    geodataframe = gpd.GeoDataFrame(
        {"instance_id": ["cell_1", "cell_2"]},
        geometry=[_polygon(0), _polygon(2)],
        index=["cell_1", "cell_2"],
    )
    geodataframe.index.name = "instance_id"
    table = ad.AnnData(obs=pd.DataFrame({"region": ["cells"], "instance_id": ["cell_1"]}, index=["obs_1"]))
    style_spec = TableColorSourceSpec(
        table_name="table",
        source_kind="obs_column",
        value_key="instance_id",
        value_kind="instance",
    )

    aligned = _align_table_color_source_to_shapes_rows(
        table=table,
        table_metadata=_table_metadata(),
        shapes_name="cells",
        shapes_element=geodataframe,
        style_spec=style_spec,
        source_row_id_by_rendered_row=(0, 1),
    )

    assert aligned.source_row_values.iloc[0] == "cell_1"
    assert pd.isna(aligned.source_row_values.iloc[1])


def test_align_table_color_source_to_shapes_rows_rejects_instance_key_column_when_index_disagrees() -> None:
    geodataframe = gpd.GeoDataFrame(
        {"instance_id": ["cell_1", "cell_2"]},
        geometry=[_polygon(0), _polygon(2)],
        index=["cell_1", "cell_99"],
    )
    geodataframe.index.name = "instance_id"
    table = ad.AnnData(obs=pd.DataFrame({"region": ["cells"], "instance_id": ["cell_1"]}, index=["obs_1"]))
    style_spec = TableColorSourceSpec(
        table_name="table",
        source_kind="obs_column",
        value_key="instance_id",
        value_kind="instance",
    )

    with pytest.raises(ValueError, match="both as a GeoDataFrame column and as the GeoDataFrame index name"):
        _align_table_color_source_to_shapes_rows(
            table=table,
            table_metadata=_table_metadata(),
            shapes_name="cells",
            shapes_element=geodataframe,
            style_spec=style_spec,
            source_row_id_by_rendered_row=(0, 1),
        )


def test_align_table_color_source_to_shapes_rows_requires_shapes_instance_identity() -> None:
    geodataframe = gpd.GeoDataFrame({"cell_type": ["T"]}, geometry=[_polygon(0)], index=["shape_a"])
    table = ad.AnnData(obs=pd.DataFrame({"region": ["cells"], "instance_id": ["cell_1"]}, index=["obs_1"]))
    style_spec = TableColorSourceSpec(
        table_name="table",
        source_kind="obs_column",
        value_key="instance_id",
        value_kind="instance",
    )

    with pytest.raises(ValueError, match="must use GeoDataFrame index"):
        _align_table_color_source_to_shapes_rows(
            table=table,
            table_metadata=_table_metadata(),
            shapes_name="cells",
            shapes_element=geodataframe,
            style_spec=style_spec,
            source_row_id_by_rendered_row=(0,),
        )


def test_align_table_color_source_to_shapes_rows_rejects_missing_shapes_instance_values() -> None:
    geodataframe = gpd.GeoDataFrame(
        {"cell_type": ["T", "B"]},
        geometry=[_polygon(0), _polygon(2)],
        index=["cell_1", None],
    )
    geodataframe.index.name = "instance_id"
    table = ad.AnnData(obs=pd.DataFrame({"region": ["cells"], "instance_id": ["cell_1"]}, index=["obs_1"]))
    style_spec = TableColorSourceSpec(
        table_name="table",
        source_kind="obs_column",
        value_key="instance_id",
        value_kind="instance",
    )

    with pytest.raises(ValueError, match="contains missing values for source row"):
        _align_table_color_source_to_shapes_rows(
            table=table,
            table_metadata=_table_metadata(),
            shapes_name="cells",
            shapes_element=geodataframe,
            style_spec=style_spec,
            source_row_id_by_rendered_row=(0,),
        )


def test_align_table_color_source_to_shapes_rows_rejects_missing_table_instance_values() -> None:
    geodataframe = gpd.GeoDataFrame(
        {},
        geometry=[_polygon(0)],
        index=pd.Index(["cell_1"], name="instance_id"),
    )
    table = ad.AnnData(obs=pd.DataFrame({"region": ["cells"], "instance_id": [None]}, index=["obs_1"]))
    style_spec = TableColorSourceSpec(
        table_name="table",
        source_kind="obs_column",
        value_key="instance_id",
        value_kind="instance",
    )

    with pytest.raises(ValueError, match="contains missing values for table row"):
        _align_table_color_source_to_shapes_rows(
            table=table,
            table_metadata=_table_metadata(),
            shapes_name="cells",
            shapes_element=geodataframe,
            style_spec=style_spec,
            source_row_id_by_rendered_row=(0,),
        )


def test_align_table_color_source_to_shapes_rows_rejects_duplicate_table_instances() -> None:
    geodataframe = gpd.GeoDataFrame(
        {},
        geometry=[_polygon(0)],
        index=pd.Index(["cell_1"], name="instance_id"),
    )
    table = ad.AnnData(
        obs=pd.DataFrame(
            {
                "region": ["cells", "cells"],
                "instance_id": ["cell_1", "cell_1"],
            },
            index=["obs_1", "obs_2"],
        )
    )
    style_spec = TableColorSourceSpec(
        table_name="table",
        source_kind="obs_column",
        value_key="instance_id",
        value_kind="instance",
    )

    with pytest.raises(ValueError, match="contains duplicate values within that region"):
        _align_table_color_source_to_shapes_rows(
            table=table,
            table_metadata=_table_metadata(),
            shapes_name="cells",
            shapes_element=geodataframe,
            style_spec=style_spec,
            source_row_id_by_rendered_row=(0,),
        )


def test_align_table_color_source_to_shapes_rows_rejects_table_instances_missing_from_shapes() -> None:
    geodataframe = gpd.GeoDataFrame(
        {},
        geometry=[_polygon(0)],
        index=pd.Index(["1"], name="instance_id"),
    )
    table = ad.AnnData(obs=pd.DataFrame({"region": ["cells"], "instance_id": [1]}, index=["obs_1"]))
    style_spec = TableColorSourceSpec(
        table_name="table",
        source_kind="obs_column",
        value_key="instance_id",
        value_kind="instance",
    )

    with pytest.raises(ValueError, match="are not present"):
        _align_table_color_source_to_shapes_rows(
            table=table,
            table_metadata=_table_metadata(),
            shapes_name="cells",
            shapes_element=geodataframe,
            style_spec=style_spec,
            source_row_id_by_rendered_row=(0,),
        )


def test_align_table_color_source_to_shapes_rows_rejects_table_instances_missing_from_named_index() -> None:
    geodataframe = gpd.GeoDataFrame(
        {"cell_type": ["T"]},
        geometry=[_polygon(0)],
        index=["cell_1"],
    )
    geodataframe.index.name = "instance_id"
    table = ad.AnnData(obs=pd.DataFrame({"region": ["cells"], "instance_id": ["cell_2"]}, index=["obs_1"]))
    style_spec = TableColorSourceSpec(
        table_name="table",
        source_kind="obs_column",
        value_key="instance_id",
        value_kind="instance",
    )

    with pytest.raises(ValueError, match="resolved shapes instance identities"):
        _align_table_color_source_to_shapes_rows(
            table=table,
            table_metadata=_table_metadata(),
            shapes_name="cells",
            shapes_element=geodataframe,
            style_spec=style_spec,
            source_row_id_by_rendered_row=(0,),
        )


def test_align_table_color_source_to_shapes_rows_rejects_empty_selected_region() -> None:
    geodataframe = gpd.GeoDataFrame(
        {},
        geometry=[_polygon(0)],
        index=pd.Index(["cell_1"], name="instance_id"),
    )
    table = ad.AnnData(obs=pd.DataFrame({"region": ["other"], "instance_id": ["cell_1"]}, index=["obs_1"]))
    style_spec = TableColorSourceSpec(
        table_name="table",
        source_kind="obs_column",
        value_key="instance_id",
        value_kind="instance",
    )

    with pytest.raises(ValueError, match="no table rows annotate"):
        _align_table_color_source_to_shapes_rows(
            table=table,
            table_metadata=_table_metadata(regions=("cells", "other")),
            shapes_name="cells",
            shapes_element=geodataframe,
            style_spec=style_spec,
            source_row_id_by_rendered_row=(0,),
        )


def test_align_table_color_source_to_shapes_rows_tracks_unannotated_instance_index_rows() -> None:
    geodataframe = gpd.GeoDataFrame(
        {},
        geometry=[_polygon(0), _polygon(2)],
        index=pd.Index(["cell_1", "cell_2"], name="instance_id"),
    )
    table = ad.AnnData(obs=pd.DataFrame({"region": ["cells"], "instance_id": ["cell_1"]}, index=["obs_1"]))
    style_spec = TableColorSourceSpec(
        table_name="table",
        source_kind="obs_column",
        value_key="instance_id",
        value_kind="instance",
    )

    aligned = _align_table_color_source_to_shapes_rows(
        table=table,
        table_metadata=_table_metadata(),
        shapes_name="cells",
        shapes_element=geodataframe,
        style_spec=style_spec,
        source_row_id_by_rendered_row=(0, 1),
    )

    assert aligned.source_row_values.index.to_list() == [0, 1]
    assert aligned.source_row_values.iloc[0] == "cell_1"
    assert pd.isna(aligned.source_row_values.iloc[1])
    assert aligned.rendered_row_has_table_row.tolist() == [True, False]


def test_align_table_color_source_to_shapes_rows_rejects_unknown_source_row_id() -> None:
    geodataframe = gpd.GeoDataFrame(
        {},
        geometry=[_polygon(0)],
        index=pd.Index(["cell_1"], name="instance_id"),
    )
    table = ad.AnnData(obs=pd.DataFrame({"region": ["cells"], "instance_id": ["cell_1"]}, index=["obs_1"]))
    style_spec = TableColorSourceSpec(
        table_name="table",
        source_kind="obs_column",
        value_key="instance_id",
        value_kind="instance",
    )

    with pytest.raises(ValueError, match="Could not align rendered shapes back"):
        _align_table_color_source_to_shapes_rows(
            table=table,
            table_metadata=_table_metadata(),
            shapes_name="cells",
            shapes_element=geodataframe,
            style_spec=style_spec,
            source_row_id_by_rendered_row=(999,),
        )


def test_apply_table_color_source_to_shapes_layer_uses_table_categorical_palette_and_table_coverage() -> None:
    geodataframe = gpd.GeoDataFrame(
        {},
        geometry=[_polygon(0), _polygon(2), _polygon(4)],
        index=pd.Index(["cell_1", "cell_2", "cell_3"], name="instance_id"),
    )
    table = ad.AnnData(
        obs=pd.DataFrame(
            {
                "region": ["cells", "cells"],
                "instance_id": ["cell_1", "cell_2"],
                "cell_type": pd.Categorical(["T", None], categories=["T", "B"]),
            },
            index=["obs_1", "obs_2"],
        )
    )
    table.uns["cell_type_colors"] = ["red", "blue"]
    layer = _make_shapes_layer(("cell_1", "cell_1", "cell_2", "cell_3"), source_shapes_index_feature_name="instance_id")
    style_spec = TableColorSourceSpec(
        table_name="table",
        source_kind="obs_column",
        value_key="cell_type",
        value_kind="categorical",
    )

    result = apply_table_color_source_to_shapes_layer(
        layer,
        sdata=_make_shapes_sdata(geodataframe, table),
        shapes_name="cells",
        style_spec=style_spec,
        source_row_id_by_rendered_row=(0, 0, 1, 2),
        source_shapes_index_feature_name="instance_id",
        fill=True,
    )

    assert result.value_kind == "categorical"
    assert result.palette_source == "stored"
    assert result.coercion_applied is False
    assert result.unannotated_source_shape_count == 1
    assert result.unannotated_rendered_shape_count == 1
    np.testing.assert_allclose(
        layer.face_color,
        np.asarray(
            [
                _rgba("red", SHAPES_FACE_ALPHA),
                _rgba("red", SHAPES_FACE_ALPHA),
                _rgba(SHAPES_MISSING_BASE_COLOR, SHAPES_FACE_ALPHA),
                _rgba(SHAPES_MISSING_BASE_COLOR, 0.0),
            ]
        ),
    )
    np.testing.assert_allclose(
        layer.edge_color,
        np.asarray(
            [
                _rgba("red", SHAPES_EDGE_ALPHA),
                _rgba("red", SHAPES_EDGE_ALPHA),
                _rgba(SHAPES_MISSING_BASE_COLOR, SHAPES_EDGE_ALPHA),
                _rgba(SHAPES_MISSING_BASE_COLOR, 0.0),
            ]
        ),
    )
    assert layer.features["cell_type"].iloc[:2].to_list() == ["T", "T"]
    assert pd.isna(layer.features["cell_type"].iloc[2])
    assert pd.isna(layer.features["cell_type"].iloc[3])


def test_apply_table_color_source_to_shapes_layer_colors_point_backed_shapes() -> None:
    geodataframe = gpd.GeoDataFrame(
        {
            "radius": [2.0, 3.0, 4.0],
        },
        geometry=[Point(0, 0), Point(2, 2), Point(4, 4)],
        index=pd.Index(["cell_1", "cell_2", "cell_3"], name="instance_id"),
    )
    table = ad.AnnData(
        obs=pd.DataFrame(
            {
                "region": ["cells", "cells"],
                "instance_id": ["cell_1", "cell_2"],
                "cell_type": pd.Categorical(["T", None], categories=["T", "B"]),
            },
            index=["obs_1", "obs_2"],
        )
    )
    table.uns["cell_type_colors"] = ["red", "blue"]
    layer = _make_points_layer(("cell_1", "cell_2", "cell_3"), source_shapes_index_feature_name="instance_id")
    original_size = layer.size.copy()
    original_symbol = layer.symbol.copy()
    style_spec = TableColorSourceSpec(
        table_name="table",
        source_kind="obs_column",
        value_key="cell_type",
        value_kind="categorical",
    )

    result = apply_table_color_source_to_shapes_layer(
        layer,
        sdata=_make_shapes_sdata(geodataframe, table),
        shapes_name="cells",
        style_spec=style_spec,
        source_row_id_by_rendered_row=range(3),
        source_shapes_index_feature_name="instance_id",
    )

    expected_colors = np.asarray(
        [
            _rgba("red", SHAPES_EDGE_ALPHA),
            _rgba(SHAPES_MISSING_BASE_COLOR, SHAPES_EDGE_ALPHA),
            _rgba(SHAPES_MISSING_BASE_COLOR, 0.0),
        ]
    )
    assert result.value_kind == "categorical"
    assert result.palette_source == "stored"
    assert result.coercion_applied is False
    assert result.unannotated_source_shape_count == 1
    assert result.unannotated_rendered_shape_count == 1
    np.testing.assert_allclose(layer.face_color, expected_colors)
    np.testing.assert_allclose(layer.border_color, expected_colors)
    np.testing.assert_allclose(layer.border_width, np.zeros(3))
    np.testing.assert_allclose(layer.size, original_size)
    np.testing.assert_array_equal(layer.symbol, original_symbol)
    assert layer.features["cell_type"].iloc[0] == "T"
    assert pd.isna(layer.features["cell_type"].iloc[1])
    assert pd.isna(layer.features["cell_type"].iloc[2])


def test_apply_table_color_source_to_shapes_layer_colors_string_instances_with_label_colormap() -> None:
    geodataframe = gpd.GeoDataFrame(
        {},
        geometry=[_polygon(0), _polygon(2), _polygon(4)],
        index=pd.Index(["cell_1", "cell_1", "cell_2"], name="instance_id"),
    )
    table = ad.AnnData(
        obs=pd.DataFrame(
            {
                "region": ["cells", "cells"],
                "instance_id": ["cell_1", "cell_2"],
            },
            index=["obs_1", "obs_2"],
        )
    )
    layer = _make_shapes_layer(("cell_1", "cell_1", "cell_2"), source_shapes_index_feature_name="instance_id")
    style_spec = TableColorSourceSpec(
        table_name="table",
        source_kind="obs_column",
        value_key="instance_id",
        value_kind="instance",
    )

    result = apply_table_color_source_to_shapes_layer(
        layer,
        sdata=_make_shapes_sdata(geodataframe, table),
        shapes_name="cells",
        style_spec=style_spec,
        source_row_id_by_rendered_row=(0, 1, 2),
        source_shapes_index_feature_name="instance_id",
    )

    expected_label_colors = label_colormap(background_value=0).map(np.asarray([1, 1, 2], dtype=np.int64))
    expected_edge = expected_label_colors.copy()
    expected_edge[:, 3] = SHAPES_EDGE_ALPHA
    expected_face = expected_label_colors.copy()
    expected_face[:, 3] = 0.0

    assert result.value_kind == "instance"
    assert result.palette_source is None
    assert result.coercion_applied is False
    assert result.unannotated_source_shape_count == 0
    assert result.unannotated_rendered_shape_count == 0
    np.testing.assert_allclose(layer.face_color, expected_face)
    np.testing.assert_allclose(layer.edge_color, expected_edge)
    assert layer.features["instance_id"].to_list() == ["cell_1", "cell_1", "cell_2"]
    assert layer.features["instance_id__value"].to_list() == ["cell_1", "cell_1", "cell_2"]


def test_apply_table_color_source_to_shapes_layer_uses_positive_integer_instance_ids_for_identity_colors() -> None:
    geodataframe = gpd.GeoDataFrame(
        {},
        geometry=[_polygon(0), _polygon(2), _polygon(4)],
        index=pd.Index([5, 10, 15], name="instance_id"),
    )
    table = ad.AnnData(
        obs=pd.DataFrame(
            {
                "region": ["cells", "cells"],
                "instance_id": [5, 10],
            },
            index=["obs_1", "obs_2"],
        )
    )
    layer = _make_shapes_layer((5, 10, 15), source_shapes_index_feature_name="instance_id")
    style_spec = TableColorSourceSpec(
        table_name="table",
        source_kind="obs_column",
        value_key="instance_id",
        value_kind="instance",
    )

    result = apply_table_color_source_to_shapes_layer(
        layer,
        sdata=_make_shapes_sdata(geodataframe, table),
        shapes_name="cells",
        style_spec=style_spec,
        source_row_id_by_rendered_row=(0, 1, 2),
        source_shapes_index_feature_name="instance_id",
    )

    expected_label_colors = label_colormap(background_value=0).map(np.asarray([5, 10, 0], dtype=np.int64))
    expected_edge = expected_label_colors.copy()
    expected_edge[:, 3] = SHAPES_EDGE_ALPHA
    expected_edge[2, 3] = 0.0

    assert result.value_kind == "instance"
    assert result.unannotated_source_shape_count == 1
    assert result.unannotated_rendered_shape_count == 1
    np.testing.assert_allclose(layer.edge_color, expected_edge)
    assert layer.features["instance_id"].to_list() == [5, 10, 15]
    assert layer.features["instance_id__value"].to_list()[:2] == [5, 10]
    assert pd.isna(layer.features["instance_id__value"].iloc[2])


def test_apply_table_color_source_to_shapes_layer_colors_sparse_x_var_continuously() -> None:
    from scipy import sparse

    geodataframe = gpd.GeoDataFrame(
        {},
        geometry=[_polygon(0), _polygon(2), _polygon(4)],
        index=pd.Index(["cell_1", "cell_2", "cell_3"], name="instance_id"),
    )
    table = ad.AnnData(
        X=sparse.csr_matrix(np.asarray([[1.0, 10.0], [2.0, 20.0]])),
        obs=pd.DataFrame(
            {
                "region": ["cells", "cells"],
                "instance_id": ["cell_1", "cell_2"],
            },
            index=["obs_1", "obs_2"],
        ),
        var=pd.DataFrame(index=["GeneA", "GeneB"]),
    )
    layer = _make_shapes_layer(("cell_1", "cell_2", "cell_3"), source_shapes_index_feature_name="instance_id")
    style_spec = TableColorSourceSpec(
        table_name="table",
        source_kind="x_var",
        value_key="GeneB",
        value_kind="continuous",
    )

    result = apply_table_color_source_to_shapes_layer(
        layer,
        sdata=_make_shapes_sdata(geodataframe, table),
        shapes_name="cells",
        style_spec=style_spec,
        source_row_id_by_rendered_row=(0, 1, 2),
        source_shapes_index_feature_name="instance_id",
    )

    rendered_row_colors = continuous_colors_for_values(
        pd.Series([10.0, 20.0, np.nan]),
        missing_color=SHAPES_MISSING_BASE_COLOR,
    )
    expected_edge = np.asarray([_rgba(color, SHAPES_EDGE_ALPHA) for color in rendered_row_colors])
    expected_edge[2, 3] = 0.0

    assert result.value_kind == "continuous"
    assert result.palette_source is None
    assert result.coercion_applied is False
    assert result.unannotated_source_shape_count == 1
    assert result.unannotated_rendered_shape_count == 1
    np.testing.assert_allclose(layer.edge_color, expected_edge)
    assert layer.features["GeneB"].to_list()[:2] == [10.0, 20.0]
    assert pd.isna(layer.features["GeneB"].iloc[2])


def test_apply_table_color_source_to_shapes_layer_coerces_string_obs_to_temporary_categorical() -> None:
    geodataframe = gpd.GeoDataFrame(
        {},
        geometry=[_polygon(0), _polygon(2)],
        index=pd.Index(["cell_1", "cell_2"], name="instance_id"),
    )
    table = ad.AnnData(
        obs=pd.DataFrame(
            {
                "region": ["cells", "cells"],
                "instance_id": ["cell_1", "cell_2"],
                "free_text": ["alpha", "beta"],
            },
            index=["obs_1", "obs_2"],
        )
    )
    table.uns["free_text_colors"] = ["red", "blue"]
    layer = _make_shapes_layer(("cell_1", "cell_2"), source_shapes_index_feature_name="instance_id")
    style_spec = TableColorSourceSpec(
        table_name="table",
        source_kind="obs_column",
        value_key="free_text",
        value_kind="categorical",
    )

    result = apply_table_color_source_to_shapes_layer(
        layer,
        sdata=_make_shapes_sdata(geodataframe, table),
        shapes_name="cells",
        style_spec=style_spec,
        source_row_id_by_rendered_row=(0, 1),
        source_shapes_index_feature_name="instance_id",
    )

    assert result.value_kind == "categorical"
    assert result.palette_source == "default_missing"
    assert result.coercion_applied is True
    assert layer.features["free_text"].to_list() == ["alpha", "beta"]


def test_apply_table_color_source_to_shapes_layer_disambiguates_style_feature_from_source_index_feature() -> None:
    geodataframe = gpd.GeoDataFrame(
        {},
        geometry=[_polygon(0)],
        index=pd.Index(["cell_1"], name="instance_id"),
    )
    table = ad.AnnData(
        obs=pd.DataFrame(
            {
                "region": ["cells"],
                "instance_id": ["cell_1"],
            },
            index=["obs_1"],
        )
    )
    layer = _make_shapes_layer(("cell_1",), source_shapes_index_feature_name="instance_id")
    style_spec = TableColorSourceSpec(
        table_name="table",
        source_kind="obs_column",
        value_key="instance_id",
        value_kind="instance",
    )

    apply_table_color_source_to_shapes_layer(
        layer,
        sdata=_make_shapes_sdata(geodataframe, table),
        shapes_name="cells",
        style_spec=style_spec,
        source_row_id_by_rendered_row=(0,),
        source_shapes_index_feature_name="instance_id",
    )

    assert layer.features["instance_id"].to_list() == ["cell_1"]
    assert layer.features["instance_id__value"].to_list() == ["cell_1"]


def test_apply_shape_column_color_source_to_shapes_layer_uses_stored_categorical_companion_palette() -> None:
    geodataframe = gpd.GeoDataFrame(
        {
            "cell_type": pd.Categorical(["T", "B", None]),
            "cell_type_colors": ["red", "blue", None],
        },
        geometry=[_polygon(0), _polygon(2), _polygon(4)],
        index=["cell_1", "cell_2", "cell_3"],
    )
    layer = _make_shapes_layer(("cell_1", "cell_1", "cell_2", "cell_3"))
    style_spec = ShapeColumnColorSourceSpec(
        source_kind="shape_column",
        value_key="cell_type",
        value_kind="categorical",
    )

    result = apply_shape_column_color_source_to_shapes_layer(
        layer,
        shapes_element=geodataframe,
        style_spec=style_spec,
        source_row_id_by_rendered_row=(0, 0, 1, 2),
        source_shapes_index_feature_name="index",
    )

    assert result.value_kind == "categorical"
    assert result.palette_source == "stored"
    assert result.coercion_applied is False
    np.testing.assert_allclose(
        layer.face_color,
        np.asarray(
            [
                _rgba("red", 0.0),
                _rgba("red", 0.0),
                _rgba("blue", 0.0),
                _rgba(SHAPES_MISSING_BASE_COLOR, 0.0),
            ]
        ),
    )
    np.testing.assert_allclose(
        layer.edge_color,
        np.asarray(
            [
                _rgba("red", SHAPES_EDGE_ALPHA),
                _rgba("red", SHAPES_EDGE_ALPHA),
                _rgba("blue", SHAPES_EDGE_ALPHA),
                _rgba(SHAPES_MISSING_BASE_COLOR, SHAPES_EDGE_ALPHA),
            ]
        ),
    )
    assert layer.features["cell_type"].iloc[:3].to_list() == ["T", "T", "B"]
    assert pd.isna(layer.features["cell_type"].iloc[3])


def test_apply_shape_column_color_source_to_shapes_layer_uses_continuous_colormap_and_missing_gray() -> None:
    geodataframe = gpd.GeoDataFrame(
        {"score": [0.0, 10.0, None]},
        geometry=[_polygon(0), _polygon(2), _polygon(4)],
        index=["cell_1", "cell_2", "cell_3"],
    )
    layer = _make_shapes_layer(("cell_1", "cell_2", "cell_3"))
    style_spec = ShapeColumnColorSourceSpec(
        source_kind="shape_column",
        value_key="score",
        value_kind="continuous",
    )

    result = apply_shape_column_color_source_to_shapes_layer(
        layer,
        shapes_element=geodataframe,
        style_spec=style_spec,
        source_row_id_by_rendered_row=(0, 1, 2),
        source_shapes_index_feature_name="index",
    )

    rendered_row_colors = continuous_colors_for_values(
        pd.Series([0.0, 10.0, np.nan]),
        missing_color=SHAPES_MISSING_BASE_COLOR,
    )
    expected_face = np.asarray([_rgba(color, 0.0) for color in rendered_row_colors])
    expected_edge = np.asarray([_rgba(color, SHAPES_EDGE_ALPHA) for color in rendered_row_colors])

    assert result.value_kind == "continuous"
    assert result.palette_source is None
    assert result.coercion_applied is False
    np.testing.assert_allclose(layer.face_color, expected_face)
    np.testing.assert_allclose(layer.edge_color, expected_edge)
    assert layer.features["score"].to_list()[:2] == [0.0, 10.0]
    assert pd.isna(layer.features["score"].iloc[2])


def test_apply_shape_column_color_source_to_shapes_layer_can_fill_faces() -> None:
    geodataframe = gpd.GeoDataFrame(
        {"score": [0.0, 10.0]},
        geometry=[_polygon(0), _polygon(2)],
        index=["cell_1", "cell_2"],
    )
    layer = _make_shapes_layer(("cell_1", "cell_2"))
    style_spec = ShapeColumnColorSourceSpec(
        source_kind="shape_column",
        value_key="score",
        value_kind="continuous",
    )

    apply_shape_column_color_source_to_shapes_layer(
        layer,
        shapes_element=geodataframe,
        style_spec=style_spec,
        source_row_id_by_rendered_row=(0, 1),
        source_shapes_index_feature_name="index",
        fill=True,
    )

    np.testing.assert_allclose(layer.face_color[:, 3], np.full(len(layer.data), SHAPES_FACE_ALPHA))
    np.testing.assert_allclose(layer.edge_color[:, 3], np.full(len(layer.data), SHAPES_EDGE_ALPHA))


@pytest.mark.parametrize(
    "colors",
    [
        ["red", "not-a-color", "green"],
        ["red", "blue", "green"],
        ["red", "red", None],
    ],
)
def test_apply_shape_column_color_source_to_shapes_layer_rejects_invalid_companion_palettes(
    colors: list[object],
) -> None:
    geodataframe = gpd.GeoDataFrame(
        {
            "group": pd.Categorical(["a", "a", "b"]),
            "group_colors": colors,
        },
        geometry=[_polygon(0), _polygon(2), _polygon(4)],
        index=["cell_1", "cell_2", "cell_3"],
    )
    layer = _make_shapes_layer(("cell_1", "cell_2", "cell_3"))
    style_spec = ShapeColumnColorSourceSpec(
        source_kind="shape_column",
        value_key="group",
        value_kind="categorical",
    )

    result = apply_shape_column_color_source_to_shapes_layer(
        layer,
        shapes_element=geodataframe,
        style_spec=style_spec,
        source_row_id_by_rendered_row=(0, 1, 2),
        source_shapes_index_feature_name="index",
    )

    assert result.value_kind == "categorical"
    assert result.palette_source == "default_invalid"
    assert result.coercion_applied is False


@pytest.mark.parametrize(
    ("values", "colors"),
    [
        ([True, False, True], ["red", "blue", "red"]),
        ([0, 1, 0], ["red", "blue", "red"]),
    ],
)
def test_apply_shape_column_color_source_to_shapes_layer_uses_stored_palettes_for_bool_and_binary_integer_columns(
    values: list[object],
    colors: list[str],
) -> None:
    geodataframe = gpd.GeoDataFrame(
        {
            "group": values,
            "group_colors": colors,
        },
        geometry=[_polygon(0), _polygon(2), _polygon(4)],
        index=["cell_1", "cell_2", "cell_3"],
    )
    layer = _make_shapes_layer(("cell_1", "cell_2", "cell_3"))
    style_spec = ShapeColumnColorSourceSpec(
        source_kind="shape_column",
        value_key="group",
        value_kind="categorical",
    )

    result = apply_shape_column_color_source_to_shapes_layer(
        layer,
        shapes_element=geodataframe,
        style_spec=style_spec,
        source_row_id_by_rendered_row=(0, 1, 2),
        source_shapes_index_feature_name="index",
    )

    assert result.palette_source == "stored"
    np.testing.assert_allclose(
        layer.edge_color,
        np.asarray(
            [
                _rgba("red", SHAPES_EDGE_ALPHA),
                _rgba("blue", SHAPES_EDGE_ALPHA),
                _rgba("red", SHAPES_EDGE_ALPHA),
            ]
        ),
    )


def test_apply_shape_column_color_source_to_shapes_layer_reports_missing_companion_palette() -> None:
    geodataframe = gpd.GeoDataFrame(
        {"group": pd.Categorical(["a", "b"])},
        geometry=[_polygon(0), _polygon(2)],
        index=["cell_1", "cell_2"],
    )
    layer = _make_shapes_layer(("cell_1", "cell_2"))
    style_spec = ShapeColumnColorSourceSpec(
        source_kind="shape_column",
        value_key="group",
        value_kind="categorical",
    )

    result = apply_shape_column_color_source_to_shapes_layer(
        layer,
        shapes_element=geodataframe,
        style_spec=style_spec,
        source_row_id_by_rendered_row=(0, 1),
        source_shapes_index_feature_name="index",
    )

    assert result.palette_source == "default_missing"
    assert result.coercion_applied is False


def test_apply_shape_column_color_source_to_shapes_layer_treats_string_object_values_as_temporary_categorical() -> None:
    geodataframe = gpd.GeoDataFrame(
        {
            "free_text": ["alpha", "beta"],
            "free_text_colors": ["red", "blue"],
        },
        geometry=[_polygon(0), _polygon(2)],
        index=["cell_1", "cell_2"],
    )
    original_dtype = geodataframe["free_text"].dtype
    layer = _make_shapes_layer(("cell_1", "cell_2"))
    style_spec = ShapeColumnColorSourceSpec(
        source_kind="shape_column",
        value_key="free_text",
        value_kind="categorical",
    )

    result = apply_shape_column_color_source_to_shapes_layer(
        layer,
        shapes_element=geodataframe,
        style_spec=style_spec,
        source_row_id_by_rendered_row=(0, 1),
        source_shapes_index_feature_name="index",
    )

    assert result.value_kind == "categorical"
    assert result.palette_source == "default_missing"
    assert result.coercion_applied is True
    assert geodataframe["free_text"].dtype == original_dtype
    assert layer.features["free_text"].to_list() == ["alpha", "beta"]


def test_apply_shape_column_color_source_to_shapes_layer_disambiguates_style_feature_from_source_index_feature() -> (
    None
):
    geodataframe = gpd.GeoDataFrame(
        {"cell_id": pd.Categorical(["A"])},
        geometry=[_polygon(0)],
        index=["cell_1"],
    )
    geodataframe.index.name = "cell_id"
    layer = _make_shapes_layer(("cell_1",), source_shapes_index_feature_name="cell_id")
    style_spec = ShapeColumnColorSourceSpec(
        source_kind="shape_column",
        value_key="cell_id",
        value_kind="categorical",
    )

    apply_shape_column_color_source_to_shapes_layer(
        layer,
        shapes_element=geodataframe,
        style_spec=style_spec,
        source_row_id_by_rendered_row=(0,),
        source_shapes_index_feature_name="cell_id",
    )

    assert layer.features["cell_id"].to_list() == ["cell_1"]
    assert layer.features["cell_id__value"].to_list() == ["A"]
    assert disambiguate_shape_style_feature_name("cell_id", "cell_id") == "cell_id__value"


def test_apply_shape_column_color_source_to_shapes_layer_allows_duplicate_source_indices() -> None:
    geodataframe = gpd.GeoDataFrame(
        {"group": pd.Categorical(["a", "b"])},
        geometry=[_polygon(0), _polygon(2)],
        index=["cell_1", "cell_1"],
    )
    layer = _make_shapes_layer(("cell_1", "cell_1"))
    style_spec = ShapeColumnColorSourceSpec(
        source_kind="shape_column",
        value_key="group",
        value_kind="categorical",
    )

    result = apply_shape_column_color_source_to_shapes_layer(
        layer,
        shapes_element=geodataframe,
        style_spec=style_spec,
        source_row_id_by_rendered_row=(0, 1),
        source_shapes_index_feature_name="index",
    )

    assert result.value_kind == "categorical"
    assert layer.features["index"].to_list() == ["cell_1", "cell_1"]
    assert layer.features["group"].to_list() == ["a", "b"]


def test_apply_rendered_row_colors_to_shapes_layer_requires_one_color_per_rendered_row() -> None:
    layer = _make_shapes_layer(("cell_1", "cell_2"))

    with pytest.raises(ValueError, match="one RGBA color for each rendered napari shape row"):
        _apply_rendered_row_colors_to_shapes_layer(layer, np.asarray([_rgba("red", 1.0)]))


def test_build_styled_shapes_layer_name_returns_stable_shape_column_variant_name() -> None:
    style_spec = ShapeColumnColorSourceSpec(
        source_kind="shape_column",
        value_key="cell_type",
        value_kind="categorical",
    )

    assert build_styled_shapes_layer_name("cell_boundaries", style_spec) == "cell_boundaries[shapes_column:cell_type]"


def test_build_styled_shapes_layer_name_returns_stable_table_variant_names() -> None:
    obs_style = TableColorSourceSpec(
        table_name="table",
        source_kind="obs_column",
        value_key="cell_type",
        value_kind="categorical",
    )
    x_style = TableColorSourceSpec(
        table_name="table",
        source_kind="x_var",
        value_key="GeneA",
        value_kind="continuous",
    )

    assert build_styled_shapes_layer_name("cell_boundaries", obs_style) == "cell_boundaries[obs:cell_type]"
    assert build_styled_shapes_layer_name("cell_boundaries", x_style) == "cell_boundaries[X:GeneA]"
