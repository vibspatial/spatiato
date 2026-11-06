from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from spatialdata import SpatialData

from spatiato.core.class_palette import default_categorical_colors, default_labeled_class_color
from spatiato.core.spatial_query import (
    CanonicalCenterQueryResult,
    SpatialAnnotationColumnChangedError,
    SpatialAnnotationPreparation,
    SpatialAnnotationQueryOutdatedError,
    apply_spatial_annotation,
    get_compatible_spatial_annotation_column_names,
    prepare_spatial_annotation,
    summarize_spatial_annotation,
)
from spatiato.core.spatial_query.canonical_centers import (
    compute_canonical_centers,
    read_canonical_centers_region,
)


def test_compatible_annotation_column_discovery_uses_shared_contract_and_table_order(
    sdata_blobs: SpatialData,
) -> None:
    table = sdata_blobs.tables["table"]
    table.obs["first"] = pd.Categorical(["A"] * table.n_obs, categories=["A"])
    table.obs["plain_string"] = pd.Series(["A"] * table.n_obs, index=table.obs.index, dtype="string")
    table.obs["integer_category"] = pd.Categorical([1] * table.n_obs, categories=[1])
    table.obs["empty"] = pd.Categorical([pd.NA] * table.n_obs, categories=[])
    table.obs["user_class"] = pd.Categorical([pd.NA] * table.n_obs, categories=[])
    table.obs["pred_class"] = pd.Categorical([1] * table.n_obs, categories=[1])
    table.obs["pred_confidence"] = pd.Categorical(["high"] * table.n_obs, categories=["high"])
    table.obs["invalid_integer_category"] = pd.Categorical([0] * table.n_obs, categories=[0])
    table.obs[1] = pd.Categorical(["ignored"] * table.n_obs, categories=["ignored"])

    assert get_compatible_spatial_annotation_column_names(sdata_blobs, "table") == [
        "first",
        "integer_category",
        "empty",
        "user_class",
    ]


def test_existing_annotation_set_extends_category_and_valid_palette_stably(
    sdata_blobs: SpatialData,
) -> None:
    table = sdata_blobs.tables["table"]
    query_result = _query_result(sdata_blobs, matched_positions=(0, 1, 2))
    values = [pd.NA] * table.n_obs
    values[1:4] = ["A", "B", "B"]
    table.obs["annotation"] = pd.Categorical(values, categories=["A", "B"], ordered=True)
    table.uns["annotation_colors"] = ["#112233", "#445566"]

    preparation = prepare_spatial_annotation(
        sdata_blobs,
        query_result=query_result,
        column_name="annotation",
        column_mode="existing",
    )
    summary = summarize_spatial_annotation(preparation, "  C  ")
    result = apply_spatial_annotation(sdata_blobs, preparation, summary)

    assert summary.annotation_value == "C"
    assert (summary.current_missing_count, summary.current_equal_count, summary.current_other_count) == (1, 0, 2)
    assert summary.changed_count == 3
    assert summary.overwrite_count == 2
    assert result.annotation_changed
    assert result.palette_changed
    assert table.obs["annotation"].iloc[:4].tolist() == ["C", "C", "C", "B"]
    assert list(table.obs["annotation"].cat.categories) == ["A", "B", "C"]
    assert table.obs["annotation"].cat.ordered
    assert table.uns["annotation_colors"] == ["#112233", "#445566", default_labeled_class_color(3)]


def test_positive_integer_annotation_appends_and_retains_vocabulary_and_palette(
    sdata_blobs: SpatialData,
) -> None:
    table = sdata_blobs.tables["table"]
    query_result = _query_result(sdata_blobs, matched_positions=(0, 1, 2))
    values = [pd.NA] * table.n_obs
    values[:3] = [3, 1, 1]
    table.obs["user_class"] = pd.Categorical(values, categories=[3, 1])
    table.uns["user_class_colors"] = ["#112233", "#445566"]

    preparation = prepare_spatial_annotation(
        sdata_blobs,
        query_result=query_result,
        column_name="user_class",
        column_mode="existing",
    )
    summary = summarize_spatial_annotation(preparation, 2)
    result = apply_spatial_annotation(sdata_blobs, preparation, summary)

    assert preparation.value_kind == "positive_integer"
    assert summary.annotation_value == 2
    assert table.obs["user_class"].iloc[:3].tolist() == [2, 2, 2]
    assert list(table.obs["user_class"].cat.categories) == [3, 1, 2]
    assert table.uns["user_class_colors"] == ["#112233", "#445566", default_labeled_class_color(3)]
    assert result.annotation_changed and result.palette_changed

    removal_preparation = prepare_spatial_annotation(
        sdata_blobs,
        query_result=query_result,
        column_name="user_class",
        column_mode="existing",
    )
    removal_summary = summarize_spatial_annotation(removal_preparation, None)
    removal_result = apply_spatial_annotation(
        sdata_blobs,
        removal_preparation,
        removal_summary,
    )

    assert table.obs["user_class"].iloc[:3].isna().all()
    assert list(table.obs["user_class"].cat.categories) == [3, 1, 2]
    assert table.uns["user_class_colors"] == ["#112233", "#445566", default_labeled_class_color(3)]
    assert removal_result.annotation_changed
    assert removal_result.palette_changed is False


def test_empty_user_class_retains_positive_integer_semantics_on_first_apply(
    sdata_blobs: SpatialData,
) -> None:
    table = sdata_blobs.tables["table"]
    query_result = _query_result(sdata_blobs, matched_positions=(0, 2))
    table.obs["user_class"] = pd.Categorical([pd.NA] * table.n_obs, categories=[])

    preparation = prepare_spatial_annotation(
        sdata_blobs,
        query_result=query_result,
        column_name="user_class",
        column_mode="existing",
    )
    summary = summarize_spatial_annotation(preparation, 4)
    result = apply_spatial_annotation(sdata_blobs, preparation, summary)

    assert preparation.value_kind == "positive_integer"
    assert table.obs["user_class"].iloc[[0, 2]].tolist() == [4, 4]
    assert table.obs["user_class"].drop(table.obs.index[[0, 2]]).isna().all()
    assert list(table.obs["user_class"].cat.categories) == [4]
    assert table.uns["user_class_colors"] == default_categorical_colors(1)
    assert result.annotation_changed and result.palette_changed


def test_annotation_summary_value_must_match_the_prepared_value_kind(
    sdata_blobs: SpatialData,
) -> None:
    table = sdata_blobs.tables["table"]
    query_result = _query_result(sdata_blobs, matched_positions=(0,))
    table.obs["string_annotation"] = pd.Categorical(["A"] * table.n_obs, categories=["A"])
    table.obs["integer_annotation"] = pd.Categorical([1] * table.n_obs, categories=[1])

    string_preparation = prepare_spatial_annotation(
        sdata_blobs,
        query_result=query_result,
        column_name="string_annotation",
        column_mode="existing",
    )
    integer_preparation = prepare_spatial_annotation(
        sdata_blobs,
        query_result=query_result,
        column_name="integer_annotation",
        column_mode="existing",
    )

    assert string_preparation.value_kind == "string"
    assert integer_preparation.value_kind == "positive_integer"
    with pytest.raises(TypeError, match="require a string"):
        summarize_spatial_annotation(string_preparation, 2)
    with pytest.raises(TypeError, match="require a Python int"):
        summarize_spatial_annotation(integer_preparation, "2")
    with pytest.raises(ValueError, match="must be positive"):
        summarize_spatial_annotation(integer_preparation, 0)
    with pytest.raises(TypeError, match="require a Python int"):
        summarize_spatial_annotation(integer_preparation, True)
    with pytest.raises(TypeError, match="require a Python int"):
        summarize_spatial_annotation(integer_preparation, np.int64(2))


def test_prediction_columns_and_new_object_classification_columns_are_rejected(
    sdata_blobs: SpatialData,
) -> None:
    table = sdata_blobs.tables["table"]
    query_result = _query_result(sdata_blobs, matched_positions=(0,))
    table.obs["pred_class"] = pd.Categorical([1] * table.n_obs, categories=[1])

    with pytest.raises(ValueError, match="classifier-owned"):
        prepare_spatial_annotation(
            sdata_blobs,
            query_result=query_result,
            column_name="pred_class",
            column_mode="existing",
        )

    for column_name in ("user_class", "pred_class", "pred_confidence"):
        if column_name in table.obs:
            table.obs.pop(column_name)
        with pytest.raises(ValueError, match="reserved for Object Classification"):
            prepare_spatial_annotation(
                sdata_blobs,
                query_result=query_result,
                column_name=column_name,
                column_mode="new",
            )


@pytest.mark.parametrize(
    ("stored_palette", "expected_palette_changed", "expected_palette"),
    [
        (None, True, default_categorical_colors(2)),
        (["#ff0000"], True, default_categorical_colors(2)),
        (["#112233", "#445566"], False, ["#112233", "#445566"]),
    ],
    ids=["missing", "invalid", "valid"],
)
def test_remove_annotation_preserves_categories_and_resolves_palette_only_on_effective_apply(
    sdata_blobs: SpatialData,
    stored_palette: list[str] | None,
    expected_palette_changed: bool,
    expected_palette: list[str],
) -> None:
    table = sdata_blobs.tables["table"]
    query_result = _query_result(sdata_blobs, matched_positions=(0, 1, 2))
    values = [pd.NA] * table.n_obs
    values[:3] = ["A", pd.NA, "B"]
    table.obs["annotation"] = pd.Categorical(values, categories=["A", "B"])
    if stored_palette is not None:
        table.uns["annotation_colors"] = stored_palette

    preparation = prepare_spatial_annotation(
        sdata_blobs,
        query_result=query_result,
        column_name="annotation",
        column_mode="existing",
    )
    summary = summarize_spatial_annotation(preparation, None)
    result = apply_spatial_annotation(sdata_blobs, preparation, summary)

    assert (summary.current_missing_count, summary.current_equal_count, summary.current_other_count) == (1, 0, 2)
    assert summary.removal_count == 2
    assert table.obs["annotation"].iloc[:3].isna().all()
    assert list(table.obs["annotation"].cat.categories) == ["A", "B"]
    assert table.uns["annotation_colors"] == expected_palette
    assert result.annotation_changed
    assert result.palette_changed is expected_palette_changed


def test_new_annotation_column_is_categorical_and_missing_outside_matching_rows(
    sdata_blobs: SpatialData,
) -> None:
    table = sdata_blobs.tables["table"]
    query_result = _query_result(sdata_blobs, matched_positions=(0, 2))

    preparation = prepare_spatial_annotation(
        sdata_blobs,
        query_result=query_result,
        column_name="  spatial_annotation  ",
        column_mode="new",
    )
    summary = summarize_spatial_annotation(preparation, "  tumor  ")
    result = apply_spatial_annotation(sdata_blobs, preparation, summary)

    annotation = table.obs["spatial_annotation"]
    assert preparation.column_name == "spatial_annotation"
    assert summary.annotation_value == "tumor"
    assert annotation.iloc[[0, 2]].tolist() == ["tumor", "tumor"]
    assert annotation.drop(annotation.index[[0, 2]]).isna().all()
    assert list(annotation.cat.categories) == ["tumor"]
    assert table.uns["spatial_annotation_colors"] == default_categorical_colors(1)
    assert result.annotation_changed and result.palette_changed


def test_noop_annotation_does_not_repair_a_missing_palette(
    sdata_blobs: SpatialData,
) -> None:
    table = sdata_blobs.tables["table"]
    query_result = _query_result(sdata_blobs, matched_positions=(0, 1))
    table.obs["annotation"] = pd.Categorical(["A"] * table.n_obs, categories=["A"])
    before = table.obs["annotation"].copy(deep=True)

    preparation = prepare_spatial_annotation(
        sdata_blobs,
        query_result=query_result,
        column_name="annotation",
        column_mode="existing",
    )
    summary = summarize_spatial_annotation(preparation, "A")
    result = apply_spatial_annotation(sdata_blobs, preparation, summary)

    assert summary.changed_count == 0
    assert result.annotation_changed is False
    assert result.palette_changed is False
    assert table.obs["annotation"].equals(before)
    assert "annotation_colors" not in table.uns


def test_apply_rejects_changed_reviewed_values_without_mutation(
    sdata_blobs: SpatialData,
) -> None:
    table = sdata_blobs.tables["table"]
    query_result = _query_result(sdata_blobs, matched_positions=(0, 1))
    table.obs["annotation"] = pd.Categorical(["A"] * table.n_obs, categories=["A", "B"])
    preparation = prepare_spatial_annotation(
        sdata_blobs,
        query_result=query_result,
        column_name="annotation",
        column_mode="existing",
    )
    summary = summarize_spatial_annotation(preparation, "B")
    table.obs.loc[table.obs.index[0], "annotation"] = "B"
    before = table.obs["annotation"].copy(deep=True)

    with pytest.raises(SpatialAnnotationColumnChangedError, match="values changed"):
        apply_spatial_annotation(sdata_blobs, preparation, summary)

    assert table.obs["annotation"].equals(before)
    assert "annotation_colors" not in table.uns


def test_apply_rejects_changed_canonical_centers_without_mutation(
    sdata_blobs: SpatialData,
) -> None:
    table = sdata_blobs.tables["table"]
    query_result = _query_result(sdata_blobs, matched_positions=(0, 1))
    table.obs["annotation"] = pd.Categorical([pd.NA] * table.n_obs, categories=["A"])
    preparation = prepare_spatial_annotation(
        sdata_blobs,
        query_result=query_result,
        column_name="annotation",
        column_mode="existing",
    )
    summary = summarize_spatial_annotation(preparation, "A")
    table.obsm["spatial_canonical"][0, 1] += 1.0

    with pytest.raises(SpatialAnnotationQueryOutdatedError, match="centers of the selected labels element changed"):
        apply_spatial_annotation(sdata_blobs, preparation, summary)

    assert table.obs["annotation"].isna().all()
    assert "annotation_colors" not in table.uns


def test_apply_rejects_canonical_centers_that_are_no_longer_valid_without_mutation(
    sdata_blobs: SpatialData,
) -> None:
    table = sdata_blobs.tables["table"]
    query_result = _query_result(sdata_blobs, matched_positions=(0, 1))
    table.obs["annotation"] = pd.Categorical([pd.NA] * table.n_obs, categories=["A"])
    preparation = prepare_spatial_annotation(
        sdata_blobs,
        query_result=query_result,
        column_name="annotation",
        column_mode="existing",
    )
    summary = summarize_spatial_annotation(preparation, "A")
    # A matrix without its record is invalid.
    del table.uns["spatial_coordinates"]["spatial_canonical"]

    with pytest.raises(SpatialAnnotationQueryOutdatedError, match="no longer valid"):
        apply_spatial_annotation(sdata_blobs, preparation, summary)

    assert table.obs["annotation"].isna().all()


def test_apply_rejects_changed_table_rows_of_the_labels_element_without_mutation(
    sdata_blobs: SpatialData,
) -> None:
    table = sdata_blobs.tables["table"]
    query_result = _query_result(sdata_blobs, matched_positions=(0, 1))
    table.obs["annotation"] = pd.Categorical([pd.NA] * table.n_obs, categories=["A"])
    preparation = prepare_spatial_annotation(
        sdata_blobs,
        query_result=query_result,
        column_name="annotation",
        column_mode="existing",
    )
    summary = summarize_spatial_annotation(preparation, "A")
    instance_ids = table.obs["instance_id"].to_numpy().copy()
    instance_ids[[0, 1]] = instance_ids[[1, 0]]
    table.obs["instance_id"] = instance_ids

    with pytest.raises(SpatialAnnotationQueryOutdatedError, match="table rows of the selected labels element changed"):
        apply_spatial_annotation(sdata_blobs, preparation, summary)

    assert table.obs["annotation"].isna().all()


def test_annotation_assignment_removes_new_column_when_palette_assignment_fails(
    sdata_blobs: SpatialData,
) -> None:
    table = sdata_blobs.tables["table"]
    query_result = _query_result(sdata_blobs, matched_positions=(0, 1))
    preparation = prepare_spatial_annotation(
        sdata_blobs,
        query_result=query_result,
        column_name="annotation",
        column_mode="new",
    )
    summary = summarize_spatial_annotation(preparation, "B")

    class FailOncePaletteAssignment(dict):
        should_fail = True

        def __setitem__(self, key, value) -> None:
            if key == "annotation_colors" and self.should_fail:
                self.should_fail = False
                raise RuntimeError("injected palette assignment failure")
            super().__setitem__(key, value)

    table.uns = FailOncePaletteAssignment(table.uns)

    with pytest.raises(RuntimeError, match="injected palette assignment failure"):
        apply_spatial_annotation(sdata_blobs, preparation, summary)

    assert "annotation" not in table.obs.columns
    assert "annotation_colors" not in table.uns


def test_preparation_rejects_row_positions_that_are_not_the_rows_of_the_matched_instances(
    sdata_blobs: SpatialData,
) -> None:
    query_result = _query_result(sdata_blobs, matched_positions=(0, 1))
    preparation = prepare_spatial_annotation(
        sdata_blobs,
        query_result=query_result,
        column_name="annotation",
        column_mode="new",
    )
    region = query_result.canonical_centers
    other_rows = region.row_positions[[2, 3]]

    # The right number of unique rows, but those of other instances.
    with pytest.raises(ValueError, match="table rows of the matched instance IDs"):
        SpatialAnnotationPreparation(
            query_result=query_result,
            column_name=preparation.column_name,
            column_mode=preparation.column_mode,
            row_positions=other_rows,
            prepared_values=preparation.prepared_values,
        )


def test_prepare_rejects_linkage_and_non_categorical_existing_columns(
    sdata_blobs: SpatialData,
) -> None:
    table = sdata_blobs.tables["table"]
    query_result = _query_result(sdata_blobs, matched_positions=(0,))
    table.obs["text_annotation"] = "A"

    with pytest.raises(ValueError, match="linkage column"):
        prepare_spatial_annotation(
            sdata_blobs,
            query_result=query_result,
            column_name="instance_id",
            column_mode="existing",
        )
    with pytest.raises(ValueError, match="must be categorical"):
        prepare_spatial_annotation(
            sdata_blobs,
            query_result=query_result,
            column_name="text_annotation",
            column_mode="existing",
        )


def _query_result(
    sdata: SpatialData,
    *,
    matched_positions: tuple[int, ...],
) -> CanonicalCenterQueryResult:
    """Calculate the table's canonical centers and match the instances at `matched_positions` of `blobs_labels`."""
    compute_canonical_centers(sdata, table_name="table")
    region = read_canonical_centers_region(sdata, table_name="table", labels_name="blobs_labels")
    matching_ids = np.sort(region.instance_ids[list(matched_positions)])
    return CanonicalCenterQueryResult(
        canonical_centers=region,
        matched_instance_ids=matching_ids,
    )
