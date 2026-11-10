"""Primary-label viewer styling for spatial annotation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from spatiato.core._color_source import TableColorSourceSpec
from spatiato.core.spatial_query.annotation import require_compatible_spatial_annotation_column
from spatiato.core.spatialdata import get_table
from spatiato.viewer.adapter import ViewerAdapter
from spatiato.viewer.labels_styling import (
    LabelsLoadResult,
    apply_neutral_labels_style,
    apply_table_color_source_to_labels_layer,
)

if TYPE_CHECKING:
    from spatialdata import SpatialData


def load_and_style_unannotated_spatial_annotation_labels(
    viewer_adapter: ViewerAdapter,
    *,
    sdata: SpatialData,
    coordinate_system: str,
    labels_name: str,
) -> LabelsLoadResult:
    """Load, neutrally style, and activate an unannotated primary labels layer."""
    load_result = viewer_adapter.ensure_labels_loaded(
        sdata,
        labels_name,
        coordinate_system,
    )
    apply_neutral_labels_style(load_result.layer)
    viewer_adapter.activate_layer(load_result.layer)

    return LabelsLoadResult(
        layer=load_result.layer,
        created=load_result.created,
        value_kind=None,
        palette_source=None,
        coercion_applied=False,
    )


def load_and_style_spatial_annotation_labels(
    viewer_adapter: ViewerAdapter,
    *,
    sdata: SpatialData,
    coordinate_system: str,
    labels_name: str,
    table_name: str,
    column_name: str,
) -> LabelsLoadResult:
    """Load, style, and activate the primary labels layer for annotation.

    The categorical string or positive-integer column is validated before the
    viewer is changed. Styling delegates table-binding validation, palette
    resolution, and labels-feature construction to the generic table-backed
    labels styling API. It changes only napari layer presentation state: it
    does not mutate the table, its persisted state, or the canonical-center
    cache.

    Parameters
    ----------
    viewer_adapter
        Viewer boundary used to load and activate the primary labels layer.
    sdata
        SpatialData object containing the labels element and annotation table.
    coordinate_system
        Coordinate system in which the primary labels layer is displayed.
    labels_name
        Name of the labels element annotated by the selected table.
    table_name
        Name of the table containing the annotation column.
    column_name
        Existing categorical string or positive-integer observation column
        used for coloring.

    Returns
    -------
    LabelsLoadResult
        The primary labels layer together with its load and palette-resolution
        information.
    """
    table = get_table(sdata, table_name)
    require_compatible_spatial_annotation_column(table, column_name)

    load_result = viewer_adapter.ensure_labels_loaded(
        sdata,
        labels_name,
        coordinate_system,
    )
    style_result = apply_table_color_source_to_labels_layer(
        load_result.layer,
        sdata=sdata,
        labels_name=labels_name,
        style_spec=TableColorSourceSpec(
            table_name=table_name,
            source_kind="obs_column",
            value_key=column_name,
            value_kind="categorical",
        ),
    )
    viewer_adapter.activate_layer(load_result.layer)

    return LabelsLoadResult(
        layer=load_result.layer,
        created=load_result.created,
        value_kind=style_result.value_kind,
        palette_source=style_result.palette_source,
        coercion_applied=style_result.coercion_applied,
    )
