from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from spatialdata import get_element_annotators
from spatialdata.models import TableModel, get_axes_names
from spatialdata.transformations import get_transformation
from xarray import DataArray

from spatiato.core.feature_matrix_metadata import RESERVED_OBSM_KEYS

if TYPE_CHECKING:
    from anndata import AnnData
    from spatialdata import SpatialData


@dataclass(frozen=True)
class SpatialDataLabelsOption:
    """A selectable labels element discovered from a loaded SpatialData object."""

    labels_name: str
    # User-facing text for the dropdown. This may include the dataset name to
    # disambiguate equal labels element names coming from different SpatialData objects.
    display_name: str
    sdata: SpatialData
    # Selectable coordinate systems for this labels element in the current
    # discovery context.
    coordinate_systems: tuple[str, ...]

    @property
    def identity(self) -> tuple[int, str]:
        """Return a stable identity for preserving widget selection across refreshes."""
        return (id(self.sdata), self.labels_name)


@dataclass(frozen=True)
class SpatialDataImageOption:
    """A selectable image element discovered from a loaded SpatialData object."""

    image_name: str
    # User-facing text for the dropdown. This may include the dataset name to
    # disambiguate equal image names coming from different SpatialData objects.
    display_name: str
    sdata: SpatialData
    # Selectable coordinate systems for this image in the current discovery
    # context. When filtered by labels, these are the shared coordinate systems
    # between the selected labels and image elements.
    coordinate_systems: tuple[str, ...]

    @property
    def identity(self) -> tuple[int, str]:
        """Return a stable identity for preserving widget selection across refreshes."""
        return (id(self.sdata), self.image_name)


@dataclass(frozen=True)
class SpatialDataShapesOption:
    """A selectable shapes element discovered from a loaded SpatialData object."""

    shapes_name: str
    display_name: str
    sdata: SpatialData
    coordinate_systems: tuple[str, ...]

    @property
    def identity(self) -> tuple[int, str]:
        """Return a stable identity for preserving widget selection across refreshes."""
        return (id(self.sdata), self.shapes_name)


@dataclass(frozen=True)
class SpatialDataPointsOption:
    """A selectable points element discovered from a loaded SpatialData object."""

    points_name: str
    display_name: str
    sdata: SpatialData
    coordinate_systems: tuple[str, ...]

    @property
    def identity(self) -> tuple[int, str]:
        """Return a stable identity for preserving widget selection across refreshes."""
        return (id(self.sdata), self.points_name)


@dataclass(frozen=True)
class SpatialDataFeatureExtractionLabelDiscovery:
    """Feature-extraction label discovery summary for one coordinate system."""

    coordinate_system: str
    eligible_label_options: list[SpatialDataLabelsOption]
    coordinate_system_labels_count: int
    unavailable_label_count: int


@dataclass(frozen=True)
class SpatialDataFeatureExtractionImageDiscovery:
    """Feature-extraction image discovery summary for one `(coordinate system, label)` pair."""

    coordinate_system: str
    labels_name: str
    eligible_image_options: list[SpatialDataImageOption]
    coordinate_system_image_count: int
    unavailable_image_count: int


@dataclass(frozen=True)
class SpatialDataTableMetadata:
    """Metadata that links a table to the spatial elements it annotates."""

    table_name: str
    region_key: str
    instance_key: str
    regions: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.regions or any(not region for region in self.regions):
            raise ValueError(f"Table `{self.table_name}` must declare at least one non-empty region.")
        if len(set(self.regions)) != len(self.regions):
            raise ValueError(f"Table `{self.table_name}` declares duplicate regions.")

    def annotates(self, element_name: str) -> bool:
        """Return whether this table can annotate the given spatial element."""
        return element_name in self.regions


def get_annotating_table_names(sdata: SpatialData, element_name: str) -> list[str]:
    """Return the table names that annotate a spatial element in a SpatialData object."""
    return sorted(get_element_annotators(sdata, element_name))


def get_table(sdata: SpatialData, table_name: str) -> AnnData:
    """Return a validated annotating table."""
    table = sdata[table_name]
    normalize_table_metadata(table)
    return TableModel.validate(table)


def normalize_table_metadata(table: AnnData) -> AnnData:
    """Normalize known SpatialData table attrs into validator-friendly types."""
    _normalize_table_model_attrs(table)
    return table


def get_table_metadata(sdata: SpatialData, table_name: str) -> SpatialDataTableMetadata:
    """Read linkage metadata without modifying or validating the table."""
    table = sdata[table_name]
    attrs = _get_table_model_attrs(table, table_name)

    return SpatialDataTableMetadata(
        table_name=table_name,
        region_key=_normalize_required_scalar_string_attr_value(
            attrs[TableModel.REGION_KEY_KEY],
            table_name=table_name,
            key=TableModel.REGION_KEY_KEY,
        ),
        instance_key=_normalize_required_scalar_string_attr_value(
            attrs[TableModel.INSTANCE_KEY],
            table_name=table_name,
            key=TableModel.INSTANCE_KEY,
        ),
        regions=_normalize_regions(attrs.get(TableModel.REGION_KEY)),
    )


def get_table_annotated_labels_names(sdata: SpatialData, table_name: str) -> list[str]:
    """Return the labels regions a table declares, validated against available labels."""
    table_metadata = get_table_metadata(sdata, table_name)
    available_labels_names = set(_get_labels_names(sdata))
    annotated_labels_names = sorted(set(table_metadata.regions))

    missing_regions = [
        labels_name for labels_name in annotated_labels_names if labels_name not in available_labels_names
    ]
    if missing_regions:
        missing = _format_name_list(missing_regions)
        raise ValueError(
            f"Table `{table_name}` declares labels element(s) {missing}, "
            "but no matching labels element exists in the selected SpatialData object."
        )

    return annotated_labels_names


def validate_table_annotation_coverage(
    sdata: SpatialData,
    table_name: str,
    labels_names: Sequence[str],
) -> SpatialDataTableMetadata:
    """Validate that a table annotates every requested labels region."""
    table_metadata = get_table_metadata(sdata, table_name)
    requested_labels_names = _normalize_requested_labels_names(labels_names)
    available_labels_names = set(_get_labels_names(sdata))

    invalid_regions = [
        labels_name for labels_name in requested_labels_names if labels_name not in available_labels_names
    ]
    if invalid_regions:
        invalid = _format_name_list(invalid_regions)
        raise ValueError(f"Labels element(s) {invalid} are not available in the selected SpatialData object.")

    annotated_labels_names = set(get_table_annotated_labels_names(sdata, table_name))
    missing_regions = [
        labels_name for labels_name in requested_labels_names if labels_name not in annotated_labels_names
    ]
    if missing_regions:
        missing = _format_name_list(missing_regions)
        raise ValueError(f"Table `{table_name}` does not annotate labels element(s) {missing}.")

    return table_metadata


def validate_table_region_instance_ids(
    sdata: SpatialData,
    table_name: str,
    *,
    labels_names: Sequence[str] | None = None,
) -> SpatialDataTableMetadata:
    """Validate per-region `instance_key` uniqueness for one table."""
    table = get_table(sdata, table_name)
    table_metadata = get_table_metadata(sdata, table_name)
    _validate_table_obs_region_and_instance_key_columns(table, table_metadata)

    if labels_names is None:
        regions_to_check = get_table_annotated_labels_names(sdata, table_name)
    else:
        validate_table_annotation_coverage(sdata, table_name, labels_names)
        regions_to_check = _normalize_requested_labels_names(labels_names)

    for labels_name in regions_to_check:
        duplicate_instances = _get_duplicate_region_instances(table, table_metadata, labels_name)
        if not duplicate_instances:
            continue

        preview = _format_duplicate_preview(duplicate_instances)
        raise ValueError(
            f"Table `{table_name}` cannot annotate labels element `{labels_name}` because "
            f"`{table_metadata.instance_key}` contains duplicate values within that region: {preview}."
        )

    return table_metadata


def validate_table_binding(sdata: SpatialData, labels_name: str, table_name: str) -> SpatialDataTableMetadata:
    """Validate that a table can be safely bound to a selected labels element."""
    table = get_table(sdata, table_name)
    table_metadata = validate_table_annotation_coverage(sdata, table_name, [labels_name])
    _validate_table_obs_region_and_instance_key_columns(table, table_metadata)

    duplicate_instances = _get_duplicate_region_instances(table, table_metadata, labels_name)
    if duplicate_instances:
        preview = _format_duplicate_preview(duplicate_instances)
        raise ValueError(
            f"Table `{table_name}` cannot be bound to labels element `{labels_name}` because `{table_metadata.instance_key}` "
            f"contains duplicate values within that region: {preview}."
        )

    return table_metadata


def get_table_obsm_keys(sdata: SpatialData, table_name: str) -> list[str]:
    """Return the `adata.obsm` keys of a table that can hold a feature matrix, sorted.

    The keys reserved for spatial coordinates (`RESERVED_OBSM_KEYS`) are left out.
    """
    table = get_table(sdata, table_name)
    return sorted(key for key in table.obsm.keys() if key not in RESERVED_OBSM_KEYS)


def get_table_var_matrix(table: AnnData, layer: str | None) -> Any:
    """Return the matrix a var source reads, as attached: `X` for `layer=None`, else `layers[layer]`.

    No value is read here, so a lazy or backed matrix stays lazy.
    """
    if layer is None:
        if table.X is None:
            raise ValueError("The selected table has no `X`.")
        return table.X
    if layer not in table.layers:
        raise ValueError(f'Layer "{layer}" is not available in the selected table.')
    return table.layers[layer]


def get_spatialdata_labels_options_from_sdata(sdata: SpatialData) -> list[SpatialDataLabelsOption]:
    """Return selectable labels elements directly from a loaded SpatialData object."""
    return [
        SpatialDataLabelsOption(
            labels_name=labels_name,
            display_name=labels_name,
            sdata=sdata,
            coordinate_systems=_get_element_coordinate_systems(sdata.labels[labels_name]),
        )
        for labels_name in _get_labels_names(sdata)
    ]


def get_spatialdata_shapes_options_from_sdata(sdata: SpatialData) -> list[SpatialDataShapesOption]:
    """Return selectable shapes elements directly from a loaded SpatialData object."""
    return [
        SpatialDataShapesOption(
            shapes_name=shapes_name,
            display_name=shapes_name,
            sdata=sdata,
            coordinate_systems=_get_element_coordinate_systems(sdata.shapes[shapes_name]),
        )
        for shapes_name in _get_shapes_names(sdata)
    ]


def get_spatialdata_points_options_from_sdata(sdata: SpatialData) -> list[SpatialDataPointsOption]:
    """Return selectable points elements directly from a loaded SpatialData object."""
    return [
        SpatialDataPointsOption(
            points_name=points_name,
            display_name=points_name,
            sdata=sdata,
            coordinate_systems=_get_element_coordinate_systems(sdata.points[points_name]),
        )
        for points_name in _get_points_names(sdata)
    ]


def get_coordinate_system_names_from_sdata(sdata: SpatialData) -> list[str]:
    """Return all coordinate-system names exposed by spatial elements in a loaded `SpatialData`."""
    coordinate_systems: set[str] = set()

    for labels_name in _get_labels_names(sdata):
        coordinate_systems.update(_get_element_coordinate_systems(sdata.labels[labels_name]))

    for image_name in _get_image_names(sdata):
        coordinate_systems.update(_get_element_coordinate_systems(sdata.images[image_name]))

    for shapes_name in _get_shapes_names(sdata):
        coordinate_systems.update(_get_element_coordinate_systems(sdata.shapes[shapes_name]))

    for points_name in _get_points_names(sdata):
        coordinate_systems.update(_get_element_coordinate_systems(sdata.points[points_name]))

    return sorted(coordinate_systems)


def get_image_channel_names_from_sdata(sdata: SpatialData, image_name: str) -> list[str]:
    """Return channel names for an image element, or an empty list without a channel axis."""
    if image_name not in _get_image_names(sdata):
        raise ValueError(f"Image element `{image_name}` is not available in the selected SpatialData object.")

    image_element = sdata.images[image_name]
    axes = get_axes_names(image_element)
    if "c" not in axes:
        return []

    if isinstance(image_element, DataArray):
        channel_values = list(image_element.coords.indexes["c"])
    else:
        scale0 = next(iter(image_element["scale0"].values()))
        channel_values = list(scale0.coords.indexes["c"])

    channel_names = [str(channel_value) for channel_value in channel_values]
    duplicates: list[str] = []
    seen: set[str] = set()
    duplicate_seen: set[str] = set()
    for channel_name in channel_names:
        if channel_name in seen and channel_name not in duplicate_seen:
            duplicates.append(channel_name)
            duplicate_seen.add(channel_name)
        seen.add(channel_name)

    if duplicates:
        duplicate_names = ", ".join(f"`{channel_name}`" for channel_name in duplicates)
        raise ValueError(
            f"Image element `{image_name}` exposes duplicate channel names ({duplicate_names}), "
            "which spatiato does not support. "
            "Update the channel names in the SpatialData object with "
            "`sdata.set_channel_names(...)`."
        )

    return channel_names


def get_spatialdata_labels_options_for_coordinate_system_from_sdata(
    *,
    sdata: SpatialData,
    coordinate_system: str,
) -> list[SpatialDataLabelsOption]:
    """Return labels options restricted to a selected coordinate system."""
    return [
        SpatialDataLabelsOption(
            labels_name=labels_name,
            display_name=labels_name,
            sdata=sdata,
            coordinate_systems=_get_element_coordinate_systems(sdata.labels[labels_name]),
        )
        for labels_name in _get_labels_names(sdata)
        if coordinate_system in _get_element_coordinate_systems(sdata.labels[labels_name])
    ]


def get_spatialdata_feature_extraction_label_discovery_for_coordinate_system_from_sdata(
    *,
    sdata: SpatialData,
    coordinate_system: str,
) -> SpatialDataFeatureExtractionLabelDiscovery:
    """Return feature-extraction label discovery summary for one coordinate system."""
    coordinate_system_labels_names = [
        labels_name
        for labels_name in _get_labels_names(sdata)
        if coordinate_system in _get_element_coordinate_systems(sdata.labels[labels_name])
    ]
    eligible_label_options = [
        SpatialDataLabelsOption(
            labels_name=labels_name,
            display_name=labels_name,
            sdata=sdata,
            coordinate_systems=_get_element_coordinate_systems(sdata.labels[labels_name]),
        )
        for labels_name in coordinate_system_labels_names
        if _is_feature_extraction_transform_supported(sdata.labels[labels_name], coordinate_system)
    ]
    coordinate_system_labels_count = len(coordinate_system_labels_names)
    return SpatialDataFeatureExtractionLabelDiscovery(
        coordinate_system=coordinate_system,
        eligible_label_options=eligible_label_options,
        coordinate_system_labels_count=coordinate_system_labels_count,
        unavailable_label_count=coordinate_system_labels_count - len(eligible_label_options),
    )


def get_spatialdata_image_options_for_coordinate_system_from_sdata(
    *,
    sdata: SpatialData,
    coordinate_system: str,
) -> list[SpatialDataImageOption]:
    """Return image options restricted to a selected coordinate system."""
    return [
        SpatialDataImageOption(
            image_name=image_name,
            display_name=image_name,
            sdata=sdata,
            coordinate_systems=_get_element_coordinate_systems(sdata.images[image_name]),
        )
        for image_name in _get_image_names(sdata)
        if coordinate_system in _get_element_coordinate_systems(sdata.images[image_name])
    ]


def get_spatialdata_shapes_options_for_coordinate_system_from_sdata(
    *,
    sdata: SpatialData,
    coordinate_system: str,
) -> list[SpatialDataShapesOption]:
    """Return shapes options restricted to a selected coordinate system."""
    return [
        SpatialDataShapesOption(
            shapes_name=shapes_name,
            display_name=shapes_name,
            sdata=sdata,
            coordinate_systems=_get_element_coordinate_systems(sdata.shapes[shapes_name]),
        )
        for shapes_name in _get_shapes_names(sdata)
        if coordinate_system in _get_element_coordinate_systems(sdata.shapes[shapes_name])
    ]


def get_spatialdata_points_options_for_coordinate_system_from_sdata(
    *,
    sdata: SpatialData,
    coordinate_system: str,
) -> list[SpatialDataPointsOption]:
    """Return points options restricted to a selected coordinate system."""
    return [
        SpatialDataPointsOption(
            points_name=points_name,
            display_name=points_name,
            sdata=sdata,
            coordinate_systems=_get_element_coordinate_systems(sdata.points[points_name]),
        )
        for points_name in _get_points_names(sdata)
        if coordinate_system in _get_element_coordinate_systems(sdata.points[points_name])
    ]


def get_spatialdata_feature_extraction_image_discovery_for_coordinate_system_and_label_from_sdata(
    *,
    sdata: SpatialData,
    coordinate_system: str,
    labels_name: str,
) -> SpatialDataFeatureExtractionImageDiscovery:
    """Return feature-extraction image discovery summary for one `(coordinate system, label)` pair."""
    available_labels_names = _get_labels_names(sdata)
    if labels_name not in available_labels_names:
        raise ValueError(f"Labels element `{labels_name}` is not available in the selected SpatialData object.")

    label_element = sdata.labels[labels_name]
    if coordinate_system not in _get_element_coordinate_systems(label_element):
        raise ValueError(f"Labels element `{labels_name}` is not available in coordinate system `{coordinate_system}`.")

    label_shape = _get_spatial_shape(label_element)
    label_affine = _get_feature_extraction_affine_matrix(label_element, coordinate_system)
    coordinate_system_image_names = [
        image_name
        for image_name in _get_image_names(sdata)
        if coordinate_system in _get_element_coordinate_systems(sdata.images[image_name])
    ]

    matches: list[SpatialDataImageOption] = []
    if label_affine is not None and label_shape:
        for image_name in coordinate_system_image_names:
            image_element = sdata.images[image_name]
            image_shape = _get_spatial_shape(image_element)
            if image_shape != label_shape:
                continue

            image_affine = _get_feature_extraction_affine_matrix(image_element, coordinate_system)
            if image_affine is None or not np.allclose(image_affine, label_affine):
                continue

            matches.append(
                SpatialDataImageOption(
                    image_name=image_name,
                    display_name=image_name,
                    sdata=sdata,
                    coordinate_systems=(coordinate_system,),
                )
            )

    coordinate_system_image_count = len(coordinate_system_image_names)
    return SpatialDataFeatureExtractionImageDiscovery(
        coordinate_system=coordinate_system,
        labels_name=labels_name,
        eligible_image_options=matches,
        coordinate_system_image_count=coordinate_system_image_count,
        unavailable_image_count=coordinate_system_image_count - len(matches),
    )


def _get_table_model_attrs(table: AnnData, table_name: str) -> dict[str, Any]:
    attrs = table.uns.get(TableModel.ATTRS_KEY)
    if not isinstance(attrs, dict):
        raise ValueError(f"Table `{table_name}` is missing `{TableModel.ATTRS_KEY}` metadata.")

    required_keys = (TableModel.REGION_KEY_KEY, TableModel.INSTANCE_KEY)
    missing_keys = [key for key in required_keys if key not in attrs]
    if missing_keys:
        missing = ", ".join(f"`{key}`" for key in missing_keys)
        raise ValueError(f"Table `{table_name}` is missing required SpatialData metadata: {missing}.")

    return attrs


def _normalize_table_model_attrs(table: AnnData) -> None:
    attrs = table.uns.get(TableModel.ATTRS_KEY)
    if not isinstance(attrs, dict):
        return

    if TableModel.REGION_KEY in attrs:
        attrs[TableModel.REGION_KEY] = _normalize_region_attr_value(attrs[TableModel.REGION_KEY])

    for key in (TableModel.REGION_KEY_KEY, TableModel.INSTANCE_KEY):
        if key in attrs:
            attrs[key] = _normalize_scalar_string_attr_value(attrs[key])


def _normalize_regions(region: Any) -> tuple[str, ...]:
    if region is None:
        return ()

    if isinstance(region, str):
        return (region,)

    return tuple(_flatten_string_values(region))


def _normalize_region_attr_value(region: Any) -> str | list[str] | None:
    if region is None or isinstance(region, str):
        return region

    return _flatten_string_values(region)


def _normalize_scalar_string_attr_value(value: Any) -> Any:
    if value is None or isinstance(value, str):
        return value

    flattened = _flatten_string_values(value)
    if len(flattened) == 1:
        return flattened[0]

    return value


def _normalize_required_scalar_string_attr_value(value: Any, *, table_name: str, key: str) -> str:
    normalized = _normalize_scalar_string_attr_value(value)
    if not isinstance(normalized, str) or not normalized:
        raise ValueError(f"Table `{table_name}` metadata `{key}` must contain exactly one non-empty string.")
    return normalized


def _flatten_string_values(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, str):
        return [value]

    if isinstance(value, bytes):
        return [value.decode()]

    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _flatten_string_values(value.item())

        flattened: list[str] = []
        for item in value.reshape(-1).tolist():
            flattened.extend(_flatten_string_values(item))
        return flattened

    if isinstance(value, np.generic):
        return [str(value.item())]

    if isinstance(value, Sequence):
        flattened: list[str] = []
        for item in value:
            flattened.extend(_flatten_string_values(item))
        return flattened

    return [str(value)]


def _is_pickable_labels_layer(layer: Any) -> bool:
    from napari.layers import Labels

    events = getattr(layer, "events", None)
    return isinstance(layer, Labels) and getattr(events, "selected_label", None) is not None


def _get_labels_names(sdata: SpatialData) -> list[str]:
    labels = getattr(sdata, "labels", {})
    return sorted(labels.keys())


def _get_image_names(sdata: SpatialData) -> list[str]:
    images = getattr(sdata, "images", {})
    return sorted(images.keys())


def _get_shapes_names(sdata: SpatialData) -> list[str]:
    shapes = getattr(sdata, "shapes", {})
    return sorted(shapes.keys())


def _get_points_names(sdata: SpatialData) -> list[str]:
    points = getattr(sdata, "points", {})
    return sorted(points.keys())


def _normalize_requested_labels_names(labels_names: Sequence[str]) -> list[str]:
    return sorted({str(labels_name) for labels_name in labels_names})


def _format_name_list(names: Sequence[str]) -> str:
    return ", ".join(f"`{name}`" for name in names)


def _format_duplicate_preview(duplicate_labels: Sequence[Any]) -> str:
    normalized_labels = [str(label) for label in duplicate_labels[:5]]
    preview = ", ".join(normalized_labels)
    if len(duplicate_labels) > 5:
        preview += ", ..."
    return preview


def _validate_table_obs_region_and_instance_key_columns(
    table: AnnData,
    table_metadata: SpatialDataTableMetadata,
) -> None:
    if table_metadata.region_key not in table.obs.columns:
        raise ValueError(
            f"Table `{table_metadata.table_name}` is missing required obs column `{table_metadata.region_key}`."
        )

    if table_metadata.instance_key not in table.obs.columns:
        raise ValueError(
            f"Table `{table_metadata.table_name}` is missing required obs column `{table_metadata.instance_key}`."
        )


def _get_duplicate_region_instances(
    table: AnnData,
    table_metadata: SpatialDataTableMetadata,
    element_name: str,
) -> list[Any]:
    region_rows = table.obs.loc[table.obs[table_metadata.region_key] == element_name]
    if region_rows.empty:
        return []

    region_instances = region_rows[table_metadata.instance_key]
    duplicate_instances = region_instances[region_instances.duplicated(keep=False)]
    if duplicate_instances.empty:
        return []

    return duplicate_instances.drop_duplicates().tolist()


def _get_reference_array(element: Any) -> Any:
    if isinstance(element, DataArray):
        return element

    try:
        scale0 = element["scale0"]
    except Exception:  # noqa: BLE001
        return element

    try:
        return next(iter(scale0.values()))
    except Exception:  # noqa: BLE001
        return element


def _get_spatial_axes(element: Any) -> tuple[str, ...]:
    axes = set(get_axes_names(_get_reference_array(element)))
    if "x" not in axes or "y" not in axes:
        return ()

    return tuple(axis for axis in ("z", "y", "x") if axis in axes)


def _get_spatial_shape(element: Any) -> tuple[int, ...]:
    reference = _get_reference_array(element)
    axes = get_axes_names(reference)
    shape = getattr(reference, "shape", None)
    if shape is None:
        return ()

    size_by_axis = {axis: int(size) for axis, size in zip(axes, shape, strict=False) if axis in {"z", "y", "x"}}
    return tuple(size_by_axis[axis] for axis in ("z", "y", "x") if axis in size_by_axis)


def _get_feature_extraction_affine_matrix(element: Any, coordinate_system: str) -> np.ndarray | None:
    spatial_axes = _get_spatial_axes(element)
    if not spatial_axes:
        return None

    transform = get_transformation(element, get_all=True).get(coordinate_system)
    if transform is None:
        return None

    try:
        matrix = np.asarray(transform.to_affine_matrix(input_axes=spatial_axes, output_axes=spatial_axes), dtype=float)
    except Exception:  # noqa: BLE001
        return None

    if not _is_pure_xy_translation_affine(matrix, spatial_axes):
        return None

    return matrix


def _is_feature_extraction_transform_supported(element: Any, coordinate_system: str) -> bool:
    return _get_feature_extraction_affine_matrix(element, coordinate_system) is not None


def _is_pure_xy_translation_affine(matrix: np.ndarray, spatial_axes: Sequence[str]) -> bool:
    axis_count = len(spatial_axes)
    if matrix.shape != (axis_count + 1, axis_count + 1):
        return False

    linear = matrix[:-1, :-1]
    if not np.allclose(linear, np.eye(axis_count)):
        return False

    expected_last_row = np.zeros(axis_count + 1, dtype=float)
    expected_last_row[-1] = 1.0
    if not np.allclose(matrix[-1], expected_last_row):
        return False

    for index, axis in enumerate(spatial_axes):
        if axis not in {"x", "y"} and not np.allclose(matrix[index, -1], 0.0):
            return False

    return True


def _get_element_coordinate_systems(element: Any) -> tuple[str, ...]:
    return tuple(sorted(get_transformation(element, get_all=True).keys()))
