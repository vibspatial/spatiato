"""Primary-label viewer styling used by object classification."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba

from spatiato.core.class_palette import (
    DEFAULT_NEUTRAL_COLOR,
    default_labeled_class_color,
    normalize_class_values,
    resolve_table_categorical_palette,
)
from spatiato.core.object_classification.annotation import (
    USER_CLASS_COLUMN,
)
from spatiato.core.spatialdata import (
    SpatialDataTableMetadata,
    get_table,
    get_table_metadata,
)
from spatiato.viewer._styling import MISSING_CONTINUOUS_COLOR
from spatiato.viewer.adapter import ViewerAdapter
from spatiato.viewer.labels_colormap import (
    _TRANSPARENT_RGBA,
    CompactLabelColormap,
    compact_categorical_label_colormap_from_values,
    compact_continuous_label_colormap_from_values,
)
from spatiato.viewer.labels_styling import _build_labels_features, _get_region_rows_by_instance
from spatiato.widgets.object_classification.controller import (
    PRED_CLASS_COLUMN,
    PRED_CONFIDENCE_COLUMN,
)

if TYPE_CHECKING:
    from anndata import AnnData
    from spatialdata import SpatialData

    from spatiato.widgets.object_classification.annotation_controller import UserClassAnnotationChange

COLOR_BY_USER_CLASS = USER_CLASS_COLUMN
COLOR_BY_PRED_CLASS = PRED_CLASS_COLUMN
COLOR_BY_PRED_CONFIDENCE = PRED_CONFIDENCE_COLUMN
COLOR_BY_OPTIONS = (
    COLOR_BY_USER_CLASS,
    COLOR_BY_PRED_CLASS,
    COLOR_BY_PRED_CONFIDENCE,
)

PRED_CONFIDENCE_COLORMAP = "plasma"


class ClassStateError(ValueError):
    """Raised when Object Classification class state is invalid for styling."""


class ViewerStylingController:
    """Manage labels-layer styling from user labels and classifier outputs."""

    def __init__(self, viewer_adapter: ViewerAdapter) -> None:
        self._viewer_adapter = viewer_adapter
        self._labels_layer: Any | None = None
        self._selected_spatialdata: SpatialData | None = None
        self._selected_labels_name: str | None = None
        self._selected_coordinate_system: str | None = None
        self._selected_table_name: str | None = None
        self._selected_table_metadata: SpatialDataTableMetadata | None = None
        self._color_by = COLOR_BY_USER_CLASS

    @property
    def color_by(self) -> str:
        """Return the current labels-layer coloring mode."""
        return self._color_by

    @property
    def labels_layer(self) -> Any | None:
        """Return the currently styled labels layer, if any."""
        return self._labels_layer

    def bind(
        self,
        sdata: SpatialData | None,
        labels_name: str | None,
        table_name: str | None,
        coordinate_system: str | None = None,
    ) -> None:
        """Bind styling to the selected labels layer and annotation table."""
        next_layer = None
        if sdata is not None and labels_name is not None:
            next_layer = self._viewer_adapter.get_loaded_primary_labels_layer(
                sdata,
                labels_name,
                coordinate_system,
            )

        next_table_metadata = None
        if sdata is not None and table_name is not None:
            next_table_metadata = get_table_metadata(sdata, table_name)

        self._labels_layer = next_layer
        self._selected_spatialdata = sdata
        self._selected_labels_name = labels_name
        self._selected_coordinate_system = coordinate_system
        self._selected_table_name = table_name
        self._selected_table_metadata = next_table_metadata

    def set_color_by(self, color_by: str) -> None:
        """Set the active coloring mode for the bound labels layer."""
        if color_by not in COLOR_BY_OPTIONS:
            raise ValueError(f"Unsupported color mode `{color_by}`.")

        self._color_by = color_by

    def refresh(self) -> None:
        """Refresh labels-layer colors and features from the current table state."""
        if self._labels_layer is None:
            return

        feature_rows = self._get_region_feature_rows()
        self.refresh_layer_colors(feature_rows=feature_rows)
        self.refresh_layer_features(feature_rows=feature_rows)

    def refresh_layer_colors(self, *, feature_rows: pd.DataFrame | None = None) -> None:
        """Apply the current `color_by` mode to the bound labels layer.

        Direct annotation happy paths should use row-scoped refresh helpers
        instead. Prediction color repainting should reach this full refresh path
        when the classifier actually writes predictions, via
        `ObjectClassificationWidget._on_classifier_prediction_state_changed()`.
        """
        if self._labels_layer is None:
            return

        if feature_rows is None:
            feature_rows = self._get_region_feature_rows()

        default_color = self._get_not_in_table_color()
        if self._color_by == COLOR_BY_PRED_CONFIDENCE:
            self._labels_layer.colormap = compact_continuous_label_colormap_from_values(
                feature_rows[PRED_CONFIDENCE_COLUMN],
                colormap_name=PRED_CONFIDENCE_COLORMAP,
                missing_color=MISSING_CONTINUOUS_COLOR,
                default_color=default_color,
                value_range=(0.0, 1.0),
            )
        else:
            class_values_by_instance = feature_rows[self._color_by]
            category_column = USER_CLASS_COLUMN
            if self._color_by == COLOR_BY_PRED_CLASS:
                category_column = PRED_CLASS_COLUMN

            # Object-classification values are class ids, not colors. Resolve
            # them through the class palette stored in table `.uns`; generic
            # styled-labels coloring intentionally uses a separate color path.
            class_color_lookup = self._get_class_color_lookup(
                category_column=category_column,
                observed_class_values=class_values_by_instance,
            )

            categories = list(class_color_lookup)
            class_values = pd.Series(
                pd.Categorical(
                    class_values_by_instance,
                    categories=categories,
                ),
                index=class_values_by_instance.index,
                name=class_values_by_instance.name,
            )
            self._labels_layer.colormap = compact_categorical_label_colormap_from_values(
                class_values,
                categories=categories,
                palette=[class_color_lookup[class_id] for class_id in categories],
                default_color=default_color,
                missing_color=DEFAULT_NEUTRAL_COLOR,
                background_value=0,
            )
        self._viewer_adapter.sync_labels_display_after_colormap_change(self._labels_layer)

    def refresh_layer_features(self, *, feature_rows: pd.DataFrame | None = None) -> None:
        """Expose current label and prediction values as napari layer features."""
        if self._labels_layer is None:
            return

        if feature_rows is None:
            feature_rows = self._get_region_feature_rows()

        instance_key = "instance_id"  # defensive fallback for the no metadata case.
        if self._selected_table_metadata is not None:
            instance_key = self._selected_table_metadata.instance_key
        self._labels_layer.features = _build_labels_features(
            feature_rows,
            instance_key=instance_key,
        )

    def refresh_user_class_colormap_and_feature(self, change: UserClassAnnotationChange) -> bool:
        """Refresh one user-class annotation in labels colors and features.

        Returns ``True`` when the row-scoped update was fully applied. Returns
        ``False`` when the caller should fall back to a normal full refresh.
        """
        if self._labels_layer is None or self._color_by != COLOR_BY_USER_CLASS:
            return False

        feature_rows = self._build_user_class_annotation_features(change)
        if feature_rows is None:
            return False

        self._refresh_compact_user_class_colormap_and_feature(change, feature_rows)
        return True

    def _refresh_compact_user_class_colormap_and_feature(
        self,
        change: UserClassAnnotationChange,
        feature_rows: pd.DataFrame,
    ) -> bool:
        colormap = getattr(self._labels_layer, "colormap", None)
        if not isinstance(colormap, CompactLabelColormap):
            raise RuntimeError(
                "Cannot update user-class annotation colors row-scoped: "
                "the labels layer is not using CompactLabelColormap."
            )

        refresh = self._labels_layer.refresh

        instance_id = int(change.instance_id)
        class_id = change.class_id
        if class_id is None:
            # Clearing the class does not remove the object from the table: its row
            # stays, with an empty `user_class`. Color it as unlabeled (neutral), not
            # as an object without a table row (transparent).
            result = colormap.set_label_missing(instance_id)
        else:
            class_id = int(class_id)
            class_color_lookup = self._get_valid_user_class_color_lookup()
            if class_color_lookup is None:
                raise RuntimeError("Cannot update compact user-class coloring without valid user-class colors.")
            class_color = class_color_lookup.get(class_id)
            if class_color is None:
                raise RuntimeError(f"Cannot update compact user-class coloring: class `{class_id}` has no color.")
            result = colormap.set_label_value(instance_id, class_id, value_color=class_color)

        self._labels_layer.features = feature_rows
        if result.texture_table_changed:
            # Only brand-new classes append a new texture-code -> RGBA row.
            # Notify vispy to upload the expanded lookup texture. Existing-
            # class edits reuse an already uploaded texture row, so they only
            # need the layer refresh below.
            self._labels_layer.events.colormap()
        # The compact mapping was mutated in place; repaint the layer without
        # asking napari to recompute the layer extent.
        refresh(extent=False)
        return True

    def refresh_user_class_feature_only(self, change: UserClassAnnotationChange) -> bool:
        """Refresh one user-class feature value without repainting label colors.

        This is the direct-annotation fast path for prediction color modes:
        annotation changes `user_class`, while `pred_class`/`pred_confidence`
        colors are refreshed only when the classifier writes predictions.
        """
        if self._labels_layer is None:
            return False

        feature_rows = self._build_user_class_annotation_features(change)
        if feature_rows is None:
            return False

        self._labels_layer.features = feature_rows
        return True

    def _build_user_class_annotation_features(
        self,
        change: UserClassAnnotationChange,
    ) -> pd.DataFrame | None:
        features = getattr(self._labels_layer, "features", None)
        if not isinstance(features, pd.DataFrame) or features.empty:
            return None
        if "index" not in features or USER_CLASS_COLUMN not in features:
            return None

        feature_index = pd.to_numeric(features["index"], errors="coerce")
        matching_rows = feature_index == int(change.instance_id)
        if int(matching_rows.sum()) != 1:
            return None

        updated_features = features.copy()
        updated_features.loc[matching_rows, USER_CLASS_COLUMN] = (
            pd.NA if change.class_id is None else int(change.class_id)
        )
        return updated_features

    def _get_valid_user_class_color_lookup(self) -> dict[int, np.ndarray] | None:
        table = self._get_bound_table()
        if table is None or USER_CLASS_COLUMN not in table.obs:
            return None

        return self._get_class_color_lookup(
            category_column=USER_CLASS_COLUMN,
        )

    def _get_bound_table(self) -> AnnData | None:
        if self._selected_spatialdata is None or self._selected_table_name is None:
            return None

        return get_table(self._selected_spatialdata, self._selected_table_name)

    def _get_not_in_table_color(self) -> Any:
        """Return the color for labels without a row in the bound table.

        Those labels are transparent, matching generic styled-labels coloring.
        Without a bound table there are no rows to contrast with, so every
        foreground label stays neutral instead of disappearing.
        """
        if self._selected_table_metadata is None or self._get_bound_table() is None:
            return DEFAULT_NEUTRAL_COLOR
        return _TRANSPARENT_RGBA

    def _get_region_rows_by_instance(self) -> pd.DataFrame:
        table = self._get_bound_table()
        metadata = self._selected_table_metadata
        if table is None or metadata is None or self._selected_labels_name is None:
            return pd.DataFrame(index=pd.Index([], dtype="int64", name="index"))

        region_rows, _ = _get_region_rows_by_instance(table, metadata, self._selected_labels_name)
        return region_rows

    def _get_region_feature_rows(self) -> pd.DataFrame:
        """Return normalized labels features for the selected segmentation region.

        The returned rows are indexed by label/instance id and include
        `user_class`, `pred_class`, and `pred_confidence`. This is scoped to the
        currently selected labels element, not necessarily the complete table.
        """
        region_rows = self._get_region_rows_by_instance()
        feature_rows = pd.DataFrame(index=region_rows.index.astype("int64", copy=False))
        feature_rows.index.name = "index"

        if USER_CLASS_COLUMN in region_rows:
            feature_rows[USER_CLASS_COLUMN] = normalize_class_values(
                region_rows[USER_CLASS_COLUMN],
                column_name=USER_CLASS_COLUMN,
            )
        else:
            feature_rows[USER_CLASS_COLUMN] = pd.Series(
                pd.NA,
                index=feature_rows.index,
                dtype="Int64",
            )

        if PRED_CLASS_COLUMN in region_rows:
            feature_rows[PRED_CLASS_COLUMN] = normalize_class_values(
                region_rows[PRED_CLASS_COLUMN],
                column_name=PRED_CLASS_COLUMN,
            )
        else:
            feature_rows[PRED_CLASS_COLUMN] = pd.Series(
                pd.NA,
                index=feature_rows.index,
                dtype="Int64",
            )

        if PRED_CONFIDENCE_COLUMN in region_rows:
            feature_rows[PRED_CONFIDENCE_COLUMN] = _to_numeric_values(
                region_rows[PRED_CONFIDENCE_COLUMN],
                PRED_CONFIDENCE_COLUMN,
            )
        else:
            feature_rows[PRED_CONFIDENCE_COLUMN] = pd.Series(
                np.nan,
                index=feature_rows.index,
                dtype="float64",
            )

        return feature_rows

    def _get_class_color_lookup(
        self,
        *,
        category_column: str,
        observed_class_values: pd.Series | None = None,
    ) -> dict[int, np.ndarray]:
        """Build the complete class-id-to-RGBA lookup used for labels-layer styling without mutating table state."""
        table = self._get_bound_table()
        observed_class_ids: set[int] = set()
        if observed_class_values is not None:
            observed_class_ids = _read_class_values_without_normalizing(observed_class_values)
            if observed_class_ids is None:
                raise ClassStateError(
                    f"Cannot style labels by `{category_column}` because the current feature rows contain "
                    "invalid class values. Class values must be positive integers or missing."
                )

        if table is None or category_column not in table.obs:
            if not observed_class_ids:
                return {}
            raise ClassStateError(
                f"Cannot style labels by `{category_column}` because the selected table is unavailable or does "
                "not contain the required categorical class column."
            )

        categories = _read_class_categories(
            table.obs[category_column],
            column_name=category_column,
        )
        unknown_observed_classes = sorted(observed_class_ids - set(categories))
        if unknown_observed_classes:
            raise ClassStateError(
                f"Cannot style labels by `{category_column}` because observed class ids "
                f"{unknown_observed_classes} are not declared as categories in the selected table column."
            )

        if category_column == PRED_CLASS_COLUMN:
            return _read_prediction_class_color_lookup(table, categories)

        if category_column != USER_CLASS_COLUMN:
            raise ValueError(f"Unsupported Object Classification category column `{category_column}`.")

        _, colors = resolve_table_categorical_palette(
            table=table,
            column_name=category_column,
            categories=categories,
        )
        return {
            class_id: _rgba_array(color)
            for class_id, color in zip(categories, colors, strict=True)
        }


def _read_class_categories(
    values: pd.Series,
    *,
    column_name: str,
) -> list[int]:
    if not isinstance(values.dtype, pd.CategoricalDtype):
        raise ClassStateError(
            f"`{column_name}` must use a categorical dtype with positive integer categories before labels "
            "can be styled."
        )

    categories: list[int] = []
    for category in values.cat.categories:
        if isinstance(category, (bool, np.bool_)) or not isinstance(category, (int, np.integer)):
            raise ClassStateError(
                f"`{column_name}` categories must be positive integers. "
                "Rows without a class must be stored as missing values."
            )
        class_id = int(category)
        if class_id <= 0:
            raise ClassStateError(
                f"`{column_name}` categories must be positive integer class ids. "
                "Rows without a class must be stored as missing values."
            )
        categories.append(class_id)

    return categories


def _read_prediction_class_color_lookup(
    table: AnnData,
    pred_categories: list[int],
) -> dict[int, np.ndarray]:
    """Derive prediction colors without reading stored ``pred_class_colors``.

    ``user_class_colors`` is the user-owned color authority. Prediction
    classes that occur in the user-class vocabulary reuse those colors, while
    prediction-only classes receive deterministic defaults.

    Classifier write operations persist the same derived result in
    ``pred_class_colors`` for AnnData persistence and external consumers.
    Viewer styling deliberately derives the lookup again instead of treating
    that stored, classifier-derived palette as a second color authority. This
    also keeps display colors aligned with the current user-class palette when
    the stored prediction palette is missing or temporarily stale.
    """
    user_color_lookup: dict[int, str] = {}
    if USER_CLASS_COLUMN in table.obs:
        user_categories = _read_class_categories(
            table.obs[USER_CLASS_COLUMN],
            column_name=USER_CLASS_COLUMN,
        )
        _, user_colors = resolve_table_categorical_palette(
            table=table,
            column_name=USER_CLASS_COLUMN,
            categories=user_categories,
        )
        user_color_lookup = dict(zip(user_categories, user_colors, strict=True))

    return {
        class_id: _rgba_array(user_color_lookup.get(class_id, default_labeled_class_color(class_id)))
        for class_id in pred_categories
    }


def _to_numeric_values(values: pd.Series, column_name: str) -> pd.Series:
    numeric_values = pd.to_numeric(values, errors="coerce").astype("float64")
    numeric_values.name = column_name
    return numeric_values


def _read_class_values_without_normalizing(values: pd.Series) -> set[int] | None:
    if pd.api.types.is_integer_dtype(values.dtype) and not pd.api.types.is_bool_dtype(values.dtype):
        # Nullable integer columns preserve missing annotations. Drop those
        # rows before the NumPy conversion so the normal styling path remains
        # vectorized even for large tables.
        non_missing_values = values.dropna()
        if len(non_missing_values) == 0:
            return set()
        raw_values = non_missing_values.to_numpy(dtype=np.int64, copy=False)
        if int(np.min(raw_values)) <= 0:
            return None
        return {int(value) for value in np.unique(raw_values)}

    categories: set[int] = set()
    for value in values.to_numpy(copy=False):
        if pd.isna(value):
            continue
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            return None
        class_id = int(value)
        if class_id <= 0:
            return None
        categories.add(class_id)

    return categories


def _rgba_color_lookup(color_lookup: dict[int, Any]) -> dict[int, np.ndarray]:
    return {class_id: _rgba_array(color) for class_id, color in color_lookup.items()}


def _rgba_array(color: Any) -> np.ndarray:
    return np.asarray(to_rgba(color), dtype=np.float32)
