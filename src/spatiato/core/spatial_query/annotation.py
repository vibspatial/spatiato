from __future__ import annotations

import copy
from dataclasses import dataclass, field
from numbers import Integral
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from spatiato.core.class_palette import (
    default_categorical_colors,
    extend_categorical_palette,
    normalize_color_sequence,
    resolve_table_categorical_palette,
)
from spatiato.core.object_classification.annotation import USER_CLASS_COLUMN
from spatiato.core.object_classification.classifier import (
    PRED_CLASS_COLUMN,
    PRED_CONFIDENCE_COLUMN,
)
from spatiato.core.spatial_query.canonical_centers import (
    CanonicalCentersRegion,
    canonical_centers_status,
    read_canonical_centers_region,
)
from spatiato.core.spatial_query.query_models import CanonicalCenterQueryResult
from spatiato.core.spatialdata import get_table, get_table_metadata
from spatiato.core.validation import normalize_spatialdata_dataframe_column_name

if TYPE_CHECKING:
    from anndata import AnnData
    from spatialdata import SpatialData

type SpatialAnnotationColumnMode = Literal["existing", "new"]
type SpatialAnnotationValueKind = Literal["string", "positive_integer"]
type SpatialAnnotationValue = str | int | None

_CLASSIFIER_OWNED_COLUMNS = frozenset((PRED_CLASS_COLUMN, PRED_CONFIDENCE_COLUMN))
_OBJECT_CLASSIFICATION_COLUMNS = frozenset((USER_CLASS_COLUMN, PRED_CLASS_COLUMN, PRED_CONFIDENCE_COLUMN))


class SpatialAnnotationColumnChangedError(ValueError):
    """The prepared annotation column changed before atomic Apply."""


class SpatialAnnotationQueryOutdatedError(ValueError):
    """The table rows or canonical centers the query used are no longer current."""


@dataclass(frozen=True)
class SpatialAnnotationPreparation:
    """Immutable table-row and target-column snapshot prepared for Apply.

    Parameters
    ----------
    query_result
        The accepted query: the rows, instance IDs and canonical centers it
        evaluated, and the instance IDs it matched.
    column_name
        The ``obs`` column the annotation is written to.
    column_mode
        ``"existing"`` to annotate an existing column, ``"new"`` to create it
        at Apply.
    row_positions
        Table row positions (for ``iloc``) of the matched instances, in the
        order of ``query_result.matched_instance_ids``.
    prepared_values
        A copy of the column's values at ``row_positions``, taken when the
        annotation was prepared; all missing for a new column. Used for the
        summary counts, the value kind, and, at Apply, the check that the
        column has not changed since.
    """

    query_result: CanonicalCenterQueryResult
    column_name: str
    column_mode: SpatialAnnotationColumnMode
    row_positions: NDArray[np.intp] = field(repr=False, compare=False)
    prepared_values: pd.Series = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.query_result, CanonicalCenterQueryResult):
            raise TypeError("Spatial annotation preparation requires a CanonicalCenterQueryResult.")
        if self.column_mode not in ("existing", "new"):
            raise ValueError("Spatial annotation column mode must be `existing` or `new`.")
        if not isinstance(self.column_name, str) or not self.column_name:
            raise ValueError("Spatial annotation column name must not be empty.")
        row_positions = np.array(self.row_positions, dtype=np.intp, copy=True)
        row_positions.flags.writeable = False
        # The query result determines the rows: each matched instance ID's row
        # in the queried region. This also makes them unique and non-negative.
        expected_row_positions = _resolve_query_row_positions(self.query_result, self.query_result.canonical_centers)
        if not np.array_equal(row_positions, expected_row_positions):
            raise ValueError("Spatial annotation row positions must be the table rows of the matched instance IDs.")
        if not isinstance(self.prepared_values, pd.Series):
            raise TypeError("Spatial annotation prepared values must be a pandas Series.")
        if len(self.prepared_values) != len(row_positions):
            raise ValueError("Spatial annotation prepared values must match the queried instance count.")
        object.__setattr__(self, "row_positions", row_positions)
        object.__setattr__(self, "prepared_values", self.prepared_values.copy(deep=True))
        _ = self.value_kind

    @property
    def value_kind(self) -> SpatialAnnotationValueKind:
        """Return the annotation kind represented by the captured column."""
        return validate_and_resolve_spatial_annotation_value_kind(
            self.prepared_values,
            column_name=self.column_name,
        )

    @property
    def canonical_centers(self) -> CanonicalCentersRegion:
        """Return the rows, instance IDs and centers the query used."""
        return self.query_result.canonical_centers


@dataclass(frozen=True)
class SpatialAnnotationSummary:
    """Exact effect of setting or removing one annotation value.

    Parameters
    ----------
    annotation_value
        Normalized string or positive integer category for Set annotation.
        ``None`` is the domain sentinel for Remove annotation: Apply clears
        the matched categorical values with ``pd.NA``. ``None`` itself is not
        stored, and the strings ``"None"`` and ``"nan"`` remain ordinary
        annotation categories.
    matched_count
        Total number of matched rows included in the operation.
    current_missing_count
        Number of matched rows whose current annotation is missing.
    current_equal_count
        Number of matched rows already equal to a Set annotation value. This is
        zero for Remove annotation.
    current_other_count
        Number of different non-missing values that Set would overwrite, or
        existing non-missing annotations that Remove would clear.
    """

    annotation_value: SpatialAnnotationValue
    matched_count: int
    current_missing_count: int
    current_equal_count: int
    current_other_count: int

    def __post_init__(self) -> None:
        annotation_value = self.annotation_value
        if annotation_value is not None:
            if isinstance(annotation_value, str):
                annotation_value = annotation_value.strip()
                if not annotation_value:
                    raise ValueError("Spatial annotation string value must not be empty.")
            elif isinstance(annotation_value, bool) or not isinstance(annotation_value, int):
                raise TypeError("Spatial annotation value must be a string, positive integer, or None.")
            elif annotation_value <= 0:
                raise ValueError("Spatial annotation integer value must be positive.")
            object.__setattr__(self, "annotation_value", annotation_value)

        counts = (
            self.matched_count,
            self.current_missing_count,
            self.current_equal_count,
            self.current_other_count,
        )
        if any(isinstance(count, bool) or not isinstance(count, Integral) or count < 0 for count in counts):
            raise ValueError("Spatial annotation summary counts must be non-negative integers.")
        if self.matched_count != sum(counts[1:]):
            raise ValueError("Spatial annotation summary counts must partition every matched row.")
        if self.is_removal and self.current_equal_count != 0:
            raise ValueError("Removal summaries must have zero equal values.")

    @property
    def is_removal(self) -> bool:
        """Return whether this summary describes annotation removal."""
        return self.annotation_value is None

    @property
    def changed_count(self) -> int:
        """Return the number of rows that would change."""
        if self.is_removal:
            return self.current_other_count
        return self.current_missing_count + self.current_other_count

    @property
    def unchanged_count(self) -> int:
        """Return the number of rows already in the requested state."""
        if self.is_removal:
            return self.current_missing_count
        return self.current_equal_count

    @property
    def overwrite_count(self) -> int:
        """Return the number of different non-missing values Set would replace."""
        return 0 if self.is_removal else self.current_other_count

    @property
    def removal_count(self) -> int:
        """Return the number of existing annotations Remove would clear."""
        return self.current_other_count if self.is_removal else 0


@dataclass(frozen=True)
class SpatialAnnotationApplyResult:
    """Report which parts of the annotation consistency unit changed."""

    annotation_changed: bool
    palette_changed: bool

    def __post_init__(self) -> None:
        if not isinstance(self.annotation_changed, bool) or not isinstance(self.palette_changed, bool):
            raise TypeError("Spatial annotation apply flags must be booleans.")
        if self.palette_changed and not self.annotation_changed:
            raise ValueError("A spatial-annotation palette changes only with an effective annotation mutation.")


def prepare_spatial_annotation(
    sdata: SpatialData,
    *,
    query_result: CanonicalCenterQueryResult,
    column_name: str,
    column_mode: SpatialAnnotationColumnMode,
) -> SpatialAnnotationPreparation:
    """Resolve queried instance IDs to live table rows without mutation."""
    if not isinstance(query_result, CanonicalCenterQueryResult):
        raise TypeError("Spatial annotation preparation requires a CanonicalCenterQueryResult.")
    if query_result.matched_instance_count == 0:
        raise ValueError("Spatial annotation preparation requires at least one matching instance.")
    if column_mode not in ("existing", "new"):
        raise ValueError("Spatial annotation column mode must be `existing` or `new`.")
    normalized_column_name = normalize_spatialdata_dataframe_column_name(column_name, "Annotation column name")

    queried = query_result.canonical_centers
    current = _read_current_canonical_centers_region(sdata, queried)
    if not _canonical_centers_region_rows_match(current, queried):
        raise SpatialAnnotationQueryOutdatedError(
            "The table rows of the selected labels element changed after the canonical-center query."
        )
    table = get_table(sdata, queried.table_name)
    table_metadata = get_table_metadata(sdata, queried.table_name)
    if normalized_column_name in (table_metadata.region_key, table_metadata.instance_key):
        raise ValueError(f"Annotation column `{normalized_column_name}` cannot be a table linkage column.")
    row_positions = _resolve_query_row_positions(query_result, current)

    if column_mode == "existing":
        current_column = require_compatible_spatial_annotation_column(table, normalized_column_name)
        prepared_values = current_column.iloc[row_positions].copy(deep=True)
    else:
        if normalized_column_name in _OBJECT_CLASSIFICATION_COLUMNS:
            raise ValueError(
                f"New annotation column `{normalized_column_name}` is reserved for Object Classification "
                "and cannot be created by Spatial Query."
            )
        if normalized_column_name in table.obs.columns:
            raise ValueError(f"New annotation column `{normalized_column_name}` already exists.")
        prepared_values = pd.Series(
            pd.Categorical([pd.NA] * len(row_positions), categories=[]),
            index=table.obs.index[row_positions],
            name=normalized_column_name,
        )

    return SpatialAnnotationPreparation(
        query_result=query_result,
        column_name=normalized_column_name,
        column_mode=column_mode,
        row_positions=row_positions,
        prepared_values=prepared_values,
    )


def summarize_spatial_annotation(
    preparation: SpatialAnnotationPreparation,
    annotation_value: SpatialAnnotationValue,
) -> SpatialAnnotationSummary:
    """Summarize a proposed Set or Remove action without mutation."""
    if not isinstance(preparation, SpatialAnnotationPreparation):
        raise TypeError("Spatial annotation summarization requires a SpatialAnnotationPreparation.")
    if annotation_value is None and preparation.column_mode == "new":
        raise ValueError("Remove annotation is not available for a new column.")

    normalized_value = annotation_value
    if normalized_value is not None:
        if preparation.value_kind == "string":
            if not isinstance(normalized_value, str):
                raise TypeError("String spatial annotation targets require a string value or None.")
            normalized_value = normalized_value.strip()
            if not normalized_value:
                raise ValueError("Spatial annotation string value must not be empty.")
        elif preparation.value_kind == "positive_integer":
            if isinstance(normalized_value, bool) or not isinstance(normalized_value, int):
                raise TypeError("Positive-integer spatial annotation targets require a Python int or None.")
            if normalized_value <= 0:
                raise ValueError("Spatial annotation integer value must be positive.")
        else:  # pragma: no cover - guarded by SpatialAnnotationPreparation
            raise ValueError(f"Unsupported spatial annotation value kind: {preparation.value_kind!r}.")

    missing = preparation.prepared_values.isna().to_numpy(dtype=bool)
    if normalized_value is None:
        equal = np.zeros(len(missing), dtype=bool)
    else:
        equal = np.asarray(preparation.prepared_values == normalized_value, dtype=bool) & ~missing
    other = ~missing & ~equal

    return SpatialAnnotationSummary(
        annotation_value=normalized_value,
        matched_count=len(missing),
        current_missing_count=int(missing.sum()),
        current_equal_count=int(equal.sum()),
        current_other_count=int(other.sum()),
    )


def apply_spatial_annotation(
    sdata: SpatialData,
    preparation: SpatialAnnotationPreparation,
    expected_summary: SpatialAnnotationSummary,
) -> SpatialAnnotationApplyResult:
    """Revalidate and atomically apply one prepared annotation mutation.

    Parameters
    ----------
    sdata
        SpatialData object containing the current table and its canonical
        centers.
    preparation
        Immutable row and target-column snapshot created before the annotation
        was applied.
    expected_summary
        Exact Set or Remove summary calculated from ``preparation``. The
        annotation value lives on this summary so its value and counts cannot
        disagree.
        Apply first verifies that the table rows, canonical centers, and
        selected column values still match ``preparation``. It then recalculates the
        summary from those accepted values and requires it to equal
        ``expected_summary`` before mutating the table. This final comparison
        protects the prepared consistency contract; live table changes are
        handled by the preceding freshness validation.

    Returns
    -------
    SpatialAnnotationApplyResult
        Whether the annotation column and companion palette changed.
    """
    if not isinstance(preparation, SpatialAnnotationPreparation):
        raise TypeError("Spatial annotation apply requires a SpatialAnnotationPreparation.")
    if not isinstance(expected_summary, SpatialAnnotationSummary):
        raise TypeError("Spatial annotation apply requires a SpatialAnnotationSummary.")
    if expected_summary.is_removal and preparation.column_mode == "new":
        raise ValueError("Remove annotation is not available for a new column.")

    _raise_if_query_outdated(sdata, preparation)
    table = get_table(sdata, preparation.canonical_centers.table_name)
    current_column: pd.Series | None = None
    if preparation.column_mode == "existing":
        current_column = require_compatible_spatial_annotation_column(table, preparation.column_name)
        current_value_kind = validate_and_resolve_spatial_annotation_value_kind(
            current_column,
            column_name=preparation.column_name,
        )
        if current_value_kind != preparation.value_kind:
            raise SpatialAnnotationColumnChangedError(
                "The annotation column value kind changed before the annotation was applied."
            )
        _require_unchanged_existing_column(preparation, current_column)
    else:
        if preparation.column_name in _OBJECT_CLASSIFICATION_COLUMNS:
            raise ValueError(
                f"New annotation column `{preparation.column_name}` is reserved for Object Classification "
                "and cannot be created by Spatial Query."
            )
        if preparation.column_name in table.obs.columns:
            raise ValueError(f"New annotation column `{preparation.column_name}` already exists.")

    fresh_summary = summarize_spatial_annotation(preparation, expected_summary.annotation_value)
    if fresh_summary != expected_summary:
        raise ValueError("The expected spatial-annotation summary does not match the prepared values.")
    if fresh_summary.changed_count == 0:
        return SpatialAnnotationApplyResult(annotation_changed=False, palette_changed=False)

    # Build the complete replacement column and optional companion palette
    # off-table. This phase reads the live AnnData state but does not mutate it.
    replacement, palette, palette_changed = _build_annotation_replacement(
        table,
        preparation,
        current_column,
        fresh_summary,
    )
    # This is the AnnData mutation boundary: install the prepared `.obs`
    # column and `.uns` palette as one unit, rolling both back on failure.
    _assign_annotation_column_and_palette_atomically(
        table,
        column_name=preparation.column_name,
        replacement=replacement,
        palette=palette,
    )
    return SpatialAnnotationApplyResult(annotation_changed=True, palette_changed=palette_changed)


def _read_current_canonical_centers_region(
    sdata: SpatialData,
    queried: CanonicalCentersRegion,
) -> CanonicalCentersRegion:
    """Read the queried labels element's rows, instance IDs and centers as they are now."""
    try:
        return read_canonical_centers_region(
            sdata,
            table_name=queried.table_name,
            labels_name=queried.labels_name,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SpatialAnnotationQueryOutdatedError(
            "The table rows or canonical centers of the selected labels element are no longer available."
        ) from exc


def _canonical_centers_region_rows_match(current: CanonicalCentersRegion, queried: CanonicalCentersRegion) -> bool:
    return (
        current.table_name == queried.table_name
        and current.labels_name == queried.labels_name
        and np.array_equal(current.row_positions, queried.row_positions)
        and np.array_equal(current.instance_ids, queried.instance_ids)
    )


def require_compatible_spatial_annotation_column(table: AnnData, column_name: str) -> pd.Series:
    """Return an existing column that can store spatial annotations."""
    if column_name not in table.obs.columns:
        raise ValueError(f"Existing annotation column `{column_name}` is not present.")
    values = table.obs[column_name]
    validate_and_resolve_spatial_annotation_value_kind(values, column_name=column_name)
    return values


def validate_and_resolve_spatial_annotation_value_kind(
    values: pd.Series,
    *,
    column_name: str,
) -> SpatialAnnotationValueKind:
    """Validate a categorical annotation column and return its value kind."""
    if column_name in _CLASSIFIER_OWNED_COLUMNS:
        raise ValueError(
            f"Existing annotation column `{column_name}` is classifier-owned and cannot be modified by Spatial Query."
        )
    if not isinstance(values, pd.Series):
        raise TypeError("Spatial annotation values must be a pandas Series.")
    if not isinstance(values.dtype, pd.CategoricalDtype):
        raise ValueError(f"Existing annotation column `{column_name}` must be categorical.")

    categories = values.cat.categories
    # `user_class` has a fixed positive-integer schema. Handle it before the
    # generic empty-category fallback because an empty categorical commonly
    # has object-typed categories and would otherwise be misclassified as a
    # string target. `all([])` intentionally accepts that empty vocabulary,
    # while a non-empty string or invalid integer vocabulary still fails.
    if column_name == USER_CLASS_COLUMN:
        if all(_is_positive_integer_category(category) for category in categories):
            return "positive_integer"
        raise ValueError(f"Existing annotation column `{column_name}` must contain only positive integer categories.")
    if len(categories) == 0:
        if pd.api.types.is_integer_dtype(categories.dtype) and not pd.api.types.is_bool_dtype(categories.dtype):
            return "positive_integer"
        return "string"
    if all(isinstance(category, str) for category in categories):
        return "string"
    if all(_is_positive_integer_category(category) for category in categories):
        return "positive_integer"
    raise ValueError(
        f"Existing annotation column `{column_name}` must contain only string categories "
        "or positive integer categories."
    )


def get_compatible_spatial_annotation_column_names(
    sdata: SpatialData,
    table_name: str,
) -> list[str]:
    """Return compatible existing annotation columns in table order."""
    table = sdata.tables[table_name]
    table_metadata = get_table_metadata(sdata, table_name)
    excluded_columns = {
        table_metadata.region_key,
        table_metadata.instance_key,
        *_CLASSIFIER_OWNED_COLUMNS,
    }
    compatible_columns: list[str] = []
    for column_name in table.obs.columns:
        if not isinstance(column_name, str) or column_name in excluded_columns:
            continue
        try:
            validate_and_resolve_spatial_annotation_value_kind(
                table.obs[column_name],
                column_name=column_name,
            )
        except (TypeError, ValueError):
            continue
        compatible_columns.append(column_name)
    return compatible_columns


def _is_positive_integer_category(value: object) -> bool:
    return not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, np.integer)) and int(value) > 0


def _resolve_query_row_positions(
    query_result: CanonicalCenterQueryResult,
    region: CanonicalCentersRegion,
) -> NDArray[np.intp]:
    sort_order = np.argsort(region.instance_ids)
    sorted_instance_ids = region.instance_ids[sort_order]
    selected_positions = np.searchsorted(sorted_instance_ids, query_result.matched_instance_ids)
    if np.any(selected_positions >= len(sorted_instance_ids)) or not np.array_equal(
        sorted_instance_ids[selected_positions], query_result.matched_instance_ids
    ):  # pragma: no cover - guarded by the result and the region row checks
        raise SpatialAnnotationQueryOutdatedError(
            "A queried instance is no longer present in the selected table region."
        )
    return np.asarray(region.row_positions[sort_order[selected_positions]], dtype=np.intp)


def _raise_if_query_outdated(
    sdata: SpatialData,
    preparation: SpatialAnnotationPreparation,
) -> None:
    """Raise ``SpatialAnnotationQueryOutdatedError`` if the table rows or canonical centers changed since the query.

    Check again, immediately before the annotation is applied, that:

    - the table's canonical centers are still valid
      (``canonical_centers_status(...)``);
    - the queried labels element's rows and instance IDs are unchanged, read
      from the table annotation, so the matched instances still resolve to the
      prepared rows;
    - its canonical centers are unchanged.

    Otherwise, raise ``SpatialAnnotationQueryOutdatedError`` and require a new
    query.

    This is necessary because the query result is calculated asynchronously
    before annotation Apply. While that result is pending, another operation
    can reload or replace the table, change its rows, or recalculate its
    canonical centers. Applying the old matching instance IDs after such a
    change could annotate rows using a geometric decision made from different
    centers.

    This validation reads only table state: the status check reads the whole
    canonical matrix, and the comparison only the rows of the queried labels
    element. It does not read labels pixels or recalculate centers. Shapes and
    transformation freshness remain controller responsibilities.
    """
    queried = preparation.query_result.canonical_centers
    status = canonical_centers_status(sdata, table_name=queried.table_name)
    if status.state != "valid":
        raise SpatialAnnotationQueryOutdatedError("The canonical centers of the table are no longer valid.")
    current = _read_current_canonical_centers_region(sdata, queried)
    # Unchanged rows and instance IDs also mean the matched instances still
    # resolve to the prepared rows: `SpatialAnnotationPreparation` requires
    # them to be the rows of the matched IDs in the queried region.
    if not _canonical_centers_region_rows_match(current, queried):
        raise SpatialAnnotationQueryOutdatedError(
            "The table rows of the selected labels element changed after annotation preparation."
        )
    if not np.array_equal(current.centers, queried.centers):
        raise SpatialAnnotationQueryOutdatedError(
            "The canonical centers of the selected labels element changed after the query."
        )


def _require_unchanged_existing_column(
    preparation: SpatialAnnotationPreparation,
    current_column: pd.Series,
) -> None:
    prepared_values = preparation.prepared_values
    current_categories = tuple(current_column.cat.categories)
    prepared_categories = tuple(prepared_values.cat.categories)
    if current_categories != prepared_categories or current_column.cat.ordered != prepared_values.cat.ordered:
        raise SpatialAnnotationColumnChangedError(
            "The annotation column categories changed before the annotation was applied."
        )
    current_values = current_column.iloc[preparation.row_positions].reset_index(drop=True)
    prepared_values = prepared_values.reset_index(drop=True)
    if not current_values.equals(prepared_values):
        raise SpatialAnnotationColumnChangedError("The annotation values changed before the annotation was applied.")


def _build_annotation_replacement(
    table: AnnData,
    preparation: SpatialAnnotationPreparation,
    current_column: pd.Series | None,
    summary: SpatialAnnotationSummary,
) -> tuple[pd.Series, list[str] | None, bool]:
    column_name = preparation.column_name
    annotation_value = summary.annotation_value
    if preparation.column_mode == "new":
        assert annotation_value is not None
        replacement = pd.Series(
            pd.Categorical([pd.NA] * table.n_obs, categories=[annotation_value]),
            index=table.obs.index,
            name=column_name,
        )
        replacement.iloc[preparation.row_positions] = annotation_value
        return replacement, default_categorical_colors(1), True

    assert current_column is not None
    current_categories = tuple(current_column.cat.categories)
    replacement = current_column.copy(deep=True)
    next_categories = current_categories
    if annotation_value is not None and annotation_value not in current_categories:
        next_categories = (*current_categories, annotation_value)
        replacement = replacement.cat.add_categories([annotation_value])
    replacement.iloc[preparation.row_positions] = pd.NA if annotation_value is None else annotation_value

    palette_source, current_palette = resolve_table_categorical_palette(
        table=table,
        column_name=column_name,
        categories=current_categories,
    )
    if next_categories != current_categories:
        palette = extend_categorical_palette(
            current_palette,
            current_categories=current_categories,
            next_categories=next_categories,
        )
        return replacement, palette, True
    if palette_source != "stored":
        return replacement, current_palette, True
    return replacement, None, False


def _assign_annotation_column_and_palette_atomically(
    table: AnnData,
    *,
    column_name: str,
    replacement: pd.Series,
    palette: list[str] | None,
) -> None:
    column_existed = column_name in table.obs.columns
    previous_column = table.obs[column_name].copy(deep=True) if column_existed else None
    palette_key = f"{column_name}_colors"
    palette_existed = palette_key in table.uns
    previous_palette = copy.deepcopy(table.uns[palette_key]) if palette_existed else None

    try:
        table.obs[column_name] = replacement
        if palette is not None:
            table.uns[palette_key] = list(palette)
        if not table.obs[column_name].equals(replacement):
            raise RuntimeError("Spatial annotation column assignment did not preserve the prepared values.")
        if palette is not None and normalize_color_sequence(table.uns.get(palette_key)) != palette:
            raise RuntimeError("Spatial annotation palette assignment did not preserve the prepared colors.")
    except BaseException:
        try:
            if column_existed:
                table.obs[column_name] = previous_column
            elif column_name in table.obs.columns:
                table.obs.pop(column_name)
            if palette_existed:
                table.uns[palette_key] = previous_palette
            else:
                table.uns.pop(palette_key, None)
        except BaseException as rollback_error:  # pragma: no cover - catastrophic mapping failure
            raise RuntimeError(
                "Spatial annotation assignment failed, and rollback could not restore the previous column and palette."
            ) from rollback_error
        raise
