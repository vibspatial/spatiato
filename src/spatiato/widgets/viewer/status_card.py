from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from spatiato.widgets.shared_styles import StatusCardKind, format_feedback_identifier, validate_status_card_kind

if TYPE_CHECKING:
    from spatiato.viewer.image_styling import ImageLoadResult
    from spatiato.viewer.labels_styling import LabelsLoadResult
    from spatiato.viewer.points_styling import PointsLoadResult
    from spatiato.viewer.shapes_styling import ShapesLoadResult
    from spatiato.widgets.viewer.labels_widget import LabelsLoadRequest
    from spatiato.widgets.viewer.points_controller import PointsLoadRequest
    from spatiato.widgets.viewer.shapes_widget import ShapesLoadRequest


@dataclass(frozen=True)
class _ViewerStatusCardSpec:
    title: str
    lines: tuple[str, ...]
    kind: StatusCardKind
    tooltip_message: str | None = None

    def __post_init__(self) -> None:
        validate_status_card_kind(self.kind)


def build_viewer_error_card_spec(title: str, lines: Sequence[str]) -> _ViewerStatusCardSpec:
    return build_viewer_feedback_card_spec(
        title=title,
        lines=lines,
        kind="error",
    )


def build_viewer_feedback_card_spec(
    message: str | None = None,
    *,
    title: str | None = None,
    lines: Sequence[str] | None = None,
    kind: StatusCardKind | None = None,
    is_error: bool | None = None,
    tooltip_message: str | None = None,
) -> _ViewerStatusCardSpec:
    if lines is None:
        resolved_lines = () if message is None else (message,)
    else:
        resolved_lines = tuple(lines)
    if kind is None:
        kind = "error" if is_error else "success"
    if title is None:
        title = "Viewer Error" if kind == "error" else "Viewer Updated"

    return _ViewerStatusCardSpec(
        title=title,
        lines=resolved_lines,
        kind=kind,
        tooltip_message=tooltip_message,
    )


def build_points_layer_card_spec(
    request: PointsLoadRequest,
    result: PointsLoadResult,
) -> _ViewerStatusCardSpec:
    selection = request.selection
    action = _created_updated_action(result.created)
    points_name = request.identity.points_name
    display_name, was_shortened = format_feedback_identifier(points_name)
    lines = [
        (
            f'{action} points layer for "{display_name}" by "{selection.index_column}" '
            f"with {selection.loaded_count:,} point(s)."
        )
    ]
    tooltip_message = (
        f'{action} points layer for "{points_name}" by "{selection.index_column}" '
        f"with {selection.loaded_count:,} point(s)."
        if was_shortened
        else None
    )
    kind: StatusCardKind = "success"
    title = f"Points Layer {action}"
    if selection.warning:
        kind = "warning"
        title = _with_warning_suffix(title)
        lines.append(selection.warning)
    if result.categorical_coloring_disabled:
        kind = "warning"
        title = _with_warning_suffix(title)
        lines.append(
            f"Categorical coloring is disabled for {result.selected_value_count:,} selected values; "
            f"using one solid color because the categorical limit is {result.categorical_limit:,}."
        )

    return _ViewerStatusCardSpec(
        title=title,
        lines=tuple(lines),
        kind=kind,
        tooltip_message=tooltip_message,
    )


def build_primary_labels_loaded_card_spec(
    request: LabelsLoadRequest,
    result: LabelsLoadResult,
) -> _ViewerStatusCardSpec:
    action = _created_updated_action(result.created)
    labels_name = request.labels_name
    display_name, was_shortened = format_feedback_identifier(labels_name)
    return _ViewerStatusCardSpec(
        title=f"Labels Layer {action}",
        lines=(f'{action} labels layer for "{display_name}".',),
        kind="success",
        tooltip_message=f'{action} labels layer for "{labels_name}".' if was_shortened else None,
    )


def build_styled_labels_card_spec(
    request: LabelsLoadRequest,
    result: LabelsLoadResult,
) -> _ViewerStatusCardSpec:
    selected_color_source = request.selected_color_source
    if selected_color_source is None:
        raise ValueError("Styled labels status requires a selected color source.")

    action = _created_updated_action(result.created)
    source_text = selected_color_source.source_label
    labels_name = request.labels_name
    display_name, was_shortened = format_feedback_identifier(labels_name)
    lines = [(f'{action} colored overlay "{source_text}" on labels element "{display_name}".')]
    tooltip_message = (
        f'{action} colored overlay "{source_text}" on labels element "{labels_name}".' if was_shortened else None
    )
    title = f"Colored Overlay {action}"
    kind = _append_palette_status_lines(
        lines,
        result=result,
        include_instance_message=True,
    )
    return _ViewerStatusCardSpec(
        title=_palette_status_title(title, result, include_instance_message=True),
        lines=tuple(lines),
        kind=kind,
        tooltip_message=tooltip_message,
    )


def build_primary_shapes_loaded_card_spec(
    request: ShapesLoadRequest,
    result: ShapesLoadResult,
) -> _ViewerStatusCardSpec:
    action = _created_updated_action(result.created)
    shapes_name = request.shapes_name
    display_name, was_shortened = format_feedback_identifier(shapes_name)
    title = f"Shapes Layer {action}"
    kind: StatusCardKind = "success"
    lines = [f'{action} shapes layer for "{display_name}".']
    if result.shapes_rendering_mode == "points":
        lines.append("Rendered point-radius shapes as napari points for faster display.")
    if result.skipped_geometry_count:
        title = _with_warning_suffix(title)
        kind = "warning"
        lines.append(_format_skipped_geometry_line(result.skipped_geometry_count))

    return _ViewerStatusCardSpec(
        title=title,
        lines=tuple(lines),
        kind=kind,
        tooltip_message=f'{action} shapes layer for "{shapes_name}".' if was_shortened else None,
    )


def build_styled_shapes_card_spec(
    request: ShapesLoadRequest,
    result: ShapesLoadResult,
) -> _ViewerStatusCardSpec:
    selected_color_source = request.selected_color_source
    if selected_color_source is None:
        raise ValueError("Styled shapes status requires a selected color source.")

    action = _created_updated_action(result.created)
    source_text = selected_color_source.source_label
    shapes_name = request.shapes_name
    display_name, was_shortened = format_feedback_identifier(shapes_name)
    lines = [(f'{action} styled shapes layer "{source_text}" on shapes element "{display_name}".')]
    if result.shapes_rendering_mode == "points":
        lines.append("Rendered point-radius shapes as napari points for faster display.")
    tooltip_message = (
        f'{action} styled shapes layer "{source_text}" on shapes element "{shapes_name}".' if was_shortened else None
    )
    title = f"Styled Shapes {action}"
    kind = _append_palette_status_lines(
        lines,
        result=result,
        include_instance_message=True,
    )
    title = _palette_status_title(title, result, include_instance_message=True)
    unannotated_source_shape_count = getattr(result, "unannotated_source_shape_count", 0)
    unannotated_rendered_shape_count = getattr(result, "unannotated_rendered_shape_count", 0)
    if unannotated_source_shape_count or unannotated_rendered_shape_count:
        if kind == "success":
            kind = "info"
        lines.append(
            _format_unannotated_shapes_line(
                source_shape_count=unannotated_source_shape_count,
                rendered_shape_count=unannotated_rendered_shape_count,
            )
        )
    if result.skipped_geometry_count:
        kind = "warning"
        title = _with_warning_suffix(title)
        lines.append(_format_skipped_geometry_line(result.skipped_geometry_count))

    return _ViewerStatusCardSpec(
        title=title,
        lines=tuple(lines),
        kind=kind,
        tooltip_message=tooltip_message,
    )


def build_image_loaded_card_spec(
    image_name: str,
    result: ImageLoadResult,
) -> _ViewerStatusCardSpec:
    display_name, was_shortened = format_feedback_identifier(image_name)
    action = _created_updated_action(result.created)
    if result.mode == "stack":
        line = f'{action} image layer for "{display_name}" in stack mode.'
        tooltip_line = f'{action} image layer for "{image_name}" in stack mode.'
    else:
        channel_text = _format_image_overlay_channels(result)
        line = f'{action} image overlay for "{display_name}" with {channel_text}.'
        tooltip_line = f'{action} image overlay for "{image_name}" with {channel_text}.'

    return _ViewerStatusCardSpec(
        title=f"Image Layer {action}",
        lines=(line,),
        kind="success",
        tooltip_message=tooltip_line if was_shortened else None,
    )


def _format_image_overlay_channels(result: ImageLoadResult) -> str:
    channel_names = tuple(getattr(result, "channel_names", ()))
    channel_values = channel_names or tuple(str(channel) for channel in result.channels)
    label = "channel" if len(channel_values) == 1 else "channels"
    formatted_channels = ", ".join(f'"{channel}"' for channel in channel_values)
    return f"{label} {formatted_channels}"


def _created_updated_action(created: bool) -> str:
    return "Created" if created else "Updated"


def _append_palette_status_lines(
    lines: list[str],
    *,
    result: LabelsLoadResult | ShapesLoadResult,
    include_instance_message: bool,
) -> StatusCardKind:
    if include_instance_message and result.value_kind == "instance":
        lines.append("Used instance colors.")
        return "success"
    if result.coercion_applied:
        lines.append("Coerced string values to categorical and used the default categorical palette.")
        return "warning"
    if result.palette_source == "stored":
        lines.append("Used the stored categorical palette.")
        return "success"
    if result.palette_source == "default_invalid":
        lines.append("The stored categorical palette was invalid, so Spatiato used the default categorical palette.")
        return "warning"
    if result.palette_source == "default_missing":
        lines.append("Used the default categorical palette because no stored palette was present.")
        return "info"
    return "success"


def _palette_status_title(
    title: str,
    result: LabelsLoadResult | ShapesLoadResult,
    *,
    include_instance_message: bool,
) -> str:
    if include_instance_message and result.value_kind == "instance":
        return title
    if result.coercion_applied or result.palette_source == "default_invalid":
        return _with_warning_suffix(title)
    return title


def _format_skipped_geometry_line(skipped_geometry_count: int) -> str:
    return (
        f"Skipped {skipped_geometry_count} empty, invalid, or unsupported geometries while loading renderable shapes."
    )


def _format_unannotated_shapes_line(*, source_shape_count: int, rendered_shape_count: int) -> str:
    if source_shape_count == rendered_shape_count:
        shape_text = _format_count(source_shape_count, "shape", "shapes")
        verb = "it has" if source_shape_count == 1 else "they have"
        return f"Rendered {shape_text} transparent because {verb} no row in the linked table."

    rendered_text = _format_count(rendered_shape_count, "shape", "shapes")
    source_text = _format_count(source_shape_count, "source shape", "source shapes")
    verb = "has" if source_shape_count == 1 else "have"
    return f"Rendered {rendered_text} transparent because {source_text} {verb} no row in the linked table."


def _format_count(count: int, singular: str, plural: str) -> str:
    label = singular if count == 1 else plural
    return f"{count:,} {label}"


def _with_warning_suffix(title: str) -> str:
    if title.endswith(" With Warning"):
        return title
    return f"{title} With Warning"
