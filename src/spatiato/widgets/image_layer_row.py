from __future__ import annotations

import math
from dataclasses import dataclass

from qtpy.QtCore import Signal
from qtpy.QtGui import QColor
from qtpy.QtWidgets import QWidget

from spatiato.viewer.adapter import ImageLayerBinding
from spatiato.viewer.image_styling import DEFAULT_OVERLAY_COLORS
from spatiato.widgets.layer_row import _LayerRow
from spatiato.widgets.overlay_color_button import OverlayColorButton


@dataclass(frozen=True)
class _ImageColormapPresentation:
    name: str
    colors: tuple[str, ...]
    solid_color: str | None


class _ImageLayerRow(_LayerRow):
    """Present one loaded image layer, with a color button for its colormap.

    It follows the contract of `_LayerRow`, and adds color intent. The color
    button emits ``color_change_requested``, and the owning Viewer or Histogram
    widget, which binds its own channel, card, or image identity while
    connecting the signal, validates the current live binding before assigning
    ``layer.colormap``. napari then emits ``layer.events.colormap``, and every
    row bound to that layer renders the accepted colormap.

    The color button is the row's control, between the label and the remove
    button. ``show_colormap=False`` hides it, for example for an RGB stack.

    ``ViewerWidget`` and ``HistogramWidget`` intentionally retain separate
    intent handlers. Although their visibility and colormap assignments are
    parallel, target resolution, stale-binding recovery, and user feedback are
    owner-specific. Shared code is limited to row presentation and pure
    layer-property helpers; layer mutation must not move into this row or a
    shared mutation handler.
    """

    color_change_requested = Signal(str)
    _object_name_prefix = "image_layer"
    binding: ImageLayerBinding

    def __init__(
        self,
        binding: ImageLayerBinding,
        *,
        display_label: str,
        accessibility_label: str,
        show_colormap: bool = True,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(
            binding,
            display_label=display_label,
            accessibility_label=accessibility_label,
            parent=parent,
        )
        self._show_colormap = show_colormap

        self.color_button = OverlayColorButton(DEFAULT_OVERLAY_COLORS[0], self)
        self.color_button.setObjectName("image_layer_color_button")
        self.color_button.setVisible(show_colormap)
        self.color_button.color_selected.connect(self._on_color_selected)
        self.add_control(self.color_button)

        if show_colormap:
            binding.layer.events.colormap.connect(self._on_layer_colormap_changed)
            self._sync_colormap_from_layer()

    def refresh_presentation(self) -> None:
        """Render visibility and colormap from the authoritative live layer."""
        super().refresh_presentation()
        if self._show_colormap:
            self._sync_colormap_from_layer()

    def _disconnect_layer_events(self) -> None:
        super()._disconnect_layer_events()
        if self._show_colormap:
            self._disconnect_layer_event(self.binding.layer.events.colormap, self._on_layer_colormap_changed)

    def _on_layer_colormap_changed(self, _event: object) -> None:
        self._sync_colormap_from_layer()

    def _on_color_selected(self, color: str) -> None:
        self.color_change_requested.emit(color)

    def _sync_colormap_from_layer(self) -> None:
        presentation = _colormap_presentation_from_layer(self.binding.layer)
        if presentation is None:
            return
        if presentation.solid_color is not None:
            self.color_button.set_color(presentation.solid_color)
            return
        self.color_button.set_colormap_preview(
            presentation.name,
            presentation.colors,
        )


def _binding_channel_index(binding: ImageLayerBinding) -> int:
    channel_index = binding.channel_index
    if not isinstance(channel_index, int) or channel_index < 0:
        raise ValueError("Overlay bindings require a non-negative channel index.")
    return channel_index


def _binding_channel_name(binding: ImageLayerBinding) -> str:
    channel_name = binding.channel_name
    if not isinstance(channel_name, str) or not channel_name:
        raise ValueError("Overlay bindings require a non-empty channel name.")
    return channel_name


def _solid_color_from_layer(layer: object) -> str | None:
    presentation = _colormap_presentation_from_layer(layer)
    return presentation.solid_color if presentation is not None else None


def _colormap_presentation_from_layer(
    layer: object,
) -> _ImageColormapPresentation | None:
    colormap = getattr(layer, "colormap", None)
    name = getattr(colormap, "name", None)
    display_name = name if isinstance(name, str) and name else "Custom colormap"
    colors = _colormap_colors_as_hex(getattr(colormap, "colors", None))
    if len(colors) == 1:
        return _ImageColormapPresentation(
            name=display_name,
            colors=colors,
            solid_color=colors[0],
        )
    if len(colors) == 2 and colors[0] == "#000000":
        return _ImageColormapPresentation(
            name=display_name,
            colors=colors,
            solid_color=colors[1],
        )
    if len(colors) >= 2:
        return _ImageColormapPresentation(
            name=display_name,
            colors=colors,
            solid_color=None,
        )

    if isinstance(name, str):
        solid_color = _normalized_color_or_none(name)
        if solid_color is not None:
            return _ImageColormapPresentation(
                name=name,
                colors=(solid_color,),
                solid_color=solid_color,
            )
    if isinstance(colormap, str):
        solid_color = _normalized_color_or_none(colormap)
        if solid_color is not None:
            return _ImageColormapPresentation(
                name=colormap,
                colors=(solid_color,),
                solid_color=solid_color,
            )
    return None


def _colormap_colors_as_hex(colors: object) -> tuple[str, ...]:
    try:
        rows = tuple(colors)  # type: ignore[arg-type]
    except TypeError:
        return ()

    converted = tuple(color for row in rows if (color := _color_row_to_hex(row)) is not None)
    return converted if len(converted) == len(rows) else ()


def _color_row_to_hex(row: object) -> str | None:
    if isinstance(row, str):
        return _normalized_color_or_none(row)

    try:
        components = tuple(float(component) for component in row)  # type: ignore[union-attr]
    except (TypeError, ValueError):
        return None
    if len(components) < 3:
        return None

    rgb = components[:3]
    if any(not math.isfinite(component) or component < 0.0 or component > 1.0 for component in rgb):
        return None
    return "#" + "".join(f"{round(component * 255):02X}" for component in rgb)


def _normalized_color_or_none(color: str) -> str | None:
    qcolor = QColor(color)
    if not qcolor.isValid():
        return None
    return qcolor.name(QColor.NameFormat.HexRgb).upper()
