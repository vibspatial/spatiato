from __future__ import annotations

from collections.abc import Callable
from html import unescape

import numpy as np
import pytest
from napari.layers import Image, Labels, Points, Shapes
from qtpy.QtWidgets import QCheckBox, QWidget

from spatiato.viewer.adapter import BaseLayerBinding, ImageLayerBinding, LabelsLayerBinding, ShapesLayerBinding
from spatiato.widgets.image_layer_row import _ImageLayerRow
from spatiato.widgets.layer_row import _LayerRow


def _labels_binding() -> LabelsLayerBinding:
    return LabelsLayerBinding(
        layer=Labels(np.zeros((4, 4), dtype=np.int32)),
        element_name="blobs_labels",
        coordinate_system="global",
        sdata_id=1,
    )


def _shapes_binding() -> ShapesLayerBinding:
    return ShapesLayerBinding(
        layer=Shapes([np.array([[0.0, 0.0], [0.0, 2.0], [2.0, 2.0]])], shape_type="polygon"),
        element_name="regions",
        coordinate_system="global",
        sdata_id=1,
    )


def _point_radius_shapes_binding() -> ShapesLayerBinding:
    return ShapesLayerBinding(
        layer=Points(np.zeros((2, 2))),
        element_name="circles",
        coordinate_system="global",
        sdata_id=1,
        shapes_rendering_mode="points",
    )


LAYER_BINDING_CASES = [
    pytest.param(_labels_binding, id="labels"),
    pytest.param(_shapes_binding, id="shapes"),
    pytest.param(_point_radius_shapes_binding, id="point-radius-shapes"),
]


def _make_row(binding: BaseLayerBinding) -> _LayerRow:
    return _LayerRow(binding, display_label="cells", accessibility_label="layer cells")


def _layout_widgets(row: _LayerRow) -> list[object]:
    layout = row.layout()
    return [layout.itemAt(index).widget() for index in range(layout.count())]


def _tab_order(row: _LayerRow, widgets: list[QWidget]) -> list[QWidget]:
    """Return `widgets` in the order the Tab key reaches them, following the row's focus chain."""
    order = []
    widget = row.nextInFocusChain()
    while widget is not row:
        if widget in widgets:
            order.append(widget)
        widget = widget.nextInFocusChain()
    return order


@pytest.mark.parametrize("make_binding", LAYER_BINDING_CASES)
def test_layer_row_emits_identity_free_intent_without_mutating_the_layer(
    qtbot,
    make_binding: Callable[[], BaseLayerBinding],
) -> None:
    binding = make_binding()
    row = _make_row(binding)
    visibility_requests: list[bool] = []
    removal_requests: list[str] = []
    qtbot.addWidget(row)
    row.visibility_change_requested.connect(visibility_requests.append)
    row.remove_requested.connect(lambda: removal_requests.append("remove"))

    row.visibility_button.click()
    row.remove_button.click()

    assert row.binding is binding
    assert row.layer_label.text() == "cells"
    assert "layer cells" in row.visibility_button.toolTip()
    assert row.remove_button.accessibleName() == "Remove layer cells from viewer"
    assert visibility_requests == [False]
    assert removal_requests == ["remove"]
    assert binding.layer.visible is True


@pytest.mark.parametrize("make_binding", LAYER_BINDING_CASES)
def test_layer_row_renders_native_visibility_events_until_disposed(
    qtbot,
    make_binding: Callable[[], BaseLayerBinding],
) -> None:
    binding = make_binding()
    row = _make_row(binding)
    visibility_requests: list[bool] = []
    qtbot.addWidget(row)
    row.visibility_change_requested.connect(visibility_requests.append)

    assert row.visibility_button.isChecked()

    binding.layer.visible = False

    assert not row.visibility_button.isChecked()
    assert "Show layer cells" in row.visibility_button.toolTip()
    assert visibility_requests == []

    row.dispose()
    row.dispose()
    binding.layer.visible = True

    assert not row.visibility_button.isChecked()


def test_layer_row_refreshes_its_presentation_from_the_layer(qtbot) -> None:
    """An owner refreshes the row after an assignment that emits no napari event."""
    binding = _labels_binding()
    row = _make_row(binding)
    qtbot.addWidget(row)

    row.visibility_button.click()
    assert not row.visibility_button.isChecked()

    row.refresh_presentation()

    assert row.visibility_button.isChecked()


def test_layer_row_shows_and_clears_a_warning(qtbot) -> None:
    row = _make_row(_labels_binding())
    qtbot.addWidget(row)
    message = "Source `obs: celltype` is no longer available."

    assert row.warning_label.isHidden()
    assert row.warning_message is None

    row.set_warning(message)

    assert not row.warning_label.isHidden()
    assert row.warning_message == message
    assert '"obs: celltype" is no longer available.' in unescape(row.warning_label.toolTip())
    assert row.warning_label.accessibleName() == f"Warning: {message}"
    assert row.layer_label.text() == "cells"

    row.set_warning(None)

    assert row.warning_label.isHidden()
    assert row.warning_message is None
    assert row.warning_label.toolTip() == ""


def test_layer_row_places_added_controls_between_the_label_and_the_warning(qtbot) -> None:
    """Controls are laid out in the order they are added; the owner connects their signals."""
    row = _make_row(_shapes_binding())
    qtbot.addWidget(row)
    fill_checkbox = QCheckBox("Fill")
    other_control = QCheckBox("Other")
    fill_requests: list[bool] = []
    fill_checkbox.toggled.connect(fill_requests.append)

    row.add_control(fill_checkbox)
    row.add_control(other_control)
    fill_checkbox.click()

    assert _layout_widgets(row) == [
        row.visibility_button,
        row.layer_label,
        fill_checkbox,
        other_control,
        row.warning_label,
        row.remove_button,
    ]
    assert fill_checkbox.parent() is row
    assert fill_requests == [True]


def test_layer_row_reaches_added_controls_with_tab_in_their_layout_order(qtbot) -> None:
    """The Tab key goes from the eye button through the controls to the remove button."""
    row = _make_row(_shapes_binding())
    qtbot.addWidget(row)
    fill_checkbox = QCheckBox("Fill")
    other_control = QCheckBox("Other")

    row.add_control(fill_checkbox)
    row.add_control(other_control)

    widgets = [row.visibility_button, fill_checkbox, other_control, row.remove_button]
    assert _tab_order(row, widgets) == widgets


def test_layer_row_has_neutral_object_names(qtbot) -> None:
    row = _make_row(_labels_binding())
    qtbot.addWidget(row)

    assert row.objectName() == "layer_row"
    assert row.visibility_button.objectName() == "layer_visibility_button"
    assert row.layer_label.objectName() == "layer_label"
    assert row.warning_label.objectName() == "layer_warning_label"
    assert row.remove_button.objectName() == "layer_remove_button"


def test_image_layer_row_is_a_layer_row_with_its_color_button_as_a_control(qtbot) -> None:
    """The image row keeps its object names, and its color button sits, and takes Tab focus, where it did."""
    layer = Image(np.zeros((4, 4)), colormap="#00FFFF")
    binding = ImageLayerBinding(
        layer=layer,
        element_name="image",
        coordinate_system="global",
        sdata_id=1,
        image_display_mode="overlay",
        channel_index=0,
        channel_name="DAPI",
    )
    row = _ImageLayerRow(binding, display_label="DAPI", accessibility_label="channel DAPI")
    qtbot.addWidget(row)

    assert isinstance(row, _LayerRow)
    assert row.objectName() == "image_layer_row"
    assert row.visibility_button.objectName() == "image_layer_visibility_button"
    assert row.layer_label.objectName() == "image_layer_label"
    assert row.color_button.objectName() == "image_layer_color_button"
    assert row.remove_button.objectName() == "image_layer_remove_button"
    assert _layout_widgets(row) == [
        row.visibility_button,
        row.layer_label,
        row.color_button,
        row.warning_label,
        row.remove_button,
    ]
    widgets = [row.visibility_button, row.color_button, row.remove_button]
    assert _tab_order(row, widgets) == widgets
