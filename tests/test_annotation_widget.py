from __future__ import annotations

from collections.abc import Callable
from html import unescape
from types import SimpleNamespace

import numpy as np
from napari.layers import Shapes
from qtpy.QtWidgets import QComboBox, QLabel
from spatialdata import SpatialData

import spatiato._app_state as app_state_module
import spatiato.widgets.annotation.widget as annotation_widget_module
import spatiato.widgets.shapes_annotation.widget as shapes_annotation_widget_module
import spatiato.widgets.spatial_query.widget as spatial_query_widget_module
from spatiato._app_state import get_or_create_app_state
from spatiato.widgets.annotation.models import AnnotationContext
from spatiato.widgets.annotation.widget import AnnotationWidget
from spatiato.widgets.shapes_annotation.widget import ShapesAnnotation

_SPACE_PAN_TIP_TEXT = (
    "Tip: while drawing in polygon, path, polyline or lasso mode, hold Space and drag to pan without ending the shape."
)


class DummyEventEmitter:
    def __init__(self) -> None:
        self._callbacks: list[Callable[[object], None]] = []

    def connect(self, callback: Callable[[object], None]) -> None:
        self._callbacks.append(callback)

    def emit(self, value: object | None = None) -> None:
        event = SimpleNamespace(value=value)
        for callback in list(self._callbacks):
            callback(event)


class DummySelection:
    def __init__(self) -> None:
        self.events = SimpleNamespace(active=DummyEventEmitter())
        self._active: object | None = None

    @property
    def active(self) -> object | None:
        return self._active

    @active.setter
    def active(self, layer: object | None) -> None:
        self._active = layer
        self.events.active.emit(layer)

    def select_only(self, layer: object) -> None:
        self.active = layer


class ProxyActiveDummySelection(DummySelection):
    @DummySelection.active.setter
    def active(self, layer: object | None) -> None:
        self._active = None if layer is None else SimpleNamespace(__wrapped__=layer)
        self.events.active.emit(layer)


class DummyLayers(list):
    def __init__(self, *, selection: DummySelection | None = None) -> None:
        super().__init__()
        self.selection = DummySelection() if selection is None else selection
        self.events = SimpleNamespace(
            inserted=DummyEventEmitter(),
            removed=DummyEventEmitter(),
            reordered=DummyEventEmitter(),
        )

    def remove(self, layer: object) -> None:
        super().remove(layer)
        self.events.removed.emit(layer)


class DummyViewer:
    def __init__(self) -> None:
        self.layers = DummyLayers()

    def add_layer(self, layer: object) -> object:
        self.layers.append(layer)
        self.layers.events.inserted.emit(layer)
        return layer


class AutoActivatingDummyViewer(DummyViewer):
    def add_layer(self, layer: object) -> object:
        self.layers.append(layer)
        self.layers.events.inserted.emit(layer)
        self.layers.selection.active = layer
        return layer


class ProxyActiveAutoActivatingDummyViewer(AutoActivatingDummyViewer):
    def __init__(self) -> None:
        self.layers = DummyLayers(selection=ProxyActiveDummySelection())


def _combo_texts(combo: QComboBox) -> list[str]:
    return [combo.itemText(index) for index in range(combo.count())]


def _combo_data(combo: QComboBox) -> list[object]:
    return [combo.itemData(index) for index in range(combo.count())]


def _combo_index_for_text(combo: QComboBox, text: str) -> int:
    for index in range(combo.count()):
        if combo.itemText(index) == text:
            return index
    return -1


def _status_text(widget: ShapesAnnotation) -> str:
    return unescape(widget.status_label.text())


def _clean_tooltip_text(text: str) -> str:
    return unescape(text).replace("&#8203;", "").replace("\u200b", "")


def _patch_coordinate_system_names(monkeypatch, coordinate_systems: list[str]) -> None:
    monkeypatch.setattr(
        annotation_widget_module,
        "get_coordinate_system_names_from_sdata",
        lambda sdata: coordinate_systems,
    )
    monkeypatch.setattr(
        shapes_annotation_widget_module,
        "get_coordinate_system_names_from_sdata",
        lambda sdata: coordinate_systems,
    )
    monkeypatch.setattr(
        app_state_module,
        "get_coordinate_system_names_from_sdata",
        lambda sdata: coordinate_systems,
    )


def _create_embedded_shapes_annotation(qtbot, viewer: DummyViewer | None = None) -> ShapesAnnotation:
    parent = AnnotationWidget(viewer)
    qtbot.addWidget(parent)
    child = parent.shapes_annotation
    # Parent tests inspect the child state reached through the committed shared
    # AnnotationContext, so retain an explicit handle to the owning parent.
    child._test_parent = parent
    return child


def _create_ready_annotation_widget(qtbot, viewer: DummyViewer, sdata: SpatialData) -> ShapesAnnotation:
    app_state = get_or_create_app_state(viewer)
    app_state.set_sdata(sdata)
    widget = _create_embedded_shapes_annotation(qtbot, viewer)
    widget.name_edit.setText("new_regions")
    return widget


def _add_polygon(layer: Shapes, offset: float = 0.0) -> None:
    layer.add_polygons(
        np.asarray(
            [
                [offset + 0.0, 0.0],
                [offset + 0.0, 2.0],
                [offset + 2.0, 2.0],
                [offset + 2.0, 0.0],
            ],
            dtype=float,
        )
    )


def _assert_layer_data_unchanged(layer: Shapes, expected_data: list[np.ndarray]) -> None:
    assert len(layer.data) == len(expected_data)
    for actual_vertices, expected_vertices in zip(layer.data, expected_data, strict=True):
        np.testing.assert_allclose(np.asarray(actual_vertices, dtype=float), expected_vertices)


def test_annotation_widget_starts_inactive_without_spatialdata(qtbot) -> None:
    parent = AnnotationWidget()
    qtbot.addWidget(parent)
    widget = parent.shapes_annotation

    assert widget.app_state.sdata is None
    assert widget.selected_spatialdata is None
    assert widget.selected_coordinate_system is None
    assert widget.validated_shapes_name is None
    header_logo = parent.findChild(QLabel, "annotation_header_logo")
    assert header_logo is not None
    pixmap = header_logo.pixmap()
    assert (pixmap is not None and not pixmap.isNull()) or header_logo.text() == "spatiato"
    assert parent.coordinate_system_combo.minimumWidth() == widget.name_edit.minimumWidth()
    assert parent.coordinate_system_combo.count() == 0
    assert parent.coordinate_system_combo.isEnabled() is False
    assert parent.shapes_combo.count() == 0
    assert parent.shapes_combo.isEnabled() is False
    assert parent.content_layout.indexOf(parent.shapes_annotation) < parent.content_layout.indexOf(parent.spatial_query)
    assert parent.spatial_query.annotation_context is parent.annotation_context
    assert parent.spatial_query.run_button.isEnabled() is False
    assert widget.create_layer_button.isEnabled() is False
    assert widget.create_holes_button.isEnabled() is False
    assert widget.save_shapes_button.isEnabled() is False
    assert "Create an editable annotation Shapes layer" in _clean_tooltip_text(widget.create_layer_button.toolTip())
    create_holes_tooltip = _clean_tooltip_text(widget.create_holes_button.toolTip())
    assert "Select one shell polygon and one or more polygons fully inside it" in create_holes_tooltip
    assert "Shift-click polygons to add them to the selection" in create_holes_tooltip
    assert "Save the current annotation layer back" in _clean_tooltip_text(widget.save_shapes_button.toolTip())
    assert "No SpatialData Loaded" in _status_text(widget)


def test_annotation_widget_destruction_unregisters_coordinate_change_participant(qtbot) -> None:
    """Ensure Qt destruction cannot leave a stale parent-owned participant."""
    viewer = DummyViewer()
    app_state = get_or_create_app_state(viewer)
    widget = AnnotationWidget(viewer)

    assert app_state._coordinate_system_change_participant is widget

    widget.deleteLater()
    qtbot.waitUntil(lambda: app_state._coordinate_system_change_participant is None)


def test_annotation_widget_refreshes_when_shared_sdata_changes(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer()
    app_state = get_or_create_app_state(viewer)
    widget = _create_embedded_shapes_annotation(qtbot, viewer)
    app_state.set_sdata(sdata_blobs)

    assert widget.selected_spatialdata is sdata_blobs
    assert _combo_texts(widget._test_parent.coordinate_system_combo) == ["global"]
    assert _combo_data(widget._test_parent.coordinate_system_combo) == ["global"]
    assert widget._test_parent.coordinate_system_combo.currentText() == "global"
    assert widget.selected_coordinate_system == "global"
    assert widget._test_parent.shapes_combo.currentText() == "Create shapes..."
    assert widget.name_edit.isEnabled() is True
    assert widget.create_layer_button.isEnabled() is False
    assert widget.save_shapes_button.isEnabled() is False
    assert "Shapes element name must not be empty" in _status_text(widget)
    assert widget._test_parent.spatial_query.selected_spatialdata is sdata_blobs
    assert widget._test_parent.spatial_query.selected_coordinate_system == "global"


def test_annotation_widget_shapes_context_updates_reuse_spatial_query_centers_status(
    qtbot,
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    """Shapes-target and dirty-state updates must not check the canonical centers again, which reads the matrix."""
    inspection_count = 0
    real_status = spatial_query_widget_module.canonical_centers_status

    def record_inspection(*args, **kwargs):
        nonlocal inspection_count
        inspection_count += 1
        return real_status(*args, **kwargs)

    monkeypatch.setattr(spatial_query_widget_module, "canonical_centers_status", record_inspection)
    viewer = DummyViewer()
    app_state = get_or_create_app_state(viewer)
    app_state.set_sdata(sdata_blobs)
    parent = AnnotationWidget(viewer)
    qtbot.addWidget(parent)
    spatial_query = parent.spatial_query

    assert spatial_query.app_state is parent.app_state
    assert inspection_count == 0
    assert spatial_query.selected_labels_name is None

    labels_index = spatial_query.labels_combo.findData("blobs_labels")
    assert labels_index >= 0
    spatial_query.labels_combo.setCurrentIndex(labels_index)
    spatial_query.new_column_edit.setText("reviewed_annotation")
    spatial_query.annotation_value_edit.setText("tumor")

    assert inspection_count == 1
    centers_status = spatial_query.centers_status
    selected_labels_name = spatial_query.selected_labels_name
    selected_table_name = spatial_query.selected_table_name

    parent.shapes_combo.setCurrentIndex(_combo_index_for_text(parent.shapes_combo, "blobs_polygons"))

    assert inspection_count == 1
    assert spatial_query.centers_status is centers_status
    assert spatial_query.selected_labels_name == selected_labels_name
    assert spatial_query.selected_table_name == selected_table_name
    assert spatial_query.run_button.isEnabled() is True

    parent.shapes_annotation.edit_session_dirty_changed.emit(True)

    assert parent.annotation_context.has_unsaved_shapes_changes is True
    assert spatial_query.annotation_context is parent.annotation_context
    assert inspection_count == 1
    assert spatial_query.centers_status is centers_status
    assert spatial_query.run_button.isEnabled() is False

    parent.shapes_annotation.edit_session_dirty_changed.emit(False)

    assert parent.annotation_context.has_unsaved_shapes_changes is False
    assert inspection_count == 1
    assert spatial_query.run_button.isEnabled() is True


def test_annotation_widget_spatial_query_labels_activation_preserves_shapes_session(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer()
    app_state = get_or_create_app_state(viewer)
    app_state.set_sdata(sdata_blobs)
    parent = AnnotationWidget(viewer)
    qtbot.addWidget(parent)
    parent.shapes_combo.setCurrentIndex(_combo_index_for_text(parent.shapes_combo, "blobs_polygons"))
    annotation_layer = parent.shapes_annotation._annotation_layer
    annotation_session = parent.shapes_annotation._annotation_session
    assert annotation_layer is not None
    assert annotation_session is not None
    assert parent.spatial_query.labels_combo.count() > 1

    parent.spatial_query.labels_combo.setCurrentIndex(1)

    assert parent.shapes_annotation._annotation_layer is annotation_layer
    assert parent.shapes_annotation._annotation_session is annotation_session
    assert annotation_layer in viewer.layers
    assert viewer.layers.selection.active is not annotation_layer


def test_annotation_widget_shapes_selector_auto_opens_existing_target(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer()
    app_state = get_or_create_app_state(viewer)
    app_state.set_sdata(sdata_blobs)
    widget = _create_embedded_shapes_annotation(qtbot, viewer)
    existing_shapes_name = "blobs_polygons"

    index = _combo_index_for_text(widget._test_parent.shapes_combo, existing_shapes_name)
    assert index >= 0
    widget._test_parent.shapes_combo.setCurrentIndex(index)

    assert len(viewer.layers) == 1
    assert (
        widget._annotation_context.shapes_target
        == shapes_annotation_widget_module._ShapesAnnotationTarget.edit_existing(existing_shapes_name)
    )
    assert widget.validated_shapes_name == existing_shapes_name
    assert widget.name_edit.isHidden() is True
    assert widget.name_edit.isEnabled() is False
    assert widget.create_layer_button.text() == "Create layer"
    assert widget.create_layer_button.isEnabled() is False
    assert widget.save_shapes_button.isEnabled() is False
    status = _status_text(widget)
    assert "Existing Shapes Opened" in status
    assert 'Edit shapes layer "blobs_polygons" in coordinate system "global".' in status
    assert _SPACE_PAN_TIP_TEXT in status


def test_annotation_widget_shapes_selector_defaults_back_to_create_when_existing_disappears(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer()
    app_state = get_or_create_app_state(viewer)
    app_state.set_sdata(sdata_blobs)
    widget = _create_embedded_shapes_annotation(qtbot, viewer)
    existing_shapes_name = next(iter(sdata_blobs.shapes))
    widget._test_parent.shapes_combo.setCurrentIndex(
        _combo_index_for_text(widget._test_parent.shapes_combo, existing_shapes_name)
    )

    sdata_blobs.shapes.pop(existing_shapes_name)
    widget._test_parent.refresh_from_sdata(sdata_blobs)

    assert (
        widget._annotation_context.shapes_target == shapes_annotation_widget_module._ShapesAnnotationTarget.create_new()
    )
    assert widget._test_parent.shapes_combo.currentText() == "Create shapes..."
    assert widget.name_edit.isHidden() is False


def test_annotation_widget_coordinate_change_updates_app_state_and_clears_spatial_query_labels(
    qtbot,
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    _patch_coordinate_system_names(monkeypatch, ["global", "local"])
    viewer = DummyViewer()
    app_state = get_or_create_app_state(viewer)
    app_state.set_sdata(sdata_blobs)
    widget = _create_embedded_shapes_annotation(qtbot, viewer)
    spatial_query = widget._test_parent.spatial_query
    labels_index = spatial_query.labels_combo.findData("blobs_labels")
    assert labels_index >= 0
    spatial_query.labels_combo.setCurrentIndex(labels_index)
    selected_layer = app_state.viewer_adapter.get_loaded_primary_labels_layer(
        sdata_blobs,
        "blobs_labels",
        "global",
    )
    assert selected_layer is not None
    assert spatial_query.centers_status is not None

    widget._test_parent.coordinate_system_combo.setCurrentIndex(1)

    assert app_state.coordinate_system == "local"
    assert widget.selected_coordinate_system == "local"
    assert selected_layer not in viewer.layers
    assert spatial_query.selected_labels_name is None
    assert spatial_query.selected_table_name is None
    assert spatial_query.centers_status is None
    assert spatial_query.run_button.isEnabled() is False


def test_annotation_widget_external_coordinate_system_change_updates_selector(
    qtbot,
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    _patch_coordinate_system_names(monkeypatch, ["global", "local"])
    viewer = DummyViewer()
    app_state = get_or_create_app_state(viewer)
    app_state.set_sdata(sdata_blobs)
    widget = _create_embedded_shapes_annotation(qtbot, viewer)
    app_state.set_coordinate_system("local", source="viewer_widget")

    assert widget._test_parent.coordinate_system_combo.currentText() == "local"
    assert widget.selected_coordinate_system == "local"


def test_annotation_widget_disables_create_when_coordinate_system_is_cleared(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer()
    app_state = get_or_create_app_state(viewer)
    app_state.set_sdata(sdata_blobs)
    widget = _create_embedded_shapes_annotation(qtbot, viewer)
    widget.name_edit.setText("new_regions")
    app_state.clear_coordinate_system(source="test")

    assert widget._test_parent.coordinate_system_combo.currentIndex() == -1
    assert widget.selected_coordinate_system is None
    assert widget.create_layer_button.isEnabled() is False
    assert "Choose Coordinate System" in _status_text(widget)


def test_annotation_widget_cancelling_coordinate_change_preserves_annotation_layer(
    qtbot,
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    _patch_coordinate_system_names(monkeypatch, ["global", "local"])
    viewer = DummyViewer()
    widget = _create_ready_annotation_widget(qtbot, viewer, sdata_blobs)
    widget.create_layer_button.click()
    layer = viewer.layers[0]
    _add_polygon(layer)
    discard_contexts: list[str] = []
    parent = widget._test_parent
    assert parent.annotation_context.has_unsaved_shapes_changes is True
    previous_context = parent.annotation_context
    published_contexts: list[object] = []
    parent.annotation_context_changed.connect(published_contexts.append)

    def cancel_discard(*, reason: str) -> bool:
        discard_contexts.append(reason)
        return False

    monkeypatch.setattr(widget, "_confirm_discard_annotation_layer", cancel_discard)

    widget._test_parent.coordinate_system_combo.setCurrentIndex(1)

    assert discard_contexts == ["coordinate_system"]
    assert published_contexts == []
    assert parent.annotation_context is previous_context
    assert widget.app_state.coordinate_system == "global"
    assert widget._test_parent.coordinate_system_combo.currentText() == "global"
    assert widget.selected_coordinate_system == "global"
    assert list(viewer.layers) == [layer]
    assert widget.app_state.viewer_adapter.layer_bindings.get_binding(layer) is not None
    assert widget._annotation_layer is layer
    assert widget.name_edit.isEnabled() is False
    assert widget.create_layer_button.isEnabled() is False
    assert widget.save_shapes_button.isEnabled() is True


def test_annotation_widget_accepted_dirty_coordinate_change_publishes_only_final_context(
    qtbot,
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    """Do not publish the intermediate clean state of the old context.

    Accepting discard closes the dirty Shapes child synchronously before the
    shared coordinate system changes. The parent blocks that intermediate
    ``dirty=False`` signal, then publishes one final context containing the
    newly accepted coordinate system and the child's final clean state.
    """
    _patch_coordinate_system_names(monkeypatch, ["global", "local"])
    viewer = DummyViewer()
    widget = _create_ready_annotation_widget(qtbot, viewer, sdata_blobs)
    widget.create_layer_button.click()
    _add_polygon(viewer.layers[0])
    parent = widget._test_parent
    assert parent.annotation_context.has_unsaved_shapes_changes is True
    published_contexts: list[AnnotationContext] = []
    parent.annotation_context_changed.connect(published_contexts.append)
    monkeypatch.setattr(widget, "_confirm_discard_annotation_layer", lambda *, reason: True)

    parent.coordinate_system_combo.setCurrentIndex(1)

    assert len(published_contexts) == 1
    assert published_contexts[0] is parent.annotation_context
    assert parent.annotation_context.coordinate_system == "local"
    assert parent.annotation_context.has_unsaved_shapes_changes is False


def test_annotation_widget_rejects_external_coordinate_change_before_losing_unsaved_geometry(
    qtbot,
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    _patch_coordinate_system_names(monkeypatch, ["global", "local"])
    viewer = DummyViewer()
    widget = _create_ready_annotation_widget(qtbot, viewer, sdata_blobs)
    widget.create_layer_button.click()
    layer = viewer.layers[0]
    _add_polygon(layer)
    expected_data = [np.asarray(vertices, dtype=float).copy() for vertices in layer.data]
    discard_contexts: list[str] = []
    coordinate_events: list[object] = []
    widget.app_state.coordinate_system_changed.connect(coordinate_events.append)

    def cancel_discard(*, reason: str) -> bool:
        discard_contexts.append(reason)
        return False

    monkeypatch.setattr(widget, "_confirm_discard_annotation_layer", cancel_discard)

    changed = widget.app_state.set_coordinate_system("local", source="object_classification_widget")

    assert changed is False
    assert discard_contexts == ["coordinate_system"]
    assert coordinate_events == []
    assert widget.app_state.coordinate_system == "global"
    assert widget._test_parent.coordinate_system_combo.currentText() == "global"
    assert list(viewer.layers) == [layer]
    assert widget._annotation_layer is layer
    assert widget.app_state.viewer_adapter.layer_bindings.get_binding(layer) is not None
    _assert_layer_data_unchanged(layer, expected_data)


def test_annotation_widget_clean_coordinate_change_closes_empty_create_layer_without_warning(
    qtbot,
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    _patch_coordinate_system_names(monkeypatch, ["global", "local"])
    viewer = DummyViewer()
    widget = _create_ready_annotation_widget(qtbot, viewer, sdata_blobs)
    widget.create_layer_button.click()
    layer = viewer.layers[0]

    def fail_if_confirmed(*, reason: str) -> bool:
        raise AssertionError(f"Clean coordinate-system switch should not warn: {reason}")

    monkeypatch.setattr(widget, "_confirm_discard_annotation_layer", fail_if_confirmed)

    widget._test_parent.coordinate_system_combo.setCurrentIndex(1)

    assert widget.app_state.coordinate_system == "local"
    assert widget._test_parent.coordinate_system_combo.currentText() == "local"
    assert widget.selected_coordinate_system == "local"
    assert list(viewer.layers) == []
    assert widget.app_state.viewer_adapter.layer_bindings.get_binding(layer) is None
    assert widget._annotation_layer is None
    assert widget._annotation_edit_guard.layer is None
    assert "_drag_modes" not in vars(layer)
    assert widget._annotation_shapes_name is None
    assert widget._annotation_coordinate_system is None
    assert widget._annotation_session is None
    assert widget.name_edit.isEnabled() is True
    assert widget.create_layer_button.isEnabled() is True
    assert widget.save_shapes_button.isEnabled() is False
    assert "Ready" in _status_text(widget)


def test_annotation_widget_cancelling_target_change_preserves_annotation_layer(
    qtbot,
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer()
    widget = _create_ready_annotation_widget(qtbot, viewer, sdata_blobs)
    widget.create_layer_button.click()
    layer = viewer.layers[0]
    existing_shapes_name = "blobs_polygons"
    _add_polygon(layer)
    discard_contexts: list[str] = []

    def cancel_discard(*, reason: str) -> bool:
        discard_contexts.append(reason)
        return False

    monkeypatch.setattr(widget, "_confirm_discard_annotation_layer", cancel_discard)

    widget._test_parent.shapes_combo.setCurrentIndex(
        _combo_index_for_text(widget._test_parent.shapes_combo, existing_shapes_name)
    )

    assert discard_contexts == ["shapes_target"]
    assert widget._test_parent.shapes_combo.currentText() == "Create shapes..."
    assert (
        widget._annotation_context.shapes_target == shapes_annotation_widget_module._ShapesAnnotationTarget.create_new()
    )
    assert list(viewer.layers) == [layer]
    assert widget.app_state.viewer_adapter.layer_bindings.get_binding(layer) is not None
    assert widget._annotation_layer is layer
    assert widget.name_edit.isHidden() is False
    assert widget.name_edit.isEnabled() is False
    assert widget.create_layer_button.isEnabled() is False
    assert widget.save_shapes_button.isEnabled() is True


def test_annotation_widget_clean_target_change_closes_empty_create_layer_without_warning(
    qtbot,
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer()
    widget = _create_ready_annotation_widget(qtbot, viewer, sdata_blobs)
    widget.create_layer_button.click()
    layer = viewer.layers[0]
    existing_shapes_name = "blobs_polygons"

    def fail_if_confirmed(*, reason: str) -> bool:
        raise AssertionError(f"Clean target switch should not warn: {reason}")

    monkeypatch.setattr(widget, "_confirm_discard_annotation_layer", fail_if_confirmed)

    widget._test_parent.shapes_combo.setCurrentIndex(
        _combo_index_for_text(widget._test_parent.shapes_combo, existing_shapes_name)
    )

    assert len(viewer.layers) == 1
    opened_layer = viewer.layers[0]
    assert opened_layer is not layer
    assert isinstance(opened_layer, Shapes)
    assert widget.app_state.viewer_adapter.layer_bindings.get_binding(layer) is None
    assert widget._annotation_layer is opened_layer
    assert widget._annotation_shapes_name == existing_shapes_name
    assert widget._annotation_coordinate_system == "global"
    assert (
        widget._annotation_context.shapes_target
        == shapes_annotation_widget_module._ShapesAnnotationTarget.edit_existing(existing_shapes_name)
    )
    assert widget.name_edit.isHidden() is True
    assert widget.create_layer_button.text() == "Create layer"
    assert widget.create_layer_button.isEnabled() is False
    assert widget.save_shapes_button.isEnabled() is False
    assert "Existing Shapes Opened" in _status_text(widget)


def test_annotation_widget_clean_saved_target_change_keeps_saved_layer_without_warning(
    qtbot,
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer()
    widget = _create_ready_annotation_widget(qtbot, viewer, sdata_blobs)
    spatial_query = widget._test_parent.spatial_query
    labels_index = spatial_query.labels_combo.findData("blobs_labels")
    assert labels_index >= 0
    spatial_query.labels_combo.setCurrentIndex(labels_index)
    spatial_query.new_column_edit.setText("reviewed_annotation")
    spatial_query.annotation_value_edit.setText("tumor")
    spatial_query_centers_status = spatial_query.centers_status
    assert spatial_query_centers_status is not None
    widget.create_layer_button.click()
    saved_layer = widget._annotation_layer
    assert saved_layer is not None
    _add_polygon(saved_layer)
    widget.save_shapes_button.click()

    assert widget._test_parent.annotation_context.saved_shapes_name == "new_regions"
    assert widget._test_parent.spatial_query.annotation_context is widget._test_parent.annotation_context
    assert widget._test_parent.spatial_query.centers_status is spatial_query_centers_status
    assert widget._test_parent.spatial_query.run_button.isEnabled() is True

    def fail_if_confirmed(*, reason: str) -> bool:
        raise AssertionError(f"Clean saved target switch should not warn: {reason}")

    monkeypatch.setattr(widget, "_confirm_discard_annotation_layer", fail_if_confirmed)

    widget._test_parent.shapes_combo.setCurrentIndex(
        _combo_index_for_text(widget._test_parent.shapes_combo, "blobs_polygons")
    )

    assert saved_layer in viewer.layers
    assert widget.app_state.viewer_adapter.layer_bindings.get_binding(saved_layer) is not None
    assert widget._annotation_shapes_name == "blobs_polygons"
    assert widget._annotation_layer is not saved_layer
    assert len([layer for layer in viewer.layers if isinstance(layer, Shapes)]) == 2


def test_annotation_widget_clean_existing_target_switch_preserves_layer_order(
    qtbot,
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    sdata_blobs.shapes["other_polygons"] = sdata_blobs.shapes["blobs_polygons"].copy()
    viewer = DummyViewer()
    app_state = get_or_create_app_state(viewer)
    app_state.set_sdata(sdata_blobs)
    widget = _create_embedded_shapes_annotation(qtbot, viewer)

    def fail_if_confirmed(*, reason: str) -> bool:
        raise AssertionError(f"Clean existing target switch should not warn: {reason}")

    monkeypatch.setattr(widget, "_confirm_discard_annotation_layer", fail_if_confirmed)

    widget._test_parent.shapes_combo.setCurrentIndex(
        _combo_index_for_text(widget._test_parent.shapes_combo, "blobs_polygons")
    )
    blobs_layer = widget._annotation_layer
    assert blobs_layer is not None
    assert widget._annotation_edit_guard.layer is blobs_layer
    assert "_drag_modes" in vars(blobs_layer)

    widget._test_parent.shapes_combo.setCurrentIndex(
        _combo_index_for_text(widget._test_parent.shapes_combo, "other_polygons")
    )
    other_layer = widget._annotation_layer
    assert other_layer is not None
    assert other_layer is not blobs_layer
    assert list(viewer.layers) == [blobs_layer, other_layer]
    assert widget._annotation_edit_guard.layer is other_layer
    assert "_drag_modes" not in vars(blobs_layer)
    assert "_drag_modes" in vars(other_layer)

    widget._test_parent.shapes_combo.setCurrentIndex(
        _combo_index_for_text(widget._test_parent.shapes_combo, "blobs_polygons")
    )

    assert widget._annotation_layer is blobs_layer
    assert list(viewer.layers) == [blobs_layer, other_layer]
    assert widget._annotation_edit_guard.layer is blobs_layer
    assert "_drag_modes" in vars(blobs_layer)
    assert "_drag_modes" not in vars(other_layer)


def test_annotation_widget_dirty_existing_target_switch_ignores_reloaded_old_active_layer(
    qtbot,
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    sdata_blobs.shapes["other_polygons"] = sdata_blobs.shapes["blobs_polygons"].copy()
    viewer = ProxyActiveAutoActivatingDummyViewer()
    app_state = get_or_create_app_state(viewer)
    app_state.set_sdata(sdata_blobs)
    widget = _create_embedded_shapes_annotation(qtbot, viewer)

    widget._test_parent.shapes_combo.setCurrentIndex(
        _combo_index_for_text(widget._test_parent.shapes_combo, "blobs_polygons")
    )
    dirty_layer = widget._annotation_layer
    assert isinstance(dirty_layer, Shapes)
    assert widget._annotation_session is not None
    _add_polygon(dirty_layer, offset=100)
    assert widget._annotation_layer_has_unsaved_changes() is True
    discard_contexts: list[str] = []
    parent = widget._test_parent
    published_contexts: list[AnnotationContext] = []
    parent.annotation_context_changed.connect(published_contexts.append)

    def confirm_discard(*, reason: str) -> bool:
        discard_contexts.append(reason)
        return True

    monkeypatch.setattr(widget, "_confirm_discard_annotation_layer", confirm_discard)

    widget._test_parent.shapes_combo.setCurrentIndex(
        _combo_index_for_text(widget._test_parent.shapes_combo, "other_polygons")
    )

    assert discard_contexts == ["shapes_target"]
    assert dirty_layer not in viewer.layers
    assert widget._annotation_layer is not None
    assert widget._annotation_layer is not dirty_layer
    assert widget._annotation_session is not None
    assert widget._annotation_session.shapes_name == "other_polygons"
    assert widget._test_parent.shapes_combo.currentText() == "other_polygons"
    assert (
        getattr(viewer.layers.selection.active, "__wrapped__", viewer.layers.selection.active)
        is widget._annotation_layer
    )
    assert len(published_contexts) == 1
    assert published_contexts[0] is parent.annotation_context
    assert parent.annotation_context.saved_shapes_name == "other_polygons"
    assert parent.annotation_context.has_unsaved_shapes_changes is False
