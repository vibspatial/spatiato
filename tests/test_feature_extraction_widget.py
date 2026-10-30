from __future__ import annotations

from html import unescape
from types import SimpleNamespace

import numpy as np
import pytest
from napari.layers import Labels
from qtpy.QtCore import QObject, Signal
from qtpy.QtWidgets import QCheckBox, QComboBox, QLabel, QLineEdit, QScrollArea
from spatialdata import SpatialData

import spatiato.widgets.feature_extraction.controller as feature_extraction_controller_module
import spatiato.widgets.feature_extraction.widget as feature_extraction_widget_module
from spatiato._app_state import TableDirtySnapshot, TableStateChangedEvent, get_or_create_app_state
from spatiato.core.feature_extraction import calculate_feature_matrix
from spatiato.core.persistence import TableComponentPath
from spatiato.core.spatialdata import (
    SpatialDataFeatureExtractionImageDiscovery,
    SpatialDataFeatureExtractionLabelDiscovery,
    SpatialDataImageOption,
    SpatialDataLabelsOption,
)
from spatiato.widgets.feature_extraction.controller import (
    FeatureExtractionBindingState,
    FeatureExtractionResult,
    FeatureExtractionTriplet,
)
from spatiato.widgets.feature_extraction.widget import FeatureExtractionWidget
from spatiato.widgets.persistence.controller import PersistenceController
from spatiato.widgets.persistence.controls import TablePersistenceControls
from spatiato.widgets.viewer.widget import ViewerWidget


class DummyEventEmitter:
    def __init__(self) -> None:
        self._callbacks = []

    def connect(self, callback) -> None:
        self._callbacks.append(callback)

    def emit(self, value: object | None = None) -> None:
        event = SimpleNamespace(value=value)
        for callback in list(self._callbacks):
            callback(event)


class DummyLayers(list):
    def __init__(self, layers: list[Labels] | None = None) -> None:
        super().__init__(layers or [])
        self.events = SimpleNamespace(
            inserted=DummyEventEmitter(),
            removed=DummyEventEmitter(),
            reordered=DummyEventEmitter(),
        )


class DummyViewer:
    def __init__(self, layers: list[Labels] | None = None) -> None:
        self.layers = DummyLayers(layers)


class _DeferredWorker(QObject):
    returned = Signal(object)
    errored = Signal(object)
    finished = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.started = False
        self.quit_called = False

    def start(self) -> None:
        self.started = True

    def quit(self) -> None:
        self.quit_called = True


def get_coordinate_system_checkbox(widget: FeatureExtractionWidget, coordinate_system: str) -> QCheckBox:
    return widget._coordinate_system_checkboxes[coordinate_system]


def _tooltip_text(widget) -> str:
    return unescape(widget.toolTip()).replace("&#8203;", "").replace("\u200b", "")


def check_coordinate_system(widget: FeatureExtractionWidget, coordinate_system: str) -> None:
    get_coordinate_system_checkbox(widget, coordinate_system).setChecked(True)


def select_segmentation(widget: FeatureExtractionWidget, coordinate_system: str, index: int) -> None:
    widget._triplet_card_widgets_by_coordinate_system[coordinate_system].segmentation_combo.setCurrentIndex(index)


def combo_texts(combo: QComboBox) -> list[str]:
    return [combo.itemText(index) for index in range(combo.count())]


def make_label_discovery(
    *,
    coordinate_system: str,
    options: list[SpatialDataLabelsOption],
    unavailable_label_count: int = 0,
) -> SpatialDataFeatureExtractionLabelDiscovery:
    return SpatialDataFeatureExtractionLabelDiscovery(
        coordinate_system=coordinate_system,
        eligible_label_options=options,
        coordinate_system_labels_count=len(options) + unavailable_label_count,
        unavailable_label_count=unavailable_label_count,
    )


def make_image_discovery(
    *,
    coordinate_system: str,
    labels_name: str,
    options: list[SpatialDataImageOption],
    unavailable_image_count: int = 0,
) -> SpatialDataFeatureExtractionImageDiscovery:
    return SpatialDataFeatureExtractionImageDiscovery(
        coordinate_system=coordinate_system,
        labels_name=labels_name,
        eligible_image_options=options,
        coordinate_system_image_count=len(options) + unavailable_image_count,
        unavailable_image_count=unavailable_image_count,
    )


def make_viewer_with_shared_sdata(sdata: SpatialData) -> DummyViewer:
    viewer = DummyViewer()
    get_or_create_app_state(viewer).set_sdata(sdata)
    return viewer


def make_blobs_labels_layer(sdata: SpatialData, labels_name: str = "blobs_labels") -> Labels:
    return Labels(
        sdata.labels[labels_name],
        name=labels_name,
        metadata={"sdata": sdata, "name": labels_name},
    )


def test_feature_extraction_widget_can_be_instantiated(qtbot) -> None:
    widget = FeatureExtractionWidget()

    qtbot.addWidget(widget)

    assert widget is not None
    assert widget.findChild(QLabel, "feature_extraction_header_logo") is not None
    assert widget.selected_segmentation_name is None
    assert widget.selected_spatialdata is None
    assert widget.selected_image_name is None
    assert widget.selected_table_name is None
    assert widget.selected_table_mode is None
    assert widget.selected_new_table_name is None
    assert widget.selected_coordinate_system is None
    assert widget.selected_feature_names == ()
    assert widget.selected_feature_key == "features"
    assert widget.overwrite_feature_key is False
    assert widget.segmentation_combo.count() == 0
    assert widget.image_combo.count() == 1
    assert widget.image_combo.itemText(0) == "No image"
    assert widget.table_combo.count() == 0
    assert widget.findChild(QLineEdit, "feature_extraction_new_table_name_line_edit").isHidden()
    assert widget.coordinate_system_combo.count() == 0
    assert widget.calculate_button.isEnabled() is False
    assert "No SpatialData Loaded" in widget.selection_status.text()
    assert "shared Spatiato state" in unescape(widget.selection_status.text())
    assert all(button.text() != "Rescan Viewer" for button in widget.findChildren(type(widget.calculate_button)))
    assert (
        widget.segmentation_combo.sizeAdjustPolicy() == QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
    )
    assert widget.image_combo.sizeAdjustPolicy() == QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
    assert widget.table_combo.sizeAdjustPolicy() == QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
    assert widget.coordinate_system_combo.sizeAdjustPolicy() == (
        QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
    )


def test_feature_extraction_widget_destruction_retains_reload_guard_until_worker_finishes(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    sdata_blobs: SpatialData,
) -> None:
    """A deleted QWidget must not expose a table still targeted by its Harpy worker to Reload."""
    viewer = make_viewer_with_shared_sdata(sdata_blobs)
    app_state = get_or_create_app_state(viewer)
    widget = FeatureExtractionWidget(viewer)
    qtbot.addWidget(widget)
    controller = widget._feature_extraction_controller
    worker = _DeferredWorker()
    monkeypatch.setattr(controller, "_create_feature_extraction_worker", lambda _job: worker)
    check_coordinate_system(widget, "global")
    select_segmentation(widget, "global", 0)
    widget.findChild(QCheckBox, "feature_checkbox_area").setChecked(True)

    assert controller in app_state._table_reload_participants
    assert widget.calculate_button.isEnabled()

    widget.calculate_button.click()
    widget.deleteLater()
    qtbot.waitUntil(lambda: worker.quit_called)

    assert controller in app_state._table_reload_participants
    assert controller.has_job_in_flight is True

    worker.finished.emit()
    qtbot.waitUntil(lambda: controller not in app_state._table_reload_participants)

    assert controller.has_job_in_flight is False


def test_feature_extraction_widget_blocks_matching_shared_reload_with_user_feedback(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    backed_sdata_blobs: SpatialData,
) -> None:
    viewer = make_viewer_with_shared_sdata(backed_sdata_blobs)
    app_state = get_or_create_app_state(viewer)
    widget = FeatureExtractionWidget(viewer)
    controls = TablePersistenceControls(app_state)
    qtbot.addWidget(widget)
    qtbot.addWidget(controls)
    worker = _DeferredWorker()
    monkeypatch.setattr(widget._feature_extraction_controller, "_create_feature_extraction_worker", lambda _job: worker)
    check_coordinate_system(widget, "global")
    select_segmentation(widget, "global", 0)
    widget.findChild(QCheckBox, "feature_checkbox_area").setChecked(True)
    controls.bind(backed_sdata_blobs, "table", "blobs_labels")
    reload_events: list[TableStateChangedEvent] = []
    app_state.table_state_changed.connect(
        lambda event: reload_events.append(event) if event.change_kind == "reloaded" else None
    )

    widget.calculate_button.click()
    controls.reload_button.click()

    assert reload_events == []
    assert "Feature Extraction is still writing it" in unescape(controls.feedback_label.text())
    assert "Wait for Feature Extraction to finish" in unescape(controls.feedback_label.text())

    worker.finished.emit()
    controls.reload_button.click()

    assert len(reload_events) == 1
    assert "Reloaded" in unescape(controls.feedback_label.text())


def test_feature_extraction_widget_seeds_from_shared_sdata_on_construction(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = make_viewer_with_shared_sdata(sdata_blobs)

    widget = FeatureExtractionWidget(viewer)

    qtbot.addWidget(widget)

    assert widget.coordinate_system_combo.count() == 1
    assert widget.coordinate_system_combo.currentIndex() == -1
    assert get_coordinate_system_checkbox(widget, "global").isChecked() is False
    assert widget.selected_coordinate_system is None
    assert widget.selected_segmentation_name is None
    assert widget.selected_image_name is None
    assert not widget._triplet_card_widgets_by_coordinate_system
    assert "Choose Coordinate Systems" in widget.selection_status.text()
    assert widget.selected_spatialdata is sdata_blobs


def test_feature_extraction_widget_shortens_long_coordinate_system_checkbox_names(
    qtbot,
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    coordinate_system = "global_long_coordinate_system_name_" + "x" * 80
    monkeypatch.setattr(
        feature_extraction_widget_module,
        "get_coordinate_system_names_from_sdata",
        lambda sdata: [coordinate_system],
    )
    viewer = make_viewer_with_shared_sdata(sdata_blobs)

    widget = FeatureExtractionWidget(viewer)
    qtbot.addWidget(widget)

    checkbox = get_coordinate_system_checkbox(widget, coordinate_system)
    assert checkbox.text() != coordinate_system
    assert "…" in checkbox.text()
    assert coordinate_system in _tooltip_text(checkbox)

    check_coordinate_system(widget, coordinate_system)

    triplet_card = widget._triplet_card_widgets_by_coordinate_system[coordinate_system].container
    assert triplet_card.title() != coordinate_system
    assert "…" in triplet_card.title()
    assert coordinate_system in _tooltip_text(triplet_card)

    selection_status_text = unescape(widget.selection_status.text())
    assert "Batch Incomplete" in selection_status_text
    assert coordinate_system not in selection_status_text
    assert "…" in selection_status_text
    assert coordinate_system in _tooltip_text(widget.selection_status)


def test_feature_extraction_widget_refreshes_when_shared_sdata_changes(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer()
    app_state = get_or_create_app_state(viewer)
    widget = FeatureExtractionWidget(viewer)

    qtbot.addWidget(widget)

    assert widget.segmentation_combo.count() == 0
    assert "No SpatialData Loaded" in widget.selection_status.text()

    app_state.set_sdata(sdata_blobs)

    assert widget.coordinate_system_combo.count() == 1
    assert widget.coordinate_system_combo.currentIndex() == -1
    assert get_coordinate_system_checkbox(widget, "global").isChecked() is False
    assert widget.selected_coordinate_system is None
    assert widget.selected_segmentation_name is None
    assert widget.selected_image_name is None
    assert not widget._triplet_card_widgets_by_coordinate_system
    assert "Choose Coordinate Systems" in widget.selection_status.text()
    assert widget.selected_spatialdata is sdata_blobs


def test_feature_extraction_widget_populates_selector_flow_from_spatialdata(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = make_viewer_with_shared_sdata(sdata_blobs)
    widget = FeatureExtractionWidget(viewer)

    qtbot.addWidget(widget)

    assert widget.coordinate_system_combo.count() == 1
    assert widget.coordinate_system_combo.currentIndex() == -1
    assert get_coordinate_system_checkbox(widget, "global").isChecked() is False
    assert widget.selected_coordinate_system is None
    assert not widget._triplet_card_widgets_by_coordinate_system
    assert "Choose Coordinate Systems" in widget.selection_status.text()

    check_coordinate_system(widget, "global")

    assert widget.segmentation_combo.count() == 2
    assert [widget.segmentation_combo.itemText(index) for index in range(widget.segmentation_combo.count())] == [
        "blobs_labels",
        "blobs_multiscale_labels",
    ]
    assert widget.segmentation_combo.currentIndex() == -1
    assert widget.segmentation_combo.placeholderText() == "Choose a labels element"
    assert widget.image_combo.count() == 1
    assert [widget.image_combo.itemText(index) for index in range(widget.image_combo.count())] == ["No image"]
    assert widget.table_combo.count() == 0
    assert widget.new_table_name_line_edit.isHidden()
    assert widget.coordinate_system_combo.count() == 1
    assert widget.coordinate_system_combo.itemText(0) == "global"
    assert widget.selected_segmentation_name is None
    assert widget.selected_spatialdata is sdata_blobs
    assert widget.selected_image_name is None
    assert widget.selected_table_name is None
    assert widget.selected_coordinate_system == "global"
    assert "Batch Incomplete" in widget.selection_status.text()
    assert widget.selection_status.toolTip() == ""


def test_feature_extraction_widget_filters_labels_and_images_by_coordinate_system(
    qtbot,
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    viewer = make_viewer_with_shared_sdata(sdata_blobs)

    monkeypatch.setattr(
        feature_extraction_widget_module,
        "get_coordinate_system_names_from_sdata",
        lambda sdata: ["aligned", "global"],
    )
    monkeypatch.setattr(
        feature_extraction_widget_module,
        "get_spatialdata_feature_extraction_label_discovery_for_coordinate_system_from_sdata",
        lambda *, sdata, coordinate_system: make_label_discovery(
            coordinate_system=coordinate_system,
            options=[
                SpatialDataLabelsOption(
                    labels_name=f"labels_{coordinate_system}",
                    display_name=f"labels_{coordinate_system}",
                    sdata=sdata,
                    coordinate_systems=(coordinate_system,),
                )
            ],
        ),
    )
    monkeypatch.setattr(
        feature_extraction_widget_module,
        "get_spatialdata_feature_extraction_image_discovery_for_coordinate_system_and_label_from_sdata",
        lambda *, sdata, coordinate_system, labels_name: make_image_discovery(
            coordinate_system=coordinate_system,
            labels_name=labels_name,
            options=[
                SpatialDataImageOption(
                    image_name=f"image_{coordinate_system}_{labels_name}",
                    display_name=f"image_{coordinate_system}_{labels_name}",
                    sdata=sdata,
                    coordinate_systems=(coordinate_system,),
                )
            ],
        ),
    )
    monkeypatch.setattr(
        feature_extraction_widget_module,
        "get_annotating_table_names",
        lambda sdata, labels_name: ["table"],
    )
    monkeypatch.setattr(
        feature_extraction_widget_module,
        "validate_table_annotation_coverage",
        lambda sdata, table_name, labels_names: None,
    )
    monkeypatch.setattr(
        feature_extraction_widget_module,
        "validate_table_region_instance_ids",
        lambda sdata, table_name, *, labels_names=None: None,
    )

    widget = FeatureExtractionWidget(viewer)
    qtbot.addWidget(widget)

    assert widget.coordinate_system_combo.count() == 2
    assert widget.coordinate_system_combo.itemText(0) == "aligned"
    assert widget.coordinate_system_combo.currentIndex() == -1
    assert widget.selected_coordinate_system is None
    assert widget.selected_segmentation_name is None
    assert not widget._triplet_card_widgets_by_coordinate_system

    check_coordinate_system(widget, "aligned")

    assert widget.selected_coordinate_system == "aligned"
    assert widget.selected_segmentation_name is None
    assert [widget.segmentation_combo.itemText(index) for index in range(widget.segmentation_combo.count())] == [
        "labels_aligned"
    ]
    assert widget.segmentation_combo.currentIndex() == -1
    assert [widget.image_combo.itemText(index) for index in range(widget.image_combo.count())] == ["No image"]
    select_segmentation(widget, "aligned", 0)
    assert widget.selected_segmentation_name == "labels_aligned"
    assert [widget.image_combo.itemText(index) for index in range(widget.image_combo.count())] == [
        "No image",
        "image_aligned_labels_aligned",
    ]
    assert widget.selected_image_name is None
    assert combo_texts(widget.table_combo) == ["table", "Create table..."]
    assert widget.selected_table_mode == "existing"
    assert widget.selected_table_name == "table"

    check_coordinate_system(widget, "global")

    assert widget.selected_coordinate_system == "global"
    assert widget.selected_segmentation_name is None
    assert [widget.segmentation_combo.itemText(index) for index in range(widget.segmentation_combo.count())] == [
        "labels_global"
    ]
    assert widget.segmentation_combo.currentIndex() == -1
    assert [widget.image_combo.itemText(index) for index in range(widget.image_combo.count())] == ["No image"]
    select_segmentation(widget, "global", 0)
    assert widget.selected_segmentation_name == "labels_global"
    assert [widget.image_combo.itemText(index) for index in range(widget.image_combo.count())] == [
        "No image",
        "image_global_labels_global",
    ]
    assert widget.selected_image_name is None
    assert combo_texts(widget.table_combo) == ["table", "Create table..."]
    assert widget.selected_table_mode == "existing"
    assert widget.selected_table_name == "table"


def test_feature_extraction_widget_renders_one_card_per_checked_coordinate_system(
    qtbot,
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    viewer = make_viewer_with_shared_sdata(sdata_blobs)

    monkeypatch.setattr(
        feature_extraction_widget_module,
        "get_coordinate_system_names_from_sdata",
        lambda sdata: ["aligned", "global"],
    )
    monkeypatch.setattr(
        feature_extraction_widget_module,
        "get_spatialdata_feature_extraction_label_discovery_for_coordinate_system_from_sdata",
        lambda *, sdata, coordinate_system: make_label_discovery(
            coordinate_system=coordinate_system,
            options=[
                SpatialDataLabelsOption(
                    labels_name=f"labels_{coordinate_system}",
                    display_name=f"labels_{coordinate_system}",
                    sdata=sdata,
                    coordinate_systems=(coordinate_system,),
                )
            ],
        ),
    )
    monkeypatch.setattr(
        feature_extraction_widget_module,
        "get_spatialdata_feature_extraction_image_discovery_for_coordinate_system_and_label_from_sdata",
        lambda *, sdata, coordinate_system, labels_name: make_image_discovery(
            coordinate_system=coordinate_system,
            labels_name=labels_name,
            options=[
                SpatialDataImageOption(
                    image_name=f"image_{coordinate_system}_{labels_name}",
                    display_name=f"image_{coordinate_system}_{labels_name}",
                    sdata=sdata,
                    coordinate_systems=(coordinate_system,),
                )
            ],
        ),
    )
    monkeypatch.setattr(
        feature_extraction_widget_module,
        "get_annotating_table_names",
        lambda sdata, labels_name: ["table"],
    )
    monkeypatch.setattr(
        feature_extraction_widget_module,
        "validate_table_annotation_coverage",
        lambda sdata, table_name, labels_names: None,
    )
    monkeypatch.setattr(
        feature_extraction_widget_module,
        "validate_table_region_instance_ids",
        lambda sdata, table_name, *, labels_names=None: None,
    )

    widget = FeatureExtractionWidget(viewer)
    qtbot.addWidget(widget)

    assert list(widget._coordinate_system_checkboxes) == ["aligned", "global"]
    assert get_coordinate_system_checkbox(widget, "aligned").isChecked() is False
    assert get_coordinate_system_checkbox(widget, "global").isChecked() is False
    assert list(widget._triplet_card_widgets_by_coordinate_system) == []
    assert widget.selected_coordinate_system is None

    check_coordinate_system(widget, "aligned")
    assert list(widget._triplet_card_widgets_by_coordinate_system) == ["aligned"]
    assert widget.selected_coordinate_system == "aligned"

    check_coordinate_system(widget, "global")

    assert list(widget._triplet_card_widgets_by_coordinate_system) == ["aligned", "global"]
    assert widget.selected_coordinate_system == "global"
    assert widget.segmentation_combo is widget._triplet_card_widgets_by_coordinate_system["global"].segmentation_combo


def test_feature_extraction_widget_restores_explicit_triplet_when_returning_to_coordinate_system(
    qtbot,
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    viewer = make_viewer_with_shared_sdata(sdata_blobs)

    aligned_label_1 = SpatialDataLabelsOption(
        labels_name="labels_aligned_1",
        display_name="labels_aligned_1",
        sdata=sdata_blobs,
        coordinate_systems=("aligned",),
    )
    aligned_label_2 = SpatialDataLabelsOption(
        labels_name="labels_aligned_2",
        display_name="labels_aligned_2",
        sdata=sdata_blobs,
        coordinate_systems=("aligned",),
    )
    global_label = SpatialDataLabelsOption(
        labels_name="labels_global",
        display_name="labels_global",
        sdata=sdata_blobs,
        coordinate_systems=("global",),
    )
    image_by_label = {
        "labels_aligned_1": SpatialDataImageOption(
            image_name="image_aligned_1",
            display_name="image_aligned_1",
            sdata=sdata_blobs,
            coordinate_systems=("aligned",),
        ),
        "labels_aligned_2": SpatialDataImageOption(
            image_name="image_aligned_2",
            display_name="image_aligned_2",
            sdata=sdata_blobs,
            coordinate_systems=("aligned",),
        ),
        "labels_global": SpatialDataImageOption(
            image_name="image_global",
            display_name="image_global",
            sdata=sdata_blobs,
            coordinate_systems=("global",),
        ),
    }

    monkeypatch.setattr(
        feature_extraction_widget_module,
        "get_coordinate_system_names_from_sdata",
        lambda sdata: ["aligned", "global"],
    )
    monkeypatch.setattr(
        feature_extraction_widget_module,
        "get_spatialdata_feature_extraction_label_discovery_for_coordinate_system_from_sdata",
        lambda *, sdata, coordinate_system: make_label_discovery(
            coordinate_system=coordinate_system,
            options=[aligned_label_1, aligned_label_2] if coordinate_system == "aligned" else [global_label],
        ),
    )
    monkeypatch.setattr(
        feature_extraction_widget_module,
        "get_spatialdata_feature_extraction_image_discovery_for_coordinate_system_and_label_from_sdata",
        lambda *, sdata, coordinate_system, labels_name: make_image_discovery(
            coordinate_system=coordinate_system,
            labels_name=labels_name,
            options=[image_by_label[labels_name]],
        ),
    )
    monkeypatch.setattr(
        feature_extraction_widget_module,
        "get_annotating_table_names",
        lambda sdata, labels_name: ["table"],
    )
    monkeypatch.setattr(
        feature_extraction_widget_module,
        "validate_table_annotation_coverage",
        lambda sdata, table_name, labels_names: None,
    )
    monkeypatch.setattr(
        feature_extraction_widget_module,
        "validate_table_region_instance_ids",
        lambda sdata, table_name, *, labels_names=None: None,
    )

    widget = FeatureExtractionWidget(viewer)
    qtbot.addWidget(widget)

    assert widget.selected_coordinate_system is None
    check_coordinate_system(widget, "aligned")

    assert widget.selected_coordinate_system == "aligned"
    assert widget.selected_segmentation_name is None
    assert widget.selected_image_name is None

    aligned_widgets = widget._triplet_card_widgets_by_coordinate_system["aligned"]
    aligned_widgets.segmentation_combo.setCurrentIndex(1)
    aligned_widgets.image_combo.setCurrentIndex(1)

    assert widget.selected_coordinate_system == "aligned"
    assert widget.selected_segmentation_name == "labels_aligned_2"
    assert widget.selected_image_name == "image_aligned_2"

    get_coordinate_system_checkbox(widget, "global").setChecked(True)

    assert widget.selected_coordinate_system == "global"
    assert widget.selected_segmentation_name is None
    assert widget.selected_image_name is None

    global_widgets = widget._triplet_card_widgets_by_coordinate_system["global"]
    global_widgets.segmentation_combo.setCurrentIndex(0)
    global_widgets.image_combo.setCurrentIndex(1)

    assert widget.selected_image_name == "image_global"

    get_coordinate_system_checkbox(widget, "aligned").setChecked(False)
    assert "aligned" not in widget._triplet_card_widgets_by_coordinate_system

    get_coordinate_system_checkbox(widget, "aligned").setChecked(True)

    assert widget.selected_coordinate_system == "aligned"
    restored_aligned_widgets = widget._triplet_card_widgets_by_coordinate_system["aligned"]
    assert restored_aligned_widgets.segmentation_combo.currentText() == "labels_aligned_2"
    assert restored_aligned_widgets.image_combo.currentText() == "image_aligned_2"
    assert widget.selected_segmentation_name == "labels_aligned_2"
    assert widget.selected_image_name == "image_aligned_2"


def test_feature_extraction_widget_excludes_duplicate_segmentation_across_visible_cards(
    qtbot,
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    viewer = make_viewer_with_shared_sdata(sdata_blobs)

    shared_label = SpatialDataLabelsOption(
        labels_name="shared_labels",
        display_name="shared_labels",
        sdata=sdata_blobs,
        coordinate_systems=("aligned", "global"),
    )
    aligned_only_label = SpatialDataLabelsOption(
        labels_name="aligned_only",
        display_name="aligned_only",
        sdata=sdata_blobs,
        coordinate_systems=("aligned",),
    )
    global_only_label = SpatialDataLabelsOption(
        labels_name="global_only",
        display_name="global_only",
        sdata=sdata_blobs,
        coordinate_systems=("global",),
    )

    monkeypatch.setattr(
        feature_extraction_widget_module,
        "get_coordinate_system_names_from_sdata",
        lambda sdata: ["aligned", "global"],
    )
    monkeypatch.setattr(
        feature_extraction_widget_module,
        "get_spatialdata_feature_extraction_label_discovery_for_coordinate_system_from_sdata",
        lambda *, sdata, coordinate_system: make_label_discovery(
            coordinate_system=coordinate_system,
            options=[shared_label, aligned_only_label]
            if coordinate_system == "aligned"
            else [shared_label, global_only_label],
        ),
    )
    monkeypatch.setattr(
        feature_extraction_widget_module,
        "get_spatialdata_feature_extraction_image_discovery_for_coordinate_system_and_label_from_sdata",
        lambda *, sdata, coordinate_system, labels_name: make_image_discovery(
            coordinate_system=coordinate_system,
            labels_name=labels_name,
            options=[],
        ),
    )
    monkeypatch.setattr(
        feature_extraction_widget_module,
        "get_annotating_table_names",
        lambda sdata, labels_name: ["table"],
    )

    widget = FeatureExtractionWidget(viewer)
    qtbot.addWidget(widget)

    check_coordinate_system(widget, "aligned")
    check_coordinate_system(widget, "global")
    select_segmentation(widget, "aligned", 0)

    global_widgets = widget._triplet_card_widgets_by_coordinate_system["global"]
    assert [
        global_widgets.segmentation_combo.itemText(index) for index in range(global_widgets.segmentation_combo.count())
    ] == ["global_only"]
    assert "shared_labels" in global_widgets.segmentation_note_label.text()
    assert "aligned" in global_widgets.segmentation_note_label.text()


def test_feature_extraction_widget_excludes_later_selected_shared_segmentation_from_earlier_card_options(
    qtbot,
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    viewer = make_viewer_with_shared_sdata(sdata_blobs)

    aligned_only_label = SpatialDataLabelsOption(
        labels_name="aligned_only",
        display_name="aligned_only",
        sdata=sdata_blobs,
        coordinate_systems=("aligned",),
    )
    shared_label = SpatialDataLabelsOption(
        labels_name="shared_labels",
        display_name="shared_labels",
        sdata=sdata_blobs,
        coordinate_systems=("aligned", "global"),
    )
    global_only_label = SpatialDataLabelsOption(
        labels_name="global_only",
        display_name="global_only",
        sdata=sdata_blobs,
        coordinate_systems=("global",),
    )

    monkeypatch.setattr(
        feature_extraction_widget_module,
        "get_coordinate_system_names_from_sdata",
        lambda sdata: ["aligned", "global"],
    )
    monkeypatch.setattr(
        feature_extraction_widget_module,
        "get_spatialdata_feature_extraction_label_discovery_for_coordinate_system_from_sdata",
        lambda *, sdata, coordinate_system: make_label_discovery(
            coordinate_system=coordinate_system,
            options=[aligned_only_label, shared_label]
            if coordinate_system == "aligned"
            else [shared_label, global_only_label],
        ),
    )
    monkeypatch.setattr(
        feature_extraction_widget_module,
        "get_spatialdata_feature_extraction_image_discovery_for_coordinate_system_and_label_from_sdata",
        lambda *, sdata, coordinate_system, labels_name: make_image_discovery(
            coordinate_system=coordinate_system,
            labels_name=labels_name,
            options=[],
        ),
    )
    monkeypatch.setattr(
        feature_extraction_widget_module,
        "get_annotating_table_names",
        lambda sdata, labels_name: ["table"],
    )

    widget = FeatureExtractionWidget(viewer)
    qtbot.addWidget(widget)

    check_coordinate_system(widget, "aligned")
    check_coordinate_system(widget, "global")

    select_segmentation(widget, "aligned", 0)
    select_segmentation(widget, "global", 0)

    aligned_widgets = widget._triplet_card_widgets_by_coordinate_system["aligned"]
    assert aligned_widgets.segmentation_combo.currentText() == "aligned_only"
    assert [
        aligned_widgets.segmentation_combo.itemText(index)
        for index in range(aligned_widgets.segmentation_combo.count())
    ] == ["aligned_only"]
    assert "shared_labels" in aligned_widgets.segmentation_note_label.text()
    assert "global" in aligned_widgets.segmentation_note_label.text()


def test_feature_extraction_widget_clears_blocked_remembered_segmentation_and_image_on_restore(
    qtbot,
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    viewer = make_viewer_with_shared_sdata(sdata_blobs)

    shared_label = SpatialDataLabelsOption(
        labels_name="shared_labels",
        display_name="shared_labels",
        sdata=sdata_blobs,
        coordinate_systems=("aligned", "global"),
    )
    aligned_only_label = SpatialDataLabelsOption(
        labels_name="aligned_only",
        display_name="aligned_only",
        sdata=sdata_blobs,
        coordinate_systems=("aligned",),
    )
    global_only_label = SpatialDataLabelsOption(
        labels_name="global_only",
        display_name="global_only",
        sdata=sdata_blobs,
        coordinate_systems=("global",),
    )
    image_by_selection = {
        ("aligned", "shared_labels"): SpatialDataImageOption(
            image_name="image_aligned_shared",
            display_name="image_aligned_shared",
            sdata=sdata_blobs,
            coordinate_systems=("aligned",),
        ),
        ("global", "shared_labels"): SpatialDataImageOption(
            image_name="image_global_shared",
            display_name="image_global_shared",
            sdata=sdata_blobs,
            coordinate_systems=("global",),
        ),
    }

    monkeypatch.setattr(
        feature_extraction_widget_module,
        "get_coordinate_system_names_from_sdata",
        lambda sdata: ["aligned", "global"],
    )
    monkeypatch.setattr(
        feature_extraction_widget_module,
        "get_spatialdata_feature_extraction_label_discovery_for_coordinate_system_from_sdata",
        lambda *, sdata, coordinate_system: make_label_discovery(
            coordinate_system=coordinate_system,
            options=[shared_label, aligned_only_label]
            if coordinate_system == "aligned"
            else [shared_label, global_only_label],
        ),
    )
    monkeypatch.setattr(
        feature_extraction_widget_module,
        "get_spatialdata_feature_extraction_image_discovery_for_coordinate_system_and_label_from_sdata",
        lambda *, sdata, coordinate_system, labels_name: make_image_discovery(
            coordinate_system=coordinate_system,
            labels_name=labels_name,
            options=[]
            if (coordinate_system, labels_name) not in image_by_selection
            else [image_by_selection[(coordinate_system, labels_name)]],
        ),
    )
    monkeypatch.setattr(
        feature_extraction_widget_module,
        "get_annotating_table_names",
        lambda sdata, labels_name: ["table"],
    )

    widget = FeatureExtractionWidget(viewer)
    qtbot.addWidget(widget)

    check_coordinate_system(widget, "aligned")
    select_segmentation(widget, "aligned", 0)
    widget._triplet_card_widgets_by_coordinate_system["aligned"].image_combo.setCurrentIndex(1)

    get_coordinate_system_checkbox(widget, "aligned").setChecked(False)

    check_coordinate_system(widget, "global")
    select_segmentation(widget, "global", 0)
    widget._triplet_card_widgets_by_coordinate_system["global"].image_combo.setCurrentIndex(1)

    get_coordinate_system_checkbox(widget, "aligned").setChecked(True)

    global_widgets = widget._triplet_card_widgets_by_coordinate_system["global"]
    assert global_widgets.segmentation_combo.currentText() == "shared_labels"
    assert global_widgets.image_combo.currentText() == "image_global_shared"

    restored_aligned_widgets = widget._triplet_card_widgets_by_coordinate_system["aligned"]
    assert restored_aligned_widgets.segmentation_combo.currentIndex() == -1
    assert restored_aligned_widgets.segmentation_combo.placeholderText() == "Choose a labels element"
    assert restored_aligned_widgets.image_combo.currentText() == "No image"
    assert "shared_labels" in restored_aligned_widgets.segmentation_note_label.text()
    assert "global" in restored_aligned_widgets.segmentation_note_label.text()
    assert widget.selected_segmentation_name is None
    assert widget._remembered_card_selection_by_coordinate_system["aligned"].label_identity is None
    assert widget._remembered_card_selection_by_coordinate_system["aligned"].image_identity is None


def test_feature_extraction_widget_hides_channel_selection_without_image(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = make_viewer_with_shared_sdata(sdata_blobs)
    widget = FeatureExtractionWidget(viewer)

    qtbot.addWidget(widget)

    assert widget.selected_image_name is None
    assert widget.selected_extraction_channel_names is None
    assert widget.selected_extraction_channel_indices is None
    assert widget.channel_selection_label.isHidden()
    assert widget.channel_selection_container.isHidden()


def test_feature_extraction_widget_shows_selected_image_channels_and_defaults_to_no_selection(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = make_viewer_with_shared_sdata(sdata_blobs)
    widget = FeatureExtractionWidget(viewer)

    qtbot.addWidget(widget)

    check_coordinate_system(widget, "global")
    select_segmentation(widget, "global", 0)
    widget.image_combo.setCurrentIndex(1)

    assert widget.selected_image_name == "blobs_image"
    assert [checkbox.text() for checkbox in widget._batch_channel_checkboxes] == ["0", "1", "2"]
    assert all(not checkbox.isChecked() for checkbox in widget._batch_channel_checkboxes)
    assert widget.selected_extraction_channel_names == ()
    assert widget.selected_extraction_channel_indices == ()
    assert not widget.channel_selection_label.isHidden()
    assert not widget.channel_selection_container.isHidden()


def test_feature_extraction_widget_blocks_intensity_features_without_selected_channels(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = make_viewer_with_shared_sdata(sdata_blobs)
    widget = FeatureExtractionWidget(viewer)

    qtbot.addWidget(widget)

    check_coordinate_system(widget, "global")
    select_segmentation(widget, "global", 0)
    widget.findChild(QCheckBox, "feature_checkbox_mean").setChecked(True)
    widget.image_combo.setCurrentIndex(1)
    widget.output_key_line_edit.setText("features")

    # No channels are selected by default, so intensity feature extraction is blocked.
    assert widget.selected_extraction_channel_names == ()
    assert widget.calculate_button.isEnabled() is False
    assert not widget.intensity_features_hint.isHidden()
    assert "select at least one channel" in widget.intensity_features_hint.text().lower()

    # The same blocker also surfaces in the primary batch status, consistent with
    # other incomplete staged extraction targets.
    selection_status_text = unescape(widget.selection_status.text())
    assert "Batch Incomplete" in selection_status_text
    assert "select at least one channel" in selection_status_text.lower()
    assert widget.controller_feedback.isHidden()

    # Selecting a channel unblocks calculation and clears the hint.
    widget._batch_channel_checkboxes[0].setChecked(True)

    assert widget.selected_extraction_channel_names == ("0",)
    assert widget.calculate_button.isEnabled() is True
    assert widget.intensity_features_hint.isHidden()


def test_feature_extraction_widget_raises_when_selected_image_has_no_channel_axis(
    qtbot,
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    viewer = make_viewer_with_shared_sdata(sdata_blobs)
    widget = FeatureExtractionWidget(viewer)

    qtbot.addWidget(widget)

    check_coordinate_system(widget, "global")
    select_segmentation(widget, "global", 0)
    widget.image_combo.setCurrentIndex(1)

    monkeypatch.setattr(
        feature_extraction_widget_module,
        "get_image_channel_names_from_sdata",
        lambda sdata, image_name: [],
    )

    with pytest.raises(
        ValueError,
        match="does not expose channel names, but feature extraction expects images with an explicit channel axis",
    ):
        widget._resolve_batch_channel_state()


def test_feature_extraction_widget_keeps_shared_channel_selector_visible_for_incompatible_later_image(
    qtbot,
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    viewer = make_viewer_with_shared_sdata(sdata_blobs)

    monkeypatch.setattr(
        feature_extraction_widget_module,
        "get_coordinate_system_names_from_sdata",
        lambda sdata: ["aligned", "global"],
    )
    monkeypatch.setattr(
        feature_extraction_widget_module,
        "get_spatialdata_feature_extraction_label_discovery_for_coordinate_system_from_sdata",
        lambda *, sdata, coordinate_system: make_label_discovery(
            coordinate_system=coordinate_system,
            options=[
                SpatialDataLabelsOption(
                    labels_name=f"labels_{coordinate_system}",
                    display_name=f"labels_{coordinate_system}",
                    sdata=sdata,
                    coordinate_systems=(coordinate_system,),
                )
            ],
        ),
    )
    monkeypatch.setattr(
        feature_extraction_widget_module,
        "get_spatialdata_feature_extraction_image_discovery_for_coordinate_system_and_label_from_sdata",
        lambda *, sdata, coordinate_system, labels_name: make_image_discovery(
            coordinate_system=coordinate_system,
            labels_name=labels_name,
            options=[
                SpatialDataImageOption(
                    image_name=f"image_{coordinate_system}",
                    display_name=f"image_{coordinate_system}",
                    sdata=sdata,
                    coordinate_systems=(coordinate_system,),
                )
            ],
        ),
    )
    monkeypatch.setattr(
        feature_extraction_widget_module,
        "get_annotating_table_names",
        lambda sdata, labels_name: ["table"],
    )
    monkeypatch.setattr(
        feature_extraction_widget_module,
        "get_image_channel_names_from_sdata",
        lambda sdata, image_name: ["0", "1", "2"] if image_name == "image_aligned" else ["0", "2"],
    )

    widget = FeatureExtractionWidget(viewer)
    qtbot.addWidget(widget)

    check_coordinate_system(widget, "aligned")
    select_segmentation(widget, "aligned", 0)
    widget._triplet_card_widgets_by_coordinate_system["aligned"].image_combo.setCurrentIndex(1)
    widget._batch_channel_checkboxes[0].setChecked(True)
    widget._batch_channel_checkboxes[2].setChecked(True)

    assert widget.selected_extraction_channel_names == ("0", "2")

    check_coordinate_system(widget, "global")
    select_segmentation(widget, "global", 0)
    widget._triplet_card_widgets_by_coordinate_system["global"].image_combo.setCurrentIndex(1)

    assert widget.selected_extraction_channel_names == ("0", "2")
    assert [checkbox.text() for checkbox in widget._batch_channel_checkboxes] == ["0", "1", "2"]
    assert not widget.channel_selection_label.isHidden()
    assert not widget.channel_selection_container.isHidden()
    assert widget.channel_selection_note_label.text() == "Channel names of selected images do not match."


def test_feature_extraction_widget_remembers_shared_channel_selection_by_schema(
    qtbot,
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    viewer = make_viewer_with_shared_sdata(sdata_blobs)

    monkeypatch.setattr(
        feature_extraction_widget_module,
        "get_coordinate_system_names_from_sdata",
        lambda sdata: ["aligned", "global"],
    )
    monkeypatch.setattr(
        feature_extraction_widget_module,
        "get_spatialdata_feature_extraction_label_discovery_for_coordinate_system_from_sdata",
        lambda *, sdata, coordinate_system: make_label_discovery(
            coordinate_system=coordinate_system,
            options=[
                SpatialDataLabelsOption(
                    labels_name=f"labels_{coordinate_system}",
                    display_name=f"labels_{coordinate_system}",
                    sdata=sdata,
                    coordinate_systems=(coordinate_system,),
                )
            ],
        ),
    )
    monkeypatch.setattr(
        feature_extraction_widget_module,
        "get_spatialdata_feature_extraction_image_discovery_for_coordinate_system_and_label_from_sdata",
        lambda *, sdata, coordinate_system, labels_name: make_image_discovery(
            coordinate_system=coordinate_system,
            labels_name=labels_name,
            options=[
                SpatialDataImageOption(
                    image_name=f"image_{coordinate_system}",
                    display_name=f"image_{coordinate_system}",
                    sdata=sdata,
                    coordinate_systems=(coordinate_system,),
                )
            ],
        ),
    )
    monkeypatch.setattr(
        feature_extraction_widget_module,
        "get_annotating_table_names",
        lambda sdata, labels_name: ["table"],
    )
    monkeypatch.setattr(
        feature_extraction_widget_module,
        "get_image_channel_names_from_sdata",
        lambda sdata, image_name: ["0", "1", "2"] if image_name == "image_global" else ["a", "b"],
    )

    widget = FeatureExtractionWidget(viewer)
    qtbot.addWidget(widget)

    check_coordinate_system(widget, "global")
    select_segmentation(widget, "global", 0)
    widget._triplet_card_widgets_by_coordinate_system["global"].image_combo.setCurrentIndex(1)
    widget._batch_channel_checkboxes[0].setChecked(True)
    widget._batch_channel_checkboxes[2].setChecked(True)

    assert widget.selected_extraction_channel_names == ("0", "2")

    get_coordinate_system_checkbox(widget, "global").setChecked(False)

    check_coordinate_system(widget, "aligned")
    select_segmentation(widget, "aligned", 0)
    widget._triplet_card_widgets_by_coordinate_system["aligned"].image_combo.setCurrentIndex(1)
    widget._batch_channel_checkboxes[1].setChecked(True)

    assert widget.selected_extraction_channel_names == ("b",)

    get_coordinate_system_checkbox(widget, "aligned").setChecked(False)
    get_coordinate_system_checkbox(widget, "global").setChecked(True)

    restored_global_widgets = widget._triplet_card_widgets_by_coordinate_system["global"]
    select_segmentation(widget, "global", 0)
    restored_global_widgets.image_combo.setCurrentIndex(1)

    assert widget.selected_extraction_channel_names == ("0", "2")


def test_feature_extraction_widget_surfaces_duplicate_channel_names_as_batch_error_for_intensity_features(
    qtbot,
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    viewer = make_viewer_with_shared_sdata(sdata_blobs)
    widget = FeatureExtractionWidget(viewer)

    qtbot.addWidget(widget)

    bind_batch_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_bind_batch(*args, **kwargs):
        bind_batch_calls.append((args, kwargs))
        return True

    widget._feature_extraction_controller.bind_batch = fake_bind_batch  # type: ignore[method-assign]
    monkeypatch.setattr(
        feature_extraction_widget_module,
        "get_image_channel_names_from_sdata",
        lambda sdata, image_name: (_ for _ in ()).throw(
            ValueError(
                "Image element `blobs_image` exposes duplicate channel names (`dup`), "
                "which spatiato does not support. "
                "Update the channel names in the SpatialData object with "
                "`sdata.set_channel_names(...)`."
            )
        ),
    )

    check_coordinate_system(widget, "global")
    select_segmentation(widget, "global", 0)
    widget.image_combo.setCurrentIndex(1)
    widget.findChild(QCheckBox, "feature_checkbox_mean").setChecked(True)

    assert widget.selected_image_name == "blobs_image"
    assert widget.channel_selection_label.isHidden()
    assert widget.channel_selection_container.isHidden()
    assert widget.intensity_features_hint.text() == (
        "One or more selected images expose duplicate channel names. "
        'Rename channels with "sdata.set_channel_names(...)" or choose a different image.'
    )
    assert bind_batch_calls
    args, kwargs = bind_batch_calls[-1]
    assert args == (sdata_blobs, (), "table", ("mean",), "features")
    assert kwargs == {"create_table": False}


def test_feature_extraction_widget_channel_selection_is_independent_from_viewer_overlay_state(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = make_viewer_with_shared_sdata(sdata_blobs)
    feature_widget = FeatureExtractionWidget(viewer)
    viewer_widget = ViewerWidget(viewer)

    qtbot.addWidget(feature_widget)
    qtbot.addWidget(viewer_widget)

    check_coordinate_system(feature_widget, "global")
    select_segmentation(feature_widget, "global", 0)
    feature_widget.image_combo.setCurrentIndex(1)
    feature_widget._batch_channel_checkboxes[1].setChecked(True)
    feature_widget._batch_channel_checkboxes[2].setChecked(True)

    assert feature_widget.selected_extraction_channel_indices == (1, 2)

    image_card = next(card for card in viewer_widget.image_cards if card.image_name == "blobs_image")
    image_card.overlay_toggle.setChecked(True)
    image_card.channel_search_input.setText("0")
    image_card.channel_search_input.returnPressed.emit()

    assert image_card.loaded_overlay_channel_indices == (0,)
    assert feature_widget.selected_extraction_channel_indices == (1, 2)


def test_feature_extraction_widget_reports_per_card_batch_ready_status_for_valid_batch(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = make_viewer_with_shared_sdata(sdata_blobs)
    widget = FeatureExtractionWidget(viewer)

    qtbot.addWidget(widget)

    check_coordinate_system(widget, "global")
    select_segmentation(widget, "global", 0)

    status_text = unescape(widget.selection_status.text())
    assert "Batch Ready" in status_text
    assert "global: blobs_labels (no image)" in status_text
    assert widget.selection_status.toolTip() == ""


def test_feature_extraction_widget_reports_one_line_per_checked_card_in_batch_ready_status(
    qtbot,
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    viewer = make_viewer_with_shared_sdata(sdata_blobs)
    monkeypatch.setattr(
        feature_extraction_widget_module,
        "get_coordinate_system_names_from_sdata",
        lambda sdata: ["aligned", "global"],
    )
    monkeypatch.setattr(
        feature_extraction_widget_module,
        "get_spatialdata_feature_extraction_label_discovery_for_coordinate_system_from_sdata",
        lambda *, sdata, coordinate_system: make_label_discovery(
            coordinate_system=coordinate_system,
            options=[
                SpatialDataLabelsOption(
                    labels_name=f"labels_{coordinate_system}",
                    display_name=f"labels_{coordinate_system}",
                    sdata=sdata,
                    coordinate_systems=(coordinate_system,),
                )
            ],
        ),
    )
    monkeypatch.setattr(
        feature_extraction_widget_module,
        "get_spatialdata_feature_extraction_image_discovery_for_coordinate_system_and_label_from_sdata",
        lambda *, sdata, coordinate_system, labels_name: make_image_discovery(
            coordinate_system=coordinate_system,
            labels_name=labels_name,
            options=[
                SpatialDataImageOption(
                    image_name=f"image_{coordinate_system}_{labels_name}",
                    display_name=f"image_{coordinate_system}_{labels_name}",
                    sdata=sdata,
                    coordinate_systems=(coordinate_system,),
                )
            ],
        ),
    )
    monkeypatch.setattr(
        feature_extraction_widget_module,
        "get_annotating_table_names",
        lambda sdata, labels_name: ["table"],
    )
    monkeypatch.setattr(
        feature_extraction_widget_module,
        "validate_table_annotation_coverage",
        lambda sdata, table_name, labels_names: None,
    )
    monkeypatch.setattr(
        feature_extraction_widget_module,
        "validate_table_region_instance_ids",
        lambda sdata, table_name, *, labels_names=None: None,
    )

    widget = FeatureExtractionWidget(viewer)

    qtbot.addWidget(widget)

    check_coordinate_system(widget, "aligned")
    check_coordinate_system(widget, "global")
    select_segmentation(widget, "aligned", 0)
    select_segmentation(widget, "global", 0)

    aligned_label = widget._triplet_card_states_by_coordinate_system["aligned"].selected_label_option
    global_label = widget._triplet_card_states_by_coordinate_system["global"].selected_label_option

    assert aligned_label is not None
    assert global_label is not None
    status_text = unescape(widget.selection_status.text())
    assert "Batch Ready" in status_text
    assert f"aligned: {aligned_label.labels_name} (no image)" in status_text
    assert f"global: {global_label.labels_name} (no image)" in status_text
    assert widget.selection_status.toolTip() == ""


def test_feature_extraction_widget_selects_create_table_when_selected_segmentation_has_no_linked_table(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = make_viewer_with_shared_sdata(sdata_blobs)
    widget = FeatureExtractionWidget(viewer)

    qtbot.addWidget(widget)
    check_coordinate_system(widget, "global")
    widget.segmentation_combo.setCurrentIndex(1)

    assert widget.selected_segmentation_name == "blobs_multiscale_labels"
    assert widget.image_combo.count() == 3
    assert widget.selected_image_name is None
    assert combo_texts(widget.table_combo) == ["Create table..."]
    assert widget.table_combo.isEnabled() is True
    assert widget.selected_table_mode == "create"
    assert widget.selected_table_name is None
    assert widget.selected_new_table_name == "features_table"
    assert not widget.new_table_name_line_edit.isHidden()
    assert widget.coordinate_system_combo.count() == 1
    assert widget.selected_coordinate_system == "global"
    assert "Batch Ready" in widget.selection_status.text()
    assert "global: blobs_multiscale_labels (no image)" in widget.selection_status.text()
    tooltip = unescape(widget.selection_status.toolTip()).replace("&#8203;", "").replace("\u200b", "")
    assert tooltip == ""


def test_feature_extraction_widget_suggests_uuid_table_name_when_default_exists(
    qtbot,
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    sdata_blobs.tables["features_table"] = sdata_blobs.tables["table"].copy()
    monkeypatch.setattr(feature_extraction_widget_module.uuid, "uuid4", lambda: "abc-123")
    viewer = make_viewer_with_shared_sdata(sdata_blobs)
    widget = FeatureExtractionWidget(viewer)

    qtbot.addWidget(widget)
    check_coordinate_system(widget, "global")
    widget.segmentation_combo.setCurrentIndex(1)

    assert widget.selected_table_mode == "create"
    assert widget.selected_new_table_name == "features_table_abc-123"
    assert widget.new_table_name_line_edit.text() == "features_table_abc-123"


def test_feature_extraction_widget_preserves_typed_create_table_name_when_toggling_modes(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = make_viewer_with_shared_sdata(sdata_blobs)
    widget = FeatureExtractionWidget(viewer)

    qtbot.addWidget(widget)
    check_coordinate_system(widget, "global")
    select_segmentation(widget, "global", 0)

    assert combo_texts(widget.table_combo) == ["table", "Create table..."]
    widget.table_combo.setCurrentIndex(1)
    widget.new_table_name_line_edit.setText("custom_features")

    assert widget.selected_table_mode == "create"
    assert widget.selected_table_name is None
    assert widget.selected_new_table_name == "custom_features"
    assert not widget.new_table_name_line_edit.isHidden()

    widget.table_combo.setCurrentIndex(0)

    assert widget.selected_table_mode == "existing"
    assert widget.selected_table_name == "table"
    assert widget.selected_new_table_name is None
    assert widget.new_table_name_line_edit.isHidden()
    assert widget.new_table_name_line_edit.text() == "custom_features"

    widget.table_combo.setCurrentIndex(1)

    assert widget.selected_table_mode == "create"
    assert widget.selected_new_table_name == "custom_features"
    assert widget.new_table_name_line_edit.text() == "custom_features"


def test_feature_extraction_widget_binds_create_table_mode_into_controller(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = make_viewer_with_shared_sdata(sdata_blobs)
    widget = FeatureExtractionWidget(viewer)

    qtbot.addWidget(widget)
    bind_batch_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_bind_batch(*args, **kwargs):
        bind_batch_calls.append((args, kwargs))
        return True

    widget._feature_extraction_controller.bind_batch = fake_bind_batch  # type: ignore[method-assign]

    check_coordinate_system(widget, "global")
    widget.segmentation_combo.setCurrentIndex(1)
    widget.findChild(QCheckBox, "feature_checkbox_area").setChecked(True)

    assert widget.selected_table_mode == "create"
    assert bind_batch_calls
    args, kwargs = bind_batch_calls[-1]
    assert args == (
        sdata_blobs,
        (
            FeatureExtractionTriplet(
                coordinate_system="global",
                labels_name="blobs_multiscale_labels",
                image_name=None,
                channels=None,
            ),
        ),
        "features_table",
        ("area",),
        "features",
    )
    assert kwargs == {"create_table": True}


def test_feature_extraction_widget_create_table_binding_matches_expected_controller_state(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = make_viewer_with_shared_sdata(sdata_blobs)
    widget = FeatureExtractionWidget(viewer)

    qtbot.addWidget(widget)
    check_coordinate_system(widget, "global")
    widget.segmentation_combo.setCurrentIndex(1)
    widget.findChild(QCheckBox, "feature_checkbox_area").setChecked(True)

    expected_state = FeatureExtractionBindingState(
        sdata=sdata_blobs,
        triplets=(
            FeatureExtractionTriplet(
                coordinate_system="global",
                labels_name="blobs_multiscale_labels",
                image_name=None,
                channels=None,
            ),
        ),
        table_name="features_table",
        create_table=True,
        feature_names=("area",),
        feature_key="features",
    )

    assert widget._expected_controller_binding_state() == expected_state
    assert widget._feature_extraction_controller.binding_state == expected_state
    assert widget.calculate_button.isEnabled() is True
    assert widget.calculate_button.toolTip() == ""
    controller_feedback_text = unescape(widget.controller_feedback.text())
    assert "Feature Extraction Ready" in controller_feedback_text
    assert 'ready to create table "features_table" and calculate.' in controller_feedback_text


def test_feature_extraction_widget_blocks_create_table_mode_for_missing_or_invalid_table_name(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = make_viewer_with_shared_sdata(sdata_blobs)
    widget = FeatureExtractionWidget(viewer)

    qtbot.addWidget(widget)
    check_coordinate_system(widget, "global")
    widget.segmentation_combo.setCurrentIndex(1)
    widget.findChild(QCheckBox, "feature_checkbox_area").setChecked(True)

    widget.new_table_name_line_edit.clear()

    assert widget.selected_table_mode == "create"
    assert widget.selected_new_table_name is None
    assert widget.calculate_button.isEnabled() is False
    assert "Table Not Ready" in widget.selection_status.text()
    assert "Enter a new table name." in widget.selection_status.text()
    assert "Enter a new table name." in unescape(widget.calculate_button.toolTip())
    assert widget._feature_extraction_controller.binding_state.triplets == ()

    widget.new_table_name_line_edit.setText("bad name")

    assert widget.calculate_button.isEnabled() is False
    assert "Choose a valid table name." in widget.selection_status.text()
    assert "Choose a valid table name." in unescape(widget.calculate_button.toolTip())
    assert widget._feature_extraction_controller.binding_state.triplets == ()


def test_feature_extraction_widget_blocks_create_table_mode_for_table_name_collision(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = make_viewer_with_shared_sdata(sdata_blobs)
    widget = FeatureExtractionWidget(viewer)

    qtbot.addWidget(widget)
    check_coordinate_system(widget, "global")
    widget.segmentation_combo.setCurrentIndex(1)
    widget.findChild(QCheckBox, "feature_checkbox_area").setChecked(True)
    widget.new_table_name_line_edit.setText("table")

    assert widget.selected_table_mode == "create"
    assert widget.calculate_button.isEnabled() is False
    assert "Table Not Ready" in widget.selection_status.text()
    assert 'Table "table" already exists. Choose a different table name.' in widget.selection_status.text()
    assert 'Table "table" already exists. Choose a different table name.' in unescape(widget.calculate_button.toolTip())
    assert widget._feature_extraction_controller.binding_state.triplets == ()

    widget.new_table_name_line_edit.setText("TABLE")

    assert widget.calculate_button.isEnabled() is False
    assert "Table Not Ready" in widget.selection_status.text()
    assert 'Table "TABLE" already exists. Choose a different table name.' in widget.selection_status.text()
    assert 'Table "TABLE" already exists. Choose a different table name.' in unescape(widget.calculate_button.toolTip())
    assert widget._feature_extraction_controller.binding_state.triplets == ()


def test_feature_extraction_widget_uses_batch_table_error_as_status_tooltip(
    qtbot,
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    viewer = make_viewer_with_shared_sdata(sdata_blobs)
    widget = FeatureExtractionWidget(viewer)

    qtbot.addWidget(widget)
    monkeypatch.setattr(
        feature_extraction_widget_module,
        "validate_table_region_instance_ids",
        lambda sdata, table_name, *, labels_names=None: (_ for _ in ()).throw(
            ValueError(
                "Table `table` cannot annotate labels element `blobs_labels` because "
                "`instance_id` contains duplicate values within that region: `1`, `2`."
            )
        ),
    )

    check_coordinate_system(widget, "global")
    select_segmentation(widget, "global", 0)

    tooltip = unescape(widget.selection_status.toolTip()).replace("&#8203;", "").replace("\u200b", "")
    assert "Table Not Ready" in widget.selection_status.text()
    assert "global: blobs_labels (no image)" in tooltip
    assert "duplicate values within that region" in tooltip


def test_feature_extraction_widget_exposes_grouped_feature_checkboxes(qtbot) -> None:
    widget = FeatureExtractionWidget()

    qtbot.addWidget(widget)

    scroll_area = widget.findChild(QScrollArea, "feature_extraction_scroll_area")
    assert scroll_area is not None
    assert scroll_area.widgetResizable()
    assert widget.intensity_features_group.title() == "Intensity Features"
    assert widget.morphology_features_group.title() == "Morphology Features"
    assert widget.findChild(QCheckBox, "feature_checkbox_mean") is not None
    assert widget.findChild(QCheckBox, "feature_checkbox_var") is not None
    assert widget.findChild(QCheckBox, "feature_checkbox_area") is not None
    assert widget.findChild(QCheckBox, "feature_checkbox_perimeter") is not None
    assert widget.findChild(QCheckBox, "feature_extraction_overwrite_feature_key_checkbox") is None
    assert widget.intensity_features_hint.isHidden()


def test_feature_extraction_widget_places_feedback_above_main_controls(qtbot) -> None:
    widget = FeatureExtractionWidget()

    qtbot.addWidget(widget)

    content_layout = widget.scroll_content.layout()

    def widget_index(target) -> int:
        for index in range(content_layout.count()):
            item = content_layout.itemAt(index)
            if item.widget() is target:
                return index
        return -1

    selection_status_index = widget_index(widget.selection_status)
    controller_feedback_index = widget_index(widget.controller_feedback)
    triplet_cards_index = widget_index(widget.triplet_cards_container)

    assert selection_status_index >= 0
    assert controller_feedback_index >= 0
    assert triplet_cards_index >= 0
    assert selection_status_index < triplet_cards_index
    assert controller_feedback_index < triplet_cards_index


def test_feature_extraction_widget_reads_back_feature_selection_and_output_key(qtbot) -> None:
    widget = FeatureExtractionWidget()

    qtbot.addWidget(widget)

    widget.findChild(QCheckBox, "feature_checkbox_mean").setChecked(True)
    widget.findChild(QCheckBox, "feature_checkbox_var").setChecked(True)
    widget.findChild(QCheckBox, "feature_checkbox_area").setChecked(True)
    widget.output_key_line_edit.setText("object_features")

    assert widget.selected_feature_names == ("mean", "var", "area")
    assert widget.selected_feature_key == "object_features"
    assert widget.overwrite_feature_key is False
    assert not widget.intensity_features_hint.isHidden()
    assert "choose an image" in widget.intensity_features_hint.text()


def test_feature_extraction_widget_hides_intensity_warning_when_image_is_selected(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = make_viewer_with_shared_sdata(sdata_blobs)
    widget = FeatureExtractionWidget(viewer)

    qtbot.addWidget(widget)
    check_coordinate_system(widget, "global")
    select_segmentation(widget, "global", 0)
    widget.image_combo.setCurrentIndex(1)
    widget.findChild(QCheckBox, "feature_checkbox_mean").setChecked(True)
    widget._batch_channel_checkboxes[0].setChecked(True)

    assert widget.selected_image_name == "blobs_image"
    assert widget.intensity_features_hint.isHidden()


def test_feature_extraction_widget_rebinds_controller_when_inputs_change(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = make_viewer_with_shared_sdata(sdata_blobs)
    widget = FeatureExtractionWidget(viewer)

    qtbot.addWidget(widget)

    bind_batch_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_bind_batch(*args, **kwargs):
        bind_batch_calls.append((args, kwargs))
        return True

    widget._feature_extraction_controller.bind_batch = fake_bind_batch  # type: ignore[method-assign]

    check_coordinate_system(widget, "global")
    select_segmentation(widget, "global", 0)
    widget.findChild(QCheckBox, "feature_checkbox_area").setChecked(True)
    widget.output_key_line_edit.setText("features")

    assert bind_batch_calls
    args, kwargs = bind_batch_calls[-1]
    assert args == (
        sdata_blobs,
        (
            FeatureExtractionTriplet(
                coordinate_system="global",
                labels_name="blobs_labels",
                image_name=None,
                channels=None,
            ),
        ),
        "table",
        ("area",),
        "features",
    )
    assert kwargs == {"create_table": False}


def test_feature_extraction_widget_binds_selected_channels_into_controller(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = make_viewer_with_shared_sdata(sdata_blobs)
    widget = FeatureExtractionWidget(viewer)

    qtbot.addWidget(widget)

    bind_batch_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_bind_batch(*args, **kwargs):
        bind_batch_calls.append((args, kwargs))
        return True

    widget._feature_extraction_controller.bind_batch = fake_bind_batch  # type: ignore[method-assign]

    check_coordinate_system(widget, "global")
    select_segmentation(widget, "global", 0)
    widget.findChild(QCheckBox, "feature_checkbox_mean").setChecked(True)
    widget.image_combo.setCurrentIndex(1)
    widget._batch_channel_checkboxes[0].setChecked(True)
    widget._batch_channel_checkboxes[2].setChecked(True)

    assert bind_batch_calls
    args, kwargs = bind_batch_calls[-1]
    assert args == (
        sdata_blobs,
        (
            FeatureExtractionTriplet(
                coordinate_system="global",
                labels_name="blobs_labels",
                image_name="blobs_image",
                channels=("0", "2"),
            ),
        ),
        "table",
        ("mean",),
        "features",
    )
    assert kwargs == {"create_table": False}


def test_feature_extraction_widget_enables_calculate_for_valid_morphology_batch(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = make_viewer_with_shared_sdata(sdata_blobs)
    widget = FeatureExtractionWidget(viewer)

    qtbot.addWidget(widget)

    check_coordinate_system(widget, "global")
    select_segmentation(widget, "global", 0)
    widget.findChild(QCheckBox, "feature_checkbox_area").setChecked(True)
    widget.output_key_line_edit.setText("features")

    assert widget.calculate_button.isEnabled() is True
    assert widget.calculate_button.toolTip() == ""
    controller_feedback_text = unescape(widget.controller_feedback.text())
    assert "Feature Extraction Ready" in controller_feedback_text
    assert "ready to calculate." in controller_feedback_text.lower()


def test_feature_extraction_widget_blocks_spatialdata_invalid_feature_key(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = make_viewer_with_shared_sdata(sdata_blobs)
    widget = FeatureExtractionWidget(viewer)

    qtbot.addWidget(widget)

    check_coordinate_system(widget, "global")
    select_segmentation(widget, "global", 0)
    widget.findChild(QCheckBox, "feature_checkbox_area").setChecked(True)
    widget.output_key_line_edit.setText("features tst")

    assert widget.calculate_button.isEnabled() is False
    tooltip = unescape(widget.calculate_button.toolTip())
    assert "choose a valid feature matrix key" in tooltip
    assert "alphanumeric characters, underscores, dots and hyphens" in tooltip
    feedback_text = unescape(widget.controller_feedback.text())
    assert "Feature Extraction Warning" in feedback_text
    assert "choose a valid feature matrix key" in feedback_text


def test_feature_extraction_widget_keeps_calculate_disabled_for_intensity_features_without_image(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = make_viewer_with_shared_sdata(sdata_blobs)
    widget = FeatureExtractionWidget(viewer)

    qtbot.addWidget(widget)

    check_coordinate_system(widget, "global")
    select_segmentation(widget, "global", 0)
    widget.findChild(QCheckBox, "feature_checkbox_mean").setChecked(True)
    widget.output_key_line_edit.setText("features")

    assert widget.calculate_button.isEnabled() is False
    assert "choose an image for every extraction target" in unescape(widget.calculate_button.toolTip()).lower()
    assert "choose an image" in widget.intensity_features_hint.text()
    status_text = unescape(widget.selection_status.text())
    assert "Batch Incomplete" in status_text
    assert "global: choose an image" in status_text


def test_feature_extraction_widget_blocks_when_no_coordinate_system_is_checked(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = make_viewer_with_shared_sdata(sdata_blobs)
    widget = FeatureExtractionWidget(viewer)

    qtbot.addWidget(widget)

    widget.findChild(QCheckBox, "feature_checkbox_area").setChecked(True)
    widget.output_key_line_edit.setText("features")
    assert get_coordinate_system_checkbox(widget, "global").isChecked() is False
    widget._bind_current_selection()

    assert widget.selected_coordinate_system is None
    assert "Choose Coordinate Systems" in widget.selection_status.text()
    assert widget.calculate_button.isEnabled() is False
    assert "choose one or more coordinate systems" in unescape(widget.calculate_button.toolTip()).lower()


def test_feature_extraction_widget_does_not_launch_controller_while_calculation_is_disabled(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = make_viewer_with_shared_sdata(sdata_blobs)
    widget = FeatureExtractionWidget(viewer)

    qtbot.addWidget(widget)
    widget.findChild(QCheckBox, "feature_checkbox_area").setChecked(True)
    widget.output_key_line_edit.setText("features")

    calls: list[bool | None] = []

    def fake_calculate(*, overwrite_feature_key: bool | None = None) -> bool:
        calls.append(overwrite_feature_key)
        return True

    widget._feature_extraction_controller.calculate = fake_calculate  # type: ignore[method-assign]

    widget.calculate_button.click()

    assert calls == []


def test_feature_extraction_widget_does_not_launch_non_overwrite_run_while_calculation_is_disabled(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = make_viewer_with_shared_sdata(sdata_blobs)
    widget = FeatureExtractionWidget(viewer)

    qtbot.addWidget(widget)
    widget.findChild(QCheckBox, "feature_checkbox_area").setChecked(True)
    widget.output_key_line_edit.setText("new_features")

    overwrite_calls: list[bool | None] = []

    def fake_calculate(*, overwrite_feature_key: bool | None = None) -> bool:
        overwrite_calls.append(overwrite_feature_key)
        return True

    widget._feature_extraction_controller.calculate = fake_calculate  # type: ignore[method-assign]

    widget.calculate_button.click()

    assert overwrite_calls == []


def test_feature_extraction_widget_does_not_prompt_before_overwriting_while_calculation_is_disabled(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = make_viewer_with_shared_sdata(sdata_blobs)
    widget = FeatureExtractionWidget(viewer)

    qtbot.addWidget(widget)
    widget.findChild(QCheckBox, "feature_checkbox_area").setChecked(True)
    widget.output_key_line_edit.setText("existing_features")
    sdata_blobs.tables["table"].obsm["existing_features"] = np.zeros((sdata_blobs.tables["table"].n_obs, 1))

    recorder = _record_launches(widget)

    widget.calculate_button.click()

    assert recorder.dialogs == []
    assert recorder.launches == []


def test_feature_extraction_widget_does_not_prompt_for_existing_feature_key_in_create_table_mode(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = make_viewer_with_shared_sdata(sdata_blobs)
    widget = FeatureExtractionWidget(viewer)

    qtbot.addWidget(widget)
    check_coordinate_system(widget, "global")
    widget.segmentation_combo.setCurrentIndex(1)
    widget.findChild(QCheckBox, "feature_checkbox_area").setChecked(True)
    widget.output_key_line_edit.setText("existing_features")
    sdata_blobs.tables["table"].obsm["existing_features"] = np.zeros((sdata_blobs.tables["table"].n_obs, 1))

    recorder = _record_launches(widget)

    assert widget.selected_table_mode == "create"
    assert widget.calculate_button.isEnabled() is True

    widget.calculate_button.click()

    # A new table has no existing entry to check: no dialog, no confirmed kind.
    assert recorder.dialogs == []
    assert recorder.launches == [{}]


class _LaunchRecorder:
    """Record the dialogs a Calculate click shows and the launches it requests."""

    def __init__(self, widget: FeatureExtractionWidget, *, confirm: bool) -> None:
        self.dialogs: list[tuple[str, tuple[object, ...]]] = []
        self.launches: list[dict[str, object]] = []
        self.dirty_snapshots: list[object] = []
        self._confirm = confirm
        widget._confirm_regional_update = self._regional  # type: ignore[method-assign]
        widget._confirm_full_replacement = self._full  # type: ignore[method-assign]
        widget._feature_extraction_controller.calculate = self._calculate  # type: ignore[method-assign]

    def _regional(self, *args: object) -> bool:
        self.dialogs.append(("regional", args))
        return self._confirm

    def _full(self, *args: object) -> bool:
        self.dialogs.append(("full", args))
        return self._confirm

    def _calculate(self, **kwargs: object) -> bool:
        self.dirty_snapshots.append(kwargs.pop("dirty_snapshot", None))
        self.launches.append(kwargs)
        return True


def _record_launches(widget: FeatureExtractionWidget, *, confirm: bool = True) -> _LaunchRecorder:
    return _LaunchRecorder(widget, confirm=confirm)


def _ready_widget(qtbot, sdata: SpatialData, feature_key: str) -> FeatureExtractionWidget:
    widget = FeatureExtractionWidget(make_viewer_with_shared_sdata(sdata))
    qtbot.addWidget(widget)
    check_coordinate_system(widget, "global")
    select_segmentation(widget, "global", 0)
    widget.findChild(QCheckBox, "feature_checkbox_area").setChecked(True)
    widget.output_key_line_edit.setText(feature_key)
    assert widget.selected_table_name == "table"
    assert widget.calculate_button.isEnabled() is True
    return widget


def test_feature_extraction_widget_launches_new_key_without_dialog(qtbot, sdata_blobs: SpatialData) -> None:
    widget = _ready_widget(qtbot, sdata_blobs, "features_new")
    recorder = _record_launches(widget)

    widget.calculate_button.click()

    assert recorder.dialogs == []
    assert recorder.launches == [{"overwrite_feature_key": False, "confirmed_existing_entry": "absent"}]
    # The launch carries the table's dirty state, for recording the job's result.
    (dirty_snapshot,) = recorder.dirty_snapshots
    assert isinstance(dirty_snapshot, TableDirtySnapshot)
    assert dirty_snapshot.sdata is sdata_blobs
    assert dirty_snapshot.table_name == "table"


def test_feature_extraction_widget_confirms_regional_update_of_compatible_matrix(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    calculate_feature_matrix(
        sdata_blobs,
        table_name="table",
        feature_key="features_new",
        triplets=(FeatureExtractionTriplet("global", "blobs_labels", None),),
        feature_names=("area",),
    )
    widget = _ready_widget(qtbot, sdata_blobs, "features_new")
    recorder = _record_launches(widget)

    widget.calculate_button.click()

    assert recorder.dialogs == [("regional", ("features_new", "table", ("blobs_labels",)))]
    assert recorder.launches == [{"overwrite_feature_key": True, "confirmed_existing_entry": "compatible"}]


@pytest.mark.parametrize("confirm", [True, False])
def test_feature_extraction_widget_confirms_full_replacement_of_incompatible_matrix(
    qtbot,
    sdata_blobs: SpatialData,
    confirm: bool,
) -> None:
    table = sdata_blobs.tables["table"]
    table.obsm["existing_features"] = np.zeros((table.n_obs, 1))
    widget = _ready_widget(qtbot, sdata_blobs, "existing_features")
    recorder = _record_launches(widget, confirm=confirm)

    widget.calculate_button.click()

    assert recorder.dialogs == [("full", ("existing_features", "it has no harpy feature metadata", ()))]
    expected_launches = [{"overwrite_feature_key": True, "confirmed_existing_entry": "incompatible"}]
    assert recorder.launches == (expected_launches if confirm else [])


def test_feature_extraction_widget_shows_check_errors_without_dialog(
    qtbot,
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    widget = _ready_widget(qtbot, sdata_blobs, "features_new")
    recorder = _record_launches(widget)

    def fail(*args: object, **kwargs: object) -> None:
        raise ValueError("rows differ")

    monkeypatch.setattr(feature_extraction_controller_module, "check_existing_feature_matrix", fail)

    widget.calculate_button.click()

    assert recorder.dialogs == []
    assert recorder.launches == []
    assert widget._feature_extraction_controller.status_kind == "error"
    assert widget._feature_extraction_controller.status_message == "Feature extraction: rows differ"


def test_feature_extraction_widget_blocks_launch_on_unsaved_feature_metadata_when_backed(
    qtbot,
    backed_sdata_blobs: SpatialData,
) -> None:
    widget = _ready_widget(qtbot, backed_sdata_blobs, "features_new")
    # As Object Classification's "register feature metadata" does for a custom matrix.
    widget._app_state.record_table_mutation(
        TableStateChangedEvent(
            sdata=backed_sdata_blobs,
            table_name="table",
            paths=frozenset({TableComponentPath("uns", ("feature_matrices", "features_new"))}),
            regions=(),
            change_kind="updated",
            source="test",
        )
    )
    recorder = _record_launches(widget)

    widget.calculate_button.click()

    assert recorder.dialogs == []
    assert recorder.launches == []
    assert widget._feature_extraction_controller.status_message == (
        'Feature extraction: feature matrix "features_new" has unsaved changes; write or reload the table first.'
    )


def test_feature_extraction_widget_refreshes_table_selection_after_controller_success(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = make_viewer_with_shared_sdata(sdata_blobs)
    widget = FeatureExtractionWidget(viewer)

    qtbot.addWidget(widget)

    calls: list[str] = []

    original_refresh_table_names = widget._refresh_table_names
    original_bind_current_selection = widget._bind_current_selection

    def recording_refresh_table_names(*, preferred_existing_table_name: str | None = None) -> None:
        calls.append("refresh_table_names")
        original_refresh_table_names(preferred_existing_table_name=preferred_existing_table_name)

    def recording_bind_current_selection() -> None:
        calls.append("bind_current_selection")
        original_bind_current_selection()

    widget._refresh_table_names = recording_refresh_table_names  # type: ignore[method-assign]
    widget._bind_current_selection = recording_bind_current_selection  # type: ignore[method-assign]

    widget._on_controller_table_selection_refresh_required(
        FeatureExtractionResult(
            job_id=1,
            labels_names=("blobs_labels",),
            table_name="table",
            feature_key="features",
            regions=("blobs_labels",),
        )
    )

    assert calls == ["refresh_table_names", "bind_current_selection"]


def test_feature_extraction_widget_promotes_created_table_to_existing_selection(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = make_viewer_with_shared_sdata(sdata_blobs)
    widget = FeatureExtractionWidget(viewer)

    qtbot.addWidget(widget)
    check_coordinate_system(widget, "global")
    select_segmentation(widget, "global", 0)
    widget.table_combo.setCurrentIndex(widget.table_combo.findText("Create table..."))
    widget.new_table_name_line_edit.setText("new_table")
    widget.findChild(QCheckBox, "feature_checkbox_area").setChecked(True)

    assert widget.selected_table_mode == "create"
    assert widget.selected_new_table_name == "new_table"
    assert widget._feature_extraction_controller.binding_state.create_table is True

    sdata_blobs.tables["new_table"] = sdata_blobs.tables["table"].copy()
    widget._on_controller_table_selection_refresh_required(
        FeatureExtractionResult(
            job_id=1,
            labels_names=("blobs_labels",),
            table_name="new_table",
            feature_key="features",
            regions=("blobs_labels",),
        )
    )

    expected_state = FeatureExtractionBindingState(
        sdata=sdata_blobs,
        triplets=(
            FeatureExtractionTriplet(
                coordinate_system="global",
                labels_name="blobs_labels",
                image_name=None,
                channels=None,
            ),
        ),
        table_name="new_table",
        create_table=False,
        feature_names=("area",),
        feature_key="features",
    )

    assert widget.selected_table_mode == "existing"
    assert widget.selected_table_name == "new_table"
    assert widget.selected_new_table_name is None
    assert widget.new_table_name_line_edit.isHidden()
    assert widget._feature_extraction_controller.binding_state == expected_state


def test_feature_extraction_widget_uses_existing_overwrite_prompt_after_created_table_promotion(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = make_viewer_with_shared_sdata(sdata_blobs)
    widget = FeatureExtractionWidget(viewer)

    qtbot.addWidget(widget)
    check_coordinate_system(widget, "global")
    select_segmentation(widget, "global", 0)
    widget.table_combo.setCurrentIndex(widget.table_combo.findText("Create table..."))
    widget.new_table_name_line_edit.setText("new_table")
    widget.findChild(QCheckBox, "feature_checkbox_area").setChecked(True)

    sdata_blobs.tables["new_table"] = sdata_blobs.tables["table"].copy()
    sdata_blobs.tables["new_table"].obsm["features"] = np.zeros((sdata_blobs.tables["new_table"].n_obs, 1))
    widget._on_controller_table_selection_refresh_required(
        FeatureExtractionResult(
            job_id=1,
            labels_names=("blobs_labels",),
            table_name="new_table",
            feature_key="features",
            regions=("blobs_labels",),
        )
    )

    recorder = _record_launches(widget)

    assert widget.selected_table_mode == "existing"
    assert widget._feature_extraction_controller.binding_state.create_table is False

    widget.calculate_button.click()

    assert recorder.dialogs == [("full", ("features", "it has no harpy feature metadata", ()))]
    assert recorder.launches == [{"overwrite_feature_key": True, "confirmed_existing_entry": "incompatible"}]


def _launch_create_table_job(qtbot, sdata: SpatialData) -> tuple[FeatureExtractionWidget, _DeferredWorker]:
    viewer = make_viewer_with_shared_sdata(sdata)
    widget = FeatureExtractionWidget(viewer)
    qtbot.addWidget(widget)
    worker = _DeferredWorker()
    widget._feature_extraction_controller._create_feature_extraction_worker = lambda _job: worker  # type: ignore[method-assign]
    check_coordinate_system(widget, "global")
    select_segmentation(widget, "global", 0)
    widget.table_combo.setCurrentIndex(widget.table_combo.findText("Create table..."))
    widget.new_table_name_line_edit.setText("new_table")
    widget.findChild(QCheckBox, "feature_checkbox_area").setChecked(True)
    widget.calculate_button.click()
    assert worker.started is True
    return widget, worker


def _return_created_table(sdata: SpatialData, worker: _DeferredWorker) -> None:
    # Stands in for harpy, which creates the table before the worker returns.
    sdata.tables["new_table"] = sdata.tables["table"].copy()
    worker.returned.emit(
        FeatureExtractionResult(
            job_id=1,
            labels_names=("blobs_labels",),
            table_name="new_table",
            feature_key="features",
            regions=("blobs_labels",),
        )
    )
    worker.finished.emit()


def test_feature_extraction_widget_reports_write_after_promoting_table_created_by_active_job(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    widget, worker = _launch_create_table_job(qtbot, sdata_blobs)
    controller = widget._feature_extraction_controller

    _return_created_table(sdata_blobs, worker)

    assert widget.selected_table_mode == "existing"
    assert widget.selected_table_name == "new_table"
    assert controller.has_job_in_flight is False
    assert controller.status_kind == "success"
    assert controller.status_message.startswith('Feature extraction: wrote "features" into table "new_table"')
    assert widget.calculate_button.isEnabled()


def test_feature_extraction_widget_lists_table_created_by_detached_job_and_keeps_selection(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    widget, worker = _launch_create_table_job(qtbot, sdata_blobs)
    widget.table_combo.setCurrentIndex(widget.table_combo.findText("table"))

    assert widget.table_combo.findText("new_table") == -1

    _return_created_table(sdata_blobs, worker)

    assert widget.table_combo.findText("new_table") >= 0
    assert widget.selected_table_mode == "existing"
    assert widget.selected_table_name == "table"
    assert widget._feature_extraction_controller.binding_state.table_name == "table"
    assert widget._feature_extraction_controller.status_message == "Feature extraction: ready to calculate."


def test_feature_extraction_widget_promotes_table_created_by_detached_job_that_is_still_the_target(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    widget, worker = _launch_create_table_job(qtbot, sdata_blobs)
    # Changing only the key detaches the job; "new_table" stays the create-table target.
    widget.output_key_line_edit.setText("features_2")

    assert widget._feature_extraction_controller.status_message == (
        "Feature extraction: waiting for the previous calculation to finish writing."
    )

    _return_created_table(sdata_blobs, worker)

    assert widget.selected_table_mode == "existing"
    assert widget.selected_table_name == "new_table"
    assert widget._feature_extraction_controller.binding_state.create_table is False
    assert widget._feature_extraction_controller.binding_state.feature_key == "features_2"


def test_feature_extraction_widget_records_feature_matrix_change_in_shared_app_state(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = make_viewer_with_shared_sdata(sdata_blobs)
    app_state = get_or_create_app_state(viewer)
    widget = FeatureExtractionWidget(viewer)

    qtbot.addWidget(widget)

    result = FeatureExtractionResult(
        job_id=1,
        labels_names=("blobs_labels",),
        table_name="table",
        feature_key="features_new",
        regions=("blobs_labels",),
        change_kind="created",
    )

    with qtbot.waitSignal(app_state.table_state_changed) as blocker:
        widget._on_controller_feature_matrix_changed(result)

    event = blocker.args[0]
    assert isinstance(event, TableStateChangedEvent)
    assert event.paths == frozenset(
        {
            TableComponentPath("obsm", ("features_new",)),
            TableComponentPath("uns", ("feature_matrices", "features_new")),
        }
    )
    assert event.regions == ("blobs_labels",)
    assert app_state.is_table_dirty(sdata_blobs, "table") is True


def test_feature_extraction_widget_records_backed_result_as_persisted_with_its_launch_snapshot(
    qtbot,
    monkeypatch,
    backed_sdata_blobs: SpatialData,
) -> None:
    viewer = make_viewer_with_shared_sdata(backed_sdata_blobs)
    app_state = get_or_create_app_state(viewer)
    widget = FeatureExtractionWidget(viewer)
    qtbot.addWidget(widget)
    unrelated_path = TableComponentPath("obs", ("user_class",))
    # Unsaved changes elsewhere in the table do not block a launch.
    app_state.record_table_mutation(
        TableStateChangedEvent(
            sdata=backed_sdata_blobs,
            table_name="table",
            paths=frozenset({unrelated_path}),
            regions=("blobs_labels",),
            change_kind="updated",
            source="test",
        )
    )
    dirty_snapshot = app_state.snapshot_table_dirty_state(backed_sdata_blobs, "table")
    result = FeatureExtractionResult(
        job_id=1,
        labels_names=("blobs_labels",),
        table_name="table",
        feature_key="features_new",
        regions=("blobs_labels",),
        change_kind="created",
        dirty_snapshot=dirty_snapshot,
    )
    recorded_snapshots: list[TableDirtySnapshot | None] = []
    record_persisted_table_change = app_state.record_persisted_table_change

    def spy(event: TableStateChangedEvent, snapshot: TableDirtySnapshot | None = None) -> None:
        recorded_snapshots.append(snapshot)
        record_persisted_table_change(event, snapshot)

    monkeypatch.setattr(app_state, "record_persisted_table_change", spy)

    with qtbot.waitSignal(app_state.table_state_changed) as blocker:
        widget._on_controller_feature_matrix_changed(result)

    assert blocker.args[0].paths == {
        TableComponentPath("obsm", ("features_new",)),
        TableComponentPath("uns", ("feature_matrices", "features_new")),
    }
    # The widget records the result with the job's own launch snapshot.
    assert len(recorded_snapshots) == 1
    assert recorded_snapshots[0] is dirty_snapshot
    # harpy already wrote the key's paths, so recording the result as persisted
    # never marks them dirty (unlike `record_table_mutation`); other unsaved
    # changes stay dirty.
    assert app_state.snapshot_table_dirty_state(backed_sdata_blobs, "table").paths == {unrelated_path}


def test_feature_extraction_widget_adopts_reloaded_feature_matrix_state_without_republishing(
    qtbot,
    backed_sdata_blobs: SpatialData,
) -> None:
    viewer = make_viewer_with_shared_sdata(backed_sdata_blobs)
    app_state = get_or_create_app_state(viewer)
    widget = FeatureExtractionWidget(viewer)
    qtbot.addWidget(widget)
    check_coordinate_system(widget, "global")
    select_segmentation(widget, "global", 0)
    widget.findChild(QCheckBox, "feature_checkbox_area").setChecked(True)
    widget.output_key_line_edit.setText("features_1")
    table = backed_sdata_blobs.tables["table"]
    del table.obsm["features_1"]
    events: list[TableStateChangedEvent] = []
    app_state.table_state_changed.connect(events.append)
    persistence = PersistenceController(app_state)
    persistence.bind(backed_sdata_blobs, "table", "blobs_labels")

    persistence.reload_table_state()

    assert "features_1" in table.obsm
    assert widget.selected_table_name == "table"
    assert widget.selected_feature_key == "features_1"
    assert widget._feature_extraction_controller.binding_state.table_name == "table"
    assert widget._feature_extraction_controller.binding_state.feature_key == "features_1"
    assert [event.change_kind for event in events] == ["reloaded"]


def test_feature_extraction_widget_clears_when_shared_sdata_is_cleared(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = make_viewer_with_shared_sdata(sdata_blobs)
    app_state = get_or_create_app_state(viewer)
    widget = FeatureExtractionWidget(viewer)

    qtbot.addWidget(widget)

    assert widget.segmentation_combo.count() == 0

    app_state.clear_sdata(discard_current=True)

    assert widget.selected_segmentation_name is None
    assert widget.selected_spatialdata is None
    assert widget.selected_image_name is None
    assert widget.selected_table_name is None
    assert widget.selected_table_mode is None
    assert widget.selected_new_table_name is None
    assert widget.selected_coordinate_system is None
    assert widget.segmentation_combo.count() == 0
    assert widget.segmentation_combo.isEnabled() is False
    assert widget.image_combo.count() == 1
    assert widget.image_combo.itemText(0) == "No image"
    assert widget.image_combo.isEnabled() is False
    assert widget.table_combo.count() == 0
    assert widget.table_combo.isEnabled() is False
    assert widget.new_table_name_line_edit.isHidden()
    assert widget.coordinate_system_combo.count() == 0
    assert widget.coordinate_system_combo.isEnabled() is False
    assert widget.calculate_button.isEnabled() is False
    assert "No SpatialData Loaded" in widget.selection_status.text()
    assert "shared Spatiato state" in unescape(widget.selection_status.text())
