from __future__ import annotations

from collections.abc import Callable
from html import unescape
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import zarr
from harpy.utils._keys import _FEATURE_MATRICES_KEY
from matplotlib.colors import to_rgba
from napari.layers import Image, Labels
from napari.utils.colormaps import DirectLabelColormap
from qtpy.QtCore import QObject, Signal
from qtpy.QtWidgets import QCheckBox, QComboBox, QLabel, QScrollArea
from spatialdata import SpatialData, read_zarr
from spatialdata.models import TableModel
from spatialdata.transformations import get_transformation

import spatiato._app_state as app_state_module
import spatiato.widgets.object_classification.annotation_controller as annotation_module
import spatiato.widgets.object_classification.controller as classifier_module
import spatiato.widgets.object_classification.widget as widget_module
import spatiato.widgets.persistence.controls as persistence_controls_module
import spatiato.widgets.viewer.widget as viewer_widget_module
from spatiato._app_state import TableStateChangedEvent, get_or_create_app_state
from spatiato.core.class_palette import (
    DEFAULT_NEUTRAL_COLOR,
    default_categorical_colors,
    default_class_colors,
)
from spatiato.core.feature_matrix_metadata import (
    CUSTOM_OBSM_SOURCE_KIND,
    HARPY_ADD_FEATURE_MATRIX_SOURCE_KIND,
    register_feature_matrix_metadata,
)
from spatiato.core.object_classification.annotation import (
    USER_CLASS_COLORS_KEY,
    USER_CLASS_COLUMN,
    UserClassStateChange,
)
from spatiato.core.object_classification.classifier_export import (
    DEFAULT_CLASSIFIER_EXPORT_SUFFIX,
    read_classifier_export_bundle,
)
from spatiato.core.persistence import TableComponentPath
from spatiato.core.spatialdata import SpatialDataLabelsOption
from spatiato.viewer.labels_colormap import CompactLabelColormap
from spatiato.widgets.annotation.models import AnnotationContext, ShapesAnnotationTarget
from spatiato.widgets.object_classification.controller import (
    CLASSIFIER_CONFIG_KEY,
    PRED_CLASS_COLORS_KEY,
    PRED_CLASS_COLUMN,
    PRED_CONFIDENCE_COLUMN,
)
from spatiato.widgets.object_classification.widget import (
    ObjectClassificationWidget,
)
from spatiato.widgets.shared_styles import STATUS_CARD_PALETTE, WIDGET_MIN_WIDTH
from spatiato.widgets.spatial_query.widget import SpatialQuery
from spatiato.widgets.viewer.widget import ViewerWidget


def _feature_table_event(
    sdata: SpatialData,
    *,
    table_name: str,
    feature_key: str,
    change_kind: str = "created",
) -> TableStateChangedEvent:
    return TableStateChangedEvent(
        sdata=sdata,
        table_name=table_name,
        paths=frozenset(
            {
                TableComponentPath("obsm", (feature_key,)),
                TableComponentPath("uns", ("feature_matrices", feature_key)),
            }
        ),
        regions=("blobs_labels",),
        change_kind=change_kind,
        source="feature_extraction",
    )


def _spatial_query_annotation_event(
    sdata: SpatialData,
    *,
    table_name: str = "table",
    paths: frozenset[TableComponentPath] | None = None,
    source: str = "spatial_query_annotation",
) -> TableStateChangedEvent:
    return TableStateChangedEvent(
        sdata=sdata,
        table_name=table_name,
        paths=(frozenset({TableComponentPath("obs", (USER_CLASS_COLUMN,))}) if paths is None else paths),
        regions=("blobs_labels",),
        change_kind="updated",
        source=source,
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


class DummyLayers(list):
    def __init__(self, layers: list[Labels] | None = None) -> None:
        super().__init__(layers or [])
        self.selection = SimpleNamespace(active=None, select_only=self._select_only)
        self.events = SimpleNamespace(
            inserted=DummyEventEmitter(),
            removed=DummyEventEmitter(),
            reordered=DummyEventEmitter(),
        )

    def _select_only(self, layer: Labels) -> None:
        self.selection.active = layer


class DummyViewer:
    def __init__(self, layers: list[Labels] | None = None, *, seed_shared_sdata: bool = True) -> None:
        self.layers = DummyLayers(layers)
        if not seed_shared_sdata:
            return

        sdata_values: list[SpatialData] = []
        for layer in self.layers:
            metadata = getattr(layer, "metadata", None)
            if not isinstance(metadata, dict):
                continue
            sdata = metadata.get("sdata")
            if isinstance(sdata, SpatialData):
                sdata_values.append(sdata)

        if sdata_values and len({id(sdata) for sdata in sdata_values}) == 1:
            app_state = get_or_create_app_state(self)
            app_state.set_sdata(sdata_values[0])
            for layer in self.layers:
                if not isinstance(layer, Labels):
                    continue
                metadata = getattr(layer, "metadata", None)
                if not isinstance(metadata, dict):
                    continue
                sdata = metadata.get("sdata")
                if sdata is not sdata_values[0]:
                    continue
                element_name = metadata.get("name", getattr(layer, "name", None))
                if not isinstance(element_name, str):
                    continue
                coordinate_system = metadata.get("coordinate_system", metadata.get("_current_cs"))
                if not isinstance(coordinate_system, str) and element_name in sdata.labels:
                    available_coordinate_systems = tuple(
                        get_transformation(sdata.labels[element_name], get_all=True).keys()
                    )
                    if len(available_coordinate_systems) == 1:
                        coordinate_system = available_coordinate_systems[0]
                    elif "global" in available_coordinate_systems:
                        coordinate_system = "global"
                app_state.viewer_adapter.register_labels_layer(
                    layer,
                    sdata=sdata,
                    labels_name=element_name,
                    coordinate_system=coordinate_system if isinstance(coordinate_system, str) else None,
                )


def make_viewer_with_shared_sdata(sdata: SpatialData, layers: list[Labels] | None = None) -> DummyViewer:
    viewer = DummyViewer(layers=layers, seed_shared_sdata=False)
    get_or_create_app_state(viewer).set_sdata(sdata)
    return viewer


def select_segmentation(widget: ObjectClassificationWidget, index: int = 0) -> None:
    widget.segmentation_combo.setCurrentIndex(index)


def _combo_texts(combo: QComboBox) -> list[str]:
    return [combo.itemText(index) for index in range(combo.count())]


def _tooltip_text(widget: object) -> str:
    return unescape(widget.toolTip()).replace("&#8203;", "").replace("\u200b", "")


def _add_feature_table_for_labels(
    sdata: SpatialData,
    *,
    table_name: str,
    labels_name: str,
    feature_key: str,
) -> None:
    table = sdata["table"].copy()
    attrs = table.uns[TableModel.ATTRS_KEY]
    region_key = attrs[TableModel.REGION_KEY_KEY]

    table.obs[region_key] = pd.Categorical([labels_name] * table.n_obs, categories=[labels_name])
    table.uns[TableModel.ATTRS_KEY] = {
        **attrs,
        TableModel.REGION_KEY: [labels_name],
    }
    for key in list(table.obsm.keys()):
        del table.obsm[key]
    table.obsm[feature_key] = np.arange(table.n_obs, dtype=np.float64).reshape(table.n_obs, 1)
    sdata.tables[table_name] = table


_SUCCESS_FEEDBACK_STYLE = STATUS_CARD_PALETTE["success"]


def _assert_persistence_success_feedback(widget: ObjectClassificationWidget, expected_message: str) -> None:
    assert "Persistence Updated" in widget.persistence_controls.feedback_label.text()
    assert expected_message in widget.persistence_controls.feedback_label.text()

    stylesheet = widget.persistence_controls.feedback_label.styleSheet()
    assert f"color: {_SUCCESS_FEEDBACK_STYLE['text']}" in stylesheet
    assert f"background-color: {_SUCCESS_FEEDBACK_STYLE['background']}" in stylesheet
    assert f"border: 1px solid {_SUCCESS_FEEDBACK_STYLE['border']}" in stylesheet


def _assert_feature_metadata_warning_card(widget: ObjectClassificationWidget) -> None:
    assert "Feature Metadata Warning" in widget.warning_status.text()

    stylesheet = widget.warning_status.styleSheet()
    warning_style = STATUS_CARD_PALETTE["warning"]
    assert f"color: {warning_style['text']}" in stylesheet
    assert f"background-color: {warning_style['background']}" in stylesheet
    assert f"border: 1px solid {warning_style['border']}" in stylesheet


def _set_feature_metadata(
    sdata: SpatialData,
    *,
    table_name: str = "table",
    feature_key: str = "features_1",
) -> None:
    table = sdata[table_name]
    n_features = int(table.obsm[feature_key].shape[1])
    table.uns.setdefault("feature_matrices", {})[feature_key] = {
        "feature_columns": [f"feature_{index}" for index in range(n_features)],
        "schema_version": 1,
        "backend": "numpy",
        "dtype": str(np.asarray(table.obsm[feature_key]).dtype),
        "source_label": "blobs_labels",
        "source_image": None,
        "coordinate_system": "global",
        "features": [f"feature_{index}" for index in range(n_features)],
        "source_kind": HARPY_ADD_FEATURE_MATRIX_SOURCE_KIND,
    }


def test_get_user_class_values_returns_unlabeled_for_missing_column() -> None:
    obs = pd.DataFrame(index=range(3))

    values = classifier_module._get_user_class_values(obs)

    assert values.isna().all()
    assert str(values.dtype) == "Int64"


def test_get_user_class_values_uses_integer_dtype_fast_path() -> None:
    obs = pd.DataFrame({USER_CLASS_COLUMN: pd.Series([1, 2, 5], dtype=np.int64)})

    values = classifier_module._get_user_class_values(obs)

    assert values.tolist() == [1, 2, 5]


def test_get_user_class_values_uses_nullable_integer_dtype_fast_path() -> None:
    obs = pd.DataFrame({USER_CLASS_COLUMN: pd.Series([1, pd.NA, 3], dtype="Int64")})

    values = classifier_module._get_user_class_values(obs)

    assert values.tolist() == [1, pd.NA, 3]


def test_get_user_class_values_uses_categorical_integer_fast_path() -> None:
    obs = pd.DataFrame(
        {
            USER_CLASS_COLUMN: pd.Categorical(
                [pd.NA, 2, 1],
                categories=[1, 2],
            )
        }
    )

    values = classifier_module._get_user_class_values(obs)

    assert values.tolist() == [pd.NA, 2, 1]


def test_get_user_class_values_preserves_missing_categorical_values() -> None:
    obs = pd.DataFrame(
        {
            USER_CLASS_COLUMN: pd.Categorical(
                [1, None, 2],
                categories=[1, 2],
            )
        }
    )

    values = classifier_module._get_user_class_values(obs)

    assert values.tolist() == [1, pd.NA, 2]


def test_get_user_class_values_rejects_string_class_values() -> None:
    obs = pd.DataFrame({USER_CLASS_COLUMN: pd.Series(["1", "bad", None, "3"], dtype="object")})

    with pytest.raises(ValueError, match="positive integer"):
        classifier_module._get_user_class_values(obs)


def _patch_coordinate_system_names(monkeypatch, coordinate_systems: list[str]) -> None:
    monkeypatch.setattr(
        widget_module,
        "get_coordinate_system_names_from_sdata",
        lambda sdata: list(coordinate_systems),
    )
    monkeypatch.setattr(
        app_state_module,
        "get_coordinate_system_names_from_sdata",
        lambda sdata: list(coordinate_systems),
    )
    monkeypatch.setattr(
        viewer_widget_module,
        "get_coordinate_system_names_from_sdata",
        lambda sdata: list(coordinate_systems),
    )


class _DeferredWorker(QObject):
    returned = Signal(object)
    errored = Signal(object)
    finished = Signal()

    def __init__(self, result: classifier_module.ClassifierJobResult) -> None:
        super().__init__()
        self._result = result
        self.started = False
        self.quit_called = False

    def start(self) -> None:
        self.started = True

    def quit(self) -> None:
        self.quit_called = True

    def emit_returned(self) -> None:
        self.returned.emit(self._result)
        self.finished.emit()


def make_blobs_labels_layer(sdata: SpatialData, labels_name: str = "blobs_labels") -> Labels:
    layer = Labels(
        sdata.labels[labels_name],
        name=labels_name,
        metadata={"sdata": sdata, "name": labels_name},
    )
    return layer


def make_multiscale_blobs_labels_layer(sdata: SpatialData, labels_name: str = "blobs_labels") -> Labels:
    base_data = np.asarray(sdata.labels[labels_name])
    multiscale_data = [base_data, base_data[::2, ::2]]
    indices = [int(value) for value in np.unique(base_data).tolist() if int(value) > 0]
    layer = Labels(
        multiscale_data,
        name=labels_name,
        metadata={"sdata": sdata, "name": labels_name, "indices": indices},
    )
    return layer


def _write_disk_table_state(
    backed_sdata_blobs: SpatialData,
    *,
    obs: pd.DataFrame,
    obsm: dict[str, object],
    uns: dict[str, object],
) -> None:
    root = zarr.open_group(backed_sdata_blobs.path, mode="a", use_consolidated=False)
    table_group = root["tables/table"]
    ad.io.write_elem(table_group, "obs", obs)
    ad.io.write_elem(table_group, "obsm", obsm)
    ad.io.write_elem(table_group, "uns", uns)


def rename_table_instance_key(sdata: SpatialData, *, table_name: str = "table", instance_key: str) -> None:
    table = sdata[table_name]
    table.obs = table.obs.rename(columns={"instance_id": instance_key})
    table.uns[TableModel.ATTRS_KEY][TableModel.INSTANCE_KEY] = instance_key


def test_widget_can_be_instantiated(qtbot) -> None:
    widget = ObjectClassificationWidget()

    qtbot.addWidget(widget)

    scroll_area = widget.findChild(QScrollArea, "object_classification_scroll_area")
    assert scroll_area is not None
    assert scroll_area.widgetResizable()
    assert widget is not None
    assert widget.findChild(QLabel, "object_classification_header_logo") is not None
    assert widget.selected_segmentation_name is None
    assert widget.selected_table_name is None
    assert widget.selected_feature_key is None
    assert widget.selected_training_scope == classifier_module.DEFAULT_TRAINING_SCOPE
    assert widget.selected_prediction_scope == classifier_module.DEFAULT_PREDICTION_SCOPE
    assert widget.selected_coordinate_system is None
    assert widget.selected_color_by == "user_class"
    assert widget.auto_train_checkbox.objectName() == "auto_train_checkbox"
    assert widget.findChild(QCheckBox, "auto_train_checkbox") is widget.auto_train_checkbox
    assert widget.auto_train_checkbox.text() == "Auto-train classifier"
    assert widget.auto_train_checkbox.isChecked() is False
    assert widget.register_feature_matrix_button.objectName() == "register_feature_matrix_button"
    assert not widget.register_feature_matrix_button.isEnabled()
    assert "Choose an annotation table and feature matrix" in _tooltip_text(widget.register_feature_matrix_button)
    assert "QCheckBox" in widget.auto_train_checkbox.styleSheet()
    assert widget._auto_train_enabled is False
    assert all(button.text() != "Rescan Viewer" for button in widget.findChildren(type(widget.retrain_button)))
    assert "No SpatialData Loaded" in widget.selection_status.text()
    assert widget.coordinate_system_combo.sizeAdjustPolicy() == (
        QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
    )
    assert (
        widget.segmentation_combo.sizeAdjustPolicy() == QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
    )
    assert widget.table_combo.sizeAdjustPolicy() == QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
    assert widget.feature_matrix_combo.sizeAdjustPolicy() == (
        QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
    )
    assert widget.training_scope_combo.currentData() == classifier_module.DEFAULT_TRAINING_SCOPE
    assert widget.prediction_scope_combo.currentData() == classifier_module.DEFAULT_PREDICTION_SCOPE


def test_widget_destruction_unregisters_table_reload_participant(qtbot) -> None:
    """Ensure Qt destruction cannot leave a stale reload participant."""
    viewer = DummyViewer(seed_shared_sdata=False)
    app_state = get_or_create_app_state(viewer)
    widget = ObjectClassificationWidget(viewer)

    assert any(participant is widget for participant in app_state._table_reload_participants)

    widget.deleteLater()
    qtbot.waitUntil(lambda: not any(participant is widget for participant in app_state._table_reload_participants))


def test_widget_initial_action_rows_fit_current_minimum_width(qtbot) -> None:
    viewer = DummyViewer(seed_shared_sdata=False)
    widget = ObjectClassificationWidget(viewer)

    qtbot.addWidget(widget)

    scrollbar_width = widget.scroll_area.verticalScrollBar().sizeHint().width()
    scroll_content_margins = widget.scroll_content.layout().contentsMargins()
    viewport_width = WIDGET_MIN_WIDTH - scrollbar_width
    available_content_width = viewport_width - scroll_content_margins.left() - scroll_content_margins.right()

    assert widget.scroll_content.sizeHint().width() <= viewport_width
    assert widget.auto_train_checkbox.minimumSizeHint().width() <= available_content_width
    assert widget.retrain_action_row.minimumSizeHint().width() <= available_content_width
    assert widget.persistence_controls.action_row.minimumSizeHint().width() <= available_content_width


def test_widget_refreshes_when_shared_sdata_changes(qtbot, sdata_blobs: SpatialData) -> None:
    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer], seed_shared_sdata=False)
    app_state = get_or_create_app_state(viewer)
    widget = ObjectClassificationWidget(viewer)

    qtbot.addWidget(widget)

    assert widget.segmentation_combo.count() == 0
    assert "No SpatialData Loaded" in widget.selection_status.text()

    app_state.set_sdata(sdata_blobs)

    assert widget.coordinate_system_combo.count() == 1
    assert widget.coordinate_system_combo.itemText(0) == "global"
    assert widget.selected_coordinate_system == "global"
    assert widget.segmentation_combo.count() == 2
    assert widget.selected_segmentation_name is None
    assert widget.selected_spatialdata is sdata_blobs
    assert widget.selected_table_name is None
    assert widget.selected_feature_key is None
    assert "Choose a labels element" in widget.selection_status.text()


def test_widget_clears_when_shared_sdata_is_cleared(qtbot, sdata_blobs: SpatialData) -> None:
    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = make_viewer_with_shared_sdata(sdata_blobs, layers=[layer])
    app_state = get_or_create_app_state(viewer)
    widget = ObjectClassificationWidget(viewer)

    qtbot.addWidget(widget)

    assert widget.segmentation_combo.count() == 2

    app_state.clear_sdata(discard_current=True)

    assert widget.selected_coordinate_system is None
    assert widget.selected_segmentation_name is None
    assert widget.selected_spatialdata is None
    assert widget.selected_table_name is None
    assert widget.selected_feature_key is None
    assert widget.coordinate_system_combo.count() == 0
    assert not widget.coordinate_system_combo.isEnabled()
    assert widget.segmentation_combo.count() == 0
    assert not widget.segmentation_combo.isEnabled()
    assert widget.table_combo.count() == 0
    assert not widget.table_combo.isEnabled()
    assert widget.feature_matrix_combo.count() == 0
    assert not widget.feature_matrix_combo.isEnabled()
    assert "No SpatialData Loaded" in widget.selection_status.text()


def test_widget_populates_segmentation_dropdown_from_spatialdata(qtbot, sdata_blobs: SpatialData) -> None:
    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])

    widget = ObjectClassificationWidget(viewer)
    qtbot.addWidget(widget)

    assert widget.coordinate_system_combo.count() == 1
    assert widget.coordinate_system_combo.itemText(0) == "global"
    assert widget.selected_coordinate_system == "global"
    assert widget.segmentation_combo.count() == 2
    assert [widget.segmentation_combo.itemText(index) for index in range(widget.segmentation_combo.count())] == [
        "blobs_labels",
        "blobs_multiscale_labels",
    ]
    assert widget.table_combo.count() == 0
    assert widget.feature_matrix_combo.count() == 0
    assert widget.color_by_combo.count() == 3
    assert [widget.color_by_combo.itemText(index) for index in range(widget.color_by_combo.count())] == [
        "user_class",
        "pred_class",
        "pred_confidence",
    ]
    assert widget.selected_segmentation_name is None
    assert widget.selected_spatialdata is sdata_blobs
    assert widget.selected_table_name is None
    assert widget.selected_feature_key is None
    assert widget.selected_color_by == "user_class"
    assert widget.selected_table_metadata is None
    assert "adata" not in layer.metadata
    assert widget.selected_instance_id is None
    assert all(button.text() != "Rescan Viewer" for button in widget.findChildren(type(widget.retrain_button)))
    assert widget.retrain_button.text() == "Train Classifier"
    assert widget.persistence_controls.write_button.text() == "Write Table State"
    assert widget.persistence_controls.reload_button.text() == "Reload Table State"
    assert not widget.persistence_controls.write_button.isEnabled()
    assert not widget.persistence_controls.reload_button.isEnabled()
    assert not widget.retrain_button.isEnabled()
    assert len(viewer.layers) == 1
    assert viewer.layers.selection.active is None
    assert "Choose a labels element" in widget.selection_status.text()
    assert widget.warning_status.isHidden()
    assert widget.warning_status.text() == ""
    assert widget.classifier_feedback.isHidden()
    assert widget.classifier_preparation_status.isHidden()
    assert widget.classifier_preparation_status.objectName() == "classifier_preparation_status"


def test_widget_populates_segmentation_choices_from_shared_sdata_without_loaded_layer(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = make_viewer_with_shared_sdata(sdata_blobs)

    widget = ObjectClassificationWidget(viewer)
    qtbot.addWidget(widget)

    assert widget.coordinate_system_combo.count() == 1
    assert widget.coordinate_system_combo.itemText(0) == "global"
    assert widget.selected_coordinate_system == "global"
    assert widget.segmentation_combo.count() == 2
    assert [widget.segmentation_combo.itemText(index) for index in range(widget.segmentation_combo.count())] == [
        "blobs_labels",
        "blobs_multiscale_labels",
    ]
    assert widget.selected_segmentation_name is None
    assert widget.selected_spatialdata is sdata_blobs
    assert widget.selected_table_name is None
    assert widget.selected_feature_key is None
    assert len(viewer.layers) == 0
    assert widget._annotation_controller.labels_layer is None
    assert widget._viewer_styling_controller.labels_layer is None
    assert "Choose a labels element" in widget.selection_status.text()
    assert not widget.apply_class_button.isEnabled()


def test_widget_filters_segmentation_choices_by_selected_coordinate_system(
    qtbot, monkeypatch, sdata_blobs: SpatialData
) -> None:
    _patch_coordinate_system_names(monkeypatch, ["cells", "global"])
    viewer = make_viewer_with_shared_sdata(sdata_blobs)

    global_option = SpatialDataLabelsOption(
        labels_name="blobs_labels",
        display_name="blobs_labels",
        sdata=sdata_blobs,
        coordinate_systems=("global",),
    )
    cells_option = SpatialDataLabelsOption(
        labels_name="blobs_multiscale_labels",
        display_name="blobs_multiscale_labels",
        sdata=sdata_blobs,
        coordinate_systems=("cells",),
    )
    monkeypatch.setattr(
        widget_module,
        "get_spatialdata_labels_options_for_coordinate_system_from_sdata",
        lambda *, sdata, coordinate_system: [global_option] if coordinate_system == "global" else [cells_option],
    )

    widget = ObjectClassificationWidget(viewer)
    qtbot.addWidget(widget)

    assert widget.coordinate_system_combo.count() == 2
    assert [
        widget.coordinate_system_combo.itemText(index) for index in range(widget.coordinate_system_combo.count())
    ] == [
        "cells",
        "global",
    ]
    assert widget.selected_coordinate_system == "cells"
    assert widget.segmentation_combo.count() == 1
    assert widget.segmentation_combo.itemText(0) == "blobs_multiscale_labels"
    assert widget.selected_segmentation_name is None
    assert widget.selected_table_name is None
    assert widget.selected_feature_key is None

    with qtbot.waitSignal(widget.app_state.coordinate_system_changed) as blocker:
        widget.coordinate_system_combo.setCurrentIndex(1)

    assert blocker.args[0].previous_coordinate_system == "cells"
    assert blocker.args[0].coordinate_system == "global"
    assert blocker.args[0].source == "object_classification_widget"
    assert widget.selected_coordinate_system == "global"
    assert widget.segmentation_combo.count() == 1
    assert widget.segmentation_combo.itemText(0) == "blobs_labels"
    assert widget.selected_segmentation_name is None
    assert widget.selected_table_name is None
    assert widget.selected_feature_key is None

    select_segmentation(widget)

    assert widget.selected_segmentation_name == "blobs_labels"
    assert widget.selected_table_name == "table"
    assert widget.selected_feature_key == "features_1"


def test_widget_coordinate_system_change_updates_viewer_widget(qtbot, monkeypatch) -> None:
    _patch_coordinate_system_names(monkeypatch, ["global", "local"])
    fake_sdata = object()
    shared_option = SpatialDataLabelsOption(
        labels_name="shared_labels",
        display_name="shared_labels",
        sdata=fake_sdata,
        coordinate_systems=("global", "local"),
    )

    monkeypatch.setattr(
        widget_module,
        "get_spatialdata_labels_options_for_coordinate_system_from_sdata",
        lambda *, sdata, coordinate_system: [shared_option],
    )
    monkeypatch.setattr(widget_module, "get_annotating_table_names", lambda sdata, labels_name: [])
    monkeypatch.setattr(viewer_widget_module, "_get_labels_in_coordinate_system", lambda sdata, coordinate_system: [])
    monkeypatch.setattr(viewer_widget_module, "_get_images_in_coordinate_system", lambda sdata, coordinate_system: [])

    viewer = DummyViewer(seed_shared_sdata=False)
    app_state = get_or_create_app_state(viewer)
    app_state.set_sdata(fake_sdata)
    viewer_widget = ViewerWidget(viewer)
    object_widget = ObjectClassificationWidget(viewer)

    qtbot.addWidget(viewer_widget)
    qtbot.addWidget(object_widget)

    assert viewer_widget.coordinate_system_combo.currentText() == "global"
    assert object_widget.coordinate_system_combo.currentText() == "global"

    with qtbot.waitSignal(app_state.coordinate_system_changed) as blocker:
        object_widget.coordinate_system_combo.setCurrentIndex(1)

    assert blocker.args[0].previous_coordinate_system == "global"
    assert blocker.args[0].coordinate_system == "local"
    assert blocker.args[0].source == "object_classification_widget"
    assert viewer_widget.coordinate_system_combo.currentText() == "local"
    assert object_widget.coordinate_system_combo.currentText() == "local"


def test_shared_coordinate_system_switch_prunes_registered_layers_and_keeps_external_layers(qtbot, monkeypatch) -> None:
    _patch_coordinate_system_names(monkeypatch, ["global", "local"])
    fake_sdata = object()
    global_image = Image(np.zeros((4, 4), dtype=np.float32), name="global_image")
    local_image = Image(np.zeros((4, 4), dtype=np.float32), name="local_image")
    external_image = Image(np.zeros((4, 4), dtype=np.float32), name="external_image")
    global_labels = Labels(np.ones((4, 4), dtype=np.int32), name="global_labels")
    local_labels = Labels(np.ones((4, 4), dtype=np.int32), name="local_labels")
    external_labels = Labels(np.ones((4, 4), dtype=np.int32), name="external_labels")

    monkeypatch.setattr(viewer_widget_module, "_get_labels_in_coordinate_system", lambda sdata, coordinate_system: [])
    monkeypatch.setattr(viewer_widget_module, "_get_images_in_coordinate_system", lambda sdata, coordinate_system: [])
    monkeypatch.setattr(
        widget_module,
        "get_spatialdata_labels_options_for_coordinate_system_from_sdata",
        lambda *, sdata, coordinate_system: [],
    )

    viewer = DummyViewer(seed_shared_sdata=False)
    viewer.layers.extend([global_image, local_image, external_image, global_labels, local_labels, external_labels])
    app_state = get_or_create_app_state(viewer)
    app_state.set_sdata(fake_sdata)
    viewer_widget = ViewerWidget(viewer)
    object_widget = ObjectClassificationWidget(viewer)

    qtbot.addWidget(viewer_widget)
    qtbot.addWidget(object_widget)

    app_state.viewer_adapter.register_image_layer(
        global_image,
        sdata=fake_sdata,
        image_name="global_image",
        coordinate_system="global",
    )
    app_state.viewer_adapter.register_image_layer(
        local_image,
        sdata=fake_sdata,
        image_name="local_image",
        coordinate_system="local",
    )
    app_state.viewer_adapter.register_labels_layer(
        global_labels,
        sdata=fake_sdata,
        labels_name="global_labels",
        coordinate_system="global",
    )
    app_state.viewer_adapter.register_labels_layer(
        local_labels,
        sdata=fake_sdata,
        labels_name="local_labels",
        coordinate_system="local",
    )

    with qtbot.waitSignal(app_state.coordinate_system_changed):
        viewer_widget.coordinate_system_combo.setCurrentIndex(1)

    assert app_state.coordinate_system == "local"
    assert viewer_widget.coordinate_system_combo.currentText() == "local"
    assert object_widget.coordinate_system_combo.currentText() == "local"
    assert list(viewer.layers) == [local_image, external_image, local_labels, external_labels]
    assert app_state.viewer_adapter.layer_bindings.get_binding(global_image) is None
    assert app_state.viewer_adapter.layer_bindings.get_binding(global_labels) is None
    assert app_state.viewer_adapter.layer_bindings.get_binding(local_image) is not None
    assert app_state.viewer_adapter.layer_bindings.get_binding(local_labels) is not None
    assert app_state.viewer_adapter.layer_bindings.get_binding(external_image) is None
    assert app_state.viewer_adapter.layer_bindings.get_binding(external_labels) is None
    assert [binding.element_name for binding in app_state.viewer_adapter.layer_bindings.iter_bindings()] == [
        "local_image",
        "local_labels",
    ]


def test_widget_clears_selected_segmentation_on_coordinate_system_change_even_when_it_is_valid(
    qtbot, monkeypatch
) -> None:
    _patch_coordinate_system_names(monkeypatch, ["global", "local"])
    fake_sdata = object()
    shared_option = SpatialDataLabelsOption(
        labels_name="shared_labels",
        display_name="shared_labels",
        sdata=fake_sdata,
        coordinate_systems=("global", "local"),
    )
    global_layer = Labels(np.ones((4, 4), dtype=np.int32), name="shared_labels")

    monkeypatch.setattr(
        widget_module,
        "get_spatialdata_labels_options_for_coordinate_system_from_sdata",
        lambda *, sdata, coordinate_system: [shared_option],
    )
    monkeypatch.setattr(widget_module, "get_annotating_table_names", lambda sdata, labels_name: [])

    viewer = DummyViewer(seed_shared_sdata=False)
    app_state = get_or_create_app_state(viewer)
    app_state.set_sdata(fake_sdata)
    viewer.layers.append(global_layer)
    app_state.viewer_adapter.register_labels_layer(
        global_layer,
        sdata=fake_sdata,
        labels_name="shared_labels",
        coordinate_system="global",
    )

    monkeypatch.setattr(
        app_state.viewer_adapter,
        "ensure_labels_loaded",
        lambda sdata, labels_name, coordinate_system: (_ for _ in ()).throw(
            AssertionError("Coordinate-system switching should not auto-load a replacement segmentation layer.")
        ),
    )

    widget = ObjectClassificationWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)

    assert widget.selected_segmentation_name == "shared_labels"
    assert widget._annotation_controller.labels_layer is global_layer
    assert viewer.layers.selection.active is global_layer

    with qtbot.waitSignal(app_state.coordinate_system_changed):
        widget.coordinate_system_combo.setCurrentIndex(1)

    assert widget.selected_coordinate_system == "local"
    assert widget.selected_segmentation_name is None
    assert widget.selected_table_name is None
    assert widget.selected_feature_key is None
    assert widget._annotation_controller.labels_layer is None
    assert widget._viewer_styling_controller.labels_layer is None
    assert list(viewer.layers) == []
    assert app_state.viewer_adapter.layer_bindings.get_binding(global_layer) is None
    assert "Choose a labels element" in widget.selection_status.text()


def test_widget_unbinds_when_selected_segmentation_is_not_valid_in_new_coordinate_system(qtbot, monkeypatch) -> None:
    _patch_coordinate_system_names(monkeypatch, ["global", "local"])
    fake_sdata = object()
    global_option = SpatialDataLabelsOption(
        labels_name="global_labels",
        display_name="global_labels",
        sdata=fake_sdata,
        coordinate_systems=("global",),
    )
    local_option = SpatialDataLabelsOption(
        labels_name="local_labels",
        display_name="local_labels",
        sdata=fake_sdata,
        coordinate_systems=("local",),
    )
    global_layer = Labels(np.ones((4, 4), dtype=np.int32), name="global_labels")

    monkeypatch.setattr(
        widget_module,
        "get_spatialdata_labels_options_for_coordinate_system_from_sdata",
        lambda *, sdata, coordinate_system: [global_option] if coordinate_system == "global" else [local_option],
    )
    monkeypatch.setattr(widget_module, "get_annotating_table_names", lambda sdata, labels_name: [])

    viewer = DummyViewer(seed_shared_sdata=False)
    app_state = get_or_create_app_state(viewer)
    app_state.set_sdata(fake_sdata)
    viewer.layers.append(global_layer)
    app_state.viewer_adapter.register_labels_layer(
        global_layer,
        sdata=fake_sdata,
        labels_name="global_labels",
        coordinate_system="global",
    )

    widget = ObjectClassificationWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)

    assert widget.selected_segmentation_name == "global_labels"
    assert widget._annotation_controller.labels_layer is global_layer

    with qtbot.waitSignal(app_state.coordinate_system_changed):
        widget.coordinate_system_combo.setCurrentIndex(1)

    assert widget.selected_coordinate_system == "local"
    assert widget.selected_segmentation_name is None
    assert widget.selected_table_name is None
    assert widget.selected_feature_key is None
    assert widget._annotation_controller.labels_layer is None
    assert widget._viewer_styling_controller.labels_layer is None
    assert list(viewer.layers) == []
    assert "Choose a labels element" in widget.selection_status.text()


def test_widget_surfaces_invalid_table_binding_for_duplicate_instance_ids(qtbot, sdata_blobs: SpatialData) -> None:
    table = sdata_blobs["table"]
    first_index, second_index = table.obs.index[:2]
    table.obs.loc[second_index, "instance_id"] = table.obs.loc[first_index, "instance_id"]
    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])

    widget = ObjectClassificationWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)

    assert widget.selected_table_name == "table"
    assert widget.warning_status.isHidden()
    assert widget.warning_status.text() == ""
    assert "Table Binding Invalid" in widget.selection_status.text()
    assert "contains duplicate values within that region" in widget.selection_status.text()
    assert not widget.color_by_combo.isEnabled()
    assert not widget.class_spinbox.isEnabled()
    assert not widget.retrain_button.isEnabled()
    assert not widget.persistence_controls.write_button.isEnabled()
    assert not widget.persistence_controls.reload_button.isEnabled()


def test_widget_rejects_invalid_user_class_without_mutation_and_styles_labels_neutrally(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    table = sdata_blobs["table"]
    table.obs[USER_CLASS_COLUMN] = pd.Categorical(
        np.zeros(table.n_obs, dtype=np.int64),
        categories=[0],
    )
    previous_obs = table.obs.copy(deep=True)
    previous_uns = table.uns.copy()
    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])
    app_state = get_or_create_app_state(viewer)

    widget = ObjectClassificationWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)

    assert widget.selected_table_name == "table"
    assert "Table Binding Invalid" in widget.selection_status.text()
    assert "Object Classification state is invalid" in widget.selection_status.text()
    assert "user_class" in widget.selection_status.text()
    assert "positive integer categories" in widget.selection_status.text()
    assert widget._annotation_controller._selected_table_name is None
    assert widget._classifier_controller._selected_table_name is None
    assert widget._viewer_styling_controller._selected_table_name is None
    assert widget.persistence_controls.controller._selected_table_name is None
    assert isinstance(layer.colormap, DirectLabelColormap)
    np.testing.assert_allclose(layer.colormap.map(0), np.zeros(4, dtype=np.float32))
    np.testing.assert_allclose(layer.colormap.map(5), np.asarray(to_rgba(DEFAULT_NEUTRAL_COLOR), dtype=np.float32))
    assert not widget.class_spinbox.isEnabled()
    assert not widget.apply_class_button.isEnabled()
    assert not widget.clear_class_button.isEnabled()
    assert not widget.feature_matrix_combo.isEnabled()
    assert not widget.register_feature_matrix_button.isEnabled()
    assert not widget.color_by_combo.isEnabled()
    assert not widget.auto_train_checkbox.isEnabled()
    assert not widget.retrain_button.isEnabled()
    assert not widget.export_classifier_button.isEnabled()
    assert not widget.persistence_controls.write_button.isEnabled()
    assert not widget.persistence_controls.reload_button.isEnabled()
    assert widget.warning_status.isHidden()
    pd.testing.assert_frame_equal(table.obs, previous_obs)
    assert table.uns == previous_uns
    assert not app_state.is_table_dirty(sdata_blobs, "table")

    table.obs[USER_CLASS_COLUMN] = pd.Categorical(
        [1, *([pd.NA] * (table.n_obs - 1))],
        categories=[1],
    )
    widget._bind_current_selection()

    assert widget._table_binding_error is None
    assert widget._annotation_controller._selected_table_name == "table"
    assert widget._classifier_controller._selected_table_name == "table"
    assert widget.color_by_combo.isEnabled()
    assert widget.auto_train_checkbox.isEnabled()


def test_widget_auto_loads_selected_segmentation_when_shared_sdata_is_set(qtbot, sdata_blobs: SpatialData) -> None:
    viewer = DummyViewer()
    app_state = get_or_create_app_state(viewer)
    widget = ObjectClassificationWidget(viewer)
    qtbot.addWidget(widget)

    assert widget.segmentation_combo.count() == 0

    app_state.set_sdata(sdata_blobs)
    assert widget.coordinate_system_combo.count() == 1
    assert widget.coordinate_system_combo.itemText(0) == "global"
    assert widget.selected_coordinate_system == "global"
    assert widget.segmentation_combo.count() == 2
    assert len(viewer.layers) == 0
    assert widget.table_combo.count() == 0
    assert widget.feature_matrix_combo.count() == 0
    assert widget.selected_segmentation_name is None
    assert widget.selected_table_name is None
    assert widget.selected_feature_key is None
    assert "Choose a labels element" in widget.selection_status.text()


def test_widget_updates_table_dropdown_when_segmentation_changes(qtbot, sdata_blobs: SpatialData) -> None:
    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])

    widget = ObjectClassificationWidget(viewer)
    qtbot.addWidget(widget)

    widget.segmentation_combo.setCurrentIndex(1)

    assert widget.selected_segmentation_name == "blobs_multiscale_labels"
    assert widget.table_combo.count() == 0
    assert not widget.table_combo.isEnabled()
    assert widget.selected_table_name is None
    assert widget.feature_matrix_combo.count() == 0
    assert not widget.feature_matrix_combo.isEnabled()
    assert widget.selected_feature_key is None


def test_widget_warns_when_loaded_segmentation_has_no_annotation_table(qtbot, sdata_blobs: SpatialData) -> None:
    primary_layer = make_blobs_labels_layer(sdata_blobs)
    base_data = np.asarray(sdata_blobs.labels["blobs_labels"])
    multiscale_layer = Labels(
        [base_data, base_data[::2, ::2]],
        name="blobs_multiscale_labels",
        metadata={
            "sdata": sdata_blobs,
            "name": "blobs_multiscale_labels",
            "indices": [int(value) for value in np.unique(base_data).tolist() if int(value) > 0],
        },
    )
    viewer = DummyViewer(layers=[primary_layer, multiscale_layer])

    widget = ObjectClassificationWidget(viewer)
    qtbot.addWidget(widget)

    widget.segmentation_combo.setCurrentIndex(1)

    assert widget.selected_segmentation_name == "blobs_multiscale_labels"
    assert widget.selected_table_name is None
    assert viewer.layers.selection.active is multiscale_layer
    assert "This labels layer is loaded, but no annotation table is linked to it." in widget.selection_status.text()
    assert not widget.class_spinbox.isEnabled()
    assert not widget.apply_class_button.isEnabled()
    assert isinstance(multiscale_layer.colormap, CompactLabelColormap)
    np.testing.assert_allclose(multiscale_layer.colormap.map(0), np.zeros(4, dtype=np.float32))
    np.testing.assert_allclose(
        multiscale_layer.colormap.map(1), np.asarray(to_rgba(DEFAULT_NEUTRAL_COLOR), dtype=np.float32)
    )


def test_widget_updates_selected_feature_key_when_feature_matrix_changes(qtbot, sdata_blobs: SpatialData) -> None:
    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])

    widget = ObjectClassificationWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)
    widget.training_scope_combo.setCurrentIndex(widget.training_scope_combo.findData("selected_segmentation_only"))

    bind_calls: list[tuple[str, str]] = []

    def record_bind(
        sdata,
        labels_name,
        table_name,
        feature_key,
        *,
        training_scope=classifier_module.DEFAULT_TRAINING_SCOPE,
        prediction_scope=classifier_module.DEFAULT_PREDICTION_SCOPE,
    ) -> bool:
        del sdata, labels_name, table_name, feature_key
        bind_calls.append((training_scope, prediction_scope))
        return True

    widget._classifier_controller.bind = record_bind  # type: ignore[method-assign]

    widget.feature_matrix_combo.setCurrentIndex(1)

    assert widget.selected_feature_key == "features_2"
    assert bind_calls == [("selected_segmentation_only", classifier_module.DEFAULT_PREDICTION_SCOPE)]
    assert "feature matrix changed" in widget.classifier_feedback.text()


def test_widget_feature_matrix_registration_button_enables_for_unregistered_matrix(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    table = sdata_blobs["table"]
    table.uns.pop(_FEATURE_MATRICES_KEY, None)
    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])

    widget = ObjectClassificationWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)

    assert widget.register_feature_matrix_button.objectName() == "register_feature_matrix_button"
    assert widget.register_feature_matrix_button.isEnabled()
    assert "Register feature-column metadata" in _tooltip_text(widget.register_feature_matrix_button)
    assert widget.warning_status.isHidden()


def test_widget_preserves_valid_custom_user_class_palette_during_binding_and_styling(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    table = sdata_blobs["table"]
    instance_ids = table.obs["instance_id"].to_numpy(dtype=np.int64)
    table.obs[USER_CLASS_COLUMN] = pd.Categorical(
        [1 if int(instance_id) == 1 else pd.NA for instance_id in instance_ids],
        categories=[1],
    )
    table.uns[USER_CLASS_COLORS_KEY] = ["#123456"]
    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])

    widget = ObjectClassificationWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)

    assert table.uns[USER_CLASS_COLORS_KEY] == ["#123456"]
    assert widget.warning_status.isHidden()
    np.testing.assert_allclose(layer.colormap.map(1), np.asarray(to_rgba("#123456"), dtype=np.float32))


def test_widget_disables_retrain_button_for_unregistered_feature_matrix_metadata(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    table = sdata_blobs["table"]
    instance_ids = table.obs["instance_id"].to_numpy(dtype=np.int64)
    table.obs[USER_CLASS_COLUMN] = pd.Categorical(
        [
            1 if int(instance_id) in {1, 2} else 2 if int(instance_id) in {24, 25} else pd.NA
            for instance_id in instance_ids
        ],
        categories=[1, 2],
    )
    table.uns.pop(_FEATURE_MATRICES_KEY, None)
    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])

    widget = ObjectClassificationWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)

    assert widget.register_feature_matrix_button.isEnabled()
    assert widget.retrain_button.isEnabled() is False
    assert "Register feature metadata" in _tooltip_text(widget.retrain_button)
    assert "Register feature metadata" in widget.classifier_preparation_status.text()

    widget.auto_train_checkbox.setChecked(True)
    layer.selected_label = 5
    widget.class_spinbox.setValue(2)
    widget.apply_class_button.click()

    assert widget._classifier_controller.is_training is False
    assert CLASSIFIER_CONFIG_KEY not in table.uns
    assert "Register feature metadata" in widget._classifier_controller.status_message


def test_widget_register_feature_matrix_button_registers_metadata_and_recovers_training(
    qtbot,
    monkeypatch,
    backed_sdata_blobs: SpatialData,
) -> None:
    table = backed_sdata_blobs["table"]
    instance_ids = table.obs["instance_id"].to_numpy(dtype=np.int64)
    table.obs[USER_CLASS_COLUMN] = pd.Categorical(
        [
            1 if int(instance_id) in {1, 2} else 2 if int(instance_id) in {24, 25} else pd.NA
            for instance_id in instance_ids
        ],
        categories=[1, 2],
    )
    table.uns.pop(_FEATURE_MATRICES_KEY, None)
    layer = make_blobs_labels_layer(backed_sdata_blobs)
    viewer = DummyViewer(layers=[layer])

    widget = ObjectClassificationWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)
    mark_dirty_reasons: list[str | None] = []
    monkeypatch.setattr(
        widget._classifier_controller,
        "mark_dirty",
        lambda *, reason=None: mark_dirty_reasons.append(reason),
    )
    emitted_events: list[object] = []
    widget.app_state.table_state_changed.connect(emitted_events.append)

    assert widget.register_feature_matrix_button.isEnabled()
    assert widget.retrain_button.isEnabled() is False
    assert widget.persistence_controls.controller.has_unsynced_table_changes is False
    assert not widget.persistence_controls.write_button.isEnabled()

    widget.register_feature_matrix_button.click()

    metadata = table.uns[_FEATURE_MATRICES_KEY]["features_1"]
    assert metadata["source_kind"] == CUSTOM_OBSM_SOURCE_KIND
    assert widget.register_feature_matrix_button.isEnabled() is False
    assert widget.retrain_button.isEnabled()
    assert widget.warning_status.isHidden()
    assert widget.persistence_controls.controller.has_unsynced_table_changes is True
    assert widget.persistence_controls.write_button.isEnabled()
    assert mark_dirty_reasons == ["feature matrix metadata registered"]
    assert len(emitted_events) == 1
    assert isinstance(emitted_events[0], TableStateChangedEvent)
    assert emitted_events[0].regions == ()
    assert emitted_events[0].source == "object_classification_feature_metadata"


def test_widget_register_feature_matrix_button_shows_error_without_dirty_side_effects(
    qtbot,
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    table = sdata_blobs["table"]
    table.uns.pop(_FEATURE_MATRICES_KEY, None)
    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])

    widget = ObjectClassificationWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)
    mark_dirty_reasons: list[str | None] = []
    monkeypatch.setattr(
        widget._classifier_controller,
        "mark_dirty",
        lambda *, reason=None: mark_dirty_reasons.append(reason),
    )

    def fail_registration(*args, **kwargs) -> None:
        del args, kwargs
        raise ValueError("registration failed")

    monkeypatch.setattr(widget_module, "register_feature_matrix_metadata", fail_registration)

    widget.register_feature_matrix_button.click()

    assert _FEATURE_MATRICES_KEY not in table.uns
    _assert_feature_metadata_warning_card(widget)
    assert "registration failed" in widget.warning_status.text()
    assert widget.register_feature_matrix_button.isEnabled()
    assert widget.persistence_controls.controller.has_unsynced_table_changes is False
    assert mark_dirty_reasons == []


def test_widget_register_feature_matrix_button_ignores_stale_click_after_external_registration(
    qtbot,
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    table = sdata_blobs["table"]
    table.uns.pop(_FEATURE_MATRICES_KEY, None)
    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])

    widget = ObjectClassificationWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)
    assert widget.register_feature_matrix_button.isEnabled()

    register_feature_matrix_metadata(table, "features_1")
    registration_calls: list[str] = []
    mark_dirty_reasons: list[str | None] = []
    monkeypatch.setattr(
        widget_module,
        "register_feature_matrix_metadata",
        lambda *args, **kwargs: registration_calls.append("register"),
    )
    monkeypatch.setattr(
        widget._classifier_controller,
        "mark_dirty",
        lambda *, reason=None: mark_dirty_reasons.append(reason),
    )

    widget.register_feature_matrix_button.click()

    assert registration_calls == []
    assert mark_dirty_reasons == []
    assert widget.persistence_controls.controller.has_unsynced_table_changes is False
    assert widget.register_feature_matrix_button.isEnabled() is False


def test_widget_feature_matrix_registration_button_disables_for_valid_custom_metadata(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    table = sdata_blobs["table"]
    register_feature_matrix_metadata(table, "features_1", overwrite=True)
    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])

    widget = ObjectClassificationWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)

    assert widget.register_feature_matrix_button.isEnabled() is False
    assert 'already registered as a custom ".obsm" feature matrix' in _tooltip_text(
        widget.register_feature_matrix_button
    )
    assert table.uns[_FEATURE_MATRICES_KEY]["features_1"]["source_kind"] == CUSTOM_OBSM_SOURCE_KIND
    assert widget.warning_status.isHidden()


def test_widget_feature_matrix_registration_button_disables_for_valid_harpy_metadata(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    _set_feature_metadata(sdata_blobs)
    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])

    widget = ObjectClassificationWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)

    assert widget.register_feature_matrix_button.isEnabled() is False
    assert "already registered from Harpy feature extraction" in _tooltip_text(widget.register_feature_matrix_button)
    assert widget.warning_status.isHidden()


def test_widget_feature_matrix_registration_button_warns_for_invalid_matrix(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    table = sdata_blobs["table"]
    table.obsm["bad_features"] = np.ones((table.n_obs, 2, 2), dtype=np.float64)
    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])

    widget = ObjectClassificationWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)
    widget.feature_matrix_combo.setCurrentIndex(widget.feature_matrix_combo.findData("bad_features"))

    assert widget.selected_feature_key == "bad_features"
    assert widget.register_feature_matrix_button.isEnabled() is False
    _assert_feature_metadata_warning_card(widget)
    assert "cannot be registered" in widget.warning_status.text()
    assert "2-dimensional" in widget.warning_status.text()
    assert "cannot be registered" in _tooltip_text(widget.register_feature_matrix_button)


def test_widget_feature_matrix_registration_button_warns_for_missing_source_kind(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    table = sdata_blobs["table"]
    table.uns[_FEATURE_MATRICES_KEY] = {
        "features_1": {
            "feature_columns": [f"feature_{index}" for index in range(table.obsm["features_1"].shape[1])],
            "features": ["custom"],
        },
    }
    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])

    widget = ObjectClassificationWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)

    assert widget.register_feature_matrix_button.isEnabled() is False
    _assert_feature_metadata_warning_card(widget)
    assert "mismatched metadata" in widget.warning_status.text()
    assert "source_kind" in widget.warning_status.text()
    assert "avoid overwriting existing metadata" in _tooltip_text(widget.register_feature_matrix_button)


def test_widget_feature_matrix_registration_button_warns_for_mismatched_metadata(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    table = sdata_blobs["table"]
    table.uns[_FEATURE_MATRICES_KEY] = {
        "features_1": {
            "feature_columns": ["feature_0"],
            "features": ["feature_0"],
            "source_kind": HARPY_ADD_FEATURE_MATRIX_SOURCE_KIND,
        },
    }
    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])

    widget = ObjectClassificationWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)

    assert widget.register_feature_matrix_button.isEnabled() is False
    _assert_feature_metadata_warning_card(widget)
    assert "mismatched metadata" in widget.warning_status.text()
    assert "metadata describes 1 feature column" in widget.warning_status.text()
    assert "avoid overwriting existing metadata" in _tooltip_text(widget.register_feature_matrix_button)


def test_widget_marks_classifier_dirty_when_training_scope_changes(
    qtbot, monkeypatch, sdata_blobs: SpatialData
) -> None:
    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])

    widget = ObjectClassificationWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)

    bind_calls: list[tuple[str, str]] = []
    mark_dirty_reasons: list[str | None] = []

    def record_bind(
        sdata,
        labels_name,
        table_name,
        feature_key,
        *,
        training_scope=classifier_module.DEFAULT_TRAINING_SCOPE,
        prediction_scope=classifier_module.DEFAULT_PREDICTION_SCOPE,
    ) -> bool:
        del sdata, labels_name, table_name, feature_key
        bind_calls.append((training_scope, prediction_scope))
        return True

    def record_mark_dirty(*, reason: str | None = None) -> None:
        mark_dirty_reasons.append(reason)

    monkeypatch.setattr(widget._classifier_controller, "bind", record_bind)
    monkeypatch.setattr(widget._classifier_controller, "mark_dirty", record_mark_dirty)

    widget.training_scope_combo.setCurrentIndex(widget.training_scope_combo.findData("selected_segmentation_only"))

    assert widget.selected_training_scope == "selected_segmentation_only"
    assert bind_calls == [("selected_segmentation_only", classifier_module.DEFAULT_PREDICTION_SCOPE)]
    assert mark_dirty_reasons == ["the training scope changed"]


def test_widget_marks_classifier_dirty_when_prediction_scope_changes(
    qtbot, monkeypatch, sdata_blobs: SpatialData
) -> None:
    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])

    widget = ObjectClassificationWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)

    bind_calls: list[tuple[str, str]] = []
    mark_dirty_reasons: list[str | None] = []

    def record_bind(
        sdata,
        labels_name,
        table_name,
        feature_key,
        *,
        training_scope=classifier_module.DEFAULT_TRAINING_SCOPE,
        prediction_scope=classifier_module.DEFAULT_PREDICTION_SCOPE,
    ) -> bool:
        del sdata, labels_name, table_name, feature_key
        bind_calls.append((training_scope, prediction_scope))
        return True

    def record_mark_dirty(*, reason: str | None = None) -> None:
        mark_dirty_reasons.append(reason)

    monkeypatch.setattr(widget._classifier_controller, "bind", record_bind)
    monkeypatch.setattr(widget._classifier_controller, "mark_dirty", record_mark_dirty)

    widget.prediction_scope_combo.setCurrentIndex(widget.prediction_scope_combo.findData("all"))

    assert widget.selected_prediction_scope == "all"
    assert bind_calls == [(classifier_module.DEFAULT_TRAINING_SCOPE, "all")]
    assert mark_dirty_reasons == ["the prediction scope changed"]


def test_widget_shows_classifier_preparation_hidden_write_notice_for_table_wide_prediction_scope(
    qtbot, sdata_blobs_multi_region: SpatialData
) -> None:
    table = sdata_blobs_multi_region["table_multi"]
    _set_feature_metadata(sdata_blobs_multi_region, table_name="table_multi")
    region_values = table.obs["region"].astype("string")
    instance_values = table.obs["instance_id"].to_numpy(dtype=np.int64)
    class_values = np.full(table.n_obs, pd.NA, dtype=object)
    class_values[(region_values == "blobs_labels").to_numpy() & np.isin(instance_values, [1, 2])] = 1
    class_values[(region_values == "blobs_labels").to_numpy() & np.isin(instance_values, [24, 25])] = 2
    table.obs[USER_CLASS_COLUMN] = pd.Categorical(class_values, categories=[1, 2])
    layer = make_blobs_labels_layer(sdata_blobs_multi_region)
    viewer = DummyViewer(layers=[layer])

    widget = ObjectClassificationWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)
    table_index = widget.table_combo.findData("table_multi")
    assert table_index >= 0
    widget.table_combo.setCurrentIndex(table_index)

    widget.prediction_scope_combo.setCurrentIndex(widget.prediction_scope_combo.findData("all"))

    assert not widget.classifier_preparation_status.isHidden()
    assert f"Prediction rows: {table.n_obs}" in widget.classifier_preparation_status.text()
    assert "Prediction scope: 2 labels elements" in widget.classifier_preparation_status.text()
    assert "Some prediction updates may not be visible in the current selection." in (
        widget.classifier_preparation_status.text()
    )
    assert f"color: {STATUS_CARD_PALETTE['success']['text']}" in widget.classifier_preparation_status.styleSheet()


def test_widget_omits_hidden_write_line_for_effectively_selected_prediction_scope(
    qtbot, sdata_blobs: SpatialData
) -> None:
    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])

    widget = ObjectClassificationWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)

    widget.prediction_scope_combo.setCurrentIndex(widget.prediction_scope_combo.findData("all"))

    assert widget.selected_prediction_scope == "all"
    assert not widget.classifier_preparation_status.isHidden()
    assert "Prediction rows:" in widget.classifier_preparation_status.text()
    assert "Some prediction updates may not be visible" not in widget.classifier_preparation_status.text()


def test_widget_shows_eligible_classifier_preparation_summary(qtbot, sdata_blobs: SpatialData) -> None:
    table = sdata_blobs["table"]
    _set_feature_metadata(sdata_blobs)
    instance_ids = table.obs["instance_id"].to_numpy(dtype=np.int64)
    table.obs[USER_CLASS_COLUMN] = pd.Categorical(
        [
            1 if int(instance_id) in {1, 2} else 2 if int(instance_id) in {24, 25} else pd.NA
            for instance_id in instance_ids
        ],
        categories=[1, 2],
    )
    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])

    widget = ObjectClassificationWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)

    assert not widget.classifier_preparation_status.isHidden()
    preparation_text = widget.classifier_preparation_status.text()
    assert "Training annotations: 4 rows" in preparation_text
    assert "Training elements: 1 labels element" in preparation_text
    assert f"Prediction rows: {table.n_obs} rows" in preparation_text
    assert "Prediction scope: selected labels element" in preparation_text
    assert 'Features: "features_1", 4 features' in preparation_text
    assert "Need at least" not in preparation_text


def test_widget_disables_retrain_button_when_preparation_is_not_trainable(qtbot, sdata_blobs: SpatialData) -> None:
    table = sdata_blobs["table"]
    instance_ids = table.obs["instance_id"].to_numpy(dtype=np.int64)
    table.obs[USER_CLASS_COLUMN] = pd.Categorical(
        [1 if int(instance_id) in {1, 2} else pd.NA for instance_id in instance_ids],
        categories=[1],
    )
    _set_feature_metadata(sdata_blobs)
    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])

    widget = ObjectClassificationWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)

    tooltip = unescape(widget.retrain_button.toolTip()).replace("&#8203;", "").replace("\u200b", "")

    assert not widget.retrain_button.isEnabled()
    assert "Need at least two labeled classes" in widget.classifier_preparation_status.text()
    assert "Need at least two labeled classes" in tooltip


def test_widget_refreshes_feature_matrix_selector_when_first_key_is_written(qtbot, sdata_blobs: SpatialData) -> None:
    table = sdata_blobs["table"]
    for key in list(table.obsm.keys()):
        del table.obsm[key]

    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])
    app_state = get_or_create_app_state(viewer)
    widget = ObjectClassificationWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)
    mark_dirty_reasons: list[str | None] = []

    def record_mark_dirty(*, reason: str | None = None) -> None:
        mark_dirty_reasons.append(reason)

    widget._classifier_controller.mark_dirty = record_mark_dirty  # type: ignore[method-assign]

    assert widget.selected_feature_key is None
    assert widget.persistence_controls.controller.has_unsynced_table_changes is False

    table.obsm["features_new"] = np.arange(table.n_obs, dtype=np.float64).reshape(table.n_obs, 1)
    app_state.record_table_mutation(
        _feature_table_event(
            sdata_blobs,
            table_name="table",
            feature_key="features_new",
        )
    )

    assert widget.feature_matrix_combo.count() == 1
    assert widget.selected_feature_key == "features_new"
    assert widget.persistence_controls.controller.has_unsynced_table_changes is True
    assert mark_dirty_reasons == []


def test_widget_invalidates_classifier_when_selected_feature_matrix_is_overwritten(
    qtbot, sdata_blobs: SpatialData
) -> None:
    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])
    app_state = get_or_create_app_state(viewer)
    widget = ObjectClassificationWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)
    table = sdata_blobs["table"]
    instance_ids = table.obs["instance_id"].to_numpy(dtype=np.int64)
    table.obs[USER_CLASS_COLUMN] = pd.Categorical(
        [
            1 if int(instance_id) in {1, 2} else 2 if int(instance_id) in {24, 25} else pd.NA
            for instance_id in instance_ids
        ],
        categories=[1, 2],
    )
    _set_feature_metadata(sdata_blobs)
    training_scope = classifier_module.ResolvedClassifierScope(
        mode="selected_segmentation_only",
        regions=("blobs_labels",),
        raw_table_row_positions=np.array([0, 1], dtype=np.int64),
        table_row_positions=np.array([0, 1], dtype=np.int64),
    )
    prediction_scope = classifier_module.ResolvedClassifierScope(
        mode="selected_segmentation_only",
        regions=("blobs_labels",),
        raw_table_row_positions=np.array([0, 1], dtype=np.int64),
        table_row_positions=np.array([0, 1], dtype=np.int64),
    )
    worker = _DeferredWorker(
        classifier_module.ClassifierJobResult(
            job_id=1,
            feature_key="features_1",
            labels_name="blobs_labels",
            table_name="table",
            pred_classes=np.array([1, 2], dtype=np.int64),
            pred_confidences=np.array([0.9, 0.8], dtype=np.float64),
            trained_at="2026-04-23T09:00:00+00:00",
            model_params=dict(classifier_module.RANDOM_FOREST_PARAMS),
            summary=classifier_module.ClassifierPreparationSummary(
                training_scope=training_scope,
                prediction_scope=prediction_scope,
                reason="Ready to train.",
                labeled_count=2,
                class_labels=(1, 2),
                n_features=2,
            ),
        )
    )

    widget._classifier_controller._create_training_worker = lambda job: worker  # type: ignore[method-assign]

    assert widget._classifier_controller.schedule_retrain(immediate=True) is True
    assert widget._classifier_controller.is_training is True

    table.obsm["features_1"] = np.arange(table.n_obs * 2, dtype=np.float64).reshape(table.n_obs, 2)
    app_state.record_table_mutation(
        _feature_table_event(
            sdata_blobs,
            table_name="table",
            feature_key="features_1",
            change_kind="updated",
        )
    )

    assert worker.quit_called is True
    assert widget._classifier_controller.is_training is False
    assert widget._classifier_controller.is_dirty is True
    assert widget.selected_feature_key == "features_1"
    assert widget.persistence_controls.controller.has_unsynced_table_changes is True
    assert "overwritten" in widget.classifier_feedback.text()


def test_widget_ignores_feature_matrix_writes_for_other_tables(qtbot, sdata_blobs: SpatialData) -> None:
    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])
    app_state = get_or_create_app_state(viewer)
    widget = ObjectClassificationWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)
    previous_items = [
        widget.feature_matrix_combo.itemText(index) for index in range(widget.feature_matrix_combo.count())
    ]

    app_state.record_table_mutation(
        _feature_table_event(
            sdata_blobs,
            table_name="other_table",
            feature_key="features_new",
        )
    )

    assert [
        widget.feature_matrix_combo.itemText(index) for index in range(widget.feature_matrix_combo.count())
    ] == previous_items
    assert widget.selected_feature_key == "features_1"
    assert widget.persistence_controls.controller.has_unsynced_table_changes is False


def test_widget_ignores_non_feature_matrix_write_events(qtbot, sdata_blobs: SpatialData) -> None:
    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])
    widget = ObjectClassificationWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)

    previous_table_items = _combo_texts(widget.table_combo)
    previous_feature_items = _combo_texts(widget.feature_matrix_combo)

    widget._on_table_state_changed(object())

    assert _combo_texts(widget.table_combo) == previous_table_items
    assert _combo_texts(widget.feature_matrix_combo) == previous_feature_items
    assert widget.selected_table_name == "table"
    assert widget.selected_feature_key == "features_1"


def test_widget_ignores_feature_matrix_writes_for_other_sdata(
    qtbot,
    sdata_blobs: SpatialData,
    sdata_blobs_multi_region: SpatialData,
) -> None:
    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])
    app_state = get_or_create_app_state(viewer)
    widget = ObjectClassificationWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)

    previous_table_items = _combo_texts(widget.table_combo)
    previous_feature_items = _combo_texts(widget.feature_matrix_combo)

    app_state.record_table_mutation(
        _feature_table_event(
            sdata_blobs_multi_region,
            table_name="table_multi",
            feature_key="features_new",
        )
    )

    assert _combo_texts(widget.table_combo) == previous_table_items
    assert _combo_texts(widget.feature_matrix_combo) == previous_feature_items
    assert widget.selected_table_name == "table"
    assert widget.selected_feature_key == "features_1"


@pytest.mark.parametrize("auto_train_enabled", [False, True])
def test_widget_consumes_spatial_query_user_class_event_without_republishing(
    qtbot,
    monkeypatch,
    sdata_blobs: SpatialData,
    auto_train_enabled: bool,
) -> None:
    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])
    app_state = get_or_create_app_state(viewer)
    widget = ObjectClassificationWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)

    table = sdata_blobs.tables["table"]
    table.obs[USER_CLASS_COLUMN] = pd.Categorical(
        [1, 2, *([pd.NA] * (table.n_obs - 2))],
        categories=[1, 2],
    )
    table.uns.pop(USER_CLASS_COLORS_KEY, None)

    mark_dirty_reasons: list[str | None] = []
    schedule_calls: list[str] = []
    full_styling_calls: list[str] = []
    monkeypatch.setattr(
        widget._classifier_controller,
        "mark_dirty",
        lambda *, reason=None: mark_dirty_reasons.append(reason),
    )
    monkeypatch.setattr(
        widget._classifier_controller,
        "schedule_retrain",
        lambda: schedule_calls.append("schedule") or True,
    )
    monkeypatch.setattr(
        widget,
        "_refresh_layer_styling",
        lambda: full_styling_calls.append("refresh"),
    )
    widget.auto_train_checkbox.setChecked(auto_train_enabled)

    emitted_events: list[TableStateChangedEvent] = []
    app_state.table_state_changed.connect(emitted_events.append)
    event = _spatial_query_annotation_event(sdata_blobs)
    app_state.record_table_mutation(event)

    assert full_styling_calls == ["refresh"]
    assert mark_dirty_reasons == ["the annotations changed"]
    assert schedule_calls == (["schedule"] if auto_train_enabled else [])
    assert USER_CLASS_COLORS_KEY not in table.uns
    assert emitted_events == [event]
    assert app_state.is_table_dirty(sdata_blobs, "table")


@pytest.mark.parametrize(
    ("event_kind", "expected_source"),
    [
        ("other_sdata", "spatial_query_annotation"),
        ("other_table", "spatial_query_annotation"),
        ("other_source", "object_classification_annotation"),
        ("other_path", "spatial_query_annotation"),
    ],
)
def test_widget_ignores_unrelated_spatial_query_annotation_events(
    qtbot,
    monkeypatch,
    sdata_blobs: SpatialData,
    sdata_blobs_multi_region: SpatialData,
    event_kind: str,
    expected_source: str,
) -> None:
    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])
    widget = ObjectClassificationWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)

    rebind_calls: list[str] = []
    monkeypatch.setattr(widget, "_bind_current_selection", lambda: rebind_calls.append("bind"))

    event_sdata = sdata_blobs_multi_region if event_kind == "other_sdata" else sdata_blobs
    table_name = "other_table" if event_kind == "other_table" else "table"
    paths = frozenset({TableComponentPath("uns", (USER_CLASS_COLORS_KEY,))}) if event_kind == "other_path" else None
    widget._on_table_state_changed(
        _spatial_query_annotation_event(
            event_sdata,
            table_name=table_name,
            paths=paths,
            source=expected_source,
        )
    )

    assert rebind_calls == []


def test_widget_rejects_invalid_spatial_query_user_class_state_without_retraining(
    qtbot,
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])
    app_state = get_or_create_app_state(viewer)
    widget = ObjectClassificationWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)

    table = sdata_blobs.tables["table"]
    table.obs[USER_CLASS_COLUMN] = pd.Categorical(
        np.zeros(table.n_obs, dtype=np.int64),
        categories=[0],
    )
    mark_dirty_reasons: list[str | None] = []
    schedule_calls: list[str] = []
    monkeypatch.setattr(
        widget._classifier_controller,
        "mark_dirty",
        lambda *, reason=None: mark_dirty_reasons.append(reason),
    )
    monkeypatch.setattr(
        widget._classifier_controller,
        "schedule_retrain",
        lambda: schedule_calls.append("schedule") or True,
    )
    widget.auto_train_checkbox.setChecked(True)

    app_state.record_table_mutation(_spatial_query_annotation_event(sdata_blobs))

    assert widget._table_binding_error is not None
    assert widget._annotation_controller.labels_layer is layer
    assert widget._classifier_controller._selected_table_name is None
    assert isinstance(layer.colormap, DirectLabelColormap)
    np.testing.assert_allclose(layer.colormap.map(0), np.zeros(4, dtype=np.float32))
    np.testing.assert_allclose(layer.colormap.map(5), np.asarray(to_rgba(DEFAULT_NEUTRAL_COLOR), dtype=np.float32))
    assert "Table Binding Invalid" in widget.selection_status.text()
    assert mark_dirty_reasons == []
    assert schedule_calls == []


@pytest.mark.parametrize(
    ("source", "paths"),
    [
        (
            "spatial_query_canonical_centers",
            frozenset(
                {
                    TableComponentPath("obsm", ("spatial_canonical",)),
                    TableComponentPath("uns", ("spatial_coordinates", "spatial_canonical")),
                }
            ),
        ),
        (
            "spatial_query_annotation",
            frozenset({TableComponentPath("obs", ("another_annotation",))}),
        ),
    ],
)
def test_widget_refreshes_persistence_for_any_selected_table_event(
    qtbot,
    monkeypatch,
    backed_sdata_blobs: SpatialData,
    source: str,
    paths: frozenset[TableComponentPath],
) -> None:
    """Keep Write Table State synchronized for every selected-table mutation.

    Persistence readiness is table-wide, so domain-specific event filtering
    must not hide changes produced by other spatiato widgets or components.
    """
    layer = make_blobs_labels_layer(backed_sdata_blobs)
    viewer = DummyViewer(layers=[layer])
    app_state = get_or_create_app_state(viewer)
    widget = ObjectClassificationWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)
    domain_calls: list[str] = []
    monkeypatch.setattr(widget, "_bind_current_selection", lambda: domain_calls.append("bind"))
    monkeypatch.setattr(
        widget._classifier_controller,
        "mark_dirty",
        lambda *, reason=None: domain_calls.append(f"dirty:{reason}"),
    )

    assert not widget.persistence_controls.write_button.isEnabled()

    app_state.record_table_mutation(
        TableStateChangedEvent(
            sdata=backed_sdata_blobs,
            table_name="table",
            paths=paths,
            regions=("blobs_labels",),
            change_kind="updated",
            source=source,
        )
    )

    assert widget.persistence_controls.write_button.isEnabled()
    assert widget.persistence_controls.controller.has_unsynced_table_changes is True
    assert domain_calls == []


@pytest.mark.parametrize("event_identity", ["other_sdata", "other_table"])
def test_widget_does_not_refresh_persistence_for_unrelated_table_event(
    qtbot,
    monkeypatch,
    backed_sdata_blobs: SpatialData,
    sdata_blobs_multi_region: SpatialData,
    event_identity: str,
) -> None:
    layer = make_blobs_labels_layer(backed_sdata_blobs)
    viewer = DummyViewer(layers=[layer])
    app_state = get_or_create_app_state(viewer)
    widget = ObjectClassificationWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)
    refresh_calls: list[str] = []
    monkeypatch.setattr(
        widget.persistence_controls,
        "refresh",
        lambda: refresh_calls.append("refresh"),
    )

    event_sdata = sdata_blobs_multi_region if event_identity == "other_sdata" else backed_sdata_blobs
    table_name = "table_multi" if event_identity == "other_sdata" else "other_table"
    app_state.record_table_mutation(
        TableStateChangedEvent(
            sdata=event_sdata,
            table_name=table_name,
            paths=frozenset({TableComponentPath("obs", ("another_annotation",))}),
            regions=("blobs_labels",),
            change_kind="updated",
            source="spatial_query_annotation",
        )
    )

    assert refresh_calls == []
    assert not widget.persistence_controls.write_button.isEnabled()


@pytest.mark.parametrize("clean_transition", ["persisted_change", "reload"])
def test_widget_disables_write_when_shared_table_event_cleans_selected_table(
    qtbot,
    backed_sdata_blobs: SpatialData,
    clean_transition: str,
) -> None:
    """Disable Write Table State after persistence or reload cleans the shared manifest."""
    layer = make_blobs_labels_layer(backed_sdata_blobs)
    viewer = DummyViewer(layers=[layer])
    app_state = get_or_create_app_state(viewer)
    widget = ObjectClassificationWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)
    dirty_event = TableStateChangedEvent(
        sdata=backed_sdata_blobs,
        table_name="table",
        paths=frozenset({TableComponentPath("obs", ("another_annotation",))}),
        regions=("blobs_labels",),
        change_kind="updated",
        source="spatial_query_annotation",
    )
    app_state.record_table_mutation(dirty_event)
    snapshot = app_state.snapshot_table_dirty_state(backed_sdata_blobs, "table")

    assert widget.persistence_controls.write_button.isEnabled()

    if clean_transition == "persisted_change":
        app_state.record_persisted_table_change(dirty_event, snapshot)
    else:
        app_state.record_table_reload(
            TableStateChangedEvent(
                sdata=backed_sdata_blobs,
                table_name="table",
                paths=dirty_event.paths,
                regions=("blobs_labels",),
                change_kind="reloaded",
                source="persistence_controller",
            )
        )

    assert app_state.is_table_dirty(backed_sdata_blobs, "table") is False
    assert not widget.persistence_controls.write_button.isEnabled()


def test_widget_discovers_new_feature_matrix_table_without_stealing_existing_selection(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])
    app_state = get_or_create_app_state(viewer)
    widget = ObjectClassificationWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)

    previous_feature_items = _combo_texts(widget.feature_matrix_combo)
    _add_feature_table_for_labels(
        sdata_blobs,
        table_name="new_table",
        labels_name="blobs_labels",
        feature_key="features_new",
    )

    app_state.record_table_mutation(
        _feature_table_event(
            sdata_blobs,
            table_name="new_table",
            feature_key="features_new",
        )
    )

    assert _combo_texts(widget.table_combo) == ["new_table", "table"]
    assert widget.selected_table_name == "table"
    assert _combo_texts(widget.feature_matrix_combo) == previous_feature_items
    assert widget.selected_feature_key == "features_1"
    assert widget.persistence_controls.controller.has_unsynced_table_changes is False


def test_widget_auto_selects_new_feature_matrix_table_when_no_table_was_available(
    qtbot,
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])
    app_state = get_or_create_app_state(viewer)
    widget = ObjectClassificationWidget(viewer)
    qtbot.addWidget(widget)
    widget.segmentation_combo.setCurrentIndex(1)
    mark_dirty_reasons: list[str | None] = []
    monkeypatch.setattr(
        widget._classifier_controller,
        "mark_dirty",
        lambda *, reason=None: mark_dirty_reasons.append(reason),
    )

    assert widget.selected_segmentation_name == "blobs_multiscale_labels"
    assert widget.selected_table_name is None
    assert widget.table_combo.count() == 0

    _add_feature_table_for_labels(
        sdata_blobs,
        table_name="new_table",
        labels_name="blobs_multiscale_labels",
        feature_key="features_new",
    )
    app_state.record_table_mutation(
        _feature_table_event(
            sdata_blobs,
            table_name="new_table",
            feature_key="features_new",
        )
    )

    assert _combo_texts(widget.table_combo) == ["new_table"]
    assert widget.selected_table_name == "new_table"
    assert widget.selected_feature_key == "features_new"
    assert widget.persistence_controls.controller.has_unsynced_table_changes is True
    assert mark_dirty_reasons == []


def test_widget_updates_color_by_mode_when_selection_changes(qtbot, sdata_blobs: SpatialData) -> None:
    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])

    widget = ObjectClassificationWidget(viewer)
    qtbot.addWidget(widget)

    widget.color_by_combo.setCurrentIndex(1)

    assert widget.selected_color_by == "pred_class"


def test_widget_tracks_picked_instance_id_from_labels_layer(qtbot, sdata_blobs: SpatialData) -> None:
    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])

    widget = ObjectClassificationWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)

    layer.selected_label = 5

    assert widget.selected_instance_id == 5
    assert widget.apply_class_button.isEnabled()
    assert "Current instance_id: 5." in widget.selection_status.text()
    assert "Current class: unlabeled." in widget.selection_status.text()


def test_widget_accepts_first_pick_when_instance_id_is_one(qtbot, sdata_blobs: SpatialData) -> None:
    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])

    widget = ObjectClassificationWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)

    layer.selected_label = 1

    assert widget.selected_instance_id == 1
    assert widget.apply_class_button.isEnabled()
    assert "Current instance_id: 1." in widget.selection_status.text()


def test_widget_automatically_enables_pick_mode_for_bound_labels_layer(qtbot, sdata_blobs: SpatialData) -> None:
    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])

    widget = ObjectClassificationWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)

    assert str(layer.mode) == "pick"
    assert viewer.layers.selection.active is layer


def test_widget_picks_multiscale_labels_layers_without_napari_pick_mode(qtbot, sdata_blobs: SpatialData) -> None:
    layer = make_multiscale_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])

    widget = ObjectClassificationWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)

    assert str(layer.mode) == "pan_zoom"
    assert viewer.layers.selection.active is layer

    coords = tuple(float(value) for value in np.argwhere(np.asarray(sdata_blobs.labels["blobs_labels"]) == 5)[0])
    event = SimpleNamespace(position=coords, view_direction=None, dims_displayed=[0, 1])
    layer.mouse_drag_callbacks[-1](layer, event)

    assert widget.selected_instance_id == 5
    assert widget.apply_class_button.isEnabled()

    widget.class_spinbox.setValue(7)
    widget.apply_class_button.click()

    table = sdata_blobs["table"]
    mask = (table.obs["region"] == "blobs_labels") & (table.obs["instance_id"] == 5)

    assert table.obs.loc[mask, USER_CLASS_COLUMN].tolist() == [7]
    assert "Assigned class 7" in widget.annotation_feedback.text()


def test_widget_auto_loads_selected_segmentation_when_it_is_not_yet_loaded(qtbot, sdata_blobs: SpatialData) -> None:
    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])

    widget = ObjectClassificationWidget(viewer)
    qtbot.addWidget(widget)

    widget.segmentation_combo.setCurrentIndex(1)
    layer.selected_label = 9

    assert widget.selected_segmentation_name == "blobs_multiscale_labels"
    assert widget.selected_instance_id is None
    assert len(viewer.layers) == 2
    assert viewer.layers[-1].name == "blobs_multiscale_labels"
    assert viewer.layers.selection.active is viewer.layers[-1]
    assert widget._annotation_controller.labels_layer is viewer.layers[-1]
    assert widget._viewer_styling_controller.labels_layer is viewer.layers[-1]
    assert 'Loaded labels element "blobs_multiscale_labels" in coordinate system "global".' in (
        widget.selection_status.text()
    )
    assert "This labels layer is loaded, but no annotation table is linked to it." in widget.selection_status.text()


def test_widget_clears_selected_segmentation_after_manual_layer_removal(qtbot, sdata_blobs: SpatialData) -> None:
    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])

    widget = ObjectClassificationWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)

    assert widget._annotation_controller.labels_layer is layer
    assert widget._viewer_styling_controller.labels_layer is layer

    viewer.layers.remove(layer)
    viewer.layers.selection.active = None
    viewer.layers.events.removed.emit(layer)

    assert widget.segmentation_combo.currentIndex() == -1
    assert widget.selected_segmentation_name is None
    assert len(viewer.layers) == 0
    assert widget._annotation_controller.labels_layer is None
    assert widget._viewer_styling_controller.labels_layer is None
    assert widget.selected_table_name is None
    assert widget.selected_feature_key is None
    assert "Choose a labels element" in widget.selection_status.text()


def test_widget_ignores_unrelated_labels_layer_removal(qtbot, monkeypatch, sdata_blobs: SpatialData) -> None:
    primary_layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[primary_layer])

    widget = ObjectClassificationWidget(viewer)
    qtbot.addWidget(widget)

    widget.segmentation_combo.setCurrentIndex(1)
    selected_layer = viewer.layers[-1]

    bind_calls: list[str | None] = []

    def _record_bind(*, classifier_dirty_reason: str | None = None) -> None:
        bind_calls.append(classifier_dirty_reason)

    monkeypatch.setattr(widget, "_bind_current_selection", _record_bind)

    viewer.layers.remove(primary_layer)
    viewer.layers.selection.active = selected_layer
    viewer.layers.events.removed.emit(primary_layer)

    assert widget.selected_segmentation_name == "blobs_multiscale_labels"
    assert widget._annotation_controller.labels_layer is selected_layer
    assert widget._viewer_styling_controller.labels_layer is selected_layer
    assert bind_calls == []


def test_widget_handles_tables_without_obsm_entries(qtbot, sdata_blobs: SpatialData) -> None:
    table = sdata_blobs["table"]
    for key in list(table.obsm.keys()):
        del table.obsm[key]

    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])

    widget = ObjectClassificationWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)

    assert widget.table_combo.count() == 1
    assert widget.table_combo.itemText(0) == "table"
    assert widget.feature_matrix_combo.count() == 0
    assert not widget.feature_matrix_combo.isEnabled()
    assert widget.selected_table_name == "table"
    assert widget.selected_feature_key is None
    assert not widget.warning_status.isHidden()
    assert 'does not contain any feature matrices in ".obsm"' in widget.warning_status.text()


def test_widget_applies_user_class_to_picked_instance(qtbot, sdata_blobs: SpatialData) -> None:
    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])

    widget = ObjectClassificationWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)
    emitted_events: list[object] = []
    widget.app_state.table_state_changed.connect(emitted_events.append)

    layer.selected_label = 5
    widget.class_spinbox.setValue(3)
    widget.apply_class_button.click()

    table = sdata_blobs["table"]
    mask = (table.obs["region"] == "blobs_labels") & (table.obs["instance_id"] == 5)

    assert USER_CLASS_COLUMN in table.obs
    assert isinstance(table.obs[USER_CLASS_COLUMN].dtype, pd.CategoricalDtype)
    assert list(table.obs[USER_CLASS_COLUMN].cat.categories) == [3]
    assert table.obs.loc[mask, USER_CLASS_COLUMN].tolist() == [3]
    assert pd.isna(table.obs.loc[table.obs["instance_id"] == 6, USER_CLASS_COLUMN].iloc[0])
    assert table.uns[USER_CLASS_COLORS_KEY] == default_categorical_colors(1)
    assert "adata" not in layer.metadata
    assert "Current class: 3." in widget.selection_status.text()
    assert "Assigned class 3" in widget.annotation_feedback.text()
    assert len(emitted_events) == 1
    event = emitted_events[0]
    assert isinstance(event, TableStateChangedEvent)
    assert event.paths == frozenset(
        {
            TableComponentPath("obs", (USER_CLASS_COLUMN,)),
            TableComponentPath("uns", (USER_CLASS_COLORS_KEY,)),
        }
    )
    assert event.regions == ("blobs_labels",)
    assert event.change_kind == "created"


def test_widget_emits_updated_table_event_when_user_class_is_already_color_source(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    table = sdata_blobs["table"]
    table.obs[USER_CLASS_COLUMN] = pd.Categorical([pd.NA] * table.n_obs, categories=[])
    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])

    widget = ObjectClassificationWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)
    emitted_events: list[object] = []
    widget.app_state.table_state_changed.connect(emitted_events.append)

    layer.selected_label = 5
    widget.class_spinbox.setValue(3)
    widget.apply_class_button.click()

    mask = (table.obs["region"] == "blobs_labels") & (table.obs["instance_id"] == 5)

    assert table.obs.loc[mask, USER_CLASS_COLUMN].tolist() == [3]
    assert len(emitted_events) == 1
    event = emitted_events[0]
    assert isinstance(event, TableStateChangedEvent)
    assert event.change_kind == "updated"


def test_widget_apply_shortcut_applies_user_class_to_picked_instance(qtbot, sdata_blobs: SpatialData) -> None:
    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])

    widget = ObjectClassificationWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)

    layer.selected_label = 5
    widget.class_spinbox.setValue(4)

    assert "Shortcut: A." in widget.apply_class_button.toolTip()
    assert widget._annotation_shortcuts[0].key().toString() == "A"

    widget._annotation_shortcuts[0].activated.emit()

    table = sdata_blobs["table"]
    mask = (table.obs["region"] == "blobs_labels") & (table.obs["instance_id"] == 5)

    assert table.obs.loc[mask, USER_CLASS_COLUMN].tolist() == [4]
    assert "Assigned class 4" in widget.annotation_feedback.text()


def test_widget_uses_table_instance_key_name_in_status_and_annotation_feedback(qtbot, sdata_blobs: SpatialData) -> None:
    rename_table_instance_key(sdata_blobs, instance_key="cell_id")

    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])

    widget = ObjectClassificationWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)

    layer.selected_label = 5
    widget.class_spinbox.setValue(3)
    widget.apply_class_button.click()

    assert "Current cell_id: 5." in widget.selection_status.text()
    assert "Assigned class 3 to cell_id 5." in widget.annotation_feedback.text()


def test_widget_can_clear_user_class_for_picked_instance(qtbot, sdata_blobs: SpatialData) -> None:
    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])

    widget = ObjectClassificationWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)

    layer.selected_label = 5
    widget.class_spinbox.setValue(2)
    widget.apply_class_button.click()
    widget.clear_class_button.click()

    table = sdata_blobs["table"]
    mask = (table.obs["region"] == "blobs_labels") & (table.obs["instance_id"] == 5)

    assert isinstance(table.obs[USER_CLASS_COLUMN].dtype, pd.CategoricalDtype)
    assert list(table.obs[USER_CLASS_COLUMN].cat.categories) == [2]
    assert table.obs.loc[mask, USER_CLASS_COLUMN].isna().all()
    assert table.uns[USER_CLASS_COLORS_KEY] == default_categorical_colors(1)
    assert "Current class: unlabeled." in widget.selection_status.text()
    assert "Cleared the user class" in widget.annotation_feedback.text()


def test_widget_clear_shortcut_clears_user_class_for_picked_instance(qtbot, sdata_blobs: SpatialData) -> None:
    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])

    widget = ObjectClassificationWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)

    layer.selected_label = 5
    widget.class_spinbox.setValue(2)
    widget.apply_class_button.click()

    assert "Shortcut: R." in widget.clear_class_button.toolTip()
    assert widget._annotation_shortcuts[1].key().toString() == "R"

    widget._annotation_shortcuts[1].activated.emit()

    table = sdata_blobs["table"]
    mask = (table.obs["region"] == "blobs_labels") & (table.obs["instance_id"] == 5)

    assert table.obs.loc[mask, USER_CLASS_COLUMN].isna().all()
    assert "Cleared the user class" in widget.annotation_feedback.text()


def test_widget_warns_when_selected_label_is_missing_from_annotation_table(
    qtbot, monkeypatch, sdata_blobs: SpatialData
) -> None:
    rename_table_instance_key(sdata_blobs, instance_key="cell_id")
    table = sdata_blobs["table"]
    keep_mask = ~((table.obs["region"] == "blobs_labels") & (table.obs["cell_id"] == 5))
    table._inplace_subset_obs(keep_mask.to_numpy())

    warnings: list[str] = []

    class DummyLogger:
        def warning(self, message: str) -> None:
            warnings.append(message)

    monkeypatch.setattr(annotation_module, "logger", DummyLogger())

    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])
    widget = ObjectClassificationWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)

    assert isinstance(layer.colormap, CompactLabelColormap)
    np.testing.assert_allclose(layer.colormap.map(5), np.zeros(4, dtype=np.float32))
    np.testing.assert_allclose(layer.colormap.map(6), np.asarray(to_rgba(DEFAULT_NEUTRAL_COLOR), dtype=np.float32))

    layer.selected_label = 5

    assert not widget.apply_class_button.isEnabled()
    assert "Selected cell_id 5 is not present in annotation table" in widget.selection_status.text()
    assert "cannot receive a user class" in widget.selection_status.text()
    assert "It is not shown in the viewer." in widget.selection_status.text()

    widget.class_spinbox.setValue(3)
    widget._apply_current_class()

    assert "Selected cell_id 5 is not present in annotation table" in widget.annotation_feedback.text()
    assert "It is not shown in the viewer." in widget.annotation_feedback.text()
    assert STATUS_CARD_PALETTE["warning"]["text"] in widget.annotation_feedback.styleSheet()
    assert USER_CLASS_COLUMN not in table.obs
    assert warnings == [widget._annotation_controller.missing_table_row_message]
    assert warnings[0] in widget.annotation_feedback.text()


def test_widget_cleared_annotation_stays_neutral_while_labels_missing_from_table_stay_hidden(
    qtbot, sdata_blobs: SpatialData
) -> None:
    """Removing a class must not hide the object.

    The table covers every blobs instance except `5`. After adding and then
    removing a class on `6`, its row stays with an empty `user_class`, so it
    returns to the neutral unlabeled color, not the transparent color used for
    instances without a table row. Instance `5` stays transparent throughout.

    This checks the resulting colors only; that Remove uses the single-object
    update is pinned in
    `test_widget_user_class_annotation_uses_sparse_refresh_for_compact_user_class`.
    """
    table = sdata_blobs["table"]
    keep_mask = ~((table.obs["region"] == "blobs_labels") & (table.obs["instance_id"] == 5))
    table._inplace_subset_obs(keep_mask.to_numpy())
    neutral_rgba = np.asarray(to_rgba(DEFAULT_NEUTRAL_COLOR), dtype=np.float32)

    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])
    widget = ObjectClassificationWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)

    layer.selected_label = 6
    widget.class_spinbox.setValue(4)
    widget.apply_class_button.click()

    assert layer.colormap.map(6)[3] > 0
    assert not np.allclose(layer.colormap.map(6), neutral_rgba)

    widget.clear_class_button.click()

    assert pd.isna(layer.features.set_index("index").loc[6, USER_CLASS_COLUMN])
    np.testing.assert_allclose(layer.colormap.map(6), neutral_rgba)
    np.testing.assert_allclose(layer.colormap.map(5), np.zeros(4, dtype=np.float32))


def test_widget_recolors_layer_from_user_class_annotations(qtbot, sdata_blobs: SpatialData) -> None:
    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])

    widget = ObjectClassificationWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)

    layer.selected_label = 5
    widget.class_spinbox.setValue(4)
    widget.apply_class_button.click()

    assert isinstance(layer.colormap, CompactLabelColormap)
    assert np.allclose(layer.colormap.map(0), np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32))
    assert len(layer.colormap.color_dict) <= 3
    assert layer.colormap.map(5)[3] > 0
    assert layer.colormap.map(6)[3] > 0
    assert not np.allclose(layer.colormap.map(5), layer.colormap.map(6))
    assert "instance_id" in layer.features.columns
    assert USER_CLASS_COLUMN in layer.features.columns
    assert layer.features.set_index("index").loc[5, "instance_id"] == 5
    assert layer.features.set_index("index").loc[5, USER_CLASS_COLUMN] == 4


def test_widget_user_class_annotation_uses_sparse_refresh_for_compact_user_class(
    qtbot,
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])
    widget = ObjectClassificationWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)
    full_refresh_calls = []
    layer_refresh_calls = []
    original_refresh = widget._viewer_styling_controller.refresh
    original_colormap = layer.colormap

    def record_full_refresh() -> None:
        full_refresh_calls.append("refresh")
        original_refresh()

    def record_layer_refresh(**kwargs) -> None:
        layer_refresh_calls.append(kwargs)

    monkeypatch.setattr(widget._viewer_styling_controller, "refresh", record_full_refresh)
    monkeypatch.setattr(layer, "refresh", record_layer_refresh)

    # Add (A): assigning a class updates one colormap entry, no full refresh.
    layer.selected_label = 5
    widget.class_spinbox.setValue(4)
    widget.apply_class_button.click()

    assert full_refresh_calls == []
    assert layer_refresh_calls == [{"extent": False}]
    assert isinstance(layer.colormap, CompactLabelColormap)
    assert layer.colormap is original_colormap
    assert len(layer.colormap.color_dict) <= 3
    assert layer.colormap.map(5)[3] > 0
    assert layer.features.set_index("index").loc[5, USER_CLASS_COLUMN] == 4

    # Remove (R): clearing the class also stays on the single-object update.
    layer_refresh_calls.clear()
    widget.clear_class_button.click()

    assert full_refresh_calls == []
    assert layer_refresh_calls == [{"extent": False}]
    assert layer.colormap is original_colormap
    np.testing.assert_allclose(layer.colormap.map(5), np.asarray(to_rgba(DEFAULT_NEUTRAL_COLOR), dtype=np.float32))
    assert pd.isna(layer.features.set_index("index").loc[5, USER_CLASS_COLUMN])


def test_widget_user_class_annotation_falls_back_to_full_refresh_when_row_scoped_refresh_fails(
    qtbot,
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])
    widget = ObjectClassificationWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)
    row_scoped_calls = []
    full_refresh_calls = []

    def record_row_scoped_refresh(change) -> bool:
        row_scoped_calls.append(change)
        return False

    def record_full_refresh() -> None:
        full_refresh_calls.append("refresh")

    monkeypatch.setattr(
        widget._viewer_styling_controller,
        "refresh_user_class_colormap_and_feature",
        record_row_scoped_refresh,
    )
    monkeypatch.setattr(widget._viewer_styling_controller, "refresh", record_full_refresh)

    layer.selected_label = 5
    widget.class_spinbox.setValue(4)
    widget.apply_class_button.click()

    assert len(row_scoped_calls) == 1
    assert row_scoped_calls[0].instance_id == 5
    assert row_scoped_calls[0].class_id == 4
    assert full_refresh_calls == ["refresh"]


def test_widget_user_class_annotation_updates_feature_only_in_prediction_color_modes(
    qtbot,
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    def run_annotation_in_color_mode(color_by: str) -> None:
        table = sdata_blobs["table"]
        if USER_CLASS_COLUMN in table.obs:
            table.obs.pop(USER_CLASS_COLUMN)
        table.uns.pop(USER_CLASS_COLORS_KEY, None)
        layer = make_blobs_labels_layer(sdata_blobs)
        viewer = DummyViewer(layers=[layer])
        widget = ObjectClassificationWidget(viewer)
        qtbot.addWidget(widget)
        select_segmentation(widget)
        widget.color_by_combo.setCurrentIndex(widget.color_by_combo.findData(color_by))
        color_refresh_calls = []
        feature_refresh_calls = []
        full_refresh_calls = []

        def record_color_refresh(change) -> bool:
            color_refresh_calls.append(change)
            return True

        def record_feature_refresh(change) -> bool:
            feature_refresh_calls.append(change)
            return True

        def record_full_refresh() -> None:
            full_refresh_calls.append("refresh")

        monkeypatch.setattr(
            widget._viewer_styling_controller,
            "refresh_user_class_colormap_and_feature",
            record_color_refresh,
        )
        monkeypatch.setattr(
            widget._viewer_styling_controller, "refresh_user_class_feature_only", record_feature_refresh
        )
        monkeypatch.setattr(widget._viewer_styling_controller, "refresh", record_full_refresh)

        layer.selected_label = 5
        widget.class_spinbox.setValue(4)
        widget.apply_class_button.click()

        assert color_refresh_calls == []
        assert [(call.instance_id, call.class_id) for call in feature_refresh_calls] == [(5, 4)]
        assert full_refresh_calls == []

    run_annotation_in_color_mode("pred_class")
    run_annotation_in_color_mode("pred_confidence")


def test_widget_user_class_annotation_falls_back_to_full_refresh_when_prediction_feature_refresh_fails(
    qtbot,
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])
    widget = ObjectClassificationWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)
    widget.color_by_combo.setCurrentIndex(widget.color_by_combo.findData("pred_class"))
    feature_refresh_calls = []
    full_refresh_calls = []

    def record_feature_refresh(change) -> bool:
        feature_refresh_calls.append(change)
        return False

    def record_full_refresh() -> None:
        full_refresh_calls.append("refresh")

    monkeypatch.setattr(widget._viewer_styling_controller, "refresh_user_class_feature_only", record_feature_refresh)
    monkeypatch.setattr(widget._viewer_styling_controller, "refresh", record_full_refresh)

    layer.selected_label = 5
    widget.class_spinbox.setValue(4)
    widget.apply_class_button.click()

    assert len(feature_refresh_calls) == 1
    assert feature_refresh_calls[0].instance_id == 5
    assert feature_refresh_calls[0].class_id == 4
    assert full_refresh_calls == ["refresh"]


def test_widget_auto_train_prediction_color_mode_keeps_immediate_refresh_feature_only(
    qtbot,
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    _set_feature_metadata(sdata_blobs)
    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])
    widget = ObjectClassificationWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)
    widget.color_by_combo.setCurrentIndex(widget.color_by_combo.findData("pred_class"))
    schedule_calls: list[str] = []
    feature_refresh_calls = []
    full_refresh_calls: list[str] = []

    def record_schedule_retrain(*args, **kwargs) -> bool:
        del args, kwargs
        schedule_calls.append("schedule")
        return False

    def record_feature_refresh(change) -> bool:
        feature_refresh_calls.append(change)
        return True

    monkeypatch.setattr(widget._classifier_controller, "schedule_retrain", record_schedule_retrain)
    monkeypatch.setattr(widget._viewer_styling_controller, "refresh_user_class_feature_only", record_feature_refresh)
    monkeypatch.setattr(widget._viewer_styling_controller, "refresh", lambda: full_refresh_calls.append("refresh"))

    widget.auto_train_checkbox.setChecked(True)
    layer.selected_label = 5
    widget.class_spinbox.setValue(4)
    widget.apply_class_button.click()

    assert schedule_calls == ["schedule"]
    assert [(call.instance_id, call.class_id) for call in feature_refresh_calls] == [(5, 4)]
    assert full_refresh_calls == []


def test_widget_annotation_defers_classifier_controls_until_selection_status(qtbot, monkeypatch) -> None:
    widget = ObjectClassificationWidget(DummyViewer())
    qtbot.addWidget(widget)
    calls: list[str] = []

    def record_selection_status() -> None:
        calls.append("selection_status")
        widget._update_classifier_controls()

    def record_schedule_retrain() -> bool:
        calls.append("schedule_retrain")
        widget._classifier_controller._set_status(
            "Classifier: model is stale. Classifier training is scheduled.",
            kind="info",
        )
        return True

    monkeypatch.setattr(widget, "_refresh_after_user_class_annotation", lambda change: calls.append("visual_refresh"))
    monkeypatch.setattr(widget, "_update_classifier_feedback", lambda: calls.append("feedback"))
    monkeypatch.setattr(widget, "_update_classifier_controls", lambda: calls.append("controls"))
    monkeypatch.setattr(widget, "_update_selection_status", record_selection_status)
    monkeypatch.setattr(widget._classifier_controller, "schedule_retrain", record_schedule_retrain)

    widget._auto_train_enabled = True
    widget._on_annotation_changed(
        annotation_module.UserClassAnnotationChange(
            instance_id=5,
            class_id=4,
            state_change=UserClassStateChange(
                user_class_changed=True,
                palette_changed=False,
            ),
            user_class_was_available_as_color_source=True,
        )
    )

    assert calls == [
        "visual_refresh",
        "feedback",
        "schedule_retrain",
        "feedback",
        "selection_status",
        "controls",
    ]

    calls.clear()
    widget._on_classifier_state_changed()

    assert calls == ["feedback", "controls"]


def test_widget_auto_train_toggle_controls_annotation_retraining(
    qtbot, monkeypatch, backed_sdata_blobs: SpatialData
) -> None:
    _set_feature_metadata(backed_sdata_blobs)
    layer = make_blobs_labels_layer(backed_sdata_blobs)
    viewer = DummyViewer(layers=[layer])
    widget = ObjectClassificationWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)
    schedule_calls: list[str] = []
    mark_dirty_reasons: list[str | None] = []
    refresh_calls: list[str] = []
    row_scoped_refresh_calls = []
    call_order: list[str] = []

    def record_schedule_retrain(*args, **kwargs) -> bool:
        del args, kwargs
        schedule_calls.append("schedule")
        call_order.append("schedule")
        return False

    def record_mark_dirty(*, reason: str | None = None) -> None:
        mark_dirty_reasons.append(reason)
        call_order.append("mark_dirty")

    def record_row_scoped_refresh(change) -> bool:
        row_scoped_refresh_calls.append(change)
        call_order.append("row_scoped_refresh")
        return True

    monkeypatch.setattr(widget._classifier_controller, "schedule_retrain", record_schedule_retrain)
    monkeypatch.setattr(widget._classifier_controller, "mark_dirty", record_mark_dirty)
    monkeypatch.setattr(
        widget._viewer_styling_controller,
        "refresh_user_class_colormap_and_feature",
        record_row_scoped_refresh,
    )
    monkeypatch.setattr(widget, "_refresh_layer_styling", lambda: refresh_calls.append("refresh"))

    assert widget.auto_train_checkbox.isChecked() is False
    assert widget._auto_train_enabled is False

    widget.auto_train_checkbox.setChecked(True)
    widget.auto_train_checkbox.setChecked(False)

    assert schedule_calls == []
    assert mark_dirty_reasons == []
    assert widget.persistence_controls.controller.has_unsynced_table_changes is False

    layer.selected_label = 5
    widget.class_spinbox.setValue(3)
    widget.apply_class_button.click()

    assert schedule_calls == []
    assert mark_dirty_reasons == ["the annotations changed"]
    assert [(call.instance_id, call.class_id) for call in row_scoped_refresh_calls] == [(5, 3)]
    assert call_order == ["row_scoped_refresh", "mark_dirty"]
    assert refresh_calls == []
    assert widget.persistence_controls.controller.has_unsynced_table_changes is True

    widget.auto_train_checkbox.setChecked(True)
    layer.selected_label = 6
    widget.class_spinbox.setValue(4)
    widget.apply_class_button.click()

    assert schedule_calls == ["schedule"]
    assert mark_dirty_reasons == ["the annotations changed", "the annotations changed"]
    assert [(call.instance_id, call.class_id) for call in row_scoped_refresh_calls] == [(5, 3), (6, 4)]
    assert call_order == [
        "row_scoped_refresh",
        "mark_dirty",
        "row_scoped_refresh",
        "mark_dirty",
        "schedule",
    ]
    assert refresh_calls == []


def test_widget_disables_sync_for_clean_backed_spatialdata(qtbot, backed_sdata_blobs: SpatialData) -> None:
    layer = make_blobs_labels_layer(backed_sdata_blobs)
    viewer = DummyViewer(layers=[layer])

    widget = ObjectClassificationWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)
    expected_table_path = Path(backed_sdata_blobs.path) / "tables" / "table"
    sync_tooltip = unescape(widget.persistence_controls.write_button.toolTip()).replace("&#8203;", "").replace("\u200b", "")
    reload_tooltip = unescape(widget.persistence_controls.reload_button.toolTip()).replace("&#8203;", "").replace("\u200b", "")

    assert not widget.persistence_controls.write_button.isEnabled()
    assert widget.persistence_controls.reload_button.isEnabled()
    assert 'The selected "table" table has no unsynced local in-memory changes to write.' in sync_tooltip
    assert f'Discard the current in-memory "table" table state and reload the table from "{expected_table_path}".' in (
        reload_tooltip
    )


def test_widget_marks_persistence_dirty_on_annotation_change_and_clears_it_on_sync(
    qtbot, monkeypatch, backed_sdata_blobs: SpatialData
) -> None:
    layer = make_blobs_labels_layer(backed_sdata_blobs)
    viewer = DummyViewer(layers=[layer])

    widget = ObjectClassificationWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)
    monkeypatch.setattr(widget._classifier_controller, "schedule_retrain", lambda *args, **kwargs: False)

    layer.selected_label = 5
    widget.class_spinbox.setValue(3)
    widget.apply_class_button.click()
    sync_tooltip = unescape(widget.persistence_controls.write_button.toolTip()).replace("&#8203;", "").replace("\u200b", "")
    reload_tooltip = unescape(widget.persistence_controls.reload_button.toolTip()).replace("&#8203;", "").replace("\u200b", "")

    assert widget.persistence_controls.controller.has_unsynced_table_changes is True
    assert widget.persistence_controls.write_button.isEnabled()
    assert "Unsynced local in-memory table changes are present." in sync_tooltip
    assert "Unsynced local in-memory table changes would be discarded." in reload_tooltip

    widget.persistence_controls.write_button.click()
    sync_tooltip = unescape(widget.persistence_controls.write_button.toolTip()).replace("&#8203;", "").replace("\u200b", "")
    reload_tooltip = unescape(widget.persistence_controls.reload_button.toolTip()).replace("&#8203;", "").replace("\u200b", "")

    assert widget.persistence_controls.controller.has_unsynced_table_changes is False
    assert not widget.persistence_controls.write_button.isEnabled()
    assert "Unsynced local in-memory table changes are present." not in sync_tooltip
    assert "Unsynced local in-memory table changes would be discarded." not in reload_tooltip


def test_widget_syncs_user_class_to_backed_zarr(qtbot, backed_sdata_blobs: SpatialData) -> None:
    layer = make_blobs_labels_layer(backed_sdata_blobs)
    viewer = DummyViewer(layers=[layer])

    widget = ObjectClassificationWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)
    expected_table_path = Path(backed_sdata_blobs.path) / "tables" / "table"

    layer.selected_label = 5
    widget.class_spinbox.setValue(3)
    widget.apply_class_button.click()
    assert widget.persistence_controls.write_button.isEnabled()
    widget.persistence_controls.write_button.click()

    reread = read_zarr(backed_sdata_blobs.path)
    mask = (reread["table"].obs["region"] == "blobs_labels") & (reread["table"].obs["instance_id"] == 5)

    assert not widget.persistence_controls.write_button.isEnabled()
    assert widget.persistence_controls.reload_button.isEnabled()
    _assert_persistence_success_feedback(
        widget,
        f'Wrote "table" annotations, predictions, and classifier metadata to "{expected_table_path}".',
    )
    assert isinstance(reread["table"].obs[USER_CLASS_COLUMN].dtype, pd.CategoricalDtype)
    assert list(reread["table"].obs[USER_CLASS_COLUMN].cat.categories) == [3]
    assert reread["table"].obs.loc[mask, USER_CLASS_COLUMN].tolist() == [3]
    assert list(reread["table"].uns[USER_CLASS_COLORS_KEY]) == default_categorical_colors(1)


def test_widget_marks_persistence_dirty_after_classifier_writes_results(qtbot, backed_sdata_blobs: SpatialData) -> None:
    table = backed_sdata_blobs["table"]
    instance_ids = table.obs["instance_id"].to_numpy(dtype=np.int64)
    table.obsm["features_1"] = np.column_stack(
        [
            (instance_ids > 13).astype(np.float64),
            instance_ids.astype(np.float64) / instance_ids.max(),
        ]
    )
    _set_feature_metadata(backed_sdata_blobs)
    table.obs[USER_CLASS_COLUMN] = pd.Categorical(
        [
            1 if int(instance_id) in {1, 2} else 2 if int(instance_id) in {24, 25} else pd.NA
            for instance_id in instance_ids
        ],
        categories=[1, 2],
    )

    layer = make_blobs_labels_layer(backed_sdata_blobs)
    viewer = DummyViewer(layers=[layer])
    widget = ObjectClassificationWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)

    assert widget.persistence_controls.controller.has_unsynced_table_changes is False

    widget.retrain_button.click()
    qtbot.waitUntil(
        lambda: (
            widget.persistence_controls.controller.has_unsynced_table_changes and table.obs[PRED_CLASS_COLUMN].notna().any()
        ),
        timeout=5000,
    )
    sync_tooltip = unescape(widget.persistence_controls.write_button.toolTip()).replace("&#8203;", "").replace("\u200b", "")
    reload_tooltip = unescape(widget.persistence_controls.reload_button.toolTip()).replace("&#8203;", "").replace("\u200b", "")

    assert widget.persistence_controls.controller.has_unsynced_table_changes is True
    assert widget.persistence_controls.write_button.isEnabled()
    assert "Unsynced local in-memory table changes are present." in sync_tooltip
    assert "Unsynced local in-memory table changes would be discarded." in reload_tooltip


def test_widget_cancels_dirty_reload_when_user_chooses_cancel(
    qtbot, monkeypatch, backed_sdata_blobs: SpatialData
) -> None:
    layer = make_blobs_labels_layer(backed_sdata_blobs)
    viewer = DummyViewer(layers=[layer])
    widget = ObjectClassificationWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)
    monkeypatch.setattr(widget._classifier_controller, "schedule_retrain", lambda *args, **kwargs: False)

    table = backed_sdata_blobs["table"]
    disk_obs = table.obs.copy()
    disk_obs[USER_CLASS_COLUMN] = pd.Categorical([pd.NA] * table.n_obs, categories=[])
    _write_disk_table_state(backed_sdata_blobs, obs=disk_obs, obsm=dict(table.obsm), uns=dict(table.uns))

    layer.selected_label = 5
    widget.class_spinbox.setValue(3)
    widget.apply_class_button.click()
    monkeypatch.setattr(
        widget.persistence_controls,
        "_prompt_dirty_reload_decision",
        lambda: persistence_controls_module._DirtyReloadDecision.CANCEL,
    )

    widget.persistence_controls.reload_button.click()

    mask = (table.obs["region"] == "blobs_labels") & (table.obs["instance_id"] == 5)
    reread = read_zarr(backed_sdata_blobs.path)
    disk_mask = (reread["table"].obs["region"] == "blobs_labels") & (reread["table"].obs["instance_id"] == 5)

    assert widget.persistence_controls.controller.has_unsynced_table_changes is True
    assert table.obs.loc[mask, USER_CLASS_COLUMN].tolist() == [3]
    assert reread["table"].obs.loc[disk_mask, USER_CLASS_COLUMN].isna().all()


def test_widget_dirty_reload_can_write_then_reload(qtbot, monkeypatch, backed_sdata_blobs: SpatialData) -> None:
    layer = make_blobs_labels_layer(backed_sdata_blobs)
    viewer = DummyViewer(layers=[layer])
    widget = ObjectClassificationWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)
    monkeypatch.setattr(widget._classifier_controller, "schedule_retrain", lambda *args, **kwargs: False)

    expected_table_path = Path(backed_sdata_blobs.path) / "tables" / "table"
    table = backed_sdata_blobs["table"]

    layer.selected_label = 5
    widget.class_spinbox.setValue(3)
    widget.apply_class_button.click()
    monkeypatch.setattr(
        widget.persistence_controls,
        "_prompt_dirty_reload_decision",
        lambda: persistence_controls_module._DirtyReloadDecision.WRITE,
    )

    widget.persistence_controls.reload_button.click()

    reread = read_zarr(backed_sdata_blobs.path)
    mask = (table.obs["region"] == "blobs_labels") & (table.obs["instance_id"] == 5)
    disk_mask = (reread["table"].obs["region"] == "blobs_labels") & (reread["table"].obs["instance_id"] == 5)

    assert widget.persistence_controls.controller.has_unsynced_table_changes is False
    assert table.obs.loc[mask, USER_CLASS_COLUMN].tolist() == [3]
    assert reread["table"].obs.loc[disk_mask, USER_CLASS_COLUMN].tolist() == [3]
    _assert_persistence_success_feedback(
        widget,
        f'Wrote local table state and reloaded "table" table state from "{expected_table_path}".',
    )


def test_widget_dirty_reload_can_discard_local_edits(qtbot, monkeypatch, backed_sdata_blobs: SpatialData) -> None:
    layer = make_blobs_labels_layer(backed_sdata_blobs)
    viewer = DummyViewer(layers=[layer])
    widget = ObjectClassificationWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)
    monkeypatch.setattr(widget._classifier_controller, "schedule_retrain", lambda *args, **kwargs: False)

    expected_table_path = Path(backed_sdata_blobs.path) / "tables" / "table"
    table = backed_sdata_blobs["table"]
    disk_obs = table.obs.copy()
    disk_obs[USER_CLASS_COLUMN] = pd.Categorical([pd.NA] * table.n_obs, categories=[])
    _write_disk_table_state(backed_sdata_blobs, obs=disk_obs, obsm=dict(table.obsm), uns=dict(table.uns))

    layer.selected_label = 5
    widget.class_spinbox.setValue(3)
    widget.apply_class_button.click()
    monkeypatch.setattr(
        widget.persistence_controls,
        "_prompt_dirty_reload_decision",
        lambda: persistence_controls_module._DirtyReloadDecision.RELOAD_DISCARD,
    )

    widget.persistence_controls.reload_button.click()

    mask = (table.obs["region"] == "blobs_labels") & (table.obs["instance_id"] == 5)
    reread = read_zarr(backed_sdata_blobs.path)
    disk_mask = (reread["table"].obs["region"] == "blobs_labels") & (reread["table"].obs["instance_id"] == 5)

    assert widget.persistence_controls.controller.has_unsynced_table_changes is False
    assert table.obs.loc[mask, USER_CLASS_COLUMN].isna().all()
    assert reread["table"].obs.loc[disk_mask, USER_CLASS_COLUMN].isna().all()
    _assert_persistence_success_feedback(widget, f'Reloaded "table" table state from "{expected_table_path}".')


def test_widget_reloads_table_state_from_backed_zarr(qtbot, backed_sdata_blobs: SpatialData) -> None:
    layer = make_blobs_labels_layer(backed_sdata_blobs)
    viewer = DummyViewer(layers=[layer])

    widget = ObjectClassificationWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)
    expected_table_path = Path(backed_sdata_blobs.path) / "tables" / "table"
    table = backed_sdata_blobs["table"]

    obs = table.obs.copy()
    obs[USER_CLASS_COLUMN] = pd.Categorical(
        [pd.NA] * (table.n_obs - 1) + [7],
        categories=[7],
    )
    obsm = dict(table.obsm)
    obsm["disk_features"] = np.arange(table.n_obs, dtype=np.float64).reshape(table.n_obs, 1)
    uns = dict(table.uns)
    _write_disk_table_state(backed_sdata_blobs, obs=obs, obsm=obsm, uns=uns)

    layer.selected_label = int(table.obs["instance_id"].iloc[-1])
    widget.persistence_controls.reload_button.click()

    mask = (table.obs["region"] == "blobs_labels") & (
        table.obs["instance_id"] == int(table.obs["instance_id"].iloc[-1])
    )

    _assert_persistence_success_feedback(widget, f'Reloaded "table" table state from "{expected_table_path}".')
    assert isinstance(table.obs[USER_CLASS_COLUMN].dtype, pd.CategoricalDtype)
    assert list(table.obs[USER_CLASS_COLUMN].cat.categories) == [7]
    assert table.obs.loc[mask, USER_CLASS_COLUMN].tolist() == [7]
    assert "disk_features" in table.obsm
    feature_matrix_items = [
        widget.feature_matrix_combo.itemText(index) for index in range(widget.feature_matrix_combo.count())
    ]
    assert feature_matrix_items == ["disk_features", "features_1", "features_2"]
    assert widget.selected_feature_key == "features_1"
    assert "Current class: 7." in widget.selection_status.text()


def test_spatial_query_reload_prepares_object_classification_for_shared_table(
    qtbot,
    monkeypatch,
    backed_sdata_blobs: SpatialData,
) -> None:
    """A Spatial Query reload must freeze Object Classification through shared app state before reloading their table."""
    layer = make_blobs_labels_layer(backed_sdata_blobs)
    viewer = DummyViewer(layers=[layer])
    object_classification = ObjectClassificationWidget(viewer)
    spatial_query = SpatialQuery(viewer)
    qtbot.addWidget(object_classification)
    qtbot.addWidget(spatial_query)
    select_segmentation(object_classification)
    spatial_query.apply_annotation_context(
        AnnotationContext(
            sdata=backed_sdata_blobs,
            coordinate_system="global",
            shapes_target=ShapesAnnotationTarget.edit_existing("blobs_circles"),
            has_unsaved_shapes_changes=False,
        )
    )
    labels_index = spatial_query.labels_combo.findData("blobs_labels")
    assert labels_index >= 0
    spatial_query.labels_combo.setCurrentIndex(labels_index)
    assert object_classification.selected_table_name == spatial_query.selected_table_name == "table"

    freeze_calls: list[str] = []
    monkeypatch.setattr(
        object_classification._classifier_controller,
        "freeze_for_reload",
        lambda: freeze_calls.append("freeze"),
    )
    reload_events: list[TableStateChangedEvent] = []
    object_classification.app_state.table_state_changed.connect(
        lambda event: reload_events.append(event) if event.change_kind == "reloaded" else None
    )

    spatial_query.persistence_controls.reload_button.click()

    assert freeze_calls == ["freeze"]
    assert len(reload_events) == 1
    assert reload_events[0].sdata is backed_sdata_blobs
    assert reload_events[0].table_name == "table"


def test_widget_reload_falls_back_when_selected_feature_key_disappears(qtbot, backed_sdata_blobs: SpatialData) -> None:
    layer = make_blobs_labels_layer(backed_sdata_blobs)
    viewer = DummyViewer(layers=[layer])

    widget = ObjectClassificationWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)
    expected_table_path = Path(backed_sdata_blobs.path) / "tables" / "table"
    table = backed_sdata_blobs["table"]

    widget.feature_matrix_combo.setCurrentIndex(1)

    assert widget.selected_feature_key == "features_2"

    obs = table.obs.copy()
    obsm = {"features_1": table.obsm["features_1"]}
    uns = dict(table.uns)
    _write_disk_table_state(backed_sdata_blobs, obs=obs, obsm=obsm, uns=uns)

    widget.persistence_controls.reload_button.click()

    feature_matrix_items = [
        widget.feature_matrix_combo.itemText(index) for index in range(widget.feature_matrix_combo.count())
    ]

    _assert_persistence_success_feedback(widget, f'Reloaded "table" table state from "{expected_table_path}".')
    assert feature_matrix_items == ["features_1"]
    assert widget.selected_feature_key == "features_1"
    assert "features_2" not in table.obsm


def test_widget_reload_freezes_classifier_worker_and_ignores_late_results(
    qtbot, monkeypatch, backed_sdata_blobs: SpatialData
) -> None:
    table = backed_sdata_blobs["table"]
    instance_ids = table.obs["instance_id"].to_numpy(dtype=np.int64)
    table.obsm["features_1"] = np.column_stack(
        [
            (instance_ids > 13).astype(np.float64),
            instance_ids.astype(np.float64) / instance_ids.max(),
        ]
    )
    _set_feature_metadata(backed_sdata_blobs)
    table.obs[USER_CLASS_COLUMN] = pd.Categorical(
        [
            1 if int(instance_id) in {1, 2} else 2 if int(instance_id) in {24, 25} else pd.NA
            for instance_id in instance_ids
        ],
        categories=[1, 2],
    )

    layer = make_blobs_labels_layer(backed_sdata_blobs)
    viewer = DummyViewer(layers=[layer])
    widget = ObjectClassificationWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)
    workers: list[_DeferredWorker] = []

    def fake_create_training_worker(job):
        result = classifier_module.ClassifierJobResult(
            job_id=job.job_id,
            feature_key=job.feature_key,
            labels_name=job.labels_name,
            table_name=job.table_name,
            pred_classes=np.full(job.prediction_scope.table_row_positions.shape, 1, dtype=np.int64),
            pred_confidences=np.full(job.prediction_scope.table_row_positions.shape, 0.91, dtype=np.float64),
            trained_at="2026-04-13T09:00:00+00:00",
            model_params=dict(classifier_module.RANDOM_FOREST_PARAMS),
            summary=job.summary,
        )
        worker = _DeferredWorker(result)
        workers.append(worker)
        return worker

    monkeypatch.setattr(widget._classifier_controller, "_create_training_worker", fake_create_training_worker)

    widget.retrain_button.click()

    assert len(workers) == 1
    assert workers[0].started is True
    assert widget._classifier_controller.is_training is True

    expected_table_path = Path(backed_sdata_blobs.path) / "tables" / "table"
    obs = table.obs.copy()
    obs[PRED_CLASS_COLUMN] = pd.Categorical(np.full(table.n_obs, 7, dtype=np.int64), categories=[7])
    obs[PRED_CONFIDENCE_COLUMN] = pd.Series(np.full(table.n_obs, 0.77), index=obs.index, dtype="float64")
    obsm = dict(table.obsm)
    uns = dict(table.uns)
    uns[CLASSIFIER_CONFIG_KEY] = {
        "model_type": "RandomForestClassifier",
        "feature_key": "features_1",
        "table_name": "table",
        "roi_mode": "none",
        "trained": True,
        "eligible": True,
        "reason": "Ready to train.",
        "training_timestamp": "2026-04-13T09:00:00+00:00",
        "n_labeled_objects": 4,
        "n_features": 2,
        "class_labels_seen": [1, 2],
        "rf_params": dict(classifier_module.RANDOM_FOREST_PARAMS),
        "training_scope": "all",
        "training_regions": ["blobs_labels"],
        "n_training_rows": int(table.n_obs),
        "prediction_scope": "selected_segmentation_only",
        "prediction_regions": ["blobs_labels"],
        "n_predicted_rows": int(table.n_obs),
    }
    _write_disk_table_state(backed_sdata_blobs, obs=obs, obsm=obsm, uns=uns)

    widget.persistence_controls.reload_button.click()

    assert workers[0].quit_called is True
    assert widget._classifier_controller.is_training is False
    assert widget._classifier_controller.is_dirty is False
    _assert_persistence_success_feedback(widget, f'Reloaded "table" table state from "{expected_table_path}".')
    assert table.obs[PRED_CLASS_COLUMN].eq(7).all()
    assert table.obs[PRED_CONFIDENCE_COLUMN].eq(0.77).all()
    assert "Loaded predictions for" in widget.classifier_feedback.text()
    assert len(workers) == 1

    workers[0].emit_returned()

    assert table.obs[PRED_CLASS_COLUMN].eq(7).all()
    assert table.obs[PRED_CONFIDENCE_COLUMN].eq(0.77).all()
    assert "Loaded predictions for" in widget.classifier_feedback.text()


def test_widget_retrain_button_recovers_after_worker_finishes(qtbot, monkeypatch, sdata_blobs: SpatialData) -> None:
    table = sdata_blobs["table"]
    instance_ids = table.obs["instance_id"].to_numpy(dtype=np.int64)
    table.obsm["features_1"] = np.column_stack(
        [
            (instance_ids > 13).astype(np.float64),
            instance_ids.astype(np.float64) / instance_ids.max(),
        ]
    )
    _set_feature_metadata(sdata_blobs)
    table.obs[USER_CLASS_COLUMN] = pd.Categorical(
        [
            1 if int(instance_id) in {1, 2} else 2 if int(instance_id) in {24, 25} else pd.NA
            for instance_id in instance_ids
        ],
        categories=[1, 2],
    )

    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])
    widget = ObjectClassificationWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)
    workers: list[_DeferredWorker] = []
    refresh_calls: list[str] = []

    def fake_create_training_worker(job):
        result = classifier_module.ClassifierJobResult(
            job_id=job.job_id,
            feature_key=job.feature_key,
            labels_name=job.labels_name,
            table_name=job.table_name,
            pred_classes=np.full(job.prediction_scope.table_row_positions.shape, 1, dtype=np.int64),
            pred_confidences=np.full(job.prediction_scope.table_row_positions.shape, 0.91, dtype=np.float64),
            trained_at="2026-04-13T09:00:00+00:00",
            model_params=dict(classifier_module.RANDOM_FOREST_PARAMS),
            summary=job.summary,
        )
        worker = _DeferredWorker(result)
        workers.append(worker)
        return worker

    monkeypatch.setattr(widget._classifier_controller, "_create_training_worker", fake_create_training_worker)
    monkeypatch.setattr(widget, "_refresh_layer_styling", lambda: refresh_calls.append("refresh"))

    widget.retrain_button.click()

    assert len(workers) == 1
    assert widget._classifier_controller.is_training is True
    assert widget.retrain_button.isEnabled() is False
    assert "currently running" in widget.retrain_button.toolTip()
    assert refresh_calls == []

    workers[0].emit_returned()

    qtbot.waitUntil(lambda: widget._classifier_controller.is_training is False, timeout=1000)
    qtbot.waitUntil(lambda: widget.retrain_button.isEnabled(), timeout=1000)

    assert refresh_calls == ["refresh"]
    assert "currently running" not in widget.retrain_button.toolTip()
    assert "write predictions for the selected prediction scope" in widget.retrain_button.toolTip()
    assert "model is up to date" in widget.classifier_feedback.text()


def test_widget_classifier_status_changes_do_not_refresh_layer_styling(
    qtbot, monkeypatch, sdata_blobs: SpatialData
) -> None:
    _set_feature_metadata(sdata_blobs)
    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])
    widget = ObjectClassificationWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)
    refresh_calls: list[str] = []

    monkeypatch.setattr(widget, "_refresh_layer_styling", lambda: refresh_calls.append("refresh"))

    widget._classifier_controller.mark_dirty(reason="the annotations changed")

    assert refresh_calls == []
    assert "annotations changed" in widget.classifier_feedback.text()

    widget._classifier_controller.schedule_retrain()
    widget._classifier_controller._debounce_timer.stop()

    assert refresh_calls == []
    assert "scheduled" in widget.classifier_feedback.text()


def test_widget_destroyed_shuts_down_classifier_controller(qtbot, monkeypatch, sdata_blobs: SpatialData) -> None:
    _set_feature_metadata(sdata_blobs)
    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])
    widget = ObjectClassificationWidget(viewer)
    select_segmentation(widget)
    controller = widget._classifier_controller
    created_job_ids: list[int] = []

    def fake_create_training_worker(job):
        created_job_ids.append(job.job_id)
        raise AssertionError("widget destruction should cancel pending classifier debounce")

    monkeypatch.setattr(controller, "_create_training_worker", fake_create_training_worker)
    controller._debounce_interval_ms = 50
    controller._debounce_timer.setInterval(50)

    assert controller._debounce_timer.parent() is widget
    assert controller.schedule_retrain() is True
    assert controller._debounce_timer.isActive() is True

    widget.deleteLater()
    qtbot.waitUntil(lambda: controller._is_shutdown, timeout=1000)
    qtbot.wait(150)

    assert created_job_ids == []
    assert controller.is_training is False
    assert controller.schedule_retrain() is False


def test_widget_retrains_classifier_after_annotation_changes(qtbot, sdata_blobs: SpatialData) -> None:
    table = sdata_blobs["table"]
    instance_ids = table.obs["instance_id"].to_numpy(dtype=np.int64)
    table.obsm["features_1"] = np.column_stack(
        [
            (instance_ids > 13).astype(np.float64),
            instance_ids.astype(np.float64) / instance_ids.max(),
        ]
    )
    _set_feature_metadata(sdata_blobs)

    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])
    widget = ObjectClassificationWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)
    emitted_events: list[object] = []
    widget.app_state.table_state_changed.connect(emitted_events.append)
    widget.auto_train_checkbox.setChecked(True)

    layer.selected_label = 1
    widget.class_spinbox.setValue(1)
    widget.apply_class_button.click()

    layer.selected_label = 24
    widget.class_spinbox.setValue(2)
    widget.apply_class_button.click()

    qtbot.waitUntil(
        lambda: PRED_CLASS_COLUMN in table.obs and table.obs[PRED_CLASS_COLUMN].notna().any(),
        timeout=5000,
    )
    qtbot.waitUntil(
        lambda: any(
            isinstance(event, TableStateChangedEvent)
            and TableComponentPath("obs", (PRED_CLASS_COLUMN,)) in event.paths
            and TableComponentPath("obs", (PRED_CONFIDENCE_COLUMN,)) in event.paths
            for event in emitted_events
        ),
        timeout=5000,
    )

    pred_class = table.obs.set_index("instance_id")[PRED_CLASS_COLUMN]
    assert isinstance(table.obs[PRED_CLASS_COLUMN].dtype, pd.CategoricalDtype)
    assert list(table.obs[PRED_CLASS_COLUMN].cat.categories) == [1, 2]
    assert table.uns[PRED_CLASS_COLORS_KEY] == default_class_colors([1, 2])
    assert pred_class.loc[1] == 1
    assert pred_class.loc[24] == 2
    assert "adata" not in layer.metadata
    assert "model is up to date" in widget.classifier_feedback.text()
    assert table.uns[CLASSIFIER_CONFIG_KEY]["trained"] is True
    assert any(
        isinstance(event, TableStateChangedEvent)
        and event.source == "object_classification_inference"
        and event.regions == ("blobs_labels",)
        and TableComponentPath("uns", (CLASSIFIER_CONFIG_KEY,)) in event.paths
        for event in emitted_events
    )


def test_widget_colors_predictions_using_pred_class_palette_in_pred_class_mode(qtbot, sdata_blobs: SpatialData) -> None:
    table = sdata_blobs["table"]
    instance_ids = table.obs["instance_id"].to_numpy(dtype=np.int64)
    table.obsm["features_1"] = np.column_stack(
        [
            (instance_ids > 13).astype(np.float64),
            instance_ids.astype(np.float64) / instance_ids.max(),
        ]
    )
    _set_feature_metadata(sdata_blobs)

    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])
    widget = ObjectClassificationWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)
    widget.auto_train_checkbox.setChecked(True)

    layer.selected_label = 1
    widget.class_spinbox.setValue(1)
    widget.apply_class_button.click()

    layer.selected_label = 24
    widget.class_spinbox.setValue(2)
    widget.apply_class_button.click()

    qtbot.waitUntil(
        lambda: PRED_CLASS_COLUMN in table.obs and table.obs[PRED_CLASS_COLUMN].notna().any(),
        timeout=5000,
    )

    assert isinstance(layer.colormap, CompactLabelColormap)
    assert not np.allclose(layer.colormap.map(1), layer.colormap.map(5))

    widget.color_by_combo.setCurrentIndex(widget.color_by_combo.findData("pred_class"))

    assert isinstance(layer.colormap, CompactLabelColormap)
    assert np.allclose(layer.colormap.map(1), layer.colormap.map(5))
    assert np.allclose(layer.colormap.map(24), layer.colormap.map(26))
    assert table.uns[PRED_CLASS_COLORS_KEY] == default_class_colors([1, 2])
    assert np.allclose(layer.colormap.map(1), np.asarray(to_rgba(default_class_colors([1])[0]), dtype=np.float32))
    assert np.allclose(layer.colormap.map(24), np.asarray(to_rgba(default_class_colors([2])[0]), dtype=np.float32))
    assert PRED_CLASS_COLUMN in layer.features.columns


def test_widget_colors_confidence_continuously_in_pred_confidence_mode(qtbot, sdata_blobs: SpatialData) -> None:
    table = sdata_blobs["table"]
    table.obs[PRED_CLASS_COLUMN] = pd.Categorical(
        np.ones(table.n_obs, dtype=np.int64),
        categories=[1],
    )
    table.obs[PRED_CONFIDENCE_COLUMN] = pd.Series(
        np.linspace(0.0, 1.0, table.n_obs),
        index=table.obs.index,
        dtype="float64",
    )

    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])
    widget = ObjectClassificationWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)

    widget.color_by_combo.setCurrentIndex(widget.color_by_combo.findData("pred_confidence"))

    assert isinstance(layer.colormap, DirectLabelColormap)
    assert not np.allclose(layer.colormap.color_dict[1], layer.colormap.color_dict[24])
    assert PRED_CONFIDENCE_COLUMN in layer.features.columns


def test_widget_exposes_label_metadata_in_napari_status_bar(qtbot, sdata_blobs: SpatialData) -> None:
    table = sdata_blobs["table"]
    mask = (table.obs["region"] == "blobs_labels") & (table.obs["instance_id"] == 5)
    table.obs[USER_CLASS_COLUMN] = pd.Categorical([pd.NA] * table.n_obs, categories=[4])
    table.obs[PRED_CLASS_COLUMN] = pd.Categorical([pd.NA] * table.n_obs, categories=[2])
    table.obs.loc[mask, USER_CLASS_COLUMN] = 4
    table.obs.loc[mask, PRED_CLASS_COLUMN] = 2
    table.obs.loc[mask, PRED_CONFIDENCE_COLUMN] = 0.95

    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])
    widget = ObjectClassificationWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)

    coords = tuple(float(value) for value in np.argwhere(np.asarray(sdata_blobs.labels["blobs_labels"]) == 5)[0])
    status = layer.get_status(position=coords, view_direction=np.array([1.0, 0.0]), dims_displayed=[0, 1])

    assert "instance_id: 5" in status["value"]
    assert "user_class: 4" in status["value"]
    assert "pred_class: 2" in status["value"]
    assert "pred_confidence: 0.95" in status["value"]


def test_widget_retrain_button_triggers_manual_retraining(qtbot, monkeypatch, sdata_blobs: SpatialData) -> None:
    table = sdata_blobs["table"]
    instance_ids = table.obs["instance_id"].to_numpy(dtype=np.int64)
    table.obs[USER_CLASS_COLUMN] = pd.Categorical(
        [
            1 if int(instance_id) in {1, 2} else 2 if int(instance_id) in {24, 25} else pd.NA
            for instance_id in instance_ids
        ],
        categories=[1, 2],
    )
    _set_feature_metadata(sdata_blobs)
    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])
    widget = ObjectClassificationWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)

    retrain_calls: list[bool] = []

    def fake_retrain_now() -> bool:
        retrain_calls.append(True)
        return True

    monkeypatch.setattr(widget._classifier_controller, "retrain_now", fake_retrain_now)

    assert widget.auto_train_checkbox.isChecked() is False
    assert widget.retrain_button.isEnabled()

    widget.retrain_button.click()

    assert retrain_calls == [True]


def test_widget_exports_classifier_with_mocked_save_dialog(
    qtbot,
    monkeypatch,
    tmp_path,
    sdata_blobs: SpatialData,
) -> None:
    table = sdata_blobs["table"]
    instance_ids = table.obs["instance_id"].to_numpy(dtype=np.int64)
    table.obs[USER_CLASS_COLUMN] = pd.Categorical(
        [
            1 if int(instance_id) in {1, 2} else 2 if int(instance_id) in {24, 25} else pd.NA
            for instance_id in instance_ids
        ],
        categories=[1, 2],
    )
    _set_feature_metadata(sdata_blobs)

    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])
    widget = ObjectClassificationWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)

    assert widget.export_classifier_button.isEnabled() is False

    widget.retrain_button.click()
    qtbot.waitUntil(lambda: widget.export_classifier_button.isEnabled(), timeout=5000)

    selected_path = tmp_path / "widget-export"
    monkeypatch.setattr(
        widget_module.QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: (str(selected_path), ""),
    )

    widget.export_classifier_button.click()

    export_path = tmp_path / f"widget-export{DEFAULT_CLASSIFIER_EXPORT_SUFFIX}"
    loaded = read_classifier_export_bundle(export_path)

    assert export_path.exists()
    assert loaded.n_features == int(table.obsm["features_1"].shape[1])
    assert "Classifier Exported" in widget.classifier_feedback.text()
    assert str(export_path) in widget.classifier_feedback.text()
