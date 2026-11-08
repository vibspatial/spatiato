from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from loguru import logger
from qtpy.QtCore import QSignalBlocker, Qt
from qtpy.QtGui import QKeySequence, QShortcut
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from spatiato._app_state import (
    CoordinateSystemChangedEvent,
    SpatiatoAppState,
    TableReloadRequest,
    TableStateChangedEvent,
    get_or_create_app_state,
)
from spatiato.core.feature_matrix_metadata import (
    FeatureMatrixMetadataState,
    inspect_feature_matrix_metadata,
    register_feature_matrix_metadata,
)
from spatiato.core.object_classification.annotation import (
    USER_CLASS_COLORS_KEY,
    USER_CLASS_COLUMN,
)
from spatiato.core.object_classification.classifier import (
    ObjectClassificationStateError,
    validate_object_classification_table_state,
)
from spatiato.core.object_classification.classifier_export import DEFAULT_CLASSIFIER_EXPORT_SUFFIX
from spatiato.core.persistence import TableComponentPath
from spatiato.core.spatialdata import (
    SpatialDataLabelsOption,
    SpatialDataTableMetadata,
    get_annotating_table_names,
    get_coordinate_system_names_from_sdata,
    get_spatialdata_labels_options_for_coordinate_system_from_sdata,
    get_table,
    get_table_metadata,
    get_table_obsm_keys,
    validate_table_binding,
)
from spatiato.viewer.labels_styling import apply_neutral_labels_style
from spatiato.widgets.object_classification.annotation_controller import (
    AnnotationController,
    UserClassAnnotationChange,
)
from spatiato.widgets.object_classification.controller import (
    DEFAULT_PREDICTION_SCOPE,
    DEFAULT_TRAINING_SCOPE,
    ClassifierController,
    ClassifierScopeMode,
    ClassifierTableStateChange,
)
from spatiato.widgets.object_classification.feature_matrix_registration import (
    _build_feature_matrix_registration_button_state,
)
from spatiato.widgets.object_classification.status_card import (
    _LabelsLayerPreparationResult,
    _ObjectClassificationStatusCardSpec,
    build_object_classification_classifier_feedback_card_spec,
    build_object_classification_classifier_preparation_card_spec,
    build_object_classification_selection_status_card_spec,
    build_object_classification_warning_status_card_spec,
)
from spatiato.widgets.object_classification.viewer_styling import (
    COLOR_BY_OPTIONS,
    COLOR_BY_PRED_CLASS,
    COLOR_BY_PRED_CONFIDENCE,
    COLOR_BY_USER_CLASS,
    ClassStateError,
    ViewerStylingController,
)
from spatiato.widgets.persistence.controls import TablePersistenceControls
from spatiato.widgets.shared_styles import (
    ACTION_BUTTON_STYLESHEET as _ACTION_BUTTON_STYLESHEET,
)
from spatiato.widgets.shared_styles import (
    CHECKBOX_STYLESHEET as _CHECKBOX_STYLESHEET,
)
from spatiato.widgets.shared_styles import (
    SMALL_ACTION_BUTTON_STYLESHEET,
    WIDGET_BORDER_COLOR,
    WIDGET_PANEL_COLOR,
    CompactComboBox,
    StatusCardKind,
    apply_scroll_content_surface,
    apply_widget_surface,
    build_input_control_stylesheet,
    create_form_label,
    create_header_logo,
    format_tooltip,
    set_status_card,
)
from spatiato.widgets.shared_styles import (
    WIDGET_MIN_WIDTH as _WIDGET_MIN_WIDTH,
)

if TYPE_CHECKING:
    import napari
    from spatialdata import SpatialData


_APPLY_CLASS_SHORTCUT = "A"
_REMOVE_CLASS_SHORTCUT = "R"
_INPUT_CONTROL_STYLESHEET = build_input_control_stylesheet("QComboBox, QSpinBox")
_TABLE_WIDE_TRAINING_SCOPE_LABEL = "All eligible annotated rows in table"
_TABLE_WIDE_PREDICTION_SCOPE_LABEL = "All eligible rows in table"
_SELECTED_SEGMENTATION_TRAINING_SCOPE_LABEL = "Selected labels element only"
_SPATIAL_QUERY_ANNOTATION_SOURCE = "spatial_query_annotation"
_CLASS_EDITOR_STYLESHEET = (
    f"QWidget#class_editor {{background-color: {WIDGET_PANEL_COLOR}; "
    f"border: 1px solid {WIDGET_BORDER_COLOR}; border-radius: 10px;}}"
)


class ObjectClassificationWidget(QWidget):
    """
    Widget for object classification.

    The widget is migrating toward the shared Spatiato app-state architecture.
    It already receives the loaded `SpatialData` object through
    `self._app_state.sdata` / `sdata_changed`, and now resolves live labels
    layers through the shared `ViewerAdapter` plus layer bindings.

    In the current transition state, the widget exposes:

    - coordinate systems from the shared loaded `SpatialData`
    - segmentation masks filtered by the selected coordinate system
    - annotating tables for the selected segmentation
    - feature matrix keys from `table.obsm`
    - the currently picked segmentation instance id from the active `Labels` layer
    """

    def __init__(self, napari_viewer: napari.Viewer | None = None) -> None:
        super().__init__()
        self.setObjectName("object_classification_widget")
        apply_widget_surface(self)
        self.setMinimumWidth(_WIDGET_MIN_WIDTH)
        self._viewer = napari_viewer
        # The napari viewer identifies which shared Spatiato session this widget
        # belongs to. We use it to attach to the per-viewer SpatiatoAppState.
        self._app_state = get_or_create_app_state(napari_viewer)
        self._annotation_controller = AnnotationController(
            self._app_state.viewer_adapter,
            on_selected_instance_changed=self._on_selected_instance_changed,
            on_annotation_changed=self._on_annotation_changed,
        )
        self._classifier_controller = ClassifierController(
            on_state_changed=self._on_classifier_state_changed,
            on_table_state_changed=self._on_classifier_table_state_changed,
            on_prediction_state_changed=self._on_classifier_prediction_state_changed,
            timer_parent=self,
        )
        # If the widget is destroyed while classifier debounce/training is still
        # pending, shut down async classifier callbacks before they can update
        # deleted Qt controls.
        # Also, capture the controller directly
        # (via classifier_controller = self._classifier_controller) so the destruction
        # callback does not need to access attributes through `self` while Qt is
        # tearing down the widget object.
        classifier_controller = self._classifier_controller
        self.destroyed.connect(classifier_controller.shutdown)
        self._viewer_styling_controller = ViewerStylingController(
            self._app_state.viewer_adapter,
        )
        self.persistence_controls = TablePersistenceControls(
            self._app_state,
            write_content_description="annotations, predictions, and classifier metadata",
            reload_source="object_classification",
            parent=self,
        )
        self._coordinate_systems: list[str] = []
        self._selected_coordinate_system: str | None = None
        self._label_options: list[SpatialDataLabelsOption] = []
        self._selected_label_option: SpatialDataLabelsOption | None = None
        self._is_preparing_labels_layer = False
        self._is_handling_coordinate_system_change = False
        self._labels_layer_preparation_result = _LabelsLayerPreparationResult(kind="none")
        self._table_names: list[str] = []
        self._selected_table_name: str | None = None
        self._table_binding_error: str | None = None
        self._feature_matrix_keys: list[str] = []
        self._selected_feature_key: str | None = None
        self._selected_training_scope: ClassifierScopeMode = DEFAULT_TRAINING_SCOPE
        self._selected_prediction_scope: ClassifierScopeMode = DEFAULT_PREDICTION_SCOPE
        self._auto_train_enabled = False
        self._is_deferring_classifier_control_updates = False
        self._layer_styling_error: str | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.scroll_area = QScrollArea()
        self.scroll_area.setObjectName("object_classification_scroll_area")
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_area.setStyleSheet("QScrollArea { border: 0px; background: transparent; }")

        self.scroll_content = QWidget()
        self.scroll_content.setObjectName("object_classification_scroll_content")
        apply_scroll_content_surface(self.scroll_content)

        content_layout = QVBoxLayout(self.scroll_content)
        content_layout.setContentsMargins(12, 12, 12, 12)
        content_layout.setSpacing(10)
        self.scroll_area.setWidget(self.scroll_content)
        layout.addWidget(self.scroll_area)

        title = create_header_logo("object_classification_header_logo")

        selector_layout = QFormLayout()
        selector_layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        selector_layout.setHorizontalSpacing(12)
        selector_layout.setVerticalSpacing(10)
        self.coordinate_system_combo = CompactComboBox()
        self.coordinate_system_combo.setObjectName("object_classification_coordinate_system_combo")
        self.coordinate_system_combo.currentIndexChanged.connect(self._on_coordinate_system_changed)
        self.coordinate_system_combo.setStyleSheet(_INPUT_CONTROL_STYLESHEET)

        self.segmentation_combo = CompactComboBox()
        self.segmentation_combo.setObjectName("segmentation_mask_combo")
        self.segmentation_combo.setPlaceholderText("Choose a labels element")
        self.segmentation_combo.currentIndexChanged.connect(self._on_segmentation_changed)
        self.segmentation_combo.setStyleSheet(_INPUT_CONTROL_STYLESHEET)

        self.table_combo = CompactComboBox()
        self.table_combo.setObjectName("annotation_table_combo")
        self.table_combo.currentIndexChanged.connect(self._on_table_changed)
        self.table_combo.setStyleSheet(_INPUT_CONTROL_STYLESHEET)

        self.feature_matrix_combo = CompactComboBox()
        self.feature_matrix_combo.setObjectName("feature_matrix_combo")
        self.feature_matrix_combo.currentIndexChanged.connect(self._on_feature_matrix_changed)
        self.feature_matrix_combo.setStyleSheet(_INPUT_CONTROL_STYLESHEET)

        self.register_feature_matrix_button = QPushButton("Register Feature Matrix")
        self.register_feature_matrix_button.setObjectName("register_feature_matrix_button")
        self.register_feature_matrix_button.clicked.connect(self._register_selected_feature_matrix_metadata)
        self.register_feature_matrix_button.setEnabled(False)
        self.register_feature_matrix_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.register_feature_matrix_button.setStyleSheet(SMALL_ACTION_BUTTON_STYLESHEET)
        self.feature_matrix_registration_row = QWidget()
        self.feature_matrix_registration_row.setObjectName("feature_matrix_registration_row")
        feature_matrix_registration_layout = QHBoxLayout(self.feature_matrix_registration_row)
        feature_matrix_registration_layout.setContentsMargins(0, 0, 0, 0)
        feature_matrix_registration_layout.setSpacing(0)
        feature_matrix_registration_layout.addWidget(self.register_feature_matrix_button, 1)

        self.training_scope_combo = CompactComboBox()
        self.training_scope_combo.setObjectName("training_scope_combo")
        self.training_scope_combo.addItem(_TABLE_WIDE_TRAINING_SCOPE_LABEL, "all")
        self.training_scope_combo.addItem(
            _SELECTED_SEGMENTATION_TRAINING_SCOPE_LABEL,
            "selected_segmentation_only",
        )
        self.training_scope_combo.setCurrentIndex(self.training_scope_combo.findData(DEFAULT_TRAINING_SCOPE))
        self.training_scope_combo.currentIndexChanged.connect(self._on_training_scope_changed)
        self.training_scope_combo.setStyleSheet(_INPUT_CONTROL_STYLESHEET)

        self.prediction_scope_combo = CompactComboBox()
        self.prediction_scope_combo.setObjectName("prediction_scope_combo")
        self.prediction_scope_combo.addItem(
            _SELECTED_SEGMENTATION_TRAINING_SCOPE_LABEL,
            "selected_segmentation_only",
        )
        self.prediction_scope_combo.addItem(_TABLE_WIDE_PREDICTION_SCOPE_LABEL, "all")
        self.prediction_scope_combo.setCurrentIndex(self.prediction_scope_combo.findData(DEFAULT_PREDICTION_SCOPE))
        self.prediction_scope_combo.currentIndexChanged.connect(self._on_prediction_scope_changed)
        self.prediction_scope_combo.setStyleSheet(_INPUT_CONTROL_STYLESHEET)

        self.color_by_combo = QComboBox()
        self.color_by_combo.setObjectName("color_by_combo")
        for color_by in COLOR_BY_OPTIONS:
            self.color_by_combo.addItem(color_by, color_by)
        self.color_by_combo.setCurrentIndex(self.color_by_combo.findData(COLOR_BY_USER_CLASS))
        self.color_by_combo.currentIndexChanged.connect(self._on_color_by_changed)
        self.color_by_combo.setStyleSheet(_INPUT_CONTROL_STYLESHEET)

        self.class_spinbox = QSpinBox()
        self.class_spinbox.setObjectName("user_class_spinbox")
        self.class_spinbox.setRange(1, 999)
        self.class_spinbox.setValue(1)
        self.class_spinbox.setStyleSheet(_INPUT_CONTROL_STYLESHEET)
        self.class_editor = QWidget()
        self.class_editor.setObjectName("class_editor")
        self.class_editor.setStyleSheet(_CLASS_EDITOR_STYLESHEET)
        class_editor_layout = QVBoxLayout(self.class_editor)
        class_editor_layout.setContentsMargins(8, 8, 8, 8)
        class_editor_layout.setSpacing(8)
        self.class_action_row = QWidget()
        self.class_action_row.setObjectName("class_action_row")
        class_action_layout = QHBoxLayout(self.class_action_row)
        class_action_layout.setContentsMargins(0, 0, 0, 0)
        class_action_layout.setSpacing(8)
        self.retrain_action_row = QWidget()
        self.retrain_action_row.setObjectName("retrain_action_row")
        retrain_action_layout = QHBoxLayout(self.retrain_action_row)
        retrain_action_layout.setContentsMargins(0, 0, 0, 0)
        retrain_action_layout.setSpacing(8)
        self.auto_train_checkbox = QCheckBox("Auto-train classifier")
        self.auto_train_checkbox.setObjectName("auto_train_checkbox")
        self.auto_train_checkbox.setChecked(False)
        self.auto_train_checkbox.setCursor(Qt.CursorShape.PointingHandCursor)
        self.auto_train_checkbox.setStyleSheet(_CHECKBOX_STYLESHEET)
        self.auto_train_checkbox.toggled.connect(self._on_auto_train_toggled)
        self._update_auto_train_tooltip()

        self.retrain_button = QPushButton("Train Classifier")
        self.retrain_button.setObjectName("retrain_button")
        self.retrain_button.clicked.connect(self._retrain_classifier)
        self.retrain_button.setEnabled(False)
        self.retrain_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.retrain_button.setMinimumHeight(28)
        self.retrain_button.setStyleSheet(_ACTION_BUTTON_STYLESHEET)

        self.export_classifier_button = QPushButton("Export Classifier")
        self.export_classifier_button.setObjectName("export_classifier_button")
        self.export_classifier_button.clicked.connect(self._export_classifier)
        self.export_classifier_button.setEnabled(False)
        self.export_classifier_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.export_classifier_button.setMinimumHeight(28)
        self.export_classifier_button.setStyleSheet(_ACTION_BUTTON_STYLESHEET)

        self.warning_status = QLabel()
        self.warning_status.setObjectName("warning_status")
        self.warning_status.setWordWrap(True)
        self.warning_status.hide()

        self.selection_status = QLabel()
        self.selection_status.setObjectName("selection_status")
        self.selection_status.setWordWrap(True)

        self.apply_class_button = QPushButton("Add (A)")
        self.apply_class_button.setObjectName("apply_class_button")
        self.apply_class_button.clicked.connect(self._apply_current_class)
        self.apply_class_button.setEnabled(False)
        self.apply_class_button.setAccessibleName("Add")
        self.apply_class_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.apply_class_button.setMinimumHeight(28)
        self.apply_class_button.setStyleSheet(_ACTION_BUTTON_STYLESHEET)

        self.clear_class_button = QPushButton("Remove (R)")
        self.clear_class_button.setObjectName("clear_class_button")
        self.clear_class_button.clicked.connect(self._clear_current_class)
        self.clear_class_button.setEnabled(False)
        self.clear_class_button.setAccessibleName("Remove")
        self.clear_class_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.clear_class_button.setMinimumHeight(28)
        self.clear_class_button.setStyleSheet(_ACTION_BUTTON_STYLESHEET)
        class_editor_layout.addWidget(self.class_spinbox)
        class_action_layout.addWidget(self.apply_class_button, 1)
        class_action_layout.addWidget(self.clear_class_button, 1)
        class_editor_layout.addWidget(self.class_action_row)
        retrain_action_layout.addWidget(self.retrain_button, 1)
        retrain_action_layout.addWidget(self.export_classifier_button, 1)
        self._annotation_shortcuts = self._create_annotation_shortcuts()

        self.annotation_feedback = QLabel()
        self.annotation_feedback.setObjectName("annotation_feedback")
        self.annotation_feedback.setWordWrap(True)
        self.annotation_feedback.hide()

        self.classifier_feedback = QLabel()
        self.classifier_feedback.setObjectName("classifier_feedback")
        self.classifier_feedback.setWordWrap(True)
        self.classifier_feedback.hide()

        self.classifier_preparation_status = QLabel()
        self.classifier_preparation_status.setObjectName("classifier_preparation_status")
        self.classifier_preparation_status.setWordWrap(True)
        self.classifier_preparation_status.hide()

        selector_layout.addRow(self._create_form_label("Coordinate system"), self.coordinate_system_combo)
        selector_layout.addRow(self._create_form_label("Labels element"), self.segmentation_combo)
        selector_layout.addRow(self._create_form_label("Table"), self.table_combo)
        selector_layout.addRow(self._create_form_label("Feature matrix"), self.feature_matrix_combo)
        selector_layout.addRow(self._create_form_label("Feature metadata"), self.feature_matrix_registration_row)
        selector_layout.addRow(self._create_form_label("Training scope"), self.training_scope_combo)
        selector_layout.addRow(self._create_form_label("Prediction scope"), self.prediction_scope_combo)
        selector_layout.addRow(self._create_form_label("Color by"), self.color_by_combo)
        selector_layout.addRow(self._create_form_label("User class"), self.class_editor)

        content_layout.addWidget(title)
        content_layout.addLayout(selector_layout)
        content_layout.addWidget(self.auto_train_checkbox)
        content_layout.addWidget(self.retrain_action_row)
        content_layout.addWidget(self.classifier_preparation_status)
        content_layout.addWidget(self.persistence_controls)
        content_layout.addWidget(self.selection_status)
        content_layout.addWidget(self.annotation_feedback)
        content_layout.addWidget(self.classifier_feedback)
        content_layout.addWidget(self.warning_status)
        content_layout.addStretch(1)

        self._app_state.sdata_changed.connect(self._on_sdata_changed)
        self._app_state.coordinate_system_changed.connect(self._on_app_state_coordinate_system_changed)
        self._app_state.viewer_adapter.primary_labels_layers_changed.connect(self._on_primary_labels_layers_changed)
        self._app_state.table_state_changed.connect(self._on_table_state_changed)
        self.refresh_from_sdata(self._app_state.sdata)
        # A table reload may be initiated by any widget that exposes the shared
        # persistence controls, not only by Object Classification. Register the
        # widget with app state so every accepted reload gives it a pre-reload
        # opportunity to freeze classifier work before the live table is
        # replaced. Handling only this widget's Reload button would miss
        # reloads initiated elsewhere.
        self._app_state.register_table_reload_participant(self)
        app_state = self._app_state
        participant = self
        self.destroyed.connect(
            lambda *_args, app_state=app_state, participant=participant: app_state.unregister_table_reload_participant(
                participant
            )
        )

    @property
    def app_state(self) -> SpatiatoAppState:
        """Return the shared Spatiato app state for this widget."""
        return self._app_state

    def _create_annotation_shortcuts(self) -> list[QShortcut]:
        apply_shortcut = QShortcut(QKeySequence(_APPLY_CLASS_SHORTCUT), self)
        apply_shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
        apply_shortcut.activated.connect(self._trigger_apply_class_shortcut)

        remove_shortcut = QShortcut(QKeySequence(_REMOVE_CLASS_SHORTCUT), self)
        remove_shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
        remove_shortcut.activated.connect(self._trigger_clear_class_shortcut)

        return [apply_shortcut, remove_shortcut]

    def _create_form_label(self, text: str) -> QLabel:
        return create_form_label(text)

    def _set_tooltip(self, widget: QWidget, message: str) -> None:
        widget.setToolTip(format_tooltip(message))

    @property
    def selected_segmentation_name(self) -> str | None:
        """Return the currently selected labels element name."""
        return None if self._selected_label_option is None else self._selected_label_option.labels_name

    @property
    def selected_spatialdata(self) -> SpatialData | None:
        """Return the loaded SpatialData object backing the current widget state."""
        return self._app_state.sdata

    @property
    def selected_coordinate_system(self) -> str | None:
        """Return the currently selected coordinate system."""
        return self._selected_coordinate_system

    @property
    def selected_table_name(self) -> str | None:
        """Return the currently selected annotation table name."""
        return self._selected_table_name

    @property
    def selected_feature_key(self) -> str | None:
        """Return the currently selected feature matrix key from `adata.obsm`."""
        return self._selected_feature_key

    @property
    def selected_training_scope(self) -> ClassifierScopeMode:
        """Return the current classifier training-scope selection."""
        return self._selected_training_scope

    @property
    def selected_prediction_scope(self) -> ClassifierScopeMode:
        """Return the current classifier prediction-scope selection."""
        return self._selected_prediction_scope

    @property
    def selected_instance_id(self) -> int | None:
        """Return the currently picked segmentation instance id."""
        return self._annotation_controller.selected_instance_id

    @property
    def selected_color_by(self) -> str:
        """Return the current labels-layer coloring mode."""
        return self._viewer_styling_controller.color_by

    @property
    def selected_table_metadata(self) -> SpatialDataTableMetadata | None:
        """Return the linkage metadata for the current table selection."""
        if self.selected_spatialdata is None or self.selected_table_name is None:
            return None

        return get_table_metadata(self.selected_spatialdata, self.selected_table_name)

    def refresh_from_sdata(self, sdata: SpatialData | None) -> None:
        """Refresh the widget from the shared Spatiato SpatialData state."""
        if sdata is None:
            self._clear_selection_inputs()
            self._bind_current_selection()
            return

        self._refresh_coordinate_systems()
        self._refresh_label_options()
        self._refresh_table_names()
        self._prepare_selected_labels_layer()
        self._bind_current_selection(classifier_dirty_reason="the labels element selection changed")

    def _on_sdata_changed(self, sdata: SpatialData | None) -> None:
        self.refresh_from_sdata(sdata)

    def _on_app_state_coordinate_system_changed(self, event: CoordinateSystemChangedEvent) -> None:
        del event
        self._is_handling_coordinate_system_change = True
        try:
            self._sync_coordinate_system_combo_selection(
                self._app_state.coordinate_system
            )  # we prefer to read .coordinate_system from authorative app state instead of from event
            self._set_selected_coordinate_system(self.coordinate_system_combo.currentIndex())
            self._refresh_label_options()
            # Coordinate-system switches intentionally clear the selected
            # segmentation instead of trying to preserve or auto-reload it in
            # the new coordinate system
            # (e.g. the case where a segmentation mask would be in two coordinate systems).
            # this is consistent with the viewer widget, where we also clear the napari viewer when switching coordinate system.
            self._clear_selected_segmentation()
            self._labels_layer_preparation_result = _LabelsLayerPreparationResult(kind="none")
            self._bind_current_selection(classifier_dirty_reason="the coordinate system changed")
        finally:
            self._is_handling_coordinate_system_change = False

    def _on_table_state_changed(self, event: object) -> None:
        if not isinstance(event, TableStateChangedEvent):
            return
        if event.sdata is not self.selected_spatialdata:
            return

        # A successful persistence reload is a table-wide lifecycle event, not
        # an Object Classification mutation source. Adopt it before the
        # feature-extraction and Spatial Query source filters below can return.
        if event.change_kind == "reloaded":
            self._adopt_reloaded_table_state(event)
            return
        if event.source == _SPATIAL_QUERY_ANNOTATION_SOURCE:
            self._consume_spatial_query_annotation(event)
            return
        if event.source != "feature_extraction":
            return

        feature_keys = tuple(path.keys[0] for path in event.paths if path.component == "obsm")
        if not feature_keys:
            return

        previous_effective_table_name = None if self._table_binding_error is not None else self.selected_table_name
        previous_table_name = self.selected_table_name
        previous_table_names = tuple(self._table_names)
        previous_feature_key = self.selected_feature_key
        self.persistence_controls.clear_feedback()
        # Prefer the event table only when the widget had no table context yet;
        # otherwise a newly created table must not steal an existing selection.
        preferred_table_name = event.table_name if previous_table_name is None and not previous_table_names else None
        self._refresh_table_names(preferred_table_name=preferred_table_name)

        next_table_binding_error = self._validate_selected_table_binding()
        next_effective_table_name = None if next_table_binding_error is not None else self.selected_table_name

        if (
            next_effective_table_name != previous_effective_table_name
            or self.selected_feature_key != previous_feature_key
        ):
            self._bind_current_selection()
            return

        if (
            event.table_name == previous_table_name
            and event.change_kind == "updated"
            and previous_feature_key in feature_keys
        ):
            self._classifier_controller.invalidate_for_feature_matrix_overwrite(previous_feature_key)

        self._update_selection_status()

    def prepare_for_table_reload(self, request: TableReloadRequest) -> None:
        """Freeze classifier work before a consumed table is reloaded.

        Classifier jobs only compute in a worker; their results are written to
        the table on the main thread. So they never make the table busy, and a
        reload does not wait for them: it cancels them here instead. A
        cancelled job's result, computed from the table being replaced, is
        discarded and never written to the table.

        This is a defense, see `TableReloadParticipant`: the controller also
        cancels these jobs when the widget receives the `reloaded` event
        (`reset_after_reload()`).
        """
        if request.sdata is not self.selected_spatialdata or request.table_name != self.selected_table_name:
            return
        self._classifier_controller.freeze_for_reload()

    def _adopt_reloaded_table_state(self, event: TableStateChangedEvent) -> None:
        """Rebind Object Classification from one successfully restored table."""
        if event.table_name != self.selected_table_name:
            return
        self._refresh_feature_matrix_keys()
        self._bind_current_selection()
        self._classifier_controller.reset_after_reload()

    def _consume_spatial_query_annotation(self, event: TableStateChangedEvent) -> None:
        """Refresh Object Classification after an external user-class annotation."""
        if event.table_name != self.selected_table_name:
            return
        if TableComponentPath("obs", (USER_CLASS_COLUMN,)) not in event.paths:
            return

        # We run self._bind_current_selection() because:
        #
        # Spatial Query has already mutated the live AnnData table
        #     → table.obs["user_class"] changed
        #     → table.uns["user_class_colors"] may also have changed
        #         ↓
        # TableStateChangedEvent is published
        #     → says which components changed
        #     → does not carry the changed instance IDs
        #         ↓
        # Object Classification receives the event
        #     → re-read and validate the current live table
        #     → refresh its controls from that table
        #     → rebuild complete Labels styling from that table
        self._bind_current_selection()
        if self._table_binding_error is not None:
            # Rebinding rejected the changed table, detached it from the
            # controllers, cancelled classifier work captured from its former
            # valid state, and applied the existing neutral invalid-table UI.
            return

        # External user-class changes invalidate classifier inputs just like
        # annotations made in this widget. mark_dirty() also rejects pending or
        # active work captured before this event; Auto-train owns the single
        # optional replacement request.
        with self._defer_classifier_control_updates():
            self._classifier_controller.mark_dirty(reason="the annotations changed")
            if self._auto_train_enabled:
                self._classifier_controller.schedule_retrain()
        self._update_selection_status()

    def _refresh_label_options(self) -> None:
        """Refresh segmentation choices from the selected coordinate system."""
        previous_identity = None if self._selected_label_option is None else self._selected_label_option.identity
        if self.selected_spatialdata is None or self.selected_coordinate_system is None:
            self._label_options = []
        else:
            self._label_options = get_spatialdata_labels_options_for_coordinate_system_from_sdata(
                sdata=self.selected_spatialdata,
                coordinate_system=self.selected_coordinate_system,
            )

        with QSignalBlocker(self.segmentation_combo):
            self.segmentation_combo.clear()
            for option in self._label_options:
                self.segmentation_combo.addItem(option.display_name)

            has_options = bool(self._label_options)
            self.segmentation_combo.setEnabled(has_options)

            # If the previously selected label is still available after a refresh,
            # keep it selected instead of resetting the user back to the first item.
            # When nothing was selected yet, stay explicitly unbound so opening the
            # widget does not auto-load or auto-bind the first segmentation.
            next_index = self._find_label_option_index(previous_identity)
            if has_options:
                self.segmentation_combo.setCurrentIndex(-1 if next_index is None else next_index)
            else:
                self.segmentation_combo.setCurrentIndex(-1)

        self._set_selected_label_option(self.segmentation_combo.currentIndex())

    def _clear_selection_inputs(self) -> None:
        self._coordinate_systems = []
        self._selected_coordinate_system = None
        self._label_options = []
        self._selected_label_option = None
        self._labels_layer_preparation_result = _LabelsLayerPreparationResult(kind="none")
        self._table_names = []
        self._selected_table_name = None
        self._table_binding_error = None
        self._feature_matrix_keys = []
        self._selected_feature_key = None

        with QSignalBlocker(self.coordinate_system_combo):
            self.coordinate_system_combo.clear()
            self.coordinate_system_combo.setEnabled(False)
            self.coordinate_system_combo.setCurrentIndex(-1)

        with QSignalBlocker(self.segmentation_combo):
            self.segmentation_combo.clear()
            self.segmentation_combo.setEnabled(False)
            self.segmentation_combo.setCurrentIndex(-1)

        with QSignalBlocker(self.table_combo):
            self.table_combo.clear()
            self.table_combo.setEnabled(False)
            self.table_combo.setCurrentIndex(-1)

        with QSignalBlocker(self.feature_matrix_combo):
            self.feature_matrix_combo.clear()
            self.feature_matrix_combo.setEnabled(False)
            self.feature_matrix_combo.setCurrentIndex(-1)

        self._apply_status_card_spec(self.classifier_preparation_status, None)
        self._set_annotation_feedback("")
        self._apply_status_card_spec(self.classifier_feedback, None)
        self.persistence_controls.clear_feedback()

    def _on_primary_labels_layers_changed(self) -> None:
        if self._is_preparing_labels_layer or self._is_handling_coordinate_system_change:
            return
        # A labels-layer insert/remove only changes live viewer availability,
        # not the shared SpatialData selection model. React narrow here:
        # clear the current segmentation if *its* live layer disappeared, or
        # rebind only if the selected segmentation was previously missing and
        # has now become available.
        self._labels_layer_preparation_result = _LabelsLayerPreparationResult(kind="none")
        if self._selected_segmentation_layer_was_removed():
            self._clear_selected_segmentation()
            self._bind_current_selection()
            return
        # If the form still points at a selected segmentation but the
        # controllers are currently unbound (for example because no live labels
        # layer was available a moment ago), then a newly inserted matching
        # labels layer should rebind the controllers. This lets the widget
        # recover when the selected segmentation becomes available again
        # without forcing a full auto-load pass on every labels-layer change.
        if self._selected_segmentation_layer_became_available():
            self._bind_current_selection()

    def _selected_segmentation_layer_was_removed(self) -> bool:
        if (
            self.selected_spatialdata is None
            or self.selected_segmentation_name is None
            or self.selected_coordinate_system is None
        ):
            return False

        loaded_layer = self._app_state.viewer_adapter.get_loaded_primary_labels_layer(
            self.selected_spatialdata,
            self.selected_segmentation_name,
            self.selected_coordinate_system,
        )
        return loaded_layer is None

    def _selected_segmentation_layer_became_available(self) -> bool:
        if (
            self.selected_spatialdata is None
            or self.selected_segmentation_name is None
            or self.selected_coordinate_system is None
            # This helper is only for the "selected in the form, but not
            # currently bound to a live labels layer" case. If annotation is
            # already bound to some labels layer, then nothing has "become
            # available" from the controller's perspective, so we can return early
            # without rebinding.
            or self._annotation_controller.labels_layer is not None
        ):
            return False

        loaded_layer = self._app_state.viewer_adapter.get_loaded_primary_labels_layer(
            self.selected_spatialdata,
            self.selected_segmentation_name,
            self.selected_coordinate_system,
        )
        return loaded_layer is not None

    def _clear_selected_segmentation(self) -> None:
        with QSignalBlocker(self.segmentation_combo):
            self.segmentation_combo.setCurrentIndex(-1)

        self._set_selected_label_option(-1)
        self._refresh_table_names()

    def _refresh_coordinate_systems(self) -> None:
        if self.selected_spatialdata is None:
            self._coordinate_systems = []
        else:
            self._coordinate_systems = get_coordinate_system_names_from_sdata(self.selected_spatialdata)

        with QSignalBlocker(self.coordinate_system_combo):
            self.coordinate_system_combo.clear()
            for coordinate_system in self._coordinate_systems:
                self.coordinate_system_combo.addItem(coordinate_system, coordinate_system)

            has_coordinate_systems = bool(self._coordinate_systems)
            self.coordinate_system_combo.setEnabled(has_coordinate_systems)

        self._sync_coordinate_system_combo_selection(self._app_state.coordinate_system)
        self._set_selected_coordinate_system(self.coordinate_system_combo.currentIndex())

    def _on_coordinate_system_changed(self, index: int) -> None:
        coordinate_system = self.coordinate_system_combo.itemData(index)
        # Publish the UI choice to shared app state. `_on_app_state_coordinate_system_changed(...)`
        # owns local selection and downstream refresh so all sources follow one path.
        changed = self._app_state.set_coordinate_system(
            coordinate_system if isinstance(coordinate_system, str) else None,
            source="object_classification_widget",
        )
        # changed is True
        #     → app state emits coordinate_system_changed
        #     → the shared event handler synchronizes all widgets
        #
        # changed is False
        #     → no event is emitted
        #     → this initiating widget must restore its own combo
        if not changed:
            self._sync_coordinate_system_combo_selection(self._app_state.coordinate_system)

    def _set_selected_coordinate_system(self, index: int) -> None:
        coordinate_system = self.coordinate_system_combo.itemData(index)
        self._selected_coordinate_system = coordinate_system if isinstance(coordinate_system, str) else None

    def _sync_coordinate_system_combo_selection(self, coordinate_system: str | None) -> None:
        with QSignalBlocker(self.coordinate_system_combo):
            if coordinate_system is None:
                self.coordinate_system_combo.setCurrentIndex(-1)
                return

            index = self.coordinate_system_combo.findData(coordinate_system)
            self.coordinate_system_combo.setCurrentIndex(index)

    def _on_segmentation_changed(self, index: int) -> None:
        self._set_selected_label_option(index)
        self._refresh_table_names()
        self._prepare_selected_labels_layer()
        self._bind_current_selection(classifier_dirty_reason="the labels element selection changed")

    def _set_selected_label_option(self, index: int) -> None:
        if index < 0 or index >= len(self._label_options):
            self._selected_label_option = None
        else:
            self._selected_label_option = self._label_options[index]

    def _find_label_option_index(self, identity: tuple[int, str] | None) -> int | None:
        if identity is None:
            return None

        for index, option in enumerate(self._label_options):
            if option.identity == identity:
                return index

        return None

    def _prepare_selected_labels_layer(self) -> _LabelsLayerPreparationResult:
        self._labels_layer_preparation_result = _LabelsLayerPreparationResult(kind="none")

        if (
            self.selected_spatialdata is None
            or self.selected_segmentation_name is None
            or self.selected_coordinate_system is None
        ):
            return self._labels_layer_preparation_result

        self._is_preparing_labels_layer = True
        try:
            existing_layer = self._app_state.viewer_adapter.get_loaded_primary_labels_layer(
                self.selected_spatialdata,
                self.selected_segmentation_name,
                self.selected_coordinate_system,
            )
            if existing_layer is not None:
                if self._app_state.viewer_adapter.layer_bindings.get_binding(existing_layer) is None:
                    self._app_state.viewer_adapter.register_labels_layer(
                        existing_layer,
                        sdata=self.selected_spatialdata,
                        labels_name=self.selected_segmentation_name,
                        coordinate_system=self.selected_coordinate_system,
                    )
                activated = self._app_state.viewer_adapter.activate_layer(existing_layer)
                if activated:
                    self._labels_layer_preparation_result = _LabelsLayerPreparationResult(
                        kind="activated",
                        labels_name=self.selected_segmentation_name,
                        coordinate_system=self.selected_coordinate_system,
                    )
                return self._labels_layer_preparation_result

            try:
                result = self._app_state.viewer_adapter.ensure_labels_loaded(
                    self.selected_spatialdata,
                    self.selected_segmentation_name,
                    self.selected_coordinate_system,
                )
            except ValueError as error:
                self._labels_layer_preparation_result = _LabelsLayerPreparationResult(kind="error", error=str(error))
                return self._labels_layer_preparation_result

            self._app_state.viewer_adapter.activate_layer(result.layer)
            self._labels_layer_preparation_result = _LabelsLayerPreparationResult(
                kind="loaded",
                labels_name=self.selected_segmentation_name,
                coordinate_system=self.selected_coordinate_system,
            )
            return self._labels_layer_preparation_result
        finally:
            self._is_preparing_labels_layer = False

    def _refresh_table_names(self, *, preferred_table_name: str | None = None) -> None:
        previous_table_name = self.selected_table_name

        if self.selected_spatialdata is None or self.selected_segmentation_name is None:
            self._table_names = []
        else:
            self._table_names = get_annotating_table_names(self.selected_spatialdata, self.selected_segmentation_name)

        with QSignalBlocker(self.table_combo):
            self.table_combo.clear()
            for table_name in self._table_names:
                self.table_combo.addItem(table_name, table_name)

            has_tables = bool(self._table_names)
            self.table_combo.setEnabled(has_tables)

            next_index = -1 if previous_table_name is None else self.table_combo.findData(previous_table_name)
            if next_index < 0 and preferred_table_name is not None:
                next_index = self.table_combo.findData(preferred_table_name)
            if has_tables:
                self.table_combo.setCurrentIndex(0 if next_index < 0 else next_index)
            else:
                self.table_combo.setCurrentIndex(-1)

        self._set_selected_table_name(self.table_combo.currentIndex())
        self._refresh_feature_matrix_keys()

    def _on_table_changed(self, index: int) -> None:
        self._set_selected_table_name(index)
        self._refresh_feature_matrix_keys()
        self._bind_current_selection(classifier_dirty_reason="the annotation table changed")

    def _refresh_feature_matrix_keys(self) -> None:
        previous_feature_key = self.selected_feature_key

        if self.selected_spatialdata is None or self.selected_table_name is None:
            self._feature_matrix_keys = []
        else:
            self._feature_matrix_keys = get_table_obsm_keys(self.selected_spatialdata, self.selected_table_name)

        with QSignalBlocker(self.feature_matrix_combo):
            self.feature_matrix_combo.clear()
            for feature_key in self._feature_matrix_keys:
                self.feature_matrix_combo.addItem(feature_key, feature_key)

            has_feature_matrices = bool(self._feature_matrix_keys)
            self.feature_matrix_combo.setEnabled(has_feature_matrices)

            next_index = (
                -1 if previous_feature_key is None else self.feature_matrix_combo.findData(previous_feature_key)
            )
            if has_feature_matrices:
                self.feature_matrix_combo.setCurrentIndex(0 if next_index < 0 else next_index)
            else:
                self.feature_matrix_combo.setCurrentIndex(-1)

        self._set_selected_feature_key(self.feature_matrix_combo.currentIndex())

    def _on_feature_matrix_changed(self, index: int) -> None:
        self._set_selected_feature_key(index)
        effective_table_name = None if self._table_binding_error is not None else self.selected_table_name
        classifier_context_changed = self._classifier_controller.bind(
            self.selected_spatialdata,
            self.selected_segmentation_name,
            effective_table_name,
            self.selected_feature_key,
            training_scope=self.selected_training_scope,
            prediction_scope=self.selected_prediction_scope,
        )
        if classifier_context_changed and effective_table_name is not None:
            self._classifier_controller.mark_dirty(reason="the feature matrix changed")
        self._refresh_layer_styling()
        self._update_selection_status()

    def _on_training_scope_changed(self, index: int) -> None:
        self._set_selected_training_scope(index)
        self._bind_current_selection(classifier_dirty_reason="the training scope changed")

    def _on_prediction_scope_changed(self, index: int) -> None:
        self._set_selected_prediction_scope(index)
        self._bind_current_selection(classifier_dirty_reason="the prediction scope changed")

    def _on_auto_train_toggled(self, checked: bool) -> None:
        self._auto_train_enabled = bool(checked)
        self._update_auto_train_tooltip()
        self._update_classifier_controls()

    def _on_color_by_changed(self, index: int) -> None:
        color_by = self.color_by_combo.itemData(index)
        if not isinstance(color_by, str):
            return

        self._viewer_styling_controller.set_color_by(color_by)
        self._refresh_layer_styling()

    def _set_selected_table_name(self, index: int) -> None:
        if index < 0 or index >= len(self._table_names):
            self._selected_table_name = None
        else:
            self._selected_table_name = self._table_names[index]

    def _set_selected_feature_key(self, index: int) -> None:
        if index < 0 or index >= len(self._feature_matrix_keys):
            self._selected_feature_key = None
        else:
            self._selected_feature_key = self._feature_matrix_keys[index]

    def _set_selected_training_scope(self, index: int) -> None:
        training_scope = self.training_scope_combo.itemData(index)
        if training_scope in ("selected_segmentation_only", "all"):
            self._selected_training_scope = training_scope
            return

        self._selected_training_scope = DEFAULT_TRAINING_SCOPE

    def _set_selected_prediction_scope(self, index: int) -> None:
        prediction_scope = self.prediction_scope_combo.itemData(index)
        if prediction_scope in ("selected_segmentation_only", "all"):
            self._selected_prediction_scope = prediction_scope
            return

        self._selected_prediction_scope = DEFAULT_PREDICTION_SCOPE

    def _validate_selected_table_binding(self) -> str | None:
        """Return why the selected table cannot be bound, without mutating it.

        Object Classification state is inspected after the structural
        labels/table relationship and before any controller adopts the table.
        Both failures therefore use the same all-or-nothing binding boundary.
        """
        if (
            self.selected_spatialdata is None
            or self.selected_segmentation_name is None
            or self.selected_table_name is None
        ):
            return None

        try:
            validate_table_binding(self.selected_spatialdata, self.selected_segmentation_name, self.selected_table_name)
        except ValueError as error:
            return str(error)

        table = get_table(self.selected_spatialdata, self.selected_table_name)
        try:
            validate_object_classification_table_state(table)
        except ObjectClassificationStateError as error:
            return str(error)

        return None

    def _bind_current_selection(self, *, classifier_dirty_reason: str | None = None) -> None:
        """Rebind every controller to the current widget selection.

        This is the central handoff point from widget-level selection state to
        the controllers. Whenever the selected segmentation, table, or feature
        matrix context changes, we call this helper so each controller updates
        which in-memory ``SpatialData`` object and table it should operate on.

        Importantly, the controllers do not own independent copies of
        ``SpatialData`` or ``AnnData``. They receive references to the current
        in-memory objects and resolve the selected table from that authoritative
        state. Rebinding here therefore "refreshes" controller state by
        updating those references, not by materializing new table objects.

        Before rebinding, the selected table is validated against the selected
        labels layer and its existing Object Classification columns are
        inspected read-only. If either contract is invalid, the table is
        intentionally downgraded to an ``effective_table_name`` of ``None`` for
        every table-dependent controller. The selected Labels layer remains
        bound and receives neutral styling, while the status card explains why
        the selected table was rejected.

        The method also:

        - validates whether the selected table can annotate the selected labels layer
          and contains valid Object Classification state
        - propagates that effective binding to annotation, classifier, styling,
          and persistence controllers
        - adopts existing class and prediction state read-only; missing
          ``user_class`` and prediction columns remain absent until an
          effective annotation or classifier write creates them
        - marks classifier outputs dirty when the classifier selection context
          changed in a way that invalidates them
        - re-applies layer styling and refreshes the user-facing status cards

        Downstream styling never mutates class columns or palettes. A missing
        or invalid ``user_class_colors`` palette receives a read-only display
        fallback. Prediction colors are derived read-only from the resolved
        user palette and class-id defaults. Invalid class-column structure is
        surfaced through the widget's table-binding status card.
        """
        self._table_binding_error = self._validate_selected_table_binding()
        effective_table_name = None if self._table_binding_error is not None else self.selected_table_name

        self._annotation_controller.bind(
            self.selected_spatialdata,
            self.selected_segmentation_name,
            effective_table_name,
            self.selected_coordinate_system,
        )
        classifier_context_changed = self._classifier_controller.bind(
            self.selected_spatialdata,
            self.selected_segmentation_name,
            effective_table_name,
            self.selected_feature_key,
            training_scope=self.selected_training_scope,
            prediction_scope=self.selected_prediction_scope,
        )
        if classifier_dirty_reason is not None and classifier_context_changed and effective_table_name is not None:
            self._classifier_controller.mark_dirty(reason=classifier_dirty_reason)
        self._viewer_styling_controller.bind(
            self.selected_spatialdata,
            self.selected_segmentation_name,
            effective_table_name,
            self.selected_coordinate_system,
        )
        self.persistence_controls.bind(
            self.selected_spatialdata,
            self.selected_table_name,
            self.selected_segmentation_name,
            binding_error=self._table_binding_error,
        )
        self._annotation_controller.activate_layer()
        self._refresh_layer_styling()
        self._set_annotation_feedback("")
        self.persistence_controls.clear_feedback()
        self._update_selection_status()

    def _update_selection_status(self) -> None:
        # The widget inspects feature metadata only to drive registration UI
        # state. Classifier training eligibility is owned by the controller.
        feature_matrix_metadata_state = self._selected_feature_matrix_metadata_state()
        self._update_warning_status_card()
        self._update_selection_status_card()
        self._update_feature_matrix_metadata_controls(feature_matrix_metadata_state)
        self._update_annotation_controls()
        self._update_color_by_controls()
        self._update_classifier_controls()
        self.persistence_controls.refresh()

    def _selected_feature_matrix_metadata_state(self) -> FeatureMatrixMetadataState | None:
        if (
            self.selected_spatialdata is None
            or self.selected_table_name is None
            or self.selected_feature_key is None
            or self._table_binding_error is not None
        ):
            return None

        table = get_table(self.selected_spatialdata, self.selected_table_name)
        return inspect_feature_matrix_metadata(table, self.selected_feature_key)

    def _register_selected_feature_matrix_metadata(self) -> None:
        if (
            self.selected_spatialdata is None
            or self.selected_table_name is None
            or self.selected_feature_key is None
            or self._table_binding_error is not None
        ):
            self._update_selection_status()
            return

        table = get_table(self.selected_spatialdata, self.selected_table_name)
        feature_matrix_metadata_state = inspect_feature_matrix_metadata(table, self.selected_feature_key)
        # Re-check the live metadata state at click time. The button may still
        # reflect an older UI state, but registration must never overwrite
        # metadata that was added or changed elsewhere.
        if feature_matrix_metadata_state.status != "unregistered":
            self._update_selection_status()
            return

        # Registration is an all-or-nothing UI action: if the core helper
        # rejects the matrix, show the error without marking persistence dirty
        # or making the classifier stale.
        try:
            register_feature_matrix_metadata(table, self.selected_feature_key)
        except ValueError as error:
            feature_matrix_metadata_state = self._selected_feature_matrix_metadata_state()
            self._update_feature_matrix_metadata_controls(feature_matrix_metadata_state)
            self._update_classifier_controls()
            self.persistence_controls.refresh()
            set_status_card(
                self.warning_status,
                title="Feature Metadata Warning",
                lines=[str(error)],
                kind="warning",
            )
            return

        self._record_table_mutation(
            frozenset(
                {
                    TableComponentPath(
                        "uns",
                        ("feature_matrices", self.selected_feature_key),
                    )
                }
            ),
            regions=(),
            source="object_classification_feature_metadata",
        )
        self._classifier_controller.mark_dirty(reason="feature matrix metadata registered")
        self._update_selection_status()

    def _update_warning_status_card(self) -> None:
        """Refresh the shared warning slot used by object-classification setup."""
        if self._table_binding_error is not None:
            # The primary selection status already reports the blocking table
            # error. Clear this secondary slot so it cannot retain a stale
            # feature-metadata or layer-styling warning for the rejected table.
            self._apply_status_card_spec(self.warning_status, None)
            return

        feature_metadata_warning_message = None
        if self._layer_styling_error is None:
            feature_matrix_metadata_state = self._selected_feature_matrix_metadata_state()
            feature_metadata_warning_message = _build_feature_matrix_registration_button_state(
                feature_matrix_metadata_state
            ).warning_message

        spec = build_object_classification_warning_status_card_spec(
            layer_styling_error=self._layer_styling_error,
            selected_table_name=self.selected_table_name,
            feature_matrix_count=self.feature_matrix_combo.count(),
            feature_metadata_warning_message=feature_metadata_warning_message,
        )
        self._apply_status_card_spec(self.warning_status, spec)

    def _update_feature_matrix_metadata_controls(
        self,
        feature_matrix_metadata_state: FeatureMatrixMetadataState | None,
    ) -> None:
        self.feature_matrix_combo.setEnabled(self._table_binding_error is None and bool(self._feature_matrix_keys))
        if self._table_binding_error is not None:
            self.register_feature_matrix_button.setEnabled(False)
            self._set_tooltip(self.register_feature_matrix_button, self._table_binding_error)
            return

        state = _build_feature_matrix_registration_button_state(feature_matrix_metadata_state)
        self.register_feature_matrix_button.setEnabled(state.enabled)
        self._set_tooltip(self.register_feature_matrix_button, state.tooltip)

    def _update_selection_status_card(self) -> None:
        spec = build_object_classification_selection_status_card_spec(
            has_spatialdata=self._app_state.sdata is not None,
            selected_coordinate_system=self.selected_coordinate_system,
            selected_segmentation_name=self.selected_segmentation_name,
            labels_layer_loaded=self._annotation_controller.labels_layer is not None,
            labels_layer_preparation_result=self._labels_layer_preparation_result,
            selected_table_name=self.selected_table_name,
            table_binding_error=self._table_binding_error,
            missing_table_row_message=self._annotation_controller.missing_table_row_message,
            selected_instance_id=self.selected_instance_id,
            instance_key_name=self._selected_instance_key_name(),
            current_user_class=self._annotation_controller.current_user_class,
        )
        self._apply_status_card_spec(self.selection_status, spec)

    def _update_annotation_controls(self) -> None:
        has_table = self.selected_table_name is not None and self._table_binding_error is None
        current_user_class = self._annotation_controller.current_user_class
        can_apply = self._annotation_controller.can_annotate
        can_clear = can_apply and current_user_class is not None

        self.class_spinbox.setEnabled(has_table)
        self.apply_class_button.setEnabled(can_apply)
        self.clear_class_button.setEnabled(can_clear)

        self._set_tooltip(
            self.apply_class_button,
            self._annotation_action_tooltip(
                enabled=can_apply,
                ready_message="Assign the selected user class to the picked object.",
                unavailable_message="Pick an annotated object in the viewer before applying a class.",
                shortcut_hint=_APPLY_CLASS_SHORTCUT,
            ),
        )
        self._set_tooltip(
            self.clear_class_button,
            self._annotation_action_tooltip(
                enabled=can_clear,
                ready_message="Clear the current user class for the picked object.",
                unavailable_message="Pick a labeled object before clearing its user class.",
                shortcut_hint=_REMOVE_CLASS_SHORTCUT,
            ),
        )

    def _annotation_action_tooltip(
        self,
        *,
        enabled: bool,
        ready_message: str,
        unavailable_message: str,
        shortcut_hint: str,
    ) -> str:
        message = ready_message if enabled else unavailable_message
        return f"{message} Shortcut: {shortcut_hint}."

    def _trigger_apply_class_shortcut(self) -> None:
        if not self.apply_class_button.isEnabled():
            return

        self._apply_current_class()

    def _trigger_clear_class_shortcut(self) -> None:
        if not self.clear_class_button.isEnabled():
            return

        self._clear_current_class()

    def _apply_current_class(self) -> None:
        class_id = self.class_spinbox.value()
        try:
            warning_message = self._annotation_controller.apply_class(class_id)
        except ValueError as error:
            self._set_annotation_feedback(str(error), kind="error")
            return

        if warning_message is not None:
            self._set_annotation_feedback(warning_message, kind="warning")
            self._update_selection_status()
            return

        self._set_annotation_feedback(
            f"Assigned class {class_id} to {self._selected_instance_key_name()} {self.selected_instance_id}.",
            kind="success",
        )
        self._update_selection_status()

    def _clear_current_class(self) -> None:
        try:
            warning_message = self._annotation_controller.clear_current_class()
        except ValueError as error:
            self._set_annotation_feedback(str(error), kind="error")
            return

        if warning_message is not None:
            self._set_annotation_feedback(warning_message, kind="warning")
            self._update_selection_status()
            return

        self._set_annotation_feedback(
            f"Cleared the user class for {self._selected_instance_key_name()} {self.selected_instance_id}.",
            kind="success",
        )
        self._update_selection_status()

    def _set_annotation_feedback(self, message: str, *, kind: StatusCardKind = "success") -> None:
        if not message:
            self.annotation_feedback.setText("")
            self.annotation_feedback.setVisible(False)
            return

        title_by_kind = {
            "error": "Annotation Error",
            "warning": "Annotation Warning",
            "success": "Annotation Updated",
        }
        set_status_card(
            self.annotation_feedback,
            title=title_by_kind.get(kind, "Annotation"),
            lines=[message],
            kind=kind,
        )

    def _apply_status_card_spec(
        self,
        label: QLabel,
        spec: _ObjectClassificationStatusCardSpec | None,
    ) -> None:
        if spec is None:
            label.setText("")
            label.setToolTip("")
            label.setVisible(False)
            return

        set_status_card(
            label,
            title=spec.title,
            lines=list(spec.lines),
            kind=spec.kind,
            tooltip_message=spec.tooltip_message,
        )

    def _update_classifier_feedback(self) -> None:
        is_visible = (
            self.selected_spatialdata is not None
            and self.selected_segmentation_name is not None
            and self.selected_table_name is not None
            and self.selected_feature_key is not None
            and self._table_binding_error is None
        )
        spec = build_object_classification_classifier_feedback_card_spec(
            is_visible=is_visible,
            message=self._classifier_controller.status_message,
            kind=self._classifier_controller.status_kind,
        )
        self._apply_status_card_spec(self.classifier_feedback, spec)

    def _selected_instance_key_name(self) -> str:
        instance_key_name = self._annotation_controller.selected_instance_key_name
        if instance_key_name is not None:
            return instance_key_name

        return "label value"

    def _update_color_by_controls(self) -> None:
        has_table = self.selected_table_name is not None and self._table_binding_error is None
        self.color_by_combo.setEnabled(has_table)

        if not has_table:
            tooltip = (
                self._table_binding_error
                if self._table_binding_error is not None
                else "Choose an annotation table before changing the labels-layer coloring mode."
            )
        elif self.selected_color_by == COLOR_BY_USER_CLASS:
            tooltip = 'Color the labels layer by "user_class".'
        elif self.selected_color_by == COLOR_BY_PRED_CLASS:
            tooltip = 'Color the labels layer by "pred_class" using the stable user-class palette.'
        elif self.selected_color_by == COLOR_BY_PRED_CONFIDENCE:
            tooltip = f'Color the labels layer by continuous "{COLOR_BY_PRED_CONFIDENCE}" values.'
        else:
            tooltip = "Choose how to color the labels layer."

        self._set_tooltip(self.color_by_combo, tooltip)

    def _update_classifier_controls(self) -> None:
        self.auto_train_checkbox.setEnabled(self._table_binding_error is None)
        self._update_auto_train_tooltip()
        can_configure_scope = (
            self.selected_spatialdata is not None
            and self.selected_segmentation_name is not None
            and self.selected_table_name is not None
            and self._table_binding_error is None
            and not self._classifier_controller.is_training
        )
        self.training_scope_combo.setEnabled(can_configure_scope)
        self.prediction_scope_combo.setEnabled(can_configure_scope)

        if (
            self.selected_spatialdata is None
            or self.selected_segmentation_name is None
            or self.selected_table_name is None
        ):
            training_scope_tooltip = (
                "Choose a labels element and annotation table before configuring classifier training scope."
            )
            prediction_scope_tooltip = (
                "Choose a labels element and annotation table before configuring classifier prediction scope."
            )
        elif self._table_binding_error is not None:
            training_scope_tooltip = self._table_binding_error
            prediction_scope_tooltip = self._table_binding_error
        elif self._classifier_controller.is_training:
            training_scope_tooltip = "A classifier training job is currently running."
            prediction_scope_tooltip = "A classifier training job is currently running."
        elif self.selected_training_scope == "all":
            training_scope_tooltip = (
                "Train on eligible labeled rows from all annotated labels elements in the selected table."
            )
            prediction_scope_tooltip = self._prediction_scope_tooltip()
        else:
            training_scope_tooltip = "Train only on eligible labeled rows from the selected labels element."
            prediction_scope_tooltip = self._prediction_scope_tooltip()

        self._set_tooltip(self.training_scope_combo, training_scope_tooltip)
        self._set_tooltip(self.prediction_scope_combo, prediction_scope_tooltip)
        classifier_preparation_summary = self._classifier_controller.describe_current_preparation()
        classifier_preparation_spec = build_object_classification_classifier_preparation_card_spec(
            selected_segmentation_name=self.selected_segmentation_name,
            selected_table_name=self.selected_table_name,
            selected_feature_key=self.selected_feature_key,
            table_binding_error=self._table_binding_error,
            summary=classifier_preparation_summary,
        )
        self._apply_status_card_spec(self.classifier_preparation_status, classifier_preparation_spec)

        can_retrain = (
            classifier_preparation_summary is not None
            and classifier_preparation_summary.eligible
            and not self._classifier_controller.is_training
        )
        self.retrain_button.setEnabled(can_retrain)
        can_export = self._classifier_controller.can_export_classifier
        self.export_classifier_button.setEnabled(can_export)

        if self.selected_spatialdata is None or self.selected_table_name is None:
            tooltip = "Choose a labels element and annotation table to enable classifier training."
        elif self._table_binding_error is not None:
            tooltip = self._table_binding_error
        elif self.selected_feature_key is None:
            tooltip = "Choose a feature matrix before training the classifier."
        elif self._classifier_controller.is_training:
            tooltip = "A classifier training job is currently running."
        elif (
            classifier_preparation_summary is not None
            and classifier_preparation_summary.training_blocker_reason is not None
        ):
            tooltip = classifier_preparation_summary.training_blocker_reason
        elif self._classifier_controller.is_dirty:
            tooltip = "The classifier model is stale. Click Train Classifier to refresh predictions."
        else:
            tooltip = (
                "Train the classifier using the current annotations and feature matrix, then write predictions "
                "for the selected prediction scope."
            )

        self._set_tooltip(self.retrain_button, tooltip)

        export_unavailable_reason = self._classifier_controller.classifier_export_unavailable_reason
        if can_export:
            export_tooltip = "Save the current fitted classifier and its feature schema to a joblib artifact."
        elif export_unavailable_reason:
            export_tooltip = export_unavailable_reason
        else:
            export_tooltip = "Train a classifier before exporting it."

        self._set_tooltip(self.export_classifier_button, export_tooltip)

    def _prediction_scope_tooltip(self) -> str:
        if self.selected_prediction_scope == "all":
            return (
                "Write predictions for all eligible rows in the selected table. In-scope rows with invalid "
                "features will be cleared."
            )

        return "Write predictions only for eligible rows from the selected labels element."

    def _on_selected_instance_changed(self, instance_id: int | None) -> None:
        del instance_id
        self._set_annotation_feedback("")
        self._update_selection_status_card()
        self._update_annotation_controls()

    def _on_annotation_changed(self, change: UserClassAnnotationChange) -> None:
        changed_paths: set[TableComponentPath] = set()
        if change.state_change.user_class_changed:
            changed_paths.add(TableComponentPath("obs", (USER_CLASS_COLUMN,)))
        if change.state_change.palette_changed:
            changed_paths.add(TableComponentPath("uns", (USER_CLASS_COLORS_KEY,)))
        self._record_table_mutation(
            frozenset(changed_paths),
            regions=() if self.selected_segmentation_name is None else (self.selected_segmentation_name,),
            change_kind=("updated" if change.user_class_was_available_as_color_source else "created"),
            source="object_classification_annotation",
        )
        self._refresh_after_user_class_annotation(change)
        # `mark_dirty(...)` and auto-train scheduling emit classifier status
        # callbacks. Let feedback update immediately, but avoid multiple
        # `_update_classifier_controls()` calls until `_update_selection_status()`.
        with self._defer_classifier_control_updates():
            self._classifier_controller.mark_dirty(reason="the annotations changed")
            if self._auto_train_enabled:
                self._classifier_controller.schedule_retrain()
        self._update_selection_status()

    def _on_classifier_table_state_changed(self, change: ClassifierTableStateChange) -> None:
        # Record every classifier-owned table mutation for persistence and
        # shared table-state notification. Prediction-specific labels-layer
        # styling is refreshed separately through
        # `_on_classifier_prediction_state_changed()`.
        self._record_table_mutation(
            change.paths,
            regions=change.regions,
            source=change.source,
        )
        # _record_table_mutation() synchronously publishes table_state_changed;
        # the shared handler refreshes persistence controls for this table.

    def _on_classifier_prediction_state_changed(self) -> None:
        # Prediction changes are the classifier-owned table changes that affect
        # labels-layer coloring/features.
        self._refresh_layer_styling()

    def _on_classifier_state_changed(self) -> None:
        self._update_classifier_feedback()
        if self._is_deferring_classifier_control_updates:
            return
        self._update_classifier_controls()

    @contextmanager
    def _defer_classifier_control_updates(self) -> Iterator[None]:
        """Defer expensive classifier-control refreshes during annotation updates."""
        previous = self._is_deferring_classifier_control_updates
        self._is_deferring_classifier_control_updates = True
        try:
            yield
        finally:
            self._is_deferring_classifier_control_updates = previous

    def _retrain_classifier(self) -> None:
        if not self._classifier_controller.can_retrain:
            self._update_classifier_controls()
            return

        self._classifier_controller.mark_dirty(reason="the user requested classifier training")
        self._classifier_controller.retrain_now()
        self._update_selection_status()

    def _export_classifier(self) -> None:
        selected_path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export Classifier",
            str(self._default_classifier_export_path()),
            "Spatiato classifier (*.spatiato-classifier.joblib);;Joblib files (*.joblib);;All files (*)",
        )
        if not selected_path:
            return

        path = self._normalize_classifier_export_path(Path(selected_path))
        try:
            bundle = self._classifier_controller.export_classifier(path)
        except ValueError as error:
            self._set_classifier_export_feedback(str(error), error=True)
            self._update_classifier_controls()
            return

        self._set_classifier_export_feedback(
            f'Exported classifier with {bundle.n_features} feature columns to "{path}".',
            error=False,
        )
        self._update_classifier_controls()

    def _default_classifier_export_path(self) -> Path:
        table_name = self.selected_table_name or "classifier"
        feature_key = self.selected_feature_key or "features"
        stem = f"{_sanitize_export_stem(table_name)}_{_sanitize_export_stem(feature_key)}"
        return Path.cwd() / f"{stem}{DEFAULT_CLASSIFIER_EXPORT_SUFFIX}"

    def _normalize_classifier_export_path(self, path: Path) -> Path:
        if str(path).endswith(".joblib"):
            return path
        return path.with_name(f"{path.name}{DEFAULT_CLASSIFIER_EXPORT_SUFFIX}")

    def _set_classifier_export_feedback(self, message: str, *, error: bool) -> None:
        set_status_card(
            self.classifier_feedback,
            title="Classifier Export Error" if error else "Classifier Exported",
            lines=[message],
            kind="error" if error else "success",
        )

    def _refresh_layer_styling(self) -> None:
        if self._table_binding_error is not None:
            layer = self._annotation_controller.labels_layer
            if layer is not None:
                # The labels layer is already loaded. Replace any previous
                # table-backed colors with the shared neutral style because
                # the rejected table cannot provide trustworthy class values.
                apply_neutral_labels_style(layer)
                self._app_state.viewer_adapter.sync_labels_display_after_colormap_change(layer)
            self._layer_styling_error = None
            self._update_warning_status_card()
            return

        try:
            self._viewer_styling_controller.refresh()
        except ClassStateError as error:
            self._layer_styling_error = str(error)
            self._update_warning_status_card()
            return

        self._layer_styling_error = None
        self._update_warning_status_card()

    def _refresh_after_user_class_annotation(self, change: UserClassAnnotationChange) -> None:
        """Refresh labels after one annotation, preferring row-scoped updates.

        To keep annotation responsive, try the narrowest safe viewer update
        before falling back to a full layer refresh:

        - `user_class` coloring: update the edited label color and feature row
        - prediction coloring: update only the edited `user_class` feature row
        - other modes or unsafe layer state: perform a full refresh
        """
        color_by = self._viewer_styling_controller.color_by
        if color_by == COLOR_BY_USER_CLASS:
            try:
                row_scoped_refresh_applied = self._viewer_styling_controller.refresh_user_class_colormap_and_feature(
                    change
                )
            except ClassStateError as error:
                self._layer_styling_error = str(error)
                self._update_warning_status_card()
                return
            if row_scoped_refresh_applied:
                self._layer_styling_error = None
                self._update_warning_status_card()
                return
            # Full refresh recovery for stale/missing feature state where the
            # edited label cannot be matched reliably for a row-scoped update.
            logger.warning(
                "Row-scoped user-class annotation refresh was unavailable for instance "
                f"`{change.instance_id}`; falling back to full labels-layer styling refresh."
            )
            self._refresh_layer_styling()
            return

        if color_by in (COLOR_BY_PRED_CLASS, COLOR_BY_PRED_CONFIDENCE):
            # Prediction color modes do not color by `user_class`; keep the
            # edited `user_class` value current in layer features only.
            feature_refresh_applied = self._viewer_styling_controller.refresh_user_class_feature_only(change)
            if feature_refresh_applied:
                return
            logger.warning(
                "Row-scoped user-class feature refresh was unavailable for instance "
                f"`{change.instance_id}`; falling back to full labels-layer styling refresh."
            )
            self._refresh_layer_styling()
            return

        self._refresh_layer_styling()

    def _update_auto_train_tooltip(self) -> None:
        if self._table_binding_error is not None:
            tooltip = self._table_binding_error
        elif self._auto_train_enabled:
            tooltip = "Automatically train the classifier after each annotation."
        else:
            tooltip = "Keep predictions stale while annotating; click Train Classifier to update predictions."

        self._set_tooltip(self.auto_train_checkbox, tooltip)

    def _record_table_mutation(
        self,
        paths: frozenset[TableComponentPath],
        *,
        regions: tuple[str, ...],
        change_kind: Literal["created", "updated", "removed", "rebuilt"] = "updated",
        source: str,
    ) -> None:
        if (
            self.selected_spatialdata is None
            or self.selected_table_name is None
            or self._table_binding_error is not None
        ):
            return

        self._app_state.record_table_mutation(
            TableStateChangedEvent(
                sdata=self.selected_spatialdata,
                table_name=self.selected_table_name,
                paths=paths,
                regions=regions,
                change_kind=change_kind,
                source=source,
            )
        )
        self.persistence_controls.clear_feedback()


def _sanitize_export_stem(value: str) -> str:
    normalized = "".join(
        character if character.isascii() and (character.isalnum() or character in ("-", "_")) else "_"
        for character in value
    )
    return normalized.strip("_") or "classifier"
