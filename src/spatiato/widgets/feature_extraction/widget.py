from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import TYPE_CHECKING, Literal

from qtpy.QtCore import QSignalBlocker, Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from spatiato._app_state import (
    SpatiatoAppState,
    TableStateChangedEvent,
    get_or_create_app_state,
)
from spatiato.core.persistence import TableComponentPath
from spatiato.core.spatialdata import (
    SpatialDataImageOption,
    SpatialDataLabelsOption,
    get_annotating_table_names,
    get_coordinate_system_names_from_sdata,
    get_image_channel_names_from_sdata,
    get_spatialdata_feature_extraction_image_discovery_for_coordinate_system_and_label_from_sdata,
    get_spatialdata_feature_extraction_label_discovery_for_coordinate_system_from_sdata,
    validate_table_annotation_coverage,
    validate_table_region_instance_ids,
)
from spatiato.core.validation import (
    normalize_spatialdata_name,
    spatialdata_element_name_exists,
)
from spatiato.widgets.feature_extraction.controller import (
    FeatureExtractionBindingState,
    FeatureExtractionController,
    FeatureExtractionResult,
    FeatureExtractionTriplet,
)
from spatiato.widgets.feature_extraction.status_card import (
    _FeatureExtractionStatusCardSpec,
    build_feature_extraction_controller_feedback_card_spec,
    build_feature_extraction_selection_status_card_spec,
    build_feature_extraction_status_card_entries,
)
from spatiato.widgets.shared_styles import (
    CALCULATE_BUTTON_STYLESHEET as _CALCULATE_BUTTON_STYLESHEET,
)
from spatiato.widgets.shared_styles import (
    CHECKBOX_STYLESHEET as _FEATURE_CHECKBOX_STYLESHEET,
)
from spatiato.widgets.shared_styles import (
    PRIMARY_BUTTON_STYLESHEET,
    SECONDARY_BUTTON_STYLESHEET,
    STATUS_CARD_PALETTE,
    WIDGET_BORDER_COLOR,
    WIDGET_PANEL_COLOR,
    WIDGET_TEXT_COLOR,
    WIDGET_WARNING_BACKGROUND_COLOR,
    WIDGET_WARNING_BORDER_COLOR,
    WIDGET_WARNING_TEXT_COLOR,
    CompactComboBox,
    apply_scroll_content_surface,
    apply_widget_surface,
    build_input_control_stylesheet,
    create_form_label,
    create_header_logo,
    format_feedback_identifier,
    format_tooltip,
    set_status_card,
)
from spatiato.widgets.shared_styles import (
    WIDGET_MIN_WIDTH as _WIDGET_MIN_WIDTH,
)
from spatiato.widgets.shared_styles import (
    WIDGET_SURFACE_COLOR as _WIDGET_SURFACE_COLOR,
)

if TYPE_CHECKING:
    import napari
    from spatialdata import SpatialData


_INPUT_CONTROL_STYLESHEET = build_input_control_stylesheet("QComboBox, QLineEdit")
_FEATURE_GROUP_STYLESHEET = (
    "QGroupBox {"
    f"background-color: {WIDGET_PANEL_COLOR}; "
    f"border: 1px solid {WIDGET_BORDER_COLOR}; "
    "border-radius: 10px; "
    f"color: {WIDGET_TEXT_COLOR}; "
    "font-weight: 700; "
    "margin-top: 10px; "
    "padding: 12px 12px 10px 12px;}"
    "QGroupBox QComboBox, QGroupBox QLineEdit {"
    "font-weight: 400;"
    "}"
    "QGroupBox::title {"
    "subcontrol-origin: margin; "
    "left: 12px; "
    f"padding: 0 6px; color: {WIDGET_TEXT_COLOR}; background-color: {_WIDGET_SURFACE_COLOR};"
    "}"
)
_FEATURE_HINT_INFO_PALETTE = STATUS_CARD_PALETTE["info"]
_FEATURE_HINT_INFO_STYLESHEET = (
    "QLabel {"
    f"background-color: {_FEATURE_HINT_INFO_PALETTE['background']}; "
    f"border: 1px solid {_FEATURE_HINT_INFO_PALETTE['border']}; "
    "border-radius: 8px; "
    f"color: {_FEATURE_HINT_INFO_PALETTE['text']}; "
    "font-size: 12px; "
    "font-weight: 500; "
    "padding: 6px 8px;}"
)
_FEATURE_HINT_WARNING_STYLESHEET = (
    "QLabel {"
    f"background-color: {WIDGET_WARNING_BACKGROUND_COLOR}; "
    f"border: 1px solid {WIDGET_WARNING_BORDER_COLOR}; "
    "border-radius: 8px; "
    f"color: {WIDGET_WARNING_TEXT_COLOR}; "
    "font-size: 12px; "
    "font-weight: 600; "
    "padding: 6px 8px;}"
)
_CHANNEL_SELECTION_PANEL_STYLESHEET = "QWidget { background: transparent; }"
_NO_IMAGE_TEXT = "No image"
_INTENSITY_FEATURES = ("sum", "mean", "var", "min", "max", "kurtosis", "skew")
_MORPHOLOGY_FEATURES = (
    "area",
    "eccentricity",
    "major_axis_length",
    "minor_axis_length",
    "perimeter",
    "convex_area",
    "equivalent_diameter",
    "major_minor_axis_ratio",
    "perim_square_over_area",
    "major_axis_equiv_diam_ratio",
    "convex_hull_resid",
    "centroid_dif",
)
_DEFAULT_FEATURE_MATRIX_KEY = "features"
_DEFAULT_NEW_TABLE_NAME = "features_table"


def _shutdown_feature_extraction_controller(
    app_state: SpatiatoAppState,
    controller: FeatureExtractionController,
) -> None:
    """Detach the destroyed widget without opening a table-reload race.

    Requesting worker cancellation prevents late results from reaching the
    widget, but an already-running ``hp.tb.add_feature_matrix()`` call may
    continue writing its table. Keep the controller registered as an app-state
    table-reload participant until that Harpy call has finished:

    widget destroyed while Harpy is writing
        → detach widget callbacks
        → retain the controller's matching-table reload guard
        → unregister the controller after the worker finishes
    """
    controller.shutdown()
    controller.release_reload_participant_when_safe(lambda: app_state.unregister_table_reload_participant(controller))


def _quoted_names(names: Sequence[str]) -> str:
    return ", ".join(f'"{name}"' for name in names)


_CREATE_TABLE_OPTION_TEXT = "Create table..."
_MAX_VISIBLE_EXTRACTION_CHANNELS = 5
_FEATURE_GROUPS_TOP_SPACING = 12
_COORDINATE_SYSTEM_CHECKBOX_MAX_LENGTH = 20
_CHOOSE_SEGMENTATION_TEXT = "Choose a labels element"
ElementIdentity = tuple[int, str]


class _FeatureExtractionTableComboMode(Enum):
    EXISTING = "existing"
    CREATE_TABLE = "create_table"


@dataclass(frozen=True)
class _FeatureExtractionTableComboData:
    mode: _FeatureExtractionTableComboMode
    table_name: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.mode, _FeatureExtractionTableComboMode):
            raise ValueError("Unknown table combo mode.")

        if self.mode is _FeatureExtractionTableComboMode.EXISTING:
            if self.table_name is None or not self.table_name.strip():
                raise ValueError("Existing table combo data requires a table name.")
            return

        if self.mode is _FeatureExtractionTableComboMode.CREATE_TABLE and self.table_name is not None:
            raise ValueError("Create-table combo data cannot carry a table name.")

    @classmethod
    def existing(cls, table_name: str) -> _FeatureExtractionTableComboData:
        return cls(_FeatureExtractionTableComboMode.EXISTING, table_name)

    @classmethod
    def create_table(cls) -> _FeatureExtractionTableComboData:
        return cls(_FeatureExtractionTableComboMode.CREATE_TABLE)


@dataclass(frozen=True)
class _FeatureExtractionTripletCardWidgets:
    coordinate_system: str
    container: QGroupBox
    segmentation_combo: CompactComboBox
    segmentation_note_label: QLabel
    image_combo: CompactComboBox
    image_note_label: QLabel


@dataclass(frozen=True)
class _FeatureExtractionTripletCardState:
    coordinate_system: str | None
    selectable_label_options: list[SpatialDataLabelsOption]
    selected_label_option: SpatialDataLabelsOption | None
    segmentation_note_text: str | None
    selectable_image_options: list[SpatialDataImageOption]
    selected_image_option: SpatialDataImageOption | None
    image_note_text: str | None


@dataclass(frozen=True)
class _FeatureExtractionCardSelection:
    label_identity: ElementIdentity | None
    image_identity: ElementIdentity | None


@dataclass(frozen=True)
class _FeatureExtractionBatchChannelState:
    """Resolved shared-channel compatibility snapshot for the checked cards.

    This state answers one narrow question for the widget:
    given the currently selected images across the checked coordinate-system
    cards, do they expose one shared ordered channel schema that can drive the
    batch channel selector?

    - `reference_coordinate_system` and `reference_image_option` identify the
      first selected image that established the shared ordered channel schema.
    - `channel_names` is that shared ordered schema, for example
      `("DAPI", "GFP")` or `("0", "1", "2")`.
    - `incompatible_coordinate_systems` and `incompatible_image_names` record
      selected images that do not match the shared schema, or whose channel
      names could not be resolved cleanly.
    - `error_text` is the shared batch-level feedback message for any channel
      incompatibility.

    This is separate from `_FeatureExtractionStagedBatchState`: channel
    compatibility is only one ingredient in staged-batch validity, while this
    object exists specifically to drive the shared channel selector and related
    intensity-feature warnings.
    """

    reference_coordinate_system: str | None
    reference_image_option: SpatialDataImageOption | None
    channel_names: tuple[str, ...]
    incompatible_coordinate_systems: tuple[str, ...]
    incompatible_image_names: tuple[str, ...]
    error_text: str | None

    @property
    def is_compatible(self) -> bool:
        return self.error_text is None


@dataclass(frozen=True)
class _FeatureExtractionStagedBatchState:
    """Resolved widget-side snapshot of the staged extraction targets.

    This state describes what the user wants to calculate on: the checked
    coordinate systems, the labels elements selected for those systems, and the
    resulting controller triplets. It is bindable only when the staged batch is
    complete enough to send to the controller.
    """

    checked_coordinate_systems: tuple[str, ...]
    labels_names: tuple[str, ...]
    triplets: tuple[FeatureExtractionTriplet, ...]
    invalid_coordinate_systems: tuple[str, ...]
    error_text: str | None

    @property
    def is_bindable(self) -> bool:
        return self.error_text is None and bool(self.triplets)


@dataclass(frozen=True)
class _FeatureExtractionTableTargetState:
    """Resolved widget-side snapshot of the table destination.

    This state describes where the feature matrix should be written. In
    existing-table mode, `table_name` names a selected table in `sdata.tables`.
    In create-table mode, `table_name` names the new table that Harpy should
    create. Incomplete or invalid UI state is represented with `table_name=None`
    or `error_text`.
    """

    table_name: str | None
    create_table: bool
    error_text: str | None = None

    @property
    def bound_table_name(self) -> str | None:
        return self.table_name if self.error_text is None else None

    @property
    def is_ready(self) -> bool:
        return self.bound_table_name is not None


@dataclass
class _FeatureExtractionChannelSelectionMemory:
    """Remember user channel selections across shared-schema refreshes.

    Terminology:

    - `schema` means the current ordered tuple of channel names exposed by the
      shared batch-compatible images, for example `("DAPI", "GFP")` or
      `("0", "1", "2")`.
    - `current_selection` is the last explicit channel subset chosen by the
      user for the schema currently visible in the widget.
    - `selections_by_schema` stores the last explicit user selection for each
      previously seen ordered schema, keyed by that exact schema tuple. For
      example, `{("0", "1", "2"): ("0", "2")}` means that when the widget
      next sees the ordered channel schema `("0", "1", "2")`, it should
      restore the user's earlier choice of channels `"0"` and `"2"`.

    We keep this memory separate from `_FeatureExtractionBatchChannelState`
    because it is longer-lived widget state rather than a pure snapshot of the
    currently resolved batch channel compatibility.
    """

    current_selection: tuple[str, ...] | None = None
    selections_by_schema: dict[tuple[str, ...], tuple[str, ...]] = field(default_factory=dict)

    def reset(self) -> None:
        """Clear all remembered channel-selection state for the loaded `SpatialData`."""
        self.current_selection = None
        self.selections_by_schema.clear()

    @staticmethod
    def is_valid_for_schema(
        schema: tuple[str, ...],
        selection: tuple[str, ...] | None,
    ) -> bool:
        """Return whether a remembered selection still fits within one schema."""
        if selection is None:
            return False
        schema_set = set(schema)
        return all(channel_name in schema_set for channel_name in selection)

    def resolve_for_schema(self, schema: tuple[str, ...]) -> tuple[str, ...]:
        """Return the best selection to restore for one ordered channel schema."""
        if self.is_valid_for_schema(schema, self.current_selection):
            return self.current_selection  # type: ignore[return-value]

        remembered_selection = self.selections_by_schema.get(schema)
        if self.is_valid_for_schema(schema, remembered_selection):
            return remembered_selection  # type: ignore[return-value]

        # Default to no channels selected so intensity feature extraction never runs on every
        # channel without an explicit choice from the user.
        return ()

    def remember_for_schema(self, schema: tuple[str, ...], selection: tuple[str, ...]) -> None:
        """Persist one explicit user selection for the current ordered schema."""
        self.current_selection = selection
        self.selections_by_schema[schema] = selection


class FeatureExtractionWidget(QWidget):
    """
    Batch-aware widget for feature extraction from a shared `SpatialData`.

    The widget stages one or more explicit
    `coordinate_system -> segmentation -> image` triplets from the loaded
    `SpatialData` object:

    - users check one or more coordinate systems and get one triplet card per
      checked coordinate system;
    - each card resolves selectable segmentation masks and matching images
      from shared discovery helpers;
    - duplicate segmentation selection is prevented across the visible batch,
      while valid image reuse remains allowed across cards;
    - shared controls below the cards resolve one batch channel selection, one
      output table, one feature-matrix key, and the shared feature groups;
    - the widget keeps a staged batch state and binds an explicit multi-target
      request into `FeatureExtractionController` only when the full checked
      batch is currently valid.
    """

    def __init__(self, napari_viewer: napari.Viewer | None = None) -> None:
        super().__init__()
        self.setObjectName("feature_extraction_widget")
        apply_widget_surface(self)
        self.setMinimumWidth(_WIDGET_MIN_WIDTH)

        # The napari viewer identifies which shared Spatiato session this widget
        # belongs to. We use it to attach to the per-viewer SpatiatoAppState
        # instead of scanning viewer layers directly.
        self._app_state = get_or_create_app_state(napari_viewer)
        self._feature_extraction_controller = FeatureExtractionController(
            on_state_changed=self._on_controller_state_changed,
            on_table_selection_refresh_required=self._on_controller_table_selection_refresh_required,
            on_feature_matrix_changed=self._on_controller_feature_matrix_changed,
        )

        self._label_options: list[SpatialDataLabelsOption] = []
        self._selected_label_option: SpatialDataLabelsOption | None = None
        self._image_options: list[SpatialDataImageOption] = []
        self._selected_image_option: SpatialDataImageOption | None = None
        self._batch_channel_names: list[str] = []
        self._batch_channel_checkboxes: list[QCheckBox] = []
        self._channel_selection_memory = _FeatureExtractionChannelSelectionMemory()
        self._batch_channel_error: str | None = None
        self._batch_channel_state = _FeatureExtractionBatchChannelState(
            reference_coordinate_system=None,
            reference_image_option=None,
            channel_names=(),
            incompatible_coordinate_systems=(),
            incompatible_image_names=(),
            error_text=None,
        )
        self._staged_batch_state = _FeatureExtractionStagedBatchState(
            checked_coordinate_systems=(),
            labels_names=(),
            triplets=(),
            invalid_coordinate_systems=(),
            error_text=None,
        )
        self._table_target_state = _FeatureExtractionTableTargetState(
            table_name=None,
            create_table=False,
        )
        # Persistent per-coordinate-system memory of the last explicit user
        # card selection while the current SpatialData object remains loaded.
        self._remembered_card_selection_by_coordinate_system: dict[str, _FeatureExtractionCardSelection] = {}
        # Existing real tables that annotate every currently staged labels element.
        self._eligible_existing_table_names: list[str] = []
        self._create_table = False
        self._selected_table_name: str | None = None
        # Proposed create-table target name.
        self._new_table_name: str | None = None
        self._coordinate_systems: list[str] = []
        self._checked_coordinate_systems: list[str] = []
        self._selected_coordinate_system: str | None = None
        self._feature_checkboxes: dict[str, QCheckBox] = {}
        self._coordinate_system_checkboxes: dict[str, QCheckBox] = {}
        # Multi-card state is split into:
        # - one widget map for the rendered card controls per coordinate system
        # - one state map for the resolved logical card state per coordinate system
        # - one active-card snapshot used as a compatibility bridge while the
        #   lower part of the widget is still largely single-card oriented
        self._triplet_card_widgets_by_coordinate_system: dict[str, _FeatureExtractionTripletCardWidgets] = {}
        self._triplet_card_states_by_coordinate_system: dict[str, _FeatureExtractionTripletCardState] = {}
        self._triplet_card_state = _FeatureExtractionTripletCardState(
            coordinate_system=None,
            selectable_label_options=[],
            selected_label_option=None,
            segmentation_note_text=None,
            selectable_image_options=[],
            selected_image_option=None,
            image_note_text=None,
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.scroll_area = QScrollArea()
        self.scroll_area.setObjectName("feature_extraction_scroll_area")
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_area.setStyleSheet("QScrollArea { border: 0px; background: transparent; }")

        self.scroll_content = QWidget()
        self.scroll_content.setObjectName("feature_extraction_scroll_content")
        apply_scroll_content_surface(self.scroll_content)

        content_layout = QVBoxLayout(self.scroll_content)
        content_layout.setContentsMargins(12, 12, 12, 12)
        content_layout.setSpacing(10)
        self.scroll_area.setWidget(self.scroll_content)
        layout.addWidget(self.scroll_area)

        title = create_header_logo("feature_extraction_header_logo")

        selector_layout = QFormLayout()
        selector_layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        selector_layout.setHorizontalSpacing(12)
        selector_layout.setVerticalSpacing(10)

        shared_controls_layout = QFormLayout()
        shared_controls_layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        shared_controls_layout.setHorizontalSpacing(12)
        shared_controls_layout.setVerticalSpacing(10)

        self._triplet_card_widgets = self._create_placeholder_triplet_card_widgets()
        self.segmentation_combo = self._triplet_card_widgets.segmentation_combo
        self.image_combo = self._triplet_card_widgets.image_combo

        self.coordinate_system_selection_container = QWidget()
        self.coordinate_system_selection_container.setObjectName(
            "feature_extraction_coordinate_system_selection_container"
        )
        coordinate_system_selection_layout = QVBoxLayout(self.coordinate_system_selection_container)
        coordinate_system_selection_layout.setContentsMargins(0, 0, 0, 0)
        coordinate_system_selection_layout.setSpacing(6)
        self.coordinate_system_selection_layout = coordinate_system_selection_layout

        self.triplet_cards_container = QWidget()
        self.triplet_cards_container.setObjectName("feature_extraction_triplet_cards_container")
        triplet_cards_layout = QVBoxLayout(self.triplet_cards_container)
        triplet_cards_layout.setContentsMargins(0, 0, 0, 0)
        triplet_cards_layout.setSpacing(10)
        self.triplet_cards_layout = triplet_cards_layout

        self.channel_selection_label = self._create_form_label("Channels")
        self.channel_selection_container = QWidget()
        self.channel_selection_container.setObjectName("feature_extraction_channel_selection_container")
        self.channel_selection_container.setStyleSheet(_CHANNEL_SELECTION_PANEL_STYLESHEET)
        self.channel_selection_container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        channel_selection_layout = QVBoxLayout(self.channel_selection_container)
        channel_selection_layout.setContentsMargins(0, 0, 0, 0)
        channel_selection_layout.setSpacing(0)

        self.channel_selection_scroll_area = QScrollArea()
        self.channel_selection_scroll_area.setObjectName("feature_extraction_channel_selection_scroll_area")
        self.channel_selection_scroll_area.setWidgetResizable(True)
        self.channel_selection_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.channel_selection_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.channel_selection_scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.channel_selection_scroll_area.setStyleSheet("QScrollArea { border: 0px; background: transparent; }")
        self.channel_selection_scroll_area.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.channel_selection_scroll_area.setMinimumWidth(self.image_combo.sizeHint().width())

        self.channel_selection_list_widget = QWidget()
        self.channel_selection_list_widget.setObjectName("feature_extraction_channel_selection_list")
        self.channel_selection_list_widget.setStyleSheet(_CHANNEL_SELECTION_PANEL_STYLESHEET)
        self.channel_selection_list_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.channel_selection_list_layout = QVBoxLayout(self.channel_selection_list_widget)
        self.channel_selection_list_layout.setContentsMargins(0, 0, 0, 0)
        self.channel_selection_list_layout.setSpacing(6)
        self.channel_selection_scroll_area.setWidget(self.channel_selection_list_widget)
        channel_selection_layout.addWidget(self.channel_selection_scroll_area)
        self.channel_selection_note_label = QLabel()
        self.channel_selection_note_label.setObjectName("feature_extraction_channel_selection_note")
        self.channel_selection_note_label.setWordWrap(True)
        self.channel_selection_note_label.setStyleSheet(_FEATURE_HINT_WARNING_STYLESHEET)
        self.channel_selection_note_label.hide()
        channel_selection_layout.addWidget(self.channel_selection_note_label)

        self.channel_selection_label.hide()
        self.channel_selection_container.hide()

        self.table_combo = CompactComboBox()
        self.table_combo.setObjectName("feature_extraction_table_combo")
        self.table_combo.currentIndexChanged.connect(self._on_table_changed)
        self.table_combo.setStyleSheet(_INPUT_CONTROL_STYLESHEET)

        self.new_table_name_label = self._create_form_label("New table name")
        self.new_table_name_label.setObjectName("feature_extraction_new_table_name_label")
        self.new_table_name_label.hide()

        self.new_table_name_line_edit = QLineEdit()
        self.new_table_name_line_edit.setObjectName("feature_extraction_new_table_name_line_edit")
        self.new_table_name_line_edit.setPlaceholderText(_DEFAULT_NEW_TABLE_NAME)
        self.new_table_name_line_edit.textChanged.connect(self._on_new_table_name_changed)
        self.new_table_name_line_edit.setStyleSheet(_INPUT_CONTROL_STYLESHEET)
        self.new_table_name_line_edit.hide()

        self.coordinate_system_combo = CompactComboBox()
        self.coordinate_system_combo.setObjectName("feature_extraction_coordinate_system_combo")
        self.coordinate_system_combo.currentIndexChanged.connect(self._on_coordinate_system_changed)
        self.coordinate_system_combo.setStyleSheet(_INPUT_CONTROL_STYLESHEET)
        self.coordinate_system_combo.hide()
        self._set_tooltip(
            self.coordinate_system_combo,
            "Choose a coordinate system in which the labels element and image are registered. In the selected coordinate system they should have the same shape, and only an identity transform, a translation, or a sequence of translations may be defined relative to it.",
        )

        self.output_key_line_edit = QLineEdit()
        self.output_key_line_edit.setObjectName("feature_extraction_output_key_line_edit")
        self.output_key_line_edit.setText(_DEFAULT_FEATURE_MATRIX_KEY)
        self.output_key_line_edit.setPlaceholderText("e.g. features")
        self.output_key_line_edit.textChanged.connect(self._on_feature_key_changed)
        self.output_key_line_edit.setStyleSheet(_INPUT_CONTROL_STYLESHEET)
        self._set_tooltip(
            self.output_key_line_edit,
            'This key names the feature matrix created by the Feature Extraction widget. It will appear in the Object Classification widget under "Feature matrix". The matrix is stored in the selected table as `.obsm[output_key]`, with companion metadata in `.uns["feature_matrices"][output_key]`.',
        )

        self.intensity_features_group = self._create_feature_group(
            "Intensity Features",
            _INTENSITY_FEATURES,
            object_name="feature_extraction_intensity_features_group",
        )
        self.intensity_features_hint = QLabel()
        self.intensity_features_hint.setObjectName("feature_extraction_intensity_features_hint")
        self.intensity_features_hint.setWordWrap(True)
        self.intensity_features_group.layout().addWidget(self.intensity_features_hint)

        self.morphology_features_group = self._create_feature_group(
            "Morphology Features",
            _MORPHOLOGY_FEATURES,
            object_name="feature_extraction_morphology_features_group",
        )

        self.calculate_action_row = QWidget()
        self.calculate_action_row.setObjectName("feature_extraction_calculate_action_row")
        calculate_action_layout = QHBoxLayout(self.calculate_action_row)
        calculate_action_layout.setContentsMargins(0, 0, 0, 0)
        calculate_action_layout.setSpacing(8)

        self.calculate_button = QPushButton("Calculate")
        self.calculate_button.setObjectName("feature_extraction_calculate_button")
        self.calculate_button.clicked.connect(self._calculate_feature_matrix)
        self.calculate_button.setEnabled(False)
        self.calculate_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.calculate_button.setMinimumHeight(28)
        self.calculate_button.setStyleSheet(_CALCULATE_BUTTON_STYLESHEET)
        self._set_tooltip(
            self.calculate_button,
            "Calculation is enabled once the shared SpatialData selection and feature choices form a valid extraction request.",
        )

        self.selection_status = QLabel()
        self.selection_status.setObjectName("feature_extraction_selection_status")
        self.selection_status.setWordWrap(True)
        self.selection_status.hide()

        self.controller_feedback = QLabel()
        self.controller_feedback.setObjectName("feature_extraction_controller_feedback")
        self.controller_feedback.setWordWrap(True)
        self.controller_feedback.hide()

        selector_layout.addRow(
            self._create_form_label("Coordinate systems"), self.coordinate_system_selection_container
        )
        shared_controls_layout.addRow(self.channel_selection_label, self.channel_selection_container)
        shared_controls_layout.addRow(self._create_form_label("Table"), self.table_combo)
        shared_controls_layout.addRow(self.new_table_name_label, self.new_table_name_line_edit)
        shared_controls_layout.addRow(self._create_form_label("Feature matrix key"), self.output_key_line_edit)

        calculate_action_layout.addWidget(self.calculate_button, 1)

        content_layout.addWidget(title)
        content_layout.addLayout(selector_layout)
        content_layout.addWidget(self.selection_status)
        content_layout.addWidget(self.controller_feedback)
        content_layout.addWidget(self.triplet_cards_container)
        content_layout.addLayout(shared_controls_layout)
        content_layout.addSpacing(_FEATURE_GROUPS_TOP_SPACING)
        content_layout.addWidget(self.intensity_features_group)
        content_layout.addWidget(self.morphology_features_group)
        content_layout.addWidget(self.calculate_action_row)
        content_layout.addStretch(1)

        self._app_state.sdata_changed.connect(self._on_sdata_changed)
        self._app_state.table_state_changed.connect(self._on_table_state_changed)
        # Feature Extraction writes through Harpy from its worker. Register the
        # controller—not this QWidget—so exact-table reload protection can
        # safely outlive Qt teardown until every non-cooperative call finishes.
        self._app_state.register_table_reload_participant(self._feature_extraction_controller)
        app_state = self._app_state
        controller = self._feature_extraction_controller
        self.destroyed.connect(
            lambda *_args, app_state=app_state, controller=controller: _shutdown_feature_extraction_controller(
                app_state,
                controller,
            )
        )
        self._update_intensity_features_hint()
        self.refresh_from_sdata(self._app_state.sdata)

    @property
    def app_state(self) -> SpatiatoAppState:
        """Return the shared Spatiato app state for this widget."""
        return self._app_state

    @property
    def selected_segmentation_name(self) -> str | None:
        """Return the currently selected labels element name."""
        return None if self._selected_label_option is None else self._selected_label_option.labels_name

    @property
    def selected_spatialdata(self) -> SpatialData | None:
        """Return the loaded SpatialData object backing the current widget state."""
        return self._app_state.sdata

    @property
    def selected_image_name(self) -> str | None:
        """Return the currently selected image element name, if any."""
        return None if self._selected_image_option is None else self._selected_image_option.image_name

    @property
    def selected_extraction_channel_names(self) -> tuple[str, ...] | None:
        """Return selected batch extraction-channel names for the current schema.

        `None` means no channel schema/selector is currently available. `()`
        means a selector is available, but no channel checkboxes are selected.
        """
        if not self._batch_channel_names:
            return None

        return tuple(checkbox.text() for checkbox in self._batch_channel_checkboxes if checkbox.isChecked())

    @property
    def selected_extraction_channel_indices(self) -> tuple[int, ...] | None:
        """Return selected batch extraction-channel indices for the current schema.

        `None` means no channel schema/selector is currently available. `()`
        means a selector is available, but no channel checkboxes are selected.
        """
        if not self._batch_channel_names:
            return None

        return tuple(index for index, checkbox in enumerate(self._batch_channel_checkboxes) if checkbox.isChecked())

    @property
    def selected_table_name(self) -> str | None:
        """Return the currently selected annotation table name."""
        return self._selected_table_name

    @property
    def selected_table_mode(self) -> Literal["existing", "create"] | None:
        """Return the current table target mode."""
        if self._create_table:
            return "create"
        if self._selected_table_name is not None:
            return "existing"
        return None

    @property
    def selected_new_table_name(self) -> str | None:
        """Return the trimmed create-table target name, if create mode is selected."""
        if not self._create_table:
            return None

        value = self._new_table_name
        if value is None:
            value = self.new_table_name_line_edit.text()
        value = value.strip()
        return value or None

    @property
    def selected_coordinate_system(self) -> str | None:
        """Return the currently selected coordinate system."""
        return self._selected_coordinate_system

    @property
    def selected_feature_names(self) -> tuple[str, ...]:
        """Return selected feature names in stable UI order."""
        ordered_features = (*_INTENSITY_FEATURES, *_MORPHOLOGY_FEATURES)
        return tuple(feature for feature in ordered_features if self._feature_checkboxes[feature].isChecked())

    @property
    def selected_feature_key(self) -> str | None:
        """Return the trimmed feature-matrix output key, if any."""
        value = self.output_key_line_edit.text().strip()
        return value or None

    @property
    def overwrite_feature_key(self) -> bool:
        """Return whether an existing feature key should be replaced."""
        return False

    def refresh_from_sdata(self, sdata: SpatialData | None) -> None:
        """Refresh the widget from the shared Spatiato SpatialData state."""
        if sdata is None:
            self._clear_selection_inputs()
            self._bind_current_selection()
            return

        self._reset_staged_triplet_state()
        self._refresh_coordinate_systems()
        self._refresh_batch_channel_options()
        self._refresh_table_names()
        self._update_intensity_features_hint()
        self._bind_current_selection()

    def _on_sdata_changed(self, sdata: SpatialData | None) -> None:
        self.refresh_from_sdata(sdata)

    def _on_table_state_changed(self, value: object) -> None:
        """Adopt restored feature state after one successful shared reload."""
        if not isinstance(value, TableStateChangedEvent) or value.change_kind != "reloaded":
            return
        if value.sdata is not self.selected_spatialdata or value.table_name != self.selected_table_name:
            return

        # The shared reload has already replaced the live table components.
        # Re-read table availability and rebind the existing UI choices so the
        # output key and overwrite readiness derive from that restored table.
        self._refresh_table_names(preferred_existing_table_name=value.table_name)
        self._bind_current_selection()

    def _clear_selection_inputs(self) -> None:
        self._reset_staged_triplet_state()
        self._set_active_card_widgets(None)
        self._clear_triplet_cards()
        self._clear_coordinate_system_checkboxes()

        with QSignalBlocker(self.segmentation_combo):
            self.segmentation_combo.clear()
            self.segmentation_combo.setEnabled(False)
            self.segmentation_combo.setCurrentIndex(-1)

        with QSignalBlocker(self.image_combo):
            self.image_combo.clear()
            self.image_combo.addItem(_NO_IMAGE_TEXT, None)
            self.image_combo.setEnabled(False)
            self.image_combo.setCurrentIndex(0)

        with QSignalBlocker(self.table_combo):
            self.table_combo.clear()
            self.table_combo.setEnabled(False)
            self.table_combo.setCurrentIndex(-1)
        self._set_create_table_name_controls_visible(False)

        with QSignalBlocker(self.coordinate_system_combo):
            self.coordinate_system_combo.clear()
            self.coordinate_system_combo.setEnabled(False)
            self.coordinate_system_combo.setCurrentIndex(-1)

    def _reset_staged_triplet_state(self) -> None:
        self._label_options = []
        self._selected_label_option = None
        self._image_options = []
        self._selected_image_option = None
        self._batch_channel_names = []
        self._batch_channel_checkboxes = []
        self._channel_selection_memory.reset()
        self._batch_channel_error = None
        self._batch_channel_state = _FeatureExtractionBatchChannelState(
            reference_coordinate_system=None,
            reference_image_option=None,
            channel_names=(),
            incompatible_coordinate_systems=(),
            incompatible_image_names=(),
            error_text=None,
        )
        self._staged_batch_state = _FeatureExtractionStagedBatchState(
            checked_coordinate_systems=(),
            labels_names=(),
            triplets=(),
            invalid_coordinate_systems=(),
            error_text=None,
        )
        self._table_target_state = _FeatureExtractionTableTargetState(
            table_name=None,
            create_table=False,
        )
        self._remembered_card_selection_by_coordinate_system = {}
        self._clear_batch_channel_options()
        self._eligible_existing_table_names = []
        self._create_table = False
        self._selected_table_name = None
        self._new_table_name = None
        if hasattr(self, "new_table_name_line_edit"):
            with QSignalBlocker(self.new_table_name_line_edit):
                self.new_table_name_line_edit.clear()
            self._set_create_table_name_controls_visible(False)
        self._coordinate_systems = []
        self._checked_coordinate_systems = []
        self._selected_coordinate_system = None
        self._coordinate_system_checkboxes = {}
        self._triplet_card_widgets_by_coordinate_system = {}
        self._triplet_card_states_by_coordinate_system = {}
        self._triplet_card_state = _FeatureExtractionTripletCardState(
            coordinate_system=None,
            selectable_label_options=[],
            selected_label_option=None,
            segmentation_note_text=None,
            selectable_image_options=[],
            selected_image_option=None,
            image_note_text=None,
        )

    def _create_form_label(self, text: str) -> QLabel:
        return create_form_label(text)

    def _create_triplet_card_note_label(self, object_name: str) -> QLabel:
        note_label = QLabel()
        note_label.setObjectName(object_name)
        note_label.setWordWrap(True)
        note_label.setStyleSheet(_FEATURE_HINT_INFO_STYLESHEET)
        note_label.hide()
        return note_label

    def _create_placeholder_triplet_card_widgets(self) -> _FeatureExtractionTripletCardWidgets:
        """Create hidden placeholder controls used when no visible card is active."""
        segmentation_combo = CompactComboBox()
        segmentation_combo.setObjectName("feature_extraction_segmentation_combo")
        segmentation_combo.setPlaceholderText(_CHOOSE_SEGMENTATION_TEXT)
        segmentation_combo.setStyleSheet(_INPUT_CONTROL_STYLESHEET)
        segmentation_combo.hide()
        segmentation_note_label = self._create_triplet_card_note_label("feature_extraction_segmentation_note")
        segmentation_note_label.hide()

        image_combo = CompactComboBox()
        image_combo.setObjectName("feature_extraction_image_combo")
        image_combo.setStyleSheet(_INPUT_CONTROL_STYLESHEET)
        image_combo.hide()
        image_note_label = self._create_triplet_card_note_label("feature_extraction_image_note")
        image_note_label.hide()

        container = QGroupBox()
        container.hide()

        return _FeatureExtractionTripletCardWidgets(
            coordinate_system="",
            container=container,
            segmentation_combo=segmentation_combo,
            segmentation_note_label=segmentation_note_label,
            image_combo=image_combo,
            image_note_label=image_note_label,
        )

    def _create_triplet_card_widgets(self, coordinate_system: str) -> _FeatureExtractionTripletCardWidgets:
        """Create one visible triplet-card widget for a coordinate system."""
        visible_coordinate_system, coordinate_system_shortened = format_feedback_identifier(
            coordinate_system,
            max_length=_COORDINATE_SYSTEM_CHECKBOX_MAX_LENGTH,
        )
        container = QGroupBox(visible_coordinate_system)
        container.setObjectName(f"feature_extraction_triplet_card_{coordinate_system}")
        container.setStyleSheet(_FEATURE_GROUP_STYLESHEET)
        if coordinate_system_shortened:
            self._set_tooltip(container, coordinate_system)

        layout = QVBoxLayout(container)
        layout.setContentsMargins(12, 18, 12, 10)
        layout.setSpacing(10)

        segmentation_combo = CompactComboBox()
        segmentation_combo.setObjectName(f"feature_extraction_segmentation_combo_{coordinate_system}")
        segmentation_combo.currentIndexChanged.connect(
            lambda index, current_coordinate_system=coordinate_system: self._on_triplet_card_segmentation_changed(
                current_coordinate_system, index
            )
        )
        segmentation_combo.setPlaceholderText(_CHOOSE_SEGMENTATION_TEXT)
        segmentation_combo.setStyleSheet(_INPUT_CONTROL_STYLESHEET)
        segmentation_note_label = self._create_triplet_card_note_label(
            f"feature_extraction_segmentation_note_{coordinate_system}"
        )

        image_combo = CompactComboBox()
        image_combo.setObjectName(f"feature_extraction_image_combo_{coordinate_system}")
        image_combo.currentIndexChanged.connect(
            lambda index, current_coordinate_system=coordinate_system: self._on_triplet_card_image_changed(
                current_coordinate_system, index
            )
        )
        image_combo.setStyleSheet(_INPUT_CONTROL_STYLESHEET)
        image_note_label = self._create_triplet_card_note_label(f"feature_extraction_image_note_{coordinate_system}")

        layout.addWidget(self._create_form_label("Labels"))
        layout.addWidget(segmentation_combo)
        layout.addWidget(segmentation_note_label)
        layout.addWidget(self._create_form_label("Image"))
        layout.addWidget(image_combo)
        layout.addWidget(image_note_label)

        return _FeatureExtractionTripletCardWidgets(
            coordinate_system=coordinate_system,
            container=container,
            segmentation_combo=segmentation_combo,
            segmentation_note_label=segmentation_note_label,
            image_combo=image_combo,
            image_note_label=image_note_label,
        )

    def _get_remembered_card_selection_for_coordinate_system(
        self,
        coordinate_system: str | None,
    ) -> _FeatureExtractionCardSelection:
        if coordinate_system is None:
            return _FeatureExtractionCardSelection(label_identity=None, image_identity=None)

        return self._remembered_card_selection_by_coordinate_system.get(
            coordinate_system,
            _FeatureExtractionCardSelection(label_identity=None, image_identity=None),
        )

    def _clear_coordinate_system_checkboxes(self) -> None:
        while self.coordinate_system_selection_layout.count():
            item = self.coordinate_system_selection_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._coordinate_system_checkboxes = {}

    def _clear_triplet_cards(self) -> None:
        while self.triplet_cards_layout.count():
            item = self.triplet_cards_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._triplet_card_widgets_by_coordinate_system = {}
        self._triplet_card_states_by_coordinate_system = {}

    def _create_feature_group(self, title: str, feature_names: tuple[str, ...], *, object_name: str) -> QGroupBox:
        group = QGroupBox(title)
        group.setObjectName(object_name)
        group.setStyleSheet(_FEATURE_GROUP_STYLESHEET)
        layout = QVBoxLayout(group)
        layout.setContentsMargins(12, 18, 12, 8)
        layout.setSpacing(8)

        feature_layout = QVBoxLayout()
        feature_layout.setContentsMargins(0, 0, 0, 0)
        feature_layout.setSpacing(6)
        for feature_name in feature_names:
            checkbox = QCheckBox(feature_name)
            checkbox.setObjectName(f"feature_checkbox_{feature_name}")
            checkbox.setStyleSheet(_FEATURE_CHECKBOX_STYLESHEET)
            checkbox.toggled.connect(self._on_feature_selection_changed)
            self._feature_checkboxes[feature_name] = checkbox
            feature_layout.addWidget(checkbox)

        layout.addLayout(feature_layout)
        return group

    def _set_tooltip(self, widget: QWidget, message: str) -> None:
        widget.setToolTip(format_tooltip(message))

    @staticmethod
    def _format_count_phrase(count: int, singular: str) -> str:
        suffix = "" if count == 1 else "s"
        return f"{count} {singular}{suffix}"

    def _set_triplet_card_note_text(self, note_label: QLabel, text: str | None) -> None:
        note_label.setText(text or "")
        note_label.setVisible(bool(text))

    def _build_segmentation_note_text(
        self,
        *,
        coordinate_system: str,
        blocked_label_options_by_owner: list[tuple[str, SpatialDataLabelsOption]],
        blocked_restored_selection: tuple[str, SpatialDataLabelsOption] | None,
        unavailable_label_count: int,
    ) -> str | None:
        fragments: list[str] = []

        if blocked_restored_selection is not None:
            owner_coordinate_system, blocked_option = blocked_restored_selection
            fragments.append(
                f'"{blocked_option.display_name}" already selected in "{owner_coordinate_system}", '
                "so choose a different labels element"
            )
        elif blocked_label_options_by_owner:
            if len(blocked_label_options_by_owner) == 1:
                owner_coordinate_system, blocked_option = blocked_label_options_by_owner[0]
                fragments.append(f'"{blocked_option.display_name}" already selected in "{owner_coordinate_system}"')
            else:
                fragments.append(
                    f"{self._format_count_phrase(len(blocked_label_options_by_owner), 'labels element')} already selected in other cards"
                )

        if unavailable_label_count > 0:
            transform_subject = "its" if unavailable_label_count == 1 else "their"
            fragments.append(
                f"{self._format_count_phrase(unavailable_label_count, 'labels element')} unavailable because "
                f'{transform_subject} supported transform relative to "{coordinate_system}" is not '
                "translation-only or identity"
            )

        if not fragments:
            return None

        return "; ".join(fragments) + "."

    def _build_image_note_text(self, *, coordinate_system: str, unavailable_image_count: int) -> str | None:
        if unavailable_image_count <= 0:
            return None

        if unavailable_image_count == 1:
            return (
                "1 image unavailable because it does not have the same shape and transform relative "
                f'to "{coordinate_system}" as the selected labels element.'
            )

        return (
            f"{unavailable_image_count} images unavailable because they do not have the same shape and "
            f'transform relative to "{coordinate_system}" as the selected labels element.'
        )

    def _sync_remembered_card_selection_from_state(
        self,
        coordinate_system: str,
        state: _FeatureExtractionTripletCardState,
    ) -> None:
        self._remembered_card_selection_by_coordinate_system[coordinate_system] = _FeatureExtractionCardSelection(
            label_identity=None if state.selected_label_option is None else state.selected_label_option.identity,
            image_identity=None if state.selected_image_option is None else state.selected_image_option.identity,
        )

    def _recompute_triplet_card_state_for_coordinate_system(
        self,
        coordinate_system: str | None,
        *,
        preferred_selection: _FeatureExtractionCardSelection | None = None,
        selected_label_identity: ElementIdentity | None = None,
        selected_label_owner_by_identity: Mapping[ElementIdentity, str] | None = None,
    ) -> _FeatureExtractionTripletCardState:
        """Resolve one coordinate-system card state from shared `SpatialData`."""
        if self.selected_spatialdata is None or coordinate_system is None:
            return _FeatureExtractionTripletCardState(
                coordinate_system=coordinate_system,
                selectable_label_options=[],
                selected_label_option=None,
                segmentation_note_text=None,
                selectable_image_options=[],
                selected_image_option=None,
                image_note_text=None,
            )

        preferred_selection = (
            self._get_remembered_card_selection_for_coordinate_system(coordinate_system)
            if preferred_selection is None
            else preferred_selection
        )
        selected_label_identity = (
            preferred_selection.label_identity if selected_label_identity is None else selected_label_identity
        )
        label_discovery = get_spatialdata_feature_extraction_label_discovery_for_coordinate_system_from_sdata(
            sdata=self.selected_spatialdata,
            coordinate_system=coordinate_system,
        )
        eligible_label_options = label_discovery.eligible_label_options
        selected_label_owner_by_identity = (
            {} if selected_label_owner_by_identity is None else dict(selected_label_owner_by_identity)
        )
        selectable_label_options = [
            option
            for option in eligible_label_options
            if selected_label_owner_by_identity.get(option.identity) in (None, coordinate_system)
        ]

        blocked_label_options_by_owner = [
            (selected_label_owner_by_identity[option.identity], option)
            for option in eligible_label_options
            if selected_label_owner_by_identity.get(option.identity) not in (None, coordinate_system)
        ]
        blocked_restored_selection: tuple[str, SpatialDataLabelsOption] | None = None
        if (
            preferred_selection.label_identity is not None
            and selected_label_identity is None
            and preferred_selection.label_identity in selected_label_owner_by_identity
            and selected_label_owner_by_identity[preferred_selection.label_identity] != coordinate_system
        ):
            blocked_option = next(
                (option for option in eligible_label_options if option.identity == preferred_selection.label_identity),
                None,
            )
            if blocked_option is not None:
                blocked_restored_selection = (
                    selected_label_owner_by_identity[preferred_selection.label_identity],
                    blocked_option,
                )
                blocked_label_options_by_owner = [
                    (owner_coordinate_system, option)
                    for owner_coordinate_system, option in blocked_label_options_by_owner
                    if option.identity != preferred_selection.label_identity
                ]

        selected_label_option = next(
            (option for option in selectable_label_options if option.identity == selected_label_identity),
            None,
        )
        segmentation_note_text = self._build_segmentation_note_text(
            coordinate_system=coordinate_system,
            blocked_label_options_by_owner=blocked_label_options_by_owner,
            blocked_restored_selection=blocked_restored_selection,
            unavailable_label_count=label_discovery.unavailable_label_count,
        )

        if selected_label_option is None:
            return _FeatureExtractionTripletCardState(
                coordinate_system=coordinate_system,
                selectable_label_options=selectable_label_options,
                selected_label_option=None,
                segmentation_note_text=segmentation_note_text,
                selectable_image_options=[],
                selected_image_option=None,
                image_note_text=None,
            )

        image_discovery = get_spatialdata_feature_extraction_image_discovery_for_coordinate_system_and_label_from_sdata(
            sdata=self.selected_spatialdata,
            coordinate_system=coordinate_system,
            labels_name=selected_label_option.labels_name,
        )
        selectable_image_options = image_discovery.eligible_image_options
        selected_image_option = next(
            (option for option in selectable_image_options if option.identity == preferred_selection.image_identity),
            None,
        )
        image_note_text = self._build_image_note_text(
            coordinate_system=coordinate_system,
            unavailable_image_count=image_discovery.unavailable_image_count,
        )

        return _FeatureExtractionTripletCardState(
            coordinate_system=coordinate_system,
            selectable_label_options=selectable_label_options,
            selected_label_option=selected_label_option,
            segmentation_note_text=segmentation_note_text,
            selectable_image_options=selectable_image_options,
            selected_image_option=selected_image_option,
            image_note_text=image_note_text,
        )

    def _apply_triplet_card_state(
        self,
        widgets: _FeatureExtractionTripletCardWidgets,
        state: _FeatureExtractionTripletCardState,
    ) -> None:
        """Apply a resolved state to one visible triplet-card widget."""
        with QSignalBlocker(widgets.segmentation_combo):
            widgets.segmentation_combo.clear()
            for option in state.selectable_label_options:
                widgets.segmentation_combo.addItem(option.display_name)

            has_label_options = bool(state.selectable_label_options)
            widgets.segmentation_combo.setEnabled(has_label_options)

            next_segmentation_index = self._find_label_option_index_in_options(
                state.selectable_label_options,
                None if state.selected_label_option is None else state.selected_label_option.identity,
            )
            widgets.segmentation_combo.setCurrentIndex(
                -1 if next_segmentation_index is None else next_segmentation_index
            )

        self._set_triplet_card_note_text(widgets.segmentation_note_label, state.segmentation_note_text)

        with QSignalBlocker(widgets.image_combo):
            widgets.image_combo.clear()
            widgets.image_combo.addItem(_NO_IMAGE_TEXT, None)
            for option in state.selectable_image_options:
                widgets.image_combo.addItem(option.display_name)

            widgets.image_combo.setEnabled(
                self.selected_spatialdata is not None
                and state.coordinate_system is not None
                and state.selected_label_option is not None
            )

            next_image_index = self._find_image_option_index_in_options(
                state.selectable_image_options,
                None if state.selected_image_option is None else state.selected_image_option.identity,
            )
            widgets.image_combo.setCurrentIndex(0 if next_image_index is None else next_image_index + 1)

        self._set_triplet_card_note_text(widgets.image_note_label, state.image_note_text)

    def _snapshot_rendered_card_selections(self) -> dict[str, _FeatureExtractionCardSelection]:
        snapshot: dict[str, _FeatureExtractionCardSelection] = {}
        for coordinate_system, state in self._triplet_card_states_by_coordinate_system.items():
            snapshot[coordinate_system] = _FeatureExtractionCardSelection(
                label_identity=None if state.selected_label_option is None else state.selected_label_option.identity,
                image_identity=None if state.selected_image_option is None else state.selected_image_option.identity,
            )

        return snapshot

    def _resolve_visible_label_selections(
        self,
        *,
        preferred_selection_by_coordinate_system: Mapping[str, _FeatureExtractionCardSelection],
        authoritative_coordinate_systems: set[str],
    ) -> tuple[dict[str, ElementIdentity | None], dict[ElementIdentity, str]]:
        """Resolve batch-visible label ownership across the checked cards.

        Feature-extraction batch selection is intentionally asymmetric:

        - labels are resolved *globally* across the checked cards, so one
          concrete segmentation element may belong to at most one visible card
          at a time;
        - images are resolved *locally* inside each card after the label is
          chosen, so the same valid image element may still be reused across
          several cards when it matches each card's selected segmentation.

        This helper implements only the first half of that rule. It walks the
        checked cards in a stable order, restores preferred label selections
        when they are still eligible, and builds a `label -> owner` map that
        prevents duplicate segmentation selection across the visible batch.
        """
        checked_coordinate_systems = self._checked_coordinate_system_names()
        authoritative_order = [
            coordinate_system
            for coordinate_system in checked_coordinate_systems
            if coordinate_system in authoritative_coordinate_systems
        ]
        restore_order = [
            coordinate_system
            for coordinate_system in checked_coordinate_systems
            if coordinate_system not in authoritative_coordinate_systems
        ]

        selected_label_identity_by_coordinate_system: dict[str, ElementIdentity | None] = {}
        selected_label_owner_by_identity: dict[ElementIdentity, str] = {}
        for coordinate_system in [*authoritative_order, *restore_order]:
            preferred_selection = preferred_selection_by_coordinate_system.get(
                coordinate_system,
                self._get_remembered_card_selection_for_coordinate_system(coordinate_system),
            )
            label_discovery = get_spatialdata_feature_extraction_label_discovery_for_coordinate_system_from_sdata(
                sdata=self.selected_spatialdata,
                coordinate_system=coordinate_system,
            )
            eligible_label_identities = {option.identity for option in label_discovery.eligible_label_options}
            selected_label_identity = preferred_selection.label_identity
            if (
                selected_label_identity is None
                or selected_label_identity not in eligible_label_identities
                or selected_label_identity in selected_label_owner_by_identity
            ):
                selected_label_identity = None

            selected_label_identity_by_coordinate_system[coordinate_system] = selected_label_identity
            if selected_label_identity is not None:
                selected_label_owner_by_identity[selected_label_identity] = coordinate_system

        return selected_label_identity_by_coordinate_system, selected_label_owner_by_identity

    def _recompute_visible_triplet_card_states(
        self,
        *,
        preferred_selection_by_coordinate_system: Mapping[str, _FeatureExtractionCardSelection] | None = None,
        authoritative_coordinate_systems: set[str] | None = None,
    ) -> None:
        checked_coordinate_systems = self._checked_coordinate_system_names()
        if preferred_selection_by_coordinate_system is None:
            preferred_selection_by_coordinate_system = {
                coordinate_system: self._get_remembered_card_selection_for_coordinate_system(coordinate_system)
                for coordinate_system in checked_coordinate_systems
            }
        if authoritative_coordinate_systems is None:
            authoritative_coordinate_systems = set(checked_coordinate_systems)

        authoritative_coordinate_systems = {
            coordinate_system
            for coordinate_system in authoritative_coordinate_systems
            if coordinate_system in checked_coordinate_systems
        }
        selected_label_identity_by_coordinate_system, selected_label_owner_by_identity = (
            self._resolve_visible_label_selections(
                preferred_selection_by_coordinate_system=preferred_selection_by_coordinate_system,
                authoritative_coordinate_systems=authoritative_coordinate_systems,
            )
        )
        next_states_by_coordinate_system: dict[str, _FeatureExtractionTripletCardState] = {}
        for coordinate_system in checked_coordinate_systems:
            state = self._recompute_triplet_card_state_for_coordinate_system(
                coordinate_system=coordinate_system,
                preferred_selection=preferred_selection_by_coordinate_system.get(
                    coordinate_system,
                    self._get_remembered_card_selection_for_coordinate_system(coordinate_system),
                ),
                selected_label_identity=selected_label_identity_by_coordinate_system.get(coordinate_system),
                selected_label_owner_by_identity=selected_label_owner_by_identity,
            )
            next_states_by_coordinate_system[coordinate_system] = state
            self._sync_remembered_card_selection_from_state(coordinate_system, state)
            widgets = self._triplet_card_widgets_by_coordinate_system.get(coordinate_system)
            if widgets is not None:
                self._apply_triplet_card_state(widgets, state)

        self._triplet_card_states_by_coordinate_system = next_states_by_coordinate_system
        self._sync_active_triplet_card_state()

    def _set_active_card_widgets(self, coordinate_system: str | None) -> None:
        if coordinate_system is None:
            self.segmentation_combo = self._triplet_card_widgets.segmentation_combo
            self.image_combo = self._triplet_card_widgets.image_combo
            return

        widgets = self._triplet_card_widgets_by_coordinate_system.get(coordinate_system)
        if widgets is None:
            self.segmentation_combo = self._triplet_card_widgets.segmentation_combo
            self.image_combo = self._triplet_card_widgets.image_combo
            return

        self.segmentation_combo = widgets.segmentation_combo
        self.image_combo = widgets.image_combo

    def _set_selected_coordinate_system_by_name(self, coordinate_system: str | None) -> None:
        self._selected_coordinate_system = coordinate_system
        self._set_active_card_widgets(coordinate_system)
        with QSignalBlocker(self.coordinate_system_combo):
            if coordinate_system is None:
                self.coordinate_system_combo.setCurrentIndex(-1)
            else:
                self.coordinate_system_combo.setCurrentIndex(self.coordinate_system_combo.findData(coordinate_system))

    def _sync_coordinate_system_combo_items(self) -> None:
        with QSignalBlocker(self.coordinate_system_combo):
            self.coordinate_system_combo.clear()
            for coordinate_system in self._coordinate_systems:
                self.coordinate_system_combo.addItem(coordinate_system, coordinate_system)
            self.coordinate_system_combo.setEnabled(bool(self._coordinate_systems))
            if self.selected_coordinate_system is None:
                self.coordinate_system_combo.setCurrentIndex(-1)
            else:
                self.coordinate_system_combo.setCurrentIndex(
                    self.coordinate_system_combo.findData(self.selected_coordinate_system)
                )

    def _sync_coordinate_system_checkboxes(self) -> None:
        self._clear_coordinate_system_checkboxes()
        checked_coordinate_systems = set(self._checked_coordinate_systems)
        for coordinate_system in self._coordinate_systems:
            visible_coordinate_system, coordinate_system_shortened = format_feedback_identifier(
                coordinate_system,
                max_length=_COORDINATE_SYSTEM_CHECKBOX_MAX_LENGTH,
            )
            checkbox = QCheckBox(visible_coordinate_system)
            checkbox.setObjectName(f"feature_extraction_coordinate_system_checkbox_{coordinate_system}")
            checkbox.setStyleSheet(_FEATURE_CHECKBOX_STYLESHEET)
            if coordinate_system_shortened:
                self._set_tooltip(checkbox, coordinate_system)
            checkbox.setChecked(coordinate_system in checked_coordinate_systems)
            checkbox.toggled.connect(
                lambda checked, current_coordinate_system=coordinate_system: (
                    self._on_coordinate_system_checkbox_toggled(current_coordinate_system, checked)
                )
            )
            self.coordinate_system_selection_layout.addWidget(checkbox)
            self._coordinate_system_checkboxes[coordinate_system] = checkbox

    def _checked_coordinate_system_names(self) -> list[str]:
        checked_coordinate_systems = set(self._checked_coordinate_systems)
        return [
            coordinate_system
            for coordinate_system in self._coordinate_systems
            if coordinate_system in checked_coordinate_systems
        ]

    def _sync_active_triplet_card_state(self) -> None:
        self._set_active_card_widgets(self.selected_coordinate_system)
        state = (
            None
            if self.selected_coordinate_system is None
            else self._triplet_card_states_by_coordinate_system.get(self.selected_coordinate_system)
        )
        if state is None:
            self._triplet_card_state = _FeatureExtractionTripletCardState(
                coordinate_system=self.selected_coordinate_system,
                selectable_label_options=[],
                selected_label_option=None,
                segmentation_note_text=None,
                selectable_image_options=[],
                selected_image_option=None,
                image_note_text=None,
            )
            self._label_options = []
            self._selected_label_option = None
            self._image_options = []
            self._selected_image_option = None
            return

        self._triplet_card_state = state
        self._label_options = state.selectable_label_options
        self._selected_label_option = state.selected_label_option
        self._image_options = state.selectable_image_options
        self._selected_image_option = state.selected_image_option

    def _rebuild_visible_triplet_cards(self) -> None:
        selected_coordinate_systems = self._checked_coordinate_system_names()
        previous_rendered_card_selections = self._snapshot_rendered_card_selections()
        self._clear_triplet_cards()
        for coordinate_system in selected_coordinate_systems:
            widgets = self._create_triplet_card_widgets(coordinate_system)
            self._triplet_card_widgets_by_coordinate_system[coordinate_system] = widgets
            self.triplet_cards_layout.addWidget(widgets.container)

        preferred_selection_by_coordinate_system = {
            coordinate_system: previous_rendered_card_selections.get(
                coordinate_system,
                self._get_remembered_card_selection_for_coordinate_system(coordinate_system),
            )
            for coordinate_system in selected_coordinate_systems
        }
        # Cards that were already rendered keep authority over their staged selections;
        # newly re-added cards only attempt to restore remembered selections around them.
        self._recompute_visible_triplet_card_states(
            preferred_selection_by_coordinate_system=preferred_selection_by_coordinate_system,
            authoritative_coordinate_systems=set(previous_rendered_card_selections),
        )

        if self.selected_coordinate_system not in selected_coordinate_systems:
            next_active_coordinate_system = selected_coordinate_systems[0] if selected_coordinate_systems else None
            self._set_selected_coordinate_system_by_name(next_active_coordinate_system)

        self._sync_active_triplet_card_state()

    def _on_triplet_card_segmentation_changed(self, coordinate_system: str, index: int) -> None:
        self._set_selected_coordinate_system_by_name(coordinate_system)
        card_state = self._triplet_card_states_by_coordinate_system.get(coordinate_system)
        preferred_selection_by_coordinate_system = self._snapshot_rendered_card_selections()
        previous_selection = preferred_selection_by_coordinate_system.get(
            coordinate_system,
            _FeatureExtractionCardSelection(label_identity=None, image_identity=None),
        )
        selected_label_option = None
        if card_state is not None and 0 <= index < len(card_state.selectable_label_options):
            selected_label_option = card_state.selectable_label_options[index]

        preferred_selection_by_coordinate_system[coordinate_system] = _FeatureExtractionCardSelection(
            label_identity=None if selected_label_option is None else selected_label_option.identity,
            image_identity=previous_selection.image_identity,
        )
        self._remembered_card_selection_by_coordinate_system[coordinate_system] = (
            preferred_selection_by_coordinate_system[coordinate_system]
        )
        # During a live edit, every currently visible card reflects active staged state,
        # so the full checked set is authoritative for duplicate-resolution.
        self._recompute_visible_triplet_card_states(
            preferred_selection_by_coordinate_system=preferred_selection_by_coordinate_system,
            authoritative_coordinate_systems=set(self._checked_coordinate_system_names()),
        )
        self._refresh_table_names()
        self._refresh_batch_channel_options()
        self._update_intensity_features_hint()
        self._bind_current_selection()

    def _on_segmentation_changed(self, index: int) -> None:
        if self.selected_coordinate_system is not None:
            self._on_triplet_card_segmentation_changed(self.selected_coordinate_system, index)

    def _find_label_option_index_in_options(
        self,
        label_options: list[SpatialDataLabelsOption],
        identity: ElementIdentity | None,
    ) -> int | None:
        if identity is None:
            return None

        for index, option in enumerate(label_options):
            if option.identity == identity:
                return index

        return None

    def _find_label_option_index(self, identity: ElementIdentity | None) -> int | None:
        return self._find_label_option_index_in_options(self._label_options, identity)

    def _on_triplet_card_image_changed(self, coordinate_system: str, index: int) -> None:
        self._set_selected_coordinate_system_by_name(coordinate_system)
        card_state = self._triplet_card_states_by_coordinate_system.get(coordinate_system)
        if card_state is None:
            return

        selected_image_option = None
        if 0 < index <= len(card_state.selectable_image_options):
            selected_image_option = card_state.selectable_image_options[index - 1]

        self._triplet_card_states_by_coordinate_system[coordinate_system] = replace(
            card_state,
            selected_image_option=selected_image_option,
        )
        self._remembered_card_selection_by_coordinate_system[coordinate_system] = _FeatureExtractionCardSelection(
            label_identity=(
                None if card_state.selected_label_option is None else card_state.selected_label_option.identity
            ),
            image_identity=None if selected_image_option is None else selected_image_option.identity,
        )
        self._sync_active_triplet_card_state()
        self._refresh_batch_channel_options()
        self._update_intensity_features_hint()
        self._bind_current_selection()

    def _on_image_changed(self, index: int) -> None:
        if self.selected_coordinate_system is not None:
            self._on_triplet_card_image_changed(self.selected_coordinate_system, index)

    def _set_selected_image_option(self, index: int) -> None:
        if index <= 0 or index > len(self._image_options):
            self._selected_image_option = None
        else:
            self._selected_image_option = self._image_options[index - 1]

    def _find_image_option_index_in_options(
        self,
        image_options: list[SpatialDataImageOption],
        identity: ElementIdentity | None,
    ) -> int | None:
        if identity is None:
            return None

        for index, option in enumerate(image_options):
            if option.identity == identity:
                return index

        return None

    def _find_image_option_index(self, identity: ElementIdentity | None) -> int | None:
        return self._find_image_option_index_in_options(self._image_options, identity)

    def _clear_batch_channel_options(self) -> None:
        self._batch_channel_names = []
        self._batch_channel_checkboxes = []

        while self.channel_selection_list_layout.count():
            item = self.channel_selection_list_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        self.channel_selection_note_label.clear()
        self.channel_selection_note_label.hide()
        self.channel_selection_label.hide()
        self.channel_selection_container.hide()
        self.channel_selection_scroll_area.setMaximumHeight(0)

    def _selected_visible_image_options(self) -> list[tuple[str, SpatialDataImageOption]]:
        selected_images: list[tuple[str, SpatialDataImageOption]] = []
        for coordinate_system in self._checked_coordinate_system_names():
            state = self._triplet_card_states_by_coordinate_system.get(coordinate_system)
            if state is None or state.selected_image_option is None:
                continue
            selected_images.append((coordinate_system, state.selected_image_option))
        return selected_images

    def _build_batch_channel_error_text(
        self,
        *,
        has_duplicate_channel_names: bool,
        has_schema_mismatch: bool,
    ) -> str | None:
        fragments: list[str] = []
        if has_schema_mismatch:
            fragments.append("Channel names of selected images do not match.")
        if has_duplicate_channel_names:
            fragments.append(
                "One or more selected images expose duplicate channel names. "
                'Rename channels with "sdata.set_channel_names(...)" or choose a different image.'
            )

        if not fragments:
            return None
        return " ".join(fragment if fragment.endswith(".") else f"{fragment}." for fragment in fragments)

    def _resolve_batch_channel_state(self) -> _FeatureExtractionBatchChannelState:
        if self.selected_spatialdata is None:
            return _FeatureExtractionBatchChannelState(
                reference_coordinate_system=None,
                reference_image_option=None,
                channel_names=(),
                incompatible_coordinate_systems=(),
                incompatible_image_names=(),
                error_text=None,
            )

        selected_images = self._selected_visible_image_options()
        if not selected_images:
            return _FeatureExtractionBatchChannelState(
                reference_coordinate_system=None,
                reference_image_option=None,
                channel_names=(),
                incompatible_coordinate_systems=(),
                incompatible_image_names=(),
                error_text=None,
            )

        reference_coordinate_system: str | None = None
        reference_image_option: SpatialDataImageOption | None = None
        reference_channel_names: tuple[str, ...] = ()
        incompatible_coordinate_systems: list[str] = []
        incompatible_image_names: list[str] = []
        has_duplicate_channel_names = False
        has_schema_mismatch = False

        for coordinate_system, image_option in selected_images:
            try:
                channel_names = tuple(
                    get_image_channel_names_from_sdata(self.selected_spatialdata, image_option.image_name)
                )
            except ValueError:
                incompatible_coordinate_systems.append(coordinate_system)
                incompatible_image_names.append(image_option.image_name)
                has_duplicate_channel_names = True
                continue

            if not channel_names:
                raise ValueError(
                    f'Image "{image_option.image_name}" in "{coordinate_system}" does not expose channel names, '
                    "but feature extraction expects images with an explicit channel axis."
                )

            if reference_image_option is None:
                reference_coordinate_system = coordinate_system
                reference_image_option = image_option
                reference_channel_names = channel_names
                continue

            if channel_names != reference_channel_names:
                incompatible_coordinate_systems.append(coordinate_system)
                incompatible_image_names.append(image_option.image_name)
                has_schema_mismatch = True

        error_text = self._build_batch_channel_error_text(
            has_duplicate_channel_names=has_duplicate_channel_names,
            has_schema_mismatch=has_schema_mismatch,
        )
        return _FeatureExtractionBatchChannelState(
            reference_coordinate_system=reference_coordinate_system,
            reference_image_option=reference_image_option,
            channel_names=reference_channel_names,
            incompatible_coordinate_systems=tuple(incompatible_coordinate_systems),
            incompatible_image_names=tuple(incompatible_image_names),
            error_text=error_text,
        )

    def _refresh_batch_channel_options(self) -> None:
        self._clear_batch_channel_options()
        self._batch_channel_state = self._resolve_batch_channel_state()
        self._batch_channel_error = self._batch_channel_state.error_text

        channel_names = self._batch_channel_state.channel_names
        if not channel_names:
            return

        current_schema = channel_names
        selected_channel_names = self._channel_selection_memory.resolve_for_schema(current_schema)

        self._batch_channel_names = list(current_schema)
        self._channel_selection_memory.remember_for_schema(current_schema, selected_channel_names)
        selected_channel_name_set = set(selected_channel_names)

        channel_rows: list[QWidget] = []
        for channel_name in current_schema:
            checkbox = QCheckBox(channel_name)
            checkbox.setObjectName(f"feature_extraction_channel_checkbox_{channel_name}")
            checkbox.setStyleSheet(_FEATURE_CHECKBOX_STYLESHEET)
            checkbox.setChecked(channel_name in selected_channel_name_set)
            checkbox.toggled.connect(self._on_channel_selection_changed)
            self.channel_selection_list_layout.addWidget(checkbox)
            self._batch_channel_checkboxes.append(checkbox)
            channel_rows.append(checkbox)

        self._set_channel_selection_scroll_height(channel_rows)
        self.channel_selection_note_label.setText(self._batch_channel_error or "")
        self.channel_selection_note_label.setVisible(bool(self._batch_channel_error))
        self.channel_selection_label.show()
        self.channel_selection_container.show()

    def _on_channel_selection_changed(self, _checked: bool) -> None:
        self._store_selected_batch_channel_names()
        self._update_intensity_features_hint()
        self._bind_current_selection()

    def _store_selected_batch_channel_names(self) -> None:
        if not self._batch_channel_names:
            return

        current_schema = tuple(self._batch_channel_names)
        selected_channel_names = tuple(
            checkbox.text() for checkbox in self._batch_channel_checkboxes if checkbox.isChecked()
        )
        self._channel_selection_memory.remember_for_schema(current_schema, selected_channel_names)

    def _set_channel_selection_scroll_height(self, channel_rows: list[QWidget]) -> None:
        visible_rows = channel_rows[:_MAX_VISIBLE_EXTRACTION_CHANNELS]
        if not visible_rows:
            self.channel_selection_scroll_area.setMaximumHeight(0)
            return

        visible_height = sum(row.sizeHint().height() for row in visible_rows)
        visible_height += self.channel_selection_list_layout.spacing() * max(0, len(visible_rows) - 1)
        margins = self.channel_selection_list_layout.contentsMargins()
        visible_height += margins.top() + margins.bottom()
        visible_height += self.channel_selection_scroll_area.frameWidth() * 2
        self.channel_selection_scroll_area.setMaximumHeight(visible_height)

    def _resolve_staged_batch_state(self) -> _FeatureExtractionStagedBatchState:
        checked_coordinate_systems = tuple(self._checked_coordinate_system_names())
        if self.selected_spatialdata is None or not checked_coordinate_systems:
            return _FeatureExtractionStagedBatchState(
                checked_coordinate_systems=checked_coordinate_systems,
                labels_names=(),
                triplets=(),
                invalid_coordinate_systems=(),
                error_text=None,
            )

        requires_image = self._has_intensity_features_selected()
        incompatible_coordinate_systems = (
            set(self._batch_channel_state.incompatible_coordinate_systems)
            if requires_image and self._batch_channel_error is not None
            else set()
        )
        labels_names: list[str] = []
        seen_labels_names: set[str] = set()
        candidate_triplets: list[FeatureExtractionTriplet] = []
        invalid_coordinate_systems: list[str] = []
        has_missing_segmentation = False
        has_missing_required_image = False
        has_empty_channel_selection = requires_image and self._has_empty_intensity_channel_selection()

        for coordinate_system in checked_coordinate_systems:
            state = self._triplet_card_states_by_coordinate_system.get(coordinate_system)
            if state is None or state.selected_label_option is None:
                has_missing_segmentation = True
                invalid_coordinate_systems.append(coordinate_system)
                continue

            labels_name = state.selected_label_option.labels_name
            if labels_name not in seen_labels_names:
                labels_names.append(labels_name)
                seen_labels_names.add(labels_name)

            image_name = None if state.selected_image_option is None else state.selected_image_option.image_name
            if requires_image and image_name is None:
                has_missing_required_image = True
                invalid_coordinate_systems.append(coordinate_system)
                continue

            if requires_image and coordinate_system in incompatible_coordinate_systems:
                invalid_coordinate_systems.append(coordinate_system)
                continue

            if has_empty_channel_selection:
                invalid_coordinate_systems.append(coordinate_system)
                continue

            candidate_triplets.append(
                FeatureExtractionTriplet(
                    coordinate_system=coordinate_system,
                    labels_name=labels_name,
                    image_name=image_name,
                    channels=self.selected_extraction_channel_names if requires_image else None,
                )
            )

        error_text: str | None = None
        if has_missing_segmentation:
            error_text = "Choose a labels element available in every checked coordinate system."
        elif has_missing_required_image:
            error_text = "Choose an image for every extraction target before calculating intensity features."
        elif requires_image and self._batch_channel_error is not None:
            error_text = self._batch_channel_error
        elif has_empty_channel_selection:
            error_text = "Select at least one channel before calculating intensity features."

        return _FeatureExtractionStagedBatchState(
            checked_coordinate_systems=checked_coordinate_systems,
            labels_names=tuple(labels_names),
            triplets=tuple(candidate_triplets) if error_text is None else (),
            invalid_coordinate_systems=tuple(invalid_coordinate_systems),
            error_text=error_text,
        )

    def _eligible_table_names_for_label_batch(self, labels_names: tuple[str, ...]) -> list[str]:
        if self.selected_spatialdata is None or not labels_names:
            return []

        table_names_by_label = [
            set(get_annotating_table_names(self.selected_spatialdata, labels_name)) for labels_name in labels_names
        ]
        if not table_names_by_label:
            return []

        return sorted(set.intersection(*table_names_by_label))

    def _can_choose_table_target(self, staged_batch_state: _FeatureExtractionStagedBatchState) -> bool:
        return (
            self.selected_spatialdata is not None
            and bool(staged_batch_state.checked_coordinate_systems)
            and bool(staged_batch_state.labels_names)
            and len(staged_batch_state.labels_names) == len(staged_batch_state.checked_coordinate_systems)
        )

    def _refresh_table_names(self, *, preferred_existing_table_name: str | None = None) -> None:
        previous_table_name = self.selected_table_name
        previous_create_table = self._create_table
        staged_batch_state = self._resolve_staged_batch_state()
        can_choose_table_target = self._can_choose_table_target(staged_batch_state)

        if not can_choose_table_target:
            self._eligible_existing_table_names = []
        else:
            self._eligible_existing_table_names = self._eligible_table_names_for_label_batch(
                staged_batch_state.labels_names
            )

        with QSignalBlocker(self.table_combo):
            self.table_combo.clear()
            for table_name in self._eligible_existing_table_names:
                self.table_combo.addItem(
                    table_name,
                    _FeatureExtractionTableComboData.existing(table_name),
                )

            if can_choose_table_target:
                self.table_combo.addItem(
                    _CREATE_TABLE_OPTION_TEXT,
                    _FeatureExtractionTableComboData.create_table(),
                )

            has_table_targets = self.table_combo.count() > 0
            self.table_combo.setEnabled(has_table_targets)

            next_index = -1
            if preferred_existing_table_name is not None:
                next_index = self._find_table_combo_target_index(
                    _FeatureExtractionTableComboData.existing(preferred_existing_table_name)
                )
            if next_index < 0:
                if previous_create_table:
                    next_index = self._find_table_combo_target_index(_FeatureExtractionTableComboData.create_table())
                elif previous_table_name is not None:
                    next_index = self._find_table_combo_target_index(
                        _FeatureExtractionTableComboData.existing(previous_table_name)
                    )

            if next_index < 0 and self._eligible_existing_table_names:
                next_index = self._find_table_combo_target_index(
                    _FeatureExtractionTableComboData.existing(self._eligible_existing_table_names[0])
                )
            if next_index < 0 and can_choose_table_target:
                next_index = self._find_table_combo_target_index(_FeatureExtractionTableComboData.create_table())

            self.table_combo.setCurrentIndex(next_index)

        self._set_selected_table_target_from_combo(self.table_combo.currentIndex())

    def _on_table_changed(self, index: int) -> None:
        self._set_selected_table_target_from_combo(index)
        self._bind_current_selection()

    def _find_table_combo_target_index(self, target_data: _FeatureExtractionTableComboData) -> int:
        for index in range(self.table_combo.count()):
            if self.table_combo.itemData(index) == target_data:
                return index
        return -1

    def _set_selected_table_target_from_combo(self, index: int) -> None:
        item_data = self.table_combo.itemData(index) if 0 <= index < self.table_combo.count() else None
        if not isinstance(item_data, _FeatureExtractionTableComboData):
            self._create_table = False
            self._selected_table_name = None
            self._set_create_table_name_controls_visible(False)
            return

        if item_data.mode is _FeatureExtractionTableComboMode.CREATE_TABLE:
            self._create_table = True
            self._selected_table_name = None
            self._ensure_new_table_name_suggestion()
            self._set_create_table_name_controls_visible(True)
            return

        self._create_table = False
        self._selected_table_name = item_data.table_name
        self._set_create_table_name_controls_visible(False)

    def _suggest_new_table_name(self) -> str:
        if self.selected_spatialdata is None:
            return _DEFAULT_NEW_TABLE_NAME
        if not spatialdata_element_name_exists(self.selected_spatialdata, _DEFAULT_NEW_TABLE_NAME):
            return _DEFAULT_NEW_TABLE_NAME
        return f"{_DEFAULT_NEW_TABLE_NAME}_{uuid.uuid4()}"

    def _ensure_new_table_name_suggestion(self) -> None:
        current_name = (
            self._new_table_name if self._new_table_name is not None else self.new_table_name_line_edit.text()
        )
        if current_name.strip():
            if self.new_table_name_line_edit.text() != current_name:
                with QSignalBlocker(self.new_table_name_line_edit):
                    self.new_table_name_line_edit.setText(current_name)
            return

        suggested_name = self._suggest_new_table_name()
        self._new_table_name = suggested_name
        with QSignalBlocker(self.new_table_name_line_edit):
            self.new_table_name_line_edit.setText(suggested_name)

    def _set_create_table_name_controls_visible(self, is_visible: bool) -> None:
        self.new_table_name_label.setVisible(is_visible)
        self.new_table_name_line_edit.setVisible(is_visible)

    def _on_new_table_name_changed(self, text: str) -> None:
        self._new_table_name = text
        self._bind_current_selection()

    def _refresh_coordinate_systems(self) -> None:
        if self.selected_spatialdata is None:
            self._coordinate_systems = []
        else:
            self._coordinate_systems = get_coordinate_system_names_from_sdata(self.selected_spatialdata)

        checked_coordinate_systems = {
            coordinate_system
            for coordinate_system in self._checked_coordinate_systems
            if coordinate_system in self._coordinate_systems
        }
        self._checked_coordinate_systems = [
            coordinate_system
            for coordinate_system in self._coordinate_systems
            if coordinate_system in checked_coordinate_systems
        ]

        if self.selected_coordinate_system not in checked_coordinate_systems:
            next_active_coordinate_system = (
                self._checked_coordinate_systems[0] if self._checked_coordinate_systems else None
            )
            self._set_selected_coordinate_system_by_name(next_active_coordinate_system)

        self._sync_coordinate_system_combo_items()
        self._sync_coordinate_system_checkboxes()
        self._rebuild_visible_triplet_cards()

    def _on_coordinate_system_changed(self, index: int) -> None:
        coordinate_system = self.coordinate_system_combo.itemData(index)
        next_coordinate_system = coordinate_system if isinstance(coordinate_system, str) else None
        if next_coordinate_system is not None and next_coordinate_system not in self._checked_coordinate_systems:
            self._checked_coordinate_systems.append(next_coordinate_system)
            self._checked_coordinate_systems = self._checked_coordinate_system_names()
            checkbox = self._coordinate_system_checkboxes.get(next_coordinate_system)
            if checkbox is not None:
                with QSignalBlocker(checkbox):
                    checkbox.setChecked(True)
            self._rebuild_visible_triplet_cards()

        self._set_selected_coordinate_system_by_name(next_coordinate_system)
        self._sync_active_triplet_card_state()
        self._refresh_table_names()
        self._refresh_batch_channel_options()
        self._update_intensity_features_hint()
        self._bind_current_selection()

    def _set_selected_coordinate_system(self, index: int) -> None:
        coordinate_system = self.coordinate_system_combo.itemData(index)
        self._selected_coordinate_system = coordinate_system if isinstance(coordinate_system, str) else None

    def _on_coordinate_system_checkbox_toggled(self, coordinate_system: str, checked: bool) -> None:
        checked_coordinate_systems = set(self._checked_coordinate_systems)
        if checked:
            checked_coordinate_systems.add(coordinate_system)
        else:
            checked_coordinate_systems.discard(coordinate_system)
        self._checked_coordinate_systems = [
            current_coordinate_system
            for current_coordinate_system in self._coordinate_systems
            if current_coordinate_system in checked_coordinate_systems
        ]

        next_active_coordinate_system = self.selected_coordinate_system
        if checked:
            next_active_coordinate_system = coordinate_system
        elif next_active_coordinate_system == coordinate_system:
            next_active_coordinate_system = (
                self._checked_coordinate_systems[0] if self._checked_coordinate_systems else None
            )

        self._set_selected_coordinate_system_by_name(next_active_coordinate_system)
        self._rebuild_visible_triplet_cards()
        self._refresh_table_names()
        self._refresh_batch_channel_options()
        self._update_intensity_features_hint()
        self._bind_current_selection()

    def _on_feature_selection_changed(self, _checked: bool) -> None:
        self._update_intensity_features_hint()
        self._bind_current_selection()

    def _on_feature_key_changed(self, _text: str) -> None:
        self._bind_current_selection()

    def _calculate_feature_matrix(self) -> None:
        controller = self._feature_extraction_controller
        binding = controller.binding_state
        sdata, table_name, feature_key = binding.sdata, binding.table_name, binding.feature_key
        if binding.create_table or not controller.can_calculate or None in (sdata, table_name, feature_key):
            # A new table has no existing entry to check. Otherwise the
            # controller cannot launch, and `calculate()` reports why.
            controller.calculate()
            return

        # Take the table's dirty state once, at launch. It is used twice:
        # - its paths go to `check_existing_feature_matrix(...)` below, which
        #   refuses to launch (backed SpatialData) while the key's paths have
        #   unsaved changes;
        # - the snapshot itself goes to `calculate(...)` and comes back in the
        #   job's result, which `_on_controller_feature_matrix_changed(...)`
        #   records with `record_persisted_table_change(...)`.
        dirty_snapshot = self._app_state.snapshot_table_dirty_state(sdata, table_name)
        # The first of two checks of the existing entry: it picks the dialog,
        # and the worker checks again against the confirmed kind
        # (see docs/development/feature_extraction.md).
        existing = controller.check_existing_feature_matrix(dirty_paths=dirty_snapshot.paths)
        if existing is None:
            return  # Blocked, or the check failed; the controller's status says why.

        match existing.kind:
            case "absent":
                confirmed = True
            case "compatible":
                confirmed = self._confirm_regional_update(
                    feature_key,
                    table_name,
                    tuple(triplet.labels_name for triplet in binding.triplets),
                )
            case "incompatible":
                confirmed = self._confirm_full_replacement(
                    feature_key,
                    existing.reason or "",
                    existing.unselected_regions,
                )
        if not confirmed:
            return

        controller.calculate(
            overwrite_feature_key=existing.kind != "absent",
            confirmed_existing_entry=existing.kind,
            dirty_snapshot=dirty_snapshot,
        )

    def _confirm_regional_update(self, feature_key: str, table_name: str, labels_names: Sequence[str]) -> bool:
        """Ask to replace the values of the selected labels elements in a compatible matrix."""
        return self._confirm_feature_matrix_replacement(
            title="Existing Feature Matrix",
            lines=[
                f'Feature matrix "{feature_key}" already exists in table "{table_name}".',
                f"Replace its values for the selected labels element(s) {_quoted_names(labels_names)}? "
                "Values for other labels elements are kept.",
            ],
            button_text="Replace selected regions",
        )

    def _confirm_full_replacement(self, feature_key: str, reason: str, lost_labels_names: Sequence[str]) -> bool:
        """Ask to remove an incompatible matrix and calculate it again."""
        lines = [f'Feature matrix "{feature_key}" exists but cannot be updated by region, because {reason}.']
        if lost_labels_names:
            lines.append(f"Replacing it removes the values for labels element(s) {_quoted_names(lost_labels_names)}.")
        return self._confirm_feature_matrix_replacement(
            title="Incompatible Feature Matrix",
            lines=lines,
            button_text="Replace entire feature matrix",
        )

    def _confirm_feature_matrix_replacement(self, *, title: str, lines: Sequence[str], button_text: str) -> bool:
        dialog = QDialog(self)
        dialog.setWindowTitle("Feature Matrix Already Exists")
        dialog.setModal(True)
        dialog.setMinimumWidth(560)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        warning_card = QLabel()
        warning_card.setWordWrap(True)
        set_status_card(warning_card, title=title, lines=list(lines), kind="warning")
        layout.addWidget(warning_card)

        button_row = QHBoxLayout()
        button_row.setSpacing(10)
        button_row.addStretch(1)

        replace_button = QPushButton(button_text)
        cancel_button = QPushButton("Cancel")

        replace_button.setStyleSheet(PRIMARY_BUTTON_STYLESHEET)
        cancel_button.setStyleSheet(SECONDARY_BUTTON_STYLESHEET)

        button_row.addWidget(replace_button)
        button_row.addWidget(cancel_button)
        layout.addLayout(button_row)

        replace_button.clicked.connect(dialog.accept)
        cancel_button.clicked.connect(dialog.reject)
        cancel_button.setDefault(True)

        return dialog.exec() == QDialog.DialogCode.Accepted

    def _has_intensity_features_selected(self) -> bool:
        return any(self._feature_checkboxes[name].isChecked() for name in _INTENSITY_FEATURES)

    def _has_selected_segmentation_without_image(self) -> bool:
        for coordinate_system in self._checked_coordinate_system_names():
            state = self._triplet_card_states_by_coordinate_system.get(coordinate_system)
            if state is None or state.selected_label_option is None:
                continue
            if state.selected_image_option is None:
                return True
        return False

    def _has_empty_intensity_channel_selection(self) -> bool:
        return bool(self._batch_channel_names) and self.selected_extraction_channel_names == ()

    def _update_intensity_features_hint(self) -> None:
        if not self._has_intensity_features_selected():
            self.intensity_features_hint.setText("")
            self.intensity_features_hint.setStyleSheet(_FEATURE_HINT_INFO_STYLESHEET)
            self.intensity_features_hint.setVisible(False)
            return

        if self._has_selected_segmentation_without_image():
            self.intensity_features_hint.setText(
                "Intensity features are selected, so choose an image for every extraction target before calculating."
            )
            self.intensity_features_hint.setStyleSheet(_FEATURE_HINT_WARNING_STYLESHEET)
            self.intensity_features_hint.setVisible(True)
            return

        if self._batch_channel_error is not None:
            self.intensity_features_hint.setText(self._batch_channel_error)
            self.intensity_features_hint.setStyleSheet(_FEATURE_HINT_WARNING_STYLESHEET)
            self.intensity_features_hint.setVisible(True)
            return

        if not self._selected_visible_image_options():
            self.intensity_features_hint.setText(
                "Intensity features are selected, so choose an image before calculating."
            )
            self.intensity_features_hint.setStyleSheet(_FEATURE_HINT_WARNING_STYLESHEET)
            self.intensity_features_hint.setVisible(True)
            return

        if self._has_empty_intensity_channel_selection():
            self.intensity_features_hint.setText(
                "Intensity features are selected, so select at least one channel before calculating."
            )
            self.intensity_features_hint.setStyleSheet(_FEATURE_HINT_WARNING_STYLESHEET)
            self.intensity_features_hint.setVisible(True)
            return

        self.intensity_features_hint.setText("")
        self.intensity_features_hint.setStyleSheet(_FEATURE_HINT_INFO_STYLESHEET)
        self.intensity_features_hint.setVisible(False)

    def _validate_selected_table_binding(
        self,
        staged_batch_state: _FeatureExtractionStagedBatchState,
    ) -> str | None:
        if (
            self._create_table
            or self.selected_spatialdata is None
            or self.selected_table_name is None
            or not staged_batch_state.labels_names
            or len(staged_batch_state.labels_names) != len(staged_batch_state.checked_coordinate_systems)
        ):
            return None

        try:
            validate_table_annotation_coverage(
                self.selected_spatialdata,
                self.selected_table_name,
                staged_batch_state.labels_names,
            )
            validate_table_region_instance_ids(
                self.selected_spatialdata,
                self.selected_table_name,
                labels_names=staged_batch_state.labels_names,
            )
        except ValueError as error:
            return str(error)

        return None

    def _validate_create_table_target(self) -> str | None:
        if not self._create_table or self.selected_spatialdata is None:
            return None

        table_name = self.selected_new_table_name
        if table_name is None:
            return "Enter a new table name."

        try:
            normalize_spatialdata_name(table_name, "table name")
        except ValueError as error:
            return f"Choose a valid table name. {error}"

        if spatialdata_element_name_exists(self.selected_spatialdata, table_name):
            return f'Table "{table_name}" already exists. Choose a different table name.'

        return None

    def _resolve_table_target(
        self,
        staged_batch_state: _FeatureExtractionStagedBatchState,
    ) -> _FeatureExtractionTableTargetState:
        if self._create_table:
            return _FeatureExtractionTableTargetState(
                table_name=self.selected_new_table_name,
                create_table=True,
                error_text=self._validate_create_table_target(),
            )

        return _FeatureExtractionTableTargetState(
            table_name=self.selected_table_name,
            create_table=False,
            error_text=self._validate_selected_table_binding(staged_batch_state),
        )

    def _bind_current_selection(self) -> None:
        self._staged_batch_state = self._resolve_staged_batch_state()
        self._table_target_state = self._resolve_table_target(self._staged_batch_state)

        triplets: tuple[FeatureExtractionTriplet, ...] = ()
        if self._staged_batch_state.is_bindable and self._table_target_state.is_ready:
            triplets = self._staged_batch_state.triplets

        self._feature_extraction_controller.bind_batch(
            self.selected_spatialdata,
            triplets,
            self._table_target_state.bound_table_name,
            self.selected_feature_names,
            self.selected_feature_key,
            create_table=self._table_target_state.create_table,
        )
        self._update_selection_status()

    def _update_selection_status(self) -> None:
        self._update_primary_status_card()
        self._update_feature_extraction_feedback()
        self._update_calculate_controls()

    def _build_selection_status_entries(self):
        checked_coordinate_systems = self._staged_batch_state.checked_coordinate_systems
        labels_names_by_coordinate_system: dict[str, str | None] = {}
        image_names_by_coordinate_system: dict[str, str | None] = {}
        blocking_reasons_by_coordinate_system: dict[str, str | None] = {}
        channel_blocking_reason = self._selection_status_channel_blocking_reason()
        incompatible_coordinate_systems = set(self._batch_channel_state.incompatible_coordinate_systems)
        requires_image = self._has_intensity_features_selected()
        has_empty_channel_selection = requires_image and self._has_empty_intensity_channel_selection()

        for coordinate_system in checked_coordinate_systems:
            state = self._triplet_card_states_by_coordinate_system.get(coordinate_system)
            selected_label_option = None if state is None else state.selected_label_option
            selected_image_option = None if state is None else state.selected_image_option
            labels_names_by_coordinate_system[coordinate_system] = (
                None if selected_label_option is None else selected_label_option.labels_name
            )
            image_names_by_coordinate_system[coordinate_system] = (
                None if selected_image_option is None else selected_image_option.image_name
            )

            blocking_reason: str | None = None
            if selected_label_option is None:
                blocking_reason = "choose a labels element"
            elif requires_image and selected_image_option is None:
                blocking_reason = "choose an image"
            elif requires_image and coordinate_system in incompatible_coordinate_systems:
                blocking_reason = channel_blocking_reason
            elif has_empty_channel_selection:
                blocking_reason = "select at least one channel"

            blocking_reasons_by_coordinate_system[coordinate_system] = blocking_reason

        return build_feature_extraction_status_card_entries(
            checked_coordinate_systems,
            labels_names_by_coordinate_system=labels_names_by_coordinate_system,
            image_names_by_coordinate_system=image_names_by_coordinate_system,
            blocking_reasons_by_coordinate_system=blocking_reasons_by_coordinate_system,
        )

    def _selection_status_channel_blocking_reason(self) -> str | None:
        if self._batch_channel_error is None:
            return None

        error_text = self._batch_channel_error.lower()
        has_schema_mismatch = "do not match" in error_text
        has_duplicate_channel_names = "duplicate channel names" in error_text

        if has_schema_mismatch and has_duplicate_channel_names:
            return "incompatible channel names"
        if has_duplicate_channel_names:
            return "duplicate channel names"
        if has_schema_mismatch:
            return "channel names do not match"
        return "incompatible channel names"

    def _selection_status_table_blocker(self) -> str | None:
        if self._table_target_state.create_table:
            if self._table_target_state.error_text is not None or self._table_target_state.table_name is None:
                return "create_invalid"
            return None

        if self.selected_table_name is None:
            return "no_eligible" if not self._eligible_existing_table_names else "choose_table"
        if self._table_target_state.error_text is not None:
            return "invalid"
        return None

    def _apply_status_card_spec(
        self,
        label: QLabel,
        spec: _FeatureExtractionStatusCardSpec | None,
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

    def _update_primary_status_card(self) -> None:
        spec = build_feature_extraction_selection_status_card_spec(
            has_spatialdata=self._app_state.sdata is not None,
            checked_coordinate_systems=self._staged_batch_state.checked_coordinate_systems,
            entries=self._build_selection_status_entries(),
            table_blocker=self._selection_status_table_blocker(),
            table_tooltip_message=self._table_target_state.error_text,
        )
        self._apply_status_card_spec(self.selection_status, spec)

    def _expected_controller_binding_state(self) -> FeatureExtractionBindingState:
        triplets: tuple[FeatureExtractionTriplet, ...] = ()
        if self._staged_batch_state.is_bindable and self._table_target_state.is_ready:
            triplets = self._staged_batch_state.triplets

        return FeatureExtractionBindingState(
            sdata=self.selected_spatialdata,
            triplets=triplets,
            table_name=self._table_target_state.bound_table_name,
            create_table=self._table_target_state.create_table,
            feature_names=self.selected_feature_names,
            feature_key=self.selected_feature_key,
        )

    def _controller_is_bound_to_staged_batch(self) -> bool:
        # Defensive guard: do not enable calculation if the controller still
        # holds an older request than the widget's currently resolved UI state.
        return self._feature_extraction_controller.binding_state == self._expected_controller_binding_state()

    def _should_show_controller_feedback(self) -> bool:
        return (
            self._staged_batch_state.is_bindable
            and self._table_target_state.is_ready
            and self._controller_is_bound_to_staged_batch()
            and bool(self._feature_extraction_controller.status_message)
        )

    def _update_feature_extraction_feedback(self) -> None:
        spec = build_feature_extraction_controller_feedback_card_spec(
            is_visible=self._should_show_controller_feedback(),
            message=self._feature_extraction_controller.status_message,
            kind=self._feature_extraction_controller.status_kind,
        )
        self._apply_status_card_spec(self.controller_feedback, spec)

    def _get_calculate_button_blocking_reason(self) -> str | None:
        if self.selected_spatialdata is None:
            return "Load a SpatialData object first."

        if not self._staged_batch_state.checked_coordinate_systems:
            return "Choose one or more coordinate systems."

        if not self._staged_batch_state.is_bindable:
            return self._staged_batch_state.error_text or "Complete all checked extraction targets before calculating."

        if self._table_target_state.error_text is not None:
            return self._table_target_state.error_text

        if self._table_target_state.create_table:
            if self._table_target_state.table_name is None:
                return "Enter a new table name."
        elif self.selected_table_name is None:
            if not self._eligible_existing_table_names:
                return "No table annotates all staged labels elements."
            return "Choose a table that annotates all staged labels elements."

        if not self.selected_feature_names:
            return "Choose at least one feature to calculate."

        if self.selected_feature_key is None:
            return "Choose an output feature key."

        if not self._controller_is_bound_to_staged_batch():
            return "Feature extraction request is still synchronizing with the current batch selection."

        if not self._feature_extraction_controller.can_calculate:
            return self._feature_extraction_controller.status_message

        return None

    def _update_calculate_controls(self) -> None:
        blocking_reason = self._get_calculate_button_blocking_reason()
        self.calculate_button.setEnabled(blocking_reason is None)
        if blocking_reason is None:
            self.calculate_button.setToolTip("")
            return

        self._set_tooltip(self.calculate_button, blocking_reason)

    def _on_controller_state_changed(self) -> None:
        self._update_feature_extraction_feedback()
        self._update_calculate_controls()

    def _on_controller_table_selection_refresh_required(self, result: FeatureExtractionResult) -> None:
        self._refresh_table_names(preferred_existing_table_name=result.table_name)
        self._bind_current_selection()

    def _on_controller_feature_matrix_changed(self, result: FeatureExtractionResult) -> None:
        sdata = self.selected_spatialdata
        if sdata is None:
            return

        event = TableStateChangedEvent(
            sdata=sdata,
            table_name=result.table_name,
            paths=frozenset(
                {
                    TableComponentPath("obsm", (result.feature_key,)),
                    TableComponentPath("uns", ("feature_matrices", result.feature_key)),
                }
            ),
            regions=result.regions,
            change_kind=result.change_kind,
            source="feature_extraction",
        )
        if sdata.is_backed():
            # harpy wrote the change to the store before this callback runs:
            # `hp.tb.add_feature_matrix()` wrote `obsm/<key>` and
            # `uns/feature_matrices/<key>`, or, for a "removed" result,
            # `hp.tb.io.remove_table_components()` deleted them. So the event is
            # recorded as persisted, not as an unsaved mutation:
            # - a path that was dirty at launch (`result.dirty_snapshot`) and that
            #   harpy wrote becomes clean, unless it changed again since launch;
            # - every other dirty path stays dirty.
            # The first case does not normally occur, because Calculate refuses to
            # launch while the key's paths have unsaved changes. A new table has
            # no snapshot.
            self._app_state.record_persisted_table_change(event, result.dirty_snapshot)
        else:
            # Nothing is on disk, so the change is an unsaved mutation.
            self._app_state.record_table_mutation(event)
