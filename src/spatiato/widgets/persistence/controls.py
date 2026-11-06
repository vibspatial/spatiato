"""Reusable controls for writing and reloading shared table state."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from spatiato._app_state import (
    SpatiatoAppState,
    TableBusyError,
    TableBusyStateChangedEvent,
    TableDirtyStateChangedEvent,
)
from spatiato.widgets.persistence.controller import PersistenceController
from spatiato.widgets.shared_styles import (
    ACTION_BUTTON_STYLESHEET,
    PRIMARY_BUTTON_STYLESHEET,
    SECONDARY_BUTTON_STYLESHEET,
    WARNING_BUTTON_STYLESHEET,
    format_tooltip,
    set_status_card,
)

if TYPE_CHECKING:
    from spatialdata import SpatialData


class _DirtyReloadDecision(Enum):
    WRITE = "write"
    RELOAD_DISCARD = "reload_discard"
    CANCEL = "cancel"


class TablePersistenceControls(QWidget):
    """Present reusable controls for one table's shared persistence state.

    Writing is generic: the control can call ``PersistenceController``
    directly, acknowledge the persisted mutation tokens, and let the shared
    dirty-state event refresh every bound control.

    While a background job writes the selected table, for example a Feature
    Extraction calculation, the table is busy: Write and Reload are disabled,
    with the reason as tooltip, and the shared busy-state event enables them
    again when the job ends.

    Reload is also generic at this boundary. This component resolves Write /
    Discard / Cancel and then captures one immutable ``TableReloadRequest``.
    Before replacing any in-memory AnnData components,
    ``PersistenceController`` passes that request through ``SpatiatoAppState``.
    App state refuses a table that became busy meanwhile, and otherwise calls
    ``prepare_for_table_reload()`` on every registered workflow.
    Unrelated workflows ignore the request; workflows using the selected table
    stop work that must not survive its replacement. For example, Object
    Classification freezes pending or running classifier work. Only then does
    ``PersistenceController`` reload the table components. Workflows adopt the
    restored table from the post-reload table-state event.

        Reload button clicked
            ↓
        controls resolve clean versus dirty-table behavior
            ↓
        PersistenceController captures the accepted request
            ↓
        SpatiatoAppState prepares every affected participant
            ↓
        PersistenceController executes the captured request
            ↓
        workflows adopt the restored state from the post-reload event
    """

    def __init__(
        self,
        app_state: SpatiatoAppState,
        *,
        write_content_description: str = "table state",
        reload_source: str = "table_persistence_controls",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        if not isinstance(write_content_description, str) or not write_content_description:
            raise ValueError("Persistence controls require a non-empty write-content description.")
        if not isinstance(reload_source, str) or not reload_source:
            raise ValueError("Persistence controls require a non-empty reload source.")

        self._app_state = app_state
        self._controller = PersistenceController(app_state)
        self._write_content_description = write_content_description
        self._reload_source = reload_source
        self._selected_spatialdata: SpatialData | None = None
        self._selected_table_name: str | None = None
        self._binding_error: str | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.action_row = QWidget()
        self.action_row.setObjectName("persistence_action_row")
        action_layout = QHBoxLayout(self.action_row)
        action_layout.setContentsMargins(0, 0, 0, 0)
        action_layout.setSpacing(8)

        self.write_button = QPushButton("Write Table State")
        self.write_button.setObjectName("sync_to_zarr_button")
        self.write_button.setEnabled(False)
        self.write_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.write_button.setMinimumHeight(28)
        self.write_button.setStyleSheet(ACTION_BUTTON_STYLESHEET)
        self.write_button.clicked.connect(self.write_table_state)

        self.reload_button = QPushButton("Reload Table State")
        self.reload_button.setObjectName("reload_from_zarr_button")
        self.reload_button.setEnabled(False)
        self.reload_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.reload_button.setMinimumHeight(28)
        self.reload_button.setStyleSheet(ACTION_BUTTON_STYLESHEET)
        self.reload_button.clicked.connect(self.reload_table_state)

        action_layout.addWidget(self.write_button, 1)
        action_layout.addWidget(self.reload_button, 1)

        self.feedback_label = QLabel()
        self.feedback_label.setObjectName("persistence_feedback")
        self.feedback_label.setWordWrap(True)
        self.feedback_label.hide()

        layout.addWidget(self.action_row)
        layout.addWidget(self.feedback_label)

        self._app_state.table_dirty_state_changed.connect(self._on_table_dirty_state_changed)
        self._app_state.table_busy_state_changed.connect(self._on_table_busy_state_changed)
        self.refresh()

    @property
    def controller(self) -> PersistenceController:
        """Return the component-level persistence service."""
        return self._controller

    def bind(
        self,
        sdata: SpatialData | None,
        table_name: str | None,
        region_name: str | None = None,
        *,
        binding_error: str | None = None,
    ) -> None:
        """Bind the controls to one selected table and refresh presentation."""
        if binding_error is not None and (not isinstance(binding_error, str) or not binding_error):
            raise ValueError("Persistence binding errors must be non-empty strings.")

        self._selected_spatialdata = sdata
        self._selected_table_name = table_name
        self._binding_error = binding_error
        effective_table_name = None if binding_error is not None else table_name
        self._controller.bind(sdata, effective_table_name, region_name)
        self.clear_feedback()
        self.refresh()

    def refresh(self) -> None:
        """Refresh button readiness and tooltips from authoritative shared state."""
        can_sync = self._controller.can_sync
        can_write = self._controller.can_write_table_state
        can_reload = self._controller.can_reload
        busy_message = self._controller.table_busy_message
        self.write_button.setEnabled(can_write and busy_message is None)
        self.reload_button.setEnabled(can_reload and busy_message is None)

        if self._selected_spatialdata is None or self._selected_table_name is None:
            write_tooltip = (
                f"Choose a backed SpatialData annotation table to enable writing "
                f"{self._write_content_description} to disk."
            )
            reload_tooltip = (
                "Choose a backed SpatialData annotation table to enable discarding the current in-memory table state "
                "and reloading the table from disk."
            )
        elif self._binding_error is not None:
            write_tooltip = self._binding_error
            reload_tooltip = self._binding_error
        elif not can_sync or not can_reload:
            write_tooltip = (
                "The selected SpatialData dataset is not backed by zarr, so the in-memory table state cannot be "
                "written to disk."
            )
            reload_tooltip = (
                "The selected SpatialData dataset is not backed by zarr, so the table state cannot be reloaded "
                "from disk."
            )
        elif busy_message is not None:
            write_tooltip = busy_message
            reload_tooltip = busy_message
        else:
            table_store_path = self._controller.selected_table_store_path
            destination = self._selected_spatialdata.path if table_store_path is None else table_store_path
            has_unsynced_changes = self._controller.has_unsynced_table_changes
            if has_unsynced_changes:
                write_tooltip = (
                    f'Write {self._write_content_description} for "{self._selected_table_name}" to "{destination}".'
                )
            else:
                write_tooltip = (
                    f'The selected "{self._selected_table_name}" table has no unsynced local in-memory changes '
                    "to write."
                )
            reload_tooltip = (
                f'Discard the current in-memory "{self._selected_table_name}" table state and reload the table '
                f'from "{destination}".'
            )
            if has_unsynced_changes:
                write_tooltip += " Unsynced local in-memory table changes are present."
                reload_tooltip += " Unsynced local in-memory table changes would be discarded."

        self.write_button.setToolTip(format_tooltip(write_tooltip))
        self.reload_button.setToolTip(format_tooltip(reload_tooltip))

    def write_table_state(
        self,
        *,
        show_feedback: bool = True,
        feedback_message: str | None = None,
    ) -> bool:
        """Write captured dirty components and optionally present the outcome."""
        try:
            self._controller.write_table_state()
        except (ValueError, TableBusyError) as error:
            self.set_feedback(str(error), error=True)
            return False

        if show_feedback:
            destination = self.selected_table_store_destination()
            message = feedback_message or (
                f'Wrote "{self._selected_table_name}" {self._write_content_description} to "{destination}".'
            )
            self.set_feedback(message)
        return True

    def reload_table_state(self) -> bool:
        """Resolve user intent and execute one participant-safe table reload.

        A clean table reloads immediately. A dirty table first presents the
        Write / Discard / Cancel decision:

        clean
            → continue with reload

        dirty
            ├── Write
            │      → persist local changes
            │      → continue with reload
            ├── Discard
            │      → continue without writing
            └── Cancel
                   → stop before capturing a TableReloadRequest

        Only an accepted transition captures a request, prepares registered
        workflows, and replaces the selected in-memory table components from
        disk.
        """
        wrote_before_reload = False
        if self._controller.has_unsynced_table_changes:
            decision = self._prompt_dirty_reload_decision()
            if decision is _DirtyReloadDecision.CANCEL:
                return False
            if decision is _DirtyReloadDecision.WRITE:
                if not self.write_table_state(show_feedback=False):
                    return False
                wrote_before_reload = True
            elif decision is not _DirtyReloadDecision.RELOAD_DISCARD:
                raise RuntimeError(f"Unhandled dirty reload decision: {decision!r}")

        try:
            request = self._controller.capture_table_reload_request(
                source=self._reload_source,
            )
            self._controller.reload_table_request(request)
        except Exception as error:  # noqa: BLE001 - this UI boundary must report participant and reload failures.
            self.set_feedback(str(error), error=True)
            return False

        destination = self.selected_table_store_destination()
        if wrote_before_reload:
            message = (
                f'Wrote local table state and reloaded "{self._selected_table_name}" table state from "{destination}".'
            )
        else:
            message = f'Reloaded "{self._selected_table_name}" table state from "{destination}".'
        self.set_feedback(message)
        return True

    def clear_feedback(self) -> None:
        """Hide any previous persistence outcome."""
        self.set_feedback("")

    def set_feedback(self, message: str, *, error: bool = False) -> None:
        """Present one persistence success or error outcome."""
        if not message:
            self.feedback_label.setText("")
            self.feedback_label.setVisible(False)
            return

        set_status_card(
            self.feedback_label,
            title="Persistence Error" if error else "Persistence Updated",
            lines=[message],
            kind="error" if error else "success",
        )

    def selected_table_store_destination(self) -> str:
        """Return the user-facing destination for the current table."""
        table_store_path = self._controller.selected_table_store_path
        if table_store_path is not None:
            return str(table_store_path)
        store_path = self._controller.selected_store_path
        return "" if store_path is None else str(store_path)

    def _prompt_dirty_reload_decision(self) -> _DirtyReloadDecision:
        dialog = QDialog(self)
        dialog.setWindowTitle("Unsynced Table Changes")
        dialog.setModal(True)
        dialog.setMinimumWidth(560)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        warning_message = (
            f'Table "{self._selected_table_name}" has in-memory changes that have not been written to zarr.'
            if self._selected_table_name is not None
            else "The selected table has in-memory changes that have not been written to zarr."
        )
        warning_card = QLabel()
        warning_card.setWordWrap(True)
        set_status_card(
            warning_card,
            title="Unsynced Changes",
            lines=[warning_message],
            kind="warning",
        )
        layout.addWidget(warning_card)

        button_row = QHBoxLayout()
        button_row.setSpacing(10)
        button_row.addStretch(1)
        write_button = QPushButton("Write table state and reload")
        discard_button = QPushButton("Reload table state and discard local edits")
        cancel_button = QPushButton("Cancel")

        write_button.setStyleSheet(PRIMARY_BUTTON_STYLESHEET)
        discard_button.setStyleSheet(WARNING_BUTTON_STYLESHEET)
        cancel_button.setStyleSheet(SECONDARY_BUTTON_STYLESHEET)

        button_row.addWidget(write_button)
        button_row.addWidget(discard_button)
        button_row.addWidget(cancel_button)
        layout.addLayout(button_row)

        write_button.clicked.connect(lambda: dialog.done(1))
        discard_button.clicked.connect(lambda: dialog.done(2))
        cancel_button.clicked.connect(dialog.reject)
        cancel_button.setDefault(True)

        result = dialog.exec()
        if result == 1:
            return _DirtyReloadDecision.WRITE
        if result == 2:
            return _DirtyReloadDecision.RELOAD_DISCARD
        return _DirtyReloadDecision.CANCEL

    def _on_table_dirty_state_changed(self, event: object) -> None:
        """Refresh Write readiness and Reload warnings after a dirty transition."""
        if not isinstance(event, TableDirtyStateChangedEvent):
            return
        if event.sdata is not self._selected_spatialdata or event.table_name != self._selected_table_name:
            return
        self.refresh()

    def _on_table_busy_state_changed(self, event: object) -> None:
        """Disable or enable Write and Reload after the table became busy or no longer busy."""
        if not isinstance(event, TableBusyStateChangedEvent):
            return
        if event.sdata is not self._selected_spatialdata or event.table_name != self._selected_table_name:
            return
        self.refresh()
