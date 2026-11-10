from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from _shapes_regression_fixtures import (
    CONCAVE_SIMPLE_POLYGON_DELETION_REGRESSION_VERTEX_INDEX,
    TRIANGULATION_REGRESSION_FAILED_MOVE_COORDINATE,
    TRIANGULATION_REGRESSION_HOLE_ANCHOR_COORDINATE,
    TRIANGULATION_REGRESSION_HOLE_ANCHOR_INDICES,
    TRIANGULATION_REGRESSION_MOVED_VERTEX_INDEX,
    TRIANGULATION_REGRESSION_NEARBY_SHELL_VERTEX_INDEX,
    TRIANGULATION_REGRESSION_SHELL_ANCHOR_COORDINATE,
    make_concave_simple_polygon_deletion_regression_vertices,
    make_triangulation_regression_pre_drag_vertices,
)
from napari.layers import Shapes
from napari.layers.base._base_constants import ActionType
from napari.layers.shapes import _shapes_utils
from napari.layers.shapes._shapes_constants import Mode

from spatiato.core.shapes_geometry import napari_polygon_vertices_to_shapely_polygon


def _make_direct_regression_layer() -> Shapes:
    vertices = make_triangulation_regression_pre_drag_vertices()
    layer = Shapes(data=[vertices], shape_type="polygon")
    layer.mode = Mode.DIRECT
    layer.selected_data = {0}
    return layer


def _direct_press_event() -> SimpleNamespace:
    return SimpleNamespace(
        type="mouse_press",
        position=TRIANGULATION_REGRESSION_HOLE_ANCHOR_COORDINATE,
        modifiers=(),
    )


def test_native_direct_first_next_runs_press_setup_and_yields_before_polygon_mutation(
    monkeypatch: pytest.MonkeyPatch,
    numba_triangulation_backend: None,
) -> None:
    """Verify that the first ``next(...)`` runs only mouse-press setup.

    It identifies raw vertex 39, initializes drag and selection state, and
    then yields. It does not move polygon data, call
    ``_data_view.edit(...)``, triangulate, mark the layer as moving, or emit a
    data-change event. Spatiato's edit guard depends on this yield boundary to
    reuse napari's press setup while retaining control before napari mutates or
    triangulates the polygon.
    """
    layer = _make_direct_regression_layer()
    before_press = np.asarray(layer.data[0], dtype=float).copy()
    data_actions: list[ActionType] = []

    layer.events.data.connect(lambda event: data_actions.append(event.action))
    monkeypatch.setattr(
        layer,
        "get_value",
        lambda position, world=True: (0, TRIANGULATION_REGRESSION_MOVED_VERTEX_INDEX),
    )

    def fail_edit(*args: object, **kwargs: object) -> None:
        raise AssertionError("The first native direct yield must occur before `_data_view.edit(...)`.")

    def fail_triangulation(*args: object, **kwargs: object) -> None:
        raise AssertionError("The first native direct yield must occur before triangulation.")

    # An unchanged row alone would not prove that these methods were never
    # entered. Use fail-fast sentinels to verify that the first yield occurs
    # before napari reaches either its edit or triangulation path.
    monkeypatch.setattr(layer._data_view, "edit", fail_edit)
    monkeypatch.setattr(layer._data_view.shapes[0], "_set_meshes", fail_triangulation)

    event = _direct_press_event()
    native_direct_callback = layer._drag_modes[Mode.DIRECT]
    direct_drag = native_direct_callback(layer, event)
    try:
        # Creating `direct_drag` does not execute the generator body. This first
        # `next(...)` runs napari's press setup and stops at its initial yield;
        # the assertions below verify that polygon mutation has not begun.
        assert next(direct_drag) is None
        assert layer._moving_value == (0, TRIANGULATION_REGRESSION_MOVED_VERTEX_INDEX)
        assert layer._drag_start is not None
        assert layer.selected_data == {0}
        assert np.array_equal(np.asarray(layer.data[0], dtype=float), before_press)
        assert layer._is_moving is False
        assert data_actions == []
    finally:
        direct_drag.close()


def test_native_direct_move_triangulation_failure_leaves_hole_anchor_unsynchronized(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    numba_triangulation_backend: None,
) -> None:
    """Verify the malformed live row left by a native triangulation failure.

    The drag starts with duplicated third-hole anchors 34 and 39 synchronized
    at Q. Napari moves only raw vertex 39 to Q' and sends the resulting row to
    the real Numba/VisPy triangulator, which raises. The failed move remains in
    the live row: index 34 is still Q, index 39 is Q', and Spatiato can no longer
    decode the third hole as closed.

    This test characterizes the failure state that Spatiato's edit guard must
    handle. It demonstrates why a rendering failure must restore the cached
    accepted row instead of leaving napari's partially applied move live.
    """
    layer = _make_direct_regression_layer()
    data_actions: list[ActionType] = []
    layer.events.data.connect(lambda event: data_actions.append(event.action))
    monkeypatch.setattr(
        layer,
        "get_value",
        lambda position, world=True: (0, TRIANGULATION_REGRESSION_MOVED_VERTEX_INDEX),
    )
    monkeypatch.setattr(
        _shapes_utils,
        "_save_failed_triangulation",
        lambda *args, **kwargs: (tmp_path / "failed.npz", tmp_path / "failed.txt"),
    )

    event = _direct_press_event()
    native_direct_callback = layer._drag_modes[Mode.DIRECT]
    direct_drag = native_direct_callback(layer, event)
    assert next(direct_drag) is None
    # Reproduce the recorded Q -> Q' move. Napari moves only raw vertex 39,
    # leaving its duplicate hole anchor at index 34 unchanged; resuming the
    # native generator then sends that unsynchronized row through the real
    # Numba/VisPy triangulator, where triangulation fails.
    event.type = "mouse_move"
    event.position = TRIANGULATION_REGRESSION_FAILED_MOVE_COORDINATE
    try:
        with pytest.raises(RuntimeError, match="Triangulation failed"):
            next(direct_drag)
    finally:
        direct_drag.close()

    vertices = np.asarray(layer.data[0], dtype=float)
    unchanged_hole_anchor_index = TRIANGULATION_REGRESSION_HOLE_ANCHOR_INDICES[0]
    np.testing.assert_array_equal(
        vertices[unchanged_hole_anchor_index],
        TRIANGULATION_REGRESSION_HOLE_ANCHOR_COORDINATE,
    )
    np.testing.assert_array_equal(
        vertices[TRIANGULATION_REGRESSION_MOVED_VERTEX_INDEX],
        TRIANGULATION_REGRESSION_FAILED_MOVE_COORDINATE,
    )
    np.testing.assert_array_equal(
        vertices[TRIANGULATION_REGRESSION_NEARBY_SHELL_VERTEX_INDEX],
        TRIANGULATION_REGRESSION_SHELL_ANCHOR_COORDINATE,
    )
    assert not np.array_equal(
        vertices[unchanged_hole_anchor_index],
        vertices[TRIANGULATION_REGRESSION_MOVED_VERTEX_INDEX],
    )
    # The failed triangulation leaves index 34 at Q and index 39 at Q', so
    # Spatiato rejects the third hole as unclosed. This is why the edit guard must
    # restore the cached accepted row when triangulation raises: that rollback
    # resynchronizes the aliases instead of leaving malformed live data behind.
    with pytest.raises(ValueError, match="each hole ring must be closed"):
        napari_polygon_vertices_to_shapely_polygon(vertices)
    assert layer._moving_value == (0, TRIANGULATION_REGRESSION_MOVED_VERTEX_INDEX)
    assert layer._is_moving is True
    assert data_actions == []


def test_native_simple_polygon_vertex_remove_accepts_invalid_shortened_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that native vertex removal accepts an invalid polygon row.

    The source is a valid concave simple polygon. Removing vertex 0 creates a
    new implicit closing edge that self-intersects, but napari's real
    ``Mode.VERTEX_REMOVE`` callback still writes the shortened row and emits
    both ``CHANGING`` and ``CHANGED`` without raising.

    Unlike the move regression, this edit does not fail during rendering. It
    completes while leaving live data that Spatiato cannot decode as a valid
    Shapely polygon. This characterizes why Spatiato's edit guard must validate a
    deletion candidate before accepting it and preserve the previous valid row
    when validation fails.
    """
    source_vertices = make_concave_simple_polygon_deletion_regression_vertices()
    layer = Shapes(data=[source_vertices], shape_type="polygon")
    layer.mode = Mode.VERTEX_REMOVE
    layer.selected_data = {0}
    data_actions: list[ActionType] = []
    layer.events.data.connect(lambda event: data_actions.append(event.action))
    monkeypatch.setattr(
        layer,
        "get_value",
        lambda position, world=True: (0, CONCAVE_SIMPLE_POLYGON_DELETION_REGRESSION_VERTEX_INDEX),
    )

    event = SimpleNamespace(position=tuple(source_vertices[0]))
    native_vertex_remove_callback = layer._drag_modes[Mode.VERTEX_REMOVE]
    assert native_vertex_remove_callback(layer, event) is None

    expected_vertices = np.delete(
        source_vertices,
        CONCAVE_SIMPLE_POLYGON_DELETION_REGRESSION_VERTEX_INDEX,
        axis=0,
    )
    assert len(layer.data) == 1
    np.testing.assert_array_equal(np.asarray(layer.data[0], dtype=float), expected_vertices)
    with pytest.raises(ValueError, match="Polygon path cannot be converted to a valid polygon"):
        napari_polygon_vertices_to_shapely_polygon(layer.data[0])
    assert data_actions == [ActionType.CHANGING, ActionType.CHANGED]
