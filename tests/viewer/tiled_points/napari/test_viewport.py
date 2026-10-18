from __future__ import annotations

from itertools import product
from uuid import uuid4

import numpy as np
import pytest

from spatiato.viewer.tiled_points import (
    TiledPointsDatasetReference,
    TiledPointsLayerModel,
    TiledPointsViewportState,
)


def _layer(**kwargs: object) -> TiledPointsLayerModel:
    return TiledPointsLayerModel(
        TiledPointsDatasetReference(
            cache_generation_id=str(uuid4()),
            points_name="spots",
            value_column="feature_name",
            value_count=3,
            x_origin=-128.0,
            y_origin=-256.0,
            x_min=-100.0,
            x_max=500.0,
            y_min=-200.0,
            y_max=600.0,
        ),
        value_palette=np.full((3, 4), 255, dtype=np.uint8),
        max_vertex_payload_bytes=1_000_000,
        **kwargs,
    )


def _record_viewports(layer: TiledPointsLayerModel) -> list[TiledPointsViewportState]:
    observed: list[TiledPointsViewportState] = []
    layer.events.viewport.connect(lambda event: observed.append(event.value))
    return observed


def _draw(
    layer: TiledPointsLayerModel,
    *,
    corners: np.ndarray | None = None,
    canvas_shape: tuple[int, int] = (100, 200),
    scale_factor: float = 0.5,
) -> None:
    if corners is None:
        corners = np.array(((2.25, 10.5), (12.75, 30.25)), dtype=np.float64)
    layer._update_draw(
        scale_factor=scale_factor,
        corner_pixels_displayed=corners,
        shape_threshold=canvas_shape,
    )


def test_identity_draw_emits_float_intrinsic_viewport_and_effective_budget() -> None:
    layer = _layer(hard_render_point_budget=100_000, target_pixels_per_point=9.0)
    observed = _record_viewports(layer)

    _draw(layer)

    assert observed == [
        TiledPointsViewportState(
            displayed_axes=(0, 1),
            x_min=10.5,
            y_min=2.25,
            x_max=30.25,
            y_max=12.75,
            canvas_width=200,
            canvas_height=100,
            hard_render_point_budget=100_000,
            screen_density_budget=2_222,
        )
    ]


@pytest.mark.parametrize(
    "layer_kwargs",
    [
        pytest.param({}, id="identity"),
        pytest.param({"translate": (17.0, -31.0)}, id="translation"),
        pytest.param({"scale": (2.0, 2.0)}, id="uniform-scale"),
        pytest.param({"scale": (2.0, 5.0)}, id="anisotropic-scale"),
        pytest.param({"rotate": 37.0}, id="rotation"),
        pytest.param({"shear": (0.35,)}, id="shear"),
    ],
)
def test_draw_inverse_transforms_all_four_world_corners_in_explicit_axis_order(
    layer_kwargs: dict[str, object],
) -> None:
    layer = _layer(**layer_kwargs)
    observed = _record_viewports(layer)
    world_bounds = np.array(((11.25, 103.5), (47.75, 182.25)), dtype=np.float64)
    world_corners = np.asarray(tuple(product(*world_bounds.T)), dtype=np.float64)
    intrinsic_yx = np.asarray(tuple(layer.world_to_data(corner) for corner in world_corners), dtype=np.float64)

    _draw(layer, corners=world_bounds)

    state = observed[-1]
    np.testing.assert_allclose(
        (state.x_min, state.y_min, state.x_max, state.y_max),
        (
            intrinsic_yx[:, 1].min(),
            intrinsic_yx[:, 0].min(),
            intrinsic_yx[:, 1].max(),
            intrinsic_yx[:, 0].max(),
        ),
        rtol=0.0,
        atol=1e-12,
    )


def test_rotation_requires_off_diagonal_world_corners_for_conservative_bounds() -> None:
    layer = _layer(rotate=45.0)
    observed = _record_viewports(layer)
    world_bounds = np.array(((0.0, 0.0), (10.0, 10.0)), dtype=np.float64)
    diagonal_yx = np.asarray(tuple(layer.world_to_data(corner) for corner in world_bounds), dtype=np.float64)

    _draw(layer, corners=world_bounds)

    state = observed[-1]
    assert state.x_min < float(diagonal_yx[:, 1].min())
    assert state.x_max > float(diagonal_yx[:, 1].max())


def test_identical_draw_is_deduplicated_but_canvas_resize_emits() -> None:
    layer = _layer(hard_render_point_budget=100_000, target_pixels_per_point=9.0)
    observed = _record_viewports(layer)

    _draw(layer)
    _draw(layer)
    _draw(layer, canvas_shape=(120, 240))

    assert len(observed) == 2
    assert (observed[-1].canvas_width, observed[-1].canvas_height) == (240, 120)
    assert observed[-1].screen_density_budget == 3_200


def test_stationary_viewport_recomputes_budget_without_point_diameter_coupling() -> None:
    layer = _layer(target_pixels_per_point=9.0)
    observed = _record_viewports(layer)

    layer.hard_render_point_budget = 500
    assert observed == []

    _draw(layer, canvas_shape=(90, 90))
    assert observed[-1].screen_density_budget == 900
    assert observed[-1].effective_point_budget == 500

    layer.hard_render_point_budget = 1_000
    assert observed[-1].effective_point_budget == 900

    layer.target_pixels_per_point = 18.0
    assert observed[-1].screen_density_budget == 450
    assert observed[-1].effective_point_budget == 450

    event_count = len(observed)
    layer.point_diameter = 12.0
    assert len(observed) == event_count


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"hard_render_point_budget": 0}, "hard_render_point_budget"),
        ({"hard_render_point_budget": True}, "hard_render_point_budget"),
        ({"target_pixels_per_point": 0.0}, "target_pixels_per_point"),
        ({"target_pixels_per_point": np.nan}, "target_pixels_per_point"),
    ],
)
def test_layer_rejects_invalid_viewport_budget_policy(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _layer(**kwargs)
