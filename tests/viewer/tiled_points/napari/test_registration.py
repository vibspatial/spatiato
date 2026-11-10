from __future__ import annotations

from collections.abc import Iterator
from types import ModuleType
from uuid import uuid4

import numpy as np
import pytest
from napari._qt.layer_controls import qt_layer_controls_container
from napari._qt.layer_controls.qt_layer_controls_container import create_qt_layer_controls
from napari._vispy.utils import visual
from napari._vispy.utils.qt_font import FontInfo
from napari._vispy.utils.visual import create_vispy_layer

from spatiato.viewer.tiled_points import (
    TiledPointsDatasetReference,
    TiledPointsLayerModel,
)
from spatiato.viewer.tiled_points.napari import registration
from spatiato.viewer.tiled_points.napari.controls import QtTiledPointsLayerControls
from spatiato.viewer.tiled_points.napari.registration import (
    TiledPointsLayerCompatibilityError,
    register_tiled_points_layer,
)
from spatiato.viewer.tiled_points.vispy.layer import VispyTiledPointsLayer


def _layer() -> TiledPointsLayerModel:
    return TiledPointsLayerModel(
        TiledPointsDatasetReference(
            cache_generation_id=str(uuid4()),
            points_name="spots",
            value_column="feature_name",
            value_count=3,
            x_origin=0.0,
            y_origin=0.0,
            x_min=3.0,
            x_max=23.0,
            y_min=2.0,
            y_max=12.0,
        ),
        value_palette=np.full((3, 4), 255, dtype=np.uint8),
        max_vertex_payload_bytes=1_000_000,
    )


@pytest.fixture
def clean_real_registration() -> Iterator[None]:
    registration._unregister_tiled_points_layer_for_testing()
    try:
        yield
    finally:
        registration._unregister_tiled_points_layer_for_testing()


def test_registration_is_idempotent_and_factories_select_owned_types(
    clean_real_registration: None,
    monkeypatch: pytest.MonkeyPatch,
    qtbot,
) -> None:
    monkeypatch.setattr("napari._vispy.layers.base.get_max_texture_sizes", lambda: (8192, 2048))
    register_tiled_points_layer()
    register_tiled_points_layer()
    layer = _layer()

    controls = create_qt_layer_controls(layer)
    qtbot.addWidget(controls)
    visual = create_vispy_layer(layer, font_info=FontInfo())
    try:
        assert isinstance(controls, QtTiledPointsLayerControls)
        assert isinstance(visual, VispyTiledPointsLayer)
        layer.visible = False
        assert not visual.node.visible
    finally:
        visual.close()


@pytest.mark.parametrize(
    ("module", "attribute", "replacement"),
    [
        (visual, "layer_to_visual", object()),
        (qt_layer_controls_container, "layer_to_controls", object()),
        (visual, "create_vispy_layer", None),
        (qt_layer_controls_container, "create_qt_layer_controls", None),
    ],
)
def test_private_registration_shape_mismatch_fails_cleanly(
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
    attribute: str,
    replacement: object,
) -> None:
    monkeypatch.setattr(module, attribute, replacement)

    with pytest.raises(TiledPointsLayerCompatibilityError, match="private visual/control registries"):
        registration._load_napari_compatibility()


class _FailingControlsRegistry(dict[type[object], type[object]]):
    def __setitem__(self, key: type[object], value: type[object]) -> None:
        raise RuntimeError("controls registry failure")


def _compatibility(
    *,
    napari_version: str = registration._MINIMUM_NAPARI_VERSION,
    vispy_version: str = registration._MINIMUM_VISPY_VERSION,
    visual_registry: dict[type[object], type[object]] | None = None,
    controls_registry: dict[type[object], type[object]] | None = None,
) -> registration._NapariCompatibility:
    return registration._NapariCompatibility(
        napari_version=napari_version,
        vispy_version=vispy_version,
        visual_registry={} if visual_registry is None else visual_registry,
        controls_registry={} if controls_registry is None else controls_registry,
    )


def test_registration_rolls_back_visual_when_controls_mutation_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    visual_registry: dict[type[object], type[object]] = {}
    compatibility = _compatibility(
        visual_registry=visual_registry,
        controls_registry=_FailingControlsRegistry(),
    )
    monkeypatch.setattr(registration, "_load_napari_compatibility", lambda: compatibility)

    with pytest.raises(TiledPointsLayerCompatibilityError, match="atomically"):
        register_tiled_points_layer()

    assert TiledPointsLayerModel not in visual_registry


def test_registration_rejects_conflict_before_mutation(monkeypatch: pytest.MonkeyPatch) -> None:
    class ConflictingVisual:
        pass

    visual_registry = {TiledPointsLayerModel: ConflictingVisual}
    controls_registry: dict[type[object], type[object]] = {}
    compatibility = _compatibility(
        visual_registry=visual_registry,
        controls_registry=controls_registry,
    )
    monkeypatch.setattr(registration, "_load_napari_compatibility", lambda: compatibility)

    with pytest.raises(TiledPointsLayerCompatibilityError, match="conflicting visual"):
        register_tiled_points_layer()

    assert visual_registry == {TiledPointsLayerModel: ConflictingVisual}
    assert controls_registry == {}


@pytest.mark.parametrize(
    ("napari_version", "vispy_version"),
    [
        ("0.9.2", "0.17.0"),  # napari below the minimum
        ("0.10.0", "0.16.2"),  # VisPy below the minimum
        ("unknown", "0.17.0"),  # unrecognized version string
    ],
)
def test_unsupported_versions_fail_before_registry_mutation(
    monkeypatch: pytest.MonkeyPatch,
    napari_version: str,
    vispy_version: str,
) -> None:
    visual_registry: dict[type[object], type[object]] = {}
    controls_registry: dict[type[object], type[object]] = {}
    compatibility = _compatibility(
        napari_version=napari_version,
        vispy_version=vispy_version,
        visual_registry=visual_registry,
        controls_registry=controls_registry,
    )
    monkeypatch.setattr(registration, "_load_napari_compatibility", lambda: compatibility)

    with pytest.raises(TiledPointsLayerCompatibilityError, match="requires napari >="):
        register_tiled_points_layer()

    assert visual_registry == {}
    assert controls_registry == {}


@pytest.mark.parametrize(
    ("napari_version", "vispy_version"),
    [
        ("0.10.0", "0.17.0"),
        ("0.10.1rc1", "0.17.0"),  # a pre-release counts as its release
        ("0.11", "0.18.2"),
        ("1.0.0.dev3+g1234abc", "1.0.0"),
    ],
)
def test_newer_versions_are_admitted(
    monkeypatch: pytest.MonkeyPatch,
    napari_version: str,
    vispy_version: str,
) -> None:
    visual_registry: dict[type[object], type[object]] = {}
    controls_registry: dict[type[object], type[object]] = {}
    compatibility = _compatibility(
        napari_version=napari_version,
        vispy_version=vispy_version,
        visual_registry=visual_registry,
        controls_registry=controls_registry,
    )
    monkeypatch.setattr(registration, "_load_napari_compatibility", lambda: compatibility)

    register_tiled_points_layer()

    assert visual_registry == {TiledPointsLayerModel: VispyTiledPointsLayer}
    assert controls_registry == {TiledPointsLayerModel: QtTiledPointsLayerControls}
