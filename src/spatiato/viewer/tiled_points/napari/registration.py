"""Explicit private napari registration for tiled-points layers."""

from __future__ import annotations

import re
from collections.abc import MutableMapping
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from spatiato.viewer.tiled_points.napari.layer import TiledPointsLayerModel

# This custom layer relies on private napari registration and rendering APIs,
# tested with napari 0.10.0 and VisPy 0.17.0. Newer versions are admitted
# without that qualification and may break, so run the tiled-points tests when
# bumping napari. `_load_napari_compatibility` still fails cleanly if napari
# moves or renames the private registries.
_MINIMUM_NAPARI_VERSION = "0.10.0"
_MINIMUM_VISPY_VERSION = "0.17.0"
# Leading release number, e.g. (0, 10, 1) for "0.10.1rc1" or "0.10.1.dev3+g1".
_RELEASE_PATTERN = re.compile(r"(\d+)\.(\d+)(?:\.(\d+))?")
_MISSING = object()


class TiledPointsLayerCompatibilityError(RuntimeError):
    """Report an unsupported or conflicting private napari integration."""


@dataclass(frozen=True)
class _NapariCompatibility:
    napari_version: str
    vispy_version: str
    visual_registry: MutableMapping[type[Any], type[Any]]
    controls_registry: MutableMapping[type[Any], type[Any]]


def register_tiled_points_layer() -> None:
    """Register the tiled-points model with napari's visual and controls factories.

    This extends the same private ``layer_to_visual`` and
    ``layer_to_controls`` registries that napari uses to dispatch its built-in
    layer models to VisPy layers and Qt controls. Napari populates its built-in
    entries statically; this function adds the Spatiato-owned mappings at runtime.
    This is not a public napari custom-layer registration API, which is why the
    integration is explicitly version- and contract-checked.

    Registration is explicit, version-checked, idempotent for the desired
    mappings, and atomic across the two private registries. Importing
    ``spatiato`` does not invoke this function. Registration installs
    factories only; it does not construct a model, controls, or visual::

        register_tiled_points_layer()
                    |
                    v
        +-- layer_to_visual
        |       TiledPointsLayerModel -> VispyTiledPointsLayer
        |
        +-- layer_to_controls
                TiledPointsLayerModel -> QtTiledPointsLayerControls

        later:

        TiledPointsLayerModel instance inserted into a GUI viewer
                    |
                    +-- napari constructs
                    |       VispyTiledPointsLayer(model, font_info)
                    |
                    +-- napari constructs
                            QtTiledPointsLayerControls(model)

    Call this before inserting the first ``TiledPointsLayerModel``. The layer
    list insertion event, not this function, triggers visual and controls
    construction through their respective factories.
    """
    compatibility = _load_napari_compatibility()
    _require_supported_versions(compatibility)

    from spatiato.viewer.tiled_points.napari.controls import QtTiledPointsLayerControls
    from spatiato.viewer.tiled_points.vispy.layer import VispyTiledPointsLayer

    desired = (
        (compatibility.visual_registry, VispyTiledPointsLayer, "visual"),
        (compatibility.controls_registry, QtTiledPointsLayerControls, "controls"),
    )
    for registry, expected, role in desired:
        existing = registry.get(TiledPointsLayerModel, _MISSING)
        if existing is not _MISSING and existing is not expected:
            raise TiledPointsLayerCompatibilityError(
                f"napari already maps TiledPointsLayerModel to a conflicting {role} class: {existing!r}."
            )

    missing_before = tuple(registry.get(TiledPointsLayerModel, _MISSING) is _MISSING for registry, _, _ in desired)
    try:
        for (registry, expected, _), is_missing in zip(desired, missing_before, strict=True):
            if is_missing:
                registry[TiledPointsLayerModel] = expected
    except Exception as error:
        for (registry, expected, _), was_missing in zip(desired, missing_before, strict=True):
            if was_missing and registry.get(TiledPointsLayerModel, _MISSING) is expected:
                del registry[TiledPointsLayerModel]
        raise TiledPointsLayerCompatibilityError(
            "Could not atomically register the tiled-points visual and controls with napari."
        ) from error


def _load_napari_compatibility() -> _NapariCompatibility:
    try:
        from napari._qt.layer_controls import qt_layer_controls_container
        from napari._vispy.utils import visual

        napari_version = version("napari")
        vispy_version = version("vispy")
    except (ImportError, PackageNotFoundError) as error:
        raise TiledPointsLayerCompatibilityError(
            "The tiled-points layer requires installed napari and VisPy packages with the expected private APIs."
        ) from error

    visual_registry = getattr(visual, "layer_to_visual", None)
    controls_registry = getattr(qt_layer_controls_container, "layer_to_controls", None)
    visual_factory = getattr(visual, "create_vispy_layer", None)
    controls_factory = getattr(qt_layer_controls_container, "create_qt_layer_controls", None)
    if (
        not isinstance(visual_registry, MutableMapping)
        or not isinstance(controls_registry, MutableMapping)
        or not callable(visual_factory)
        or not callable(controls_factory)
    ):
        raise TiledPointsLayerCompatibilityError(
            "The installed napari private visual/control registries do not match the supported integration contract."
        )
    return _NapariCompatibility(
        napari_version=napari_version,
        vispy_version=vispy_version,
        visual_registry=visual_registry,
        controls_registry=controls_registry,
    )


def _require_supported_versions(compatibility: _NapariCompatibility) -> None:
    if not (
        _is_at_least(compatibility.napari_version, _MINIMUM_NAPARI_VERSION)
        and _is_at_least(compatibility.vispy_version, _MINIMUM_VISPY_VERSION)
    ):
        raise TiledPointsLayerCompatibilityError(
            "The tiled-points renderer requires "
            f"napari >= {_MINIMUM_NAPARI_VERSION} with VisPy >= {_MINIMUM_VISPY_VERSION}; "
            f"found napari {compatibility.napari_version} with VisPy {compatibility.vispy_version}."
        )


def _is_at_least(installed: str, minimum: str) -> bool:
    """Return whether the installed release number is at least the minimum.

    Only the release number counts, so a pre-release or development build of
    the minimum release is admitted. An unrecognized version string is not.
    """
    installed_release = _release(installed)
    minimum_release = _release(minimum)
    if installed_release is None or minimum_release is None:
        return False
    return installed_release >= minimum_release


def _release(version_text: str) -> tuple[int, int, int] | None:
    match = _RELEASE_PATTERN.match(version_text)
    if match is None:
        return None
    major, minor, patch = match.groups()
    return int(major), int(minor), int(patch or 0)


def _unregister_tiled_points_layer_for_testing() -> None:
    """Remove only Spatiato-owned mappings to isolate registration tests."""
    compatibility = _load_napari_compatibility()

    from spatiato.viewer.tiled_points.napari.controls import QtTiledPointsLayerControls
    from spatiato.viewer.tiled_points.vispy.layer import VispyTiledPointsLayer

    for registry, expected in (
        (compatibility.visual_registry, VispyTiledPointsLayer),
        (compatibility.controls_registry, QtTiledPointsLayerControls),
    ):
        if registry.get(TiledPointsLayerModel, _MISSING) is expected:
            del registry[TiledPointsLayerModel]
