<p align="center">
  <img src="https://raw.githubusercontent.com/vibspatial/spatiato/main/src/spatiato/_static/logo.svg" alt="Spatiato logo" width="400">
</p>

<h1 align="center">spatiato: a spatial omics interface for napari.</h1>

[![PyPI](https://img.shields.io/pypi/v/spatiato.svg)](https://pypi.org/project/spatiato/)

Built around [`SpatialData`](https://spatialdata.scverse.org/en/stable/) and [`Harpy`](https://harpy.readthedocs.io/en/latest/) for interactive exploration, feature extraction, and object classification.

`spatiato` is a napari plugin for viewing, exploring, and analyzing
`SpatialData` datasets. It includes its own viewer for loading and
browsing data inside napari, alongside annotation, feature extraction, and
interactive object classification workflows.

## Installation

Install from [PyPI](https://pypi.org/project/spatiato/):

```bash
pip install spatiato
```

## Quickstart

The quickest way to try the plugin is to create a small example `SpatialData`
object, write it to a temporary zarr store, read it back as an on-disk dataset,
and launch the Spatiato napari interface with `Interactive`.

```python
import tempfile
from pathlib import Path

import harpy as hp

from spatiato import Interactive
from spatiato.datasets import blobs_multi_region

zarr_path = Path(tempfile.mkdtemp()) / "blobs_multi_region.zarr"

sdata = blobs_multi_region()
sdata.write(zarr_path)

sdata = hp.io.read_zarr(zarr_path)
Interactive(sdata)
```

This opens napari with the Spatiato widgets docked and the
`blobs_multi_region` dataset available in the shared viewer state.
`hp.io.read_zarr` opens table matrices lazily, as Spatiato's own reader does;
`spatialdata.read_zarr` also works, and loads them into memory.
Shapes triangulation uses the fast Bermuda backend by default. To use Numba
instead, launch with `Interactive(sdata, triangulation_backend="numba")`.

The current repository contains four working widgets:

- `Viewer`
- `Feature Extraction`
- `Object Classification`
- `Annotation`

Today the plugin supports:

- loading and viewing `SpatialData` through the Spatiato viewer widget
- selecting a labels element, optional image, compatible coordinate system, and
  linked table from the shared loaded `SpatialData`
- calculating intensity and morphology features through Harpy
- writing feature matrices into the selected `AnnData` table linked to the
  labels element, as `.obsm[feature_key]`, with companion metadata in
  `.uns["feature_matrices"][feature_key]`
- interactive manual annotation of instances in labels elements
- interactive creation and editing of polygon shapes annotations
- background `RandomForestClassifier` retraining on the selected feature
  matrix stored in `.obsm[feature_key]` of the `AnnData` table linked to the
  labels element
- live prediction updates and labels recoloring
- explicit write-back of in-memory table state to zarr
- explicit reload of on-disk table state back into memory
- multi-sample workflows through multi-region tables and explicit
  labels/image/coordinate-system matching
- headless feature extraction and classifier application for scripted or batch
  processing

Example spatiato session:

<p align="center">
  <img src="https://raw.githubusercontent.com/vibspatial/spatiato/main/docs/_static/viewer.png" alt="spatiato viewer example screenshot" width="900">
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/vibspatial/spatiato/main/docs/_static/object_classification.png" alt="spatiato object classification example screenshot" width="900">
</p>

## Headless and Multi-Sample Workflows

For scripted or batch processing, use the public `spatiato.headless` module.
It can apply an exported classifier to an existing feature matrix, or compute
the required features before applying the classifier.

The headless APIs accept one labels element or a sequence of labels elements.
For multi-sample data, pass matching labels, image, and coordinate-system
sequences so Harpy can build or apply a shared table-level feature matrix across
the selected samples. When features need image intensities, the channel
selection is read from the exported classifier's `source_channels` metadata.

```python
import harpy as hp

from spatiato import headless

sdata = hp.io.read_zarr("experiment.zarr")

result = headless.apply_classifier_with_feature_extraction_from_path(
    sdata,
    "classifier.spatiato-classifier.joblib",
    table_name="table_multi",
    labels_name=["sample_1_labels", "sample_2_labels"],
    coordinate_system=["sample_1", "sample_2"],
    image_name=["sample_1_image", "sample_2_image"],
)
```

Predictions go to `pred_class` and `pred_class_confidence`. If you pass
another `pred_class_column`, the confidences go to
`<class column>_confidence`. How the predictions were made is recorded in
`table.uns["classifier_config"][<class column>]`. If the Object
Classification widget shows the same data, click Reload in the widget after
the headless run to show the new predictions. If the run wrote `pred_class`,
the widget shows those predictions, but has no model of its own for them:
its status asks you to train, and Export is unavailable until you do.

## Local development

Create the development environment:

```bash
./create_env.sh
```

Then launch napari:

```bash
source .venv/bin/activate
napari
```

Open the widgets from the napari plugin menu:

- `Plugins -> spatiato -> Viewer`
- `Plugins -> spatiato -> Feature Extraction`
- `Plugins -> spatiato -> Object Classification`
- `Plugins -> spatiato -> Annotation`

### Optional real-OpenGL renderer qualification

The tiled-points renderer includes one opt-in real-canvas qualification test at
[`tests/viewer/tiled_points/vispy/test_real_canvas.py`](tests/viewer/tiled_points/vispy/test_real_canvas.py).
It is skipped during ordinary test runs because it requires a working
Qt, VisPy, and OpenGL environment. Run it explicitly with:

```bash
SPATIATO_RUN_REAL_GL_TESTS=1 \
.venv/bin/pytest -q tests/viewer/tiled_points/vispy/test_real_canvas.py
```

The focused renderer unit tests remain part of ordinary test runs; this
additional qualification creates a real canvas, compiles and executes the
custom shaders, and compares the framebuffer result with standard VisPy
markers.

## Debug script

A small local debug script is available at
[`scripts/debug_widget.py`](scripts/debug_widget.py).

It creates a temporary `blobs_multi_region` zarr store, loads it into napari,
and docks the Spatiato widgets automatically.
This is useful for quickly reproducing widget behavior during development.

Run it with:

```bash
source .venv/bin/activate
python scripts/debug_widget.py
```
