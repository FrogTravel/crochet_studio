"""Crochet Studio — YOLO-OBB pipeline for crochet stitch detection.

Modules in this package mirror the three steps of
``notebooks/full_pipeline_YOLO_OBB.ipynb``:

* :mod:`config`            — class metadata and default hyperparameters.
* :mod:`data_generation`   — Step 1: synthetic data generation.
* :mod:`training`          — Step 2: training and evaluation.
* :mod:`inference`         — Step 3: adaptive tiled inference.
* :mod:`rendering`         — visualisation (overlays + reconstructed scheme).
* :mod:`label_studio`      — tile + emit ``tasks.json`` for Label Studio.
* :mod:`generation`        — upstream Gemini image generation.
* :mod:`pipeline`          — end-to-end orchestration.
"""

__version__: str = "0.1.0"
