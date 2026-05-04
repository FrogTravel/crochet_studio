# Crochet Studio

Crochet stitch detection and scheme reconstruction. Generate a crochet
diagram with Gemini (or upload your own), detect every stitch with a
YOLOv8n-OBB model, and export a reconstructed black-line scheme plus
structured JSON.

For a complete walk-through of the technological stack and the three
main pipeline steps, see
[`notebooks/full_pipeline_YOLO_OBB.ipynb`](notebooks/full_pipeline_YOLO_OBB.ipynb).

## Layout

```
crochet_studio/
├── main.py                  # CLI entry point (subcommands)
├── app.py                   # Streamlit entry point
├── README.md
├── requirements.txt
├── data/                    # raw photos, templates, synthetic dataset
├── notebooks/
│   └── full_pipeline_YOLO_OBB.ipynb
└── src/                     # all pipeline logic
    ├── config.py            # class metadata + default hyperparameters
    ├── data_generation.py   # Step 1: synthetic dataset
    ├── training.py          # Step 2: training + evaluation
    ├── inference.py         # Step 3: adaptive tiled inference
    ├── rendering.py         # overlays + reconstructed scheme
    ├── label_studio.py      # tile + emit tasks.json for Label Studio
    ├── generation.py        # upstream Gemini image generation
    └── pipeline.py          # end-to-end orchestration
```

## Quick start

```bash
pip install -r requirements.txt

# Step 1 — synthetic dataset
python main.py generate --output-dir data/synthetic

# Step 2 — train
python main.py train --data data/synthetic/data.yaml

# Step 3 — inference on a real image
python main.py infer --image data/raw/easy/0.png \
                     --weights runs/obb/obb_train/weights/best.pt

# Streamlit app
streamlit run app.py
```
