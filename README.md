# Crochet

Crochet stitch detection and scheme reconstruction. Generate a crochet
diagram with Gemini (or upload your own), detect every stitch with a
YOLOv8n OBB model, and export a reconstructed black-line scheme plus
structured JSON.

## Project layout

```
Crochet/
├── app.py                     # Streamlit entry point (public UI)
├── pages/                     # Streamlit multi-page pages
│   └── 1_Admin_Demo.py        # pipeline walk-through for demos
├── crochet/                   # Main package
│   ├── config.py              # CLASS_CONFIG, defaults
│   ├── detection.py           # Detection dataclass + tiled / adaptive YOLO inference
│   ├── generation.py          # Google Gemini image generation
│   ├── pipeline.py            # end-to-end generate + classify
│   ├── json_export.py         # detection → JSON
│   ├── dsl.py                 # JSON → CrochetPARADE DSL
│   ├── streamlit_ui.py        # shared UI components
│   ├── mcp_server.py          # CrochetDesigner MCP tool
│   ├── rendering/             # matplotlib + SVG renderers
│   │   ├── icons.py
│   │   ├── figures.py
│   │   └── svg.py
│   └── classical/             # classical-CV baseline (HOG + SVM)
│       ├── classifier.py
│       ├── scheme_parser.py
│       ├── extract_segments.py
│       ├── labeler.py
│       └── dataset.py
├── scripts/                   # CLI entry points
│   ├── run_pipeline.py
│   ├── classical_cli.py
│   └── json_to_crochetparade.py
├── tests/
├── notebooks/                 # exploratory notebooks
├── data/                      # raw + processed data
│   └── samples/               # input.json, label_studio_import.json
├── dataset/                   # YOLO dataset
├── runs/                      # YOLO training runs (weights under runs/obb/…)
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Quick start

```bash
# Install deps
pip install -r requirements.txt
# Optional extras
pip install google-genai mcp

# Run the Streamlit app
streamlit run app.py

# Run the CLI pipeline
export GEMINI_API_KEY=...
python scripts/run_pipeline.py \
    --image-out free_output.png \
    --figure-out free_output_predictions.png

# Re-run inference on an existing image (no Gemini call)
python scripts/run_pipeline.py --skip-generation --image-out free_output.png
```

## MCP server

Register with Claude Desktop:

```json
{
  "mcpServers": {
    "CrochetDesigner": {
      "command": "python",
      "args": ["-m", "crochet.mcp_server"]
    }
  }
}
```

End-to-end smoke test:

```bash
python tests/test_mcp_server.py
```

## Classical-CV baseline

```bash
# Extract + label crops, train, predict
python scripts/classical_cli.py parse data/raw/easy/2.png \
    --out data/segments/unlabeled
python scripts/classical_cli.py label \
    --crops data/segments/unlabeled --out data/segments/labeled
python scripts/classical_cli.py train \
    --data data/segments/labeled --model model.joblib
python scripts/classical_cli.py predict data/raw/easy/3.png \
    --model model.joblib --visualize
```
