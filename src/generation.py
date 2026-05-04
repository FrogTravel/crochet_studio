"""Upstream image generation via Google Gemini."""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_PROMPT: str = (
    "A clean black-and-white crochet stitch diagram on white paper, "
    "showing a top-down view of a circular doily, no annotations, no text."
)
"""Prompt used when the caller does not provide one."""


def generate_image(prompt: str = DEFAULT_PROMPT,
                   output_path: str | Path = "free_output.png",
                   model: str = "gemini-3.1-flash-image-preview",
                   api_key: str | None = None) -> Path:
    """Generate a chart image with Gemini and save it to disk."""
    try:
        from google import genai
        from google.genai import types as genai_types
    except ImportError as exc:  # pragma: no cover — optional dep
        raise RuntimeError(
            "google-genai is not installed. "
            "Install it via `pip install google-genai`."
        ) from exc

    api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Pass api_key=... or set the env var."
        )

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=genai_types.GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"],
        ),
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for part in response.candidates[0].content.parts:
        if getattr(part, "inline_data", None) is None:
            continue
        output_path.write_bytes(part.inline_data.data)
        return output_path.resolve()

    raise RuntimeError("Gemini response contained no image data.")
