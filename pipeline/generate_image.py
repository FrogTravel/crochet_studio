"""Generate a crochet stitch diagram image using Google Gemini.

Mirrors the logic of `gemini_image_generation.ipynb` but exposed as a
reusable function so it can be called from the pipeline runner.
"""

from __future__ import annotations

import os
from pathlib import Path


DEFAULT_PROMPT = (
    "A high-resolution, technical crochet stitch diagram (scheme) on a clean "
    "white background. The image features professional black line art showing "
    "intricate crochet symbols: chains (ovals), double crochets (T-shapes with "
    "slashes), and slip stitches (dots). The layout is a perfectly symmetrical "
    "circular mandala pattern, showing clear stitch intersections and "
    "structural details. Minimalist aesthetic, sharp vector-like lines, no "
    "text, no hands, no 3D yarn—only the 2D schematic symbols. Professional "
    "craft book style, 8k resolution, top-down flat lay view"
)

DEFAULT_MODEL = "gemini-3.1-flash-image-preview"


def generate_image(
    prompt: str = DEFAULT_PROMPT,
    output_path: str | Path = "free_output.png",
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
) -> Path:
    """Generate an image with Gemini and save it to ``output_path``.

    Parameters
    ----------
    prompt:
        Text prompt describing the crochet diagram to generate.
    output_path:
        Where the resulting PNG should be written.
    model:
        Gemini image model name. Defaults to the free-tier image preview.
    api_key:
        Optional API key. If omitted the function reads ``GEMINI_API_KEY``
        (and falls back to ``GOOGLE_API_KEY``) from the environment.

    Returns
    -------
    Path
        Path to the saved image.

    Raises
    ------
    RuntimeError
        If no image data is returned or the API call fails.
    """
    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError(
            "No Gemini API key provided. Set GEMINI_API_KEY in your environment "
            "or pass api_key=... to generate_image()."
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    from google import genai  # lazy import so --help works without the dep

    client = genai.Client(api_key=key)

    try:
        response = client.models.generate_content(model=model, contents=prompt)
    except Exception as exc:  # pragma: no cover - network dependent
        if "429" in str(exc):
            raise RuntimeError(
                "Gemini rate limit hit (10 requests/minute on the free tier)."
            ) from exc
        raise RuntimeError(f"Gemini generation failed: {exc}") from exc

    for part in response.candidates[0].content.parts:
        if part.inline_data:
            img = part.as_image()
            img.save(output_path)
            print(f"[generate_image] Saved image to {output_path}")
            return output_path

    raise RuntimeError(
        "No image data returned from Gemini. The prompt may have been filtered."
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate a crochet diagram with Gemini")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Generation prompt")
    parser.add_argument("--output", default="free_output.png", help="Output PNG path")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Gemini model name")
    args = parser.parse_args()

    generate_image(prompt=args.prompt, output_path=args.output, model=args.model)
