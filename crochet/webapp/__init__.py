"""Crochet Instruction Studio - a FastAPI-powered standalone web app.

Run with::

    python -m crochet.webapp             # default http://127.0.0.1:8765
    python -m crochet.webapp --port 9000

The app wraps :mod:`crochet.routing` behind a single-page frontend that
matches the typographic style of the generated tutorials (Cormorant
Garamond + Jost, cream/deep palette). Progress streams back to the
browser as Server-Sent Events so the user sees each phase, generated
image, and palette update the moment they happen.
"""

from .server import app  # noqa: F401

__all__ = ["app"]
