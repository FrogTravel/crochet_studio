"""Launch the Crochet Instruction Studio.

Example::

    python -m crochet.webapp                # http://127.0.0.1:8765
    python -m crochet.webapp --port 9000
    python -m crochet.webapp --host 0.0.0.0 --port 9000 --reload
"""

from __future__ import annotations

import os

# Router code plots on a worker thread; the default macOS matplotlib backend
# crashes off the main thread. Pin Agg before any other import pulls pyplot in.
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import webbrowser
from threading import Timer

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m crochet.webapp",
        description="Launch the Crochet Instruction Studio web app.",
    )
    parser.add_argument("--host", default="127.0.0.1",
                        help="Interface to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765,
                        help="Port to listen on (default: 8765)")
    parser.add_argument("--reload", action="store_true",
                        help="Reload on code changes (dev only)")
    parser.add_argument("--no-browser", action="store_true",
                        help="Do not open a browser tab on launch")
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}/"
    print(f"\n  Crochet Instruction Studio")
    print(f"  --------------------------")
    print(f"  Serving on {url}\n")

    if not args.no_browser:
        def _open() -> None:
            try:
                webbrowser.open(url)
            except Exception:
                pass
        Timer(1.2, _open).start()

    uvicorn.run(
        "crochet.webapp.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
