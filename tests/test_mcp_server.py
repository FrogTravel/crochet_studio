"""End-to-end test for the CrochetDesigner MCP server.

Spawns the server as a subprocess, speaks MCP JSON-RPC over stdio as
Claude Desktop does, and exercises both the ``image_path`` and
``image_base64`` paths.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_IMAGE = REPO_ROOT / "data" / "raw" / "big" / "11.jpg"


class MCPClient:
    """Minimal MCP JSON-RPC client over stdio."""

    def __init__(self, server_cmd: list[str]) -> None:
        self.proc = subprocess.Popen(
            server_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(REPO_ROOT),
            text=False,
            bufsize=0,
        )
        self.next_id = 1

    def send(self, method: str, params: dict | None = None, notify: bool = False) -> dict | None:
        msg: dict = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        if not notify:
            msg["id"] = self.next_id
            self.next_id += 1
        data = (json.dumps(msg) + "\n").encode("utf-8")
        self.proc.stdin.write(data)
        self.proc.stdin.flush()
        if notify:
            return None
        return self._read_response()

    def _read_response(self, timeout: float = 120.0) -> dict:
        deadline = time.time() + timeout
        while time.time() < deadline:
            chunk = self.proc.stdout.readline()
            if not chunk:
                if self.proc.poll() is not None:
                    stderr = self.proc.stderr.read().decode("utf-8", errors="replace")
                    raise RuntimeError(
                        f"Server died unexpectedly (exit {self.proc.returncode}).\n"
                        f"STDERR:\n{stderr}"
                    )
                time.sleep(0.05)
                continue
            line = chunk.strip()
            if not line:
                continue
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                sys.stderr.write(f"[skipped non-JSON stdout line]: {line[:200]!r}\n")
                continue
        raise TimeoutError(f"No response within {timeout}s")

    def drain_stderr(self) -> str:
        """Non-blocking drain of anything in stderr."""
        import fcntl
        fd = self.proc.stderr.fileno()
        flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
        data = b""
        try:
            while True:
                chunk = self.proc.stderr.read(4096)
                if not chunk:
                    break
                data += chunk
        except Exception:
            pass
        return data.decode("utf-8", errors="replace")

    def close(self) -> None:
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def _banner(msg: str) -> None:
    print("\n" + "═" * 70)
    print(msg)
    print("═" * 70)


def _dump_content(resp: dict, show_rows: bool = False) -> None:
    if "error" in resp:
        print(f"✗ Tool error: {resp['error']}")
        return
    content = resp.get("result", {}).get("content", [])
    print(f"Got {len(content)} content items:")
    for c in content:
        if c.get("type") == "text":
            text = c["text"]
            print(f"  [text] {len(text)} chars")
            try:
                parsed = json.loads(text)
            except Exception:
                print(f"    preview: {text[:200]}")
                continue
            if "error" in parsed:
                print(f"    ERROR in payload: {parsed['error']}")
                continue
            print(f"    summary: {parsed.get('summary', {})}")
            if show_rows:
                print("    row_descriptions (first 3):")
                for rd in parsed.get("row_descriptions", [])[:3]:
                    print(f"      {rd[:120]}")
        elif c.get("type") == "image":
            print(f"  [image] {c.get('mimeType')} — {len(c.get('data', ''))} b64 chars")


def main() -> None:
    if not TEST_IMAGE.is_file():
        print(f"✗ Test image missing: {TEST_IMAGE}")
        sys.exit(1)
    print(f"Test image: {TEST_IMAGE} ({TEST_IMAGE.stat().st_size} bytes)")

    _banner("STEP 1 — Launch MCP server")
    client = MCPClient([sys.executable, "-m", "crochet.mcp_server"])
    print(f"Server PID: {client.proc.pid}")

    try:
        _banner("STEP 2 — Initialize handshake")
        resp = client.send("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test-client", "version": "0.0.0"},
        })
        print(json.dumps(resp, indent=2)[:500])
        assert "result" in resp, f"initialize failed: {resp}"

        client.send("notifications/initialized", notify=True)

        _banner("STEP 3 — List tools")
        resp = client.send("tools/list")
        tools = resp["result"]["tools"]
        print(f"Found {len(tools)} tool(s):")
        for t in tools:
            print(f"  - {t['name']}: {t.get('description', '')[:100]}")
            print(f"      inputSchema: {json.dumps(t['inputSchema'])[:200]}")
        assert any(t["name"] == "analyze_crochet_chart" for t in tools), "tool not found"

        _banner("STEP 4 — Call tool via image_path")
        resp = client.send("tools/call", {
            "name": "analyze_crochet_chart",
            "arguments": {"image_path": str(TEST_IMAGE)},
        })
        _dump_content(resp)

        _banner("STEP 5 — Call tool via image_base64")
        b64 = base64.b64encode(TEST_IMAGE.read_bytes()).decode("ascii")
        print(f"Encoded to {len(b64)} b64 chars")
        resp = client.send("tools/call", {
            "name": "analyze_crochet_chart",
            "arguments": {"image_base64": b64},
        })
        _dump_content(resp, show_rows=True)

        _banner("STEP 6 — Call tool with neither arg (should error cleanly)")
        resp = client.send("tools/call", {
            "name": "analyze_crochet_chart",
            "arguments": {},
        })
        for c in resp.get("result", {}).get("content", []):
            if c.get("type") == "text":
                print(f"  [text] {c['text'][:300]}")

        _banner("STEP 7 — Server stderr log")
        log = client.drain_stderr()
        print(log[-2000:] if log.strip() else "(empty)")
        _banner("DONE")
    finally:
        client.close()


if __name__ == "__main__":
    main()
