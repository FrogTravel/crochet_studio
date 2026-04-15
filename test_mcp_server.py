"""
End-to-end test for crochet_tool.py MCP server.

Spawns the server as a subprocess, speaks MCP JSON-RPC over stdio exactly like
Claude Desktop does, and exercises both `image_path` and `image_base64` paths.
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SERVER_PATH = os.path.join(SCRIPT_DIR, "crochet_tool.py")
TEST_IMAGE  = os.path.join(SCRIPT_DIR, "data", "raw", "big", "11.jpg")


class MCPClient:
    """Minimal MCP JSON-RPC client over stdio."""
    def __init__(self, server_cmd: list[str]):
        self.proc = subprocess.Popen(
            server_cmd,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd=SCRIPT_DIR, text=False, bufsize=0,
        )
        self.next_id = 1

    def send(self, method: str, params: dict | None = None, notify: bool = False) -> dict | None:
        msg = {"jsonrpc": "2.0", "method": method}
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
        line = b""
        while time.time() < deadline:
            chunk = self.proc.stdout.readline()
            if not chunk:
                # Process may have exited
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
                # Some servers send log lines — ignore non-JSON
                sys.stderr.write(f"[skipped non-JSON stdout line]: {line[:200]!r}\n")
                continue
        raise TimeoutError(f"No response within {timeout}s")

    def drain_stderr(self) -> str:
        """Non-blocking drain of anything in stderr."""
        import fcntl, os as _os
        fd = self.proc.stderr.fileno()
        flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, flags | _os.O_NONBLOCK)
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

    def close(self):
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def banner(msg):
    print("\n" + "═" * 70)
    print(msg)
    print("═" * 70)


def main():
    # Verify test image exists
    if not os.path.isfile(TEST_IMAGE):
        print(f"✗ Test image missing: {TEST_IMAGE}")
        sys.exit(1)
    print(f"Test image: {TEST_IMAGE} ({os.path.getsize(TEST_IMAGE)} bytes)")

    banner("STEP 1 — Launch MCP server")
    client = MCPClient([sys.executable, SERVER_PATH])
    print(f"Server PID: {client.proc.pid}")

    try:
        banner("STEP 2 — Initialize handshake")
        resp = client.send("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test-client", "version": "0.0.0"},
        })
        print(json.dumps(resp, indent=2)[:500])
        assert "result" in resp, f"initialize failed: {resp}"

        client.send("notifications/initialized", notify=True)

        banner("STEP 3 — List tools")
        resp = client.send("tools/list")
        tools = resp["result"]["tools"]
        print(f"Found {len(tools)} tool(s):")
        for t in tools:
            print(f"  - {t['name']}: {t.get('description','')[:100]}")
            print(f"      inputSchema: {json.dumps(t['inputSchema'])[:200]}")
        assert any(t["name"] == "analyze_crochet_chart" for t in tools), "tool not found"

        banner("STEP 4 — Call tool via image_path")
        resp = client.send("tools/call", {
            "name": "analyze_crochet_chart",
            "arguments": {"image_path": TEST_IMAGE},
        })
        if "error" in resp:
            print(f"✗ Tool error: {resp['error']}")
        else:
            content = resp["result"].get("content", [])
            print(f"Got {len(content)} content items:")
            for c in content:
                if c.get("type") == "text":
                    text = c["text"]
                    print(f"  [text] {len(text)} chars")
                    try:
                        parsed = json.loads(text)
                        if "error" in parsed:
                            print(f"    ERROR in payload: {parsed['error']}")
                        else:
                            print(f"    summary: {parsed.get('summary', {})}")
                    except Exception:
                        print(f"    preview: {text[:200]}")
                elif c.get("type") == "image":
                    print(f"  [image] {c.get('mimeType')} — {len(c.get('data',''))} b64 chars")

        banner("STEP 5 — Call tool via image_base64")
        with open(TEST_IMAGE, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        print(f"Encoded to {len(b64)} b64 chars")
        resp = client.send("tools/call", {
            "name": "analyze_crochet_chart",
            "arguments": {"image_base64": b64},
        })
        if "error" in resp:
            print(f"✗ Tool error: {resp['error']}")
        else:
            content = resp["result"].get("content", [])
            print(f"Got {len(content)} content items:")
            for c in content:
                if c.get("type") == "text":
                    print(f"  [text] {len(c['text'])} chars")
                    try:
                        parsed = json.loads(c["text"])
                        if "error" in parsed:
                            print(f"    ERROR: {parsed['error']}")
                        else:
                            print(f"    summary: {parsed.get('summary', {})}")
                            print(f"    row_descriptions (first 3):")
                            for rd in parsed.get("row_descriptions", [])[:3]:
                                print(f"      {rd[:120]}")
                    except Exception as e:
                        print(f"    parse failed: {e}")
                        print(f"    preview: {c['text'][:300]}")
                elif c.get("type") == "image":
                    print(f"  [image] {c.get('mimeType')} — {len(c.get('data',''))} b64 chars")

        banner("STEP 6 — Call tool with neither arg (should error cleanly)")
        resp = client.send("tools/call", {
            "name": "analyze_crochet_chart",
            "arguments": {},
        })
        content = resp.get("result", {}).get("content", [])
        for c in content:
            if c.get("type") == "text":
                print(f"  [text] {c['text'][:300]}")

        banner("STEP 7 — Server stderr log")
        log = client.drain_stderr()
        if log.strip():
            print(log[-2000:])
        else:
            print("(empty)")

        banner("DONE")
    finally:
        client.close()


if __name__ == "__main__":
    main()
