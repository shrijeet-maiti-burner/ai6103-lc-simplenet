"""Exercise the actual local HTTP boundary without downloading a model."""

import importlib.util
import json
import threading
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("serve_demo", ROOT / "scripts/serve_demo.py")
server_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server_module)


class Runtime:
    def describe(self):
        return {"mode": "recorded predictions"}

    def inspect(self, request):
        if not isinstance(request, dict):
            raise ValueError("invalid inspection request")
        return {"query": request, "mode": "recorded predictions"}

    def sweep(self, request):
        return {"rows": [self.inspect(request)] * 9}


@pytest.fixture
def connection():
    server = ThreadingHTTPServer(("127.0.0.1", 0), server_module.handler_for(Runtime()))
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    client = HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    try:
        yield client
    finally:
        client.close()
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)


def request(client, method, path, body=None, headers=None):
    client.request(method, path, body, headers or {})
    response = client.getresponse()
    return response.status, dict(response.getheaders()), response.read()


def test_static_assets_catalog_and_measured_record_routes(connection):
    for path in ("/", "/app.js", "/style.css"):
        code, headers, body = request(connection, "GET", path)
        assert code == 200 and body
        assert headers["Cache-Control"] == "no-store"
        assert headers["X-Content-Type-Options"] == "nosniff"
    code, _, body = request(connection, "GET", "/api/catalog")
    assert code == 200 and json.loads(body)["mode"] == "recorded predictions"
    for path in ("/api/frame", "/api/sweep"):
        code, _, body = request(connection, "POST", path, "{}")
        assert code == 200
        assert json.loads(body)


@pytest.mark.parametrize("body", ["invalid json", "[]", "x" * 8193, ""])
def test_invalid_requests_return_json_without_inference(connection, body):
    code, _, data = request(connection, "POST", "/api/frame", body)
    assert code == 400 and json.loads(data)["error"]


def test_cross_origin_and_arbitrary_filesystem_routes_rejected(connection):
    code, _, _ = request(
        connection, "POST", "/api/frame", "{}", {"Origin": "https://unrelated.example"}
    )
    assert code == 403
    for path in ("/../AGENTS.md", "/results/private.json", "/api/unknown"):
        assert request(connection, "GET", path)[0] == 404
    assert request(connection, "POST", "/api/unknown", "{}")[0] == 404
