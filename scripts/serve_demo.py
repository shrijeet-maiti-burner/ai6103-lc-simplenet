"""Serve the local illumination inspection app; data and heads never leave localhost."""

import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from lc_simplenet.demo import InspectionRuntime, RecordedRuntime

ROOT = Path(__file__).resolve().parents[1]


def handler_for(runtime):
    class Handler(BaseHTTPRequestHandler):
        def local_host(self):
            try:
                host = urlsplit("http://" + self.headers.get("Host", ""))
                return (
                    host.hostname in {"127.0.0.1", "localhost"}
                    and (host.port or 80) == self.server.server_port
                )
            except ValueError:
                return False

        def reply(self, code, body, content_type="application/json"):
            if not isinstance(body, bytes):
                body = json.dumps(body, allow_nan=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            if not self.local_host():
                self.reply(403, {"error": "use the printed localhost address"})
                return
            route = urlsplit(self.path).path
            if route == "/api/catalog":
                self.reply(200, runtime.describe())
                return
            assets = {
                "/": "index.html",
                "/index.html": "index.html",
                "/app.js": "app.js",
                "/style.css": "style.css",
            }
            if route not in assets:
                self.reply(404, {"error": "not found"})
                return
            path = ROOT / "demo/dist" / assets[route]
            self.reply(200, path.read_bytes(), mimetypes.guess_type(path)[0] or "text/plain")

        def do_POST(self):  # noqa: N802
            if not self.local_host():
                self.reply(403, {"error": "use the printed localhost address"})
                return
            # Local UI only; cross-origin browser requests are rejected.
            origin = self.headers.get("Origin")
            if origin and origin != f"http://{self.headers.get('Host')}":
                self.reply(403, {"error": "cross-origin requests are unsupported"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                upload = self.path in {"/api/upload", "/api/upload-sweep"}
                if not 0 < length <= (24 * 2**20 if upload else 8192):
                    raise ValueError("invalid request size")
                request = json.loads(self.rfile.read(length))
                if self.path == "/api/frame":
                    self.reply(200, runtime.inspect(request))
                elif self.path == "/api/sweep":
                    self.reply(200, runtime.sweep(request))
                elif upload:
                    method = "inspect_upload" if self.path == "/api/upload" else "sweep_upload"
                    if not hasattr(runtime, method):
                        raise ValueError("local image upload requires the live model runtime")
                    self.reply(200, getattr(runtime, method)(request))
                else:
                    self.reply(404, {"error": "not found"})
            except (ValueError, TypeError, KeyError) as error:
                self.reply(400, {"error": str(error)})
            except Exception as error:
                self.reply(500, {"error": f"inference failed: {type(error).__name__}: {error}"})

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path)
    parser.add_argument("--root", type=Path, help="local ad 2 normal-data root")
    parser.add_argument("--recording", type=Path, help="serve measured offline predictions")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--port", type=int, default=18763)
    args = parser.parse_args()
    import torch

    torch.set_num_threads(args.threads)
    if args.recording:
        runtime = RecordedRuntime(args.recording)
    elif args.runs and args.root:
        runtime = InspectionRuntime(args.runs, args.root, args.device)
    else:
        parser.error("provide --recording, or both --runs and --root")

    class LocalServer(ThreadingHTTPServer):
        allow_reuse_address = False

    server = LocalServer(("127.0.0.1", args.port), handler_for(runtime))
    print(f"illumination lab: http://127.0.0.1:{server.server_port}", flush=True)
    print(runtime.describe()["scope"] + "; press ctrl+c to stop", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
