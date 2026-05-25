import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

import time
import chainfl.interact as ci


class ChallengeHandler(BaseHTTPRequestHandler):
    def _send(self, code: int, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        try:
            parsed = urlparse(self.path)
            qs = parse_qs(parsed.query)
            path = parsed.path

            if path == "/health":
                return self._send(200, {"ok": True})

            if path == "/register":
                client = qs.get("client", [None])[0]
                if not client:
                    return self._send(400, {"ok": False, "error": "missing client"})
                ok = ci.chain_proxy.pol_register_client(str(client))
                return self._send(200, {"ok": bool(ok)})

            if path == "/issue":
                client = qs.get("client", [None])[0]
                if not client:
                    return self._send(400, {"ok": False, "error": "missing client"})
                idx0 = int(qs.get("idx0", [0])[0])
                idx1 = int(qs.get("idx1", [1])[0])
                deadline = int(qs.get("deadline", [int(time.time()) + 3600])[0])
                cid = ci.chain_proxy.issue_challenge(str(client), idx0=idx0, idx1=idx1, deadline_ts=deadline)
                if not cid:
                    return self._send(500, {"ok": False, "error": "issue failed"})
                return self._send(200, {"ok": True, "cid": cid})

            if path == "/get":
                cid = qs.get("cid", [None])[0]
                if not cid:
                    return self._send(400, {"ok": False, "error": "missing cid"})
                info = ci.chain_proxy.get_challenge(str(cid))
                return self._send(200, {"ok": True, "challenge": info})

            return self._send(404, {"ok": False, "error": "not found"})
        except Exception as e:  # pragma: no cover
            return self._send(500, {"ok": False, "error": str(e)})


def start_server(host: str = "127.0.0.1", port: int = 0):
    httpd = HTTPServer((host, port), ChallengeHandler)
    th = threading.Thread(target=httpd.serve_forever, daemon=True)
    th.start()
    return httpd, th


def stop_server(httpd: HTTPServer):
    httpd.shutdown()
    httpd.server_close()

