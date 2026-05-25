import json, base64, io, threading, time
from http.server import HTTPServer
from server.committee.AggregatorNode import _Handler
import torch

# Start a local AggregatorNode on ephemeral port

def _run_server(srv):
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass

def test_aggregate_summary_only():
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    host, port = srv.server_address
    th = threading.Thread(target=_run_server, args=(srv,), daemon=True)
    th.start()
    try:
        import urllib.request
        req = urllib.request.Request(
            f"http://{host}:{port}/aggregate",
            data=json.dumps({"mode":"summary_only","selected_ids":[1,2,3],"passed_ids":[1,2],"failed_ids":[3]}).encode("utf-8"),
            headers={"Content-Type":"application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as r:
            body = json.loads(r.read().decode("utf-8"))
            assert body.get("ok") is True
            assert body.get("counts",{}).get("selected") == 3
    finally:
        srv.shutdown(); th.join(timeout=2)

def test_aggregate_weights_b64():
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    host, port = srv.server_address
    th = threading.Thread(target=_run_server, args=(srv,), daemon=True)
    th.start()
    try:
        # two tiny state_dicts
        sd1 = {"w": torch.ones(3)}
        sd2 = {"w": torch.zeros(3)}
        buf = io.BytesIO(); torch.save([sd1, sd2], buf)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        import urllib.request
        req = urllib.request.Request(
            f"http://{host}:{port}/aggregate",
            data=json.dumps({"mode":"weights_b64","models_b64": b64}).encode("utf-8"),
            headers={"Content-Type":"application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as r:
            body = json.loads(r.read().decode("utf-8"))
            assert body.get("ok") is True
            out_b64 = body.get("aggregated_b64")
            assert isinstance(out_b64, str) and len(out_b64) > 0
            out = torch.load(io.BytesIO(base64.b64decode(out_b64.encode("ascii"))))
            assert torch.allclose(out["w"], torch.full((3,), 0.5))
    finally:
        srv.shutdown(); th.join(timeout=2)

def test_aggregate_weights_b64_weighted():
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    host, port = srv.server_address
    th = threading.Thread(target=_run_server, args=(srv,), daemon=True)
    th.start()
    try:
        sd1 = {"w": torch.ones(3)}
        sd2 = {"w": torch.zeros(3)}
        buf = io.BytesIO(); torch.save([sd1, sd2], buf)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        import urllib.request
        req = urllib.request.Request(
            f"http://{host}:{port}/aggregate",
            data=json.dumps({"mode":"weights_b64","models_b64": b64, "client_sizes": [3, 1]}).encode("utf-8"),
            headers={"Content-Type":"application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as r:
            body = json.loads(r.read().decode("utf-8"))
            assert body.get("ok") is True
            assert body.get("weights_used") == [0.75, 0.25]
            out_b64 = body.get("aggregated_b64")
            out = torch.load(io.BytesIO(base64.b64decode(out_b64.encode("ascii"))))
            assert torch.allclose(out["w"], torch.full((3,), 0.75))
    finally:
        srv.shutdown(); th.join(timeout=2)
