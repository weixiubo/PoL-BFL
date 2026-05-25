"""
Minimal AggregatorNode HTTP service (V0 minimal viable)
- POST /aggregate:
  - mode=summary_only: accept summary {selected_ids, passed_ids, failed_ids} and return ok/status
  - mode=weights_b64: accept base64-serialized list of state_dict and return aggregated_b64 (FedAvg)
- GET  /health: health check with recent latency stats
"""
from __future__ import annotations
import json, base64, io
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Dict, Any, List
import torch
import math

import time
from collections import deque

# recent aggregation latencies (seconds)
_RECENT_LATENCIES = deque(maxlen=64)
_LAST_ERROR = None

def _percentile(values, p):
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    # nearest-rank method
    k = int(round((p / 100.0) * (len(s) - 1)))
    k = max(0, min(k, len(s) - 1))
    return float(s[k])

def _health_payload():
    p50 = _percentile(list(_RECENT_LATENCIES), 50)
    p95 = _percentile(list(_RECENT_LATENCIES), 95)
    return {"ok": True, "service": "AggregatorNode", "recent_agg_latency_p50": p50, "recent_agg_latency_p95": p95, "last_error": _LAST_ERROR}

HOST = "127.0.0.1"
PORT = 8188


def _json_response(handler: BaseHTTPRequestHandler, code: int, payload: Dict[str, Any]):
    out = json.dumps(payload).encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(out)))
    handler.end_headers()
    handler.wfile.write(out)


def _normalize_weights(data: Dict[str, Any], num_models: int) -> List[float]:
    """Return explicit or data-size weights; default to equal FedAvg."""
    raw_weights = data.get("weights")
    raw_sizes = data.get("client_sizes")

    if raw_weights is None and raw_sizes is not None:
        raw_weights = raw_sizes

    if raw_weights is None:
        return [1.0 / float(num_models) for _ in range(num_models)]

    weights = [float(x) for x in list(raw_weights)]
    if len(weights) != num_models:
        raise ValueError(f"weights length {len(weights)} != models length {num_models}")
    if any((not math.isfinite(w)) or w < 0.0 for w in weights):
        raise ValueError("weights must be finite non-negative numbers")

    total = float(sum(weights))
    if total <= 0.0:
        raise ValueError("weights sum must be positive")
    return [w / total for w in weights]


def _weighted_fedavg(model_list: List[Dict[str, torch.Tensor]], weights: List[float]) -> Dict[str, torch.Tensor]:
    first = model_list[0]
    aggregated = {}
    for k in first.keys():
        acc = None
        for w, m in zip(weights, model_list):
            if k not in m:
                raise ValueError(f"model is missing key {k}")
            term = m[k] * float(w)
            acc = term.clone() if acc is None else acc + term
        aggregated[k] = acc
    return aggregated


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/health"):
            _json_response(self, 200, _health_payload())
            return
        _json_response(self, 404, {"error": "not_found"})

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
            data = json.loads(body or "{}")
        except Exception as e:
            _json_response(self, 400, {"error": f"bad_request: {e}"})
            return

        if self.path.startswith("/aggregate"):
            # Mode 1: real aggregation with weights (Phase B minimal viable)
            try:
                mode = str(data.get('mode', 'summary_only'))
            except Exception:
                mode = 'summary_only'

            if mode == 'weights_b64' and 'models_b64' in data:
                start = time.time()
                try:
                    raw = base64.b64decode(data['models_b64'].encode('ascii'))
                    buf = io.BytesIO(raw)
                    model_list = torch.load(buf)
                    if not isinstance(model_list, list) or not model_list:
                        _json_response(self, 400, {"ok": False, "error": "bad_models"})
                        return
                    num = len(model_list)
                    weights = _normalize_weights(data, num)
                    aggregated = _weighted_fedavg(model_list, weights)
                    out_buf = io.BytesIO()
                    torch.save(aggregated, out_buf)
                    out_b64 = base64.b64encode(out_buf.getvalue()).decode('ascii')
                    _json_response(self, 200, {
                        "ok": True,
                        "aggregated_b64": out_b64,
                        "weights_used": weights,
                    })
                    _RECENT_LATENCIES.append(max(0.0, time.time() - start))
                    return
                except Exception as e:
                    global _LAST_ERROR
                    _LAST_ERROR = f"aggregate_failed: {e}"
                    _json_response(self, 500, {"ok": False, "error": _LAST_ERROR})
                    return

            # Mode 2: summary only (existing behavior)
            try:
                sel = len(list(data.get('selected_ids', [])))
                pas = len(list(data.get('passed_ids', [])))
                fai = len(list(data.get('failed_ids', [])))
                print(f"AggregatorNode received summary: selected={sel}, passed={pas}, failed={fai}")
            except Exception:
                sel = pas = fai = None
            _json_response(self, 200, {"ok": True, "status": "received", "counts": {
                "selected": sel,
                "passed": pas,
                "failed": fai,
            }, "details": data})
            return

        _json_response(self, 404, {"error": "not_found"})


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", type=str, default=HOST)
    ap.add_argument("--port", type=int, default=PORT)
    args = ap.parse_args()

    server = HTTPServer((args.host, args.port), _Handler)
    print(f"AggregatorNode listening on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
