#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VerifierNode HTTP service (Phase A: distance-only verification over serialized checkpoints).
- POST /verify_pairs: JSON with base64-serialized response + commitment + pair_indices
- POST /verify_full:  JSON with base64-serialized response + commitment

It validates Merkle membership and checks parameter distance between successive
checkpoints against delta. Does NOT replay training (no dataset/model required).

Run:
  python -m server.committee.VerifierNode --host 127.0.0.1 --port 8088
  or set env POL_VERIFIER_HOST / POL_VERIFIER_PORT
"""
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
import argparse
import logging
import sys
from pathlib import Path
from typing import Tuple, List, Dict

# strict_replay enables torch deterministic algorithms inside PoLVerifier. Set
# this before any lazy torch import in request handlers so standalone verifier
# processes work the same way as launcher-managed ones.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

# Optional signing libs for Phase B receipts
try:
    from eth_account import Account  # type: ignore
    from eth_account.messages import encode_defunct  # type: ignore
except Exception:
    Account = None  # type: ignore
    encode_defunct = None  # type: ignore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VerifierNode")

_CODE_ROOT = Path(__file__).resolve().parents[2]
for _path in (
    _CODE_ROOT,
    _CODE_ROOT / "experiments",
    _CODE_ROOT / "experiments" / "scripts",
    _CODE_ROOT / "experiments" / "scripts" / "utils",
):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)


def _decode_response(b64_str: str) -> Dict:
    import base64, io, torch
    raw = base64.b64decode(b64_str.encode('utf-8'))
    buf = io.BytesIO(raw)
    try:
        return torch.load(buf, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(buf, map_location='cpu')


def _decode_torch_object(b64_str: str):
    import base64, io, torch
    raw = base64.b64decode(b64_str.encode('utf-8'))
    buf = io.BytesIO(raw)
    try:
        return torch.load(buf, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(buf, map_location='cpu')


def _distance_only_verify(response: Dict, commitment: str, pair_indices: List[int], params: Dict) -> Dict:
    from server.pol.PoLVerifier import PoLVerifier
    delta = float(params.get('delta', 0.01))
    metric = str(params.get('distance_metric', 'l2'))
    mpsr = float(params.get('min_pair_success_rate', 0.99))
    device = os.getenv('POL_VERIFIER_DEVICE', '').strip().lower()
    if not device:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device.startswith('cuda') and not torch.cuda.is_available():
        device = 'cpu'
    pv = PoLVerifier({'delta': delta, 'distance_metric': metric, 'device': device, 'min_pair_success_rate': mpsr})

    checkpoints = response.get('checkpoints', [])
    if not checkpoints or len(checkpoints) < 2:
        return {"valid": False, "reason": "insufficient_checkpoints", "pairs_total": 0, "pairs_passed": 0, "success_rate": 0.0}

    # Merkle membership
    if not pv._verify_merkle_membership(checkpoints, commitment):
        return {"valid": False, "reason": "merkle_fail", "pairs_total": 0, "pairs_passed": 0, "success_rate": 0.0}

    total_pairs = len(checkpoints) - 1 if not pair_indices else len(pair_indices)
    if not pair_indices:
        indices = list(range(len(checkpoints) - 1))
    else:
        indices = [i for i in sorted(set(int(x) for x in pair_indices)) if 0 <= i < len(checkpoints) - 1]
        if not indices:
            return {"valid": False, "reason": "empty_indices", "pairs_total": 0, "pairs_passed": 0, "success_rate": 0.0}

    passed = 0
    for i in indices:
        s1 = checkpoints[i]['data']['model_state']
        s2 = checkpoints[i + 1]['data']['model_state']
        dist = pv._compute_parameter_distance(s1, s2, metric)
        if dist <= delta:
            passed += 1

    success_rate = (passed / len(indices)) if indices else 0.0
    return {
        "valid": bool(success_rate >= mpsr),
        "pairs_total": len(indices),
        "pairs_passed": passed,
        "success_rate": success_rate,
        "mode": "distance_only"
    }


def _optimizer_class_from_meta(train_meta: Dict):
    import torch
    opt_name = str((train_meta or {}).get('optimizer', 'SGD')).lower()
    if 'adam' in opt_name:
        return torch.optim.Adam
    return torch.optim.SGD


def _build_dataloader(payload: Dict):
    from torch.utils.data import DataLoader
    dataset_b64 = payload.get('ser_dataset')
    dataloader_b64 = payload.get('ser_dataloader')
    if dataloader_b64:
        return _decode_torch_object(str(dataloader_b64))
    if not dataset_b64:
        raise ValueError("strict_replay missing ser_dataset")
    dataset = _decode_torch_object(str(dataset_b64))
    meta = dict(payload.get('loader_meta', {}) or {})
    batch_size = int(meta.get('batch_size', 1) or 1)
    drop_last = bool(meta.get('drop_last', False))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=drop_last, num_workers=0)


def _build_model(payload: Dict):
    ser_model = payload.get('ser_model')
    if ser_model:
        return _decode_torch_object(str(ser_model))
    spec = dict(payload.get('model_spec', {}) or {})
    if not spec:
        raise ValueError("strict_replay missing ser_model/model_spec")
    factory = str(spec.get('factory', ''))
    if factory != 'experiments.scripts.utils.models.create_model':
        raise ValueError(f"unsupported model factory: {factory}")
    from models import create_model
    model_name = str(spec.get('model_name', ''))
    num_classes = int(spec.get('num_classes', 10))
    input_channels = int(spec.get('input_channels', 3))
    return create_model(model_name, num_classes=num_classes, input_channels=input_channels)


def _strict_replay_verify(response: Dict, commitment: str, pair_indices: List[int], params: Dict, train_meta: Dict, payload: Dict) -> Dict:
    from server.pol.PoLVerifier import PoLVerifier
    import torch

    if not payload.get('ser_model') and not payload.get('model_spec'):
        return {"valid": False, "reason": "missing_strict_model", "mode": "strict_replay"}

    delta = float(params.get('delta', 0.01))
    metric = str(params.get('distance_metric', 'l2'))
    mpsr = float(params.get('min_pair_success_rate', 0.99))
    device = os.getenv('POL_VERIFIER_DEVICE', '').strip().lower()
    if not device:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device.startswith('cuda') and not torch.cuda.is_available():
        device = 'cpu'
    pv = PoLVerifier({'delta': delta, 'distance_metric': metric, 'device': device, 'min_pair_success_rate': mpsr})

    try:
        model = _build_model(payload)
    except Exception as e:
        return {"valid": False, "reason": f"model_build_failed: {e}", "mode": "strict_replay"}
    dataloader = _build_dataloader(payload)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer_class = _optimizer_class_from_meta(train_meta)
    lr = float((train_meta or {}).get('lr', 0.01))
    challenge = dict(payload.get('challenge', {}) or {})

    checkpoints = response.get('checkpoints', [])
    if len(checkpoints) < 2:
        return {"valid": False, "reason": "insufficient_checkpoints", "pairs_total": 0, "pairs_passed": 0, "success_rate": 0.0, "mode": "strict_replay"}

    if pair_indices:
        indices = [i for i in sorted(set(int(x) for x in pair_indices)) if 0 <= i < len(checkpoints) - 1]
        if not indices:
            return {"valid": False, "reason": "empty_indices", "pairs_total": 0, "pairs_passed": 0, "success_rate": 0.0, "mode": "strict_replay"}
        valid = bool(pv.verify_on_pairs_indices(
            challenge=challenge,
            response=response,
            commitment=commitment,
            model=model,
            dataloader=dataloader,
            criterion=criterion,
            optimizer_class=optimizer_class,
            lr=lr,
            pair_indices=indices,
        ))
        return {
            "valid": valid,
            "pairs_total": len(indices),
            "pairs_passed": len(indices) if valid else 0,
            "success_rate": 1.0 if valid else 0.0,
            "mode": "strict_replay",
            "strict_context": dict(payload.get('strict_context', {}) or {}),
        }

    valid = bool(pv.verify_response(
        challenge=challenge,
        response=response,
        commitment=commitment,
        model=model,
        dataloader=dataloader,
        criterion=criterion,
        optimizer_class=optimizer_class,
        lr=lr,
    ))
    n_pairs = max(0, len(checkpoints) - 1)
    return {
        "valid": valid,
        "pairs_total": n_pairs,
        "pairs_passed": n_pairs if valid else 0,
        "success_rate": 1.0 if valid else 0.0,
        "mode": "strict_replay",
        "strict_context": dict(payload.get('strict_context', {}) or {}),
    }


def _sign_receipt(msg: dict) -> dict:
    """Return {'msg': msg, 'sig': hex, 'addr': address} or {'msg': msg} if signing unavailable."""
    try:
        pk = os.getenv('VERIFIER_PRIV_KEY', '').strip()
        if not pk or Account is None or encode_defunct is None:
            if not pk:
                logger.warning("VERIFIER_PRIV_KEY not set; returning unsigned receipt")
            return {'msg': msg}
        # Normalize private key to hex with 0x prefix
        if not pk.startswith('0x'):
            pk_hex = '0x' + pk
        else:
            pk_hex = pk
        message_text = json.dumps(msg, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
        signed = Account.sign_message(encode_defunct(text=message_text), private_key=pk_hex)
        addr = Account.from_key(pk_hex).address
        return {'msg': msg, 'sig': signed.signature.hex(), 'addr': addr}
    except Exception as e:
        logger.warning(f"sign_receipt_error: {e}")
        return {'msg': msg}



class _JSONHandler(BaseHTTPRequestHandler):
    server_version = "VerifierNode/0.2"

    def _read_json(self) -> Tuple[dict, int]:
        try:
            length = int(self.headers.get('Content-Length', '0'))
            data = self.rfile.read(length) if length > 0 else b"{}"
            obj = json.loads(data.decode('utf-8') or '{}')
            return obj, 200
        except Exception as e:
            logger.exception("Failed to parse JSON body")
            return {"error": f"invalid_json: {e}"}, 400

    def _send_json(self, status: int, obj: dict):
        body = json.dumps(obj).encode('utf-8')
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):  # noqa: N802
        if self.path in ("/verify_pairs", "/verify_full"):
            payload, _ = self._read_json()
            try:
                b64 = payload.get('ser_response')
                if not b64:
                    raise ValueError("missing ser_response")
                response = _decode_response(b64)
                commitment = str(payload.get('commitment', ''))
                params = dict(payload.get('verifier_params', {}))
                if self.path == "/verify_pairs":
                    pair_indices = list(map(int, payload.get('pair_indices', []) or []))
                else:
                    pair_indices = []  # infer all pairs
                mode = str(payload.get('mode') or os.getenv('POL_REMOTE_MODE', 'distance_only')).lower()
                if mode == 'strict_replay':
                    result = _strict_replay_verify(response, commitment, pair_indices, params, dict(payload.get('train_meta', {}) or {}), payload)
                else:
                    result = _distance_only_verify(response, commitment, pair_indices, params)
                status = 200 if 'valid' in result else 500
                # Build signed receipt (Phase B)
                try:
                    rid = self.headers.get('X-Request-ID') or str(payload.get('request_id') or os.getenv('POL_REQUEST_ID',''))
                except Exception:
                    rid = ''
                try:
                    round_no = payload.get('round') or self.headers.get('X-Round')
                except Exception:
                    round_no = None
                try:
                    client_id = payload.get('client_id') or self.headers.get('X-Client-ID')
                except Exception:
                    client_id = None
                rmsg = {
                    'request_id': rid,
                    'round': round_no,
                    'client_id': client_id,
                    'commitmentRoot': commitment,
                    'pair_indices': list(pair_indices or []),
                    'verifier_params': {
                        'delta': float(params.get('delta', 0.01)),
                        'distance_metric': str(params.get('distance_metric', 'l2')),
                        'min_pair_success_rate': float(params.get('min_pair_success_rate', 0.99)),
                    },
                    'valid': bool(result.get('valid', False)),
                    'mode': str(result.get('mode', mode)),
                }
                out = dict(result)
                out['receipt'] = _sign_receipt(rmsg)
                self._send_json(status, out)
            except Exception as e:
                logger.exception("verification_error")
                self._send_json(500, {"error": f"verification_error: {e}"})
            return

        self._send_json(404, {"error": "not_found", "path": self.path})

    def do_GET(self):  # healthcheck
        if self.path in ("/health", "/", ""):
            self._send_json(200, {"ok": True, "service": "VerifierNode"})
            return
        self._send_json(404, {"error": "not_found", "path": self.path})

    # Silence default noisy logging
    def log_message(self, format, *args):  # noqa: A003
        logger.debug("%s - %s", self.address_string(), format % args)


def main():
    parser = argparse.ArgumentParser(description="VerifierNode HTTP service")
    parser.add_argument("--host", default=os.getenv("POL_VERIFIER_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("POL_VERIFIER_PORT", "8088")))
    args = parser.parse_args()

    server_address = (args.host, args.port)
    try:
        from http.server import ThreadingHTTPServer
        server_cls = ThreadingHTTPServer
    except Exception:
        server_cls = HTTPServer
    httpd = server_cls(server_address, _JSONHandler)
    logger.info("VerifierNode listening on http://%s:%d", args.host, args.port)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        logger.info("VerifierNode stopped")


if __name__ == "__main__":
    main()
