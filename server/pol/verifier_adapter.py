"""
Verifier Adapter abstraction for PoL verification.

Stage A (MVP):
- LocalVerifierAdapter wraps existing PoLVerifier for in-process verification
- RemoteVerifierAdapter posts JSON to a remote verifier node (HTTP) and returns boolean
  - If remote call fails, callers should fallback to LocalVerifierAdapter

No external dependencies required (uses urllib from stdlib).
"""
from __future__ import annotations
import os
import json
import logging
from typing import List, Dict, Optional, Any

try:
    # Optional typing only
    from collections import OrderedDict  # noqa: F401
except Exception:
    pass

logger = logging.getLogger(__name__)


class VerifierAdapter:
    """Abstract adapter interface for verification backends."""

    def verify_on_pairs_indices(
        self,
        challenge: Dict,
        response: Dict,
        commitment: str,
        model: Any,
        dataloader: Any,
        criterion: Any,
        optimizer_class: Any,
        lr: float,
        pair_indices: List[int],
    ) -> bool:
        raise NotImplementedError

    def verify_response(
        self,
        challenge: Dict,
        response: Dict,
        commitment: str,
        model: Any,
        dataloader: Any,
        criterion: Any,
        optimizer_class: Any,
        lr: float,
    ) -> bool:
        raise NotImplementedError


class LocalVerifierAdapter(VerifierAdapter):
    """Thin wrapper around the in-process PoLVerifier."""

    def __init__(self, pol_verifier: Any):
        self._pv = pol_verifier

    def verify_on_pairs_indices(self, challenge, response, commitment, model, dataloader, criterion, optimizer_class, lr, pair_indices):
        return bool(self._pv.verify_on_pairs_indices(
            challenge=challenge,
            response=response,
            commitment=commitment,
            model=model,
            dataloader=dataloader,
            criterion=criterion,
            optimizer_class=optimizer_class,
            lr=lr,
            pair_indices=pair_indices,
        ))

    def verify_response(self, challenge, response, commitment, model, dataloader, criterion, optimizer_class, lr):
        return bool(self._pv.verify_response(
            challenge=challenge,
            response=response,
            commitment=commitment,
            model=model,
            dataloader=dataloader,
            criterion=criterion,
            optimizer_class=optimizer_class,
            lr=lr,
        ))


class RemoteVerifierAdapter(VerifierAdapter):
    """
    Minimal HTTP client for remote verification.
    Expects a verifier node exposing:
      - POST /verify_pairs
      - POST /verify_full
    Payload and response are JSON.

    MVP transport: serialize heavy Python objects (checkpoints with tensors)
    using torch.save and base64, avoiding custom JSON encoders.
    The remote node will run a distance-only check by default.
    """

    def __init__(self, endpoints: List[str], timeout_sec: int = 60, verifier_params: Optional[Dict] = None, mode: str = 'distance_only'):
        self._endpoints = list(endpoints or [])
        # allow env override for timeout
        try:
            timeout_env = int(os.getenv('POL_REMOTE_TIMEOUT_SEC', '0'))
            if timeout_env > 0:
                timeout_sec = timeout_env
        except Exception:
            pass
        self._timeout = timeout_sec
        self._verifier_params = verifier_params or {}
        self._mode = mode
        # health tracking
        self._health = {ep: True for ep in self._endpoints}
        self._fails = {ep: 0 for ep in self._endpoints}
        self._last_fail_ts = {ep: 0.0 for ep in self._endpoints}
        self._rr = 0
        self._max_fails = int(os.getenv('POL_REMOTE_MAX_FAILS', '2'))
        self._cooldown_sec = int(os.getenv('POL_REMOTE_COOLDOWN_SEC', '60'))
        # observability (per-round; aggregator will snapshot and reset)
        self._latencies: list = []
        self._error_counts: Dict[str, int] = {'timeout': 0, 'network': 0, 'invalid': 0, 'business': 0}
    def metrics_snapshot_and_reset(self) -> Dict:
        """Return a snapshot of per-round observability metrics and reset counters.
        Keys: latencies (list of floats), error_counts (dict).
        """
        try:
            snap = {
                'latencies': list(self._latencies or []),
                'error_counts': dict(self._error_counts or {}),
            }
        except Exception:
            snap = {'latencies': [], 'error_counts': {}}
        # reset
        self._latencies = []
        self._error_counts = {'timeout': 0, 'network': 0, 'invalid': 0, 'business': 0}
        return snap


    def _now(self) -> float:
        import time
        return time.time()

    def _is_healthy(self, ep: str) -> bool:
        # cooldown-based half-open
        if not self._health.get(ep, True):
            last = self._last_fail_ts.get(ep, 0.0)
            if self._now() - last >= self._cooldown_sec:
                # recover
                self._health[ep] = True
                self._fails[ep] = 0
                logger.info(f"Remote verifier recovered from cooldown: {ep}")
        return self._health.get(ep, True)

    def _record_success(self, ep: str):
        self._health[ep] = True
        self._fails[ep] = 0

    def _record_failure(self, ep: str):
        self._fails[ep] = int(self._fails.get(ep, 0)) + 1
        if self._fails[ep] >= self._max_fails and self._health.get(ep, True):
            self._health[ep] = False
            self._last_fail_ts[ep] = self._now()
            logger.warning(f"Remote verifier circuit open: {ep} (fails={self._fails[ep]} >= max={self._max_fails})")

    def _healthy_endpoints(self) -> List[str]:
        return [ep for ep in self._endpoints if self._is_healthy(ep)]

    def _post(self, path: str, payload: Dict) -> Optional[Dict]:
        if not self._endpoints:
            raise RuntimeError("No verifier endpoints configured")
        # Strategy: any (default) tries endpoints round-robin until one succeeds;
        # majority will query all endpoints and return the majority vote of 'valid'.
        try:
            strategy = str(os.getenv('POL_REMOTE_STRATEGY', 'any')).lower()
        except Exception:
            strategy = 'any'
        import urllib.request, socket, time
        # attach request context id if present
        try:
            _rid = os.getenv('POL_REQUEST_ID', '').strip()
        except Exception:
            _rid = ''
        try:
            if _rid:
                payload = dict(payload)
                payload['request_id'] = _rid
        except Exception:
            pass
        data = json.dumps(payload).encode('utf-8')

        def _classify_error(ex: Exception) -> str:
            msg = str(ex).lower()
            if isinstance(ex, socket.timeout) or 'timed out' in msg:
                return 'timeout'
            # urllib error types
            if getattr(ex, 'reason', None) is not None:
                return 'network'
            if 'http error' in msg or '404' in msg or '500' in msg:
                return 'network'
            return 'business'

        def _call(ep: str) -> Optional[Dict]:
            url = ep.rstrip('/') + path
            t0 = time.time()
            try:
                headers = {'Content-Type': 'application/json'}
                if _rid:
                    headers['X-Request-ID'] = _rid
                req = urllib.request.Request(url, data=data, headers=headers)
                with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                    body = resp.read().decode('utf-8')
                    out = json.loads(body)
                    # record latency on success only
                    self._latencies.append(max(0.0, time.time() - t0))
                    return out
            except Exception as e:
                # record latency for failed attempt as well
                self._latencies.append(max(0.0, time.time() - t0))
                et = _classify_error(e)
                try:
                    self._error_counts[et] = int(self._error_counts.get(et, 0)) + 1
                except Exception:
                    pass
                if _rid:
                    logger.warning(f"[ctx={_rid}] Remote verifier POST {url} failed: {e}")
                else:
                    logger.warning(f"Remote verifier POST {url} failed: {e}")
                return None

        if strategy == 'majority':
            votes = []
            responders = 0
            receipts = []
            eps = self._healthy_endpoints()
            if not eps:
                return None
            for ep in eps:
                res = _call(ep)
                if res is not None and 'valid' in res:
                    responders += 1
                    votes.append(bool(res['valid']))
                    # collect receipt when present
                    try:
                        rcpt = res.get('receipt')
                        if rcpt:
                            receipts.append(rcpt)
                    except Exception:
                        pass
                    self._record_success(ep)
                else:
                    self._record_failure(ep)
            if responders == 0:
                return None
            yes = votes.count(True)
            valid = yes > (len(votes) // 2)
            logger.info(f"Remote verifier majority: responders={responders}, yes={yes}, valid={valid}")
            out = {'valid': valid, 'responders': responders, 'yes': yes}
            if receipts:
                out['receipts'] = receipts
            return out
        else:
            start = getattr(self, '_rr', 0)
            n = len(self._endpoints)
            tried = 0
            for i in range(n):
                ep = self._endpoints[(start + i) % n]
                if not self._is_healthy(ep):
                    continue
                tried += 1
                res = _call(ep)
                if res is not None:
                    # advance round-robin pointer to next after this ep
                    self._record_success(ep)
                    self._rr = ((start + i + 1) % n)
                    logger.info(f"Remote verifier succeeded via {ep}")
                    return res
                else:
                    self._record_failure(ep)
            return None

    def _compact_response_for_pairs(self, response: Dict, pair_indices: Optional[List[int]] = None) -> Dict:
        """
        Remove optimizer states that strict replay will never consume.

        Replay of pair i -> i+1 needs checkpoint i optimizer_state, but not the
        target checkpoint's optimizer_state. Merkle verification hashes only
        model_state, so pruning unused optimizer states preserves integrity while
        shrinking the remote payload substantially for momentum optimizers.
        """
        try:
            enabled = str(os.getenv('POL_COMPACT_REMOTE_RESPONSE', '1')).strip().lower() in ('1', 'true', 'yes', 'on')
            if not enabled:
                return response
            checkpoints = list((response or {}).get('checkpoints', []) or [])
            if not checkpoints:
                return response
            if pair_indices:
                keep_optimizer_at = {
                    int(i) for i in pair_indices
                    if 0 <= int(i) < max(0, len(checkpoints) - 1)
                }
            else:
                keep_optimizer_at = set(range(max(0, len(checkpoints) - 1)))
            compact_checkpoints = []
            for pos, ckpt in enumerate(checkpoints):
                ckpt_copy = dict(ckpt)
                data = dict(ckpt_copy.get('data', {}) or {})
                if pos not in keep_optimizer_at:
                    data.pop('optimizer_state', None)
                ckpt_copy['data'] = data
                compact_checkpoints.append(ckpt_copy)
            compact = dict(response)
            compact['checkpoints'] = compact_checkpoints
            compact.setdefault('transport_hints', {})
            try:
                compact['transport_hints'] = dict(compact.get('transport_hints') or {})
                compact['transport_hints']['optimizer_state_kept_at'] = sorted(keep_optimizer_at)
                compact['transport_hints']['response_compacted'] = True
            except Exception:
                pass
            return compact
        except Exception:
            return response

    def _serialize_response(self, response: Dict, pair_indices: Optional[List[int]] = None) -> str:
        """Serialize response (with tensors) into base64 string using torch.save."""
        try:
            import io, base64, torch
            buf = io.BytesIO()
            response = self._compact_response_for_pairs(response, pair_indices=pair_indices)
            # Use CPU tensors to minimize device-specific state
            torch.save(response, buf)
            return base64.b64encode(buf.getvalue()).decode('utf-8')
        except Exception as e:
            logger.warning(f"Serialize response failed: {e}")
            raise

    def _serialize_object(self, obj: Any) -> str:
        """Serialize a replay dependency for strict remote verification."""
        try:
            import io, base64, torch
            buf = io.BytesIO()
            torch.save(obj, buf)
            return base64.b64encode(buf.getvalue()).decode('utf-8')
        except Exception as e:
            logger.warning(f"Serialize strict replay object failed: {e}")
            raise

    def _model_spec(self, model: Any) -> Optional[Dict[str, Any]]:
        """Describe known experiment models so strict replay need not pickle weights."""
        try:
            model_name = str(getattr(model, '__class__', type(model)).__name__)
            num_classes = None
            input_channels = None
            fc = getattr(model, 'fc', None)
            if fc is not None and hasattr(fc, 'out_features'):
                num_classes = int(fc.out_features)
            fc2 = getattr(model, 'fc2', None)
            if num_classes is None and fc2 is not None and hasattr(fc2, 'out_features'):
                num_classes = int(fc2.out_features)
            classifier = getattr(model, 'classifier', None)
            if num_classes is None and classifier is not None:
                try:
                    last = list(classifier.children())[-1]
                    if hasattr(last, 'out_features'):
                        num_classes = int(last.out_features)
                except Exception:
                    pass
            conv1 = getattr(model, 'conv1', None)
            if conv1 is not None and hasattr(conv1, 'in_channels'):
                input_channels = int(conv1.in_channels)
            if model_name not in {'SimpleCNN', 'ResNet18', 'ResNet34', 'VGG11'}:
                return None
            return {
                'factory': 'experiments.scripts.utils.models.create_model',
                'model_name': model_name,
                'num_classes': int(num_classes if num_classes is not None else 10),
                'input_channels': int(input_channels if input_channels is not None else 3),
            }
        except Exception:
            return None

    def _strict_replay_fields(self, model: Any, dataloader: Any, criterion: Any) -> Dict[str, Any]:
        dataset = getattr(dataloader, 'dataset', None)
        if model is None or dataset is None:
            raise ValueError("strict_replay requires model and dataloader.dataset")
        loader_meta = {
            'batch_size': int(getattr(dataloader, 'batch_size', 1) or 1),
            'drop_last': bool(getattr(dataloader, 'drop_last', False)),
            'shuffle': False,
        }
        try:
            loader_meta['num_workers'] = int(getattr(dataloader, 'num_workers', 0) or 0)
        except Exception:
            loader_meta['num_workers'] = 0
        indices = []
        try:
            raw_indices = getattr(dataset, 'indices', None)
            if raw_indices is not None:
                indices = [int(x) for x in list(raw_indices)]
        except Exception:
            indices = []
        indices_hash = ''
        if indices:
            try:
                import hashlib
                h = hashlib.sha256()
                for idx in indices:
                    h.update(str(int(idx)).encode('utf-8'))
                    h.update(b',')
                indices_hash = h.hexdigest()
            except Exception:
                indices_hash = ''
        model_spec = self._model_spec(model)
        fields = {
            'model_spec': model_spec,
            'ser_dataset': self._serialize_object(dataset),
            'loader_meta': loader_meta,
            'strict_context': {
                'transport': 'model_spec_plus_torch_pickle_dataset',
                'criterion': str(getattr(criterion, '__class__', type(criterion)).__name__),
                'dataset_class': str(getattr(dataset, '__class__', type(dataset)).__name__),
                'dataset_size': len(dataset) if hasattr(dataset, '__len__') else None,
                'partition_indices_hash': indices_hash,
                'partition_size': len(indices),
                'model_class': str(getattr(model, '__class__', type(model)).__name__),
                'seed': os.getenv('POL_DET_SEED') or os.getenv('PYTHONHASHSEED') or '',
            },
        }
        if model_spec is None or os.getenv('POL_STRICT_REPLAY_SERIALIZE_MODEL', '0') == '1':
            fields['ser_model'] = self._serialize_object(model)
            fields['strict_context']['transport'] = 'torch_pickle_inline'
        return fields

    def verify_on_pairs_indices(self, challenge, response, commitment, model, dataloader, criterion, optimizer_class, lr, pair_indices):
        payload = {
            'mode': self._mode,
            'challenge': challenge or {},
            'ser_response': self._serialize_response(response, pair_indices=pair_indices),
            'commitment': commitment,
            'verifier_params': {
                'delta': float(self._verifier_params.get('delta', 0.01)),
                'distance_metric': str(self._verifier_params.get('distance_metric', 'l2')),
                'min_pair_success_rate': float(self._verifier_params.get('min_pair_success_rate', 0.99)),
            },
            'train_meta': {
                'optimizer': str(getattr(optimizer_class, '__name__', 'SGD')),
                'lr': float(lr),
            },
            'pair_indices': list(map(int, pair_indices or [])),
        }
        if str(self._mode).lower() == 'strict_replay':
            payload.update(self._strict_replay_fields(model, dataloader, criterion))
        res = self._post('/verify_pairs', payload)
        if not res or 'valid' not in res:
            raise RuntimeError('Remote verifier returned no result')
        # Return full response dict when available (e.g., majority strategy provides
        # {'valid': bool, 'responders': int, 'yes': int}); fallback to boolean.
        return res if isinstance(res, dict) else bool(res)

    def verify_response(self, challenge, response, commitment, model, dataloader, criterion, optimizer_class, lr):
        payload = {
            'mode': self._mode,
            'challenge': challenge or {},
            'ser_response': self._serialize_response(response),
            'commitment': commitment,
            'verifier_params': {
                'delta': float(self._verifier_params.get('delta', 0.01)),
                'distance_metric': str(self._verifier_params.get('distance_metric', 'l2')),
                'min_pair_success_rate': float(self._verifier_params.get('min_pair_success_rate', 0.99)),
            },
            'train_meta': {
                'optimizer': str(getattr(optimizer_class, '__name__', 'SGD')),
                'lr': float(lr),
            },
        }
        if str(self._mode).lower() == 'strict_replay':
            payload.update(self._strict_replay_fields(model, dataloader, criterion))
        res = self._post('/verify_full', payload)
        if not res or 'valid' not in res:
            raise RuntimeError('Remote verifier returned no result')
        # Return full response dict when available; fallback to boolean.
        return res if isinstance(res, dict) else bool(res)
