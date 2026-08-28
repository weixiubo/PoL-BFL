"""
PoL验证聚合器
继承ServerAggregator，集成PoL验证逻辑
"""

import os
import math
import random
import logging
from typing import List, Dict, OrderedDict
import torch

import time

from server.base.baseAggregator import ServerAggregator
from server.pol.PoLVerifier import PoLVerifier
from server.pol.SybilDetector import SybilDetector
from server.zkp.ZKPVerifier import ZKPVerifier
from client.clients import Client

from server.pol.verifier_adapter import LocalVerifierAdapter, RemoteVerifierAdapter


from client.PoLClient import PoLClient
# Attempt to import on-chain proxy; fall back explicitly if unavailable
try:
    from chainfl.interact import chain_proxy  # requires Brownie/Ganache
    _CHAIN_PROXY_AVAILABLE = True
except Exception as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"Blockchain integration unavailable (chain_proxy import failed): {e}")
    _CHAIN_PROXY_AVAILABLE = False
    class _ChainProxyStub:
        def pol_register_client(self, *args, **kwargs):
            return None
        def batch_record_pol_verification(self, *args, **kwargs):
            return None
        def issue_challenge(self, *args, **kwargs):
            return None
        def challenge_proof(self, *args, **kwargs):
            return None
        def get_stake_info(self, *args, **kwargs):
            return {}
        def penalize(self, *args, **kwargs):
            return None
        def fund_reward_pool(self, *args, **kwargs):
            return None
        def get_incentive_stats(self, *args, **kwargs):
            return {'penalty_pool': 0}
        def distribute_rewards(self, *args, **kwargs):
            return None
    chain_proxy = _ChainProxyStub()

from server.incentive.RewardCalculator import RewardCalculator
from server.incentive.ReputationSystem import ReputationSystem
from server.incentive.StakingManager import StakingManager

logger = logging.getLogger(__name__)


class PoLVerifyAggregator(ServerAggregator):
    """
    集成PoL验证的聚合器

    核心功能:
    1. 在聚合前随机选择客户端进行PoL验证
    2. 过滤掉验证失败的客户端
    3. 对通过验证的客户端进行FedAvg聚合
    4. 记录验证结果到区块链
    """

    def __init__(self, model=None, args=None):
        """
        初始化PoL验证聚合器

        Args:
            model: 全局模型
            args: 参数字典，应包含:
                - enable_pol: 是否启用PoL验证
                - verification_rate: 验证比例 (0-1)
                - pol_delta: 距离阈值
                - pol_distance_metric: 距离度量
                - device: 计算设备
                - top_q: Top-Q验证的Q值（可选）
        """
        super().__init__(model, args)

        self.enable_pol = args.get('enable_pol', False)
        self.verification_rate = args.get('verification_rate', 0.3)
        self.use_top_q = args.get('use_top_q', False)
        self.top_q = args.get('top_q', 5)
        self.robust_aggregation = args.get('robust_aggregation', None)
        self.robust_aggregation_args = args.get('robust_aggregation_args', {}) or {}
        # Hybrid sampling parameters (root-cause improvement)
        try:
            self.always_verify_last_k = int(os.getenv('POL_ALWAYS_VERIFY_LAST_K', str(args.get('always_verify_last_k', 2))))
        except Exception:
            self.always_verify_last_k = 2
        try:
            self.random_q = int(os.getenv('POL_RANDOM_Q', str(args.get('random_q', 3))))
        except Exception:
            self.random_q = 3
        self.challenge_selected_pairs = str(
            os.getenv('POL_CHALLENGE_SELECTED_PAIRS', str(args.get('challenge_selected_pairs', '0')))
        ).strip().lower() in ('1', 'true', 'yes', 'on')

        # 初始化PoL验证器
        if self.enable_pol:
            verifier_args = {
                'delta': args.get('pol_delta', 0.01),
                'distance_metric': args.get('pol_distance_metric', 'l2'),
                'device': args.get('device', 'cpu'),
                'top_q': self.top_q if self.use_top_q else None
            }
            # Optional pass-through of min_pair_success_rate (falls back to verifier default 0.99)
            try:
                _mpsr_env = os.getenv('POL_MIN_PAIR_SUCCESS_RATE')
                if args.get('min_pair_success_rate') is not None:
                    verifier_args['min_pair_success_rate'] = float(args.get('min_pair_success_rate'))
                elif _mpsr_env is not None:
                    verifier_args['min_pair_success_rate'] = float(_mpsr_env)
            except Exception:
                pass

            self.pol_verifier = PoLVerifier(verifier_args)
            # Separate final-consistency delta (can be different from pairwise delta)
            _final_env = os.getenv('POL_FINAL_DELTA_OVERRIDE')
            try:
                _final_from_args = float(args.get('pol_final_delta')) if args.get('pol_final_delta') is not None else None
            except Exception:
                _final_from_args = None
            self.final_delta = float(_final_from_args if _final_from_args is not None else (_final_env if _final_env is not None else self.pol_verifier.delta))

            logger.info("PoL verification enabled")
            logger.info(f"  Verification rate: {self.verification_rate}")
            logger.info(f"  Use Top-Q: {self.use_top_q}")
            logger.info(f"  Pairwise delta: {self.pol_verifier.delta}")
            logger.info(f"  Final-consistency delta: {self.final_delta}")
            logger.info(f"  Always verify last K: {self.always_verify_last_k}")
            logger.info(f"  Random Q: {self.random_q}")
            logger.info(f"  Challenge selected pairs only: {self.challenge_selected_pairs}")
            logger.info(f"  Min pair success rate: {getattr(self.pol_verifier, 'min_pair_success_rate', 'n/a')}")
        else:
            self.pol_verifier = None
        # Verification adapter setup (supports decentralized verification)
        self.decentralized_verification = bool(args.get('decentralized_verification', False) or os.getenv('POL_DECENT_MODE', '0') in ('1', 'true', 'True'))
        verifier_endpoints_env = os.getenv('POL_VERIFIER_ENDPOINTS', '').strip()
        self._local_verifier_adapter = LocalVerifierAdapter(self.pol_verifier) if self.pol_verifier else None
        self._remote_verifier_adapter = None
        if self.decentralized_verification and verifier_endpoints_env:
            try:
                endpoints = [e.strip() for e in verifier_endpoints_env.split(',') if e.strip()]
                if endpoints:
                    vparams = {
                        'delta': float(getattr(self.pol_verifier, 'delta', args.get('pol_delta', 0.01))) if self.pol_verifier else float(args.get('pol_delta', 0.01)),
                        'distance_metric': str(getattr(self.pol_verifier, 'distance_metric', args.get('pol_distance_metric', 'l2'))) if self.pol_verifier else str(args.get('pol_distance_metric', 'l2')),
                        'min_pair_success_rate': float(getattr(self.pol_verifier, 'min_pair_success_rate', args.get('min_pair_success_rate', 0.99))) if self.pol_verifier else float(args.get('min_pair_success_rate', 0.99)),
                    }
                    self._remote_verifier_adapter = RemoteVerifierAdapter(endpoints=endpoints, verifier_params=vparams, mode=os.getenv('POL_REMOTE_MODE', 'distance_only'))
                    logger.info(f"Decentralized verification enabled. Endpoints: {endpoints}; mode={os.getenv('POL_REMOTE_MODE', 'distance_only')}; strategy={os.getenv('POL_REMOTE_STRATEGY', 'any')}")
            except Exception as e:
                logger.warning(f"Failed to initialize RemoteVerifierAdapter: {e}")
        # Default adapter
        self.verifier_adapter = self._remote_verifier_adapter or self._local_verifier_adapter

        # Phase A: enforce decoupling switches (remote-only verification / external-only aggregation)
        try:
            self.require_remote_verifier = str(os.getenv('POL_REQUIRE_REMOTE_VERIFIER', '0')).lower() in ('1','true','yes')
        except Exception:
            self.require_remote_verifier = False
        try:
            self.require_external_aggregator = str(os.getenv('POL_REQUIRE_EXTERNAL_AGGREGATOR', '0')).lower() in ('1','true','yes')
        except Exception:
            self.require_external_aggregator = False
        if self.require_remote_verifier:
            # Disable local verifier fallback; remote adapter must be configured
            self._local_verifier_adapter = None
            if getattr(self, '_remote_verifier_adapter', None) is None:
                raise RuntimeError('POL_REQUIRE_REMOTE_VERIFIER=1 but no POL_VERIFIER_ENDPOINTS configured')
            self.verifier_adapter = self._remote_verifier_adapter
            logger.info("POL_REQUIRE_REMOTE_VERIFIER=1: local verifier fallback disabled")
        if self.require_external_aggregator and not str(os.getenv('POL_AGGREGATOR_ENDPOINT', '')).strip():
            logger.warning("POL_REQUIRE_EXTERNAL_AGGREGATOR=1 but POL_AGGREGATOR_ENDPOINT is not set; external aggregation will be required and local fallback is disabled")

        self.enable_sybil_detector = str(os.getenv('POL_ENABLE_SYBIL_DETECTOR', str(args.get('enable_sybil_detector', '1')))).lower() in ('1', 'true', 'yes')
        self.sybil_detector = SybilDetector() if self.enable_sybil_detector else None


        # ZKP verifier (optional)
        self.enable_zkp = args.get('enable_zkp', False)
        self.zkp_use_simulation = args.get('zkp_use_simulation', False)
        self.zkp_verifier = None
        if self.enable_zkp:
            try:
                vkey_path = args.get('zkp_vkey_path', 'circuits/build/parameter_update.vkey.json')
                self.zkp_verifier = ZKPVerifier(
                    verification_key_path=vkey_path,
                    use_simulation=self.zkp_use_simulation,
                    use_onchain=False
                )
                logger.info(f"ZKP verification enabled (simulation={self.zkp_use_simulation})")
            except Exception as e:
                logger.warning(f"Failed to initialize ZKPVerifier: {e}")
                self.enable_zkp = False

        # 验证结果记录
        # Track client pool and commitments for verification
        self._client_pool: List[Client] = []
        self._client_ids: List[str] = []
        self._pol_commitments: Dict[str, dict] = {}

        # Initialize verification tracking structures
        self.verification_results = {}
        self.verified_clients = set()
        self.failed_clients = set()

        # Incentive systems (on-chain integrated)
        self.enable_incentives = args.get('enable_incentives', True)
        self.base_reward_per_round_wei = int(args.get('base_reward_per_round_wei', 0))
        self.contribution_weight = float(args.get('contribution_weight', 0.3))
        self.reputation_weight = float(args.get('reputation_weight', 0.2))

        self.reputation_system = ReputationSystem(chain_proxy=chain_proxy)
        self.staking_manager = StakingManager(chain_proxy=chain_proxy)
        self.reward_calculator = RewardCalculator(
            base_reward_per_round=float(self.base_reward_per_round_wei),
            contribution_weight=self.contribution_weight,
            reputation_weight=self.reputation_weight,
            chain_proxy=chain_proxy
        )

        # End of __init__ prelude; the adapter helpers follow
    def _maybe_send_to_external_aggregator(self, selected_ids: List[str]):
        """Optional hook: send a summary of verified clients to an external AggregatorNode.
        Controlled by env POL_AGGREGATOR_ENDPOINT. Non-blocking best-effort.
        """
        try:
            import os, json, urllib.request
            endpoint = os.getenv('POL_AGGREGATOR_ENDPOINT', '').strip()
            if not endpoint:
                return
            url = endpoint.rstrip('/') + '/aggregate'
            payload = {
                'mode': 'summary_only',
                'selected_ids': list(selected_ids or []),
                'passed_ids': sorted(list(getattr(self, 'verified_clients', set()))),
                'failed_ids': sorted(list(getattr(self, 'failed_clients', set()))),
                'commitments': {cid: self._pol_commitments.get(cid, {}) for cid in (selected_ids or [])}
            }
            try:
                rid = os.getenv('POL_REQUEST_ID', '').strip()
                if rid:
                    payload['request_id'] = rid
            except Exception:
                rid = ''
            data = json.dumps(payload).encode('utf-8')

            headers = {'Content-Type': 'application/json'}
            if rid:
                headers['X-Request-ID'] = rid
            req = urllib.request.Request(url, data=data, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as resp:
                _ = resp.read()
            if rid:
                logger.info(f"[ctx={rid}] External aggregator hook sent to {url} (passed={len(payload.get('passed_ids', []))})")
            else:
                logger.info(f"External aggregator hook sent to {url} (passed={len(payload.get('passed_ids', []))})")
        except Exception as e:
            logger.warning(f"External aggregator hook failed: {e}")

    def _select_pair_indices(self, n_pairs: int, checkpoints: List[Dict] = None) -> List[int]:
        """Choose checkpoint-pair indices using the same hybrid policy in all paths."""
        n_pairs = max(0, int(n_pairs or 0))
        if n_pairs <= 0:
            return []

        base_pairs = list(range(n_pairs))
        k = max(0, min(int(self.always_verify_last_k), n_pairs))
        last_k = list(range(n_pairs - k, n_pairs)) if k > 0 else []

        top_q_indices = []
        if self.use_top_q and self.top_q and checkpoints is not None:
            try:
                top_q_indices = self.pol_verifier.select_top_q_checkpoints(checkpoints, int(self.top_q))
            except Exception:
                top_q_indices = []

        excluded = set(last_k) | set(top_q_indices)
        remaining = [i for i in base_pairs if i not in excluded]
        r = max(0, min(int(self.random_q), len(remaining)))
        rand_sel = random.sample(remaining, r) if r > 0 else []
        return sorted(set(last_k + rand_sel + top_q_indices))

    def _preselect_challenge_steps(self, checkpoint_steps: List[int]):
        """Request only checkpoints needed by selected verification pairs."""
        steps = [int(s) for s in (checkpoint_steps or [])]
        if (
            not self.challenge_selected_pairs
            or len(steps) < 2
            or (self.use_top_q and self.top_q)
        ):
            return steps, None

        selected_original_pairs = self._select_pair_indices(len(steps) - 1)
        if not selected_original_pairs:
            return steps, None

        needed_positions = {len(steps) - 1}
        for pair_idx in selected_original_pairs:
            if 0 <= pair_idx < len(steps) - 1:
                needed_positions.add(pair_idx)
                needed_positions.add(pair_idx + 1)

        ordered_positions = sorted(needed_positions)
        selected_steps = [steps[pos] for pos in ordered_positions]
        response_position_by_original = {
            original_pos: response_pos
            for response_pos, original_pos in enumerate(ordered_positions)
        }
        response_pair_indices = []
        for pair_idx in selected_original_pairs:
            left = response_position_by_original.get(pair_idx)
            right = response_position_by_original.get(pair_idx + 1)
            if left is not None and right == left + 1:
                response_pair_indices.append(left)

        if not response_pair_indices:
            return steps, None
        return selected_steps, sorted(set(response_pair_indices))


    def _verify_pairs_via_adapter(self, *, challenge, response, commitment, model, dataloader, criterion, optimizer_class, lr, pair_indices) -> bool:
        """Try remote adapter first (if configured), then fallback to local adapter.
        Returns boolean validity.
        """
        # reset last receipts for this call
        self._last_remote_receipts = []
        # Phase A: remote-only enforcement path
        if getattr(self, 'require_remote_verifier', False):
            if getattr(self, '_remote_verifier_adapter', None) is None:
                raise RuntimeError('remote_verifier_required_but_unavailable')
            try:
                res = self._remote_verifier_adapter.verify_on_pairs_indices(
                    challenge=challenge,
                    response=response,
                    commitment=commitment,
                    model=model,
                    dataloader=dataloader,
                    criterion=criterion,
                    optimizer_class=optimizer_class,
                    lr=lr,
                    pair_indices=pair_indices,
                )
                if isinstance(res, dict):
                    self._metrics = getattr(self, '_metrics', {})
                    self._metrics['remote_majority_yes'] = int(res.get('yes', 0))
                    self._metrics['remote_majority_responders'] = int(res.get('responders', 0))
                    # collect receipts (majority strategy returns receipts=[], any returns single receipt)
                    rlist = []
                    try:
                        if res.get('receipts'):
                            rlist = list(res.get('receipts') or [])
                        elif res.get('receipt'):
                            rlist = [res.get('receipt')]
                    except Exception:
                        rlist = []
                    self._last_remote_receipts = rlist
                    return bool(res.get('valid', False))
                return bool(res)
            except Exception as e:
                logger.error(f"Remote-only verify_on_pairs_indices failed: {e}")
                raise RuntimeError(f"remote_verifier_failed: {e}")

        # Prefer selected adapter (default behavior)
        adapters = []
        if self.verifier_adapter is not None:
            adapters.append(self.verifier_adapter)
        # Ensure local fallback present
        if self._local_verifier_adapter and (self.verifier_adapter is not self._local_verifier_adapter):
            adapters.append(self._local_verifier_adapter)
        last_err = None
        for ad in adapters:
            try:
                res = ad.verify_on_pairs_indices(
                    challenge=challenge,
                    response=response,
                    commitment=commitment,
                    model=model,
                    dataloader=dataloader,
                    criterion=criterion,
                    optimizer_class=optimizer_class,
                    lr=lr,
                    pair_indices=pair_indices,
                )
                if isinstance(res, dict):
                    self._metrics = getattr(self, '_metrics', {})
                    self._metrics['remote_majority_yes'] = int(res.get('yes', 0))
                    self._metrics['remote_majority_responders'] = int(res.get('responders', 0))
                    # collect receipts if present
                    rlist = []
                    try:
                        if res.get('receipts'):
                            rlist = list(res.get('receipts') or [])
                        elif res.get('receipt'):
                            rlist = [res.get('receipt')]
                    except Exception:
                        rlist = []
                    self._last_remote_receipts = rlist
                    return bool(res.get('valid', False))
                if res is not None:
                    return bool(res)
            except Exception as e:
                last_err = e
                logger.warning(f"Adapter verify_on_pairs_indices failed, trying next: {e}")
        if last_err:
            logger.error(f"All adapters failed for verify_on_pairs_indices: {last_err}")
        return False

    def _verify_full_via_adapter(self, *, challenge, response, commitment, model, dataloader, criterion, optimizer_class, lr) -> bool:
        # Phase A: remote-only enforcement path
        if getattr(self, 'require_remote_verifier', False):
            if getattr(self, '_remote_verifier_adapter', None) is None:
                raise RuntimeError('remote_verifier_required_but_unavailable')
            try:
                res = self._remote_verifier_adapter.verify_response(
                    challenge=challenge,
                    response=response,
                    commitment=commitment,
                    model=model,
                    dataloader=dataloader,
                    criterion=criterion,
                    optimizer_class=optimizer_class,
                    lr=lr,
                )
                if isinstance(res, dict):
                    self._metrics = getattr(self, '_metrics', {})
                    self._metrics['remote_majority_yes'] = int(res.get('yes', 0))
                    self._metrics['remote_majority_responders'] = int(res.get('responders', 0))
                    rlist = []
                    try:
                        if res.get('receipts'):
                            rlist = list(res.get('receipts') or [])
                        elif res.get('receipt'):
                            rlist = [res.get('receipt')]
                    except Exception:
                        rlist = []
                    self._last_remote_receipts = rlist
                    return bool(res.get('valid', False))
                return bool(res)
            except Exception as e:
                logger.error(f"Remote-only verify_response failed: {e}")
                raise RuntimeError(f"remote_verifier_failed: {e}")

        adapters = []
        if self.verifier_adapter is not None:
            adapters.append(self.verifier_adapter)
        if self._local_verifier_adapter and (self.verifier_adapter is not self._local_verifier_adapter):
            adapters.append(self._local_verifier_adapter)
        last_err = None
        for ad in adapters:
            try:
                res = ad.verify_response(
                    challenge=challenge,
                    response=response,
                    commitment=commitment,
                    model=model,
                    dataloader=dataloader,
                    criterion=criterion,
                    optimizer_class=optimizer_class,
                    lr=lr,
                )
                if isinstance(res, dict):
                    self._metrics = getattr(self, '_metrics', {})
                    self._metrics['remote_majority_yes'] = int(res.get('yes', 0))
                    self._metrics['remote_majority_responders'] = int(res.get('responders', 0))
                    rlist = []
                    try:
                        if res.get('receipts'):
                            rlist = list(res.get('receipts') or [])
                        elif res.get('receipt'):
                            rlist = [res.get('receipt')]
                    except Exception:
                        rlist = []
                    self._last_remote_receipts = rlist
                    return bool(res.get('valid', False))
                return bool(res)
            except Exception as e:
                last_err = e
                logger.warning(f"Adapter verify_response failed, trying next: {e}")
        if last_err:
            logger.error(f"All adapters failed for verify_response: {last_err}")
        return False
    def _verify_m_of_n(self, receipts: list, commitment: str):
        """Verify signed receipts against POL_VERIFIER_ADDRESSES and POL_M_OF_N.
        Returns (ok, m, n). If not configured, returns (True, 0, 0).
        """
        try:
            wl_raw = os.getenv('POL_VERIFIER_ADDRESSES', '').strip()
            m_of_n = os.getenv('POL_M_OF_N', '').strip()
            if not wl_raw or not m_of_n or not receipts:
                return True, 0, 0
            whitelist = [a.strip().lower() for a in wl_raw.split(',') if a.strip()]
            if not whitelist:
                return True, 0, 0
            # parse POL_M_OF_N like "2/3" or "2"
            req_m, req_n = None, None
            try:
                if '/' in m_of_n:
                    a, b = m_of_n.split('/', 1)
                    req_m = int(a)
                    req_n = int(b)
                else:
                    req_m = int(m_of_n)
            except Exception:
                logger.warning(f"Invalid POL_M_OF_N: {m_of_n}")
                return True, 0, 0
            try:
                from eth_account import Account
                from eth_account.messages import encode_defunct
            except Exception as e:
                logger.error(f"eth_account unavailable for M-of-N verification: {e}")
                return False, 0, 0
            agg_addr = str(os.getenv('POL_AGGREGATOR_ADDR', '')).strip().lower()
            seen = {}
            yes_count = 0
            for rcpt in receipts:
                try:
                    msg = rcpt.get('msg') if isinstance(rcpt, dict) else None
                    sig = rcpt.get('sig') if isinstance(rcpt, dict) else None
                    addr = str(rcpt.get('addr', '')).strip().lower() if isinstance(rcpt, dict) else ''
                    if not msg or not sig:
                        continue
                    # verify signature
                    text = json.dumps(msg, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
                    rec_addr = str(Account.recover_message(encode_defunct(text=text), signature=sig)).lower()
                    # basic checks
                    if addr and rec_addr != addr:
                        continue
                    if rec_addr == agg_addr and agg_addr:
                        raise RuntimeError('aggregator_must_not_sign')
                    if rec_addr not in whitelist:
                        continue
                    if msg.get('commitmentRoot') and commitment and str(msg.get('commitmentRoot')) != str(commitment):
                        # mismatch commitment, ignore
                        continue
                    if rec_addr in seen:
                        continue
                    seen[rec_addr] = True
                    if bool(msg.get('valid', False)):
                        yes_count += 1
                except Exception:
                    # Fallback: if signature recovery fails, trust provided addr if whitelisted
                    try:
                        msg = rcpt.get('msg') if isinstance(rcpt, dict) else None
                        addr_fb = str((rcpt.get('addr','') if isinstance(rcpt, dict) else '')).strip().lower()
                        if msg and addr_fb and addr_fb in whitelist:
                            if msg.get('commitmentRoot') and commitment and str(msg.get('commitmentRoot')) != str(commitment):
                                pass
                            elif agg_addr and addr_fb == agg_addr:
                                raise RuntimeError('aggregator_must_not_sign')
                            elif addr_fb not in seen:
                                seen[addr_fb] = True
                                if bool(msg.get('valid', False)):
                                    yes_count += 1
                    except Exception:
                        pass
                    continue
            n = len(seen)
            m = yes_count
            # Accept if at least M valid distinct signers; do not require all N to respond
            ok = m >= int(req_m)
            # store metrics
            self._metrics = getattr(self, '_metrics', {})
            self._metrics['receipt_m'] = int(m)
            self._metrics['receipt_n'] = int(n)
            return ok, m, n
        except Exception as e:
            logger.warning(f"M-of-N verification error: {e}")
            return False, 0, 0




    def receive_upload(self, client_pool: List[Client]):
        """
        Override to capture client objects, models, and PoL commitments.
        """
        # Reset pools for this round
        self.model_pool = []
        self._client_pool = list(client_pool)
        self._client_ids = [c.client_id for c in client_pool]
        self._pol_commitments = {}

        for client in client_pool:
            # collect model state
            self.model_pool.append(client.get_model_state_dict())
            # collect PoL commitment if available
            if isinstance(client, PoLClient):
                commit = client.get_pol_commitment()
                if commit:
                    self._pol_commitments[client.client_id] = commit

            else:
                # Fallback: try to derive PoL commitment from attached trainer (for tests/non-PoLClient objects)
                try:
                    pm = getattr(getattr(client, 'trainer', None), 'pol_manager', None)
                    if pm is not None:
                        commit = {
                            'commitment': pm.generate_commitment(),
                            'num_checkpoints': pm.get_checkpoint_count(),
                            'data_hash': ''
                        }
                        if commit.get('commitment'):
                            self._pol_commitments[client.client_id] = commit
                except Exception as e:
                    logger.debug(f"Deriving PoL commitment from client.trainer failed: {e}")

        self.verification_results = {}  # {client_id: bool}
        self.verified_clients = set()
        self.failed_clients = set()

        # Reset per-round decentralized/aggregation metrics
        try:
            self._metrics = {
                'external_agg_success': False,
                'external_agg_latency_s': 0.0,
                'remote_majority_responders': 0,
                'remote_majority_yes': 0,
                # observability extensions (not all emitted by RQ1 yet)
                'remote_verify_latency_p50_s': 0.0,
                'remote_verify_latency_p95_s': 0.0,
                'remote_error_timeout': 0,
                'remote_error_network': 0,
                'remote_error_invalid': 0,
                'remote_error_business': 0,
                'pol_verify_time_s': 0.0,
            }
        except Exception:
            self._metrics = {}

    def release_round_payloads(self):
        """Release per-round objects that can retain model/checkpoint tensors."""
        try:
            self.model_pool = []
        except Exception:
            pass
        self._client_pool = []
        self._pol_commitments = {}
        try:
            self._receipts_by_client = {}
        except Exception:
            pass
        try:
            self._filtered_indices = []
        except Exception:
            pass

    def _on_before_aggregation(
        self, raw_client_model_or_grad_list: List[OrderedDict]
    ) -> List[OrderedDict]:
        """
        聚合前的处理：执行PoL验证

        Args:
            raw_client_model_or_grad_list: 客户端模型列表

        Returns:
            filtered_list: 过滤后的模型列表（移除验证失败的客户端）
        """
        if not self.enable_pol or self.pol_verifier is None:
            # PoL未启用，直接返回
            return raw_client_model_or_grad_list

        # 随机选择客户端进行验证
        num_clients = len(raw_client_model_or_grad_list)
        num_to_verify = max(1, int(num_clients * self.verification_rate))

        # 这里简化处理，实际应该从client_pool中获取客户端信息
        # 假设每个模型都有对应的client_id（需要在实际集成时调整）
        clients_to_verify = random.sample(range(num_clients), num_to_verify)

        logger.info(f"Verifying {num_to_verify}/{num_clients} clients")
        logger.info(f"  Selected indices: {clients_to_verify}")

        # 1) send challenge, 2) receive response, 3) verify with PoLVerifier, 4) record to chain
        # Require client_pool from receive_upload; if absent, do not silently pass.
        failed_indices = set()
        selected_ids = []
        responses_by_client = {}
        commitments_by_client = {}
        idx_by_client = {}

        # start total PoL verification timer
        _pol_t0 = time.time()

        # initialize per-client receipts box for anchoring
        self._receipts_by_client = {}

        for idx in clients_to_verify:
            if idx >= len(self._client_pool):
                continue
            client = self._client_pool[idx]
            client_id = client.client_id
            # selected_ids will be appended once during actual verification loop

            # Ensure client is registered on-chain for challenge flow
            try:
                chain_proxy.pol_register_client(client_id)
            except Exception as e:
                logger.debug(f"Client {client_id} register on-chain skipped: {e}")

        if not getattr(self, '_client_pool', None):
            logger.error("PoL verification requires calling receive_upload(client_pool) with PoLClient instances.")
            return raw_client_model_or_grad_list

        for idx in clients_to_verify:
            if idx >= len(self._client_pool):
                continue
            client = self._client_pool[idx]
            client_id = client.client_id
            selected_ids.append(client_id)

            # Build a challenge: request all checkpoint steps
            # FIX: Use actual checkpoint steps instead of indices to avoid mismatch
            # when batch_counter accumulates across rounds
            checkpoint_steps = []
            preselected_pair_indices = None
            commit = self._pol_commitments.get(client_id)

            try:
                # Try to get actual steps from pol_manager metadata
                pol_manager = getattr(client.trainer, 'pol_manager', None)
                logger.info(f"{client_id}: pol_manager exists: {pol_manager is not None}")
                if pol_manager and hasattr(pol_manager, 'metadata'):
                    ckpt_list = pol_manager.metadata.get('checkpoints', [])
                    logger.info(f"{client_id}: Found {len(ckpt_list)} checkpoints in metadata")
                    for ckpt_meta in ckpt_list:
                        step = ckpt_meta.get('step')
                        if step is not None:
                            checkpoint_steps.append(int(step))
                    logger.info(f"{client_id}: Extracted {len(checkpoint_steps)} checkpoint steps from metadata")
                else:
                    logger.info(f"{client_id}: pol_manager or metadata not available")
            except Exception as e:
                logger.warning(f"{client_id}: Failed to get checkpoint steps from metadata: {e}", exc_info=True)

            # Fallback: use old index-based logic if no steps found
            if not checkpoint_steps:
                num_ckpts = None
                if commit:
                    num_ckpts = int(commit.get('num_checkpoints', 0))
                if num_ckpts is None or num_ckpts <= 1:
                    try:
                        num_ckpts = int(getattr(client.trainer, 'pol_manager').checkpoint_count)
                    except Exception:
                        num_ckpts = 0

                # Old logic: indices [1..num_ckpts] mapped to steps [save_freq, 2*save_freq, ...]
                checkpoint_indices = list(range(1, max(1, num_ckpts) + 1))
                logger.warning(f"{client_id}: Using fallback index-based challenge with {len(checkpoint_indices)} indices")
                challenge = {
                    'checkpoint_indices': checkpoint_indices,
                    'include_data_indices': True
                }
            else:
                # New logic: use actual steps
                challenge_steps, preselected_pair_indices = self._preselect_challenge_steps(checkpoint_steps)
                challenge = {
                    'checkpoint_steps': challenge_steps,
                    'include_data_indices': True
                }
                if preselected_pair_indices is not None:
                    logger.info(
                        f"{client_id}: Using selected-pair step challenge with "
                        f"{len(challenge_steps)}/{len(checkpoint_steps)} steps and "
                        f"{len(preselected_pair_indices)} pair(s)"
                    )
                else:
                    logger.info(f"{client_id}: Using step-based challenge with {len(checkpoint_steps)} steps")


            # Lock required checkpoints on client to avoid premature cleanup during verification
            try:
                pm = getattr(client.trainer, 'pol_manager', None)
                if pm and hasattr(pm, 'begin_inflight_verification'):
                    steps_to_lock = challenge.get('checkpoint_steps', [])
                    if not steps_to_lock:
                        idxs = challenge.get('checkpoint_indices', [])
                        sf = getattr(pm, 'save_freq', 1)
                        steps_to_lock = [int(i) * int(sf) for i in idxs]
                    tag = str(commit.get('commitment')) if commit else None
                    pm.begin_inflight_verification(steps_to_lock, tag=tag)
            except Exception:
                pass

            response = None
            try:
                response = client.respond_to_challenge(challenge)
            except Exception as e:
                logger.error(f"Challenge failed for {client_id}: {e}")
                response = None

            # If no response or no commitment, mark failed
            if not response or not commit:
                self.verification_results[client_id] = False
                self.failed_clients.add(client_id)
                failed_indices.add(idx)
                try:
                    pm = getattr(client.trainer, 'pol_manager', None)
                    if pm and hasattr(pm, 'complete_inflight_verification'):
                        pm.complete_inflight_verification(tag=str(commit.get('commitment')) if commit else None)
                except Exception:
                    pass
                continue

            responses_by_client[client_id] = response
            commitments_by_client[client_id] = commit
            idx_by_client[client_id] = idx

            # Enforce minimal training progress if configured (guards against lazy-training free-riding)
            try:
                min_epochs = int(os.getenv('POL_MIN_EPOCHS', '0'))
                if min_epochs > 0:
                    ds_len = None
                    bs = None
                    try:
                        ds_len = len(getattr(client.dataloader, 'dataset', []))
                    except Exception:
                        ds_len = None
                    try:
                        bs = getattr(client.dataloader, 'batch_size', None)
                        if bs is None:
                            # Fallback for custom loaders without batch_size attribute
                            bs = int(os.getenv('POL_DEFAULT_BATCH_SIZE', '32'))
                    except Exception:
                        bs = int(os.getenv('POL_DEFAULT_BATCH_SIZE', '32'))

                    if ds_len is not None and ds_len > 0 and bs and bs > 0:
                        steps_per_epoch = int(math.ceil(ds_len / float(bs)))
                        expected_min_steps = int(min_epochs * steps_per_epoch)
                        frac = float(os.getenv('POL_MIN_STEPS_FRACTION', '1.0'))
                        threshold_steps = int(expected_min_steps * frac)
                        actual_steps = int(commit.get('total_steps', 0))
                        if actual_steps < threshold_steps:
                            logger.warning(f"{client_id}: total_steps {actual_steps} < threshold {threshold_steps} (min_epochs={min_epochs}, steps/epoch={steps_per_epoch}). Marking verification failed.")
                            self.verification_results[client_id] = False
                            self.failed_clients.add(client_id)
                            failed_indices.add(idx)
                            continue
            except Exception as e:
                logger.debug(f"Minimal steps check skipped due to error: {e}")

            # Bind the PoL trajectory to the server-accepted data view. This is
            # the check that can expose label/data poisoning: the client hashes
            # the data actually used by its PoL trainer, while the server
            # recomputes the hash over the clean registered client dataset.
            try:
                committed_data_hash = str(commit.get('data_hash', '') or '')
                if committed_data_hash:
                    pm = getattr(client.trainer, 'pol_manager', None)
                    clean_dataset = getattr(getattr(client, 'dataloader', None), 'dataset', None)
                    if pm is not None and clean_dataset is not None and hasattr(pm, 'compute_data_hash'):
                        expected_data_hash = str(pm.compute_data_hash(clean_dataset) or '')
                        if expected_data_hash and expected_data_hash != committed_data_hash:
                            logger.warning(
                                f"{client_id}: data_hash mismatch "
                                f"(commit={committed_data_hash[:16]}..., expected={expected_data_hash[:16]}...). "
                                "Marking verification failed."
                            )
                            self.verification_results[client_id] = False
                            self.failed_clients.add(client_id)
                            failed_indices.add(idx)
                            continue
                    else:
                        logger.debug(f"{client_id}: data_hash check skipped; missing pol_manager or clean dataset")
            except Exception as e:
                logger.warning(f"{client_id}: data_hash check failed with error: {e}")

            # Optional: verify ZKP proof with binding and record on-chain independently
            zkp_ok = None
            try:
                if self.enable_zkp and self.zkp_verifier and response.get('zkp') and len(response.get('checkpoints', [])) >= 2:
                    z = response['zkp']
                    current_ckpt = response['checkpoints'][0]
                    next_ckpt = response['checkpoints'][1]
                    zkp_ok = self.zkp_verifier.verify_proof_with_binding(
                        current_ckpt=current_ckpt,
                        next_ckpt=next_ckpt,
                        data_indices=response.get('data_indices', []),
                        proof=z.get('proof', {}),
                        public_signals=z.get('public_signals', {})
                    )
                    # On-chain audit trail for ZKP: issue challenge and immediately resolve
                    if zkp_ok is True:
                        pair = z.get('pair', [0, 1])
                        idx0 = int(pair[0]) if len(pair) > 0 else 0
                        idx1 = int(pair[1]) if len(pair) > 1 else 1
                        deadline_ts = int(time.time()) + 3600
                        try:
                            chal_id = chain_proxy.issue_challenge(client_id, idx0, idx1, deadline_ts)
                            if isinstance(chal_id, str) and len(chal_id) > 0:
                                chain_proxy.challenge_proof(chal_id, z.get('public_signals', {}), True, reason="zkp_ok")
                        except Exception as e:
                            logger.warning(f"On-chain challengeProof record failed for {client_id}: {e}")
            except Exception as e:
                logger.error(f"ZKP verification error for {client_id}: {e}")
                zkp_ok = False

            # Fallback: if ZKP not present but we have 2+ checkpoints, still issue an on-chain challenge record for auditability
            try:
                if (zkp_ok is None or zkp_ok is False) and self.enable_zkp and len(response.get('checkpoints', [])) >= 2:
                    # Represent absent public signals explicitly when no proof was submitted.
                    pair = [response['checkpoints'][0]['index'], response['checkpoints'][1]['index']]
                    idx0 = int(pair[0])
                    idx1 = int(pair[1])
                    deadline_ts = int(time.time()) + 3600
                    chal_id = chain_proxy.issue_challenge(client_id, idx0, idx1, deadline_ts)
                    if isinstance(chal_id, str) and len(chal_id) > 0:
                        # Record absence of ZKP without synthesizing public inputs.
                        chain_proxy.challenge_proof(chal_id, {}, False, reason="zkp_absent_or_fail")
            except Exception as e:
                logger.warning(f"On-chain fallback challengeProof record failed for {client_id}: {e}")

            # Prepare verification inputs
            criterion = torch.nn.CrossEntropyLoss()
            # optimizer mapping
            opt_name = str(client.args.get('optimizer', 'SGD')).lower()
            optimizer_class = torch.optim.SGD if 'sgd' in opt_name else torch.optim.Adam
            lr = float(client.args.get('lr', 0.01))

            # Run verification
            try:
                # Hybrid sampling: always verify last K pairs + random R from the rest + (optional) Top-Q
                checkpoints = response.get('checkpoints', [])
                n_ckpts = len(checkpoints)
                if n_ckpts < 2:
                    is_valid = False

                else:
                    n_pairs = n_ckpts - 1
                    if preselected_pair_indices is not None:
                        pair_indices = [
                            int(i) for i in preselected_pair_indices
                            if 0 <= int(i) < n_pairs
                        ]
                    else:
                        pair_indices = self._select_pair_indices(n_pairs, checkpoints=checkpoints)

                    if pair_indices:
                        is_valid = self._verify_pairs_via_adapter(
                            challenge=challenge,
                            response=response,
                            commitment=commit['commitment'],
                            model=self.model,
                            dataloader=client.dataloader,
                            criterion=criterion,
                            optimizer_class=optimizer_class,
                            lr=lr,
                            pair_indices=pair_indices
                        )
                    else:
                        if preselected_pair_indices is not None:
                            is_valid = False
                        else:
                            # Fallback to full verification if indices selection failed
                            is_valid = self._verify_full_via_adapter(
                                challenge=challenge,
                                response=response,
                                commitment=commit['commitment'],
                                model=self.model,
                                dataloader=client.dataloader,
                                criterion=criterion,
                                optimizer_class=optimizer_class,
                                lr=lr
                            )

                # CRITICAL: Verify final model consistency
                # Check if the uploaded model matches the last checkpoint
                # This detects Byzantine attacks (noise injection after training)
                if is_valid and len(response.get('checkpoints', [])) > 0:
                    try:
                        # Select checkpoint at the maximum data.step to avoid ordering issues
                        ckpts = response.get('checkpoints', [])
                        def _get_step(ck):
                            try:
                                return int(ck.get('data', {}).get('step', -1))
                            except Exception:
                                return -1
                        try:
                            last_ckpt = max(ckpts, key=_get_step)
                        except Exception:
                            last_ckpt = ckpts[-1] if ckpts else None
                        if last_ckpt is None:
                            raise RuntimeError("No checkpoints available for final consistency check")
                        last_step = _get_step(last_ckpt)
                        logger.debug(f"{client_id}: Final consistency check using step={last_step}")
                        last_ckpt_state = last_ckpt.get('data', {}).get('model_state', None)
                        if last_ckpt_state is None:
                            raise RuntimeError("Missing model_state in last checkpoint data")

                        # Get the uploaded model state
                        uploaded_model_state = client.model.state_dict()

                        # Compute distance between uploaded model and last checkpoint
                        final_distance = self.pol_verifier._compute_parameter_distance(
                            uploaded_model_state,
                            last_ckpt_state,
                            self.pol_verifier.distance_metric
                        )

                        # Check if distance exceeds threshold (final-consistency may use different delta)
                        _th = getattr(self, 'final_delta', self.pol_verifier.delta)
                        if final_distance > _th:
                            logger.warning(f"{client_id}: Final model inconsistent with last checkpoint "
                                           f"(distance={final_distance:.6f} > final_delta={_th})")
                            is_valid = False
                        else:
                            logger.debug(f"{client_id}: Final model consistent "
                                         f"(distance={final_distance:.6f} <= final_delta={_th})")
                    except Exception as e:
                        logger.error(f"Final model verification error for {client_id}: {e}")
                        is_valid = False

                # If ZKP is enabled and client provided proof, verify and bind
                if is_valid and self.enable_zkp and self.zkp_verifier and response.get('zkp'):
                    z = response['zkp']
                    # Use the first two checkpoints in response for binding
                    if len(response.get('checkpoints', [])) >= 2:
                        current_ckpt = response['checkpoints'][0]
                        next_ckpt = response['checkpoints'][1]
                        try:
                            is_zkp_ok = self.zkp_verifier.verify_proof_with_binding(
                                current_ckpt=current_ckpt,
                                next_ckpt=next_ckpt,
                                data_indices=response.get('data_indices', []),
                                proof=z.get('proof', {}),
                                public_signals=z.get('public_signals', {})
                            )
                            if not is_zkp_ok:
                                is_valid = False
                        except Exception as e:
                            logger.error(f"ZKP verification error for {client_id}: {e}")
                            is_valid = False

            except Exception as e:
                logger.error(f"Verification error for {client_id}: {e}")
                is_valid = False

            # Phase B-2: M-of-N gate using signed receipts (if configured)
            try:
                receipts = list(getattr(self, '_last_remote_receipts', []) or [])
                # store per-client receipts for anchoring later
                try:
                    self._receipts_by_client[client_id] = receipts
                except Exception:
                    pass
                if receipts:
                    ok, m, n = self._verify_m_of_n(receipts, commitment=str(commit.get('commitment')) if commit else '')
                    if not ok:
                        is_valid = False
            except Exception as _:
                pass

            self.verification_results[client_id] = bool(is_valid)
            if is_valid:
                self.verified_clients.add(client_id)
            else:
                self.failed_clients.add(client_id)
                failed_indices.add(idx)
            # Release locks on client regardless of pass/fail
            try:
                pm = getattr(client.trainer, 'pol_manager', None)
                if pm and hasattr(pm, 'complete_inflight_verification'):
                    pm.complete_inflight_verification(tag=str(commit.get('commitment')) if commit else None)
            except Exception:
                pass

        try:
            if self.sybil_detector is not None and responses_by_client:
                suspects = self.sybil_detector.detect(responses_by_client, commitments_by_client)
                self._metrics = getattr(self, '_metrics', {})
                self._metrics['sybil_suspects'] = int(len(suspects))
                for cid, reasons in suspects.items():
                    idx = idx_by_client.get(cid)
                    if idx is None:
                        continue
                    self.verification_results[cid] = False
                    self.failed_clients.add(cid)
                    self.verified_clients.discard(cid)
                    failed_indices.add(idx)
                    logger.warning(f"{cid}: Sybil detector failed client ({'; '.join(reasons)})")
        except Exception as e:
            logger.warning(f"Sybil detector skipped due to error: {e}")

        # total PoL verify time (s)
        try:
            self._metrics = getattr(self, '_metrics', {})
            self._metrics['pol_verify_time_s'] = float(max(0.0, time.time() - _pol_t0))
        except Exception:
            pass

        # Record results on chain (best-effort)
        try:
            if selected_ids:
                results = [self.verification_results[cid] for cid in selected_ids]
                chain_proxy.batch_record_pol_verification(selected_ids, results)
        except Exception as e:
            logger.warning(f"Failed to record verification results on chain: {e}")


        # Filter out failed clients and remember indices for weighted aggregation
        self._filtered_indices = [i for i in range(len(raw_client_model_or_grad_list)) if i not in failed_indices]
        filtered_list = [raw_client_model_or_grad_list[i] for i in self._filtered_indices]

        logger.info("Verification complete:")
        # Optional: forward summary to external aggregator node (Phase B hook)
        try:
            self._maybe_send_to_external_aggregator(selected_ids)
        except Exception as e:
            logger.debug(f"External aggregator hook error (ignored): {e}")

        logger.info(f"  Selected: {len(selected_ids)}; Passed: {len(self.verified_clients)}; Failed: {len(self.failed_clients)}")

        # Snapshot remote adapter observability and compute latency percentiles
        try:
            if getattr(self, '_remote_verifier_adapter', None):
                snap = self._remote_verifier_adapter.metrics_snapshot_and_reset()
                lats = list(map(float, snap.get('latencies', []) or []))
                if lats:
                    s = sorted(lats)
                    def _pct(p):
                        if not s:
                            return 0.0
                        k = int(round((p/100.0) * (len(s)-1)))
                        k = max(0, min(k, len(s)-1))
                        return float(s[k])
                    self._metrics['remote_verify_latency_p50_s'] = _pct(50)
                    self._metrics['remote_verify_latency_p95_s'] = _pct(95)
                errs = snap.get('error_counts', {}) or {}
                self._metrics['remote_error_timeout'] = int(errs.get('timeout', 0))
                self._metrics['remote_error_network'] = int(errs.get('network', 0))
                self._metrics['remote_error_invalid'] = int(errs.get('invalid', 0))
                self._metrics['remote_error_business'] = int(errs.get('business', 0))
        except Exception as e:
            logger.debug(f"Remote adapter metrics snapshot failed: {e}")

        return filtered_list

    def _preflight_check(self) -> bool:
        """Lightweight startup self-check for decentralized verification and external aggregator."""
        try:
            ok = True
            # Check verifier endpoints if remote adapter is configured
            if getattr(self, '_remote_verifier_adapter', None):
                import urllib.request
                endpoints = list(getattr(self._remote_verifier_adapter, '_endpoints', []) or [])
                for ep in endpoints:
                    url = ep.rstrip('/') + '/health'
                    try:
                        with urllib.request.urlopen(url, timeout=2) as _:
                            pass
                    except Exception:
                        logger.warning(f"Verifier endpoint not healthy: {url}")
                        ok = False
            # Check external aggregator endpoint if configured
            endpoint = os.getenv('POL_AGGREGATOR_ENDPOINT', '').strip()
            if endpoint:
                import urllib.request
                url = endpoint.rstrip('/') + '/health'
                try:
                    with urllib.request.urlopen(url, timeout=2) as _:
                        pass
                except Exception:
                    logger.warning(f"Aggregator endpoint not healthy: {url}")
                    ok = False
            return ok
        except Exception as e:
            logger.warning(f"Preflight check error: {e}")
            return False


    def _external_aggregate_weights(self, model_list: List[OrderedDict]):
        """Send filtered client models to external AggregatorNode for aggregation.
        Returns (aggregated_state_dict or None, latency_seconds). Best-effort.
        """
        try:
            import os, io, base64, json, urllib.request, time, torch, socket
            endpoint = os.getenv('POL_AGGREGATOR_ENDPOINT', '').strip()
            if not endpoint:
                return None, 0.0
            url = endpoint.rstrip('/') + '/aggregate'
            buf = io.BytesIO()
            torch.save(model_list, buf)
            payload = {
                'mode': 'weights_b64',
                'models_b64': base64.b64encode(buf.getvalue()).decode('ascii'),
            }
            try:
                indices = list(getattr(self, '_filtered_indices', []) or [])
                if not indices and getattr(self, '_client_pool', None) and len(self._client_pool) >= len(model_list):
                    indices = list(range(len(model_list)))
                sizes = []
                if indices and getattr(self, '_client_pool', None) and len(indices) == len(model_list):
                    for idx in indices:
                        try:
                            size = len(getattr(self._client_pool[idx].dataloader, 'dataset', []))
                        except Exception:
                            size = 0
                        sizes.append(int(size))
                if sizes and sum(sizes) > 0:
                    payload['client_sizes'] = sizes
                    total = float(sum(sizes))
                    payload['weights'] = [float(s) / total for s in sizes]
            except Exception as e:
                logger.debug(f"External aggregation weights unavailable; falling back to node default: {e}")
            rid = ''
            try:
                rid = os.getenv('POL_REQUEST_ID', '').strip()
                if rid:
                    payload['request_id'] = rid
            except Exception:
                rid = ''
            data = json.dumps(payload).encode('utf-8')
            t0 = time.time()
            headers = {'Content-Type': 'application/json'}
            if rid:
                headers['X-Request-ID'] = rid
            req = urllib.request.Request(url, data=data, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                resp_data = json.loads(resp.read().decode('utf-8'))
            latency = time.time() - t0
            if not resp_data.get('ok'):
                return None, latency
            out_b64 = resp_data.get('aggregated_b64')
            if not out_b64:
                return None, latency
            out_bytes = base64.b64decode(out_b64.encode('ascii'))
            out_buf = io.BytesIO(out_bytes)
            aggregated = torch.load(out_buf)
            return aggregated, latency
        except Exception as e:
            logger.warning(f"External aggregation failed: {e}")
            # classify error type
            try:
                etype = 'business'
                msg = str(e).lower()
                if isinstance(e, socket.timeout) or 'timed out' in msg:
                    etype = 'timeout'
                else:
                    try:
                        from urllib.error import URLError, HTTPError
                        if isinstance(e, (URLError, HTTPError)):
                            etype = 'network'
                    except Exception:
                        pass
                self._metrics = getattr(self, '_metrics', {})
                # store the last error type (counter optional)
                self._metrics['external_agg_error_type'] = etype
            except Exception:
                pass
            return None, 0.0

    def _maybe_anchor_onchain(self):
        """Best-effort on-chain anchoring of this round's verification/aggregation.
        Controlled by POL_ANCHOR_ONCHAIN=1 and requires chain_proxy with anchor_round.
        """
        try:
            if str(os.getenv('POL_ANCHOR_ONCHAIN', '0')).lower() not in ('1', 'true', 'yes'):
                return
            rid = str(getattr(self, '_request_id', '') or os.getenv('POL_REQUEST_ID', '')).strip()
            # build commit set for verified clients
            verified = sorted(list(self.verified_clients))
            commits = []
            for cid in verified:
                try:
                    c = (self._pol_commitments or {}).get(cid, {})
                    commits.append({'cid': cid, 'commitment': str(c.get('commitment',''))})
                except Exception:
                    commits.append({'cid': cid, 'commitment': ''})
            import hashlib, json as _json
            commit_bytes = _json.dumps({'rid': rid, 'commits': commits}, sort_keys=True, separators=(',', ':')).encode('utf-8')
            commit_hash_hex = hashlib.sha256(commit_bytes).hexdigest()
            # build sigset from receipts (distinct addresses over verified clients)
            addrs = []
            try:
                for cid in verified:
                    for rc in (getattr(self, '_receipts_by_client', {}).get(cid) or []):
                        try:
                            a = str((rc or {}).get('addr','')).strip().lower()
                            if a:
                                addrs.append(a)
                        except Exception:
                            pass
            except Exception:
                pass
            addrs = sorted(list(set(addrs)))
            sigset_bytes = _json.dumps({'rid': rid, 'addrs': addrs}, sort_keys=True, separators=(',', ':')).encode('utf-8')
            sigset_hash_hex = hashlib.sha256(sigset_bytes).hexdigest()
            # try chain_proxy native API if available
            txid, blockno = None, None
            try:
                fn = getattr(chain_proxy, 'anchor_round', None)
                if callable(fn):
                    r = fn(rid, commit_hash_hex, sigset_hash_hex)
                    if isinstance(r, dict):
                        txid = r.get('txid') or r.get('tx')
                        blockno = r.get('blockNumber') or r.get('block')
            except Exception as e:
                logger.debug(f"chain_proxy.anchor_round failed: {e}")
            # record metrics
            self._metrics = getattr(self, '_metrics', {})
            if txid:
                self._metrics['anchor_txid'] = str(txid)
            if blockno is not None:
                try:
                    self._metrics['anchor_block'] = int(blockno)
                except Exception:
                    self._metrics['anchor_block'] = 0
            logger.info(f"Anchored round (rid={rid}) commitHash={commit_hash_hex[:10]}.. sigsetHash={sigset_hash_hex[:10]}.. txid={txid}")
        except Exception as e:
            logger.warning(f"Anchor on-chain skipped: {e}")




    def _aggregate_alg(self, raw_client_model_or_grad_list: List[OrderedDict] = None) -> OrderedDict:
        """
        聚合算法：FedAvg

        Args:
            raw_client_model_or_grad_list: 客户端模型列表

        Returns:
            aggregated_model: 聚合后的模型
        """
        if raw_client_model_or_grad_list is None:
            raw_client_model_or_grad_list = self.model_pool

        if not raw_client_model_or_grad_list:
            logger.warning("No client models to aggregate")
            return self.model.state_dict() if self.model else {}

        # Optional robust aggregation after PoL filtering. This is used by the
        # composability experiments; default remains weighted FedAvg.
        if self.robust_aggregation:
            try:
                from experiments.scripts.utils.baselines import create_aggregator

                robust_kwargs = dict(self.robust_aggregation_args)
                if self.robust_aggregation in ("Krum", "Bulyan", "SDEA") and "num_byzantine" not in robust_kwargs:
                    robust_kwargs["num_byzantine"] = int(self.args.get("num_byzantine", 0) or 0)
                if self.robust_aggregation == "Trimmed_Mean" and "trim_ratio" not in robust_kwargs:
                    robust_kwargs["trim_ratio"] = float(self.args.get("trim_ratio", 0.1) or 0.1)
                aggregator = create_aggregator(self.robust_aggregation, **robust_kwargs)
                aggregated = aggregator.aggregate(raw_client_model_or_grad_list)
                logger.debug("Aggregated %d client models using robust method %s", len(raw_client_model_or_grad_list), self.robust_aggregation)
                return aggregated
            except Exception as e:
                logger.warning("Robust aggregation %s failed; falling back to FedAvg: %s", self.robust_aggregation, e)

        # FedAvg聚合


        num_clients = len(raw_client_model_or_grad_list)

        aggregated_model = {}

        # 获取第一个模型的所有键
        first_model = raw_client_model_or_grad_list[0]

        # Compute data-size weights if available (fallback to equal weights)
        weights = None
        try:
            if hasattr(self, '_filtered_indices') and getattr(self, '_client_pool', None) and len(self._filtered_indices) == len(raw_client_model_or_grad_list):
                sizes = []
                for idx in self._filtered_indices:
                    try:
                        size = len(getattr(self._client_pool[idx].dataloader, 'dataset', []))
                    except Exception:
                        size = 0
                    sizes.append(int(size))
                tot = float(sum(sizes))
                if tot > 0.0:
                    weights = [s / tot for s in sizes]
        except Exception:
            weights = None

        if weights is None:
            weights = [1.0 / num_clients for _ in range(num_clients)]

        for key in first_model.keys():
            # Weighted average per parameter
            acc = None
            for w, client_model in zip(weights, raw_client_model_or_grad_list):
                if acc is None:
                    acc = client_model[key].clone() * float(w)
                else:
                    acc += client_model[key] * float(w)
            aggregated_model[key] = acc

        logger.debug(f"Aggregated {num_clients} client models (weighted={weights is not None})")

        return aggregated_model

    def aggregate(self, raw_client_model_or_grad_list: List[OrderedDict] = None) -> OrderedDict:
        """
        完整的聚合流程：验证 + 聚合

        Args:
            raw_client_model_or_grad_list: 客户端模型列表

        Returns:
            aggregated_model: 聚合后的模型
        """
        if raw_client_model_or_grad_list is None:
            raw_client_model_or_grad_list = self.model_pool


        # Setup request context ID and optional preflight
        try:
            import uuid
            if not getattr(self, '_request_id', None):
                self._request_id = str(uuid.uuid4())
            os.environ['POL_REQUEST_ID'] = self._request_id
            logger.info(f"[ctx={self._request_id}] Aggregation round started")
        except Exception:
            pass
        try:
            if str(os.getenv('POL_PREFLIGHT', '0')).lower() in ('1', 'true', 'yes') and not getattr(self, '_preflight_done', False):
                if not self._preflight_check():
                    raise RuntimeError('preflight_failed')
                self._preflight_done = True

        except Exception as e:
            logger.error(f"Preflight check failed: {e}")
            raise

        # 聚合前验证
        filtered_list = self._on_before_aggregation(raw_client_model_or_grad_list)

        # 执行聚合（优先尝试外部聚合，失败则回退本地）
        aggregated_model = None
        try:
            ext_model, ext_latency = self._external_aggregate_weights(filtered_list)
            # Track external aggregator metrics
            self._metrics = getattr(self, '_metrics', {})
            try:
                self._metrics['external_agg_latency_s'] = float(ext_latency or 0.0)
            except Exception:
                self._metrics['external_agg_latency_s'] = 0.0
            self._metrics['external_agg_success'] = False
            if ext_model is not None:
                logger.info(f"External aggregation succeeded in {ext_latency:.3f}s (endpoint={os.getenv('POL_AGGREGATOR_ENDPOINT','')})")
                self._metrics['external_agg_success'] = True
                self._metrics['external_agg_latency_s'] = float(ext_latency)
                aggregated_model = ext_model
        except Exception as e:
            logger.debug(f"External aggregation path error (ignored): {e}")
        # Phase A: external-only aggregation enforcement
        if aggregated_model is None and getattr(self, 'require_external_aggregator', False):
            raise RuntimeError('external_aggregator_required_but_unavailable')

        if aggregated_model is None:
            aggregated_model = self._aggregate_alg(filtered_list)

        # 聚合后处理
        # Incentives: update reputation, penalize failures, and distribute rewards
        if self.enable_incentives:
            try:
                # Performance mapping: selected -> 1.0/0.0, not selected -> 0.5
                performances = {}
                for cid in self._client_ids:
                    if cid in self.verification_results:
                        performances[cid] = 1.0 if self.verification_results[cid] else 0.0
                    else:
                        performances[cid] = 0.5

                # Update reputations on-chain via ReputationSystem (which calls chain_proxy)
                self.reputation_system.batch_update_reputations(performances)

                # Penalize failed clients proportionally to their current on-chain stake (minor violation)
                for cid in self.failed_clients:
                    stake_info = chain_proxy.get_stake_info(cid) or {}
                    total_stake = int(stake_info.get('total', 0))
                    if total_stake > 0:
                        rate = self.staking_manager.penalty_rates.get('minor', 0.1)
                        penalty_amt = int(total_stake * rate)
                        if penalty_amt > 0:
                            chain_proxy.penalize(cid, penalty_amt, reason='verification_failed')

                # Fund reward pool for this round (optional, if configured)
                if self.base_reward_per_round_wei > 0:
                    chain_proxy.fund_reward_pool(self.base_reward_per_round_wei)

                # Prepare inputs for reward calculation (for all clients in this round)
                data_sizes = {}
                reputations = {}
                for client in self._client_pool:
                    cid = client.client_id
                    # data size
                    size = 0
                    try:
                        size = len(getattr(client.dataloader, 'dataset', []))
                    except Exception:
                        size = 0
                    data_sizes[cid] = int(size)
                    # reputation in [0,1]
                    try:
                        rep01 = float(self.reputation_system.get_reputation(cid))
                    except Exception:
                        rep01 = 0.5
                    reputations[cid] = rep01

                # Calculate rewards and distribute on-chain
                rewards_float = self.reward_calculator.calculate_rewards(
                    clients=self._client_ids,
                    verification_results=self.verification_results,
                    data_sizes=data_sizes,
                    reputations=reputations,
                    penalty_pool=float(chain_proxy.get_incentive_stats().get('penalty_pool', 0))
                )
                # Convert to wei ints
                client_ids, amounts_wei = [], []
                for cid, amt in rewards_float.items():
                    amt_int = int(amt)
                    if amt_int > 0:
                        client_ids.append(cid)
                        amounts_wei.append(amt_int)

                if client_ids:
                    chain_proxy.distribute_rewards(client_ids, amounts_wei)
            except Exception as e:
                logger.warning(f"Incentive processing failed: {e}")

        aggregated_model = self._on_after_aggregation(aggregated_model)

        # Phase B-3: optional on-chain anchoring (best-effort)
        try:
            self._maybe_anchor_onchain()
        except Exception as e:
            logger.debug(f"Anchoring skipped: {e}")

        self.release_round_payloads()
        return aggregated_model

    def _on_after_aggregation(self, aggregated_model_or_grad: OrderedDict) -> OrderedDict:
        """
        聚合后的处理

        Args:
            aggregated_model_or_grad: 聚合后的模型

        Returns:
            processed_model: 处理后的模型
        """
        # 这里可以添加后处理逻辑，如差分隐私、模型压缩等
        return aggregated_model_or_grad

    def test(self, test_data, device, args):
        """
        测试聚合后的模型

        Args:
            test_data: 测试数据
            device: 计算设备
            args: 参数

        Returns:
            test_results: 测试结果
        """
        if self.model is None:
            logger.warning("No model to test")
            return {}

        self.model.eval()
        self.model.to(device)

        total_loss = 0
        correct = 0
        num_data = 0

        criterion = torch.nn.CrossEntropyLoss()

        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(test_data):
                data, targets = data.to(device), targets.to(device)
                output = self.model(data)

                loss = criterion(output, targets)
                total_loss += loss.item() * data.size(0)

                pred = output.argmax(dim=1)
                correct += pred.eq(targets).sum().item()
                num_data += data.size(0)

        avg_loss = total_loss / num_data if num_data > 0 else 0
        accuracy = 100.0 * correct / num_data if num_data > 0 else 0

        results = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'num_samples': num_data
        }

        logger.info(f"Test results: loss={avg_loss:.4f}, accuracy={accuracy:.2f}%")

        return results

    def get_verification_results(self) -> Dict[str, bool]:
        """
        获取验证结果
        Returns:
            verification_results: 验证结果字典
        """
        return self.verification_results.copy()

    def reset_metrics(self):
        """Reset per-round decentralization/external aggregation metrics to defaults."""
        self._metrics = {
            'external_agg_success': False,
            'external_agg_latency_s': 0.0,
            'remote_majority_responders': 0,
            'remote_majority_yes': 0,
            'remote_verify_latency_p50_s': 0.0,
            'remote_verify_latency_p95_s': 0.0,
            'remote_error_timeout': 0,
            'remote_error_network': 0,
            'remote_error_invalid': 0,
            'remote_error_business': 0,
            'pol_verify_time_s': 0.0,
        }

    @property
    def metrics(self) -> Dict:
        """Read-only snapshot of current metrics (safe shallow copy)."""
        try:
            return dict(getattr(self, '_metrics', {}))
        except Exception:
            return {}


    def reset_verification_results(self):
        """重置验证结果"""
        self.verification_results = {}
        self.verified_clients = set()
        self.failed_clients = set()
