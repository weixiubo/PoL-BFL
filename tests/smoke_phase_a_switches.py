import os
import sys
sys.path.append('.')

import json
from collections import OrderedDict

# Ensure repo-relative imports work from PoL-BFL/代码 as CWD
from server.aggregation_alg.PoLVerifyAggregator import PoLVerifyAggregator


def _reset_env():
    for k in (
        'POL_REQUIRE_REMOTE_VERIFIER',
        'POL_REQUIRE_EXTERNAL_AGGREGATOR',
        'POL_VERIFIER_ENDPOINTS',
        'POL_DECENT_MODE',
        'POL_REMOTE_MODE',
        'POL_REMOTE_STRATEGY',
        'POL_AGGREGATOR_ENDPOINT',
    ):
        if k in os.environ:
            os.environ.pop(k)


def _base_args(enable_pol=True):
    return {
        'enable_pol': enable_pol,
        'verification_rate': 0.1,
        'pol_delta': 0.1,
        'pol_distance_metric': 'l2',
        'device': 'cpu',
        'use_top_q': False,
        'top_q': 3,
    }


def test_remote_required_without_endpoints():
    _reset_env()
    os.environ['POL_REQUIRE_REMOTE_VERIFIER'] = '1'
    try:
        PoLVerifyAggregator(model=None, args=_base_args(True))
        return {'name': 'remote_required_without_endpoints', 'ok': False, 'msg': 'expected RuntimeError'}
    except RuntimeError as e:
        return {'name': 'remote_required_without_endpoints', 'ok': 'POL_REQUIRE_REMOTE_VERIFIER=1' in str(e), 'msg': str(e)}


def test_remote_required_with_endpoints():
    _reset_env()
    os.environ['POL_REQUIRE_REMOTE_VERIFIER'] = '1'
    os.environ['POL_DECENT_MODE'] = '1'
    os.environ['POL_VERIFIER_ENDPOINTS'] = 'http://dummy:8088'
    agg = PoLVerifyAggregator(model=None, args=_base_args(True))
    # Monkeypatch remote verify to avoid network
    agg._remote_verifier_adapter.verify_response = lambda **kwargs: {'valid': True, 'yes': 3, 'responders': 3}
    ok = agg._verify_full_via_adapter(
        challenge=None, response=None, commitment=None,
        model=None, dataloader=None, criterion=None,
        optimizer_class=None, lr=0.0
    )
    return {'name': 'remote_required_with_endpoints', 'ok': bool(ok)}


def test_external_agg_required_raises_without_result():
    _reset_env()
    os.environ['POL_REQUIRE_EXTERNAL_AGGREGATOR'] = '1'
    # Keep PoL disabled to skip verify complexity in this smoke
    agg = PoLVerifyAggregator(model=None, args=_base_args(False))
    # Bypass pre-aggregation logic and external call
    agg._on_before_aggregation = lambda lst: lst or [OrderedDict()]
    agg._external_aggregate_weights = lambda lst: (None, 0.01)
    try:
        agg.aggregate(raw_client_model_or_grad_list=[OrderedDict()])
        return {'name': 'external_agg_required_raises_without_result', 'ok': False, 'msg': 'expected RuntimeError'}
    except RuntimeError as e:
        return {'name': 'external_agg_required_raises_without_result', 'ok': 'external_aggregator_required_but_unavailable' in str(e), 'msg': str(e)}


def main():
    results = [
        test_remote_required_without_endpoints(),
        test_remote_required_with_endpoints(),
        test_external_agg_required_raises_without_result(),
    ]
    print(json.dumps({'results': results}, ensure_ascii=False))


if __name__ == '__main__':
    main()

