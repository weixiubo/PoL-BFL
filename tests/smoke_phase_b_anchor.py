import os
import sys
sys.path.append('.')
import unittest

from server.aggregation_alg.PoLVerifyAggregator import PoLVerifyAggregator, chain_proxy


class DummyChainProxy:
    def __init__(self):
        self.calls = []
    def anchor_round(self, rid, commit_hash_hex, sigset_hash_hex):
        self.calls.append((rid, commit_hash_hex, sigset_hash_hex))
        return {'txid': '0xabc', 'blockNumber': 123}


class TestAnchor(unittest.TestCase):
    def setUp(self):
        os.environ['POL_ANCHOR_ONCHAIN'] = '1'
    def tearDown(self):
        if 'POL_ANCHOR_ONCHAIN' in os.environ:
            del os.environ['POL_ANCHOR_ONCHAIN']

    def test_anchor_invoked_and_metrics_recorded(self):
        # monkeypatch chain_proxy
        orig = chain_proxy
        try:
            import server.aggregation_alg.PoLVerifyAggregator as mod
            mod.chain_proxy = DummyChainProxy()
            agg = PoLVerifyAggregator(model=None, args={})
            agg._request_id = 'RID1'
            agg.verified_clients = {'c1','c2'}
            agg._pol_commitments = {
                'c1': {'commitment': 'aa'},
                'c2': {'commitment': 'bb'},
            }
            agg._receipts_by_client = {
                'c1': [{'addr': '0x111'}],
                'c2': [{'addr': '0x222'}],
            }
            agg._metrics = {}
            agg._maybe_anchor_onchain()
            # check metrics
            self.assertIn('anchor_txid', agg._metrics)
            self.assertEqual(agg._metrics['anchor_txid'], '0xabc')
        finally:
            mod.chain_proxy = orig


if __name__ == '__main__':
    unittest.main()

