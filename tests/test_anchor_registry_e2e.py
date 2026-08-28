"""
End-to-end test for AnchorRegistry on-chain anchoring functionality.

Tests:
1. AnchorRegistry contract deployment
2. anchor_round() method in chainProxy
3. Integration with PoLVerifyAggregator._maybe_anchor_onchain()
4. Query and verification of anchored rounds
"""

import os
import sys
import unittest
import hashlib

sys.path.append('.')

from chainfl.interact import chainProxy


class TestAnchorRegistryE2E(unittest.TestCase):
    """End-to-end test for on-chain anchoring via AnchorRegistry"""

    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        # Enable on-chain anchoring
        os.environ['POL_ANCHOR_ONCHAIN'] = '1'

        # Initialize chain proxy (will deploy contracts)
        try:
            cls.chain_proxy = chainProxy()
            cls.has_blockchain = cls.chain_proxy.anchor_registry is not None
        except Exception as e:
            print(f"Blockchain not available: {e}")
            cls.has_blockchain = False

    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        if 'POL_ANCHOR_ONCHAIN' in os.environ:
            del os.environ['POL_ANCHOR_ONCHAIN']

    def test_01_anchor_registry_deployed(self):
        """Test that AnchorRegistry contract is deployed"""
        if not self.has_blockchain:
            self.skipTest("Blockchain not available")

        self.assertIsNotNone(self.chain_proxy.anchor_registry)
        print(f"[PASS] AnchorRegistry deployed at: {self.chain_proxy.anchor_registry.address}")

    def test_02_anchor_round_basic(self):
        """Test basic anchor_round() functionality"""
        if not self.has_blockchain:
            self.skipTest("Blockchain not available")

        # Prepare test data
        round_id = "test_round_001"
        commit_hash = hashlib.sha256(b"client1_commit,client2_commit").hexdigest()
        sigset_hash = hashlib.sha256(b"verifier1,verifier2").hexdigest()

        # Anchor the round
        result = self.chain_proxy.anchor_round(round_id, commit_hash, sigset_hash)

        # Verify result
        self.assertIn('txid', result)
        self.assertIn('blockNumber', result)
        self.assertIsNotNone(result['txid'])
        self.assertGreater(result['blockNumber'], 0)

        print(f"[PASS] Round anchored: txid={result['txid']}, block={result['blockNumber']}")

    def test_03_anchor_round_query(self):
        """Test querying anchored round from contract"""
        if not self.has_blockchain:
            self.skipTest("Blockchain not available")

        from brownie import web3

        # Prepare test data
        round_id = "test_round_002"
        commit_hash = hashlib.sha256(b"client3_commit,client4_commit").hexdigest()
        sigset_hash = hashlib.sha256(b"verifier3,verifier4").hexdigest()

        # Anchor the round
        result = self.chain_proxy.anchor_round(round_id, commit_hash, sigset_hash)

        # Query the anchor from contract
        round_id_bytes32 = web3.keccak(text=round_id)
        anchor_info = self.chain_proxy.anchor_registry.getAnchor(round_id_bytes32)

        # Verify anchor info
        self.assertEqual(len(anchor_info), 5)  # (commitHash, sigsetHash, aggregator, timestamp, blockNumber)

        # Convert bytes32 to hex for comparison
        stored_commit_hash = anchor_info[0].hex()
        stored_sigset_hash = anchor_info[1].hex()

        self.assertEqual(stored_commit_hash, commit_hash)
        self.assertEqual(stored_sigset_hash, sigset_hash)
        self.assertGreater(anchor_info[3], 0)  # timestamp > 0
        self.assertEqual(anchor_info[4], result['blockNumber'])

        print(f"[PASS] Anchor verified: commit={stored_commit_hash[:16]}..., sigset={stored_sigset_hash[:16]}...")

    def test_04_anchor_round_duplicate_prevention(self):
        """Test that duplicate round anchoring is prevented"""
        if not self.has_blockchain:
            self.skipTest("Blockchain not available")

        # Prepare test data
        round_id = "test_round_003"
        commit_hash = hashlib.sha256(b"client5_commit").hexdigest()
        sigset_hash = hashlib.sha256(b"verifier5").hexdigest()

        # Anchor the round first time
        result1 = self.chain_proxy.anchor_round(round_id, commit_hash, sigset_hash)
        self.assertIsNotNone(result1['txid'])

        # Try to anchor the same round again (should fail)
        with self.assertRaises(Exception) as context:
            self.chain_proxy.anchor_round(round_id, commit_hash, sigset_hash)

        # Verify error message contains "already anchored"
        error_msg = str(context.exception).lower()
        self.assertTrue('already anchored' in error_msg or 'revert' in error_msg)

        print(f"[PASS] Duplicate anchoring prevented: {error_msg[:100]}")

    def test_05_integration_with_aggregator(self):
        """Test integration with PoLVerifyAggregator._maybe_anchor_onchain()"""
        if not self.has_blockchain:
            self.skipTest("Blockchain not available")

        from server.aggregation_alg.PoLVerifyAggregator import PoLVerifyAggregator
        import server.aggregation_alg.PoLVerifyAggregator as agg_module

        # Temporarily replace chain_proxy in aggregator module
        original_chain_proxy = agg_module.chain_proxy
        try:
            agg_module.chain_proxy = self.chain_proxy

            # Create aggregator instance
            agg = PoLVerifyAggregator(model=None, args={})
            agg._request_id = 'integration_test_round'
            agg.verified_clients = {'client1', 'client2'}
            agg._pol_commitments = {
                'client1': {'commitment': 'commit_hash_1'},
                'client2': {'commitment': 'commit_hash_2'},
            }
            agg._receipts_by_client = {
                'client1': [{'addr': '0x111'}],
                'client2': [{'addr': '0x222'}],
            }
            agg._metrics = {}

            # Call _maybe_anchor_onchain
            agg._maybe_anchor_onchain()

            # Verify metrics were recorded
            self.assertIn('anchor_txid', agg._metrics)
            self.assertIn('anchor_block', agg._metrics)
            self.assertIsNotNone(agg._metrics['anchor_txid'])
            self.assertGreater(agg._metrics['anchor_block'], 0)

            print(f"[PASS] Aggregator integration successful: txid={agg._metrics['anchor_txid']}, block={agg._metrics['anchor_block']}")

        finally:
            # Restore original chain_proxy
            agg_module.chain_proxy = original_chain_proxy

    def test_06_anchor_verification(self):
        """Test verifyAnchor() contract method"""
        if not self.has_blockchain:
            self.skipTest("Blockchain not available")

        from brownie import web3

        # Prepare test data
        round_id = "test_round_004"
        commit_hash = hashlib.sha256(b"client6_commit").hexdigest()
        sigset_hash = hashlib.sha256(b"verifier6").hexdigest()

        # Anchor the round
        self.chain_proxy.anchor_round(round_id, commit_hash, sigset_hash)

        # Verify with correct hashes
        round_id_bytes32 = web3.keccak(text=round_id)
        commit_hash_bytes32 = bytes.fromhex(commit_hash)
        sigset_hash_bytes32 = bytes.fromhex(sigset_hash)

        is_valid = self.chain_proxy.anchor_registry.verifyAnchor(
            round_id_bytes32,
            commit_hash_bytes32,
            sigset_hash_bytes32
        )
        self.assertTrue(is_valid)

        # Verify with wrong commit hash (should fail)
        wrong_commit_hash = hashlib.sha256(b"wrong_commit").hexdigest()
        wrong_commit_bytes32 = bytes.fromhex(wrong_commit_hash)

        is_valid_wrong = self.chain_proxy.anchor_registry.verifyAnchor(
            round_id_bytes32,
            wrong_commit_bytes32,
            sigset_hash_bytes32
        )
        self.assertFalse(is_valid_wrong)

        print(f"[PASS] Anchor verification working: correct=True, wrong=False")

    def test_07_statistics(self):
        """Test getStats() contract method"""
        if not self.has_blockchain:
            self.skipTest("Blockchain not available")

        # Get initial stats
        stats_before = self.chain_proxy.anchor_registry.getStats()
        total_before = stats_before[0]

        # Anchor a new round
        round_id = f"test_round_stats_{total_before}"
        commit_hash = hashlib.sha256(f"commit_{total_before}".encode()).hexdigest()
        sigset_hash = hashlib.sha256(f"sigset_{total_before}".encode()).hexdigest()

        self.chain_proxy.anchor_round(round_id, commit_hash, sigset_hash)

        # Get stats after
        stats_after = self.chain_proxy.anchor_registry.getStats()
        total_after = stats_after[0]

        # Verify total increased by 1
        self.assertEqual(total_after, total_before + 1)

        print(f"[PASS] Statistics working: total anchors increased from {total_before} to {total_after}")


def run_tests():
    """Run all tests and print summary"""
    print("=" * 80)
    print("AnchorRegistry End-to-End Tests")
    print("=" * 80)

    # Run tests
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAnchorRegistryE2E)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")

    if result.wasSuccessful():
        print("\n[PASS] All tests passed.")
        return 0
    else:
        print("\n[FAIL] Some tests failed")
        return 1


if __name__ == '__main__':
    sys.exit(run_tests())

