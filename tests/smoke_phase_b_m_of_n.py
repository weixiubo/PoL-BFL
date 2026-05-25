import os, json
import sys
sys.path.append('.')
import unittest

try:
    from eth_account import Account
    from eth_account.messages import encode_defunct
    HAVE_ETH = True
except Exception:
    HAVE_ETH = False

from server.aggregation_alg.PoLVerifyAggregator import PoLVerifyAggregator


def make_receipt(priv, msg):
    text = json.dumps(msg, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    pk = priv if priv.startswith('0x') else '0x'+priv
    acc = Account.from_key(pk)
    sig = Account.sign_message(encode_defunct(text=text), acc.key).signature.hex()
    return {'msg': msg, 'sig': sig, 'addr': acc.address}


@unittest.skipUnless(HAVE_ETH, 'eth_account not available')
class TestMofN(unittest.TestCase):
    def setUp(self):
        os.environ['POL_M_OF_N'] = '2/3'
        self.commitment = 'deadbeef'
        self.pk1 = Account.create().key.hex()
        self.pk2 = Account.create().key.hex()
        self.pk3 = Account.create().key.hex()
        self.addr1 = Account.from_key(self.pk1).address
        self.addr2 = Account.from_key(self.pk2).address
        self.addr3 = Account.from_key(self.pk3).address
        os.environ['POL_VERIFIER_ADDRESSES'] = ','.join([self.addr1, self.addr2, self.addr3])

    def tearDown(self):
        for k in ('POL_M_OF_N','POL_VERIFIER_ADDRESSES','POL_AGGREGATOR_ADDR'):
            if k in os.environ:
                del os.environ[k]

    def test_accept_2_of_3(self):
        msg_yes = {'valid': True, 'commitmentRoot': self.commitment}
        r1 = make_receipt(self.pk1, msg_yes)
        r2 = make_receipt(self.pk2, msg_yes)
        r3 = make_receipt(self.pk3, {'valid': False, 'commitmentRoot': self.commitment})

        agg = PoLVerifyAggregator(model=None, args={})
        ok, m, n = agg._verify_m_of_n([r1, r2, r3], self.commitment)
        self.assertTrue(ok)
        self.assertEqual(m, 2)
        self.assertEqual(n, 3)

    def test_reject_1_of_3(self):
        msg_yes = {'valid': True, 'commitmentRoot': self.commitment}
        r1 = make_receipt(self.pk1, msg_yes)
        r2 = make_receipt(self.pk2, {'valid': False, 'commitmentRoot': self.commitment})
        r3 = make_receipt(self.pk3, {'valid': False, 'commitmentRoot': self.commitment})
        agg = PoLVerifyAggregator(model=None, args={})
        ok, m, n = agg._verify_m_of_n([r1, r2, r3], self.commitment)
        self.assertFalse(ok)
        self.assertEqual(m, 1)
        self.assertEqual(n, 3)

    def test_agg_must_not_sign(self):
        # If POL_AGGREGATOR_ADDR equals signer, should raise/return False
        os.environ['POL_AGGREGATOR_ADDR'] = self.addr1
        msg_yes = {'valid': True, 'commitmentRoot': self.commitment}
        r1 = make_receipt(self.pk1, msg_yes)
        agg = PoLVerifyAggregator(model=None, args={})
        ok, m, n = agg._verify_m_of_n([r1], self.commitment)
        self.assertFalse(ok)

    def test_commitment_mismatch_ignored(self):
        msg_yes = {'valid': True, 'commitmentRoot': 'other'}
        r1 = make_receipt(self.pk1, msg_yes)
        r2 = make_receipt(self.pk2, {'valid': True, 'commitmentRoot': self.commitment})
        r3 = make_receipt(self.pk3, {'valid': True, 'commitmentRoot': self.commitment})
        agg = PoLVerifyAggregator(model=None, args={})
        ok, m, n = agg._verify_m_of_n([r1, r2, r3], self.commitment)
        self.assertTrue(ok)
        self.assertEqual(m, 2)
        self.assertEqual(n, 2)


if __name__ == '__main__':
    unittest.main()

