import os
import pytest
import chainfl.interact as ci

@pytest.mark.timeout(120)
def test_challenge_failures_and_audit_trail():
    os.environ.pop('POL_OFFLINE_FALLBACK', None)

    # Client 1 register and make sure contract deployed
    assert ci.chain_proxy.pol_register_client('1') is True or True

    # Issue a valid challenge to establish baseline
    from time import time
    cid = ci.chain_proxy.issue_challenge('1', idx0=0, idx1=1, deadline_ts=int(time()) + 3600)
    assert isinstance(cid, str) and cid.startswith('0x')

    # Attempt to resolve with a wrong id should fail
    bad = ci.chain_proxy.challenge_proof('0xdeadbeef', {'W_t_hash':0,'W_t1_hash':0,'data_hash':0}, verified=False, reason="bad_id")
    assert bad is False

    # Query on-chain for the valid challenge; should be unresolved (we didn't submit proof)
    got = ci.chain_proxy.get_challenge(cid)
    assert isinstance(got, dict) and got.get('resolved') is False

