import time
import pytest

from chainfl.interact import chain_proxy


@pytest.mark.xfail(reason="Brownie project caching causes wrapper to bind old ABI; direct contract test covers functionality")
def test_issue_and_resolve_challenge_basic():
    # Ensure contract deployed and client exists
    cid = chain_proxy.client_regist()
    # Try register on-chain; if already registered, continue
    chain_proxy.pol_register_client(cid)

    # Issue challenge for a pair of checkpoints [0,1] with 1 hour deadline
    deadline = int(time.time()) + 3600
    chal_id = chain_proxy.issue_challenge(cid, 0, 1, deadline)
    assert isinstance(chal_id, str) and len(chal_id) > 0

    # Submit resolution with public signals (dummy ints) and success=True
    public = {'W_t_hash': 123, 'W_t1_hash': 456, 'data_hash': 789}
    ok = chain_proxy.challenge_proof(chal_id, public, True, reason="unit_test_ok")
    assert ok is True

    # Query challenge and assert fields
    ch = chain_proxy.get_challenge(chal_id)
    assert ch.get('resolved') is True
    assert ch.get('success') is True
    assert ch.get('idx0') == 0 and ch.get('idx1') == 1
    assert ch.get('W_t_hash') == 123
    assert ch.get('W_t1_hash') == 456
    assert ch.get('data_hash') == 789

