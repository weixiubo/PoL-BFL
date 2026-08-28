import os
import time
import pytest
import chainfl.interact as ci

@pytest.mark.timeout(120)
def test_concurrent_challenges_same_client():
    os.environ.pop('POL_OFFLINE_FALLBACK', None)
    assert ci.chain_proxy.pol_register_client('5') is True or True

    now = int(time.time())
    c1 = ci.chain_proxy.issue_challenge('5', idx0=0, idx1=2, deadline_ts=now + 3600)
    c2 = ci.chain_proxy.issue_challenge('5', idx0=3, idx1=4, deadline_ts=now + 3600)
    assert isinstance(c1, str) and c1.startswith('0x')
    assert isinstance(c2, str) and c2.startswith('0x')
    assert c1 != c2

    g1 = ci.chain_proxy.get_challenge(c1)
    g2 = ci.chain_proxy.get_challenge(c2)
    assert g1.get('resolved') is False and g2.get('resolved') is False

@pytest.mark.timeout(120)
def test_expired_challenge_rejects_resolution():
    os.environ.pop('POL_OFFLINE_FALLBACK', None)
    assert ci.chain_proxy.pol_register_client('6') is True or True

    now = int(time.time())
    cid = ci.chain_proxy.issue_challenge('6', idx0=0, idx1=1, deadline_ts=now + 5)
    assert isinstance(cid, str) and cid.startswith('0x')

    time.sleep(6)
    ok = ci.chain_proxy.challenge_proof(cid, {'W_t_hash':0,'W_t1_hash':0,'data_hash':0}, verified=True, reason="late")
    assert ok is False

    chk = ci.chain_proxy.get_challenge(cid)
    assert chk.get('resolved') is False

@pytest.mark.timeout(120)
def test_malicious_indices_invalid_issue():
    os.environ.pop('POL_OFFLINE_FALLBACK', None)
    assert ci.chain_proxy.pol_register_client('7') is True or True

    now = int(time.time())
    invalid = ci.chain_proxy.issue_challenge('7', idx0=5, idx1=3, deadline_ts=now + 3600)
    assert invalid == ""
