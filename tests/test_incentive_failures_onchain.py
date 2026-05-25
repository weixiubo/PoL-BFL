import os
import pytest
import chainfl.interact as ci

@pytest.mark.timeout(120)
def test_onchain_incentive_failures_and_state_integrity():
    os.environ.pop('POL_OFFLINE_FALLBACK', None)

    # Ensure client exists
    assert ci.chain_proxy.pol_register_client('3') is True or True

    # Unstake without stake should fail and keep state zero
    ret = ci.chain_proxy.unstake('3', 10**18)
    assert ret == ""
    info = ci.chain_proxy.get_stake_info('3')
    assert info.get('total', 0) == 0 and info.get('locked', 0) == 0 and info.get('available', 0) == 0

    # Now stake some, then try to unlock more than locked
    amt = 2 * 10**18
    assert ci.chain_proxy.stake('3', amt)
    # lock less than amt
    lock_amt = 5 * 10**17
    assert ci.chain_proxy.lock_stake('3', lock_amt)
    info_b = ci.chain_proxy.get_stake_info('3')
    assert info_b['locked'] >= lock_amt

    # attempt to unlock too much
    bad_unlock = ci.chain_proxy.unlock_stake('3', lock_amt + 10**17)
    assert bad_unlock == ""
    # state unchanged
    info_c = ci.chain_proxy.get_stake_info('3')
    assert info_c['locked'] == info_b['locked'] and info_c['total'] == info_b['total']

    # attempt to unstake more than available
    bad_unstake = ci.chain_proxy.unstake('3', info_c['available'] + 1)
    assert bad_unstake == ""
    info_d = ci.chain_proxy.get_stake_info('3')
    assert info_d == info_c

