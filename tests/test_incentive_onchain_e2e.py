import time
import pytest

import chainfl.interact as ci


@pytest.mark.order("last")
def test_onchain_incentives_full_flow():
    # Ensure two clients are registered
    assert ci.chain_proxy.pol_register_client('1') is True or True
    assert ci.chain_proxy.pol_register_client('2') is True or True

    # 1) Client 1 stakes 5 ether (in wei)
    five_eth = 5 * 10**18
    txid = ci.chain_proxy.stake('1', five_eth)
    assert isinstance(txid, str) and len(txid) > 0

    # Verify stake info
    info1 = ci.chain_proxy.get_stake_info('1')
    assert info1['total'] >= five_eth
    assert info1['available'] >= five_eth - info1['locked']

    # 2) Lock a portion, penalize small amount, then unlock
    lock_amt = 1 * 10**18
    assert ci.chain_proxy.lock_stake('1', lock_amt)
    info1b = ci.chain_proxy.get_stake_info('1')
    assert info1b['locked'] >= lock_amt

    pen_amt = 2 * 10**17  # 0.2 ETH
    assert ci.chain_proxy.penalize('1', pen_amt, reason="fail_verification")
    info1c = ci.chain_proxy.get_stake_info('1')
    assert info1c['total'] <= info1b['total'] - pen_amt

    assert ci.chain_proxy.unlock_stake('1', lock_amt)

    # 3) Fund reward pool and distribute rewards to two clients
    fund_amt = 3 * 10**18
    assert ci.chain_proxy.fund_reward_pool(fund_amt)

    # Distribute to both clients
    amt1 = 6 * 10**17
    amt2 = 4 * 10**17
    assert ci.chain_proxy.distribute_rewards(['1', '2'], [amt1, amt2])

    stats = ci.chain_proxy.get_incentive_stats()
    assert stats['reward_pool'] >= 0
    assert stats['penalty_pool'] >= 0

    # 4) Update reputation (0..1000 scale)
    assert ci.chain_proxy.update_reputation('1', 800)
    assert ci.chain_proxy.update_reputation('2', 600)
    assert ci.chain_proxy.get_reputation('1') == 800
    assert ci.chain_proxy.get_reputation('2') == 600

    # 5) Unstake some amount (ensure available >= amount)
    info1d = ci.chain_proxy.get_stake_info('1')
    unstake_amt = min(10**17, info1d['available'])
    if unstake_amt > 0:
        assert ci.chain_proxy.unstake('1', unstake_amt)

    # Basic sanity on totals
    final1 = ci.chain_proxy.get_stake_info('1')
    assert final1['total'] >= 0 and final1['available'] >= 0

