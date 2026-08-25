import time
import pytest
brownie = pytest.importorskip("brownie")
from brownie import project, accounts, network


def test_contract_issue_and_resolve_direct():
    p = project.load(project_path="chainEnv", name="chainServerTest", raise_if_loaded=False)
    p.load_config()
    from brownie.project.chainServerTest import PoLContract
    try:
        if not network.is_connected():
            network.connect('development')
    except Exception:
        pass

    # Deploy fresh
    owner = accounts[0]
    pc = PoLContract.deploy({'from': owner})

    # Register client 1
    client = accounts[1]
    pc.registerClient({'from': client})

    # Issue challenge
    deadline = int(time.time()) + 3600
    tx = pc.issueChallenge(client.address, 0, 1, deadline, {'from': owner})
    ev = tx.events['ChallengeIssued'] if 'ChallengeIssued' in tx.events else tx.events[0]
    chal_id = ev['challengeId'] if isinstance(ev, dict) else ev[0]['challengeId']

    # Resolve challenge
    tx2 = pc.challengeProof(chal_id, 123, 456, 789, True, 'ok', {'from': owner})
    assert 'ChallengeResolved' in tx2.events

    # Query and assert
    res = pc.getChallenge(chal_id)
    assert res[5] is True and res[6] is True  # resolved, success
    assert int(res[8]) == 123 and int(res[9]) == 456 and int(res[10]) == 789

