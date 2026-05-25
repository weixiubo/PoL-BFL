import os
import pytest


@pytest.fixture(autouse=True)
def fresh_pol_contract():
    """Ensure each test runs against a fresh PoL contract instance and that
    chainfl.interact.chain_proxy is bound to the latest deployment when Brownie
    is available; otherwise, do nothing to avoid import errors in environments
    without blockchain tooling.
    """
    try:
        # Import lazily to avoid hard dependency when not needed
        from brownie import network
        import chainfl.interact as ci

        # Ensure development network is connected
        try:
            if not network.is_connected():
                network.connect('development')
        except Exception:
            pass

        # Deploy fresh PoL contract and rebind chain proxy
        try:
            ci.PoLContract.deploy({'from': ci.server_accounts})
        except Exception:
            # If deployment fails for any reason, continue; bind last instance
            pass

        try:
            # If a mock proxy is present (missing on-chain methods), replace with a real one
            if not hasattr(ci.chain_proxy, 'stake') or not hasattr(ci.chain_proxy, 'issue_challenge'):
                ci.chain_proxy = ci.chainProxy()
            # Bind to the latest deployed instance
            if len(ci.PoLContract) > 0:
                ci.chain_proxy.pol_contract = ci.PoLContract[len(ci.PoLContract)-1]
            # Clear in-memory caches
            setattr(ci.chain_proxy, '_issued_challenges', [])
            setattr(ci.chain_proxy, '_resolved_challenges', [])
        except Exception:
            pass
    except Exception:
        # Brownie not available; skip fixture actions
        pass

    # Make sure offline fallback is not forced by default
    if os.environ.get('POL_OFFLINE_FALLBACK') == '1':
        del os.environ['POL_OFFLINE_FALLBACK']

    yield

    # Teardown: nothing specific; Brownie handles ephemeral chain state on dev network

