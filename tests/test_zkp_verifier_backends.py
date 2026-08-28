from server.zkp.ZKPVerifier import ZKPVerifier


PROOF = {
    "pi_a": ["1", "2", "1"],
    "pi_b": [["3", "4"], ["5", "6"], ["1", "0"]],
    "pi_c": ["7", "8", "1"],
    "protocol": "groth16",
    "curve": "bn128",
}
PUBLIC_SIGNALS = {
    "W_t_hash": "11",
    "W_t1_hash": "12",
    "data_hash": "13",
    "max_distance": "14",
}


def test_onchain_backend_uses_the_injected_contract_adapter():
    calls = []

    def contract_adapter(proof, public_signals):
        calls.append((proof, public_signals))
        return True

    verifier = ZKPVerifier(
        use_simulation=False,
        use_onchain=True,
        onchain_verifier=contract_adapter,
    )

    assert verifier.verify_proof(PROOF, PUBLIC_SIGNALS) is True
    assert calls == [(PROOF, PUBLIC_SIGNALS)]


def test_onchain_backend_fails_closed_without_offchain_downgrade():
    def unavailable_contract(_proof, _public_signals):
        raise ConnectionError("contract unavailable")

    verifier = ZKPVerifier(
        use_simulation=False,
        use_onchain=True,
        onchain_verifier=unavailable_contract,
    )
    verifier._verify_offchain = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("on-chain requests must not use the off-chain backend")
    )

    assert verifier.verify_proof(PROOF, PUBLIC_SIGNALS) is False


def test_onchain_backend_rejects_malformed_inputs_before_contract_call():
    verifier = ZKPVerifier(
        use_simulation=False,
        use_onchain=True,
        onchain_verifier=lambda *_args: True,
    )

    assert verifier.verify_proof({}, PUBLIC_SIGNALS) is False
    assert verifier.verify_proof(PROOF, {"W_t_hash": "11"}) is False


def test_verification_backend_selection_is_unambiguous():
    try:
        ZKPVerifier(use_simulation=True, use_onchain=True)
    except ValueError as exc:
        assert "exactly one" in str(exc)
    else:
        raise AssertionError("ambiguous backend selection must be rejected")
