import os
import json
import importlib


def test_sign_receipt_unsigned(monkeypatch):
    mod = importlib.import_module('server.committee.VerifierNode')
    # ensure no key
    monkeypatch.delenv('VERIFIER_PRIV_KEY', raising=False)
    msg = {
        'request_id': 'rid-1',
        'round': 1,
        'client_id': 'clientA',
        'commitmentRoot': 'abc',
        'pair_indices': [0, 1],
        'verifier_params': {'delta': 0.01, 'distance_metric': 'l2', 'min_pair_success_rate': 0.99},
        'valid': True,
        'mode': 'distance_only',
    }
    out = mod._sign_receipt(msg)
    assert 'msg' in out and out['msg'] == msg
    assert 'sig' not in out and 'addr' not in out


def test_sign_receipt_with_key(monkeypatch):
    mod = importlib.import_module('server.committee.VerifierNode')
    # Known private key from web3 docs
    pk = '0x4c0883a69102937d623414e8fca3a39e6b5e3a9c2a1f9b0f8a1d5f7f8a9e6d8f'
    monkeypatch.setenv('VERIFIER_PRIV_KEY', pk)
    msg = {
        'request_id': 'rid-2',
        'round': 2,
        'client_id': 'clientB',
        'commitmentRoot': 'def',
        'pair_indices': [2, 3],
        'verifier_params': {'delta': 0.5, 'distance_metric': 'l2', 'min_pair_success_rate': 0.9},
        'valid': False,
        'mode': 'distance_only',
    }
    out = mod._sign_receipt(msg)
    assert 'msg' in out and out['msg'] == msg
    assert isinstance(out.get('sig'), str) and out['sig'].startswith('0x')
    assert isinstance(out.get('addr'), str) and out['addr'].startswith('0x')

