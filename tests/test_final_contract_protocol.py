import json
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).parents[1]


def test_contract_node_engine_matches_bundled_ganache_uws():
    package = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))
    lock = json.loads((ROOT / "package-lock.json").read_text(encoding="utf-8"))
    assert package["engines"]["node"] == ">=18 <21"
    assert lock["packages"][""]["engines"] == package["engines"]

    binaries = {
        path.name
        for path in (
            ROOT
            / "node_modules"
            / "ganache"
            / "node_modules"
            / "@trufflesuite"
            / "uws-js-unofficial"
            / "binaries"
        ).glob("uws_linux_x64_*.node")
    }
    assert {"uws_linux_x64_108.node", "uws_linux_x64_115.node"} <= binaries


def test_contract_replay_deadlines_use_chain_time_not_wall_clock():
    if not (ROOT / "node_modules" / "ganache").exists():
        pytest.skip("contract test dependencies are not installed")
    process = subprocess.run(
        [
            "node",
            str(ROOT / "tests" / "contracts" / "contract_replay_timing.cjs"),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    assert process.returncode == 0, process.stderr or process.stdout
    result = json.loads(process.stdout.strip().splitlines()[-1])
    assert result == {
        "advanced_timestamp": 202,
        "already_past_timestamp": 251,
    }


def test_real_contract_transitions_and_paper_gas_gates():
    if not (ROOT / "node_modules" / "solc").exists():
        pytest.skip("contract test dependencies are not installed")
    process = subprocess.run(
        ["node", str(ROOT / "tests" / "contracts" / "polbfl_protocol_e2e.cjs")],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=240,
        check=False,
    )
    assert process.returncode == 0, process.stderr or process.stdout
    result = json.loads(process.stdout.strip().splitlines()[-1])
    assert result["runtime_bytes"] < 24_576
    assert max(map(int, result["commitment_gas"])) <= 85_000
    assert max(map(int, result["receipt_gas"])) <= 120_000
    assert int(result["slash_gas"]) <= 65_000
    assert int(result["reward_claim_gas"]) <= 45_000
    assert result["slashed_clients"] == 2
