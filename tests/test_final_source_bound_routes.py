from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_every_formal_matrix_entry_point_has_a_clean_source_gate():
    runners = sorted((ROOT / "experiments" / "final").glob("run_*matrix.py"))
    assert runners
    for path in runners:
        text = path.read_text(encoding="utf-8")
        assert "source_identity" in text, path
        assert '"dirty"' in text, path
        assert '"commit"' in text, path


def test_every_final_route_owner_uses_canonical_evidence_sealing():
    owners = (
        "experiments/final/aggregate_table2.py",
        "experiments/final/aggregate_table3.py",
        "experiments/final/aggregate_table4.py",
        "experiments/final/aggregate_table5.py",
        "experiments/final/run_economics.py",
        "experiments/final/run_table7_matrix.py",
        "experiments/final/aggregate_scalability.py",
        "experiments/final/aggregate_noniid.py",
        "experiments/final/aggregate_table10.py",
        "experiments/final/aggregate_table11.py",
        "experiments/final/aggregate_table12.py",
        "scripts/contract_gas_benchmark.py",
        "experiments/final/convergence.py",
        "experiments/final/reputation_evolution.py",
        "experiments/final/aggregate_sensitivity.py",
        "experiments/final/gas_price_stress.py",
        "experiments/final/aggregate_figure6.py",
    )
    assert len(owners) == 17
    for label in owners:
        text = (ROOT / label).read_text(encoding="utf-8")
        assert "seal_evidence" in text, label
        assert "analysis_root=" in text or "analysis_source=" in text, label


def test_every_training_backed_route_declares_formal_result_paths():
    owners = (
        "experiments/final/aggregate_table2.py",
        "experiments/final/aggregate_table3.py",
        "experiments/final/aggregate_table4.py",
        "experiments/final/aggregate_table5.py",
        "experiments/final/run_table7_matrix.py",
        "experiments/final/aggregate_scalability.py",
        "experiments/final/aggregate_noniid.py",
        "experiments/final/convergence.py",
        "experiments/final/reputation_evolution.py",
        "experiments/final/aggregate_sensitivity.py",
        "experiments/final/aggregate_figure6.py",
    )
    assert len(owners) == 11
    for label in owners:
        text = (ROOT / label).read_text(encoding="utf-8")
        assert "formal_result_paths" in text, label


def test_every_gpu_matrix_execution_uses_the_runtime_exclusivity_supervisor():
    runners = (
        "run_adaptive_matrix.py",
        "run_composability_matrix.py",
        "run_cross_hardware_matrix.py",
        "run_layer_matrix.py",
        "run_matrix.py",
        "run_noniid_matrix.py",
        "run_scalability_matrix.py",
        "run_sensitivity_matrix.py",
        "run_sybil_matrix.py",
        "run_table4_matrix.py",
        "run_table5_matrix.py",
    )
    for name in runners:
        text = (ROOT / "experiments" / "final" / name).read_text(encoding="utf-8")
        assert "supervised_gpu_command" in text, name
