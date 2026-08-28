#!/usr/bin/env python3
"""缩减规模检查 RQ1 MNIST 轻量验证实验数据是否符合预期。

不画图，只打印：
- 每个攻击 × 方法的最终精度 / TPR / FPR
- 若干关键预期关系是否满足（PoL 优于基线、无攻击场景正常、标签翻转为已知局限等）

运行位置：PoL-BFL/Code
用法：python3 experiments/scripts/analyze_rq1_mnist_validation.py
"""

import csv
from pathlib import Path
from statistics import mean

ATTACKS = [
    "no_attack",
    "byzantine_random_noise",
    "byzantine_model_replacement",
    "byzantine_gradient_inversion",
    "byzantine_label_flipping",
    "byzantine_alie",
    "byzantine_ipm",
    "byzantine_minmax",
    "free_riding_no_training",
    "free_riding_lazy_training",
    "free_riding_minimal_update",
]

METHODS_GPU0 = ["Vanilla_FL", "Krum", "Trimmed_Mean", "Median"]
METHODS_GPU1 = ["ShapleyFL", "FoolsGold", "PoL_FL"]
ALL_METHODS = METHODS_GPU0 + METHODS_GPU1

BASE_DIR_GPU0 = Path("experiments/results/validation/rq1_mnist_smoke_gpu0")
BASE_DIR_GPU1 = Path("experiments/results/validation/rq1_mnist_smoke_gpu1")


def load_last_row(path: Path):
    """读取 CSV 最后一行及表头。"""
    if not path.exists():
        return None, None
    header = None
    last = None
    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return None, None
        for row in reader:
            if row:
                last = row
    return header, last


def extract_metrics(header, row):
    """Extract the metrics used by this report."""
    if header is None or row is None:
        return None
    idx = {name: i for i, name in enumerate(header)}

    def get(name, default=0.0):
        i = idx.get(name)
        if i is None or i >= len(row) or row[i] == "":
            return default
        try:
            return float(row[i])
        except ValueError:
            return default

    return {
        "test_accuracy": get("test_accuracy"),
        "tpr": get("detection_tpr"),
        "fpr": get("detection_fpr"),
        "verify_rate": get("verification_pass_rate"),
    }


def find_metrics(attack: str, method: str):
    """在两个输出目录中查找相应 CSV 并提取最后一轮指标。"""
    name = f"rq1_rounds_MNIST_{attack}_{method}.csv"
    for base in (BASE_DIR_GPU0, BASE_DIR_GPU1):
        path = base / name
        header, row = load_last_row(path)
        if header is not None:
            m = extract_metrics(header, row)
            if m is not None:
                return m
    return None


def summarize_all():
    print("=" * 100)
    print("RQ1 MNIST 轻量验证结果总览（最终一轮指标）")
    print("=" * 100)
    print()

    all_data = {}

    for attack in ATTACKS:
        print(f"[攻击] {attack}")
        all_data[attack] = {}
        row_strs = []
        for method in ALL_METHODS:
            m = find_metrics(attack, method)
            if m is None:
                row_strs.append(f"  - {method:12s}: MISSING")
            else:
                all_data[attack][method] = m
                row_strs.append(
                    f"  - {method:12s}: acc={m['test_accuracy']:.4f}, TPR={m['tpr']:.3f}, FPR={m['fpr']:.3f}"
                )
        print("\n".join(row_strs))
        print()

    return all_data


def check_expectations(all_data):
    print("=" * 100)
    print("与预期的符合程度检查（启发式）")
    print("=" * 100)
    print()

    issues = []

    # 1) 无攻击场景：精度应较高，TPR/FPR ≈ 0
    attack = "no_attack"
    if attack in all_data:
        print("[检查] 无攻击场景 (no_attack)")
        for method, m in all_data[attack].items():
            acc, tpr, fpr = m["test_accuracy"], m["tpr"], m["fpr"]
            flags = []
            if acc < 0.95:
                flags.append("acc<0.95")
            if tpr > 0.05:
                flags.append("TPR>0.05")
            if fpr > 0.05:
                flags.append("FPR>0.05")
            if flags:
                msg = f"  - {method}: 可能异常 ({', '.join(flags)}), acc={acc:.4f}, TPR={tpr:.3f}, FPR={fpr:.3f}"
                print(msg)
                issues.append((attack, method, msg))
            else:
                print(
                    f"  - {method}: OK acc={acc:.4f}, TPR={tpr:.3f}, FPR={fpr:.3f}"
                )
        print()

    # 2) 标签翻转：PoL_FL 难以检测，TPR 接近 0 反而是预期
    attack = "byzantine_label_flipping"
    if attack in all_data and "PoL_FL" in all_data[attack]:
        m = all_data[attack]["PoL_FL"]
        tpr = m["tpr"]
        print("[检查] 标签翻转 (PoL 已知局限)")
        if tpr > 0.3:
            msg = f"  - PoL_FL: TPR={tpr:.3f} exceeds the configured reference range"
            print(msg)
            issues.append((attack, "PoL_FL", msg))
        else:
            print(f"  - PoL_FL: OK TPR={tpr:.3f} (接近 0，符合局限性预期)")
        print()

    # 3) 其他攻击：PoL_FL 在 TPR / FPR / acc 上应整体优于或不差于鲁棒基线
    robust_baselines = ["Krum", "Trimmed_Mean", "Median"]
    for attack in ATTACKS:
        if attack in ("no_attack", "byzantine_label_flipping"):
            continue
        if attack not in all_data or "PoL_FL" not in all_data[attack]:
            continue
        pol = all_data[attack]["PoL_FL"]
        pol_acc, pol_tpr, pol_fpr = pol["test_accuracy"], pol["tpr"], pol["fpr"]
        print(f"[检查] 攻击 {attack} 下 PoL_FL vs 鲁棒基线")
        best_acc = None
        best_tpr = None
        best_fpr = None
        for method in robust_baselines:
            m = all_data[attack].get(method)
            if not m:
                continue
            acc, tpr, fpr = m["test_accuracy"], m["tpr"], m["fpr"]
            best_acc = acc if best_acc is None else max(best_acc, acc)
            best_tpr = tpr if best_tpr is None else max(best_tpr, tpr)
            best_fpr = fpr if best_fpr is None else min(best_fpr, fpr)

        if best_acc is None:
            print("  - 无鲁棒基线数据，跳过")
            print()
            continue

        # 启发式规则：
        # - PoL_FL 精度不能明显低于最佳鲁棒基线
        # - PoL_FL TPR 不应明显低于最佳鲁棒基线
        # Compare PoL-FL FPR with the lowest-FPR robust baseline.
        acc_gap = best_acc - pol_acc
        tpr_gap = best_tpr - pol_tpr
        fpr_gap = pol_fpr - best_fpr
        flags = []
        if acc_gap > 0.05:
            flags.append(f"acc 落后 {acc_gap:.3f}")
        if tpr_gap > 0.2:
            flags.append(f"TPR 落后 {tpr_gap:.3f}")
        if fpr_gap > 0.2:
            flags.append(f"FPR 明显更高 {fpr_gap:.3f}")

        if flags:
            msg = (
                f"  - PoL_FL 在 {attack} 下相对鲁棒基线可能偏弱: "
                f"acc={pol_acc:.4f}, TPR={pol_tpr:.3f}, FPR={pol_fpr:.3f}, "
                f"问题: {', '.join(flags)}"
            )
            print(msg)
            issues.append((attack, "PoL_FL", msg))
        else:
            print(
                f"  - PoL_FL: OK acc={pol_acc:.4f}, TPR={pol_tpr:.3f}, FPR={pol_fpr:.3f}"
            )
        print()

    print("=" * 100)
    if not issues:
        print("总体结论：未发现明显违反预期的点（在当前启发式规则下）。")
    else:
        print(f"总体结论：发现 {len(issues)} 个可能需要进一步人工检查的点：")
        for attack, method, msg in issues:
            print(f"  - [{attack}] {method}: {msg}")
    print("=" * 100)


def main():
    if not BASE_DIR_GPU0.exists() or not BASE_DIR_GPU1.exists():
        print("The result directory is unavailable; run the RQ1 MNIST reduced-scale configuration first.")
        return
    all_data = summarize_all()
    check_expectations(all_data)


if __name__ == "__main__":
    main()
