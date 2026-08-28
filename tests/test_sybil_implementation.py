#!/usr/bin/env python3
"""
Sybil Attack 实现验证测试脚本
Validates the three-layer Sybil-defense implementation.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_sybil_attack_import():
    """Test 1: 验证 Sybil Attack 导入"""
    print("=" * 60)
    print("Test 1: 验证 Sybil Attack 导入")
    print("=" * 60)
    try:
        from experiments.attacks.sybil_attacks import SybilAttack, create_sybil_attack
        print("[PASS] Sybil Attack 导入成功")
        return True
    except Exception as e:
        print(f"[FAIL] 导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_sybil_attack_creation():
    """Test 2: 创建 Sybil Attack 实例"""
    print("\n" + "=" * 60)
    print("Test 2: 创建 Sybil Attack 实例")
    print("=" * 60)
    try:
        from experiments.attacks.sybil_attacks import SybilAttack
        sybil = SybilAttack(num_identities=5, shared_data_ratio=1.0)
        identities = sybil.create_identities()
        print(f"[PASS] 创建成功，虚假身份数: {len(identities)}")
        print(f"   虚假身份: {identities}")
        assert len(identities) == 5, f"Expected 5 identities, got {len(identities)}"
        return True
    except Exception as e:
        print(f"[FAIL] 创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_metrics_import():
    """Test 3: 验证指标函数导入"""
    print("\n" + "=" * 60)
    print("Test 3: 验证指标函数导入")
    print("=" * 60)
    try:
        from experiments.scripts.utils.metrics import (
            compute_sybil_detection_rate,
            compute_identity_correlation,
            compute_sybil_attack_success_rate,
            compute_sybil_attack_cost,
            compute_reward_dilution,
            compute_reputation_penalty
        )
        print("[PASS] 所有指标函数导入成功")
        return True
    except Exception as e:
        print(f"[FAIL] 导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_metrics_functions():
    """Test 4: 测试指标函数"""
    print("\n" + "=" * 60)
    print("Test 4: 测试指标函数")
    print("=" * 60)
    try:
        from experiments.scripts.utils.metrics import (
            compute_sybil_detection_rate,
            compute_sybil_attack_success_rate,
            compute_sybil_attack_cost,
            compute_reward_dilution,
            compute_reputation_penalty
        )

        # Test compute_sybil_detection_rate
        detected = ["attacker_sybil_0", "attacker_sybil_1"]
        actual = ["attacker_sybil_0", "attacker_sybil_1", "attacker_sybil_2"]
        rate = compute_sybil_detection_rate(detected, actual)
        print(f"[PASS] compute_sybil_detection_rate: {rate:.2%}")
        assert 0 <= rate <= 1, f"Detection rate should be between 0 and 1, got {rate}"

        # Test compute_sybil_attack_success_rate
        results = {"id1": True, "id2": False, "id3": True}
        success = compute_sybil_attack_success_rate(results)
        print(f"[PASS] compute_sybil_attack_success_rate: {success:.2%}")
        assert 0 <= success <= 1, f"Success rate should be between 0 and 1, got {success}"

        # Test compute_sybil_attack_cost
        cost = compute_sybil_attack_cost(staking_cost=100, computation_cost=10, num_identities=5)
        print(f"[PASS] compute_sybil_attack_cost: {cost}")
        assert cost == 550, f"Expected cost 550, got {cost}"

        # Test compute_reward_dilution
        dilution = compute_reward_dilution(total_reward=1000, num_identities=5)
        print(f"[PASS] compute_reward_dilution: {dilution:.2%}")
        assert 0 <= dilution <= 1, f"Dilution should be between 0 and 1, got {dilution}"

        # Test compute_reputation_penalty
        changes = {"id1": -0.3, "id2": -0.4, "id3": -0.5}
        penalty = compute_reputation_penalty(changes)
        print(f"[PASS] compute_reputation_penalty: {penalty:.2f}")
        assert penalty < 0, f"Penalty should be negative, got {penalty}"

        return True
    except Exception as e:
        print(f"[FAIL] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_rq1_config():
    """Test 5: 验证 RQ1 配置"""
    print("\n" + "=" * 60)
    print("Test 5: 验证 RQ1 配置")
    print("=" * 60)
    try:
        from experiments.scripts.runners.run_rq1_security import RQ1_CONFIG

        # Check if sybil_attack is in attacks
        assert 'sybil_attack' in RQ1_CONFIG['attacks'], "sybil_attack not in RQ1_CONFIG"
        print("[PASS] RQ1_CONFIG 包含 sybil_attack")

        # Check sybil_attack configuration
        sybil_config = RQ1_CONFIG['attacks']['sybil_attack']
        assert 'malicious_ratios' in sybil_config, "malicious_ratios not in sybil_attack config"
        assert 'num_identities' in sybil_config, "num_identities not in sybil_attack config"
        assert 'shared_data_ratio' in sybil_config, "shared_data_ratio not in sybil_attack config"
        print(f"[PASS] RQ1 Sybil Attack 配置: {sybil_config}")

        return True
    except Exception as e:
        print(f"[FAIL] 验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_rq4_config():
    """Test 6: 验证 RQ4 配置"""
    print("\n" + "=" * 60)
    print("Test 6: 验证 RQ4 配置")
    print("=" * 60)
    try:
        from experiments.scripts.runners.run_rq4_incentive import RQ4_CONFIG

        # Check if sybil_attack is in scenarios
        assert 'sybil_attack' in RQ4_CONFIG['scenarios'], "sybil_attack not in RQ4_CONFIG scenarios"
        print("[PASS] RQ4_CONFIG 包含 sybil_attack 场景")

        # Check sybil_config
        assert 'sybil_config' in RQ4_CONFIG, "sybil_config not in RQ4_CONFIG"
        sybil_config = RQ4_CONFIG['sybil_config']
        assert 'base_scenario' in sybil_config, "base_scenario not in sybil_config"
        assert 'identities_per_attacker' in sybil_config, "identities_per_attacker not in sybil_config"
        print(f"[PASS] RQ4 Sybil Attack 配置: {sybil_config}")

        return True
    except Exception as e:
        print(f"[FAIL] 验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("Sybil Attack 实现验证测试")
    print("=" * 60)

    results = []
    results.append(("Sybil Attack 导入", test_sybil_attack_import()))
    results.append(("Sybil Attack 创建", test_sybil_attack_creation()))
    results.append(("指标函数导入", test_metrics_import()))
    results.append(("指标函数测试", test_metrics_functions()))
    results.append(("RQ1 配置验证", test_rq1_config()))
    results.append(("RQ4 配置验证", test_rq4_config()))

    # Summary
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "[PASS] 通过" if result else "[FAIL] 失败"
        print(f"{status}: {test_name}")

    print(f"\n总体结果: {passed}/{total} 通过")

    if passed == total:
        print("\n[PASS] 所有验证测试通过。")
        return 0
    else:
        print(f"\n[WARNING]  有 {total - passed} 个测试失败")
        return 1

if __name__ == "__main__":
    sys.exit(main())
