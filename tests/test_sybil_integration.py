#!/usr/bin/env python3
"""
Sybil Attack 集成测试脚本
用于验证 Sybil Attack 与现有框架的集成
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_rq1_sybil_attack_handling():
    """Test 1: 验证 RQ1 中的 Sybil Attack 处理"""
    print("=" * 60)
    print("Test 1: 验证 RQ1 中的 Sybil Attack 处理")
    print("=" * 60)
    try:
        from experiments.scripts.runners.run_rq1_security import RQ1_CONFIG
        from experiments.attacks.sybil_attacks import SybilAttack
        
        # Check if sybil_attack is properly configured
        sybil_config = RQ1_CONFIG['attacks']['sybil_attack']
        
        # Create Sybil Attack instance with RQ1 config
        sybil = SybilAttack(
            num_identities=sybil_config['num_identities'],
            shared_data_ratio=sybil_config['shared_data_ratio']
        )
        
        identities = sybil.create_identities()
        print(f"✅ RQ1 Sybil Attack 处理正确")
        print(f"   配置: {sybil_config}")
        print(f"   创建的虚假身份: {identities}")
        
        return True
    except Exception as e:
        print(f"❌ 验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_rq4_sybil_attack_handling():
    """Test 2: 验证 RQ4 中的 Sybil Attack 处理"""
    print("\n" + "=" * 60)
    print("Test 2: 验证 RQ4 中的 Sybil Attack 处理")
    print("=" * 60)
    try:
        from experiments.scripts.runners.run_rq4_incentive import RQ4_CONFIG
        from experiments.attacks.sybil_attacks import SybilAttack
        
        # Check if sybil_attack is in scenarios
        assert 'sybil_attack' in RQ4_CONFIG['scenarios'], "sybil_attack not in scenarios"
        
        # Check sybil_config
        sybil_config = RQ4_CONFIG['sybil_config']
        
        # Create Sybil Attack instance with RQ4 config
        sybil = SybilAttack(
            num_identities=sybil_config['identities_per_attacker'],
            shared_data_ratio=1.0
        )
        
        identities = sybil.create_identities()
        print(f"✅ RQ4 Sybil Attack 处理正确")
        print(f"   配置: {sybil_config}")
        print(f"   创建的虚假身份: {identities}")
        
        return True
    except Exception as e:
        print(f"❌ 验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_metrics_consistency():
    """Test 3: 验证指标函数的一致性"""
    print("\n" + "=" * 60)
    print("Test 3: 验证指标函数的一致性")
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
        
        # Test edge cases
        # Empty inputs
        assert compute_sybil_detection_rate([], []) == 0.0
        assert compute_sybil_attack_success_rate({}) == 0.0
        assert compute_reward_dilution(0, 0) == 0.0
        assert compute_reputation_penalty({}) == 0.0
        
        # Normal cases
        assert 0 <= compute_sybil_detection_rate(["a"], ["a", "b"]) <= 1
        assert 0 <= compute_sybil_attack_success_rate({"a": True}) <= 1
        assert compute_sybil_attack_cost(100, 10, 5) == 550
        assert 0 <= compute_reward_dilution(1000, 5) <= 1
        
        print("✅ 所有指标函数一致性验证通过")
        return True
    except Exception as e:
        print(f"❌ 验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_sybil_attack_methods():
    """Test 4: 验证 Sybil Attack 类的所有方法"""
    print("\n" + "=" * 60)
    print("Test 4: 验证 Sybil Attack 类的所有方法")
    print("=" * 60)
    try:
        from experiments.attacks.sybil_attacks import SybilAttack
        from collections import OrderedDict
        import torch
        
        sybil = SybilAttack(num_identities=3, shared_data_ratio=1.0)
        
        # Test create_identities
        identities = sybil.create_identities()
        assert len(identities) == 3, f"Expected 3 identities, got {len(identities)}"
        print(f"✅ create_identities: {identities}")
        
        # Test get_shared_model_update
        model_state = OrderedDict([
            ('layer1.weight', torch.randn(10, 5)),
            ('layer1.bias', torch.randn(10))
        ])
        shared_update = sybil.get_shared_model_update(model_state)
        assert isinstance(shared_update, OrderedDict), "Should return OrderedDict"
        print(f"✅ get_shared_model_update: 返回 OrderedDict")
        
        # Test get_identity_correlation
        correlation = sybil.get_identity_correlation(identities[0], identities[1])
        assert 0 <= correlation <= 1, f"Correlation should be between 0 and 1, got {correlation}"
        print(f"✅ get_identity_correlation: {correlation:.2f}")
        
        # Test get_data_commitment_correlation
        data_corr = sybil.get_data_commitment_correlation(identities[0], identities[1])
        assert 0 <= data_corr <= 1, f"Data correlation should be between 0 and 1, got {data_corr}"
        print(f"✅ get_data_commitment_correlation: {data_corr:.2f}")
        
        # Test get_model_update_similarity
        update1 = OrderedDict([('w', torch.randn(5, 5))])
        update2 = OrderedDict([('w', torch.randn(5, 5))])
        similarity = sybil.get_model_update_similarity(update1, update2)
        assert 0 <= similarity <= 1, f"Similarity should be between 0 and 1, got {similarity}"
        print(f"✅ get_model_update_similarity: {similarity:.2f}")
        
        return True
    except Exception as e:
        print(f"❌ 验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_no_gpu_usage():
    """Test 5: 验证代码可以在 CPU 上运行"""
    print("\n" + "=" * 60)
    print("Test 5: 验证代码可以在 CPU 上运行")
    print("=" * 60)
    try:
        import torch
        
        # Force CPU usage
        device = torch.device('cpu')
        
        from experiments.attacks.sybil_attacks import SybilAttack
        from collections import OrderedDict
        
        sybil = SybilAttack(num_identities=2)
        identities = sybil.create_identities()
        
        # Create model state on CPU
        model_state = OrderedDict([
            ('layer1.weight', torch.randn(10, 5, device=device)),
            ('layer1.bias', torch.randn(10, device=device))
        ])
        
        shared_update = sybil.get_shared_model_update(model_state)
        
        print(f"✅ 代码可以在 CPU 上正常运行")
        print(f"   设备: {device}")
        print(f"   虚假身份: {identities}")
        
        return True
    except Exception as e:
        print(f"❌ 验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """运行所有集成测试"""
    print("\n" + "=" * 60)
    print("Sybil Attack 集成测试")
    print("=" * 60)
    
    results = []
    results.append(("RQ1 Sybil Attack 处理", test_rq1_sybil_attack_handling()))
    results.append(("RQ4 Sybil Attack 处理", test_rq4_sybil_attack_handling()))
    results.append(("指标函数一致性", test_metrics_consistency()))
    results.append(("Sybil Attack 方法", test_sybil_attack_methods()))
    results.append(("CPU 运行验证", test_no_gpu_usage()))
    
    # Summary
    print("\n" + "=" * 60)
    print("集成测试总结")
    print("=" * 60)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{status}: {test_name}")
    
    print(f"\n总体结果: {passed}/{total} 通过")
    
    if passed == total:
        print("\n🎉 所有集成测试通过！")
        return 0
    else:
        print(f"\n⚠️  有 {total - passed} 个测试失败")
        return 1

if __name__ == "__main__":
    sys.exit(main())

