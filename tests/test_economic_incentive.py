#!/usr/bin/env python3
"""
经济激励系统测试脚本
验证质押、奖励、声誉等机制
"""

import sys
import logging
from server.incentive.EconomicIncentiveSystem import EconomicIncentiveSystem

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_registration_and_eligibility():
    """测试注册和资格验证"""
    logger.info("\n" + "="*60)
    logger.info("[1/4] Testing Registration and Eligibility...")
    logger.info("="*60)

    try:
        # 初始化系统
        args = {
            'min_stake': 100.0,
            'min_stake_for_participation': 100.0,
            'reputation_threshold': 0.3,
            'base_reward': 100.0,
            'reward_pool': 10000.0
        }
        system = EconomicIncentiveSystem(args)

        # 注册客户端
        success, msg = system.register_client('client_1', 150.0)
        if not success:
            logger.error(f"[FAIL] Registration failed: {msg}")
            return False

        logger.info("[PASS] Client registered successfully")

        # 验证资格
        eligible, reason = system.verify_client_eligibility('client_1')
        if not eligible:
            logger.error(f"[FAIL] Eligibility check failed: {reason}")
            return False

        logger.info("[PASS] Client is eligible")

        # 获取客户端状态
        status = system.get_client_status('client_1')
        logger.info(f"[PASS] Client status: stake={status['stake']}, reputation={status['reputation']:.4f}")

        return True

    except Exception as e:
        logger.error(f"[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_verification_and_rewards():
    """测试验证和奖励"""
    logger.info("\n" + "="*60)
    logger.info("[2/4] Testing Verification and Rewards...")
    logger.info("="*60)

    try:
        # 初始化系统
        args = {
            'min_stake': 100.0,
            'min_stake_for_participation': 100.0,
            'reputation_threshold': 0.3,
            'base_reward': 100.0,
            'reward_pool': 10000.0
        }
        system = EconomicIncentiveSystem(args)

        # 注册客户端
        system.register_client('client_1', 150.0)
        system.register_client('client_2', 150.0)

        # 处理成功验证
        result1 = system.process_verification_result(
            'client_1',
            is_verified=True,
            training_steps=100,
            total_steps=100
        )

        logger.info(f"[PASS] Successful verification: reward={result1['reward']:.2f}")

        # 处理失败验证
        result2 = system.process_verification_result(
            'client_2',
            is_verified=False,
            training_steps=0,
            total_steps=100
        )

        logger.info(f"[PASS] Failed verification: slash={result2['slash']:.2f}")

        # 检查客户端状态
        status1 = system.get_client_status('client_1')
        status2 = system.get_client_status('client_2')

        logger.info(f"[PASS] Client 1 reputation: {status1['reputation']:.4f}")
        logger.info(f"[PASS] Client 2 reputation: {status2['reputation']:.4f}")

        return True

    except Exception as e:
        logger.error(f"[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multiple_rounds():
    """测试多轮运行"""
    logger.info("\n" + "="*60)
    logger.info("[3/4] Testing Multiple Rounds...")
    logger.info("="*60)

    try:
        # 初始化系统
        args = {
            'min_stake': 100.0,
            'min_stake_for_participation': 100.0,
            'reputation_threshold': 0.3,
            'base_reward': 100.0,
            'reward_pool': 10000.0,
            'decay_rate': 0.01
        }
        system = EconomicIncentiveSystem(args)

        # 注册客户端
        for i in range(3):
            system.register_client(f'client_{i}', 150.0)

        # 运行多轮
        for round_num in range(3):
            logger.info(f"\n--- Round {round_num + 1} ---")

            # 处理验证
            for i in range(3):
                is_verified = (i + round_num) % 2 == 0  # 交替成功和失败
                system.process_verification_result(
                    f'client_{i}',
                    is_verified=is_verified,
                    training_steps=100 if is_verified else 0,
                    total_steps=100
                )

            # 结束轮次
            round_stats = system.end_round()
            logger.info(f"[PASS] Round {round_num + 1} completed")

        # 获取系统统计
        stats = system.get_system_statistics()
        logger.info(f"[PASS] System statistics:")
        logger.info(f"   Total rounds: {stats['total_rounds']}")
        logger.info(f"   Total rewards: {stats['total_rewards_distributed']:.2f}")
        logger.info(f"   Total slashed: {stats['total_stakes_slashed']:.2f}")

        return True

    except Exception as e:
        logger.error(f"[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_system_statistics():
    """测试系统统计"""
    logger.info("\n" + "="*60)
    logger.info("[4/4] Testing System Statistics...")
    logger.info("="*60)

    try:
        # 初始化系统
        args = {
            'min_stake': 100.0,
            'min_stake_for_participation': 100.0,
            'reputation_threshold': 0.3,
            'base_reward': 100.0,
            'reward_pool': 10000.0
        }
        system = EconomicIncentiveSystem(args)

        # 注册客户端
        for i in range(5):
            system.register_client(f'client_{i}', 150.0)

        # 获取统计
        stats = system.get_system_statistics()

        logger.info("[PASS] System statistics retrieved:")
        logger.info(f"   Total rounds: {stats['total_rounds']}")
        logger.info(f"   Total clients: {stats['total_clients']}")
        logger.info(f"   Total staked: {stats['total_staked']:.2f}")
        logger.info(f"   Total rewards distributed: {stats['total_rewards_distributed']:.2f}")
        logger.info(f"   Total stakes slashed: {stats['total_stakes_slashed']:.2f}")

        return True

    except Exception as e:
        logger.error(f"[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """运行所有测试"""
    logger.info("\n" + "="*60)
    logger.info("[TEST] Economic Incentive System Tests")
    logger.info("="*60)

    results = []

    # 运行所有测试
    results.append(("Registration and Eligibility", test_registration_and_eligibility()))
    results.append(("Verification and Rewards", test_verification_and_rewards()))
    results.append(("Multiple Rounds", test_multiple_rounds()))
    results.append(("System Statistics", test_system_statistics()))

    # 总结
    logger.info("\n" + "="*60)
    logger.info("[RESULT] Test Summary")
    logger.info("="*60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "[PASS] PASS" if result else "[FAIL] FAIL"
        logger.info(f"{status}: {name}")

    logger.info(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        logger.info("\n[PASS] All economic incentive system tests passed.")
        return 0
    else:
        logger.error(f"\n[FAIL] {total - passed} test(s) failed")
        return 1

if __name__ == '__main__':
    sys.exit(main())

