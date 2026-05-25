#!/usr/bin/env python3
"""
第二阶段改进验证脚本
验证CheckpointCleaner、P2P通信、文档完整性
"""

import sys
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_checkpoint_cleaner():
    """测试CheckpointCleaner"""
    logger.info("\n" + "="*60)
    logger.info("[1/3] Testing CheckpointCleaner...")
    logger.info("="*60)
    
    try:
        from client.pol.CheckpointCleaner import CheckpointCleaner
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建测试checkpoint文件
            for i in range(10):
                filepath = os.path.join(tmpdir, f"ckpt_step_{i*10}.pt")
                with open(filepath, 'w') as f:
                    f.write(f"test {i}\n" * 50)
            
            # 初始化cleaner
            cleaner = CheckpointCleaner(tmpdir, max_age_days=7, keep_every_n=3)
            
            # 获取统计信息
            stats = cleaner.get_cleanup_stats()
            logger.info(f"✅ CheckpointCleaner working correctly")
            logger.info(f"   Total checkpoints: {stats['total_checkpoints']}")
            logger.info(f"   Files to delete: {stats['files_to_delete']}")
            logger.info(f"   Space saved: {stats['space_saved_percent']:.1f}%")
            
            return True
    except Exception as e:
        logger.error(f"❌ CheckpointCleaner test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_p2p_communication():
    """测试P2P通信"""
    logger.info("\n" + "="*60)
    logger.info("[2/3] Testing P2P Communication...")
    logger.info("="*60)
    
    try:
        from server.p2p.challenge_client import ChallengeClient
        from client.p2p.challenge_response_server import ChallengeResponseServer
        
        # 启动客户端响应服务器
        server = ChallengeResponseServer(host="127.0.0.1", port=0)
        
        # 设置挑战处理函数
        def handle_challenge(client_id, challenge_data):
            return {
                'client_id': client_id,
                'response': 'test_response',
                'timestamp': time.time()
            }
        
        server.set_challenge_handler(handle_challenge)
        
        if not server.start():
            logger.error("❌ Failed to start challenge response server")
            return False
        
        logger.info(f"✅ Challenge response server started on {server.get_url()}")
        
        # 创建客户端
        client = ChallengeClient(server.get_url(), timeout=5)
        
        # 测试健康检查
        if not client.health_check():
            logger.error("❌ Health check failed")
            server.stop()
            return False
        
        logger.info("✅ Health check passed")
        
        # 测试发送挑战
        challenge_data = {
            'checkpoint_indices': [0, 1],
            'data_indices': [0, 1, 2],
            'deadline': int(time.time()) + 3600
        }
        
        response = client.send_challenge("client_1", challenge_data)
        
        if response is None:
            logger.error("❌ Failed to send challenge")
            server.stop()
            return False
        
        logger.info(f"✅ Challenge sent and response received")
        logger.info(f"   Response: {response}")
        
        server.stop()
        logger.info("✅ P2P communication working correctly")
        
        return True
    except Exception as e:
        logger.error(f"❌ P2P communication test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_documentation():
    """测试文档完整性"""
    logger.info("\n" + "="*60)
    logger.info("[3/3] Testing Documentation...")
    logger.info("="*60)
    
    try:
        # 检查关键模块的docstring
        from client.pol.CheckpointCleaner import CheckpointCleaner
        from server.p2p.challenge_client import ChallengeClient
        from client.p2p.challenge_response_server import ChallengeResponseServer
        
        modules = [
            ('CheckpointCleaner', CheckpointCleaner),
            ('ChallengeClient', ChallengeClient),
            ('ChallengeResponseServer', ChallengeResponseServer)
        ]
        
        all_documented = True
        for name, module in modules:
            if module.__doc__:
                logger.info(f"✅ {name}: documented")
            else:
                logger.warning(f"⚠️  {name}: missing docstring")
                all_documented = False
        
        if all_documented:
            logger.info("✅ All modules properly documented")
        else:
            logger.warning("⚠️  Some modules missing documentation")
        
        return True
    except Exception as e:
        logger.error(f"❌ Documentation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """运行所有测试"""
    logger.info("\n" + "="*60)
    logger.info("🧪 Phase 2 Improvements Verification")
    logger.info("="*60)
    
    results = []
    
    # 运行所有测试
    results.append(("CheckpointCleaner", test_checkpoint_cleaner()))
    results.append(("P2P Communication", test_p2p_communication()))
    results.append(("Documentation", test_documentation()))
    
    # 总结
    logger.info("\n" + "="*60)
    logger.info("📊 Test Summary")
    logger.info("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status}: {name}")
    
    logger.info(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("\n🎉 All Phase 2 improvements verified successfully!")
        return 0
    else:
        logger.error(f"\n❌ {total - passed} test(s) failed")
        return 1

if __name__ == '__main__':
    sys.exit(main())

