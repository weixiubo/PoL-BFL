#!/usr/bin/env python3
"""
测试CheckpointCleaner功能
"""

import os
import sys
import tempfile
import shutil
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_checkpoint_cleaner():
    """测试CheckpointCleaner"""
    
    from client.pol.CheckpointCleaner import CheckpointCleaner
    
    logger.info("=" * 60)
    logger.info("Testing CheckpointCleaner")
    logger.info("=" * 60)
    
    # 创建临时目录
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = os.path.join(tmpdir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 创建模拟的checkpoint文件
        logger.info("\n[1] Creating mock checkpoint files...")
        for step in range(0, 100, 10):
            filepath = os.path.join(checkpoint_dir, f"ckpt_step_{step}.pt")
            with open(filepath, 'w') as f:
                f.write(f"Mock checkpoint at step {step}\n" * 100)
            logger.info(f"  Created: {filepath}")
        
        # 初始化CheckpointCleaner
        logger.info("\n[2] Initializing CheckpointCleaner...")
        cleaner = CheckpointCleaner(
            checkpoint_dir=checkpoint_dir,
            max_age_days=7,
            keep_every_n=3,
            min_keep_count=3
        )
        
        # 获取清理统计信息
        logger.info("\n[3] Getting cleanup statistics...")
        stats = cleaner.get_cleanup_stats()
        logger.info(f"  Total checkpoints: {stats['total_checkpoints']}")
        logger.info(f"  Total size: {stats['total_size_mb']:.2f} MB")
        logger.info(f"  Files to delete: {stats['files_to_delete']}")
        logger.info(f"  Space to save: {stats['delete_size_mb']:.2f} MB ({stats['space_saved_percent']:.1f}%)")
        
        # 执行dry-run清理
        logger.info("\n[4] Running dry-run cleanup...")
        cleanup_stats = cleaner.cleanup(dry_run=True)
        logger.info(f"  Total checkpoints: {cleanup_stats['total_checkpoints']}")
        logger.info(f"  Files to delete: {cleanup_stats['files_to_delete']}")
        logger.info(f"  Dry run: {cleanup_stats['dry_run']}")
        
        # 执行实际清理
        logger.info("\n[5] Running actual cleanup...")
        cleanup_stats = cleaner.cleanup(dry_run=False)
        logger.info(f"  Deleted: {len(cleanup_stats['deleted_files'])} files")
        if cleanup_stats['failed_deletions']:
            logger.warning(f"  Failed: {len(cleanup_stats['failed_deletions'])} files")
        
        # 验证清理结果
        logger.info("\n[6] Verifying cleanup results...")
        remaining_files = cleaner.get_checkpoint_files()
        logger.info(f"  Remaining checkpoints: {len(remaining_files)}")
        for step, filepath in remaining_files:
            logger.info(f"    - Step {step}: {os.path.basename(filepath)}")
        
        # 测试cleanup_by_count
        logger.info("\n[7] Testing cleanup_by_count...")
        # 重新创建一些文件
        for step in range(100, 150, 10):
            filepath = os.path.join(checkpoint_dir, f"ckpt_step_{step}.pt")
            with open(filepath, 'w') as f:
                f.write(f"Mock checkpoint at step {step}\n" * 100)
        
        logger.info(f"  Created additional checkpoints")
        current_count = len(cleaner.get_checkpoint_files())
        logger.info(f"  Current count: {current_count}")
        
        # 保留最近5个
        logger.info(f"  Keeping only 5 most recent...")
        stats = cleaner.cleanup_by_count(keep_count=5, dry_run=False)
        logger.info(f"  Deleted: {len(stats['deleted_files'])} files")
        
        final_count = len(cleaner.get_checkpoint_files())
        logger.info(f"  Final count: {final_count}")
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ CheckpointCleaner test completed successfully!")
        logger.info("=" * 60)
        
        return True

if __name__ == '__main__':
    try:
        success = test_checkpoint_cleaner()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

