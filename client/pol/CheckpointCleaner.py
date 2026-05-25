"""
Checkpoint自动清理器
负责定期清理过期的checkpoint，减少存储开销
"""

import os
import glob
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class CheckpointCleaner:
    """
    自动清理过期checkpoint的工具类
    
    支持多种清理策略：
    1. 基于时间的清理：删除超过指定天数的checkpoint
    2. 基于数量的清理：保留最近N个checkpoint
    3. 基于间隔的清理：保留每N个checkpoint中的一个
    4. 混合策略：结合多种清理方式
    """
    
    def __init__(self, checkpoint_dir: str, 
                 max_age_days: int = 7,
                 keep_every_n: int = 10,
                 min_keep_count: int = 5):
        """
        初始化CheckpointCleaner
        
        Args:
            checkpoint_dir: checkpoint保存目录
            max_age_days: 最大保留天数（超过此天数的checkpoint将被删除）
            keep_every_n: 保留间隔（保留每N个checkpoint中的一个）
            min_keep_count: 最少保留数量（即使满足删除条件也至少保留此数量）
        """
        self.checkpoint_dir = checkpoint_dir
        self.max_age_days = max_age_days
        self.keep_every_n = keep_every_n
        self.min_keep_count = min_keep_count
        
        logger.info(f"CheckpointCleaner initialized")
        logger.info(f"  Directory: {checkpoint_dir}")
        logger.info(f"  Max age: {max_age_days} days")
        logger.info(f"  Keep every N: {keep_every_n}")
        logger.info(f"  Min keep count: {min_keep_count}")
    
    def get_checkpoint_files(self) -> List[Tuple[int, str]]:
        """
        获取所有checkpoint文件
        
        Returns:
            List of (step, filepath) tuples, sorted by step
        """
        if not os.path.exists(self.checkpoint_dir):
            return []
        
        checkpoints = []
        # 查找所有checkpoint文件（支持压缩和未压缩）
        for pattern in ["ckpt_step_*.pt", "ckpt_step_*.pt.gz"]:
            for filepath in glob.glob(os.path.join(self.checkpoint_dir, pattern)):
                try:
                    # 从文件名提取step
                    filename = os.path.basename(filepath)
                    step_str = filename.split('_')[2].split('.')[0]
                    step = int(step_str)
                    checkpoints.append((step, filepath))
                except (ValueError, IndexError):
                    logger.warning(f"Failed to parse checkpoint file: {filepath}")
        
        # 按step排序
        checkpoints.sort(key=lambda x: x[0])
        return checkpoints
    
    def get_file_age_days(self, filepath: str) -> float:
        """
        获取文件的年龄（天数）
        
        Args:
            filepath: 文件路径
            
        Returns:
            文件年龄（天数）
        """
        if not os.path.exists(filepath):
            return float('inf')
        
        mtime = os.path.getmtime(filepath)
        file_time = datetime.fromtimestamp(mtime)
        age = datetime.now() - file_time
        return age.total_seconds() / (24 * 3600)
    
    def get_files_to_delete(self) -> List[str]:
        """
        确定应该删除的checkpoint文件
        
        使用混合策略：
        1. 删除超过max_age_days的文件
        2. 保留最近的min_keep_count个文件
        3. 在保留的文件中，只保留每keep_every_n个中的一个
        
        Returns:
            应该删除的文件路径列表
        """
        checkpoints = self.get_checkpoint_files()
        
        if len(checkpoints) <= self.min_keep_count:
            # 文件数量少于最少保留数量，不删除
            return []
        
        to_delete = []
        
        # 策略1：删除超过max_age_days的文件
        for step, filepath in checkpoints:
            age = self.get_file_age_days(filepath)
            if age > self.max_age_days:
                to_delete.append(filepath)
        
        # 策略2：在剩余文件中应用keep_every_n策略
        remaining = [f for f in checkpoints if f[1] not in to_delete]
        
        if len(remaining) > self.min_keep_count:
            # 保留最后min_keep_count个文件
            keep_recent = set(f[1] for f in remaining[-self.min_keep_count:])
            
            # 在其他文件中应用keep_every_n策略
            for idx, (step, filepath) in enumerate(remaining[:-self.min_keep_count]):
                if idx % self.keep_every_n != 0:
                    if filepath not in keep_recent:
                        to_delete.append(filepath)
        
        return list(set(to_delete))  # 去重
    
    def cleanup(self, dry_run: bool = True) -> Dict[str, any]:
        """
        执行清理操作
        
        Args:
            dry_run: 如果为True，只报告将删除的文件，不实际删除
            
        Returns:
            清理统计信息
        """
        to_delete = self.get_files_to_delete()
        
        stats = {
            'total_checkpoints': len(self.get_checkpoint_files()),
            'files_to_delete': len(to_delete),
            'deleted_files': [],
            'failed_deletions': [],
            'dry_run': dry_run
        }
        
        if not to_delete:
            logger.info("No checkpoints to clean up")
            return stats
        
        logger.info(f"Found {len(to_delete)} checkpoint(s) to delete")
        
        for filepath in to_delete:
            try:
                if dry_run:
                    logger.info(f"[DRY RUN] Would delete: {filepath}")
                    stats['deleted_files'].append(filepath)
                else:
                    os.remove(filepath)
                    logger.info(f"Deleted: {filepath}")
                    stats['deleted_files'].append(filepath)
            except Exception as e:
                logger.error(f"Failed to delete {filepath}: {e}")
                stats['failed_deletions'].append((filepath, str(e)))
        
        return stats
    
    def cleanup_by_count(self, keep_count: int, dry_run: bool = True) -> Dict[str, any]:
        """
        保留最近N个checkpoint，删除其他的
        
        Args:
            keep_count: 保留的checkpoint数量
            dry_run: 如果为True，只报告将删除的文件，不实际删除
            
        Returns:
            清理统计信息
        """
        checkpoints = self.get_checkpoint_files()
        
        stats = {
            'total_checkpoints': len(checkpoints),
            'files_to_delete': 0,
            'deleted_files': [],
            'failed_deletions': [],
            'dry_run': dry_run
        }
        
        if len(checkpoints) <= keep_count:
            logger.info(f"Only {len(checkpoints)} checkpoints, no cleanup needed")
            return stats
        
        # 保留最后keep_count个
        to_keep = set(f[1] for f in checkpoints[-keep_count:])
        to_delete = [f[1] for f in checkpoints if f[1] not in to_keep]
        
        stats['files_to_delete'] = len(to_delete)
        
        logger.info(f"Keeping {keep_count} recent checkpoints, deleting {len(to_delete)}")
        
        for filepath in to_delete:
            try:
                if dry_run:
                    logger.info(f"[DRY RUN] Would delete: {filepath}")
                    stats['deleted_files'].append(filepath)
                else:
                    os.remove(filepath)
                    logger.info(f"Deleted: {filepath}")
                    stats['deleted_files'].append(filepath)
            except Exception as e:
                logger.error(f"Failed to delete {filepath}: {e}")
                stats['failed_deletions'].append((filepath, str(e)))
        
        return stats
    
    def get_cleanup_stats(self) -> Dict[str, any]:
        """
        获取清理统计信息（不实际删除）
        
        Returns:
            清理统计信息
        """
        checkpoints = self.get_checkpoint_files()
        to_delete = self.get_files_to_delete()
        
        total_size = sum(os.path.getsize(f[1]) for f in checkpoints if os.path.exists(f[1]))
        delete_size = sum(os.path.getsize(f) for f in to_delete if os.path.exists(f))
        
        return {
            'total_checkpoints': len(checkpoints),
            'total_size_mb': total_size / (1024 * 1024),
            'files_to_delete': len(to_delete),
            'delete_size_mb': delete_size / (1024 * 1024),
            'space_saved_percent': (delete_size / total_size * 100) if total_size > 0 else 0
        }

