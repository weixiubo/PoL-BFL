"""
PoL数据管理器
负责checkpoint的保存、加载、哈希计算和Merkle树构建
"""

import os
import json
import hashlib
import pickle
import gzip
import copy
import numpy as np
import torch
from typing import Dict, List, Any, Optional
import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime

from client.pol.MerkleTree import MerkleTree
from client.pol.CheckpointCleaner import CheckpointCleaner

logger = logging.getLogger(__name__)


class PoLManager:
    """
    管理PoL数据的存储、检索和哈希计算

    核心功能:
    1. Checkpoint保存和加载
    2. 数据索引记录
    3. Merkle树构建
    4. 哈希计算
    """

    def __init__(self, client_id: str, save_dir: str, save_freq: int,
                 compress: bool = True, async_save: bool = False,
                 save_to_disk: bool = True, memory_limit: int = 5,
                 enable_auto_cleanup: bool = False, auto_cleanup_interval: int = 50):
        """
        初始化PoL管理器

        Args:
            client_id: 客户端ID
            save_dir: 保存目录
            save_freq: checkpoint保存频率（每N个batch）
            compress: 是否压缩checkpoint
            async_save: 是否在单独I/O线程中持久化checkpoint
            save_to_disk: 是否保存checkpoint到磁盘
            memory_limit: 内存模式下保留的checkpoint数量
            enable_auto_cleanup: 是否启用自动清理
            auto_cleanup_interval: 自动清理间隔
        """
        self.client_id = client_id
        self.save_dir = os.path.join(save_dir, f"client_{client_id}")
        self.save_freq = save_freq
        self.compress = compress
        self.async_save = async_save
        self.save_to_disk = save_to_disk
        self._save_lock = threading.Lock()
        self._pending_saves: Dict[int, Future] = {}
        self._save_executor = (
            ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix=f"pol-checkpoint-{client_id}",
            )
            if self.async_save and self.save_to_disk
            else None
        )
        try:
            self.memory_limit = max(1, int(memory_limit))
        except Exception:
            self.memory_limit = 5
        self.enable_auto_cleanup = enable_auto_cleanup
        self.auto_cleanup_interval = auto_cleanup_interval

        # Cleanup deferral: by default, defer cleanup until safe (after challenge or next round)
        self.defer_cleanup: bool = True
        self._deferred_cleanup_pending: bool = False
        # Verification/cleanup coordination
        self._verification_pending: bool = False
        self._last_commitment: Optional[str] = None
        self._inflight_active: bool = False
        self._inflight_steps: set = set()


        # Metadata and data-index files are still persisted in memory mode.
        os.makedirs(self.save_dir, exist_ok=True)
        if self.save_to_disk:
            os.makedirs(os.path.join(self.save_dir, "checkpoints"), exist_ok=True)

        self.checkpoint_count = 0
        self.checkpoint_hashes = []  # 用于生成Merkle tree
        # Merkle tree (built on demand)
        self.merkle_tree: Optional[MerkleTree] = None

        # 内存模式下的checkpoint存储
        self.memory_checkpoints = {}  # {step: checkpoint_data}

        self.data_indices = []  # 记录数据索引序列

        # 元数据
        self.metadata = {
            'client_id': client_id,
            'save_freq': save_freq,
            'created_at': datetime.now().isoformat(),
            'checkpoints': []
        }

        logger.info(f"PoLManager initialized for client {client_id}")
        logger.info(f"  Save directory: {self.save_dir}")
        logger.info(f"  Save frequency: {save_freq}")
        logger.info(f"  Compression: {compress}")
        logger.info(f"  Save to disk: {save_to_disk}")
        if not save_to_disk:
            logger.info(f"  Memory checkpoint limit: {self.memory_limit}")
        if enable_auto_cleanup:
            logger.info(f"  Auto cleanup interval: {auto_cleanup_interval}")

    def record_data_indices(self, indices: List[int]):
        """
        记录训练使用的数据索引

        Args:
            indices: 数据索引列表
        """
        self.data_indices.extend(indices)

    def _snapshot_for_storage(self, value: Any) -> Any:
        """Clone tensors to CPU so in-memory checkpoints are immutable snapshots."""
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {k: self._snapshot_for_storage(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._snapshot_for_storage(v) for v in value]
        if isinstance(value, tuple):
            return tuple(self._snapshot_for_storage(v) for v in value)
        try:
            return copy.deepcopy(value)
        except Exception:
            return value

    def _write_checkpoint_file(self, checkpoint_snapshot: Dict, path: str) -> str:
        """Atomically persist one checkpoint and return its final path."""
        temporary = f"{path}.tmp-{threading.get_ident()}"
        try:
            if self.compress:
                with gzip.open(temporary, 'wb') as stream:
                    torch.save(checkpoint_snapshot, stream)
            else:
                torch.save(checkpoint_snapshot, temporary)
            os.replace(temporary, path)
            return path
        finally:
            if os.path.exists(temporary):
                os.remove(temporary)

    def _wait_for_checkpoint(self, step: int) -> None:
        with self._save_lock:
            future = self._pending_saves.get(int(step))
        if future is not None:
            future.result()
            with self._save_lock:
                self._pending_saves.pop(int(step), None)

    def flush_pending_saves(self) -> None:
        """Wait until every scheduled checkpoint is durable on disk."""
        with self._save_lock:
            pending = list(self._pending_saves.items())
        for step, future in pending:
            future.result()
            with self._save_lock:
                self._pending_saves.pop(step, None)

    def close(self) -> None:
        """Flush checkpoint I/O and release the optional worker thread."""
        self.flush_pending_saves()
        if self._save_executor is not None:
            self._save_executor.shutdown(wait=True, cancel_futures=False)
            self._save_executor = None

    def save_checkpoint(self, step: int, checkpoint_data: Dict) -> str:
        """
        保存checkpoint（磁盘或内存模式）

        Args:
            step: 当前步数
            checkpoint_data: checkpoint数据，包含:
                - model_state: 模型参数
                - optimizer_state: 优化器状态
                - epoch: 当前epoch
                - step: 当前步数
                - loss: 当前loss（可选）

        Returns:
            checkpoint_hash: checkpoint的哈希值
        """
        checkpoint_snapshot = self._snapshot_for_storage(checkpoint_data)

        # 计算checkpoint哈希
        ckpt_hash = self._compute_checkpoint_hash(checkpoint_snapshot)
        self.checkpoint_hashes.append(ckpt_hash)
        self.checkpoint_count += 1

        if self.save_to_disk:
            # 磁盘模式：保存到文件
            checkpoint_path = os.path.join(
                self.save_dir, "checkpoints", f"ckpt_step_{step}.pt"
            )

            if self.compress:
                checkpoint_path += ".gz"

            if self._save_executor is None:
                if self.compress:
                    with gzip.open(checkpoint_path, 'wb') as stream:
                        torch.save(checkpoint_snapshot, stream)
                else:
                    torch.save(checkpoint_snapshot, checkpoint_path)
            else:
                self._wait_for_checkpoint(step)
                future = self._save_executor.submit(
                    self._write_checkpoint_file,
                    checkpoint_snapshot,
                    checkpoint_path,
                )
                with self._save_lock:
                    self._pending_saves[int(step)] = future

            # 更新元数据
            self.metadata['checkpoints'].append({
                'step': step,
                'hash': ckpt_hash,
                'path': checkpoint_path,
                'timestamp': datetime.now().isoformat()
            })

            logger.debug(f"Saved checkpoint to disk at step {step}, hash: {ckpt_hash[:16]}...")
        else:
            # 内存模式：保存到内存
            self.memory_checkpoints[step] = checkpoint_snapshot

            # 限制内存中的checkpoint数量
            if len(self.memory_checkpoints) > self.memory_limit:
                # 删除最旧的checkpoint
                oldest_step = min(self.memory_checkpoints.keys())
                del self.memory_checkpoints[oldest_step]
                logger.debug(f"Removed earliest checkpoint from memory: step {oldest_step}")

            # 更新元数据（不包含path）
            self.metadata['checkpoints'].append({
                'step': step,
                'hash': ckpt_hash,
                'path': 'memory',
                'timestamp': datetime.now().isoformat()
            })

            logger.debug(f"Saved checkpoint to memory at step {step}, hash: {ckpt_hash[:16]}...")

        # 自动清理检查（默认延后到安全时间点）
        if self.enable_auto_cleanup and self.checkpoint_count % self.auto_cleanup_interval == 0:
            if getattr(self, 'defer_cleanup', False):
                self._deferred_cleanup_pending = True
                logger.debug(f"Deferred cleanup scheduled at checkpoint {self.checkpoint_count}")
            else:
                self._auto_cleanup()

        return ckpt_hash

    def load_checkpoint(self, step: int) -> Optional[Dict]:
        """
        加载指定step的checkpoint

        Args:
            step: 步数

        Returns:
            checkpoint_data: checkpoint数据，如果不存在返回None
        """
        if not self.save_to_disk:
            checkpoint = self.memory_checkpoints.get(step)
            if checkpoint is not None:
                return self._snapshot_for_storage(checkpoint)

        if self._save_executor is not None:
            self._wait_for_checkpoint(step)

        # 尝试压缩版本
        checkpoint_path = os.path.join(
            self.save_dir, "checkpoints", f"ckpt_step_{step}.pt.gz"
        )

        if os.path.exists(checkpoint_path):
            with gzip.open(checkpoint_path, 'rb') as f:
                return torch.load(f, map_location=torch.device("cpu"))

        # 尝试未压缩版本
        checkpoint_path = os.path.join(
            self.save_dir, "checkpoints", f"ckpt_step_{step}.pt"
        )

        if os.path.exists(checkpoint_path):
            return torch.load(checkpoint_path, map_location=torch.device("cpu"))

        logger.warning(f"Checkpoint at step {step} not found")
        return None

    def save_data_indices(self, indices: List[int]):
        """
        保存数据索引序列

        Args:
            indices: 数据索引列表
        """
        self.data_indices = indices
        indices_path = os.path.join(self.save_dir, "data_indices.npy")
        np.save(indices_path, np.array(indices))
        logger.debug(f"Saved {len(indices)} data indices")

    def load_data_indices(self) -> Optional[np.ndarray]:
        """
        加载数据索引序列

        Returns:
            indices: 数据索引数组，如果不存在返回None
        """
        indices_path = os.path.join(self.save_dir, "data_indices.npy")
        if os.path.exists(indices_path):
            return np.load(indices_path)
        return None

    def compute_data_hash(self, dataset) -> str:
        """
        计算数据集的哈希

        The compatibility path hashes recorded samples when they are available
        and otherwise hashes dataset metadata.

        Args:
            dataset: 数据集对象

        Returns:
            data_hash: 数据哈希（十六进制字符串）
        """
        m = hashlib.sha256()

        # 如果有记录的数据索引，哈希实际使用的数据
        if self.data_indices:
            fast_hash = getattr(dataset, 'fast_data_hash_for_global_indices', None)
            if callable(fast_hash):
                try:
                    data_hash = str(fast_hash([int(i) for i in self.data_indices]))
                    logger.debug(
                        "Computed fast data hash for %d recorded indices: %s",
                        len(self.data_indices),
                        data_hash[:16],
                    )
                    return data_hash
                except Exception as e:
                    logger.debug(f"Fast data hash unavailable, falling back to item hashing: {e}")

            logger.debug(f"Computing hash for {len(self.data_indices)} data samples...")
            subset_local_from_global = None
            try:
                if hasattr(dataset, 'indices'):
                    idx_list = list(getattr(dataset, 'indices'))
                    subset_local_from_global = {int(g): i for i, g in enumerate(idx_list)}
            except Exception as e:
                logger.debug(f"Build subset index mapping for data hash failed: {e}")

            # 对每个使用的数据样本进行哈希
            for idx in self.data_indices:
                try:
                    lookup_idx = int(idx)
                    if subset_local_from_global is not None:
                        lookup_idx = subset_local_from_global.get(int(idx), None)
                        if lookup_idx is None:
                            logger.warning(f"Failed to hash data at global index {idx}: not found in subset mapping")
                            m.update(str(idx).encode('utf-8'))
                            continue
                    # 获取数据和标签
                    item = dataset[lookup_idx]
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        data, label = item[0], item[1]
                    else:
                        data, label = item, 0

                    # 哈希数据张量
                    if hasattr(data, 'numpy'):
                        m.update(data.numpy().tobytes())
                    else:
                        m.update(str(data).encode('utf-8'))

                    # 哈希标签
                    m.update(str(label).encode('utf-8'))

                except Exception as e:
                    logger.warning(f"Failed to hash data at index {idx}: {e}")
                    # 如果无法访问数据，至少哈希索引
                    m.update(str(idx).encode('utf-8'))
        else:
            # 如果没有数据索引记录，使用数据集元信息
            logger.warning("No data indices recorded, using dataset metadata for hash")
            m.update(str(len(dataset)).encode('utf-8'))
            m.update(str(type(dataset).__name__).encode('utf-8'))

        data_hash = m.hexdigest()
        logger.debug(f"Computed data hash: {data_hash[:16]}...")

        return data_hash

    def generate_commitment(self) -> str:
        """
        生成PoL承诺（Merkle root）

        Returns:
            commitment: Merkle root（十六进制字符串）
        """
        if not self.checkpoint_hashes:
            logger.warning("No checkpoints to generate commitment")
            return ""

        # 构建Merkle树并缓存用于后续生成证明
        self.merkle_tree = MerkleTree(self.checkpoint_hashes)
        merkle_root = self.merkle_tree.get_root()

        logger.info(f"Generated PoL commitment: {merkle_root[:16]}...")

        return merkle_root

    def get_checkpoint_count(self) -> int:
        """获取checkpoint数量"""
        return self.checkpoint_count

    def get_metadata(self) -> Dict:
        """获取元数据"""
        return self.metadata.copy()

    def save_metadata(self):
        """保存元数据到文件（原子写入）"""
        metadata_path = os.path.join(self.save_dir, "metadata.json")
        tmp_path = metadata_path + ".tmp"
        with open(tmp_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        try:
            os.replace(tmp_path, metadata_path)
        except Exception:
            # Fallback for environments without atomic replace
            import shutil
            shutil.move(tmp_path, metadata_path)
        logger.debug("Saved metadata (atomic)")

    def mark_commitment_generated(self, commitment: str):
        """标记生成了新的承诺，等待验证完成后再清理"""
        try:
            self._last_commitment = str(commitment)
        except Exception:
            self._last_commitment = str(commitment)
        self._verification_pending = True

    def begin_inflight_verification(self, steps: list, tag: Optional[str] = None):
        """在发起挑战时由聚合器调用，锁定所需的checkpoint"""
        self._inflight_active = True
        try:
            self._inflight_steps.update(int(s) for s in (steps or []))
        except Exception:
            pass
        logger.debug(f"Begin inflight verification: steps={len(self._inflight_steps)}, tag={tag or ''}")

    def complete_inflight_verification(self, tag: Optional[str] = None):
        """在验证完成后由聚合器调用，释放锁并允许清理"""
        self._inflight_active = False
        self._inflight_steps.clear()
        # 当前承诺已完成验证，允许后续清理
        self._verification_pending = False
        logger.debug(f"Complete inflight verification: tag={tag or ''}")


    def _compute_checkpoint_hash(self, checkpoint_data: Dict) -> str:
        """
        计算单个checkpoint的哈希

        Args:
            checkpoint_data: checkpoint数据

        Returns:
            hash_value: SHA256哈希值（十六进制字符串）
        """
        # 只哈希模型参数
        model_state = checkpoint_data.get('model_state', {})
        m = hashlib.sha256()

        for key in sorted(model_state.keys()):
            param_bytes = model_state[key].cpu().numpy().tobytes()
            m.update(param_bytes)

        return m.hexdigest()

    def _build_merkle_tree(self, leaves: List[str]) -> str:
        """
        构建Merkle树并返回根哈希

        Args:
            leaves: 叶子节点哈希列表

        Returns:
            root: Merkle根哈希
        """
        if not leaves:
            return ""

        if len(leaves) == 1:
            return leaves[0]

        # 构建树的层级
        current_level = leaves[:]

        while len(current_level) > 1:
            next_level = []

            for i in range(0, len(current_level), 2):
                left = current_level[i]
                right = current_level[i+1] if i+1 < len(current_level) else left

                # 哈希一对节点
                parent = self._hash_pair(left, right)
                next_level.append(parent)

            current_level = next_level

        return current_level[0]

    def _hash_pair(self, left: str, right: str) -> str:
        """
        哈希一对节点

        Args:
            left: 左节点哈希
            right: 右节点哈希

        Returns:
            parent: 父节点哈希
        """
        combined = left + right
        return hashlib.sha256(combined.encode('utf-8')).hexdigest()

    def cleanup_old_checkpoints(self, keep_every_n: int = 10):
        """
        清理旧的checkpoint，保留每N个中的一个

        Args:
            keep_every_n: 每N个checkpoint保留一个
        """
        if keep_every_n <= 0:
            raise ValueError("keep_every_n must be positive")
        if self._save_executor is not None:
            self.flush_pending_saves()
        checkpoint_dir = os.path.join(self.save_dir, "checkpoints")
        if not os.path.exists(checkpoint_dir):
            return

        files = os.listdir(checkpoint_dir)
        checkpoint_files = sorted([f for f in files if f.startswith("ckpt_step_")])

        removed_count = 0
        for idx, filename in enumerate(checkpoint_files):
            # 保留每N个checkpoint
            if idx % keep_every_n != 0:
                filepath = os.path.join(checkpoint_dir, filename)
                os.remove(filepath)
                removed_count += 1

        logger.info(f"Removed {removed_count} expired checkpoints")

    def _auto_cleanup(self):
        """
        自动清理checkpoint
        使用CheckpointCleaner进行智能清理
        """
        # Guard: never cleanup while verification is pending or inflight
        if getattr(self, '_verification_pending', False) or getattr(self, '_inflight_active', False):
            logger.info("Skip auto cleanup: verification pending/inflight")
            return
        if self.save_to_disk:
            if self._save_executor is not None:
                self.flush_pending_saves()
            # 磁盘模式：使用CheckpointCleaner进行智能清理
            checkpoint_dir = os.path.join(self.save_dir, "checkpoints")
            cleaner = CheckpointCleaner(
                checkpoint_dir=checkpoint_dir,
                max_age_days=7,  # 保留7天内的checkpoint
                keep_every_n=self.auto_cleanup_interval // 5,  # 保留每N个中的一个
                min_keep_count=5  # 至少保留5个最近的checkpoint
            )

            # 执行清理（非dry-run）
            stats = cleaner.cleanup(dry_run=False)

            logger.info(f"Auto cleanup completed at checkpoint {self.checkpoint_count}")
            logger.info(f"  Total checkpoints: {stats['total_checkpoints']}")
            logger.info(f"  Deleted: {stats['files_to_delete']}")
            if stats['failed_deletions']:
                logger.warning(f"  Failed deletions: {len(stats['failed_deletions'])}")
            # 同步元数据与哈希到磁盘状态
            self._sync_after_cleanup()

        else:
            # 内存模式：已经在save_checkpoint中处理了
            pass

        logger.debug(f"Auto cleanup triggered at checkpoint {self.checkpoint_count}")

    def _sync_after_cleanup(self):
        """
        清理后同步 metadata 和 checkpoint_hashes, 确保与磁盘文件一致。
        防止验证阶段请求已被清理的 step。
        """
        if not self.save_to_disk:
            return

        if self._save_executor is not None:
            self.flush_pending_saves()

        checkpoint_dir = os.path.join(self.save_dir, "checkpoints")
        try:
            files = sorted(os.listdir(checkpoint_dir))
        except Exception as e:
            logger.warning(f"Cannot list checkpoint dir for sync: {e}")
            return

        existing = {}
        for name in files:
            if not name.startswith("ckpt_step_"):
                continue
            if not (name.endswith(".pt") or name.endswith(".pt.gz")):
                continue
            step_str = name[len("ckpt_step_"):].split(".")[0]
            try:
                step = int(step_str)
            except Exception:
                continue
            existing[step] = os.path.join(checkpoint_dir, name)

        # 仅保留仍然存在（或内存模式）的条目，并更新路径
        new_meta = []
        for meta in self.metadata.get("checkpoints", []):
            step = meta.get("step")
            path = meta.get("path")
            if path == "memory":
                new_meta.append(meta)
            elif step in existing and os.path.exists(existing[step]):
                meta["path"] = existing[step]
                new_meta.append(meta)
            else:
                logger.debug(f"Pruned missing checkpoint from metadata: step {step}")

        self.metadata["checkpoints"] = new_meta
        # 使 checkpoint_hashes 与元数据顺序对齐
        self.checkpoint_hashes = [m.get("hash") for m in new_meta if "hash" in m]
        self.checkpoint_count = len(new_meta)
        self.merkle_tree = None

        # 持久化同步后的元数据（忽略失败）
        try:
            self.save_metadata()
        except Exception as e:
            logger.warning(f"Failed to save synced metadata: {e}")

        # 令 MerkleTree 失效，按需重建
    def run_deferred_cleanup_if_any(self):
        """
        If deferred cleanup is pending and auto-cleanup is enabled, run it now.
        This should be called after challenge response is constructed, or
        at the beginning of the next local training round.
        """
        if not self.save_to_disk:
            return
        if self.enable_auto_cleanup and getattr(self, '_deferred_cleanup_pending', False):
            # Do not run cleanup while verification is pending or inflight
            if getattr(self, '_verification_pending', False) or getattr(self, '_inflight_active', False):
                logger.debug("Deferred cleanup kept pending due to verification in progress")
            else:
                try:
                    # Temporarily disable deferral for this run
                    old = self.defer_cleanup
                    self.defer_cleanup = False
                    self._auto_cleanup()
                finally:
                    self.defer_cleanup = old
                self._deferred_cleanup_pending = False
                logger.info("Ran deferred checkpoint cleanup safely.")

        try:
            self.save_metadata()
        except Exception as e:
            logger.warning(f"Failed to save synced metadata: {e}")


    def get_memory_checkpoint(self, step: int) -> Optional[Dict]:
        """
        从内存获取checkpoint（仅内存模式）

        Args:
            step: checkpoint步数

        Returns:
            checkpoint_data: checkpoint数据，如果不存在返回None
        """
        if not self.save_to_disk:
            return self.memory_checkpoints.get(step)
        else:
            logger.warning("get_memory_checkpoint called in disk mode")
            return None

    def get_memory_checkpoint_count(self) -> int:
        """
        获取内存中的checkpoint数量

        Returns:
            count: checkpoint数量
        """
        if not self.save_to_disk:
            return len(self.memory_checkpoints)
        else:
            return 0

    def clear_memory_checkpoints(self):
        """
        清空内存中的所有checkpoint
        """
        if not self.save_to_disk:
            self.memory_checkpoints.clear()
            logger.info("Cleared all memory checkpoints")
        else:
            logger.warning("clear_memory_checkpoints called in disk mode")

    def get_leaf_index_for_step(self, step: int) -> Optional[int]:
        """
        Get the Merkle leaf index corresponding to a checkpoint step.

        Returns None if not found.
        """
        for i, meta in enumerate(self.metadata.get('checkpoints', [])):
            if meta.get('step') == step:
                return i
        return None

    def get_merkle_proof_by_index(self, index: int):
        """
        Return Merkle proof (list of (hash, position)) for a leaf index.
        """
        if not self.checkpoint_hashes:
            return []
        if self.merkle_tree is None:
            self.merkle_tree = MerkleTree(self.checkpoint_hashes)
        return self.merkle_tree.get_proof(index)

    def get_merkle_proof_by_step(self, step: int):
        """
        Return Merkle proof for a checkpoint saved at the given step.
        """
        idx = self.get_leaf_index_for_step(step)
        if idx is None:
            return []
        return self.get_merkle_proof_by_index(idx)
