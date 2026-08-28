"""
PoL Trainer
继承BaseTrainer，集成Proof-of-Learning机制
"""

import copy
import logging
import os
import torch
from torch import nn
from typing import Dict, List, Optional
import numpy as np

from client.base.baseTrainer import BaseTrainer
from client.pol.PoLManager import PoLManager
from client.zkp.ZKPProver import ZKPProver

logger = logging.getLogger(__name__)


class PoLTrainer(BaseTrainer):
    """
    集成PoL机制的Trainer

    核心功能:
    1. 在训练过程中周期性保存checkpoint
    2. 记录数据索引序列
    3. 生成PoL承诺（Merkle root）
    4. 响应验证挑战
    """

    def __init__(self, model, dataloader, criterion, args={}, watermarks={}):
        """
        初始化PoLTrainer

        Args:
            model: 模型
            dataloader: 数据加载器
            criterion: 损失函数
            args: 训练参数，应包含:
                - enable_pol: 是否启用PoL (bool)
                - pol_save_freq: checkpoint保存频率 (int)
                - pol_save_dir: PoL数据保存目录 (str)
                - pol_compress: 是否压缩checkpoint (bool, 默认True)
                - client_id: 客户端ID (str)
            watermarks: 水印参数
        """
        super().__init__(model, dataloader, criterion, args, watermarks)

        # PoL配置
        self.enable_pol = args.get('enable_pol', False)
        self.pol_save_freq = args.get('pol_save_freq', 10)
        self.pol_save_dir = args.get('pol_save_dir', 'pol_data')
        self.pol_compress = args.get('pol_compress', True)
        self.client_id = args.get('client_id', 'unknown')

        # 初始化PoLManager
        if self.enable_pol:
            # 从配置中获取新的参数
            from config.pol_config import POL_CONFIG

            self.pol_manager = PoLManager(
                client_id=self.client_id,
                save_dir=self.pol_save_dir,
                save_freq=self.pol_save_freq,
                compress=self.pol_compress,
                async_save=POL_CONFIG.get('async_save', False),
                save_to_disk=POL_CONFIG.get('save_checkpoints_to_disk', True),
                memory_limit=POL_CONFIG.get('memory_checkpoint_limit', 5),
                enable_auto_cleanup=POL_CONFIG.get('enable_auto_cleanup', False),
                auto_cleanup_interval=POL_CONFIG.get('auto_cleanup_interval', 50)
            )
            logger.info(f"PoL enabled for client {self.client_id}")
            logger.info(f"  Save frequency: {self.pol_save_freq}")
            logger.info(f"  Save to disk: {POL_CONFIG.get('save_checkpoints_to_disk', True)}")
        else:
            self.pol_manager = None
            logger.info("PoL disabled")

        # 训练状态
        self.batch_counter = 0
        self.data_indices = []  # 记录使用的数据索引
        self.criterion = criterion if criterion is not None else torch.nn.CrossEntropyLoss()

    @staticmethod
    def _set_dataset_replay_context(dataset, *, round_num=None, epoch=None):
        """Set deterministic replay context on wrapped datasets when supported."""
        seen = set()

        def visit(obj):
            if obj is None or id(obj) in seen:
                return
            seen.add(id(obj))
            setter = getattr(obj, 'set_replay_context', None)
            if callable(setter):
                setter(round_num=round_num, epoch=epoch)
            else:
                if round_num is not None and hasattr(obj, 'current_round'):
                    setattr(obj, 'current_round', int(round_num))
                if epoch is not None and hasattr(obj, 'current_epoch'):
                    setattr(obj, 'current_epoch', int(epoch))
            for attr in ('dataset', 'base', 'subset'):
                visit(getattr(obj, attr, None))

        visit(dataset)

    def train(self, total_epoch):
        """Override to run any deferred cleanup from previous round before training."""
        try:
            if getattr(self, 'enable_pol', False) and getattr(self, 'pol_manager', None) is not None:
                self.pol_manager.run_deferred_cleanup_if_any()
        except Exception as e:
            logger.debug(f"Deferred cleanup before training skipped: {e}")
        return super().train(total_epoch)

    def _train_epoch(self, epoch):
        """
        训练一个epoch，集成PoL checkpoint保存

        Args:
            epoch: 当前epoch编号

        Returns:
            ret: 训练结果字典
        """
        model = self.model
        args = self.args
        device = args.get("device", "cpu")

        model.to(device)
        model.train()
        self._set_dataset_replay_context(
            getattr(self.dataloader, 'dataset', None),
            round_num=args.get('round_num', args.get('round', 0)),
            epoch=epoch,
        )

        batch_loss = []

        for batch_idx, batch in enumerate(self.dataloader):
            # 支持 (x, y) 或 (x, y, idx) 形式
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                x, labels, idxs = batch
            else:
                x, labels = batch
                idxs = None
            x, labels = x.to(device), labels.to(device)
            batch_indices = ()

            # 记录数据索引（优先使用 dataloader 返回的真实全局索引）
            if self.enable_pol:
                batch_indices = None
                if idxs is not None:
                    try:
                        if hasattr(idxs, 'tolist'):
                            batch_indices = [int(i) for i in idxs.tolist()]
                        elif isinstance(idxs, (list, tuple)):
                            batch_indices = [int(i) for i in idxs]
                        else:
                            batch_indices = [int(idxs)]
                    except Exception:
                        batch_indices = None
                # 回退：根据批次号推断（仅当无法直接拿到真实索引时）
                if batch_indices is None:
                    batch_indices = self._get_batch_indices(batch_idx)

                if batch_indices is not None:
                    # 记录到 PoLManager（用于后续commitment与哈希）
                    self.pol_manager.record_data_indices(batch_indices)
                    # 同步维护到 trainer 自身（严格重放需要完整序列，禁止去重/重排）
                    try:
                        self.data_indices.extend([int(i) for i in list(batch_indices)])
                    except Exception:
                        pass

            # 前向传播
            log_probs = model(x)
            loss = self.criterion(log_probs, labels)

            # NaN/Inf guard on loss before backward
            if not torch.isfinite(loss):
                if os.getenv("POL_INTEGRITY", "0") == "1":
                    raise FloatingPointError(
                        f"non-finite loss at optimizer step {self.batch_counter + 1}"
                    )
                logger.warning(f"[NaNGuard] Non-finite loss at global_step={self.batch_counter+1}: {float(loss)}. Skipping this step.")
                self.optimizer.zero_grad()
                continue

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping to prevent explosion
            try:
                clip_value = self.args.get("clip_norm", 5.0)
                if clip_value is not None:
                    clip_norm = float(clip_value)
                    if clip_norm <= 0:
                        raise ValueError("clip norm must be positive or None")
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            except Exception:
                if os.getenv("POL_INTEGRITY", "0") == "1":
                    raise

            self.optimizer.step()

            # Post-step parameter NaN/Inf guard and adaptive LR backoff
            try:
                sanitized = False
                with torch.no_grad():
                    for name, p in model.named_parameters():
                        if p is None or (not p.requires_grad):
                            continue
                        if torch.isnan(p).any() or torch.isinf(p).any():
                            if os.getenv("POL_INTEGRITY", "0") == "1":
                                raise FloatingPointError(
                                    f"non-finite parameter {name} at optimizer step {self.batch_counter + 1}"
                                )
                            logger.warning(f"[NaNGuard] Param {name} has NaN/Inf at step {self.batch_counter+1}. Sanitizing and reducing LR.")
                            torch.nan_to_num_(p, nan=0.0, posinf=1e6, neginf=-1e6)
                            sanitized = True
                if sanitized:
                    for g in self.optimizer.param_groups:
                        try:
                            old_lr = float(g.get('lr', 0.0))
                            g['lr'] = max(old_lr * 0.5, 1e-5)
                        except Exception:
                            pass
            except Exception:
                pass

            batch_loss.append(loss.item())
            self.batch_counter += 1

            self._record_protocol_step(
                step=self.batch_counter,
                epoch=epoch,
                model=model,
                optimizer=self.optimizer,
                batch_data=x,
                batch_labels=labels,
                batch_indices=tuple(batch_indices or ()),
                activations={"logits": log_probs.detach()},
            )

            # PoL: 周期性保存checkpoint
            if self.enable_pol and self.batch_counter % self.pol_save_freq == 0:
                self._save_checkpoint(epoch, batch_idx, loss.item())

        # 计算epoch平均loss
        if len(batch_loss) == 0:
            epoch_loss = 0.0
        else:
            epoch_loss = sum(batch_loss) / len(batch_loss)

        ret = {
            'loss': epoch_loss,
            'epoch': epoch,
            'batch_count': len(batch_loss)
        }

        return ret

    def _record_protocol_step(self, **_kwargs):
        """Extension hook for the paper protocol recorder."""
        return None

    def _save_checkpoint(self, epoch: int, batch_idx: int, loss: float):
        """
        保存checkpoint

        Args:
            epoch: 当前epoch
            batch_idx: 当前batch索引
            loss: 当前loss
        """
        if not self.enable_pol or self.pol_manager is None:
            return

        checkpoint_data = {
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'epoch': epoch,
            'round': int(self.args.get('round_num', self.args.get('round', 0))),
            'step': self.batch_counter,
            'batch_idx': batch_idx,
            'loss': loss
        }

        ckpt_hash = self.pol_manager.save_checkpoint(
            step=self.batch_counter,
            checkpoint_data=checkpoint_data
        )

        logger.debug(f"Saved checkpoint at step {self.batch_counter}, hash: {ckpt_hash[:16]}...")

    def _get_batch_indices(self, batch_idx: int) -> Optional[List[int]]:
        """
        获取当前batch的真实数据索引

        Args:
            batch_idx: batch索引

        Returns:
            batch_indices: 数据索引列表，如果无法获取则返回None
        """
        try:
            # 方法1: 从Subset获取索引
            if hasattr(self.dataloader.dataset, 'indices'):
                start_idx = batch_idx * self.dataloader.batch_size
                end_idx = min(start_idx + self.dataloader.batch_size,
                             len(self.dataloader.dataset))
                batch_indices = self.dataloader.dataset.indices[start_idx:end_idx]
                return batch_indices.tolist() if hasattr(batch_indices, 'tolist') else list(batch_indices)

            # 方法2: 从batch_sampler获取
            elif hasattr(self.dataloader, 'batch_sampler') and hasattr(self.dataloader.batch_sampler, 'sampler'):
                # 对于SequentialSampler或RandomSampler
                sampler = self.dataloader.batch_sampler.sampler
                if hasattr(sampler, 'data_source'):
                    start_idx = batch_idx * self.dataloader.batch_size
                    end_idx = min(start_idx + self.dataloader.batch_size,
                                 len(sampler.data_source))
                    return list(range(start_idx, end_idx))

            # 方法3: 假设顺序采样（fallback）
            else:
                start_idx = batch_idx * self.dataloader.batch_size
                end_idx = min(start_idx + self.dataloader.batch_size,
                             len(self.dataloader.dataset))
                return list(range(start_idx, end_idx))

        except Exception as e:
            logger.warning(f"Failed to get batch indices: {e}")
            return None

    def finalize_pol(self, epoch: int, dataset=None) -> Optional[Dict]:
        """
        完成训练后生成PoL承诺

        Args:
            epoch: 最终epoch
            dataset: 数据集（用于计算数据哈希）

        Returns:
            pol_commitment: PoL承诺数据，包含:
                - commitment: Merkle root
                - num_checkpoints: checkpoint数量
                - data_hash: 数据哈希
                - client_id: 客户端ID
                - total_steps: 总步数
        """
        if not self.enable_pol or self.pol_manager is None:
            logger.warning("PoL not enabled, cannot generate commitment")
            return None

        # 保存最终checkpoint（避免重复保存同一步）
        try:
            last_meta = self.pol_manager.metadata.get('checkpoints', [])[-1] if self.pol_manager else None
            last_step = int(last_meta.get('step', -1)) if last_meta else -1
            will_save = (last_step != int(self.batch_counter))
            logger.info(f"[PoL finalize] client={self.client_id} last_meta_step={last_step} "
                        f"batch_counter={int(self.batch_counter)} will_save_final_ckpt={will_save}")
            if will_save:
                self._save_checkpoint(epoch, -1, 0.0)
        except Exception as e:
            logger.warning(f"[PoL finalize] fallback to save final checkpoint due to error: {e}")
            self._save_checkpoint(epoch, -1, 0.0)

        # 保存数据索引
        if self.data_indices:
            self.pol_manager.save_data_indices(self.data_indices)

        # 计算数据哈希
        data_hash = ""
        if dataset is not None:
            data_hash = self.pol_manager.compute_data_hash(dataset)

        # 生成Merkle root
        commitment = self.pol_manager.generate_commitment()

        # 标记本轮承诺已生成，进入待验证状态
        try:
            self.pol_manager.mark_commitment_generated(commitment)
        except Exception:
            pass

        # 保存元数据
        self.pol_manager.save_metadata()

        pol_commitment = {
            'commitment': commitment,
            'num_checkpoints': self.pol_manager.get_checkpoint_count(),
            'data_hash': data_hash,
            'client_id': self.client_id,
            'total_steps': self.batch_counter,
            'save_freq': self.pol_save_freq
        }

        logger.info(f"Generated PoL commitment for client {self.client_id}")
        logger.info(f"  Commitment: {commitment[:16]}...")
        logger.info(f"  Checkpoints: {pol_commitment['num_checkpoints']}")
        logger.info(f"  Total steps: {pol_commitment['total_steps']}")

        return pol_commitment

    def respond_to_challenge(self, challenge_data: Dict) -> Optional[Dict]:
        """
        响应验证挑战

        Args:
            challenge_data: 挑战数据，包含:
                - checkpoint_steps: 需要验证的checkpoint步数列表 (NEW)
                - checkpoint_indices: 需要验证的checkpoint索引列表 (DEPRECATED, for backward compatibility)

        Returns:
            response: 响应数据，包含:
                - checkpoints: checkpoint数据列表
                - data_indices: 对应的数据索引
        """
        if not self.enable_pol or self.pol_manager is None:
            logger.warning("PoL not enabled, cannot respond to challenge")
            return None

        # Support checkpoint_steps and the checkpoint_indices compatibility field.
        checkpoint_steps = challenge_data.get('checkpoint_steps', [])
        if not checkpoint_steps:
            # Use the compatibility field when checkpoint_steps is unavailable.
            checkpoint_indices = challenge_data.get('checkpoint_indices', [])
            checkpoint_steps = [idx * self.pol_save_freq for idx in checkpoint_indices]
            logger.debug(f"Using index-based challenge: {len(checkpoint_indices)} indices -> {len(checkpoint_steps)} steps")
        else:
            logger.debug(f"Using step-based challenge: {len(checkpoint_steps)} steps")

        checkpoints = []
        for step in checkpoint_steps:
            ckpt = self.pol_manager.load_checkpoint(step)
            if ckpt is not None:
                # 生成对应的Merkle证明
                proof = self.pol_manager.get_merkle_proof_by_step(step)
                leaf_index = self.pol_manager.get_leaf_index_for_step(step)
                checkpoints.append({
                    'index': leaf_index,
                    'step': step,
                    'data': ckpt,
                    'merkle_proof': proof
                })
            else:
                logger.warning(f"Checkpoint at step {step} not found (requested in challenge)")

        response = {
            'client_id': self.client_id,
            'checkpoints': checkpoints,
        }
        # Attach data_indices if available: prefer trainer copy; fallback to PoLManager's record
        indices = self.data_indices
        if (not indices) and getattr(self, 'pol_manager', None) is not None:
            try:
                indices = list(getattr(self.pol_manager, 'data_indices', []))
            except Exception:
                indices = []
        if indices:
            response['data_indices'] = indices

        # Optional: attach ZKP proof for the first adjacent pair if enabled
        try:
            if self.args.get('enable_zkp', False) and len(checkpoints) >= 2:
                use_sim = self.args.get('zkp_use_simulation', False)
                prover = ZKPProver(use_simulation=use_sim)
                # Flatten to vectors using model_state; prover handles dicts
                proof, public = prover.generate_proof(
                    checkpoints[0]['data']['model_state'],
                    checkpoints[1]['data']['model_state'],
                    self.data_indices,
                    max_distance=None
                )
                response['zkp'] = {
                    'pair': [checkpoints[0]['index'], checkpoints[1]['index']],
                    'proof': proof,
                    'public_signals': public,
                }
        except Exception as e:
            logger.warning(f"Failed to attach ZKP proof: {e}")

        logger.info(f"Responded to challenge with {len(checkpoints)} checkpoints")

        # After constructing response, it's safe to cleanup deferred checkpoints
        try:
            if getattr(self, 'enable_pol', False) and getattr(self, 'pol_manager', None) is not None:
                self.pol_manager.run_deferred_cleanup_if_any()
        except Exception as e:
            logger.debug(f"Deferred cleanup after challenge skipped: {e}")

        return response

    def _on_before_upload(self, epoch):
        """上传前的处理"""
        pass

    def _on_after_download(self, epoch):
        """下载后的处理"""
        pass

    def _upload_model(self, epoch):
        """上传模型到区块链"""
        uploaded_para = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'client_id': self.id if hasattr(self, 'id') else self.client_id
        }

        # 如果启用PoL，附加PoL承诺
        if self.enable_pol and self.pol_manager is not None:
            pol_commitment = self.finalize_pol(epoch)
            if pol_commitment:
                uploaded_para['pol_commitment'] = pol_commitment

        self.pipe.upload_model(uploaded_para)

    def _download_model(self, epoch):
        """从区块链下载模型"""
        download_params = self.pipe.download_model()
        self.model.load_state_dict(download_params['state_dict'])
