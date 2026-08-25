"""
PoL验证器
实现参数距离计算和验证逻辑
"""

import os
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from collections import OrderedDict

from client.pol.MerkleTree import MerkleTree

logger = logging.getLogger(__name__)


class PoLVerifier:
    """
    PoL验证器

    核心功能:
    1. 重放训练步骤
    2. 计算参数距离
    3. 验证checkpoint的有效性
    4. Top-Q验证策略
    """

    def __init__(self, args: Dict):
        """
        初始化验证器

        Args:
            args: 验证参数，包含:
                - delta: 距离阈值
                - distance_metric: 距离度量 ('l1', 'l2', 'linf', 'cosine')
                - device: 计算设备
                - top_q: Top-Q验证的Q值（可选）
        """
        import os
        self.delta = args.get('delta', 0.01)
        self.distance_metric = args.get('distance_metric', 'l2')
        self.device = args.get('device', 'cpu')
        self.top_q = args.get('top_q', None)
        # Minimum pairwise success rate to accept verification (root-cause fix)
        _mpsr_env = os.getenv('POL_MIN_PAIR_SUCCESS_RATE')
        try:
            _mpsr_from_args = float(args.get('min_pair_success_rate')) if args.get('min_pair_success_rate') is not None else None
        except Exception:
            _mpsr_from_args = None
        self.min_pair_success_rate = float(_mpsr_from_args if _mpsr_from_args is not None else (_mpsr_env if _mpsr_env is not None else 0.99))

        # Try to enforce determinism to safely support near-zero tolerance
        try:
            import torch
            if hasattr(torch, 'use_deterministic_algorithms'):
                torch.use_deterministic_algorithms(True)
            import torch.backends.cudnn as cudnn
            cudnn.benchmark = False
            cudnn.deterministic = True
        except Exception:
            pass

        logger.info(f"PoLVerifier initialized")
        logger.info(f"  Delta: {self.delta}")
        logger.info(f"  Distance metric: {self.distance_metric}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Min pair success rate: {self.min_pair_success_rate}")
        if self.top_q:
            logger.info(f"  Top-Q: {self.top_q}")

    def verify_response(self, challenge: Dict, response: Dict,
                       commitment: str, model, dataloader,
                       criterion, optimizer_class, lr: float) -> bool:
        """
        验证客户端的响应

        Args:
            challenge: 挑战数据
            response: 客户端响应
            commitment: PoL承诺（Merkle root）
            model: 模型架构
            dataloader: 数据加载器
            criterion: 损失函数
            optimizer_class: 优化器类
            lr: 学习率

        Returns:
            is_valid: 验证是否通过
        """
        checkpoints = response.get('checkpoints', [])

        if not checkpoints:
            logger.warning("No checkpoints in response")
            return False

        # 先验证Merkle成员关系，确保checkpoint确属该承诺
        if not self._verify_merkle_membership(checkpoints, commitment):
            logger.warning("Merkle membership verification failed")
            return False

        # 至少需要两个checkpoint才能验证相邻步
        if len(checkpoints) < 2:
            logger.warning("Need at least 2 checkpoints for verification")
            return False

        # 验证相邻checkpoint对（共有 N-1 对）
        valid_count = 0
        total_pairs = max(1, len(checkpoints) - 1)

        for i in range(total_pairs):
            current_ckpt = checkpoints[i]
            next_ckpt = checkpoints[i + 1]

            # 重放训练或退化为参数距离校验
            is_valid = self._verify_single_step(
                current_ckpt=current_ckpt,
                next_ckpt=next_ckpt,
                model=model,
                dataloader=dataloader,
                criterion=criterion,
                optimizer_class=optimizer_class,
                lr=lr,
                data_indices=response.get('data_indices')
            )

            if is_valid:
                valid_count += 1

        # 判断验证是否通过（基于对数，而非checkpoint数）
        success_rate = valid_count / total_pairs
        is_valid = success_rate >= float(self.min_pair_success_rate)

        logger.info(f"Verification result: {valid_count}/{total_pairs} pairs passed")
        logger.info(f"  Success rate: {success_rate:.2%}")
        logger.info(f"  Threshold: {self.min_pair_success_rate}")
        logger.info(f"  Overall valid: {is_valid}")

        return is_valid

    def _compute_checkpoint_hash(self, checkpoint_data: Dict) -> str:
        """
        Compute SHA256 over sorted model_state tensors, matching client hashing.
        """
        import hashlib
        model_state = checkpoint_data.get('model_state', {})
        m = hashlib.sha256()
        for key in sorted(model_state.keys()):
            param = model_state[key]
            if isinstance(param, torch.Tensor):
                m.update(param.cpu().numpy().tobytes())
        return m.hexdigest()

    def _verify_merkle_membership(self, checkpoints: List[Dict], commitment_root: str) -> bool:
        """
        Verify each checkpoint's hash is included in the commitment Merkle root via provided proofs.
        """
        for ckpt in checkpoints:
            proof = ckpt.get('merkle_proof', None)
            data = ckpt.get('data', {})
            if proof is None:
                logger.warning("Missing merkle_proof in checkpoint")
                return False
            leaf_hash = self._compute_checkpoint_hash(data)
            if not MerkleTree.verify_proof(leaf_hash, proof, commitment_root):
                logger.warning("Merkle proof verification failed for a checkpoint")
                return False
        return True

    def _verify_single_step(self, current_ckpt: Dict, next_ckpt: Dict,
                           model, dataloader, criterion,
                           optimizer_class, lr: float,
                           data_indices: Optional[List[int]] = None) -> bool:
        """
        严格按设计：对相邻 checkpoint 进行逐步重放验证（不做参数距离步长缩放）。
        要求提供完整 data_indices（训练期间记录的样本索引顺序）。
        """
        import copy, torch
        # 1) 基本校验
        if data_indices is None or len(data_indices) == 0:
            logger.warning("Missing data_indices for strict replay; fail by design.")
            return False
        try:
            curr_step = int(current_ckpt.get('data', {}).get('step'))
            next_step = int(next_ckpt.get('data', {}).get('step'))
        except Exception:
            logger.warning("Missing step numbers in checkpoints; cannot replay.")
            return False
        if next_step <= curr_step:
            logger.warning(f"Non-increasing steps: curr={curr_step}, next={next_step}")
            return False
        steps_to_replay = next_step - curr_step

        # 2) 克隆模型并加载当前checkpoint参数与优化器状态
        local_model = copy.deepcopy(model).to(self.device)
        local_model.load_state_dict(current_ckpt['data']['model_state'])
        local_model.train()
        optimizer = optimizer_class(local_model.parameters(), lr=lr)
        try:
            opt_state = current_ckpt['data'].get('optimizer_state', None)
            if opt_state is not None:
                optimizer.load_state_dict(opt_state)
        except Exception as e:
            logger.debug(f"Load optimizer_state failed (continue with provided hyperparams): {e}")

        # 3) 根据 dataloader 的 batch_size / drop_last / dataset 长度 复原批次边界
        dataset = getattr(dataloader, 'dataset', None)
        if dataset is None:
            logger.warning("Dataloader.dataset missing; cannot replay.")
            return False
        bs = int(getattr(dataloader, 'batch_size', 32))
        drop_last = bool(getattr(dataloader, 'drop_last', False))
        ds_len = int(len(dataset)) if hasattr(dataset, '__len__') else None
        if ds_len is None:
            logger.warning("Dataset length unknown; cannot replay.")
            return False
        if drop_last:
            batches_per_epoch = ds_len // bs
            last_batch_size = bs
        else:
            batches_per_epoch = (ds_len + bs - 1) // bs
            last_batch_size = ds_len - bs * (batches_per_epoch - 1) if batches_per_epoch > 0 else 0
            if last_batch_size <= 0:
                last_batch_size = bs

        def batch_size_for_step(step_idx_1based: int) -> int:
            idx_in_epoch = (step_idx_1based - 1) % batches_per_epoch
            return bs if (idx_in_epoch < batches_per_epoch - 1 or drop_last) else last_batch_size

        def consumed_samples_upto(step_1based: int) -> int:
            if step_1based <= 0:
                return 0
            full_epochs = (step_1based) // batches_per_epoch
            rem_steps = (step_1based) % batches_per_epoch
            total = full_epochs * ((batches_per_epoch - 1) * bs + (last_batch_size if not drop_last else bs))
            for i in range(1, rem_steps + 1):
                total += batch_size_for_step(i)
            return total

        start_offset = consumed_samples_upto(curr_step)
        expected_window = sum(
            batch_size_for_step(curr_step + offset + 1)
            for offset in range(steps_to_replay)
        )
        compact_window = len(data_indices) < start_offset + expected_window
        if compact_window and len(data_indices) < steps_to_replay:
            logger.warning(
                "Recorded data_indices cannot provide one sample per replay step"
            )
            return False

        def _set_dataset_replay_context(dataset_obj, *, round_num=None, epoch=None):
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

            visit(dataset_obj)

        # 4) 若 dataset 为 Subset，建立 global->local 的索引映射
        subset_local_from_global = None
        try:
            if hasattr(dataset, 'indices'):
                idx_list = list(getattr(dataset, 'indices'))
                subset_local_from_global = {int(g): i for i, g in enumerate(idx_list)}
        except Exception as e:
            logger.debug(f"Build subset index mapping failed: {e}")

        # 5) 逐步重放 Δ 个 step
        # OPTIMIZATION: Use DataLoader for efficient batch loading instead of individual sample access
        total_consumed = 0
        for s in range(steps_to_replay):
            step_1based = curr_step + s + 1
            replay_epoch = int((step_1based - 1) // batches_per_epoch)
            replay_round = int(current_ckpt.get('data', {}).get('round', 0))
            _set_dataset_replay_context(dataset, round_num=replay_round, epoch=replay_epoch)
            if compact_window:
                quotient, remainder = divmod(
                    len(data_indices), steps_to_replay
                )
                need = quotient + (1 if s < remainder else 0)
                begin = s * quotient + min(s, remainder)
                end = begin + need
            else:
                need = batch_size_for_step(step_1based)
                begin = start_offset + total_consumed
                end = begin + need
            if end > len(data_indices):
                logger.warning("Recorded data_indices shorter than required for replay window")
                return False
            batch_idx_list = data_indices[begin:end]

            # CRITICAL FIX: Use DataLoader for batch loading instead of individual dataset[idx] calls
            # This reduces I/O overhead from ~20s to ~2s per verification pair
            try:
                from torch.utils.data import Subset, DataLoader

                # Convert global indices to local indices if using Subset
                if subset_local_from_global is not None:
                    local_indices = []
                    for gidx in batch_idx_list:
                        local_idx = subset_local_from_global.get(int(gidx), None)
                        if local_idx is None:
                            logger.warning(f"Global index {gidx} not found in Subset mapping")
                            return False
                        local_indices.append(local_idx)
                    batch_subset = Subset(dataset, local_indices)
                else:
                    batch_subset = Subset(dataset, [int(idx) for idx in batch_idx_list])

                # Use DataLoader with optimized settings for fast batch loading
                # num_workers=0 to avoid multiprocessing overhead for small batches
                # pin_memory=True if using GPU for faster transfer
                batch_loader = DataLoader(
                    batch_subset,
                    batch_size=len(batch_idx_list),
                    shuffle=False,
                    num_workers=0,  # Single-threaded for small batches
                    pin_memory=(self.device != 'cpu')
                )

                # Load the entire batch at once
                batch_data = next(iter(batch_loader))

                # Handle different dataset return formats: (x, y) or (x, y, idx)
                if isinstance(batch_data, (list, tuple)):
                    if len(batch_data) >= 2:
                        x_batch, y_batch = batch_data[0], batch_data[1]
                    else:
                        x_batch = batch_data[0]
                        y_batch = torch.zeros(len(x_batch), dtype=torch.long)
                else:
                    x_batch = batch_data
                    y_batch = torch.zeros(len(x_batch), dtype=torch.long)

                # Move to device
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device, dtype=torch.long)

            except Exception as e:
                # Fallback to old method if DataLoader fails (should be rare)
                logger.warning(f"DataLoader batch loading failed, falling back to individual loading: {e}")
                xs, ys = [], []
                for gidx in batch_idx_list:
                    try:
                        if subset_local_from_global is not None:
                            local_idx = subset_local_from_global.get(int(gidx), None)
                            if local_idx is None:
                                logger.warning("Global index not found in Subset mapping")
                                return False
                            val = dataset[local_idx]
                        else:
                            val = dataset[int(gidx)]
                        # Support (x,y) or (x,y,idx)
                        if isinstance(val, (list, tuple)) and len(val) >= 2:
                            x, y = val[0], val[1]
                        else:
                            x, y = val, None
                        xs.append(x)
                        ys.append(int(y) if (y is not None and not isinstance(y, torch.Tensor)) else (int(y.item()) if isinstance(y, torch.Tensor) else 0))
                    except Exception as e2:
                        logger.warning(f"Failed to fetch sample {gidx}: {e2}")
                        return False
                try:
                    x_batch = torch.stack(xs).to(self.device)
                except Exception:
                    x_batch = torch.stack([torch.as_tensor(x) for x in xs]).to(self.device)
                y_batch = torch.as_tensor(ys, dtype=torch.long, device=self.device)

            # Match training-time guards and behavior
            optimizer.zero_grad(set_to_none=True)
            logits = local_model(x_batch)
            loss = criterion(logits, y_batch)

            # Pre-backward finite-loss guard
            try:
                if not torch.isfinite(loss):
                    logger.warning(f"[NaNGuard-Verify] Non-finite loss at replay_step={s+1}: {float(loss)}. Skipping this step.")
                    # Skip this step without updating model; do not advance steps_to_replay counter explicitly
                    # We still consume the same batch of indices to align offsets
                    total_consumed += need
                    continue
            except Exception:
                pass

            # Backward
            loss.backward()

            # Gradient clipping (mirror training default or POL_CLIP_NORM override)
            try:
                clip_norm = 5.0
                try:
                    clip_norm = float(os.getenv('POL_CLIP_NORM', '5.0'))
                except Exception:
                    pass
                torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=clip_norm)
            except Exception:
                pass

            optimizer.step()

            # Post-step parameter NaN/Inf guard and adaptive LR backoff (mirror training)
            try:
                sanitized = False
                with torch.no_grad():
                    for name, p in local_model.named_parameters():
                        if p is None or (not p.requires_grad):
                            continue
                        if torch.isnan(p).any() or torch.isinf(p).any():
                            logger.warning(f"[NaNGuard-Verify] Param {name} has NaN/Inf at replay_step={s+1}. Sanitizing and reducing LR.")
                            torch.nan_to_num_(p, nan=0.0, posinf=1e6, neginf=-1e6)
                            sanitized = True
                if sanitized:
                    for g in optimizer.param_groups:
                        try:
                            old_lr = float(g.get('lr', 0.0))
                            g['lr'] = max(old_lr * 0.5, 1e-5)
                        except Exception:
                            pass
            except Exception:
                pass

            total_consumed += need

        produced = local_model.state_dict()
        target = next_ckpt['data']['model_state']
        dist = self._compute_parameter_distance(produced, target, self.distance_metric)
        ok = (dist <= float(self.delta))
        # Elevate to INFO and include key context to diagnose persistent 0/3 pass cases
        logger.info(f"[DIAG] Replay verification: distance={dist:.6f}, threshold={self.delta}, valid={ok}, curr_step={curr_step}, next_step={next_step}, steps_to_replay={steps_to_replay}")
        return ok

    def _compute_parameter_distance(self, state1: OrderedDict,
                                   state2: OrderedDict,
                                   metric: str = 'l2') -> float:
        """
        计算两个模型参数之间的距离

        Args:
            state1: 第一个模型的state_dict
            state2: 第二个模型的state_dict
            metric: 距离度量 ('l1', 'l2', 'linf', 'cosine')

        Returns:
            distance: 参数距离
        """
        # 将参数展平为向量
        params1 = self._flatten_parameters(state1)
        params2 = self._flatten_parameters(state2)

        # 确保在同一设备上
        params1 = params1.to(self.device)
        params2 = params2.to(self.device)
        # DIAG: Check for NaNs/Infs before distance to explain NaN distances
        try:
            n1 = torch.isnan(params1).sum().item(); p1 = torch.isinf(params1).sum().item()
            n2 = torch.isnan(params2).sum().item(); p2 = torch.isinf(params2).sum().item()
            if (n1 + p1 + n2 + p2) > 0:
                logger.warning(f"[DIAG] NaN/Inf detected in states before distance: params1_nan={n1}, params1_inf={p1}, params2_nan={n2}, params2_inf={p2}")
        except Exception:
            pass


        # 计算距离
        if metric == 'l1':
            distance = torch.norm(params1 - params2, p=1).item()
        elif metric == 'l2':
            distance = torch.norm(params1 - params2, p=2).item()
        elif metric == 'linf':
            distance = torch.norm(params1 - params2, p=float('inf')).item()
        elif metric == 'cosine':
            # 余弦距离 = 1 - 余弦相似度
            # 使用torch.nn.functional.cosine_similarity以获得更稳定的计算
            from torch.nn.functional import cosine_similarity
            # 确保参数不为零向量
            norm1 = torch.norm(params1)
            norm2 = torch.norm(params2)
            if norm1 > 1e-8 and norm2 > 1e-8:
                cos_sim = torch.dot(params1, params2) / (norm1 * norm2)
                distance = (1 - cos_sim).item()
            else:
                # 如果任一向量为零，返回最大距离
                distance = 2.0
        else:
            raise ValueError(f"Unknown distance metric: {metric}")

        return distance

    def _flatten_parameters(self, state_dict: OrderedDict) -> torch.Tensor:
        """
        将state_dict展平为一维向量

        Args:
            state_dict: 模型的state_dict

        Returns:
            flattened: 展平的参数向量
        """
        params = []
        exclude_substrings = (
            'running_mean', 'running_var', 'num_batches_tracked'
        )
        for key in sorted(state_dict.keys()):
            if any(sub in key for sub in exclude_substrings):
                continue  # Exclude BN buffers and counters
            param = state_dict[key]
            if isinstance(param, torch.Tensor):
                params.append(param.reshape(-1))

        if not params:
            return torch.tensor([])

        return torch.cat(params)

    def select_top_q_checkpoints(self, checkpoints: List[Dict], q: int) -> List[int]:
        """
        选择Top-Q个计算量最大的checkpoint进行验证

        Args:
            checkpoints: checkpoint列表
            q: 选择的数量

        Returns:
            indices: 选中的checkpoint索引列表
        """
        if len(checkpoints) <= q:
            return list(range(len(checkpoints)))

        # 计算相邻checkpoint之间的参数变化量
        distances = []
        for i in range(len(checkpoints) - 1):
            state1 = checkpoints[i]['data']['model_state']
            state2 = checkpoints[i + 1]['data']['model_state']
            dist = self._compute_parameter_distance(state1, state2, 'l2')
            distances.append((i, dist))

        # 选择距离最大的Q个
        distances.sort(key=lambda x: x[1], reverse=True)
        top_q_indices = [idx for idx, _ in distances[:q]]
        top_q_indices.sort()

        logger.info(f"Selected top-{q} checkpoints: {top_q_indices}")
        return top_q_indices

    def verify_on_pairs_indices(self, challenge: Dict, response: Dict,
                                commitment: str, model, dataloader,
                                criterion, optimizer_class, lr: float,
                                pair_indices: List[int]) -> bool:
        """
        验证指定的相邻checkpoint对（由pair_indices给出起点索引）。
        例如 pair_indices=[0,3] 表示验证 (0->1) 和 (3->4) 两对。
        """
        checkpoints = response.get('checkpoints', [])
        if not checkpoints:
            logger.warning("No checkpoints in response")
            return False

        # Merkle成员校验
        if not self._verify_merkle_membership(checkpoints, commitment):
            logger.warning("Merkle membership verification failed")
            return False

        # 去重并过滤非法索引
        pair_indices = sorted(set(int(i) for i in pair_indices if isinstance(i, (int, np.integer))))
        pair_indices = [i for i in pair_indices if (0 <= i < len(checkpoints) - 1)]
        if not pair_indices:
            logger.warning("Empty pair_indices after filtering; nothing to verify")
            return False

        valid_count = 0
        for idx in pair_indices:
            current_ckpt = checkpoints[idx]
            next_ckpt = checkpoints[idx + 1]
            ok = self._verify_single_step(
                current_ckpt=current_ckpt,
                next_ckpt=next_ckpt,
                model=model,
                dataloader=dataloader,
                criterion=criterion,
                optimizer_class=optimizer_class,
                lr=lr,
                data_indices=response.get('data_indices')
            )
            if ok:
                valid_count += 1

        success_rate = valid_count / len(pair_indices)
        is_valid = success_rate >= float(self.min_pair_success_rate)
        logger.info(f"Custom-indices verification: {valid_count}/{len(pair_indices)} passed")
        logger.info(f"  Success rate: {success_rate:.2%}")
        logger.info(f"  Threshold: {self.min_pair_success_rate}")
        logger.info(f"  Overall valid: {is_valid}")
        return is_valid

    def verify_with_top_q(self, challenge: Dict, response: Dict,
                         commitment: str, model, dataloader,
                         criterion, optimizer_class, lr: float,
                         q: int) -> bool:
        """
        使用Top-Q策略验证

        Args:
            challenge: 挑战数据
            response: 客户端响应
            commitment: PoL承诺
            model: 模型
            dataloader: 数据加载器
            criterion: 损失函数
            optimizer_class: 优化器类
            lr: 学习率
            q: Top-Q的Q值

        Returns:
            is_valid: 验证是否通过
        """
        checkpoints = response.get('checkpoints', [])

        if not checkpoints:
            logger.warning("No checkpoints in response")
            return False

        # 先验证Merkle成员关系，确保checkpoint确属该承诺
        if not self._verify_merkle_membership(checkpoints, commitment):
            logger.warning("Merkle membership verification failed")
            return False

        # 选择Top-Q checkpoint
        top_q_indices = self.select_top_q_checkpoints(checkpoints, q)

        # 只验证选中的checkpoint
        valid_count = 0
        for idx in top_q_indices:
            if idx + 1 >= len(checkpoints):
                continue

            current_ckpt = checkpoints[idx]
            next_ckpt = checkpoints[idx + 1]

            is_valid = self._verify_single_step(
                current_ckpt=current_ckpt,
                next_ckpt=next_ckpt,
                model=model,
                dataloader=dataloader,
                criterion=criterion,
                optimizer_class=optimizer_class,
                lr=lr,
                data_indices=response.get('data_indices')
            )

            if is_valid:
                valid_count += 1

        # 判断验证是否通过
        success_rate = valid_count / len(top_q_indices) if top_q_indices else 0
        is_valid = success_rate >= float(self.min_pair_success_rate)

        logger.info(f"Top-Q verification: {valid_count}/{len(top_q_indices)} passed")
        logger.info(f"  Success rate: {success_rate:.2%}")
        logger.info(f"  Threshold: {self.min_pair_success_rate}")

        return is_valid
