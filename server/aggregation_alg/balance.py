from ..base.baseAggregator import ServerAggregator
import numpy as np
import torch
from collections import OrderedDict

class balanceAggregator(ServerAggregator):
    def __init__(self, gamma, kappa, init_model):
        super().__init__()
        self.gamma = gamma
        self.kappa = kappa
        self.saved_model = init_model
        
    def _on_before_aggregation(self, raw_client_model_or_grad_list):
        return raw_client_model_or_grad_list
        
    def _on_after_aggregation(self, aggregated_model_or_grad):
        return aggregated_model_or_grad
        
    def test(self, test_data=None, device=None, args=None):
        return {
            'gamma': self.gamma,
            'kappa': self.kappa,
            'pooled_updates': len(self.model_pool),
        }

    @staticmethod
    def _flatten(model):
        if isinstance(model, dict):
            parts = []
            for key in sorted(model):
                value = model[key]
                if torch.is_tensor(value):
                    value = value.detach().cpu().numpy()
                parts.append(np.asarray(value, dtype=np.float64).reshape(-1))
            return np.concatenate(parts)
        return np.asarray(model, dtype=np.float64).reshape(-1)
        

    def _aggregate_alg(self, raw_client_model_or_grad_list=None, t=None):
        if raw_client_model_or_grad_list is None:
            raw_client_model_or_grad_list = self.model_pool
        if not raw_client_model_or_grad_list:
            raise ValueError("balance aggregation requires client updates")

        # 假设 w_global 是当前的全局模型参数
        w_global = self.saved_model  # 需要确保 self.global_model 已定义

        # 计算 w_global 的范数
        global_vector = self._flatten(w_global)
        w_global_norm = np.linalg.norm(global_vector)

        # 定义 lambda 函数，假设 lambda(t) = 1 / (1 + t)
        lambda_t = 1 / (1 + t) if t is not None else 1

        # 计算距离阈值
        gamma = self.gamma  # 需要确保 gamma 已定义
        kappa = self.kappa  # 需要确保 kappa 已定义
        distance_threshold = gamma * np.exp(-kappa * lambda_t) * w_global_norm

        # 筛选符合条件的客户端模型参数
        filtered_models = []
        for model in raw_client_model_or_grad_list:
            # 计算模型参数与 w_global 的距离
            model_vector = self._flatten(model)
            if model_vector.shape != global_vector.shape:
                raise ValueError("balance client updates have incompatible shapes")
            distance = np.linalg.norm(model_vector - global_vector)
            if distance < distance_threshold:
                filtered_models.append(model)
            else:
                continue

        if not filtered_models:
            return w_global

        # 对筛选后的模型参数进行平均聚合
        if isinstance(filtered_models[0], dict):
            keys = filtered_models[0].keys()
            aggregated_model = OrderedDict()

            for key in keys:
                values = [model[key] for model in filtered_models]
                if torch.is_tensor(values[0]):
                    aggregated_model[key] = torch.stack(values).mean(dim=0)
                else:
                    aggregated_model[key] = np.mean(values, axis=0)

            return aggregated_model

        elif isinstance(filtered_models[0], list):
            # 将列表转换为 NumPy 数组以利用向量化运算
            grads = np.array(filtered_models)
            mean_grad = np.mean(grads, axis=0)
            return mean_grad.tolist()

        else:
            # 处理其他数据类型的情况（例如，单一数值）
            try:
                aggregated_value = np.mean(filtered_models)
                return aggregated_value
            except Exception as e:
                raise ValueError("Unsupported data type for aggregation") from e
