from ..base.baseAggregator import ServerAggregator
import numpy as np

class krumAggregator(ServerAggregator):
    def __init__(self, num_byzantine=None):
        super().__init__()
        if num_byzantine is not None and num_byzantine < 0:
            raise ValueError("num_byzantine cannot be negative")
        self.num_byzantine = num_byzantine
        self.last_scores = None

    def _on_before_aggregation(self, raw_client_model_or_grad_list):
        return raw_client_model_or_grad_list

    def _on_after_aggregation(self, aggregated_model_or_grad):
        return aggregated_model_or_grad

    def test(self, test_data=None, device=None, args=None):
        return {
            'scores': self.last_scores.tolist() if self.last_scores is not None else [],
            'num_byzantine': self.num_byzantine,
        }

    @staticmethod
    def _flatten(update):
        if isinstance(update, dict):
            parts = []
            for key in sorted(update):
                value = update[key]
                if hasattr(value, 'detach'):
                    value = value.detach().cpu().numpy()
                parts.append(np.asarray(value, dtype=np.float64).reshape(-1))
            return np.concatenate(parts)
        return np.asarray(update, dtype=np.float64).reshape(-1)

    def _aggregate_alg(self, raw_client_model_or_grad_list=None):
        if raw_client_model_or_grad_list is None:
            raw_client_model_or_grad_list = self.model_pool
        if not raw_client_model_or_grad_list:
            raise ValueError("Krum requires at least one client update")
        vectors = [self._flatten(update) for update in raw_client_model_or_grad_list]
        if any(vector.size != vectors[0].size for vector in vectors):
            raise ValueError("Krum client updates have incompatible shapes")
        matrix = np.stack(vectors)
        n = len(matrix)
        f = self.num_byzantine if self.num_byzantine is not None else n // 4
        if n < 2 * f + 3:
            raise ValueError("Krum requires n >= 2f + 3 clients")
        neighbor_count = n - f - 2
        distances = np.sum(
            (matrix[:, None, :] - matrix[None, :, :]) ** 2,
            axis=2,
        )
        scores = np.empty(n, dtype=np.float64)
        for index in range(n):
            peers = np.delete(distances[index], index)
            scores[index] = np.sort(peers)[:neighbor_count].sum()
        self.last_scores = scores
        return raw_client_model_or_grad_list[int(np.argmin(scores))]
