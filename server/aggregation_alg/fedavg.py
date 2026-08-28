"""Federated Averaging aggregator."""

from collections import OrderedDict

from ..base.baseAggregator import ServerAggregator

class fedavgAggregator(ServerAggregator):
    def __init__(self):
        super().__init__()

    def _on_before_aggregation(self, raw_client_model_or_grad_list):
        return raw_client_model_or_grad_list

    def _on_after_aggregation(self, aggregated_model_or_grad):
        return aggregated_model_or_grad

    def test(self, test_data=None, device=None, args=None):
        return {'pooled_updates': len(self.model_pool)}

    def _aggregate_alg(self, raw_client_model_or_grad_list=None):
        if raw_client_model_or_grad_list is None:
            raw_client_model_or_grad_list = self.model_pool
        if not raw_client_model_or_grad_list:
            raise ValueError("FedAvg requires at least one client update")
        if isinstance(raw_client_model_or_grad_list[0], dict):
            num_clients = len(raw_client_model_or_grad_list)
            expected_keys = tuple(raw_client_model_or_grad_list[0])
            if any(tuple(model) != expected_keys for model in raw_client_model_or_grad_list):
                raise ValueError("FedAvg client updates have incompatible keys")
            aggregated_model = OrderedDict()
            for key in raw_client_model_or_grad_list[0].keys():
                sum_values = sum(client_model[key] for client_model in raw_client_model_or_grad_list)
                aggregated_model[key] = sum_values / num_clients
            return aggregated_model
        elif isinstance(raw_client_model_or_grad_list[0], list):
            num_clients = len(raw_client_model_or_grad_list)
            width = len(raw_client_model_or_grad_list[0])
            if any(len(gradient) != width for gradient in raw_client_model_or_grad_list):
                raise ValueError("FedAvg gradient updates have incompatible lengths")
            grads_sum = [sum(grad[i] for grad in raw_client_model_or_grad_list) for i in range(len(raw_client_model_or_grad_list[0]))]
            return [grad_sum / num_clients for grad_sum in grads_sum]
        raise TypeError("FedAvg updates must be mappings or gradient lists")


if __name__ == '__main__':
    client1_model = {'weight': 0.5, 'bias': 0.2}
    client2_model = {'weight': 0.3, 'bias': 0.1}
    client3_model = {'weight': 0.6, 'bias': 0.3}

    aggregator = fedavgAggregator()
    aggregated_model = aggregator._aggregate_alg([client1_model, client2_model, client3_model])

    print(aggregated_model)
