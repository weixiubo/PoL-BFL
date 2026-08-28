from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, OrderedDict
from client.clients import Client
class ServerAggregator(ABC):
    """Abstract interface for server-side federated learning aggregation."""

    def __init__(self, model=None, args=None):
        self.model = model
        self.id = 0
        self.args = args
        self.model_pool = []
    def set_id(self, aggregator_id):
        self.id = aggregator_id
    def receive_upload(self,client_pool:List[Client]):
        for client in client_pool:
            self.model_pool.append(client.get_model_state_dict())

    @abstractmethod
    def _aggregate_alg(self,raw_client_model_or_grad_list:List[OrderedDict]=None):
        """Aggregate client state dictionaries into one server model state."""
        pass


    @abstractmethod
    def _on_before_aggregation(
        self, raw_client_model_or_grad_list: List[OrderedDict]
    ) -> List[OrderedDict]:
        """Apply optional filtering or transformations before aggregation."""
        pass

    def aggregate(self, raw_client_model_or_grad_list: [OrderedDict]=None) -> OrderedDict:
        if(raw_client_model_or_grad_list is None): raw_client_model_or_grad_list = self.model_pool
        return self._aggregate_alg(raw_client_model_or_grad_list)

    def _on_after_aggregation(self, aggregated_model_or_grad: OrderedDict) -> OrderedDict:
        """Apply an optional post-aggregation transformation.

        The base implementation is the identity transformation.
        """
        return aggregated_model_or_grad

    @abstractmethod
    def test(self, test_data, device, args):
        pass

