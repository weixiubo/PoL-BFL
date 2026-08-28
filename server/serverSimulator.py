#This implement the base class of server
from server.base.baseAggregator import ServerAggregator
import os
import torch
from copy import deepcopy
from pathlib import Path
from typing import OrderedDict

class serverSimulator:
    # A local serverSimulator to simulate the blockchain
    # This serverSimulator act as the same with the serverProxy in chainfl.
    
    def __init__(
        self,
        aggregator:ServerAggregator,
        client_num=10,
        args = None
        #args params give the server some other information to initialize
        #args is a dict which can contain below keys:
        # 1.checkpoint_folder: indicate the saved model path
        # 2.model: indicate the saved model name
        # 3.train_method: indicate the aggregated method used,
        #                 attention fedavg is used if provided method doesn't implemented
        #
    ) -> None:
        self.aggregator = aggregator
        self.client_num = client_num
        self.global_model = None
        self.args = dict(args or {})
        self.upload_model_list = []
    def _clear_upload_model_list(self):
        self.upload_model_list = []
        
    def _is_all_client_upload(self) -> bool:
        return len(self.upload_model_list) >= self.client_num

    def _set_global_model(self, global_model):
        self.global_model = global_model

    def _set_test_dataset(self, test_dataset):
        if test_dataset is None:
            raise ValueError("test dataset is required")
        self.test_dataset = test_dataset
        self.test_batch_size = test_dataset.batch_size
        
    def _load_model(self):
        file_path = Path(self.args['checkpoint_folder']) / str(self.args['model'])
        state = torch.load(file_path, map_location='cpu')
        if hasattr(self.global_model, 'load_state_dict'):
            self.global_model.load_state_dict(state)
        else:
            self.global_model = state
        return self.global_model

    def save_model(self,file_name='saved_model'):
        if self.global_model is None:
            raise RuntimeError("global model is unavailable")
        save_path = Path(self.args['checkpoint_folder'])
        save_path.mkdir(parents=True, exist_ok=True)
        file_path = save_path / file_name
        count = 0
        while file_path.exists():
            count += 1
            file_path = save_path / f"{file_name}.{count}"
        payload = (
            self.global_model.state_dict()
            if hasattr(self.global_model, 'state_dict')
            else self.global_model
        )
        torch.save(payload, file_path)
        return str(file_path)
    
    def upload_model(self,upload_params:dict):
        if 'state_dict' not in upload_params:
            raise ValueError("upload parameters require state_dict")
        model_state_dict = upload_params['state_dict']
        self.upload_model_list.append(model_state_dict)
        if(self._is_all_client_upload()):
            trained_model = self.aggregator.aggregate(self.upload_model_list)
            self._set_global_model(trained_model)
            self._clear_upload_model_list()
        
    def download_model(self,params=None) -> OrderedDict:
        if self.global_model is None:
            raise RuntimeError("global model is unavailable")
        return deepcopy(self.global_model)
    
    def test(
        self,
        model=None,
        test_dataset=None,
        criterion=None,
        device='cpu',
    ):
        """Evaluate the current global state on a configured data loader."""
        loader = test_dataset or getattr(self, 'test_dataset', None)
        if loader is None:
            raise RuntimeError("test dataset is unavailable")
        evaluated = model or (
            self.global_model if hasattr(self.global_model, 'eval') else None
        )
        if evaluated is None:
            raise RuntimeError("a model architecture is required for evaluation")
        if not hasattr(self.global_model, 'eval'):
            evaluated.load_state_dict(self.global_model, strict=True)
        evaluated = evaluated.to(device)
        evaluated.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        with torch.no_grad():
            for batch in loader:
                inputs, labels = batch[:2]
                inputs = inputs.to(device)
                labels = labels.to(device)
                logits = evaluated(inputs)
                if criterion is not None:
                    total_loss += float(criterion(logits, labels).item()) * len(labels)
                correct += int((logits.argmax(dim=1) == labels).sum().item())
                total += int(labels.numel())
        if total == 0:
            raise ValueError("test dataset is empty")
        return {
            'accuracy': correct / total,
            'loss': total_loss / total if criterion is not None else None,
            'samples': total,
        }
    
if __name__ == '__main__':
    import torch.nn as nn
    class LinearModel(nn.Module):
        def __init__(self, h_dims):
            super(LinearModel,self).__init__()

            models = []
            for i in range(len(h_dims) - 1):
                models.append(nn.Linear(h_dims[i], h_dims[i + 1]))
                if i != len(h_dims) - 2:
                    models.append(nn.ReLU()) 
            self.models = nn.Sequential(*models)
        def forward(self, X):
            return self.models(X)
    
    from server.aggregation_alg.fedavg import fedavgAggregator

    test_sample_pool = [LinearModel([10, 1]) for _ in range(10)]
    a = fedavgAggregator()
    server = serverSimulator(a)
    for i in test_sample_pool:
        upload_param = {'state_dict': i.state_dict()}
        server.upload_model(upload_param)
