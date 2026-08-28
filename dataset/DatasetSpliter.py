import logging
import numpy as np
import numpy.random
import random
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from collections import defaultdict

logger = logging.getLogger(__name__)


class DatasetSpliter:
    '''
    Receive a dataset object. Provided with some method to random divided the dataset.

    For Federated Learning: 
    1. Random Split
    2. Non-IID Split with params of dirichlet distribution. 
    '''
    def __init__(self) -> None:
        self.last_partition = None

    def _sample_random(self, dataset: Dataset, client_list: dict):
        client_ids = list(client_list)
        if not client_ids:
            raise ValueError("at least one client is required")
        indices = np.arange(len(dataset), dtype=np.int64)
        numpy.random.shuffle(indices)
        chunks = np.array_split(indices, len(client_ids))
        partition = defaultdict(list)
        for client_id, chunk in zip(client_ids, chunks):
            partition[client_id] = [int(index) for index in chunk]
        self.last_partition = partition
        return partition
    
    def _sample_dirichlet(self, dataset: Dataset, client_list: dict, alpha: int) -> defaultdict(list):
        client_ids = list(client_list)
        client_num = len(client_ids)
        if client_num == 0:
            raise ValueError("at least one client is required")
        if alpha <= 0:
            raise ValueError("Dirichlet alpha must be positive")
        per_class_list = defaultdict(list)
        
        #get each class index
        for ind, (_, label) in enumerate(dataset):
            if hasattr(label, 'item'):
                label = label.item()
            per_class_list[label].append(ind)
        
        #split the dataset(distribute each dataset sample to client by dirichlet probability distribution)
        per_client_list = defaultdict(list)
        for class_indices in per_class_list.values():
            random.shuffle(class_indices)
            probabilities = numpy.random.dirichlet(
                np.full(client_num, float(alpha), dtype=np.float64)
            )
            counts = numpy.random.multinomial(len(class_indices), probabilities)
            offset = 0
            for client_id, count in zip(client_ids, counts):
                next_offset = offset + int(count)
                per_client_list[client_id].extend(
                    class_indices[offset:next_offset]
                )
                offset = next_offset
            if offset != len(class_indices):
                raise RuntimeError("Dirichlet partition did not consume every sample")
        for client_id in client_ids:
            per_client_list[client_id]
        self.last_partition = per_client_list
        return per_client_list 
        
    def _build_dataloaders(
        self,
        dataset: Dataset,
        partition: dict,
        batch_size: int,
    ) -> dict:
        if batch_size <= 0:
            raise ValueError("batch size must be positive")
        return {
            client_id: DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                sampler=SubsetRandomSampler(indices),
                num_workers=4,
            )
            for client_id, indices in partition.items()
        }

    def dirichlet_split(self, dataset: Dataset, client_list: dict, batch_size: int = 32, alpha: float = 1) -> dict:
        #get each client samples
        split_list = self._sample_dirichlet(dataset = dataset, 
                                            client_list = client_list,
                                            alpha = alpha)
        return self._build_dataloaders(dataset, split_list, batch_size)
    
    def random_split(self, dataset: Dataset, client_list: dict, batch_size: int = 32) -> dict[DataLoader]:
        split_list = self._sample_random(dataset, client_list)
        return self._build_dataloaders(dataset, split_list, batch_size)
