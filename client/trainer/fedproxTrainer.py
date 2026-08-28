"""FedProx trainer implementation."""
import copy
import logging

import torch
from torch import nn
import torch.utils.data
from client.base.baseTrainer import BaseTrainer

logger = logging.getLogger(__name__)

class fedproxTrainer(BaseTrainer):
    def __init__(self, model: nn.Module, dataloader: torch.utils.data.DataLoader, criterion, args: dict, mu:int =0.5):
        # BaseTrainer provides the data and optimizer lifecycle.
        super().__init__(model, dataloader, criterion, args)
        self.mu = mu
        self.criterion = torch.nn.CrossEntropyLoss()

    def _train_epoch(self, epoch):
        #FedProx Algorithm

        model = self.model
        args = self.args
        device = args["device"]

        model.to(device)
        model.train()

        previous_model = copy.deepcopy(model.state_dict())
        optimizer = self.optimizer
        batch_loss = []
        for _, batch in enumerate(self.dataloader):
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                x, labels, _ = batch
            else:
                x, labels = batch
            x, labels = x.to(device), labels.to(device)
            self.optimizer.zero_grad()
            log_probs = model(x)
            loss = self.criterion(log_probs, labels)  # pylint: disable=E1102
            fed_prox_reg = 0.0
            for name, param in model.named_parameters():
                fed_prox_reg += ((self.mu / 2) * \
                                 torch.norm((param - previous_model[name].data.to(device))) ** 2)
            loss += fed_prox_reg

            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())
        if len(batch_loss) == 0:
            epoch_loss = 0.0
        else:
            epoch_loss = (sum(batch_loss) / len(batch_loss))

        ret = dict()
        ret['loss'] = epoch_loss
        return ret
