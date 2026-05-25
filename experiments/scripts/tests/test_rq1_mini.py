"""
Minimal RQ1 Test

Quick test of RQ1 experiment with minimal configuration.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import numpy as np
import logging

from data_utils import load_dataset, partition_data_iid
from models import create_model
from metrics import compute_accuracy
from baselines import create_aggregator
from attacks.byzantine_attacks import create_attack

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_client(model, dataloader, device, epochs=1):
    """Train client model"""
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()


def main():
    """Run minimal RQ1 test"""
    logger.info("Starting Minimal RQ1 Test")
    
    device = torch.device('cpu')  # Use CPU for quick test
    
    # Load data
    logger.info("Loading data...")
    train_dataset = load_dataset('MNIST', train=True)
    test_dataset = load_dataset('MNIST', train=False)
    
    # Partition data
    client_datasets = partition_data_iid(train_dataset, num_clients=5)
    train_loaders = [
        torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True)
        for ds in client_datasets
    ]
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Create global model
    global_model = create_model('SimpleCNN', num_classes=10, input_channels=1).to(device)
    
    # Test 1: Vanilla FL (no attack)
    logger.info("\n=== Test 1: Vanilla FL (No Attack) ===")
    aggregator = create_aggregator('Vanilla_FL')
    
    for round_num in range(3):  # Only 3 rounds for quick test
        logger.info(f"Round {round_num + 1}/3")
        
        # Train clients
        client_models = []
        for i in range(3):  # Only 3 clients per round
            client_model = create_model('SimpleCNN', num_classes=10, input_channels=1).to(device)
            client_model.load_state_dict(global_model.state_dict())
            train_client(client_model, train_loaders[i], device, epochs=1)
            client_models.append(client_model.state_dict())
        
        # Aggregate
        aggregated_state = aggregator.aggregate(client_models)
        global_model.load_state_dict(aggregated_state)
        
        # Evaluate
        test_acc = compute_accuracy(global_model, test_loader, device)
        logger.info(f"  Test Accuracy: {test_acc:.4f}")
    
    # Test 2: Vanilla FL with Byzantine attack
    logger.info("\n=== Test 2: Vanilla FL with Byzantine Attack ===")
    global_model = create_model('SimpleCNN', num_classes=10, input_channels=1).to(device)
    aggregator = create_aggregator('Vanilla_FL')
    
    for round_num in range(3):
        logger.info(f"Round {round_num + 1}/3")
        
        # Train clients (1 malicious)
        client_models = []
        for i in range(3):
            client_model = create_model('SimpleCNN', num_classes=10, input_channels=1).to(device)
            client_model.load_state_dict(global_model.state_dict())
            
            if i == 0:  # First client is malicious
                attack = create_attack('random_noise', noise_scale=1.0)
                attacked_state = attack.apply(client_model.state_dict())
                client_model.load_state_dict(attacked_state)
            else:
                train_client(client_model, train_loaders[i], device, epochs=1)
            
            client_models.append(client_model.state_dict())
        
        # Aggregate
        aggregated_state = aggregator.aggregate(client_models)
        global_model.load_state_dict(aggregated_state)
        
        # Evaluate
        test_acc = compute_accuracy(global_model, test_loader, device)
        logger.info(f"  Test Accuracy: {test_acc:.4f}")
    
    # Test 3: Krum with Byzantine attack
    logger.info("\n=== Test 3: Krum with Byzantine Attack ===")
    global_model = create_model('SimpleCNN', num_classes=10, input_channels=1).to(device)
    aggregator = create_aggregator('Krum', num_byzantine=1)
    
    for round_num in range(3):
        logger.info(f"Round {round_num + 1}/3")
        
        # Train clients (1 malicious)
        client_models = []
        for i in range(3):
            client_model = create_model('SimpleCNN', num_classes=10, input_channels=1).to(device)
            client_model.load_state_dict(global_model.state_dict())
            
            if i == 0:  # First client is malicious
                attack = create_attack('random_noise', noise_scale=1.0)
                attacked_state = attack.apply(client_model.state_dict())
                client_model.load_state_dict(attacked_state)
            else:
                train_client(client_model, train_loaders[i], device, epochs=1)
            
            client_models.append(client_model.state_dict())
        
        # Aggregate
        aggregated_state = aggregator.aggregate(client_models)
        global_model.load_state_dict(aggregated_state)
        
        # Evaluate
        test_acc = compute_accuracy(global_model, test_loader, device)
        logger.info(f"  Test Accuracy: {test_acc:.4f}")
    
    logger.info("\n=== Minimal RQ1 Test Completed ===")


if __name__ == '__main__':
    main()

