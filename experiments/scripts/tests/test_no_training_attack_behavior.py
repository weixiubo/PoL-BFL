"""
Test script to verify NoTrainingAttack client behavior
Confirms whether NoTrainingAttack clients:
1. Skip training completely
2. Generate no checkpoints
3. Return empty/None response to challenges
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import logging

from client.PoLClient import PoLClient
from client.trainer.PoLTrainer import PoLTrainer
from experiments.attacks.free_riding_attacks import NoTrainingAttack
from experiments.scripts.utils.models import create_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_dummy_dataloader(num_samples=100, input_size=28*28, num_classes=10):
    """Create a dummy dataloader for testing"""
    X = torch.randn(num_samples, 1, 28, 28)
    y = torch.randint(0, num_classes, (num_samples,))
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=32, shuffle=True)

def test_no_training_attack():
    """Test NoTrainingAttack client behavior"""
    logger.info("="*80)
    logger.info("Testing NoTrainingAttack Client Behavior")
    logger.info("="*80)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model('SimpleCNN', num_classes=10, input_channels=1).to(device)
    dataloader = create_dummy_dataloader()
    
    # Create PoLTrainer with PoL enabled
    trainer_args = {
        'device': device,
        'lr': 0.01,
        'momentum': 0.9,
        'weight_decay': 0.0001,
        'optimizer': 'SGD',
        'enable_pol': True,
        'pol_save_freq': 5,
        'pol_save_dir': '/tmp/test_no_training_attack',
        'pol_compress': False,
    }
    
    trainer = PoLTrainer(
        model=model,
        dataloader=dataloader,
        criterion=nn.CrossEntropyLoss(),
        args=trainer_args
    )
    
    # Create PoLClient
    client = PoLClient(
        client_id='test_malicious_client',
        dataloader=dataloader,
        model=model,
        trainer=trainer,
        args=trainer_args
    )
    
    # Test 1: Check if NoTrainingAttack.should_train() returns False
    attack = NoTrainingAttack()
    logger.info(f"\nTest 1: NoTrainingAttack.should_train() = {attack.should_train()}")
    assert attack.should_train() == False, "NoTrainingAttack should return False for should_train()"
    
    # Test 2: Simulate the experiment behavior (skip training if should_train() is False)
    logger.info("\nTest 2: Simulating experiment behavior...")
    if attack.should_train():
        logger.info("  Training client...")
        client.train(total_epoch=2, dataset=None)
    else:
        logger.info("  Skipping training (as per NoTrainingAttack)")
    
    # Test 3: Check if client has PoL commitment
    pol_commitment = client.get_pol_commitment()
    logger.info(f"\nTest 3: PoL commitment exists: {pol_commitment is not None}")
    if pol_commitment:
        logger.info(f"  Commitment: {pol_commitment.get('commitment', 'N/A')[:16]}...")
        logger.info(f"  Num checkpoints: {pol_commitment.get('num_checkpoints', 0)}")
    
    # Test 4: Try to respond to challenge
    logger.info("\nTest 4: Responding to challenge...")
    challenge = {
        'checkpoint_steps': [5, 10, 15],
        'include_data_indices': True
    }
    response = client.respond_to_challenge(challenge)
    logger.info(f"  Response is None: {response is None}")
    if response:
        checkpoints = response.get('checkpoints', [])
        logger.info(f"  Number of checkpoints in response: {len(checkpoints)}")
    
    # Test 5: Check pol_manager state
    logger.info("\nTest 5: Checking PoL manager state...")
    if hasattr(trainer, 'pol_manager') and trainer.pol_manager:
        num_ckpts = trainer.pol_manager.get_checkpoint_count()
        logger.info(f"  PoL manager checkpoint count: {num_ckpts}")
        logger.info(f"  Batch counter: {trainer.batch_counter}")
    else:
        logger.info("  PoL manager not available")
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    logger.info(f"1. should_train() returns False: ✓")
    logger.info(f"2. Training was skipped: ✓")
    logger.info(f"3. PoL commitment exists: {pol_commitment is not None}")
    logger.info(f"4. Challenge response is None: {response is None}")
    logger.info(f"5. Checkpoint count: {num_ckpts if hasattr(trainer, 'pol_manager') and trainer.pol_manager else 'N/A'}")
    
    # Expected behavior
    logger.info("\nEXPECTED BEHAVIOR:")
    logger.info("- NoTrainingAttack client should NOT train")
    logger.info("- Should have NO checkpoints")
    logger.info("- Should return None or empty response to challenge")
    logger.info("- PoL verifier should detect this as malicious (return False)")
    
    logger.info("\n" + "="*80)

if __name__ == '__main__':
    test_no_training_attack()

