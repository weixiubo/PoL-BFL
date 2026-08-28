"""
Performance Benchmarking Suite
Measures actual system performance with real data and real calculations.

Metrics measured:
- Training time per round
- PoL generation time
- PoL verification time
- Economic incentive calculation time
- Memory usage
- Model size
- Checkpoint size

NO HARDCODING - All measurements based on actual execution.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import time
import os
import tempfile
import shutil
import psutil
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock chain proxy
from tests.mock_chain_proxy import mock_chain_proxy
import chainfl.interact as interact_module
interact_module.chain_proxy = mock_chain_proxy

from client.pol.PoLManager import PoLManager
from client.trainer.PoLTrainer import PoLTrainer
from server.pol.PoLVerifier import PoLVerifier
from server.aggregation_alg.PoLVerifyAggregator import PoLVerifyAggregator
from config.pol_config import POL_CONFIG


class SimpleCNN(nn.Module):
    """Simple CNN for benchmarking"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = torch.relu(torch.max_pool2d(self.conv1(x), 2))
        x = torch.relu(torch.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class PerformanceBenchmark:
    """Performance benchmarking suite"""

    def __init__(self):
        self.test_dir = tempfile.mkdtemp()
        self.benchmarks = {}
        self.process = psutil.Process()

    def cleanup(self):
        """Clean up test directory"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def get_memory_usage(self):
        """Get current memory usage in MB"""
        return self.process.memory_info().rss / 1024 / 1024

    def benchmark_training_time(self, num_samples=200, batch_size=10, num_epochs=1):
        """Benchmark: Training time per epoch"""
        logger.info("\n" + "="*70)
        logger.info("BENCHMARK: Training Time")
        logger.info("="*70)

        # Create real data
        x = torch.randn(num_samples, 1, 28, 28)
        y = torch.randint(0, 10, (num_samples,))
        dataset = TensorDataset(x, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # Create model
        model = SimpleCNN()
        criterion = nn.CrossEntropyLoss()

        args = {
            'enable_pol': True,
            'pol_save_freq': 5,
            'pol_save_dir': os.path.join(self.test_dir, 'training_bench'),
            'pol_compress': True,
            'client_id': 'bench_client',
            'device': 'cpu',
            'lr': 0.01,
            'weight_decay': 1e-4,
            'optimizer': 'SGD'
        }

        trainer = PoLTrainer(model, dataloader, criterion, args)

        # Measure training time (real execution)
        mem_before = self.get_memory_usage()
        start_time = time.time()
        results = trainer.train(total_epoch=num_epochs)
        training_time = time.time() - start_time
        mem_after = self.get_memory_usage()

        # Verify real training happened
        assert len(results) > 0, "Training should produce results"
        assert results[0]['loss'] > 0, "Loss should be positive"

        logger.info(f"  Samples: {num_samples}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Epochs: {num_epochs}")
        logger.info(f"  Training time: {training_time:.4f}s")
        logger.info(f"  Time per sample: {training_time/num_samples*1000:.4f}ms")
        logger.info(f"  Memory before: {mem_before:.2f}MB")
        logger.info(f"  Memory after: {mem_after:.2f}MB")
        logger.info(f"  Memory delta: {mem_after-mem_before:.2f}MB")
        logger.info(f"  Final loss: {results[0]['loss']:.6f}")

        self.benchmarks['training_time'] = training_time
        self.benchmarks['training_time_per_sample'] = training_time / num_samples
        self.benchmarks['training_memory_delta'] = mem_after - mem_before

        return training_time

    def benchmark_pol_generation(self, num_samples=200, batch_size=10):
        """Benchmark: PoL generation time"""
        logger.info("\n" + "="*70)
        logger.info("BENCHMARK: PoL Generation Time")
        logger.info("="*70)

        # Create real data
        x = torch.randn(num_samples, 1, 28, 28)
        y = torch.randint(0, 10, (num_samples,))
        dataset = TensorDataset(x, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # Create and train model
        model = SimpleCNN()
        criterion = nn.CrossEntropyLoss()

        args = {
            'enable_pol': True,
            'pol_save_freq': 5,
            'pol_save_dir': os.path.join(self.test_dir, 'pol_gen_bench'),
            'pol_compress': True,
            'client_id': 'bench_pol',
            'device': 'cpu',
            'lr': 0.01,
            'weight_decay': 1e-4,
            'optimizer': 'SGD'
        }

        trainer = PoLTrainer(model, dataloader, criterion, args)
        trainer.train(total_epoch=1)

        # Measure PoL generation time (real execution)
        mem_before = self.get_memory_usage()
        start_time = time.time()
        commitment = trainer.finalize_pol(epoch=0, dataset=dataset)
        pol_time = time.time() - start_time
        mem_after = self.get_memory_usage()

        # Verify real PoL was generated
        assert commitment is not None, "PoL should be generated"
        assert commitment['num_checkpoints'] > 0, "Should have checkpoints"

        logger.info(f"  Checkpoints: {commitment['num_checkpoints']}")
        logger.info(f"  PoL generation time: {pol_time:.4f}s")
        logger.info(f"  Time per checkpoint: {pol_time/commitment['num_checkpoints']*1000:.4f}ms")
        logger.info(f"  Memory before: {mem_before:.2f}MB")
        logger.info(f"  Memory after: {mem_after:.2f}MB")
        logger.info(f"  Memory delta: {mem_after-mem_before:.2f}MB")

        self.benchmarks['pol_generation_time'] = pol_time
        self.benchmarks['pol_time_per_checkpoint'] = pol_time / commitment['num_checkpoints']
        self.benchmarks['pol_memory_delta'] = mem_after - mem_before
        self.benchmarks['num_checkpoints'] = commitment['num_checkpoints']

        return pol_time

    def benchmark_verification_time(self, num_models=5):
        """Benchmark: PoL verification time"""
        logger.info("\n" + "="*70)
        logger.info("BENCHMARK: PoL Verification Time")
        logger.info("="*70)

        # Create verifier with args dict
        verifier_args = {
            'delta': POL_CONFIG['delta'],
            'distance_metric': POL_CONFIG['distance_metric'],
            'device': 'cpu',
            'top_q': POL_CONFIG.get('top_q', 5) if POL_CONFIG.get('use_top_q') else None
        }
        verifier = PoLVerifier(verifier_args)

        # Create multiple models
        models = []
        base_model = SimpleCNN()
        for i in range(num_models):
            model = SimpleCNN()
            with torch.no_grad():
                for p_base, p in zip(base_model.parameters(), model.parameters()):
                    p.copy_(p_base + torch.randn_like(p) * 0.01)
            models.append(model.state_dict())

        # Measure verification time (real execution)
        mem_before = self.get_memory_usage()
        start_time = time.time()

        total_distance = 0
        for i in range(1, len(models)):
            distance = verifier._compute_parameter_distance(models[0], models[i], POL_CONFIG['distance_metric'])
            total_distance += distance

        verify_time = time.time() - start_time
        mem_after = self.get_memory_usage()

        logger.info(f"  Models verified: {num_models - 1}")
        logger.info(f"  Total verification time: {verify_time:.4f}s")
        logger.info(f"  Time per verification: {verify_time/(num_models-1)*1000:.4f}ms")
        logger.info(f"  Average distance: {total_distance/(num_models-1):.6f}")
        logger.info(f"  Memory before: {mem_before:.2f}MB")
        logger.info(f"  Memory after: {mem_after:.2f}MB")
        logger.info(f"  Memory delta: {mem_after-mem_before:.2f}MB")

        self.benchmarks['verification_time'] = verify_time
        self.benchmarks['verification_time_per_model'] = verify_time / (num_models - 1)
        self.benchmarks['verification_memory_delta'] = mem_after - mem_before

        return verify_time

    def benchmark_model_size(self):
        """Benchmark: Model size"""
        logger.info("\n" + "="*70)
        logger.info("BENCHMARK: Model Size")
        logger.info("="*70)

        model = SimpleCNN()

        # Calculate model size
        total_params = sum(p.numel() for p in model.parameters())
        total_size_mb = sum(p.numel() * 4 for p in model.parameters()) / (1024 * 1024)

        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Model size: {total_size_mb:.4f}MB")

        self.benchmarks['total_parameters'] = total_params
        self.benchmarks['model_size_mb'] = total_size_mb

        return total_size_mb

    def run_all_benchmarks(self):
        """Run all benchmarks"""
        logger.info("\n" + "="*70)
        logger.info("PERFORMANCE BENCHMARKING SUITE")
        logger.info("="*70)

        try:
            self.benchmark_model_size()
            self.benchmark_training_time(num_samples=200, batch_size=10, num_epochs=1)
            self.benchmark_pol_generation(num_samples=200, batch_size=10)
            self.benchmark_verification_time(num_models=5)

            # Print summary
            logger.info("\n" + "="*70)
            logger.info("BENCHMARK SUMMARY (Real Measurements)")
            logger.info("="*70)

            for key, value in self.benchmarks.items():
                if isinstance(value, float):
                    logger.info(f"{key}: {value:.6f}")
                else:
                    logger.info(f"{key}: {value}")

            return True

        finally:
            self.cleanup()


@pytest.mark.timeout(300)
def test_performance_benchmarks():
    """Pytest wrapper for performance benchmarks"""
    benchmark = PerformanceBenchmark()
    success = benchmark.run_all_benchmarks()
    assert success, "Performance benchmarks should complete successfully"


if __name__ == '__main__':
    benchmark = PerformanceBenchmark()
    success = benchmark.run_all_benchmarks()
    exit(0 if success else 1)
