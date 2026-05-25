#!/usr/bin/env python3
"""
快速环境检查脚本
验证PoL-BFL实验框架的完整性和可用性
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def check_environment():
    """检查基础环境"""
    print("\n" + "="*70)
    print("🔍 ENVIRONMENT CHECK")
    print("="*70)
    
    checks = {}
    
    # Python version
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    checks['python_version'] = py_version
    print(f"✓ Python: {py_version}")
    
    # PyTorch
    checks['pytorch_version'] = torch.__version__
    print(f"✓ PyTorch: {torch.__version__}")
    
    # CUDA
    cuda_available = torch.cuda.is_available()
    checks['cuda_available'] = cuda_available
    if cuda_available:
        gpu_count = torch.cuda.device_count()
        checks['gpu_count'] = gpu_count
        print(f"✓ CUDA: Available ({gpu_count} GPUs)")
        for i in range(gpu_count):
            print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("⚠ CUDA: Not available")
    
    # NumPy
    checks['numpy_version'] = np.__version__
    print(f"✓ NumPy: {np.__version__}")
    
    return checks

def check_datasets():
    """检查数据集"""
    print("\n" + "="*70)
    print("📊 DATASET CHECK")
    print("="*70)
    
    checks = {}
    
    try:
        from dataset.DatasetFactory import DatasetFactory
        factory = DatasetFactory()
        
        for dataset_name in ['MNIST', 'CIFAR10', 'CIFAR100']:
            try:
                train_ds = factory.get_dataset(dataset_name, train=True)
                test_ds = factory.get_dataset(dataset_name, train=False)
                checks[dataset_name] = {
                    'train_size': len(train_ds),
                    'test_size': len(test_ds),
                    'status': 'OK'
                }
                print(f"✓ {dataset_name}: train={len(train_ds)}, test={len(test_ds)}")
            except Exception as e:
                checks[dataset_name] = {'status': 'FAILED', 'error': str(e)}
                print(f"✗ {dataset_name}: {e}")
    except Exception as e:
        print(f"✗ Dataset factory error: {e}")
        checks['error'] = str(e)
    
    return checks

def check_models():
    """检查模型"""
    print("\n" + "="*70)
    print("🧠 MODEL CHECK")
    print("="*70)
    
    checks = {}
    
    try:
        from model.ModelFactory import ModelFactory
        factory = ModelFactory()
        
        models = [
            ('SimpleCNN', 10),
            ('ResNet18', 10),
        ]
        
        for model_name, num_classes in models:
            try:
                model = factory.get_model(model_name, num_classes)
                param_count = sum(p.numel() for p in model.parameters())
                checks[model_name] = {
                    'num_classes': num_classes,
                    'param_count': param_count,
                    'status': 'OK'
                }
                print(f"✓ {model_name}: {param_count:,} parameters")
            except Exception as e:
                checks[model_name] = {'status': 'FAILED', 'error': str(e)}
                print(f"✗ {model_name}: {e}")
    except Exception as e:
        print(f"✗ Model factory error: {e}")
        checks['error'] = str(e)
    
    return checks

def check_pol_components():
    """检查PoL组件"""
    print("\n" + "="*70)
    print("🔐 PoL COMPONENTS CHECK")
    print("="*70)
    
    checks = {}
    
    components = [
        ('PoLManager', 'client.pol.PoLManager'),
        ('PoLVerifier', 'server.pol.PoLVerifier'),
        ('PoLTrainer', 'client.trainer.PoLTrainer'),
        ('MerkleTree', 'client.pol.MerkleTree'),
    ]
    
    for comp_name, module_path in components:
        try:
            parts = module_path.split('.')
            module = __import__(module_path, fromlist=[parts[-1]])
            cls = getattr(module, comp_name)
            checks[comp_name] = {'status': 'OK'}
            print(f"✓ {comp_name}: Loaded")
        except Exception as e:
            checks[comp_name] = {'status': 'FAILED', 'error': str(e)}
            print(f"✗ {comp_name}: {e}")
    
    return checks

def check_experiment_scripts():
    """检查实验脚本"""
    print("\n" + "="*70)
    print("📝 EXPERIMENT SCRIPTS CHECK")
    print("="*70)
    
    checks = {}
    scripts_dir = PROJECT_ROOT / 'experiments' / 'scripts' / 'runners'
    
    scripts = [
        'run_rq1_security.py',
        'run_rq2_ablation.py',
        'run_rq3_overhead.py',
        'run_rq4_incentive.py',
    ]
    
    for script in scripts:
        script_path = scripts_dir / script
        if script_path.exists():
            checks[script] = {'status': 'OK', 'path': str(script_path)}
            print(f"✓ {script}: Found")
        else:
            checks[script] = {'status': 'MISSING'}
            print(f"✗ {script}: Not found")
    
    return checks

def check_config():
    """检查配置"""
    print("\n" + "="*70)
    print("⚙️  CONFIGURATION CHECK")
    print("="*70)
    
    checks = {}
    
    try:
        from config.pol_config import POL_CONFIG, EXPERIMENT_CONFIG
        
        # PoL config
        pol_keys = ['enable', 'save_freq', 'verification_rate', 'delta']
        pol_check = {}
        for key in pol_keys:
            if key in POL_CONFIG:
                pol_check[key] = POL_CONFIG[key]
        checks['POL_CONFIG'] = pol_check
        print(f"✓ POL_CONFIG: {json.dumps(pol_check, indent=2)}")
        
        # Experiment config
        exp_keys = ['dataset', 'model', 'num_clients', 'num_rounds']
        exp_check = {}
        for key in exp_keys:
            if key in EXPERIMENT_CONFIG:
                exp_check[key] = EXPERIMENT_CONFIG[key]
        checks['EXPERIMENT_CONFIG'] = exp_check
        print(f"✓ EXPERIMENT_CONFIG: {json.dumps(exp_check, indent=2)}")
    except Exception as e:
        print(f"✗ Config error: {e}")
        checks['error'] = str(e)
    
    return checks

def main():
    """主函数"""
    print("\n" + "="*70)
    print("🚀 PoL-BFL ENVIRONMENT & FRAMEWORK CHECK")
    print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    all_checks = {}
    
    # Run all checks
    all_checks['environment'] = check_environment()
    all_checks['datasets'] = check_datasets()
    all_checks['models'] = check_models()
    all_checks['pol_components'] = check_pol_components()
    all_checks['experiment_scripts'] = check_experiment_scripts()
    all_checks['config'] = check_config()
    
    # Summary
    print("\n" + "="*70)
    print("📋 SUMMARY")
    print("="*70)
    
    # Count status
    total_checks = 0
    passed_checks = 0
    
    for category, results in all_checks.items():
        if isinstance(results, dict):
            for key, value in results.items():
                if isinstance(value, dict) and 'status' in value:
                    total_checks += 1
                    if value['status'] == 'OK':
                        passed_checks += 1
    
    print(f"✓ Passed: {passed_checks}/{total_checks}")
    
    if passed_checks == total_checks:
        print("\n✅ All checks passed! Framework is ready for experiments.")
        return 0
    else:
        print(f"\n⚠️  {total_checks - passed_checks} checks failed. Please review above.")
        return 1

if __name__ == '__main__':
    sys.exit(main())

