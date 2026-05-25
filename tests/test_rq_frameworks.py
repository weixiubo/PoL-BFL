#!/usr/bin/env python
"""
Verify that all RQ frameworks can be imported and initialized
"""

import sys
import os
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'experiments' / 'scripts' / 'utils'))
sys.path.insert(0, str(Path(__file__).parent / 'experiments'))

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_rq1_framework():
    """Test RQ1 framework"""
    try:
        from experiments.scripts.runners.run_rq1_security import SecurityExperiment, RQ1_SIMPLE_CONFIG
        logger.info("✓ RQ1 framework imported successfully")
        
        # Try to initialize
        exp = SecurityExperiment(RQ1_SIMPLE_CONFIG)
        logger.info("✓ RQ1 SecurityExperiment initialized")
        return True
    except Exception as e:
        logger.error(f"✗ RQ1 framework failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_rq2_framework():
    """Test RQ2 framework"""
    try:
        from experiments.scripts.runners.run_rq2_ablation import AblationStudyExperiment, RQ2_ABLATION_CONFIG
        logger.info("✓ RQ2 framework imported successfully")
        
        # Try to initialize
        exp = AblationStudyExperiment(RQ2_ABLATION_CONFIG)
        logger.info("✓ RQ2 AblationStudyExperiment initialized")
        return True
    except Exception as e:
        logger.error(f"✗ RQ2 framework failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_rq3_framework():
    """Test RQ3 framework"""
    try:
        from experiments.scripts.runners.run_rq3_overhead import OverheadExperiment, RQ3_CONFIG
        logger.info("✓ RQ3 framework imported successfully")
        
        # Try to initialize
        exp = OverheadExperiment(RQ3_CONFIG)
        logger.info("✓ RQ3 OverheadExperiment initialized")
        return True
    except Exception as e:
        logger.error(f"✗ RQ3 framework failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_rq4_framework():
    """Test RQ4 framework"""
    try:
        from experiments.scripts.runners.run_rq4_incentive import IncentiveExperiment, RQ4_CONFIG
        logger.info("✓ RQ4 framework imported successfully")
        
        # Try to initialize
        exp = IncentiveExperiment(RQ4_CONFIG)
        logger.info("✓ RQ4 IncentiveExperiment initialized")
        return True
    except Exception as e:
        logger.error(f"✗ RQ4 framework failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all framework tests"""
    logger.info("="*70)
    logger.info("RQ Framework Verification")
    logger.info("="*70)
    
    tests = [
        ("RQ1 Security Evaluation", test_rq1_framework),
        ("RQ2 Ablation Study", test_rq2_framework),
        ("RQ3 Overhead Analysis", test_rq3_framework),
        ("RQ4 Incentive Mechanism", test_rq4_framework),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\nTesting: {test_name}")
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("Framework Verification Summary")
    logger.info("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status}: {test_name}")
    
    logger.info(f"\nTotal: {passed}/{total} frameworks verified")
    logger.info("="*70)
    
    return passed == total

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

