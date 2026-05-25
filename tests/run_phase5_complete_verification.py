#!/usr/bin/env python3
"""
Phase 5: Complete Verification Script
Runs all Phase 5 tests and generates final report.

This script:
1. Runs comprehensive integration tests
2. Runs performance benchmarks
3. Generates verification report
4. Verifies academic compliance
5. Generates final summary

NO HARDCODING - All results based on actual test execution.
"""

import subprocess
import sys
import logging
import json
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Phase5Verifier:
    """Runs complete Phase 5 verification"""
    
    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'phase': 5,
            'tests': {},
            'summary': {}
        }
        self.test_dir = Path('tests')
    
    def run_test(self, test_name, test_file):
        """Run a single test file"""
        logger.info(f"\n{'='*70}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*70}")
        
        try:
            # Run pytest on the test file
            result = subprocess.run(
                [sys.executable, '-m', 'pytest', str(self.test_dir / test_file), '-v', '-s'],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            # Check result
            success = result.returncode == 0
            
            logger.info(f"\nTest Output:")
            logger.info(result.stdout)
            
            if result.stderr:
                logger.warning(f"\nStderr:")
                logger.warning(result.stderr)
            
            # Store result
            self.results['tests'][test_name] = {
                'status': 'PASSED' if success else 'FAILED',
                'return_code': result.returncode,
                'output': result.stdout,
                'error': result.stderr
            }
            
            logger.info(f"\n✅ {test_name}: {'PASSED' if success else 'FAILED'}")
            return success
            
        except subprocess.TimeoutExpired:
            logger.error(f"❌ {test_name}: TIMEOUT")
            self.results['tests'][test_name] = {
                'status': 'TIMEOUT',
                'return_code': -1
            }
            return False
        except Exception as e:
            logger.error(f"❌ {test_name}: ERROR - {e}")
            self.results['tests'][test_name] = {
                'status': 'ERROR',
                'error': str(e)
            }
            return False
    
    def run_all_tests(self):
        """Run all Phase 5 tests"""
        logger.info("\n" + "="*70)
        logger.info("PHASE 5: COMPLETE VERIFICATION")
        logger.info("="*70)
        
        tests = [
            ("Comprehensive Integration Test", "test_phase5_comprehensive_integration.py"),
            ("Performance Benchmarks", "test_phase5_performance_benchmarks.py"),
            ("Verification Report", "test_phase5_verification_report.py"),
        ]
        
        results = []
        for test_name, test_file in tests:
            success = self.run_test(test_name, test_file)
            results.append((test_name, success))
        
        return results
    
    def run_existing_tests(self):
        """Run existing test suite to verify no regression"""
        logger.info("\n" + "="*70)
        logger.info("RUNNING EXISTING TEST SUITE (Regression Check)")
        logger.info("="*70)
        
        existing_tests = [
            ("PoL Manager Tests", "test_pol_manager.py"),
            ("PoL Verifier Tests", "test_pol_verifier.py"),
            ("End-to-End Tests", "test_end_to_end.py"),
        ]
        
        results = []
        for test_name, test_file in existing_tests:
            success = self.run_test(test_name, test_file)
            results.append((test_name, success))
        
        return results
    
    def generate_summary(self, phase5_results, existing_results):
        """Generate final summary"""
        logger.info("\n" + "="*70)
        logger.info("FINAL SUMMARY")
        logger.info("="*70)
        
        # Count results
        phase5_passed = sum(1 for _, success in phase5_results if success)
        phase5_total = len(phase5_results)
        
        existing_passed = sum(1 for _, success in existing_results if success)
        existing_total = len(existing_results)
        
        total_passed = phase5_passed + existing_passed
        total_tests = phase5_total + existing_total
        
        # Print summary
        logger.info(f"\nPhase 5 Tests: {phase5_passed}/{phase5_total} passed")
        for test_name, success in phase5_results:
            status = "✅ PASS" if success else "❌ FAIL"
            logger.info(f"  {status}: {test_name}")
        
        logger.info(f"\nExisting Tests (Regression Check): {existing_passed}/{existing_total} passed")
        for test_name, success in existing_results:
            status = "✅ PASS" if success else "❌ FAIL"
            logger.info(f"  {status}: {test_name}")
        
        logger.info(f"\nTotal: {total_passed}/{total_tests} tests passed")
        
        # Store summary
        self.results['summary'] = {
            'phase5_tests': {
                'passed': phase5_passed,
                'total': phase5_total,
                'percentage': (phase5_passed / phase5_total * 100) if phase5_total > 0 else 0
            },
            'existing_tests': {
                'passed': existing_passed,
                'total': existing_total,
                'percentage': (existing_passed / existing_total * 100) if existing_total > 0 else 0
            },
            'total': {
                'passed': total_passed,
                'total': total_tests,
                'percentage': (total_passed / total_tests * 100) if total_tests > 0 else 0
            },
            'status': 'COMPLETE' if total_passed == total_tests else 'INCOMPLETE',
            'ready_for_deployment': total_passed == total_tests
        }
        
        # Print final status
        logger.info("\n" + "="*70)
        logger.info("DEPLOYMENT READINESS")
        logger.info("="*70)
        
        if self.results['summary']['ready_for_deployment']:
            logger.info("✅ READY FOR DEPLOYMENT")
            logger.info("All tests passed. System is production-ready.")
        else:
            logger.info("❌ NOT READY FOR DEPLOYMENT")
            logger.info("Some tests failed. Please review and fix issues.")
        
        return self.results['summary']['ready_for_deployment']
    
    def save_results(self):
        """Save results to file"""
        output_file = Path('PHASE5_COMPLETE_VERIFICATION_RESULTS.json')
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"\n✅ Results saved to: {output_file}")
    
    def run_complete_verification(self):
        """Run complete verification"""
        try:
            # Run Phase 5 tests
            phase5_results = self.run_all_tests()
            
            # Run existing tests for regression check
            existing_results = self.run_existing_tests()
            
            # Generate summary
            ready = self.generate_summary(phase5_results, existing_results)
            
            # Save results
            self.save_results()
            
            return ready
            
        except Exception as e:
            logger.error(f"Error during verification: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main entry point"""
    verifier = Phase5Verifier()
    success = verifier.run_complete_verification()
    
    exit(0 if success else 1)


if __name__ == '__main__':
    main()

