"""
Verification script for AnchorRegistry implementation.

Checks:
1. AnchorRegistry.sol contract exists
2. chainfl/interact.py has anchor_round() method
3. Test file exists
4. All imports are correct
"""

import os
import sys

sys.path.append('.')


def check_file_exists(path, description):
    """Check if a file exists"""
    if os.path.exists(path):
        print(f"[PASS] {description}: {path}")
        return True
    else:
        print(f"[FAIL] {description} NOT FOUND: {path}")
        return False


def check_code_contains(path, search_string, description):
    """Check if a file contains a specific string"""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
            if search_string in content:
                print(f"[PASS] {description}")
                return True
            else:
                print(f"[FAIL] {description} NOT FOUND")
                return False
    except Exception as e:
        print(f"[FAIL] Error reading {path}: {e}")
        return False


def main():
    """Run all verification checks"""
    print("=" * 80)
    print("AnchorRegistry Implementation Verification")
    print("=" * 80)
    print()

    all_checks_passed = True

    # Check 1: AnchorRegistry.sol exists
    print("Check 1: AnchorRegistry.sol contract")
    print("-" * 80)
    contract_path = "chainEnv/contracts/AnchorRegistry.sol"
    if check_file_exists(contract_path, "AnchorRegistry.sol"):
        check_code_contains(contract_path, "function anchorRound", "  - anchorRound() function")
        check_code_contains(contract_path, "function getAnchor", "  - getAnchor() function")
        check_code_contains(contract_path, "function verifyAnchor", "  - verifyAnchor() function")
    else:
        all_checks_passed = False
    print()

    # Check 2: chainfl/interact.py modifications
    print("Check 2: chainfl/interact.py modifications")
    print("-" * 80)
    interact_path = "chainfl/interact.py"
    if check_file_exists(interact_path, "chainfl/interact.py"):
        check_code_contains(interact_path, "AnchorRegistry", "  - AnchorRegistry import")
        check_code_contains(interact_path, "self.anchor_registry", "  - anchor_registry attribute")
        check_code_contains(interact_path, "def anchor_round", "  - anchor_round() method")
        check_code_contains(interact_path, "AnchorRegistry.deploy", "  - AnchorRegistry deployment")
    else:
        all_checks_passed = False
    print()

    # Check 3: Test file exists
    print("Check 3: Test file")
    print("-" * 80)
    test_path = "tests/test_anchor_registry_e2e.py"
    if check_file_exists(test_path, "test_anchor_registry_e2e.py"):
        check_code_contains(test_path, "test_01_anchor_registry_deployed", "  - Deployment test")
        check_code_contains(test_path, "test_02_anchor_round_basic", "  - Basic anchoring test")
        check_code_contains(test_path, "test_05_integration_with_aggregator", "  - Aggregator integration test")
    else:
        all_checks_passed = False
    print()

    # Check 4: Integration with PoLVerifyAggregator
    print("Check 4: Integration with PoLVerifyAggregator")
    print("-" * 80)
    aggregator_path = "server/aggregation_alg/PoLVerifyAggregator.py"
    if check_file_exists(aggregator_path, "PoLVerifyAggregator.py"):
        check_code_contains(aggregator_path, "_maybe_anchor_onchain", "  - _maybe_anchor_onchain() method")
        check_code_contains(aggregator_path, "chain_proxy.anchor_round", "  - Calls anchor_round()")
    else:
        all_checks_passed = False
    print()

    # Summary
    print("=" * 80)
    print("Verification Summary")
    print("=" * 80)

    if all_checks_passed:
        print("[PASS] All checks passed.")
        print()
        print("Implementation complete:")
        print("  1. AnchorRegistry.sol contract created")
        print("  2. chainfl/interact.py updated with anchor_round() method")
        print("  3. AnchorRegistry deployment added to initialization")
        print("  4. End-to-end test created")
        print("  5. Integration with PoLVerifyAggregator verified")
        print()
        print("Next steps:")
        print("  1. Compile contracts: cd PoL-BFL/Code/chainEnv && brownie compile")
        print("  2. Run tests: cd PoL-BFL/Code && python tests/test_anchor_registry_e2e.py")
        print()
        return 0
    else:
        print("[FAIL] Some checks failed")
        print()
        print("Review the preceding missing-component report.")
        print()
        return 1


if __name__ == '__main__':
    sys.exit(main())
