#!/usr/bin/env python3
"""
Test runner for BTC Trading System

This script runs all tests with proper configuration and reporting.
"""
import sys
import os
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, description):
    """Run a command and print status"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"\n❌ {description} FAILED")
        return False
    else:
        print(f"\n✅ {description} PASSED")
        return True


def install_test_dependencies():
    """Install test dependencies"""
    print("Installing test dependencies...")
    cmd = [sys.executable, "-m", "pip", "install", "-r", "tests/requirements.txt"]
    return run_command(cmd, "Install test dependencies")


def run_unit_tests():
    """Run unit tests"""
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/unit",
        "-v",
        "-m", "unit",
        "--tb=short"
    ]
    return run_command(cmd, "Unit tests")


def run_integration_tests():
    """Run integration tests"""
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/integration",
        "-v",
        "-m", "integration",
        "--tb=short"
    ]
    return run_command(cmd, "Integration tests")


def run_e2e_tests():
    """Run end-to-end tests"""
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/e2e",
        "-v",
        "-m", "e2e",
        "--tb=short"
    ]
    return run_command(cmd, "End-to-end tests")


def run_all_tests():
    """Run all tests with coverage"""
    cmd = [
        sys.executable, "-m", "pytest",
        "tests",
        "-v",
        "--cov=src",
        "--cov-report=term-missing",
        "--cov-report=html",
        "--tb=short"
    ]
    return run_command(cmd, "All tests with coverage")


def run_specific_test(test_path):
    """Run a specific test file or directory"""
    cmd = [
        sys.executable, "-m", "pytest",
        test_path,
        "-v",
        "--tb=short"
    ]
    return run_command(cmd, f"Tests in {test_path}")


def check_code_quality():
    """Run code quality checks (if tools are installed)"""
    quality_checks = []
    
    # Check if flake8 is available
    try:
        subprocess.run(["flake8", "--version"], capture_output=True, check=True)
        quality_checks.append(([
            "flake8", "src", "--max-line-length=120", "--exclude=__pycache__"
        ], "Flake8 linting"))
    except:
        print("⚠️  Flake8 not installed, skipping linting")
    
    # Check if black is available
    try:
        subprocess.run(["black", "--version"], capture_output=True, check=True)
        quality_checks.append(([
            "black", "src", "--check", "--diff"
        ], "Black formatting check"))
    except:
        print("⚠️  Black not installed, skipping formatting check")
    
    # Run available checks
    all_passed = True
    for cmd, description in quality_checks:
        if not run_command(cmd, description):
            all_passed = False
    
    return all_passed


def main():
    """Main test runner"""
    parser = argparse.ArgumentParser(description="Run tests for BTC Trading System")
    parser.add_argument(
        "type",
        nargs="?",
        default="all",
        choices=["all", "unit", "integration", "e2e", "quality", "specific"],
        help="Type of tests to run"
    )
    parser.add_argument(
        "--path",
        help="Specific test path (used with 'specific' type)"
    )
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Install test dependencies first"
    )
    parser.add_argument(
        "--no-coverage",
        action="store_true",
        help="Skip coverage report"
    )
    
    args = parser.parse_args()
    
    # Change to project root
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    # Install dependencies if requested
    if args.install_deps:
        if not install_test_dependencies():
            print("Failed to install dependencies")
            sys.exit(1)
    
    # Run requested tests
    success = True
    
    if args.type == "unit":
        success = run_unit_tests()
    elif args.type == "integration":
        success = run_integration_tests()
    elif args.type == "e2e":
        success = run_e2e_tests()
    elif args.type == "quality":
        success = check_code_quality()
    elif args.type == "specific":
        if not args.path:
            print("Error: --path required for specific tests")
            sys.exit(1)
        success = run_specific_test(args.path)
    else:  # all
        if args.no_coverage:
            # Run each type separately
            success = (
                run_unit_tests() and
                run_integration_tests() and
                run_e2e_tests()
            )
        else:
            success = run_all_tests()
    
    # Print summary
    print("\n" + "="*60)
    if success:
        print("✅ All tests PASSED!")
        if not args.no_coverage and args.type == "all":
            print("\nCoverage report available at: htmlcov/index.html")
    else:
        print("❌ Some tests FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()