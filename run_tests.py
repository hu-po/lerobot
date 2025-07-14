#!/usr/bin/env python

"""
Test runner script for the modified lerobot components.

This script demonstrates how to run the tests for:
1. Atari Teleoperator (with improvements from TODOs A-1, A-2, A-3)
2. Tatbot Robot (with improvements from TODOs T-1 through T-8)
3. IP Camera functionality

Usage:
    python run_tests.py [test_name]
    
Examples:
    python run_tests.py                    # Run all tests
    python run_tests.py atari              # Run only Atari teleoperator tests
    python run_tests.py tatbot             # Run only Tatbot robot tests
    python run_tests.py ip_camera          # Run only IP camera tests
"""

import sys
import subprocess
from pathlib import Path


def run_tests(test_name=None):
    """Run the specified tests."""
    
    # Define test files
    test_files = {
        'atari': 'tests/teleoperators/test_atari_teleoperator.py',
        'tatbot': 'tests/robots/test_tatbot.py', 
        'ip_camera': 'tests/cameras/test_ip_camera.py'
    }
    
    if test_name is None:
        # Run all tests
        print("Running all tests...")
        for name, file_path in test_files.items():
            print(f"\n{'='*50}")
            print(f"Running {name} tests...")
            print(f"{'='*50}")
            run_single_test(file_path)
    elif test_name in test_files:
        # Run specific test
        print(f"Running {test_name} tests...")
        run_single_test(test_files[test_name])
    else:
        print(f"Unknown test: {test_name}")
        print(f"Available tests: {', '.join(test_files.keys())}")
        return False
    
    return True


def run_single_test(test_file):
    """Run a single test file."""
    try:
        # Run pytest with verbose output
        result = subprocess.run([
            sys.executable, '-m', 'pytest', 
            test_file, 
            '-v',  # Verbose output
            '--tb=short',  # Short traceback format
            '--color=yes'  # Colored output
        ], capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"Error running test {test_file}: {e}")
        return False


def main():
    """Main function."""
    if len(sys.argv) > 1:
        test_name = sys.argv[1]
    else:
        test_name = None
    
    success = run_tests(test_name)
    
    if success:
        print("\n✅ All tests completed successfully!")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main() 