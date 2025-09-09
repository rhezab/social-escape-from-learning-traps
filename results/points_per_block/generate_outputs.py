#!/usr/bin/env python3
"""
Generate all outputs for the points per block analysis.

This script runs all analyses in the points_per_block folder:
1. Generates points per block plots (grid by dataset)
2. Generates cumulative points plots (grid by dataset)
"""

import os
import sys
import subprocess
from datetime import datetime


def run_script(script_name, description):
    """Run a Python script and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Script: {script_name}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print('='*60)
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=True,
            text=True,
            check=True
        )
        print("SUCCESS")
        if result.stdout:
            print("\nOutput:")
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("FAILED")
        print(f"\nError: {e}")
        if e.stdout:
            print("\nStdout:")
            print(e.stdout)
        if e.stderr:
            print("\nStderr:")
            print(e.stderr)
        return False


def main():
    """Run all analysis scripts in the correct order."""
    print("POINTS PER BLOCK ANALYSIS - GENERATE ALL OUTPUTS")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Ensure we're in the correct directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Track success
    all_success = True
    
    # Run analyses
    scripts = [
        ("plot_points_per_block.py", "Points per block analysis and visualization")
    ]
    
    for script, description in scripts:
        if os.path.exists(script):
            success = run_script(script, description)
            all_success = all_success and success
        else:
            print(f"\nWARNING: Script {script} not found, skipping...")
            all_success = False
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if all_success:
        print("\nAll analyses completed successfully!")
        print("\nGenerated outputs:")
        print("  - outputs/points_per_block_by_dataset.svg")
        print("  - outputs/cumulative_points_by_dataset.svg")
        print("  - outputs/analysis_summary.md")
        print("  - outputs/trapped_learners_stats.csv")
        print("  - outputs/trapped_learners_stats_summary.md")
    else:
        print("\nSome analyses failed. Check the error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()