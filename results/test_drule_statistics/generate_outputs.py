#!/usr/bin/env python3
"""
Generate Outputs for Test Decision Rule Statistics Analysis

This script calculates test decision rule statistics and generates documentation.

Author: Generated with Claude Code
"""

import sys
import os
from pathlib import Path

# Add repository root to path to import script_utils
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from script_utils import run_scripts_in_order, print_pipeline_header

def main():
    """Generate all test decision rule statistics outputs."""
    print_pipeline_header("TEST DECISION RULE STATISTICS ANALYSIS PIPELINE")
    
    # Define the scripts to run in order
    scripts_to_run = [
        'calculate_test_drule_stats.py',
        'plot.py'
    ]
    
    # Run the scripts
    success_count, total_count, failed_scripts = run_scripts_in_order(
        scripts=scripts_to_run,
        working_dir=os.path.dirname(os.path.abspath(__file__)),
        description="Test Decision Rule Statistics Analysis"
    )
    
    # Exit with appropriate code
    if failed_scripts:
        print(f"\n⚠️  {len(failed_scripts)} script(s) failed in test decision rule statistics analysis.")
        sys.exit(1)
    else:
        print(f"\n🎉 Test decision rule statistics analysis completed successfully!")
        print(f"\nGenerated outputs in: results/test_drule_statistics/outputs/")

if __name__ == "__main__":
    main()