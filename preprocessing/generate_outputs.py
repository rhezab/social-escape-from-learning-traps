#!/usr/bin/env python3
"""
Generate Outputs for Data Preprocessing

This script runs all data preprocessing steps to generate the core datasets.

Author: Generated with Claude Code
"""

import sys
import os
from pathlib import Path

# Add repository root to path to import script_utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from script_utils import run_scripts_in_order, print_pipeline_header

def main():
    """Generate all preprocessing outputs."""
    print_pipeline_header("DATA PREPROCESSING PIPELINE")
    
    # Define the scripts to run in order
    scripts_to_run = [
        'main.py'
    ]
    
    # Run the scripts
    success_count, total_count, failed_scripts = run_scripts_in_order(
        scripts=scripts_to_run,
        working_dir=os.path.dirname(os.path.abspath(__file__)),
        description="Data Preprocessing Pipeline"
    )
    
    # Exit with appropriate code
    if failed_scripts:
        print(f"\n⚠️  {len(failed_scripts)} script(s) failed in preprocessing pipeline.")
        sys.exit(1)
    else:
        print(f"\n🎉 All preprocessing steps completed successfully!")
        print(f"\nGenerated outputs in: preprocessing/outputs/")

if __name__ == "__main__":
    main()