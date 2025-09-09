#!/usr/bin/env python3
"""
Generate Outputs for Partner Prediction Analysis

This script runs the partner prediction performance analysis and creates visualizations.

Author: Generated with Claude Code
"""

import sys
import os
from pathlib import Path

# Add repository root to path to import script_utils
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from script_utils import run_scripts_in_order, print_pipeline_header

def main():
    """Generate all partner prediction analysis outputs."""
    print_pipeline_header("PARTNER PREDICTION ANALYSIS PIPELINE")
    
    # Define the scripts to run in order
    scripts_to_run = [
        'calculate_stats.py',
        'plot.py'
    ]
    
    # Run the scripts
    success_count, total_count, failed_scripts = run_scripts_in_order(
        scripts=scripts_to_run,
        working_dir=os.path.dirname(os.path.abspath(__file__)),
        description="Partner Prediction Performance Analysis and Visualization"
    )
    
    # Exit with appropriate code
    if failed_scripts:
        print(f"\n⚠️  {len(failed_scripts)} script(s) failed in partner prediction analysis.")
        sys.exit(1)
    else:
        print(f"\n🎉 All partner prediction analysis completed successfully!")
        print(f"\nGenerated outputs in: results/partner_prediction/outputs/")
        print(f"  - Statistical results: partner_prediction_results.csv")
        print(f"  - JSON data: partner_prediction_results.json")
        print(f"  - Analysis summary: partner_prediction_significance_results.md")
        print(f"  - Combined figure: partner_prediction_combined_figure.svg/.pdf")
        print(f"    - Panel A: Histogram of partner prediction performance")
        print(f"    - Panel B: Learning outcomes by prediction performance")

if __name__ == "__main__":
    main()