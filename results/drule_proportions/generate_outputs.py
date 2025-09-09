#!/usr/bin/env python3
"""
Generate Outputs for Decision Rule Proportions Analysis

This script calculates decision rule proportions and creates the corresponding figure.

Author: Generated with Claude Code
"""

import sys
import os
from pathlib import Path

# Add repository root to path to import script_utils
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from script_utils import run_scripts_in_order, print_pipeline_header

def main():
    """Generate all decision rule proportions outputs."""
    print_pipeline_header("DECISION RULE PROPORTIONS ANALYSIS PIPELINE")
    
    # Define the scripts to run in order
    scripts_to_run = [
        'calculate_proportions.py',
        'calculate_test_proportions.py',
        'plot.py'
    ]
    
    # Run the scripts
    success_count, total_count, failed_scripts = run_scripts_in_order(
        scripts=scripts_to_run,
        working_dir=os.path.dirname(os.path.abspath(__file__)),
        description="Decision Rule Proportions Analysis (Learning & Test Phases) and Plotting"
    )
    
    # Exit with appropriate code
    if failed_scripts:
        print(f"\n⚠️  {len(failed_scripts)} script(s) failed in decision rule proportions analysis.")
        sys.exit(1)
    else:
        print(f"\n🎉 All decision rule proportions analysis completed successfully!")
        print(f"\nGenerated outputs in: results/drule_proportions/outputs/")
        print(f"  - Learning phase proportions: decision_rule_proportions.csv")
        print(f"  - Test phase proportions: test_decision_rule_proportions.csv")
        print(f"  - Learning phase visualization: drule-proportions-bar-plot.svg/.pdf")
        print(f"  - First test overview: first_test_overview.svg/.pdf")
        print(f"  - Trapped learner comparison: trapped_learner_comparison.svg/.pdf")
        print(f"  - Original test visualization: test_decision_rule_proportions.svg/.pdf")

if __name__ == "__main__":
    main()