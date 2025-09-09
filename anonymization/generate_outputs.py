"""
Generate Anonymized Datasets

This script runs the complete anonymization pipeline:
1. Anonymizes all experimental datasets
2. Verifies the anonymization was successful
3. Generates reports

The output files are saved in the anonymization/outputs/ directory.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from script_utils import run_scripts_in_order, print_pipeline_header


def main():
    print_pipeline_header("DATA ANONYMIZATION PIPELINE")
    
    # Define the scripts to run in order
    scripts_to_run = [
        "anonymize.py",
        "test_anonymize.py"
    ]
    
    # Run the scripts
    success_count, total_count, failed_scripts = run_scripts_in_order(
        scripts=scripts_to_run,
        working_dir=os.path.dirname(os.path.abspath(__file__)),
        description="Data Anonymization Pipeline"
    )
    
    # Summary
    if failed_scripts:
        print(f"\n⚠️  {len(failed_scripts)} script(s) failed in anonymization pipeline.")
        print("Failed scripts:", ', '.join(failed_scripts))
        return 1
    else:
        print("\n✅ Anonymization pipeline completed successfully!")
        print(f"   Anonymized datasets are in: data/")
        print(f"   Reports are in: anonymization/outputs/")
        return 0


if __name__ == "__main__":
    sys.exit(main())