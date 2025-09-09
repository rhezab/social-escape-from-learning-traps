#!/usr/bin/env python3
"""
Generate Simulation Data Outputs

This script generates simulated experimental data for the dyadic learning trap experiment.
It serves as the entry point for the simulation pipeline when run as part of the complete
analysis pipeline.

The script generates:
- Simulated behavioral data (simulated_data.json)
- Simulation metadata (simulation_metadata.json)
- Analysis outputs (simulated_players_df.csv, simulated_blocks_df.csv)

Author: Generated with Claude Code
"""

import sys
import os
from pathlib import Path
from datetime import datetime


def main():
    """Generate all simulation outputs."""
    
    print("=" * 60)
    print("SIMULATION DATA GENERATION")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Ensure we're in the simulation directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Create outputs directory
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)
    
    print(f"Working directory: {os.getcwd()}")
    print(f"Output directory: {outputs_dir.absolute()}")
    print()
    
    try:
        # Step 1: Generate simulation data
        print("[STEP 1/2] Generating simulated experimental data...")
        from generate_simulation_data import main as generate_main
        generate_main()
        print("✓ Simulation data generated successfully")
        
        # Step 2: Test simulated data and generate analysis files
        print("\n[STEP 2/2] Testing simulated data and generating analysis outputs...")
        from test_simulated_data import main as test_main
        test_main()
        print("✓ Analysis outputs generated successfully")
        
        print("\n" + "=" * 60)
        print("SIMULATION OUTPUTS COMPLETED SUCCESSFULLY")
        print("=" * 60)
        
        # List generated files
        print("\nGenerated files:")
        for file_path in sorted(outputs_dir.glob("*")):
            if file_path.is_file():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                print(f"  - {file_path.name} ({size_mb:.2f} MB)")
        
        print(f"\nAll outputs saved to: {outputs_dir.absolute()}")
        
    except Exception as e:
        print(f"\n✗ ERROR: Simulation generation failed: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()