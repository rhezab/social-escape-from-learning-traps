#!/usr/bin/env python3
"""
Generate all outputs for the reward/punishment heatmap analysis.

This script runs the heatmap generation for different participant groups
in the dyadic learning trap experiment.
"""

import os
import sys
from datetime import datetime
from pathlib import Path


def main():
    """Generate all heatmap outputs."""
    print("REWARD/PUNISHMENT HEATMAP ANALYSIS - GENERATE ALL OUTPUTS")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Ensure we're in the correct directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Create outputs directory
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)
    
    print(f"Working directory: {os.getcwd()}")
    print(f"Output directory: {outputs_dir.absolute()}")
    
    try:
        # Import and run only the trapped-to-optimal grid generation
        print("\nGenerating trapped-to-optimal heatmap grid...")
        from make_trapped_to_optimal_grid import main as generate_grid
        generate_grid()
        
        # Summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print('='*60)
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\nTrapped-to-optimal heatmap grid generated successfully!")
        
        # List generated files
        print("\nGenerated outputs:")
        for svg_file in sorted(outputs_dir.glob("*.svg")):
            print(f"  - {svg_file.name}")
        
    except Exception as e:
        print(f"\n✗ ERROR: Heatmap grid generation failed: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()