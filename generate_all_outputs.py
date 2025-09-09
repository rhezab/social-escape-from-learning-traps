#!/usr/bin/env python3
"""
Run All Analyses for Dyadic Learning Trap Experiment

This script auto-discovers and runs all analysis pipelines in the correct order:
1. Simulation data generation (simulation/generate_outputs.py)
2. Data preprocessing (preprocessing/generate_outputs.py)
3. All results analyses (results/*/generate_outputs.py)

Author: Generated with Claude Code
"""

import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime
from script_utils import print_pipeline_header, print_pipeline_summary

def find_generate_outputs_scripts(repo_root):
    """
    Find all generate_outputs.py scripts in the repository.
    
    Returns:
        tuple: (simulation_script, preprocessing_script, results_scripts)
    """
    simulation_script = None
    preprocessing_script = None
    results_scripts = []
    
    # Check for simulation script
    simulation_path = os.path.join(repo_root, 'simulation', 'generate_outputs.py')
    if os.path.exists(simulation_path):
        simulation_script = simulation_path
    
    # Check for preprocessing script
    preprocessing_path = os.path.join(repo_root, 'preprocessing', 'generate_outputs.py')
    if os.path.exists(preprocessing_path):
        preprocessing_script = preprocessing_path
    
    # Find all results scripts
    results_dir = os.path.join(repo_root, 'results')
    if os.path.exists(results_dir):
        for item in os.listdir(results_dir):
            item_path = os.path.join(results_dir, item)
            if os.path.isdir(item_path):
                script_path = os.path.join(item_path, 'generate_outputs.py')
                if os.path.exists(script_path):
                    results_scripts.append((item, script_path))
    
    # Sort results scripts by folder name for consistent ordering
    results_scripts.sort(key=lambda x: x[0])
    
    return simulation_script, preprocessing_script, results_scripts

def run_generate_outputs_script(script_path, description):
    """
    Run a generate_outputs.py script.
    
    Args:
        script_path: Path to the generate_outputs.py script
        description: Description for logging
        
    Returns:
        bool: True if successful, False if failed
    """
    working_dir = os.path.dirname(script_path)
    
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"Script: {os.path.relpath(script_path)}")
    print(f"Working directory: {os.path.relpath(working_dir)}")
    print(f"{'='*60}")
    
    try:
        # Build command
        cmd = [sys.executable, 'generate_outputs.py']
        
        # Run the script
        result = subprocess.run(
            cmd,
            cwd=working_dir,
            capture_output=False,  # Show output in real-time
            text=True,
            check=True
        )
        
        print(f"\n✓ SUCCESS: {description} completed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n✗ ERROR: {description} failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"\n✗ ERROR: Failed to run {description}: {str(e)}")
        return False

def main():
    """Run all analyses in the correct order using auto-discovery."""
    start_time = datetime.now()
    
    print_pipeline_header("DYADIC LEARNING TRAP EXPERIMENT - COMPLETE ANALYSIS PIPELINE")
    
    # Get the repository root directory
    repo_root = os.path.dirname(os.path.abspath(__file__))
    
    # Auto-discover all generate_outputs.py scripts
    print("Auto-discovering analysis scripts...")
    simulation_script, preprocessing_script, results_scripts = find_generate_outputs_scripts(repo_root)
    
    print(f"Found simulation script: {'✓' if simulation_script else '✗'}")
    print(f"Found preprocessing script: {'✓' if preprocessing_script else '✗'}")
    print(f"Found {len(results_scripts)} results analysis scripts:")
    for folder_name, _ in results_scripts:
        print(f"  - results/{folder_name}/")
    
    # Track success/failure
    successful_folders = []
    failed_folders = []
    
    # Calculate total steps
    total_steps = 0
    if simulation_script:
        total_steps += 1
    if preprocessing_script:
        total_steps += 1
    total_steps += len(results_scripts)
    
    current_step = 0
    
    # Step 1: Run simulation (must be first)
    if simulation_script:
        current_step += 1
        print(f"\n[STEP {current_step}/{total_steps}] SIMULATION")
        success = run_generate_outputs_script(
            simulation_script, 
            "Simulation Data Generation"
        )
        
        if success:
            successful_folders.append("simulation")
        else:
            failed_folders.append("simulation")
            print(f"\n⚠️  WARNING: Simulation failed. This may affect subsequent analyses.")
    else:
        print(f"\n⚠️  WARNING: No simulation script found at simulation/generate_outputs.py")
        failed_folders.append("simulation (not found)")
    
    # Step 2: Run preprocessing
    if preprocessing_script:
        current_step += 1
        print(f"\n[STEP {current_step}/{total_steps}] PREPROCESSING")
        success = run_generate_outputs_script(
            preprocessing_script, 
            "Data Preprocessing Pipeline"
        )
        
        if success:
            successful_folders.append("preprocessing")
        else:
            failed_folders.append("preprocessing")
            print(f"\n⚠️  WARNING: Preprocessing failed. This may affect subsequent analyses.")
    else:
        print(f"\n⚠️  WARNING: No preprocessing script found at preprocessing/generate_outputs.py")
        failed_folders.append("preprocessing (not found)")
    
    # Step 3+: Run all results analyses
    for folder_name, script_path in results_scripts:
        current_step += 1
        print(f"\n[STEP {current_step}/{total_steps}] RESULTS/{folder_name.upper()}")
        
        success = run_generate_outputs_script(
            script_path,
            f"Results Analysis: {folder_name}"
        )
        
        if success:
            successful_folders.append(f"results/{folder_name}")
        else:
            failed_folders.append(f"results/{folder_name}")
            print(f"\n⚠️  WARNING: {folder_name} analysis failed. Continuing with remaining analyses...")
    
    # Calculate total time
    end_time = datetime.now()
    total_time = str(end_time - start_time).split('.')[0]  # Remove microseconds
    
    # Final summary
    success = print_pipeline_summary(successful_folders, failed_folders, total_time)
    
    if success:
        print(f"\nOutput files have been generated in:")
        if simulation_script:
            print(f"  - simulation/outputs/")
        print(f"  - preprocessing/outputs/")
        for folder_name, _ in results_scripts:
            print(f"  - results/{folder_name}/outputs/")
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()