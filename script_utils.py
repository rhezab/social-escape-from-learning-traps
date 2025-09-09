#!/usr/bin/env python3
"""
Script Utilities for Dyadic Learning Trap Experiment

Shared utility functions for running analysis scripts across the project.

Author: Generated with Claude Code
"""

import subprocess
import sys
import os
from datetime import datetime

def run_scripts_in_order(scripts, working_dir=None, description=None):
    """
    Run a list of Python scripts in order with error handling and progress tracking.
    
    Args:
        scripts (list): List of script filenames to run in order
        working_dir (str, optional): Directory to run scripts from. Defaults to current directory.
        description (str, optional): Description of what this set of scripts does
        
    Returns:
        tuple: (success_count, total_count, failed_scripts)
    """
    if working_dir is None:
        working_dir = os.getcwd()
    
    if description is None:
        description = f"Scripts in {os.path.basename(working_dir)}"
    
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"Working directory: {working_dir}")
    print(f"Scripts: {', '.join(scripts)}")
    print(f"{'='*60}")
    
    successful_scripts = []
    failed_scripts = []
    
    for i, script in enumerate(scripts, 1):
        script_path = os.path.join(working_dir, script)
        
        print(f"\n[{i}/{len(scripts)}] Running {script}...")
        print(f"{'─'*40}")
        
        # Check if script exists
        if not os.path.exists(script_path):
            print(f"✗ ERROR: Script {script} not found at {script_path}")
            failed_scripts.append(script)
            continue
        
        try:
            # Build command
            cmd = [sys.executable, script]
            
            # Run the script
            result = subprocess.run(
                cmd,
                cwd=working_dir,
                capture_output=False,  # Show output in real-time
                text=True,
                check=True
            )
            
            print(f"\n✓ SUCCESS: {script} completed successfully")
            successful_scripts.append(script)
            
        except subprocess.CalledProcessError as e:
            print(f"\n✗ ERROR: {script} failed with exit code {e.returncode}")
            failed_scripts.append(script)
        except Exception as e:
            print(f"\n✗ ERROR: Failed to run {script}: {str(e)}")
            failed_scripts.append(script)
    
    # Summary for this set of scripts
    print(f"\n{'─'*60}")
    print(f"SUMMARY: {description}")
    print(f"{'─'*60}")
    print(f"Total scripts: {len(scripts)}")
    print(f"Successful: {len(successful_scripts)}")
    print(f"Failed: {len(failed_scripts)}")
    
    if successful_scripts:
        print(f"✓ Successful: {', '.join(successful_scripts)}")
    
    if failed_scripts:
        print(f"✗ Failed: {', '.join(failed_scripts)}")
        
    return len(successful_scripts), len(scripts), failed_scripts

def print_pipeline_header(title):
    """Print a formatted header for analysis pipelines."""
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def print_pipeline_summary(successful_folders, failed_folders, total_time=None):
    """Print a formatted summary for analysis pipelines."""
    print(f"\n{'='*70}")
    print("ANALYSIS PIPELINE SUMMARY")
    print(f"{'='*70}")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if total_time:
        print(f"Total time: {total_time}")
    print(f"Total folders: {len(successful_folders) + len(failed_folders)}")
    print(f"Successful: {len(successful_folders)}")
    print(f"Failed: {len(failed_folders)}")
    
    if successful_folders:
        print(f"\n✓ SUCCESSFUL FOLDERS:")
        for folder in successful_folders:
            print(f"  - {folder}")
    
    if failed_folders:
        print(f"\n✗ FAILED FOLDERS:")
        for folder in failed_folders:
            print(f"  - {folder}")
    
    if not failed_folders:
        print(f"\n🎉 ALL ANALYSES COMPLETED SUCCESSFULLY!")
    else:
        print(f"\n⚠️  Some analyses failed. Check the error messages above for details.")
        
    return len(failed_folders) == 0