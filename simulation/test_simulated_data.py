#!/usr/bin/env python3
"""
Test that simulated data works with the preprocessing pipeline.

This script verifies that the generated simulation data can be successfully processed
by the same preprocessing functions used for real experimental data.
"""

import sys
from pathlib import Path
import pandas as pd

# Add preprocessing directory to path
preprocessing_dir = Path(__file__).parent.parent / 'preprocessing'
sys.path.insert(0, str(preprocessing_dir))

from dataset import load_data
from players_df import create_players_df, add_test_drules_to_players_df
from blocks_df import create_blocks_df

def test_data_loading(json_file):
    """Test that simulated data can be loaded."""
    print(f"Testing data loading for: {json_file}")
    
    try:
        data_dict = load_data(json_file)
        print(f"✓ Successfully loaded {len(data_dict)} participants")
        
        # Check structure of first participant
        first_id = list(data_dict.keys())[0]
        first_participant = data_dict[first_id]
        
        required_fields = [
            'id', 'recruitment_service', 'done', 'points', 
            'task_data', 'relevantFeatures', 'relevantFeaturesBadValues'
        ]
        
        missing_fields = [field for field in required_fields 
                         if field not in first_participant]
        
        if missing_fields:
            print(f"✗ Missing required fields: {missing_fields}")
            return False
        else:
            print("✓ All required fields present")
            
        # Check that recruitment_service is 'sim'
        if first_participant['recruitment_service'] != 'sim':
            print(f"✗ Expected recruitment_service='sim', got '{first_participant['recruitment_service']}'")
            return False
        else:
            print("✓ Recruitment service correctly set to 'sim'")
            
        return data_dict
        
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return None

def test_players_df_creation(data_dict, condition):
    """Test creating players_df from simulated data."""
    print(f"\nTesting players_df creation for {condition} condition...")
    
    try:
        # Debug: check participant structure
        print(f"  Input data: {len(data_dict)} participants")
        for i, (pid, pdata) in enumerate(list(data_dict.items())[:3]):
            print(f"    Participant {i}: recruitment_service={pdata.get('recruitment_service')}, done={pdata.get('done')}")
            if 'conditions' in pdata:
                print(f"      Game type: {pdata['conditions'].get('gameType')}")
                if 'partnerRule' in pdata['conditions']:
                    print(f"      Partner rule: {pdata['conditions']['partnerRule']}")
            else:
                print(f"      No conditions field")
        
        players_df = create_players_df(data_dict, dataset='sim')
        print(f"✓ Successfully created players_df with {len(players_df)} participants")
        
        # Check required columns
        required_cols = ['id', 'game_type', 'points', 'first_test_points', 'second_test_points']
        missing_cols = [col for col in required_cols if col not in players_df.columns]
        
        if missing_cols:
            print(f"✗ Missing required columns: {missing_cols}")
            return None
        else:
            print("✓ All required columns present")
        
        # Check for external aid filtering (should be False for simulation)
        if 'external_aid' in players_df.columns:
            external_aid_counts = players_df['external_aid'].value_counts()
            print(f"  External aid usage: {dict(external_aid_counts)}")
        
        # Check game type distribution
        game_type_counts = players_df['game_type'].value_counts()
        print(f"  Game types: {dict(game_type_counts)}")
        
        # Check partner rule distribution for social conditions
        if 'partner_rule' in players_df.columns:
            partner_rule_counts = players_df['partner_rule'].value_counts()
            print(f"  Partner rules: {dict(partner_rule_counts)}")
        
        return players_df
        
    except Exception as e:
        print(f"✗ Error creating players_df: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_blocks_df_creation(data_dict, condition):
    """Test creating blocks_df from simulated data."""
    print(f"\nTesting blocks_df creation for {condition} condition...")
    
    try:
        # For combined simulation data, all participants have 4 phases
        n_phases = 4  
        phase_blocks = [4, 2, 4, 2]  # Training, test, training, test
        
        # Get participant IDs
        participant_ids = list(data_dict.keys())
        
        blocks_df = create_blocks_df(
            data_dict, 
            epsilon=1, 
            n_phases=n_phases,
            phase_blocks=phase_blocks,
            ids=participant_ids,
            partner_prediction_phase=None,  # No partner prediction in simulation
            sim_data=True  # Important: this is simulation data
        )
        
        print(f"✓ Successfully created blocks_df with {len(blocks_df)} blocks")
        
        # Check required columns
        required_cols = ['id', 'phase', 'block', 'points', 'drule', 'drule_error_2d', 'drule_error_1d_a', 'drule_error_1d_b']
        missing_cols = [col for col in required_cols if col not in blocks_df.columns]
        
        if missing_cols:
            print(f"✗ Missing required columns: {missing_cols}")
            return None
        else:
            print("✓ All required columns present")
        
        # Check decision rule distribution
        drule_counts = blocks_df['drule'].value_counts()
        print(f"  Decision rules: {dict(drule_counts)}")
        
        # Check phase distribution
        phase_counts = blocks_df['phase'].value_counts().sort_index()
        print(f"  Blocks per phase: {dict(phase_counts)}")
        
        # Check that we have the expected number of phases
        expected_phases = list(range(n_phases))
        actual_phases = sorted(blocks_df['phase'].unique())
        if actual_phases != expected_phases:
            print(f"✗ Expected phases {expected_phases}, got {actual_phases}")
            return None
        else:
            print(f"✓ Correct phases present: {actual_phases}")
        
        return blocks_df
        
    except Exception as e:
        print(f"✗ Error creating blocks_df: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_drule_addition(players_df, blocks_df, condition):
    """Test adding decision rules to players_df."""
    print(f"\nTesting decision rule addition for {condition} condition...")
    
    try:
        # For combined simulation data, test phases are always 1 and 3
        first_test_phase = 1
        second_test_phase = 3
        
        players_with_drules = add_test_drules_to_players_df(
            players_df, blocks_df,
            epsilon=2,
            first_test_phase=first_test_phase,
            second_test_phase=second_test_phase
        )
        
        print(f"✓ Successfully added decision rules")
        
        # Check new columns
        expected_cols = ['first_test_drule', 'first_test_drule_gen', 'second_test_drule', 'second_test_drule_gen']
        
        missing_cols = [col for col in expected_cols if col not in players_with_drules.columns]
        
        if missing_cols:
            print(f"✗ Missing new columns: {missing_cols}")
            return None
        else:
            print("✓ All decision rule columns added")
        
        # Check decision rule distributions
        for col in ['first_test_drule', 'second_test_drule']:
            if col in players_with_drules.columns:
                drule_counts = players_with_drules[col].value_counts()
                print(f"  {col}: {dict(drule_counts)}")
        
        return players_with_drules
        
    except Exception as e:
        print(f"✗ Error adding decision rules: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_condition(condition, json_file):
    """Test complete preprocessing pipeline for one condition."""
    print("="*60)
    print(f"TESTING CONDITION: {condition.upper()}")
    print("="*60)
    
    # Test data loading
    data_dict = test_data_loading(json_file)
    if data_dict is None:
        print(f"✗ Failed to load data for {condition}")
        return False
    
    # Test players_df creation
    players_df = test_players_df_creation(data_dict, condition)
    if players_df is None:
        print(f"✗ Failed to create players_df for {condition}")
        return False
    
    # Test blocks_df creation
    blocks_df = test_blocks_df_creation(data_dict, condition)
    if blocks_df is None:
        print(f"✗ Failed to create blocks_df for {condition}")
        return False
    
    # Test decision rule addition
    players_with_drules = test_drule_addition(players_df, blocks_df, condition)
    if players_with_drules is None:
        print(f"✗ Failed to add decision rules for {condition}")
        return False
    
    # Save processed dataframes
    outputs_dir = Path(__file__).parent / 'outputs'
    players_output_file = outputs_dir / f'simulated_players_df_{condition}.csv'
    blocks_output_file = outputs_dir / f'simulated_blocks_df_{condition}.csv'
    
    players_with_drules.to_csv(players_output_file, index=False)
    blocks_df.to_csv(blocks_output_file, index=False)
    
    print(f"✓ Saved processed players_df to: {players_output_file}")
    print(f"✓ Saved processed blocks_df to: {blocks_output_file}")
    print(f"✓ All tests passed for {condition}!")
    return True

def main():
    """Test simulated data with preprocessing pipeline."""
    
    outputs_dir = Path(__file__).parent / 'outputs'
    json_file = outputs_dir / 'simulated_data.json'
    
    if not outputs_dir.exists():
        print("Outputs directory not found. Please run generate_simulation_data.py first.")
        return
    
    if not json_file.exists():
        print(f"✗ Simulation data file not found: {json_file}")
        return
    
    print("="*60)
    print("TESTING SIMULATED DATA")
    print("="*60)
    
    try:
        # Test data loading
        data_dict = test_data_loading(str(json_file))
        if data_dict is None:
            print("✗ Failed to load simulated data")
            return
        
        # Test players_df creation
        players_df = test_players_df_creation(data_dict, 'sim')
        if players_df is None:
            print("✗ Failed to create players_df")
            return
        
        # Test blocks_df creation - use 4 phases for all participants
        blocks_df = test_blocks_df_creation(data_dict, 'sim')
        if blocks_df is None:
            print("✗ Failed to create blocks_df")
            return
        
        # Test decision rule addition
        players_with_drules = test_drule_addition(players_df, blocks_df, 'sim')
        if players_with_drules is None:
            print("✗ Failed to add decision rules")
            return
        
        # Save processed dataframes
        players_output_file = outputs_dir / 'simulated_players_df.csv'
        blocks_output_file = outputs_dir / 'simulated_blocks_df.csv'
        
        players_with_drules.to_csv(players_output_file, index=False)
        blocks_df.to_csv(blocks_output_file, index=False)
        
        print(f"✓ Saved simulated players_df to: {players_output_file}")
        print(f"✓ Saved simulated blocks_df to: {blocks_output_file}")
        
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print("✓ All tests passed! Simulated data is compatible with preprocessing pipeline.")
        print("\nYour simulation data can now be used with:")
        print("1. The preprocessing pipeline (preprocessing/main.py)")
        print("2. All analysis scripts in results/ folders")
        print("3. The same data structures as real experimental data")
        print(f"\nMain data file: {json_file}")
        print(f"Processed data saved to: {outputs_dir}/")
        
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()