import pandas as pd
import numpy as np
from utils import get_last_substring


def create_blocks_df(data_dict, epsilon=1, n_phases=4, phase_blocks=[4,2,4,2], 
                     ids=None, partner_prediction_phase=None, sim_data=False):
    """ 
    Create a DataFrame of block-level data for each participant.
    
    Data Format Requirements:
        data_dict structure: {participant_id: participant_data, ...}
        
        participant_data structure:
            - recruitment_service: str, 'prolific' or 'sim'
            - done: bool, completion status (ignored for 'sim')
            - conditions: dict with 'gameType': str ('solo'/'duo')
            - relevantFeatures: list[int], [feature1_idx, feature2_idx] (0-3)
            - relevantFeaturesBadValues: list[int], [bad_val1, bad_val2] (0 or 1)
            - task_data: list[list] or dict[str, dict], trial data by phase
                - If dict format (from load_data): task_data[str(phase)][str(trial)]
                - Trial structure:
                    - pointsPreTrial/pointsPostTrial: int, for block scoring
                    - stimulusParams: dict, {0: val, 1: val, 2: val, 3: val} (int keys!)
                    - response: str, 'avoid' or 'approach'
                    - partnerResponse: str, 'avoid' or 'approach' (duo only)
                    - predictionPoints: int, for partner prediction phase only
        
        Block structure assumptions:
            - Each phase contains phase_blocks[i] blocks
            - Each block contains exactly 16 trials
            - Trials indexed as: task_data[phase][block*16:(block+1)*16]
    
    Parameters
    ----------
    data_dict : dict
        Dictionary containing participant data
    epsilon : float, optional
        Threshold for determining decision rule type, by default 1 (0-16 scale)
    n_phases : int, optional
        Number of phases in the experiment, by default 4
    phase_blocks : list, optional
        Number of blocks in each phase, by default [4,2,4,2]
    ids : list, optional
        List of participant IDs to include, by default None (includes all)
    partner_prediction_phase : int, optional
        Phase number for partner prediction, by default None
    sim_data : bool, optional
        True if simulated data, to bypass filtering for Prolific participants. 
        
    Returns
    -------
    pandas.DataFrame
        DataFrame containing block-level data for each participant
    """
    new_df = []
    
    # Track skipped participants for better error reporting
    skipped_counts = {
        'not_done': 0,
        'not_prolific': 0,
        'not_in_ids': 0,
        'total_skipped': 0
    }
    
    for id, data in data_dict.items():
        skip_participant = False
        
        if not sim_data:
            # Skip if not completed or not from Prolific
            if not data.get('done'):
                skipped_counts['not_done'] += 1
                skip_participant = True
            elif data.get('recruitment_service') != 'prolific':
                skipped_counts['not_prolific'] += 1
                skip_participant = True
                
        # Skip if id not in provided ids list (applies to both sim and real data)
        if ids is not None and id not in ids:
            skipped_counts['not_in_ids'] += 1
            skip_participant = True
            
        if skip_participant:
            skipped_counts['total_skipped'] += 1
            continue

        # Get basic info that was previously from players_df
        short_id = id[:3]+'-'+get_last_substring(id)
        game_type = data['conditions']['gameType'] if ('conditions' in data and 'gameType' in data['conditions']) else 'solo'

        idx = 0
        for phase in range(n_phases):
            for block in range(phase_blocks[phase]):
                newrow = {}
                newrow['phase'] = phase
                newrow['block'] = block
                newrow['id'] = id
                newrow['short_id'] = short_id
                newrow['game_type'] = game_type

                task_data = get_task_data_by_block(data_dict, id, phase, block)
                if phase == partner_prediction_phase:
                    try:
                        # Count correct predictions, handling missing partner predictionkeys at trial level
                        # e.g. missing for participant '735c91cd-ed50-4758-9f47-62bdd2e039ba-p30'
                        correct_count = 0
                        missing_count = 0
                        for t in task_data:
                            if 'partnerPrediction' in t and 'partnerResponse' in t:
                                if t['partnerPrediction'] == t['partnerResponse']:
                                    correct_count += 1
                            else:
                                missing_count += 1
                        
                        newrow['partner_predictions_correct'] = correct_count
                        newrow['partner_predictions_missing'] = missing_count
                        
                    except Exception as e:
                        raise RuntimeError(f"Unexpected error calculating partner prediction correctness for participant {id}: {str(e)}")
                    
                try:
                    pre_points = task_data[0]['pointsPreTrial']
                    post_points = task_data[-1]['pointsPostTrial']
                    
                    if pre_points is None or post_points is None:
                        newrow['points'] = None  # Default to 0 when points data is None
                    else:
                        newrow['points'] = post_points - pre_points
                except (KeyError, IndexError) as e:
                    raise KeyError(f"Missing points data for participant {id}, phase {phase}, block {block}: {str(e)}")
                except Exception as e:
                    raise RuntimeError(f"Unexpected error calculating points for participant {id}: {str(e)}")

                dmat = get_decision_matrix_by_block(data_dict, id, phase, block, normalized=False)
                ideal_dmat = get_ideal_decision_matrix(data_dict, id, normalized=False)
                error = np.sum(np.abs(dmat - ideal_dmat))
                dmat1, dmat2 = get_1d_decision_matrices(data_dict, id, normalized=False)
                error_1d_1 = np.sum(np.abs(dmat - dmat1))
                error_1d_2 = np.sum(np.abs(dmat - dmat2)) # save errors for test block classification??

                # Verify that at most one error is <= epsilon
                errors_below_threshold = sum([error <= epsilon, error_1d_1 <= epsilon, error_1d_2 <= epsilon])
                assert errors_below_threshold <= 1, f"Multiple errors <= epsilon for {id}, phase {phase}, block {block}: error={error}, error_1d_1={error_1d_1}, error_1d_2={error_1d_2}, epsilon={epsilon}"

                # Classify decision rules
                if error <= epsilon:
                    newrow['drule'] = '2d'
                elif error_1d_1 <= epsilon:
                    newrow['drule'] = '1d_a'
                elif error_1d_2 <= epsilon:
                    newrow['drule'] = '1d_b'
                else:
                    newrow['drule'] = 'neither'

                # Save errors 
                newrow['drule_error_2d'] = error
                newrow['drule_error_1d_a'] = error_1d_1
                newrow['drule_error_1d_b'] = error_1d_2
                newrow['drule_error_1d'] = np.amin([error_1d_1, error_1d_2])

                # Add approach behavior counts
                rewards_approached, punishment_approached = get_approach_behavior_by_block(data_dict, id, phase, block)
                newrow['rewards_approached'] = rewards_approached
                newrow['punishment_approached'] = punishment_approached

                new_df.append(newrow)
                idx += 1
    new_df = pd.DataFrame(new_df)
    
    # Check if new_df is empty
    if len(new_df) == 0:
        error_msg = ["No blocks were created. Debugging information:"]
        error_msg.append(f"  Total participants in data_dict: {len(data_dict)}")
        error_msg.append(f"  Total participants skipped: {skipped_counts['total_skipped']}")
        
        if skipped_counts['total_skipped'] > 0:
            error_msg.append("  Reasons for skipping:")
            if skipped_counts['not_done'] > 0:
                error_msg.append(f"    - Not marked as done: {skipped_counts['not_done']}")
            if skipped_counts['not_prolific'] > 0:
                error_msg.append(f"    - Not from Prolific: {skipped_counts['not_prolific']}")
            if skipped_counts['not_in_ids'] > 0:
                error_msg.append(f"    - Not in provided IDs list: {skipped_counts['not_in_ids']}")
        
        error_msg.append(f"  Configuration: n_phases={n_phases}, phase_blocks={phase_blocks}")
        error_msg.append(f"  sim_data setting: {sim_data}")
        
        if not sim_data and skipped_counts['not_prolific'] > 0:
            error_msg.append("\n  Note: If processing simulated data, make sure to set sim_data=True")
        
        raise ValueError("\n".join(error_msg))
    
    new_df.loc[:, 'drule'] = pd.Categorical(new_df['drule'], categories=['2d', '1d_a', '1d_b', 'neither'])
    return new_df


# Helpers
def get_task_data_by_block(data_dict, id, phase, block):
    """
    Get task data for a specific block.
    
    Data format required:
        data_dict[id]['task_data']: list[list] of trial data
            - task_data[phase]: list of trials for this phase
            - Returns 16 trials: trials [block*16:(block+1)*16]
    """
    return data_dict[id]['task_data'][phase][block*16:(block+1)*16]

def get_decision_matrix_by_block(data_dict, id, phase, block, partner_response=False, normalized=True):
    """
    Get 2x2 matrix of decisions by relevant_features.
    
    Quantifies the proportion of choices that were 'avoid' in each cell of the 2x2 matrix.
    Each cell represents one combination of the two relevant features (0 or 1).
    Since there are 4 trials per cell in a 16-trial block, normalized values are 0, 0.25, 0.5, 0.75, or 1.0.
    
    Data format required:
        data_dict[id]['relevantFeatures']: list[int], [feature1_idx, feature2_idx] (0-3)
        Block trial data (from get_task_data_by_block):
            trial['stimulusParams']: dict, {0: val, 1: val, 2: val, 3: val} (int keys!)
            trial['response']: str, 'avoid' or 'approach'
            trial['partnerResponse']: str, 'avoid' or 'approach' (if partner_response=True)
    
    Parameters:
    -----------
    normalized : bool, default True
        If True, returns proportions (0-1). If False, returns raw counts (0-4).
    """
    relevant_features = data_dict[id]['relevantFeatures']
    block_data = get_task_data_by_block(data_dict, id, phase, block)
    mat = np.zeros((2,2))
    for trial in block_data:
        stim = trial['stimulusParams']
        # Handle both int and str keys in stimulusParams (JSON converts int keys to strings)
        rf0 = stim[str(relevant_features[0])] if str(relevant_features[0]) in stim else stim[relevant_features[0]]
        rf1 = stim[str(relevant_features[1])] if str(relevant_features[1]) in stim else stim[relevant_features[1]]
        if partner_response:
            resp = trial['partnerResponse']
        else:
            resp = trial['response']
        if resp == 'avoid':
            mat[rf0,rf1] += 1
    return mat / 4 if normalized else mat

def get_ideal_decision_matrix(data_dict, id, normalized=True):
    """
    Get ideal decision matrix based on relevantFeaturesBadValues.
    
    Returns the optimal decision pattern where 'avoid' responses should occur.
    Only the cell corresponding to the bad feature combination should be avoided (value 1).
    
    Data format required:
        data_dict[id]['relevantFeaturesBadValues']: list[int], [bad_val1, bad_val2] (0 or 1)
    
    Parameters:
    -----------
    normalized : bool, default True
        If True, returns proportions (0 or 1). If False, returns raw counts (0 or 4).
    """
    rfbv = data_dict[id]['relevantFeaturesBadValues']
    mat = np.zeros((2,2))
    mat[rfbv[0], rfbv[1]] = 1 if normalized else 4
    return mat

def get_1d_decision_matrices(data_dict, id, normalized=True):
    """
    Get 1D decision matrices for single-feature avoidance strategies.
    
    Returns two matrices representing decisions based on only one relevant feature:
    - mat1: avoid when first relevant feature is bad (regardless of second feature)
    - mat2: avoid when second relevant feature is bad (regardless of first feature)
    
    Data format required:
        data_dict[id]['relevantFeaturesBadValues']: list[int], [bad_val1, bad_val2] (0 or 1)
    
    Parameters:
    -----------
    normalized : bool, default True
        If True, returns proportions (0 or 1). If False, returns raw counts (0 or 4).
    """
    rfbv = data_dict[id]['relevantFeaturesBadValues']
    mat1 = np.zeros((2,2))
    mat2 = np.zeros((2,2))
    value = 1 if normalized else 4
    mat1[rfbv[0],:] = value
    mat2[:,rfbv[1]] = value
    return mat1, mat2

def get_approach_behavior_by_block(data_dict, id, phase, block):
    """
    Get approach behavior counts for rewarding and punishing stimuli.
    
    Counts how many rewarding and punishing stimuli were approached in a block.
    
    Data format required:
        Block trial data (from get_task_data_by_block):
            trial['response']: str, 'avoid' or 'approach'  
            trial['approachOutcome']: int, -1 for punishing, +1 for rewarding
    
    Returns:
    --------
    tuple: (rewards_approached, punishment_approached)
        rewards_approached: int, count of rewarding stimuli that were approached
        punishment_approached: int, count of punishing stimuli that were approached
    """
    block_data = get_task_data_by_block(data_dict, id, phase, block)
    
    rewards_approached = 0
    punishment_approached = 0
    
    for trial in block_data:
        if trial['response'] == 'approach':
            if trial['approachOutcome'] == 1:  # Rewarding stimulus
                rewards_approached += 1
            elif trial['approachOutcome'] == -1:  # Punishing stimulus
                punishment_approached += 1
    
    return rewards_approached, punishment_approached


def merge_blocks_dataframes(blocks_dataframes, dataset_labels=None):
    """
    Merge multiple blocks dataframes into one with dataset labels.
    
    Parameters:
    -----------
    blocks_dataframes : list of pandas.DataFrame
        List of blocks dataframes to merge
    dataset_labels : list of str, optional
        Labels for each dataset. If None, uses '1', '2', '3', etc.
        
    Returns:
    --------
    pandas.DataFrame
        Merged dataframe with 'dataset' column added
    """
    if dataset_labels is None:
        dataset_labels = [str(i+1) for i in range(len(blocks_dataframes))]
    
    if len(blocks_dataframes) != len(dataset_labels):
        raise ValueError("Number of dataframes must match number of dataset labels")
    
    # Add dataset column to each dataframe
    labeled_dataframes = []
    for blocks_df, label in zip(blocks_dataframes, dataset_labels):
        df_copy = blocks_df.copy()
        df_copy['dataset'] = label
        labeled_dataframes.append(df_copy)
    
    # Merge all dataframes
    merged_df = pd.concat(labeled_dataframes, ignore_index=True)
    
    return merged_df

def analyze_blocks_dataframe(blocks_df):
    """
    Print analysis metrics for merged blocks dataframe.
    
    Analyzes NaN values by column and dataset, providing a summary of data quality.
    
    Parameters:
    -----------
    blocks_df : pandas.DataFrame
        Merged blocks dataframe with 'dataset' column
    """
    print(f"\nMerged blocks dataframe size: {len(blocks_df)}")
    
    print("\nChecking for NaN values in blocks_all...")
    
    # Get unique ids with NaN values in any column
    ids_with_nans = blocks_df[blocks_df.isna().any(axis=1)]['id'].unique()
    print(f"Number of unique IDs with NaN values: {len(ids_with_nans)}")
    
    # Print which columns have NaN values
    nan_cols = blocks_df.columns[blocks_df.isna().any()].tolist()
    print("\nColumns with NaN values:")
    for col in nan_cols:
        num_nans = blocks_df[col].isna().sum()
        print(f"  {col}: {num_nans} NaN values")
    
    # Print breakdown by dataset
    print("\nBreakdown of IDs with NaNs by dataset:")
    for dataset in blocks_df['dataset'].unique():
        dataset_ids = blocks_df[blocks_df['dataset'] == dataset]['id'].unique()
        dataset_nan_ids = [id for id in ids_with_nans if id in dataset_ids]
        print(f"  {dataset}: {len(dataset_nan_ids)} IDs with NaNs")

def add_generalized_drule_column(blocks_df):
    """
    Add a generalized decision rule column to blocks dataframe.
    
    Converts specific decision rules ('1d_a', '1d_b') to generalized categories ('1d').
    
    Parameters:
    -----------
    blocks_df : pandas.DataFrame
        DataFrame with 'drule' column containing specific decision rules
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added 'drule_gen' column containing generalized categories
    """
    # Create a new column 'drule_gen' that generalizes the decision rules
    blocks_df['drule_gen'] = blocks_df['drule'].map({
        '2d': '2d',
        '1d_a': '1d', 
        '1d_b': '1d',
        'neither': 'neither'
    })
    
    # Convert to categorical with specific order
    blocks_df['drule_gen'] = pd.Categorical(blocks_df['drule_gen'], 
                                           categories=['2d', '1d', 'neither'])
    
    return blocks_df