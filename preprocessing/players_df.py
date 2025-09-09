import pandas as pd
import numpy as np
from utils import get_last_substring, time_on_page

def create_players_df(data_dict, dataset=None):
    """
    Create a DataFrame of player data with configurable fields based on dataset.
    
    Args:
        data_dict: Dictionary containing player data (previously d.data)
        dataset: Dataset identifier ('1', '2', '3', '4', '5', '6', 'sim')
    
    Data Format Requirements:
        data_dict structure: {participant_id: participant_data, ...}
        
        participant_data structure (core fields for all datasets):
            - id: str, participant identifier
            - recruitment_service: str, 'prolific' (d1-d6) or 'sim'
            - done: bool, completion status (ignored for 'sim')
            - points: int, total points earned
            - conditions: dict containing:
                - gameType: str, 'solo' or 'duo'
                - partnerRule: str, d2-d6+sim only ('2d'/'1d_a'/'1d_b'/'neither')
            - task_data: list[list] or dict[str, dict], trial data by phase:
                - Format: task_data[phase][trial] with trial fields:
                    - pointsPreTrial/pointsPostTrial: int, for scoring
                    - stimulusParams: dict, feature values {0,1,2,3: 0|1}
                    - response: str, 'avoid' or 'approach'
                    - partnerResponse: str, 'avoid' or 'approach' (duo only)
        
        Real datasets only (1-6):
            - questionnaireQuestions: list, external aid question at index 1:-1, 2:4, 3/4/5/6:5
        
        Dataset-specific fields:
            - bonus_points: int, datasets 2-6 only
            - route_order/route_times: list, dataset 4 only (for time_on_share_page)
            - user_data: dict with 'partner': str, dataset 1 only (partner ID)
        
        Phase structure (task_data length):
            - datasets 1-4,6+sim: 4 phases, test phases at [1,3]
            - dataset 5: 5 phases, test phases at [1,4], partner prediction at phase 2
    
    Raises:
        ValueError: If dataset is not one of the supported datasets
    """
    SUPPORTED_DATASETS = ['1', '2', '3', '4', '5', '6', 'sim']
    
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"Dataset must be one of {SUPPORTED_DATASETS}, got {dataset}")

    external_aid_key = {'Yes': True, 'No': False}
    partner_judgement_key = {
        'Better than me': 'Better',
        'About equal to me': 'Equal',
        'Worse than me': 'Worse',
        'No idea': 'IDK'
    }

    to_df = []
    for id, data in data_dict.items():
        # Skip specific IDs if needed
        if dataset == '1' and id == '00d725e7-19d0-479a-971e-7f9a2f966d76-p249':
            continue

        row = {}
        # Personal info
        row['id'] = id
        row['short_id'] = id[:3]+'-'+get_last_substring(id)

        # Handle recruitment service
        if dataset == 'sim':
            pass
        elif data['recruitment_service'] == 'prolific':
            # Only include completed participants
            if not data['done']:
                continue
        else:
            continue

        # Questionnaire handling
        if dataset == 'sim':
            row['external_aid'] = False
        elif bool(data.get('questionnaireQuestions')):
            try:
                # Different datasets have different question indices
                if dataset in ['2']:
                    q_index = 4
                elif dataset in ['3', '4', '5', '6']:
                    q_index = 5
                elif dataset in ['1']:
                    q_index = -1
                else:
                    raise ValueError(f"Dataset {dataset} not implemented for questionnaire handling")
                
                q = data['questionnaireQuestions'][q_index]
                if "external aid" in q['text'].lower():
                    row['external_aid'] = external_aid_key[q['responses']]
            except:
                continue
        else:
            continue

        # Partner judgement (for duo games)
        if dataset in ['3', '4'] and data['conditions']['gameType'] == 'duo':
            q_index = -3 if dataset == '3' else -2
            try:
                q = data['questionnaireQuestions'][q_index]
                if "How did your partner perform" in q['text']:
                    row['partner_judgement'] = partner_judgement_key[q['responses']]
            except:
                row['partner_judgement'] = None

        # Game metrics (common across all datasets)
        row['game_type'] = data['conditions']['gameType'] if ('conditions' in data and 'gameType' in data['conditions']) else 'solo'
        row['points'] = data['points']
        
        # Handle different test phases for d5
        if dataset == '5':
            first_test_phase = 1
            second_test_phase = 3
            partner_pred_phase = 2
        else:
            first_test_phase = 1
            second_test_phase = 3

        # Calculate test points using appropriate phases
        if data['task_data'][first_test_phase][-1]['pointsPostTrial'] is None:
            continue
        row['first_test_points'] = data['task_data'][first_test_phase][-1]['pointsPostTrial'] - data['task_data'][first_test_phase][0]['pointsPreTrial']
        row['second_test_points'] = data['task_data'][second_test_phase][-1]['pointsPostTrial'] - data['task_data'][second_test_phase][0]['pointsPreTrial']
        row['first_half_points'] = data['task_data'][first_test_phase][-1]['pointsPostTrial']
        row['second_half_points'] = data['task_data'][second_test_phase][-1]['pointsPostTrial'] - row['first_half_points']

        # Dataset-specific fields
        if dataset in ['2', '3', '4', '5', 'sim']:
            row['partner_rule'] = data['conditions'].get('partnerRule')

        if dataset in ['2', '3', '4', '5', '6']:
            row['bonus_points'] = data.get('bonus_points')

        if dataset == '4':
            row['time_on_share_page'] = time_on_page(data, 'midgameshare')['seconds']

        if dataset == '5':
            row['partner_predictions_correct'] = data['task_data'][partner_pred_phase][-1]['cumulativeCorrectPredictions']

        if dataset == '1':
            # Partner info specific to d1
            partner_id = data['user_data']['partner']
            if partner_id == 'none':
                partner_id = None
            row['partner_id'] = partner_id
            if partner_id and (partner_id in data_dict):
                row['partner_short_id'] = partner_id[:3] + '-' + get_last_substring(partner_id)
                row['partner_points'] = data_dict[partner_id]['points']
            else:
                row['partner_short_id'] = None

        to_df.append(row)
    
    return pd.DataFrame(to_df)

def add_partner_rules_d1(df_d1):
    """
    Add partner rule information to d1 DataFrame.
    
    This function:
    1. Adds partner_rule based on first_test_drule
    2. Optionally adds partner_rule_gen if first_test_drule_gen exists
    
    Parameters
    ----------
    df_d1 : pandas.DataFrame
        d1 DataFrame containing participant data with 'id', 'partner_id', and 'first_test_drule' columns
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with added partner rule columns
    """
    # Add partner rules
    partner_rules = dict(zip(df_d1['id'], df_d1['first_test_drule']))
    df_d1['partner_rule'] = df_d1['partner_id'].map(partner_rules)
    
    return df_d1


def filter_d1_participants(df_d1):
    """
    Filter d1 participants to only include those with valid partner data.
    
    For duo participants: only include those whose partner also exists in the dataset
    and has matching game type.
    For solo participants: include all (no partner filtering needed).
    
    Parameters
    ----------
    df_d1 : pandas.DataFrame
        DataFrame containing d1 participant data
        
    Returns
    -------
    pandas.DataFrame
        Filtered DataFrame containing valid participants
    """
    # Get all participant IDs in d1
    d1_ids = set(df_d1['id'])
    
    # Separate solo and duo participants
    d1_solo_participants = df_d1[df_d1['game_type'] == 'solo'].copy()
    d1_duo_participants = df_d1[df_d1['game_type'] == 'duo'].copy()
    
    # For duo participants, filter for valid partners
    if len(d1_duo_participants) > 0:
        # Create mask for matching game types between partners
        game_type_dict = dict(zip(d1_duo_participants['id'], d1_duo_participants['game_type']))
        matching_game_type_mask = d1_duo_participants['partner_id'].map(game_type_dict) == d1_duo_participants['game_type']
        
        # Combine masks for valid partners and matching game types
        valid_partner_mask = d1_duo_participants['partner_id'].isin(d1_ids) & matching_game_type_mask
        
        # Filter duo participants
        valid_duo_participants = d1_duo_participants[valid_partner_mask].copy()
    else:
        valid_duo_participants = d1_duo_participants.copy()
    
    # Combine solo and filtered duo participants
    df_filtered = pd.concat([d1_solo_participants, valid_duo_participants], ignore_index=True)
    
    return df_filtered


def filter_and_add_partner_rules_d1(df_d1):
    """
    Filter d1 participants and add partner rule information.
    
    This function combines filtering for valid partners and adding partner rules.
    
    Parameters
    ----------
    df_d1 : pandas.DataFrame
        DataFrame containing d1 participant data
        
    Returns
    -------
    pandas.DataFrame
        Filtered DataFrame with partner rule information added
    """
    # First filter for valid participants
    df_filtered = filter_d1_participants(df_d1)
    
    # Then add partner rules
    df_with_rules = add_partner_rules_d1(df_filtered)
    
    return df_with_rules


# new function
def add_test_drules_to_players_df(players_df, blocks_df, epsilon=2, first_test_phase=1, second_test_phase=3):
    """
    Add columns for test drules to players_df.
    
    Aggregates error values from blocks_df for each test phase and classifies decision rules
    based on total error across all blocks in the test phase.
    
    Parameters:
    -----------
    players_df : pandas.DataFrame
        DataFrame containing participant data with 'id' column
    blocks_df : pandas.DataFrame  
        DataFrame with columns 'id', 'phase', 'error_2d', 'error_1d_a', 'error_1d_b', 'error_1d'
    epsilon : float, default 2
        Threshold for classifying decision rule (total error <= epsilon)
    first_test_phase : int, default 1
        Phase number for first test
    second_test_phase : int, default 3  
        Phase number for second test
        
    Returns:
    --------
    pandas.DataFrame
        players_df with added columns: first_test_drule, second_test_drule,
        first_test_drule_gen, second_test_drule_gen
    """
    # Create a copy to avoid modifying the original
    players_df = players_df.copy()
    
    # Define phase mapping for column names
    phase_mapping = {
        first_test_phase: 'first',
        second_test_phase: 'second'
    }
    
    # For each participant
    for idx, player in players_df.iterrows():
        player_id = player['id']
        
        # For each test phase
        for phase_num in [first_test_phase, second_test_phase]:
            phase_label = phase_mapping[phase_num]
            
            # Get all blocks for this participant and phase
            phase_blocks = blocks_df[(blocks_df['id'] == player_id) & (blocks_df['phase'] == phase_num)]
            
            if len(phase_blocks) == 0:
                # No data for this phase, set to None
                players_df.loc[idx, f'{phase_label}_test_drule'] = None
                players_df.loc[idx, f'{phase_label}_test_drule_gen'] = None
                continue
            
            # Sum errors across all blocks in this phase
            total_error_2d = phase_blocks['drule_error_2d'].sum()
            total_error_1d_a = phase_blocks['drule_error_1d_a'].sum() 
            total_error_1d_b = phase_blocks['drule_error_1d_b'].sum()
            
            # Verify that at most one error is <= epsilon
            errors_below_threshold = sum([
                total_error_2d <= epsilon, 
                total_error_1d_a <= epsilon, 
                total_error_1d_b <= epsilon
            ])
            assert errors_below_threshold <= 1, f"Multiple errors <= epsilon for {player_id}, phase {phase_num}: error_2d={total_error_2d}, error_1d_a={total_error_1d_a}, error_1d_b={total_error_1d_b}, epsilon={epsilon}"
            
            # Classify decision rule based on lowest error <= epsilon
            if total_error_2d <= epsilon:
                drule = '2d'
                drule_gen = '2d'
            elif total_error_1d_a <= epsilon:
                drule = '1d_a'
                drule_gen = '1d'
            elif total_error_1d_b <= epsilon:
                drule = '1d_b' 
                drule_gen = '1d'
            else:
                drule = 'neither'
                drule_gen = 'neither'
            
            # Set the values
            players_df.loc[idx, f'{phase_label}_test_drule'] = drule
            players_df.loc[idx, f'{phase_label}_test_drule_gen'] = drule_gen
    
    # Convert to categorical with proper ordering
    for phase_label in ['first', 'second']:
        if f'{phase_label}_test_drule' in players_df.columns:
            players_df[f'{phase_label}_test_drule'] = pd.Categorical(
                players_df[f'{phase_label}_test_drule'], 
                categories=['2d', '1d_a', '1d_b', 'neither']
            )
        if f'{phase_label}_test_drule_gen' in players_df.columns:
            players_df[f'{phase_label}_test_drule_gen'] = pd.Categorical(
                players_df[f'{phase_label}_test_drule_gen'], 
                categories=['2d', '1d', 'neither']
            )
    
    return players_df


def add_partner_rule_extensions(players_df):
    """
    Add partner_rule_gen and partner_rule_rel columns to merged players dataframe.
    
    - partner_rule_gen: Generalized partner rule ('2d', '1d', 'neither') for all datasets
    - partner_rule_rel: Relative partner rule classification ('2d', 'other-1d', 'same-1d', 'neither')
    
    Parameters:
    -----------
    players_df : pandas.DataFrame
        Merged players dataframe from all datasets
        
    Returns:
    --------
    pandas.DataFrame
        Players dataframe with added partner_rule_gen and partner_rule_rel columns
    """
    players_df = players_df.copy()
    
    def generalize_partner_rule(partner_rule):
        """Convert specific partner_rule to generalized version"""
        if pd.isna(partner_rule) or partner_rule is None:
            return None
        if partner_rule in ['1d_a', '1d_b']:
            return '1d'
        elif partner_rule in ['2d', 'neither']:
            return partner_rule
        else:
            return None
    
    def compute_partner_rule_rel(own_rule, partner_rule):
        """Compute relative partner rule classification - only valid for trapped learners (1d rule in first test)"""
        # Only compute for trapped learners (those with 1d rule in first test)
        if pd.isna(own_rule) or own_rule not in ['1d_a', '1d_b']:
            return None
        
        if pd.isna(partner_rule) or partner_rule is None:
            return None
        if partner_rule == '2d':
            return '2d'
        elif partner_rule == 'neither':
            return 'neither'
        elif partner_rule in ['1d_a', '1d_b']:
            if own_rule == partner_rule:
                return 'same-1d'  # both 1d_a or both 1d_b
            else:
                return 'other-1d'  # one is 1d_a, other is 1d_b
        else:
            return None
    
    # Add partner_rule_gen for all datasets d1-d6
    players_df['partner_rule_gen'] = players_df['partner_rule'].apply(generalize_partner_rule)
    
    # Add partner_rule_rel for all datasets d1-d6
    # Remember -- this only really valid for trapped learners
    players_df['partner_rule_rel'] = players_df.apply(
        lambda row: compute_partner_rule_rel(row['first_test_drule'], row['partner_rule']),
        axis=1
    )
    
    # Convert to categorical with proper ordering (None/NaN values will remain as NaN)
    players_df['partner_rule_gen'] = pd.Categorical(
        players_df['partner_rule_gen'], 
        categories=['2d', '1d', 'neither']
    )
    
    players_df['partner_rule_rel'] = pd.Categorical(
        players_df['partner_rule_rel'], 
        categories=['2d', 'same-1d', 'other-1d', 'neither']
    )
    
    return players_df
