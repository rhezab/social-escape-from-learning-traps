import csv
from utils import load_json

def load_data(fname_or_dict):
    """
    Load and process behavioral data from a JSON file or dictionary.
    
    This function loads participant data and converts the task data structure
    from a dictionary format to a list format for easier processing.
    
    Parameters
    ----------
    fname_or_dict : str or dict
        Either a path to a JSON file containing the behavioral data,
        or a dictionary containing the data directly.
        
    Returns
    -------
    dict
        A dictionary where:
        - Keys are participant IDs
        - Values are dictionaries containing participant data with task_data
          converted to a list format
        
    Raises
    ------
    TypeError
        If fname_or_dict is neither a string nor a dictionary
        
    Notes
    -----
    The task_data structure is converted from:
    {phase: {trial: data}} to [[trial_data]]
    where each inner list contains the trial data for a phase
    """
    # load data
    if isinstance(fname_or_dict, str):
        data_list = load_json(fname_or_dict)
        data = {x['id']: x['data'] for x in data_list}
    elif isinstance(fname_or_dict, dict):
        data = fname_or_dict
    else:
        raise TypeError("Invalid input type. Expected a string or a dictionary.")

    # Convert task data to be indexed by ints, not strings
    for player_id, player_data in data.items():
        new_task_data = []
        for phase,phase_data in player_data['task_data'].items():
            new_task_data.append([trial_data for trial_data in phase_data.values()])
        player_data['task_data'] = new_task_data
    
    return data

# Helper functions
def write_to_csv(data, filename):
    """Write a list of tuples or list of lists to a CSV file."""
    if not filename.endswith('.csv'):
        filename += '.csv'
    with open(filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(data)

def load_csv(filename):
    """
    Load a CSV file into a list of tuples.
    
    Parameters:
    - filename: str, path to the CSV file
    
    Returns:
    - List of tuples containing the data from the CSV file
    """
    if not filename.endswith('.csv'):
        filename += '.csv'
    with open(filename, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        data = [tuple(row) for row in csv_reader]
    return data

def get_last_substring(input_str, delimiter='-'):
    substrings = input_str.split(delimiter)
    return substrings[-1]



