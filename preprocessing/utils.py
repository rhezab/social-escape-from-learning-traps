import csv
import json

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

def load_json(file_path):
    """Load data from a JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def get_last_substring(input_str, delimiter='-'):
    """Get the last substring after splitting by delimiter."""
    substrings = input_str.split(delimiter)
    return substrings[-1] 

def time_elapsed(time1, time2):
    """Calculate time difference between two timestamps in seconds and nanoseconds."""
    # Convert times to nanoseconds
    time1_ns = time1['seconds'] * 1e9 + time1['nanoseconds']
    time2_ns = time2['seconds'] * 1e9 + time2['nanoseconds']

    # Find the difference
    diff_ns = abs(time2_ns - time1_ns)

    # Convert back to seconds and nanoseconds
    diff_s = diff_ns // 1e9
    diff_ns = diff_ns % 1e9

    return {'seconds': int(diff_s), 'nanoseconds': int(diff_ns)}

def find_first_index(lst, condition):
    """Find index of first element in list that satisfies condition."""
    return next((i for i, x in enumerate(lst) if condition(x)), None)

def time_on_page(data, page):
    """Calculate time spent on a specific page from route data."""
    route_order = data['route_order']
    route_times = data['route_times']
    cond = lambda x: x==page
    i = find_first_index(route_order, cond)
    time_enter_page = route_times[i]
    time_leave_page = route_times[i+1]
    return time_elapsed(time_enter_page, time_leave_page)