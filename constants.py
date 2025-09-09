#!/usr/bin/env python3
"""
Global constants for the Dyadic Learning Trap experiment analysis pipeline.

This file contains shared constants used across preprocessing and results analysis
to ensure consistency in dataset ordering and processing.
"""

# Dataset processing order - sim first, then numbered datasets
# This order ensures simulated data appears first in all outputs and analyses
ALL_DATASETS = ['sim', '1', '2', '3', '4', '5', '6']
REAL_DATASETS = ['1', '2', '3', '4', '5', '6']  # For operations that only apply to real data\
PUBLICATION_DATASETS = ['sim', '1', '2', '3', '4', '5']
PUBLICATION_REAL_DATASETS = ['1', '2', '3', '4', '5']

# Phase structure for each dataset
# Most datasets have 4 phases (0-3) with test phases at 1 and 3
# Dataset 5 has 5 phases (0-4) with test phases at 1 and 4, and a partner prediction phase at 2
DATASET_PHASES = {
    'sim': {'total': 4, 'learning': [0, 2], 'test': [1, 3]},
    '1': {'total': 4, 'learning': [0, 2], 'test': [1, 3]},
    '2': {'total': 4, 'learning': [0, 2], 'test': [1, 3]},
    '3': {'total': 4, 'learning': [0, 2], 'test': [1, 3]},
    '4': {'total': 4, 'learning': [0, 2], 'test': [1, 3]},
    '5': {'total': 5, 'learning': [0, 3], 'test': [1, 4], 'partner_prediction': [2]},
    '6': {'total': 4, 'learning': [0, 2], 'test': [1, 3]}
}

# Default phase structure (for most datasets)
DEFAULT_PHASES = {'total': 4, 'learning': [0, 2], 'test': [1, 3]}




def get_ordered_datasets(available_datasets):
    """
    Filter ALL_DATASETS to only include available datasets.
    hm n
    Parameters
    ----------
    available_datasets : set or list
        Set or list of dataset labels that are available
        
    Returns
    -------
    list
        Ordered list of dataset labels that exist in available_datasets
    """
    if isinstance(available_datasets, set):
        return [d for d in ALL_DATASETS if d in available_datasets]
    else:
        available_set = set(available_datasets)
        return [d for d in ALL_DATASETS if d in available_set]