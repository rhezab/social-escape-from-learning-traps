# Preprocessing Pipeline

This directory contains the data processing pipeline that transforms raw experimental JSON files into structured CSV dataframes for analysis.

## Overview

The preprocessing pipeline converts raw experimental data from six datasets (d1-d6) into two main structured dataframes:

- **`players_df_all.csv`**: Participant-level data with performance metrics and decision rules
- **`blocks_all.csv`**: Block-level data with decision matrices and error calculations
- **`players_df_all_filtered.csv`**: Filtered participant-level data (external aid users removed)
- **`blocks_all_filtered.csv`**: Filtered block-level data (external aid users removed)

The unfiltered versions have already been filtered for valid partners (in the d1 case), while the filtered versions additionally exclude participants who used external aid.

## Data Flow

```
Raw JSON Files → Dataset Loading → Players DataFrames → Blocks DataFrames → Test Rules → Merging → External Aid Filtering → Final CSV Files
```

### Pipeline Steps

1. **Load Raw Data** (`dataset.py`)
   - Load JSON files from `../data/` directory
   - Convert task data structure for easier processing

2. **Create Players DataFrames** (`players_df.py`)
   - Extract participant-level information
   - Apply dataset-specific filtering rules

3. **Create Blocks DataFrames** (`blocks_df.py`)
   - Generate block-level decision matrices
   - Calculate error metrics for rule classification

4. **Add Test Decision Rules** (`players_df.py`)
   - Classify participant strategies during test phases
   - Add decision rule columns to players dataframes

5. **Merge DataFrames**
   - Combine all datasets into unified dataframes
   - Add dataset labels and partner rule extensions

6. **Filter External Aid Users**
   - Remove participants who reported using external assistance
   - Create filtered versions of both dataframes

## Key Scripts

### `generate_outputs.py`
Main execution script that runs the entire pipeline. Run with:
```bash
python generate_outputs.py
```

### Core Modules

- **`dataset.py`**: Raw data loading and preprocessing
- **`players_df.py`**: Participant-level dataframe creation and filtering
- **`blocks_df.py`**: Block-level dataframe creation and decision rule classification
- **`utils.py`**: Helper functions for data manipulation

## Dataset Versions

The pipeline processes six experimental datasets:

| Dataset | Description | Phases | Test Phases | Special Features |
|---------|-------------|--------|-------------|------------------|
| d1 | Baseline | 4 | 1, 3 | Partner validation required |
| d2 | Version 2 | 4 | 1, 3 | Prolific ID whitelist |
| d3 | Score sharing | 4 | 1, 3 | Partner judgement questions |
| d4 | Rule sharing | 4 | 1, 3 | Rule sharing, time tracking |
| d5 | Partner prediction | 5 | 1, 4 | Partner prediction phase (2) |
| d6 | Full information | 4 | 1, 3 | - |

## Output Files

### `players_df_all.csv` - Participant-Level Data

Contains one row per participant with the following columns:

#### Core Information
- **`dataset`**: Dataset identifier (1-6)
- **`id`**: Full participant identifier
- **`short_id`**: Shortened participant ID for display
- **`prolific_id`**: Prolific recruitment platform ID
- **`game_type`**: 'solo' or 'duo'

#### Performance Metrics
- **`points`**: Total points earned across all phases
- **`first_test_points`**: Points earned during first test phase
- **`second_test_points`**: Points earned during second test phase
- **`first_half_points`**: Cumulative points at end of first half
- **`second_half_points`**: Points earned in second half
- **`bonus_points`**: Additional bonus points (d2-d6)

#### Decision Rules
- **`first_test_drule`**: Decision rule in first test ('2d', '1d_a', '1d_b', 'neither')
- **`second_test_drule`**: Decision rule in second test ('2d', '1d_a', '1d_b', 'neither')
- **`first_test_drule_gen`**: Generalized first test rule ('2d', '1d', 'neither')
- **`second_test_drule_gen`**: Generalized second test rule ('2d', '1d', 'neither')

#### Partner Information
- **`partner_id`**: Partner's full ID (d1 only)
- **`partner_short_id`**: Partner's shortened ID (d1 only)
- **`partner_points`**: Partner's total points (d1 only)
- **`partner_rule`**: Partner's first test decision rule
- **`partner_rule_gen`**: Partner's generalized decision rule ('2d', '1d', 'neither')
- **`partner_rule_rel`**: Relative partner rule classification ('2d', 'same-1d', 'other-1d', 'neither')
  - **Note**: Only valid for trapped learners (participants with 1d rule in first test)
- **`partner_judgement`**: Subjective partner performance rating (d3-d4)

#### Additional Metrics
- **`external_aid`**: Whether participant used external assistance
- **`partner_predictions_correct`**: Correct partner predictions (d5 only)
- **`time_on_share_page`**: Time spent on sharing page in seconds (d4 only)

### `blocks_all.csv` - Block-Level Data

Contains one row per block per participant with the following columns:

#### Block Information
- **`dataset`**: Dataset identifier (1-6)
- **`phase`**: Phase number (0-indexed)
- **`block`**: Block number within phase (0-indexed)
- **`id`**: Participant identifier
- **`short_id`**: Shortened participant ID
- **`game_type`**: 'solo' or 'duo'

#### Performance
- **`points`**: Points earned in this specific block

#### Decision Rules and Errors
- **`drule`**: Block-level decision rule ('2d', '1d_a', '1d_b', 'neither')
- **`drule_gen`**: Generalized decision rule ('2d', '1d', 'neither')
- **`drule_error_2d`**: Error when compared to optimal 2d strategy (0-16 scale)
- **`drule_error_1d_a`**: Error when compared to 1d_a strategy (0-16 scale)
- **`drule_error_1d_b`**: Error when compared to 1d_b strategy (0-16 scale)
- **`drule_error_1d`**: Minimum of 1d_a and 1d_b errors

## Decision Rule Classification

Participants' strategies are classified by comparing their decision matrices to ideal patterns:

### Rule Types
- **2d**: Uses both relevant features (optimal strategy)
- **1d_a**: Uses only first relevant feature
- **1d_b**: Uses only second relevant feature  
- **neither**: No clear rule detected

### Error Calculation
- Decision matrices are 2x2 grids representing choices based on relevant features
- Error = Manhattan distance between participant matrix and ideal matrix
- Block-level classification: error ≤ 1 (epsilon)
- Test phase classification: total error ≤ 2 (epsilon)

### Partner Rule Relationships
For trapped learners (1d rule in first test), partner relationships are classified as:
- **2d**: Partner uses optimal strategy
- **same-1d**: Partner uses same 1d rule (both 1d_a or both 1d_b)
- **other-1d**: Partner uses different 1d rule (one 1d_a, other 1d_b)
- **neither**: Partner shows no clear rule

## Filtering Steps

### Dataset-Specific Filtering

#### d1 (Baseline)
- Only completed Prolific participants
- **Partner validation**: Only include participants whose partner also exists in dataset and has matching game type
- Excludes participant '00d725e7-19d0-479a-971e-7f9a2f966d76-p249'

#### d2 (Version 2)
- **Prolific ID whitelist**: Only include specific participants or those who completed
- Whitelist: '576ad50bc8a90000010d99c8', '61267e282a445af9eb95beae', '6331b9456956d880a1e4632f'

#### d3-d6
- Only completed Prolific participants

### External Aid Filtering
- **External aid exclusion**: Remove participants who reported using external assistance (calculators, notes, etc.)
- Applied to both players and blocks dataframes to create filtered versions
- This is the only difference between unfiltered and filtered versions

### Final Counts
- **Original**: 1,116 participants, 14,276 blocks
- **Filtered**: 1,080 participants, 13,728 blocks
- **Excluded**: 36 participants who used external aid

## Phase Structure

### Standard (d1-d4, d6)
- Phase 0: Training (4 blocks)
- Phase 1: **First test** (2 blocks)
- Phase 2: Training (4 blocks)  
- Phase 3: **Second test** (2 blocks)

### Extended (d5)
- Phase 0: Training (4 blocks)
- Phase 1: **First test** (2 blocks)
- Phase 2: **Partner prediction** (4 blocks)
- Phase 3: Training (4 blocks)
- Phase 4: **Second test** (2 blocks)

## Usage

To run the complete preprocessing pipeline:

```bash
cd preprocessing/
python generate_outputs.py
```

This will generate all output files in the `outputs/` directory:
- `players_df_all.csv` and `players_df_all_filtered.csv`
- `blocks_all.csv` and `blocks_all_filtered.csv`
- `processing_summary.txt`

## Dependencies

- pandas: Data manipulation and analysis
- numpy: Numerical operations for decision matrices
- json: JSON data loading
- csv: CSV file operations