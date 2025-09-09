# Decision Rule Proportions Analysis

This analysis calculates and visualizes proportions of decision rule usage (2D, 1D, neither) across different experimental conditions and datasets for the dyadic learning trap experiment.

## Overview

The decision rule proportions analysis examines how participants use different decision strategies:
- **2D**: Optimal strategy using both relevant features
- **1D**: Suboptimal strategy using only one relevant feature (1D_a or 1D_b)
- **Neither**: No consistent rule detected

The analysis covers both learning phases and test phases, with special focus on how partner interactions affect decision rule adoption among "trapped learners" (participants who used 1D rules in the first test phase).

## Pipeline Structure

### Scripts

1. **`calculate_proportions.py`**
   - Calculates decision rule proportions for learning phases
   - Uses block-level data to track rule usage over time
   - Outputs: `decision_rule_proportions.csv`

2. **`calculate_test_proportions.py`**
   - Calculates decision rule proportions for test phases
   - Uses participant-level test columns directly
   - Outputs: `test_decision_rule_proportions.csv`, `test_decision_rule_proportions_info.md`

3. **`plot.py`**
   - Creates all visualizations
   - Three main plotting functions (see Figures section)
   - Outputs: Multiple SVG files

4. **`generate_outputs.py`**
   - Orchestrates the full pipeline
   - Runs all scripts in correct order
   - Provides progress reporting

### Data Flow

```
Preprocessed Data
├── players_df_all_filtered.csv
└── blocks_all_filtered.csv
                ↓
        calculate_proportions.py
                ↓
        decision_rule_proportions.csv
                ↓
        calculate_test_proportions.py
                ↓
        test_decision_rule_proportions.csv
                ↓
            plot.py
                ↓
    Multiple visualization files
```

## Usage

### Run Full Pipeline
```bash
python generate_outputs.py
```

### Run Individual Components
```bash
python calculate_proportions.py      # Learning phase proportions
python calculate_test_proportions.py # Test phase proportions  
python plot.py                       # All visualizations
```

## Output Files

### Data Files
- **`decision_rule_proportions.csv`**: Learning phase proportions by dataset, condition, and block
- **`test_decision_rule_proportions.csv`**: Test phase proportions by dataset, condition, and test phase type
- **`decision_rule_proportions_info.txt`**: Learning phase analysis documentation
- **`test_decision_rule_proportions_info.md`**: Test phase analysis documentation with tables

### Figures

#### 1. Learning Phase Visualization
- **File**: `drule-proportions-bar-plot.svg`
- **Description**: 6×4 grid showing decision rule proportions across learning blocks
- **Format**: Stacked bar plots (1D=blue, Neither=white, 2D=orange)
- **Conditions**: first, 2D partner, other-1D partner, asocial control

#### 2. First Test Phase Overview
- **File**: `first_test_overview.svg`
- **Description**: Single plot showing natural distribution of decision rules in first test phase
- **Population**: All participants before any filtering
- **Purpose**: Shows baseline decision rule distribution across datasets

#### 3. Trapped Learner Comparison
- **File**: `trapped_learner_comparison.svg`
- **Description**: 2×3 grid comparing trapped learners' performance
- **Bars**: T1 Baseline + T2 conditions (2D Partner, Other-1D Partner, Asocial)
- **Population**: Only participants who used 1D rules in first test phase
- **Purpose**: Shows intervention effects on trapped learners

#### 4. Original Test Phase Visualization (Reference)
- **File**: `test_decision_rule_proportions.svg`
- **Description**: Original combined test phase visualization
- **Note**: Kept for reference; may be less clear methodologically

## Methodology

### Experimental Conditions

#### Learning Phase Conditions
- **first**: First learning phase (all participants)
- **2d**: Participants paired with 2D-using partner
- **other_1d**: Participants paired with complementary 1D-using partner  
- **asocial**: Solo control condition

#### Test Phase Analysis
- **Population Filtering**: Test phase conditions (2d, other_1d, asocial) only include participants who used 1D rules in the first test phase ("trapped learners")
- **Baseline Comparison**: T1 baseline in trapped learner analysis represents the combined first test performance of all participants who later appear in T2 conditions

### Decision Rule Classification
- Rules are classified using error thresholds comparing participant decision matrices to ideal patterns
- **2D**: Uses both relevant features (optimal)
- **1D**: Uses single relevant feature (1D_a or 1D_b combined)
- **Neither**: No clear rule detected

### Data Processing
- **Learning phases**: Uses block-level decision matrices from `blocks_df`
- **Test phases**: Uses participant-level test columns (`first_test_drule_gen`, `second_test_drule_gen`, etc.)
- **Filtering**: Excludes participants who used external aid
- **Datasets**: Covers sim + datasets 1-5 (publication datasets)

## Key Analysis Features

### Phase Structure Handling
- Accounts for dataset-specific phase structures (dataset 5 has different phase indices)
- Uses `DATASET_PHASES` dictionary for consistent phase mapping
- Combines learning and test phases appropriately for each dataset

### Trapped Learner Focus
- Identifies participants who used 1D rules in first test phase
- Tracks how these participants perform in second test phase after different partner experiences
- Provides baseline comparison using the same participants' first test performance

### Statistical Validation
- T1 baseline in trapped learner analysis serves as sanity check (should be ~100% 1D)
- Weighted averages used when combining conditions
- Participant counts (N) displayed for all conditions

## Dependencies

### Required Data Files
- `../../preprocessing/outputs/players_df_all_filtered.csv`
- `../../preprocessing/outputs/blocks_all_filtered.csv`

### Python Packages
- `pandas`: Data manipulation
- `numpy`: Numerical operations  
- `matplotlib`: Visualization
- `os`, `sys`: File operations
- `datetime`: Timestamps

### Constants
- `constants.py`: Dataset definitions (`ALL_DATASETS`, `PUBLICATION_DATASETS`, `DATASET_PHASES`)

## Notes

- All figures use consistent color scheme and styling
- Participant counts (N) displayed on all visualizations
- Gray dashed lines separate different analysis phases
- Documentation files include detailed methodology and participant breakdowns
- Analysis handles missing data gracefully (e.g., asocial condition in dataset 5)