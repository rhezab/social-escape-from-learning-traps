# Test Decision Rule Statistics Analysis

## Overview
Calculates comprehensive statistics for decision rule usage during test phases, with two-sided permutation significance testing and bootstrap confidence intervals to compare social vs asocial learning conditions.

## Analysis Components

### First Test Phase Analysis
- **Who**: All participants after individual learning
- **What**: Counts and proportions following each decision rule (2d, 1d, neither) with bootstrap CIs
- **When**: First test phase (after initial individual learning)

### Second Test Phase Analysis (Trapped Learners Only)
- **Who**: Participants showing 1D rule in first test ("trapped learners")  
- **What**: Decision rule changes by condition with statistical testing
- **Conditions analyzed**:
  - **Solo (asocial)**: Control baseline with bootstrap CIs
  - **Duo (social)** with permutation tests vs. solo:
    - Partner with 2D rule
    - Partner with other-1D rule  
    - Partner with same-1D rule

## Outputs
- `first_test_drule_statistics.csv`: Rule usage after individual learning with bootstrap CIs
- `second_test_drule_statistics.csv`: Rule changes for trapped learners with permutation p-values and bootstrap CIs
- `test_drule_statistics_info.md`: Detailed analysis report with methodology

## Usage
```bash
cd results/test_drule_statistics/
python generate_outputs.py
```

## Dependencies
- Requires preprocessed data: `../../preprocessing/outputs/players_df_all_filtered.csv`
- Uses statistical functions from `../../stats.py`