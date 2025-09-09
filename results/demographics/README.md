# Demographics Analysis

## Overview
Calculates participant demographics and filtering statistics for each dataset. 

## What it calculates

### For each dataset (sim, d1-d6):
1. **Raw data counts**: Total participants in original JSON files
2. **After initial processing**: 
   - Total participants after completion filtering
   - Number excluded for using external aid
   - Special handling for d1: filters duo participants to valid partner pairs only
3. **After external aid filtering**:
   - Final participant counts
   - Breakdown by game type (solo vs duo)

### Special analyses:
- **Dataset 1 (d1) dyad analysis**: Complete vs incomplete partner pairs

## Outputs
- `demographics_report.md`: Comprehensive markdown report with:
  - Summary table across all datasets
  - Detailed breakdown by dataset
  - Dataset 1 dyad statistics
  - Filtering methodology explanation

## Usage
```bash
cd results/demographics/
python calculate_demographics.py
```

## Dependencies
- Requires preprocessed data: `../../preprocessing/outputs/players_df_*.csv`
- Requires raw data files: `../../data/exp-*-data.json`, `../../simulation/outputs/simulated_data.json`