# Points Per Block Analysis

## Overview
Analyzes and visualizes participant performance (points per block) across experimental phases, focusing on trapped learners and comparing their performance under different social learning conditions (asocial control, 2D partner, other-1D partner).

## Pipeline
The analysis pipeline consists of:
- **`generate_outputs.py`** - Main execution script
- **`plot_points_per_block.py`** - Core analysis generating both per-block and cumulative plots

## Outputs
- `points_per_block_by_dataset.svg` - Grid of per-block points across datasets
- `cumulative_points_by_dataset.svg` - Grid of cumulative points across datasets  
- `analysis_summary.md` - Statistical summary of findings
- `trapped_learners_stats.csv` - Detailed statistics by condition
- `trapped_learners_stats_summary.md` - Summary of trapped learner statistics

## Usage
```bash
cd results/points_per_block/
python generate_outputs.py
```

## Dependencies
- Requires preprocessed data: `../../preprocessing/outputs/players_df_all_filtered.csv`, `blocks_all_filtered.csv`
- Uses constants from `../../constants.py` 