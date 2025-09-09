# Reward/Punishment Heatmap Analysis

## Overview
Generates approach/avoid heatmaps showing exploration patterns for participants who transition from trapped (1D rule) in phase 1 to optimal (2D rule) in phase 2. Demonstrates that social learning allows for vicarious learning without broad
exploration.

## Pipeline
The analysis pipeline consists of:
- **`generate_outputs.py`** - Main execution script
- **`make_trapped_to_optimal_grid.py`** - Generates 2x3 heatmap grid with marginals
- **`heatmap.py`** - Utility functions for reward/punishment counting and plotting


## Outputs
- 2x3 grid of heatmaps showing reward vs punishment patterns
- Marginal distributions for each heatmap
- SVG format for publication quality

## Usage
```bash
cd results/reward_punishment_heatmap/
python generate_outputs.py
```

## Dependencies
- Requires preprocessed data: `../../preprocessing/outputs/players_df_all_filtered.csv`, `blocks_all_filtered.csv`
- Uses constants from `../../constants.py` 