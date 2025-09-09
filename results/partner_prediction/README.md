# Partner Prediction Analysis

## Overview
Analyzes how partner prediction accuracy affects learning outcomes for trapped learners. Tests whether accurately predicting partner behavior improves escape from learning traps.

## Pipeline
- **`generate_outputs.py`** → **`calculate_stats.py`** → **`plot.py`**

## Outputs
- `partner_prediction_results.csv` - Statistical results
- `partner_prediction_results.json` - JSON format data
- `partner_prediction_significance_results.md` - Analysis summary with significance tests
- `partner_prediction_combined_figure.svg/.pdf` - Two-panel figure:
  - Panel A: Histogram of partner prediction performance
  - Panel B: Learning outcomes by prediction performance (2D vs other-1D partners)

## Usage
```bash
cd results/partner_prediction/
python generate_outputs.py
```

## Dependencies
- Requires preprocessed data: `../../preprocessing/outputs/players_df_all_filtered.csv`
- Uses constants from `../../constants.py`
- Only analyzes dataset 5 (which includes partner prediction questions) 