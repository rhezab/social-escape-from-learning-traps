# Social Escape from Learning Traps

Analysis pipeline for the paper "Decision rule inference limits social escape from learning traps." 

## Repository Structure

```
social-escape-learning-traps/
├── data/                    # Raw experimental data (JSON files)
├── preprocessing/           # Data preprocessing pipeline
├── results/                 # Analysis pipelines for each research question
├── simulation/              # ALCOVE-RL simulation framework
└── Root files              # Utilities, constants, and run all script. 
```

See individual folder READMEs for detailed documentation of each component.

## Paper figures

| Figure | Description | File Path |
|--------|-------------|-----------|
| Figure 2 | Decision rule distribution after initial individual learning phase. | `./results/drule_proportions/outputs/first_test_overview.pdf` |
| Figure 3 | Decision rule distribution for trapped learners after second learning phase.  | `./results/drule_proportions/outputs/trapped_learner_comparison.pdf` |
| Figure 4 | Exploration approach bad/good heatmaps for trapped learners who learned optimal rule during second learning phase. | `./results/reward_punishment_heatmap/outputs/trapped_to_optimal_grid.pdf` |
| Figure 5 | Effect of social decision rule inference success on observational learning success. | `./results/partner_prediction/outputs/partner_prediction_combined_figure.pdf` |
| Figure 6 | Decision rule distributions for each learning and test block, by second learning phase condition. | `./results/drule_proportions/outputs/drule-proportions-bar-plot.pdf` |

## Root Level Files

### Core Pipeline
- **`generate_all_outputs.py`** - Master script that runs the entire analysis pipeline:
  1. Simulation data generation
  2. Data preprocessing 
  3. All statistical analyses and visualizations
  
### Utilities
- **`constants.py`** - Global constants (dataset names, phase structures)
- **`stats.py`** - Statistical functions (permutation tests, bootstrap CIs)
- **`script_utils.py`** - Helper functions for running analysis scripts
- **`.gitignore`** - Standard Python/data science exclusions

## Quick Start

### Run Complete Pipeline
```bash
python generate_all_outputs.py
```

This will:
1. Generate simulated data using ALCOVE-RL model
2. Preprocess all experimental and simulated data
3. Run all analyses and generate figures

### Run Individual Components
Each component can also be run independently:

```bash
# Preprocessing only
cd preprocessing/
python generate_outputs.py

# Specific analysis
cd results/drule_proportions/
python generate_outputs.py
```

## Key Analyses

| Analysis | Location | Description |
|----------|----------|-------------|
| Demographics | `results/demographics/` | Participant counts and filtering statistics |
| Decision Rule Proportions | `results/drule_proportions/` | Decision rule proportions across conditions. |
| Test Phase Decision Rule Stats | `results/test_drule_statistics/` | Decision rule proportions at test phases, across conditions, with additional stats like CIs and significance testing comparisons. |
| Points Analysis | `results/points_per_block/` | Task performance of trapped learners |
| Partner Prediction Phase Analyses (Exp 5) | `results/partner_prediction/` | Effect of partner prediction phase accuracy on learning (Exp 5) |
| Exploration heatmaps | `results/reward_punishment_heatmap/` | Exploration pattern heatmap (proportion approached safe/dangeorus stimuli) |

## Data Flow

```
Raw JSON → Preprocessing → Filtered CSVs → Individual Analyses → Figures/Tables
                ↑
         Simulated Data
```

## Data Anonymization

**⚠️ Note: The anonymization scripts in `anonymization/` are included for transparency and reproducibility documentation, but they operate on the original unanonymized data files which are not included in this pub
 release.**

This repository contains anonymized experimental data. The original raw data (containing personally identifiable information) has been processed through an anonymization pipeline that:

## Dependencies

Standard Python data science stack:
- pandas, numpy
- matplotlib, seaborn
- scipy (for statistical tests)

## Acknowledgments

This analysis pipeline was developed with assistance from Claude Code (Anthropic). 
Files with more wholesale Claude generated code are marked with authorship comments at the top of the file.