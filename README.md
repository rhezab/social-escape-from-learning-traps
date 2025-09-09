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

### Removed Fields
- **Browser fingerprints** (IP addresses, user agents, browser info)
- **Recruitment identifiers** (Prolific IDs, session IDs, study IDs) 
- **ZIP codes** (directly identifying demographic information)

### Preserved Fields
- **Demographics** (age, gender, race, education, income, country - except ZIP codes)
- **Timestamps** (useful for analysis, not personally identifying)
- **All task data** (trials, responses, performance metrics)

## Dependencies

Standard Python data science stack:
- pandas, numpy
- matplotlib, seaborn
- scipy (for statistical tests)

## Acknowledgments

This analysis pipeline was developed with assistance from Claude Code (Anthropic). 
Files with more wholesale Claude generated code are marked with authorship comments at the top of the file.