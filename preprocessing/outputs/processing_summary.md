# Dyadic Learning Trap Experiment - Data Processing Summary

Generated: 2025-09-09 16:25:30

## Participant Totals

### Raw Data Counts

| Dataset | Participants |
|---------|--------------|
| sim | 1000 |
| 1 | 236 |
| 2 | 188 |
| 3 | 206 |
| 4 | 200 |
| 5 | 197 |
| 6 | 100 |
| **Total** | **2127** |

### Processed Participant Counts

| Dataset | Participants | Blocks |
|---------|--------------|--------|
| sim | 1000 | 12000 |
| 1 | 227 | 2820 |
| 2 | 188 | 2256 |
| 3 | 206 | 2472 |
| 4 | 199 | 2388 |
| 5 | 197 | 3152 |
| 6 | 99 | 1188 |
| **Total** | **2116** | **26276** |

### Final Filtered Counts (Excluding External Aid)

| Dataset | Participants | Blocks |
|---------|--------------|--------|
| sim | 1000 | 12000 |
| 1 | 220 | 2640 |
| 2 | 184 | 2208 |
| 3 | 199 | 2388 |
| 4 | 192 | 2304 |
| 5 | 192 | 3072 |
| 6 | 93 | 1116 |
| **Total** | **2080** | **25728** |

Participants excluded due to external aid: 36

## External Aid Usage Breakdown

| Dataset | Used Aid | No Aid | % Using Aid |
|---------|----------|--------|-------------|
| sim | 0 | 1000 | 0.0% |
| 1 | 7 | 220 | 3.1% |
| 2 | 4 | 184 | 2.1% |
| 3 | 7 | 199 | 3.4% |
| 4 | 7 | 192 | 3.5% |
| 5 | 5 | 192 | 2.5% |
| 6 | 6 | 93 | 6.1% |
| **Total** | **36** | **2080** | **1.7%** |

## Preliminary Results: Decision Rule Analysis

*Note: Analysis based on filtered data (excluding external aid users)*

### First Test Phase Decision Rules

| Dataset | 2d | 1d | neither | Total |
|---------|----|----|---------|---------
| sim | 300 (30.0%) | 635 (63.5%) | 65 (6.5%) | 1000 |
| 1 | 33 (15.0%) | 104 (47.3%) | 83 (37.7%) | 220 |
| 2 | 23 (12.5%) | 88 (47.8%) | 73 (39.7%) | 184 |
| 3 | 19 (9.5%) | 83 (41.7%) | 97 (48.7%) | 199 |
| 4 | 15 (7.8%) | 76 (39.6%) | 101 (52.6%) | 192 |
| 5 | 12 (6.2%) | 90 (46.9%) | 90 (46.9%) | 192 |
| 6 | 10 (10.8%) | 32 (34.4%) | 51 (54.8%) | 93 |
| **Total** | **412 (19.8%)** | **1108 (53.3%)** | **560 (26.9%)** | **2080** |

### Second Test Phase Decision Rules

| Dataset | 2d | 1d | neither | Total |
|---------|----|----|---------|---------
| sim | 711 (71.1%) | 256 (25.6%) | 33 (3.3%) | 1000 |
| 1 | 45 (20.5%) | 119 (54.1%) | 56 (25.5%) | 220 |
| 2 | 41 (22.3%) | 86 (46.7%) | 57 (31.0%) | 184 |
| 3 | 50 (25.1%) | 78 (39.2%) | 71 (35.7%) | 199 |
| 4 | 65 (33.9%) | 82 (42.7%) | 45 (23.4%) | 192 |
| 5 | 47 (24.5%) | 79 (41.1%) | 66 (34.4%) | 192 |
| 6 | 39 (41.9%) | 14 (15.1%) | 40 (43.0%) | 93 |
| **Total** | **998 (48.0%)** | **714 (34.3%)** | **368 (17.7%)** | **2080** |

## Output Files

The following files have been generated:

- **players_df_all.csv**: All participants (before filtering)
- **blocks_all.csv**: All blocks (before filtering)
- **players_df_all_filtered.csv**: Participants excluding external aid users
- **blocks_all_filtered.csv**: Blocks excluding external aid users
- **processing_summary.md**: This summary file

## Data Structure

**Players dataframe columns** (24 total):
`bonus_points, dataset, external_aid, first_half_points, first_test_drule, first_test_drule_gen, first_test_points, game_type, id, partner_id, partner_judgement, partner_points, partner_predictions_correct, partner_rule, partner_rule_gen, partner_rule_rel, partner_short_id, points, second_half_points, second_test_drule, second_test_drule_gen, second_test_points, short_id, time_on_share_page`

**Blocks dataframe columns** (17 total):
`block, dataset, drule, drule_error_1d, drule_error_1d_a, drule_error_1d_b, drule_error_2d, drule_gen, game_type, id, partner_predictions_correct, partner_predictions_missing, phase, points, punishment_approached, rewards_approached, short_id`
