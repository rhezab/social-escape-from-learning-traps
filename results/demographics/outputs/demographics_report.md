# Demographics Report

**Generated:** 2025-09-09 16:25:33

This report shows participant demographics for each dataset in the dyadic learning trap experiment.

## Filtering Process

The data goes through multiple filtering stages:

1. **Raw Data**: All participants in the original JSON files
2. **Initial Processing**: Filters for completed participants and valid partner pairs (for d1 duo condition only)
3. **External Aid Filtering**: Additionally excludes participants who reported using external assistance

Note: For dataset 1 (d1), the initial processing stage filters duo participants to only include those with valid partner pairs, while solo participants are retained regardless of partner status.

## Summary Table

| Dataset | Raw Data | After Initial Processing |  | After External Aid Filtering |  |  |
|---------|----------|--------------------------|--|------------------------------|--|--|
| | **Total** | **Total** | **External Aid** | **Total** | **Solo** | **Duo** |
| sim | 1000 | 1000 | 0 | 1000 | 250 | 750 |
| 1 | 236 | 227 | 7 | 220 | 117 | 103 |
| 2 | 188 | 188 | 4 | 184 | 44 | 140 |
| 3 | 206 | 206 | 7 | 199 | 94 | 105 |
| 4 | 200 | 199 | 7 | 192 | 53 | 139 |
| 5 | 197 | 197 | 5 | 192 | 0 | 192 |
| 6 | 100 | 99 | 6 | 93 | 93 | 0 |

## Detailed Breakdown

### Dataset sim

**Raw Data:**
- Total participants in JSON file: 1000

**After Initial Processing (completion + partner filtering for d1):**
- Total participants: 1000
- Participants using external aid: 0

**After External Aid Filtering:**
- Total participants: 1000
- Solo (asocial) condition: 250
- Duo (social) condition: 750

### Dataset 1

**Raw Data:**
- Total participants in JSON file: 236

**After Initial Processing (completion + partner filtering for d1):**
- Total participants: 227
- Participants using external aid: 7

**After External Aid Filtering:**
- Total participants: 220
- Solo (asocial) condition: 117
- Duo (social) condition: 103

### Dataset 2

**Raw Data:**
- Total participants in JSON file: 188

**After Initial Processing (completion + partner filtering for d1):**
- Total participants: 188
- Participants using external aid: 4

**After External Aid Filtering:**
- Total participants: 184
- Solo (asocial) condition: 44
- Duo (social) condition: 140

### Dataset 3

**Raw Data:**
- Total participants in JSON file: 206

**After Initial Processing (completion + partner filtering for d1):**
- Total participants: 206
- Participants using external aid: 7

**After External Aid Filtering:**
- Total participants: 199
- Solo (asocial) condition: 94
- Duo (social) condition: 105

### Dataset 4

**Raw Data:**
- Total participants in JSON file: 200

**After Initial Processing (completion + partner filtering for d1):**
- Total participants: 199
- Participants using external aid: 7

**After External Aid Filtering:**
- Total participants: 192
- Solo (asocial) condition: 53
- Duo (social) condition: 139

### Dataset 5

**Raw Data:**
- Total participants in JSON file: 197

**After Initial Processing (completion + partner filtering for d1):**
- Total participants: 197
- Participants using external aid: 5

**After External Aid Filtering:**
- Total participants: 192
- Solo (asocial) condition: 0
- Duo (social) condition: 192

### Dataset 6

**Raw Data:**
- Total participants in JSON file: 100

**After Initial Processing (completion + partner filtering for d1):**
- Total participants: 99
- Participants using external aid: 6

**After External Aid Filtering:**
- Total participants: 93
- Solo (asocial) condition: 93
- Duo (social) condition: 0

## Overall Totals

**Raw Data:**
- Total participants across all JSON files: 2127

**After Initial Processing:**
- Total participants across all datasets: 2116
- Total participants using external aid: 36
- Participants excluded in initial processing: 11

**After External Aid Filtering:**
- Total participants across all datasets: 2080
- Total solo (asocial) participants: 651
- Total duo (social) participants: 1429
- Participants excluded due to external aid: 36
- Total participants excluded across all stages: 47

## Experiment 1 Dyad Analysis

In experiment 1, all participants (both solo and duo conditions) are organized into dyads.
This analysis counts unique dyads in the final filtered sample.

**Dyad Statistics:**
- Total participants in experiment 1: 220
  - Solo condition: 117
  - Duo condition: 103
- Total unique dyads: 123
- Complete dyads (both members present): 97
- Incomplete dyads (one member filtered out): 26

**Methodology:**
Dyads are identified using sorted tuples of (participant_id, partner_id), ensuring the same
dyad identifier regardless of which member is accessed. Complete dyads have both members
present in the filtered dataset, while incomplete dyads have one member filtered out due to
completion requirements, partner pairing issues, or external aid usage.