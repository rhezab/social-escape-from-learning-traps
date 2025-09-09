# Test Decision Rule Statistics Analysis

**Generated:** 2025-09-09 16:25:55

## Description

This analysis examines decision rule usage during test phases of the dyadic learning trap experiment. Two separate analyses are conducted:

1. **First Test Phase:** Decision rule counts and proportions for all participants
2. **Second Test Phase:** Trapped learner analysis comparing social conditions to asocial control using two-sided permutation significance tests

## Output Files

- `first_test_drule_statistics.csv`: First test phase statistics for all participants
- `second_test_drule_statistics.csv`: Second test phase statistics for trapped learners

## First Test Phase Results

**All participants by dataset and decision rule:**

| Dataset | Total | 2D % | 1D % | Neither % | 2D Count | 1D Count | Neither Count | 95% CI (2D) | 95% CI (1D) | 95% CI (Neither) |
|---------|-------|------|------|-----------|----------|----------|---------------|-------------|-------------|---------------|
| sim | 1000 | 30.0% | 63.5% | 6.5% | 300 | 635 | 65 | [0.272, 0.329] | [0.605, 0.664] | [0.050, 0.080] |
| 1 | 220 | 15.0% | 47.3% | 37.7% | 33 | 104 | 83 | [0.105, 0.200] | [0.409, 0.541] | [0.314, 0.445] |
| 2 | 184 | 12.5% | 47.8% | 39.7% | 23 | 88 | 73 | [0.082, 0.174] | [0.408, 0.549] | [0.326, 0.467] |
| 3 | 199 | 9.5% | 41.7% | 48.7% | 19 | 83 | 97 | [0.055, 0.141] | [0.347, 0.487] | [0.417, 0.558] |
| 4 | 192 | 7.8% | 39.6% | 52.6% | 15 | 76 | 101 | [0.042, 0.120] | [0.328, 0.464] | [0.458, 0.594] |
| 5 | 192 | 6.2% | 46.9% | 46.9% | 12 | 90 | 90 | [0.031, 0.099] | [0.396, 0.542] | [0.396, 0.542] |
| 6 | 93 | 10.8% | 34.4% | 54.8% | 10 | 32 | 51 | [0.054, 0.172] | [0.247, 0.441] | [0.452, 0.645] |

## Second Test Phase Results (Trapped Learners Only)

**Trapped learners by condition and decision rule:**

| Dataset | Condition | Total | 2D % | 1D % | Neither % | 2D Count | 1D Count | Neither Count | 95% CI (2D) | 95% CI (1D) | 95% CI (Neither) | p-perm (2D) | p-perm (1D) | p-perm (Neither) |
|---------|-----------|-------|------|------|-----------|----------|----------|---------------|-------------|-------------|------------------|-------------|-------------|------------------|
| sim | asocial | 152 | 20.4% | 77.0% | 2.6% | 31 | 117 | 4 | [0.145, 0.270] | [0.697, 0.836] | [0.007, 0.053] | - | - | - |
| sim | 2d | 159 | 92.5% | 0.0% | 7.5% | 147 | 0 | 12 | [0.881, 0.962] | [0.000, 0.000] | [0.038, 0.119] | 0.000 | 0.000 | 0.067 |
| sim | other-1d | 159 | 94.3% | 0.0% | 5.7% | 150 | 0 | 9 | [0.906, 0.975] | [0.000, 0.000] | [0.025, 0.094] | 0.000 | 0.000 | 0.257 |
| 1 | asocial | 57 | 1.8% | 93.0% | 5.3% | 1 | 53 | 3 | [0.000, 0.053] | [0.860, 0.982] | [0.000, 0.123] | - | - | - |
| 1 | 2d | 10 | 10.0% | 90.0% | 0.0% | 1 | 9 | 0 | [0.000, 0.300] | [0.700, 1.000] | [0.000, 0.000] | 0.280 | 1.000 | 1.000 |
| 1 | other-1d | 8 | 0.0% | 100.0% | 0.0% | 0 | 8 | 0 | [0.000, 0.000] | [1.000, 1.000] | [0.000, 0.000] | 1.000 | 1.000 | 1.000 |
| 2 | asocial | 19 | 0.0% | 89.5% | 10.5% | 0 | 17 | 2 | [0.000, 0.000] | [0.737, 1.000] | [0.000, 0.263] | - | - | - |
| 2 | 2d | 27 | 33.3% | 63.0% | 3.7% | 9 | 17 | 1 | [0.148, 0.519] | [0.444, 0.815] | [0.000, 0.111] | 0.006 | 0.086 | 0.557 |
| 2 | other-1d | 24 | 8.3% | 83.3% | 8.3% | 2 | 20 | 2 | [0.000, 0.208] | [0.667, 0.958] | [0.000, 0.208] | 0.491 | 0.669 | 1.000 |
| 3 | asocial | 30 | 3.3% | 93.3% | 3.3% | 1 | 28 | 1 | [0.000, 0.100] | [0.833, 1.000] | [0.000, 0.100] | - | - | - |
| 3 | 2d | 28 | 35.7% | 46.4% | 17.9% | 10 | 13 | 5 | [0.179, 0.536] | [0.286, 0.643] | [0.036, 0.321] | 0.002 | 0.000 | 0.099 |
| 3 | other-1d | 25 | 20.0% | 72.0% | 8.0% | 5 | 18 | 2 | [0.040, 0.360] | [0.520, 0.880] | [0.000, 0.200] | 0.078 | 0.065 | 0.590 |
| 4 | asocial | 19 | 0.0% | 100.0% | 0.0% | 0 | 19 | 0 | [0.000, 0.000] | [1.000, 1.000] | [0.000, 0.000] | - | - | - |
| 4 | 2d | 25 | 80.0% | 20.0% | 0.0% | 20 | 5 | 0 | [0.640, 0.960] | [0.040, 0.360] | [0.000, 0.000] | 0.000 | 0.000 | 1.000 |
| 4 | other-1d | 32 | 31.2% | 68.8% | 0.0% | 10 | 22 | 0 | [0.156, 0.469] | [0.531, 0.844] | [0.000, 0.000] | 0.007 | 0.008 | 1.000 |
| 5 | asocial | 0 | 0.0% | 0.0% | 0.0% | 0 | 0 | 0 | [nan, nan] | [nan, nan] | [nan, nan] | - | - | - |
| 5 | 2d | 44 | 56.8% | 38.6% | 4.5% | 25 | 17 | 2 | [nan, nan] | [nan, nan] | [nan, nan] | nan | nan | nan |
| 5 | other-1d | 21 | 28.6% | 71.4% | 0.0% | 6 | 15 | 0 | [nan, nan] | [nan, nan] | [nan, nan] | nan | nan | nan |
| 6 | asocial | 32 | 59.4% | 21.9% | 18.8% | 19 | 7 | 6 | [0.406, 0.750] | [0.094, 0.375] | [0.062, 0.344] | - | - | - |
| 6 | 2d | 0 | 0.0% | 0.0% | 0.0% | 0 | 0 | 0 | [nan, nan] | [nan, nan] | [nan, nan] | nan | nan | nan |
| 6 | other-1d | 0 | 0.0% | 0.0% | 0.0% | 0 | 0 | 0 | [nan, nan] | [nan, nan] | [nan, nan] | nan | nan | nan |

## Methodology

### Trapped Learners
Participants showing 1d rule in first test phase. Second test analysis limited to these participants only.

### Conditions (Second Test)
- **asocial**: Trapped learners in solo games (control)
- **2d**: Trapped learners paired with 2d partner
- **other-1d**: Trapped learners paired with different 1d partner

### Statistical Analysis
Two complementary statistical approaches (10,000 iterations each):

- **Permutation tests**: Compare social conditions to asocial control by pooling participants and randomly reassigning to conditions
- **Bootstrap confidence intervals**: Estimate uncertainty in proportion estimates using 95% percentile-based CIs

## Summary Statistics

- **First test rows:** 21
- **Second test rows:** 63
- **Datasets analyzed:** ['1', '2', '3', '4', '5', '6', 'sim']

### Trapped Learners by Dataset

| Dataset | Trapped Learners | Total | Percentage |
|---------|------------------|-------|------------|
| sim | 635 | 1000 | 63.5% |
| 1 | 104 | 220 | 47.3% |
| 2 | 88 | 184 | 47.8% |
| 3 | 83 | 199 | 41.7% |
| 4 | 76 | 192 | 39.6% |
| 5 | 90 | 192 | 46.9% |
| 6 | 32 | 93 | 34.4% |
