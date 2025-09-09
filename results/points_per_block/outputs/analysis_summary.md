# Points Per Block Analysis Summary

Generated: 2025-09-09 16:25:44

## Overview

This analysis examines participant performance (points per block) across different phases of the dyadic learning trap experiment, focusing on trapped learners and their performance under different social learning conditions.

## Conditions Analyzed

### all_trapped
- Participants: 1108
- Blocks: 13656
- Datasets represented: ['1', '2', '3', '4', '5', '6', 'sim']

### asocial
- Participants: 309
- Blocks: 3708
- Datasets represented: ['1', '2', '3', '4', '6', 'sim']

### 2d_partner
- Participants: 333
- Blocks: 4172
- Datasets represented: ['1', '2', '3', '4', '5', 'sim']

### other_1d_partner
- Participants: 309
- Blocks: 3792
- Datasets represented: ['1', '2', '3', '4', '5', 'sim']

## Output Files

- `points_per_block_by_dataset.png`: Grid plot showing points per block for each dataset (excluding partner prediction phase for dataset 5)
- `cumulative_points_by_dataset.png`: Grid plot showing cumulative points across blocks for each dataset

Each grid contains 6 subplots (2x3), one for each dataset in PUBLICATION_DATASETS.
Dataset-specific phase structures are used (e.g., dataset 5 has different learning/test phases).

## Performance Benchmarks

- **1D Rule Performance**: 8 points per block (gray dashed line)
- **2D Rule Performance**: 12 points per block (gray dashed line)

## Dataset-Specific Phase Structures

Different datasets have different phase structures:

### Dataset sim
- Total phases: 4
- Learning phases: [0, 2]
- Test phases: [1, 3]

### Dataset 1
- Total phases: 4
- Learning phases: [0, 2]
- Test phases: [1, 3]

### Dataset 2
- Total phases: 4
- Learning phases: [0, 2]
- Test phases: [1, 3]

### Dataset 3
- Total phases: 4
- Learning phases: [0, 2]
- Test phases: [1, 3]

### Dataset 4
- Total phases: 4
- Learning phases: [0, 2]
- Test phases: [1, 3]

### Dataset 5
- Total phases: 5
- Learning phases: [0, 3]
- Test phases: [1, 4]
- Partner prediction phase: [2]

## Block Labels

- **L1-L4**: Learning blocks
- **T1-T2**: Test blocks
- Phase numbers determine which learning/test phase blocks belong to
