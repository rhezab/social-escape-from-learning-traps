# Data Anonymization Pipeline

This directory contains scripts to anonymize the raw experimental data by removing personally identifiable information (PII) while preserving all data necessary for scientific analysis.

## Overview

The anonymization process removes only the minimal set of truly identifying fields while preserving all research-relevant data including timestamps, demographics (except zipcode), and all experimental measurements.

## Fields Removed

The following fields are removed during anonymization:

1. **`browser_fingerprint`** - Entire object containing:
   - IP address
   - User agent string
   - Browser language
   - WebDriver status

2. **`recruitment_info`** - Entire object containing:
   - `prolific_id` - Prolific platform identifier
   - `session_id` - Prolific session identifier
   - `study_id` - Prolific study identifier

3. **`demographic_form.zipcode`** - US ZIP code (directly identifying)

## Fields Preserved

All other fields are preserved, including:

- **Participant IDs** - Random UUIDs used only for internal data linking
- **Task data** - All experimental trials and responses
- **Demographics** - Age, gender, race, education, income, country, etc. (except zipcode)
- **Performance metrics** - Points, bonuses, accuracy, etc.
- **Timestamps** - starttime, endtime, route_times (useful for analysis, not identifying)
- **Questionnaire responses** - Text responses about bee learning strategies (non-sensitive)
- **Experimental conditions** - All parameters and settings
- **Partner information** - Partner IDs and relationships (using anonymous IDs)

## Usage

### Quick Start

Run the complete anonymization pipeline:

```bash
cd anonymization/
python generate_outputs.py
```

This will:
1. Anonymize all experimental datasets (exp-1 through exp-6)
2. Run verification tests
3. Generate reports

### Individual Scripts

```bash
# Run just the anonymization
python anonymize.py

# Run just the verification tests
python test_anonymize.py
```

## Output Files

- **`data/exp-{1-6}-data-anonymized.json`** - Anonymized datasets ready for public release
- **`anonymization/outputs/anonymization_report.txt`** - Summary of the anonymization process
- **`anonymization/outputs/test_report.txt`** - Verification report confirming PII removal

The original raw data files (`data/exp-{1-6}-data.json`) are excluded from git via .gitignore.

## Verification

The `test_anonymize.py` script performs comprehensive verification:

1. **Confirms removal** of all PII fields (browser_fingerprint, recruitment_info, zipcode)
2. **Confirms preservation** of all research-relevant fields
3. **Validates participant counts** match between original and anonymized
4. **Generates detailed report** of any issues found

A successful test run indicates the anonymized datasets are ready for public release.

## Data Structure

The anonymized JSON files maintain the exact same structure as the originals:

```json
[
  {
    "id": "participant-uuid",
    "data": {
      "task_data": {...},
      "demographic_form": {
        "age": "33",
        "gender": "Female",
        "country": "United States",
        // zipcode removed
        ...
      },
      // browser_fingerprint removed
      // recruitment_info removed
      ...
    }
  },
  ...
]
```

## Dependencies

- Python 3.x
- Standard library only (json, pathlib, datetime, copy)
