"""
Anonymization Script for Social Escape Learning Traps Experiment Data
(written with substantial assistance from Claude Code)

This script generates anonymized versions of the raw experimental datasets by removing
personally identifiable information (PII) while preserving all data necessary for 
scientific analysis and reproduction.

DEMOGRAPHIC FIELDS IN RAW DATA:
The demographic_form contains the following fields:
- age (exact age as string, e.g., "33")
- gender (e.g., "Female", "Male", etc.)
- race (e.g., "Caucasian/White", etc.)
- hispanic (Yes/No)
- country (e.g., "United States")
- zipcode (5-digit US ZIP code)
- education_level (e.g., "High School Diploma", "Bachelor's Degree", etc.)
- household_income (ranges like "$40,000–$59,999")
- fluent_english (Yes/No)
- normal_vision (Yes/No)
- color_blind (Yes/No)3
- psychiatric_disorder (Yes/No)
- neurodevelopmental_disorder (Yes/No)
- learning_disability (Yes/No)

FIELDS TO BE REMOVED:
1. browser_fingerprint - Entire object (contains IP address, user agent, language)
2. recruitment_info - Entire object (contains prolific_id, session_id, study_id)
3. demographic_form.zipcode - ZIP code (directly identifying)

FIELDS PRESERVED:
- All task_data (experimental trials and responses)
- Performance metrics (points, bonuses, etc.)
- All demographics EXCEPT zipcode (age, gender, education, race, income, country, etc.)
- questionnaireQuestions (text responses about bee learning strategies - non-sensitive)
- Participant IDs (for internal data linking)
- Experimental conditions and parameters
- All behavioral/cognitive measures
- Timestamps (starttime, endtime, route_times)

TESTING PLAN:
To verify anonymization is complete, run test_anonymize.py which will:
1. Load each anonymized file
2. Verify NO participant has any of the following fields:
   - browser_fingerprint (entire object)
   - recruitment_info (entire object)
   - demographic_form.zipcode
3. Verify all expected fields ARE still present:
   - task_data with all trials
   - demographic_form (minus zipcode)
   - questionnaireQuestions
   - All performance metrics
4. Compare participant counts between original and anonymized
5. Generate a test report confirming all PII has been removed

Output:
- Creates anonymized JSON files in data/ directory (exp-*-data-anonymized.json)
- Generates summary report in anonymization/outputs/
- Preserves original file structure for compatibility with existing analysis code

Usage:
    python anonymize.py                 # Anonymize all data
    python test_anonymize.py            # Verify anonymization completeness
    
This will process all experiment files (exp-1-data.json through exp-6-data.json)
and create corresponding anonymized versions.
"""

import json
from pathlib import Path
from typing import Dict, Any
import copy
from datetime import datetime

def anonymize_participant_data(participant_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove PII from a single participant's data while preserving research data.
    
    Args:
        participant_data: Dictionary containing all data for one participant
        
    Returns:
        Anonymized copy of the participant data
    """
    # Create deep copy to avoid modifying original
    anon_data = copy.deepcopy(participant_data)
    
    # Remove browser fingerprint entirely (contains IP, user agent, etc.)
    if 'browser_fingerprint' in anon_data:
        del anon_data['browser_fingerprint']
    
    # Remove entire recruitment_info object (contains only prolific_id, session_id, study_id)
    if 'recruitment_info' in anon_data:
        del anon_data['recruitment_info']
    
    # Remove zipcode from demographics (keep country for analysis)
    if 'demographic_form' in anon_data:
        if 'zipcode' in anon_data['demographic_form']:
            del anon_data['demographic_form']['zipcode']
    
    # Timestamps are preserved (starttime, endtime, route_times) - useful for analysis
    # questionnaireQuestions are preserved as-is (non-sensitive bee learning responses)
    
    return anon_data

def anonymize_experiment_data(input_path: Path, output_path: Path) -> Dict[str, int]:
    """
    Anonymize an entire experiment dataset file.
    
    Args:
        input_path: Path to original JSON file
        output_path: Path where anonymized JSON will be saved
        
    Returns:
        Dictionary with statistics about the anonymization
    """
    print(f"Processing {input_path.name}...")
    
    # Load original data
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    # Track statistics
    stats = {
        'total_participants': len(data),
        'fields_removed': set()
    }
    
    # Anonymize each participant
    anonymized_data = []
    for participant in data:
        participant_id = participant['id']
        original_data = participant['data']
        
        # Track what fields are being removed
        if 'browser_fingerprint' in original_data:
            stats['fields_removed'].add('browser_fingerprint')
        if 'recruitment_info' in original_data:
            if 'prolific_id' in original_data['recruitment_info']:
                stats['fields_removed'].add('recruitment_info.prolific_id')
        if 'demographic_form' in original_data:
            if 'zipcode' in original_data['demographic_form']:
                stats['fields_removed'].add('demographic_form.zipcode')
        
        # No need to check text responses since they're non-sensitive
        
        # Anonymize the data
        anonymized_participant_data = anonymize_participant_data(original_data)
        
        # Preserve the structure
        anonymized_data.append({
            'id': participant_id,
            'data': anonymized_participant_data
        })
    
    # Save anonymized data
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(anonymized_data, f, separators=(',', ':'))
    
    print(f"  Saved to {output_path}")
    stats['fields_removed'] = list(stats['fields_removed'])
    return stats

def main():
    """Main function to anonymize all experiment datasets."""
    
    # Setup paths
    data_dir = Path('../data')
    output_dir = Path('../data')  # Save anonymized files in data directory
    reports_dir = Path('outputs')  # Save reports in outputs directory
    reports_dir.mkdir(exist_ok=True)
    
    # Process each experiment file
    all_stats = {}
    for exp_num in range(1, 7):
        input_file = data_dir / f'exp-{exp_num}-data.json'
        if not input_file.exists():
            print(f"Warning: {input_file} not found, skipping...")
            continue
            
        output_file = output_dir / f'exp-{exp_num}-data-anonymized.json'
        stats = anonymize_experiment_data(input_file, output_file)
        all_stats[f'exp-{exp_num}'] = stats
    
    # Generate summary report (save in outputs directory)
    report_path = reports_dir / 'anonymization_report.txt'
    with open(report_path, 'w') as f:
        f.write("Anonymization Report\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for exp_name, stats in all_stats.items():
            f.write(f"\n{exp_name}:\n")
            f.write(f"  Total participants: {stats['total_participants']}\n")
            f.write(f"  Fields removed: {', '.join(sorted(stats['fields_removed']))}\n")
        
        f.write("\n" + "=" * 50 + "\n")
        f.write("Note: questionnaireQuestions (text responses about bee learning) have been\n")
        f.write("preserved as they contain no sensitive information.\n")
    
    print(f"\nAnonymization complete! Report saved to {report_path}")

if __name__ == "__main__":
    main()