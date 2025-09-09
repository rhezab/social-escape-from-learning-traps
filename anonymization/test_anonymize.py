"""
Test Script for Verifying Data Anonymization
(written with substantial assistance from Claude Code)

This script verifies that the anonymization process has correctly removed all
personally identifiable information while preserving necessary research data.

TESTING PLAN:
1. Load each anonymized file from anonymization/outputs/
2. Verify NO participant has any of the following fields:
   - browser_fingerprint (entire object)
   - recruitment_info (entire object)
   - demographic_form.zipcode
3. Verify all expected fields ARE still present:
   - task_data with all trials
   - demographic_form (minus zipcode)
   - questionnaireQuestions
   - All performance metrics (points, bonuses, etc.)
   - Timestamps (starttime, endtime, route_times)
   - Participant IDs (id, seedID, user_data.playerId, etc.)
4. Compare participant counts between original and anonymized
5. Generate a test report confirming all PII has been removed

Usage:
    python test_anonymize.py
    
This will test all anonymized files and generate a verification report.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime


def check_field_exists(data: Dict, field_path: str) -> bool:
    """
    Check if a field exists in nested dictionary.
    
    Args:
        data: Dictionary to check
        field_path: Dot-separated path to field (e.g., "demographic_form.zipcode")
        
    Returns:
        True if field exists, False otherwise
    """
    parts = field_path.split('.')
    current = data
    
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return False
    return True


def verify_removed_fields(participant_data: Dict) -> List[str]:
    """
    Check if any fields that should be removed are still present.
    
    Args:
        participant_data: Data for one participant
        
    Returns:
        List of field names that should have been removed but are still present
    """
    fields_that_should_be_removed = [
        'browser_fingerprint',
        'recruitment_info',
        'demographic_form.zipcode'
    ]
    
    violations = []
    for field_path in fields_that_should_be_removed:
        if check_field_exists(participant_data, field_path):
            violations.append(field_path)
    
    return violations


def verify_preserved_fields(anonymized_data: Dict, original_data: Dict) -> List[str]:
    """
    Check if all fields that existed in the original are preserved (except those meant to be removed).
    
    Args:
        anonymized_data: Anonymized data for one participant
        original_data: Original data for the same participant
        
    Returns:
        List of field names that were in original but are missing in anonymized
    """
    # Fields that we intentionally remove
    intentionally_removed = {
        'browser_fingerprint',
        'recruitment_info'
    }
    
    # Fields that commonly exist and should be preserved if present
    fields_to_check = [
        'task_data',
        'demographic_form',
        'questionnaireQuestions',
        'points',
        'seedID',
        'user_data',
        'starttime',
        'endtime',
        'route_times',
        'conditions',
        'totalDuration',
        'done',
        'consented',
        'partner_found',
        'relevantFeatures',
        'relevantFeaturesBadValues'
    ]
    
    missing = []
    
    # Check each field that existed in original
    for field in fields_to_check:
        if field in original_data and field not in intentionally_removed:
            if field not in anonymized_data:
                missing.append(field)
    
    # Special check for demographic_form subfields
    if 'demographic_form' in original_data and 'demographic_form' in anonymized_data:
        for demo_field, demo_value in original_data['demographic_form'].items():
            # Skip zipcode as it should be removed
            if demo_field == 'zipcode':
                continue
            # Check if other demo fields are preserved
            if demo_field not in anonymized_data['demographic_form']:
                missing.append(f'demographic_form.{demo_field}')
    
    return missing


def test_anonymized_file(original_path: Path, anonymized_path: Path) -> Dict:
    """
    Test a single anonymized file against its original.
    
    Args:
        original_path: Path to original data file
        anonymized_path: Path to anonymized data file
        
    Returns:
        Dictionary with test results
    """
    print(f"\nTesting {anonymized_path.name}...")
    
    # Load both files
    with open(original_path, 'r') as f:
        original_data = json.load(f)
    
    with open(anonymized_path, 'r') as f:
        anonymized_data = json.load(f)
    
    results = {
        'file': anonymized_path.name,
        'original_count': len(original_data),
        'anonymized_count': len(anonymized_data),
        'count_match': len(original_data) == len(anonymized_data),
        'participants_with_violations': [],
        'participants_missing_fields': [],
        'all_violations': [],
        'all_missing': []
    }
    
    # Create a lookup dictionary for original data
    original_lookup = {p['id']: p['data'] for p in original_data}
    
    # Check each participant
    for participant in anonymized_data:
        participant_id = participant['id']
        participant_data = participant['data']
        
        # Get the original data for this participant
        original_participant_data = original_lookup.get(participant_id, {})
        
        # Check for fields that should be removed
        violations = verify_removed_fields(participant_data)
        if violations:
            results['participants_with_violations'].append(participant_id)
            results['all_violations'].extend(violations)
        
        # Check for fields that should be preserved (compare against original)
        missing = verify_preserved_fields(participant_data, original_participant_data)
        if missing:
            results['participants_missing_fields'].append(participant_id)
            results['all_missing'].extend(missing)
    
    # Remove duplicates from all_violations and all_missing
    results['all_violations'] = list(set(results['all_violations']))
    results['all_missing'] = list(set(results['all_missing']))
    
    # Print summary for this file
    if results['count_match']:
        print(f"  ✓ Participant count matches: {results['original_count']}")
    else:
        print(f"  ✗ Participant count mismatch: {results['original_count']} → {results['anonymized_count']}")
    
    if not results['participants_with_violations']:
        print(f"  ✓ All PII fields successfully removed")
    else:
        print(f"  ✗ {len(results['participants_with_violations'])} participants still have PII fields")
        print(f"    Fields found: {', '.join(results['all_violations'])}")
    
    if not results['participants_missing_fields']:
        print(f"  ✓ All required fields preserved")
    else:
        print(f"  ✗ {len(results['participants_missing_fields'])} participants missing required fields")
        print(f"    Fields missing: {', '.join(results['all_missing'])}")
    
    return results


def main():
    """Main function to test all anonymized datasets."""
    
    # Setup paths
    data_dir = Path('../data')
    anonymized_dir = Path('../data')  # Anonymized files are in data directory
    
    if not anonymized_dir.exists():
        print(f"Error: Anonymized data directory '{anonymized_dir}' does not exist.")
        print("Please run anonymize.py first.")
        return
    
    # Test each experiment file
    all_results = []
    for exp_num in range(1, 7):
        original_file = data_dir / f'exp-{exp_num}-data.json'
        anonymized_file = anonymized_dir / f'exp-{exp_num}-data-anonymized.json'
        
        if not original_file.exists():
            print(f"Warning: Original file {original_file} not found, skipping...")
            continue
            
        if not anonymized_file.exists():
            print(f"Warning: Anonymized file {anonymized_file} not found, skipping...")
            continue
        
        results = test_anonymized_file(original_file, anonymized_file)
        all_results.append(results)
    
    # Generate test report (save in outputs directory)
    reports_dir = Path('outputs')
    reports_dir.mkdir(exist_ok=True)
    report_path = reports_dir / 'test_report.txt'
    with open(report_path, 'w') as f:
        f.write("Anonymization Verification Report\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Overall summary
        total_files = len(all_results)
        files_passed = sum(1 for r in all_results if not r['participants_with_violations'] and not r['participants_missing_fields'] and r['count_match'])
        
        f.write(f"OVERALL SUMMARY\n")
        f.write(f"Files tested: {total_files}\n")
        f.write(f"Files passed all tests: {files_passed}/{total_files}\n\n")
        
        # Detailed results for each file
        for result in all_results:
            f.write(f"\n{result['file']}:\n")
            f.write(f"  Participant count: {result['anonymized_count']} ")
            f.write(f"({'matches' if result['count_match'] else 'MISMATCH'} original)\n")
            
            if result['participants_with_violations']:
                f.write(f"  ⚠️  PII FIELDS FOUND: {', '.join(result['all_violations'])}\n")
                f.write(f"     Affected participants: {len(result['participants_with_violations'])}\n")
            else:
                f.write(f"  ✓ All PII fields removed\n")
            
            if result['participants_missing_fields']:
                f.write(f"  ⚠️  MISSING FIELDS: {', '.join(result['all_missing'])}\n")
                f.write(f"     Affected participants: {len(result['participants_missing_fields'])}\n")
            else:
                f.write(f"  ✓ All required fields preserved\n")
        
        # Final verdict
        f.write("\n" + "=" * 50 + "\n")
        if files_passed == total_files:
            f.write("✓ ALL TESTS PASSED - Anonymization successful!\n")
            f.write("The anonymized datasets are ready for public release.\n")
        else:
            f.write("✗ TESTS FAILED - Issues found in anonymization\n")
            f.write("Please review the issues above and re-run anonymize.py\n")
    
    print(f"\n{'=' * 50}")
    print(f"Test complete! Report saved to {report_path}")
    
    if files_passed == total_files:
        print("✓ All anonymization tests passed!")
    else:
        print(f"✗ {total_files - files_passed} file(s) failed tests. See report for details.")


if __name__ == "__main__":
    main()