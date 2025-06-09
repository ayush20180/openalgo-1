"""
Test Application Script
---------------------
This script runs comprehensive tests on the trading journal application.
"""

import os
import sys
import json
from datetime import datetime

# Add the project directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import testing module
from app.utils import testing

def main():
    """
    Run comprehensive tests and generate a report.
    """
    print("Starting comprehensive testing of Trading Journal Application...")
    print(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Generate test dataset
    print("\n1. Generating test dataset...")
    test_file_path = os.path.join('data', 'comprehensive_test_dataset.csv')
    
    # Ensure data directory exists
    os.makedirs(os.path.dirname(test_file_path), exist_ok=True)
    
    # Generate test dataset (passing None to generate a new dataset)
    print("Generating new test dataset...")
    test_results = testing.run_comprehensive_tests(None)
    
    # Generate test report
    print("\n3. Generating test report...")
    report_path = os.path.join('data', 'test_report.md')
    testing.generate_test_report(test_results, report_path)
    
    # Print summary
    print("\n=== TEST SUMMARY ===")
    print(f"Overall Status: {test_results['overall_status']}")
    print(f"Data Generation: {test_results['data_generation'].get('status', 'N/A')}")
    print(f"Data Integrity: {test_results['data_integrity'].get('overall_status', 'N/A')}")
    print(f"Data Loading: {test_results['data_loading'].get('status', 'N/A')}")
    print(f"Metrics Calculation: {test_results['metrics_calculation'].get('status', 'N/A')}")
    print(f"Advanced Metrics: {test_results['advanced_metrics'].get('status', 'N/A')}")
    
    print(f"\nDetailed test report saved to: {report_path}")
    
    # Save test results as JSON for further analysis
    json_path = os.path.join('data', 'test_results.json')
    with open(json_path, 'w') as f:
        json.dump(test_results, f, indent=2, default=str)
    
    print(f"Test results saved to: {json_path}")
    
    return test_results['overall_status'] == 'PASS'

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
