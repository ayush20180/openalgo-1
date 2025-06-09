"""
Integration Testing Script
------------------------
This script performs integration testing across all features of the trading journal application.
"""

import os
import sys
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import dash
from dash.testing.application_runners import import_app

# Add the project directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_large_dataset():
    """
    Test application performance with a large dataset.
    
    Returns:
        Dictionary with test results
    """
    from app.utils import testing, data_loader
    
    results = {
        'status': 'PASS',
        'errors': [],
        'metrics': {}
    }
    
    try:
        # Generate large dataset (1000 trades)
        print("Generating large dataset (1000 trades)...")
        large_file_path = os.path.join('data', 'large_test_dataset.csv')
        df_large = testing.generate_test_dataset(num_trades=1000, output_file=large_file_path)
        
        # Measure loading time
        start_time = time.time()
        with open(large_file_path, 'r') as f:
            file_content = f.read()
        df = data_loader.load_trade_csv(file_content)
        loading_time = time.time() - start_time
        
        # Measure preprocessing time
        start_time = time.time()
        processed_df = data_loader.preprocess_data(df)
        preprocessing_time = time.time() - start_time
        
        # Record metrics
        results['metrics']['dataset_size'] = len(df)
        results['metrics']['loading_time'] = loading_time
        results['metrics']['preprocessing_time'] = preprocessing_time
        results['metrics']['total_processing_time'] = loading_time + preprocessing_time
        
        # Check if processing time is acceptable (under 5 seconds for 1000 trades)
        if results['metrics']['total_processing_time'] > 5:
            results['status'] = 'WARNING'
            results['errors'].append(f"Processing time of {results['metrics']['total_processing_time']:.2f}s exceeds recommended threshold of 5s")
    
    except Exception as e:
        results['status'] = 'FAIL'
        results['errors'].append(str(e))
    
    return results

def test_calculation_accuracy():
    """
    Test calculation accuracy across different metrics.
    
    Returns:
        Dictionary with test results
    """
    from app.utils import metrics_calculator, advanced_metrics
    
    results = {
        'status': 'PASS',
        'errors': [],
        'metrics': {}
    }
    
    try:
        # Create a controlled dataset with known outcomes
        data = {
            'TradeID': ['T1', 'T2', 'T3', 'T4', 'T5'],
            'OpenTimestamp': pd.date_range(start='2025-01-01', periods=5),
            'CloseTimestamp': pd.date_range(start='2025-01-02', periods=5),
            'Symbol': ['AAPL', 'MSFT', 'AAPL', 'GOOGL', 'MSFT'],
            'PositionType': ['Long', 'Short', 'Long', 'Long', 'Short'],
            'EntryPrice': [150.0, 300.0, 155.0, 2500.0, 310.0],
            'ExitPrice': [160.0, 290.0, 150.0, 2600.0, 320.0],
            'Quantity': [10, 5, 10, 1, 5],
            'Commission': [5.0, 5.0, 5.0, 5.0, 5.0],
            'SwapFees': [0.0, 0.0, 0.0, 0.0, 0.0],
            'NetP&L': [95.0, 45.0, -55.0, 95.0, -55.0]
        }
        df = pd.DataFrame(data)
        
        # Test win rate calculation
        stats = metrics_calculator.calculate_summary_stats(df)
        expected_win_rate = 3/5  # 3 winning trades out of 5
        if abs(stats['win_rate'] - expected_win_rate) > 0.001:
            results['status'] = 'FAIL'
            results['errors'].append(f"Win rate calculation incorrect. Expected: {expected_win_rate}, Got: {stats['win_rate']}")
        
        # Test profit factor calculation
        expected_profit_factor = (95 + 45 + 95) / (55 + 55)  # (sum of profits) / (sum of losses)
        if abs(stats['profit_factor'] - expected_profit_factor) > 0.001:
            results['status'] = 'FAIL'
            results['errors'].append(f"Profit factor calculation incorrect. Expected: {expected_profit_factor}, Got: {stats['profit_factor']}")
        
        # Test algorithm-specific stats
        df['AlgorithmID'] = ['ALGO1', 'ALGO1', 'ALGO2', 'ALGO2', 'ALGO1']
        algo_stats = metrics_calculator.get_stats_per_algorithm(df)
        
        # ALGO1 should have 2 wins, 1 loss
        expected_algo1_win_rate = 2/3
        if abs(algo_stats['ALGO1']['win_rate'] - expected_algo1_win_rate) > 0.001:
            results['status'] = 'FAIL'
            results['errors'].append(f"Algorithm win rate calculation incorrect. Expected: {expected_algo1_win_rate}, Got: {algo_stats['ALGO1']['win_rate']}")
        
        # Test advanced metrics
        # Add cumulative P&L for drawdown calculation
        df = df.sort_values('OpenTimestamp')
        df['CumulativeP&L'] = df['NetP&L'].cumsum()
        
        # Test max drawdown calculation
        drawdown = advanced_metrics.calculate_max_drawdown(df['CumulativeP&L'])
        expected_max_drawdown = -55.0  # The largest drop is from trade 3
        if abs(drawdown['max_drawdown'] - expected_max_drawdown) > 0.001:
            results['status'] = 'FAIL'
            results['errors'].append(f"Max drawdown calculation incorrect. Expected: {expected_max_drawdown}, Got: {drawdown['max_drawdown']}")
        
        # Record metrics
        results['metrics']['win_rate'] = stats['win_rate']
        results['metrics']['profit_factor'] = stats['profit_factor']
        results['metrics']['max_drawdown'] = drawdown['max_drawdown']
    
    except Exception as e:
        results['status'] = 'FAIL'
        results['errors'].append(str(e))
    
    return results

def test_visualization_correctness():
    """
    Test visualization correctness by validating figure data.
    
    Returns:
        Dictionary with test results
    """
    import plotly.graph_objects as go
    from app.utils import metrics_calculator
    
    results = {
        'status': 'PASS',
        'errors': [],
        'metrics': {}
    }
    
    try:
        # Create a controlled dataset
        data = {
            'TradeID': ['T1', 'T2', 'T3', 'T4', 'T5'],
            'OpenTimestamp': pd.date_range(start='2025-01-01', periods=5),
            'CloseTimestamp': pd.date_range(start='2025-01-02', periods=5),
            'Symbol': ['AAPL', 'MSFT', 'AAPL', 'GOOGL', 'MSFT'],
            'PositionType': ['Long', 'Short', 'Long', 'Long', 'Short'],
            'EntryPrice': [150.0, 300.0, 155.0, 2500.0, 310.0],
            'ExitPrice': [160.0, 290.0, 150.0, 2600.0, 320.0],
            'Quantity': [10, 5, 10, 1, 5],
            'Commission': [5.0, 5.0, 5.0, 5.0, 5.0],
            'SwapFees': [0.0, 0.0, 0.0, 0.0, 0.0],
            'NetP&L': [95.0, 45.0, -55.0, 95.0, -55.0]
        }
        df = pd.DataFrame(data)
        
        # Calculate cumulative P&L
        df = df.sort_values('OpenTimestamp')
        df = metrics_calculator.calculate_cumulative_pnl(df)
        
        # Create equity curve figure
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['OpenTimestamp'],
            y=df['CumulativeP&L'],
            mode='lines+markers',
            name='Equity Curve'
        ))
        
        # Validate figure data
        expected_y_values = [95.0, 140.0, 85.0, 180.0, 125.0]
        actual_y_values = fig.data[0].y.tolist()
        
        if len(actual_y_values) != len(expected_y_values):
            results['status'] = 'FAIL'
            results['errors'].append(f"Equity curve has incorrect number of points. Expected: {len(expected_y_values)}, Got: {len(actual_y_values)}")
        
        for i, (expected, actual) in enumerate(zip(expected_y_values, actual_y_values)):
            if abs(expected - actual) > 0.001:
                results['status'] = 'FAIL'
                results['errors'].append(f"Equity curve point {i} is incorrect. Expected: {expected}, Got: {actual}")
        
        # Record metrics
        results['metrics']['figure_points'] = len(actual_y_values)
        results['metrics']['data_accuracy'] = 'PASS' if results['status'] == 'PASS' else 'FAIL'
    
    except Exception as e:
        results['status'] = 'FAIL'
        results['errors'].append(str(e))
    
    return results

def test_ui_responsiveness():
    """
    Test UI responsiveness by simulating different screen sizes.
    
    Returns:
        Dictionary with test results
    """
    results = {
        'status': 'PASS',
        'errors': [],
        'metrics': {}
    }
    
    try:
        # This is a simplified test since we can't easily test the UI in this environment
        # In a real environment, we would use Selenium or similar tools
        
        # Check if the app uses responsive design components
        with open('app/main.py', 'r') as f:
            main_content = f.read()
        
        # Check for responsive design patterns
        responsive_patterns = [
            'fluid=True',
            'className="container',
            'dbc.Row',
            'dbc.Col',
            'width='
        ]
        
        responsive_score = 0
        for pattern in responsive_patterns:
            if pattern in main_content:
                responsive_score += 1
        
        # Calculate percentage of responsive patterns found
        responsive_percentage = (responsive_score / len(responsive_patterns)) * 100
        
        # Record metrics
        results['metrics']['responsive_score'] = responsive_score
        results['metrics']['responsive_percentage'] = responsive_percentage
        
        # Check if score is acceptable
        if responsive_percentage < 80:
            results['status'] = 'WARNING'
            results['errors'].append(f"UI responsiveness score ({responsive_percentage:.1f}%) is below recommended threshold of 80%")
    
    except Exception as e:
        results['status'] = 'FAIL'
        results['errors'].append(str(e))
    
    return results

def run_integration_tests():
    """
    Run all integration tests and generate a report.
    
    Returns:
        Dictionary with all test results
    """
    results = {
        'overall_status': 'PASS',
        'large_dataset': {},
        'calculation_accuracy': {},
        'visualization_correctness': {},
        'ui_responsiveness': {}
    }
    
    try:
        # Test large dataset performance
        print("\nTesting large dataset performance...")
        results['large_dataset'] = test_large_dataset()
        if results['large_dataset']['status'] == 'FAIL':
            results['overall_status'] = 'FAIL'
        
        # Test calculation accuracy
        print("\nTesting calculation accuracy...")
        results['calculation_accuracy'] = test_calculation_accuracy()
        if results['calculation_accuracy']['status'] == 'FAIL':
            results['overall_status'] = 'FAIL'
        
        # Test visualization correctness
        print("\nTesting visualization correctness...")
        results['visualization_correctness'] = test_visualization_correctness()
        if results['visualization_correctness']['status'] == 'FAIL':
            results['overall_status'] = 'FAIL'
        
        # Test UI responsiveness
        print("\nTesting UI responsiveness...")
        results['ui_responsiveness'] = test_ui_responsiveness()
        if results['ui_responsiveness']['status'] == 'FAIL':
            results['overall_status'] = 'FAIL'
    
    except Exception as e:
        results['overall_status'] = 'FAIL'
        results['error'] = str(e)
    
    return results

def generate_integration_report(test_results, output_file='integration_test_report.md'):
    """
    Generate a markdown report from integration test results.
    
    Args:
        test_results: Dictionary with test results
        output_file: Path to save the report
        
    Returns:
        Path to the generated report
    """
    report = []
    
    # Add header
    report.append("# Trading Journal Application Integration Test Report")
    report.append(f"\nGenerated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"\n## Overall Status: {test_results['overall_status']}")
    
    # Large Dataset Performance
    report.append("\n## 1. Large Dataset Performance")
    large_dataset = test_results.get('large_dataset', {})
    report.append(f"- **Status**: {large_dataset.get('status', 'N/A')}")
    
    if large_dataset.get('metrics'):
        metrics = large_dataset['metrics']
        report.append(f"- **Dataset Size**: {metrics.get('dataset_size', 'N/A')} trades")
        report.append(f"- **Loading Time**: {metrics.get('loading_time', 'N/A'):.3f} seconds")
        report.append(f"- **Preprocessing Time**: {metrics.get('preprocessing_time', 'N/A'):.3f} seconds")
        report.append(f"- **Total Processing Time**: {metrics.get('total_processing_time', 'N/A'):.3f} seconds")
    
    if large_dataset.get('errors'):
        report.append("\n### Errors")
        for error in large_dataset['errors']:
            report.append(f"- {error}")
    
    # Calculation Accuracy
    report.append("\n## 2. Calculation Accuracy")
    calc_accuracy = test_results.get('calculation_accuracy', {})
    report.append(f"- **Status**: {calc_accuracy.get('status', 'N/A')}")
    
    if calc_accuracy.get('metrics'):
        metrics = calc_accuracy['metrics']
        report.append(f"- **Win Rate**: {metrics.get('win_rate', 'N/A'):.2%}")
        report.append(f"- **Profit Factor**: {metrics.get('profit_factor', 'N/A'):.2f}")
        report.append(f"- **Max Drawdown**: ${metrics.get('max_drawdown', 'N/A'):.2f}")
    
    if calc_accuracy.get('errors'):
        report.append("\n### Errors")
        for error in calc_accuracy['errors']:
            report.append(f"- {error}")
    
    # Visualization Correctness
    report.append("\n## 3. Visualization Correctness")
    viz_correctness = test_results.get('visualization_correctness', {})
    report.append(f"- **Status**: {viz_correctness.get('status', 'N/A')}")
    
    if viz_correctness.get('metrics'):
        metrics = viz_correctness['metrics']
        report.append(f"- **Figure Points**: {metrics.get('figure_points', 'N/A')}")
        report.append(f"- **Data Accuracy**: {metrics.get('data_accuracy', 'N/A')}")
    
    if viz_correctness.get('errors'):
        report.append("\n### Errors")
        for error in viz_correctness['errors']:
            report.append(f"- {error}")
    
    # UI Responsiveness
    report.append("\n## 4. UI Responsiveness")
    ui_resp = test_results.get('ui_responsiveness', {})
    report.append(f"- **Status**: {ui_resp.get('status', 'N/A')}")
    
    if ui_resp.get('metrics'):
        metrics = ui_resp['metrics']
        report.append(f"- **Responsive Score**: {metrics.get('responsive_score', 'N/A')}/5")
        report.append(f"- **Responsive Percentage**: {metrics.get('responsive_percentage', 'N/A'):.1f}%")
    
    if ui_resp.get('errors'):
        report.append("\n### Errors")
        for error in ui_resp['errors']:
            report.append(f"- {error}")
    
    # Write report to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(report))
    
    return output_file

def main():
    """
    Run integration tests and generate a report.
    """
    print("Starting integration testing of Trading Journal Application...")
    print(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run integration tests
    test_results = run_integration_tests()
    
    # Generate integration test report
    report_path = os.path.join('data', 'integration_test_report.md')
    generate_integration_report(test_results, report_path)
    
    # Print summary
    print("\n=== INTEGRATION TEST SUMMARY ===")
    print(f"Overall Status: {test_results['overall_status']}")
    print(f"Large Dataset Performance: {test_results['large_dataset'].get('status', 'N/A')}")
    print(f"Calculation Accuracy: {test_results['calculation_accuracy'].get('status', 'N/A')}")
    print(f"Visualization Correctness: {test_results['visualization_correctness'].get('status', 'N/A')}")
    print(f"UI Responsiveness: {test_results['ui_responsiveness'].get('status', 'N/A')}")
    
    print(f"\nDetailed integration test report saved to: {report_path}")
    
    return test_results['overall_status'] == 'PASS'

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
