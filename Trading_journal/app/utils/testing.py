"""
Testing and Validation Module
---------------------------
This module contains functions for comprehensive testing and validation of the application.
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
import random


def generate_test_dataset(num_trades=100, output_file='data/test_dataset.csv'):
    """
    Generate a comprehensive test dataset with realistic trading data.
    
    Args:
        num_trades: Number of trades to generate (default: 100)
        output_file: Path to save the CSV file (default: 'data/test_dataset.csv')
        
    Returns:
        DataFrame containing the generated test data
    """
    # Define possible values for categorical fields
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NFLX', 'NVDA', 'AMD', 'INTC',
               'JPM', 'BAC', 'GS', 'MS', 'C', 'WFC', 'V', 'MA', 'PYPL', 'SQ',
               'DIS', 'NFLX', 'CMCSA', 'T', 'VZ', 'TMUS', 'CHTR', 'DISH', 'ROKU', 'SPOT']
    
    exchanges = ['NASDAQ', 'NYSE', 'AMEX', 'LSE', 'TSE']
    
    position_types = ['Long', 'Short']
    
    algorithms = ['ALGO001', 'ALGO002', 'ALGO003', 'ALGO004', 'ALGO005']
    
    exit_signals = ['TAKE_PROFIT', 'STOP_LOSS', 'TRAILING_STOP', 'TIME_EXIT', 'MANUAL']
    
    product_types = ['Stock', 'ETF', 'Option', 'Future', 'Forex', 'Crypto']
    
    # Parameter templates for algorithms
    parameter_templates = {
        'ALGO001': {'ma_period': [10, 20, 50], 'rsi_threshold': [30, 70]},
        'ALGO002': {'bollinger_period': [14, 20, 30], 'std_dev': [1.5, 2.0, 2.5]},
        'ALGO003': {'macd_fast': [8, 12], 'macd_slow': [21, 26], 'signal': [9]},
        'ALGO004': {'ema_short': [5, 8, 13], 'ema_long': [21, 34, 55]},
        'ALGO005': {'atr_period': [7, 14, 21], 'atr_multiplier': [1.5, 2.0, 3.0]}
    }
    
    # Generate random trades
    trades = []
    
    # Start date (3 months ago)
    start_date = datetime.now() - timedelta(days=90)
    
    for i in range(1, num_trades + 1):
        # Generate trade ID
        trade_id = f"T{i:05d}"
        
        # Generate random open timestamp within the last 3 months
        days_ago = random.randint(0, 89)
        open_timestamp = start_date + timedelta(days=days_ago)
        
        # Add random hours and minutes
        open_timestamp = open_timestamp.replace(
            hour=random.randint(9, 16),
            minute=random.randint(0, 59),
            second=0
        )
        
        # Generate random trade duration (minutes to days)
        duration_minutes = random.randint(15, 60 * 24 * 3)  # 15 min to 3 days
        close_timestamp = open_timestamp + timedelta(minutes=duration_minutes)
        
        # Select random values for categorical fields
        symbol = random.choice(symbols)
        exchange = random.choice(exchanges)
        position_type = random.choice(position_types)
        algorithm_id = random.choice(algorithms)
        exit_signal = random.choice(exit_signals)
        product_type = random.choice(product_types)
        
        # Generate realistic prices based on symbol
        base_price = {
            'AAPL': 150.0, 'MSFT': 300.0, 'GOOGL': 120.0, 'AMZN': 130.0, 'TSLA': 200.0,
            'META': 300.0, 'NFLX': 400.0, 'NVDA': 450.0, 'AMD': 100.0, 'INTC': 35.0
        }.get(symbol, random.uniform(50.0, 500.0))
        
        # Add some randomness to base price
        base_price *= random.uniform(0.95, 1.05)
        
        # Generate entry and exit prices based on position type and exit signal
        price_change_pct = random.uniform(-0.05, 0.05)
        
        if position_type == 'Long':
            entry_price = base_price
            if exit_signal == 'TAKE_PROFIT':
                price_change_pct = abs(price_change_pct)  # Ensure profit
            elif exit_signal == 'STOP_LOSS':
                price_change_pct = -abs(price_change_pct)  # Ensure loss
            exit_price = entry_price * (1 + price_change_pct)
        else:  # Short
            entry_price = base_price
            if exit_signal == 'TAKE_PROFIT':
                price_change_pct = -abs(price_change_pct)  # Ensure profit (price went down)
            elif exit_signal == 'STOP_LOSS':
                price_change_pct = abs(price_change_pct)  # Ensure loss (price went up)
            exit_price = entry_price * (1 + price_change_pct)
        
        # Generate quantity based on price (higher prices get lower quantities)
        quantity = max(1, int(10000 / entry_price))
        
        # Generate commission and swap fees
        commission = round(random.uniform(5.0, 15.0), 2)
        swap_fees = round(random.uniform(0.0, 5.0), 2) if duration_minutes > 60 * 24 else 0.0
        
        # Round prices first to ensure consistency
        entry_price_rounded = round(entry_price, 2)
        exit_price_rounded = round(exit_price, 2)
        
        # Calculate P&L using rounded prices
        if position_type == 'Long':
            gross_pnl = (exit_price_rounded - entry_price_rounded) * quantity
        else:  # Short
            gross_pnl = (entry_price_rounded - exit_price_rounded) * quantity
        
        gross_pnl = round(gross_pnl, 2)  # Round to 2 decimal places
        net_pnl = round(gross_pnl - commission - swap_fees, 2)  # Round after calculation
        
        # Generate algorithm parameters
        if algorithm_id in parameter_templates:
            params = {}
            for param, values in parameter_templates[algorithm_id].items():
                params[param] = random.choice(values)
            parameters = json.dumps(params)
        else:
            parameters = "{}"
        
        # Create trade record
        trade = {
            'TradeID': trade_id,
            'OpenTimestamp': open_timestamp,
            'CloseTimestamp': close_timestamp,
            'Symbol': symbol,
            'Exchange': exchange,
            'PositionType': position_type,
            'EntryPrice': entry_price_rounded,
            'ExitPrice': exit_price_rounded,
            'Quantity': quantity,
            'Commission': commission,
            'SwapFees': swap_fees,
            'GrossP&L': gross_pnl,
            'NetP&L': net_pnl,
            'AlgorithmID': algorithm_id,
            'Parameters': parameters,
            'SignalName_Exit': exit_signal,
            'ProductType': product_type
        }
        
        trades.append(trade)
    
    # Create DataFrame
    df = pd.DataFrame(trades)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_file, index=False, quoting=1)  # Use QUOTE_ALL for safe CSV
    
    return df


def validate_data_integrity(df):
    """
    Validate data integrity of the DataFrame.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Dictionary with validation results
    """
    results = {
        'missing_values': {},
        'data_type_issues': {},
        'logical_issues': [],
        'overall_status': 'PASS'
    }
    
    # Check for missing values
    for column in df.columns:
        missing_count = df[column].isna().sum()
        if missing_count > 0:
            results['missing_values'][column] = missing_count
            results['overall_status'] = 'FAIL'
    
    # Check data types
    expected_types = {
        'TradeID': str,
        'OpenTimestamp': 'datetime',
        'CloseTimestamp': 'datetime',
        'Symbol': str,
        'Exchange': str,
        'PositionType': str,
        'EntryPrice': 'numeric',
        'ExitPrice': 'numeric',
        'Quantity': 'numeric',
        'Commission': 'numeric',
        'SwapFees': 'numeric',
        'GrossP&L': 'numeric',
        'NetP&L': 'numeric',
        'AlgorithmID': str,
        'Parameters': str,
        'SignalName_Exit': str,
        'ProductType': str
    }
    
    for column, expected_type in expected_types.items():
        if column not in df.columns:
            results['data_type_issues'][column] = 'Column missing'
            results['overall_status'] = 'FAIL'
            continue
        
        if expected_type == 'datetime':
            if not pd.api.types.is_datetime64_any_dtype(df[column]):
                results['data_type_issues'][column] = f'Expected datetime, got {df[column].dtype}'
                results['overall_status'] = 'FAIL'
        elif expected_type == 'numeric':
            if not pd.api.types.is_numeric_dtype(df[column]):
                results['data_type_issues'][column] = f'Expected numeric, got {df[column].dtype}'
                results['overall_status'] = 'FAIL'
        else:
            # For string types, we're more lenient
            pass
    
    # Check logical consistency
    
    # 1. CloseTimestamp should be after OpenTimestamp
    if 'OpenTimestamp' in df.columns and 'CloseTimestamp' in df.columns:
        invalid_timestamps = df[df['CloseTimestamp'] < df['OpenTimestamp']].shape[0]
        if invalid_timestamps > 0:
            results['logical_issues'].append(f'Found {invalid_timestamps} trades where close time is before open time')
            results['overall_status'] = 'FAIL'
    
    # 2. Check P&L calculation consistency
    if all(col in df.columns for col in ['PositionType', 'EntryPrice', 'ExitPrice', 'Quantity', 'GrossP&L']):
        df['calculated_gross_pnl'] = np.where(
            df['PositionType'] == 'Long',
            (df['ExitPrice'] - df['EntryPrice']) * df['Quantity'],
            (df['EntryPrice'] - df['ExitPrice']) * df['Quantity']
        )
        
        # Allow for small floating point differences
        pnl_diff = (df['calculated_gross_pnl'] - df['GrossP&L']).abs()
        inconsistent_pnl = df[pnl_diff > 0.01].shape[0]
        
        if inconsistent_pnl > 0:
            results['logical_issues'].append(f'Found {inconsistent_pnl} trades with inconsistent P&L calculations')
            results['overall_status'] = 'FAIL'
    
    # 3. Check Net P&L calculation
    if all(col in df.columns for col in ['GrossP&L', 'Commission', 'SwapFees', 'NetP&L']):
        df['calculated_net_pnl'] = df['GrossP&L'] - df['Commission'] - df['SwapFees']
        
        # Allow for small floating point differences
        net_pnl_diff = (df['calculated_net_pnl'] - df['NetP&L']).abs()
        inconsistent_net_pnl = df[net_pnl_diff > 0.01].shape[0]
        
        if inconsistent_net_pnl > 0:
            results['logical_issues'].append(f'Found {inconsistent_net_pnl} trades with inconsistent Net P&L calculations')
            results['overall_status'] = 'FAIL'
    
    return results


def test_data_loading(file_path):
    """
    Test data loading functionality.
    
    Args:
        file_path: Path to the test CSV file
        
    Returns:
        Dictionary with test results
    """
    from app.utils import data_loader
    
    results = {
        'status': 'PASS',
        'errors': []
    }
    
    try:
        # Read file content
        with open(file_path, 'r') as f:
            file_content = f.read()
        
        # Test CSV loading
        df = data_loader.load_trade_csv(file_content)
        results['loaded_rows'] = len(df)
        
        # Test data preprocessing
        processed_df = data_loader.preprocess_data(df)
        results['processed_rows'] = len(processed_df)
        
        # Validate data types after preprocessing
        for col in ['OpenTimestamp', 'CloseTimestamp']:
            if not pd.api.types.is_datetime64_any_dtype(processed_df[col]):
                results['status'] = 'FAIL'
                results['errors'].append(f'Column {col} is not datetime type after preprocessing')
        
        for col in ['EntryPrice', 'ExitPrice', 'Quantity', 'Commission', 'SwapFees', 'GrossP&L', 'NetP&L']:
            if not pd.api.types.is_numeric_dtype(processed_df[col]):
                results['status'] = 'FAIL'
                results['errors'].append(f'Column {col} is not numeric type after preprocessing')
    
    except Exception as e:
        results['status'] = 'FAIL'
        results['errors'].append(str(e))
    
    return results


def test_metrics_calculation(df):
    """
    Test metrics calculation functionality.
    
    Args:
        df: DataFrame with preprocessed trade data
        
    Returns:
        Dictionary with test results
    """
    from app.utils import metrics_calculator
    
    results = {
        'status': 'PASS',
        'errors': [],
        'metrics': {}
    }
    
    try:
        # Test trade duration calculation
        df_with_duration = metrics_calculator.calculate_trade_duration(df)
        if 'TradeDuration' not in df_with_duration.columns:
            results['status'] = 'FAIL'
            results['errors'].append('TradeDuration column not added')
        
        # Test cumulative P&L calculation
        df_with_cum_pnl = metrics_calculator.calculate_cumulative_pnl(df)
        if 'CumulativeP&L' not in df_with_cum_pnl.columns:
            results['status'] = 'FAIL'
            results['errors'].append('CumulativeP&L column not added')
        
        # Test summary statistics calculation
        stats = metrics_calculator.calculate_summary_stats(df)
        results['metrics']['summary_stats'] = stats
        
        # Validate summary statistics
        required_metrics = ['total_pnl', 'total_trades', 'win_rate', 'profit_factor']
        for metric in required_metrics:
            if metric not in stats:
                results['status'] = 'FAIL'
                results['errors'].append(f'Required metric {metric} missing from summary stats')
        
        # Test algorithm-specific statistics
        algo_stats = metrics_calculator.get_stats_per_algorithm(df)
        results['metrics']['algo_stats'] = {k: v for k, v in algo_stats.items()}
        
        # Validate algorithm stats
        if len(algo_stats) == 0 and 'AlgorithmID' in df.columns:
            results['status'] = 'FAIL'
            results['errors'].append('No algorithm-specific statistics calculated')
    
    except Exception as e:
        results['status'] = 'FAIL'
        results['errors'].append(str(e))
    
    return results


def test_advanced_metrics(df):
    """
    Test advanced metrics calculation functionality.
    
    Args:
        df: DataFrame with preprocessed trade data
        
    Returns:
        Dictionary with test results
    """
    from app.utils import advanced_metrics
    
    results = {
        'status': 'PASS',
        'errors': [],
        'metrics': {}
    }
    
    try:
        # Ensure TradeDuration is calculated
        if 'TradeDuration' not in df.columns:
            from app.utils import metrics_calculator
            df = metrics_calculator.calculate_trade_duration(df)
        
        # Test average holding time calculation
        holding_time = advanced_metrics.calculate_avg_holding_time(df)
        results['metrics']['avg_holding_time'] = {
            'overall': str(holding_time['overall_avg']),
            'winners': str(holding_time['winners_avg']),
            'losers': str(holding_time['losers_avg'])
        }
        
        # Test max drawdown calculation
        df = df.sort_values('OpenTimestamp')
        df['CumulativeP&L'] = df['NetP&L'].cumsum()
        drawdown = advanced_metrics.calculate_max_drawdown(df['CumulativeP&L'])
        results['metrics']['max_drawdown'] = drawdown['max_drawdown']
        results['metrics']['max_drawdown_pct'] = drawdown['max_drawdown_pct']
        
        # Test Sharpe ratio calculation
        # Group by date to get daily P&L
        df['Date'] = df['OpenTimestamp'].dt.date
        daily_pnl = df.groupby('Date')['NetP&L'].sum()
        sharpe = advanced_metrics.calculate_sharpe_ratio(daily_pnl)
        results['metrics']['sharpe_ratio'] = sharpe
        
        # Test Sortino ratio calculation
        sortino = advanced_metrics.calculate_sortino_ratio(daily_pnl)
        results['metrics']['sortino_ratio'] = sortino
        
        # Test position type performance analysis
        position_perf = advanced_metrics.calculate_performance_by_positiontype(df)
        results['metrics']['position_performance'] = {k: v['total_pnl'] for k, v in position_perf.items()}
        
        # Test exit signal analysis
        exit_analysis = advanced_metrics.analyze_exit_signals(df)
        results['metrics']['exit_signals'] = {k: v['win_rate'] for k, v in exit_analysis.items()}
        
        # Test consecutive wins/losses calculation
        streak_data = advanced_metrics.calculate_consecutive_wins_losses(df)
        results['metrics']['max_consecutive_wins'] = streak_data['max_consecutive_wins']
        results['metrics']['max_consecutive_losses'] = streak_data['max_consecutive_losses']
        
        # Test volatility calculation
        volatility = advanced_metrics.calculate_volatility(daily_pnl)
        results['metrics']['volatility_last'] = volatility.iloc[-1] if len(volatility) > 0 else None
        
        # Test trade cluster analysis
        clusters = advanced_metrics.analyze_trade_clusters(df, ['Symbol', 'PositionType'])
        results['metrics']['clusters_count'] = len(clusters)
    
    except Exception as e:
        results['status'] = 'FAIL'
        results['errors'].append(str(e))
    
    return results


def run_comprehensive_tests(test_file_path=None):
    """
    Run comprehensive tests on all application features.
    
    Args:
        test_file_path: Path to test CSV file (if None, generates a new one)
        
    Returns:
        Dictionary with all test results
    """
    results = {
        'overall_status': 'PASS',
        'data_generation': {},
        'data_integrity': {},
        'data_loading': {},
        'metrics_calculation': {},
        'advanced_metrics': {}
    }
    
    try:
        # Generate test data if not provided
        if test_file_path is None:
            test_file_path = 'data/test_dataset.csv'
            df = generate_test_dataset(num_trades=100, output_file=test_file_path)
            results['data_generation'] = {
                'status': 'PASS',
                'file_path': test_file_path,
                'rows_generated': len(df)
            }
        else:
            # Load existing test data
            df = pd.read_csv(test_file_path)
            results['data_generation'] = {
                'status': 'PASS',
                'file_path': test_file_path,
                'rows_loaded': len(df)
            }
        
        # Validate data integrity
        results['data_integrity'] = validate_data_integrity(df)
        if results['data_integrity']['overall_status'] == 'FAIL':
            results['overall_status'] = 'FAIL'
        
        # Test data loading
        results['data_loading'] = test_data_loading(test_file_path)
        if results['data_loading']['status'] == 'FAIL':
            results['overall_status'] = 'FAIL'
        
        # Preprocess data for further tests
        from app.utils import data_loader
        with open(test_file_path, 'r') as f:
            file_content = f.read()
        df = data_loader.load_trade_csv(file_content)
        processed_df = data_loader.preprocess_data(df)
        
        # Test metrics calculation
        results['metrics_calculation'] = test_metrics_calculation(processed_df)
        if results['metrics_calculation']['status'] == 'FAIL':
            results['overall_status'] = 'FAIL'
        
        # Test advanced metrics
        results['advanced_metrics'] = test_advanced_metrics(processed_df)
        if results['advanced_metrics']['status'] == 'FAIL':
            results['overall_status'] = 'FAIL'
    
    except Exception as e:
        results['overall_status'] = 'FAIL'
        results['error'] = str(e)
    
    return results


def generate_test_report(test_results, output_file='test_report.md'):
    """
    Generate a markdown report from test results.
    
    Args:
        test_results: Dictionary with test results
        output_file: Path to save the report
        
    Returns:
        Path to the generated report
    """
    report = []
    
    # Add header
    report.append("# Trading Journal Application Test Report")
    report.append(f"\nGenerated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"\n## Overall Status: {test_results['overall_status']}")
    
    # Data Generation
    report.append("\n## 1. Data Generation")
    data_gen = test_results.get('data_generation', {})
    report.append(f"- **Status**: {data_gen.get('status', 'N/A')}")
    report.append(f"- **File Path**: {data_gen.get('file_path', 'N/A')}")
    report.append(f"- **Rows**: {data_gen.get('rows_generated', data_gen.get('rows_loaded', 'N/A'))}")
    
    # Data Integrity
    report.append("\n## 2. Data Integrity")
    data_int = test_results.get('data_integrity', {})
    report.append(f"- **Status**: {data_int.get('overall_status', 'N/A')}")
    
    if data_int.get('missing_values'):
        report.append("\n### Missing Values")
        for col, count in data_int['missing_values'].items():
            report.append(f"- {col}: {count} missing values")
    
    if data_int.get('data_type_issues'):
        report.append("\n### Data Type Issues")
        for col, issue in data_int['data_type_issues'].items():
            report.append(f"- {col}: {issue}")
    
    if data_int.get('logical_issues'):
        report.append("\n### Logical Issues")
        for issue in data_int['logical_issues']:
            report.append(f"- {issue}")
    
    # Data Loading
    report.append("\n## 3. Data Loading")
    data_load = test_results.get('data_loading', {})
    report.append(f"- **Status**: {data_load.get('status', 'N/A')}")
    report.append(f"- **Loaded Rows**: {data_load.get('loaded_rows', 'N/A')}")
    report.append(f"- **Processed Rows**: {data_load.get('processed_rows', 'N/A')}")
    
    if data_load.get('errors'):
        report.append("\n### Errors")
        for error in data_load['errors']:
            report.append(f"- {error}")
    
    # Metrics Calculation
    report.append("\n## 4. Metrics Calculation")
    metrics = test_results.get('metrics_calculation', {})
    report.append(f"- **Status**: {metrics.get('status', 'N/A')}")
    
    if metrics.get('errors'):
        report.append("\n### Errors")
        for error in metrics['errors']:
            report.append(f"- {error}")
    
    if metrics.get('metrics', {}).get('summary_stats'):
        report.append("\n### Summary Statistics")
        stats = metrics['metrics']['summary_stats']
        report.append(f"- Total P&L: ${stats.get('total_pnl', 'N/A'):.2f}")
        report.append(f"- Total Trades: {stats.get('total_trades', 'N/A')}")
        report.append(f"- Win Rate: {stats.get('win_rate', 'N/A'):.2%}")
        report.append(f"- Profit Factor: {stats.get('profit_factor', 'N/A'):.2f}")
    
    # Advanced Metrics
    report.append("\n## 5. Advanced Metrics")
    adv_metrics = test_results.get('advanced_metrics', {})
    report.append(f"- **Status**: {adv_metrics.get('status', 'N/A')}")
    
    if adv_metrics.get('errors'):
        report.append("\n### Errors")
        for error in adv_metrics['errors']:
            report.append(f"- {error}")
    
    if adv_metrics.get('metrics'):
        m = adv_metrics['metrics']
        report.append("\n### Key Metrics")
        report.append(f"- Sharpe Ratio: {m.get('sharpe_ratio', 'N/A'):.2f}")
        report.append(f"- Sortino Ratio: {m.get('sortino_ratio', 'N/A'):.2f}")
        report.append(f"- Max Drawdown: ${m.get('max_drawdown', 'N/A'):.2f}")
        report.append(f"- Max Drawdown %: {m.get('max_drawdown_pct', 'N/A')*100:.2f}%")
        report.append(f"- Max Consecutive Wins: {m.get('max_consecutive_wins', 'N/A')}")
        report.append(f"- Max Consecutive Losses: {m.get('max_consecutive_losses', 'N/A')}")
    
    # Write report to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(report))
    
    return output_file

# --- Unit Tests for Journal Management ---
import unittest
# Assuming testing.py is in app/utils, and journal_management.py is also in app/utils
from .journal_management import add_trade 
import io # Required for pd.read_json on string data in add_trade

class TestJournalManagement(unittest.TestCase):

    def test_add_trade_datetime_conversion(self):
        # 1. Setup initial data
        initial_trades_list = [{
            'TradeID': 'T1',
            'OpenTimestamp': '2023-12-01T10:00:00.000Z', # ISO format
            'CloseTimestamp': '2023-12-01T11:00:00.000Z',# ISO format
            'Symbol': 'AAPL',
            'Exchange': 'NASDAQ',
            'PositionType': 'Long',
            'EntryPrice': 150.0,
            'ExitPrice': 151.0,
            'Quantity': 10.0,
            'Commission': 1.0,
            'SwapFees': 0.0,
            'GrossP&L': 10.0,
            'NetP&L': 9.0,
            'AlgorithmID': 'Algo1',
            'Parameters': '{}',
            'SignalName_Exit': 'Manual',
            'ProductType': 'Stock'
        }]
        initial_df = pd.DataFrame(initial_trades_list)
        # Ensure OpenTimestamp and CloseTimestamp are datetime objects before converting to JSON
        initial_df['OpenTimestamp'] = pd.to_datetime(initial_df['OpenTimestamp'])
        initial_df['CloseTimestamp'] = pd.to_datetime(initial_df['CloseTimestamp'])
        initial_json_data = initial_df.to_json(date_format='iso', orient='split')

        # New trade inputs (simulating form inputs)
        trade_id_new = "T2"
        symbol_new = "MSFT"
        exchange_new = "NASDAQ"
        open_date_new = "2024-01-15" # Date string from DatePicker
        open_time_new = "09:30:00"   # Time string from Input
        close_date_new = "2024-01-15"
        close_time_new = "10:30:00"
        position_type_new = "Long"
        product_type_new = "Stock"
        quantity_new = 100.0
        entry_price_new = 300.0
        exit_price_new = 305.0
        commission_new = 5.0
        swap_fees_new = 0.0
        # P&L typically calculated or entered, ensure these are float
        gross_pnl_new = 500.0 # (305-300)*100
        net_pnl_new = 495.0   # 500 - 5
        algorithm_id_new = "Algo2"
        exit_signal_new = "TAKE_PROFIT"
        parameters_new = "{}"
        # Other fields that might be in new_trade but not explicitly part of test logic can be added if required by add_trade
        # For example, 'Tags' or 'Notes' if they were mandatory. Assuming they are not for this test.

        # 2. Execution
        try:
            updated_json_data, message = add_trade( # Expect two return values
                n_clicks=1,
                json_data=initial_json_data,
                trade_id=trade_id_new,
                symbol=symbol_new,
                exchange=exchange_new,
                open_date=open_date_new,
                open_time=open_time_new,
                close_date=close_date_new,
                close_time=close_time_new,
                position_type=position_type_new,
                product_type=product_type_new,
                quantity=quantity_new,
                entry_price=entry_price_new,
                exit_price=exit_price_new,
                commission=commission_new,
                swap_fees=swap_fees_new,
                gross_pnl=gross_pnl_new,
                net_pnl=net_pnl_new,
                algorithm_id=algorithm_id_new,
                exit_signal=exit_signal_new,
                parameters=parameters_new
            )
        except ValueError as e:
            self.fail(f"add_trade raised ValueError unexpectedly: {e}")
        except Exception as e: # Catch any other unexpected error
            self.fail(f"add_trade raised an unexpected exception: {e}")


        # 3. Assertions
        self.assertIsNotNone(updated_json_data, "Updated JSON data should not be None")
        self.assertIsNotNone(message, "Message should not be None")
        self.assertEqual(message, f"Trade {trade_id_new} added successfully.") # Specific success message
        
        # Parse the output JSON
        updated_df = pd.read_json(io.StringIO(updated_json_data), orient='split')
        
        self.assertEqual(len(updated_df), 2, "DataFrame should contain two trades (initial + new)")
        
        # Verify the new trade is present
        new_trade_row_df = updated_df[updated_df['TradeID'] == trade_id_new]
        self.assertFalse(new_trade_row_df.empty, f"New trade with ID {trade_id_new} not found")
        
        # Verify timestamps for all trades are datetime objects after pd.read_json
        # This confirms they were stored in a recognizable ISO format.
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(updated_df['OpenTimestamp']),
                        "OpenTimestamp column should be of datetime type after parsing JSON.")
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(updated_df['CloseTimestamp']),
                        "CloseTimestamp column should be of datetime type after parsing JSON.")

        # Verify the new trade's OpenTimestamp content
        expected_new_open_ts_str = f"{open_date_new}T{open_time_new}"
        expected_new_open_ts = pd.to_datetime(expected_new_open_ts_str)
        actual_new_open_ts = pd.to_datetime(new_trade_row_df['OpenTimestamp'].iloc[0])
        # Compare by normalizing (e.g. removing timezone if one has it and other doesn't, or converting both to UTC)
        self.assertEqual(actual_new_open_ts.tz_localize(None) if actual_new_open_ts.tzinfo else actual_new_open_ts,
                         expected_new_open_ts.tz_localize(None) if expected_new_open_ts.tzinfo else expected_new_open_ts,
                         "New trade's OpenTimestamp does not match expected value.")

        # Verify the new trade's CloseTimestamp content
        expected_new_close_ts_str = f"{close_date_new}T{close_time_new}"
        expected_new_close_ts = pd.to_datetime(expected_new_close_ts_str)
        actual_new_close_ts = pd.to_datetime(new_trade_row_df['CloseTimestamp'].iloc[0])
        self.assertEqual(actual_new_close_ts.tz_localize(None) if actual_new_close_ts.tzinfo else actual_new_close_ts,
                         expected_new_close_ts.tz_localize(None) if expected_new_close_ts.tzinfo else expected_new_close_ts,
                         "New trade's CloseTimestamp does not match expected value.")

        # Verify the original trade's OpenTimestamp content is preserved
        original_trade_row_df = updated_df[updated_df['TradeID'] == 'T1']
        expected_original_open_ts = pd.to_datetime(initial_trades_list[0]['OpenTimestamp'])
        actual_original_open_ts = pd.to_datetime(original_trade_row_df['OpenTimestamp'].iloc[0])
        self.assertEqual(actual_original_open_ts.tz_localize(None) if actual_original_open_ts.tzinfo else actual_original_open_ts,
                         expected_original_open_ts.tz_localize(None) if expected_original_open_ts.tzinfo else expected_original_open_ts,
                         "Original trade's OpenTimestamp was altered.")

# To make this runnable directly for testing purposes:
# if __name__ == '__main__':
#     # This part is tricky if testing.py is inside a package like app.utils
#     # You might need to adjust sys.path or run as a module: python -m app.utils.testing
#     unittest.main()

# If you intend to run this file directly as a script (e.g., python app/utils/testing.py),
# and it's part of a package, Python's import system can be tricky.
# Using `python -m unittest app.utils.testing` from the project root is often more robust.
# The `from .journal_management import add_trade` assumes standard package structure.

from unittest.mock import patch, MagicMock
import dash
from .journal_management import handle_select_deselect_all, delete_trade, save_table_changes # Added save_table_changes

class TestJournalManagement(unittest.TestCase):

    @staticmethod
    def _create_sample_trade_df(trade_details_list):
        required_cols = ['TradeID', 'OpenTimestamp', 'CloseTimestamp', 'Symbol', 'Exchange', 
                         'PositionType', 'EntryPrice', 'ExitPrice', 'Quantity', 'Commission', 
                         'SwapFees', 'GrossP&L', 'NetP&L', 'AlgorithmID', 'Parameters', 
                         'SignalName_Exit', 'ProductType']
        
        df = pd.DataFrame(trade_details_list)
        for col in required_cols:
            if col not in df.columns:
                if 'Timestamp' in col: df[col] = pd.NaT
                elif col in ['EntryPrice', 'ExitPrice', 'Quantity', 'Commission', 'SwapFees', 'GrossP&L', 'NetP&L']: df[col] = 0.0
                elif col == 'TradeID': df[col] = [f"AutoGenID{i}" for i in range(len(df))] # Ensure TradeID if missing
                else: df[col] = ""
        
        df['OpenTimestamp'] = pd.to_datetime(df['OpenTimestamp'], errors='coerce')
        df['CloseTimestamp'] = pd.to_datetime(df['CloseTimestamp'], errors='coerce')
        for col in ['EntryPrice', 'ExitPrice', 'Quantity', 'Commission', 'SwapFees', 'GrossP&L', 'NetP&L']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        # Ensure TradeID is string for consistency, as it's used for merging/lookup
        if 'TradeID' in df.columns:
            df['TradeID'] = df['TradeID'].astype(str)
        return df

    def test_add_trade_datetime_conversion(self):
        # 1. Setup initial data
        initial_trades_list = [{
            'TradeID': 'T1',
            'OpenTimestamp': '2023-12-01T10:00:00.000Z', # ISO format
            'CloseTimestamp': '2023-12-01T11:00:00.000Z',# ISO format
            'Symbol': 'AAPL',
            'Exchange': 'NASDAQ',
            'PositionType': 'Long',
            'EntryPrice': 150.0,
            'ExitPrice': 151.0,
            'Quantity': 10.0,
            'Commission': 1.0,
            'SwapFees': 0.0,
            'GrossP&L': 10.0,
            'NetP&L': 9.0,
            'AlgorithmID': 'Algo1',
            'Parameters': '{}',
            'SignalName_Exit': 'Manual',
            'ProductType': 'Stock'
        }]
        initial_df = pd.DataFrame(initial_trades_list)
        # Ensure OpenTimestamp and CloseTimestamp are datetime objects before converting to JSON
        initial_df['OpenTimestamp'] = pd.to_datetime(initial_df['OpenTimestamp'])
        initial_df['CloseTimestamp'] = pd.to_datetime(initial_df['CloseTimestamp'])
        initial_json_data = initial_df.to_json(date_format='iso', orient='split')

        # New trade inputs (simulating form inputs)
        trade_id_new = "T2"
        symbol_new = "MSFT"
        exchange_new = "NASDAQ"
        open_date_new = "2024-01-15" # Date string from DatePicker
        open_time_new = "09:30:00"   # Time string from Input
        close_date_new = "2024-01-15"
        close_time_new = "10:30:00"
        position_type_new = "Long"
        product_type_new = "Stock"
        quantity_new = 100.0
        entry_price_new = 300.0
        exit_price_new = 305.0
        commission_new = 5.0
        swap_fees_new = 0.0
        # P&L typically calculated or entered, ensure these are float
        gross_pnl_new = 500.0 # (305-300)*100
        net_pnl_new = 495.0   # 500 - 5
        algorithm_id_new = "Algo2"
        exit_signal_new = "TAKE_PROFIT"
        parameters_new = "{}"
        # Other fields that might be in new_trade but not explicitly part of test logic can be added if required by add_trade
        # For example, 'Tags' or 'Notes' if they were mandatory. Assuming they are not for this test.

        # 2. Execution
        try:
            updated_json_data = add_trade(
                n_clicks=1,
                json_data=initial_json_data,
                trade_id=trade_id_new,
                symbol=symbol_new,
                exchange=exchange_new,
                open_date=open_date_new,
                open_time=open_time_new,
                close_date=close_date_new,
                close_time=close_time_new,
                position_type=position_type_new,
                product_type=product_type_new,
                quantity=quantity_new,
                entry_price=entry_price_new,
                exit_price=exit_price_new,
                commission=commission_new,
                swap_fees=swap_fees_new,
                gross_pnl=gross_pnl_new,
                net_pnl=net_pnl_new,
                algorithm_id=algorithm_id_new,
                exit_signal=exit_signal_new,
                parameters=parameters_new
            )
        except ValueError as e:
            self.fail(f"add_trade raised ValueError unexpectedly: {e}")
        except Exception as e: # Catch any other unexpected error
            self.fail(f"add_trade raised an unexpected exception: {e}")


        # 3. Assertions
        self.assertIsNotNone(updated_json_data, "Updated JSON data should not be None")
        
        # Parse the output JSON
        updated_df = pd.read_json(io.StringIO(updated_json_data), orient='split')
        
        self.assertEqual(len(updated_df), 2, "DataFrame should contain two trades (initial + new)")
        
        # Verify the new trade is present
        new_trade_row_df = updated_df[updated_df['TradeID'] == trade_id_new]
        self.assertFalse(new_trade_row_df.empty, f"New trade with ID {trade_id_new} not found")
        
        # Verify timestamps for all trades are datetime objects after pd.read_json
        # This confirms they were stored in a recognizable ISO format.
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(updated_df['OpenTimestamp']),
                        "OpenTimestamp column should be of datetime type after parsing JSON.")
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(updated_df['CloseTimestamp']),
                        "CloseTimestamp column should be of datetime type after parsing JSON.")

        # Verify the new trade's OpenTimestamp content
        expected_new_open_ts_str = f"{open_date_new}T{open_time_new}"
        expected_new_open_ts = pd.to_datetime(expected_new_open_ts_str)
        actual_new_open_ts = pd.to_datetime(new_trade_row_df['OpenTimestamp'].iloc[0])
        # Compare by normalizing (e.g. removing timezone if one has it and other doesn't, or converting both to UTC)
        self.assertEqual(actual_new_open_ts.tz_localize(None) if actual_new_open_ts.tzinfo else actual_new_open_ts,
                         expected_new_open_ts.tz_localize(None) if expected_new_open_ts.tzinfo else expected_new_open_ts,
                         "New trade's OpenTimestamp does not match expected value.")

        # Verify the new trade's CloseTimestamp content
        expected_new_close_ts_str = f"{close_date_new}T{close_time_new}"
        expected_new_close_ts = pd.to_datetime(expected_new_close_ts_str)
        actual_new_close_ts = pd.to_datetime(new_trade_row_df['CloseTimestamp'].iloc[0])
        self.assertEqual(actual_new_close_ts.tz_localize(None) if actual_new_close_ts.tzinfo else actual_new_close_ts,
                         expected_new_close_ts.tz_localize(None) if expected_new_close_ts.tzinfo else expected_new_close_ts,
                         "New trade's CloseTimestamp does not match expected value.")

        # Verify the original trade's OpenTimestamp content is preserved
        original_trade_row_df = updated_df[updated_df['TradeID'] == 'T1']
        expected_original_open_ts = pd.to_datetime(initial_trades_list[0]['OpenTimestamp'])
        actual_original_open_ts = pd.to_datetime(original_trade_row_df['OpenTimestamp'].iloc[0])
        self.assertEqual(actual_original_open_ts.tz_localize(None) if actual_original_open_ts.tzinfo else actual_original_open_ts,
                         expected_original_open_ts.tz_localize(None) if expected_original_open_ts.tzinfo else expected_original_open_ts,
                         "Original trade's OpenTimestamp was altered.")

    def test_add_trade_validation_failure(self):
        initial_json_data = None # Or some valid minimal JSON if needed by internal logic
        
        # Call add_trade with a missing required field (e.g., trade_id is None)
        returned_json_data, message = add_trade(
            n_clicks=1,
            json_data=initial_json_data,
            trade_id=None, # Missing trade_id
            symbol="MSFT",
            exchange="NASDAQ",
            open_date="2024-01-15",
            open_time="09:30:00",
            close_date="2024-01-15",
            close_time="10:30:00",
            position_type="Long",
            product_type="Stock",
            quantity=100.0,
            entry_price=300.0,
            exit_price=305.0,
            commission=5.0,
            swap_fees=0.0,
            gross_pnl=500.0,
            net_pnl=495.0,
            algorithm_id="Algo2",
            exit_signal="TAKE_PROFIT",
            parameters="{}"
        )

        self.assertEqual(returned_json_data, initial_json_data, 
                         "JSON data should be unchanged on validation failure.")
        self.assertTrue(message, "A message should be returned on validation failure.")
        self.assertIn("required fields missing", message.lower(), 
                      "Message should indicate missing required fields.")
        self.assertIn("trade id", message.lower(), 
                      "Message should specifically mention 'Trade ID' as missing.")

    @patch('dash.callback_context', MagicMock())
    def test_handle_select_all_button_click(self):
        # Simulate 'select-all-button' was clicked
        dash.callback_context.triggered = [{'prop_id': 'select-all-button.n_clicks'}]
        
        sample_data_1 = [{'id': 1, 'value': 'A'}, {'id': 2, 'value': 'B'}, {'id': 3, 'value': 'C'}]
        result_1 = handle_select_deselect_all(1, None, sample_data_1)
        self.assertEqual(result_1, list(range(len(sample_data_1))), "Should select all rows.")

        # Test with empty data
        sample_data_2 = []
        result_2 = handle_select_deselect_all(1, None, sample_data_2)
        self.assertEqual(result_2, [], "Should return empty list for empty data.")

    @patch('dash.callback_context', MagicMock())
    def test_deselect_all_button_click(self):
        # Simulate 'deselect-all-button' was clicked
        dash.callback_context.triggered = [{'prop_id': 'deselect-all-button.n_clicks'}]
        
        sample_data = [{'id': 1, 'value': 'A'}, {'id': 2, 'value': 'B'}]
        result = handle_select_deselect_all(None, 1, sample_data)
        self.assertEqual(result, [], "Should deselect all rows, returning an empty list.")

    @patch('dash.callback_context', MagicMock())
    def test_no_relevant_trigger(self):
        sample_data = [{'id': 1, 'value': 'A'}]

        # Scenario 1: No trigger (empty triggered list)
        dash.callback_context.triggered = []
        result_no_trigger = handle_select_deselect_all(None, None, sample_data)
        self.assertEqual(result_no_trigger, dash.no_update, "Should return dash.no_update if context.triggered is empty.")

        # Scenario 2: Trigger from an unrelated button
        dash.callback_context.triggered = [{'prop_id': 'other-button.n_clicks'}]
        result_other_trigger = handle_select_deselect_all(None, None, sample_data)
        self.assertEqual(result_other_trigger, dash.no_update, "Should return dash.no_update for unrelated trigger.")

    # --- Tests for delete_trade ---
    def test_delete_trade_successful(self):
        sample_details = [
            {'TradeID': 'T1', 'Symbol': 'S1', 'OpenTimestamp': '2023-01-01T10:00:00Z'},
            {'TradeID': 'T2', 'Symbol': 'S2', 'OpenTimestamp': '2023-01-02T10:00:00Z'},
            {'TradeID': 'T3', 'Symbol': 'S3', 'OpenTimestamp': '2023-01-03T10:00:00Z'}
        ]
        initial_trades_df = self._create_sample_trade_df(sample_details)
        initial_json_data = initial_trades_df.to_json(date_format='iso', orient='split')
        table_data_view = initial_trades_df.to_dict('records')
        
        # Select T1 and T3 for deletion (indices 0 and 2)
        selected_indices = [0, 2] 
        trade_ids_to_delete = [table_data_view[0]['TradeID'], table_data_view[2]['TradeID']]

        updated_json_data, message = delete_trade(
            n_clicks=1, 
            json_data=initial_json_data, 
            selected_rows_indices=selected_indices, 
            table_data_view=table_data_view
        )

        self.assertIsNotNone(updated_json_data, "JSON data should not be None after deletion.")
        updated_df = pd.read_json(io.StringIO(updated_json_data), orient='split')
        
        self.assertEqual(len(updated_df), 1, "DataFrame should have one trade left.")
        self.assertNotIn(trade_ids_to_delete[0], updated_df['TradeID'].astype(str).values)
        self.assertNotIn(trade_ids_to_delete[1], updated_df['TradeID'].astype(str).values)
        self.assertIn('T2', updated_df['TradeID'].astype(str).values, "T2 should remain.")
        self.assertEqual(message, "2 trade(s) deleted successfully.")

    def test_delete_trade_no_selection(self):
        sample_details = [{'TradeID': 'T1', 'Symbol': 'S1', 'OpenTimestamp': '2023-01-01T10:00:00Z'}]
        initial_trades_df = self._create_sample_trade_df(sample_details)
        initial_json_data = initial_trades_df.to_json(date_format='iso', orient='split')
        table_data_view = initial_trades_df.to_dict('records')

        returned_json_data, message = delete_trade(
            n_clicks=1, 
            json_data=initial_json_data, 
            selected_rows_indices=[], # No selection
            table_data_view=table_data_view
        )
        self.assertEqual(returned_json_data, initial_json_data, "Data should be unchanged if no rows selected.")
        self.assertEqual(message, "No trades selected for deletion.")

    def test_delete_trade_id_not_found(self):
        initial_trades_df = self._create_sample_trade_df([
            {'TradeID': 'T1', 'Symbol': 'S1', 'OpenTimestamp': '2023-01-01T10:00:00Z'}
        ])
        initial_json_data = initial_trades_df.to_json(date_format='iso', orient='split')
        
        # table_data_view has a trade 'T_NONEXISTENT' that is not in initial_json_data
        table_data_view_with_nonexistent = [
            {'TradeID': 'T_NONEXISTENT', 'Symbol': 'S_BAD', 'OpenTimestamp': '2023-01-05T10:00:00Z'}
        ] 
        # Fill other required fields for table_data_view to be a valid list of dicts for delete_trade
        # This part might need adjustment based on actual fields `delete_trade` expects in `table_data_view`
        # For this test, only TradeID from table_data_view is critical.
        # However, the helper function creates a full DF, so we can use it to create a consistent dict.
        temp_df_for_view = self._create_sample_trade_df(table_data_view_with_nonexistent)
        populated_table_data_view = temp_df_for_view.to_dict('records')


        returned_json_data, message = delete_trade(
            n_clicks=1, 
            json_data=initial_json_data, 
            selected_rows_indices=[0], # Selected the non-existent trade from table_data_view
            table_data_view=populated_table_data_view 
        )

        self.assertEqual(returned_json_data, initial_json_data, "Data should be unchanged if selected TradeID not in stored data.")
        self.assertEqual(message, "No trades were deleted. Selected trades might have already been removed or did not match existing trade IDs.")

    def test_delete_trade_no_initial_data(self):
        # table_data_view for the case where selection happens on client but server store is None
        table_data_view_sample = self._create_sample_trade_df([
            {'TradeID': 'T1', 'Symbol': 'S1', 'OpenTimestamp': '2023-01-01T10:00:00Z'}
        ]).to_dict('records')

        returned_data, message = delete_trade(
            n_clicks=1, 
            json_data=None, 
            selected_rows_indices=[0], 
            table_data_view=table_data_view_sample
        )
        self.assertIsNone(returned_data, "Returned data should be None if initial data is None.")
        self.assertEqual(message, "Cannot delete trades: No trade data exists.")

    def test_delete_trade_no_n_clicks(self):
        initial_json_data = self._create_sample_trade_df([
             {'TradeID': 'T1', 'Symbol': 'S1', 'OpenTimestamp': '2023-01-01T10:00:00Z'}
        ]).to_json(date_format='iso', orient='split')
        table_data_view = [] # Not relevant for this test path

        # Test with n_clicks = None
        returned_data_none_clicks, message_none_clicks = delete_trade(
            n_clicks=None, 
            json_data=initial_json_data, 
            selected_rows_indices=[0], 
            table_data_view=table_data_view
        )
        self.assertEqual(returned_data_none_clicks, initial_json_data, "Data should be unchanged if n_clicks is None.")
        self.assertEqual(message_none_clicks, dash.no_update, "Message should be dash.no_update if n_clicks is None.")

        # Test with n_clicks = 0
        returned_data_zero_clicks, message_zero_clicks = delete_trade(
            n_clicks=0, 
            json_data=initial_json_data, 
            selected_rows_indices=[0], 
            table_data_view=table_data_view
        )
        self.assertEqual(returned_data_zero_clicks, initial_json_data, "Data should be unchanged if n_clicks is 0.")
        self.assertEqual(message_zero_clicks, dash.no_update, "Message should be dash.no_update if n_clicks is 0.")

    def test_delete_trade_empty_table_data_view_but_selected(self):
        """ Test when selected_rows_indices has values but table_data_view is empty or None. """
        initial_trades_df = self._create_sample_trade_df([
            {'TradeID': 'T1', 'Symbol': 'S1', 'OpenTimestamp': '2023-01-01T10:00:00Z'}
        ])
        initial_json_data = initial_trades_df.to_json(date_format='iso', orient='split')

        returned_json_data, message = delete_trade(
            n_clicks=1,
            json_data=initial_json_data,
            selected_rows_indices=[0], # A row is selected
            table_data_view=None # But the view data is None
        )
        self.assertEqual(returned_json_data, initial_json_data)
        self.assertEqual(message, "No valid TradeIDs found in the current selection to delete.")

        returned_json_data_empty, message_empty = delete_trade(
            n_clicks=1,
            json_data=initial_json_data,
            selected_rows_indices=[0], # A row is selected
            table_data_view=[] # Or the view data is empty
        )
        self.assertEqual(returned_json_data_empty, initial_json_data)
        self.assertEqual(message_empty, "No valid TradeIDs found in the current selection to delete.")

    # --- Tests for save_table_changes ---
    def test_save_table_changes_successful_update(self):
        initial_details = [
            {'TradeID': 'T1', 'Symbol': 'OLD_SYM', 'EntryPrice': 100.0, 'OpenTimestamp': '2023-01-01T10:00:00Z', 'NetP&L': 10.0},
            {'TradeID': 'T2', 'Symbol': 'XYZ', 'EntryPrice': 200.0, 'OpenTimestamp': '2023-01-02T10:00:00Z', 'NetP&L': 20.0}
        ]
        initial_trades_df = self._create_sample_trade_df(initial_details)
        initial_json_data = initial_trades_df.to_json(date_format='iso', orient='split')

        # Create table_data (list of dicts) for the Dash DataTable
        # T1 is edited, T2 is present but unchanged from initial_df's perspective for this test
        edited_T1_dict = initial_trades_df[initial_trades_df['TradeID'] == 'T1'].to_dict('records')[0]
        edited_T1_dict['Symbol'] = 'NEW_SYM'
        edited_T1_dict['EntryPrice'] = 105.0
        # Ensure OpenTimestamp is string as it would be from table
        edited_T1_dict['OpenTimestamp'] = pd.to_datetime(edited_T1_dict['OpenTimestamp']).strftime('%Y-%m-%d %H:%M:%S')


        T2_dict_from_df = initial_trades_df[initial_trades_df['TradeID'] == 'T2'].to_dict('records')[0]
        # Ensure T2's OpenTimestamp is also string
        T2_dict_from_df['OpenTimestamp'] = pd.to_datetime(T2_dict_from_df['OpenTimestamp']).strftime('%Y-%m-%d %H:%M:%S')
        if 'CloseTimestamp' in T2_dict_from_df and pd.notna(T2_dict_from_df['CloseTimestamp']):
             T2_dict_from_df['CloseTimestamp'] = pd.to_datetime(T2_dict_from_df['CloseTimestamp']).strftime('%Y-%m-%d %H:%M:%S')
        else:
            T2_dict_from_df['CloseTimestamp'] = None # Or some other appropriate string representation for NaT if needed by table


        edited_table_data = [edited_T1_dict, T2_dict_from_df]


        updated_json_data, message = save_table_changes(
            n_clicks=1,
            json_data=initial_json_data,
            table_data=edited_table_data
        )

        self.assertIsNotNone(updated_json_data)
        self.assertEqual(message, "Changes saved successfully.")
        
        updated_df = pd.read_json(io.StringIO(updated_json_data), orient='split')
        updated_df['OpenTimestamp'] = pd.to_datetime(updated_df['OpenTimestamp']) # Convert back for comparison

        self.assertEqual(len(updated_df), 2)
        
        updated_T1_series = updated_df[updated_df['TradeID'].astype(str) == 'T1'].iloc[0]
        self.assertEqual(updated_T1_series['Symbol'], 'NEW_SYM')
        self.assertEqual(updated_T1_series['EntryPrice'], 105.0)
        
        # Check T2 (should be unchanged in terms of key values)
        original_T2_series = initial_trades_df[initial_trades_df['TradeID'] == 'T2'].iloc[0]
        updated_T2_series = updated_df[updated_df['TradeID'].astype(str) == 'T2'].iloc[0]
        self.assertEqual(updated_T2_series['Symbol'], original_T2_series['Symbol'])
        self.assertEqual(updated_T2_series['EntryPrice'], original_T2_series['EntryPrice'])


    def test_save_table_changes_no_effective_change(self):
        initial_details = [{'TradeID': 'T1', 'Symbol': 'SYM', 'EntryPrice': 100.0, 'OpenTimestamp': '2023-01-01T10:00:00Z'}]
        initial_trades_df = self._create_sample_trade_df(initial_details)
        initial_json_data = initial_trades_df.to_json(date_format='iso', orient='split')
        
        # table_data is identical to initial_json_data content
        table_data_dicts = initial_trades_df.to_dict('records')
        # Convert datetime objects in dicts to strings, as they might be in the table
        for record in table_data_dicts:
            if 'OpenTimestamp' in record and pd.notna(record['OpenTimestamp']):
                record['OpenTimestamp'] = pd.to_datetime(record['OpenTimestamp']).strftime('%Y-%m-%d %H:%M:%S')
            if 'CloseTimestamp' in record and pd.notna(record['CloseTimestamp']):
                record['CloseTimestamp'] = pd.to_datetime(record['CloseTimestamp']).strftime('%Y-%m-%d %H:%M:%S')
            else: # Ensure NaT is represented as None or empty string if that's how table gives it
                record['CloseTimestamp'] = None


        updated_json_data, message = save_table_changes(
            n_clicks=1,
            json_data=initial_json_data,
            table_data=table_data_dicts
        )
        self.assertEqual(message, "Changes saved successfully.") # Current function doesn't detect "no change"
        # Compare content as re-serialization might occur
        updated_df = pd.read_json(io.StringIO(updated_json_data), orient='split')
        pd.testing.assert_frame_equal(initial_trades_df.reset_index(drop=True), updated_df.reset_index(drop=True), check_dtype=False)


    def test_save_table_changes_empty_table_data(self):
        initial_details = [{'TradeID': 'T1', 'Symbol': 'SYM', 'EntryPrice': 100.0, 'OpenTimestamp': '2023-01-01T10:00:00Z'}]
        initial_trades_df = self._create_sample_trade_df(initial_details)
        initial_json_data = initial_trades_df.to_json(date_format='iso', orient='split')

        returned_json_data, message = save_table_changes(
            n_clicks=1,
            json_data=initial_json_data,
            table_data=[] # Empty table data
        )
        self.assertEqual(returned_json_data, initial_json_data, "JSON data should be unchanged if table_data is empty.")
        self.assertEqual(message, "No changes to save from the table (table view is empty).")

    def test_save_table_changes_initial_save_to_empty_store(self):
        initial_json_data = None
        
        new_trades_details = [
            {'TradeID': 'N1', 'Symbol': 'NEW1', 'EntryPrice': 50.0, 'OpenTimestamp': '2023-02-01T10:00:00Z'},
            {'TradeID': 'N2', 'Symbol': 'NEW2', 'EntryPrice': 60.0, 'OpenTimestamp': '2023-02-02T10:00:00Z'}
        ]
        # Use helper to create dicts with all necessary fields for table_data
        table_data_df = self._create_sample_trade_df(new_trades_details)
        table_data_list_of_dicts = table_data_df.to_dict('records')
        # Convert datetimes to strings for table_data simulation
        for record in table_data_list_of_dicts:
            if 'OpenTimestamp' in record and pd.notna(record['OpenTimestamp']):
                record['OpenTimestamp'] = pd.to_datetime(record['OpenTimestamp']).strftime('%Y-%m-%d %H:%M:%S')
            if 'CloseTimestamp' in record and pd.notna(record['CloseTimestamp']):
                 record['CloseTimestamp'] = pd.to_datetime(record['CloseTimestamp']).strftime('%Y-%m-%d %H:%M:%S')
            else:
                record['CloseTimestamp'] = None


        updated_json_data, message = save_table_changes(
            n_clicks=1,
            json_data=initial_json_data,
            table_data=table_data_list_of_dicts
        )
        self.assertIsNotNone(updated_json_data)
        self.assertEqual(message, "Changes saved successfully.")
        
        updated_df = pd.read_json(io.StringIO(updated_json_data), orient='split')
        self.assertEqual(len(updated_df), 2)
        self.assertIn('N1', updated_df['TradeID'].astype(str).values)
        self.assertIn('N2', updated_df['TradeID'].astype(str).values)

    def test_save_table_changes_add_new_trade_to_existing(self):
        initial_details = [{'TradeID': 'T1', 'Symbol': 'SYM', 'EntryPrice': 100.0, 'OpenTimestamp': '2023-01-01T10:00:00Z'}]
        initial_trades_df = self._create_sample_trade_df(initial_details)
        initial_json_data = initial_trades_df.to_json(date_format='iso', orient='split')

        # table_data includes original T1 and a new trade T3
        t1_dict = initial_trades_df[initial_trades_df['TradeID'] == 'T1'].to_dict('records')[0]
        t1_dict['OpenTimestamp'] = pd.to_datetime(t1_dict['OpenTimestamp']).strftime('%Y-%m-%d %H:%M:%S')
        if 'CloseTimestamp' in t1_dict and pd.notna(t1_dict['CloseTimestamp']):
             t1_dict['CloseTimestamp'] = pd.to_datetime(t1_dict['CloseTimestamp']).strftime('%Y-%m-%d %H:%M:%S')
        else:
            t1_dict['CloseTimestamp'] = None


        new_trade_t3_details = [{'TradeID': 'T3', 'Symbol': 'NEW_ADD', 'EntryPrice': 300.0, 'OpenTimestamp': '2023-03-01T10:00:00Z'}]
        t3_df = self._create_sample_trade_df(new_trade_t3_details)
        t3_dict = t3_df.to_dict('records')[0]
        t3_dict['OpenTimestamp'] = pd.to_datetime(t3_dict['OpenTimestamp']).strftime('%Y-%m-%d %H:%M:%S')
        if 'CloseTimestamp' in t3_dict and pd.notna(t3_dict['CloseTimestamp']):
            t3_dict['CloseTimestamp'] = pd.to_datetime(t3_dict['CloseTimestamp']).strftime('%Y-%m-%d %H:%M:%S')
        else:
            t3_dict['CloseTimestamp'] = None

        table_data = [t1_dict, t3_dict]


        updated_json_data, message = save_table_changes(
            n_clicks=1,
            json_data=initial_json_data,
            table_data=table_data
        )
        self.assertEqual(message, "Changes saved successfully.")
        updated_df = pd.read_json(io.StringIO(updated_json_data), orient='split')
        self.assertEqual(len(updated_df), 2)
        self.assertIn('T1', updated_df['TradeID'].astype(str).values)
        self.assertIn('T3', updated_df['TradeID'].astype(str).values)
        self.assertEqual(updated_df[updated_df['TradeID'] == 'T3'].iloc[0]['Symbol'], 'NEW_ADD')


    def test_save_table_changes_error_malformed_data(self):
        initial_details = [{'TradeID': 'T1', 'Symbol': 'SYM', 'EntryPrice': 100.0, 'OpenTimestamp': '2023-01-01T10:00:00Z'}]
        initial_trades_df = self._create_sample_trade_df(initial_details)
        initial_json_data = initial_trades_df.to_json(date_format='iso', orient='split')

        # Malformed table_data: TradeID set to a list, which will cause error during set_index or astype(str)
        malformed_table_data = [
            {'TradeID': ['T1_bad_id'], 'Symbol': 'BAD', 'EntryPrice': 'not-a-number', 'OpenTimestamp': 'not-a-date'}
        ]

        returned_json_data, message = save_table_changes(
            n_clicks=1,
            json_data=initial_json_data,
            table_data=malformed_table_data
        )
        self.assertEqual(returned_json_data, initial_json_data, "Original JSON data should be returned on error.")
        self.assertTrue("An error occurred while saving changes:" in message, f"Message should indicate error. Got: {message}")

    def test_save_table_changes_no_n_clicks(self):
        initial_json_data = self._create_sample_trade_df([
             {'TradeID': 'T1', 'Symbol': 'S1', 'OpenTimestamp': '2023-01-01T10:00:00Z'}
        ]).to_json(date_format='iso', orient='split')
        table_data = [] # Not relevant for this test path if n_clicks is None/0

        returned_data_none, msg_none = save_table_changes(None, initial_json_data, table_data)
        self.assertEqual(returned_data_none, initial_json_data)
        self.assertEqual(msg_none, dash.no_update)

        returned_data_zero, msg_zero = save_table_changes(0, initial_json_data, table_data)
        self.assertEqual(returned_data_zero, initial_json_data)
        self.assertEqual(msg_zero, dash.no_update)
