"""
Metrics Calculator Module
-----------------------
This module handles calculation of trading metrics and statistics.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional


def calculate_trade_duration(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the duration of each trade.
    
    Args:
        df: DataFrame containing trade data with OpenTimestamp and CloseTimestamp
        
    Returns:
        DataFrame with added TradeDuration column
    """
    result_df = df.copy()
    result_df['TradeDuration'] = result_df['CloseTimestamp'] - result_df['OpenTimestamp']
    return result_df


def calculate_cumulative_pnl(df: pd.DataFrame, pnl_column: str = 'NetP&L') -> pd.DataFrame:
    """
    Calculate cumulative P&L over time.
    
    Args:
        df: DataFrame containing trade data
        pnl_column: Column name for P&L values (default: 'NetP&L')
        
    Returns:
        DataFrame with added CumulativeP&L column
    """
    # Ensure trades are sorted by OpenTimestamp
    result_df = df.copy().sort_values('OpenTimestamp')
    
    # Calculate cumulative sum of P&L
    result_df['CumulativeP&L'] = result_df[pnl_column].cumsum()
    
    return result_df


def calculate_summary_stats(df: pd.DataFrame, pnl_column: str = 'NetP&L') -> Dict[str, Any]:
    """
    Calculate summary statistics for trading performance.
    
    Args:
        df: DataFrame containing trade data
        pnl_column: Column name for P&L values (default: 'NetP&L')
        
    Returns:
        Dictionary containing summary statistics
    """
    if df.empty:
        return {
            'total_pnl': 0,
            'total_trades': 0,
            'win_rate': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'profit_factor': 0,
            'expectancy': 0
        }
    
    # Calculate basic metrics
    total_pnl = df[pnl_column].sum()
    total_trades = len(df)
    
    # Winning and losing trades
    winning_trades = df[df[pnl_column] > 0]
    losing_trades = df[df[pnl_column] < 0]
    
    win_count = len(winning_trades)
    loss_count = len(losing_trades)
    
    # Win rate
    win_rate = win_count / total_trades if total_trades > 0 else 0
    
    # Average win and loss
    avg_win = winning_trades[pnl_column].mean() if win_count > 0 else 0
    avg_loss = losing_trades[pnl_column].mean() if loss_count > 0 else 0
    
    # Profit factor
    gross_profit = winning_trades[pnl_column].sum() if win_count > 0 else 0
    gross_loss = abs(losing_trades[pnl_column].sum()) if loss_count > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0
    
    # Expectancy
    expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss) if total_trades > 0 else 0
    
    return {
        'total_pnl': total_pnl,
        'total_trades': total_trades,
        'win_count': win_count,
        'loss_count': loss_count,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'expectancy': expectancy
    }


def get_stats_per_algorithm(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Calculate summary statistics grouped by AlgorithmID.
    
    Args:
        df: DataFrame containing trade data
        
    Returns:
        Dictionary where keys are AlgorithmID and values are stats dictionaries
    """
    if 'AlgorithmID' not in df.columns:
        return {}
    
    result = {}
    
    # Group by AlgorithmID
    for algo_id, group in df.groupby('AlgorithmID'):
        result[algo_id] = calculate_summary_stats(group)
    
    return result


# --- New Detailed Performance Metrics Calculation ---

def _prepare_df_for_detailed_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares the DataFrame with necessary columns for detailed metric calculations.
    Adds HoldingPeriod, HoldingDays, CapitalDeployed, PnlPct, and LossPct.
    """
    if df.empty:
        # Add empty columns if df is empty to ensure consistency in downstream processing
        df['HoldingPeriod'] = pd.Series(dtype='timedelta64[ns]')
        df['HoldingDays'] = pd.Series(dtype='float64')
        df['CapitalDeployed'] = pd.Series(dtype='float64')
        df['PnlPct'] = pd.Series(dtype='float64')
        df['LossPct'] = pd.Series(dtype='float64')
        return df

    df_copy = df.copy()

    # Ensure Timestamps are datetime objects
    if 'OpenTimestamp' in df_copy.columns:
        df_copy['OpenTimestamp'] = pd.to_datetime(df_copy['OpenTimestamp'], errors='coerce')
    if 'CloseTimestamp' in df_copy.columns:
        df_copy['CloseTimestamp'] = pd.to_datetime(df_copy['CloseTimestamp'], errors='coerce')

    # Calculate HoldingPeriod and HoldingDays
    if 'OpenTimestamp' in df_copy.columns and 'CloseTimestamp' in df_copy.columns:
        df_copy['HoldingPeriod'] = df_copy['CloseTimestamp'] - df_copy['OpenTimestamp']
        df_copy['HoldingDays'] = df_copy['HoldingPeriod'].dt.total_seconds() / (24 * 60 * 60)
    else:
        df_copy['HoldingPeriod'] = pd.NaT
        df_copy['HoldingDays'] = np.nan
        
    # Calculate CapitalDeployed
    if 'Quantity' in df_copy.columns and 'EntryPrice' in df_copy.columns:
        df_copy['CapitalDeployed'] = df_copy['Quantity'] * df_copy['EntryPrice']
        # Handle potential zero or negative capital deployment for percentage calculations
        df_copy['CapitalDeployed'] = df_copy['CapitalDeployed'].replace(0, np.nan) # Avoid division by zero
    else:
        df_copy['CapitalDeployed'] = np.nan

    # Calculate PnlPct and LossPct
    if 'NetP&L' in df_copy.columns and 'CapitalDeployed' in df_copy.columns:
        df_copy['PnlPct'] = (df_copy['NetP&L'] / df_copy['CapitalDeployed']) * 100
        
        df_copy['LossPct'] = np.nan # Initialize LossPct column
        losing_trades_mask = df_copy['NetP&L'] < 0
        # Calculate LossPct only for losing trades and where CapitalDeployed is not NaN
        df_copy.loc[losing_trades_mask & df_copy['CapitalDeployed'].notna(), 'LossPct'] = \
            (df_copy.loc[losing_trades_mask, 'NetP&L'].abs() / df_copy.loc[losing_trades_mask, 'CapitalDeployed']) * 100
    else:
        df_copy['PnlPct'] = np.nan
        df_copy['LossPct'] = np.nan
        
    return df_copy


def calculate_winning_trades_details(df_wins: pd.DataFrame, total_trades: int) -> Dict[str, Any]:
    """Calculates detailed metrics for winning trades."""
    num_winning_trades = len(df_wins)
    if num_winning_trades == 0:
        return {
            'num_winning_trades': 0, 'pct_winning_trades': 0,
            'total_profit_numeric': 0, 'avg_pnl_winning_trades': 0,
            'avg_pnl_pct_winning_trades': 0, 'avg_capital_deployed_winning_trades': 0,
            'max_profit_numeric': 0, 'max_profit_pct': 0, 'avg_holding_days_winning': 0
        }

    total_profit_numeric = df_wins['NetP&L'].sum()
    return {
        'num_winning_trades': num_winning_trades,
        'pct_winning_trades': num_winning_trades / total_trades if total_trades > 0 else 0,
        'total_profit_numeric': total_profit_numeric,
        'avg_pnl_winning_trades': df_wins['NetP&L'].mean(),
        'avg_pnl_pct_winning_trades': df_wins['PnlPct'].mean(), # Assumes PnlPct column exists
        'avg_capital_deployed_winning_trades': df_wins['CapitalDeployed'].mean(), # Assumes CapitalDeployed exists
        'max_profit_numeric': df_wins['NetP&L'].max(),
        'max_profit_pct': df_wins['PnlPct'].max(),
        'avg_holding_days_winning': df_wins['HoldingDays'].mean() # Assumes HoldingDays exists
    }

def calculate_losing_trades_details(df_losses: pd.DataFrame, total_trades: int) -> Dict[str, Any]:
    """Calculates detailed metrics for losing trades."""
    num_losing_trades = len(df_losses)
    if num_losing_trades == 0:
        return {
            'num_losing_trades': 0, 'pct_losing_trades': 0,
            'total_loss_numeric': 0, 'avg_pnl_losing_trades': 0,
            'avg_loss_pct_losing_trades': 0, 'avg_capital_deployed_losing_trades': 0,
            'max_loss_numeric': 0, 'max_loss_pct': 0, 'avg_holding_days_losing': 0
        }
    
    total_loss_numeric = abs(df_losses['NetP&L'].sum()) # Sum of negative P&Ls, then absolute
    return {
        'num_losing_trades': num_losing_trades,
        'pct_losing_trades': num_losing_trades / total_trades if total_trades > 0 else 0,
        'total_loss_numeric': total_loss_numeric,
        'avg_pnl_losing_trades': df_losses['NetP&L'].mean(), # Will be negative
        'avg_loss_pct_losing_trades': df_losses['LossPct'].mean(), # Assumes LossPct column exists
        'avg_capital_deployed_losing_trades': df_losses['CapitalDeployed'].mean(), # Assumes CapitalDeployed exists
        'max_loss_numeric': df_losses['NetP&L'].min(), # Min P&L, most negative
        'max_loss_pct': df_losses['LossPct'].max(), # Max of positive LossPct values
        'avg_holding_days_losing': df_losses['HoldingDays'].mean() # Assumes HoldingDays exists
    }

def calculate_overall_pnl_details(df_prepared: pd.DataFrame) -> Dict[str, Any]:
    """Calculates overall average P&L details."""
    if df_prepared.empty:
        return {
            'overall_avg_pnl_numeric': 0, 'overall_avg_pnl_pct': 0,
            'avg_capital_deployed_all_trades': 0, 'avg_holding_days_all': 0
        }
    return {
        'overall_avg_pnl_numeric': df_prepared['NetP&L'].mean(),
        'overall_avg_pnl_pct': df_prepared['PnlPct'].mean(), # Avg of PnlPct (includes negative PnlPct for losses)
        'avg_capital_deployed_all_trades': df_prepared['CapitalDeployed'].mean(),
        'avg_holding_days_all': df_prepared['HoldingDays'].mean()
    }


def calculate_advanced_ratios(
    summary_stats: Dict[str, Any], 
    winning_details: Dict[str, Any], 
    losing_details: Dict[str, Any]
) -> Dict[str, Any]:
    """Calculates advanced trading ratios."""
    avg_pnl_winning = winning_details.get('avg_pnl_winning_trades', 0)
    avg_pnl_losing = losing_details.get('avg_pnl_losing_trades', 0) # This is negative
    
    avg_pnl_pct_winning = winning_details.get('avg_pnl_pct_winning_trades', 0)
    avg_loss_pct_losing = losing_details.get('avg_loss_pct_losing_trades', 0) # This is positive

    total_profit = winning_details.get('total_profit_numeric', 0)
    total_loss = losing_details.get('total_loss_numeric', 0) # This is positive absolute sum of losses

    win_rate = summary_stats.get('win_rate', 0)
    # avg_win_from_summary = summary_stats.get('avg_win', 0) # This is avg_pnl_winning
    # avg_loss_from_summary = summary_stats.get('avg_loss', 0) # This is avg_pnl_losing (negative)

    # Avg Risk/Reward Ratio (using absolute numeric P&Ls)
    avg_risk_reward_ratio = np.nan
    if avg_pnl_losing != 0:
        avg_risk_reward_ratio = abs(avg_pnl_winning / avg_pnl_losing)
        
    # R-Multiple Average (using P&L percentages)
    r_multiple_avg_pct = np.nan
    if avg_loss_pct_losing != 0 and avg_loss_pct_losing is not np.nan : # Ensure it's not NaN
        r_multiple_avg_pct = avg_pnl_pct_winning / avg_loss_pct_losing

    # Gain/Loss Ratio (Total Profit / Total Loss) - This is essentially the Profit Factor
    gain_loss_ratio_numeric = np.nan
    if total_loss != 0:
        gain_loss_ratio_numeric = total_profit / total_loss
        
    # Optimal F (Kelly Criterion simplified)
    optimal_f = np.nan
    # W = win_rate
    # R = avg_win / abs(avg_loss)
    if avg_pnl_losing != 0:
        R = avg_pnl_winning / abs(avg_pnl_losing)
        if R != 0 : # Avoid division by zero if R is 0 (e.g. avg_pnl_winning is 0)
             optimal_f = ((R + 1) * win_rate - 1) / R
        elif win_rate == 1: # If win rate is 100%, optimal_f is 1 (bet everything)
            optimal_f = 1.0
        else: # If R is 0 and win_rate is not 1, it implies no gain, so optimal_f is effectively 0 or negative
            optimal_f = 0.0 # Or np.nan if preferred for "cannot compute"
    elif win_rate == 1 and avg_pnl_winning > 0 : # All trades win, avg_loss is 0
        optimal_f = 1.0 
    elif avg_pnl_winning == 0 and avg_pnl_losing == 0: # No profit, no loss
        optimal_f = 0.0

    return {
        'avg_risk_reward_ratio_numeric': avg_risk_reward_ratio,
        'r_multiple_avg_pct': r_multiple_avg_pct,
        'gain_loss_ratio_numeric': gain_loss_ratio_numeric, # Same as profit_factor
        'optimal_f': optimal_f if optimal_f is not np.nan and optimal_f > 0 else 0 # Cap at 0 if negative/NaN
    }

def calculate_detailed_performance_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculates a comprehensive set of detailed performance metrics.
    """
    if df is None or df.empty:
        # Return a dictionary with default/NaN values for all expected keys
        return {
            **calculate_summary_stats(pd.DataFrame()), # Get default summary stats
            **calculate_winning_trades_details(pd.DataFrame(), 0),
            **calculate_losing_trades_details(pd.DataFrame(), 0),
            **calculate_overall_pnl_details(pd.DataFrame()),
            **calculate_advanced_ratios({}, {}, {}) # Pass empty dicts
        }

    df_prepared = _prepare_df_for_detailed_metrics(df)
    
    total_trades = len(df_prepared)
    
    df_wins = df_prepared[df_prepared['NetP&L'] > 0].copy()
    df_losses = df_prepared[df_prepared['NetP&L'] < 0].copy()

    summary_stats = calculate_summary_stats(df_prepared) # Basic summary
    winning_details = calculate_winning_trades_details(df_wins, total_trades)
    losing_details = calculate_losing_trades_details(df_losses, total_trades)
    overall_pnl_details = calculate_overall_pnl_details(df_prepared)
    
    advanced_ratios = calculate_advanced_ratios(summary_stats, winning_details, losing_details)

    # Consolidate all metrics
    all_metrics = {
        **summary_stats,
        **winning_details,
        **losing_details,
        **overall_pnl_details,
        **advanced_ratios
    }
    
    # Clean up NaN to None or 0 for JSON compatibility or dashboard display if needed
    for key, value in all_metrics.items():
        if isinstance(value, (float, np.floating)) and np.isnan(value):
            all_metrics[key] = None # Or 0, depending on desired representation
        elif isinstance(value, (float, np.floating)) and np.isinf(value):
             all_metrics[key] = None # Or a very large number string, or 0

    return all_metrics
