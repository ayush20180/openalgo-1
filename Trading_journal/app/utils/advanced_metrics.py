"""
Advanced Metrics Calculator Module
--------------------------------
This module extends the basic metrics calculator with advanced trading statistics.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional


def calculate_avg_holding_time(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate average holding time for all trades, winners, and losers.
    
    Args:
        df: DataFrame containing trade data with TradeDuration and NetP&L columns
        
    Returns:
        Dictionary with average holding times
    """
    if 'TradeDuration' not in df.columns:
        raise ValueError("DataFrame must contain TradeDuration column")
    
    if 'NetP&L' not in df.columns:
        raise ValueError("DataFrame must contain NetP&L column")
    
    # Calculate overall average
    overall_avg = df['TradeDuration'].mean()
    
    # Calculate average for winning trades
    winners = df[df['NetP&L'] > 0]
    winners_avg = winners['TradeDuration'].mean() if not winners.empty else pd.Timedelta(0)
    
    # Calculate average for losing trades
    losers = df[df['NetP&L'] < 0]
    losers_avg = losers['TradeDuration'].mean() if not losers.empty else pd.Timedelta(0)
    
    return {
        'overall_avg': overall_avg,
        'winners_avg': winners_avg,
        'losers_avg': losers_avg
    }


def calculate_max_drawdown(cumulative_pnl_series: pd.Series) -> Dict[str, Any]:
    """
    Calculate maximum drawdown from a cumulative P&L series.
    
    Args:
        cumulative_pnl_series: Series containing cumulative P&L values
        
    Returns:
        Dictionary with drawdown metrics
    """
    if cumulative_pnl_series.empty:
        return {
            'max_drawdown': 0,
            'max_drawdown_pct': 0,
            'peak_idx': None,
            'peak_value': 0,
            'max_drawdown_idx': None,
            'recovery_idx': None,
            'recovery_value': None,
            'recovery_duration': None
        }
    # Calculate running maximum
    running_max = cumulative_pnl_series.cummax()
    
    # Calculate drawdown in dollars
    drawdown = cumulative_pnl_series - running_max
    
    # Find maximum drawdown and its index
    # Handle empty drawdown series (e.g., if all cumulative P&L values are the same or series is too short)
    if drawdown.empty or drawdown.min() >= 0: # No drawdown or all positive
        max_drawdown = 0
        max_drawdown_idx = None
    else:
        max_drawdown = drawdown.min()
        max_drawdown_idx = drawdown.idxmin()
    
    # If max_drawdown_idx is None (no drawdown), we can't proceed with peak/recovery calculations meaningfully
    if max_drawdown_idx is None:
        return {
            'max_drawdown': 0,
            'max_drawdown_pct': 0,
            'peak_idx': None,
            'peak_value': cumulative_pnl_series.iloc[-1] if not cumulative_pnl_series.empty else 0,
            'max_drawdown_idx': None,
            'recovery_idx': None,
            'recovery_value': None,
            'recovery_duration': None
        }
    
    # Find the peak before the maximum drawdown
    peak_idx = running_max.loc[:max_drawdown_idx].idxmax()
    peak_value = running_max.loc[peak_idx]
    
    # Calculate drawdown percentage
    max_drawdown_pct = max_drawdown / peak_value if peak_value != 0 else 0
    
    # Find recovery point (if any)
    try:
        recovery_idx = cumulative_pnl_series.loc[max_drawdown_idx:].ge(peak_value).idxmax()
        recovery_value = cumulative_pnl_series.loc[recovery_idx]
        recovery_duration = recovery_idx - max_drawdown_idx
    except:
        recovery_idx = None
        recovery_value = None
        recovery_duration = None
    
    return {
        'max_drawdown': max_drawdown,
        'max_drawdown_pct': max_drawdown_pct,
        'peak_idx': peak_idx,
        'peak_value': peak_value,
        'max_drawdown_idx': max_drawdown_idx,
        'recovery_idx': recovery_idx,
        'recovery_value': recovery_value,
        'recovery_duration': recovery_duration
    }


def calculate_sharpe_ratio(pnl_series: pd.Series, risk_free_rate: float = 0.0, annualization_factor: int = 252) -> float:
    """
    Calculate Sharpe ratio from a series of P&L values.
    
    Args:
        pnl_series: Series containing P&L values
        risk_free_rate: Annual risk-free rate (default: 0.0)
        annualization_factor: Factor to annualize returns (default: 252 for daily returns)
        
    Returns:
        Sharpe ratio value
    """
    # Calculate mean return and standard deviation
    mean_return = pnl_series.mean()
    std_dev = pnl_series.std()
    
    # Avoid division by zero
    if std_dev == 0:
        return 0
    
    # Calculate daily Sharpe ratio
    daily_sharpe = (mean_return - risk_free_rate / annualization_factor) / std_dev
    
    # Annualize Sharpe ratio
    sharpe_ratio = daily_sharpe * np.sqrt(annualization_factor)
    
    return sharpe_ratio


def calculate_sortino_ratio(pnl_series: pd.Series, risk_free_rate: float = 0.0, annualization_factor: int = 252) -> float:
    """
    Calculate Sortino ratio from a series of P&L values.
    
    Args:
        pnl_series: Series containing P&L values
        risk_free_rate: Annual risk-free rate (default: 0.0)
        annualization_factor: Factor to annualize returns (default: 252 for daily returns)
        
    Returns:
        Sortino ratio value
    """
    # Calculate mean return
    mean_return = pnl_series.mean()
    
    # Calculate downside deviation (standard deviation of negative returns only)
    negative_returns = pnl_series[pnl_series < 0]
    downside_dev = negative_returns.std() if len(negative_returns) > 0 else 0
    
    # Avoid division by zero
    if downside_dev == 0:
        return 0
    
    # Calculate daily Sortino ratio
    daily_sortino = (mean_return - risk_free_rate / annualization_factor) / downside_dev
    
    # Annualize Sortino ratio
    sortino_ratio = daily_sortino * np.sqrt(annualization_factor)
    
    return sortino_ratio


def calculate_performance_by_positiontype(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Calculate performance metrics grouped by position type (Long/Short).
    
    Args:
        df: DataFrame containing trade data
        
    Returns:
        Dictionary with performance metrics by position type
    """
    if 'PositionType' not in df.columns:
        raise ValueError("DataFrame must contain PositionType column")
    
    result = {}
    
    # Group by position type
    for position_type, group in df.groupby('PositionType'):
        # Calculate basic stats for this position type
        from app.utils.metrics_calculator import calculate_summary_stats
        stats = calculate_summary_stats(group)
        
        # Add additional metrics
        stats['count'] = len(group)
        stats['avg_duration'] = group['TradeDuration'].mean() if 'TradeDuration' in group.columns else None
        
        result[position_type] = stats
    
    return result


def analyze_exit_signals(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Analyze exit signals and their performance.
    
    Args:
        df: DataFrame containing trade data
        
    Returns:
        Dictionary with exit signal analysis
    """
    if 'SignalName_Exit' not in df.columns:
        raise ValueError("DataFrame must contain SignalName_Exit column")
    
    result = {}
    
    # Group by exit signal
    for signal, group in df.groupby('SignalName_Exit'):
        # Calculate basic stats for this exit signal
        from app.utils.metrics_calculator import calculate_summary_stats
        stats = calculate_summary_stats(group)
        
        # Add additional metrics
        stats['count'] = len(group)
        stats['frequency'] = len(group) / len(df)
        
        result[signal] = stats
    
    return result


def calculate_consecutive_wins_losses(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate consecutive wins and losses statistics.
    
    Args:
        df: DataFrame containing trade data with NetP&L column
        
    Returns:
        Dictionary with consecutive wins/losses metrics
    """
    if 'NetP&L' not in df.columns:
        raise ValueError("DataFrame must contain NetP&L column")
    
    # Sort by timestamp if available
    if 'OpenTimestamp' in df.columns:
        df = df.sort_values('OpenTimestamp')
    
    # Create a series indicating win (1) or loss (-1)
    results = np.where(df['NetP&L'] > 0, 1, np.where(df['NetP&L'] < 0, -1, 0))
    
    # Calculate consecutive wins and losses
    streaks = []
    current_streak = 0
    
    for result in results:
        if result == 0:  # Breakeven trade
            if current_streak != 0:
                streaks.append(current_streak)
                current_streak = 0
        elif (current_streak > 0 and result > 0) or (current_streak < 0 and result < 0):
            # Continuing the streak
            current_streak += result
        else:
            # New streak
            if current_streak != 0:
                streaks.append(current_streak)
            current_streak = result
    
    # Add the last streak if there is one
    if current_streak != 0:
        streaks.append(current_streak)
    
    # Calculate metrics
    win_streaks = [s for s in streaks if s > 0]
    loss_streaks = [abs(s) for s in streaks if s < 0]
    
    max_consecutive_wins = max(win_streaks) if win_streaks else 0
    max_consecutive_losses = max(loss_streaks) if loss_streaks else 0
    avg_win_streak = sum(win_streaks) / len(win_streaks) if win_streaks else 0
    avg_loss_streak = sum(loss_streaks) / len(loss_streaks) if loss_streaks else 0
    
    return {
        'max_consecutive_wins': max_consecutive_wins,
        'max_consecutive_losses': max_consecutive_losses,
        'avg_win_streak': avg_win_streak,
        'avg_loss_streak': avg_loss_streak,
        'win_streaks': win_streaks,
        'loss_streaks': loss_streaks
    }


def calculate_volatility(pnl_series: pd.Series, window: int = 20) -> pd.Series:
    """
    Calculate rolling volatility of P&L.
    
    Args:
        pnl_series: Series containing P&L values
        window: Rolling window size (default: 20)
        
    Returns:
        Series with rolling volatility values
    """
    # Calculate rolling standard deviation
    volatility = pnl_series.rolling(window=window).std()
    
    return volatility


def analyze_trade_clusters(df: pd.DataFrame, features: List[str] = None) -> Dict[str, Any]:
    """
    Analyze trade clusters based on specified features.
    
    Args:
        df: DataFrame containing trade data
        features: List of features to use for clustering (default: None, uses Symbol and PositionType)
        
    Returns:
        Dictionary with cluster analysis
    """
    if features is None:
        features = ['Symbol', 'PositionType']
    
    # Ensure all features exist in DataFrame
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        raise ValueError(f"DataFrame missing required features: {missing_features}")
    
    # Group by the specified features
    clusters = {}
    
    for name, group in df.groupby(features):
        # Convert name to string if it's a tuple
        cluster_name = '_'.join(name) if isinstance(name, tuple) else str(name)
        
        # Calculate basic stats for this cluster
        from app.utils.metrics_calculator import calculate_summary_stats
        stats = calculate_summary_stats(group)
        
        # Add additional metrics
        stats['count'] = len(group)
        stats['frequency'] = len(group) / len(df)
        
        clusters[cluster_name] = stats
    
    return clusters
