"""
Data Loader Module
-----------------
This module handles loading and preprocessing of trading data from CSV files.
"""

import pandas as pd
import io
from typing import Dict, Any, Optional
import os
from glob import glob # This import is no longer used, can be removed later if desired


def load_trade_csv(file_content_string: str) -> pd.DataFrame:
    """
    Load trading data from a CSV file content string.
    
    Args:
        file_content_string: Decoded file content string from dcc.Upload
        
    Returns:
        Pandas DataFrame containing the trading data
    """
    try:
        # Use StringIO to convert string to file-like object for pandas
        string_io = io.StringIO(file_content_string)
        # Use quoting to handle JSON in Parameters column which contains commas
        df = pd.read_csv(string_io, quoting=1)  # QUOTE_ALL = 1
        return df
    except Exception as e:
        raise ValueError(f"Error loading CSV data: {str(e)}")


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the raw trading data.
    
    Args:
        df: Raw DataFrame from load_trade_csv
        
    Returns:
        Processed DataFrame with correct data types and validations
    """
    # Validate required columns
    required_columns = [
        'TradeID', 'OpenTimestamp', 'CloseTimestamp', 'Symbol', 'Exchange',
        'PositionType', 'EntryPrice', 'ExitPrice', 'Quantity', 'Commission',
        'SwapFees', 'GrossP&L', 'NetP&L', 'AlgorithmID', 'Parameters',
        'SignalName_Exit', 'ProductType'
    ]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
    
    # Create a copy to avoid modifying the original DataFrame
    processed_df = df.copy()
    
    # Convert timestamps to datetime
    try:
        processed_df['OpenTimestamp'] = pd.to_datetime(processed_df['OpenTimestamp'])
        processed_df['CloseTimestamp'] = pd.to_datetime(processed_df['CloseTimestamp'])
    except Exception as e:
        raise ValueError(f"Error converting timestamps: {str(e)}")

    # Extract day of the week from OpenTimestamp
    if 'OpenTimestamp' in processed_df.columns and pd.api.types.is_datetime64_any_dtype(processed_df['OpenTimestamp']):
        processed_df['OpenDayOfWeek'] = processed_df['OpenTimestamp'].dt.day_name()
    else:
        pass

    # Convert numeric columns to float
    numeric_columns = [
        'EntryPrice', 'ExitPrice', 'Quantity', 'Commission', 
        'SwapFees', 'GrossP&L', 'NetP&L'
    ]
    
    for col in numeric_columns:
        try:
            processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
        except Exception as e:
            raise ValueError(f"Error converting {col} to numeric: {str(e)}")
    
    return processed_df


def load_data_csv() -> pd.DataFrame:
    """
    Loads data from 'data/data.csv', performs type conversions, and returns a DataFrame.
    Returns an empty DataFrame if the file doesn't exist or an error occurs.
    """
    data_file_path = os.path.join("data", "data.csv")
    if not os.path.exists(data_file_path):
        return pd.DataFrame()

    try:
        df = pd.read_csv(data_file_path, quoting=1)
        if df.empty:
            return pd.DataFrame()

        # Data Type Conversions
        timestamp_cols = ['OpenTimestamp', 'CloseTimestamp', 'AddedTimestamp']
        for col in timestamp_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

        numeric_cols = [
            'EntryPrice', 'ExitPrice', 'Quantity', 'Commission',
            'SwapFees', 'GrossP&L', 'NetP&L'
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        return df

    except Exception as e:
        # Keep this print for critical load errors
        print(f"Warning: Could not load or process data/data.csv: {e}")
        return pd.DataFrame()


def save_data_csv(df: pd.DataFrame) -> None:
    """
    Saves the given DataFrame to 'data/data.csv'.
    Ensures the 'data/' directory exists.
    """
    data_file_path = os.path.join("data", "data.csv")
    try:
        os.makedirs("data", exist_ok=True)
        df.to_csv(data_file_path, index=False, quoting=1)
    except Exception as e:
        print(f"Error saving data to data/data.csv: {e}")
