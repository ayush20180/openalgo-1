# Project Structure and Technical Notes

This document provides an overview of the Algorithmic Trading Dashboard project structure and key technical details to help developers understand the codebase and contribute effectively.

## Project Overview

The Algorithmic Trading Dashboard is a Dash-based web application that allows users to upload their trading data (in CSV format), visualize performance metrics, analyze strategies, and manage a trade journal.

## Core Technologies

- **Dash:** The main framework for building the web application and interactive dashboard.
- **Plotly:** Used for creating interactive plots and visualizations.
- **Pandas:** Extensively used for data loading, preprocessing, manipulation, and analysis.
- **Dash Bootstrap Components:** Provides pre-built Bootstrap components for styling the application layout.

## Project Structure

The project is organized as follows:

```
.
├── app/
│   ├── __init__.py
│   ├── main.py
│   └── utils/
│       ├── __init__.py
│       ├── advanced_dashboard.py
│       ├── advanced_metrics.py
│       ├── data_loader.py
│       ├── global_filters.py
│       ├── journal_management.py
│       ├── metrics_calculator.py
│       └── testing.py
├── doc/
│   ├── prd.md
│   ├── structure.md  <- This file
│   └── task.md
├── integration_test.py
├── README.md
├── requirements.txt
├── test_application.py
└── USER_GUIDE.md
```

- **`app/`**: Contains the main application code.
    - **`main.py`**: The entry point of the Dash application. It defines the overall layout, integrates components from utility modules, and registers core callbacks for file upload and overall dashboard updates.
    - **`utils/`**: Contains various modules for specific functionalities.
        - **`data_loader.py`**: Handles loading trade data from CSV files and performing initial preprocessing (data type conversion, validation).
        - **`metrics_calculator.py`**: Contains functions for calculating basic trading performance metrics (total P&L, win rate, profit factor, etc.).
        - **`advanced_metrics.py`**: Provides functions for calculating more advanced trading statistics (drawdown, Sharpe ratio, Sortino ratio, consecutive wins/losses, etc.).
        - **`advanced_dashboard.py`**: Defines the layout and registers callbacks specifically for the "Advanced Analytics" tab, utilizing functions from `advanced_metrics.py` to display detailed analysis.
        - **`global_filters.py`**: Defines the layout and registers callbacks for the global filtering section, allowing users to filter data based on various criteria (date range, symbol, algorithm, etc.).
        - **`journal_management.py`**: Defines the layout and registers callbacks for the "Journal Management" tab, enabling manual entry, editing, and deletion of trade records.
        - **`testing.py`**: (Based on file name) Likely contains utility functions or data for testing purposes.
- **`doc/`**: Contains project documentation.
    - **`prd.md`**: Product Requirements Document.
    - **`structure.md`**: This document, detailing the project structure.
    - **`task.md`**: (Based on file name) Likely contains task-related documentation.
- **`integration_test.py`**: (Based on file name) Contains integration tests for the application.
- **`README.md`**: Provides a general overview and instructions for the project.
- **`requirements.txt`**: Lists the project's dependencies.
- **`test_application.py`**: (Based on file name) Contains unit or functional tests for the application.
- **`USER_GUIDE.md`**: Provides instructions for users on how to use the application.

## Key Components and Functionality

### `app/main.py`

- Initializes the Dash app with `dbc.themes.BOOTSTRAP`.
- Defines the main `app.layout` using `dbc.Container` and `dbc.Row`/`dbc.Col` for structure.
- Includes a `dcc.Upload` component for CSV file uploads.
- Uses `dcc.Store` components (`trade-data-store`, `filtered-data-store`) to store raw and filtered trade data in JSON format.
- Organizes the main dashboard content into `dbc.Tabs`:
    - **Overview:** Displays overall performance metrics, equity curve, and P&L distribution.
    - **Algorithm Analysis:** Allows analysis of performance per algorithm.
    - **Advanced Analytics:** Integrates the advanced metrics and visualizations from `advanced_dashboard.py`.
    - **Trade Details:** Shows a table of trade details.
    - **Journal Management:** Provides the interface for manual trade management from `journal_management.py`.
- Registers callbacks to:
    - Handle file uploads, decode content, load/preprocess data using `data_loader`, and store in `trade-data-store`.
    - Update overall metrics, equity curve, P&L histogram, and algorithm selector options based on `trade-data-store`.
    - Update P&L by period chart based on `trade-data-store` and period selection.
    - Update algorithm-specific metrics and equity curve based on `trade-data-store` and algorithm selection.
    - Update the basic trade table based on `trade-data-store` and algorithm selection.
- Registers callbacks from `advanced_dashboard.py`, `global_filters.py`, and `journal_management.py`.

### `app/utils/data_loader.py`

- **`load_trade_csv(file_content_string)`**:
    - Takes a string of CSV content.
    - Uses `io.StringIO` and `pd.read_csv` to create a DataFrame.
    - Handles quoting issues with `quoting=1`.
    - Includes error handling for loading.
- **`preprocess_data(df)`**:
    - Takes a DataFrame.
    - Validates the presence of required columns.
    - Converts 'OpenTimestamp' and 'CloseTimestamp' to datetime objects.
    - Converts numeric columns ('EntryPrice', 'ExitPrice', 'Quantity', 'Commission', 'SwapFees', 'GrossP&L', 'NetP&L') to numeric types.
    - Returns the processed DataFrame.

### `app/utils/metrics_calculator.py`

- **`calculate_trade_duration(df)`**: Calculates the time difference between `CloseTimestamp` and `OpenTimestamp`.
- **`calculate_cumulative_pnl(df, pnl_column='NetP&L')`**: Calculates the running sum of P&L after sorting by `OpenTimestamp`.
- **`calculate_summary_stats(df, pnl_column='NetP&L')`**: Calculates core metrics: total P&L, total trades, win count, loss count, win rate, average win, average loss, profit factor, and expectancy. Handles empty DataFrames.
- **`get_stats_per_algorithm(df)`**: Groups data by 'AlgorithmID' and applies `calculate_summary_stats` to each group.

### `app/utils/advanced_metrics.py`

- **`calculate_avg_holding_time(df)`**: Calculates average trade duration for all trades, winners, and losers.
- **`calculate_max_drawdown(cumulative_pnl_series)`**: Calculates maximum drawdown, percentage drawdown, and related peak/recovery points from a cumulative P&L series.
- **`calculate_sharpe_ratio(pnl_series, risk_free_rate=0.0, annualization_factor=252)`**: Calculates the Sharpe ratio.
- **`calculate_sortino_ratio(pnl_series, risk_free_rate=0.0, annualization_factor=252)`**: Calculates the Sortino ratio.
- **`calculate_performance_by_positiontype(df)`**: Groups data by 'PositionType' and calculates summary stats and average duration.
- **`analyze_exit_signals(df)`**: Groups data by 'SignalName_Exit' and calculates summary stats and frequency.
- **`calculate_consecutive_wins_losses(df)`**: Analyzes streaks of wins and losses.
- **`calculate_volatility(pnl_series, window=20)`**: Calculates rolling standard deviation of P&L.
- **`analyze_trade_clusters(df, features=None)`**: Groups data by specified features and calculates summary stats and frequency for each cluster.

### `app/utils/advanced_dashboard.py`

- **`create_advanced_metrics_layout()`**: Defines the layout for the advanced analytics tab with sections for various advanced metrics visualizations.
- **`register_advanced_callbacks(app)`**: Registers callbacks to update the advanced analytics charts and metrics displays based on data from the data stores, using functions from `advanced_metrics.py`. Includes callbacks for drawdown, risk-adjusted metrics, position type analysis, exit signal analysis, consecutive wins/losses, volatility, and trade clusters.

### `app/utils/global_filters.py`

- **`create_global_filters_layout()`**: Defines the layout for the global filters card with various input components.
- **`register_filter_callbacks(app)`**: Registers callbacks to:
    - Populate filter dropdown options and P&L slider range based on loaded data.
    - Apply selected filters to the data and store the result in `filtered-data-store`.
    - Reset filters.

### `app/utils/journal_management.py`

- **`create_journal_entry_layout()`**: Defines the layout for the journal management tab, including a form for adding trades and a table for managing existing trades.
- **`register_journal_callbacks(app)`**: Registers callbacks to:
    - Update the editable trade table based on data, search, and sort.
    - Populate the trade entry form for editing or clear it.
    - Calculate P&L in the form based on inputs.
    - Add a new trade or update an existing one in `trade-data-store` from the form.
    - Delete a selected trade from `trade-data-store`.
    - Save changes made directly in the editable table back to `trade-data-store`.

## Data Structure (Expected CSV Columns)

The application expects a CSV file with at least the following columns:

- `TradeID` (Unique identifier for each trade)
- `OpenTimestamp` (Timestamp when the trade was opened)
- `CloseTimestamp` (Timestamp when the trade was closed)
- `Symbol` (Trading instrument symbol)
- `Exchange` (Exchange where the trade occurred)
- `PositionType` (e.g., 'Long', 'Short')
- `EntryPrice` (Price at which the trade was entered)
- `ExitPrice` (Price at which the trade was exited)
- `Quantity` (Size of the position)
- `Commission` (Commission paid for the trade)
- `SwapFees` (Swap fees incurred)
- `GrossP&L` (Profit or Loss before commissions and fees)
- `NetP&L` (Profit or Loss after commissions and fees)
- `AlgorithmID` (Identifier for the algorithm that generated the trade)
- `Parameters` (JSON string or similar, containing algorithm parameters for the trade)
- `SignalName_Exit` (Name of the signal that triggered the exit)
- `ProductType` (e.g., 'Stock', 'Forex', 'Crypto')

Additional columns may be present but are not strictly required by the core functionality.

## Adding New Features

To add new features, consider the following:

1.  **Identify the relevant module(s):** Determine which existing module(s) the new feature aligns with (e.g., adding a new metric would involve `metrics_calculator.py` or `advanced_metrics.py`, adding a new filter would involve `global_filters.py`). If the feature is a major new section, you might need a new module in the `app/utils/` directory.
2.  **Implement core logic:** Write the Python code for the new feature's logic within the appropriate module(s).
3.  **Update layout:** Modify `app/main.py` or the relevant dashboard module (`advanced_dashboard.py`, `journal_management.py`) to add the necessary Dash components to the layout.
4.  **Register callbacks:** Add new callbacks in `app/main.py` or the relevant dashboard module to connect user interactions or data changes to your new logic and update the UI.
5.  **Update documentation:** Add details about the new feature to this `structure.md` file and potentially the `USER_GUIDE.md` if it affects user interaction.
6.  **Add tests:** Write unit or integration tests for your new code in the `test_application.py` or `integration_test.py` files.

This structure is designed to be modular, allowing for the addition of new features by extending existing modules or adding new ones without significantly altering the core application flow in `main.py`.
