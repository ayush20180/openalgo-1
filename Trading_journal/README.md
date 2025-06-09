# Trading Journal Application

A comprehensive trading journal application built with Plotly Dash, designed to help traders track, analyze, and improve their trading performance.

## Features

- **Data Management**
  - CSV import with automatic data preprocessing
  - Manual trade entry and editing
  - Trade deletion and tagging

- **Dashboard & Visualizations**
  - Equity curve and P&L distribution
  - Period-based P&L analysis (daily, weekly, monthly)
  - Algorithm-specific performance metrics

- **Advanced Analytics**
  - Drawdown analysis
  - Risk-adjusted returns (Sharpe, Sortino ratios)
  - Position type and exit signal analysis
  - Win/loss streak tracking
  - P&L volatility monitoring
  - Trade cluster analysis

- **Journal Management**
  - Comprehensive trade entry form
  - Trade editing and deletion
  - Tagging and categorization
  - Notes and parameters tracking

- **Global Filtering**
  - Date range filtering
  - Symbol, exchange, and position type filtering
  - Algorithm and exit signal filtering
  - P&L range filtering

## Installation

1. Clone the repository
2. Create a virtual environment:
   ```
   python -m venv venv
   ```
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`
4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Activate the virtual environment (if not already activated)
2. Run the application:
   ```
   python -m app.main
   ```
3. Open your web browser and navigate to `http://localhost:8050`

## Data Format

The application expects CSV files with the following columns:

- `TradeID`: Unique identifier for each trade
- `OpenTimestamp`: Date and time when the trade was opened
- `CloseTimestamp`: Date and time when the trade was closed
- `Symbol`: Trading symbol (e.g., AAPL, MSFT)
- `Exchange`: Exchange where the trade was executed
- `PositionType`: Long or Short
- `EntryPrice`: Price at which the trade was entered
- `ExitPrice`: Price at which the trade was exited
- `Quantity`: Number of shares/contracts
- `Commission`: Trading commission
- `SwapFees`: Overnight fees (if applicable)
- `GrossP&L`: Gross profit/loss
- `NetP&L`: Net profit/loss after fees
- `AlgorithmID`: Identifier for the trading algorithm
- `Parameters`: JSON string of algorithm parameters
- `SignalName_Exit`: Exit signal type
- `ProductType`: Type of product traded

## Project Structure

```
trading_dashboard/
├── app/
│   ├── __init__.py
│   ├── main.py
│   └── utils/
│       ├── data_loader.py
│       ├── metrics_calculator.py
│       ├── advanced_metrics.py
│       ├── advanced_dashboard.py
│       ├── global_filters.py
│       └── journal_management.py
├── data/
│   └── sample_trades.csv
├── venv/
├── requirements.txt
├── test_application.py
└── integration_test.py
```

## Testing

The application includes comprehensive testing:

1. Run unit and component tests:
   ```
   python test_application.py
   ```

2. Run integration tests:
   ```
   python integration_test.py
   ```

## Performance

The application has been tested with datasets of up to 1000 trades and performs well:
- Loading time: ~0.01 seconds
- Preprocessing time: ~0.005 seconds
- Total processing time: ~0.015 seconds

## License

This project is licensed under the MIT License - see the LICENSE file for details.
