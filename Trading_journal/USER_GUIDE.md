"""
User Guide - Trading Journal Application
--------------------------------------
A comprehensive guide for using the Trading Journal Application.
"""

# Trading Journal Application - User Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Data Import](#data-import)
4. [Dashboard Overview](#dashboard-overview)
5. [Advanced Analytics](#advanced-analytics)
6. [Journal Management](#journal-management)
7. [Global Filtering](#global-filtering)
8. [Interface Customization](#interface-customization)
9. [Tips and Best Practices](#tips-and-best-practices)
10. [Troubleshooting](#troubleshooting)

## Introduction

The Trading Journal Application is a powerful tool designed to help traders track, analyze, and improve their trading performance. This comprehensive platform provides detailed analytics, visualizations, and journal management features to gain insights into your trading patterns and make data-driven decisions.

## Getting Started

### System Requirements
- Python 3.8 or higher
- Modern web browser (Chrome, Firefox, Edge, or Safari)
- 4GB RAM minimum (8GB recommended for large datasets)

### Installation
1. Ensure Python is installed on your system
2. Open a terminal or command prompt
3. Navigate to the application directory
4. Create a virtual environment:
   ```
   python -m venv venv
   ```
5. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`
   - bash: `source venv/Scripts/activate`
   
6. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Launching the Application
1. Activate the virtual environment (if not already activated)
2. Run the application:
   ```
   python -m app.main
   ```
3. Open your web browser and navigate to `http://localhost:8050`

## Data Import

### CSV Import
1. Prepare your trading data in CSV format with the required columns
2. From the application home screen, click "Upload CSV File"
3. Select your CSV file
4. The application will automatically process and validate your data
5. Once imported, you'll be redirected to the dashboard

### Required CSV Format
Your CSV file should include the following columns:
- `TradeID`: Unique identifier for each trade
- `OpenTimestamp`: Date and time when the trade was opened (YYYY-MM-DD HH:MM:SS)
- `CloseTimestamp`: Date and time when the trade was closed (YYYY-MM-DD HH:MM:SS)
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
- `Parameters`: JSON string of algorithm parameters (e.g., `{"ma_period": 20}`)
- `SignalName_Exit`: Exit signal type
- `ProductType`: Type of product traded

## Dashboard Overview

### Overview Tab
The Overview tab provides a high-level summary of your trading performance:

1. **Overall Performance Metrics**
   - Total P&L
   - Win Rate
   - Profit Factor
   - Total Trades

2. **Equity Curve**
   - Visual representation of your account balance over time
   - Helps identify trends and periods of growth or drawdown

3. **P&L Distribution**
   - Histogram showing the distribution of your winning and losing trades
   - Helps understand the consistency and variability of your returns

4. **P&L by Period**
   - Daily, weekly, or monthly breakdown of your trading performance
   - Helps identify your most profitable trading periods

### Algorithm Analysis Tab
This tab allows you to analyze the performance of specific trading algorithms:

1. **Algorithm Selection**
   - Dropdown to select a specific algorithm for detailed analysis

2. **Algorithm Metrics**
   - Performance metrics specific to the selected algorithm
   - Comparison to overall performance

3. **Algorithm Equity Curve**
   - Performance chart for the selected algorithm
   - Helps evaluate algorithm effectiveness over time

## Advanced Analytics

The Advanced Analytics tab provides deeper insights into your trading performance:

1. **Drawdown Analysis**
   - Visual representation of drawdowns over time
   - Maximum drawdown calculation and annotation

2. **Risk-Adjusted Returns**
   - Sharpe Ratio: Measures return relative to risk
   - Sortino Ratio: Focuses on downside risk
   - Maximum Drawdown: Largest peak-to-trough decline

3. **Position Type Analysis**
   - Performance breakdown by position type (Long vs Short)
   - Win rate, profit factor, and total P&L for each position type

4. **Exit Signal Analysis**
   - Performance metrics by exit signal type
   - Helps identify which exit strategies are most effective

5. **Win/Loss Streaks**
   - Analysis of consecutive winning and losing trades
   - Streak distribution visualization

6. **P&L Volatility**
   - 20-day rolling volatility of your trading returns
   - Helps identify periods of increased risk

7. **Trade Clusters**
   - Groups trades by similar characteristics
   - Identifies patterns in your most profitable trade setups

## Journal Management

The Journal Management tab allows you to manually add, edit, and manage your trades:

1. **Add New Trade**
   - Complete form for entering all trade details
   - Automatic P&L calculation based on entry/exit prices and quantity
   - Parameter storage for algorithm settings

2. **Manage Trades**
   - Search and filter existing trades
   - Sort by various criteria (date, P&L, symbol)
   - Edit or delete individual trades
   - Add tags and notes to trades for better organization

3. **Trade Table**
   - Comprehensive view of all trades
   - Color-coded for quick identification of winning/losing trades
   - Pagination for easy navigation through large datasets

## Global Filtering

The Global Filters panel at the top of the dashboard allows you to filter your data across all views:

1. **Date Range Filter**
   - Select specific time periods for analysis

2. **Symbol Filter**
   - Focus on specific trading symbols

3. **Exchange Filter**
   - Filter by trading exchange

4. **Position Type Filter**
   - Filter by Long or Short positions

5. **Product Type Filter**
   - Filter by product type (Stock, ETF, Option, etc.)

6. **Algorithm Filter**
   - Filter by specific trading algorithms

7. **Exit Signal Filter**
   - Filter by exit signal types

8. **P&L Range Filter**
   - Focus on trades within a specific P&L range

## Interface Customization

### Theme Switching (Dark/Light Mode)

The application offers both a light and a dark theme to suit your visual preference.

- **How to Switch Themes:** You can find the theme toggle switch (labeled with a moon icon "🌙 Dark Mode") in the header section of the dashboard, typically to the right of the main title "Algorithmic Trading Dashboard".
- **Functionality:** Click the switch to toggle between the light (default) and dark themes. The change will apply globally across the application.
- **Persistence:** Your theme preference is saved locally in your browser, so the application will remember your choice for future sessions.

## Tips and Best Practices

1. **Regular Data Import**
   - Import your trading data regularly to maintain an up-to-date journal
   - Consider setting up a weekly or monthly review process

2. **Detailed Notes**
   - Use the notes field to record your thoughts, emotions, and market conditions
   - This qualitative data can provide valuable context to your quantitative metrics

3. **Tagging System**
   - Develop a consistent tagging system for your trades
   - Examples: #Breakout, #Reversal, #News, #Earnings

4. **Performance Review**
   - Use the advanced analytics to regularly review your performance
   - Look for patterns in your winning and losing trades

5. **Algorithm Refinement**
   - Use the algorithm analysis to identify which strategies are working
   - Document parameter changes in the Parameters field

## Troubleshooting

1. **CSV Import Issues**
   - Ensure your CSV file has all required columns
   - Check date formats (YYYY-MM-DD HH:MM:SS)
   - Verify that numeric fields contain valid numbers

2. **Performance Issues**
   - For very large datasets (>10,000 trades), consider filtering by date range
   - Close other resource-intensive applications when running the application

3. **Visualization Problems**
   - If charts are not displaying correctly, try refreshing the page
   - Ensure your browser is updated to the latest version

4. **Calculation Discrepancies**
   - If you notice P&L calculation discrepancies, check your commission and swap fee values
   - Verify that your entry/exit prices and quantities are correct

For additional support or to report issues, please contact the development team.
