"""
Global Filtering Module
---------------------
This module implements global filtering functionality for the dashboard.
"""

import dash
from dash import dcc, html, callback, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
from datetime import datetime, timedelta


def create_global_filters_layout():
    """
    Create the layout for global filters.
    
    Returns:
        Dash layout components for global filters
    """
    layout = dbc.Card([
        # dbc.CardHeader("Global Filters"), # Removed as per subtask
        dbc.CardBody([
            # Date Range Filter
            dbc.Row([
                dbc.Col(html.Label("Date Range:", className="col-form-label"), width=12, lg=4),
                dbc.Col(dcc.DatePickerRange(
                    id='date-range-picker',
                    start_date_placeholder_text="Start Date",
                    end_date_placeholder_text="End Date",
                    calendar_orientation='horizontal',
                    clearable=True,
                    reopen_calendar_on_clear=True,
                    with_portal=True,  # Explicitly set, though often default
                    with_full_screen_portal=False, # Key change for better visibility
                    style={'width': '100%'}
                ), width=12, lg=8)
            ], className="mb-3"),

            # Symbol Filter
            dbc.Row([
                dbc.Col(html.Label("Symbol:", className="col-form-label"), width=12, lg=4),
                dbc.Col(dcc.Dropdown(
                    id='symbol-selector',
                    multi=True,
                    placeholder="Select Symbol(s)",
                ), width=12, lg=8)
            ], className="mb-3"),

            # Exchange Filter
            dbc.Row([
                dbc.Col(html.Label("Exchange:", className="col-form-label"), width=12, lg=4),
                dbc.Col(dcc.Dropdown(
                    id='exchange-selector',
                    multi=True,
                    placeholder="Select Exchange(s)",
                ), width=12, lg=8)
            ], className="mb-3"),

            # Position Type Filter
            dbc.Row([
                dbc.Col(html.Label("Position Type:", className="col-form-label"), width=12, lg=4),
                dbc.Col(dcc.Dropdown(
                    id='position-type-selector',
                    options=[
                        {'label': 'Long', 'value': 'Long'},
                        {'label': 'Short', 'value': 'Short'}
                    ],
                    multi=True,
                    placeholder="Select Position Type(s)",
                ), width=12, lg=8)
            ], className="mb-3"),

            # Product Type Filter
            dbc.Row([
                dbc.Col(html.Label("Product Type:", className="col-form-label"), width=12, lg=4),
                dbc.Col(dcc.Dropdown(
                    id='product-type-selector',
                    multi=True,
                    placeholder="Select Product Type(s)",
                ), width=12, lg=8)
            ], className="mb-3"),

            # Algorithm Filter
            dbc.Row([
                dbc.Col(html.Label("Algorithm:", className="col-form-label"), width=12, lg=4),
                dbc.Col(dcc.Dropdown(
                    id='global-algorithm-selector',
                    multi=True,
                    placeholder="Select Algorithm(s)",
                ), width=12, lg=8)
            ], className="mb-3"),

            # Exit Signal Filter
            dbc.Row([
                dbc.Col(html.Label("Exit Signal:", className="col-form-label"), width=12, lg=4),
                dbc.Col(dcc.Dropdown(
                    id='exit-signal-selector',
                    multi=True,
                    placeholder="Select Exit Signal(s)",
                ), width=12, lg=8)
            ], className="mb-3"),
            
            # P&L Range Filter
            dbc.Row([
                dbc.Col(html.Label("P&L Range:", className="col-form-label"), width=12, lg=4), # Keep label similar to other filters
                dbc.Col([
                    dbc.Button("Set P&L Range", id="pnl-range-modal-button", color="light", outline=True, className="me-2 w-100"), # Button to open modal
                    html.Div(id="selected-pnl-range-display", children="All P&L", className="mt-1 form-text text-muted small") # To display selected range
                ], width=12, lg=8)
            ], className="mb-3 align-items-center"), # align-items-center for button and text

            # Day of Week Filter
            dbc.Row([
                dbc.Col(html.Label("Day of Week:", className="col-form-label"), width=12, lg=4),
                dbc.Col(dcc.Dropdown(
                    id='day-of-week-selector',
                    options=[
                        {'label': 'Monday', 'value': 'Monday'},
                        {'label': 'Tuesday', 'value': 'Tuesday'},
                        {'label': 'Wednesday', 'value': 'Wednesday'},
                        {'label': 'Thursday', 'value': 'Thursday'},
                        {'label': 'Friday', 'value': 'Friday'},
                        {'label': 'Saturday', 'value': 'Saturday'},
                        {'label': 'Sunday', 'value': 'Sunday'}
                    ],
                    multi=True,
                    placeholder="Select Day(s) of Week",
                ), width=12, lg=8)
            ], className="mb-3"),

            # Trade Type Filter
            dbc.Row([
                dbc.Col(html.Label("Trade Type:", className="col-form-label"), width=12, lg=4),
                dbc.Col(dcc.Dropdown(
                    id='trade-type-filter', # New ID
                    options=[
                        {'label': 'Manual', 'value': 'manual'},
                        {'label': 'Algo', 'value': 'Algo'}
                    ],
                    multi=True,
                    placeholder="Select Trade Type(s)",
                ), width=12, lg=8)
            ], className="mb-3"),
            
            # Apply/Reset Buttons
            dbc.Row([
                dbc.Col([
                    dbc.Button("Apply Filters", id="apply-filters-button", color="primary", className="me-2"),
                    dbc.Button("Reset Filters", id="reset-filters-button", color="secondary")
                ], width=12, className="d-flex justify-content-end")
            ], className="mt-4 apply-reset-buttons-row") # Added class for potential specific CSS targeting if needed
        ])
    ], className="global-filters-card") # Use a more specific class, remove mb-4 if accordion handles spacing
    
    return layout


def register_filter_callbacks(app):
    """
    Register callbacks for global filters.
    
    Args:
        app: Dash application instance
    """
    
    @app.callback(
        [
            Output('symbol-selector', 'options'),
            Output('exchange-selector', 'options'),
            Output('product-type-selector', 'options'),
            Output('global-algorithm-selector', 'options'),
            Output('exit-signal-selector', 'options'),
            Output('modal-pnl-range-slider', 'min'), 
            Output('modal-pnl-range-slider', 'max'),  
            Output('modal-pnl-range-slider', 'marks'),
            Output('modal-pnl-range-slider', 'value'), 
            Output('stored-pnl-range', 'data', allow_duplicate=True),
            Output('selected-pnl-range-display', 'children', allow_duplicate=True)
        ],
        Input('trade-data-store', 'data'),
        prevent_initial_call=True # Added to prevent issues on initial load before full setup
    )
    def update_filter_options(json_data): 
        if json_data is None:
            empty_options = []
            # Default values for modal-pnl-range-slider and new outputs
            return empty_options, empty_options, empty_options, empty_options, empty_options, -1000, 1000, {-1000: '-$1000', 0: '$0', 1000: '$1000'}, [-1000, 1000], [-1000, 1000], "All P&L"
        
        # Convert JSON to DataFrame
        df = pd.read_json(json_data, orient='split')
        
        # Get unique values for each filter
        symbol_options = [{'label': symbol, 'value': symbol} for symbol in sorted(df['Symbol'].unique())]
        exchange_options = [{'label': exchange, 'value': exchange} for exchange in sorted(df['Exchange'].unique())]
        product_options = [{'label': product, 'value': product} for product in sorted(df['ProductType'].unique())]
        algo_options = [{'label': algo, 'value': algo} for algo in sorted(df['AlgorithmID'].unique())]
        exit_options = [{'label': signal, 'value': signal} for signal in sorted(df['SignalName_Exit'].unique())]
        
        # Calculate min and max P&L for range slider
        min_pnl = min(df['NetP&L'].min(), -100)  # Ensure at least -100
        max_pnl = max(df['NetP&L'].max(), 100)   # Ensure at least 100
        
        # Round to nearest 100
        min_pnl = int(min_pnl / 100) * 100
        max_pnl = int(max_pnl / 100 + 1) * 100
        
        # Create marks for slider
        step = max(int((max_pnl - min_pnl) / 5), 100) # Ensure step is reasonable
        # Ensure marks are generated correctly, especially if min_pnl and max_pnl are close
        # Generate marks from min_pnl to max_pnl with the calculated step
        marks = {i: f'${i}' for i in range(min_pnl, max_pnl + step, step) if i <= max_pnl} # Ensure marks do not exceed max_pnl
        if not marks: # Fallback if marks are empty (e.g. min_pnl == max_pnl)
            marks = {min_pnl: f'${min_pnl}', max_pnl: f'${max_pnl}'}
        
        initial_pnl_range = [min_pnl, max_pnl]
        initial_display_text = f"Range: ${min_pnl} to ${max_pnl}"
        if min_pnl == initial_pnl_range[0] and max_pnl == initial_pnl_range[1] and df['NetP&L'].min() == min_pnl and df['NetP&L'].max() == max_pnl:
             # If the range covers all available data, display "All P&L"
             if min_pnl == df['NetP&L'].min() and max_pnl == df['NetP&L'].max():
                 initial_display_text = "All P&L"


        return symbol_options, exchange_options, product_options, algo_options, exit_options, min_pnl, max_pnl, marks, initial_pnl_range, initial_pnl_range, initial_display_text
    
    @app.callback(
        Output('filtered-data-store', 'data'),
        [
            Input('apply-filters-button', 'n_clicks'),
            Input('reset-filters-button', 'n_clicks')
        ],
        [
            State('trade-data-store', 'data'),
            State('date-range-picker', 'start_date'),
            State('date-range-picker', 'end_date'),
            State('symbol-selector', 'value'),
            State('exchange-selector', 'value'),
            State('position-type-selector', 'value'),
            State('product-type-selector', 'value'),
            State('global-algorithm-selector', 'value'),
            State('exit-signal-selector', 'value'),
            State('stored-pnl-range', 'data'), # Changed to stored P&L range
            State('day-of-week-selector', 'value'),
            State('trade-type-filter', 'value') # Added Trade Type filter
        ]
    )
    def apply_filters(apply_clicks, reset_clicks, json_data, start_date, end_date, 
                     symbols, exchanges, position_types, product_types, algorithms, 
                     exit_signals, pnl_range, day_of_week_selected, trade_type_selected): # Added trade_type_selected
        # Check which button was clicked
        ctx = dash.callback_context
        if not ctx.triggered:
            # No button clicked yet, return None (no filtering)
            return None
        
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        # If reset button clicked or no data, return None (no filtering)
        if button_id == 'reset-filters-button' or json_data is None:
            return None
        
        # Convert JSON to DataFrame
        df = pd.read_json(json_data, orient='split')
        
        # Apply filters
        if start_date and end_date:
            df = df[(df['OpenTimestamp'] >= start_date) & (df['OpenTimestamp'] <= end_date)]
        
        if symbols and len(symbols) > 0:
            df = df[df['Symbol'].isin(symbols)]
        
        if exchanges and len(exchanges) > 0:
            df = df[df['Exchange'].isin(exchanges)]
        
        if position_types and len(position_types) > 0:
            df = df[df['PositionType'].isin(position_types)]
        
        if product_types and len(product_types) > 0:
            df = df[df['ProductType'].isin(product_types)]
        
        if algorithms and len(algorithms) > 0:
            df = df[df['AlgorithmID'].isin(algorithms)]
        
        if exit_signals and len(exit_signals) > 0:
            df = df[df['SignalName_Exit'].isin(exit_signals)]
        
        if pnl_range:
            df = df[(df['NetP&L'] >= pnl_range[0]) & (df['NetP&L'] <= pnl_range[1])]

        if day_of_week_selected and len(day_of_week_selected) > 0:
            # Ensure 'OpenDayOfWeek' column exists from data_loader
            if 'OpenDayOfWeek' in df.columns:
                 df = df[df['OpenDayOfWeek'].isin(day_of_week_selected)]
            else:
                # Optionally print a warning or handle if the column is somehow missing
                print("Warning: 'OpenDayOfWeek' column not found for filtering.")

        if trade_type_selected and len(trade_type_selected) > 0:
            # Ensure 'Trade type' column exists in df to avoid KeyError
            if 'Trade type' in df.columns:
                df = df[df['Trade type'].isin(trade_type_selected)]
            else:
                # Optionally print a warning if the column is missing,
                # though data should consistently have it now.
                print("Warning: 'Trade type' column not found for filtering.")
        
        # Convert filtered DataFrame back to JSON
        filtered_json = df.to_json(date_format='iso', orient='split')
        
        return filtered_json

    @app.callback(
        Output('pnl-range-modal', 'is_open'),
        Output('modal-pnl-range-slider', 'value', allow_duplicate=True), # Explicitly set value on open
        Input('pnl-range-modal-button', 'n_clicks'),
        State('stored-pnl-range', 'data'),
        State('modal-pnl-range-slider', 'min'), # Get current min from modal slider (set by update_filter_options)
        State('modal-pnl-range-slider', 'max'), # Get current max from modal slider (set by update_filter_options)
        prevent_initial_call=True
    )
    def toggle_pnl_modal(n_clicks, stored_range, current_min, current_max):
        if n_clicks:
            if stored_range:
                # Ensure stored range is within current min/max from data
                slider_value = [
                    max(stored_range[0], current_min),
                    min(stored_range[1], current_max)
                ]
                return True, slider_value
            else:
                # Default to full range if nothing stored
                return True, [current_min, current_max] 
        return False, dash.no_update

    @app.callback(
        Output('stored-pnl-range', 'data'),
        Output('selected-pnl-range-display', 'children'),
        Output('pnl-range-modal', 'is_open', allow_duplicate=True), # To close the modal
        Input('apply-pnl-range-button', 'n_clicks'),
        State('modal-pnl-range-slider', 'value'),
        prevent_initial_call=True
    )
    def apply_pnl_range_from_modal(n_clicks, selected_value):
        if n_clicks and selected_value:
            display_text = f"Range: ${selected_value[0]} to ${selected_value[1]}"
            # Check if the selected range is the full range available from the slider
            # This requires knowing the slider's current min/max, which might be complex if they changed
            # For simplicity, we just display the selected range.
            # A more advanced check could compare with values from trade_data_store if needed.
            return selected_value, display_text, False # Store value, update display, close modal
        return dash.no_update, dash.no_update, dash.no_update

    @app.callback(
        Output('pnl-range-modal', 'is_open', allow_duplicate=True), # To close the modal
        Input('cancel-pnl-range-button', 'n_clicks'),
        prevent_initial_call=True
    )
    def cancel_pnl_modal(n_clicks):
        if n_clicks:
            return False # Close modal
        return dash.no_update
