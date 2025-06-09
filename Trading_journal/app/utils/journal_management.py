import io
"""
Journal Entry Management Module
-----------------------------
This module handles manual trade entry, editing, and deletion functionality.
"""

import dash
from dash import dcc, html, callback, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import json
from datetime import datetime
import os # Added for file path operations
import re # Added for regex in TradeID hint
from app.utils.data_loader import load_data_csv, save_data_csv # Updated import
from datetime import datetime # Ensure datetime is imported
import pandas as pd # Ensure pandas is imported


def create_journal_entry_layout():
    """
    Create the layout for journal entry management.
    
    Returns:
        Dash layout components for journal entry management
    """
    layout = html.Div([
        dbc.Row([
            dbc.Col([
                html.H3("Trade Journal Management", className="mt-4"),
                html.P("Add, edit, and manage your trading records", className="text-muted")
            ])
        ]),
        
        # --- Notification Area ---
        dbc.Row([
            dbc.Col([
                html.Div(id='trade-action-notification-div', children=[]) # Empty initially, will be populated by callbacks
            ], className="mb-3") # Added some bottom margin for spacing
        ]),
        # --- End Notification Area ---
        
        # Trade Entry Form
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Add New Trade"),
                    dbc.CardBody([
                        dbc.Form([
                            # Row 1: Basic Trade Info
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Trade ID:"),
                                    dbc.InputGroup([
                                        dbc.Input(id="trade-id-input", type="text", placeholder="Enter unique ID"),
                                        dbc.Button("Generate ID", id="generate-trade-id-button", outline=True, color="primary", size="sm", n_clicks=0)
                                    ]),
                                    dbc.FormText(html.Span(id='max-trade-id-display', children=["Hint: Enter a unique TradeID."]), color="muted")
                                ], width=4),
                                dbc.Col([
                                    html.Label("Symbol:"),
                                    dbc.Input(id="symbol-input", type="text", placeholder="e.g., AAPL")
                                ], width=4),
                                dbc.Col([
                                    html.Label("Exchange:"),
                                    dbc.Input(id="exchange-input", type="text", placeholder="e.g., NASDAQ")
                                ], width=4)
                            ], className="mb-3"),
                            
                            # Row 2: Timestamps
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Open Date/Time:"),
                                    dcc.DatePickerSingle(
                                        id="open-date-input",
                                        placeholder="Select date",
                                        display_format="YYYY-MM-DD"
                                    ),
                                    dbc.Input(id="open-time-input", type="text", placeholder="HH:MM:SS", className="mt-2")
                                ], width=6),
                                dbc.Col([
                                    html.Label("Close Date/Time:"),
                                    dcc.DatePickerSingle(
                                        id="close-date-input",
                                        placeholder="Select date",
                                        display_format="YYYY-MM-DD"
                                    ),
                                    dbc.Input(id="close-time-input", type="text", placeholder="HH:MM:SS", className="mt-2")
                                ], width=6)
                            ], className="mb-3"),
                            
                            # Row 3: Position Details
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Position Type:"),
                                    dbc.Select(
                                        id="position-type-input",
                                        options=[
                                            {"label": "Long", "value": "Long"},
                                            {"label": "Short", "value": "Short"}
                                        ]
                                    )
                                ], width=4),
                                dbc.Col([
                                    html.Label("Product Type:"),
                                    dbc.Input(id="product-type-input", type="text", placeholder="e.g., Stock")
                                ], width=4),
                                dbc.Col([
                                    html.Label("Quantity:"),
                                    dbc.Input(id="quantity-input", type="number", placeholder="Enter quantity")
                                ], width=4)
                            ], className="mb-3"),
                            
                            # Row 4: Price Information
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Entry Price:"),
                                    dbc.InputGroup([
                                        dbc.InputGroupText("$"),
                                        dbc.Input(id="entry-price-input", type="number", placeholder="0.00")
                                    ])
                                ], width=4),
                                dbc.Col([
                                    html.Label("Exit Price:"),
                                    dbc.InputGroup([
                                        dbc.InputGroupText("$"),
                                        dbc.Input(id="exit-price-input", type="number", placeholder="0.00")
                                    ])
                                ], width=4),
                                dbc.Col([
                                    html.Label("Commission:"),
                                    dbc.InputGroup([
                                        dbc.InputGroupText("$"),
                                        dbc.Input(id="commission-input", type="number", placeholder="0.00")
                                    ])
                                ], width=4)
                            ], className="mb-3"),
                            
                            # Row 5: P&L and Fees
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Swap Fees:"),
                                    dbc.InputGroup([
                                        dbc.InputGroupText("$"),
                                        dbc.Input(id="swap-fees-input", type="number", placeholder="0.00")
                                    ])
                                ], width=4),
                                dbc.Col([
                                    html.Label("Gross P&L:"),
                                    dbc.InputGroup([
                                        dbc.InputGroupText("$"),
                                        dbc.Input(id="gross-pnl-input", type="number", placeholder="0.00")
                                    ])
                                ], width=4),
                                dbc.Col([
                                    html.Label("Net P&L:"),
                                    dbc.InputGroup([
                                        dbc.InputGroupText("$"),
                                        dbc.Input(id="net-pnl-input", type="number", placeholder="0.00")
                                    ])
                                ], width=4)
                            ], className="mb-3"),
                            
                            # Row 6: Algorithm and Exit Signal
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Algorithm ID:"),
                                    dbc.Input(id="algorithm-id-input", type="text", placeholder="e.g., ALGO001")
                                ], width=4),
                                dbc.Col([
                                    html.Label("Exit Signal:"),
                                    dbc.Select(
                                        id="exit-signal-input",
                                        options=[
                                            {"label": "TAKE_PROFIT", "value": "TAKE_PROFIT"},
                                            {"label": "STOP_LOSS", "value": "STOP_LOSS"},
                                            {"label": "MANUAL", "value": "MANUAL"},
                                            {"label": "SIGNAL", "value": "SIGNAL"}
                                        ]
                                    )
                                ], width=4),
                                dbc.Col([
                                    html.Label("Parameters:"),
                                    dbc.Input(id="parameters-input", type="text", placeholder='e.g., {"ma_period":20}')
                                ], width=4)
                            ], className="mb-3"),
                            
                            # Row 7: Tags
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Tags:"),
                                    dbc.Input(id="tags-input", type="text", placeholder="Comma-separated tags")
                                ], width=12)
                            ], className="mb-3"),
                            
                            # Row 8: Notes
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Notes:"),
                                    dbc.Textarea(id="notes-input", placeholder="Enter trade notes", style={"height": "100px"})
                                ], width=12)
                            ], className="mb-3"),
                            
                            # Row 9: Buttons
                            dbc.Row([
                                dbc.Col([
                                    dbc.Button("Add Trade", id="add-trade-button", color="success", className="me-2"),
                                    dbc.Button("Calculate P&L", id="calculate-pnl-button", color="primary", className="me-2"),
                                    dbc.Button("Clear Form", id="clear-form-button", color="secondary")
                                ], width=12, className="d-flex justify-content-end")
                            ])
                        ])
                    ])
                ], className="mb-4")
            ])
        ]),
        
        # Trade Management
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Manage Trades"),
                    dbc.CardBody([
                        # Search and Filter
                        dbc.Row([
                            dbc.Col([
                                html.Label("Search Trades:"),
                                dbc.Input(id="trade-search-input", type="text", placeholder="Search by ID, Symbol, etc.")
                            ], width=8),
                            dbc.Col([
                                html.Label("Sort By:"),
                                dbc.Select(
                                    id="trade-sort-select",
                                    options=[
                                        {"label": "Date (Newest First)", "value": "date_desc"},
                                        {"label": "Date (Oldest First)", "value": "date_asc"},
                                        {"label": "P&L (Highest First)", "value": "pnl_desc"},
                                        {"label": "P&L (Lowest First)", "value": "pnl_asc"},
                                        {"label": "Symbol (A-Z)", "value": "symbol_asc"},
                                        {"label": "Symbol (Z-A)", "value": "symbol_desc"}
                                    ],
                                    value="date_desc"
                                )
                            ], width=4)
                        ], className="mb-3"),
                        
                        # Trade Table
                        dash_table.DataTable(
                            id='editable-trade-table',
                            columns=[
                                {'name': 'TradeID', 'id': 'TradeID', 'editable': False},
                                {'name': 'Date', 'id': 'OpenTimestamp', 'editable': False},
                                {'name': 'Symbol', 'id': 'Symbol', 'editable': True},
                                {'name': 'Position', 'id': 'PositionType', 'editable': True},
                                {'name': 'Entry', 'id': 'EntryPrice', 'editable': True, 'type': 'numeric', 'format': {'specifier': '$.2f'}},
                                {'name': 'Exit', 'id': 'ExitPrice', 'editable': True, 'type': 'numeric', 'format': {'specifier': '$.2f'}},
                                {'name': 'Quantity', 'id': 'Quantity', 'editable': True, 'type': 'numeric'},
                                {'name': 'Net P&L', 'id': 'NetP&L', 'editable': True, 'type': 'numeric', 'format': {'specifier': '$.2f'}},
                                {'name': 'Algorithm', 'id': 'AlgorithmID', 'editable': True},
                                {'name': 'Exit Signal', 'id': 'SignalName_Exit', 'editable': True}
                            ],
                            data=[],
                            editable=True,
                            row_selectable='multi', # CHANGED HERE
                            page_size=10,
                            style_table={'overflowX': 'auto'},
                            style_cell={
                                'height': 'auto',
                                'minWidth': '80px', 'width': '100px', 'maxWidth': '180px',
                                'whiteSpace': 'normal'
                            },
                            style_data_conditional=[
                                {
                                    'if': {'filter_query': '{NetP&L} > 0'},
                                    'backgroundColor': 'rgba(40, 167, 69, 0.3)', 
                                    'color': '#f0f0f0'
                                },
                                {
                                    'if': {'filter_query': '{NetP&L} < 0'},
                                    'backgroundColor': 'rgba(220, 53, 69, 0.3)', 
                                    'color': '#f0f0f0'
                                },
                                {
                                    'if': {'filter_query': '{NetP&L} = 0'}, 
                                    'backgroundColor': '#3a3a40', 
                                    'color': '#f0f0f0'
                                }
                            ]
                        ),
                        
                        # Action Buttons
                        dbc.Row([
                            dbc.Col([
                                dbc.Button("Select All Page", id="select-all-button", color="info", className="me-1"),
                                dbc.Button("Deselect All", id="deselect-all-button", color="warning", className="me-2"),
                                dbc.Button("Edit Selected", id="edit-trade-button", color="primary", className="me-2"),
                                dbc.Button("Delete Selected", id="delete-trade-button", color="danger", className="me-2"),
                                dbc.Button("Save Changes", id="save-changes-button", color="success")
                            ], width=12, className="d-flex justify-content-end mt-3")
                        ])
                    ])
                ])
            ])
        ])
    ])
    
    return layout


def register_journal_callbacks(app):
    """
    Register callbacks for journal entry management.
    
    Args:
        app: Dash application instance
    """
    
    @app.callback(
        [
            Output('editable-trade-table', 'data'),
            Output('editable-trade-table', 'selected_rows')
        ],
        [
            Input('filtered-data-store', 'data'), # Primary
            State('trade-data-store', 'data'),    # Fallback
            Input('trade-search-input', 'value'),
            Input('trade-sort-select', 'value')
        ]
    )
    def update_editable_table(filtered_json_data, raw_json_data, search_term, sort_by):
        json_data_to_use = filtered_json_data
        if json_data_to_use is None:
            json_data_to_use = raw_json_data

        if json_data_to_use is None:
            return [], []
        
        # Convert JSON to DataFrame
        # Use StringIO to handle json string
        df = pd.read_json(io.StringIO(json_data_to_use), orient='split')

        # Ensure timestamp columns are datetime objects, coercing errors to NaT
        if 'OpenTimestamp' in df.columns:
            df['OpenTimestamp'] = pd.to_datetime(df['OpenTimestamp'], errors='coerce')
        if 'CloseTimestamp' in df.columns: # Proactively convert CloseTimestamp as well
            df['CloseTimestamp'] = pd.to_datetime(df['CloseTimestamp'], errors='coerce')
        
        # Apply search filter if provided
        if search_term and len(search_term) > 0:
            search_term = search_term.lower()
            # Search across multiple columns
            mask = (
                df['TradeID'].astype(str).str.lower().str.contains(search_term) |
                df['Symbol'].astype(str).str.lower().str.contains(search_term) |
                df['AlgorithmID'].astype(str).str.lower().str.contains(search_term) |
                df['SignalName_Exit'].astype(str).str.lower().str.contains(search_term)
            )
            df = df[mask]
        
        # Apply sorting
        if sort_by == 'date_desc':
            df = df.sort_values('OpenTimestamp', ascending=False)
        elif sort_by == 'date_asc':
            df = df.sort_values('OpenTimestamp', ascending=True)
        elif sort_by == 'pnl_desc':
            df = df.sort_values('NetP&L', ascending=False)
        elif sort_by == 'pnl_asc':
            df = df.sort_values('NetP&L', ascending=True)
        elif sort_by == 'symbol_asc':
            df = df.sort_values('Symbol', ascending=True)
        elif sort_by == 'symbol_desc':
            df = df.sort_values('Symbol', ascending=False)
        
        # Helper function for robust datetime formatting
        def format_datetime_for_table(dt_val):
            if pd.isna(dt_val):
                return ""  # Use empty string for NaT in table display
            if isinstance(dt_val, str): # If it's already a string, return as is
                return dt_val
            try:
                return dt_val.strftime('%Y-%m-%d %H:%M')
            except AttributeError: # Fallback if not a datetime object
                return str(dt_val)

        # Format datetime columns for display
        if 'OpenTimestamp' in df.columns:
            df['OpenTimestamp'] = df['OpenTimestamp'].apply(format_datetime_for_table)
        
        # Convert to table data
        table_data = df.to_dict('records')
        
        return table_data, []
    
    @app.callback(
        [
            Output('trade-id-input', 'value', allow_duplicate=True),
            Output('symbol-input', 'value'),
            Output('exchange-input', 'value'),
            Output('open-date-input', 'date'),
            Output('open-time-input', 'value'),
            Output('close-date-input', 'date'),
            Output('close-time-input', 'value'),
            Output('position-type-input', 'value'),
            Output('product-type-input', 'value'),
            Output('quantity-input', 'value'),
            Output('entry-price-input', 'value'),
            Output('exit-price-input', 'value'),
            Output('commission-input', 'value'),
            Output('swap-fees-input', 'value'),
            Output('gross-pnl-input', 'value'),
            Output('net-pnl-input', 'value'),
            Output('algorithm-id-input', 'value'),
            Output('exit-signal-input', 'value'),
            Output('parameters-input', 'value')
        ],
        [
            Input('edit-trade-button', 'n_clicks'),
            Input('clear-form-button', 'n_clicks')
        ],
        [
            State('editable-trade-table', 'selected_rows'),
            State('editable-trade-table', 'data')
        ],
        prevent_initial_call=True
    )
    def handle_trade_form_actions(edit_clicks, clear_clicks, selected_rows, table_data):
        # Check which button was clicked
        ctx = dash.callback_context
        if not ctx.triggered:
            # No button clicked yet, return empty form
            return ["", "", "", None, "", None, "", None, "", None, None, None, None, None, None, None, "", None, ""]
        
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        # Clear form
        if button_id == 'clear-form-button':
            return ["", "", "", None, "", None, "", None, "", None, None, None, None, None, None, None, "", None, ""]
        
        # Edit selected trade
        if button_id == 'edit-trade-button' and selected_rows and len(selected_rows) > 0 and table_data:
            selected_trade = table_data[selected_rows[0]]
            
            # Parse timestamps
            open_date = None
            open_time = ""
            close_date = None
            close_time = ""
            
            if 'OpenTimestamp' in selected_trade:
                try:
                    dt = datetime.strptime(selected_trade['OpenTimestamp'], '%Y-%m-%d %H:%M')
                    open_date = dt.strftime('%Y-%m-%d')
                    open_time = dt.strftime('%H:%M:%S')
                except:
                    pass
            
            if 'CloseTimestamp' in selected_trade:
                try:
                    dt = datetime.strptime(selected_trade['CloseTimestamp'], '%Y-%m-%d %H:%M')
                    close_date = dt.strftime('%Y-%m-%d')
                    close_time = dt.strftime('%H:%M:%S')
                except:
                    pass
            
            # Parse parameters
            parameters = ""
            if 'Parameters' in selected_trade and selected_trade['Parameters']:
                try:
                    if isinstance(selected_trade['Parameters'], str):
                        parameters = selected_trade['Parameters']
                    else:
                        parameters = json.dumps(selected_trade['Parameters'])
                except:
                    pass
            
            return [
                selected_trade.get('TradeID', ""),
                selected_trade.get('Symbol', ""),
                selected_trade.get('Exchange', ""),
                open_date,
                open_time,
                close_date,
                close_time,
                selected_trade.get('PositionType', None),
                selected_trade.get('ProductType', ""),
                selected_trade.get('Quantity', None),
                selected_trade.get('EntryPrice', None),
                selected_trade.get('ExitPrice', None),
                selected_trade.get('Commission', None),
                selected_trade.get('SwapFees', None),
                selected_trade.get('GrossP&L', None),
                selected_trade.get('NetP&L', None),
                selected_trade.get('AlgorithmID', ""),
                selected_trade.get('SignalName_Exit', None),
                parameters
            ]
        
        # Default - empty form
        return ["", "", "", None, "", None, "", None, "", None, None, None, None, None, None, None, "", None, ""]
    
    @app.callback(
        [
            Output('gross-pnl-input', 'value', allow_duplicate=True),
            Output('net-pnl-input', 'value', allow_duplicate=True)
        ],
        Input('calculate-pnl-button', 'n_clicks'),
        [
            State('position-type-input', 'value'),
            State('entry-price-input', 'value'),
            State('exit-price-input', 'value'),
            State('quantity-input', 'value'),
            State('commission-input', 'value'),
            State('swap-fees-input', 'value')
        ],
        prevent_initial_call=True
    )
    def calculate_pnl(n_clicks, position_type, entry_price, exit_price, quantity, commission, swap_fees):
        if None in [position_type, entry_price, exit_price, quantity]:
            return None, None
        
        # Convert to float
        entry_price = float(entry_price)
        exit_price = float(exit_price)
        quantity = float(quantity)
        commission = float(commission) if commission is not None else 0
        swap_fees = float(swap_fees) if swap_fees is not None else 0
        
        # Calculate P&L based on position type
        if position_type == 'Long':
            gross_pnl = (exit_price - entry_price) * quantity
        else:  # Short
            gross_pnl = (entry_price - exit_price) * quantity
        
        # Calculate net P&L
        net_pnl = gross_pnl - commission - swap_fees
        
        return gross_pnl, net_pnl
    
    @app.callback(
        [Output('trade-data-store', 'data', allow_duplicate=True),
         Output('trade-action-notification-div', 'children', allow_duplicate=True)],
        Input('add-trade-button', 'n_clicks'),
        [
            State('trade-data-store', 'data'),
            State('trade-id-input', 'value'),
            State('symbol-input', 'value'),
            State('exchange-input', 'value'),
            State('open-date-input', 'date'),
            State('open-time-input', 'value'),
            State('close-date-input', 'date'),
            State('close-time-input', 'value'),
            State('position-type-input', 'value'),
            State('product-type-input', 'value'),
            State('quantity-input', 'value'),
            State('entry-price-input', 'value'),
            State('exit-price-input', 'value'),
            State('commission-input', 'value'),
            State('swap-fees-input', 'value'),
            State('gross-pnl-input', 'value'),
            State('net-pnl-input', 'value'),
            State('algorithm-id-input', 'value'),
            State('exit-signal-input', 'value'),
            State('parameters-input', 'value')
        ],
        prevent_initial_call=True
    )
    def add_trade(n_clicks, json_data, trade_id, symbol, exchange, open_date, open_time, 
                 close_date, close_time, position_type, product_type, quantity, entry_price, 
                 exit_price, commission, swap_fees, gross_pnl, net_pnl, algorithm_id, 
                 exit_signal, parameters):
        if n_clicks is None or n_clicks == 0:
            return json_data, dash.no_update

        # manual_trades_file_path = os.path.join("data", "manual_trades.csv") # Old logic

        # 1. Validation for required fields
        required_field_names = {
            'Trade ID': trade_id, 'Symbol': symbol, 'Exchange': exchange, 
            'Open Date': open_date, 'Position Type': position_type, 
            'Quantity': quantity, 'Entry Price': entry_price, 'Exit Price': exit_price
        }
        missing_details = [name for name, val in required_field_names.items() if val is None or str(val).strip() == ""]
        
        if missing_details:
            message = f"Failed to add/update trade. Required fields missing: {', '.join(missing_details)}."
            return json_data, message

        try:
            # 2. Timestamp string creation and conversion
            open_timestamp_str = f"{open_date} {open_time if open_time and str(open_time).strip() else '00:00:00'}"
            try:
                open_dt = pd.to_datetime(open_timestamp_str)
            except ValueError:
                return json_data, f"Failed to add/update trade: Invalid format for Open Date/Time ('{open_timestamp_str}'). Expected YYYY-MM-DD HH:MM:SS."

            close_dt = open_dt # Default if close_date is not provided
            if close_date and str(close_date).strip():
                close_timestamp_str = f"{close_date} {close_time if close_time and str(close_time).strip() else '00:00:00'}"
                try:
                    close_dt = pd.to_datetime(close_timestamp_str)
                except ValueError:
                    return json_data, f"Failed to add/update trade: Invalid format for Close Date/Time ('{close_timestamp_str}'). Expected YYYY-MM-DD HH:MM:SS."
            
            # 3. Numeric conversions
            try:
                entry_price_val = float(entry_price)
            except ValueError:
                return json_data, f"Failed to add/update trade: Entry Price ('{entry_price}') must be a number."
            try:
                exit_price_val = float(exit_price)
            except ValueError:
                return json_data, f"Failed to add/update trade: Exit Price ('{exit_price}') must be a number."
            try:
                quantity_val = float(quantity)
            except ValueError:
                return json_data, f"Failed to add/update trade: Quantity ('{quantity}') must be a number."

            def safe_float_conversion(value, field_name, default=0.0):
                if value is None or str(value).strip() == "":
                    return default
                try:
                    return float(value)
                except ValueError:
                    raise ValueError(f"Invalid format for {field_name} ('{value}'). Value must be a number.")

            try:
                commission_val = safe_float_conversion(commission, "Commission")
                swap_fees_val = safe_float_conversion(swap_fees, "Swap Fees")
                gross_pnl_val = safe_float_conversion(gross_pnl, "Gross P&L")
                net_pnl_val = safe_float_conversion(net_pnl, "Net P&L")
            except ValueError as ve_numeric:
                 return json_data, f"Failed to add/update trade: {str(ve_numeric)}"

            new_trade = {
                'TradeID': trade_id, 'OpenTimestamp': open_dt, 'CloseTimestamp': close_dt,
                'Symbol': symbol, 'Exchange': exchange, 'PositionType': position_type,
                'EntryPrice': entry_price_val, 'ExitPrice': exit_price_val, 'Quantity': quantity_val,
                'Commission': commission_val, 'SwapFees': swap_fees_val,
                'GrossP&L': gross_pnl_val, 'NetP&L': net_pnl_val,
                'AlgorithmID': algorithm_id if algorithm_id and str(algorithm_id).strip() else "",
                'Parameters': parameters if parameters and str(parameters).strip() else "{}",
                'SignalName_Exit': exit_signal if exit_signal and str(exit_signal).strip() else "",
                'ProductType': product_type if product_type and str(product_type).strip() else "Stock",
                'Trade type': "manual", # Added Trade type
                'AddedTimestamp': pd.to_datetime(datetime.now()) # Added Timestamp
            }
            
            existing_df = load_data_csv()
            new_trade_df = pd.DataFrame([new_trade])

            combined_df = pd.concat([existing_df, new_trade_df], ignore_index=True)

            # Ensure correct dtypes, especially for AddedTimestamp before sorting and deduplication
            combined_df['AddedTimestamp'] = pd.to_datetime(combined_df['AddedTimestamp'], errors='coerce')
            # Ensure other key timestamp columns are also datetime
            if 'OpenTimestamp' in combined_df.columns:
                 combined_df['OpenTimestamp'] = pd.to_datetime(combined_df['OpenTimestamp'], errors='coerce')
            if 'CloseTimestamp' in combined_df.columns:
                 combined_df['CloseTimestamp'] = pd.to_datetime(combined_df['CloseTimestamp'], errors='coerce')


            combined_df = combined_df.sort_values(by='AddedTimestamp', ascending=False, na_position='last')
            final_df = combined_df.drop_duplicates(subset=['TradeID'], keep='first')

            save_data_csv(final_df)
            
            store_data_for_return = final_df.to_json(date_format='iso', orient='split')
            action_message = "added/updated" if trade_id in existing_df['TradeID'].values else "added"
            success_message = f"Trade {trade_id} {action_message} successfully. All data refreshed."
            return store_data_for_return, success_message

        except Exception as e:
            # Fallback: try to load existing data to prevent data loss in store on error
            current_data_df = load_data_csv()
            current_store_data = current_data_df.to_json(date_format='iso', orient='split') if not current_data_df.empty else None
            error_message = f"An unexpected error occurred: {str(e)}"
            return current_store_data, error_message
    
    @app.callback(
        [Output('trade-data-store', 'data', allow_duplicate=True),
         Output('trade-action-notification-div', 'children', allow_duplicate=True)],
        Input('delete-trade-button', 'n_clicks'),
        [
            State('trade-data-store', 'data'),
            State('editable-trade-table', 'selected_rows'),
            State('editable-trade-table', 'data')
        ],
        prevent_initial_call=True
    )
    def delete_trade(n_clicks, json_data, selected_rows_indices, table_data_view):
        if n_clicks is None or n_clicks == 0:
            return json_data, dash.no_update

        # manual_trades_file_path = os.path.join("data", "manual_trades.csv") # Old logic

        df = load_data_csv()

        if df.empty:
            return None, "Cannot delete trades: No data exists."

        if not selected_rows_indices or len(selected_rows_indices) == 0:
            return json_data, "No trades selected for deletion."

        trade_ids_to_delete = []
        if table_data_view and selected_rows_indices:
            for row_index in selected_rows_indices:
                if row_index < len(table_data_view):
                    trade_to_delete = table_data_view[row_index]
                    if trade_to_delete and 'TradeID' in trade_to_delete:
                        trade_ids_to_delete.append(str(trade_to_delete['TradeID']))
        
        if not trade_ids_to_delete:
            return json_data, "No valid TradeIDs found in the current selection to delete."
            
        try:
            # df = pd.read_csv(manual_trades_file_path) # Old logic
            # Timestamp columns should already be datetime from load_data_csv

            if 'TradeID' not in df.columns:
                 return json_data, "TradeID column not found in the data."
            df['TradeID'] = df['TradeID'].astype(str) # Ensure consistent type for comparison

            initial_count = len(df)
            df = df[~df['TradeID'].isin(trade_ids_to_delete)]
            deleted_count = initial_count - len(df)
            
            if deleted_count > 0:
                save_data_csv(df)
                message = f"{deleted_count} trade(s) deleted successfully. All data refreshed."
            else:
                message = "No trades were deleted. Selected trades might have already been removed or did not match existing trade IDs."
            
            store_data_for_return = df.to_json(date_format='iso', orient='split') if not df.empty else None
            return store_data_for_return, message
        except Exception as e:
            current_data_df = load_data_csv()
            current_store_data = current_data_df.to_json(date_format='iso', orient='split') if not current_data_df.empty else None
            return current_store_data, f"An error occurred during deletion: {str(e)}"
    
    @app.callback(
        [Output('trade-data-store', 'data', allow_duplicate=True),
         Output('trade-action-notification-div', 'children', allow_duplicate=True)],
        Input('save-changes-button', 'n_clicks'),
        [
            State('trade-data-store', 'data'),
            State('editable-trade-table', 'data')
        ],
        prevent_initial_call=True
    )
    def save_table_changes(n_clicks, json_data, table_data):
        if n_clicks is None or n_clicks == 0:
            return json_data, dash.no_update

        if not table_data: # This table_data is from the editable UI component
            return json_data, "No changes to save from the table (table view is empty)."

        try:
            original_full_df = load_data_csv() # Load the single source of truth
            edited_table_df = pd.DataFrame(table_data)

            if 'TradeID' not in edited_table_df.columns:
                 return json_data, "Error: Edited data must contain 'TradeID' column."
            edited_table_df['TradeID'] = edited_table_df['TradeID'].astype(str)

            # Prepare edited_table_df: ensure types and add timestamps for new rows
            processed_rows = []
            existing_trade_ids = original_full_df['TradeID'].astype(str).tolist() if 'TradeID' in original_full_df.columns else []

            for _, row in edited_table_df.iterrows():
                trade_id = str(row['TradeID'])
                if trade_id not in existing_trade_ids:
                    row['Trade type'] = "manual"
                    row['AddedTimestamp'] = pd.to_datetime(datetime.now())
                else:
                    # Preserve original AddedTimestamp and Trade type if trade already exists
                    original_row = original_full_df[original_full_df['TradeID'] == trade_id]
                    if not original_row.empty:
                        if 'AddedTimestamp' in original_row.columns:
                             row['AddedTimestamp'] = original_row['AddedTimestamp'].iloc[0]
                        if 'Trade type' in original_row.columns:
                             row['Trade type'] = original_row['Trade type'].iloc[0]

                # Ensure essential timestamp columns are datetime
                if 'OpenTimestamp' in row: row['OpenTimestamp'] = pd.to_datetime(row['OpenTimestamp'], errors='coerce')
                if 'CloseTimestamp' in row: row['CloseTimestamp'] = pd.to_datetime(row['CloseTimestamp'], errors='coerce')
                if 'AddedTimestamp' in row: row['AddedTimestamp'] = pd.to_datetime(row['AddedTimestamp'], errors='coerce')

                processed_rows.append(row)

            if not processed_rows: # Should not happen if table_data was not empty
                 return json_data, "No data processed from table."

            edited_final_df = pd.DataFrame(processed_rows)

            # Merge strategy: Update existing, append new
            # Set index for efficient update/lookup
            if not original_full_df.empty and 'TradeID' in original_full_df.columns:
                original_full_df['TradeID'] = original_full_df['TradeID'].astype(str)
                original_full_df_indexed = original_full_df.set_index('TradeID')
            else: # original_full_df is empty or has no TradeID
                original_full_df_indexed = pd.DataFrame(columns=edited_final_df.columns).set_index('TradeID')


            if not edited_final_df.empty and 'TradeID' in edited_final_df.columns:
                 edited_final_df['TradeID'] = edited_final_df['TradeID'].astype(str)
                 edited_final_df_indexed = edited_final_df.set_index('TradeID')
            else: # This case implies edited_final_df is empty or lacks TradeID; should be caught earlier
                 return json_data, "Critical error: Processed edited data is invalid."


            # Update existing rows in original_full_df_indexed
            # The .update() method modifies in place and aligns on index (TradeID)
            original_full_df_indexed.update(edited_final_df_indexed)

            # Identify new trades from edited_final_df_indexed and add them
            new_trades_mask = edited_final_df_indexed.index.difference(original_full_df_indexed.index)
            if not new_trades_mask.empty:
                new_trades_to_add = edited_final_df_indexed.loc[new_trades_mask]
                # Need to reset index to concat if original_full_df_indexed was initially empty and gained columns
                # or if new_trades_to_add has columns not in original_full_df_indexed
                temp_concat_df = pd.concat([original_full_df_indexed.reset_index(), new_trades_to_add.reset_index()], ignore_index=True)
            else:
                temp_concat_df = original_full_df_indexed.reset_index()

            # Final deduplication and sort as a safeguard
            if 'AddedTimestamp' in temp_concat_df.columns:
                 temp_concat_df['AddedTimestamp'] = pd.to_datetime(temp_concat_df['AddedTimestamp'], errors='coerce')
                 temp_concat_df = temp_concat_df.sort_values(by='AddedTimestamp', ascending=False, na_position='last')

            if 'TradeID' in temp_concat_df.columns:
                final_df_to_save = temp_concat_df.drop_duplicates(subset=['TradeID'], keep='first')
            else: # Should not happen as TradeID is critical
                final_df_to_save = temp_concat_df

            save_data_csv(final_df_to_save)

            store_data_for_return = final_df_to_save.to_json(date_format='iso', orient='split') if not final_df_to_save.empty else None
            return store_data_for_return, "Changes saved successfully. All data refreshed."

        except Exception as e:
            current_data_df = load_data_csv()
            current_store_data = current_data_df.to_json(date_format='iso', orient='split') if not current_data_df.empty else None
            return current_store_data, f"An error occurred while saving changes: {str(e)}"

    @app.callback(
        Output('editable-trade-table', 'selected_rows', allow_duplicate=True),
        [
            Input('select-all-button', 'n_clicks'),
            Input('deselect-all-button', 'n_clicks')
        ],
        [State('editable-trade-table', 'data')],
        prevent_initial_call=True
    )
    def handle_select_deselect_all(select_all_clicks, deselect_all_clicks, table_data):
        ctx = dash.callback_context
        if not ctx.triggered:
            return dash.no_update
        
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if button_id == 'select-all-button':
            if table_data:
                return list(range(len(table_data)))
            return []
        elif button_id == 'deselect-all-button':
            return []
        
        return dash.no_update

    @app.callback(
        Output('max-trade-id-display', 'children'),
        Input('trade-data-store', 'data')
    )
    def update_max_trade_id_hint(json_data):
        if json_data is None:
            return "Hint: No trades yet. Start with e.g., T00001."

        try:
            df = pd.read_json(io.StringIO(json_data), orient='split')
            if df.empty or 'TradeID' not in df.columns:
                return "Hint: Ensure TradeIDs are unique. Example: T00001"

            trade_ids = df['TradeID'].astype(str)
            
            numeric_ids = []
            # Regex to find IDs starting with "T" followed by numbers
            pattern = re.compile(r"^T(\d+)$") 
            
            for id_str in trade_ids:
                match = pattern.match(id_str)
                if match:
                    numeric_ids.append(int(match.group(1)))
            
            if not numeric_ids:
                # Fallback if no "T<number>" pattern IDs are found
                # Attempt to find any numeric IDs if no "T" prefix
                plain_numeric_ids = [int(s) for s in trade_ids if s.isdigit()]
                if plain_numeric_ids:
                    max_plain_num = max(plain_numeric_ids)
                    return f"Hint: Max numeric ID found is {max_plain_num}. Ensure new ID is unique."
                
                # If still no numeric IDs, show the overall max string ID
                highest_id_str = trade_ids.max() 
                if pd.isna(highest_id_str) or not highest_id_str: # Check if max is NaN or empty
                     return "Hint: Ensure TradeIDs are unique. Example: T00001"
                return f"Hint: Max existing TradeID is '{highest_id_str}'. Ensure new ID is unique."

            max_num = max(numeric_ids)
            # Suggest next ID, assuming 5-digit padding for the number part
            suggestion = f"T{max_num + 1:05d}"
            return f"Hint: Last 'T' prefixed ID was T{max_num:05d}. Use 'Generate ID' for the next ID ({suggestion}), or ensure manual entries are unique."

        except Exception:
            # Catch errors from read_json or other processing
            return "Hint: Could not determine next TradeID. Please ensure uniqueness manually."

    @app.callback(
        Output('trade-id-input', 'value', allow_duplicate=True),
        Input('generate-trade-id-button', 'n_clicks'),
        State('trade-data-store', 'data'),
        prevent_initial_call=True
    )
    def generate_trade_id_callback(n_clicks, json_data):
        if not n_clicks: # Handles None or 0 clicks
            return dash.no_update

        default_start_id = "T00001"
        prefix = "T"
        num_digits = 5 # For padding like T00001

        if json_data is None:
            return default_start_id

        try:
            # Use io.StringIO if json_data is a string, otherwise it might be None
            df = pd.read_json(io.StringIO(json_data), orient='split')
            if df.empty or 'TradeID' not in df.columns:
                return default_start_id
        except Exception: # Broad exception for parsing issues
            return default_start_id

        trade_ids = df['TradeID'].astype(str)
        
        numeric_parts = []
        # Regex to find IDs like T001, T12345
        id_pattern = re.compile(rf"^{prefix}(\d+)$") 

        for id_str in trade_ids:
            match = id_pattern.match(id_str)
            if match:
                numeric_parts.append(int(match.group(1)))
        
        if not numeric_parts:
            # Fallback: try to see if any IDs are purely numeric strings
            plain_numeric_ids = []
            for id_str in trade_ids:
                if id_str.isdigit(): # Check if the string consists of digits only
                    try:
                        plain_numeric_ids.append(int(id_str))
                    except ValueError:
                        continue # Should not happen if isdigit() is true, but as a safeguard
            
            if plain_numeric_ids:
                max_num = max(plain_numeric_ids)
                return f"{prefix}{(max_num + 1):0{num_digits}d}"
            else:
                # No "T<number>" pattern and no plain numeric IDs found
                return default_start_id
        
        max_num = max(numeric_parts)
        next_id_num = max_num + 1
        return f"{prefix}{next_id_num:0{num_digits}d}"
