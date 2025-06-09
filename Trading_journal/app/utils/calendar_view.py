import dash
from dash import dcc, html, callback, Input, Output, State, ALL, ctx
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px # For potential mini-charts or if table styling needs it
from dash import dash_table
from datetime import datetime, date
import calendar # For generating calendar days
import io # For StringIO with pandas

def aggregate_daily_pnl(df_in):
    if df_in is None or df_in.empty:
        return pd.DataFrame(columns=['Date', 'NetP&L', 'TradeCount', 'TotalVolume']) # Added TotalVolume
    
    # Ensure OpenTimestamp is datetime
    df = df_in.copy() # Avoid SettingWithCopyWarning
    df['OpenTimestamp'] = pd.to_datetime(df['OpenTimestamp'])
    df['Date'] = df['OpenTimestamp'].dt.date # Extract just the date part

    daily_summary = df.groupby('Date').agg(
        NetPnl=('NetP&L', 'sum'),
        TradeCount=('TradeID', 'count'),
        TotalVolume=('Quantity', 'sum') # Example: sum of quantities or other relevant metric
    ).reset_index()
    daily_summary.rename(columns={'NetPnl': 'NetP&L'}, inplace=True) # Ensure consistent naming
    return daily_summary

def create_calendar_layout():
    current_year = datetime.now().year
    current_month = datetime.now().month
    return html.Div([
        dbc.Row([
            dbc.Col(dcc.Dropdown(
                id='calendar-year-selector',
                options=[{'label': str(y), 'value': y} for y in range(current_year - 5, current_year + 6)], # Example year range
                value=current_year,
                clearable=False
            ), width=3),
            dbc.Col(dcc.Dropdown(
                id='calendar-month-selector',
                options=[{'label': calendar.month_name[i], 'value': i} for i in range(1, 13)],
                value=current_month,
                clearable=False
            ), width=3)
        ], className="mb-3"),
        dbc.Row([
            dbc.Col(html.Div(id='calendar-grid-container'), width=12)
        ]),
        # Placeholder for future elements if needed
    ])

def register_calendar_callbacks(app):
    @app.callback(
        Output('calendar-grid-container', 'children'),
        [Input('calendar-year-selector', 'value'),
         Input('calendar-month-selector', 'value'),
         Input('filtered-data-store', 'data')],
        [State('trade-data-store', 'data')]
    )
    def update_calendar_grid(selected_year, selected_month, filtered_json_data, raw_json_data):
        json_data_to_use = filtered_json_data if filtered_json_data is not None else raw_json_data
        if json_data_to_use is None:
            return html.Div("No trade data available to display calendar.", className="text-center mt-3")

        try:
            df_trades = pd.read_json(io.StringIO(json_data_to_use), orient='split')
            if df_trades.empty:
                 return html.Div("No trade data for the selected period/filters.", className="text-center mt-3")
        except ValueError:
            return html.Div("Error reading trade data.", className="text-center mt-3")

        daily_pnl_data = aggregate_daily_pnl(df_trades)
        
        # Convert 'Date' in daily_pnl_data to datetime.date objects if they are strings
        if not daily_pnl_data.empty and isinstance(daily_pnl_data['Date'].iloc[0], str):
            daily_pnl_data['Date'] = pd.to_datetime(daily_pnl_data['Date']).dt.date

        month_calendar = calendar.monthcalendar(selected_year, selected_month)
        
        # For P&L color scaling
        min_pnl = daily_pnl_data['NetP&L'].min() if not daily_pnl_data.empty else 0
        max_pnl = daily_pnl_data['NetP&L'].max() if not daily_pnl_data.empty else 0

        def get_pnl_color_class(pnl):
            if pnl == 0 or pd.isna(pnl): return "calendar-pnl-neutral"
            if pnl > 0:
                # Normalize profit: higher P&L -> darker green
                if max_pnl > 0:
                    intensity = (pnl / max_pnl)
                    if intensity > 0.66: return "calendar-pnl-profit-high"
                    if intensity > 0.33: return "calendar-pnl-profit-medium"
                    return "calendar-pnl-profit-low"
                return "calendar-pnl-profit-low" # Default for any profit if max_pnl is 0 or less
            else: # pnl < 0
                # Normalize loss: lower P&L (more negative) -> darker red
                if min_pnl < 0:
                    intensity = (pnl / min_pnl) # pnl and min_pnl are negative, so ratio is positive
                    if intensity > 0.66: return "calendar-pnl-loss-high" # Most negative
                    if intensity > 0.33: return "calendar-pnl-loss-medium"
                    return "calendar-pnl-loss-low"
                return "calendar-pnl-loss-low" # Default for any loss if min_pnl is 0 or more

        header_row = dbc.Row([dbc.Col(html.Strong(day_name), className="text-center calendar-weekday-header") for day_name in calendar.weekheader(2).split()])
        
        weeks_rows = []
        for week in month_calendar:
            day_cells = []
            for day_num in week:
                day_content = ""
                # day_pnl_info = None # Not needed here
                cell_class = "calendar-day-cell p-0" # Adjusted padding to p-0 as button will handle it
                cell_id = None

                if day_num == 0: # Day not in current month
                    # Render a disabled button for empty cells to maintain grid structure and styling consistency
                    day_cells.append(dbc.Col(
                        dbc.Button(html.Div(""), className="w-100 h-100 calendar-day-button-style calendar-day-empty", disabled=True),
                        className=cell_class + " calendar-day-empty", width=True
                    ))
                    continue

                current_date_obj = date(selected_year, selected_month, day_num)
                day_pnl_info_series = daily_pnl_data[daily_pnl_data['Date'] == current_date_obj]

                if not day_pnl_info_series.empty:
                    pnl = day_pnl_info_series['NetP&L'].iloc[0]
                    trade_count = day_pnl_info_series['TradeCount'].iloc[0]
                    # total_volume = day_pnl_info_series['TotalVolume'].iloc[0] # If you want to display it
                    
                    pnl_color_class = get_pnl_color_class(pnl)
                    # cell_class += f" {pnl_color_class} calendar-day-active" # Class added to button instead
                    cell_id = {'type': 'calendar-day-button', 'date': current_date_obj.isoformat()}
                    
                    day_content = html.Div([
                        html.Div(str(day_num), className="calendar-day-number"),
                        html.Div(f"P&L: ${pnl:,.2f}", className="calendar-pnl"),
                        html.Div(f"Trades: {trade_count}", className="calendar-trade-count")
                    ])
                    button_extra_class = pnl_color_class + " calendar-day-active"
                else:
                    day_content = html.Div(str(day_num), className="calendar-day-number")
                    # cell_class += " calendar-day-no-trades" # Class added to button instead
                    button_extra_class = "calendar-day-no-trades"
                
                button_props = {
                    "children": day_content,
                    "className": f"w-100 h-100 calendar-day-button-style {button_extra_class}",
                    "disabled": (cell_id is None)
                }
                if cell_id is not None:
                    button_props["id"] = cell_id
                
                day_cells.append(dbc.Col(
                    dbc.Button(**button_props),
                    className=cell_class, width=True
                ))
            weeks_rows.append(dbc.Row(day_cells, className="g-1")) # g-1 for small gutters

        return html.Div([header_row] + weeks_rows, className="mt-2")

    @app.callback(
        [Output('daily-trades-modal', 'is_open'),
         Output('daily-trades-modal-title', 'children'),
         Output('daily-trades-content', 'children')],
        [Input({'type': 'calendar-day-button', 'date': ALL}, 'n_clicks'),
         Input('close-daily-trades-modal', 'n_clicks')], # New input
        [State('filtered-data-store', 'data'),
         State('trade-data-store', 'data'),
         State('daily-trades-modal', 'is_open')]
    )
    def display_daily_trades_popup(n_clicks_days, n_clicks_close, filtered_json_data, raw_json_data, is_open_current):
        triggered_input_id = ctx.triggered_id
        
        if not triggered_input_id:
            return dash.no_update, dash.no_update, dash.no_update

        # Handle close button click
        if triggered_input_id == 'close-daily-trades-modal':
            if n_clicks_close and n_clicks_close > 0:
                return False, dash.no_update, dash.no_update # Close modal, no change to title/content
            return dash.no_update, dash.no_update, dash.no_update

        # Handle calendar day click (existing logic)
        # The triggered_id for pattern-matching is a dict
        if isinstance(triggered_input_id, dict) and triggered_input_id.get('type') == 'calendar-day-button':
            # Original logic for day click:
            if not any(filter(None,n_clicks_days)): # Should not happen if this specific day button was the trigger, but good check
                return False, "Daily Trades", None 

            clicked_date_str = triggered_input_id['date']
            clicked_date_obj = date.fromisoformat(clicked_date_str)
            modal_title = f"Trades for {clicked_date_obj.strftime('%B %d, %Y')}"

            json_data_to_use = filtered_json_data if filtered_json_data is not None else raw_json_data
        if json_data_to_use is None:
            return True, modal_title, html.Div("No trade data available.")

        try:
            df_all_trades = pd.read_json(io.StringIO(json_data_to_use), orient='split')
        except ValueError:
             return True, modal_title, html.Div("Error reading trade data.")

        df_all_trades['OpenTimestamp'] = pd.to_datetime(df_all_trades['OpenTimestamp'])
        
        # Filter trades for the clicked day
        df_daily_trades = df_all_trades[df_all_trades['OpenTimestamp'].dt.date == clicked_date_obj].copy()

        if df_daily_trades.empty:
            return True, modal_title, html.Div("No trades found for this day.")

        # Format columns for display (similar to main trade table)
        if 'OpenTimestamp' in df_daily_trades.columns:
            df_daily_trades.loc[:, 'OpenTimestamp'] = df_daily_trades['OpenTimestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        if 'CloseTimestamp' in df_daily_trades.columns and not df_daily_trades['CloseTimestamp'].isnull().all():
            df_daily_trades.loc[:, 'CloseTimestamp'] = pd.to_datetime(df_daily_trades['CloseTimestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
        
        display_cols = ['TradeID', 'Symbol', 'OpenTimestamp', 'PositionType', 'EntryPrice', 'ExitPrice', 'Quantity', 'NetP&L']
        table_cols = [{"name": i, "id": i} for i in display_cols if i in df_daily_trades.columns]
        
        if 'NetP&L' in df_daily_trades.columns:
             df_daily_trades.loc[:, 'NetP&L'] = df_daily_trades['NetP&L'].map('${:,.2f}'.format)

        trades_table = dash_table.DataTable(
            data=df_daily_trades.to_dict('records'),
            columns=table_cols,
            page_size=10,
            style_table={'overflowX': 'auto', 'minWidth': '100%'},
        )
        return True, modal_title, trades_table
            
        # Default: no relevant trigger, no update to modal state
        return dash.no_update, dash.no_update, dash.no_update
