import dash
from dash import dcc, html, callback, Input, Output, State # Added callback, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd # Added pandas
import io # Added io
from datetime import datetime # Added datetime
import numpy as np # For np.isinf, np.nan
from app.utils import metrics_calculator # Corrected import path
import plotly.graph_objects as go # ADDED THIS IMPORT

def create_wmy_layout():
    layout = dbc.Container([
        dcc.Store(id='wmy-trade-data-store'), # To store intermediate data for WMY calculations
        
        dbc.Row([
            dbc.Col(html.H2("WMY Analysis (Weekly, Monthly, Yearly)"), width=12, className="mb-4 mt-4")
        ]),
        
        dbc.Card([
            dbc.CardHeader("WMY Filters"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Select Periodicity:"),
                        dbc.RadioItems(
                            id='wmy-periodicity-selector',
                            options=[
                                {'label': 'Weekly', 'value': 'W'},
                                {'label': 'Monthly', 'value': 'M'},
                                {'label': 'Yearly', 'value': 'Y'}
                            ],
                            value='M', # Default to Monthly
                            inline=True,
                            className="mb-2"
                        )
                    ], width=12, md=4),
                    dbc.Col([
                        dbc.Label("Select Year(s):"),
                        dcc.Dropdown(
                            id='wmy-year-filter',
                            multi=True,
                            placeholder="Select Year(s)"
                        )
                    ], width=12, md=4),
                    dbc.Col([ # Placeholder for Month filter, visibility controlled by callback
                        dbc.Label("Select Month(s):", html_for='wmy-month-filter'), # MODIFIED LABEL + html_for
                        dcc.Dropdown(
                            id='wmy-month-filter',
                            multi=True,
                            placeholder="Select Month(s)"
                            # REMOVED style={'display': 'none'} from here
                        )
                    ], width=12, md=4, id='wmy-month-filter-col', style={'display': 'none'}), # Initial state for the COL is hidden
                ], className="mb-3"),
                dbc.Row([
                    dbc.Col(
                        dbc.Button("Apply WMY Filters", id="apply-wmy-filters-button", color="primary", className="w-100"),
                        width=12
                    )
                ])
            ])
        ], className="mb-4"),
        
        dbc.Row([
            dbc.Col(html.Div(id='wmy-metrics-display-area'), width=12)
        ])
        # REMOVED SUMMARY CHART LAYOUT as it's now part of wmy-metrics-display-area
    ], fluid=True)
    return layout

def register_wmy_callbacks(app):
    @app.callback(
        Output('wmy-month-filter-col', 'style'),
        Input('wmy-periodicity-selector', 'value')
    )
    def toggle_month_filter_visibility(selected_periodicity):
        if selected_periodicity in ['W', 'M']: # Show for Weekly or Monthly
            return {'display': 'block', 'width': '100%'} # Ensure width is also set
        return {'display': 'none'} # Hide for Yearly

    @app.callback(
        [Output('wmy-year-filter', 'options'),
         Output('wmy-month-filter', 'options')],
        Input('trade-data-store', 'data') # Trigger when main data is loaded/changed
    )
    def populate_wmy_filter_options(json_data):
        if json_data is None:
            return [], []
        
        df = pd.read_json(io.StringIO(json_data), orient='split')
        if df.empty:
            return [], []

        # Ensure 'OpenTimestamp' is datetime
        df['OpenTimestamp'] = pd.to_datetime(df['OpenTimestamp'], errors='coerce')
        
        # Populate Year options
        years = sorted(df['OpenTimestamp'].dt.year.unique(), reverse=True)
        year_options = [{'label': str(year), 'value': year} for year in years]
        
        # Populate Month options (standard for all years)
        month_options = [
            {'label': datetime.strptime(str(i), "%m").strftime('%B'), 'value': i} 
            for i in range(1, 13)
        ]
        
        return year_options, month_options

    @app.callback(
        Output('wmy-trade-data-store', 'data'),
        Input('apply-wmy-filters-button', 'n_clicks'),
        [State('wmy-periodicity-selector', 'value'),
         State('wmy-year-filter', 'value'),
         State('wmy-month-filter', 'value'),
         State('trade-data-store', 'data')] # Main data source
    )
    def prepare_wmy_data(n_clicks, periodicity, selected_years, selected_months, json_data):
        if not n_clicks or json_data is None:
            return None # Or dash.no_update if preferred for no-op

        df = pd.read_json(io.StringIO(json_data), orient='split')
        if df.empty:
            return None

        df['OpenTimestamp'] = pd.to_datetime(df['OpenTimestamp'], errors='coerce')
        
        # Filter by years
        if selected_years: # Not None and not empty
            df = df[df['OpenTimestamp'].dt.year.isin(selected_years)]
        
        # Filter by months (if periodicity is Weekly or Monthly and months are selected)
        if periodicity in ['W', 'M'] and selected_months: # Not None and not empty
            df = df[df['OpenTimestamp'].dt.month.isin(selected_months)]
        
        if df.empty:
            return None
            
        # Store the filtered DataFrame (or just relevant columns) for metric calculation
        # The actual resampling/grouping might be better done in the next callback that consumes this store
        # to avoid storing large resampled objects if not necessary.
        # For now, store the filtered (but not yet resampled) data.
        return df.to_json(date_format='iso', orient='split')

    @app.callback(
        Output('wmy-metrics-display-area', 'children'), # Single output now
        Input('wmy-trade-data-store', 'data'),
        State('wmy-periodicity-selector', 'value')
    )
    def display_wmy_metrics(json_filtered_data, periodicity):
        if json_filtered_data is None:
            return dbc.Alert("Please apply WMY filters to see analysis.", color="info", className="mt-3")

        df_filtered = pd.read_json(io.StringIO(json_filtered_data), orient='split')
        if df_filtered.empty:
            return dbc.Alert("No trades match the selected WMY criteria.", color="warning", className="mt-3")

        df_filtered['OpenTimestamp'] = pd.to_datetime(df_filtered['OpenTimestamp'])
        df_filtered = df_filtered.sort_values(by='OpenTimestamp').set_index('OpenTimestamp')

        if periodicity == 'W':
            rule = 'W-MON'
            period_name_format = "Week of %Y-%m-%d"
        elif periodicity == 'M':
            rule = 'M'
            period_name_format = "%B %Y"
        elif periodicity == 'Y':
            rule = 'Y'
            period_name_format = "%Y"
        else:
            return dbc.Alert("Invalid periodicity selected.", color="danger", className="mt-3")

        resampled_groups = df_filtered.resample(rule)
        
        metrics_to_chart = [
            {'id': 'total_pnl', 'name': 'Net P&L', 'is_currency': True, 'is_percentage': False, 'is_pct_of_capital': False},
            {'id': 'total_trades', 'name': 'Total Trades', 'precision': 0},
            {'id': 'win_rate', 'name': 'Win Rate', 'is_percentage': True},
            {'id': 'profit_factor', 'name': 'Profit Factor'},
            {'id': 'overall_avg_pnl_numeric', 'name': 'Overall Avg P&L', 'is_currency': True},
            {'id': 'total_profit_numeric', 'name': 'Total Profit', 'is_currency': True},
            {'id': 'total_loss_numeric', 'name': 'Total Loss', 'is_currency': True},
            {'id': 'avg_pnl_winning_trades', 'name': 'Avg P&L Wins', 'is_currency': True},
            {'id': 'avg_pnl_pct_winning_trades', 'name': 'Avg P&L Wins %', 'is_pct_of_capital': True},
            {'id': 'avg_pnl_losing_trades', 'name': 'Avg P&L Losses', 'is_currency': True},
            {'id': 'avg_loss_pct_losing_trades', 'name': 'Avg Loss % Losses', 'is_pct_of_capital': True},
            {'id': 'num_winning_trades', 'name': 'Num Wins', 'precision': 0},
            {'id': 'pct_winning_trades', 'name': 'Pct Wins', 'is_percentage': True},
            {'id': 'num_losing_trades', 'name': 'Num Losses', 'precision': 0},
            {'id': 'pct_losing_trades', 'name': 'Pct Losses', 'is_percentage': True},
            {'id': 'max_profit_numeric', 'name': 'Max Profit', 'is_currency': True},
            {'id': 'max_profit_pct', 'name': 'Max Profit %', 'is_pct_of_capital': True},
            {'id': 'max_loss_numeric', 'name': 'Max Loss', 'is_currency': True},
            {'id': 'max_loss_pct', 'name': 'Max Loss %', 'is_pct_of_capital': True},
            {'id': 'avg_capital_deployed_winning_trades', 'name': 'Avg Cap Deployed Wins', 'is_currency': True},
            {'id': 'avg_capital_deployed_losing_trades', 'name': 'Avg Cap Deployed Losses', 'is_currency': True},
            {'id': 'avg_holding_days_winning', 'name': 'Avg Holding Days Wins'},
            {'id': 'avg_holding_days_losing', 'name': 'Avg Holding Days Losses'},
            {'id': 'avg_risk_reward_ratio_numeric', 'name': 'Avg Risk/Reward'},
            {'id': 'r_multiple_avg_pct', 'name': 'R Multiple (Avg Pct)'}, # Assuming this is a direct percentage value
            {'id': 'gain_loss_ratio_numeric', 'name': 'Gain/Loss Ratio'},
            {'id': 'optimal_f', 'name': 'Optimal F', 'is_percentage': True},
        ]

        all_metrics_data = {metric['id']: [] for metric in metrics_to_chart}
        period_labels_for_chart = []

        for period_timestamp, group_df in resampled_groups:
            if group_df.empty:
                continue
            
            period_df_for_calc = group_df.reset_index()
            metrics = metrics_calculator.calculate_detailed_performance_metrics(period_df_for_calc)
            period_label_str = period_timestamp.strftime(period_name_format)
            period_labels_for_chart.append(period_label_str)

            for metric_info in metrics_to_chart:
                value = metrics.get(metric_info['id'])
                # Handle potential NaN/inf values, replace with 0 for charting if problematic, or handle in chart generation
                if value is None or np.isinf(value) or pd.isna(value):
                    value = 0  # Or np.nan if charts handle it gracefully
                all_metrics_data[metric_info['id']].append(value)

        if not period_labels_for_chart:
            return dbc.Alert("No data to display for the selected periods after processing. Try adjusting year/month filters or ensure trades exist in the selected timeframe.", color="info", className="mt-3")

        charts_components = []
        for metric_info in metrics_to_chart:
            metric_id = metric_info['id']
            metric_name = metric_info['name']
            metric_values = all_metrics_data[metric_id]

            fig = go.Figure()
            
            # Determine bar colors, especially for P&L
            current_bar_colors = 'rgba(50, 150, 250, 0.7)' # Default blue
            if metric_id == 'total_pnl':
                current_bar_colors = ['rgba(40, 167, 69, 0.7)' if v >= 0 else 'rgba(220, 53, 69, 0.7)' for v in metric_values]
            
            fig.add_trace(go.Bar(
                x=period_labels_for_chart,
                y=metric_values,
                name=metric_name,
                marker_color=current_bar_colors
            ))
            
            yaxis_title = metric_name
            if metric_info.get('is_currency'):
                yaxis_title += " ($)"
            elif metric_info.get('is_percentage') or metric_info.get('is_pct_of_capital'):
                 yaxis_title += " (%)"
                 # For percentage metrics, raw values are 0.xx, so multiply by 100 for display if needed or format tick labels
                 # If calculate_detailed_performance_metrics returns e.g. win_rate as 0.55, display as 55%
                 # Plotly can format y-axis ticks as percentages if data is in 0-1 range: tickformat=".0%"

            fig.update_layout(
                title_text=f"{metric_name} per Period",
                xaxis_title="Period",
                yaxis_title=yaxis_title,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#f0f0f0'),
                height=350, # Standard height for charts
                margin=dict(t=50, b=80) # Adjust top/bottom margin for title/labels
            )
            
            if metric_info.get('is_percentage') or metric_info.get('is_pct_of_capital'):
                fig.update_yaxes(tickformat=".0%") # Format y-axis as percentage if data is 0.xx

            # Each graph component will be a Col spanning full width, then wrapped in its own Row
            charts_components.append(dbc.Col(dcc.Graph(figure=fig), width=12, className="mb-3"))

        # Arrange each chart component in its own Row
        individual_chart_rows = []
        for chart_col in charts_components:
            individual_chart_rows.append(dbc.Row([chart_col]))
            
        return html.Div(individual_chart_rows, className="mt-4")
