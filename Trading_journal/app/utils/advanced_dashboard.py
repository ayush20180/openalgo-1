"""
Advanced Metrics Integration Module
---------------------------------
This module integrates advanced metrics into the dashboard.
"""

import dash
from dash import dcc, html, callback, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
from io import StringIO

# Import the main metrics_calculator module
from app.utils import metrics_calculator, advanced_metrics # Added metrics_calculator


def create_advanced_metrics_layout():
    """
    Create the layout for advanced metrics section.
    
    Returns:
        Dash layout components for advanced metrics
    """
    layout = html.Div([
        dbc.Row([
            dbc.Col([
                html.H3("Advanced Performance Analytics", className="mt-4"),
                html.P("Detailed analysis of trading performance with advanced metrics", className="text-muted")
            ])
        ]),
        
        # Drawdown Analysis
        dbc.Row([
            dbc.Col([
                html.H4("Drawdown Analysis", className="mt-3"),
                dcc.Graph(id='drawdown-chart')
            ])
        ]),
        
        # Risk-Adjusted Returns
        dbc.Row([
            dbc.Col([
                html.H4("Risk-Adjusted Returns", className="mt-3"),
                html.Div(id='risk-adjusted-metrics')
            ])
        ]),
        
        # Position Type Analysis
        dbc.Row([
            dbc.Col([
                html.H4("Position Type Analysis", className="mt-3"),
                dcc.Graph(id='position-type-chart')
            ], width=6),
            dbc.Col([
                html.H4("Exit Signal Analysis", className="mt-3"),
                dcc.Graph(id='exit-signal-chart')
            ], width=6)
        ]),
        
        # Consecutive Wins/Losses
        dbc.Row([
            dbc.Col([
                html.H4("Win/Loss Streaks", className="mt-3"),
                html.Div(id='consecutive-wins-losses')
            ])
        ]),
        
        # Volatility Chart
        dbc.Row([
            dbc.Col([
                html.H4("P&L Volatility (20-day rolling)", className="mt-3"),
                dcc.Graph(id='volatility-chart')
            ])
        ]),
        
        # Trade Clusters
        dbc.Row([
            dbc.Col([
                html.H4("Trade Clusters", className="mt-3"),
                html.P("Performance grouped by similar characteristics", className="text-muted"),
                dcc.Dropdown(
                    id='cluster-feature-selector',
                    options=[
                        {'label': 'Symbol & Position Type', 'value': 'symbol_position'},
                        {'label': 'Symbol & Exit Signal', 'value': 'symbol_exit'},
                        {'label': 'Exchange & Product Type', 'value': 'exchange_product'}
                    ],
                    value='symbol_position',
                    clearable=False
                ),
                dcc.Graph(id='cluster-analysis-chart')
            ])
        ]),

        # --- New Detailed Metrics Sections ---
        html.H3("Detailed Performance Metrics", className="mt-5 mb-4 text-center"),

        # Card for Win/Loss Analysis
        dbc.Card([
            dbc.CardHeader(html.H5("Win/Loss Analysis", className="mb-0")),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([html.H6("Winning Trades:", className="text-muted small"), html.Div(id='adv-num-winning-trades-val', className="h5"), html.Div(id='adv-pct-winning-trades-val', className="small text-muted")], md=3, className="mb-3 text-center border-end"),
                    dbc.Col([html.H6("Losing Trades:", className="text-muted small"), html.Div(id='adv-num-losing-trades-val', className="h5"), html.Div(id='adv-pct-losing-trades-val', className="small text-muted")], md=3, className="mb-3 text-center border-end"),
                    dbc.Col([html.H6("Win Rate:", className="text-muted small"), html.Div(id='adv-win-rate-val', className="h5")], md=3, className="mb-3 text-center border-end"), # From summary_stats
                    dbc.Col([html.H6("Profit Factor:", className="text-muted small"), html.Div(id='adv-profit-factor-val', className="h5")], md=3, className="mb-3 text-center"), # From summary_stats
                ])
            ])
        ], className="mb-4 shadow-sm"),

        # Card for P&L Details
        dbc.Card([
            dbc.CardHeader(html.H5("P&L Details", className="mb-0")),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([html.H6("Avg P&L (Wins):", className="text-muted small"), html.Div(id='adv-avg-pnl-wins-numeric-val', className="h5"), html.Div(id='adv-avg-pnl-wins-pct-val', className="small text-muted")], md=3, className="mb-3 text-center border-end"),
                    dbc.Col([html.H6("Avg P&L (Losses):", className="text-muted small"), html.Div(id='adv-avg-pnl-losses-numeric-val', className="h5"), html.Div(id='adv-avg-pnl-losses-pct-val', className="small text-muted")], md=3, className="mb-3 text-center border-end"),
                    dbc.Col([html.H6("Total Profit:", className="text-muted small"), html.Div(id='adv-total-profit-numeric-val', className="h5")], md=3, className="mb-3 text-center border-end"),
                    dbc.Col([html.H6("Total Loss:", className="text-muted small"), html.Div(id='adv-total-loss-numeric-val', className="h5")], md=3, className="mb-3 text-center"),
                ]),
                dbc.Row([
                    dbc.Col([html.H6("Overall Avg P&L:", className="text-muted small"), html.Div(id='adv-overall-avg-pnl-numeric-val', className="h5"), html.Div(id='adv-overall-avg-pnl-pct-val', className="small text-muted")], md=3, className="mb-3 text-center border-end"),
                    dbc.Col([html.H6("Expectancy:", className="text-muted small"), html.Div(id='adv-expectancy-val', className="h5")], md=3, className="mb-3 text-center"), # From summary_stats
                ])
            ])
        ], className="mb-4 shadow-sm"),
        
        # Card for Extreme Performance
        dbc.Card([
            dbc.CardHeader(html.H5("Extreme Performance", className="mb-0")),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([html.H6("Max Profit:", className="text-muted small"), html.Div(id='adv-max-profit-numeric-val', className="h5"), html.Div(id='adv-max-profit-pct-val', className="small text-muted")], md=3, className="mb-3 text-center border-end"),
                    dbc.Col([html.H6("Max Loss:", className="text-muted small"), html.Div(id='adv-max-loss-numeric-val', className="h5"), html.Div(id='adv-max-loss-pct-val', className="small text-muted")], md=3, className="mb-3 text-center"),
                ])
            ])
        ], className="mb-4 shadow-sm"),

        # Card for Position & Holding Analysis
        dbc.Card([
            dbc.CardHeader(html.H5("Position & Holding Analysis", className="mb-0")),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([html.H6("Avg Capital (Wins):", className="text-muted small"), html.Div(id='adv-avg-capital-wins-val', className="h5")], md=3, className="mb-3 text-center border-end"),
                    dbc.Col([html.H6("Avg Capital (Losses):", className="text-muted small"), html.Div(id='adv-avg-capital-losses-val', className="h5")], md=3, className="mb-3 text-center border-end"),
                    dbc.Col([html.H6("Avg Holding (Wins):", className="text-muted small"), html.Div(id='adv-avg-holding-wins-val', className="h5")], md=3, className="mb-3 text-center border-end"),
                    dbc.Col([html.H6("Avg Holding (Losses):", className="text-muted small"), html.Div(id='adv-avg-holding-losses-val', className="h5")], md=3, className="mb-3 text-center"),
                ]),
                 dbc.Row([
                    dbc.Col([html.H6("Avg Capital (All):", className="text-muted small"), html.Div(id='adv-avg-capital-all-val', className="h5")], md=3, className="mb-3 text-center border-end"),
                    dbc.Col([html.H6("Avg Holding (All):", className="text-muted small"), html.Div(id='adv-avg-holding-all-val', className="h5")], md=3, className="mb-3 text-center"),
                ])
            ])
        ], className="mb-4 shadow-sm"),

        # Card for Advanced Ratios
        dbc.Card([
            dbc.CardHeader(html.H5("Advanced Trading Ratios", className="mb-0")),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([html.H6("Avg Risk/Reward (Numeric):", className="text-muted small"), html.Div(id='adv-avg-risk-reward-numeric-val', className="h5")], md=3, className="mb-3 text-center border-end"),
                    dbc.Col([html.H6("R-Multiple (Avg Pct):", className="text-muted small"), html.Div(id='adv-r-multiple-avg-pct-val', className="h5")], md=3, className="mb-3 text-center border-end"),
                    dbc.Col([html.H6("Gain/Loss Ratio (Numeric):", className="text-muted small"), html.Div(id='adv-gain-loss-ratio-numeric-val', className="h5")], md=3, className="mb-3 text-center border-end"), # This is Profit Factor
                    dbc.Col([html.H6("Optimal F (%):", className="text-muted small"), html.Div(id='adv-optimal-f-val', className="h5")], md=3, className="mb-3 text-center"),
                ])
            ])
        ], className="mb-4 shadow-sm"),
        
        # Profit vs Loss Bar Chart
        dbc.Card([
            dbc.CardHeader(html.H5("Profit vs Loss Summary", className="mb-0")),
            dbc.CardBody([
                dcc.Graph(id='adv-profit-loss-barchart', style={'height': '400px'})
            ])
        ], className="mt-4 mb-4 shadow-sm"), # Added mb-4 for spacing

    ]) # End of main layout Div
    
    return layout


def register_advanced_callbacks(app):
    """
    Register callbacks for advanced metrics.
    
    Args:
        app: Dash application instance
    """
    
    @app.callback(
        Output('drawdown-chart', 'figure'),
        Input('trade-data-store', 'data'),
        Input('filtered-data-store', 'data')
    )
    def update_drawdown_chart(json_data, filtered_json_data):
        # Use filtered data if available, otherwise use all data
        data_to_use = filtered_json_data if filtered_json_data is not None else json_data
        
        if data_to_use is None:
            # Return empty figure
            empty_fig = go.Figure()
            empty_fig.update_layout(
                title="No data available for drawdown analysis",
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
            return empty_fig
        
        # Convert JSON to DataFrame using StringIO to avoid deprecation warning
        from io import StringIO
        df = pd.read_json(StringIO(data_to_use), orient='split')

        if df.empty:
            empty_fig = go.Figure()
            empty_fig.update_layout(
                title="No data available for drawdown analysis after filtering",
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
            return empty_fig

        # Ensure timestamps are datetime objects
        if 'OpenTimestamp' in df.columns:
            df['OpenTimestamp'] = pd.to_datetime(df['OpenTimestamp'])
        if 'CloseTimestamp' in df.columns:
            df['CloseTimestamp'] = pd.to_datetime(df['CloseTimestamp'])
        
        # Sort by timestamp
        df = df.sort_values('OpenTimestamp')
        
        # Calculate cumulative P&L
        df['CumulativeP&L'] = df['NetP&L'].cumsum()
        
        # Calculate drawdown
        drawdown_data = advanced_metrics.calculate_max_drawdown(df['CumulativeP&L'])
        
        # Create figure with two traces
        fig = go.Figure()
        
        # Add cumulative P&L trace
        fig.add_trace(go.Scatter(
            x=df['OpenTimestamp'],
            y=df['CumulativeP&L'],
            mode='lines',
            name='Cumulative P&L',
            line=dict(color='green', width=2)
        ))
        
        # Add drawdown trace
        running_max = df['CumulativeP&L'].cummax()
        drawdown = df['CumulativeP&L'] - running_max
        
        fig.add_trace(go.Scatter(
            x=df['OpenTimestamp'],
            y=drawdown,
            mode='lines',
            name='Drawdown',
            line=dict(color='red', width=2),
            fill='tozeroy',
            fillcolor='rgba(255, 0, 0, 0.2)'
        ))
        
        # Add annotations for maximum drawdown
        if drawdown_data['max_drawdown_idx'] is not None:
            max_dd_date = df.loc[drawdown_data['max_drawdown_idx'], 'OpenTimestamp']
            fig.add_annotation(
                x=max_dd_date,
                y=drawdown_data['max_drawdown'],
                text=f"Max DD: ${drawdown_data['max_drawdown']:.2f}",
                showarrow=True,
                arrowhead=1,
                ax=0,
                ay=40
            )
        
        # Update layout
        fig.update_layout(
            title="Equity Curve and Drawdown Analysis",
            xaxis_title="Date",
            yaxis_title="Value ($)",
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    @app.callback(
        Output('risk-adjusted-metrics', 'children'),
        Input('trade-data-store', 'data'),
        Input('filtered-data-store', 'data')
    )
    def update_risk_adjusted_metrics(json_data, filtered_json_data):
        # Use filtered data if available, otherwise use all data
        data_to_use = filtered_json_data if filtered_json_data is not None else json_data
        
        if data_to_use is None:
            return html.Div("No data available for risk-adjusted metrics")
        
        # Convert JSON to DataFrame using StringIO to avoid deprecation warning
        df = pd.read_json(StringIO(data_to_use), orient='split')
        # Ensure timestamps are datetime objects
        if 'OpenTimestamp' in df.columns:
            df['OpenTimestamp'] = pd.to_datetime(df['OpenTimestamp'])
        if 'CloseTimestamp' in df.columns:
            df['CloseTimestamp'] = pd.to_datetime(df['CloseTimestamp'])

        # Group by date to get daily P&L
        df['Date'] = df['OpenTimestamp'].dt.date
        daily_pnl = df.groupby('Date')['NetP&L'].sum().reset_index()
        
        # Calculate risk-adjusted metrics
        sharpe_ratio = advanced_metrics.calculate_sharpe_ratio(daily_pnl['NetP&L'])
        sortino_ratio = advanced_metrics.calculate_sortino_ratio(daily_pnl['NetP&L'])
        
        # Calculate max drawdown
        df = df.sort_values('OpenTimestamp')
        df['CumulativeP&L'] = df['NetP&L'].cumsum()
        drawdown_data = advanced_metrics.calculate_max_drawdown(df['CumulativeP&L'])
        
        # Create metrics cards
        metrics_cards = dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardHeader("Sharpe Ratio"),
                dbc.CardBody(html.H4(f"{sharpe_ratio:.2f}"))
            ]), width=3),
            dbc.Col(dbc.Card([
                dbc.CardHeader("Sortino Ratio"),
                dbc.CardBody(html.H4(f"{sortino_ratio:.2f}"))
            ]), width=3),
            dbc.Col(dbc.Card([
                dbc.CardHeader("Max Drawdown"),
                dbc.CardBody(html.H4(f"${abs(drawdown_data['max_drawdown']):.2f}"))
            ]), width=3),
            dbc.Col(dbc.Card([
                dbc.CardHeader("Max Drawdown %"),
                dbc.CardBody(html.H4(f"{abs(drawdown_data['max_drawdown_pct'])*100:.2f}%"))
            ]), width=3)
        ])
        
        return metrics_cards
    
    @app.callback(
        Output('position-type-chart', 'figure'),
        Input('trade-data-store', 'data'),
        Input('filtered-data-store', 'data')
    )
    def update_position_type_chart(json_data, filtered_json_data):
        # Use filtered data if available, otherwise use all data
        data_to_use = filtered_json_data if filtered_json_data is not None else json_data
        
        if data_to_use is None:
            # Return empty figure
            empty_fig = go.Figure()
            empty_fig.update_layout(
                title="No data available for position type analysis",
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
            return empty_fig
        
        # Convert JSON to DataFrame using StringIO to avoid deprecation warning
        df = pd.read_json(StringIO(data_to_use), orient='split')
        # Ensure timestamps are datetime objects
        if 'OpenTimestamp' in df.columns:
            df['OpenTimestamp'] = pd.to_datetime(df['OpenTimestamp'])
        if 'CloseTimestamp' in df.columns:
            df['CloseTimestamp'] = pd.to_datetime(df['CloseTimestamp'])
        
        # Calculate trade duration if not already present
        if 'TradeDuration' not in df.columns:
            df['TradeDuration'] = df['CloseTimestamp'] - df['OpenTimestamp']
        
        # Get position type performance
        position_performance = advanced_metrics.calculate_performance_by_positiontype(df)
        
        # Convert to DataFrame for plotting
        position_df = pd.DataFrame.from_dict(position_performance, orient='index')

        # Check if DataFrame is empty or essential columns are missing
        if position_df.empty or 'win_rate' not in position_df.columns or \
           'profit_factor' not in position_df.columns or 'total_pnl' not in position_df.columns:
            empty_fig = go.Figure()
            empty_fig.update_layout(
                title="No data available for position type analysis after processing",
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
            return empty_fig
        
        # Create figure with subplots
        fig = go.Figure()
        
        # Add win rate bars
        fig.add_trace(go.Bar(
            x=position_df.index,
            y=position_df['win_rate'],
            name='Win Rate',
            marker_color='lightgreen',
            text=[f"{x:.1%}" for x in position_df['win_rate']],
            textposition='auto'
        ))
        
        # Add profit factor bars
        fig.add_trace(go.Bar(
            x=position_df.index,
            y=position_df['profit_factor'],
            name='Profit Factor',
            marker_color='lightblue',
            text=[f"{x:.2f}" for x in position_df['profit_factor']],
            textposition='auto',
            visible='legendonly'  # Hide by default
        ))
        
        # Add total P&L bars
        fig.add_trace(go.Bar(
            x=position_df.index,
            y=position_df['total_pnl'],
            name='Total P&L',
            marker_color='gold',
            text=[f"${x:.2f}" for x in position_df['total_pnl']],
            textposition='auto'
        ))
        
        # Update layout
        fig.update_layout(
            title="Performance by Position Type",
            xaxis_title="Position Type",
            yaxis_title="Value",
            barmode='group',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    @app.callback(
        Output('exit-signal-chart', 'figure'),
        Input('trade-data-store', 'data'),
        Input('filtered-data-store', 'data')
    )
    def update_exit_signal_chart(json_data, filtered_json_data):
        # Use filtered data if available, otherwise use all data
        data_to_use = filtered_json_data if filtered_json_data is not None else json_data
        
        if data_to_use is None:
            # Return empty figure
            empty_fig = go.Figure()
            empty_fig.update_layout(
                title="No data available for exit signal analysis",
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
            return empty_fig
        
        # Convert JSON to DataFrame using StringIO to avoid deprecation warning
        df = pd.read_json(StringIO(data_to_use), orient='split')
        
        # Get exit signal performance
        exit_performance = advanced_metrics.analyze_exit_signals(df)
        
        # Convert to DataFrame for plotting
        exit_df = pd.DataFrame.from_dict(exit_performance, orient='index')
        
        # Create figure with subplots
        fig = go.Figure()
        
        # Add win rate bars
        fig.add_trace(go.Bar(
            x=exit_df.index,
            y=exit_df['win_rate'],
            name='Win Rate',
            marker_color='lightgreen',
            text=[f"{x:.1%}" for x in exit_df['win_rate']],
            textposition='auto'
        ))
        
        # Add frequency bars
        fig.add_trace(go.Bar(
            x=exit_df.index,
            y=exit_df['frequency'],
            name='Frequency',
            marker_color='lightblue',
            text=[f"{x:.1%}" for x in exit_df['frequency']],
            textposition='auto',
            visible='legendonly'  # Hide by default
        ))
        
        # Add average P&L bars
        exit_df['avg_pnl'] = exit_df['total_pnl'] / exit_df['count']
        fig.add_trace(go.Bar(
            x=exit_df.index,
            y=exit_df['avg_pnl'],
            name='Avg P&L per Trade',
            marker_color='gold',
            text=[f"${x:.2f}" for x in exit_df['avg_pnl']],
            textposition='auto'
        ))
        
        # Update layout
        fig.update_layout(
            title="Performance by Exit Signal",
            xaxis_title="Exit Signal",
            yaxis_title="Value",
            barmode='group',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    # Callback for Detailed Performance Metrics
    @app.callback(
        [
            # Win/Loss Analysis Card
            Output('adv-num-winning-trades-val', 'children'),
            Output('adv-pct-winning-trades-val', 'children'),
            Output('adv-num-losing-trades-val', 'children'),
            Output('adv-pct-losing-trades-val', 'children'),
            Output('adv-win-rate-val', 'children'),
            Output('adv-profit-factor-val', 'children'),
            # P&L Details Card
            Output('adv-avg-pnl-wins-numeric-val', 'children'),
            Output('adv-avg-pnl-wins-pct-val', 'children'),
            Output('adv-avg-pnl-losses-numeric-val', 'children'),
            Output('adv-avg-pnl-losses-pct-val', 'children'),
            Output('adv-total-profit-numeric-val', 'children'),
            Output('adv-total-loss-numeric-val', 'children'),
            Output('adv-overall-avg-pnl-numeric-val', 'children'),
            Output('adv-overall-avg-pnl-pct-val', 'children'),
            Output('adv-expectancy-val', 'children'),
            # Extreme Performance Card
            Output('adv-max-profit-numeric-val', 'children'),
            Output('adv-max-profit-pct-val', 'children'),
            Output('adv-max-loss-numeric-val', 'children'),
            Output('adv-max-loss-pct-val', 'children'),
            # Position & Holding Analysis Card
            Output('adv-avg-capital-wins-val', 'children'),
            Output('adv-avg-capital-losses-val', 'children'),
            Output('adv-avg-holding-wins-val', 'children'),
            Output('adv-avg-holding-losses-val', 'children'),
            Output('adv-avg-capital-all-val', 'children'),
            Output('adv-avg-holding-all-val', 'children'),
            # Advanced Ratios Card
            Output('adv-avg-risk-reward-numeric-val', 'children'),
            Output('adv-r-multiple-avg-pct-val', 'children'),
            Output('adv-gain-loss-ratio-numeric-val', 'children'),
            Output('adv-optimal-f-val', 'children'),
            # New Bar Chart Output
            Output('adv-profit-loss-barchart', 'figure'),
        ],
        [Input('trade-data-store', 'data'), Input('filtered-data-store', 'data')]
    )
    def update_detailed_performance_metrics_display(json_data, filtered_json_data):
        data_to_use = filtered_json_data if filtered_json_data is not None else json_data
        
        # Initialize empty figure for the bar chart
        profit_loss_fig = go.Figure()
        # Default title, will be updated if data is available
        profit_loss_fig.update_layout(
            title_text="Total Profit vs Total Loss (No data)",
            xaxis_title="Category",
            yaxis_title="Amount ($)"
        )

        if data_to_use is None:
            na_val = "N/A"
            # Return N/A for all text fields and empty fig for chart
            return tuple([na_val] * 29 + [profit_loss_fig])

        df = pd.read_json(StringIO(data_to_use), orient='split')
        if df.empty:
            na_val = "N/A"
            return tuple([na_val] * 29 + [profit_loss_fig])

        metrics = metrics_calculator.calculate_detailed_performance_metrics(df)

        # Populate the bar chart for total profit vs total loss
        total_profit = metrics.get('total_profit_numeric')
        total_loss = metrics.get('total_loss_numeric') # This is a positive value

        if total_profit is not None and total_loss is not None and (total_profit != 0 or total_loss != 0):
            categories = ['Total Profit', 'Total Loss']
            # Ensure total_loss is positive for bar chart height
            values = [total_profit, total_loss if total_loss >= 0 else -total_loss] 
            colors = ['green', 'red']
            
            profit_loss_fig = go.Figure(data=[
                go.Bar(
                    x=categories, 
                    y=values, 
                    marker_color=colors,
                    text=[f"${v:,.2f}" for v in values], # Format text on bars
                    textposition='auto'
                )
            ])
            profit_loss_fig.update_layout(
                title_text="Total Profit vs. Total Loss",
                xaxis_title="Category",
                yaxis_title="Amount ($)",
                showlegend=False # No legend needed for two bars
            )
        # else, the default "No data" or "values missing" title remains

        def fmt_pct(value, default="N/A"):
            return f"{value*100:.2f}%" if value is not None else default
        
        def fmt_num(value, precision=2, default="N/A"):
            return f"{value:.{precision}f}" if value is not None else default

        def fmt_currency(value, default="N/A"):
            return f"${value:,.2f}" if value is not None else default

        # Win/Loss Analysis
        num_winning_trades_val = fmt_num(metrics.get('num_winning_trades'), 0)
        pct_winning_trades_val = fmt_pct(metrics.get('pct_winning_trades'))
        num_losing_trades_val = fmt_num(metrics.get('num_losing_trades'), 0)
        pct_losing_trades_val = fmt_pct(metrics.get('pct_losing_trades'))
        win_rate_val = fmt_pct(metrics.get('win_rate'))
        profit_factor_val = fmt_num(metrics.get('profit_factor'))

        # P&L Details
        avg_pnl_wins_numeric_val = fmt_currency(metrics.get('avg_pnl_winning_trades'))
        avg_pnl_wins_pct_val = f"({fmt_pct(metrics.get('avg_pnl_pct_winning_trades'))})" if metrics.get('avg_pnl_pct_winning_trades') is not None else ""
        avg_pnl_losses_numeric_val = fmt_currency(metrics.get('avg_pnl_losing_trades'))
        avg_pnl_losses_pct_val = f"({fmt_pct(metrics.get('avg_loss_pct_losing_trades'))})" if metrics.get('avg_loss_pct_losing_trades') is not None else "" # Note: avg_loss_pct is positive
        total_profit_numeric_val = fmt_currency(metrics.get('total_profit_numeric'))
        total_loss_numeric_val = fmt_currency(metrics.get('total_loss_numeric')) # This is positive
        overall_avg_pnl_numeric_val = fmt_currency(metrics.get('overall_avg_pnl_numeric'))
        overall_avg_pnl_pct_val = f"({fmt_pct(metrics.get('overall_avg_pnl_pct'))})" if metrics.get('overall_avg_pnl_pct') is not None else ""
        expectancy_val = fmt_currency(metrics.get('expectancy'))
        
        # Extreme Performance
        max_profit_numeric_val = fmt_currency(metrics.get('max_profit_numeric'))
        max_profit_pct_val = f"({fmt_pct(metrics.get('max_profit_pct'))})" if metrics.get('max_profit_pct') is not None else ""
        max_loss_numeric_val = fmt_currency(metrics.get('max_loss_numeric')) # This is negative
        max_loss_pct_val = f"({fmt_pct(metrics.get('max_loss_pct'))})" if metrics.get('max_loss_pct') is not None else "" # This is positive

        # Position & Holding Analysis
        avg_capital_wins_val = fmt_currency(metrics.get('avg_capital_deployed_winning_trades'))
        avg_capital_losses_val = fmt_currency(metrics.get('avg_capital_deployed_losing_trades'))
        avg_holding_wins_val = f"{fmt_num(metrics.get('avg_holding_days_winning'))} days"
        avg_holding_losses_val = f"{fmt_num(metrics.get('avg_holding_days_losing'))} days"
        avg_capital_all_val = fmt_currency(metrics.get('avg_capital_deployed_all_trades'))
        avg_holding_all_val = f"{fmt_num(metrics.get('avg_holding_days_all'))} days"

        # Advanced Ratios
        avg_risk_reward_numeric_val = fmt_num(metrics.get('avg_risk_reward_ratio_numeric'))
        r_multiple_avg_pct_val = fmt_num(metrics.get('r_multiple_avg_pct'))
        gain_loss_ratio_numeric_val = fmt_num(metrics.get('gain_loss_ratio_numeric')) # Profit Factor
        optimal_f_val = fmt_pct(metrics.get('optimal_f'))

        return (
            num_winning_trades_val, pct_winning_trades_val, num_losing_trades_val, pct_losing_trades_val,
            win_rate_val, profit_factor_val,
            avg_pnl_wins_numeric_val, avg_pnl_wins_pct_val, avg_pnl_losses_numeric_val, avg_pnl_losses_pct_val,
            total_profit_numeric_val, total_loss_numeric_val, overall_avg_pnl_numeric_val, overall_avg_pnl_pct_val,
            expectancy_val,
            max_profit_numeric_val, max_profit_pct_val, max_loss_numeric_val, max_loss_pct_val,
            avg_capital_wins_val, avg_capital_losses_val, avg_holding_wins_val, avg_holding_losses_val,
            avg_capital_all_val, avg_holding_all_val,
            avg_risk_reward_numeric_val, r_multiple_avg_pct_val, gain_loss_ratio_numeric_val, optimal_f_val,
            profit_loss_fig # Add the figure to the return tuple
        )

    @app.callback(
        Output('consecutive-wins-losses', 'children'),
        Input('trade-data-store', 'data'),
        Input('filtered-data-store', 'data')
    )
    def update_consecutive_wins_losses(json_data, filtered_json_data):
        # Use filtered data if available, otherwise use all data
        data_to_use = filtered_json_data if filtered_json_data is not None else json_data
        
        if data_to_use is None:
            return html.Div("No data available for win/loss streak analysis")
        
        # Convert JSON to DataFrame using StringIO to avoid deprecation warning
        df = pd.read_json(StringIO(data_to_use), orient='split')
        if df.empty: # Added check for empty df after potential filtering
            return html.Div("No data available for win/loss streak analysis after filtering")

        # Calculate consecutive wins/losses
        streak_data = advanced_metrics.calculate_consecutive_wins_losses(df) # df here is already filtered
        
        # Create metrics cards
        metrics_cards = dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardHeader("Max Consecutive Wins"),
                dbc.CardBody(html.H4(f"{streak_data['max_consecutive_wins']}"))
            ]), width=3),
            dbc.Col(dbc.Card([
                dbc.CardHeader("Max Consecutive Losses"),
                dbc.CardBody(html.H4(f"{streak_data['max_consecutive_losses']}"))
            ]), width=3),
            dbc.Col(dbc.Card([
                dbc.CardHeader("Avg Win Streak"),
                dbc.CardBody(html.H4(f"{streak_data['avg_win_streak']:.2f}"))
            ]), width=3),
            dbc.Col(dbc.Card([
                dbc.CardHeader("Avg Loss Streak"),
                dbc.CardBody(html.H4(f"{streak_data['avg_loss_streak']:.2f}"))
            ]), width=3)
        ])
        
        # Create streak distribution chart
        win_streak_counts = pd.Series(streak_data['win_streaks']).value_counts().sort_index()
        loss_streak_counts = pd.Series(streak_data['loss_streaks']).value_counts().sort_index()
        
        fig = go.Figure()
        
        # Add win streak bars
        fig.add_trace(go.Bar(
            x=win_streak_counts.index,
            y=win_streak_counts.values,
            name='Win Streaks',
            marker_color='green'
        ))
        
        # Add loss streak bars
        fig.add_trace(go.Bar(
            x=loss_streak_counts.index,
            y=loss_streak_counts.values,
            name='Loss Streaks',
            marker_color='red'
        ))
        
        # Update layout
        fig.update_layout(
            title="Streak Distribution",
            xaxis_title="Streak Length",
            yaxis_title="Frequency",
            barmode='group'
        )
        
        return html.Div([
            metrics_cards,
            dcc.Graph(figure=fig)
        ])
    
    @app.callback(
        Output('volatility-chart', 'figure'),
        Input('trade-data-store', 'data'),
        Input('filtered-data-store', 'data')
    )
    def update_volatility_chart(json_data, filtered_json_data):
        # Use filtered data if available, otherwise use all data
        data_to_use = filtered_json_data if filtered_json_data is not None else json_data
        
        if data_to_use is None:
            # Return empty figure
            empty_fig = go.Figure()
            empty_fig.update_layout(
                title="No data available for volatility analysis",
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
            return empty_fig
        
        # Convert JSON to DataFrame
        df = pd.read_json(data_to_use, orient='split')
        if df.empty: # Added check for empty df after potential filtering
             empty_fig = go.Figure()
             empty_fig.update_layout(
                 title="No data available for volatility analysis after filtering",
                 xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                 yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
             )
             return empty_fig

        # Ensure 'OpenTimestamp' is in datetime format
        df['OpenTimestamp'] = pd.to_datetime(df['OpenTimestamp'])
        
        # Group by date to get daily P&L
        df['Date'] = df['OpenTimestamp'].dt.date
        daily_pnl = df.groupby('Date')['NetP&L'].sum().reset_index()
        
        # Calculate volatility
        daily_pnl['Volatility'] = advanced_metrics.calculate_volatility(daily_pnl['NetP&L'])
        
        # Create figure
        fig = go.Figure()
        
        # Add daily P&L bars
        fig.add_trace(go.Bar(
            x=daily_pnl['Date'],
            y=daily_pnl['NetP&L'],
            name='Daily P&L',
            marker_color=np.where(daily_pnl['NetP&L'] >= 0, 'green', 'red')
        ))
        
        # Add volatility line
        fig.add_trace(go.Scatter(
            x=daily_pnl['Date'],
            y=daily_pnl['Volatility'],
            name='20-Day Volatility',
            line=dict(color='blue', width=2)
        ))
        
        # Update layout
        fig.update_layout(
            title="Daily P&L and Volatility",
            xaxis_title="Date",
            yaxis_title="Value ($)",
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    @app.callback(
        Output('cluster-analysis-chart', 'figure'),
        Input('trade-data-store', 'data'),
        Input('filtered-data-store', 'data'),
        Input('cluster-feature-selector', 'value')
    )
    def update_cluster_analysis(json_data, filtered_json_data, cluster_feature):
        # Use filtered data if available, otherwise use all data
        data_to_use = filtered_json_data if filtered_json_data is not None else json_data
        
        if data_to_use is None or cluster_feature is None:
            # Return empty figure
            empty_fig = go.Figure()
            empty_fig.update_layout(
                title="No data available for cluster analysis",
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
            return empty_fig
        
        # Convert JSON to DataFrame
        df = pd.read_json(data_to_use, orient='split')
        if df.empty: # Added check for empty df after potential filtering
            empty_fig = go.Figure()
            empty_fig.update_layout(
                title="No data available for cluster analysis after filtering",
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
            return empty_fig
            
        # Define features based on selection
        if cluster_feature == 'symbol_position':
            features = ['Symbol', 'PositionType']
        elif cluster_feature == 'symbol_exit':
            features = ['Symbol', 'SignalName_Exit']
        elif cluster_feature == 'exchange_product':
            features = ['Exchange', 'ProductType']
        else:
            features = ['Symbol']
        
        # Get cluster analysis
        cluster_data = advanced_metrics.analyze_trade_clusters(df, features)
        
        # Convert to DataFrame for plotting
        cluster_df = pd.DataFrame.from_dict(cluster_data, orient='index')

        # Check if DataFrame is empty or essential columns are missing
        if cluster_df.empty or 'total_pnl' not in cluster_df.columns:
            empty_fig = go.Figure()
            empty_fig.update_layout(
                title="No data available for cluster analysis after processing",
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
            return empty_fig
        
        # Sort by total P&L
        cluster_df = cluster_df.sort_values('total_pnl', ascending=False)
        
        # Limit to top 10 clusters
        cluster_df = cluster_df.head(10)
        
        # Create figure
        fig = go.Figure()
        
        # Add total P&L bars
        fig.add_trace(go.Bar(
            x=cluster_df.index,
            y=cluster_df['total_pnl'],
            name='Total P&L',
            marker_color='lightblue',
            text=[f"${x:.2f}" for x in cluster_df['total_pnl']],
            textposition='auto'
        ))
        
        # Add win rate as a line
        fig.add_trace(go.Scatter(
            x=cluster_df.index,
            y=cluster_df['win_rate'],
            name='Win Rate',
            mode='markers+lines',
            marker=dict(size=10, color='green'),
            yaxis='y2'
        ))
        
        # Update layout
        fig.update_layout(
            title=f"Top 10 Clusters by P&L ({', '.join(features)})",
            xaxis_title="Cluster",
            yaxis_title="Total P&L ($)",
            yaxis2=dict(
                title="Win Rate",
                tickfont=dict(color='green'),
                overlaying='y',
                side='right',
                range=[0, 1]
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
