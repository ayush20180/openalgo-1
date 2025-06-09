"""
Trading Journal Application - Main Application
--------------------------------------------
This is the main Dash application file that defines the layout and callbacks
for the trading journal application.
"""

import dash
from dash import dcc, html, dash_table, callback, Input, Output, State, no_update
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import base64
import io
import json
from io import StringIO
import plotly.io as pio
import os 
from datetime import datetime

# Import utility modules
from app.utils.data_loader import load_trade_csv, preprocess_data, load_data_csv, save_data_csv
from app.utils import metrics_calculator, advanced_metrics, advanced_dashboard, global_filters, journal_management, calendar_view
from app.utils import wmy_analysis
from app.ui_components import upload_data_page_content # Import for the new page

# Initialize the Dash app with Bootstrap theme and Font Awesome icons
app = dash.Dash(
    __name__, 
    external_stylesheets=[dbc.themes.DARKLY, dbc.icons.FONT_AWESOME],
    suppress_callback_exceptions=True
)
pio.templates.default = "plotly_dark"

# --- Page Content Definitions ---
overview_section_content = html.Div([
    dbc.Card([
        dbc.CardHeader(html.H4("Overall Performance Metrics", className="card-title")),
        dbc.CardBody([html.Div(id='overall-metrics-display')])
    ], className="mb-4"),
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader(html.H4("Equity Curve", className="card-title")),
            dbc.CardBody([dcc.Graph(id='overall-equity-curve', style={'height': '400px', 'width': '100%'})])
        ], className="mb-4"), width=12, lg=6),
        dbc.Col(dbc.Card([
            dbc.CardHeader(html.H4("P&L Distribution", className="card-title")),
            dbc.CardBody([dcc.Graph(id='pnl-distribution-histogram', style={'height': '400px', 'width': '100%'})])
        ], className="mb-4"), width=12, lg=6)
    ]),
    dbc.Card([
        dbc.CardHeader(html.H4("P&L by Period", className="card-title")),
        dbc.CardBody([
            dbc.Row([dbc.Col([dcc.RadioItems(id='pnl-period-selector', options=[{'label': 'Daily', 'value': 'D'},{'label': 'Weekly', 'value': 'W'},{'label': 'Monthly', 'value': 'M'}], value='D', inline=True, className="mb-2")])]),
            dcc.Graph(id='pnl-by-period-chart', style={'height': '400px', 'width': '100%'})
        ])
    ], className="mb-4"),
])

algo_analysis_section_content = html.Div([
    dbc.Card([
        dbc.CardHeader(html.H4("Algorithm Selection & Metrics", className="card-title")),
        dbc.CardBody([
            dcc.Dropdown(id='algorithm-selector', placeholder="Select Algorithm ID", className="mb-3"),
            html.Div(id='algo-metrics-display')
        ])
    ], className="mb-4"),
    dbc.Card([
        dbc.CardHeader(html.H4("Algorithm Equity Curve", className="card-title")),
        dbc.CardBody([dcc.Graph(id='algo-equity-curve', style={'height': '400px', 'width': '100%'})])
    ], className="mb-4"),
])

advanced_analytics_section_content = html.Div([
    dbc.Card([
        dbc.CardHeader(html.H4("Advanced Analytics", className="card-title")),
        dbc.CardBody([advanced_dashboard.create_advanced_metrics_layout()])
    ], className="mb-4"),
])

trade_details_section_content = html.Div([
    dbc.Card([
        dbc.CardHeader(html.H4("Trade Details", className="card-title")),
        dbc.CardBody([
            dash_table.DataTable(
                id='basic-trade-table', page_size=10, style_table={'overflowX': 'auto', 'minWidth': '100%'},
                style_cell={'height': 'auto', 'minWidth': '80px', 'width': '100px', 'maxWidth': '180px', 'whiteSpace': 'normal'},
                style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': 'rgb(248, 248, 248)'}]
            )
        ])
    ], className="mb-4"),
])

journal_management_section_content = html.Div([
    dbc.Card([
        dbc.CardHeader(html.H4("Journal Management", className="card-title")),
        dbc.CardBody([journal_management.create_journal_entry_layout()])
    ], className="mb-4"),
])

# --- App Layout ---
app.layout = dbc.Container(id='app-container', children=[
    dcc.Location(id='url', refresh=False), # dcc.Location needs to be part of the layout for callbacks reacting to pathname
    dcc.Store(id='trade-data-store'),
    dcc.Store(id='filtered-data-store'),
    dcc.Store(id='sidebar-state-store', data=0),
    dcc.Store(id='stored-pnl-range'), 

    html.Div([ 
        dbc.Row([
            dbc.Col(html.H4("Navigation", className="mb-0 mt-1"), width='auto'),
            dbc.Col(dbc.Button(html.I(className="fas fa-bars"), id="sidebar-toggle-button", n_clicks=0, className="ms-auto border-0", style={'backgroundColor': 'transparent', 'fontSize': '1.5rem'}), width='auto', className="d-flex justify-content-end")
        ], align="center", className="mb-2"),
        html.Hr(className="mt-1 mb-3"),
        dbc.Nav([
            dbc.NavLink([html.I(className="fas fa-home me-2"), "Overview"], href="/overview", active="exact", id="nav-overview"),
            dbc.NavLink([html.I(className="fas fa-cogs me-2"), "Algorithm Analysis"], href="/algo-analysis", active="exact", id="nav-algo-analysis"),
            dbc.NavLink([html.I(className="fas fa-chart-pie me-2"), "Advanced Analytics"], href="/advanced-analytics", active="exact", id="nav-advanced-analytics"),
            dbc.NavLink([html.I(className="fas fa-table me-2"), "Trade Details"], href="/trade-details", active="exact", id="nav-trade-details"),
            dbc.NavLink([html.I(className="fas fa-book me-2"), "Journal Management"], href="/journal-management", active="exact", id="nav-journal-management"),
            dbc.NavLink([html.I(className="fas fa-calendar-alt me-2"), "Calendar View"], href="/calendar-view", active="exact", id="nav-calendar-view"),
            dbc.NavLink([html.I(className="fas fa-calendar-week me-2"), "WMY Analysis"], href="/wmy-analysis", active="exact", id="nav-wmy-analysis"),
            dbc.NavLink([html.I(className="fas fa-upload me-2"), "Upload Data"], href="/upload-data", active="exact", id="nav-upload-data"),
        ], vertical=True, pills=True, id="sidebar-nav"),
        html.Hr(className="mt-3"),
        dbc.Accordion([dbc.AccordionItem(global_filters.create_global_filters_layout(), title="Global Filters", item_id="global-filters-accordion")], start_collapsed=True, flush=True, id="sidebar-accordion-filters", className="mt-3")
    ], id="sidebar", className="sidebar-active p-3"),

    html.Div([ 
        dbc.Row([dbc.Col(html.H1("Algorithmic Trading Dashboard", className="my-4"),width=True)], align="center", className="mb-4 header-row"),
        html.Div(id='dynamic-content-area', style={'display': 'none'}, children=[html.Div(id='page-content')])
    ], id="page-content-wrapper", className="content-shifted"),

    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle(id='daily-trades-modal-title')),
        dbc.ModalBody(html.Div(id='daily-trades-content')),
        dbc.ModalFooter(dbc.Button("Close", id="close-daily-trades-modal", className="ms-auto", n_clicks=0))
    ], id='daily-trades-modal', size="xl", is_open=False),
    
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Select P&L Range")),
        dbc.ModalBody([
            dcc.RangeSlider(id='modal-pnl-range-slider', min=-1000, max=1000, step=100, value=[-1000, 1000], tooltip={'placement': 'bottom', 'always_visible': True}),
            html.Div(id='modal-pnl-slider-container-helper', style={'padding': '10px 0'}) 
        ]),
        dbc.ModalFooter([
            dbc.Button("Cancel", id="cancel-pnl-range-button", color="secondary", outline=True),
            dbc.Button("Apply", id="apply-pnl-range-button", color="primary", className="ms-2")
        ])
    ], id='pnl-range-modal', is_open=False, size="lg", centered=True),
], fluid=True)

# --- Callbacks ---

@callback(
    Output('trade-data-store', 'data', allow_duplicate=True),
    Output('upload-sync-status', 'children'), 
    Input('upload-data-page-component', 'contents'), 
    State('upload-data-page-component', 'filename'), 
    prevent_initial_call=True
)
def update_output(contents, filename):
    if contents is None: return no_update, None 
    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string).decode('utf-8')
        df = load_trade_csv(decoded)
        processed_df = preprocess_data(df)
        processed_df['Trade type'] = "Uploaded" 
        processed_df['AddedTimestamp'] = pd.to_datetime(datetime.now())
        existing_df = load_data_csv()
        combined_df = pd.concat([existing_df, processed_df], ignore_index=True)
        combined_df['AddedTimestamp'] = pd.to_datetime(combined_df['AddedTimestamp'], errors='coerce')
        combined_df = combined_df.sort_values(by='AddedTimestamp', ascending=False, na_position='last')
        final_df = combined_df.drop_duplicates(subset=['TradeID'], keep='first')
        save_data_csv(final_df)
        updated_store_data = final_df.to_json(date_format='iso', orient='split') if not final_df.empty else None
        return updated_store_data, dbc.Alert(f"Successfully loaded and processed {filename}.", color="success", dismissable=True)
    except Exception as e:
        return no_update, dbc.Alert(f"Error processing file: {str(e)}", color="danger", dismissable=True)

@callback(
    Output('trade-data-store', 'data', allow_duplicate=True),
    Output('upload-sync-status', 'children', allow_duplicate=True), 
    Input('sync-data-button', 'n_clicks'),
    prevent_initial_call=True
)
def sync_data_from_file(n_clicks):
    if n_clicks is None: return no_update, dbc.Alert("Sync not triggered.", color="info", dismissable=True)
    try:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        sync_csv_path = os.path.join(project_root, 'strategy_live', 'trading_journal.csv')
        if not os.path.exists(sync_csv_path): return no_update, dbc.Alert("Error: trading_journal.csv not found.", color="danger", dismissable=True)
        synced_df = pd.read_csv(sync_csv_path, quoting=1)
        if synced_df.empty: return no_update, dbc.Alert("trading_journal.csv is empty.", color="warning", dismissable=True)
        processed_synced_df = preprocess_data(synced_df)
        processed_synced_df['Trade type'] = "Synced"
        processed_synced_df['AddedTimestamp'] = pd.to_datetime(datetime.now())
        existing_df = load_data_csv()
        combined_df = pd.concat([existing_df, processed_synced_df], ignore_index=True)
        combined_df['AddedTimestamp'] = pd.to_datetime(combined_df['AddedTimestamp'], errors='coerce')
        combined_df = combined_df.sort_values(by='AddedTimestamp', ascending=False, na_position='last')
        final_df = combined_df.drop_duplicates(subset=['TradeID'], keep='first')
        save_data_csv(final_df)
        updated_store_data = final_df.to_json(date_format='iso', orient='split') if not final_df.empty else None
        return updated_store_data, dbc.Alert(f"Synced {len(processed_synced_df)} trades. Total: {len(final_df)}.", color="success", dismissable=True)
    except ValueError as ve: return no_update, dbc.Alert(f"Error processing synced data: {str(ve)}", color="danger", dismissable=True)
    except Exception as e:
        print(f"Error during sync: {e}"); traceback.print_exc()
        return no_update, dbc.Alert(f"Unexpected sync error: {str(e)}", color="danger", dismissable=True)

@app.callback(
    Output('trade-data-store', 'data', allow_duplicate=True),
    Input('url', 'pathname'), # Initial data load based on accessing the app
    prevent_initial_call='initial_duplicate' 
)
def load_initial_data(pathname): # Pathname is an input, but not directly used to decide *what* to load, just *when*
    master_df = load_data_csv()
    if master_df is not None and not master_df.empty:
        return master_df.to_json(date_format='iso', orient='split')
    return None

# MODIFIED CALLBACK:
@app.callback(
    Output('dynamic-content-area', 'style'),
    Input('trade-data-store', 'data'),
    Input('url', 'pathname') # Added pathname
)
def toggle_dynamic_content_visibility(trade_data_json, pathname):
    if pathname == '/upload-data': # Always show content area for upload page
        return {'display': 'block'}
    
    if trade_data_json is not None:
        try:
            df = pd.read_json(io.StringIO(trade_data_json), orient='split')
            if not df.empty:
                return {'display': 'block'}
        except ValueError: pass
    return {'display': 'none'}

@callback(
    Output('overall-metrics-display', 'children'), Output('overall-equity-curve', 'figure'), Output('pnl-distribution-histogram', 'figure'),
    Input('filtered-data-store', 'data'), State('trade-data-store', 'data'), Input('url', 'pathname')
)
def update_overall_metrics(filtered_json_data, raw_json_data, pathname):
    if pathname != '/overview':
        empty_fig = go.Figure().update_layout(title="No data available", xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
        return None, empty_fig, empty_fig
    json_data_to_use = filtered_json_data if filtered_json_data else raw_json_data
    if not json_data_to_use:
        empty_fig = go.Figure().update_layout(title="No data available", xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
        return None, empty_fig, empty_fig
    df = pd.read_json(StringIO(json_data_to_use), orient='split')
    stats = metrics_calculator.calculate_summary_stats(df)
    df_with_cum_pnl = metrics_calculator.calculate_cumulative_pnl(df)
    metrics_cards = dbc.Row([
        dbc.Col(dbc.Card([dbc.CardHeader("Total P&L"), dbc.CardBody(html.H4(f"${stats['total_pnl']:.2f}"))]), width=12, sm=6, md=4, lg=3, className="mb-3"),
        dbc.Col(dbc.Card([dbc.CardHeader("Total Trades"), dbc.CardBody(html.H4(f"{stats['total_trades']}"))]), width=12, sm=6, md=4, lg=3, className="mb-3"),
        dbc.Col(dbc.Card([dbc.CardHeader("Win Rate"), dbc.CardBody(html.H4(f"{stats['win_rate']:.2%}"))]), width=12, sm=6, md=4, lg=3, className="mb-3"),
        dbc.Col(dbc.Card([dbc.CardHeader("Profit Factor"), dbc.CardBody(html.H4(f"{stats['profit_factor']:.2f}"))]), width=12, sm=6, md=4, lg=3, className="mb-3")
    ])
    equity_fig = px.line(df_with_cum_pnl, x='OpenTimestamp', y='CumulativeP&L', title='Equity Curve').update_layout(xaxis_title="Date", yaxis_title="Cumulative P&L ($)", hovermode="x unified")
    hist_fig = px.histogram(df, x='NetP&L', nbins=30, title='P&L Distribution').update_layout(xaxis_title="Net P&L ($)", yaxis_title="Number of Trades", bargap=0.1)
    return metrics_cards, equity_fig, hist_fig

@app.callback(
    Output('pnl-by-period-chart', 'figure'),
    Input('filtered-data-store', 'data'), State('trade-data-store', 'data'), Input('pnl-period-selector', 'value')
)
def update_pnl_by_period(filtered_json_data, raw_json_data, period):
    json_data_to_use = filtered_json_data if filtered_json_data else raw_json_data
    if not json_data_to_use: return go.Figure().update_layout(title="No data available")
    df = pd.read_json(StringIO(json_data_to_use), orient='split')
    if 'OpenTimestamp' not in df.columns: return go.Figure().update_layout(title="'OpenTimestamp' column missing")
    df['OpenTimestamp'] = pd.to_datetime(df['OpenTimestamp'], errors='coerce')
    period_label = "Daily" if period == 'D' else "Weekly" if period == 'W' else "Monthly"
    df['Period'] = df['OpenTimestamp'].dt.to_period(period)
    period_pnl = df.groupby('Period')['NetP&L'].sum().reset_index(); period_pnl['Period'] = period_pnl['Period'].astype(str)
    fig = px.bar(period_pnl, x='Period', y='NetP&L', title=f'{period_label} P&L', color='NetP&L', color_continuous_scale=['red', 'green'], color_continuous_midpoint=0)
    fig.update_layout(xaxis_title=f"{period_label} Period", yaxis_title="Net P&L ($)", coloraxis_showscale=False); return fig

@app.callback(
    Output('algorithm-selector', 'options'),
    Input('filtered-data-store', 'data'), State('trade-data-store', 'data'), Input('url', 'pathname')
)
def set_algorithm_options(filtered_json_data, raw_json_data, pathname):
    if pathname != '/algo-analysis': return []
    json_data_to_use = filtered_json_data if filtered_json_data else raw_json_data
    if not json_data_to_use: return []
    try:
        df = pd.read_json(StringIO(json_data_to_use), orient='split')
        return [{'label': algo, 'value': algo} for algo in sorted(df['AlgorithmID'].unique())] if 'AlgorithmID' in df.columns else []
    except Exception as e: print(f"Error setting options: {e}"); return []

@app.callback(
    Output('algo-metrics-display', 'children'), Output('algo-equity-curve', 'figure'),
    Input('filtered-data-store', 'data'), State('trade-data-store', 'data'), Input('algorithm-selector', 'value'),
    prevent_initial_call=True
)
def update_algorithm_analysis(filtered_json_data, raw_json_data, selected_algo):
    json_data_to_use = filtered_json_data if filtered_json_data else raw_json_data
    empty_fig = go.Figure().update_layout(title="No data/selection", xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
    if not json_data_to_use or not selected_algo: return None, empty_fig
    df = pd.read_json(StringIO(json_data_to_use), orient='split'); df['OpenTimestamp'] = pd.to_datetime(df['OpenTimestamp'], errors='coerce')
    algo_df = df[df['AlgorithmID'] == selected_algo]
    if algo_df.empty: return None, empty_fig
    stats = metrics_calculator.calculate_summary_stats(algo_df)
    algo_df_with_cum_pnl = metrics_calculator.calculate_cumulative_pnl(algo_df)
    metrics_cards = dbc.Row([
        dbc.Col(dbc.Card([dbc.CardHeader("Algo P&L"), dbc.CardBody(html.H4(f"${stats['total_pnl']:.2f}"))]), width=12, sm=6, md=4,lg=3, className="mb-3"),
        dbc.Col(dbc.Card([dbc.CardHeader("Algo Trades"), dbc.CardBody(html.H4(f"{stats['total_trades']}"))]), width=12, sm=6,md=4, lg=3, className="mb-3"),
        dbc.Col(dbc.Card([dbc.CardHeader("Algo Win Rate"), dbc.CardBody(html.H4(f"{stats['win_rate']:.2%}"))]), width=12, sm=6,md=4, lg=3, className="mb-3"),
        dbc.Col(dbc.Card([dbc.CardHeader("Algo Profit Factor"), dbc.CardBody(html.H4(f"{stats['profit_factor']:.2f}"))]),width=12, sm=6, md=4, lg=3, className="mb-3")
    ])
    equity_fig = px.line(algo_df_with_cum_pnl, x='OpenTimestamp', y='CumulativeP&L', title=f'Algo {selected_algo} Equity').update_layout(xaxis_title="Date", yaxis_title="Cum P&L ($)", hovermode="x unified")
    return metrics_cards, equity_fig

@app.callback(
    Output('basic-trade-table', 'data'), Output('basic-trade-table', 'columns'),
    Output('basic-trade-table', 'style_header'), Output('basic-trade-table', 'style_cell'),
    Output('basic-trade-table', 'style_data_conditional'),
    Input('filtered-data-store', 'data'), State('trade-data-store', 'data')
)
def update_trade_table(filtered_json_data, raw_json_data):
    data_to_use = filtered_json_data if filtered_json_data else raw_json_data; empty_table_response = [], [], {}, {}, []
    if not data_to_use: return empty_table_response
    df = pd.read_json(StringIO(data_to_use), orient='split')
    if df.empty: return empty_table_response
    df['OpenTimestamp'] = pd.to_datetime(df['OpenTimestamp'], errors='coerce'); df['CloseTimestamp'] = pd.to_datetime(df['CloseTimestamp'], errors='coerce')
    display_cols_src = ['TradeID','OpenTimestamp','CloseTimestamp','Symbol','PositionType','EntryPrice','ExitPrice','Quantity','NetP&L']
    display_cols = [col for col in display_cols_src if col in df.columns]
    num_cols = ['EntryPrice','ExitPrice','Quantity','NetP&L']
    for col in num_cols:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
    df_disp = df.copy()
    if 'OpenTimestamp' in df_disp.columns: df_disp['OpenTimestamp'] = df_disp['OpenTimestamp'].dt.strftime('%Y-%m-%d %H:%M')
    if 'CloseTimestamp' in df_disp.columns: df_disp['CloseTimestamp'] = df_disp['CloseTimestamp'].dt.strftime('%Y-%m-%d %H:%M')
    tbl_data = df_disp[display_cols].to_dict('records')
    tbl_cols = [{"name": c, "id": c, "type": ("numeric" if c in num_cols else "any"), "format": (dash_table.FormatTemplate.money(2) if c == 'NetP&L' else None)} for c in display_cols]
    style_header={'backgroundColor':'#343a40','color':'#f0f0f0','fontWeight':'bold','border':'1px solid #444'}
    style_cell={'height':'auto','minWidth':'80px','width':'100px','maxWidth':'180px','whiteSpace':'normal','padding':'8px','textAlign':'left','backgroundColor':'#23232A','color':'#f0f0f0','border':'1px solid #444'}
    style_data_cond=[
        {'if':{'row_index':'odd'},'backgroundColor':'#2E2E36'},{'if':{'row_index':'even'},'backgroundColor':'#23232A'},
        {'if':{'filter_query':'{NetP&L}>0'},'backgroundColor':'rgba(40,167,69,0.3)','color':'#f0f0f0'},
        {'if':{'filter_query':'{NetP&L}<0'},'backgroundColor':'rgba(220,53,69,0.3)','color':'#f0f0f0'},
        {'if':{'filter_query':'{NetP&L}=0'},'backgroundColor':'#3a3a40','color':'#f0f0f0'},
        {'if':{'column_type':'numeric'},'textAlign':'right'},
        {'if':{'state':'active'},'backgroundColor':'rgba(0,123,255,0.15)','border':'1px solid #007bff'}
    ]
    return tbl_data, tbl_cols, style_header, style_cell, style_data_cond

SIDEBAR_STATES = [
    {'name':'EXPANDED','width':'430px','marginLeft':'430px','sidebar_class':'sidebar-active p-3 sidebar-expanded','content_class':'content-shifted'},
    {'name':'ICON_ONLY','width':'80px','marginLeft':'80px','sidebar_class':'sidebar-active p-3 sidebar-icon-only','content_class':'content-shifted-narrow'}
]
@app.callback(
    [Output('sidebar','className'),Output('page-content-wrapper','className'),Output('page-content-wrapper','style'),Output('sidebar','style'),Output('sidebar-state-store','data')],
    [Input('sidebar-toggle-button','n_clicks')],[State('sidebar-state-store','data')]
)
def toggle_sidebar_visibility(n_clicks, current_state_index):
    idx = 0 if (n_clicks is None or n_clicks == 0) else (1 if current_state_index == 0 else 0)
    s = SIDEBAR_STATES[idx]
    return s['sidebar_class'],s['content_class'],{'marginLeft':s['marginLeft']},{'width':s['width']},idx

@app.callback(Output('page-content', 'children'), Input('url', 'pathname'))
def display_page(pathname):
    if pathname in ['/overview', '/', None]: return overview_section_content
    elif pathname == '/algo-analysis': return algo_analysis_section_content
    elif pathname == '/advanced-analytics': return advanced_analytics_section_content
    elif pathname == '/trade-details': return trade_details_section_content
    elif pathname == '/journal-management': return journal_management_section_content
    elif pathname == '/calendar-view': return calendar_view.create_calendar_layout()
    elif pathname == '/wmy-analysis': return wmy_analysis.create_wmy_layout()
    elif pathname == '/upload-data': return upload_data_page_content
    return overview_section_content

advanced_dashboard.register_advanced_callbacks(app)
global_filters.register_filter_callbacks(app)
journal_management.register_journal_callbacks(app)
calendar_view.register_calendar_callbacks(app)
wmy_analysis.register_wmy_callbacks(app)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8051)
