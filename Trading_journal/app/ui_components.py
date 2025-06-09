import dash_bootstrap_components as dbc
from dash import html, dcc

# Content for Upload Data Page
upload_data_page_content = html.Div([
    html.H3("Data Management", className="mb-4"),
    dbc.Card([ # Wrapping in a card for better visual grouping
        dbc.CardBody([
            dcc.Upload(
                id='upload-data-page-component', # Changed ID to avoid conflict
                children=html.Div(['Drag and Drop or ', html.A('Select Trade CSV File')]), # Slightly changed text
                style={
                    'width': '100%', 'height': '60px', 'lineHeight': '60px',
                    'borderWidth': '1px', 'borderStyle': 'dashed',
                    'borderRadius': '5px', 'textAlign': 'center', 'marginBottom': '20px' # Adjusted margin
                },
                multiple=False
            ),
            html.Div(id='upload-page-status', className="mb-3"), # Status for this upload component
            dbc.Button("Sync Data", id="sync-data-button", color="primary", className="mt-2"), # Adjusted margin
            html.Div(id='upload-sync-status', className="mt-3") # Status for sync button
        ])
    ])
])
