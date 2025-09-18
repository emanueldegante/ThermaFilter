# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 12:52:27 2025

@author: emanu
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import dash
from dash import dcc, html, Input, Output, State
import matplotlib.pyplot as plt
import io
import base64
import smtplib
from email.message import EmailMessage
import dash_bootstrap_components as dbc
import os

# -------------------- USER AUTH --------------------
VALID_USERNAME_PASSWORD_PAIRS = {"UNB_User": "geophysics"}

# -------------------- Load Data --------------------
def load_deep_data(filepath: str, start_col: int = 52, end_col: int = 143) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df_ele = df.iloc[:, start_col:end_col]
    time = df.iloc[:, [0]]  # ensure it's a DataFrame
    df_shallow = pd.concat([time, df_ele], axis=1)
    return df_shallow

df_deep = load_deep_data("DTS_daily_Total.csv")

# -------------------- Encode Logos --------------------
def encode_image(image_file):
    with open(image_file, 'rb') as f:
        return base64.b64encode(f.read()).decode()

logo1_encoded = encode_image("NB_Power.png")
logo2_encoded = encode_image("UNB_Logo.png")

# -------------------- Dash App --------------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
latest_plot_png = None  # Store latest PNG for email or download

# -------------------- Layout --------------------
app.layout = dbc.Container([
    # Header with logos (always visible)
    dbc.Row([
        dbc.Col(html.Img(src=f'data:image/png;base64,{logo1_encoded}', style={'height':'80px'}), width="auto"),
        dbc.Col(html.H1("ThermaFilter - Temperature Gradient Profiles Dashboard",
                        style={'color':'white', 'textAlign':'center'}), width=True),
        dbc.Col(html.Img(src=f'data:image/png;base64,{logo2_encoded}', style={'height':'80px'}), width="auto"),
    ], align="center", className="mb-4"),

    # Login section (shown only if not logged in)
    html.Div(id="login-section", children=[
        dbc.Row([
            dbc.Col(html.H2("DTScope", style={'color':'white', 'textAlign':'center'})),
        ], justify='center'),
        dbc.Row([
            dbc.Col(dcc.Input(id="username", type="text", placeholder="Username", style={'width':'100%'}), width=6),
        ], justify='center', className="mb-2"),
        dbc.Row([
            dbc.Col(dcc.Input(id="password", type="password", placeholder="Password", style={'width':'100%'}), width=6),
        ], justify='center', className="mb-2"),
        dbc.Row([
            dbc.Col(html.Button("Login", id="login-btn", n_clicks=0, style={
                            'backgroundColor': 'blue',  # light blue
                            'color': 'white',
                            'fontSize': '16px',
                            'fontWeight': 'bold',
                            'borderRadius': '8px',
                            'border': 'none',
                            'padding': '10px 20px',
                            'width': '100%',
                            'cursor': 'pointer'
                        }), width=6),
        ], justify='center'),
        html.Div(id="login-message", style={'color':'red', 'textAlign':'center', 'marginTop':'10px'})
    ], style={'marginTop':'50px'}),

    # Dashboard content (hidden until login)
    html.Div(id="dashboard-content", style={"display": "none"}, children=[
        # Controls
        dbc.Row([
            dbc.Col([
                html.Label("Select Year", style={'color': 'white'}),
                dcc.Dropdown(
                    id="year-dropdown",
                    options=[{"label": y, "value": y} for y in sorted(df_deep.iloc[:, 0].apply(lambda d: int(str(d)[:4])).unique())],
                    value=int(str(df_deep.iloc[0, 0])[:4])
                ),
                html.Br(),

                html.Label("Select Month", style={'color': 'white'}),
                dcc.Dropdown(
                    id="month-dropdown",
                    options=[{"label": m, "value": m} for m in range(1, 13)],
                    value=1
                ),
                html.Br(),

                html.Label("Select Day", style={'color': 'white'}),
                dcc.Dropdown(
                    id="day-dropdown",
                    options=[{"label": d, "value": d} for d in range(1, 32)],
                    value=1
                ),
                html.Br(),

                html.Label("Total Days to Analyze:", style={'color': 'white'}),
                dcc.Input(id='days-input', type='number', value=10, style={'width': '100%'}),
                html.Br(), html.Br(),

                html.Label("Step Size (days):", style={'color': 'white'}),
                dcc.Input(id='step-input', type='number', value=1, style={'width': '100%'}),
                html.Br(), html.Br(),

                html.Label("Offset (decimal allowed):", style={'color': 'white'}),
                dcc.Input(id='offset-input', type='number', value=0.0, step=0.1, style={'width': '100%'}),
                html.Br(), html.Br(),

                html.Label("Apply Median Filter:", style={'color': 'white'}),
                dcc.Checklist(
                    id='median-filter',
                    options=[{'label': 'Apply Median Filter', 'value': 'Y'}],
                    value=[],
                    inputStyle={"margin-right": "10px", "margin-left": "5px"}
                ),
                html.Br(),

                html.Button("Generate Plot", id='generate-btn', n_clicks=0,
                            style={'backgroundColor': 'blue', 'color': 'white', 'fontSize': '16px',
                                   'fontWeight': 'bold', 'borderRadius': '8px', 'border': 'none',
                                   'padding': '10px 20px', 'width': '100%', 'cursor': 'pointer'}),
            ], width=4),
        ]),

        html.Div(id='output-plot', style={'marginTop': 30}),
        html.Hr(),

        html.Label("Recipient Email:", style={'color': 'white'}),
        dcc.Input(id='email-input', type='email', placeholder="Enter recipient email",
                  style={'width': '100%', 'marginTop': '5px'}),
        html.Br(), html.Br(),

        html.Button("Send Email with Plot", id="send-email-btn", n_clicks=0,
                    style={'backgroundColor': 'blue', 'color': 'white', 'width': '100%'}),
        html.Div(id="email-status", style={'marginTop': '10px', 'color': 'white'})
    ])
], fluid=True, style={'backgroundColor': 'orange', 'padding': '20px', 'minHeight': '100vh'})

# -------------------- LOGIN CALLBACK --------------------
@app.callback(
    Output("login-section", "style"),
    Output("dashboard-content", "style"),
    Output("login-message", "children"),
    Input("login-btn", "n_clicks"),
    State("username", "value"),
    State("password", "value")
)
def login(n_clicks, username, password):
    if n_clicks > 0:
        if username in VALID_USERNAME_PASSWORD_PAIRS and VALID_USERNAME_PASSWORD_PAIRS[username] == password:
            # Hide login section, show dashboard
            return {"display": "none"}, {"display": "block"}, ""
        else:
            # Keep login visible, hide dashboard, show error
            return {"display": "block"}, {"display": "none"}, "Invalid username or password"
    # Initial state
    return {"display": "block"}, {"display": "none"}, ""

# -------------------- Plot Generation Callback --------------------
@app.callback(
    Output('output-plot', 'children'),
    Input('generate-btn', 'n_clicks'),
    State('year-dropdown', 'value'),
    State('month-dropdown', 'value'),
    State('day-dropdown', 'value'),
    State('days-input', 'value'),
    State('step-input', 'value'),
    State('offset-input', 'value'),
    State('median-filter', 'value')
)
def generate_plot(n_clicks, year, month, day, total_days, step_size, offset, median_filter):
    global latest_plot_png
    if n_clicks > 0:
        try:
            date_1 = f"{year}-{month:02d}-{day:02d}"
            offset = float(offset)
            df = df_deep
            df_ele = df.iloc[:,1:]
            elevations = np.array(df_ele.columns.tolist(), dtype=np.float64)
            temp_ave = pd.DataFrame(elevations)
            dates_final = []

            for i in range(0, total_days+1, step_size):
                date_3 = (datetime.strptime(date_1, "%Y-%m-%d") + timedelta(days=i)).strftime("%Y-%m-%d")
                norm_1 = df[df.iloc[:,0]==date_3].iloc[:,1:]
                if norm_1.empty: continue
                norm_1 = norm_1.transpose().to_numpy().reshape(-1)
                t_grad= (np.diff(norm_1))/.146
                temp_grad = pd.DataFrame({i: t_grad})
                temp_ave = pd.concat([temp_ave, temp_grad], axis=1)
                dates_final.append(date_3)

            apply_median = 'Y' in median_filter
            kl = temp_ave.iloc[:,1:].to_numpy()
            if apply_median:
                med_fil = np.median(kl, axis=1)
            else:
                med_fil = np.zeros(kl.shape[0])
            kl = kl - med_fil[:, None]

            a = kl[:,0].reshape(-1,1)
            jj = offset
            for i in range(kl.shape[1]):
                kl_div = kl[:,i] + offset
                a = np.concatenate([a, kl_div.reshape(-1,1)], axis=1)
                offset += jj

            y_slan_min = 50.0684 - 1.19585 * elevations[-1]
            y_slan_max = 50.0684 - 1.19585 * elevations[0]

            fig = plt.figure(figsize=(20,8), dpi=160, facecolor='white')
            ax1 = plt.subplot(111, facecolor='white')
            ax2 = ax1.twinx()
            ax2.set_ylabel(' Slant Depth (m)', fontsize=20)
            ax1.set_ylabel(' Elevation (m)', fontsize=20)
            ax2.set_ylim(elevations[-1],elevations[0])
            ax2.yaxis.set_tick_params(labelsize=20)
            ax1.yaxis.set_tick_params(labelsize=20)
            ax1.xaxis.set_tick_params(labelsize=20)
            ax1.set_ylim(elevations[-1], elevations[0])
            ax2.set_ylim(y_slan_min,y_slan_max)

            ax1.plot(a, elevations, color='black')
            ax1.set_xlabel('Temperature gradient profiles with offset ⁰C/m', fontsize=20)
            plt.title(f'Temperature gradient profiles\nInitial date {date_1} Final Date {dates_final[-1]}', fontsize=20)

            buf = io.BytesIO()
            plt.savefig(buf, format='png', facecolor=fig.get_facecolor())
            buf.seek(0)
            encoded_image = base64.b64encode(buf.read()).decode('utf-8')
            buf.seek(0)
            latest_plot_png = buf.read()
            plt.close(fig)

            return html.Div([
                html.P(f"Plot generated.", style={'color':'white'}),
                html.Img(src=f'data:image/png;base64,{encoded_image}', style={'width':'90%', 'height':'auto'})
            ])
        except Exception as e:
            return html.Div([html.P(f"Error: {str(e)}", style={'color':'white'})])
    return ""

# -------------------- Email Callback --------------------
@app.callback(
    Output("email-status", "children"),
    Input("send-email-btn", "n_clicks"),
    State("email-input", "value"),
    prevent_initial_call=True
)
def send_email(n_clicks, recipient):
    global latest_plot_png
    if not recipient:
        return "❌ Please enter a recipient email."
    if latest_plot_png is None:
        return "⚠️ No plot generated yet. Please generate a plot first."

    sender = "your_email@example.com"
    password = "YOUR_APP_PASSWORD"

    msg = EmailMessage()
    msg["Subject"] = "Temperature Gradient Plot"
    msg["From"] = sender
    msg["To"] = recipient
    msg.set_content("Attached is the requested temperature gradient plot.")
    msg.add_attachment(latest_plot_png, maintype="image", subtype="png", filename="temperature_plot.png")

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender, password)
            server.send_message(msg)
        return f"✅ Email sent successfully to {recipient}"
    except Exception as e:
        return f"❌ Failed to send email: {e}"

# -------------------- Run App --------------------
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8050))  # Render sets the PORT environment variable
    app.run_server(host='0.0.0.0', port=port)
