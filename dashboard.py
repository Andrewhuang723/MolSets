import dash
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_table
import dash_core_components as dcc
from dash import dash_table, dcc, html, Input, Output, State
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import pickle
import threading
import torch
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from scipy.optimize import LinearConstraint, NonlinearConstraint, differential_evolution
from data_utils import RevIndexedDataset
from dmpnn import MolSets_DMPNN
from prepared import predict, MIXTURE, create_mixtures, objective_function, constraint_eq, iteration_callback

external_stylesheets = [dbc.themes.BOOTSTRAP, "assets/object_properties_style.css"]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server


## Standardization from "training dataset"
dataset = RevIndexedDataset('./data/data_list_v2.pkl')
target_mean = dataset.target_mean
target_std = dataset.target_std

## Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load("./results/DMPNN_6_h256_e256_reg0.0001_norm.pt")
best_model = checkpoint["best_model"]
hyperpars = checkpoint["hyperparameters"]
model = MolSets_DMPNN(n_node_features=133, hidden_dim=hyperpars['hidden_dim'], emb_dim=hyperpars['emb_dim'], 
                    output_dim=1, n_conv_layers=hyperpars['n_conv_layers'], after_readout=hyperpars['after_readout']).to(device)

model.load_state_dict(best_model)

def plot_contour(x, y, z, title, xname, yname):
    fig = go.Figure(data=go.Contour(z=z ,x=x, y=y, 
                                    colorscale='Magma', 
                                    contours=dict(size=0.01, 
                                                showlabels=True),
                                    colorbar=dict(
                                                title="Conductivity (mS/cm)",                # Title of the color bar
                                                titleside="top",                # Position title on the right
                                                titlefont=dict(size=14, color="black",family="Arial, sans-serif", )  # Title font properties
                                            )
                                    )
                    )
    fig.update_layout(
        title=title,
        xaxis=dict(
            title=xname,
            title_font=dict(
                size=30,                    # Font size for Y-axis label
                family="Arial, sans-serif",
                color="black"
            ),
            tickfont=dict(
                family="Arial, sans-serif",
                size=30,
                color="black"
            )
        ),
        yaxis=dict(
            title=yname,
            title_font=dict(
                size=30,                    # Font size for Y-axis label
                family="Arial, sans-serif",
                color="black"
            ),
            tickfont=dict(
                family="Arial, sans-serif",
                size=30,
                color="black"
            )
        ),
    )
    return fig


with open("space.md", "r") as f:
    howto_md = f.read()


modal_overlay = dbc.Modal(
    [
        dbc.ModalBody(html.Div([dcc.Markdown(howto_md)], id="howto-md")),
        dbc.ModalFooter(dbc.Button("Close", id="howto-close", className="howto-bn")),
    ],
    id="modal",
    size="lg",
)

# Define Header Layout
header = dbc.Navbar(
    dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        html.A(
                            html.Img(
                                src="assets/Foxconn.png",
                                height="30px",
                            ),
                            href="https://www.foxconn.com/",
                        )
                    ),
                    dbc.Col(dbc.NavbarBrand("An Electrolyte Conductivity Prediction & Recipe Screening App by 高階分析實驗室")),
                    modal_overlay,
                ],
                align="center",
            ),
        ],
        fluid=True,
    ),
    color="dark",
    dark=True,
)

data = {
    "Name": ["EC", "PC", "EMC", "DMC", "DEC", "FEC", "Salt (mol/kg)"],
    "Value": ["1.00", "0.00", "0.00", "0.00", "0.00", "0.00", "1.14"],
}
df = pd.DataFrame(data)


predict_card = dbc.Card(
    [
        dbc.CardHeader(html.H2("Ion Conductivity Prediction", style={'color': "black"})),
        dbc.CardBody([
            dbc.Row([
                html.H4("Salt", style={"color": "black"}),
                dcc.Dropdown(
                    id="prediction-salt-dropdown",
                    options=[
                        {"label": "LiPF6", "value": "LiPF6"},
                        {"label": "LiBOB", "value": "LiBOB"},
                        {"label": "LiBF4", "value": "LiBF4"},
                    ],
                    value="LiPF6",  # Default selected value
                    clearable=False,  # Disable clearing of the selection
                    style={"width": "50%"}
                )
            ]),
            dbc.Row(
                dbc.Col(
                   [
                    dash_table.DataTable(
                        id='predict-table',
                        columns=[{"name": i, "id": i, "editable": True} for i in df.columns],  # Set columns as editable
                        data=df.to_dict('records'),  # Fill table data from DataFrame
                        editable=True,               # Make entire table editable
                        style_cell=dict(textAlign='left', color='white'),
                        style_header=dict(backgroundColor="darkblue", fontWeight="bold", fontColor="white"),
                        style_data=dict(backgroundColor="black")
                    ),
                   ]
                )
            )
        ]),
        dbc.CardBody(
                dbc.Button(
                    'RUN PREDICTION', 
                    id='run-prediction', 
                    n_clicks=0,
                    style={'background-color': 'blue', 'color': 'white'}
                )
        ),
        dbc.CardBody(
                "Predicted Conductivity (mS/cm): ", id="prediction-output",
                style={
                "fontSize": "20px",    # Set font size
                "fontWeight": "bold"   # Make text bold
            }
        )
    ],
    style={
            "width": "900px",  # Set the width of the card
            "height": "600px", # Set the height of the card
    }
)

data2 = data.copy()
data2.pop("Value")
data2.update(
    {
        "Upper Bound": [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
        "Lower Bound": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        "Optimal Value": [None] * 7
    }
)
df2 = pd.DataFrame(data2)
df2_cols = []
for i in df2.columns:
    if i != 'Optimal Value':
        df2_cols.append({"name": i, "id": i, "editable": True})
    else:
        df2_cols.append({"name": i, "id": i, "editable": False})

screening_card = dbc.Card(
    [
        dbc.CardHeader(html.H2("Electrolyte Screening", style={'color': "black"})),
        dbc.CardBody([
            dbc.Row([
                    html.H4("Salt", style={"color": "black"}),
                    dcc.Dropdown(
                        id="screening-salt-dropdown",
                        options=[
                            {"label": "LiPF6", "value": "LiPF6"},
                            {"label": "LiBOB", "value": "LiBOB"},
                            {"label": "LiBF4", "value": "LiBF4"},
                        ],
                        value="LiPF6",  # Default selected value
                        clearable=False,  # Disable clearing of the selection
                        style={"width": "50%"}
                    )
            ]),
            dbc.Row(
                dbc.Col(
                   [
                    dash_table.DataTable(
                        id='screening-table',
                        columns=[{"name": i, "id": i, "editable": True} for i in df2.columns],  # Set columns as editable
                        data=df2.to_dict('records'),  # Fill table data from DataFrame
                        editable=True,               # Make entire table editable
                        style_cell=dict(textAlign='left', color='white'),
                        style_header=dict(backgroundColor="darkblue", fontWeight="bold", fontColor="white"),
                        style_data=dict(backgroundColor="black")
                    ),
                   ]
                )
            )
        ]),
        dbc.CardBody(
            dbc.Row([
                dbc.Col([
                    dbc.Button(
                        'RUN SCREENING', 
                        id='run-screening', 
                        n_clicks=0,
                        style={'background-color': 'blue', 'color': 'white', "width": "50%"}
                    ),
                    html.Pre(
                    id="current-output",
                    style={"whiteSpace": "pre-wrap", "marginTop": 20}
                    ),
                    dcc.Interval(id="interval", interval=100, n_intervals=0, disabled=True),
                ]),
                dbc.Col([
                    dbc.CardBody(
                        "Predicted Conductivity (mS/cm): ", id="optimal-prediction-output",
                        style={
                        "fontSize": "20px",    # Set font size
                        "fontWeight": "bold"   # Make text bold
                        }
                    )
                ])
            ])
            
        ),
        
        
    ],
    style={
            "width": "900px",  # Set the width of the card
            "height": "600px", # Set the height of the card
    }
)


is_filtered = None

contour_card = dbc.Card([   
        dbc.CardHeader(html.H2("Visualization-Condition", style={'color': "black"})),
        dbc.CardBody([
            dbc.Row([   
                dbc.Col([
                    html.H4("Solvent-1", style={"color": "black"}),
                    dcc.Dropdown(
                        id="solvent-1-dropdown",
                        options=[
                            {"label": "None", "value": "None"},
                            {"label": "EC", "value": "EC"},
                            {"label": "PC", "value": "PC"},
                            {"label": "EMC", "value": "EMC"},
                            {"label": "DMC", "value": "DMC"},
                            {"label": "DEC", "value": "DEC"},
                            {"label": "FEC", "value": "FEC"}
                        ],
                        value="None",  # Default selected value
                        clearable=False,  # Disable clearing of the selection
                        style={"width": "100%"}
                    )
                ]),
                dbc.Col([
                    html.H4("+", style={"color": "black"}),
                ]),
                dbc.Col([
                    html.H4("Solvent-2", style={"color": "black"}),
                    dcc.Dropdown(
                        id="solvent-2-dropdown",
                        options=[
                            {"label": "None", "value": "None"},
                            {"label": "EC", "value": "EC"},
                            {"label": "PC", "value": "PC"},
                            {"label": "EMC", "value": "EMC"},
                            {"label": "DMC", "value": "DMC"},
                            {"label": "DEC", "value": "DEC"},
                            {"label": "FEC", "value": "FEC"}
                        ],
                        value="None",  # Default selected value
                        clearable=False,  # Disable clearing of the selection
                        style={"width": "100%"}
                    )
                ]),
                dbc.Col([
                    html.H4("=", style={"color": "black"}),
                ]),
                dbc.Col([
                    html.H4("Range", style={"color": "black"}),
                    dcc.Slider(
                        id="solvents-slider",
                        min=0,               # Minimum value
                        max=1,             # Maximum value
                        step=0.01,              # Step size
                        value=0.5,            # Default value
                        marks={i: str(i) for i in range(0, 1, 10)},  # Custom marks every 10 units
                        tooltip={"placement": "bottom", "always_visible": True}  # Show tooltip
                    )
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    html.H4("Salt", style={"color": "black"}),
                    dcc.Dropdown(
                        id="salt-dropdown",
                        options=[
                            {"label": "LiPF6", "value": "LiPF6"},
                            {"label": "LiBOB", "value": "LiBOB"},
                            {"label": "LiBF4", "value": "LiBF4"},
                        ],
                        value="LiPF6",  # Default selected value
                        clearable=False,  # Disable clearing of the selection
                        style={"width": "100%"}
                    )
                ]),
                dbc.Col([
                    html.H4("=", style={"color": "black"}),
                ]),
                dbc.Col([
                    html.H4("Range", style={"color": "black"}),
                    dcc.RangeSlider(
                        id="salt-slider",
                        min=0,               # Minimum value
                        max=5,             # Maximum value
                        step=0.5,              # Step size
                        value=[0.5, 1.5],            # Default value
                        marks={i: str(i) for i in range(0, 1, 100)},  # Custom marks every 10 units
                        tooltip={"placement": "bottom", "always_visible": True}  # Show tooltip
                    )
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    html.H4("Condition table", style={"color": "black"}),
                    dash_table.DataTable(
                        id='condition-table',
                        columns=[{"name": i, "id": i, "editable": True} for i in df.columns],
                        data=df.to_dict('records'),
                        editable=True,
                        style_cell=dict(textAlign='left', color='white'),
                        style_header=dict(backgroundColor="darkblue", fontWeight="bold", fontColor="white"),
                        style_data=dict(backgroundColor="black"),
                    ),
                ])
            ]),
            dbc.CardBody(
                id="condition-output",
                style={
                "fontSize": "20px",    # Set font size
                "fontWeight": "bold"   # Make text bold
                }            
            ),
            dbc.Row([
                dbc.Col([
                    dbc.Button(
                        'RUN PLOT', 
                        id='run-plot', 
                        n_clicks=0,
                        style={'background-color': 'blue', 'color': 'white'}
                    ),
                    dcc.Interval(id="progress-interval", interval=500),
                    dbc.Progress(id="progress-bar", value=0, label="0%", striped=True, animated=True, style={'width': '50%'}),
                ])        
            ]),
        ]),

    ],
    style={
            "width": "900px",  # Set the width of the card
            "height": "700px", # Set the height of the card
    }
)

describe_card = dbc.Card(
    [
        dbc.CardHeader(html.H2("Visualization Contour Map", style={'color': 'black'})),
        dbc.CardBody(
            dbc.Row([
                dcc.Graph(
                        id='contour-map', 
                        style={"width": "800px", "height": "600px"}
                    )
                ])
        )
    ],
    style={
            "width": "900px",  # Set the width of the card
            "height": "700px", # Set the height of the card
    }
)
        

app.layout = html.Div(
    [
        header,        
        dbc.Container(
            [
             dbc.Row([dbc.Col(predict_card), dbc.Col(screening_card)]),
             dbc.Row([dbc.Col(contour_card), dbc.Col(describe_card)]),
             ],
            
            fluid=True,
        ),
    ]
)

# Callback to capture and display the updated table data
@app.callback(
    Output('prediction-output', 'children'),
    Input('run-prediction', 'n_clicks'),
    State('predict-table', 'data'),
    State('prediction-salt-dropdown', 'value'),
    prevent_initial_call=True
)
def display_updated_data(n_clicks, rows, salt):
    pred_x = list(map(lambda x: float(x['Value']), rows))
    buffer = 1e-3
    try:
        if (sum(pred_x[:-1]) > (1 + buffer)) or (sum(pred_x[:-1]) < (1 - buffer)):
            return "Sum of EC, PC, EMC, DMC, DEC, FEC should be equal to 1! But yours is %.4f" % sum(pred_x[:-1])
        pred_x.insert(-1, salt) # insert salt name  
        mixture = [MIXTURE(*pred_x)]  # A simple 2D function (minimizes at [0, 0])
        samples = create_mixtures(mixtures=mixture)
        with open("./data/temp_samples.pkl", "wb") as f:
            pickle.dump(samples, f)
        test_data = RevIndexedDataset('./data/temp_samples.pkl', mean=target_mean, std=target_std)

        out = predict(test_data=test_data, model=model)
        out = np.power(10, dataset.get_orig(out)) * 1000
        return "Predicted Conductivity (mS/cm): %.4f" % out
    except ValueError:
        raise Exception(ValueError)
    
iteration_counts = 0
current_result = {}
is_running = False
result = None

def iteration_callback(xk, convergence):
    # Update the global variable with the latest result only
    global current_result
    current_result = {
        "solution": xk,
        "Recipe": "EC: %.2f || PC: %.2f || EMC: %.2f || DMC: %.2f || DEC: %.2f || FEC: %.2f  || Salt: %.2f " % (xk[0], xk[1], xk[2], xk[3], xk[4], xk[5], xk[6]),
        "objective_value": objective_function(xk) * -1,
        "convergence": convergence
    }


def run_optimization(rows, salt="LiPF6"):
    ## run optimization
    global is_running
    is_running = True

    bounds = list(map(lambda x: (float(x['Lower Bound']), float(x['Upper Bound'])), rows))
    A = [[1, 1, 1, 1, 1, 1, 0]]
    lc_lower_bound = 1.0
    uc_upper_bound = 1.0
    lc = LinearConstraint(A=A, ub=uc_upper_bound, lb=lc_lower_bound)

    global result
    result = differential_evolution(lambda x: objective_function(x, salt=salt), bounds=bounds, constraints=lc, 
                                    seed=42, callback=iteration_callback)
    
    is_running = False



# Combined callback to start the optimization and update the output
@app.callback(
    Output("interval", "disabled"),
    Output("current-output", "children"),
    Output("screening-table", "data"),
    Output("optimal-prediction-output", "children"),
    Input("run-screening", "n_clicks"),
    Input("interval", "n_intervals"),
    State('screening-table', 'data'),
    State('screening-salt-dropdown', 'value')
)
def update_output(n_clicks, n_intervals, rows, salt):
    global is_running
    
    # Check if the start button was clicked
    ctx = dash.callback_context
    if ctx.triggered and ctx.triggered[0]["prop_id"].split(".")[0] == "run-screening":
        if not is_running and n_clicks > 0:
            # Clear previous results and start a new optimization in a background thread
            current_result.clear()
            threading.Thread(target=lambda: run_optimization(rows, salt)).start()
            return False, "Optimization started...\n", dash.no_update, dash.no_update
    
    # Update the output with only the current iteration result
    if current_result:
        output_text = (f"{current_result['Recipe']}\n"
                       f"Conductivity = {float(current_result['objective_value']):.4f} mS/cm\n")
        out_message = ""
        if not is_running:
            output_text += "Optimization Complete!"
            opt_x = result.x
            opt_x = [float(opt_x[i]) for i in range(len(opt_x))]
            opt_x.insert(-1, salt) # insert salt name
            mixture = [MIXTURE(*opt_x)]
            samples = create_mixtures(mixtures=mixture)
            with open("./data/temp_samples.pkl", "wb") as f:
                pickle.dump(samples, f)
            test_data = RevIndexedDataset('./data/temp_samples.pkl', mean=target_mean, std=target_std)

            out = predict(test_data=test_data, model=model)
            out = np.power(10, dataset.get_orig(out)) * 1000
            out_message = "Best Conductivity (mS/cm): %.4f" % out
            opt_x.pop(-2) # remove salt name
            for i in range(len(opt_x)):
                rows[i]["Optimal Value"] = opt_x[i]
        return dash.no_update, output_text, rows, out_message
    
    return dash.no_update, "Press RUN SCREENING to start optimization...", dash.no_update, dash.no_update


progress = {'value': 0}

@app.callback(
    Output('progress-bar', 'value'),
    Output('progress-bar', 'label'),
    Input('progress-interval', 'n_intervals'),
    prevent_initial_call=True
)
def update_progress(n):
    return progress['value'], f"{progress['value']}%"


@app.callback(
    Output("condition-table", "style_data_conditional"),
    Input("solvent-1-dropdown", "value"),
    Input("solvent-2-dropdown", "value"),
    Input("salt-dropdown", "value"),
    prevent_initial_call=True
)
def display_condition(solvent_1, solvent_2, salt):
### Redirect to condition-table
    is_filtered = [{
        "if": {"filter_query": f'{{Name}} = "{name}"'},  # Condition to make row with ID = 1 non-editable
        "pointer-events": "none",            # Disable interactions on row with ID = 1
        "backgroundColor": "#f0f0f0"         # Optional styling to indicate non-editable
    } for name in [solvent_1, solvent_2, f"Salt (mol/kg)"]]

    return is_filtered

@app.callback(
    Output("condition-output", "children"),
    Output("contour-map", "figure"),
    Input("run-plot", "n_clicks"),
    State("solvent-1-dropdown", "value"),
    State("solvent-2-dropdown", "value"),
    State("solvents-slider", "value"),
    State("salt-dropdown", "value"),
    State("salt-slider", "value"),
    State("condition-table", "data"),
    prevent_initial_call=True
)
def display_contour(n_clicks, solvent_1, solvent_2, solvent_range: list, salt, salt_range: list, rows):
    solvent_list = ["EC", "PC", "EMC", "DMC", "DEC", "FEC"]
    xs = np.linspace(0, solvent_range, 25)
    ys = np.linspace(*salt_range, 25)
    x1_idx = solvent_list.index(solvent_1)
    x2_idx = solvent_list.index(solvent_2)
    solvent_list.pop(x1_idx)
    solvent_list.pop(x2_idx)

    message = ""
    other_solvent_sum = sum([float(solvent['Value']) for solvent in rows if solvent['Name'] in solvent_list])
    other_solvent_string = " ".join(solvent_list)
    buffer = 1e-3
    if ((1 - solvent_range) < (other_solvent_sum - buffer)) or ((1 - solvent_range) > (other_solvent_sum + buffer)):
        message += "Sum of %s should be equal to %.4f! But yours is %.4f" % (other_solvent_string, (1 - solvent_range), other_solvent_sum)
        return message, dash.no_update
    else:
        message += "Please look at the contour map at the right hand side card"
        X, Y = np.meshgrid(xs, ys)
        Z = np.zeros_like(X)
        Z = Z.reshape(-1)
        for i, (x, y) in enumerate(zip(X.reshape(-1), Y.reshape(-1))):
            progress['value'] = int(i / 625 * 100)
            s = [float(solvent['Value']) for solvent in rows]
            s[x1_idx] = float(x) # solvent 1
            s[x2_idx] = float(solvent_range) - float(x) # solvent 2
            s[-1] = float(y) # salt concentration
            s.insert(-1, salt) # insert salt name
            mixture = [MIXTURE(*s)]
            samples = create_mixtures(mixtures=mixture)
            with open("./data/temp_samples2.pkl", "wb") as f:
                pickle.dump(samples, f)
            test_data = RevIndexedDataset('./data/temp_samples2.pkl', mean=target_mean, std=target_std)
            out = predict(test_data=test_data, model=model)
            out = dataset.get_orig(out)
            out = np.power(10, out) * 1000
            Z[i] += out
        Z = Z.reshape(X.shape)

        return message, plot_contour(x=xs, y=ys, z=Z, title=f"{solvent_1}/{solvent_2}_{salt}_map",
                                    xname=f"{solvent_1} (1 - {solvent_2})", yname=f"{salt} mol/kg")


if __name__ == "__main__":
    app.run_server(debug=True, port=5020)