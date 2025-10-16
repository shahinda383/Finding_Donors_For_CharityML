# Smart Donation Prediction Dashboard - Vivid Colors + Extra Visualizations
# Requirements:
# pip install dash dash-bootstrap-components plotly scikit-learn pandas numpy xgboost

import os
import io
import base64
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb

import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, dash_table, callback_context
import dash_bootstrap_components as dbc

# --------------------- 🎨 Theme / Colors (Vivid Modern: Pink / Orange / Teal) ---------------------
BG = "#FFF9F7"             # Soft warm background
CARD = "#FFFFFF"           # Card background
PINK = "#FF6B6B"           # vivid pink
ORANGE = "#FFA94D"         # vivid orange
TEAL = "#2DD4BF"           # vivid teal
YELLOW = "#FFD43B"         # accent yellow
MUTED_TEXT = "#6B7280"
TEXT = "#0F172A"

# Plotly template
PLOTLY_TEMPLATE = {
    "layout": {
        "plot_bgcolor": BG,
        "paper_bgcolor": BG,
        "font": {"color": TEXT, "family": "Inter, sans-serif"},
        "margin": {"t": 40, "l": 20, "r": 20, "b": 20},
        "title_x": 0.5,
        "xaxis": {"gridcolor": "#FFEFE6"},
        "yaxis": {"gridcolor": "#FFEFE6"},
        "colorway": [TEAL, ORANGE, PINK, YELLOW]
    }
}

# --------------------- Helper: load df ---------------------
DEFAULT_FILE = "cleaned_census.csv"

def read_df_from_file(path=DEFAULT_FILE):
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        # Fallback demo dataset
        return pd.DataFrame({
            "age":[25,34,45,50,23,40,55,60,28,38],
            "education_num":[10,12,16,9,14,13,11,17,15,12],
            "capital_gain":[0,5000,0,0,0,12000,0,8000,0,0],
            "capital_loss":[0,0,0,0,0,0,0,0,0,0],
            "hours_per_week":[40,36,50,30,20,45,40,55,35,40],
            "fnlwgt":[220000,180000,310000,190000,250000,280000,200000,270000,240000,230000],
            "workclass_Private":[1,1,1,0,0,0,1,0,1,0],
            "occupation_Exec":[0,1,0,1,0,0,0,1,0,0],
            "race_White":[1,1,0,1,1,1,1,1,1,0],
            "sex_Male":[1,0,1,1,0,1,1,0,1,0],
            "donation_success":[0,1,0,0,1,1,0,1,0,1]
        }).rename(columns={'donation_success': 'donate'})

# initial load
df = read_df_from_file(DEFAULT_FILE)
if "donate" not in df.columns:
    if df.empty:
        df['donate'] = 0
    else:
        df['donate'] = (df['age'].fillna(0) > 30).astype(int)

# Identify numeric features (exclude target)
numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c!="donate"]
INPUT_FEATURES = numeric_cols[:6] if len(numeric_cols) >= 6 else numeric_cols

# --------------------- Train model function (XGBoost) ---------------------
def train_model(dataframe):
    if "donate" not in dataframe.columns or len(dataframe) < 10:
        feature_list = dataframe.drop("donate", axis=1).columns.tolist() if "donate" in dataframe.columns else []
        return {
            "model": None, "accuracy": 0.0, "confusion_matrix": np.array([[0,0],[0,0]]),
            "precision_recall_fscore": ([0,0],[0,0],[0,0],[0,0]),
            "feature_importances": pd.Series(0, index=feature_list),
            "fpr": None, "tpr": None, "roc_auc": None,
            "feature_names": feature_list
        }

    X = dataframe.drop("donate", axis=1)
    y = dataframe["donate"]
    X_num = X.select_dtypes(include=[np.number])
    feature_names = X_num.columns.tolist()
    strat = y if y.nunique() > 1 and y.value_counts().min() > 1 else None

    try:
        X_train, X_test, y_train, y_test = train_test_split(X_num, y, test_size=0.25, random_state=42, stratify=strat)
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('xgb', xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, n_estimators=200, n_jobs=-1))
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        prf = precision_recall_fscore_support(y_test, y_pred, zero_division=0)

        roc_auc, fpr, tpr = None, None, None
        try:
            if hasattr(pipeline, "predict_proba"):
                probs = pipeline.predict_proba(X_test)[:,1]
                fpr, tpr, _ = roc_curve(y_test, probs)
                roc_auc = auc(fpr, tpr)
        except Exception:
            pass

        try:
            xgb_model = pipeline.named_steps['xgb']
            importances = pd.Series(xgb_model.feature_importances_, index=feature_names).sort_values(ascending=False)
        except Exception:
            importances = pd.Series([0]*len(feature_names), index=feature_names)

        return {
            "model": pipeline, "accuracy": acc, "confusion_matrix": cm,
            "precision_recall_fscore": prf, "feature_importances": importances,
            "fpr": fpr, "tpr": tpr, "roc_auc": roc_auc,
            "feature_names": feature_names
        }

    except ValueError as e:
        print(f"Model Training Failed (ValueError): {e}")
        return {
            "model": None, "accuracy": 0.0, "confusion_matrix": np.array([[0,0],[0,0]]),
            "precision_recall_fscore": ([0,0],[0,0],[0,0],[0,0]),
            "feature_importances": pd.Series(0, index=feature_names),
            "fpr": None, "tpr": None, "roc_auc": None,
            "feature_names": feature_names
        }

model_bundle = train_model(df)
dataset_size = len(df)
feature_count = len(numeric_cols)

# --------------------- Plot helpers ---------------------
def fig_confusion(cm):
    z = cm
    x = ["Pred: No Donation","Pred: Donation"]
    y = ["True: No Donation","True: Donation"]
    fig = go.Figure(data=go.Heatmap(z=z, x=x, y=y, colorscale=[[0, TEAL], [1, PINK]], showscale=False))
    annotations=[]
    for i in range(2):
        for j in range(2):
            if i < z.shape[0] and j < z.shape[1]:
                 annotations.append(dict(x=x[j], y=y[i], text=str(z[i][j]), showarrow=False,
                                         font=dict(color="white" if z[i][j]>z.max()/2 else TEXT)))
    fig.update_layout(title="Confusion Matrix", annotations=annotations, template=PLOTLY_TEMPLATE)
    return fig

def fig_feature_importance(importances, top_n=10):
    if importances.empty or importances.sum() == 0:
        fig = go.Figure()
        fig.add_annotation(text="Feature importances not available (Model failed or data too small)", showarrow=False, font=dict(color=PINK))
        fig.update_layout(template=PLOTLY_TEMPLATE)
        return fig
    top = importances.head(top_n)
    fig = px.bar(x=top.values[::-1], y=top.index[::-1], orientation='h', labels={'x':'Importance Score','y':'Feature'}, title=f"Top {top_n} Feature Importances")
    fig.update_traces(marker_color=TEAL)
    fig.update_layout(template=PLOTLY_TEMPLATE)
    return fig

def fig_roc(fpr, tpr, roc_auc):
    if fpr is None or tpr is None or roc_auc is None:
        fig = go.Figure()
        fig.add_annotation(text="ROC not available (Model training unstable)", showarrow=False, font=dict(color=PINK))
        fig.update_layout(template=PLOTLY_TEMPLATE)
        return fig
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'AUC = {roc_auc:.3f}' if roc_auc else 'ROC', line=dict(color=ORANGE, width=3)))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(dash='dash', color=MUTED_TEXT), showlegend=False))
    fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", template=PLOTLY_TEMPLATE)
    return fig

def fig_prf_metrics(prf_tuple):
    if prf_tuple is None:
        fig = go.Figure()
        fig.add_annotation(text="Precision/Recall metrics not available", showarrow=False, font=dict(color=PINK))
        fig.update_layout(template=PLOTLY_TEMPLATE)
        return fig
    precision, recall, fscore, _ = prf_tuple
    data = pd.DataFrame({
        'Metric': ['Precision', 'Recall', 'F1-Score'] * 2,
        'Value': [precision[0], recall[0], fscore[0], precision[1], recall[1], fscore[1]],
        'Class': ['No Donation (0)']*3 + ['Donation (1)']*3
    })
    fig = px.bar(data, x='Metric', y='Value', color='Class', barmode='group',
                 title='Precision, Recall, F1-Score by Class',
                 color_discrete_map={'No Donation (0)': ORANGE, 'Donation (1)': TEAL})
    fig.update_layout(template=PLOTLY_TEMPLATE, yaxis_tickformat=".2f", yaxis_range=[0, 1])
    return fig

# NEW: Violin plot for top features (distribution by target)
def fig_violin_top_features(df_local, importances, top_n=3):
    if importances.empty or importances.sum() == 0:
        fig = go.Figure()
        fig.add_annotation(text="Violin plot not available (no importances)", showarrow=False, font=dict(color=PINK))
        fig.update_layout(template=PLOTLY_TEMPLATE)
        return fig
    top = importances.head(top_n).index.tolist()
    fig = make_violin_grid(df_local, top)
    fig.update_layout(template=PLOTLY_TEMPLATE, title=f"Distribution of Top {top_n} Features by Donation")
    return fig

def make_violin_grid(df_local, cols):
    # Create subplots with one violin per feature (grouped by 'donate')
    fig = go.Figure()
    for col in cols:
        if col in df_local.columns:
            fig.add_trace(go.Violin(x=df_local['donate'].astype(str), y=df_local[col], name=col, box_visible=True, meanline_visible=True))
    fig.update_traces(orientation='v', width=0.8)
    fig.update_layout(legend_title="Feature", yaxis_title="Value", xaxis_title="Donate (0/1)")
    return fig

# NEW: Lift chart (by deciles) - uses model if available
def fig_lift_chart(dataframe, model_bundle_local):
    try:
        if model_bundle_local["model"] is None:
            raise ValueError("Model not available")
        # get numeric data used in training
        X_num = dataframe.select_dtypes(include=[np.number]).drop(columns=['donate'], errors='ignore')
        # ensure same columns as model
        cols = model_bundle_local['feature_names']
        # create DataFrame with missing columns zero-filled
        X_for_pred = pd.DataFrame(0, index=X_num.index, columns=cols)
        for c in X_num.columns:
            if c in X_for_pred.columns:
                X_for_pred[c] = X_num[c]
        probs = model_bundle_local['model'].predict_proba(X_for_pred)[:,1]
        tmp = pd.DataFrame({'y': dataframe['donate'].values, 'proba': probs})
        tmp = tmp.sort_values('proba', ascending=False).reset_index(drop=True)
        tmp['decile'] = pd.qcut(tmp.index+1, 10, labels=False)
        lift = tmp.groupby('decile').agg({'y':'mean', 'proba':'mean'}).reset_index()
        # cumulative
        lift['cum_actual'] = tmp['y'].expanding().mean().values[:len(lift)]  # not exact; alternative below
        # Plotting decile actual vs average probability
        fig = go.Figure()
        fig.add_trace(go.Bar(x=lift['decile']+1, y=lift['y'], name='Actual Rate', marker_color=ORANGE))
        fig.add_trace(go.Line(x=lift['decile']+1, y=lift['proba'], name='Avg Predicted Prob', marker_color=TEAL))
        fig.update_layout(title='Lift by Decile (Actual rate vs Predicted probability)', xaxis_title='Decile (1 = highest prob)', yaxis_title='Rate', template=PLOTLY_TEMPLATE)
        return fig
    except Exception:
        fig = go.Figure()
        fig.add_annotation(text="Lift chart not available (need model + numeric data)", showarrow=False, font=dict(color=PINK))
        fig.update_layout(template=PLOTLY_TEMPLATE)
        return fig

# Helper function to get metric text
def get_safe_metric_value(bundle, key):
    val = bundle.get(key)
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    if key == 'accuracy':
        return f"{val*100:.2f} %"
    if key == 'roc_auc':
        return f"{val:.3f}"
    if key == 'feature_importances' and not val.empty and val.sum() > 0:
        return val.index[0]
    return "N/A"

def render_metric_card(title, value_id, color):
    # color is one of: 'pink','orange','teal' (we will use classes for color)
    return dbc.Card(
        dbc.CardBody([
            html.H6(title, className="card-subtitle mb-2 text-muted"),
            html.H3(id=value_id, children="...", className=""),
        ]),
        className="shadow-sm border-0 rounded-lg"
    )

# --------------------- Layout Components ---------------------
header = html.Div(
    style={
        'background': f'linear-gradient(90deg, {PINK} 0%, {ORANGE} 50%, {TEAL} 100%)',
        'padding': '1.5rem 0',
        'color': 'white',
        'boxShadow': '0 4px 6px rgba(0,0,0,0.08)'
    },
    children=dbc.Container([
        html.H1("💰 Smart Donor Predictor Dashboard 🎯", className="fw-bold mb-0"),
        html.P("✨ XGBoost Power: Predict Propensity. Maximize Impact. 🚀", className="lead mb-0")
    ])
)

upload = dcc.Upload(
    id='upload-data',
    children=html.Div(['📁 Drag & Drop or ', html.A('Select a CSV file', style={"color": TEAL, "fontWeight": "bold"})]),
    style={'width': '100%', 'padding': '15px', 'borderWidth': '2px', 'borderStyle': 'dashed',
           'borderRadius': '12px','textAlign':'center', 'backgroundColor':CARD, 'borderColor': TEAL, 'cursor': 'pointer'},
    multiple=False
)

# --------------------- TAB 1: Overview ---------------------
tab_overview = dbc.Card(
    dbc.CardBody([
        html.H4("Project Overview & Summary Statistics 📊", className="mb-4", style={"color": TEAL}),
        html.P(
            children=[
                "Welcome to your ", html.B("Impact Center!"), " 💡 This interactive tool predicts the ",
                html.B("propensity of donation"), " using a high-performance ", html.B("XGBoost Classifier"), " 🚀."
            ],
            className="lead"
        ),
        html.Hr(className="my-4"),
        html.H5("System Status", className="mb-3"),
        dbc.Row([
            dbc.Col(render_metric_card("Dataset Size (Rows)", "metric-dataset-size", "teal"), md=4),
            dbc.Col(render_metric_card("Features Used", "metric-feature-count", "teal"), md=4),
            dbc.Col(render_metric_card("Current Model Accuracy", "metric-accuracy-val", "teal"), md=4),
        ], className="g-4 mb-4"),
        dbc.Row([
            dbc.Col(dbc.Button("Retrain Model", id="btn-retrain", color="dark", className="w-100 rounded-pill shadow-sm"), md=3),
            dbc.Col(html.Div(id="retrain-status"), md=9)
        ])
    ]),
    className="mt-3 shadow-lg border-0"
)

# --------------------- TAB 2: Model & Graphs ---------------------
tab_model_graphs = dbc.Card(
    dbc.CardBody([
        dbc.Row([
            dbc.Col(render_metric_card("Top Feature", "metric-top-feature-val", "pink"), md=4),
            dbc.Col(render_metric_card("ROC AUC Score", "metric-roc-auc-val", "teal"), md=4),
            dbc.Col(render_metric_card("Model Accuracy", "metric-accuracy-val-2", "teal"), md=4),
        ], className="g-4 mb-4"),
        dbc.Row([
            dbc.Col([
                dbc.Card(dbc.CardBody([
                    html.H6("XGBoost Feature Importances", className="mb-3 text-muted"),
                    dcc.Graph(id="feat-imp", figure=fig_feature_importance(model_bundle['feature_importances']))
                ]), className="mb-4"),
                dbc.Card(dbc.CardBody([
                    html.H6("Class-wise Performance Metrics", className="mb-3 text-muted"),
                    dcc.Graph(id="prf-metrics-fig", figure=fig_prf_metrics(model_bundle['precision_recall_fscore']))
                ])),
            ], md=6),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("Model Performance", className="mb-3 text-muted"),
                dcc.Graph(id="cm-fig", figure=fig_confusion(model_bundle['confusion_matrix'])),
                html.Hr(className="my-2"),
                dcc.Graph(id="roc-fig", figure=fig_roc(model_bundle['fpr'], model_bundle['tpr'], model_bundle['roc_auc']))
            ])), md=6)
        ], className="g-4"),
        html.Hr(),
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("Distribution — Top Features (Violin)", className="mb-3 text-muted"),
                dcc.Graph(id="violin-fig", figure=fig_violin_top_features(df, model_bundle['feature_importances'], top_n=3))
            ])), md=6),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("Lift Chart by Decile", className="mb-3 text-muted"),
                dcc.Graph(id="lift-fig", figure=fig_lift_chart(df, model_bundle))
            ])), md=6)
        ], className="g-4")
    ]),
    className="mt-3 shadow-lg border-0"
)

# Function to render prediction inputs dynamically
def render_prediction_inputs(features):
    return dbc.Row([
        dbc.Col(html.Div([
            html.Label(col, className="form-label fw-bold text-sm text-muted"),
            dcc.Input(id=f"inp-{col}", type="number", placeholder=col, className="form-control rounded-pill"),
        ], className="mb-3")) for col in features
    ], className="g-3")

# --------------------- TAB 3: Predict Donation ---------------------
tab_predict = dbc.Card(
    dbc.CardBody(
        id='tab-predict-content',
        children=[
            html.H4("Quick Prediction Tool", className="mb-4", style={"color": TEAL}),
            html.P("Enter values for the top numeric features below. The model predicts the probability of donation success.", className="text-muted"),
            render_prediction_inputs(INPUT_FEATURES),
            html.Div(className="mt-4"),
            dbc.Row([
                dbc.Col(dbc.Button("Auto-Fill Example", id="btn-example", color="secondary", className="rounded-pill shadow-sm w-100"), md=3),
                dbc.Col(dbc.Button("PREDICT DONATION", id="btn-predict", color="success", className="rounded-pill shadow-lg w-100"), md=3),
                dbc.Col(html.Div(id="predict-output", className="ms-3 p-2"), md=6)
            ], className="align-items-center g-3")
        ]
    ),
    className="mt-3 shadow-lg border-0"
)

# --------------------- TAB 4: View Dataset ---------------------
tab_view_dataset = dbc.Card(
    dbc.CardBody([
        html.H4("Dataset Preview", className="mb-4", style={"color": TEAL}),
        html.P(f"Showing the first 10 rows of the current dataset ({dataset_size} total rows). Upload a new CSV to replace the data.", className="text-muted"),
        html.Hr(),
        dash_table.DataTable(
            id='current-data-table',
            data=df.head(10).to_dict('records'),
            columns=[{"name": i, "id": i} for i in df.columns],
            style_header={'backgroundColor': TEAL, 'color': 'white', 'fontWeight': 'bold', 'fontSize': '14px'},
            style_cell={'textAlign': 'left', 'minWidth': '100px', 'backgroundColor': BG, 'color': TEXT},
            style_table={'overflowX': 'auto', 'borderRadius': '8px', 'border': '1px solid #E5E7EB'},
            page_action='none',
        )
    ]),
    className="mt-3 shadow-lg border-0"
)

# Group tabs
tabs = dcc.Tabs(
    id='tabs-container',
    value='tab-1',
    children=[
        dcc.Tab(label='1️⃣ Overview', value='tab-1', children=tab_overview),
        dcc.Tab(label='2️⃣ Model & Graphs', value='tab-2', children=tab_model_graphs),
        dcc.Tab(label='3️⃣ Predict Donation', value='tab-3', children=tab_predict),
        dcc.Tab(label='4️⃣ View Dataset', value='tab-4', children=tab_view_dataset),
    ],
    className="mt-4", style={"fontWeight": "bold", "color": TEXT, "borderBottom": f"3px solid {TEAL}"}
)

# --------------------- Dash App Layout ---------------------
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Smart Donation Prediction Dashboard"

app.layout = html.Div(style={"backgroundColor": BG, "minHeight": "100vh"}, children=[
    header,
    dbc.Container([
        html.Br(),
        dbc.Row([dbc.Col(upload, md=12)], className="mb-4"),
        tabs,
        html.Footer("Smart Donation Prediction Dashboard • Powered by XGBoost", style={"textAlign":"center","color":MUTED_TEXT,"padding":"18px 0","marginTop":"30px"})
    ], className="py-4")
])

# --------------------- Callbacks ---------------------
def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            new_df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            if 'donate' not in new_df.columns and not new_df.empty:
                new_df['donate'] = (new_df.iloc[:, 0].fillna(0) > new_df.iloc[:, 0].mean()).astype(int)
            elif 'donate' not in new_df.columns:
                new_df['donate'] = 0
            return new_df
        else:
            return None
    except Exception as e:
        print("Upload parsing error:", e)
        return None

# Prediction callback
@app.callback(
    Output("predict-output", "children"),
    Input("btn-predict", "n_clicks"),
    Input("btn-example", "n_clicks"),
    [State(f"inp-{col}", "value") for col in INPUT_FEATURES]
)
def handle_predict(n_click, n_example, *vals):
    ctx = callback_context
    if model_bundle["model"] is None:
        return dbc.Alert("Model is not trained. Data is too small or training failed.", color="danger")
    if not ctx.triggered:
        return ""
    trigger = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger == "btn-example":
        if INPUT_FEATURES and not df.empty:
            ex = df[INPUT_FEATURES].iloc[0].tolist()
            padded = ex + [0]*(len(model_bundle['feature_names']) - len(ex))
            try:
                input_df = pd.DataFrame([padded], columns=model_bundle['feature_names'])
                proba = model_bundle['model'].predict_proba(input_df)[0][1] if hasattr(model_bundle['model'], "predict_proba") else None
                pred = model_bundle['model'].predict(input_df)[0]
            except Exception as e:
                return dbc.Alert(f"Prediction failed with example data: {e}. Check input feature count.", color="danger")
            txt_pred = f"Prediction: {'DONATE (1)' if pred==1 else 'NO DONATION (0)'}"
            if proba is not None:
                txt_proba = f"Probability: {proba*100:.2f}%"
                return dbc.Alert([html.Div(f"Example Used: {txt_pred}", style={"fontWeight":"700"}), html.Div(txt_proba, style={"fontSize":"13px","color":MUTED_TEXT})], color="info")
            else:
                return dbc.Alert(f"Example Used: {txt_pred}", color="info")

    else:
        current_vals = list(vals)[:len(INPUT_FEATURES)]
        if any(v is None or v == '' for v in current_vals):
            return dbc.Alert("Please provide values for all input fields (or click Auto-Fill Example).", color="warning")
        try:
            inp = [float(v) for v in current_vals]
            padded = inp + [0.0]*(len(model_bundle['feature_names']) - len(inp))
            input_df = pd.DataFrame([padded], columns=model_bundle['feature_names'])
            proba = model_bundle['model'].predict_proba(input_df)[0][1] if hasattr(model_bundle['model'], "predict_proba") else None
            pred = model_bundle['model'].predict(input_df)[0]
            color = "success" if pred==1 else "secondary"
            if proba is not None:
                return dbc.Alert([html.Div(f"Prediction: {'DONATE (1)' if pred==1 else 'NO DONATION (0)'}", style={"fontWeight":"700"}),
                                  html.Div(f"Probability: {proba*100:.2f}%", style={"fontSize":"13px","color":MUTED_TEXT})], color=color)
            else:
                return dbc.Alert(f"Prediction: {'DONATE (1)' if pred==1 else 'NO DONATION (0)'}", color=color)
        except Exception as e:
            return dbc.Alert(f"Prediction Error: {e}", color="danger")

# Retrain / update all UI
@app.callback(
    Output("feat-imp", "figure"),
    Output("cm-fig", "figure"),
    Output("roc-fig", "figure"),
    Output("prf-metrics-fig", "figure"),
    Output("metric-dataset-size", "children"),
    Output("metric-feature-count", "children"),
    Output("metric-accuracy-val", "children"),
    Output("metric-top-feature-val", "children"),
    Output("metric-roc-auc-val", "children"),
    Output("metric-accuracy-val-2", "children"),
    Output("retrain-status", "children"),
    Output("tab-predict-content", "children"),
    Output("current-data-table", "data"),
    Output("current-data-table", "columns"),
    Input("btn-retrain", "n_clicks"),
    Input("upload-data", "contents"),
    State("upload-data", "filename")
)
def update_all_ui(n_clicks, contents, filename):
    global df, model_bundle, numeric_cols, INPUT_FEATURES, dataset_size, feature_count
    triggered = callback_context.triggered
    status = ""

    if triggered and 'upload-data' in triggered[0]['prop_id']:
        if contents is not None:
            new_df = parse_contents(contents, filename)
            if new_df is None or "donate" not in new_df.columns:
                status = dbc.Alert("Failed to parse uploaded file or 'donate' column is missing.", color="danger")
            else:
                df = new_df
                numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c!="donate"]
                INPUT_FEATURES = numeric_cols[:6] if len(numeric_cols)>=6 else numeric_cols
                dataset_size = len(df)
                feature_count = len(numeric_cols)
                model_bundle = train_model(df)
                if model_bundle["model"] is None:
                    status = dbc.Alert("Data uploaded. However, model retrain failed (data issue).", color="danger")
                else:
                    status = dbc.Alert(f"Data uploaded ({len(df)} rows). Model retrained successfully!", color="success")

    elif triggered and 'btn-retrain' in triggered[0]['prop_id']:
        model_bundle = train_model(df)
        if model_bundle["model"] is None:
            status = dbc.Alert("Retrain failed: Data too small or unstable.", color="danger")
        else:
            dataset_size = len(df)
            feature_count = len(numeric_cols)
            status = dbc.Alert("Model successfully retrained.", color="success")

    # Prepare updated prediction content (reflect new INPUT_FEATURES)
    updated_prediction_content = [
        html.H4("Quick Prediction Tool", className="mb-4", style={"color": TEAL}),
        html.P("Enter values for the top numeric features below. The model predicts the probability of donation success.", className="text-muted"),
        render_prediction_inputs(INPUT_FEATURES),
        html.Div(className="mt-4"),
        dbc.Row([
            dbc.Col(dbc.Button("Auto-Fill Example", id="btn-example", color="secondary", className="rounded-pill shadow-sm w-100"), md=3),
            dbc.Col(dbc.Button("PREDICT DONATION", id="btn-predict", color="success", className="rounded-pill shadow-lg w-100"), md=3),
            dbc.Col(html.Div(id="predict-output", className="ms-3 p-2"), md=6)
        ], className="align-items-center g-3")
    ]

    # Metrics
    acc_val = html.H3(get_safe_metric_value(model_bundle, 'accuracy'))
    top_feat_val = html.H3(get_safe_metric_value(model_bundle, 'feature_importances'))
    roc_auc_val = html.H3(get_safe_metric_value(model_bundle, 'roc_auc'))

    # Data table
    data_table_data = df.head(10).to_dict('records')
    data_table_cols = [{"name": i, "id": i} for i in df.columns]

    # Figures
    feat_fig = fig_feature_importance(model_bundle['feature_importances'])
    cm_fig = fig_confusion(model_bundle['confusion_matrix'])
    roc_fig = fig_roc(model_bundle['fpr'], model_bundle['tpr'], model_bundle['roc_auc'])
    prf_fig = fig_prf_metrics(model_bundle['precision_recall_fscore'])

    return (
        feat_fig,
        cm_fig,
        roc_fig,
        prf_fig,
        html.H3(dataset_size),
        html.H3(feature_count),
        acc_val,
        top_feat_val,
        roc_auc_val,
        acc_val,
        status,
        updated_prediction_content,
        data_table_data,
        data_table_cols
    )

# --------------------- Run server ---------------------
if __name__ == "__main__":
    app.run(debug=True, port=8050)