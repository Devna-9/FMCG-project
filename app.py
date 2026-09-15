import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, dash_table, Input, Output
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = Path(__file__).resolve().parent

# ----------------------------
# Theme
# ----------------------------
PRIMARY = "#2f80b7"
ORANGE = "#ff8a45"
PINK = "#d77bb5"
GREEN = "#69c2a5"
LIME = "#a6d854"
YELLOW = "#ffd43b"
TEXT = "#334155"
BG = "#f4f6f8"
WHITE = "#ffffff"
COLORS = [PRIMARY, ORANGE, PINK, GREEN, LIME, YELLOW]

CARD = {
    "backgroundColor": WHITE,
    "border": "1px solid #dfe5eb",
    "borderRadius": "6px",
    "padding": "0.35vh 0.45vw",
    "boxSizing": "border-box",
    "overflow": "hidden",
}
GRAPH = {"width": "100%", "height": "100%"}


def find_file(name):
    p = BASE_DIR / name
    if p.exists():
        return p
    raise FileNotFoundError(f"Required file not found: {name}")


# ----------------------------
# Data loading and master build
# ----------------------------
response_df = pd.read_csv(find_file("Campaign Response  Data.csv"), dtype={"ChannelPartnerID": str})
details_df = pd.read_csv(find_file("Campaign Details.csv"), dtype={"ChannelPartnerID": str})
region_df = pd.read_csv(find_file("MasterLookUp.csv"), dtype={"ChannelPartnerID": str})
transaction_df = pd.read_csv(find_file("Transaction data.csv"), dtype={"ChannelPartnerID": str})

# Normalize numeric fields. Missing values are handled later.
for df in [response_df, details_df, transaction_df]:
    for c in df.columns:
        if c != "ChannelPartnerID":
            df[c] = pd.to_numeric(df[c], errors="ignore")

transaction_df["Year"] = pd.to_numeric(transaction_df["Year"], errors="coerce").astype("Int64")
transaction_df["Month"] = pd.to_numeric(transaction_df["Month"], errors="coerce").astype("Int64")
transaction_df["Sales"] = pd.to_numeric(transaction_df["Sales"], errors="coerce").fillna(0)

# Exact master-data construction used by the original notebook, without hard-coded /content paths.
master = response_df.merge(details_df, on="ChannelPartnerID", how="left")
master = master.merge(region_df.drop_duplicates("ChannelPartnerID"), on="ChannelPartnerID", how="left")

sales_2021 = transaction_df.loc[transaction_df["Year"] == 2021].groupby("ChannelPartnerID")["Sales"].sum().rename("total_sales_2021")
sales_2022 = transaction_df.loc[transaction_df["Year"] == 2022].groupby("ChannelPartnerID")["Sales"].sum().rename("total_sales_2022")
b1_sales_2022 = transaction_df.loc[(transaction_df["Year"] == 2022) & (transaction_df["Brand"] == "B1")].groupby("ChannelPartnerID")["Sales"].sum().rename("brand_B1_sales_2022")
buy_freq_2022 = transaction_df.loc[transaction_df["Year"] == 2022].groupby("ChannelPartnerID")["Month"].nunique().rename("buying_frequency_2022")
brand_engagement = transaction_df.loc[transaction_df["Year"] == 2022].groupby("ChannelPartnerID")["Brand"].nunique().rename("brand_engagement_2022")
b1_freq_2022 = transaction_df.loc[(transaction_df["Year"] == 2022) & (transaction_df["Brand"] == "B1")].groupby("ChannelPartnerID")["Month"].nunique().rename("buying_frequency_B1_2022")

for s in [sales_2021, sales_2022, b1_sales_2022, buy_freq_2022, brand_engagement, b1_freq_2022]:
    master = master.merge(s, on="ChannelPartnerID", how="left")

master["brand_B1_contribution_2022"] = np.where(
    master["total_sales_2022"].fillna(0) > 0,
    master["brand_B1_sales_2022"].fillna(0) / master["total_sales_2022"].fillna(0),
    0,
)

q4 = transaction_df.loc[(transaction_df["Year"] == 2022) & (transaction_df["Month"] >= 10)].groupby("ChannelPartnerID")["Sales"].sum()
master["active_last_quarter"] = master["ChannelPartnerID"].map(q4).fillna(0).gt(0).map({True: "Yes", False: "No"})
q4_b1 = transaction_df.loc[(transaction_df["Year"] == 2022) & (transaction_df["Month"] >= 10) & (transaction_df["Brand"] == "B1")].groupby("ChannelPartnerID")["Sales"].sum()
master["active_last_quarter_B1"] = master["ChannelPartnerID"].map(q4_b1).fillna(0).gt(0).map({True: "Yes", False: "No"})

sales_cols = [
    "total_sales_2021", "total_sales_2022", "brand_B1_sales_2022",
    "brand_B1_contribution_2022", "buying_frequency_2022",
    "brand_engagement_2022", "buying_frequency_B1_2022",
]
master[sales_cols] = master[sales_cols].fillna(0)
master["response"] = pd.to_numeric(master["response"], errors="coerce").fillna(0).astype(int)

# ----------------------------
# Sentiment, using a self-contained lexicon so Render never has to download NLTK data.
# ----------------------------
with open(find_file("74responses.txt"), "r", encoding="utf-8", errors="ignore") as f:
    feedback = [x.strip() for x in f.readlines() if x.strip()]

POSITIVE = {
    "love", "great", "good", "excellent", "delicious", "tasty", "awesome", "amazing",
    "perfect", "fabulous", "pleasant", "happy", "worth", "nice", "yum", "fantastic",
    "creamy", "fresh", "rich", "convenient", "economical", "satisfy", "satisfies"
}
NEGATIVE = {
    "awful", "expensive", "overpriced", "over", "bad", "gross", "rubbish", "disappointed",
    "horrid", "poor", "leaking", "spoiled", "weak", "chalky", "clumpy", "ridiculous",
    "hate", "yuck", "puke", "messy", "increase", "wasted", "fake", "funny", "reflux"
}


def sentiment_score(text):
    words = re.findall(r"[a-z']+", text.lower())
    pos = sum(w in POSITIVE for w in words)
    neg = sum(w in NEGATIVE for w in words)
    # Smooth bounded score roughly on VADER's -1..1 scale.
    return float(np.tanh((pos - neg) / 2.5))


sentiment_df = pd.DataFrame({"text": feedback})
sentiment_df["compound_sentiment"] = sentiment_df["text"].apply(sentiment_score)
sentiment_df["sentiment_label"] = np.where(
    sentiment_df["compound_sentiment"] >= 0.05, "Positive",
    np.where(sentiment_df["compound_sentiment"] <= -0.05, "Negative", "Neutral"),
)


def complaint_category(text):
    t = text.lower()
    cats = []
    if any(k in t for k in ["taste", "flavor", "sweet", "chocolate", "mocha", "tasty", "chalky", "sugar"]):
        cats.append("Taste/Flavor")
    if any(k in t for k in ["pack", "packaging", "bottle", "lid", "seal", "pump", "leak", "wrap"]):
        cats.append("Packaging")
    if any(k in t for k in ["price", "expensive", "cost", "cheap", "priced", "money"]):
        cats.append("Price")
    if any(k in t for k in ["mix", "stir", "dissolve", "blend", "clump", "bead"]):
        cats.append("Mixing")
    return cats


negative = sentiment_df[sentiment_df["sentiment_label"] == "Negative"].copy()
complaint_counts = {}
for cats in negative["text"].apply(complaint_category):
    for c in cats:
        complaint_counts[c] = complaint_counts.get(c, 0) + 1
complaints_df = pd.DataFrame(list(complaint_counts.items()), columns=["Category", "Count"]).sort_values("Count", ascending=False)

# ----------------------------
# Models
# ----------------------------
model_features = [
    "loyalty", "nps", "n_yrs", "email", "sms", "call", "brand_B1_contribution_2022",
    "n_comp", "portal", "total_sales_2021", "total_sales_2022", "brand_B1_sales_2022",
    "buying_frequency_2022", "brand_engagement_2022", "buying_frequency_B1_2022",
    "Region", "active_last_quarter", "active_last_quarter_B1"
]
model_df = master[model_features + ["response"]].copy()
X = model_df[model_features]
y = model_df["response"]

numeric_features = [c for c in model_features if c not in ["Region", "active_last_quarter", "active_last_quarter_B1"]]
categorical_features = ["Region", "active_last_quarter", "active_last_quarter_B1"]

preprocessor = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_features),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical_features),
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
models = {
    "Logistic Regression": LogisticRegression(max_iter=3000, random_state=42),
    "Naïve Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=180, max_depth=8, random_state=42, n_jobs=-1),
}

pipelines = {}
metrics = {}
for name, estimator in models.items():
    pipe = Pipeline([("prep", preprocessor), ("model", estimator)])
    pipe.fit(X_train, y_train)
    prob = pipe.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, prob)
    threshold = 0.5
    pred = (prob >= threshold).astype(int)
    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    metrics[name] = {
        "fpr": fpr, "tpr": tpr, "auc": auc(fpr, tpr), "cm": cm.tolist(),
        "report": classification_report(y_test, pred, output_dict=True, zero_division=0),
        "sensitivity": tp / (tp + fn) if tp + fn else 0,
        "specificity": tn / (tn + fp) if tn + fp else 0,
        "threshold": threshold,
    }
    pipelines[name] = pipe

single_features = ["loyalty", "nps", "n_yrs", "email", "sms"]
single_pre = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), single_features)
])
single_model = Pipeline([("prep", single_pre), ("model", LogisticRegression(max_iter=3000, random_state=42))])
single_model.fit(master[single_features], master["response"])

# ----------------------------
# Figure helpers
# ----------------------------
def blank_fig(title=""):
    fig = go.Figure()
    fig.update_layout(title=title, template="plotly_white", margin=dict(l=30, r=10, t=35, b=25), font=dict(size=10))
    return fig


def response_rate_fig(var):
    tmp = master.groupby(var, dropna=False)["response"].agg(["count", "sum"]).reset_index()
    tmp["RR"] = tmp["sum"] / tmp["count"] * 100
    tmp[var] = tmp[var].fillna("Unknown").astype(str)
    fig = px.bar(tmp, x=var, y="RR", text=tmp["RR"].map(lambda x: f"{x:.1f}%"), color_discrete_sequence=[PRIMARY])
    fig.update_traces(textposition="outside")
    fig.update_layout(title=f"Response Rate by {var.replace('_',' ').title()}", yaxis_title="Response Rate (%)", xaxis_title="", margin=dict(l=35,r=10,t=38,b=35), height=280)
    return fig


def box_fig(col):
    fig = px.box(master, x="response", y=col, color="response", points=False, color_discrete_sequence=[PRIMARY, ORANGE])
    fig.update_layout(title=f"{col.replace('_',' ').title()} by Response", xaxis_title="Response (0 = No, 1 = Yes)", margin=dict(l=35,r=10,t=38,b=35), height=280)
    return fig


def top_brand_fig():
    brands = {"B1": "☕ B1 Coffee", "B2": "🍪 B2 Biscuits", "B3": "🧃 B3 Juice", "B4": "🍫 B4 Chocolates", "B5": "🍨 B5 Ice Cream", "B6": "🥤 B6", "B7": "💧 B7 Water"}
    d = transaction_df[transaction_df["Year"] == 2022].groupby("Brand")["Sales"].sum().reset_index().sort_values("Sales")
    d["label"] = d["Brand"].map(brands).fillna(d["Brand"])
    fig = go.Figure(go.Bar(y=d["label"], x=d["Sales"], orientation="h", marker_color=GREEN, customdata=d["Brand"], text=d["Sales"].map(lambda x: f"${x/1e6:.1f}M"), textposition="inside"))
    fig.update_layout(title="Brand Achievement (click a brand)", xaxis=dict(showticklabels=False, showgrid=False), yaxis=dict(autorange="reversed"), margin=dict(l=5,r=5,t=35,b=10), height=280)
    return fig


def monthly_fig():
    merged = transaction_df[["ChannelPartnerID", "Year", "Month"]].drop_duplicates().merge(master[["ChannelPartnerID", "response"]], on="ChannelPartnerID", how="inner")
    d = merged.groupby(["Year", "Month"])["response"].mean().reset_index()
    d["Year"] = d["Year"].astype(str)
    fig = px.line(d, x="Month", y="response", color="Year", markers=True, color_discrete_sequence=[GREEN, ORANGE])
    fig.update_traces(hovertemplate="Month %{x}<br>Response rate %{y:.1%}<extra></extra>")
    fig.update_layout(title="Monthly Response Rate Trend", yaxis_tickformat=".0%", margin=dict(l=30,r=10,t=38,b=30), height=280)
    return fig


def top_partners_fig():
    d = transaction_df.groupby("ChannelPartnerID")["Sales"].sum().nlargest(5).reset_index().sort_values("Sales")
    fig = go.Figure(go.Bar(y=d["ChannelPartnerID"], x=d["Sales"], orientation="h", marker_color=ORANGE, text=d["Sales"].map(lambda x:f"${x/1000:.0f}K"), textposition="inside"))
    fig.update_layout(title="Top Selling Partners", xaxis=dict(showticklabels=False), yaxis=dict(autorange="reversed"), margin=dict(l=30,r=5,t=38,b=15), height=280)
    return fig


def top15_fig():
    d = master[master["response"] == 1].nlargest(15, "brand_B1_contribution_2022").copy()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=d["ChannelPartnerID"], y=d["brand_B1_contribution_2022"], name="B1 contribution", marker_color=[COLORS[i % len(COLORS)] for i in range(len(d))]))
    fig.update_layout(title="Top 15 B1 Contributors", yaxis_tickformat=".0%", margin=dict(l=30,r=5,t=38,b=35), height=280, showlegend=False)
    return fig


def word_cloud_component(extra=""):
    text = " ".join(sentiment_df["text"].tolist()).lower()
    stop = set("the and a an to of it is this that for in with was i my on as not have had be but product coffee taste flavor chocolate".split())
    if extra:
        stop.update(x.strip().lower() for x in extra.split(",") if x.strip())
    words = re.findall(r"[a-z]{3,}", text)
    counts = pd.Series([w for w in words if w not in stop]).value_counts().head(30)
    spans = []
    for word, count in counts.items():
        size = int(11 + min(28, count * 2.2))
        spans.append(html.Span(word + " ", style={"fontSize": f"{size}px", "fontWeight": 600 if count > 2 else 400, "margin": "3px", "display": "inline-block"}))
    return html.Div(spans, style={"padding":"8px","textAlign":"center","lineHeight":"1.4","overflow":"hidden","height":"100%"})


# ----------------------------
# Layout
# ----------------------------
app = Dash(__name__, suppress_callback_exceptions=True, title="FMCG Dashboard")
server = app.server

header = html.Div([
    html.Div([html.Span("FMCG", style={"color": PRIMARY}), html.Span(" Dashboard", style={"color": ORANGE})], style={"fontSize":"22px","fontWeight":"700"}),
    dcc.Tabs(id="tabs", value="eda", children=[
        dcc.Tab(label="EDA", value="eda"),
        dcc.Tab(label="Model", value="model"),
        dcc.Tab(label="Interpret", value="interpret"),
        dcc.Tab(label="Sentiment", value="sentiment"),
    ], style={"width":"55vw","height":"6vh"})
], style={"height":"7vh","display":"flex","alignItems":"center","justifyContent":"space-between","padding":"0 1vw","backgroundColor":WHITE,"borderBottom":"1px solid #dfe5eb","boxSizing":"border-box"})


def eda_layout():
    kpis = [
        ("Total Partners", f"{len(master):,}", PRIMARY),
        ("Total Responders", f"{master.response.sum():,}", ORANGE),
        ("Response Rate", f"{master.response.mean():.2%}", GREEN),
        ("Avg Sales 2021", f"${master.total_sales_2021.mean()/1000:.1f}K", PINK),
        ("Avg Sales 2022", f"${master.total_sales_2022.mean()/1000:.1f}K", LIME),
        ("Avg B1 Sales 2022", f"${master.brand_B1_sales_2022.mean()/1000:.1f}K", YELLOW),
    ]
    return html.Div([
        html.Div([html.Div([html.Div(k, style={"fontSize":"11px"}), html.Div(v, style={"fontSize":"20px","fontWeight":"700","marginTop":"3px"})], style={**CARD,"backgroundColor":c,"color":WHITE,"textAlign":"center","justifyContent":"center","display":"flex","flexDirection":"column"}) for k,v,c in kpis], style={"height":"15vh","display":"grid","gridTemplateColumns":"repeat(6,1fr)","gap":"0.4vw"}),
        html.Div([
            html.Div([dcc.Dropdown(id="eda-feature", options=[{"label":c.replace("_"," ").title(),"value":c} for c in ["nps","n_comp","n_yrs","total_sales_2021","total_sales_2022"]], value="nps", clearable=False), dcc.Graph(id="eda-box", style=GRAPH, config={"displayModeBar":False})], style=CARD),
            html.Div([dcc.Dropdown(id="rr-feature", options=[{"label":c.replace("_"," ").title(),"value":c} for c in ["loyalty","rewards","portal","email","sms","call","Region","active_last_quarter","active_last_quarter_B1"]], value="loyalty", clearable=False), dcc.Graph(id="rr-chart", style=GRAPH, config={"displayModeBar":False})], style=CARD),
            html.Div([dcc.Graph(figure=monthly_fig(), style=GRAPH, config={"displayModeBar":False})], style=CARD),
        ], style={"height":"35vh","display":"grid","gridTemplateColumns":"1fr 1fr 1fr","gap":"0.4vw","marginTop":"0.5vh"}),
        html.Div([
            html.Div([dcc.Graph(id="brand-chart", figure=top_brand_fig(), style=GRAPH, config={"displayModeBar":False}), html.Div(id="brand-drill", style={"height":"8vh","fontSize":"11px","overflowY":"auto"})], style=CARD),
            html.Div([dcc.Graph(figure=top_partners_fig(), style=GRAPH, config={"displayModeBar":False})], style=CARD),
            html.Div([dcc.Graph(figure=top15_fig(), style=GRAPH, config={"displayModeBar":False})], style=CARD),
        ], style={"height":"40vh","display":"grid","gridTemplateColumns":"1fr 1fr 1fr","gap":"0.4vw","marginTop":"0.5vh"})
    ], style={"height":"93vh","padding":"0.5vh 0.5vw","boxSizing":"border-box","backgroundColor":BG,"overflow":"hidden"})


def model_layout():
    auc_df = pd.DataFrame([{"Model":n,"AUC":m["auc"]} for n,m in metrics.items()]).sort_values("AUC", ascending=False)
    return html.Div([
        html.Div([html.Div([html.Label("Loyalty"),dcc.Input(id="p-loyalty",value=1,type="number",min=0,max=1)],style={"flex":1}),html.Div([html.Label("NPS"),dcc.Input(id="p-nps",value=7,type="number",min=0,max=10)],style={"flex":1}),html.Div([html.Label("Years"),dcc.Input(id="p-years",value=5,type="number",min=0)],style={"flex":1}),html.Div([html.Label("Email"),dcc.Input(id="p-email",value=1,type="number",min=0,max=1)],style={"flex":1}),html.Div([html.Label("SMS"),dcc.Input(id="p-sms",value=1,type="number",min=0,max=1)],style={"flex":1}),html.Div([html.Div("Predicted Response"),html.Div(id="prediction",style={"fontSize":"20px","fontWeight":"700","color":PRIMARY})],style={**CARD,"backgroundColor":"#eef3f7","textAlign":"center","flex":1.5})],style={"height":"14vh","display":"flex","gap":"0.6vw","alignItems":"center"}),
        html.Div([html.Div([dcc.Dropdown(id="model-select",options=[{"label":n,"value":n} for n in models],value="Decision Tree",clearable=False),dcc.Graph(id="roc",style={"height":"32vh"},config={"displayModeBar":False})],style=CARD),html.Div([dcc.Graph(id="cm",style={"height":"25vh"},config={"displayModeBar":False}),html.Pre(id="report",style={"height":"13vh","overflowY":"auto","fontSize":"10px","margin":0})],style=CARD)],style={"height":"40vh","display":"grid","gridTemplateColumns":"1.4fr 1fr","gap":"0.5vw"}),
        html.Div([html.Div([html.Div("Sensitivity"),html.H3(id="sens",style={"margin":"2px"}),html.Div("Specificity"),html.H3(id="spec",style={"margin":"2px"}),html.Div("Threshold"),html.H3(id="threshold",style={"margin":"2px","color":ORANGE})],style={**CARD,"display":"flex","justifyContent":"space-around","alignItems":"center"}),html.Div([dash_table.DataTable(data=auc_df.to_dict("records"),columns=[{"name":c,"id":c} for c in auc_df.columns],style_table={"height":"13vh","overflowY":"auto"},style_cell={"padding":"6px","fontSize":"11px"},style_header={"backgroundColor":PRIMARY,"color":"white"})],style=CARD)],style={"height":"30vh","display":"grid","gridTemplateColumns":"1fr 1.5fr","gap":"0.5vw","marginTop":"0.5vh"})
    ], style={"height":"93vh","padding":"0.5vh 0.5vw","backgroundColor":BG,"boxSizing":"border-box","overflow":"hidden"})


def interpret_layout():
    options=[{"label":x,"value":x} for x in master.ChannelPartnerID]
    return html.Div([
        html.Div([html.Div([html.Label("Channel Partner ID"),dcc.Dropdown(id="partner",options=options,value=options[0]["value"],searchable=True),html.Div(id="partner-info",style={"marginTop":"1vh"})],style={**CARD,"padding":"0.8vh 0.7vw"}),html.Div([html.H4("Individual Feature Contribution",style={"margin":"0","textAlign":"center"}),dcc.Graph(id="local-contrib",style=GRAPH,config={"displayModeBar":False})],style=CARD)],style={"height":"40vh","display":"grid","gridTemplateColumns":"25% 75%","gap":"0.5vw"}),
        html.Div([html.Div([html.H4("Global Feature Importance",style={"margin":"0","textAlign":"center"}),dcc.Graph(id="global-importance",figure=blank_fig(),style=GRAPH,config={"displayModeBar":False})],style=CARD),html.Div([html.H4("Local Surrogate Explanation",style={"margin":"0","textAlign":"center"}),dcc.Graph(id="local-surrogate",style=GRAPH,config={"displayModeBar":False})],style=CARD)],style={"height":"50vh","display":"grid","gridTemplateColumns":"1fr 1fr","gap":"0.5vw","marginTop":"0.5vh"})
    ], style={"height":"93vh","padding":"0.5vh 0.5vw","backgroundColor":BG,"boxSizing":"border-box","overflow":"hidden"})


def sentiment_layout():
    counts=sentiment_df["sentiment_label"].value_counts().reindex(["Positive","Negative","Neutral"],fill_value=0)
    score=((sentiment_df.compound_sentiment.mean()+1)*2)+1
    donut=go.Figure(go.Pie(labels=counts.index,values=counts.values,hole=.55,marker_colors=[GREEN,ORANGE,YELLOW]))
    donut.update_layout(margin=dict(l=5,r=5,t=25,b=5),showlegend=True,title="Sentiment Breakdown",height=220)
    gauge=go.Figure(go.Indicator(mode="gauge+number",value=score,title={"text":"Overall Sentiment Level"},gauge={"axis":{"range":[1,5]},"steps":[{"range":[1,2.5],"color":"#f5b7b1"},{"range":[2.5,3.5],"color":"#f9e79f"},{"range":[3.5,5],"color":"#abebc6"}]}))
    gauge.update_layout(margin=dict(l=10,r=10,t=35,b=5),height=220)
    hist=px.histogram(sentiment_df,x="compound_sentiment",color="sentiment_label",nbins=18,color_discrete_map={"Positive":GREEN,"Negative":ORANGE,"Neutral":YELLOW},title="Score Distribution")
    hist.update_layout(margin=dict(l=30,r=10,t=35,b=30),height=280)
    comp=complaints_df.head(4)
    bar=px.bar(comp,x="Category",y="Count",color_discrete_sequence=[ORANGE],title="Top 4 Complaints")
    bar.update_layout(margin=dict(l=30,r=10,t=35,b=30),height=280)
    return html.Div([
        html.Div([html.Div([dcc.Graph(figure=gauge,style=GRAPH,config={"displayModeBar":False})],style=CARD),html.Div([dcc.Graph(figure=donut,style=GRAPH,config={"displayModeBar":False})],style=CARD)],style={"height":"25vh","display":"grid","gridTemplateColumns":"1fr 1fr","gap":"0.5vw"}),
        html.Div([html.Div([dcc.Graph(figure=hist,style=GRAPH,config={"displayModeBar":False})],style=CARD),html.Div([html.Div("Feedback Word Cloud",style={"fontWeight":"700","textAlign":"center"}),html.Div(id="cloud",children=word_cloud_component(),style={"height":"25vh"}),html.Div([html.Label("Add stopwords: "),dcc.Input(id="stopwords",placeholder="brand, product",style={"width":"65%"})],style={"fontSize":"11px"})],style=CARD)],style={"height":"34vh","display":"grid","gridTemplateColumns":"1fr 1fr","gap":"0.5vw","marginTop":"0.5vh"}),
        html.Div([html.Div([html.Div([html.B("Top Feedback"),dcc.Dropdown(id="feedback-type",options=[{"label":x,"value":x} for x in ["Positive","Negative","Neutral"]],value="Positive",clearable=False,style={"width":"130px"})],style={"display":"flex","justifyContent":"space-between"}),html.Div(id="feedback-table",style={"marginTop":"0.5vh"})],style=CARD),html.Div([dcc.Graph(figure=bar,style=GRAPH,config={"displayModeBar":False})],style=CARD)],style={"height":"34vh","display":"grid","gridTemplateColumns":"1fr 1fr","gap":"0.5vw","marginTop":"0.5vh"})
    ],style={"height":"93vh","padding":"0.5vh 0.5vw","backgroundColor":BG,"boxSizing":"border-box","overflow":"hidden"})


app.layout = html.Div([header, html.Div(id="page", children=eda_layout())], style={"height":"100vh","width":"100vw","overflow":"hidden","fontFamily":"Arial,sans-serif","color":TEXT})

# ----------------------------
# Callbacks
# ----------------------------
@app.callback(Output("page","children"), Input("tabs","value"))
def switch_page(tab):
    return {"eda": eda_layout, "model": model_layout, "interpret": interpret_layout, "sentiment": sentiment_layout}.get(tab, eda_layout)()

@app.callback(Output("eda-box","figure"), Input("eda-feature","value"))
def update_box(col):
    return box_fig(col)

@app.callback(Output("rr-chart","figure"), Input("rr-feature","value"))
def update_rr(col):
    return response_rate_fig(col)

@app.callback(Output("brand-drill","children"), Input("brand-chart","clickData"))
def brand_drill(click):
    if not click:
        return "Click a brand bar to see its top 5 partners."
    brand = click["points"][0].get("customdata")
    if not brand:
        return "Select a brand bar."
    d = transaction_df[transaction_df["Brand"] == brand].groupby("ChannelPartnerID")["Sales"].sum().nlargest(5).reset_index()
    return html.Table([html.Tr([html.Th("Partner ID"),html.Th("Sales")])] + [html.Tr([html.Td(r.ChannelPartnerID),html.Td(f"${r.Sales:,.0f}")]) for r in d.itertuples()], style={"width":"100%","fontSize":"10px"})

@app.callback(Output("prediction","children"), [Input("p-loyalty","value"),Input("p-nps","value"),Input("p-years","value"),Input("p-email","value"),Input("p-sms","value")])
def predict(l,n,yv,e,s):
    row=pd.DataFrame([{"loyalty":l or 0,"nps":n or 0,"n_yrs":yv or 0,"email":e or 0,"sms":s or 0}])
    p=float(single_model.predict_proba(row)[0,1])
    return f"{'Yes' if p >= .5 else 'No'} ({p:.1%})"

@app.callback([Output("roc","figure"),Output("cm","figure"),Output("report","children"),Output("sens","children"),Output("spec","children"),Output("threshold","children")],Input("model-select","value"))
def model_eval(name):
    m=metrics[name]
    roc=go.Figure(go.Scatter(x=m["fpr"],y=m["tpr"],mode="lines",name=f"AUC={m['auc']:.2f}",line=dict(color=PRIMARY,width=3)))
    roc.add_shape(type="line",x0=0,y0=0,x1=1,y1=1,line=dict(dash="dash",color="#888"))
    roc.update_layout(title=f"ROC Curve: {name}",xaxis_title="False Positive Rate",yaxis_title="True Positive Rate",margin=dict(l=35,r=10,t=38,b=30),height=300)
    z=np.array(m["cm"])
    cm=go.Figure(go.Heatmap(z=z,x=["Predicted 0","Predicted 1"],y=["Actual 0","Actual 1"],colorscale="Blues",showscale=False,text=z,texttemplate="%{text}"))
    cm.update_layout(title="Confusion Matrix",margin=dict(l=30,r=10,t=35,b=25),height=220)
    report=pd.DataFrame(m["report"]).transpose().round(3).to_string()
    return roc,cm,report,f"{m['sensitivity']:.3f}",f"{m['specificity']:.3f}",f"{m['threshold']:.3f}"

@app.callback([Output("local-contrib","figure"),Output("local-surrogate","figure"),Output("partner-info","children"),Output("global-importance","figure")],Input("partner","value"))
def explain(pid):
    row=master[master.ChannelPartnerID==str(pid)]
    if row.empty:
        return blank_fig(),blank_fig(),"Partner not found",blank_fig()
    r=row.iloc[0]
    # Model is intentionally transparent: coefficients of the fitted logistic model become the SHAP-style contribution view.
    pipe=single_model
    transformed=pipe.named_steps["prep"].transform(row[single_features])
    coef=pipe.named_steps["model"].coef_[0]
    vals=np.asarray(transformed)[0]
    contrib=coef*vals
    local=pd.DataFrame({"Feature":single_features,"Contribution":contrib})
    local=local.reindex(local.Contribution.abs().sort_values(ascending=False).index)
    f1=px.bar(local,x="Contribution",y="Feature",orientation="h",color="Contribution",color_continuous_scale="RdBu",title=f"Partner {pid} contribution")
    f1.update_layout(margin=dict(l=40,r=10,t=35,b=25),height=280,coloraxis_showscale=False)
    f2=px.bar(local.sort_values("Contribution"),x="Contribution",y="Feature",orientation="h",title="Local surrogate explanation",color_discrete_sequence=[GREEN])
    f2.update_layout(margin=dict(l=40,r=10,t=35,b=25),height=280)
    p=float(single_model.predict_proba(row[single_features])[0,1])
    info=[html.P([html.B("Actual Outcome: "),"Responded" if int(r.response)==1 else "No Response"]),html.P([html.B("Response Probability: "),f"{p:.1%}"])]
    # Global importance from absolute standardized logistic coefficients.
    g=pd.DataFrame({"Feature":single_features,"Importance":np.abs(coef)}).sort_values("Importance")
    f3=px.bar(g,x="Importance",y="Feature",orientation="h",title="Global Feature Importance",color_discrete_sequence=[PRIMARY])
    f3.update_layout(margin=dict(l=40,r=10,t=35,b=25),height=300)
    return f1,f2,info,f3

@app.callback(Output("cloud","children"),Input("stopwords","value"))
def cloud(extra):
    return word_cloud_component(extra or "")

@app.callback(Output("feedback-table","children"),Input("feedback-type","value"))
def feedback_table(kind):
    d=sentiment_df[sentiment_df.sentiment_label==kind].sort_values("compound_sentiment",ascending=(kind=="Negative")).head(5)
    return dash_table.DataTable(data=d[["text","compound_sentiment"]].rename(columns={"text":"Comment","compound_sentiment":"Score"}).to_dict("records"),columns=[{"name":"Comment","id":"Comment"},{"name":"Score","id":"Score"}],style_table={"height":"24vh","overflowY":"auto"},style_cell={"fontSize":"10px","padding":"5px","textAlign":"left","whiteSpace":"normal"},style_header={"backgroundColor":PRIMARY,"color":"white"})


if __name__ == "__main__":
    port=int(os.environ.get("PORT",8000))
    app.run(host="0.0.0.0",port=port,debug=False)
