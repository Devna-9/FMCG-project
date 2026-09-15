# app.py
# FMCG Dashboard - Render-ready Dash application
# Built from the supplied campaign, transaction, lookup and feedback files.

from __future__ import annotations

import base64
import os
import re
from collections import Counter
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html, dash_table
from plotly.subplots import make_subplots

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, classification_report, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from wordcloud import WordCloud

# Optional XAI dependencies. The dashboard still starts if one of these
# packages is unavailable, and falls back to transparent coefficient-based
# explanations instead of crashing at import time.
try:
    import shap
except Exception:
    shap = None

try:
    from lime.lime_tabular import LimeTabularExplainer
except Exception:
    LimeTabularExplainer = None

# -----------------------------
# Paths and constants
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent

PRIMARY_BLUE = "#4a75a7"
ACCENT_ORANGE = "#e37b00"
COLORS = px.colors.qualitative.Set2
BG = "#f5f7fa"
CARD = "#ffffff"
TEXT = "#34495e"

DATA_FILES = {
    "campaign_response": BASE_DIR / "Campaign Response  Data.csv",
    "campaign_details": BASE_DIR / "Campaign Details.csv",
    "lookup": BASE_DIR / "MasterLookUp.csv",
    "transactions": BASE_DIR / "Transaction data.csv",
    "feedback": BASE_DIR / "74responses.txt",
}

BRAND_LABELS = {
    "B1": "☕ B1",
    "B2": "🍪 B2",
    "B3": "🧃 B3",
    "B4": "🍫 B4",
    "B5": "🍨 B5",
    "B6": "B6",
    "B7": "💧 B7",
}

EDA_FEATURES = ["nps", "n_comp", "n_yrs", "total_sales_2021", "total_sales_2022"]
RR_FEATURES = [
    "loyalty", "rewards", "portal", "email", "sms", "call",
    "Region", "active_last_quarter", "active_last_quarter_B1",
]

INFERENCE_TEXT = {
    "nps": "Responders have a higher median NPS score, suggesting stronger customer advocacy is associated with campaign response.",
    "n_comp": "Responder and non-responder complaint distributions can be compared to assess whether complaint burden is associated with response.",
    "n_yrs": "Tenure differences show whether longer partner relationships are associated with stronger campaign response.",
    "total_sales_2021": "The 2021 sales distribution compares campaign response across lower- and higher-sales partners.",
    "total_sales_2022": "The 2022 sales distribution compares campaign response across lower- and higher-sales partners.",
}

SINGLE_FEATURES = ["loyalty", "nps", "n_yrs", "email", "sms"]
FULL_NUMERIC = [
    "n_comp", "nps", "n_yrs", "total_sales_2021", "total_sales_2022",
    "brand_B1_sales_2022", "buying_frequency_2022",
    "brand_engagement_2022", "buying_frequency_B1_2022",
    "brand_B1_contribution_2022",
]
FULL_BASE_FEATURES = [
    "loyalty", "nps", "n_yrs", "email", "sms", "call",
    "brand_B1_contribution_2022", "n_comp", "portal",
    "total_sales_2021", "total_sales_2022", "brand_B1_sales_2022",
    "buying_frequency_2022", "brand_engagement_2022",
    "buying_frequency_B1_2022",
]
XAI_FEATURES = [
    "loyalty", "nps", "n_yrs", "email", "sms", "call",
    "brand_B1_contribution_2022",
]
XAI_NUMERIC = ["nps", "n_yrs", "brand_B1_contribution_2022"]

MODEL_THRESHOLDS = {
    "Logistic Regression": 0.3947,
    "Naïve Bayes": 0.3232,
    "Decision Tree": 0.3375,
    "Random Forest": 0.3915,
}


# -----------------------------
# Utility helpers
# -----------------------------

def require_files() -> None:
    missing = [str(p.name) for p in DATA_FILES.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing dashboard data files: " + ", ".join(missing) +
            ". Put these files in the same GitHub repository as app.py."
        )


def safe_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df


def b64_png(fig_or_plot_func, figsize=(9, 4)) -> str:
    """Create a PNG from a matplotlib plotting function."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=figsize)
    fig_or_plot_func()
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=130)
    plt.close()
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def figure_to_image(fig) -> html.Img:
    # For matplotlib figures saved into BytesIO.
    import matplotlib.pyplot as plt
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=130)
    plt.close(fig)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return html.Img(
        src=f"data:image/png;base64,{encoded}",
        style={"width": "100%", "height": "100%", "objectFit": "contain"},
    )


# -----------------------------
# Data preparation
# -----------------------------

def load_data():
    require_files()

    campaign_response = pd.read_csv(
        DATA_FILES["campaign_response"], dtype={"ChannelPartnerID": "string"}
    )
    campaign_details = pd.read_csv(
        DATA_FILES["campaign_details"], dtype={"ChannelPartnerID": "string"}
    )
    lookup = pd.read_csv(
        DATA_FILES["lookup"], dtype={"ChannelPartnerID": "string"}
    )
    transactions = pd.read_csv(
        DATA_FILES["transactions"], dtype={"ChannelPartnerID": "string"}
    )
    feedback = pd.read_csv(
        DATA_FILES["feedback"], header=None, names=["text"], dtype="string"
    )

    # Make the transaction fields predictable.
    transactions["Month"] = pd.to_numeric(transactions["Month"], errors="coerce").fillna(0).astype(int)
    transactions["Year"] = pd.to_numeric(transactions["Year"], errors="coerce").fillna(0).astype(int)
    transactions["Sales"] = pd.to_numeric(transactions["Sales"], errors="coerce").fillna(0.0)
    transactions["Brand"] = transactions["Brand"].astype("string")

    # Build the final master table directly from the four supplied source files.
    master = (
        campaign_response
        .merge(campaign_details, on="ChannelPartnerID", how="left")
        .merge(lookup, on="ChannelPartnerID", how="left")
    )

    master["ChannelPartnerID"] = master["ChannelPartnerID"].astype("string")

    numeric_master = [
        "response", "n_comp", "loyalty", "portal", "rewards", "nps", "n_yrs",
        "email", "sms", "call"
    ]
    master = safe_numeric(master, numeric_master)

    # Partner-level sales and activity features used by the original dashboard.
    sales_2021 = transactions.loc[transactions["Year"].eq(2021)].groupby("ChannelPartnerID")["Sales"].sum()
    sales_2022 = transactions.loc[transactions["Year"].eq(2022)].groupby("ChannelPartnerID")["Sales"].sum()
    b1_2022 = transactions.loc[
        transactions["Year"].eq(2022) & transactions["Brand"].eq("B1")
    ].groupby("ChannelPartnerID")["Sales"].sum()

    freq_2022 = transactions.loc[transactions["Year"].eq(2022)].groupby("ChannelPartnerID")["Month"].nunique()
    engagement_2022 = transactions.loc[transactions["Year"].eq(2022)].groupby("ChannelPartnerID")["Brand"].nunique()
    freq_b1_2022 = transactions.loc[
        transactions["Year"].eq(2022) & transactions["Brand"].eq("B1")
    ].groupby("ChannelPartnerID")["Month"].nunique()

    master["total_sales_2021"] = master["ChannelPartnerID"].map(sales_2021).fillna(0)
    master["total_sales_2022"] = master["ChannelPartnerID"].map(sales_2022).fillna(0)
    master["brand_B1_sales_2022"] = master["ChannelPartnerID"].map(b1_2022).fillna(0)
    master["buying_frequency_2022"] = master["ChannelPartnerID"].map(freq_2022).fillna(0)
    master["brand_engagement_2022"] = master["ChannelPartnerID"].map(engagement_2022).fillna(0)
    master["buying_frequency_B1_2022"] = master["ChannelPartnerID"].map(freq_b1_2022).fillna(0)

    master["brand_B1_contribution_2022"] = np.where(
        master["total_sales_2022"] > 0,
        master["brand_B1_sales_2022"] / master["total_sales_2022"],
        0.0,
    )

    # Quarter was not present in the supplied transaction CSV, so derive Q4
    # from months 10-12 rather than assuming a missing column exists.
    q4 = transactions.loc[
        transactions["Year"].eq(2022) & transactions["Month"].between(10, 12)
    ].groupby("ChannelPartnerID")["Sales"].sum()
    q4_b1 = transactions.loc[
        transactions["Year"].eq(2022)
        & transactions["Month"].between(10, 12)
        & transactions["Brand"].eq("B1")
    ].groupby("ChannelPartnerID")["Sales"].sum()

    master["active_last_quarter"] = (
        master["ChannelPartnerID"].map(q4).fillna(0).gt(0).map({True: "Yes", False: "No"})
    )
    master["active_last_quarter_B1"] = (
        master["ChannelPartnerID"].map(q4_b1).fillna(0).gt(0).map({True: "Yes", False: "No"})
    )

    return master, transactions, feedback


# -----------------------------
# Sentiment
# -----------------------------

def sentiment_setup():
    """Use VADER when available; otherwise use a deterministic fallback."""
    try:
        import nltk
        from nltk.sentiment.vader import SentimentIntensityAnalyzer

        try:
            nltk.data.find("sentiment/vader_lexicon.zip")
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)

        analyzer = SentimentIntensityAnalyzer()
        return analyzer
    except Exception:
        return None


def fallback_sentiment(text: str) -> float:
    positive = {
        "good", "great", "excellent", "delicious", "love", "liked", "tasty",
        "amazing", "awesome", "pleasant", "perfect", "fresh", "nice", "yum",
        "fantastic", "worth", "happy", "fabulous", "rich", "creamy",
    }
    negative = {
        "bad", "awful", "expensive", "overpriced", "horrid", "gross", "disappointed",
        "rubbish", "poor", "yuck", "spoiled", "weak", "chalky", "ridiculous",
        "leaking", "leak", "pricey", "hate", "worst", "awful", "funny",
    }
    words = re.findall(r"[a-z']+", str(text).lower())
    if not words:
        return 0.0
    pos = sum(w in positive for w in words)
    neg = sum(w in negative for w in words)
    return float(np.clip((pos - neg) / max(1, len(words) ** 0.5), -1, 1))


def prepare_sentiment(feedback: pd.DataFrame):
    analyzer = sentiment_setup()
    df = feedback.copy()
    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df = df[df["text"].ne("")].copy()

    if analyzer is not None:
        df["compound_sentiment"] = df["text"].apply(
            lambda x: analyzer.polarity_scores(x)["compound"]
        )
    else:
        df["compound_sentiment"] = df["text"].apply(fallback_sentiment)

    df["sentiment_label"] = np.select(
        [df["compound_sentiment"] >= 0.05, df["compound_sentiment"] <= -0.05],
        ["Positive", "Negative"],
        default="Neutral",
    )

    return df


COMPLAINT_KEYWORDS = {
    "Taste/Flavor": [
        "taste", "flavor", "sweet", "sweetener", "mocha", "chocolate",
        "delicious", "tasty", "chalky", "weak",
    ],
    "Packaging": [
        "packaging", "package", "bottle", "lid", "seal", "wrap", "pump",
        "leaking", "leak", "box",
    ],
    "Price": [
        "expensive", "price", "priced", "cheap", "cost", "pricey", "overpriced",
    ],
    "Mixing": [
        "mix", "mixed", "mixes", "stir", "stirring", "dissolve", "clump",
        "clumpy", "blend", "blended", "beads",
    ],
}


def complaint_counts(sentiment_df: pd.DataFrame) -> pd.DataFrame:
    negative = sentiment_df.loc[sentiment_df["sentiment_label"].eq("Negative")]
    counts = Counter()

    for text in negative["text"]:
        words = set(re.findall(r"[a-z]+", text.lower()))
        for category, keywords in COMPLAINT_KEYWORDS.items():
            if any(k in words or k in text.lower() for k in keywords):
                counts[category] += 1

    out = pd.DataFrame(
        [{"Category": k, "Count": v} for k, v in counts.items()]
    )
    if out.empty:
        out = pd.DataFrame({"Category": [], "Count": []})
    return out.sort_values("Count", ascending=False).head(4)


def make_wordcloud(sentiment_df: pd.DataFrame, extra_stopwords=None) -> str:
    stop_words = {
        "the", "and", "a", "an", "to", "of", "it", "is", "this", "for",
        "in", "i", "my", "with", "on", "was", "but", "that", "have", "had",
        "be", "as", "so", "very", "not", "you", "they", "we", "me",
        "coffee", "taste", "chocolate", "flavor", "common", "milk",
    }
    if extra_stopwords:
        stop_words.update(
            s.strip().lower() for s in extra_stopwords if s.strip()
        )

    text = " ".join(sentiment_df["text"].tolist())
    tokens = [
        w for w in re.findall(r"[A-Za-z']+", text.lower())
        if w not in stop_words and len(w) > 2
    ]
    if not tokens:
        return ""

    wc = WordCloud(
        background_color="white",
        width=1100,
        height=500,
        colormap="Set2",
        prefer_horizontal=0.9,
    ).generate(" ".join(tokens))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(11, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# -----------------------------
# Models
# -----------------------------

def train_models(master: pd.DataFrame):
    # Single-instance predictor
    scaler_single = StandardScaler()
    single_X = master[SINGLE_FEATURES].copy()
    single_X[["nps", "n_yrs"]] = scaler_single.fit_transform(single_X[["nps", "n_yrs"]])
    single_model = LogisticRegression(max_iter=5000, random_state=42)
    single_model.fit(single_X, master["response"])

    # Full model dataset
    proc = master.copy()
    for col in ["active_last_quarter", "active_last_quarter_B1"]:
        proc[col] = proc[col].map({"Yes": 1, "No": 0}).fillna(0).astype(float)

    X = pd.get_dummies(
        proc[FULL_BASE_FEATURES + ["Region"]],
        columns=["Region"],
        dtype=float,
    )
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = proc["response"].astype(int)

    # Keep a fixed feature list for stable inference.
    feature_columns = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    logistic_scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    numerical_for_scale = [c for c in FULL_NUMERIC if c in X.columns]
    X_train_scaled[numerical_for_scale] = logistic_scaler.fit_transform(
        X_train[numerical_for_scale]
    )
    X_test_scaled[numerical_for_scale] = logistic_scaler.transform(
        X_test[numerical_for_scale]
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=5000, random_state=42),
        "Naïve Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=20, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, max_depth=8, random_state=42, n_jobs=-1
        ),
    }

    results = {}
    fitted = {}

    for name, model in models.items():
        if name == "Logistic Regression":
            model.fit(X_train_scaled, y_train)
            probs = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            probs = model.predict_proba(X_test)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, probs)
        threshold = MODEL_THRESHOLDS[name]
        preds = (probs >= threshold).astype(int)
        cm = confusion_matrix(y_test, preds)
        report = classification_report(y_test, preds, output_dict=True, zero_division=0)

        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0

        results[name] = {
            "fpr": fpr,
            "tpr": tpr,
            "auc": auc(fpr, tpr),
            "cm": cm.tolist(),
            "report": report,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "threshold": threshold,
        }
        fitted[name] = model

    # XAI logistic model on the same 7-feature structure used by the supplied app.
    xai_scaler = StandardScaler()
    X_xai = master[XAI_FEATURES].copy()
    X_xai_scaled = X_xai.copy()
    X_xai_scaled[XAI_NUMERIC] = xai_scaler.fit_transform(X_xai[XAI_NUMERIC])

    xai_model = LogisticRegression(max_iter=5000, random_state=42)
    xai_model.fit(X_xai_scaled, master["response"])

    shap_explainer = None
    lime_explainer = None

    if shap is not None:
        try:
            background = X_xai_scaled.sample(
                min(200, len(X_xai_scaled)), random_state=42
            )
            shap_explainer = shap.LinearExplainer(xai_model, background)
        except Exception:
            shap_explainer = None

    if LimeTabularExplainer is not None:
        try:
            lime_explainer = LimeTabularExplainer(
                training_data=X_xai_scaled.values,
                feature_names=XAI_FEATURES,
                class_names=["No Response", "Response"],
                mode="classification",
                discretize_continuous=True,
                random_state=42,
            )
        except Exception:
            lime_explainer = None

    return {
        "single_scaler": scaler_single,
        "single_model": single_model,
        "feature_columns": feature_columns,
        "logistic_scaler": logistic_scaler,
        "models": fitted,
        "results": results,
        "xai_scaler": xai_scaler,
        "xai_model": xai_model,
        "xai_data": X_xai_scaled,
        "shap_explainer": shap_explainer,
        "lime_explainer": lime_explainer,
    }


# -----------------------------
# Figures
# -----------------------------

def chart_layout(fig, title=None, height=None):
    fig.update_layout(
        title=title or fig.layout.title.text,
        margin=dict(l=35, r=15, t=38, b=35),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(size=10, color=TEXT),
        height=height,
    )
    return fig


def build_figures(master: pd.DataFrame, transactions: pd.DataFrame):
    # Monthly response rate based on unique partners transacting each month.
    t = transactions.copy()
    t["response"] = t["ChannelPartnerID"].map(
        master.set_index("ChannelPartnerID")["response"]
    )
    monthly = (
        t.groupby(["Year", "Month", "ChannelPartnerID"])["response"]
        .first()
        .reset_index()
        .groupby(["Year", "Month"])["response"]
        .mean()
        .reset_index(name="monthly_response_rate")
    )
    monthly["Year_str"] = monthly["Year"].astype(str)

    fig_monthly = px.line(
        monthly,
        x="Month",
        y="monthly_response_rate",
        color="Year_str",
        markers=True,
        color_discrete_sequence=[COLORS[0], COLORS[1]],
        labels={"monthly_response_rate": "Monthly Response Rate", "Month": "Month"},
    )
    fig_monthly.update_yaxes(tickformat=".0%")
    chart_layout(fig_monthly, "Monthly Response Rate Trend")

    top_partners = (
        transactions.groupby("ChannelPartnerID", as_index=False)["Sales"]
        .sum()
        .sort_values("Sales", ascending=False)
        .head(5)
    )
    fig_top = go.Figure(
        go.Bar(
            y=top_partners["ChannelPartnerID"].astype(str),
            x=top_partners["Sales"],
            orientation="h",
            text=[f"${x/1000:.0f}K" for x in top_partners["Sales"]],
            textposition="inside",
            marker_color=COLORS[1],
            hovertemplate="<b>Partner: %{y}</b><br>Sales: $%{x:,.0f}<extra></extra>",
        )
    )
    fig_top.update_yaxes(autorange="reversed")
    chart_layout(fig_top, "Top Selling Partners")

    brand = (
        transactions.loc[transactions["Year"].eq(2022)]
        .groupby("Brand", as_index=False)["Sales"]
        .sum()
        .rename(columns={"Sales": "achieved"})
    )
    brand["target"] = brand["achieved"] * 1.75
    brand["label"] = brand["Brand"].map(BRAND_LABELS).fillna(brand["Brand"])

    fig_brand = go.Figure()
    fig_brand.add_bar(
        y=brand["label"],
        x=brand["target"],
        orientation="h",
        marker_color="#e9ecef",
        width=0.65,
        hoverinfo="skip",
        name="Target",
    )
    fig_brand.add_bar(
        y=brand["label"],
        x=brand["achieved"],
        orientation="h",
        marker_color=COLORS[0],
        width=0.38,
        customdata=brand["Brand"],
        text=[f"${x/1e6:.1f}M" for x in brand["achieved"]],
        textposition="inside",
        name="Achieved",
        hovertemplate="<b>%{customdata}</b><br>Achieved: $%{x:,.0f}<extra></extra>",
    )
    fig_brand.update_layout(barmode="overlay", showlegend=False)
    fig_brand.update_yaxes(autorange="reversed")
    chart_layout(fig_brand, "Brand Achievement (click a brand)")

    top15 = (
        master.loc[master["response"].eq(1)]
        .sort_values("brand_B1_contribution_2022", ascending=False)
        .head(15)
        .copy()
    )
    fig_top15 = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        subplot_titles=("By Region", "By Activity")
    )
    region_colors = [
        COLORS[i % len(COLORS)]
        for i in range(len(top15))
    ]
    activity_colors = [
        COLORS[0] if x == "Yes" else COLORS[1]
        for x in top15["active_last_quarter_B1"]
    ]

    fig_top15.add_trace(
        go.Bar(
            x=top15["ChannelPartnerID"].astype(str),
            y=top15["brand_B1_contribution_2022"],
            marker_color=region_colors,
            showlegend=False,
            hovertemplate="Partner: %{x}<br>B1 Contribution: %{y:.2%}<extra></extra>",
        ),
        row=1, col=1,
    )
    fig_top15.add_trace(
        go.Bar(
            x=top15["ChannelPartnerID"].astype(str),
            y=top15["brand_B1_contribution_2022"],
            marker_color=activity_colors,
            showlegend=False,
            hovertemplate="Partner: %{x}<br>B1 Contribution: %{y:.2%}<extra></extra>",
        ),
        row=2, col=1,
    )
    fig_top15.update_yaxes(tickformat=".0%", row=1, col=1)
    fig_top15.update_yaxes(tickformat=".0%", row=2, col=1)
    chart_layout(fig_top15, "Top 15 B1 Contributors", height=300)

    return {
        "monthly": fig_monthly,
        "top_partners": fig_top,
        "brand": fig_brand,
        "top15": fig_top15,
    }


# -----------------------------
# Layout
# -----------------------------

CARD_STYLE = {
    "backgroundColor": CARD,
    "border": "1px solid #dee2e6",
    "borderRadius": "6px",
    "padding": "6px",
    "boxSizing": "border-box",
    "overflow": "hidden",
}

APP_STYLE = {
    "minHeight": "100vh",
    "backgroundColor": BG,
    "fontFamily": "Arial, sans-serif",
    "color": TEXT,
}

GRAPH_CONFIG = {"displayModeBar": False, "responsive": True}


def kpi(title, value, color):
    return html.Div(
        [
            html.Div(title, style={"fontSize": "12px", "opacity": 0.95}),
            html.Div(value, style={"fontSize": "22px", "fontWeight": "700", "marginTop": "4px"}),
        ],
        style={
            "flex": "1",
            "backgroundColor": color,
            "color": "white",
            "borderRadius": "6px",
            "padding": "10px",
            "textAlign": "center",
            "minWidth": "130px",
        },
    )


def render_eda(figures, master, transactions):
    total_partners = len(master)
    total_responders = int(master["response"].sum())
    response_rate = master["response"].mean()
    avg21 = master["total_sales_2021"].mean()
    avg22 = master["total_sales_2022"].mean()
    avgb1 = master["brand_B1_sales_2022"].mean()

    return html.Div(
        [
            html.Div(
                [
                    kpi("Total Partners", f"{total_partners:,}", "#5bb89f"),
                    kpi("Total Responders", f"{total_responders:,}", "#ff8a5c"),
                    kpi("Response Rate", f"{response_rate:.2%}", "#8ea2d0"),
                    kpi("Avg Sales 2021", f"${avg21/1000:.1f}K", "#dc7eb6"),
                    kpi("Avg Sales 2022", f"${avg22/1000:.1f}K", "#a5d84d"),
                    kpi("Avg B1 Sales 2022", f"${avgb1/1000:.1f}K", "#ffd333"),
                ],
                style={"display": "flex", "gap": "6px", "marginBottom": "6px", "flexWrap": "wrap"},
            ),
            html.Div(
                [
                    html.Div(
                        [
                            dcc.Dropdown(
                                id="eda-dropdown",
                                options=[
                                    {"label": c.replace("_", " ").title(), "value": c}
                                    for c in EDA_FEATURES
                                ],
                                value="nps",
                                clearable=False,
                            ),
                            dcc.Graph(id="eda-graph", config=GRAPH_CONFIG, style={"height": "calc(100% - 55px)"}),
                            html.Div(id="eda-inference", style={"fontSize": "12px", "background": "#f1f3f5", "padding": "6px"}),
                        ],
                        style={**CARD_STYLE, "flex": "1"},
                    ),
                    html.Div(
                        [
                            dcc.Dropdown(
                                id="rr-dropdown",
                                options=[
                                    {"label": c.replace("_", " ").title(), "value": c}
                                    for c in RR_FEATURES
                                ],
                                value="loyalty",
                                clearable=False,
                            ),
                            dcc.Graph(id="rr-bar-chart", config=GRAPH_CONFIG, style={"height": "calc(100% - 55px)"}),
                        ],
                        style={**CARD_STYLE, "flex": "1"},
                    ),
                    html.Div(
                        dcc.Graph(figure=figures["monthly"], config=GRAPH_CONFIG, style={"height": "100%"}),
                        style={**CARD_STYLE, "flex": "1"},
                    ),
                ],
                style={"display": "flex", "gap": "6px", "height": "360px", "marginBottom": "6px"},
            ),
            html.Div(
                [
                    html.Div(
                        [
                            dcc.Graph(
                                id="brand-achievement-graph",
                                figure=figures["brand"],
                                config=GRAPH_CONFIG,
                                style={"height": "70%"},
                            ),
                            html.Div(
                                "Click a brand bar to see the top five selling partners.",
                                id="brand-drilldown-box",
                                style={"fontSize": "12px", "overflowY": "auto", "padding": "5px"},
                            ),
                        ],
                        style={**CARD_STYLE, "flex": "1"},
                    ),
                    html.Div(
                        dcc.Graph(figure=figures["top_partners"], config=GRAPH_CONFIG, style={"height": "100%"}),
                        style={**CARD_STYLE, "flex": "1"},
                    ),
                    html.Div(
                        dcc.Graph(figure=figures["top15"], config=GRAPH_CONFIG, style={"height": "100%"}),
                        style={**CARD_STYLE, "flex": "1"},
                    ),
                ],
                style={"display": "flex", "gap": "6px", "height": "330px"},
            ),
        ],
        style={"padding": "6px", "overflowY": "auto"},
    )


def render_model(models_state):
    results = models_state["results"]
    auc_df = pd.DataFrame(
        [{"Model": name, "AUC": r["auc"]} for name, r in results.items()]
    ).sort_values("AUC", ascending=False)
    best = auc_df.iloc[0]

    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H4("Response Predictor (Logistic Regression)", style={"margin": "0 0 8px"}),
                            html.Div(
                                [
                                    html.Div([html.Label("Loyalty"), dcc.Input(id="input-loyalty", type="number", value=1, min=0, max=1, step=1, style={"width": "100%"})], style={"flex": 1}),
                                    html.Div([html.Label("NPS"), dcc.Input(id="input-nps", type="number", value=7, min=0, max=10, step=1, style={"width": "100%"})], style={"flex": 1}),
                                    html.Div([html.Label("Years"), dcc.Input(id="input-n-yrs", type="number", value=5, min=0, step=1, style={"width": "100%"})], style={"flex": 1}),
                                    html.Div([html.Label("Email"), dcc.Input(id="input-email", type="number", value=1, min=0, max=1, step=1, style={"width": "100%"})], style={"flex": 1}),
                                    html.Div([html.Label("SMS"), dcc.Input(id="input-sms", type="number", value=1, min=0, max=1, step=1, style={"width": "100%"})], style={"flex": 1}),
                                    html.Div(
                                        [html.Div("Predicted Response", style={"fontWeight": "700"}), html.Div(id="prediction-result-display", style={"fontSize": "22px", "fontWeight": "700", "color": PRIMARY_BLUE})],
                                        style={**CARD_STYLE, "backgroundColor": "#eef3f8", "border": f"2px solid {ACCENT_ORANGE}", "flex": 1.4, "textAlign": "center"},
                                    ),
                                ],
                                style={"display": "flex", "gap": "8px", "alignItems": "stretch"},
                            ),
                        ],
                        style={**CARD_STYLE, "flex": 1},
                    ),
                ],
                style={"display": "flex", "marginBottom": "6px"},
            ),
            html.Div(
                [
                    html.Div(
                        [
                            dcc.Dropdown(
                                id="model-selector-dropdown",
                                options=[{"label": n, "value": n} for n in results],
                                value="Decision Tree",
                                clearable=False,
                            ),
                            dcc.Graph(id="evaluation-roc-graph", config=GRAPH_CONFIG, style={"height": "calc(100% - 45px)"}),
                        ],
                        style={**CARD_STYLE, "flex": 1.25},
                    ),
                    html.Div(
                        [
                            dcc.Graph(id="evaluation-cm-heatmap", config=GRAPH_CONFIG, style={"height": "48%"}),
                            html.Pre(id="evaluation-report-text", style={"height": "46%", "overflow": "auto", "fontSize": "11px", "margin": "0", "background": "#f8f9fa", "padding": "5px"}),
                        ],
                        style={**CARD_STYLE, "flex": 1},
                    ),
                ],
                style={"display": "flex", "gap": "6px", "height": "430px", "marginBottom": "6px"},
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(["Sensitivity", html.B(id="summary-sensitivity")]),
                            html.Div(["Specificity", html.B(id="summary-specificity")]),
                            html.Div(["Opt. Threshold", html.B(id="summary-threshold", style={"color": ACCENT_ORANGE})]),
                        ],
                        style={**CARD_STYLE, "flex": 1, "display": "flex", "justifyContent": "space-around", "alignItems": "center"},
                    ),
                    html.Div(
                        [
                            dash_table.DataTable(
                                data=auc_df.round(4).to_dict("records"),
                                columns=[{"name": c, "id": c} for c in auc_df.columns],
                                style_table={"overflowX": "auto"},
                                style_cell={"padding": "5px", "fontSize": "12px"},
                                style_header={"backgroundColor": PRIMARY_BLUE, "color": "white", "fontWeight": "700"},
                            ),
                            html.Div(f"Best Model: {best['Model']} ({best['AUC']:.2f} AUC)", style={"fontWeight": "700", "color": "#278a4b", "padding": "5px"}),
                        ],
                        style={**CARD_STYLE, "flex": 1.5},
                    ),
                ],
                style={"display": "flex", "gap": "6px", "height": "110px"},
            ),
        ],
        style={"padding": "6px", "overflowY": "auto"},
    )


def render_interpret(master, models_state):
    ids = master["ChannelPartnerID"].astype(str).unique()
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H4("Channel Partner ID", style={"margin": "0 0 6px"}),
                            dcc.Dropdown(
                                id="interpret-partner-dropdown",
                                options=[{"label": x, "value": x} for x in ids],
                                value=ids[0] if len(ids) else None,
                                searchable=True,
                                clearable=False,
                            ),
                            html.Div(id="partner-prediction-info", style={"marginTop": "12px", "padding": "8px", "background": "#f8f9fa"}),
                        ],
                        style={**CARD_STYLE, "flex": "0 0 25%"},
                    ),
                    html.Div(
                        [
                            html.H4("Individual Feature Contribution (SHAP Waterfall)", style={"textAlign": "center", "margin": "0 0 5px"}),
                            html.Div(id="shap-waterfall-container", style={"height": "calc(100% - 35px)"}),
                        ],
                        style={**CARD_STYLE, "flex": "1"},
                    ),
                ],
                style={"display": "flex", "gap": "6px", "height": "350px", "marginBottom": "6px"},
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.H4("Global Feature Importance (SHAP)", style={"textAlign": "center", "margin": "0"}),
                            dcc.Graph(id="global-shap-graph", config=GRAPH_CONFIG, style={"height": "calc(100% - 35px)"}),
                        ],
                        style={**CARD_STYLE, "flex": 1},
                    ),
                    html.Div(
                        [
                            html.H4("Local Surrogate Explanation (LIME)", style={"textAlign": "center", "margin": "0"}),
                            html.Div(id="lime-explanation-container", style={"height": "calc(100% - 35px)"}),
                        ],
                        style={**CARD_STYLE, "flex": 1},
                    ),
                ],
                style={"display": "flex", "gap": "6px", "height": "420px"},
            ),
        ],
        style={"padding": "6px", "overflowY": "auto"},
    )


def render_sentiment(sentiment_df):
    counts = sentiment_df["sentiment_label"].value_counts().reindex(
        ["Positive", "Negative", "Neutral"], fill_value=0
    )
    avg_compound = sentiment_df["compound_sentiment"].mean()
    scaled_score = float(np.clip(((avg_compound + 1) * 2) + 1, 1, 5))
    pct = (counts / max(1, counts.sum()) * 100).round().astype(int)

    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=scaled_score,
            title={"text": "<b>Overall Sentiment Level</b><br><span style='font-size:0.8em'>(1=Negative, 5=Positive)</span>"},
            gauge={
                "axis": {"range": [1, 5]},
                "bar": {"color": PRIMARY_BLUE},
                "steps": [
                    {"range": [1, 2.5], "color": "#e74c3c"},
                    {"range": [2.5, 3.5], "color": "#f1c40f"},
                    {"range": [3.5, 5], "color": "#2ecc71"},
                ],
            },
        )
    )
    gauge.update_layout(margin=dict(l=10, r=10, t=30, b=5), paper_bgcolor="white")

    donut = go.Figure(
        go.Pie(
            labels=counts.index,
            values=counts.values,
            hole=0.45,
            marker={"colors": ["#2ecc71", "#e74c3c", "#f1c40f"]},
            textinfo="percent+label",
        )
    )
    donut.update_layout(margin=dict(l=0, r=0, t=5, b=0), paper_bgcolor="white", showlegend=False)

    hist = px.histogram(
        sentiment_df,
        x="compound_sentiment",
        color="sentiment_label",
        nbins=20,
        color_discrete_map={"Positive": "#2ecc71", "Negative": "#e74c3c", "Neutral": "#f1c40f"},
        title="Score Distribution",
    )
    chart_layout(hist, "Score Distribution")

    complaints = complaint_counts(sentiment_df)
    complaint_fig = px.bar(
        complaints,
        x="Category",
        y="Count",
        title="Top 4 Complaints",
        color_discrete_sequence=[ACCENT_ORANGE],
    )
    chart_layout(complaint_fig, "Top 4 Complaints")

    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [dcc.Graph(figure=gauge, config=GRAPH_CONFIG, style={"height": "100%"})],
                        style={**CARD_STYLE, "flex": 1},
                    ),
                    html.Div(
                        [
                            html.Div(
                                dcc.Graph(figure=donut, config=GRAPH_CONFIG, style={"height": "100%"}),
                                style={"flex": "0 0 65%", "height": "100%"},
                            ),
                            html.Div(
                                [
                                    html.Div(["👍 ", html.B(f"{pct['Positive']}%"), " Positive"]),
                                    html.Div(["👎 ", html.B(f"{pct['Negative']}%"), " Negative"]),
                                    html.Div(["😐 ", html.B(f"{pct['Neutral']}%"), " Neutral"]),
                                ],
                                style={"flex": "0 0 35%", "display": "flex", "flexDirection": "column", "justifyContent": "center", "gap": "8px"},
                            ),
                        ],
                        style={**CARD_STYLE, "flex": 1, "display": "flex"},
                    ),
                ],
                style={"display": "flex", "gap": "6px", "height": "220px", "marginBottom": "6px"},
            ),
            html.Div(
                [
                    html.Div(
                        [
                            dcc.Graph(figure=hist, config=GRAPH_CONFIG, style={"height": "100%"}),
                        ],
                        style={**CARD_STYLE, "flex": 1},
                    ),
                    html.Div(
                        [
                            html.H4("Feedback Word Cloud", style={"textAlign": "center", "margin": "0 0 4px"}),
                            html.Div(
                                [
                                    html.Label("Add Stopwords:"),
                                    dcc.Input(id="stopword-input", type="text", placeholder="comma separated (e.g., brand, product)", style={"flex": 1}),
                                ],
                                style={"display": "flex", "gap": "6px", "alignItems": "center"},
                            ),
                            html.Div(id="wordcloud-container", style={"height": "calc(100% - 55px)", "display": "flex", "justifyContent": "center"}),
                        ],
                        style={**CARD_STYLE, "flex": 1},
                    ),
                ],
                style={"display": "flex", "gap": "6px", "height": "360px", "marginBottom": "6px"},
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H4("Top Feedback", style={"margin": "0"}),
                                    dcc.Dropdown(
                                        id="sentiment-feedback-dropdown",
                                        options=[{"label": x, "value": x} for x in ["Positive", "Negative", "Neutral"]],
                                        value="Positive",
                                        clearable=False,
                                        style={"width": "130px"},
                                    ),
                                ],
                                style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"},
                            ),
                            html.Div(id="top-feedback-table-container", style={"height": "calc(100% - 45px)"}),
                        ],
                        style={**CARD_STYLE, "flex": 1},
                    ),
                    html.Div(
                        dcc.Graph(figure=complaint_fig, config=GRAPH_CONFIG, style={"height": "100%"}),
                        style={**CARD_STYLE, "flex": 1},
                    ),
                ],
                style={"display": "flex", "gap": "6px", "height": "300px"},
            ),
        ],
        style={"padding": "6px", "overflowY": "auto"},
    )


# -----------------------------
# Application initialization
# -----------------------------

master_df, transaction_df, sentiment_df = load_data()
sentiment_df = prepare_sentiment(sentiment_df)
models_state = train_models(master_df)
figures = build_figures(master_df, transaction_df)

app = Dash(
    __name__,
    title="FMCG Dashboard",
    suppress_callback_exceptions=True,
)
server = app.server

app.layout = html.Div(
    [
        html.Div(
            [
                html.H2(
                    [
                        html.Span("FMCG", style={"color": PRIMARY_BLUE, "fontWeight": "700"}),
                        html.Span(" Dashboard", style={"color": ACCENT_ORANGE}),
                    ],
                    style={"margin": 0},
                ),
                dcc.Tabs(
                    id="main-tabs-selector",
                    value="tab-eda",
                    children=[
                        dcc.Tab(label="EDA", value="tab-eda"),
                        dcc.Tab(label="Model", value="tab-model"),
                        dcc.Tab(label="Explainable AI", value="tab-interpret"),
                        dcc.Tab(label="Sentiment", value="tab-sentiment"),
                    ],
                ),
            ],
            style={
                "backgroundColor": "white",
                "padding": "8px 18px",
                "borderBottom": "1px solid #dee2e6",
                "display": "flex",
                "justifyContent": "space-between",
                "alignItems": "center",
                "position": "sticky",
                "top": 0,
                "zIndex": 10,
            },
        ),
        html.Div(id="main-tab-content"),
    ],
    style=APP_STYLE,
)


# -----------------------------
# Callbacks
# -----------------------------

@app.callback(
    Output("main-tab-content", "children"),
    Input("main-tabs-selector", "value"),
)
def update_tab(tab):
    if tab == "tab-eda":
        return render_eda(figures, master_df, transaction_df)
    if tab == "tab-model":
        return render_model(models_state)
    if tab == "tab-interpret":
        return render_interpret(master_df, models_state)
    if tab == "tab-sentiment":
        return render_sentiment(sentiment_df)
    return html.Div("Tab not found.")


@app.callback(
    Output("eda-graph", "figure"),
    Output("eda-inference", "children"),
    Input("eda-dropdown", "value"),
)
def update_eda(feature):
    fig = px.box(
        master_df,
        x="response",
        y=feature,
        color="response",
        color_discrete_sequence=[COLORS[0], COLORS[1]],
        labels={"response": "Response", feature: feature.replace("_", " ").title()},
    )
    fig.update_xaxes(tickvals=[0, 1], ticktext=["No", "Yes"])
    chart_layout(fig, feature.replace("_", " ").title())
    return fig, INFERENCE_TEXT.get(feature, "")


@app.callback(
    Output("rr-bar-chart", "figure"),
    Input("rr-dropdown", "value"),
)
def update_rr(feature):
    df = master_df.copy()
    if feature not in df.columns:
        return go.Figure()

    # Correct response-rate definition: responders / all partners in each group.
    grouped = (
        df.groupby(feature, dropna=False)["response"]
        .agg(total="count", responders="sum")
        .reset_index()
    )
    grouped["response_rate"] = np.where(
        grouped["total"] > 0,
        grouped["responders"] / grouped["total"] * 100,
        0,
    )
    grouped[feature] = grouped[feature].fillna("Missing").astype(str)

    fig = px.bar(
        grouped,
        x=feature,
        y="response_rate",
        text=grouped["response_rate"].map(lambda x: f"{x:.1f}%"),
        labels={feature: feature.replace("_", " ").title(), "response_rate": "Response Rate (%)"},
        color_discrete_sequence=[COLORS[0]],
    )
    fig.update_traces(textposition="outside")
    chart_layout(fig, f"Response Rate by {feature.replace('_', ' ').title()}")
    return fig


@app.callback(
    Output("brand-drilldown-box", "children"),
    Input("brand-achievement-graph", "clickData"),
)
def brand_drilldown(click_data):
    if not click_data:
        return "Click a brand bar to see the top five selling partners."

    point = click_data["points"][0]
    selected_brand = point.get("customdata")
    if selected_brand is None:
        return "Select a brand bar."

    df = (
        transaction_df.loc[transaction_df["Brand"].eq(selected_brand)]
        .groupby("ChannelPartnerID", as_index=False)["Sales"]
        .sum()
        .sort_values("Sales", ascending=False)
        .head(5)
    )
    rows = [
        html.Tr([
            html.Td(str(r["ChannelPartnerID"])),
            html.Td(f"${r['Sales']:,.0f}"),
        ])
        for _, r in df.iterrows()
    ]
    return html.Div([
        html.B(f"Top 5 Partners for {selected_brand}"),
        html.Table(
            [html.Thead(html.Tr([html.Th("Partner ID"), html.Th("Sales")]))]
            + [html.Tbody(rows)],
            style={"width": "100%", "fontSize": "12px"},
        ),
    ])


@app.callback(
    Output("prediction-result-display", "children"),
    Input("input-loyalty", "value"),
    Input("input-nps", "value"),
    Input("input-n-yrs", "value"),
    Input("input-email", "value"),
    Input("input-sms", "value"),
)
def predict_response(loyalty, nps, years, email, sms):
    values = [loyalty, nps, years, email, sms]
    if any(v is None for v in values):
        return "Enter all values"

    row = pd.DataFrame([{
        "loyalty": float(loyalty),
        "nps": float(nps),
        "n_yrs": float(years),
        "email": float(email),
        "sms": float(sms),
    }])
    row[["nps", "n_yrs"]] = models_state["single_scaler"].transform(row[["nps", "n_yrs"]])
    probability = models_state["single_model"].predict_proba(row[SINGLE_FEATURES])[0, 1]
    return f"{'Yes' if probability >= 0.5 else 'No'} ({probability:.1%})"


def cm_figure(cm, model_name):
    z = np.asarray(cm)
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=["Predicted: 0", "Predicted: 1"],
            y=["Actual: 0", "Actual: 1"],
            colorscale="Blues",
            showscale=False,
            text=z,
            texttemplate="%{text}",
        )
    )
    fig.update_layout(
        title=f"Confusion Matrix: {model_name}",
        margin=dict(l=40, r=10, t=35, b=35),
        paper_bgcolor="white",
        font=dict(size=10),
    )
    return fig


@app.callback(
    Output("evaluation-roc-graph", "figure"),
    Output("evaluation-cm-heatmap", "figure"),
    Output("evaluation-report-text", "children"),
    Output("summary-sensitivity", "children"),
    Output("summary-specificity", "children"),
    Output("summary-threshold", "children"),
    Input("model-selector-dropdown", "value"),
)
def update_model_evaluation(model_name):
    r = models_state["results"][model_name]
    roc_fig = go.Figure(
        go.Scatter(
            x=r["fpr"],
            y=r["tpr"],
            mode="lines",
            name=f"ROC (AUC={r['auc']:.2f})",
            line={"color": PRIMARY_BLUE, "width": 3},
        )
    )
    roc_fig.add_shape(
        type="line",
        x0=0, y0=0, x1=1, y1=1,
        line={"dash": "dash", "color": "#777"},
    )
    roc_fig.update_layout(
        title=f"ROC Curve: {model_name}",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        margin=dict(l=45, r=15, t=40, b=40),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    report_text = pd.DataFrame(r["report"]).transpose().round(3).to_string()
    return (
        roc_fig,
        cm_figure(r["cm"], model_name),
        report_text,
        f"{r['sensitivity']:.4f}",
        f"{r['specificity']:.4f}",
        f"{r['threshold']:.4f}",
    )


def coefficient_fallback(model, features, values):
    coef = model.coef_[0]
    contribution = coef * values
    order = np.argsort(np.abs(contribution))[::-1]
    fig = go.Figure(
        go.Bar(
            x=contribution[order],
            y=np.array(features)[order],
            orientation="h",
        )
    )
    fig.update_layout(
        title="Coefficient-based feature contribution (SHAP package unavailable)",
        margin=dict(l=30, r=10, t=45, b=25),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig


@app.callback(
    Output("shap-waterfall-container", "children"),
    Output("lime-explanation-container", "children"),
    Output("partner-prediction-info", "children"),
    Input("interpret-partner-dropdown", "value"),
)
def update_xai(partner_id):
    if partner_id is None:
        return "", "", ""

    raw = master_df.loc[
        master_df["ChannelPartnerID"].astype(str).eq(str(partner_id))
    ]
    if raw.empty:
        return "Partner not found.", "", ""

    x = raw.iloc[0][XAI_FEATURES].to_frame().T.copy()
    x_scaled = x.copy()
    x_scaled[XAI_NUMERIC] = models_state["xai_scaler"].transform(x[XAI_NUMERIC])
    probability = models_state["xai_model"].predict_proba(x_scaled)[0, 1]
    actual = "Responded" if int(raw.iloc[0]["response"]) == 1 else "No Response"

    info = html.Div([
        html.B("Actual Outcome: "), actual,
        html.Br(),
        html.B("Response Probability: "), f"{probability:.2%}",
    ])

    # SHAP waterfall
    if models_state["shap_explainer"] is not None and shap is not None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            sv = models_state["shap_explainer"](x_scaled)
            fig = plt.figure(figsize=(8, 4))
            shap.plots.waterfall(sv[0], show=False, max_display=len(XAI_FEATURES))
            plt.tight_layout()
            shap_buf = BytesIO()
            plt.savefig(shap_buf, format="png", bbox_inches="tight", dpi=120)
            plt.close()
            shap_src = base64.b64encode(shap_buf.getvalue()).decode("utf-8")
            shap_component = html.Img(
                src=f"data:image/png;base64,{shap_src}",
                style={"width": "100%", "height": "100%", "objectFit": "contain"},
            )
        except Exception:
            shap_component = html.Div(
                dcc.Graph(
                    figure=coefficient_fallback(
                        models_state["xai_model"], XAI_FEATURES, x_scaled.iloc[0].values
                    ),
                    config=GRAPH_CONFIG,
                    style={"height": "100%"},
                )
            )
    else:
        shap_component = dcc.Graph(
            figure=coefficient_fallback(
                models_state["xai_model"], XAI_FEATURES, x_scaled.iloc[0].values
            ),
            config=GRAPH_CONFIG,
            style={"height": "100%"},
        )

    # Global SHAP
    if models_state["shap_explainer"] is not None and shap is not None:
        try:
            sample = models_state["xai_data"].sample(
                min(200, len(models_state["xai_data"])), random_state=42
            )
            sv = models_state["shap_explainer"](sample)
            importance = np.abs(sv.values).mean(axis=0)
        except Exception:
            importance = np.abs(models_state["xai_model"].coef_[0])
    else:
        importance = np.abs(models_state["xai_model"].coef_[0])

    global_fig = go.Figure(
        go.Bar(
            x=importance,
            y=XAI_FEATURES,
            orientation="h",
        )
    )
    global_fig.update_layout(
        title="Global Feature Importance",
        margin=dict(l=130, r=15, t=35, b=30),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    # LIME
    if models_state["lime_explainer"] is not None:
        try:
            def predict_fn(arr):
                frame = pd.DataFrame(arr, columns=XAI_FEATURES)
                return models_state["xai_model"].predict_proba(frame)

            explanation = models_state["lime_explainer"].explain_instance(
                x_scaled.iloc[0].values,
                predict_fn,
                num_features=len(XAI_FEATURES),
            )
            lime_fig = go.Figure(
                go.Bar(
                    x=[v for _, v in explanation.as_list()],
                    y=[k for k, _ in explanation.as_list()],
                    orientation="h",
                )
            )
            lime_fig.update_layout(
                title=f"LIME: Partner {partner_id}",
                margin=dict(l=150, r=15, t=35, b=30),
                paper_bgcolor="white",
                plot_bgcolor="white",
            )
            lime_component = dcc.Graph(
                figure=lime_fig, config=GRAPH_CONFIG, style={"height": "100%"}
            )
        except Exception:
            lime_component = dcc.Graph(
                figure=coefficient_fallback(
                    models_state["xai_model"], XAI_FEATURES, x_scaled.iloc[0].values
                ),
                config=GRAPH_CONFIG,
                style={"height": "100%"},
            )
    else:
        lime_component = dcc.Graph(
            figure=coefficient_fallback(
                models_state["xai_model"], XAI_FEATURES, x_scaled.iloc[0].values
            ),
            config=GRAPH_CONFIG,
            style={"height": "100%"},
        )

    return shap_component, html.Div([
        dcc.Graph(id="global-shap-inner", figure=global_fig, config=GRAPH_CONFIG, style={"height": "100%"}),
        html.Div("Global importance is calculated from SHAP values when SHAP is available; otherwise model coefficients are used.", style={"fontSize": "10px", "padding": "2px"}),
    ]), info


@app.callback(
    Output("wordcloud-container", "children"),
    Input("stopword-input", "value"),
)
def update_wordcloud(extra):
    extra_words = [x.strip().lower() for x in (extra or "").split(",") if x.strip()]
    encoded = make_wordcloud(sentiment_df, extra_words)
    if not encoded:
        return html.Div("No words available.")
    return html.Img(
        src=f"data:image/png;base64,{encoded}",
        style={"width": "100%", "height": "100%", "objectFit": "contain"},
    )


@app.callback(
    Output("top-feedback-table-container", "children"),
    Input("sentiment-feedback-dropdown", "value"),
)
def update_feedback_table(label):
    df = sentiment_df.loc[sentiment_df["sentiment_label"].eq(label)].copy()
    ascending = label == "Negative"
    df = df.sort_values("compound_sentiment", ascending=ascending).head(5)

    table = dash_table.DataTable(
        data=[
            {"Comment": row["text"], "Score": round(float(row["compound_sentiment"]), 3)}
            for _, row in df.iterrows()
        ],
        columns=[
            {"name": "Comment", "id": "Comment"},
            {"name": "Score", "id": "Score"},
        ],
        style_table={"overflowY": "auto", "height": "100%"},
        style_cell={"textAlign": "left", "fontSize": "11px", "padding": "5px", "whiteSpace": "normal", "height": "auto"},
        style_header={"backgroundColor": PRIMARY_BLUE, "color": "white", "fontWeight": "700"},
    )
    return table


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8050"))
    app.run(host="0.0.0.0", port=port, debug=False)
