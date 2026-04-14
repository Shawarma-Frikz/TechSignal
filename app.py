import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.data_loader import load_stock_data
from src.features import engineer_features
from src.preprocessing import split_and_scale, FEATURE_COLS
from src.model import (train_logistic_regression,
                       train_random_forest,
                       train_xgboost)
import ta

st.set_page_config(
    page_title="Stock Movement Predictor",
    page_icon="📈",
    layout="wide"
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Manrope:wght@400;500;700&display=swap');

    .stApp {
        background: radial-gradient(circle at 15% 20%, #eef6ff 0%, #f8fbff 45%, #f3f8f7 100%);
        font-family: 'Manrope', sans-serif;
    }

    .stApp::before,
    .stApp::after {
        content: "";
        position: fixed;
        width: 320px;
        height: 320px;
        border-radius: 50%;
        filter: blur(40px);
        z-index: -1;
        opacity: 0.35;
    }

    .stApp::before {
        background: #8fc8ff;
        top: -90px;
        right: -70px;
    }

    .stApp::after {
        background: #9ee5c0;
        bottom: -120px;
        left: -100px;
    }

    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif !important;
        letter-spacing: -0.02em;
    }

    .hero {
        border: 1px solid #d5e4f3;
        border-radius: 18px;
        padding: 1.2rem 1.4rem;
        background: linear-gradient(120deg, #ffffff 0%, #f0f7ff 100%);
        box-shadow: 0 8px 28px rgba(22, 72, 122, 0.08);
        margin-bottom: 1rem;
    }

    .hero-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0d2f4f;
        margin-bottom: 0.1rem;
    }

    .hero-subtitle {
        color: #355575;
        margin-bottom: 0;
    }

    .footer-note {
        border-top: 1px solid #d6e4ef;
        margin-top: 1.2rem;
        padding-top: 0.8rem;
        color: #3e5d7a;
        font-size: 0.95rem;
        text-align: center;
    }

    [data-baseweb="tab-list"] {
        gap: 0.25rem;
    }

    [data-baseweb="tab"] {
        border-radius: 999px;
        padding: 0.45rem 0.9rem;
        border: 1px solid #d8e6f4;
        background: #ffffff;
    }

    [data-baseweb="tab-highlight"] {
        background-color: #2e6ea8 !important;
        height: 3px !important;
    }

    .analysis-card {
        border: 1px solid #d2e2f2;
        border-radius: 14px;
        background: linear-gradient(145deg, #ffffff 0%, #f6fbff 100%);
        padding: 0.9rem 1rem;
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
        box-shadow: 0 6px 20px rgba(19, 83, 133, 0.06);
    }

    .analysis-title {
        color: #12395f;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .analysis-text {
        color: #34597b;
        margin: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">Stock Movement Predictor</div>
        <p class="hero-subtitle">A decision-support dashboard for significant stock move prediction using technical indicators and explainable ML.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.header("Analysis Settings")
    st.caption("Built by Idris Foudhaili")

    ticker = st.text_input(
        "Stock ticker",
        value="AAPL",
        help="e.g. AAPL, TSLA, MSFT, MC.PA"
    )

    start_date = st.date_input(
        "Start date",
        value=pd.to_datetime("2018-01-01")
    )

    end_date = st.date_input(
        "End date",
        value=pd.to_datetime("2024-01-01")
    )

    threshold = st.slider(
        "Prediction threshold",
        min_value=0.3,
        max_value=0.7,
        value=0.5,
        step=0.05,
        help="Higher = only flag high confidence predictions"
    )

    run_button = st.button("Run Analysis", type="primary")

# ── Stop if button not clicked ────────────────────────────
if not run_button:
    st.info("Configure settings in the sidebar and click Run Analysis")
    st.stop()

# ── Cached pipeline ───────────────────────────────────────
@st.cache_resource(show_spinner=False)
def run_pipeline(ticker, start, end):
    df = load_stock_data(ticker, str(start), str(end))
    df = engineer_features(df)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(df)

    lr  = train_logistic_regression(X_train, y_train)
    rf  = train_random_forest(X_train, y_train)
    xgb = train_xgboost(X_train, y_train)

    explainer   = shap.TreeExplainer(xgb)
    shap_values = explainer.shap_values(X_test)

    return df, X_train, X_test, y_train, y_test, scaler, lr, rf, xgb, explainer, shap_values

# ── Run pipeline ──────────────────────────────────────────
with st.spinner(f"Loading {ticker} data and training models..."):
    df, X_train, X_test, y_train, y_test, scaler, lr, rf, xgb, explainer, shap_values = run_pipeline(
        ticker, start_date, end_date
    )

models = {
    "Logistic Regression": lr,
    "Random Forest": rf,
    "XGBoost": xgb
}

results = []
model_probs = {}
for name, model in models.items():
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    model_probs[name] = y_prob
    results.append({
        "Model": name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Balanced Accuracy": round(balanced_accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "F1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "ROC-AUC": round(roc_auc_score(y_test, y_prob), 4),
        "PR-AUC": round(average_precision_score(y_test, y_prob), 4),
        "Brier": round(brier_score_loss(y_test, y_prob), 4),
    })

results_df = pd.DataFrame(results)
best_model_row = results_df.sort_values(
    by=["F1", "PR-AUC", "ROC-AUC", "Accuracy"], ascending=False
).iloc[0]
best_model_name = best_model_row["Model"]

price_start = float(df["Close"].iloc[0])
price_end = float(df["Close"].iloc[-1])
total_return_pct = (price_end / price_start - 1) * 100

recent_window = min(20, len(df) - 1)
if recent_window > 0:
    recent_return_pct = (float(df["Close"].iloc[-1]) / float(df["Close"].iloc[-(recent_window + 1)]) - 1) * 100
else:
    recent_return_pct = 0.0

daily_returns = df["Close"].pct_change().dropna()
rolling_vol_pct = daily_returns.tail(20).std() * (252 ** 0.5) * 100 if len(daily_returns) >= 5 else 0.0
sma20 = df["Close"].rolling(20).mean().iloc[-1]
sma50 = df["Close"].rolling(50).mean().iloc[-1]
trend_regime = "bullish momentum" if sma20 > sma50 else "cooling momentum"

if total_return_pct > 20:
    long_term_comment = "The selected period shows a strong long-term upward move."
elif total_return_pct > 0:
    long_term_comment = "The selected period is positive overall, but with moderate gain."
else:
    long_term_comment = "The selected period is negative overall, indicating a weaker regime."

if rolling_vol_pct > 40:
    vol_comment = "Volatility is elevated, so predictions may be less stable day to day."
elif rolling_vol_pct > 25:
    vol_comment = "Volatility is moderate, with a balanced risk profile."
else:
    vol_comment = "Volatility is relatively calm, often favoring smoother signal behavior."

# ── Dashboard tabs ────────────────────────────────────────
tab_overview, tab_metrics, tab_explainability, tab_prediction = st.tabs(
    ["Overview", "Metrics & Curves", "Explainability", "Tomorrow Signal"]
)

with tab_overview:
    st.subheader("Dataset Snapshot")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Trading Days", f"{len(df):,}")
    col2.metric("Train Window", f"{int(len(df)*0.8):,}")
    col3.metric("Test Window", f"{int(len(df)*0.2):,}")
    col4.metric("Move Definition", "±1%")

    st.markdown("#### Model Accuracy Comparison")
    
    # Model comparison bar chart - Blue for Accuracy, Pale Teal for ROC-AUC
    comparison_fig, comparison_ax = plt.subplots(figsize=(10, 4))
    x_pos = range(len(results_df))
    width = 0.35
    
    accuracy_vals = results_df["Accuracy"].values
    rocauc_vals = results_df["ROC-AUC"].values
    model_names = results_df["Model"].values
    
    comparison_ax.bar(
        [p - width/2 for p in x_pos],
        accuracy_vals,
        width,
        label="Accuracy",
        color="#1F4E79",  # Blue
        alpha=0.85
    )
    comparison_ax.bar(
        [p + width/2 for p in x_pos],
        rocauc_vals,
        width,
        label="ROC-AUC",
        color="#5DADE2",  # Pale teal
        alpha=0.75
    )
    
    comparison_ax.set_ylabel("Score")
    comparison_ax.set_title("Model Performance Comparison")
    comparison_ax.set_xticks(x_pos)
    comparison_ax.set_xticklabels(model_names, fontsize=9)
    comparison_ax.legend(loc="best", fontsize=9)
    comparison_ax.set_ylim([0.4, 0.65])
    comparison_ax.grid(axis="y", alpha=0.3)
    comparison_ax.spines["top"].set_visible(False)
    comparison_ax.spines["right"].set_visible(False)
    
    st.pyplot(comparison_fig)
    plt.close()
    
    st.caption(
        "Scoreboard guide: Accuracy = overall hit rate, Precision = quality of UP signals, Recall = how many true UP moves are captured, F1 = precision/recall balance, PR-AUC and ROC-AUC = ranking quality, Brier = probability calibration (lower is better)."
    )
    st.dataframe(
        results_df.style.background_gradient(
            cmap="Blues", subset=["Accuracy", "Balanced Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
        ),
        use_container_width=True,
    )

    st.markdown(
        f"""
        <div class="analysis-card">
            <div class="analysis-title">Scoreboard Analysis</div>
            <p class="analysis-text">
                Best current performer is <strong>{best_model_name}</strong> with F1={best_model_row['F1']:.3f},
                PR-AUC={best_model_row['PR-AUC']:.3f}, ROC-AUC={best_model_row['ROC-AUC']:.3f}.
                This means it currently offers the strongest precision/recall balance for significant-move classification under the selected threshold.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Price Action & Predicted Signals")
    
    # Generate predictions for visualization
    xgb_probs = model_probs["XGBoost"]
    xgb_preds = (xgb_probs >= threshold).astype(int)
    
    # Map predictions to test set indices
    test_start_idx = int(len(df) * 0.8)
    signal_indices = list(range(test_start_idx, len(df)))
    
    # Create price action chart with signal overlays
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Plot closing price
    ax.plot(df.index, df["Close"], linewidth=1.5, color="#1F4E79", label="Close Price", zorder=2)
    
    # Plot SMA-20 as dashed line
    df["SMA20"] = df["Close"].rolling(window=20).mean()
    ax.plot(df.index, df["SMA20"], linestyle="--", linewidth=1.2, color="#7B8A99", label="SMA-20", alpha=0.7, zorder=1)
    
    # Plot UP signals (green triangles) - ▲
    up_signals = [(signal_indices[i], df["Close"].iloc[signal_indices[i]]) 
                  for i in range(len(xgb_preds)) if xgb_preds[i] == 1]
    if up_signals:
        up_x, up_y = zip(*up_signals)
        ax.scatter(up_x, up_y, marker="^", s=150, color="#27AE60", label="UP Signal (>+1%)", zorder=3, edgecolors="#1E8449", linewidth=0.5)
    
    # Plot DOWN signals (red squares) - ■
    down_signals = [(signal_indices[i], df["Close"].iloc[signal_indices[i]]) 
                    for i in range(len(xgb_preds)) if xgb_preds[i] == 0]
    if down_signals:
        down_x, down_y = zip(*down_signals)
        ax.scatter(down_x, down_y, marker="s", s=120, color="#E74C3C", label="DOWN Signal (>-1%)", zorder=3, edgecolors="#C0392B", linewidth=0.5)
    
    ax.set_ylabel("Price (USD)", fontsize=10)
    ax.set_xlabel("")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.2, linestyle=":")
    
    st.pyplot(fig)
    plt.close()

    st.markdown(
        f"""
        <div class="analysis-card">
            <div class="analysis-title">Price Action Analysis</div>
            <p class="analysis-text">
                Total return over selected window is <strong>{total_return_pct:.2f}%</strong>,
                with recent {recent_window}-day return at <strong>{recent_return_pct:.2f}%</strong>.
                Trend regime appears to be <strong>{trend_regime}</strong> (SMA20 vs SMA50).
                Annualized short-term volatility is <strong>{rolling_vol_pct:.2f}%</strong>.
                {long_term_comment} {vol_comment}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # RSI Indicator with shading
    st.markdown("#### Technical Indicator: RSI-14")
    rsi_fig, rsi_ax = plt.subplots(figsize=(12, 3.5))
    
    df["RSI"] = ta.momentum.rsi(df["Close"], window=14)
    
    rsi_ax.plot(df.index, df["RSI"], linewidth=1.5, color="#1F4E79", label="RSI-14")
    
    # Shade bullish region (RSI > 50, green)
    rsi_ax.fill_between(df.index, 50, df["RSI"], where=(df["RSI"] > 50), 
                        color="#27AE60", alpha=0.3, label="Bullish (RSI > 50)")
    
    # Shade bearish region (RSI < 50, red)
    rsi_ax.fill_between(df.index, df["RSI"], 50, where=(df["RSI"] <= 50), 
                        color="#E74C3C", alpha=0.3, label="Bearish (RSI ≤ 50)")
    
    # Overbought/oversold lines
    rsi_ax.axhline(y=70, color="#E74C3C", linestyle="--", linewidth=1, alpha=0.6, label="Overbought (70)")
    rsi_ax.axhline(y=30, color="#27AE60", linestyle="--", linewidth=1, alpha=0.6, label="Oversold (30)")
    rsi_ax.axhline(y=50, color="#7B8A99", linestyle="-", linewidth=0.8, alpha=0.4)
    
    rsi_ax.set_ylabel("RSI", fontsize=10)
    rsi_ax.set_xlabel("")
    rsi_ax.set_ylim([0, 100])
    rsi_ax.legend(loc="upper left", fontsize=8, ncol=3, framealpha=0.95)
    rsi_ax.spines["top"].set_visible(False)
    rsi_ax.spines["right"].set_visible(False)
    rsi_ax.grid(alpha=0.2, linestyle=":")
    
    st.pyplot(rsi_fig)
    plt.close()

with tab_metrics:
    st.subheader("Rich Evaluation Metrics")
    st.caption(f"Classification threshold in use: {threshold:.2f}")
    st.dataframe(results_df, use_container_width=True)

    selected_model = st.selectbox("Inspect confusion matrix for", list(models.keys()), index=2)
    selected_probs = model_probs[selected_model]
    selected_preds = (selected_probs >= threshold).astype(int)
    cm = confusion_matrix(y_test, selected_preds)

    cm_fig, cm_ax = plt.subplots(figsize=(4.6, 4.2))
    im = cm_ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            cm_ax.text(j, i, cm[i, j], ha="center", va="center", color="#0A2C4E", fontsize=12)
    cm_ax.set_title(f"{selected_model} Confusion Matrix")
    cm_ax.set_xticks([0, 1], ["DOWN", "UP"])
    cm_ax.set_yticks([0, 1], ["DOWN", "UP"])
    cm_ax.set_xlabel("Predicted")
    cm_ax.set_ylabel("Actual")
    cm_fig.colorbar(im, ax=cm_ax, fraction=0.046, pad=0.04)
    st.pyplot(cm_fig)
    plt.close()

    curve_col1, curve_col2 = st.columns(2)

    with curve_col1:
        roc_fig, roc_ax = plt.subplots(figsize=(6.2, 4.2))
        for name in models:
            fpr, tpr, _ = roc_curve(y_test, model_probs[name])
            auc = roc_auc_score(y_test, model_probs[name])
            roc_ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
        roc_ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
        roc_ax.set_title("ROC Curves")
        roc_ax.set_xlabel("False Positive Rate")
        roc_ax.set_ylabel("True Positive Rate")
        roc_ax.legend(fontsize=8)
        roc_ax.grid(alpha=0.2)
        st.pyplot(roc_fig)
        plt.close()

    with curve_col2:
        pr_fig, pr_ax = plt.subplots(figsize=(6.2, 4.2))
        for name in models:
            precision, recall, _ = precision_recall_curve(y_test, model_probs[name])
            pr_auc = average_precision_score(y_test, model_probs[name])
            pr_ax.plot(recall, precision, label=f"{name} (AP={pr_auc:.3f})")
        pr_ax.set_title("Precision-Recall Curves")
        pr_ax.set_xlabel("Recall")
        pr_ax.set_ylabel("Precision")
        pr_ax.legend(fontsize=8)
        pr_ax.grid(alpha=0.2)
        st.pyplot(pr_fig)
        plt.close()

with tab_explainability:
    st.subheader("What Drives the Predictions")
    shap.summary_plot(
        shap_values,
        X_test,
        feature_names=FEATURE_COLS,
        show=False,
    )
    st.pyplot(plt.gcf())
    plt.close()

with tab_prediction:
    st.subheader(f"Tomorrow's Prediction for {ticker}")
    latest_features = df[FEATURE_COLS].iloc[-1:].values
    latest_scaled = scaler.transform(latest_features)

    prob_up = xgb.predict_proba(latest_scaled)[0][1]
    prediction = "UP" if prob_up >= threshold else "DOWN"
    confidence = prob_up if prob_up >= threshold else 1 - prob_up

    col1, col2, col3 = st.columns(3)
    col1.metric("Direction", prediction)
    col2.metric("Confidence", f"{confidence:.1%}")
    col3.metric("Decision Threshold", f"{threshold:.2f}")

    if prediction == "UP":
        st.success("Model indicates a potential significant UP move on the next trading day.")
    else:
        st.warning("Model indicates a potential significant DOWN move on the next trading day.")

    st.markdown("#### Explore More Tickers")
    st.write("Update the symbol in the sidebar and run again (e.g., TSLA, MSFT, GOOGL, MC.PA, TTE.PA).")

st.markdown(
    f"""
    <div class="footer-note">
        © {pd.Timestamp.now().year} Idris Foudhaili. All rights reserved. Educational and research purpose only, not financial advice.
    </div>
    """,
    unsafe_allow_html=True,
)