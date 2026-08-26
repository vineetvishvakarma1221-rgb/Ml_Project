import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import joblib as jb
import shap
from datetime import datetime
import plotly.express as px
from database import *
from predict import *
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

ASSETS_DIR = BASE_DIR/'assets'
MODEL_DIR = BASE_DIR/'models'
SRC_DIR = BASE_DIR/'src'
#  PAGE CONFIG

st.set_page_config(
    page_title='Electricity Fraud Detection',
    page_icon="⚡", layout='wide',
    initial_sidebar_state='expanded'
)

#  THEME / CSS  ("Substation" theme — dark grid, amber spark accent)

def load_css():
    try:
        with open(SRC_DIR/"style.css",'r',encoding='utf-8') as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception:
        pass

load_css()

#  SMALL UI HELPERS

def section_header(icon, title, subtitle=""):
    st.markdown(f"""
    <div class="sec-head">
        <span class="ico">{icon}</span>
        <span class="ttl">{title}</span>
        <span class="sub">{subtitle}</span>
    </div>
    <div class="sec-divider"></div>
    """, unsafe_allow_html=True)

def kpi_card(label, value, delta=""):
    st.markdown(f"""
    <div class="kpi">
        <div class="kpi-top"><span>{label}</span></div>
        <div class="kpi-val">{value}</div>
        <div class="kpi-delta">{delta}</div>
    </div>
    """, unsafe_allow_html=True)

def risk_badge(risk):
    cls = {"High": "b-danger", "Medium": "b-warn", "Low": "b-safe"}.get(risk, "b-warn")
    dot = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(risk, "⚪")
    st.markdown(f'<span class="badge {cls}">{dot} {risk} Risk</span>', unsafe_allow_html=True)

def alert_card(kind, title, message):
    cls = {"danger": "alert-danger", "warn": "alert-warn", "safe": "alert-safe"}[kind]
    icon = {"danger": "🚨", "warn": "⚠️", "safe": "✅"}[kind]
    st.markdown(f"""
    <div class="alert-card {cls}">
        <b>{icon} {title}</b><br>
        <span style="color:var(--text-dim); font-size:13.5px;">{message}</span>
    </div>
    """, unsafe_allow_html=True)

def plotly_theme(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e9edf5", family="Inter"),
        margin=dict(t=50, b=20, l=10, r=10),
    )
    return fig

# ============================================================
#  HERO HEADER
# ============================================================
st.markdown("""
<div class="hero-wrap">
    <div class="hero-eyebrow"><span class="hero-dot"></span> LIVE GRID MONITORING</div>
    <div class="hero-title">⚡ Electricity Fraud Detection System</div>
    <div class="hero-sub">
        Screen customer consumption patterns for signs of electricity theft using a
        Random Forest classifier with SHAP-based explainability, batch scoring and
        historical analytics — all in one place.
    </div>
</div>
""", unsafe_allow_html=True)

try:
    st.image(ASSETS_DIR/"banner.png", use_container_width=True)
except Exception:
    pass

try:
    st.sidebar.image(ASSETS_DIR/"logo.png", width=100)
except Exception:
    st.sidebar.markdown("### ⚡")

# ============================================================
#  CACHED LOADERS
# ============================================================
@st.cache_resource
def load_feature_name():
    return jb.load(MODEL_DIR/"feature_names.pkl")

@st.cache_resource
def load_explainer():
    return jb.load(MODEL_DIR/"shap_explainer.pkl")

# ============================================================
#  SIDEBAR
# ============================================================
if "prediction_count" not in st.session_state:
    st.session_state.prediction_count = 0

st.sidebar.title('✨ Navigation')
st.sidebar.caption("Use the tabs on the main page to move between sections.")
st.sidebar.markdown('---')

st.sidebar.header("Project")
st.sidebar.write("Electricity Fraud Detection")
st.sidebar.markdown("---")

st.sidebar.header("Model")
st.sidebar.write("Random Forest Classifier")
st.sidebar.markdown('---')

st.sidebar.header("Developer")
st.sidebar.write("Vineet Vishvakarma")
st.sidebar.markdown("---")

st.sidebar.header("📌 Session")
st.sidebar.metric("Predictions this session", st.session_state.prediction_count)
st.sidebar.caption(f"🕒 {datetime.now().strftime('%d %b %Y, %H:%M:%S')}")
st.sidebar.markdown("---")

with st.sidebar.expander("ℹ️ How to use this app"):
    st.markdown("""
    1. Go to **🔍 Live Prediction** and enter customer readings, or load a preset.
    2. Click **Predict Fraud** to score the customer.
    3. Check **🧠 Explainability** to see which features drove the decision.
    4. Use **📂 Batch Prediction** to score an entire CSV of customers.
    5. Review **📜 History** and **📊 Analytics** for trends over time.
    """)

# ============================================================
#  TABS
# ============================================================
tab_overview, tab_predict, tab_explain, tab_batch, tab_history, tab_analytics = st.tabs(
    ["🏠 Overview", "🔍 Live Prediction", "🧠 Explainability", "📂 Batch Prediction", "📜 History", "📊 Analytics"]
)

# ------------------------------------------------------------
#  TAB: OVERVIEW
# ------------------------------------------------------------
with tab_overview:
    section_header("🧾", "Model Information", "Snapshot of the deployed classifier")
    c1, c2, c3 = st.columns(3)
    with c1:
        kpi_card("Model", "Random Forest")
    with c2:
        kpi_card("ROC AUC", "0.746")
    with c3:
        kpi_card("Accuracy", "66.37%")

    st.write("")
    section_header("🧩", "Selected Features", "The 8 engineered features fed into the model")
    feat_list = [
        ("01", "Standard Deviation"), ("02", "Stability"),
        ("03", "Coefficient of Variance"), ("04", "Longest Missing Streak"),
        ("05", "Highest Consumption Ratio"), ("06", "Peak-to-Average Ratio"),
        ("07", "Zero Consumption Ratio"), ("08", "Longest Zero Streak"),
    ]
    cols = st.columns(4)
    for i, (num, name) in enumerate(feat_list):
        with cols[i % 4]:
            st.markdown(f"""
            <div class="feat-card">
                <div class="feat-num">{num}</div>
                <div class="feat-name">{name}</div>
            </div>
            """, unsafe_allow_html=True)
            st.write("")

    st.write("")
    section_header("📚", "About this System")
    st.markdown("""
    <div class="feat-card">
    Developed using:

    - Random Forest Classifier
    - Streamlit
    - SHAP Explainability
    - Python
    - Scikit-Learn
    </div>
    """, unsafe_allow_html=True)

# ------------------------------------------------------------
#  TAB: LIVE PREDICTION
# ------------------------------------------------------------
result = None
customer = None

with tab_predict:
    section_header("🎛️", "Customer Input", "Enter consumption statistics for a single customer")

    # ---- Presets (purely fill the same inputs, no model logic touched) ----
    preset_defaults = {
        "std_cons": 3.5, "Stability": 0.80, "cv": 0.30, "longest_missing_streak": 5,
        "high_cons_ratio": 0.04, "PAR": 3.0, "zero_ratio": 0.02, "longest_zero_streak": 2,
    }
    preset_normal = {
        "std_cons": 2.1, "Stability": 0.92, "cv": 0.18, "longest_missing_streak": 1,
        "high_cons_ratio": 0.03, "PAR": 2.4, "zero_ratio": 0.01, "longest_zero_streak": 0,
    }
    preset_suspicious = {
        "std_cons": 8.7, "Stability": 0.35, "cv": 1.15, "longest_missing_streak": 14,
        "high_cons_ratio": 0.68, "PAR": 7.9, "zero_ratio": 0.42, "longest_zero_streak": 11,
    }

    if "input_values" not in st.session_state:
        st.session_state.input_values = dict(preset_defaults)

    pcol1, pcol2, pcol3, pcol4 = st.columns(4)
    with pcol1:
        if st.button("↩️ Reset Defaults", use_container_width=True):
            st.session_state.input_values = dict(preset_defaults)
            st.rerun()
    with pcol2:
        if st.button("🟢 Load Normal Example", use_container_width=True):
            st.session_state.input_values = dict(preset_normal)
            st.rerun()
    with pcol3:
        if st.button("🔴 Load Suspicious Example", use_container_width=True):
            st.session_state.input_values = dict(preset_suspicious)
            st.rerun()
    with pcol4:
        st.caption("Presets only pre-fill the fields below — you can still edit every value before predicting.")

    st.write("")
    iv = st.session_state.input_values
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        std_cons = st.number_input(
            'Standard Deviation of Consumption',
            min_value=0.0, value=float(iv["std_cons"]), step=0.1,
            help="Spread of the customer's daily/monthly consumption readings."
        )
        Stability = st.number_input(
            'Stability', value=float(iv["Stability"]), step=0.1,
            help="How stable the consumption pattern is over time (closer to 1 = more stable)."
        )

    with c2:
        cv = st.number_input(
            'Coeffient of Variation', value=float(iv["cv"]), step=0.01,
            help="Standard deviation relative to the mean consumption."
        )
        longest_missing_streak = st.number_input(
            'Longest Missing Streak', min_value=0, value=int(iv["longest_missing_streak"]),
            help="Longest run of missing meter readings."
        )

    with c3:
        high_cons_ratio = st.number_input(
            "High Consumptionn Ratio", value=float(iv["high_cons_ratio"]), step=0.01,
            help="Share of readings that are unusually high."
        )
        par = st.number_input(
            'Peak-to-Average ratio', value=float(iv["PAR"]), step=0.1,
            help="Ratio of peak consumption to average consumption."
        )

    with c4:
        zero_ratio = st.number_input(
            'Zero Cosumption Ratio', value=float(iv["zero_ratio"]), step=0.01,
            help="Share of readings that are exactly zero."
        )
        longest_zero_streak = st.number_input(
            'Longest Zero Streak', min_value=0, value=int(iv["longest_zero_streak"]),
            help="Longest run of consecutive zero-consumption readings."
        )

    st.write("")
    predict_btn = st.button(
        "🔍 Predict Fraud 🔍",
        use_container_width=True,
        type="primary"
    )

    if predict_btn:
        customer = {
            "std_cons": std_cons,
            "Stability": Stability,
            "cv": cv,
            "longest_missing_streak": longest_missing_streak,
            "high_cons_ratio": high_cons_ratio,
            "PAR": par,
            "zero_ratio": zero_ratio,
            "longest_zero_streak": longest_zero_streak
        }
        with st.spinner('⚡ Running fraud detection model...'):
            result = predict_customer(customer)
        st.session_state.prediction_count += 1

    st.markdown("---")
    section_header("📟", "Prediction Result")

    if result is not None:
        c1, c2, c3 = st.columns(3)
        with c1:
            kpi_card("Fraud Probability", f'{result["fraud_pobability"]*100:.2f}%')
        with c2:
            kpi_card("Prediction", "🚨 FRAUD" if result['prediction'] == 1 else "✅ NORMAL")
        with c3:
            st.markdown('<div class="kpi"><div class="kpi-top">Risk Level</div><div style="margin-top:10px;">', unsafe_allow_html=True)
            risk_badge(result['risk'])
            st.markdown('</div></div>', unsafe_allow_html=True)

        st.write("")
        if result['prediction'] == 1:
            alert_card("danger", "High Possibility of Electricity Theft Detected",
                       "This customer's pattern closely matches known theft signatures.")
        else:
            alert_card("safe", "Customer Appears Normal",
                       "No strong indicators of electricity theft were found.")

        if result['risk'] == 'High':
            alert_card("danger", "Immediate Investigation Recommended", "Escalate this case for field inspection.")
        elif result['risk'] == 'Medium':
            alert_card("warn", "Customer Should Be Monitored", "Keep this customer on a watchlist for recurring checks.")
        else:
            alert_card("safe", "Low Fraud Risk", "No immediate action required.")

        st.write("")
        st.markdown("**Fraud Probability**")
        st.progress(float(result['fraud_pobability']))
        st.caption(f'Fruad Probability : {result["fraud_pobability"]:.2%}')

        with st.expander("🔎 Prediction Details (raw JSON)"):
            st.json(result)

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=result["fraud_pobability"]*100,
                title={"text": "Fraud Probability (%)"},
                number={"font": {"color": "#e9edf5"}},
                gauge={"axis": {"range": [0, 100], "tickcolor": "#8b96ab"},
                       "bar": {"color": "#ffb020"},
                       "bgcolor": "rgba(0,0,0,0)",
                       "steps": [
                           {"range": [0, 30], "color": "rgba(34,197,94,0.35)"},
                           {"range": [30, 60], "color": "rgba(245,179,1,0.35)"},
                           {"range": [60, 100], "color": "rgba(255,85,102,0.35)"}
                       ],
                       "threshold": {
                           "line": {"color": "#ff5566", "width": 4},
                           "thickness": 0.8,
                           "value": result["fraud_pobability"]*100
                       }
                       }
            )
        )
        st.plotly_chart(plotly_theme(fig), use_container_width=True)

        st.markdown("##### Risk Level Legend")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<span class="badge b-safe">🟢 Low Risk (0–30%)</span>', unsafe_allow_html=True)
        with col2:
            st.markdown('<span class="badge b-warn">🟡 Medium Risk (30–60%)</span>', unsafe_allow_html=True)
        with col3:
            st.markdown('<span class="badge b-danger">🔴 High Risk (60–100%)</span>', unsafe_allow_html=True)

        prob = result["fraud_pobability"]
        st.write("")
        if prob < 0.30:
            alert_card("safe", "Low Probability of Theft", "The customer has a low probability of electricity theft.")
        elif prob < 0.60:
            alert_card("warn", "Moderate Fraud Behaviour", "The customer shows moderate fraud behaviour. Monitoring is recommended.")
        else:
            alert_card("danger", "High Probability of Theft", "The customer has a high probability of electricity theft. Immediate investigation is recommended.")

        section_header("🎯", "Prediction Confidence")
        confidence = max(prob, 1 - prob)
        c1, c2 = st.columns([1, 2])
        with c1:
            kpi_card("Model Confidence", f'{confidence*100:.2f}%')
        with c2:
            st.progress(float(confidence))

        st.markdown("---")
        section_header("🗂️", "Customer Feature Summary")
        summary_df = pd.DataFrame(customer.items(), columns=['Feature', 'Value'])

        def feature_status(feature, value):
            if feature == "zero_ratio":
                return "🔴 High" if value > 0.30 else "🟢 Normal"
            elif feature == "high_cons_ratio":
                return "🔴 High" if value > 0.60 else "🟢 Normal"
            elif feature == "cv":
                return "🟠 Variable" if value > 0.80 else "🟢 Stable"
            elif feature == "PAR":
                return "🔴 High" if value > 6 else "🟢 Normal"
            return "-"

        summary_df["Status"] = summary_df.apply(
            lambda row: feature_status(row["Feature"], row["Value"]), axis=1
        )
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        report = pd.DataFrame({
            "Fraud Probability": [result["fraud_pobability"]],
            "Prediction": [result["prediction"]],
            "Risk": [result["risk"]]
        })
        csv = report.to_csv(index=False).encode("utf-8")
        dcol1, dcol2 = st.columns([1, 1])
        with dcol1:
            st.download_button(
                label="📥 Download Prediction Report",
                data=csv,
                file_name="prediction_report.csv",
                mime="text/csv",
                use_container_width=True
            )
        with dcol2:
            st.caption(f"Prediction generated on : {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}")

        if result['risk'] == 'Low' and result['prediction'] == 0:
            st.balloons()
    else:
        st.info("👆 Fill in the customer readings above (or load a preset) and click **Predict Fraud** to see results here.")

# ------------------------------------------------------------
#  EXPLAINABILITY LOGIC (unchanged) + TAB
# ------------------------------------------------------------
feature_names = load_feature_name()
explainer = load_explainer()
jb.dump(explainer, MODEL_DIR/"shap_explainer.pkl")
Exp_path = MODEL_DIR/"shap_explainer.pkl"
explainer = load_explainer()

def explain_prediction(data_input):
    df = pd.DataFrame([data_input])
    df = df[feature_names]
    shap_values = explainer(df)
    constributions = pd.DataFrame({
        'Feature': feature_names,
        'SHAP Value': shap_values.values[0, :, 1]
    })
    constributions['Impact'] = constributions['SHAP Value'].abs()
    constributions = constributions.sort_values('Impact', ascending=False)
    return constributions

with tab_explain:
    section_header("🧠", "Why did the model make this prediction?", "SHAP feature attributions")
    if customer is not None:
        explanation = explain_prediction(customer)
        st.dataframe(explanation.head(10), use_container_width=True, hide_index=True)

        fig = px.bar(
            explanation.head(10), x='SHAP Value', y='Feature',
            orientation='h', title='Top Feature Contribution',
            color='SHAP Value', color_continuous_scale=["#22c55e", "#ffb020", "#ff5566"]
        )
        fig.update_layout(yaxis=dict(autorange='reversed'))
        st.plotly_chart(plotly_theme(fig), use_container_width=True)

        top_feature = explanation.iloc[0]
        alert_card("warn", f"Most Influential Feature: {top_feature['Feature']}",
                   f"SHAP Contribution: {top_feature['SHAP Value']:.4f}. Higher absolute SHAP values indicate a stronger influence on the model's decision.")

        positive = explanation[explanation["SHAP Value"] > 0]
        negative = explanation[explanation["SHAP Value"] < 0]
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### 🔺 Features Increasing Fraud Risk")
            st.dataframe(positive.head(5), use_container_width=True, hide_index=True)
        with col2:
            st.markdown("##### 🔻 Features Decreasing Fraud Risk")
            st.dataframe(negative.head(5), use_container_width=True, hide_index=True)
    else:
        st.info("Run a prediction in the **🔍 Live Prediction** tab first — the explanation for that customer will appear here.")

# ------------------------------------------------------------
#  TAB: BATCH PREDICTION
# ------------------------------------------------------------
with tab_batch:
    section_header("📂", "Batch Prediction", "Score an entire CSV of customers at once")
    upload_file = st.file_uploader('Upload Customer CSV', type=['csv'])

    if upload_file is not None:
        batch_df = pd.read_csv(upload_file)
        st.markdown("##### Uploaded Dataset (preview)")
        st.dataframe(batch_df.head(), use_container_width=True)

        req_features = feature_names
        missing = [col for col in req_features if col not in batch_df.columns]
        if missing:
            batch_df=extract_model_feature(batch_df)
            #st.error(f'Missing Columns : {missing}')
            #st.stop()

        with st.spinner("⚡ Scoring uploaded customers..."):
            result_df = batch_predict(batch_df)

        st.markdown("##### Prediction Results")
        st.dataframe(result_df, use_container_width=True)
        save_batch_prediction_mysql(result_df)
        fraud = (result_df['prediction'] == 1).sum()
        normal = (result_df['prediction'] == 0).sum()
        c1, c2 = st.columns(2)
        with c1:
            kpi_card("Fraud Customers", fraud)
        with c2:
            kpi_card("Normal Customers",normal)

        risk_counts = result_df['risk'].value_counts()
        fig = px.bar(x=risk_counts.index, y=risk_counts.values, title="Risk Distribution (Batch)",
                     labels={"x": "Risk", "y": "Count"}, color=risk_counts.index,
                     color_discrete_map={"Low": "#22c55e", "Medium": "#ffb020", "High": "#ff5566"})
        st.plotly_chart(plotly_theme(fig), use_container_width=True)

        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button('📥 Download Prediction Results', csv, 'Batch_Prediction.csv', 'text/csv')

        fraud_rate = (fraud / len(result_df)) * 100
        alert_card("warn", "Batch Prediction Summary",
                   f"Total Customers: {len(result_df)} &nbsp;|&nbsp; Fraud Detected: {fraud} &nbsp;|&nbsp; Fraud Rate: {fraud_rate:.2f}%")
    else:
        st.info("Upload a CSV containing the required feature columns to run batch scoring.")

# ------------------------------------------------------------
#  PERSISTENCE (unchanged)
# ------------------------------------------------------------
from utils import *

#save_prediction(customer,result)
try:
    save_prediction_mysql(customer,result)
except Exception as e:
    st.warning(f"MySQL save Failed : {e}")

# ------------------------------------------------------------
#  TAB: HISTORY
# ------------------------------------------------------------
history = load_prediction_history()

with tab_history:
    section_header("📜", "Prediction History")

    if history.empty:
        st.info("No history available.")
    else:
        st.dataframe(history, use_container_width=True, hide_index=True)

    search = st.text_input("🔎 Search Prediction History")

    if search:
        filtered = history[
            history.astype(str).apply(
                lambda row: row.str.contains(search, case=False).any(), axis=1
            )
        ]
        st.dataframe(filtered, use_container_width=True, hide_index=True)

    if not history.empty:
        history['Prediction'] = history['prediction'].replace({0: 'Normal', 1: 'Fraud'})
        history.rename(
            columns={
                'fraud_probability': "Fraud Probability",
                'risk': 'Risk',
                'prediction_time': 'Timestamp'
            },
            inplace=True
        )
        csv = history.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Prediction History", csv, "Prediction_History.csv", "text/csv")

    if st.button("🗑 Clear Prediction History"):
        #clear_history()
        clear_prediction_history()
        st.success("Prediction history deleted.")
        st.rerun()

    if not history.empty:
        st.markdown("##### History Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            kpi_card("Total Predictions", len(history))
        with col2:
            kpi_card("Fraud Cases", (history["Prediction"] == 'Fraud').sum())
        with col3:
            kpi_card("Normal Cases", (history["Prediction"] == 'Normal').sum())

# ------------------------------------------------------------
#  TAB: ANALYTICS
# ------------------------------------------------------------
with tab_analytics:
    section_header("📊", "Dashboard Analytics")

    if history.empty:
        st.warning("No history available for analytics.")
    else:
        prediction_counts = history["Prediction"].value_counts()
        fig = px.pie(
            values=prediction_counts.values, names=prediction_counts.index,
            title="Fraud vs Normal Predictions", hole=0.55,
            color=prediction_counts.index,
            color_discrete_map={"Fraud": "#ff5566", "Normal": "#22c55e"}
        )
        st.plotly_chart(plotly_theme(fig), use_container_width=True)

        risk = history["Risk"].value_counts()
        fig = px.bar(x=risk.index, y=risk.values, text=risk.values, title="Risk Distribution",
                     color=risk.index, color_discrete_map={"Low": "#22c55e", "Medium": "#ffb020", "High": "#ff5566"})
        st.plotly_chart(plotly_theme(fig), use_container_width=True)

        fig = px.histogram(history, x="fraud_pobability", nbins=25, title="Fraud Probability Distribution",
                            color_discrete_sequence=["#ffb020"])
        st.plotly_chart(plotly_theme(fig), use_container_width=True)

        history["Timestamp"] = pd.to_datetime(history["Timestamp"])
        timeline = (history.groupby(history["Timestamp"].dt.date).size().reset_index(name="Predictions"))

        fig = px.line(timeline, x="Timestamp", y="Predictions", markers=True, title="Prediction Timeline",
                       color_discrete_sequence=["#22d3ee"])
        st.plotly_chart(plotly_theme(fig), use_container_width=True)

        avg_prob = history["fraud_pobability"].mean() * 100
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=avg_prob,
            title={"text": "Average Fraud Probability"},
            number={"font": {"color": "#e9edf5"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#8b96ab"},
                "bar": {"color": "#22d3ee"},
                "bgcolor": "rgba(0,0,0,0)",
                "steps": [
                    {"range": [0, 30], "color": "rgba(34,197,94,0.35)"},
                    {"range": [30, 60], "color": "rgba(245,179,1,0.35)"},
                    {"range": [60, 100], "color": "rgba(255,85,102,0.35)"}
                ]
            }
        ))
        st.plotly_chart(plotly_theme(fig), use_container_width=True)

        st.markdown("##### 🕒 Recent Predictions")
        st.dataframe(history.tail(10), use_container_width=True, hide_index=True)

        st.markdown("##### Overall Statistics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            kpi_card("Total Predictions", len(history))
        with col2:
            kpi_card("Fraud Cases", (history["Prediction"] == 'Fraud').sum())
        with col3:
            kpi_card("Average Probability", f"{history['fraud_pobability'].mean()*100:.2f}%")
        with col4:
            kpi_card("High Risk Cases", (history["Risk"] == "High").sum())

st.markdown('<div class="footer-note">⚡ Electricity Fraud Detection System — Random Forest · SHAP · Streamlit</div>', unsafe_allow_html=True)
