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

st.set_page_config(
    page_title='Electricity Fraud Detection',
    page_icon="⚡",layout='wide',
    initial_sidebar_state='expanded' 
)
def load_css():
    with open(r"C:\Users\vinee\ML_Project\src\style.css") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True        )
load_css()

st.markdown("""
<div class='main-title'>
⚡ Electricity Fraud Detection System
</div>
""",
unsafe_allow_html=True)
st.image(r"C:\Users\vinee\ML_Project\assets\banner.png",use_container_width=True)
st.sidebar.image(r"C:\Users\vinee\ML_Project\assets\logo.png",width=120)

@st.cache_resource
def load_feature_name():
    return jb.load(r"C:\Users\vinee\ML_Project\models\feature_names.pkl")
@st.cache_resource
def load_explainer():
    return jb.load(r"C:\Users\vinee\ML_Project\models\shap_explainer.pkl")

st.sidebar.title('✨ Navigation')
st.sidebar.markdown('---')
st.sidebar.header('Project')
st.sidebar.write("Electricity Fraud Detection")
st.sidebar.markdown("---")
st.sidebar.header("Model")
st.sidebar.write("Random Forest Classifier")
st.sidebar.markdown('---')
st.sidebar.header("Developer")
st.sidebar.write("Vineet Vishvakarma")
st.sidebar.markdown("---")

# Model information 

st.subheader('Model Information')
c1,c2,c3 = st.columns(3)
with c1 :
    st.metric('Model','Random Forest')
with c2 :
    st.metric("ROC AUC",'0.746')
with c3 :
    st.metric("Accuracy",'66.37%')

st.subheader("Selected Features")
#st.markdown("---")
#st.markdown("Selected Features")
col1,col2,col3,col4 = st.columns(4)
with col1 :
    st.info(
        """
        1. Standard Daviation 
        2. Stability 
    """
    )
with col2 :
    st.success(
        """
        3. Coeffient of Variance
        4. Longest missing streak
        """
        )
with col3 :
    st.info(
        """
        5. Highest consumption ratio
        6. Peak-to-Average Ratio
    """
    )
with col4 :
    st.success(
        """
        7. Zero consumption ratio
        8. Longest Zero streak
        """
        )

st.subheader("Prediction Dashboard")
st.subheader("Customer Input")
c1,c2,c3,c4 = st.columns(4)

with c1 :
    std_cons = st.number_input(
        'Standard Deviation of Consumption',
        min_value=0.0,value=3.5,step=0.1
        )
    Stability = st.number_input(
        'Stability',value=0.80,step=0.1
    )

with c2 :
    cv = st.number_input(
        'Coeffient of Variation',value=0.30,step=0.01
    )
    longest_missing_streak = st.number_input(
        'Longest Missing Streak',min_value=0,value=5
    )

with c3:
    high_cons_ratio = st.number_input(
        "High Consumptionn Ratio",value=0.04,step=0.01
    ) 
    par = st.number_input(
        'Peak-to-Average ratio',value=3.0,step=0.1
    )

with c4:
    zero_ratio = st.number_input(
        'Zero Cosumption Ratio',value=0.02,step=0.01
    )
    longest_zero_streak = st.number_input(
        'Longest Zero Streak',min_value=0,value=2
    )

predict_btn = st.button(
    "🔍 Predict Fraud 🔍",
    use_container_width=True
)
result = None
customer = None
if predict_btn :
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
    with st.spinner('Running Fraud detection model....'):
        result = predict_customer(customer)
    
st.markdown("---")
st.subheader("Prediction Result")

if result is not None :
    c1,c2,c3 = st.columns(3)
    with c1 :
        st.metric(
            "Fraud Probability",
            f'{result["fraud_pobability"]*100:.2f}%'
        )

    with c2 :
        if result['prediction']==1 :
            st.metric('Prediction','FRAUD')
        else :
            st.metric('Prediction','NORMAL')
   
    with c3 :
        st.metric('Risk Level',result['risk'])


    if result['prediction']==1:
        st.error('High Possibility of electricity theft detected.')
    else :
        st.success("Customer appears to be normal")


    if result['risk']=='High':
        st.warning('Immediate investigation is recommened')
    elif result['risk']=='Medium':
        st.info("Customer should be monitored")
    else :
        st.success("Customer has low fraud risk")
    

    st.subheader('Fraud Probability')
    st.progress(
        float(result['fraud_pobability'])
    )
    st.caption(
        f'Fruad Probability : {result['fraud_pobability']:.2%}'
    )
    with st.expander("Prediction Details"):
        st.json(result)

    fig = go.Figure(
    go.Indicator(
        mode="gauge+number",
        value=result["fraud_pobability"]*100,
        title={"text":"Fraud Probability (%)"},
        gauge={"axis":{"range":[0,100]},
                "bar":{"color":"darkred"},
                "steps":[
                {"range":[0,30],"color":"lightgreen"},
                {"range":[30,60],"color":"gold"},
                {"range":[60,100],"color":"salmon"}
                ],

            "threshold":{
                "line":{"color":"red","width":4},
                "thickness":0.8,
                "value":result["fraud_pobability"]*100
            }
        }
    )
)
    st.plotly_chart(fig,use_container_width=True)
    st.markdown("### Risk Levels")
    col1,col2,col3 = st.columns(3)

    with col1:
        st.success("🟢 Low Risk (0–30%)")

    with col2:
        st.warning("🟡 Medium Risk (30–60%)")

    with col3:
        st.error("🔴 High Risk (60–100%)")

    prob = result["fraud_pobability"]

    if prob < 0.30:
        st.success(
        "The customer has a low probability of electricity theft."
    )
    elif prob < 0.60:
        st.warning(
        "The customer shows moderate fraud behaviour. Monitoring is recommended."
    )
    else:
        st.error(
        "The customer has a high probability of electricity theft. Immediate investigation is recommended."
    )

    st.subheader("Prediction Confidence")
    confidence = max(prob,1-prob)
    st.metric(
        'Model Confidence',f'{confidence*100:.2f}%'
    )

    st.markdown("---")
    st.subheader("Customer Feature Summary")
    summary_df = pd.DataFrame(
        customer.items(),columns=['Feature','Value']
    )
    st.dataframe(
        summary_df,use_container_width=True,hide_index=True
    )
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
        lambda row: feature_status(row["Feature"], row["Value"]),
        axis=1
    )
    st.dataframe(
        summary_df,
        use_container_width=True,
        hide_index=True
    )
    report = pd.DataFrame({
        "Fraud Probability":[result["fraud_pobability"]],
        "Prediction":[result["prediction"]],
        "Risk":[result["risk"]]
    })
    csv = report.to_csv(index=False).encode("utf-8")
    st.download_button(
    label="📥 Download Prediction Report",
    data=csv,
    file_name="prediction_report.csv",
    mime="text/csv"
)
    st.caption(
        f"Prediction generated on : {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}"
    )
    st.markdown("---")
    st.markdown(
"""
### ⚡ Electricity Fraud Detection System

Developed using:

- Random Forest Classifier
- Streamlit
- SHAP Explainability
- Python
- Scikit-Learn
"""
)

feature_names = load_feature_name()
explainer = load_explainer()
jb.dump(explainer,r"C:\Users\vinee\ML_Project\models\shap_explainer.pkl")
Exp_path = r"C:\Users\vinee\ML_Project\models\shap_explainer.pkl"
explainer = load_explainer()
def explain_prediction(data_input):
    df = pd.DataFrame([data_input])
    df = df[feature_names]
    shap_values = explainer(df)
    constributions = pd.DataFrame({
        'Feature':feature_names,
        'SHAP Value':shap_values.values[0,:,1]
    })
    constributions['Impact'] = constributions['SHAP Value'].abs()
    constributions = constributions.sort_values('Impact',ascending = False)
    return constributions
    
st.markdown("---")
st.subheader("Why did the mdoel make this prediction ?")
if customer is not None :
    explanation = explain_prediction(customer)
    st.dataframe(
    explanation.head(10),use_container_width=True,hide_index=True
)

    fig = px.bar(
    explanation.head(10),x='SHAP Value',y='Feature',
    orientation='h',title='Top Feature contribution'
)
    fig.update_layout(
    yaxis = dict(autorange='reversed')
)
    st.plotly_chart(fig,use_container_width=True)
    top_feature = explanation.iloc[0]
    st.info(
    f"""
The most influential feature for this prediction is **{top_feature['Feature']}**.

SHAP Contribution: **{top_feature['SHAP Value']:.4f}**

Higher absolute SHAP values indicate a stronger influence on the model's decision.
"""
)
    positive = explanation[explanation["SHAP Value"] > 0]
    negative = explanation[explanation["SHAP Value"] < 0]
    col1, col2 = st.columns(2)
    with col1:
        st.success("### Features Increasing Fraud Risk")
        st.dataframe(
        positive.head(5),
        use_container_width=True,
        hide_index=True
    )

    with col2:
        st.info("### Features Decreasing Fraud Risk")
        st.dataframe(
        negative.head(5),
        use_container_width=True,
        hide_index=True
        )

st.markdown("---")
st.header('Batch Prediction')
upload_file = st.file_uploader(
    'Upload Customer CSV',type=['csv']
)
if upload_file is not None :
    batch_df = pd.read_csv(upload_file)
    st.subheader("Uploaded Dataset")
    st.dataframe(batch_df.head())

    req_features = feature_names 
    missing = [col for col in req_features if col not in batch_df.columns]
    if missing :
        extract_model_feature(batch_df)
        st.error(f'Missing Columns : {missing}')
        st.stop()

    result_df = batch_predict(batch_df)
    st.subheader("Prediction reuslt")
    st.dataframe(result_df,use_container_width=True)

    fraud = (result_df['prediction']==1).sum()
    normal = (result_df['prediction']==0).sum()
    c1,c2 = st.columns(2)
    with c1 :
        st.metric('Fraud Customers',fraud)
    with c2 :
        st.metric('Normal Customers')
    risk_counts = result_df['Risk'].value_counts()
    st.bar_chart(risk_counts)

    csv = result_df.to_csv(index=False).encode('utf-8')
    st.download_button(
    'Download Prediction Results',csv,'Batch_Prediction.csv','text/csv'
)
    fraud_rate = (fraud / len(result_df)) * 100

    st.info(

    f"""
### Batch Prediction Summary

- Total Customers : {len(result_df)}
- Fraud Detected : {fraud}
- Fraud Rate : {fraud_rate:.2f}%
"""
)

from utils import *

save_prediction(customer,result)
try :
    save_prediction_mysql(customer,result)
except Exception as e :
    st.warning(f"MySQL save Failed : {e}")

st.markdown("---")
st.header("📜 Prediction History")
history = load_prediction_history()

if history.empty:

    st.info("No  history available.")

else:

    st.dataframe(
        history,
        use_container_width=True,
        hide_index=True
    )
search = st.text_input(
    "Search Prediction History"
)

if search:

    filtered = history[
        history.astype(str)
        .apply(
            lambda row: row.str.contains(
                search,
                case=False
            ).any(),
            axis=1
        )
    ]

    st.dataframe(
        filtered,
        use_container_width=True,
        hide_index=True
    )

if not history.empty:
    history['Prediction']=history['prediction'].replace({
        0:'Normal',1:'Fraud'
    })
    history.rename(
        columns={
            'fraud_probability':"Fraud Probability",
            'risk':'Risk',
            'prediction_time':'Timestamp'
        },
        inplace=True
    )
    csv = history.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download Prediction History",
        csv,
        "Prediction_History.csv",
        "text/csv"
    )
if st.button("🗑 Clear Prediction History"):
    clear_history()
    st.success("Prediction history deleted.")
    st.rerun()

if not history.empty:
    st.subheader("History Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Total Predictions",
            len(history)
        )
    with col2:
        st.metric(
            "Fraud Cases",
            (history["Prediction"] == 'Fruad').sum()
        )
    with col3:
        st.metric(
            "Normal Cases",
            (history["Prediction"] == 'Normal').sum()
        ) 

st.markdown("---")
st.header("📊 Dashboard Analytics")
if history.empty:
    st.warning("No history available for analytics.")
else:
    prediction_counts = history["Prediction"].value_counts()
    fig = px.pie(
    values=prediction_counts.values,
    names=prediction_counts.index,
    title="Fraud vs Normal Predictions",
    hole=0.45)
    st.plotly_chart(
    fig,use_container_width=True)


    risk = history["Risk"].value_counts()
    fig = px.bar(
    x=risk.index,
    y=risk.values,
    text=risk.values,
    title="Risk Distribution")
    st.plotly_chart(
    fig,use_container_width=True)

    fig = px.histogram(
    history,
    x="fraud_pobability",
    nbins=25,
    title="Fraud Probability Distribution")
    st.plotly_chart(
    fig,use_container_width=True)


    history["Timestamp"] = pd.to_datetime(
    history["Timestamp"])
    timeline = (
    history.groupby(history["Timestamp"].dt.date).size().reset_index(name="Predictions"))


    fig = px.line(
    timeline,
    x="Timestamp",
    y="Predictions",
    markers=True,
    title="Prediction Timeline")

    st.plotly_chart(
    fig,use_container_width=True)

    avg_prob = history["fraud_pobability"].mean() * 100
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=avg_prob,
        title={"text": "Average Fraud Probability"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "darkblue"},
            "steps": [
                {"range": [0, 30], "color": "lightgreen"},
                {"range": [30, 60], "color": "gold"},
                {"range": [60, 100], "color": "salmon"}
            ]
        }
    )
)

    st.plotly_chart(
    fig,use_container_width=True)

    st.subheader("🕒 Recent Predictions")
    st.dataframe(
    history.tail(10),
    use_container_width=True,
    hide_index=True)

    st.subheader("Overall Statistics")

    col1,col2,col3,col4 = st.columns(4)
    with col1:
        st.metric(
        "Total Predictions",
        len(history))

    with col2:
        st.metric(
        "Fraud Cases",
        (history["Prediction"]=='Fraud').sum()
    )
    with col3:
        st.metric(
        "Average Probability",
        f"{history['fraud_pobability'].mean()*100:.2f}%"
    )

    with col4:
        st.metric(
        "High Risk Cases",
        (history["Risk"]=="High").sum()
    )







