from __future__ import annotations 
import io 
from datetime import datetime 
from typing import Optional,Iterable
import pandas as pd 

Model_feature = [
    'std_cons','Stability','cv','longest_missing_streak',
    'high_cons_ratio','PAR','zero_ratio','longest_zero_streak'
]
prediction_col = [ 'CONS_NO',
    'std_cons','Stability','cv','longest_missing_streak',
    'high_cons_ratio','PAR','zero_ratio','longest_zero_streak',
    'fraud_pobability','prediction','risk','prediction_time'
]

def _copy_dataframe(data) -> pd.DataFrame :
    'Convert supported input into a safe dataframe'
    if data is None :
        return pd.DataFrame()
    if isinstance(data,pd.DataFrame):
        return data.copy()
    if isinstance(data,dict):
        return pd.DataFrame(data)
    if isinstance(data,(list,tuple)):
        return pd.DataFrame(data)
    raise TypeError("Report data must be pandas dataframe,dict,list,tuple")

def _find_column(df:pd.DataFrame,candidates:Iterable[str])->Optional[str]:
    "Find the fisrst matching column from a list of possible names."
    normalized = {str(column).strip().lower():column for column in df.columns}
    for candidate in candidates :
        key = str(candidate).strip().lower()
        if key is normalized :
            return normalized[key]
    return None

def _normalize_prediction_column(df:pd.DataFrame)->pd.DataFrame:
    'Normalize prediction-based column without changing schema'
    df = df.copy()
    probability_column = _find_column(
        df,['fraud_pobability','fraud_probability','Fraud Probability','Fraud Pobability (%)',],
    )

    if probability_column and probability_column != 'fraud_pobability':
        df['fraud_pobability'] = df[probability_column]

    prediction_col = _find_column(
        df,['prediction','Prediction','predicted','Predicted',],
    )
    
    if prediction_col and prediction_col != 'prediction' :
        df['prediction'] = df[prediction_col]

    risk_col = _find_column(df,
        ['risk','Risk','risk_level','Risk Level',],
    )

    if risk_col and risk_col != 'risk':
        df['risk']=df[risk_col]

    timestamp_col = _find_column(
        df,
        [
            "prediction_time",
            "Prediction Time",
            "timestamp",
            "Timestamp",
        ],
    )

    if timestamp_col and timestamp_col != "prediction_time":
        df["prediction_time"] = df[timestamp_col]

    return df

def _normalize_prediction_values(df : pd.DataFrame) -> pd.DataFrame :
    # Normalize prediction value into : 0 = normal , 1 = fraud
    df = df.copy()
    if 'prediction' not in df.columns :
        return df 
    def normalize(value):
        if pd.isna(value):
            return value
        if isinstance(value,str):
            value_clean = value.strip().lower()
            if value_clean in {'fraud','fraudulent','1','true','yes'}:
                return 1
            if value_clean in {'normal','not fraud','0','false','no'}:
                return 0     

        try : 
            return int(value)
        except (TypeError,ValueError):
            return value 

    df['prediction']=df['prediction'].apply(normalize)
    return df 

def _normalize_risk_values(df:pd.DataFrame)-> pd.DataFrame:
    'normalize risk as Low,Medium,High'
    df = df.copy()
    if 'risk' not in df.columns :
        return df
    df['risk']=(df['risk'].astype(str).str.strip().str.title())
    return df

def prepare_report_dataframe(data) -> pd.DataFrame :
    'prepare prediction data'
    df = _copy_dataframe(data)
    if df.empty :
        return df

    df = _normalize_prediction_column(df)
    df = _normalize_prediction_values(df)
    df = _normalize_risk_values(df)
    return df 


def calculate_report_summ(data)->dict :
    'calculate high level statistics from prediction result '
    df = prepare_report_dataframe(data)
    if df.empty :
        return {
            'total_predictions':0,
            "fraud_cases": 0,
            "normal_cases": 0,
            "fraud_rate": 0.0,
            "average_fraud_probability": 0.0,
            "low": 0,
            "medium": 0,
            "high": 0,
        }

    total = len(df)
    fraud_cases = 0
    if 'prediction' in df.columns :
        fraud_cases = int((df['prediction']==1).sum())
        
    normal_cases = total - fraud_cases
    fraud_rate = ((fraud_cases/total)*100 if total>0 else 0.0)
    average_probability = 0.0
    if 'fraud_pobability' in df.columns :
        probabilities = pd.to_numeric(df['fraud_pobability'],errors='coerce')
        average_probability = (probabilities.mean()*100 if probabilities.notna().any() else 0.0)

    low = 0
    medium = 0
    high = 0 
    if 'risk' in df.columns :
        low = int((df['risk']=='Low').sum())
        medium = int((df['risk']=='Medium').sum())
        high = int((df['risk']=='High').sum())

    return {
        'total_predictions':total,
        "fraud_cases": fraud_cases,
        "normal_cases": normal_cases,
        "fraud_rate": round(fraud_rate,2),
        "average_fraud_probability": round(average_probability,2),
        "low_risk": low,
        "medium_risk": medium,
        "high_risk": high,
    }


def report_metadata(
    report_type : str = 'Prediction Report',
    generated_by : str = 'Electricity Fraud Detection system'
)->dict:
     
     'generating metadata for report'
     return {
        'report_type':report_type,
        'generated_by':generated_by,
        'generated_at' : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
     }

def generate_executive_csv(data)->bytes :
    'generate csv'
    summ = calculate_report_summ(data)
    metadata = report_metadata(report_type='Executive Analytics Report')
    report_rows = [{
            "Report Type": metadata["report_type"],
            "Generated At": metadata["generated_at"],
            "Metric": "Total Predictions",
            "Value": summ["total_predictions"],
        },
        {
            "Report Type": metadata["report_type"],
            "Generated At": metadata["generated_at"],
            "Metric": "Fraud Cases",
            "Value": summ["fraud_cases"],
        },
        {
            "Report Type": metadata["report_type"],
            "Generated At": metadata["generated_at"],
            "Metric": "Normal Cases",
            "Value": summ["normal_cases"],
        },
        {
            "Report Type": metadata["report_type"],
            "Generated At": metadata["generated_at"],
            "Metric": "Fraud Rate (%)",
            "Value": summ["fraud_rate"],
        },
        {
            "Report Type": metadata["report_type"],
            "Generated At": metadata["generated_at"],
            "Metric": "Average Fraud Probability (%)",
            "Value": summ[
                "average_fraud_probability"
            ],
        },
        {
            "Report Type": metadata["report_type"],
            "Generated At": metadata["generated_at"],
            "Metric": "Low Risk Cases",
            "Value": summ["low_risk"],
        },
        {
            "Report Type": metadata["report_type"],
            "Generated At": metadata["generated_at"],
            "Metric": "Medium Risk Cases",
            "Value": summ["medium_risk"],
        },
        {
            "Report Type": metadata["report_type"],
            "Generated At": metadata["generated_at"],
            "Metric": "High Risk Cases",
            "Value": summ["high_risk"],
        },
    ]
    report_df = pd.DataFrame(report_rows)
    return report_df.to_csv(index=False).encode('utf-8')

def generate_prediction_csv(data,include_feature : bool = True)->bytes:
    'generate a detailed prediction csv'
    df = prepare_report_dataframe(data)
    if df.empty:
        empty_df = pd.DataFrame(
            columns= ['CONS_NO','Prediction',"Fraud Probability","Risk","Prediction Time"]
        )
        return empty_df.to_csv(index=False).encode('utf-8')

    output = pd.DataFrame()

    consumer_col = _find_column(df,['CONS_NO','consumer_id','Consumer ID','consumer'],)
    if consumer_col :
        output['CONS_NO']=df[consumer_col]

    if include_feature :
        for feature in Model_feature :
            if feature in df.columns :
                output[feature] = df[feature]

    if 'prediction' in df.columns:
        output['Prediction'] = df['prediction'].map({0:'Normal',1:'Fraud'}).fillna(df['prediction'])

    if 'fraud_pobability' in df.columns :
        probability = pd.to_numeric(df['fraud_pobability'],errors='coerce')
        output['Fraud Probability']=(probability,round(5))
        output['Fraud Probability (%)'] = (probability*100).round(2)

    if 'risk' in df.columns:
        output['Risk']=df['risk']

    if 'prediction_time' in df.columns:
        output['Prediction Time']=(
            pd.to_datetime(df['prediction_time'],errors='coerce')
            .dt.strftime("%Y-%m-%d %H:%M:%S")
        )

    return output.to_csv(index=False).encode('utf-8')




def generate_customer_csv(customer_data,Prediction_result)->bytes:
    'Generate a single customer prediction report'
    customer = _copy_dataframe(customer_data)
    if customer.empty:
        customer = pd.DataFrame([customer_data])

    result = (Prediction_result if isinstance(Prediction_result,dict) else {})
    row = {}
    if not customer.empty:
        customer_row = customer.iloc[0].to_dict()
        for feature in Model_feature :
            if feature in customer_row:
                row[feature]=(customer_row[feature])

    
    prediction = result.get('prediction')
    if prediction == 1 :
        prediction_label = 'Fraud'
    if prediction == 0 :
        prediction_label = 'Normal'
    else:
        prediction_label = prediction
    row['Prediction']=prediction_label


    probability = result.get("fraud_pobability")
    if probability is not None:
        try:
            probability = float(probability)
            row['Fraud Probability'] = round(probability,5)
            row['Fraud Probability (%)'] = round(probability*100,2)
        except (TypeError,ValueError):
            row['Fraud Probability'] = probability


    row['Risk'] = result.get('risk','unknown')

    row['Report Generated At'] = (datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    report_df = pd.DataFrame([row])
    
    return report_df.to_csv(index=False).encode('utf-8')


def dataframe_to_csv_bytes(data,index:bool=False,)->bytes:
    'Generic helper for streamlit download btton'
    df = _copy_dataframe(data)
    return df.to_csv(index=False).encode('utf-8')


def make_report_filename(prefix:str,extension:str='csv',)->str:
    'generate timestamped report filenames.'
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    extension = extension.lstrip(".")
    return (f'{prefix}_{timestamp}.'
            f"{extension}"
    )


#TESTING
if __name__ == "__main__":

    print("=" * 60)
    print("REPORT GENERATOR TEST")
    print("=" * 60)

    sample_data = pd.DataFrame(
        [
            {
                "CONS_NO": "TEST001",
                "std_cons": 3.5,
                "Stability": 0.82,
                "cv": 0.31,
                "longest_missing_streak": 4,
                "high_cons_ratio": 0.61,
                "PAR": 3.5,
                "zero_ratio": 0.12,
                "longest_zero_streak": 2,
                "fraud_pobability": 0.82,
                "prediction": 1,
                "risk": "High",
                "prediction_time": datetime.now(),
            },
            {
                "CONS_NO": "TEST002",
                "std_cons": 1.5,
                "Stability": 0.91,
                "cv": 0.12,
                "longest_missing_streak": 1,
                "high_cons_ratio": 0.21,
                "PAR": 1.8,
                "zero_ratio": 0.03,
                "longest_zero_streak": 1,
                "fraud_pobability": 0.18,
                "prediction": 0,
                "risk": "Low",
                "prediction_time": datetime.now(),
            },
        ]
    )

    print("\n[1] Summary")

    summary = calculate_report_summ(
        sample_data
    )

    for key, value in summary.items():
        print(
            f"{key}: {value}"
        )

    print("\n[2] Generating detailed CSV...")

    csv_bytes = generate_prediction_csv(
        sample_data
    )

    with open(
        "test_prediction_report.csv",
        "wb"
    ) as file:

        file.write(csv_bytes)

    print(
        "Created: test_prediction_report.csv"
    )

    print("\n[3] Generating executive CSV...")

    executive_bytes = (
        generate_executive_csv(
            sample_data
        )
    )

    with open(
        "test_executive_report.csv",
        "wb"
    ) as file:

        file.write(executive_bytes)

    print(
        "Created: test_executive_report.csv"
    )

    print("\n[4] Test completed successfully.")