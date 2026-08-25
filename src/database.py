import mysql.connector
from mysql.connector import Error
from datetime import datetime
from predict import * 
import pandas as pd 

DB_config = {
    'host':'localhost',
    'port':3306,
    'user':'root',
    'password':'root123',
    'database':'electricity_fraud_db'
}
def get_connect():
    try :
        conn = mysql.connector.connect(**DB_config)
        if conn.is_connected():
            return conn
        return None

    except Error as e :
        print('MySQL connection Error',e)
        return None

def test_connection():
    try:
        conn = get_connect()
        return True
        conn.close()
    except :
        return False

def show_table():
    conn = get_connect()
    cursor = conn.cursor()
    try :
        cursor.execute("SHOW TABLES")
        tables = cursor.fetchall()
        return tables
    finally:
        cursor.close()
        conn.close()

def total_record():
    conn = get_connect()
    cursor = conn.cursor()
    try:
        cursor.execute(
        'SELECT COUNT(*) FROM prediction_history'
    )
        total = cursor.fetchone()[0]
        return total
    finally:
        cursor.close()
        conn.close()

def save_prediction_mysql(customer,result):
    if customer is None or result is None :
        return
    conn = get_connect()
    cursor = conn.cursor()
    query = """
    INSERT INTO prediction_history(
    std_cons,Stability,cv,longest_missing_streak,high_cons_ratio,
    PAR,zero_ratio,longest_zero_streak,fraud_pobability,prediction,
    risk,prediction_time
    )
    VALUES(
    %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s
    )"""
    values = (
        customer['std_cons'],
        customer['Stability'],
        customer['cv'],
        customer['longest_missing_streak'],
        customer['high_cons_ratio'],
        customer['PAR'],
        customer['zero_ratio'],
        customer['longest_zero_streak'],
        float(result['fraud_pobability']),
        int(result['prediction']),
        result['risk'],
        datetime.now()
    )
    cursor.execute(query,values)
    conn.commit()
    cursor.close()
    conn.close()

def save_batch_prediction_mysql(result_df):
    conn = get_connect()
    cursor = conn.cursor()
    query = query = """
    INSERT INTO prediction_history(
    std_cons,Stability,cv,longest_missing_streak,high_cons_ratio,
    PAR,zero_ratio,longest_zero_streak,fraud_pobability,prediction,
    risk,prediction_time
    )
    VALUES(
    %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s
    )"""
    values = []
    for _, row in result_df.iterrows():
        values.append((
        row["std_cons"],
        row["Stability"],
        row["cv"],
        row["longest_missing_streak"],
        row["high_cons_ratio"],
        row["PAR"],
        row["zero_ratio"],
        row["longest_zero_streak"],
        float(row["fraud_pobability"]),
        int(row["prediction"]),
        row["risk"],
        datetime.now()
        ))
    cursor.executemany(query,values)
    conn.commit()
    cursor.close()
    conn.close()

    

def load_prediction_history():
    conn = get_connect()
    if conn is None :
        return pd.DataFrame()
    query = """SELECT id,
    std_cons,Stability,cv,longest_missing_streak,high_cons_ratio,
    PAR,zero_ratio,longest_zero_streak,fraud_pobability,prediction,
    risk,prediction_time
    FROM prediction_history
    ORDER BY prediction_time DESC
    """
    try : 
        history = pd.read_sql(query,conn)
    except Exception :
        history = pd.DataFrame()
    finally:
        conn.close()
    return history

def search_prediction_history(search_text):
    conn = get_connect()
    if conn is None :
        return pd.DataFrame()
    query = '''SELECT * FROM prediction_history 
    WHERE risk LIKE %s OR prediction LIKE %s OR CAST(id AS CHAR) LIKE %s
    ORDER BY prediction_time DESC '''
    keyword = f'%{search_text}%'
    df = pd.read_sql(query,conn,params=(keyword,keyword,keyword))
    conn.close()
    return df

def filter_by_risk(risk):
    conn = get_connect()
    if conn is None :
        return pd.DataFrame()
    query = '''SELECT * FROM prediction_history 
    WHERE risk=%s
    ORDER BY prediction_time DESC '''
    df = pd.read_sql(query,conn,params=(risk,))
    conn.close()
    return df

def filter_by_date(start_date,end_date):
    conn = get_connect()
    if conn is None :
        return pd.DataFrame()
    query = '''SELECT * FROM prediction_history 
    WHERE DATE(prediction_time) BETWEEN %s AND %s
    ORDER BY prediction_time DESC '''
    df = pd.read_sql(query,conn,params=(start_date,end_date))
    conn.close()
    return df

def dashboard_statistics():
    conn = get_connect()
    if conn is None :
        return pd.DataFrame()
    query = """
    SELECT
        COUNT(*) Total,
        SUM(prediction=1) Fraud,
        AVG(fraud_pobability) AvgProbability,
        SUM(risk='High') HighRisk
    FROM prediction_history
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df.iloc[0]

def risk_distribution():
    conn = get_connect()
    if conn is None :
        return pd.DataFrame()
    query = """
    SELECT
        risk,
        COUNT(*) Total
    FROM prediction_history
    GROUP BY risk
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def prediction_distribution():
    conn = get_connect()
    if conn is None :
        return pd.DataFrame()
    query = """
    SELECT
        prediction,
        COUNT(*) Total
    FROM prediction_history
    GROUP BY prediction
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def prediction_timeline():
    conn = get_connect()
    if conn is None :
        return pd.DataFrame()
    query = """
    SELECT
        DATE(prediction_time) Date,
        COUNT(*) Predictions
    FROM prediction_history
    GROUP BY DATE(prediction_time)
    ORDER BY Date
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def load_prediction_history_page(limit=100,offset=0):
    conn =get_connect()
    if conn is None :
        return pd.DataFrame()
    query = """ SELECT * FROM prediction_history
    ORDER BY prediction_time DESC LIMIT %s OFFSET %s"""
    df = pd.read_sql(query,conn,params=(limit,offset))
    conn.close()
    return df

def clear_prediction_history():
    conn=get_connect()
    if conn is None:
        return False
    cursor = conn.cursor()
    try :
        cursor.execute('DELETE FROM prediction_history')
        conn.commit()
        return True
    except Exception as e :
        print(e)
        return False
    finally:
        cursor.close()
        conn.close()


if __name__=="__main__":
    table = show_table()
    for t in table :
        print('ok',t[0])
    print()
    print('total record')
    print("  ",total_record())
    