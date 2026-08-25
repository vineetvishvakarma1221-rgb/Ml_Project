CREATE DATABASE IF NOT EXISTS electricity_fraud_db;

USE electricity_fraud_db;

CREATE TABLE IF NOT EXISTS prediction_history (
    id INT AUTO_INCREMENT PRIMARY KEY,
    std_cons FLOAT,
    Stability FLOAT,
    cv FLOAT,
    longest_missing_streak INT,
    high_cons_ratio FLOAT,
    PAR FLOAT,
    zero_ratio FLOAT,
    longest_zero_streak INT,
    Fraud Probability FLOAT,
    prediction INT,
    risk VARCHAR(20),
    prediction_time DATETIME
);