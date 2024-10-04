# feature_engineering_script.py

import pandas as pd
import numpy as np

# 1. Aggregate Features

def create_aggregate_features(data):

    agg_data = data.groupby('AccountId').agg(
        total_transaction_amount=pd.NamedAgg(column='Amount', aggfunc='sum'),
        average_transaction_amount=pd.NamedAgg(column='Amount', aggfunc='mean'),
        transaction_count=pd.NamedAgg(column='TransactionId', aggfunc='count'),
        transaction_std=pd.NamedAgg(column='Amount', aggfunc='std')
    ).reset_index()
    
    return agg_data

# 2. Date-Time Features

def extract_date_time_features(data):

    data['TransactionStartTime'] = pd.to_datetime(data['TransactionStartTime'])
    data['transaction_hour'] = data['TransactionStartTime'].dt.hour
    data['transaction_day'] = data['TransactionStartTime'].dt.day
    data['transaction_month'] = data['TransactionStartTime'].dt.month
    data['transaction_year'] = data['TransactionStartTime'].dt.year
    
    return data

# 3. Encoding Categorical Variables

def one_hot_encode(data, columns):

    return pd.get_dummies(data, columns=columns, drop_first=True)

def label_encode(data, columns):

    from sklearn.preprocessing import LabelEncoder
    label_encoders = {}
    
    for col in columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le
    
    return data, label_encoders

# 4. RFM Analysis

def create_rfm_features(data):

    # Convert TransactionStartTime to datetime if not already
    data['TransactionStartTime'] = pd.to_datetime(data['TransactionStartTime'])
    
    # Calculate Recency (days since last transaction)
    current_date = data['TransactionStartTime'].max()
    recency = data.groupby('AccountId')['TransactionStartTime'].apply(lambda x: (current_date - x.max()).days).reset_index()
    recency.columns = ['AccountId', 'recency']
    
    # Calculate Frequency (number of transactions per customer)
    frequency = data.groupby('AccountId')['TransactionId'].count().reset_index()
    frequency.columns = ['AccountId', 'frequency']
    
    # Calculate Monetary (total transaction value per customer)
    monetary = data.groupby('AccountId')['Amount'].sum().reset_index()
    monetary.columns = ['AccountId', 'monetary']
    
    # Merge R, F, M into one dataframe
    rfm = recency.merge(frequency, on='AccountId').merge(monetary, on='AccountId')
    
    return rfm

# 5. Credit-to-Debit Ratio

def create_credit_debit_ratio(data):

    credit_debit = data.groupby('AccountId').apply(lambda x: pd.Series({
        'total_credit': x[x['Amount'] < 0]['Amount'].abs().sum(),
        'total_debit': x[x['Amount'] > 0]['Amount'].sum()
    })).reset_index()

    credit_debit['credit_debit_ratio'] = credit_debit['total_credit'] / (credit_debit['total_debit'] + 1e-9)  # Add a small number to avoid division by zero

    return credit_debit[['AccountId', 'credit_debit_ratio']]

# 6. Maximum Transaction

def create_max_transaction(data):

    max_transaction = data.groupby('AccountId')['Amount'].max().reset_index()
    max_transaction.columns = ['AccountId', 'max_transaction_amount']
    
    return max_transaction

# 7. Number of Subscriptions and Fraud Relationship

def create_subscription_features(data):

    # Number of subscriptions tied to each account
    subscription_count = data.groupby('AccountId')['SubscriptionId'].nunique().reset_index()
    subscription_count.columns = ['AccountId', 'subscription_count']
    
    # Relationship between subscription count and fraud (fraud rate by subscription count)
    fraud_relation = data.groupby('subscription_count')['FraudResult'].mean().reset_index()
    fraud_relation.columns = ['subscription_count', 'fraud_rate']
    
    return subscription_count, fraud_relation
