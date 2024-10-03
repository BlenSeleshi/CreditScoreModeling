# eda_script.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import logging 

def load_data(file_path):
    logging.info("Loading the dataset....")
    """
    Load dataset from the provided file path.
    """
    data = pd.read_csv(file_path)
    return data

def overview_data(data):
    logging.info("Printing overview of the data....")
    """
    Overview of the dataset: number of rows, columns, data types.
    """
    print("Data Overview:")
    print(f"Number of rows: {data.shape[0]}")
    print(f"Number of columns: {data.shape[1]}")
    print("\nData Types:\n", data.dtypes)

def summary_statistics(data):
    
    logging.info("Printing summary statistics of the data....")
    """
    Summary statistics for numerical features.
    """
    print("\nSummary Statistics:")
    return data.describe()

def plot_numerical_distributions(data, numerical_columns):
    
    logging.info("Visualizing the distribution of numerical features....")
    """
    Visualize distribution of numerical features using histograms.
    """
    for column in numerical_columns:
        plt.figure(figsize=(8, 6))
        sns.histplot(data[column], kde=True, bins=30)
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.show()

def plot_categorical_distributions(data, categorical_columns):
    
    logging.info("Visualizing the distribution of categorical features....")
    """
    Visualize distribution of categorical features using bar plots.
    """
    for column in categorical_columns:
        plt.figure(figsize=(8, 6))
        sns.countplot(data[column])
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.xticks(rotation=45)
        plt.show()

def correlation_analysis(data, numerical_columns):
    
    logging.info("Plotting the correlation of numerical features....")
    """
    Correlation matrix for numerical features to understand relationships.
    """
    correlation_matrix = data[numerical_columns].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.show()


def detect_outliers(data, numerical_columns):
    
    logging.info("Detecting outliers....")
    """
    Boxplots to detect outliers in numerical columns.
    """
    for column in numerical_columns:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=data[column])
        plt.title(f'Boxplot of {column}')
        plt.show()

# Extra EDA for answering question 1 (Proxy Variable for High-Risk Users)
def transaction_frequency_per_user(data):
    
    logging.info("Calculating transaction frequence per user....")
    """
    Calculate transaction frequency per user (AccountId) and visualize.
    """
    transaction_counts = data.groupby('AccountId')['TransactionId'].count().sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    sns.histplot(transaction_counts, bins=50, kde=False)
    plt.title('Transaction Frequency per User')
    plt.xlabel('Number of Transactions')
    plt.ylabel('Number of Users')
    plt.show()

    return transaction_counts

def amount_distribution_by_fraud(data):
    
    logging.info("Visualizing the distribution of transaction amounts based on fraud results....")
    """
    Visualize the distribution of transaction amounts based on fraud results.
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='FraudResult', y='Amount', data=data)
    plt.title('Transaction Amount Distribution by Fraud Result')
    plt.xlabel('Fraud Result (1=Fraud, 0=No Fraud)')
    plt.ylabel('Transaction Amount')
    plt.yscale('log')  # Log scale to handle extreme outliers
    plt.show()

def avg_transaction_amount_per_user(data):
    
    logging.info("Calculating average transaction frequence per user....")
    """
    Calculate the average transaction amount per user and visualize.
    """
    avg_transaction_per_user = data.groupby('AccountId')['Amount'].mean().sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    sns.histplot(avg_transaction_per_user, bins=50, kde=False)
    plt.title('Average Transaction Amount per User')
    plt.xlabel('Average Transaction Amount')
    plt.ylabel('Number of Users')
    plt.show()

    return avg_transaction_per_user

# Extra EDA 
def fraud_correlation_analysis(data):
    
    logging.info("Plotting the correlation of numerical features with the fraud result....")
    """
    Correlate numerical features with the fraud result.
    """
    fraud_corr = data.corr()['FraudResult'].sort_values(ascending=False)
    print(fraud_corr)

    # Visualize top correlated features
    plt.figure(figsize=(8, 6))
    fraud_corr.drop('FraudResult').plot(kind='bar')
    plt.title('Correlation of Features with FraudResult')
    plt.ylabel('Correlation Coefficient')
    plt.show()

    return fraud_corr

def feature_importance_random_forest(data):
    
    logging.info("Estimating feature importance for predicting fraud/default....")
    """
    Use a Random Forest to estimate feature importance for predicting fraud/default.
    """
    # Select relevant features for modeling
    features = ['Amount', 'Value', 'ProviderId', 'ProductCategory', 'ChannelId']
    X = data[features]
    y = data['FraudResult']

    # Handle categorical variables
    X = pd.get_dummies(X, drop_first=True)

    # Train Random Forest model
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)

    # Feature importance
    importances = pd.Series(rf.feature_importances_, index=X.columns)
    importances.sort_values(ascending=False).plot(kind='bar', figsize=(10, 6))
    plt.title('Feature Importance for Predicting Fraud/Default')
    plt.ylabel('Importance')
    plt.show()

    return importances

def extra_eda(data):
    
    """
    Additional EDA steps that may be useful for deeper understanding.
    Examples could include checking skewness, kurtosis, or 
    any feature engineering possibilities.
    """
    # Example: Skewness of numerical features
    print("\nSkewness of numerical features:")
    skewness = data.skew(numeric_only=True)
    print(skewness)

    # Checking for duplicated rows
    print("\nChecking for duplicated rows:")
    duplicates = data.duplicated().sum()
    print(f"Number of duplicate rows: {duplicates}")
