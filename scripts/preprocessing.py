import pandas as pd
import numpy as np
import scipy.stats
import logging
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path):
    logging.info("Loading the dataset....")
    """
    Load dataset from the provided file path.
    """
    data = pd.read_csv(file_path)
    return data

def remove_columns(df, columns_to_remove):
    
    logging.info("Removing Unnecessary Columns....")
    # Check if all columns are in the dataframe
    columns_to_remove = [col for col in columns_to_remove if col in df.columns]
    
    # Drop the columns
    df = df.drop(columns=columns_to_remove)
    
    print(f"Removed columns: {columns_to_remove}")
    
    return df
# Function to display missing values and their percentage in the DataFrame
def missing_values_table(df):
    logging.info("Displaying Missing Value Percentages for Each Column....")
    mis_val = df.isnull().sum()
    
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    
    mis_val_dtype = df.dtypes
    
    mis_val_table = pd.concat([mis_val, mis_val_percent, mis_val_dtype], axis = 1)
    
    mis_val_table_ren_columns = mis_val_table.rename (
        columns={0: 'Missing Values', 1: '% of Total Values', 2: 'DType'}
    )
    
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
            '% of Total Values', ascending=False).round(1)
    
    print ("The dataframe has " + str(df.shape[1]) + "columns.\n"
           "There are " + str(mis_val_table_ren_columns.shape[0]) +
           " columns that have missing values.\n")
           
    return mis_val_table_ren_columns

# Handle missing values (two strategies: median or zero)
def handle_missing_values_median(df, columns):
    """
    Fills missing values with the median for the specified columns.
    :param df: DataFrame to process
    :param columns: List of column names to fill missing values with the median
    :return: DataFrame with missing values filled
    """
    logging.info("Replacing Missing Values with Median....")
    for col in columns:
        df[col].fillna(df[col].median(), inplace=True)
    return df

def handle_missing_values_zero(df, columns):
    """
    Fills missing values with zero for the specified columns.
    :param df: DataFrame to process
    :param columns: List of column names to fill missing values with zero
    :return: DataFrame with missing values filled
    """
    logging.info("Replacing missing values with 0....")
    for col in columns:
        df[col].fillna(0, inplace=True)
    return df
 
# Function to drop rows with missing values in identifier columns
def handle_missing_identifiers(df, columns):
    logging.info("Dropping rows with missing values under the stated column....")
    for column in columns:
        if column in df.columns:
            initial_rows = df.shape[0]
            df.dropna(subset=[column], inplace=True)
            final_rows = df.shape[0]
            print(f"Dropped {initial_rows - final_rows} rows due to missing values in '{column}'")
    return df

# Function to fill missing values for numerical columns based on a percentage threshold
def handle_missing_numerical(df, columns, threshold=65, fill_strategy='mean'):
    logging.info("Filling missing values with mean....")
    for column in columns:
        if column in df.columns:
            missing_percent = df[column].isnull().sum() * 100 / len(df)
            if missing_percent > threshold:
                df.drop(column, axis=1, inplace=True)
                print(f"Dropped column '{column}' due to {missing_percent:.1f}% missing values")
            else:
                if fill_strategy == 'mean':
                    df[column].fillna(df[column].mean(), inplace=True)
                elif fill_strategy == 'median':
                    df[column].fillna(df[column].median(), inplace=True)
                print(f"Filled missing values in '{column}' using {fill_strategy}")
    return df

# Function to fill missing values for categorical columns with 'Unknown'
def handle_missing_categorical(df, columns, fill_value='Unknown'):
    logging.info("Filling missing categorical valuesw with 'Unknown'....")
    for column in columns:
        if column in df.columns:
            df[column].fillna(fill_value, inplace=True)
            print(f"Missing values in categorical column '{column}' filled with '{fill_value}'")
    return df

def detect_outliers(data, numerical_columns):
    
    logging.info("Detecting Outliers....")
    """
    Boxplots to detect outliers in numerical columns.
    """
    for column in numerical_columns:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=data[column])
        plt.title(f'Boxplot of {column}')
        plt.show()

def fix_outlier(df, columns):
    logging.info("Replacing values higher than 95th percentile with median....")
    for column in columns:
        if column in df.columns:
            # Calculate 95th percentile and median
            percentile_95 = df[column].quantile(0.95)
            median_value = df[column].median()
            
            # Replace values higher than 95th percentile with median
            df[column] = np.where(df[column] > percentile_95, median_value, df[column])
           # print(f"Outliers in column {column} fixed (values above 95th percentile replaced with median).")
    
    return df

def remove_outliers(df, columns_to_process, z_threshold=3):
    logging.info("Removing outlier with z-score > 3....")
    for column in columns_to_process:
        if column in df.columns:
            z_scores = zscore(df[column])
            outlier_column = column + '_Outlier'
            
            # Flag rows as outliers
            df[outlier_column] = (np.abs(z_scores) > z_threshold).astype(int)
            
            # Filter out the rows with outliers
            df = df[df[outlier_column] == 0]
            
            # Drop the outlier flag column
            df = df.drop(columns=[outlier_column], errors='ignore')
            #print(f"Outliers removed from column {column} based on Z-score threshold of {z_threshold}.")
    
    return df

# Function to identify categorical columns in a DataFrame
def get_categorical_columns(df):
    logging.info("Fetching the Categorical Columns....")
    categorical_columns = df.select_dtypes(include=['object', 'category','bool']).columns.tolist()
    return categorical_columns


def get_unique_values_count(df, categorical_columns):
    result = []
    
    for col in categorical_columns:
        unique_vals = df[col].value_counts(normalize=True)
        
        for value, percentage in unique_vals.items():
            count = int(df[col].eq(value).sum())
            
            result.append({
                'Column': col,
                'Unique Value': value,
                'Count': count,
                'Percentage': round(percentage * 100, 2)
            })
    
    df_result = pd.DataFrame(result)
    
    # Sort by Column and Count descending
    df_result = df_result.sort_values(['Column', 'Count'], ascending=[True, False])
    
    return df_result
