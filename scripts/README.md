# Credit Scoring Analysis Project

## Overview

This project aims to analyze credit transaction data to assess customers' risk using various preprocessing, exploratory data analysis (EDA), and feature engineering techniques. The analysis focuses on identifying key patterns and relationships within the data that can be leveraged to develop a predictive model for fraud detection and credit scoring.

## Repository Structure

```plaintext
.
├── preprocessing.py
├── eda_script.py
├── feature_engineering_script.py
└── README.md
```

## Prerequisites

- Python 3.x
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- SciPy

You can install the required libraries using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

## Scripts

### 1. `preprocessing.py`

This script includes functions for loading and preprocessing the dataset. The preprocessing steps are crucial for preparing the data for analysis and modeling.

#### Key Functions:

- **`load_data(file_path)`**: Loads the dataset from a specified CSV file path.
- **`remove_columns(df, columns_to_remove)`**: Removes specified unnecessary columns from the DataFrame.

- **`missing_values_table(df)`**: Displays the missing values and their percentage in each column.

- **`handle_missing_values_median(df, columns)`**: Fills missing values in specified columns with the median.

- **`handle_missing_values_zero(df, columns)`**: Fills missing values in specified columns with zero.

- **`handle_missing_identifiers(df, columns)`**: Drops rows with missing values in specified identifier columns.

- **`handle_missing_numerical(df, columns, threshold, fill_strategy)`**: Handles missing values in numerical columns based on a specified threshold and filling strategy.

- **`handle_missing_categorical(df, columns, fill_value)`**: Fills missing values in categorical columns with a specified value (default is 'Unknown').

- **`detect_outliers(data, numerical_columns)`**: Visualizes outliers in numerical columns using boxplots.

- **`fix_outlier(df, columns)`**: Replaces values higher than the 95th percentile with the median.

- **`remove_outliers(df, columns_to_process, z_threshold)`**: Removes rows flagged as outliers based on the z-score threshold.

- **`get_categorical_columns(df)`**: Identifies categorical columns in the DataFrame.

- **`get_unique_values_count(df, categorical_columns)`**: Returns a DataFrame with unique values, their counts, and percentages for each categorical column.

### 2. `eda_script.py`

This script conducts exploratory data analysis to understand the data better and visualize key relationships.

#### Key Functions:

- **`overview_data(data)`**: Provides an overview of the dataset, including the number of rows, columns, and data types.

- **`summary_statistics(data)`**: Displays summary statistics for numerical features.

- **`plot_numerical_distributions(data, numerical_columns)`**: Visualizes the distribution of numerical features using histograms.

- **`plot_categorical_distributions(data, categorical_columns)`**: Visualizes the distribution of categorical features using bar plots.

- **`correlation_analysis(data, numerical_columns)`**: Plots a correlation matrix for numerical features.

- **`detect_outliers(data, numerical_columns)`**: Visualizes outliers in numerical columns using boxplots.

- **`transaction_frequency_per_user(data)`**: Calculates and visualizes transaction frequency per user.

- **`amount_distribution_by_fraud(data)`**: Visualizes transaction amounts based on fraud results.

- **`avg_transaction_amount_per_user(data)`**: Calculates and visualizes the average transaction amount per user.

- **`fraud_correlation_analysis(data)`**: Analyzes correlations of numerical features with the fraud result.

- **`feature_importance_random_forest(data)`**: Estimates feature importance using a Random Forest model.

- **`extra_eda(data)`**: Additional EDA steps, including checking skewness and duplicated rows.

### 3. `feature_engineering_script.py`

This script contains functions for feature engineering, creating new features that enhance model performance.

#### Key Functions:

- **`create_aggregate_features(data)`**: Creates aggregated features based on each account ID.

- **`extract_date_time_features(data)`**: Extracts features from the transaction start time.

- **`one_hot_encode(data, columns)`**: Encodes categorical columns using one-hot encoding.

- **`label_encode(data, columns)`**: Encodes categorical columns using label encoding.

- **`create_rfm_features(data)`**: Calculates Recency, Frequency, and Monetary (RFM) metrics for customer behavior.

- **`create_credit_debit_ratio(data)`**: Calculates the credit-to-debit ratio for each account.

- **`create_max_transaction(data)`**: Calculates the maximum transaction amount for each account.

- **`create_subscription_features(data)`**: Extracts features related to subscriptions for each account.

- **`calculate_rfm_score(rfm)`**: Calculates the RFM score based on recency, frequency, and monetary metrics.

## Usage

1. **Preprocessing the Data**:

   - Import the `preprocessing_script.py` and use the provided functions to clean and prepare your dataset.

2. **Exploratory Data Analysis**:

   - Import the `eda_script.py` to visualize and analyze the data, helping to uncover patterns and relationships.

3. **Feature Engineering**:
   - Use the functions in `feature_engineering_script.py` to create new features that will enhance the model's predictive capability.

## Example Usage

Here’s a brief example of how you might use these scripts in your main analysis workflow:

```python
import pandas as pd
from preprocessing_script import *
from eda_script import *
from feature_engineering_script import *

# Load your data
data = load_data('path/to/your/data.csv')

# Preprocess your data
data = remove_columns(data, ['UnnecessaryColumn'])
data = handle_missing_values_median(data, ['ColumnWithNaN'])

# Conduct EDA
overview_data(data)
plot_numerical_distributions(data, ['Amount', 'Value'])

# Perform feature engineering
agg_features = create_aggregate_features(data)
rfm_features = create_rfm_features(data)
```

Feel free to modify any sections to better fit your project's specifics or to add additional features and functions as needed!
