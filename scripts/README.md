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

## RFMS Visualization and Classification (`rfms_visualization.py`)

This script provides functions to visualize RFMS data and classify users based on their RFMS scores.

### Functions:

#### 1.1. `visualize_rfms(data: pd.DataFrame)`

Visualizes the relationship between Recency, Frequency, and Monetary values.

- **Arguments**:
  - `data`: A pandas DataFrame with columns for `recency`, `frequency`, `monetary`, and an optional `RFM_Score` column.
- **Returns**: None, shows plots for Recency vs Monetary, Frequency vs Monetary, and Recency vs Frequency.

#### 1.2. `classify_rfms(data: pd.DataFrame, threshold: int = 7)`

Classifies users into 'good' (1) and 'bad' (0) categories based on their RFM score.

- **Arguments**:
  - `data`: A pandas DataFrame containing the `RFM_Score` column.
  - `threshold`: A threshold for classifying good and bad users (default is 7).
- **Returns**: DataFrame with an added `label` column (1 for good, 0 for bad).

#### 1.3. `visualize_rfm_distribution(data: pd.DataFrame)`

Plots the distribution of RFMS scores in a histogram.

- **Arguments**:
  - `data`: A pandas DataFrame with an `RFM_Score` column.
- **Returns**: None, shows a plot of the RFMS score distribution.

#### 1.4. `visualize_label_counts(data: pd.DataFrame)`

Visualizes the counts of 'good' and 'bad' users based on the `label` column.

- **Arguments**:
  - `data`: A pandas DataFrame containing the `label` column.
- **Returns**: None, shows a bar plot of 'good' vs 'bad' labels.

### Example Usage:

```python
import pandas as pd
from rfms_visualization import visualize_rfms, classify_rfms, visualize_rfm_distribution, visualize_label_counts

# Example DataFrame
df = pd.DataFrame({
    'recency': [10, 20, 30],
    'frequency': [5, 10, 15],
    'monetary': [100, 200, 300],
    'RFM_Score': [8, 6, 5]
})

# Visualize RFMS relationships
visualize_rfms(df)

# Classify users into good and bad based on RFM score
df = classify_rfms(df, threshold=7)

# Visualize RFMS score distribution
visualize_rfm_distribution(df)

# Visualize counts of good and bad users
visualize_label_counts(df)
```

## 2. Weight of Evidence (WoE) and Information Value (IV) Calculation (`woe_iv_calculator.py`)

This script calculates Weight of Evidence (WoE) and Information Value (IV) for binary classification models and helps with feature selection by ranking features based on IV.

### Functions:

#### 2.1. `woe_binning(data: pd.DataFrame, feature: str, target: str, bins: int = 5)`

Bins a feature into equal-sized bins and calculates WoE for each bin.

- **Arguments**:
  - `data`: A pandas DataFrame containing the feature and target columns.
  - `feature`: The name of the feature to bin.
  - `target`: The binary target column (1 for positive, 0 for negative).
  - `bins`: The number of bins to create (default is 5).
- **Returns**: A pandas Series of WoE values and a DataFrame with bin statistics.

#### 2.2. `calculate_information_value(data: pd.DataFrame, feature: str, target: str, bins: int = 5)`

Calculates the Information Value (IV) for a feature.

- **Arguments**:
  - `data`: A pandas DataFrame containing the feature and target columns.
  - `feature`: The name of the feature to calculate IV for.
  - `target`: The binary target column.
  - `bins`: The number of bins to create (default is 5).
- **Returns**: A float representing the IV of the feature.

#### 2.3. `apply_woe_binning_to_features(data: pd.DataFrame, features: list, target: str, bins: int = 5)`

Applies WoE binning to a list of features and calculates IV for each feature.

- **Arguments**:
  - `data`: A pandas DataFrame containing the features and target.
  - `features`: A list of feature names to apply WoE binning to.
  - `target`: The binary target column.
  - `bins`: The number of bins for each feature (default is 5).
- **Returns**: A dictionary of IV values for each feature.

#### 2.4. `get_top_features_by_iv(iv_values: dict, n: int = 5)`

Retrieves the top N features by IV.

- **Arguments**:
  - `iv_values`: A dictionary of IV values for each feature.
  - `n`: The number of top features to return (default is 5).
- **Returns**: A list of the top N features sorted by IV.

#### 2.5. `visualize_woe_bins(data: pd.DataFrame, features_to_bin: list, target: str, bins: int = 5)`

Visualizes the WoE bins for each feature and the mean target values.

- **Arguments**:
  - `data`: A pandas DataFrame containing the features and target.
  - `features_to_bin`: A list of features to visualize.
  - `target`: The binary target column.
  - `bins`: The number of bins for each feature (default is 5).
- **Returns**: None, shows bar plots of WoE bins and mean target values.

### Example Usage:

```python
import pandas as pd
from woe_iv_calculator import apply_woe_binning_to_features, get_top_features_by_iv, visualize_woe_bins

# Example DataFrame
df = pd.DataFrame({
    'total_transaction_amount': [10000, 20000, 30000],
    'average_transaction_amount': [1000, 2000, 3000],
    'transaction_count': [10, 20, 30],
    'label': [1, 0, 1]
})

# Apply WoE binning and calculate IV for selected features
iv_values = apply_woe_binning_to_features(df, ['total_transaction_amount', 'average_transaction_amount'], 'label')

# Get top features by IV
top_features = get_top_features_by_iv(iv_values, n=2)

# Visualize WoE bins for the top features
visualize_woe_bins(df, ['total_transaction_amount', 'average_transaction_amount'], 'label')
```

Feel free to modify any sections to better fit your project's specifics or to add additional features and functions as needed!
