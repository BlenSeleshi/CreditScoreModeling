# Credit Scoring EDA and Feature Engineering Notebooks

## Overview

This project includes two Jupyter notebooks focused on Exploratory Data Analysis (EDA) and Feature Engineering for a credit scoring model. The goal is to analyze transaction data to assess customer risk and enhance model performance through well-defined features.

## Repository Structure

```plaintext
.
├── eda_and_preprocessing.ipynb
└── feature_engineering.ipynb
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

## EDA Notebook (`eda_notebook.ipynb`)

This notebook conducts a comprehensive Exploratory Data Analysis to understand the dataset's structure, visualize relationships, and identify patterns.

### 1. Importing the Necessary Packages

```python
import pandas as pd
import numpy as np
import sys, os
import warnings
warnings.filterwarnings('ignore')
```

### 2. Setting the Path for the Script

```python
sys.path.append(os.path.abspath('../scripts'))
import preprocessing as psr
import eda_script as eda
```

### 3. Loading the Dataset

````

### 4. Initial Exploration

- **Overview of the Dataset**:
  ```python
  df.head()
  df.info()
````

- **Checking for Missing Values**:
  ```python
  psr.missing_values_table(df)
  eda.overview_data(df)
  ```

### 5. Summary Statistics

```python
summary = eda.summary_statistics(df)
print(summary)
```

### 6. Distribution of Numerical and Categorical Data

- Define numerical and categorical columns:

  ```python
  numerical_columns = ['Amount', 'Value', 'CountryCode', 'PricingStrategy', 'FraudResult']
  categorical_columns = ['ProviderId', 'ProductCategory', 'ChannelId']
  ```

- Plotting distributions:
  ```python
  eda.plot_numerical_distributions(df, numerical_columns)
  eda.plot_categorical_distributions(df, categorical_columns)
  ```

### 7. Correlation Analysis

```python
eda.correlation_analysis(df, numerical_columns)
```

### 8. Detecting Outliers

```python
eda.detect_outliers(df, numerical_columns)
```

### 9. Extra EDA for Defining High-Risk Users

- **Transaction Frequency per User**:
  ```python
  eda.transaction_frequency_per_user(df)
  ```
- **Amount Distribution by Fraud**:
  ```python
  eda.amount_distribution_by_fraud(df)
  ```
- **Average Transaction Amount per User**:
  ```python
  eda.avg_transaction_amount_per_user(df)
  ```
  Got it! Here's a simplified and structured README for the EDA and Feature Engineering notebooks, similar in style to the previous one without going into detailed code explanations.

---

# Credit Scoring Model EDA and Feature Engineering

This repository contains two notebooks focused on Exploratory Data Analysis (EDA) and Feature Engineering. The aim is to prepare and explore transaction data for building a credit scoring model.

## Table of Contents

1. [Project Structure](#project-structure)
2. [EDA Notebook](#eda-notebook)
3. [Feature Engineering Notebook](#feature-engineering-notebook)
4. [Usage](#usage)
5. [License](#license)

---

## Project Structure

```plaintext
.
├── eda_notebook.ipynb                 # Notebook for Exploratory Data Analysis
├── feature_engineering_notebook.ipynb  # Notebook for Feature Engineering
└── README.md                          # This file
```

## EDA Notebook

The **EDA notebook** (`eda_notebook.ipynb`) is responsible for analyzing and visualizing the dataset. It includes steps to:

- Overview the dataset structure
- Handle missing values
- Generate summary statistics
- Visualize distributions of numerical and categorical variables
- Analyze correlations
- Detect outliers
- Perform additional analyses, such as transaction frequency and fraud-related patterns

---

## Feature Engineering Notebook

The **Feature Engineering notebook** (`feature_engineering_notebook.ipynb`) is focused on generating new features that improve the model's ability to predict credit risk. The key steps include:

- Creating aggregated features based on user transactions
- Extracting date and time features
- Encoding categorical variables
- Creating Recency, Frequency, and Monetary (RFM) features
- Calculating credit-to-debit ratios
- Extracting maximum transaction amount and subscription-related features
- Merging all features into a final dataset

---

## Usage

1. **Exploratory Data Analysis**:

   - Use the **EDA notebook** to gain insights into the dataset and identify important patterns, relationships, and anomalies.

2. **Feature Engineering**:
   - Leverage the **Feature Engineering notebook** to create new features for your model, ensuring you capture the most important aspects of user behavior and transaction patterns.

### How to Run

- Ensure the necessary dependencies are installed.
- Open the notebooks in Jupyter or Google Colab.
- Update the file paths to point to your dataset.
- Run each notebook step-by-step, following the instructions inside.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Feel free to modify this README file based on your specific requirements.
