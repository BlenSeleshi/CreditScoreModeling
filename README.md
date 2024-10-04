# Credit Scoring Model

## Overview

This project focuses on analyzing credit transaction data to assess customer risk by performing Exploratory Data Analysis (EDA), data preprocessing, and feature engineering. The data is prepared and processed for further predictive modeling.

## Repository Structure

```plaintext
.
├── preprocessing_script.py             # Script for data preprocessing
├── eda_script.py                       # Script for exploratory data analysis
├── feature_engineering_script.py       # Script for feature engineering
├── eda_notebook.ipynb                  # Notebook for EDA and preprocessing
├── feature_engineering_notebook.ipynb  # Notebook for feature engineering
└── README.md                           # This file
```

## Prerequisites

- Python 3.x
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

You can install the required libraries using:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Notebooks

### 1. EDA and Preprocessing (`eda_notebook.ipynb`)

This notebook covers:

- Data loading
- Missing value analysis
- Data overview (basic statistics)
- Visualizing numerical and categorical distributions
- Correlation analysis
- Outlier detection
- High-risk user analysis

### 2. Feature Engineering (`feature_engineering_notebook.ipynb`)

This notebook includes:

- Creation of aggregated features
- Extraction of date and time features
- One-hot encoding of categorical variables
- Calculating Recency, Frequency, and Monetary (RFM) features
- Calculating credit-to-debit ratios and maximum transaction amounts
- Analyzing subscription count per account

## Scripts

### 1. Preprocessing (`preprocessing_script.py`)

Handles data cleaning, missing value imputation, and outlier detection.

### 2. EDA (`eda_script.py`)

Performs exploratory data analysis, summary statistics, and correlation visualization.

### 3. Feature Engineering (`feature_engineering_script.py`)

Generates new features like aggregated metrics, RFM features, and credit-debit ratios.

## Usage

1. **EDA and Preprocessing**: Use `eda_notebook.ipynb` to explore the data and preprocess it.
2. **Feature Engineering**: Use `feature_engineering_notebook.ipynb` to create and enhance features for modeling.
3. **Modular Workflow**: You can also run the individual functions from the scripts directly in your custom workflow.
