# Credit Scoring Model

## Overview

This project aims to build a credit scoring model using RFMS (Recency, Frequency, Monetary) analysis, Weight of Evidence (WoE) techniques, and various exploratory data analysis (EDA) and feature engineering processes to assess customer risk based on eCommerce transaction data. The project encompasses data loading, preprocessing, feature engineering, modeling, and evaluation.

## Repository Structure

```plaintext
.
├── data/
│   ├── aggregated_data.csv         # Raw dataset
│   ├── filtered_data.csv           # Processed dataset for modeling
│
├── models/
│   ├── random_forest_model.pkl      # Trained Random Forest model
│   ├── gradient_boosting_model.pkl   # Trained Gradient Boosting model
│   └── logistic_regression_model.pkl  # Trained Logistic Regression model
│
├── scripts/
│   ├── bin_feature.py               # Functions for binning features
│   ├── preprocessing_script.py       # Script for data preprocessing
│   ├── eda_script.py                 # Script for exploratory data analysis
│   ├── feature_engineering_script.py  # Script for feature engineering
│   ├── rfms_visualization.py        # Functions for RFMS visualization
│   └── woe_iv_calculator.py         # Functions for WoE and IV calculations
|
│├── notebooks/
├── eda_notebook.ipynb                  # Notebook for EDA and preprocessing
├── feature_engineering_notebook.ipynb  # Notebook for feature engineering
├── rfms_woe_analysis.ipynb             # Notebook for RFMS and WoE analysis
└── modeling.ipynb                       # Notebook for modeling and evaluation
└── README.md                           # This file
```

## Prerequisites

To run this project, you will need:

- Python 3.x
- Libraries:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`

You can install the required libraries using pip:

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

### 3. RFMS WoE Analysis (`rfms_woe_analysis.ipynb`)

This notebook performs RFMS analysis and WoE calculations on the dataset. It includes data loading, preprocessing, visualization, classification of users, and saving the filtered dataset for modeling.

### 4. Modeling (`modeling.ipynb`)

This notebook focuses on modeling, where different machine learning algorithms (Logistic Regression, Random Forest, Gradient Boosting) are trained on the filtered dataset. It also includes hyperparameter tuning, evaluation metrics, and model serialization.

## Scripts

### 1. Preprocessing (`preprocessing_script.py`)

Handles data cleaning, missing value imputation, and outlier detection.

### 2. EDA (`eda_script.py`)

Performs exploratory data analysis, summary statistics, and correlation visualization.

### 3. Feature Engineering (`feature_engineering_script.py`)

Generates new features like aggregated metrics, RFM features, and credit-debit ratios.

### 4. Binning Features (`bin_feature.py`)

Contains functions for binning continuous features to categorize them into bins for analysis.

### 5. RFMS Visualization (`rfms_visualization.py`)

Contains functions to visualize RFMS scores, classify users, and visualize distribution and counts.

### 6. WoE and IV Calculator (`woe_iv_calculator.py`)

Contains functions to calculate Weight of Evidence (WoE), Information Value (IV), visualize WoE bins, and get the top features based on IV.

## Usage

1. **EDA and Preprocessing**: Use `eda_notebook.ipynb` to explore the data and preprocess it.
2. **Feature Engineering**: Use `feature_engineering_notebook.ipynb` to create and enhance features for modeling.
3. **RFMS WoE Analysis**: Use `rfms_woe_analysis.ipynb` to perform RFMS and WoE analysis on the dataset.
4. **Modeling**: Use `modeling.ipynb` to load the filtered dataset, train different machine learning models, evaluate their performance, and save the trained models.
5. **Modular Workflow**: You can also run the individual functions from the scripts directly in your custom workflow.

## Results

The results of the model evaluations will be printed to the console, displaying the performance metrics for each model.

## Future Work

Future enhancements may include:

- Exploring additional machine learning algorithms.
- Hyperparameter tuning for other models.
- Implementing cross-validation for more robust evaluation.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Feel free to customize any section further to fit your specific project details!
