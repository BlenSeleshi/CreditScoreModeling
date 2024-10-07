import pandas as pd
import numpy as np

def woe_iv(data, feature, target, bins=10):
    """
    Calculate WoE (Weight of Evidence) and IV (Information Value) for a given feature.

    Args:
        data (pd.DataFrame): Input DataFrame containing the feature and target variable.
        feature (str): The feature for which to calculate WoE and IV.
        target (str): The target variable (e.g., 'user_class').
        bins (int): Number of bins for discretization.

    Returns:
        pd.DataFrame: WoE and IV for each bin of the feature.
        float: Total IV value for the feature.
    """
    # Ensure target is numeric
    data[target] = pd.to_numeric(data[target], errors='coerce').fillna(0)

    # Bin the feature into equal-sized buckets
    data['bin'] = pd.qcut(data[feature], q=bins, duplicates='drop')

    # Calculate event and non-event counts
    grouped = data.groupby('bin')[target].agg(['count', 'sum'])
    grouped.columns = ['total', 'good']
    
    # Check types in grouped for debugging
    print(grouped.dtypes)
    
    # Ensure correct types for arithmetic
    grouped['good'] = pd.to_numeric(grouped['good'], errors='coerce')
    grouped['total'] = pd.to_numeric(grouped['total'], errors='coerce')

    grouped['bad'] = grouped['total'] - grouped['good']

    # Calculate distributions
    grouped['good_dist'] = grouped['good'] / grouped['good'].sum()
    grouped['bad_dist'] = grouped['bad'] / grouped['bad'].sum()

    # Calculate WoE while avoiding division by zero
    grouped['WoE'] = np.log(grouped['good_dist'] / grouped['bad_dist'].replace(0, np.nan))

    # Calculate IV
    grouped['IV'] = (grouped['good_dist'] - grouped['bad_dist']) * grouped['WoE']

    # Calculate total IV
    total_iv = grouped['IV'].sum()

    # Prepare and return the output
    return grouped[['total', 'good', 'bad', 'WoE', 'IV']], total_iv

