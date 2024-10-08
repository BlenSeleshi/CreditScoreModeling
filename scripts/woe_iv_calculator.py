import pandas as pd
import numpy as np

def calculate_woe_iv(df, feature, target):
    """
    Calculates WoE and IV for a given feature.
    """
    # Create a DataFrame to store the bins and WoE values
    df_woe = pd.DataFrame()
    
    # Create bins for the feature and group by target
    df['bin'] = pd.qcut(df[feature], q=10, duplicates='drop')
    grouped = df.groupby('bin')[target].agg(['count', 'sum'])
    
    # Calculate WoE and IV
    grouped['bad'] = grouped['sum']
    grouped['good'] = grouped['count'] - grouped['bad']
    grouped['bad_rate'] = grouped['bad'] / grouped['count']
    grouped['good_rate'] = grouped['good'] / grouped['count']
    grouped['woe'] = np.log((grouped['good_rate'] / grouped['bad_rate']).replace(0, np.nan))
    grouped['iv'] = (grouped['good_rate'] - grouped['bad_rate']) * grouped['woe']
    
    # Add the bin, WoE, and IV values to the df_woe
    df_woe[feature + '_bin'] = grouped.index
    df_woe['WoE'] = grouped['woe'].fillna(0)
    df_woe['IV'] = grouped['iv'].fillna(0)
    
    return df_woe, grouped['iv'].sum()

def apply_woe_binning(df, features, target):
    """
    Applies WoE binning for multiple features and returns a DataFrame with WoE values.
    """
    woe_iv_summary = {}
    for feature in features:
        woe_df, iv = calculate_woe_iv(df, feature, target)
        woe_iv_summary[feature] = iv
        print(f"{feature} Information Value (IV): {iv}")
    return woe_iv_summary

