import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def woe_binning(data, feature, target, bins=5):
    """
    Bins a given feature into equal-sized bins and calculates WoE for each bin.
    
    Args:
        data (pd.DataFrame): The dataset containing the feature and target.
        feature (str): The feature to calculate WoE for.
        target (str): The target column (should be binary: 1 for good, 0 for bad).
        bins (int): The number of bins to create for the feature.
    
    Returns:
        pd.Series: WoE values for the feature.
        pd.DataFrame: DataFrame with the binned data and corresponding WoE values.
    """
    logger.info(f"Starting WoE binning for feature: {feature}")
    
    # Ensure no missing values
    data = data[[feature, target]].dropna()
    
    # Create equal-frequency bins
    try:
        data['bin'], bin_edges = pd.qcut(data[feature], q=bins, retbins=True, duplicates='drop')
    except Exception as e:
        logger.error(f"Error in binning for feature {feature}: {e}")
        return pd.Series(), pd.DataFrame()

    # Calculate total good (target == 1) and bad (target == 0)
    total_good = len(data[data[target] == 1])
    total_bad = len(data[data[target] == 0])
    
    # Group by bins and calculate counts
    bin_stats = data.groupby('bin').agg(
        good_count=(target, lambda x: (x == 1).sum()),
        bad_count=(target, lambda x: (x == 0).sum())
    )
    
    # Calculate percentage of good and bad in each bin
    bin_stats['good_pct'] = bin_stats['good_count'] / total_good
    bin_stats['bad_pct'] = bin_stats['bad_count'] / total_bad
    
    # Replace 0 to avoid division by zero in WoE calculation
    bin_stats['good_pct'].replace(0, 1e-5, inplace=True)
    bin_stats['bad_pct'].replace(0, 1e-5, inplace=True)
    
    # Calculate WoE
    bin_stats['WoE'] = np.log(bin_stats['good_pct'] / bin_stats['bad_pct'])
    
    # Add WoE back to original data
    data = data.merge(bin_stats[['WoE']], left_on='bin', right_index=True, how='left')
    
    logger.info(f"Completed WoE binning for feature: {feature}")
    
    return data['WoE'], bin_stats

def calculate_information_value(data, feature, target, bins=5):
    """
    Calculates the Information Value (IV) for a given feature.
    
    Args:
        data (pd.DataFrame): The dataset containing the feature and target.
        feature (str): The feature to calculate IV for.
        target (str): The target column (should be binary: 1 for good, 0 for bad).
        bins (int): The number of bins to create for the feature.
    
    Returns:
        float: The Information Value (IV) for the feature.
    """
    # Perform WoE binning
    _, bin_stats = woe_binning(data, feature, target, bins=bins)
    
    if bin_stats.empty:
        logger.error(f"No valid IV calculated for {feature}. Returning NaN.")
        return np.nan
    
    # Calculate IV
    bin_stats['IV'] = (bin_stats['good_pct'] - bin_stats['bad_pct']) * bin_stats['WoE']
    iv = bin_stats['IV'].sum()
    
    logger.info(f"Information Value (IV) for {feature}: {iv}")
    return iv

def apply_woe_binning_to_features(data, features, target, bins=5):
    """
    Applies WoE binning to a list of features and calculates IV for each.
    
    Args:
        data (pd.DataFrame): The dataset containing the features and target.
        features (list): List of feature names to apply WoE binning to.
        target (str): The target column.
        bins (int): Number of bins for each feature.
    
    Returns:
        dict: Dictionary of IV values for each feature.
    """
    iv_values = {}
    for feature in features:
        iv = calculate_information_value(data, feature, target, bins)
        iv_values[feature] = iv
    return iv_values

def get_top_features_by_iv(iv_values, n=5):
    """
    Retrieves the top N features by Information Value (IV).
    
    Args:
        iv_values (dict): Dictionary of IV values for each feature.
        n (int): The number of top features to return.
    
    Returns:
        list: Sorted list of top N features by IV.
    """
    valid_ivs = [(feature, iv) for feature, iv in iv_values.items() if not np.isnan(iv)]
    return sorted(valid_ivs, key=lambda x: x[1], reverse=True)[:n]

def visualize_woe_bins(data, features_to_bin, target, bins=5):
    """
    Visualizes WoE bins and target mean for each feature.
    
    Args:
        data (pd.DataFrame): The dataset containing the features and target.
        features_to_bin (list): List of features to visualize.
        target (str): The target column.
        bins (int): Number of bins for each feature.
    """
    logger.info("Visualizing WoE bins")
    
    fig, axes = plt.subplots(nrows=len(features_to_bin), ncols=1, figsize=(12, 20))
    
    for i, feature in enumerate(features_to_bin):
        # Perform WoE binning
        data[f'{feature}_woe'], bin_stats = woe_binning(data, feature, target, bins)
        
        if not bin_stats.empty:
            # Plot mean target by WoE bin
            bin_means = data.groupby(f'{feature}_woe')[target].mean()
            axes[i].bar(bin_means.index.astype(str), bin_means.values)
            axes[i].set_title(f'Mean {target} by {feature} WoE bin')
    
    plt.tight_layout()
    plt.show()

def plot_feature_importance(feature_importances, top_n=10):
    """
    Plots the Information Value of the top N features.
    
    Args:
        feature_importances (list): List of (feature, IV) tuples.
        top_n (int): Number of top features to plot.
    """
    top_features = feature_importances[:top_n]
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(top_features)), [iv for _, iv in top_features])
    plt.xticks(range(len(top_features)), [feature for feature, _ in top_features], rotation=45)
    plt.title('Information Value of Top Features')
    plt.xlabel('Feature')
    plt.ylabel('Information Value')
    plt.tight_layout()
    plt.show()
