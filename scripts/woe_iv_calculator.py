# woe_binning_utils.py

import pandas as pd
import numpy as np
from scipy.special import xlogy
import matplotlib.pyplot as plt

def woe_binning(data, feature):
    # Split data into good and bad groups
    good = data[data['label'] == 1]
    bad = data[data['label'] == 0]

    # Initialize bins
    bins = np.linspace(min(data[feature]), max(data[feature]), 20)

    # Calculate WoE for each bin
    woebins = []
    for i in range(len(bins) - 1):
        bin_data = data[(data[feature] >= bins[i]) & (data[feature] < bins[i+1])]
        good_count = len(bin_data[bin_data['label'] == 1])
        bad_count = len(bin_data[bin_data['label'] == 0])

        if good_count == 0 or bad_count == 0:
            woebins.append(np.nan)
        else:
            woebins.append(xlogy(good_count / len(good), bad_count / len(bad)) - xlogy((len(data) - good_count) / len(good), (len(data) - bad_count) / len(bad)))

    # Handle edge cases
    woebins[-1] = np.mean([x for x in woebins[:-1] if not np.isnan(x)])

    return pd.cut(data[feature], bins=bins, labels=woebins, include_lowest=True)

def visualize_woe_bins(data, features_to_bin):
    fig, axes = plt.subplots(nrows=len(features_to_bin), ncols=1, figsize=(12, 20))
    for i, feature in enumerate(features_to_bin):
        axes[i].bar(range(len(data[f'{feature}_woe'].unique())), data.groupby(f'{feature}_woe')['label'].mean())
        axes[i].set_title(f'Mean Label by {feature} WoE Bin')
    plt.tight_layout()
    plt.show()

def calculate_information_value(data, features_to_bin):
    iv_values = {}
    for feature in features_to_bin:
        iv = np.sum(data.groupby(f'{feature}_woe')[f'{feature}_woe'].value_counts(normalize=True) * data.groupby(f'{feature}_woe')['label'].mean().diff())
        iv_values[feature] = iv
    return iv_values

def get_top_features_by_iv(iv_values, n=5):
    return sorted(iv_values.items(), key=lambda x: x[1], reverse=True)[:n]

def plot_feature_importance(feature_importances, top_n=10):
    top_features = feature_importances[:top_n]
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(top_features)), [iv for _, iv in top_features])
    plt.xticks(range(len(top_features)), [feature for feature, _ in top_features], rotation=45)
    plt.title('Information Value of Top Features')
    plt.xlabel('Feature')
    plt.ylabel('Information Value')
    plt.tight_layout()
    plt.show()

