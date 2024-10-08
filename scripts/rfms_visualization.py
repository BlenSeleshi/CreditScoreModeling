# rfms_visualization.py

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def visualize_rfms(data):
    """
    Visualize the RFMS space (Recency, Frequency, Monetary, Subscription count).

    Args:
        data (pd.DataFrame): DataFrame with recency, frequency, monetary, and subscription columns.

    Returns:
        None
    """
    # Plot RFMS with recency, frequency, and monetary on different axes
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    sns.scatterplot(x=data['recency'], y=data['monetary'], hue=data['RFM_Score'], ax=axs[0])
    axs[0].set_title("Recency vs Monetary")

    sns.scatterplot(x=data['frequency'], y=data['monetary'], hue=data['RFM_Score'], ax=axs[1])
    axs[1].set_title("Frequency vs Monetary")

    sns.scatterplot(x=data['recency'], y=data['frequency'], hue=data['RFM_Score'], ax=axs[2])
    axs[2].set_title("Recency vs Frequency")

    plt.show()

def classify_rfms(data, threshold=7):
    """
    Classify users into 'good' and 'bad' categories based on RFM Score.

    Args:
        data (pd.DataFrame): DataFrame containing RFM_Score.
        threshold (int): Threshold value for classifying good and bad users.

    Returns:
        pd.DataFrame: DataFrame with an added 'user_class' column.
    """
    data['label'] = np.where(data['RFM_Score'] > threshold, 1, 0)
    return data

def visualize_rfm_distribution(df):
    """
    Plots histograms of RFMS scores to visualize the distribution.
    """
    plt.figure(figsize=(12, 6))
    sns.histplot(df['RFM_Score'], kde=True, bins=20)
    plt.title('RFMS Score Distribution')
    plt.xlabel('RFM Score')
    plt.ylabel('Frequency')
    plt.show()

def visualize_label_counts(data):
    """
    Visualize the counts of 'good' and 'bad' labels.

    Args:
        data (pd.DataFrame): DataFrame containing the 'user_class' column.

    Returns:
        None
    """
    # Count the occurrences of each label
    label_counts = data['label'].value_counts()

    # Set the visual style
    sns.set(style="whitegrid")

    # Create a bar plot
    plt.figure(figsize=(8, 5))
    sns.barplot(x=label_counts.index, y=label_counts.values, palette='Set2')
    
    # Add titles and labels
    plt.title('Counts of Good and Bad Labels')
    plt.xlabel('User Class')
    plt.ylabel('Count')
    plt.xticks(rotation=0)  # Rotate x labels if needed
    plt.show()