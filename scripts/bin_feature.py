import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def bin_feature(data, feature, method='cut', num_bins=5):
    # Check feature variability
    print(data[feature].describe())
    
    # Plot the distribution
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(data[feature], bins=30)
    plt.title(f'Distribution of {feature}')
    
    plt.subplot(1, 2, 2)
    sns.boxplot(x=data[feature])
    plt.title(f'Boxplot of {feature}')
    plt.show()
    
    # Binning based on the method specified
    if method == 'cut':
        data['bin'] = pd.cut(data[feature], bins=num_bins)
    elif method == 'qcut':
        data['bin'] = pd.qcut(data[feature], q=num_bins, duplicates='drop')

    # Check the created bins
    print(data['bin'].value_counts())
    
    return data

# Usage
# Assuming 'data' is your DataFrame
#data = bin_feature(data, 'total_transaction_amount', method='qcut', num_bins=5)
