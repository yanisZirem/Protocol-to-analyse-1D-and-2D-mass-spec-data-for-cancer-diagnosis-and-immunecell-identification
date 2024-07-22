# Necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import kruskal
from itertools import combinations
from statannot import add_stat_annotation

# Function for peak picking based on local maxima 
def peak_picking(data, min_sn=10):
    import numpy as np
    from scipy.signal import find_peaks
    
    # List to store dataframes of peaks
    peaks_df_list = []
    
    # Drop the 'Class' column from the data as it's not part of the spectra
    ms_data = data.drop(["Class"], axis=1)
    
    # Iterate over each spectrum
    for i in range(len(ms_data)):
        spectrum = ms_data.iloc[i].values  # Extract the spectrum values
        noise_std = np.std(spectrum)  # Calculate the standard deviation of the noise
        threshold = noise_std * min_sn  # Set the peak detection threshold
        
        # Find peaks in the spectrum (peak picking using local maxima) 
        peaks, _ = find_peaks(spectrum, height=threshold)
        
        # Create a dataframe for the detected peaks
        peaks_df_i = pd.DataFrame({
            'spectrum_index': i,
            'm/z': ms_data.columns[peaks],
            'intensity': spectrum[peaks],
        })
        
        # Append the dataframe to the list
        peaks_df_list.append(peaks_df_i)
    
    # Concatenate all peak dataframes into one
    peaks_df = pd.concat(peaks_df_list, ignore_index=True)
    peaks_df = peaks_df.dropna(subset=['m/z'])  # Drop rows with NaN values in 'm/z'
    
    # Pivot the dataframe to have spectra indices as rows and m/z values as columns
    peaks_df = peaks_df.pivot_table(index='spectrum_index', columns='m/z', values='intensity')
    
    # Concatenate the 'Class' column back to the peak data
    data_pick_picked = pd.concat([peaks_df, data['Class']], axis=1)
    
    # Fill NaN values with 0
    data_pick_picked = data_pick_picked.fillna(0) # for the unpicked peaks in certain samples 
    
    return data_pick_picked

# Function for unsupervised clustering heatmap
def create_heatmap(data, cmap='RdYlGn_r', distance_metric='cosine', z_score=0):
    # Create a clustered heatmap of the mean values for each class
    sns.clustermap(data.groupby('Class').mean().T, cmap=cmap, center=0, col_cluster=False, row_cluster=True, metric=distance_metric, z_score=z_score, cbar_kws={'label': ''}, cbar=True, xticklabels=True, yticklabels=False)
    plt.show()

# Function to identify significant features using the Kruskal-Wallis test
def significant_features(data, alpha=0.05):
    x = 'Class'
    y_columns = data.columns.tolist()
    
    # Ensure the 'Class' column is present in the data
    if x in y_columns:
        y_columns.remove(x)
    else:
        print("'Class' column not found in the data.")
        return None
    
    order = data[x].unique()  # Get unique class labels
    significant_columns = []
    num_comparisons = len(y_columns)  # Number of features to test
    corrected_alpha = alpha / num_comparisons  # Bonferroni correction for multiple comparisons
    
    # Perform Kruskal-Wallis test for each feature
    for col in y_columns:
        data_dict = {group: data[col][data[x] == group] for group in order}
        test_statistic, p_value = kruskal(*data_dict.values())
        if p_value <= corrected_alpha:
            significant_columns.append(col)
    
    return significant_columns

# Function to display boxplots of significant features
def boxplot_significant_features(data, mz_values, class_colors=None, test='Kruskal', loc='inside', show_scatter=False):
    label = 'Class'
    order = sorted(data[label].unique())  # Get sorted class labels
    box_pairs = list(combinations(order, 2))  # Create pairs of class labels for statistical annotation
    print("Class labels in dataset:", order)
    
    # Custom color palette for classes
    custom_palette = {class_label: class_colors.get(class_label, 'blue') for class_label in order}
    
    num_mz_values = len(mz_values)
    num_cols = int(num_mz_values ** 0.5)  # Determine number of columns in the plot grid
    num_rows = (num_mz_values + num_cols - 1) // num_cols  # Determine number of rows in the plot grid
    
    figsize_x = 16
    figsize_y = 5 * num_rows
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(figsize_x, figsize_y), dpi=100, squeeze=False)

    # Plot boxplots (and optionally scatter plots) for each m/z value
    for i, mz in enumerate(mz_values):
        x = "Class"
        y = mz
        row = i // num_cols
        col = i % num_cols
        ax = axes[row, col]
        
        sns.boxplot(data=data, x=x, y=y, order=order, ax=ax, palette=custom_palette)
        if show_scatter:
            sns.swarmplot(data=data, x=x, y=y, order=order, ax=ax, color=".25")
        
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        add_stat_annotation(ax, data=data, x=x, y=y, order=order, box_pairs=box_pairs,
                            test=test, text_format='star', loc='inside', verbose=0)
        ax.tick_params(axis='y', labelsize=8)
        ax.set_title(f'Feature: {mz}')
    
    # Remove empty subplots
    for i in range(num_mz_values, num_rows * num_cols):
        fig.delaxes(axes.flatten()[i])
    
    plt.tight_layout()
    plt.show()

# Function to display individual boxplot of a specific m/z value
def one_box_plot(data, mz, test='Kruskal', class_colors=None, show_scatter=False):
    label = 'Class'
    order = sorted(data[label].unique())  # Get sorted class labels
    box_pairs = list(combinations(order, 2))  # Create pairs of class labels for statistical annotation
    print("Class labels in dataset:", order)
    
    x = "Class"
    y = mz
    custom_palette = {class_label: class_colors.get(class_label, 'blue') for class_label in order}
    
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    
    sns.boxplot(data=data, x=x, y=y, order=order, ax=ax, palette=custom_palette)
    if show_scatter:
        sns.swarmplot(data=data, x=x, y=y, order=order, ax=ax, color=".25")
    
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
    add_stat_annotation(ax, data=data, x=x, y=y, order=order, box_pairs=box_pairs,
                        test=test, text_format='star', loc='outside', verbose=2)
    plt.show()

# Function to display violin plots of significant features
def violinplot_significant_features(data, mz_values, class_colors=None, test='Kruskal', loc='inside', show_scatter=False):
    label = 'Class'
    order = sorted(data[label].unique())  
    box_pairs = list(combinations(order, 2))  
    print("Class labels in dataset:", order)
    
    # Custom color palette for classes
    custom_palette = {class_label: class_colors.get(class_label, 'blue') for class_label in order}
    
    num_mz_values = len(mz_values)
    num_cols = int(num_mz_values ** 0.5)  
    num_rows = (num_mz_values + num_cols - 1) // num_cols  
    
    figsize_x = 16
    figsize_y = 5 * num_rows
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(figsize_x, figsize_y), dpi=100, squeeze=False)

    # Plot boxplots (and optionally scatter plots) for each m/z value
    for i, mz in enumerate(mz_values):
        x = "Class"
        y = mz
        row = i // num_cols
        col = i % num_cols
        ax = axes[row, col]
        
        sns.violinplot(data=data, x=x, y=y, order=order, ax=ax, palette=custom_palette)
        if show_scatter:
            sns.swarmplot(data=data, x=x, y=y, order=order, ax=ax, color=".25")
        
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        add_stat_annotation(ax, data=data, x=x, y=y, order=order, box_pairs=box_pairs,
                            test=test, text_format='star', loc='inside', verbose=0)
        ax.tick_params(axis='y', labelsize=8)
        ax.set_title(f'Feature: {mz}')
    
    # Remove empty subplots
    for i in range(num_mz_values, num_rows * num_cols):
        fig.delaxes(axes.flatten()[i])
    
    plt.tight_layout()
    plt.show()

# Function to display individual violinplot of a specific m/z value
def one_violin_plot(data, mz, test='Kruskal', class_colors=None, show_scatter=False):
    label = 'Class'
    order = sorted(data[label].unique())  
    box_pairs = list(combinations(order, 2)) 
    print("Class labels in dataset:", order)
    
    x = "Class"
    y = mz
    custom_palette = {class_label: class_colors.get(class_label, 'blue') for class_label in order}
    
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    
    sns.violinplot(data=data, x=x, y=y, order=order, ax=ax, palette=custom_palette)
    if show_scatter:
        sns.swarmplot(data=data, x=x, y=y, order=order, ax=ax, color=".25")
    
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
    add_stat_annotation(ax, data=data, x=x, y=y, order=order, box_pairs=box_pairs,
                        test=test, text_format='star', loc='outside', verbose=2)
    plt.show()