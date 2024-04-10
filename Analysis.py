# Saloni Jain
#Capstone

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_distribution(df, column_name, kde=True, bins=30, title=None):
    """
    Plots the distribution of a specified column in a histogram.
    """
    title = title or f'Distribution of {column_name}'
    sns.histplot(df[column_name].dropna(), kde=kde, bins=bins)
    plt.title(title)
    plt.xlabel(column_name)
    plt.ylabel('Frequency')
    plt.show()

def plot_dual_distribution(df, column1, column2, kde=True, bins=30, title=None):
    """
    Plots the distributions of two specified columns in a histogram.
    """
    title = title or f'Distribution of {column1} and {column2}'
    sns.histplot(df[column1].dropna(), kde=kde, color='blue', bins=bins)
    sns.histplot(df[column2].dropna(), kde=kde, color='red', bins=bins)
    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()


def plot_correlation_matrix_exclude_datetime(df, figsize=(12, 8)):
    """
    Generates and plots the correlation matrix for numeric variables of a given DataFrame,
    excluding 'Collect DateTime' column.

    Parameters:
    df (DataFrame): The pandas DataFrame for which the correlation matrix is to be computed.
    figsize (tuple): The size of the figure (width, height).
    """
    # Exclude 'Collect DateTime' and select only numeric columns
    if 'Collect DateTime' in df.columns:
        numeric_df = df.drop('Collect DateTime', axis=1).select_dtypes(include=[np.number])
    else:
        numeric_df = df.select_dtypes(include=[np.number])
    
    # Compute the correlation matrix
    correlation_matrix = numeric_df.corr()
    
    # Set up the matplotlib figure
    plt.figure(figsize=figsize)
    
    # Generate a heatmap
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm',
                xticklabels=correlation_matrix.columns,
                yticklabels=correlation_matrix.columns)
    
    # Add title
    plt.title('Correlation Matrix ')
    
    # Show plot
    plt.show()
    
    # Return the correlation matrix
    return correlation_matrix

