import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Function to plot distribution of a single measurement
def plot_measurement_distribution(col_name, df, ax, exclude_zeros=False, location_num=None):
    """
    Plot distribution (histogram + KDE) for a single measurement column.
    
    Args:
        col_name (str): Name of the column to plot
        df (pd.DataFrame): DataFrame containing the data
        ax (matplotlib.axes): Axis to plot on (required)
        exclude_zeros (bool): Whether to exclude zero values (sensor resets). Default False.
        location_num (int): Location number for title (1-indexed). If None, extracted from col_name.
    
    Returns:
        matplotlib.axes: The axis object
    """
    data = df[col_name].copy()
    
    # Exclude zeros if requested
    if exclude_zeros:
        data = data[data != 0]
    
    # Plot histogram
    ax.hist(data, bins=50, alpha=0.6, density=True, edgecolor='black')
    
    # Plot KDE (kernel density estimate)
    try:
        if len(data) > 0:
            kde = stats.gaussian_kde(data)
            x_range = np.linspace(data.min(), data.max(), 200)
            ax.plot(x_range, kde(x_range), 'r-', linewidth=2, alpha=0.8)
    except:
        pass
    
    # Add statistics
    mean_val = data.mean()
    std_val = data.std()
    median_val = data.median()
    zero_count = (df[col_name] == 0).sum()  # Count zeros from original data
    
    ax.axvline(mean_val, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.axvline(median_val, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Extract location number if not provided (1-indexed: Measurement0 -> Location 1)
    if location_num is None:
        try:
            meas_num = int(col_name.replace('Stage1.Output.Measurement', '').replace('.U.Actual', ''))
            location_num = meas_num + 1
        except:
            location_num = col_name
    
    # Set title with location number and statistics
    ax.set_title(f'Location {location_num}\nμ={mean_val:.2f}, σ={std_val:.2f}, Zeros: {zero_count}', 
                 fontsize=9, pad=3)
    ax.set_xlabel('Value', fontsize=7)
    ax.set_ylabel('Density', fontsize=7)
    ax.tick_params(labelsize=6)
    ax.grid(True, alpha=0.3)
    
    return ax


# Function to plot machine variables in a grid layout
def plot_machine_variables(machine_cols, machine_name, df):
    """
    Plots all variables for a given machine in a grid layout.

    Args:
        machine_cols (list): List of column names for the machine.
        machine_name (str): Name of the machine (str for figure title).
        df (pd.DataFrame): DataFrame containing the data.
    """
    if len(machine_cols) > 0:
        n_cols_plot = min(4, len(machine_cols))
        n_rows_plot = (len(machine_cols) + n_cols_plot - 1) // n_cols_plot

        fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(16, 4 * n_rows_plot))
        fig.suptitle(f'{machine_name} Variables', fontsize=14)
        if n_rows_plot == 1:
            axes = axes.reshape(1, -1) if n_cols_plot > 1 else [axes]
        axes = axes.flatten()

        for idx, col in enumerate(machine_cols):
            ax = axes[idx]
            ax.plot(df['time_stamp'], df[col], linewidth=0.8, alpha=0.8)
            ax.set_title(col, fontsize=8, pad=2)
            ax.set_xlabel('Time', fontsize=6)
            ax.set_ylabel('Value', fontsize=6)
            ax.tick_params(labelsize=6)
            ax.grid(True, alpha=0.3)

        for idx in range(len(machine_cols), len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.show()