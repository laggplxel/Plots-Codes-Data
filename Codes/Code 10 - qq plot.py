import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_qq_plot_from_file():
    """
    Generates a Quantile-Quantile (Q-Q) plot to compare the empirical flip-gap
    data against the theoretical model using a pre-computed CDF file.
    """
    # --- 1. Load Empirical Data (The actual S values from your experiment) ---
    try:
        data_filepath = 'data/zeta_slope_gaps_1_2000000.csv'
        print(f"Loading empirical data from '{data_filepath}'...")
        data = pd.read_csv(data_filepath, usecols=['S', 'flip'])
        
        # Filter for flips and sort to get empirical quantiles
        empirical_quantiles = np.sort(data[data['flip'] == 1]['S'].values)
        print(f"Found {len(empirical_quantiles):,} empirical data points (flips).")
        
    except FileNotFoundError:
        print(f"Error: The data file '{data_filepath}' was not found.")
        print("Please ensure it is in a 'data' subfolder.")
        return

    # --- 2. Load Theoretical Model CDF (Your pre-computed file) ---
    try:
        cdf_filepath = 'data/flip_model_cdf.csv'
        print(f"Loading model CDF from '{cdf_filepath}'...")
        cdf_data = pd.read_csv(cdf_filepath)
        
        # Extract the columns for interpolation
        s_from_cdf_file = cdf_data['S'].values
        model_cdf_values = cdf_data['model_cdf'].values
        print("Model CDF loaded successfully.")

    except FileNotFoundError:
        print(f"Error: The CDF file '{cdf_filepath}' was not found.")
        print("Please ensure it is in a 'data' subfolder.")
        return

    # --- 3. Calculate Theoretical Quantiles ---
    print("Calculating theoretical quantiles using interpolation...")
    
    # Create a set of probability points corresponding to the empirical data
    n = len(empirical_quantiles)
    # These are the "plotting positions" for the quantiles, from ~0 to ~1
    prob_points = (np.arange(n) + 0.5) / n

    # Use interpolation to find the S-value (quantile) for each probability point.
    # This is the Inverse CDF (or PPF) lookup.
    # np.interp(x_new, x_known, y_known)
    theoretical_quantiles = np.interp(prob_points, model_cdf_values, s_from_cdf_file)
    print("Theoretical quantiles generated.")


    # --- 4. Generate the Q-Q Plot ---
    print("Generating plot...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(9, 9))

    # Scatter plot of theoretical vs. empirical quantiles
    # Use small points and transparency due to the large number of points
    ax.scatter(theoretical_quantiles, empirical_quantiles, s=2, alpha=0.1, color='dodgerblue', rasterized=True)
    
    # Add the y = x line for a perfect-fit reference
    max_val = max(np.max(theoretical_quantiles), np.max(empirical_quantiles))
    ax.plot([0, max_val], [0, max_val], 'r-', lw=2, label='y = x (Perfect Fit)')
    
    ax.set_xlabel("Theoretical Quantiles (from Ψ_hybrid Model)", fontsize=13)
    ax.set_ylabel("Empirical Quantiles (from Data)", fontsize=13)
    ax.set_title("Q-Q Plot: Model Fit for Flip-Gap Distribution", fontsize=15)
    ax.legend(fontsize=11)
    
    # Make the plot square for accurate visual interpretation of the reference line
    ax.set_aspect('equal', 'box')
    ax.grid(True)
    
    # Set axis limits to zoom in on the main data body if needed
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)

    # Use a tight layout
    plt.tight_layout()
    plt.show()

# Run the function
generate_qq_plot_from_file()