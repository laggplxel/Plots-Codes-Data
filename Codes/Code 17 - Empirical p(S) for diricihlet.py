import pandas as pd
import matplotlib.pyplot as plt

# --- Configuration ---
# The specific data file for the Dirichlet L-function analysis
filepath = 'data/Lchi3_slope_gaps_1_10000.csv'

# The window size for the rolling average
ROLLING_WINDOW = 500

def generate_dirichlet_empirical_pS():
    """
    Loads the L(s, χ₃) data and plots the empirical p(S) curve using
    a rolling average to estimate the conditional flip probability.
    """
    # --- 1. Load the data ---
    try:
        print(f"Loading data from '{filepath}'...")
        data = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        print("Please ensure the file is in a 'data' subfolder or provide the correct path.")
        return

    # --- 2. Sort the data by 'S' ---
    print(f"Sorting {len(data)} data points by gap size 'S'...")
    data_sorted = data.sort_values(by='S').reset_index(drop=True)

    # --- 3. Calculate the rolling average ---
    print(f"Calculating rolling average with a window of {ROLLING_WINDOW}...")
    empirical_pS_values = data_sorted['flip'].rolling(
        window=ROLLING_WINDOW,
        center=True,
        min_periods=1
    ).mean()
    
    empirical_S_values = data_sorted['S']

    # --- 4. Generate the plot ---
    print("Generating plot...")
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 7))

    plt.plot(empirical_S_values, empirical_pS_values, color='dodgerblue',
             label=f'Empirical p(S) (Window = {ROLLING_WINDOW})', lw=2)

    # --- Formatting ---
    # *** CORRECTED LINE ***
    # Using r'...' creates a raw string and $...$ enables math rendering.
    plt.title(r'Empirical Conditional Flip Probability for $L(s, \chi_3)$', fontsize=16)
    
    plt.xlabel('Normalized Gap S', fontsize=14)
    plt.ylabel('Empirical Flip Probability p(S)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.xlim(0, max(empirical_S_values) * 1.05)
    plt.ylim(0, 1.0)

    plt.tight_layout()
    plt.show()

# Run the function to generate and display the plot
generate_dirichlet_empirical_pS()