import pandas as pd
import matplotlib.pyplot as plt

# --- Configuration ---
# The specific data file for the Dirichlet L-function analysis
filepath = 'data/Lchi3_slope_gaps_1_10000.csv'

# Number of bins to use for the histogram.
# For ~5000 data points, 75-100 bins is usually a good starting point.
NUM_BINS = 75

def generate_dirichlet_flip_histogram():
    """
    Loads the L(s, χ₃) data and plots the probability density histogram
    of the normalized gaps (S) where a flip occurred.
    """
    # --- 1. Load the data ---
    try:
        print(f"Loading data from '{filepath}'...")
        data = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        print("Please ensure the file is in a 'data' subfolder or provide the correct path.")
        return

    # --- 2. Filter the data for flips ---
    # We only want the 'S' values from rows where 'flip' is 1.
    flip_data = data[data['flip'] == 1]['S']
    print(f"Found {len(flip_data)} flip events to include in the histogram.")

    # --- 3. Generate the plot ---
    print("Generating histogram...")
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 7))

    # Create the histogram. `density=True` is crucial for it to be a PDF.
    plt.hist(flip_data, bins=NUM_BINS, density=True, alpha=0.75,
             label='Empirical Flip-Gap Density')

    # --- Formatting ---
    # Use LaTeX rendering for the title for correct symbol display
    plt.title(r'Flip-Gap Probability Density Function $\Psi(S)$ for $L(s, \chi_3)$', fontsize=16)
    
    plt.xlabel('Normalized Gap S', fontsize=14)
    plt.ylabel('Probability Density', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    
    # Set axis limits to focus on the main distribution
    plt.xlim(0, max(flip_data) * 1.05)

    plt.tight_layout()
    plt.show()

# Run the function to generate and display the plot
generate_dirichlet_flip_histogram()