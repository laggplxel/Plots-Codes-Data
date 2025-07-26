import pandas as pd

# Update this path if needed
file_path = "zeta_slope_gaps_1_2000000.csv"

# Load CSV
df = pd.read_csv(file_path)

# Compute average flip probability
flip_prob = df['flip'].mean()
print(f"Average flip probability: {flip_prob:.6f}")