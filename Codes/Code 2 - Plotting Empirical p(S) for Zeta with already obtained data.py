import pandas as pd
import matplotlib.pyplot as plt

# === Configuration ===
# The window size for the rolling average
ROLLING_WINDOW = 200000

# === Static Filename ===
# Hardcoded path to the data file
gap_csv = "data/zeta_slope_gaps_1_2000000.csv"

# === Load gap data ===
try:
    df_gap = pd.read_csv(gap_csv)
except FileNotFoundError:
    raise FileNotFoundError(f"Cannot find gap file: {gap_csv}")

# --- Rolling average of flip probability ---
print("Sorting data by 'S' and calculating rolling average...")
df_sorted = df_gap.sort_values('S').reset_index(drop=True)

# Calculate the smoothed probability using the specified window
df_sorted['smoothed_prob'] = df_sorted['flip']\
    .rolling(ROLLING_WINDOW, center=True, min_periods=1)\
    .mean()
print("Calculation complete.")

# --- Plot smoothed flip probability ---
print("Generating plot...")
plt.figure(figsize=(12, 7))
plt.plot(df_sorted['S'], df_sorted['smoothed_prob'], lw=2, color='dodgerblue', label='Smoothed Flip Probability')
plt.xlabel('Normalized Gap S', fontsize=12)
plt.ylabel('Flip Probability', fontsize=12)
plt.title(f'Smoothed Flip Probability (Rolling Window = {ROLLING_WINDOW:,})', fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()

# Display the plot on the screen
plt.show()