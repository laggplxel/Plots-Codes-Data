import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load your flip-gap data
df = pd.read_csv("data/zeta_slope_gaps_1_2000000.csv")
s_flips = df[df['flip'] == 1]['S'].values

# Sort and compute ECDF
s_sorted = np.sort(s_flips)
ecdf = np.arange(1, len(s_sorted)+1) / len(s_sorted)

# Plot empirical CDF
plt.step(s_sorted, ecdf, where='post')
plt.xlabel("Normalized Gap S")
plt.ylabel("Empirical CDF")
plt.title("Empirical CDF of Flip Gaps")
plt.grid(True)
plt.show()