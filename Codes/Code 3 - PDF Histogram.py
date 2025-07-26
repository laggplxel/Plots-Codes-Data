import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === Configuration ===
CSV_PATH = "data/zeta_slope_gaps_1_2000000.csv"
BINS     = 200       # adjust for smoother/coarser PDF
OUTPUT_PNG = "data/psi_empirical.pdf.png"

# === Load and filter ===
df = pd.read_csv(CSV_PATH)
s_flips = df.loc[df['flip']==1, 'S'].values

# === Compute histogram as a density ===
hist, edges = np.histogram(s_flips, bins=BINS, density=True)
centers = 0.5*(edges[:-1] + edges[1:])

# === Plot ===
plt.figure(figsize=(8,5))
plt.bar(centers, hist, width=(edges[1]-edges[0]), alpha=0.6, 
        label=r'Empirical $\Psi(S)$', edgecolor='black')
plt.xlabel('Normalized gap $S$')
plt.ylabel('Probability density')
plt.title('Empirical Flip-Gap PDF $\\Psi(S)$')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_PNG)
print(f"Saved empirical PDF plot to '{OUTPUT_PNG}'")