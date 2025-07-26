import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma as gamma_func

# Load data
df = pd.read_csv('Data/zeta_slope_gaps_1_2000000.csv')
S = df['S'].values

# Common histogram settings
bins = np.linspace(0, 4, 100)
hist_kwargs = dict(bins=bins, range=(0,4), density=True, alpha=0.6, color='C0')

# 1) Wigner surmise PDF
def wigner_pdf(s):
    return (32 / np.pi**2) * s**2 * np.exp(-4 * s**2 / np.pi)

# 2) Generalized–Gamma (Stacy) PDF
d, p, a = 2.38342, 1.98133, 1.01145
ggsd_const = p / (a**d * gamma_func(d/p))
def ggsd_pdf(s):
    return ggsd_const * s**(d - 1) * np.exp(- (s / a)**p)

# 3) Gamma distribution PDF
kappa = 5.57356
gamma_const = kappa**kappa / gamma_func(kappa)
def gamma_pdf(s):
    return gamma_const * s**(kappa - 1) * np.exp(-kappa * s)

# Prepare grid for overlay curves
s_vals = np.linspace(0, 4, 500)

# Create subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

# Plot 1: Wigner surmise
axes[0].hist(S, **hist_kwargs, label='Empirical')
axes[0].plot(s_vals, wigner_pdf(s_vals), 'r-', lw=2, label="Wigner surmise")
axes[0].set_title("Wigner surmise")
axes[0].set_xlabel('S')
axes[0].set_ylabel('Density')
axes[0].legend()
axes[0].grid(True)

# Plot 2: Generalized–Gamma (Stacy)
axes[1].hist(S, **hist_kwargs, label='Empirical')
axes[1].plot(s_vals, ggsd_pdf(s_vals), 'r-', lw=2, label="GGSD (Stacy)")
axes[1].set_title("Generalized–Gamma (Stacy)")
axes[1].set_xlabel('S')
axes[1].legend()
axes[1].grid(True)

# Plot 3: Gamma distribution
axes[2].hist(S, **hist_kwargs, label='Empirical')
axes[2].plot(s_vals, gamma_pdf(s_vals), 'r-', lw=2, label="Gamma($\\kappa$=5.57356)")
axes[2].set_title("Gamma distribution")
axes[2].set_xlabel('S')
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.show()