import os
import pandas as pd
import numpy as np
from scipy.stats import norm, skewnorm
import matplotlib.pyplot as plt

def psi_body(S, p):
    A1, mu1, sigma1, A2, mu2, sigma2, A3, mu3, sigma3, alpha3, A4, mu4, sigma4 = p
    return (
        A1 * norm.pdf(S, loc=mu1, scale=sigma1) +
        A2 * norm.pdf(S, loc=mu2, scale=sigma2) +
        A3 * skewnorm.pdf(S, a=alpha3, loc=mu3, scale=sigma3) +
        A4 * norm.pdf(S, loc=mu4, scale=sigma4)
    )

def wigner_surmise_gue(S):
    pi = np.pi
    return (32 / (pi**2)) * (S + 1e-12)**2 * np.exp(-4 * (S + 1e-12)**2 / pi)

def hybrid_psi(S, p):
    C_tail = 1.0
    body = psi_body(S, p)
    tail = C_tail * wigner_surmise_gue(S)
    cutoff = 2.25
    width = 0.25
    sigmoid = 1 / (1 + np.exp(-(S - cutoff) / width))
    return (1 - sigmoid) * body + sigmoid * tail

params = [
    0.067693, 0.289448, 0.081570,
    0.446319, 0.727425, 0.112153,
    0.406473, 1.147052, 0.167306,
    1.184845, 0.074950, 1.823121, 0.142196
]

try:
    events_path = os.path.join('1999999_gaps', 'zeta_slope_gaps_1_2000000.csv')
    events_df = pd.read_csv(events_path)
    S_flips = events_df[events_df['flip'] == 1]['S']
except FileNotFoundError as e:
    print(f"ERROR: Data file not found. {e}")
    exit()

plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(15, 8))
plt.hist(S_flips, bins=250, density=True, alpha=0.6, label='Empirical Flip Density')
S_plot = np.linspace(0, 4.0, 1000)
psi_curve = hybrid_psi(S_plot, params)
plt.plot(S_plot, psi_curve, 'r-', linewidth=3, label='Analytical $\Psi(S)$')
plt.xlabel('Normalized Gap S', fontsize=12)
plt.ylabel('Probability Density', fontsize=12)
plt.legend()
plt.xlim(0, 3.5)
plt.ylim(bottom=0)
plt.grid(True)
plt.tight_layout()
plt.show()