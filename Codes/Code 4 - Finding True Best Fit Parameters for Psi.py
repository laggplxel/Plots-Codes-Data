import os
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, skewnorm
from scipy.integrate import quad
import matplotlib.pyplot as plt

def psi_body(S, p):
    A1, mu1, sigma1, A2, mu2, sigma2, A3, mu3, sigma3, alpha3, A4, mu4, sigma4 = p
    return (
        A1 * norm.pdf(S, loc=mu1, scale=sigma1) +
        A2 * norm.pdf(S, loc=mu2, scale=sigma2) +
        A3 * skewnorm.pdf(S, a=alpha3, loc=mu3, scale=sigma3) +
        A4 * norm.pdf(S, loc=mu4, scale=sigma4)
    )

def tail(S):
    pi = np.pi
    return (32 / (pi**2)) * (S + 1e-12)**2 * np.exp(-4 * (S + 1e-12)**2 / pi)

def hybrid_psi_model(S, p):
    C_tail = 1.0
    body = psi_body(S, p)
    tail = C_tail * tail(S)
    cutoff = 2.25
    width = 0.25
    sigmoid = 1 / (1 + np.exp(-(S - cutoff) / width))
    return (1 - sigmoid) * body + sigmoid * tail

S_flips_hist, bin_centers_hist = (None, None)
iteration_counter = 0

def cost_function(p):
    if any(val < 0 for val in p): return np.inf
    y_predicted = psi_body(bin_centers_hist, p)
    fit_error = np.sum((y_predicted - S_flips_hist)**2)
    integral, _ = quad(lambda s: hybrid_psi_model(s, p), 0, np.inf, limit=100)
    normalization_error = (integral - 1.0)**2
    penalty_weight = 1e6
    return fit_error + penalty_weight * normalization_error

def progress_callback(xk):
    global iteration_counter
    iteration_counter += 1
    if iteration_counter % 50 == 0:
        cost = cost_function(xk)
        print(f"Iteration: {iteration_counter:<5} | Current Cost: {cost:.6f}")

try:
    events_path = os.path.join('1999999_gaps', 'zeta_slope_gaps_1_2000000.csv')
    events_df = pd.read_csv(events_path)
    S_flips = events_df[events_df['flip'] == 1]['S']
    print("Data loaded successfully.")
except FileNotFoundError as e:
    print(f"ERROR: Data file not found. {e}")
    exit()

S_flips_hist, bin_edges_hist = np.histogram(S_flips, bins=250, density=True)
bin_centers_hist = (bin_edges_hist[:-1] + bin_edges_hist[1:]) / 2

p0 = [
    0.067693, 0.289448, 0.081570,        # Gaussian 1
    0.446319, 0.727425, 0.112153,        # Gaussian 2
    0.406473, 1.147052, 0.167306, 1.184845,  # Skew-normal
    0.074950, 1.823121, 0.142196         # Gaussian 3
]


print("\nRunning optimization...")

result = minimize(
    cost_function, 
    p0, 
    method='Nelder-Mead', 
    callback=progress_callback,
    options={'maxiter': 20000, 'adaptive': True}
)

optimized_params = result.x
print("\nOptimization complete.")

param_names = ['A1', 'mu1', 'sigma1', 'A2', 'mu2', 'sigma2',
               'A3', 'mu3', 'sigma3', 'alpha3', 'A4', 'mu4', 'sigma4']

print("\nOptimized Parameters:")
for name, val in zip(param_names, optimized_params):
    print(f"  {name:<7}: {val:.6f}")

integral_val, _ = quad(lambda s: hybrid_psi_model(s, optimized_params), 0, np.inf)
print(f"\nIntegral of Psi_hybrid(S): {integral_val:.8f}")

C = 0.5
S_plot = np.linspace(0.01, 4.0, 2000)
psi_vals = hybrid_psi_model(S_plot, optimized_params)
pS_vals = C * (psi_vals / wigner_surmise_gue(S_plot))

plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(15, 8))
plt.plot(S_plot, pS_vals, 'r-', linewidth=3, label='Analytical Model for p(S)')
plt.xlabel('Normalized Gap S', fontsize=14)
plt.ylabel('Flip Probability p(S)', fontsize=14)
plt.legend(fontsize=12)
plt.xlim(0, 4.0)
plt.ylim(0, 1.1)
plt.grid(True)
plt.show()