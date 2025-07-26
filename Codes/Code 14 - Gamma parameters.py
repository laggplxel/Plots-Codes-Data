import os
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, skewnorm
from scipy.integrate import quad
from scipy.special import gamma
import matplotlib.pyplot as plt

# -----------------------------
# Body: 3 Gaussians + Skew-Normal
# -----------------------------
def psi_body(S, p):
    A1, mu1, sigma1, A2, mu2, sigma2, A3, mu3, sigma3, alpha3, A4, mu4, sigma4 = p
    return (
        A1 * norm.pdf(S, loc=mu1, scale=sigma1) +
        A2 * norm.pdf(S, loc=mu2, scale=sigma2) +
        A3 * skewnorm.pdf(S, a=alpha3, loc=mu3, scale=sigma3) +
        A4 * norm.pdf(S, loc=mu4, scale=sigma4)
    )

# -----------------------------
# Tail: Gamma PDF
# -----------------------------
def gamma_tail_pdf(S):
    kappa = 5.57356
    coeff = (kappa ** kappa) / gamma(kappa)
    return coeff * (S + 1e-12)**(kappa - 1) * np.exp(-kappa * (S + 1e-12))

# -----------------------------
# Hybrid Model
# -----------------------------
def hybrid_psi_model(S, p):
    C_tail = 1.0
    body = psi_body(S, p)
    tail = C_tail * gamma_tail_pdf(S)
    cutoff = 2.25
    width = 0.25
    sigmoid = 1 / (1 + np.exp(-(S - cutoff) / width))
    return (1 - sigmoid) * body + sigmoid * tail

# Globals for fitting
S_flips_hist, bin_centers_hist = (None, None)
iteration_counter = 0

# -----------------------------
# Cost Function
# -----------------------------
def cost_function(p):
    if any(val < 0 for val in p):
        return np.inf
    y_predicted = psi_body(bin_centers_hist, p)
    fit_error = np.sum((y_predicted - S_flips_hist)**2)
    integral, _ = quad(lambda s: hybrid_psi_model(s, p), 0, np.inf, limit=100)
    normalization_error = (integral - 1.0)**2
    penalty_weight = 1e6
    return fit_error + penalty_weight * normalization_error

# -----------------------------
# Callback for Monitoring
# -----------------------------
def progress_callback(xk):
    global iteration_counter
    iteration_counter += 1
    if iteration_counter % 50 == 0:
        cost = cost_function(xk)
        print(f"Iteration: {iteration_counter:<5} | Current Cost: {cost:.6f}")

# -----------------------------
# Data Loading
# -----------------------------
try:
    events_path = os.path.join('Data', 'zeta_slope_gaps_1_2000000.csv')
    events_df = pd.read_csv(events_path)
    S_flips = events_df[events_df['flip'] == 1]['S']
    print("Data loaded successfully.")
except FileNotFoundError as e:
    print(f"ERROR: Data file not found. {e}")
    exit()

# Histogram for fitting
S_flips_hist, bin_edges_hist = np.histogram(S_flips, bins=250, density=True)
bin_centers_hist = (bin_edges_hist[:-1] + bin_edges_hist[1:]) / 2

# -----------------------------
# Initial Parameter Guesses
# -----------------------------
p0 = [
    0.067693, 0.289448, 0.081570,        # Gaussian 1
    0.446319, 0.727425, 0.112153,        # Gaussian 2
    0.406473, 1.147052, 0.167306, 1.184845,  # Skew-normal
    0.074950, 1.823121, 0.142196         # Gaussian 3
]

# -----------------------------
# Run Optimization
# -----------------------------
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

# -----------------------------
# Print Optimized Parameters
# -----------------------------
param_names = ['A1', 'mu1', 'sigma1', 'A2', 'mu2', 'sigma2',
               'A3', 'mu3', 'sigma3', 'alpha3', 'A4', 'mu4', 'sigma4']

print("\nOptimized Parameters:")
for name, val in zip(param_names, optimized_params):
    print(f"  {name:<7}: {val:.6f}")

# -----------------------------
# Check Normalization
# -----------------------------
integral_val, _ = quad(lambda s: hybrid_psi_model(s, optimized_params), 0, np.inf)
print(f"\nIntegral of Psi_hybrid(S): {integral_val:.8f}")

# -----------------------------
# Plotting
# -----------------------------
C = 0.5
S_plot = np.linspace(0.01, 4.0, 2000)
psi_vals = hybrid_psi_model(S_plot, optimized_params)
pS_vals = C * (psi_vals / gamma_tail_pdf(S_plot))

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