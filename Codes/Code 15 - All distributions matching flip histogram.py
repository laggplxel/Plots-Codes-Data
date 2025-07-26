import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.special import gamma

# --- 1. Define All Model Parameters ---

# --- Sigmoid Transition Parameters (shared by all models) ---
S_c = 2.25
w = 0.25

# --- Body Parameters for Ψ_hybrid_GUE (from original table) ---
params_gue = {
    'A1': 0.067693, 'mu1': 0.289448, 'sigma1': 0.081570,
    'A2': 0.446319, 'mu2': 0.727425, 'sigma2': 0.112153,
    'A3': 0.406473, 'mu3': 1.147052, 'sigma3': 0.167306, 'alpha3': 1.184845,
    'A4': 0.074950, 'mu4': 1.823121, 'sigma4': 0.142196
}

# --- Body Parameters for Ψ_hybrid_GGSD (from first new image) ---
params_ggsd = {
    'A1': 0.065801, 'mu1': 0.289778, 'sigma1': 0.080372,
    'A2': 0.443794, 'mu2': 0.727445, 'sigma2': 0.111845,
    'A3': 0.403737, 'mu3': 1.147317, 'sigma3': 0.166821, 'alpha3': 1.184899,
    'A4': 0.072240, 'mu4': 1.822925, 'sigma4': 0.139481
}

# --- Body Parameters for Ψ_hybrid_Gamma (from second new image) ---
params_gamma = {
    'A1': 0.067063, 'mu1': 0.289559, 'sigma1': 0.081171,
    'A2': 0.445485, 'mu2': 0.727432, 'sigma2': 0.112053,
    'A3': 0.405568, 'mu3': 1.147131, 'sigma3': 0.167155, 'alpha3': 1.185053,
    'A4': 0.074044, 'mu4': 1.823058, 'sigma4': 0.141291
}

# --- Distribution Parameters for the Tails ---
# GGSD
d_ggsd = 2.38342
p_ggsd = 1.98133
a_ggsd = 1.01145
# Gamma
kappa_gamma = 5.57356


# --- 2. Define All Model Functions ---

def psi_body(s, p):
    """Generic mixture model for the body. Takes a parameter dictionary 'p'."""
    term1 = p['A1'] * norm.pdf(s, loc=p['mu1'], scale=p['sigma1'])
    term2 = p['A2'] * norm.pdf(s, loc=p['mu2'], scale=p['sigma2'])
    z = (s - p['mu3']) / p['sigma3']
    term3 = p['A3'] * (2 / p['sigma3']) * norm.pdf(z) * norm.cdf(p['alpha3'] * z)
    term4 = p['A4'] * norm.pdf(s, loc=p['mu4'], scale=p['sigma4'])
    return term1 + term2 + term3 + term4

def weight(s, s_c=S_c, width=w):
    """Shared sigmoid transition function."""
    return 1 / (1 + np.exp(-(s - s_c) / width))

# --- Tail Distribution Functions ---
def f_gue(s):
    """GUE Wigner Surmise PDF."""
    return (32 / np.pi**2) * s**2 * np.exp(-4 * s**2 / np.pi)

def f_ggsd(s):
    """Generalized-Gamma (Stacy) PDF."""
    term1 = p_ggsd / (a_ggsd**d_ggsd * gamma(d_ggsd / p_ggsd))
    term2 = s**(d_ggsd - 1)
    term3 = np.exp(-(s / a_ggsd)**p_ggsd)
    return term1 * term2 * term3

def f_gamma(s):
    """Gamma PDF."""
    return ((kappa_gamma**kappa_gamma) / gamma(kappa_gamma)) * s**(kappa_gamma - 1) * np.exp(-kappa_gamma * s)

# --- Full Hybrid Model Functions ---
def psi_hybrid_gue(s):
    w_s = weight(s)
    body = psi_body(s, params_gue)
    tail = f_gue(s)
    return (1 - w_s) * body + w_s * tail

def psi_hybrid_ggsd(s):
    w_s = weight(s)
    body = psi_body(s, params_ggsd)
    tail = f_ggsd(s)
    return (1 - w_s) * body + w_s * tail

def psi_hybrid_gamma(s):
    w_s = weight(s)
    body = psi_body(s, params_gamma)
    tail = f_gamma(s)
    return (1 - w_s) * body + w_s * tail


# --- 3. Main Plotting Script ---

def generate_hybrid_fit_plots():
    """Loads data and plots the three model fits against the empirical histogram."""
    
    # --- Load and filter the empirical data ---
    try:
        filepath = 'data/zeta_slope_gaps_1_2000000.csv'
        print(f"Loading data from '{filepath}'...")
        data = pd.read_csv(filepath, usecols=['S', 'flip'])
        flipped_s = data[data['flip'] == 1]['S'].values
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return

    # --- Setup for plotting ---
    s_smooth = np.linspace(0.01, 4.5, 1000)
    
    # Calculate the analytical curves
    psi_gue_vals = psi_hybrid_gue(s_smooth)
    psi_ggsd_vals = psi_hybrid_ggsd(s_smooth)
    psi_gamma_vals = psi_hybrid_gamma(s_smooth)
    
    # --- Create the 1x3 horizontal plot ---
    print("Generating plots...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axs = plt.subplots(1, 3, figsize=(21, 6), sharey=True)
    fig.suptitle('Comparison of Optimized Ψ_hybrid Models to Empirical Flip-Gap Histogram', fontsize=18)

    # Common histogram settings
    hist_kwargs = {'bins': 250, 'density': True, 'alpha': 0.6, 'range': (0, 4.5)}
    
    # --- Plot 1: GUE-based Model ---
    axs[0].hist(flipped_s, **hist_kwargs, label='Empirical Flip Density')
    axs[0].plot(s_smooth, psi_gue_vals, 'r-', lw=2, label='Ψ_hybrid (GUE Tail)')
    axs[0].set_title('GUE-based Model Fit', fontsize=14)
    axs[0].set_ylabel('Probability Density', fontsize=12)
    axs[0].set_xlabel('Normalized Gap S', fontsize=12)
    axs[0].legend()
    
    # --- Plot 2: GGSD-based Model ---
    axs[1].hist(flipped_s, **hist_kwargs)
    axs[1].plot(s_smooth, psi_ggsd_vals, 'g-', lw=2, label='Ψ_hybrid (GGSD Tail)')
    axs[1].set_title('GGSD-based Model Fit', fontsize=14)
    axs[1].set_xlabel('Normalized Gap S', fontsize=12)
    axs[1].legend()

    # --- Plot 3: Gamma-based Model ---
    axs[2].hist(flipped_s, **hist_kwargs)
    axs[2].plot(s_smooth, psi_gamma_vals, 'purple', lw=2, label='Ψ_hybrid (Gamma Tail)')
    axs[2].set_title('Gamma-based Model Fit', fontsize=14)
    axs[2].set_xlabel('Normalized Gap S', fontsize=12)
    axs[2].legend()
    
    for ax in axs:
        ax.set_xlim(0, 4.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# Run the plotting function
generate_hybrid_fit_plots()