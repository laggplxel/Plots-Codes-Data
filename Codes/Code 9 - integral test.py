import numpy as np
from scipy.stats import norm
from scipy.integrate import quad

S_c = 2.25
w = 0.25
params_body = {
    'A1': 0.067693, 'mu1': 0.289448, 'sigma1': 0.081570,
    'A2': 0.446319, 'mu2': 0.727425, 'sigma2': 0.112153,
    'A3': 0.406473, 'mu3': 1.147052, 'sigma3': 0.167306, 'alpha3': 1.184845,
    'A4': 0.074950, 'mu4': 1.823121, 'sigma4': 0.142196
}

def psi_body(s, p=params_body):
    """Mixture model for the main distribution body."""
    term1 = p['A1'] * norm.pdf(s, loc=p['mu1'], scale=p['sigma1'])
    term2 = p['A2'] * norm.pdf(s, loc=p['mu2'], scale=p['sigma2'])
    z = (s - p['mu3']) / p['sigma3']
    term3 = p['A3'] * (2 / p['sigma3']) * norm.pdf(z) * norm.cdf(p['alpha3'] * z)
    term4 = p['A4'] * norm.pdf(s, loc=p['mu4'], scale=p['sigma4'])
    return term1 + term2 + term3 + term4

def weight(s, s_c=S_c, width=w):
    """Logistic mixing weight (sigmoid function)."""
    return 1 / (1 + np.exp(-(s - s_c) / width))

def f_wigner(s):
    """Wigner Surmise PDF (GUE), which is the tail of the hybrid model."""
    
    if np.any(s < 0):
        return 0
    return (32 / np.pi**2) * s**2 * np.exp(-4 * s**2 / np.pi)

def psi_hybrid(s):
    """
    The complete analytical PDF model. This is the function to be integrated.
    It must accept a single argument `s` to be compatible with scipy.integrate.quad.
    """
    w_s = weight(s)
    body = psi_body(s)
    tail = f_wigner(s)
    return (1 - w_s) * body + w_s * tail

integral_value, error_estimate = quad(psi_hybrid, 0, np.inf)

print("--- Integral of the Ψ_hybrid(S) Model ---")
print(f"Integrating from S = 0 to S = infinity...")
print("-" * 45)
print(f"Calculated Integral Value: {integral_value:.10f}")
print(f"Estimated Absolute Error: {error_estimate:.2e}")
