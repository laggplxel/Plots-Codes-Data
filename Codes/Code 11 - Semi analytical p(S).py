import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

S_c = 2.25
w = 0.25
params_body = {
    'A1': 0.067693, 'mu1': 0.289448, 'sigma1': 0.081570,
    'A2': 0.446319, 'mu2': 0.727425, 'sigma2': 0.112153,
    'A3': 0.406473, 'mu3': 1.147052, 'sigma3': 0.167306, 'alpha3': 1.184845,
    'A4': 0.074950, 'mu4': 1.823121, 'sigma4': 0.142196
}

def psi_body(s, p=params_body):
    term1 = p['A1'] * norm.pdf(s, loc=p['mu1'], scale=p['sigma1'])
    term2 = p['A2'] * norm.pdf(s, loc=p['mu2'], scale=p['sigma2'])
    z = (s - p['mu3']) / p['sigma3']
    term3 = p['A3'] * (2 / p['sigma3']) * norm.pdf(z) * norm.cdf(p['alpha3'] * z)
    term4 = p['A4'] * norm.pdf(s, loc=p['mu4'], scale=p['sigma4'])
    return term1 + term2 + term3 + term4

def weight(s, s_c=S_c, width=w):
    return 1 / (1 + np.exp(-(s - s_c) / width))

def f_wigner(s):
    s_safe = s + 1e-9
    return (32 / np.pi**2) * s_safe**2 * np.exp(-4 * s_safe**2 / np.pi)

def psi_hybrid(s):
    w_s = weight(s)
    body = psi_body(s)
    tail = f_wigner(s)
    return (1 - w_s) * body + w_s * tail

def theoretical_pS(s):
    numerator = psi_hybrid(s)
    denominator = f_wigner(s)
    denominator[denominator < 1e-9] = 1e-9 # Avoid division by zero
    return 0.5 * numerator / denominator

def generate_three_curve_comparison():
    

    try:
        filepath = 'data/zeta_slope_gaps_1_2000000.csv'
        print(f"Loading empirical data from '{filepath}'...")
        data = pd.read_csv(filepath, usecols=['S', 'flip'])
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return

    print("Sorting data and calculating rolling averages...")
    data_sorted = data.sort_values(by='S').reset_index(drop=True)

    
    window_20k = 20000
    window_200k = 200000
    
    empirical_pS_20k = data_sorted['flip'].rolling(window=window_20k, center=True, min_periods=1).mean()
    empirical_pS_200k = data_sorted['flip'].rolling(window=window_200k, center=True, min_periods=1).mean()
    
    empirical_S_values = data_sorted['S']

    
    print("Calculating the smooth theoretical p(S) curve...")
    theoretical_S_values = np.linspace(0.01, 4.0, 1000)
    theoretical_pS_values = theoretical_pS(theoretical_S_values)
    
    
    print("Generating plot...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 8))

    
    ax.plot(empirical_S_values, empirical_pS_200k, color='dodgerblue',
            label=f'Empirical p(S) (Window = {window_200k:,})', lw=2.5, alpha=0.8)
            
    
    ax.plot(empirical_S_values, empirical_pS_20k, color='mediumseagreen',
            label=f'Empirical p(S) (Window = {window_20k:,})', lw=1.5, alpha=0.9)

    
    ax.plot(theoretical_S_values, theoretical_pS_values, color='red',
            label=r'Theoretical $p(S)$', lw=2.5)

    
    ax.set_title("Comparison of Theoretical and Empirical Conditional Flip Probability", fontsize=16)
    ax.set_xlabel("Normalized Gap S", fontsize=14)
    ax.set_ylabel("Conditional Flip Probability p(S)", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True)
    
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 1.25)

    plt.tight_layout()
    plt.show()

generate_three_curve_comparison()