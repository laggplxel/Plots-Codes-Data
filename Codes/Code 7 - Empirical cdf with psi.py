import pandas as pd
import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import os

# ── 1) Configuration ────────────────────────────────────────────────
NUM_CORES = 14   # adjust as desired
INPUT_CSV = "data/zeta_slope_gaps_1_2000000.csv"
OUTPUT_CSV = "data/flip_model_cdf.csv"

# ── 2) High precision ───────────────────────────────────────────────
mp.mp.dps = 50

# ── 3) Load & sort flip-gap S values ─────────────────────────────────
df = pd.read_csv(INPUT_CSV)
s_sorted = np.sort(df.loc[df['flip']==1, 'S'].values)
N = len(s_sorted)

# ── 4) Define Ψ_hybrid and its CDF ───────────────────────────────────
phi = lambda x: mp.exp(-x**2/2)/mp.sqrt(2*mp.pi)
Phi = lambda x: 0.5*(1 + mp.erf(x/mp.sqrt(2)))
A1, mu1, s1 = 0.067693, 0.289448, 0.081570
A2, mu2, s2 = 0.446319, 0.727425, 0.112153
A3, mu3, s3 = 0.406473, 1.147052, 0.167306
alpha3      = 1.184845
A4, mu4, s4 = 0.074950, 1.823121, 0.142196
Sc, w       = 2.25, 0.25

def norm_pdf(S, m, σ):
    z = (S-m)/σ
    return phi(z)/σ

def skew_pdf(S, α, m, σ):
    z = (S-m)/σ
    return 2*phi(z)/σ * Phi(α*z)

def psi_body(S):
    return (A1*norm_pdf(S,mu1,s1)
          + A2*norm_pdf(S,mu2,s2)
          + A3*skew_pdf(S,alpha3,mu3,s3)
          + A4*norm_pdf(S,mu4,s4))

def psi_tail(S):
    return (32/mp.pi**2) * S**2 * mp.exp(-4*S**2/mp.pi)

def sigma_fun(S):
    return 1/(1 + mp.e**(-(S-Sc)/w))

def psi_hybrid(S):
    if S < 0:
        return mp.mpf('0')
    return (1 - sigma_fun(S))*psi_body(S) + sigma_fun(S)*psi_tail(S)

def F_model_float(S):
    """Return the model CDF F(S) as a float."""
    return float(mp.quad(psi_hybrid, [0, S]))

# ── 5) Main block: parallel compute + save ───────────────────────────
if __name__ == '__main__':
    # make sure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    # allocate array for model CDF
    model_cdf = np.empty(N, dtype=float)

    # compute in parallel with progress bar
    with Pool(processes=NUM_CORES) as pool:
        for i, val in enumerate(tqdm(pool.imap(F_model_float, s_sorted),
                                     total=N,
                                     desc="Computing model CDF")):
            model_cdf[i] = val

    # save S vs. model CDF to CSV
    out_df = pd.DataFrame({
        'S': s_sorted,
        'model_cdf': model_cdf
    })
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved model CDF to '{OUTPUT_CSV}'")

    # optional: plot empirical vs. model CDF
    ecdf = np.arange(1, N+1) / N
    plt.figure(figsize=(8,5))
    plt.step(s_sorted, ecdf, where='post', label='Empirical CDF')
    plt.plot(s_sorted, model_cdf, 'r-', lw=1, label='Model CDF')
    plt.xlabel("Normalized gap S")
    plt.ylabel("CDF")
    plt.title("Empirical vs. Model CDF of Flip Gaps")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()