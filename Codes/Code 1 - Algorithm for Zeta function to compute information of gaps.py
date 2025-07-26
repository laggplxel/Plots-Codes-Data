import time, os
import numpy as np
import pandas as pd
import mpmath
from numpy.polynomial.legendre import leggauss
from multiprocessing import Pool, Manager
import matplotlib.pyplot as plt

# === Configuration ===
ZEROS_FILE  = 'Data/zeros.txt'       # file of nontrivial zeta zeros
START_ZERO  = 1                 # 1-based index of first zero to process
NUM_ZEROS   = 2000000           # number of zeros (so 1999999 gaps)
MPMATH_DPS  = 15                # mpmath precision
GL_POINTS   = 10                # Gauss–Legendre nodes
NUM_CORES   = 14                # CPU cores for parallel
PRINT_EVERY = 100              # print progress intervals
SMOOTH_FRAC = 0.025             

# === Filenames (all data in data/ folder) ===
date_range        = f"{START_ZERO}_{START_ZERO+NUM_ZEROS-1}"
raw_view_csv      = f"data/zeta_slope_raw_for_viewing_{date_range}.csv"
gap_csv           = f"data/zeta_slope_gaps_{date_range}.csv"
smoothed_csv      = f"data/zeta_slope_smoothed_{date_range}.csv"
plot_png_smoothed = f"data/flip_prob_smoothed_{date_range}.png"

# === Derivative of zeta (complex-step) ===
def zeta_prime(s, h=1e-8):
    return mpmath.zeta(s + 1j*h) / h

# === Gauss–Legendre nodes & weights ===
nodes, weights = leggauss(GL_POINTS)
sigmas = 0.5 * (nodes + 1)
weights *= 0.5

# === Compute sign of slope integral ===
def slope_sign(t):
    prods = []
    for sigma in sigmas:
        d = zeta_prime(mpmath.mpc(sigma, t))
        prods.append(float(d.real * d.imag))
    return 1 if np.dot(prods, weights) > 0 else 0

# === Worker for parallel ===
def process_zero(arg):
    idx, t = arg
    return idx, t, slope_sign(t)

if __name__ == '__main__':
    # --- Load zeros ---
    if not os.path.exists(ZEROS_FILE):
        raise FileNotFoundError(f"Cannot find '{ZEROS_FILE}'.")
    with open(ZEROS_FILE) as f:
        all_zeros = [float(line.strip()) for line in f if line.strip()]
    ts = all_zeros[START_ZERO-1 : START_ZERO-1 + NUM_ZEROS]
    mpmath.mp.dps = MPMATH_DPS

    # --- Parallel slope-sign computation ---
    inputs = list(enumerate(ts, start=1))
    N = len(inputs)
    mgr = Manager()
    counter = mgr.Value('i', 0)
    results = mgr.list()

    def callback(res):
        results.append(res)
        counter.value += 1
        if counter.value % PRINT_EVERY == 0 or counter.value == N:
            print(f"Processed {counter.value}/{N}")

    print(f"Starting slope-sign for zeros {START_ZERO}-{START_ZERO+NUM_ZEROS-1}")
    t0 = time.time()
    with Pool(NUM_CORES) as pool:
        for arg in inputs:
            pool.apply_async(process_zero, args=(arg,), callback=callback)
        pool.close(); pool.join()
    print(f"Computed in {time.time()-t0:.1f}s")

    # --- Sort and unpack results ---
    raw = sorted(results, key=lambda x: x[0])
    idxs, zeros, signs = zip(*raw)

    # --- Write raw CSV for view---
    with open(raw_view_csv, 'w') as f:
        f.write('# Raw Slope Signs\n')
        f.write('Index, Zero (t_n), Sign\n')
        for idx, t, s in zip(idxs, zeros, signs):
            f.write(f'{idx}, {t:.12f}, {"Positive" if s else "Negative"}\n\n')
    print(f"Saved view raw CSV to '{raw_view_csv}'")

    # --- Compute gaps and flips ---
    deltas    = np.diff(zeros)
    mean_gaps = 2 * np.pi / np.log(np.array(zeros[1:]) / (2 * np.pi))
    S_vals    = deltas / mean_gaps
    flips     = [int(signs[i] != signs[i-1]) for i in range(1, N)]

    # --- Build DataFrame for gap data and save ---
    df_gap = pd.DataFrame({
        'idx':   idxs[1:],
        't_n':   zeros[1:],
        'delta': deltas,
        'S':     S_vals,
        'flip':  flips
    })
    df_gap.to_csv(gap_csv, index=False)
    print(f"Saved gap data CSV to '{gap_csv}'")

    # --- Rolling average of flip probability ---
    df_sorted = df_gap.sort_values('S').reset_index(drop=True)
    window = int(SMOOTH_FRAC * len(df_sorted))
    df_sorted['smoothed_prob'] = df_sorted['flip'].rolling(window, center=True, min_periods=1).mean()

    # --- Save smoothed data CSV ---
    df_sorted[['S', 'flip', 'smoothed_prob']].to_csv(smoothed_csv, index=False)
    print(f"Saved smoothed data CSV to '{smoothed_csv}'")

    # --- Plot smoothed flip probability ---
    plt.figure(figsize=(10, 6))
    plt.plot(df_sorted['S'], df_sorted['smoothed_prob'], lw=2, label='Smoothed Flip Probability')
    plt.xlabel('Normalized Gap S')
    plt.ylabel('Flip Probability')
    plt.title(f'Smoothed Flip Probability (window={window})\nZeros {START_ZERO}-{START_ZERO+NUM_ZEROS-1}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_png_smoothed)
    print(f"Saved smoothed plot to '{plot_png_smoothed}'")