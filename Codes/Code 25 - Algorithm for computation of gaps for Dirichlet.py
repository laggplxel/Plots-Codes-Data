import time, os
import numpy as np
import pandas as pd
import mpmath
from numpy.polynomial.legendre import leggauss
from multiprocessing import Pool, Manager

# === Configuration ===
ZEROS_FILE  = 'Dirichlet_zeros.txt'
START_ZERO  = 1
NUM_ZEROS   = 10000
MPMATH_DPS  = 15
GL_POINTS   = 10
NUM_CORES   = 14
PRINT_EVERY = 100
SMOOTH_FRAC = 0.025

# === Output directory ===
gaps_count = NUM_ZEROS - 1
output_dir = f"{gaps_count}_gaps_Lchi3"
os.makedirs(output_dir, exist_ok=True)

# === Filenames ===
date_range        = f"{START_ZERO}_{START_ZERO+NUM_ZEROS-1}"
raw_pretty_csv    = os.path.join(output_dir, f"Lchi3_slope_raw_pretty_{date_range}.csv")
gap_csv           = os.path.join(output_dir, f"Lchi3_slope_gaps_{date_range}.csv")

# === Dirichlet character χ₃ ===
def chi3(n: int) -> int:
    r = n % 3
    if r == 0: return 0
    return 1 if r == 1 else -1

q = 3
a = 1
tau_chi = mpmath.sqrt(q)

def eps_factor(s):
    return (tau_chi *
            mpmath.power(mpmath.pi/q, 0.5 - s) *
            mpmath.gamma((1 - s + a)/2) /
            mpmath.gamma((s + a)/2))

def L_chi3(s):
    t = abs(mpmath.im(s))
    N = int(mpmath.floor(mpmath.sqrt(q*(t + 3)/mpmath.pi))) + 8
    main = mpmath.nsum(lambda n: chi3(n)/n**s, [1, N])
    dual = mpmath.nsum(lambda n: chi3(n)/n**(1 - s), [1, N])
    return main + eps_factor(s)*dual

def L_chi3_prime(s, h=1e-8):
    return (L_chi3(s + 1j*h)) / h

# === Gauss–Legendre nodes & weights ===
nodes, weights = leggauss(GL_POINTS)
sigmas = 0.5 * (nodes + 1)
weights *= 0.5

def slope_sign(t):
    prods = []
    for sigma in sigmas:
        d = L_chi3_prime(mpmath.mpc(sigma, t))
        prods.append(float(d.real * d.imag))
    return 1 if np.dot(prods, weights) > 0 else 0

def process_zero(arg):
    idx, t = arg
    return idx, t, slope_sign(t)

if __name__ == '__main__':
    if not os.path.exists(ZEROS_FILE):
        raise FileNotFoundError(f"Cannot find '{ZEROS_FILE}'.")

    with open(ZEROS_FILE) as f:
        all_zeros = [float(line.strip()) for line in f if line.strip()]
    ts = all_zeros[START_ZERO-1 : START_ZERO-1 + NUM_ZEROS]
    mpmath.mp.dps = MPMATH_DPS

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

    print(f"Starting slope-sign for Dirichlet L(s, χ₃) zeros {START_ZERO}-{START_ZERO+NUM_ZEROS-1}")
    t0 = time.time()
    with Pool(NUM_CORES) as pool:
        for arg in inputs:
            pool.apply_async(process_zero, args=(arg,), callback=callback)
        pool.close(); pool.join()
    print(f"Done in {time.time()-t0:.1f} seconds")

    raw = sorted(results, key=lambda x: x[0])
    idxs, zeros, signs = zip(*raw)

    with open(raw_pretty_csv, 'w') as f:
        f.write('# Raw Slope Signs (L-function)\n')
        f.write('Index, Zero (t_n), Sign\n')
        for idx, t, s in zip(idxs, zeros, signs):
            f.write(f'{idx}, {t:.12f}, {"Positive" if s else "Negative"}\n\n')
    print(f"Saved slope signs to '{raw_pretty_csv}'")

    deltas    = np.diff(zeros)
    mean_gaps = 2 * np.pi / np.log(np.array(zeros[1:]) / (2 * np.pi))
    S_vals    = deltas / mean_gaps
    flips     = [int(signs[i] != signs[i-1]) for i in range(1, N)]

    df_gap = pd.DataFrame({
        'idx':   idxs[1:],
        't_n':   zeros[1:],
        'delta': deltas,
        'S':     S_vals,
        'flip':  flips
    })
    df_gap.to_csv(gap_csv, index=False)
    print(f"Saved gap+flip data to '{gap_csv}'")