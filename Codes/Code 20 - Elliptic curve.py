import numpy as np
import matplotlib.pyplot as plt
import mpmath as mp
from math import gcd

# Increase precision for mpmath
mp.mp.dps = 20

# Disable LaTeX rendering
plt.rcParams['text.usetex'] = False

# --- Build the primitive Dirichlet character mod 37 ---
mod = 37
def order(a, m):
    for k in range(1, m):
        if pow(a, k, m) == 1:
            return k
    return None

# find a primitive root g mod 37
g = next(x for x in range(2, mod) if order(x, mod) == mod-1)
log_map = {pow(g, i, mod): i for i in range(mod-1)}

# choose k so that chi(3) = i -> k/(mod-1) = 1/4 -> k = 9
k = (mod - 1) // 4

def chi(n):
    r = n % mod
    if r == 0 or gcd(r, mod) != 1:
        return 0
    return mp.e**(2j * mp.pi * k * log_map[r] / (mod - 1))

# Truncated Dirichlet series for L(s)
def L_ec(s, N=1000):
    total = mp.mpc(0)
    for n in range(1, N+1):
        total += chi(n) * n**(-s)
    return total

# Completed L-function Lambda(s) = 37^(s/2)*Gamma_C(s)*L(s)
def Lambda_ec(s):
    gamma_c = 2 * (2*mp.pi)**(-s) * mp.gamma(s)
    return (37**(s/2)) * gamma_c * L_ec(s)

# Real-part grid
x = np.linspace(0, 1, 300)
# First two nontrivial zeros' imaginary parts
t_values = [9.032345851694468357832029856913, 9.819966817497618882768827794755]

# Plot side by side
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for idx, (ax, t) in enumerate(zip(axes, t_values)):
    s_vals = [xv + 1j*t for xv in x]
    lam_vals = [Lambda_ec(s) for s in s_vals]
    re_vals = [mp.re(val) for val in lam_vals]
    im_vals = [mp.im(val) for val in lam_vals]
    
    ax.plot(re_vals, im_vals, color='orange')
    ordinal = "fourth" if idx == 0 else "fifth"
    ax.set_title(f"Elliptic curve over $\\mathbb{{Q}}$\nOn the {ordinal} zero,\nim = {t:.4f}")
    ax.set_xlabel("Re Λ(s)")
    ax.set_ylabel("Im Λ(s)")
    ax.grid(True)

plt.tight_layout()
plt.show()