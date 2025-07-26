import numpy as np
import matplotlib.pyplot as plt
import mpmath as mp

# Increase precision for mpmath
mp.mp.dps = 20

# Disable LaTeX rendering
plt.rcParams['text.usetex'] = False

# Known Dirichlet series coefficients for the classical modular form (level 300)
coeffs = {
    1: 1,
    3: 1j,
    7: 1j,
    9: -1,
    13: -1j,
    19: 1,
    21: -1,
    27: -1j,
    31: -1,
    37: -2j,
    39: 1,
    43: -1j,
    57: 1j,
    61: -1,
    63: -1j,
    67: 1j
}

# Truncated Dirichlet series L(s)
def L_modular(s, N=1000):
    return mp.nsum(lambda k: coeffs.get(k, 0) * k**(-s), [1, max(coeffs.keys())])

# Completed L-function: Lambda(s) = 300^(s/2) * Gamma_C(s) * L(s)
def Lambda_modular(s):
    gamma_c = 2 * (2*mp.pi)**(-s) * mp.gamma(s)
    return (300**(s/2)) * gamma_c * L_modular(s)

# Grid for real part sigma
x = np.linspace(0, 1, 300)
# First two nontrivial zeros' imaginary parts
t_values = [4.69009675734075972868552749914, 5.29423475529553211433386184685]

# Plot side by side
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for idx, (ax, t) in enumerate(zip(axes, t_values)):
    S = [xv + 1j*t for xv in x]
    vals = [Lambda_modular(s) for s in S]
    re_vals = [mp.re(v) for v in vals]
    im_vals = [mp.im(v) for v in vals]

    ax.plot(re_vals, im_vals, color='blue')
    ordinal = "fifth" if idx == 0 else "sixth"
    ax.set_title(f"Classical modular form\non the {ordinal} zero,\nim = {t:.4f}")
    ax.set_xlabel("Re Λ(s)")
    ax.set_ylabel("Im Λ(s)")
    ax.grid(True)

plt.tight_layout()
plt.show()