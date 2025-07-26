import numpy as np
import matplotlib.pyplot as plt
import mpmath as mp

# Increase precision for mpmath
mp.mp.dps = 20

# Disable LaTeX rendering
plt.rcParams['text.usetex'] = False

# --- GL4 Maass form Dirichlet coefficients from LMFDB ---
coeffs = {
    1: 1+0j,
    2: complex(-0.784, -0.248),
    3: complex(-0.129,  0.502),
    4: complex(-0.0239, 0.390),
    5: complex(1.23,   -0.0481),
    6: complex(0.226,  -0.362),
    7: complex(0.0795,  1.18),
    8: complex(-0.215,  0.0922),
    9: complex(0.304,  -0.130),
    10: complex(-0.983, -0.270),
    11: complex(0.119,  -0.526),
    12: complex(-0.193, -0.0626),
    13: complex(0.678,   0.515),
    14: complex(0.233,  -0.952),
    15: complex(-0.135,  0.628),
    16: complex(-0.116, -0.244),
    17: complex(0.0156,  0.230)
}

def L_GL4(s):
    """Truncated Dirichlet series L(s)."""
    total = mp.mpc(0)
    for n, a in coeffs.items():
        total += a * n**(-s)
    return total

def gamma_R(s):
    """Real gamma factor Gamma_R(s) = pi^{-s/2} * Gamma(s/2)."""
    return mp.pi**(-s/2) * mp.gamma(s/2)

def Lambda_GL4(s):
    """Completed GL4 Maass form L-function."""
    return (
        gamma_R(s - 21.7j) *
        gamma_R(s - 2.12j) *
        gamma_R(s + 6.75j) *
        gamma_R(s + 17.1j) *
        L_GL4(s)
    )

# Grid for real part sigma in [0,1]
x = np.linspace(0, 1, 300)

# First two positive zeros (imaginary parts)
t_values = [
    1.15611850,  # first zero
    5.78697355   # second zero
]

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for idx, (ax, t) in enumerate(zip(axes, t_values)):
    s_vals = [mp.mpc(xv, t) for xv in x]
    lam_vals = [Lambda_GL4(s) for s in s_vals]
    re_vals = [mp.re(v) for v in lam_vals]
    im_vals = [mp.im(v) for v in lam_vals]

    ax.plot(re_vals, im_vals, lw=2)
    ordinal = "first" if idx == 0 else "second"
    ax.set_title(f"GL$_4$ Maass form\non the {ordinal} zero\nIm = {t:.5f}")
    ax.set_xlabel("Re Λ(s)")
    ax.set_ylabel("Im Λ(s)")
    ax.grid(True)

plt.tight_layout()
plt.show()