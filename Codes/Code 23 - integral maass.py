# GL4 Maass form 
import numpy as np
import mpmath as mp
from numpy.polynomial.legendre import leggauss

mp.mp.dps = 20
coeffs = {
    1:1+0j, 2:-0.784-0.248j, 3:-0.129+0.502j, 4:-0.0239+0.390j,
    5:1.23-0.0481j, 6:0.226-0.362j, 7:0.0795+1.18j, 8:-0.215+0.0922j,
    9:0.304-0.130j, 10:-0.983-0.270j, 11:0.119-0.526j, 12:-0.193-0.0626j,
    13:0.678+0.515j, 14:0.233-0.952j, 15:-0.135+0.628j, 16:-0.116-0.244j,
    17:0.0156+0.230j
}
def L_GL4(s):
    total = mp.mpc(0)
    for n,a in coeffs.items():
        total += a * n**(-s)
    return total
def Gamma_R(s):
    return mp.pi**(-s/2)*mp.gamma(s/2)
def Lambda_GL4(s):
    return Gamma_R(s-21.7j)*Gamma_R(s-2.12j)*Gamma_R(s+6.75j)*Gamma_R(s+17.1j)*L_GL4(s)
def Lambda_prime(s,h=1e-8):
    up   = Lambda_GL4(mp.mpc(mp.re(s), mp.im(s)+h))
    down = Lambda_GL4(mp.mpc(mp.re(s), mp.im(s)-h))
    return (up-down)/(2*h)*1j

t_vals = [
    1.15611850, 5.78697355, 8.85174703, 9.70332000, 9.95642352,
    11.63624613, 13.54487303, 13.67353387, 15.72141489, 17.52006026,
    18.16953036, 18.54436507, 20.62681146, 21.35057097, 22.70682375,
    24.31409552, 24.74458435
]

nodes, w = leggauss(10)
sigmas = 0.5*(nodes+1)
w *= 0.5

for t in t_vals:
    prods = []
    for σ, weight in zip(sigmas, w):
        d = Lambda_prime(mp.mpc(σ, t))
        prods.append(d.real * d.imag)
    I = float(np.dot(prods, w))
    print(f"t={float(t):.10f} → I={I:.3e}", "Positive" if I>0 else "Negative")