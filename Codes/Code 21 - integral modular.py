# Classical modular form (level 300)
import numpy as np
import mpmath as mp
from numpy.polynomial.legendre import leggauss

mp.mp.dps = 20
coeffs = {
    1:1, 3:1j, 7:1j, 9:-1, 13:-1j, 19:1, 21:-1,
    27:-1j, 31:-1, 37:-2j, 39:1, 43:-1j, 57:1j,
    61:-1, 63:-1j, 67:1j
}
def L_modular(s):
    return mp.nsum(lambda k: coeffs.get(k,0)*k**(-s), [1, max(coeffs)])
def Lambda_mod(s):
    return (300**(s/2)) * 2*(2*mp.pi)**(-s)*mp.gamma(s) * L_modular(s)
def Lambda_prime(s,h=1e-8):
    up   = Lambda_mod(mp.mpc(mp.re(s), mp.im(s)+h))
    down = Lambda_mod(mp.mpc(mp.re(s), mp.im(s)-h))
    return (up-down)/(2*h)*1j

t_vals = [
    1.51565973822216203176734940741, 2.63549969015813420091347628242, 3.80587507684635087489723777263,
    4.69009675734075972868552749914, 5.29423475529553211433386184685, 5.76908640453884171589561568387,
    6.10915812908886137955795875640, 7.05929133007431612511485815057, 7.44423218924716347098249162256,
    7.72608082543691412317778600171, 8.611554033917077913091286940942, 8.873022695828254400712013405300,
    9.302592474543976489563525055459, 9.929993887286861233702550480290, 10.24535189912948783281184977550,
    11.11616451565845491899036330357, 11.12320549034651232018049959416, 11.78061059495766263985688721443,
    12.01372016912809342749285172174
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