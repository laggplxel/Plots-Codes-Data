# Elliptic curve over Q (character mod 37)
import numpy as np
import mpmath as mp
from numpy.polynomial.legendre import leggauss
from math import gcd

mp.mp.dps = 20
mod = 37
def order(a,m):
    for k in range(1,m):
        if pow(a,k,m)==1: return k
    return None
g = next(x for x in range(2,mod) if order(x,mod)==mod-1)
log_map = {pow(g,i,mod):i for i in range(mod-1)}
k = (mod-1)//4
def chi(n):
    r = n % mod
    return mp.mpc(0) if r==0 or gcd(r,mod)!=1 else mp.e**(2j*mp.pi*k*log_map[r]/(mod-1))
def L_ec(s):
    total = mp.mpc(0)
    for n in range(1,1001):
        total += chi(n)*n**(-s)
    return total
def Lambda_ec(s):
    return (mod**(s/2)) * 2*(2*mp.pi)**(-s)*mp.gamma(s) * L_ec(s)
def Lambda_prime(s,h=1e-8):
    up   = Lambda_ec(mp.mpc(mp.re(s), mp.im(s)+h))
    down = Lambda_ec(mp.mpc(mp.re(s), mp.im(s)-h))
    return (up-down)/(2*h)*1j

t_vals = [
    3.50910294340479626549928321873, 5.44973416215471530225317007593, 7.59911177067371416247669368567,
    9.032345851694468357832029856913, 9.819966817497618882768827794755, 11.79612239530835321782087991524,
    13.04504337075354428108495487834, 14.26876170526783296623721964932,
    14.80595468440372872355464099506, 16.72216150673137017128071742797
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