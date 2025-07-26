import numpy as np
import matplotlib.pyplot as plt
from mpmath import zeta, zetazero, re, im, mp

mp.dps = 50

im_parts_zeta = [zetazero(1).imag, zetazero(2).imag]

re_vals = np.linspace(0.01, 0.99, 400)

s_values_1 = [complex(re_val, im_parts_zeta[0]) for re_val in re_vals]
s_values_2 = [complex(re_val, im_parts_zeta[1]) for re_val in re_vals]

zeta_vals_1 = [zeta(s) for s in s_values_1]
zeta_vals_2 = [zeta(s) for s in s_values_2]

zeta_re_1, zeta_im_1 = zip(*[(re(z), im(z)) for z in zeta_vals_1])
zeta_re_2, zeta_im_2 = zip(*[(re(z), im(z)) for z in zeta_vals_2])

im_part_0 = float(im_parts_zeta[0])
im_part_1 = float(im_parts_zeta[1])

# Plotting
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(zeta_re_1, zeta_im_1)
plt.xlabel("Re(ζ(s))")
plt.ylabel("Im(ζ(s))")
plt.title(f"Argand Plot of ζ(s) at Im(s) = {im_part_0:.6f}")
plt.grid(True)
plt.axis('equal')

plt.subplot(1, 2, 2)
plt.plot(zeta_re_2, zeta_im_2)
plt.xlabel("Re(ζ(s))")
plt.ylabel("Im(ζ(s))")
plt.title(f"Argand Plot of ζ(s) at Im(s) = {im_part_1:.6f}")
plt.grid(True)
plt.axis('equal')

plt.tight_layout()
plt.show()