import numpy as np
from scipy.special import log_ndtr
from scipy.stats import norm
from scipy.optimize import brentq

V = 151936                      # Qwen
P0 = 1 - 1 / V                  # P_e(0)

z = np.linspace(-10, 10, 20001)  # bruit du vrai token
dz = z[1] - z[0]

def Pe(t):
    b = t / (1 - t)                                
    win = np.exp((V - 1) * log_ndtr(z + b))         
    return 1 - np.sum(norm.pdf(z) * win) * dz     

def tau(t):
    return 1 - Pe(t) / P0

u = brentq(lambda t: tau(t) - 0.5, 0.3, 0.999)
print(f"u = tau^-1(1/2) = {u:.4f}")

for N in (2, 3, 4):
    grid = [brentq(lambda t: tau(t) - n / N, 0.3, 0.999) for n in range(1, N)]
    print(f"N = {N} : t_n = {[round(g, 3) for g in grid]}")

lo = brentq(lambda t: tau(t) - 0.05, 0.3, 0.999)
hi = brentq(lambda t: tau(t) - 0.95, 0.3, 0.999)
print(f"tau de 5 % à 95 % : t de {lo:.3f} à {hi:.3f}")

print("\n   t    P_e/P_e(0)")
for t in [0.5, 0.7, 0.75, 0.8, u, 0.85, 0.9]:
    print(f"{t:6.3f}   {Pe(t) / P0:.4f}")

# monte carlo vs intégrale
rng = np.random.default_rng(0)
n_slots = 2000
print("\nMonte-Carlo (", n_slots, "slots ) vs intégrale")
for t in [0.75, 0.8, u, 0.85]:
    x0 = rng.standard_normal((n_slots, V), dtype=np.float32)
    xt = (1 - t) * x0
    xt[:, 0] += t                                
    err = np.mean(xt.argmax(axis=1) != 0)
    print(f"t = {t:.3f} :  MC {err / P0:.3f}   intégrale {Pe(t) / P0:.3f}")

ts = np.r_[np.arange(0, 0.7, 0.1), np.arange(0.7, 0.92, 0.005), 0.95, 1 - 1e-9]
print("\nTikZ :", " ".join(f"({t:.3f},{Pe(t) / P0:.4f})" for t in ts))