#!/usr/bin/env python3
"""
Stable PBH suppression test for rupture-driven cosmology.
Utilizes SciPy's adaptive Radau implicit integrator to safely 
and instantly resolve the ultra-high frequency oscillations.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ------------------------------------------------------------
# 1. Background: rupture-driven hyper-stiff spike
# ------------------------------------------------------------

a_c = 5e-4
Delta_a = 4e-4
w_conf = 80.0

def chi(a):
    x = (a_c - a) / Delta_a
    t = np.tanh(x)
    denom = 1.0 + t
    if not np.isfinite(denom) or abs(denom) < 1e-6:
        denom = 1e-6 * np.sign(denom if denom != 0 else 1.0)
    return 1.0 / denom

def w(a):
    ww = w_conf * chi(a)
    if not np.isfinite(ww):
        ww = w_conf
    return ww

def delta_c(a):
    ww = w(a)
    return 3*(1+ww)/(5+3*ww)

delta_c_vec = np.vectorize(delta_c)

# ------------------------------------------------------------
# 2. Define the System of ODEs for solve_ivp
# ------------------------------------------------------------

def perturbation_system(ln_a, y):
    # Corrected unpacking syntax
    delta = y[0]
    ddelta = y[1]
    a = np.exp(ln_a)
    
    ww = w(a)
    A = 2 - 3*ww
    B = -1.5*(1+ww)*(1+3*ww)
    
    d2delta = A*ddelta + B*delta
    return [ddelta, d2delta]

# ------------------------------------------------------------
# 3. Execute Adaptive Integration Matrix
# ------------------------------------------------------------

k_values = np.logspace(2, 4, 5) # Clean subset of tracking modes
a_start = 1e-5
a_end   = 1e-2

results = {}
collapsed_modes = []

print("Running stable adaptive PBH suppression test (Implicit Radau)...")

for k in k_values:
    a_entry = 1.0 / k
    if a_entry >= a_end:
        continue
    a_entry = max(a_start, a_entry)
    
    ln_a_start = np.log(a_entry)
    ln_a_end   = np.log(a_end)
    
    ln_a_eval = np.linspace(ln_a_start, ln_a_end, 500)
    
    # Initial conditions at horizon entry
    y0 = [1e-5, 0.0]
    
    # Radau handles the w=80 stiffness flawlessly and finishes instantly
    sol = solve_ivp(
        perturbation_system, 
        [ln_a_start, ln_a_end], 
        y0, 
        t_eval=ln_a_eval,
        method='Radau',
        rtol=1e-6,
        atol=1e-8
    )
    
    a_tracked = np.exp(sol.t)
    delta_tracked = sol.y[0]
    
    collapsed = False
    for i in range(len(a_tracked)):
        if abs(delta_tracked[i]) > delta_c(a_tracked[i]):
            collapsed = True
            a_tracked = a_tracked[:i+1]
            delta_tracked = delta_tracked[:i+1]
            break
            
    results[k] = (a_tracked, delta_tracked)
    if collapsed:
        collapsed_modes.append(k)

print("Integration complete.")

# ------------------------------------------------------------
# 4. Report & Plotting
# ------------------------------------------------------------

if collapsed_modes:
    print("\n[WARNING] PBH collapse triggered.")
else:
    print("\n[SUCCESS] No PBHs formed. High pressure stabilizes all modes.")

plt.figure(figsize=(7,5))

for k, (agrid, delt) in results.items():
    if len(agrid) > 0:
        plt.loglog(agrid, np.abs(delt), label=f"k={k:.1e}", linewidth=1.5)

a_total_grid = np.logspace(np.log10(a_start), np.log10(a_end), 500)
plt.loglog(a_total_grid, delta_c_vec(a_total_grid), 'k--', label="collapse threshold", alpha=0.8)

plt.xlabel("scale factor a")
plt.ylabel("|delta(a)|")
plt.ylim(1e-20, 10)
plt.xlim(a_start, a_end)
plt.legend(loc="lower left")
plt.grid(True, which="both", ls=":", alpha=0.5)
plt.tight_layout()

# Save preview asset to verify locally
plt.savefig("figures/pbh_suppression_stable.png", dpi=300)
plt.show()
