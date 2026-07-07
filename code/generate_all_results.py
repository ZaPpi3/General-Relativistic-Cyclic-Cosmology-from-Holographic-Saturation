"""
generate_all_results.py
=======================
Single, self-contained pipeline that computes EVERY figure and every quoted
number in the revised manuscript. No plotted quantity is asserted: all curves
are outputs of the computations below.

Outputs (into ./figures/):
  eigenmodes.png            - Fig. 1: emergent eigenmodes, 8x8 torus kernel, alpha=6
  rho0_convergence.png      - Fig. 2: continuum convergence of rho_0 across lattice sizes
  attractor_genericity.png  - Fig. 3: inhomogeneity decay under the damping ansatz
                                       (multiple coefficients -> genericity of attractor)
  weff_profile.png          - Fig. 4: computed w_eff(a) for the Eq. (27) confinement profile
  tensor_spectrum.png       - Fig. 5: Bogoliubov spectra |beta_k|^2 for delta = 0.4/0.5/0.6

Printed to console:
  - rho_0 table (L = 4..16) and power-law fit exponent
  - Weyl vs fluid dimension table (6x6, fit window n = 1..12)
  - degenerate-tier eigenvalues for Fig. 1
  - tensor tail slopes, Wronskian checks, per-mode |beta_k|^2

Usage:
  python3 generate_all_results.py                # everything (~2-4 min; tensor part dominates)
  python3 generate_all_results.py --skip-tensor  # fast parts only (~20 s)

Dependencies: numpy, scipy, matplotlib.
"""
import os
import sys
import numpy as np
import scipy.linalg as la
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "figures"
os.makedirs(OUT, exist_ok=True)
SKIP_TENSOR = "--skip-tensor" in sys.argv

# ============================================================
# A. Relational kernel on a toroidal lattice
# ============================================================

def torus_distances(L):
    x, y = np.meshgrid(np.arange(L), np.arange(L))
    pos = np.vstack([x.ravel(), y.ravel()]).T
    N = L * L
    D = np.zeros((N, N))
    for i in range(N):
        dx = np.minimum(np.abs(pos[i, 0] - pos[:, 0]), L - np.abs(pos[i, 0] - pos[:, 0]))
        dy = np.minimum(np.abs(pos[i, 1] - pos[:, 1]), L - np.abs(pos[i, 1] - pos[:, 1]))
        D[i] = np.sqrt(dx**2 + dy**2)
    np.fill_diagonal(D, 0.0)
    return D

def mi_laplacian(D, alpha):
    I = np.zeros_like(D)
    mask = D > 0
    I[mask] = 1.0 / (D[mask] ** alpha)
    return np.diag(I.sum(axis=1)) - I

# ============================================================
# B. Fig. 1 - emergent eigenmodes (8x8, alpha = 6)
# ============================================================

def figure_eigenmodes(L=8, alpha=6.0):
    Lap = mi_laplacian(torus_distances(L), alpha)
    vals, vecs = la.eigh(Lap)
    print(f"[Fig 1] 8x8 kernel, alpha={alpha}: lowest 6 eigenvalues = {np.round(vals[:6], 4)}")
    deg = np.sum(np.isclose(vals, vals[1], atol=1e-8))
    print(f"[Fig 1] first excited tier degeneracy = {deg} at lambda = {vals[1]:.4f}")

    fig, axes = plt.subplots(2, 3, figsize=(9.5, 6.2), dpi=200)
    for m, ax in enumerate(axes.ravel()):
        field = vecs[:, m].reshape(L, L)
        im = ax.imshow(field, cmap="RdBu_r")
        ax.set_title(f"Mode {m}, $\\lambda$={vals[m]:.3f}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Emergent eigenmodes of the relational kernel "
                 f"($L={L}$ torus, $\\alpha={alpha:g}$)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "eigenmodes.png"), bbox_inches="tight")
    plt.close()
    return vals[1], deg

# ============================================================
# C. Fig. 2 - rho_0 continuum convergence
# ============================================================

def figure_rho0_convergence(Ls=(4, 6, 8, 10, 12, 14, 16), alpha=6.0):
    rows = []
    for L in Ls:
        vals = np.sort(la.eigvalsh(mi_laplacian(torus_distances(L), alpha)))
        active = vals[vals > 1e-8]
        rho0 = active.mean()                      # spectral centroid, Eq. (rho0)
        rows.append((L, L * L, active.min(), active.max(), len(active), rho0))

    print("\n[Fig 2] rho_0 convergence (alpha = 6):")
    print(f"{'L':>4} {'N':>5} {'lam_min':>9} {'lam_max':>9} {'modes':>6} {'rho_0':>8}")
    for r in rows:
        print(f"{r[0]:>4} {r[1]:>5} {r[2]:>9.4f} {r[3]:>9.4f} {r[4]:>6} {r[5]:>8.4f}")

    Ns = np.array([r[1] for r in rows], float)
    rho0s = np.array([r[5] for r in rows])
    p = np.polyfit(np.log(Ns), np.log(rho0s), 1)[0]
    print(f"[Fig 2] power-law fit rho_0 ~ N^p: p = {p:.4f}  "
          f"(|p| << 1 => N-independent continuum value)")
    print(f"[Fig 2] continuum estimate rho_0 = {rho0s[-1]:.3f} "
          f"(largest lattice); spread over L=4..16 = {rho0s.max()-rho0s.min():.3f}")

    plt.figure(figsize=(6.4, 4.4), dpi=200)
    plt.plot(1.0 / Ns, rho0s, "o-", color="darkblue")
    for (Lv, Nv, *_ , r0) in rows:
        plt.annotate(f"L={Lv}", (1.0 / Nv, r0), textcoords="offset points",
                     xytext=(6, 4), fontsize=8)
    plt.xlabel("$1/N$   (N = number of sites)")
    plt.ylabel(r"$\rho_0$  (lattice units)")
    plt.title(r"Continuum convergence of $\rho_0$ at $\alpha=6$"
              f"   (fit exponent $p={p:.3f}$)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "rho0_convergence.png"), bbox_inches="tight")
    plt.close()
    return rho0s, p

# ============================================================
# D. Table I - Weyl vs fluid dimension (6x6, fit window n = 1..12)
# ============================================================

def table_dimensions(L=6, alphas=(1.5, 2.5, 3.5, 4.5, 6.0, 7.0), nfit=12):
    D = torus_distances(L)
    print(f"\n[Table I] Weyl vs fluid dimension ({L}x{L} torus, fit window n=1..{nfit}):")
    print(f"{'alpha':>6} {'w_eff':>8} {'d_Weyl':>8} {'d_fluid':>8}")
    out = []
    for alpha in alphas:
        vals = np.sort(la.eigvalsh(mi_laplacian(D, alpha)))
        active = vals[vals > 1e-10][:nfit]
        n = np.arange(1, len(active) + 1)
        slope = np.polyfit(np.log(n), np.log(active), 1)[0]
        d_weyl = 2.0 / slope
        w = alpha / 3.0 - 1.0
        out.append((alpha, w, d_weyl, alpha))
        print(f"{alpha:>6.1f} {w:>+8.3f} {d_weyl:>8.2f} {alpha:>8.1f}")
    return out

# ============================================================
# E. Fig. 3 - attractor genericity under the damping ANSATZ
# ============================================================

def figure_attractor_genericity(L=8, beta=0.1, cycles=30, growth=2.5,
                                coeffs=(0.05, 0.22, 1.0)):
    """The damping law D_n = exp(-c N / rho_throat) is an ANSATZ. This figure
    demonstrates that convergence of the renormalised inhomogeneity ratio to
    zero is GENERIC across damping coefficients c, because any 0 < D_n < 1
    combined with geometrically growing capacity S_max forces S~ -> 0. The
    figure therefore illustrates the ansatz's qualitative behaviour; it is
    not a derivation of a specific rate."""
    N = L * L
    D = torus_distances(L)

    def rho_throat(cycle):
        alpha_n = 0.005 + 0.3 / (1.0 + 0.5 * cycle)     # near-critical schedule
        vals = np.sort(la.eigvalsh(mi_laplacian(D, alpha_n)))
        act = vals[vals > 1e-10]
        wgt = np.exp(-beta * act)
        return float(np.sum(act * wgt) / np.sum(wgt))

    rhos = [rho_throat(c) for c in range(cycles)]
    plt.figure(figsize=(6.4, 4.4), dpi=200)
    print("\n[Fig 3] attractor genericity (ansatz illustration):")
    for c in coeffs:
        S = 0.30
        traj = []
        for n in range(cycles):
            Dn = np.exp(-c * N / rhos[n])
            S = S * Dn / growth                      # S~ <- S~ * D_n / g
            traj.append(S)
        plt.semilogy(np.arange(1, cycles + 1), traj, "o-",
                     label=f"$c={c}$", markersize=3.5)
        print(f"  c={c}: S~(final) = {traj[-1]:.3e}")
    plt.xlabel("cycle $n$")
    plt.ylabel(r"$\tilde{S}(n)$")
    plt.title("Inhomogeneity decay under the exponential damping ansatz\n"
              "(convergence is generic across damping coefficients)")
    plt.legend(); plt.grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "attractor_genericity.png"), bbox_inches="tight")
    plt.close()

# ============================================================
# F. Background from the Eq. (27) profile + radiation
# ============================================================

def chi_of_a(a, a_c=1.0, delta_a=0.5):
    return 0.5 * (1.0 + np.tanh((a_c - a) / delta_a))

def build_background(delta_a, rho_r0=1.0, N_lo=-1.8, N_hi=1.8, n_grid=6000):
    N = np.linspace(N_lo, N_hi, n_grid)
    a = np.exp(N)
    chi = chi_of_a(a, 1.0, delta_a)
    peak_shape = (1.0 / 9.0) * (8.0 / 9.0) ** 8       # analytic max of chi(1-chi)^8
    rho0 = 3.0 / peak_shape                            # spike peak density = 3
    rho_conf = rho0 * chi * (1.0 - chi) ** 8
    rho_rad = rho_r0 * a ** (-4)
    rho_tot = rho_conf + rho_rad

    w_tot = -1.0 - np.gradient(np.log(rho_tot), N) / 3.0    # from continuity eq.
    H = np.sqrt(rho_tot / 3.0)
    calH = a * H
    eps = 1.5 * (1.0 + w_tot)
    U = calH ** 2 * (2.0 - eps)                              # a''/a, conformal time

    dtau = 0.5 * (1.0 / calH[1:] + 1.0 / calH[:-1]) * np.diff(N)
    tau = np.concatenate(([0.0], np.cumsum(dtau)))
    return dict(N=N, a=a, tau=tau, calH=calH, U=U, w_tot=w_tot)

def figure_weff(delta_a=0.5):
    bg = build_background(delta_a)
    print(f"\n[Fig 4] Eq.(27) profile, delta={delta_a}: "
          f"computed peak w_eff = {bg['w_tot'].max():.3f}  (adiabatic c_s^2 = w would exceed 1)")
    plt.figure(figsize=(6.4, 4.4), dpi=200)
    plt.plot(bg["a"], bg["w_tot"], "darkblue", lw=2)
    plt.axhline(1.0, color="crimson", ls="--", lw=1.4, label=r"$w=1$ (causal bound, $c_s^2=w$)")
    plt.axhline(bg["w_tot"].max(), color="gray", ls=":", lw=1.2,
                label=f"computed peak $w={bg['w_tot'].max():.2f}$")
    plt.xscale("log")
    plt.xlabel("scale factor $a$  ($a_c=1$)")
    plt.ylabel(r"$w_{\rm eff}(a)$")
    plt.title(r"Equation of state from the confinement profile ($\delta=%.1f$)" % delta_a)
    plt.legend(fontsize=9); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "weff_profile.png"), bbox_inches="tight")
    plt.close()
    return bg["w_tot"].max()

# ============================================================
# G. Tensor Mukhanov-Sasaki integration -> Bogoliubov spectra
# ============================================================

def bogoliubov_at_k(k, tau, U, rtol=1e-7, atol=1e-10):
    U_spline = CubicSpline(tau, U)
    def rhs(t, y):
        Uv = U_spline(t)
        return [y[2], y[3], -(k * k - Uv) * y[0], -(k * k - Uv) * y[1]]
    norm = 1.0 / np.sqrt(2 * k)
    y0 = [norm, 0.0, 0.0, -k * norm]                # adiabatic vacuum e^{-ik tau}
    step = min((tau[-1] - tau[0]) / 300, 0.15 / k)
    sol = solve_ivp(rhs, [tau[0], tau[-1]], y0, method="RK45",
                    max_step=step, rtol=rtol, atol=atol)
    vr, vi, vpr, vpi = sol.y[:, -1]
    v, vp = vr + 1j * vi, vpr + 1j * vpi
    t = sol.t[-1]
    u = np.exp(-1j * k * t) / np.sqrt(2 * k); up = -1j * k * u
    W = u * np.conj(up) - up * np.conj(u)
    B = (vp * u - v * up) / W
    A = (v * np.conj(up) - vp * np.conj(u)) / W
    return np.abs(B) ** 2, np.abs(A) ** 2 - np.abs(B) ** 2   # (beta^2, Wronskian)

def figure_tensor_spectrum(deltas=(0.4, 0.5, 0.6),
                           ks=np.array([80., 120., 160., 240., 320.])):
    print("\n[Fig 5] tensor Bogoliubov spectra (vacuum ICs, Wronskian-checked):")
    colors = {0.4: "#1f77b4", 0.5: "#d62728", 0.6: "#2ca02c"}
    plt.figure(figsize=(6.6, 4.6), dpi=200)
    ref = None
    slopes = {}
    for d in deltas:
        bg = build_background(d)
        b2, wrons = [], []
        for k in ks:
            bb, ww = bogoliubov_at_k(k, bg["tau"], bg["U"])
            b2.append(bb); wrons.append(ww)
            print(f"  delta={d}: k={k:>6.0f}  |beta|^2={bb:.4e}  "
                  f"|alpha|^2-|beta|^2-1={ww-1:+.2e}")
        b2 = np.array(b2)
        slope = np.polyfit(np.log(ks), np.log(b2), 1)[0]
        res = np.log(b2) - np.polyval(np.polyfit(np.log(ks), np.log(b2), 1), np.log(ks))
        slopes[d] = slope
        print(f"  delta={d}: tail slope = {slope:.4f}  "
              f"(k^3-weighted n_T = {slope + 3:.3f}), max|residual| = {np.abs(res).max():.4f}")
        plt.loglog(ks, b2, "o-", color=colors.get(d), label=fr"$\delta={d}$: slope$={slope:.2f}$")
        if ref is None:
            ref = b2[0]
    bg05 = build_background(0.5)
    b2_05 = np.array([bogoliubov_at_k(k, bg05["tau"], bg05["U"])[0] for k in ks[:1]])
    plt.loglog(ks, b2_05[0] * (ks / ks[0]) ** -4, "k--", lw=1, alpha=0.6,
               label=r"$k^{-4}$ reference")
    plt.xlabel(r"$k$  (units of spike-era $aH$)")
    plt.ylabel(r"$|\beta_k|^2$")
    plt.title("Computed tensor particle-production spectra")
    plt.legend(fontsize=9); plt.grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "tensor_spectrum.png"), bbox_inches="tight")
    plt.close()
    return slopes

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print(" generate_all_results.py - every figure/number in the manuscript")
    print("=" * 70)

    figure_eigenmodes()
    figure_rho0_convergence()
    table_dimensions()
    figure_attractor_genericity()
    figure_weff()

    if SKIP_TENSOR:
        print("\n[skipped] tensor spectra (--skip-tensor). Run without the flag "
              "for Fig. 5 (~2-4 min).")
    else:
        figure_tensor_spectrum()

    print("\nAll outputs written to ./figures/")
