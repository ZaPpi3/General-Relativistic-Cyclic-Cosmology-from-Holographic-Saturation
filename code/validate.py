import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.ndimage import gaussian_filter1d

# ============================================================
# GLOBAL SETTINGS
# ============================================================

w = 1/3                 # equation of state
rho_sat = 1.0           # saturation density (sets bounce scale)
t_max = 5.0
t_eval = np.linspace(-t_max, t_max, 2001)

k_mode = 5.0            # perturbation mode

# ============================================================
# MODIFIED FRIEDMANN MODEL (FROM PAPER)
#   H^2 = rho (1 - rho / rho_sat)
#   drho/dt = -3 H (1+w) rho
#   dH/dt   = -3/2 (1+w) rho (1 - 2 rho / rho_sat)
# ============================================================

def background_rhs(t, y):
    a, rho, H = y
    # modified Friedmann is used to *check* consistency, but we evolve H via dH/dt
    drho = -3.0 * H * (1.0 + w) * rho
    dH   = -1.5 * (1.0 + w) * rho * (1.0 - 2.0 * rho / rho_sat)
    da   = a * H
    return [da, drho, dH]

def evolve_background():
    # bounce at t=0: rho = rho_sat, H = 0, a = 1
    a0   = 1.0
    rho0 = rho_sat
    H0   = 0.0

    # integrate backward
    sol_back = solve_ivp(
        background_rhs,
        (0.0, -t_max),
        [a0, rho0, H0],
        t_eval=np.linspace(0.0, -t_max, 1001),
        max_step=0.01
    )

    # integrate forward
    sol_fwd = solve_ivp(
        background_rhs,
        (0.0, t_max),
        [a0, rho0, H0],
        t_eval=np.linspace(0.0, t_max, 1001),
        max_step=0.01
    )

    # stitch
    t_back  = sol_back.t[::-1]
    a_back  = sol_back.y[0][::-1]
    rho_back= sol_back.y[1][::-1]
    H_back  = sol_back.y[2][::-1]

    t_fwd   = sol_fwd.t[1:]
    a_fwd   = sol_fwd.y[0][1:]
    rho_fwd = sol_fwd.y[1][1:]
    H_fwd   = sol_fwd.y[2][1:]

    t_full   = np.concatenate([t_back, t_fwd])
    a_full   = np.concatenate([a_back, a_fwd])
    rho_full = np.concatenate([rho_back, rho_fwd])
    H_full   = np.concatenate([H_back, H_fwd])

    # interpolate onto uniform t_eval
    a_interp   = np.interp(t_eval, t_full, a_full)
    rho_interp = np.interp(t_eval, t_full, rho_full)
    H_interp   = np.interp(t_eval, t_full, H_full)

    return t_eval, a_interp, rho_interp, H_interp

# ============================================================
# EFFECTIVE FLUID FOR PERTURBATIONS
#   rho_eff = 3 H^2
#   p_eff   = -2 dH/dt - rho_eff
# ============================================================

def compute_effective_fluid(t, H):
    # numerical dH/dt
    dHdt = np.gradient(H, t)
    rho_eff = 3.0 * H**2
    p_eff   = -2.0 * dHdt - rho_eff
    return rho_eff, p_eff, dHdt

# ============================================================
# PERTURBATION PIPELINE: z, z''/z, v_k
# ============================================================

def compute_z_and_zpp_over_z(t, a, rho_eff, p_eff, H):
    cs2 = w
    Hsafe = np.where(H == 0, 1e-8, H)
    z = a * np.sqrt(np.abs((rho_eff + p_eff) / (cs2 * Hsafe**2)))
    z_s = gaussian_filter1d(z, sigma=5)

    zp = np.gradient(z_s, t)
    zpp = np.gradient(zp, t)
    zpp_over_z = np.nan_to_num(zpp / z_s, nan=0.0, posinf=0.0, neginf=0.0)
    zpp_over_z = np.clip(zpp_over_z, -1e6, 1e6)

    return z_s, zpp_over_z

def evolve_vk(t, zpp_over_z):
    def ms_rhs(ti, y):
        v, vp = y
        zppz = np.interp(ti, t, zpp_over_z)
        dv = vp
        dvp = -(w * k_mode**2 - zppz) * v
        return [dv, dvp]

    sol = solve_ivp(
        ms_rhs,
        (t[0], t[-1]),
        [1e-3, 0.0],
        t_eval=t,
        method="Radau",
        rtol=1e-8,
        atol=1e-10
    )
    return sol.y[0]

# ============================================================
# RUN BACKGROUND + PERTURBATIONS
# ============================================================

t, a, rho, H = evolve_background()
rho_eff, p_eff, dHdt = compute_effective_fluid(t, H)
z, zppz = compute_z_and_zpp_over_z(t, a, rho_eff, p_eff, H)
v_k = evolve_vk(t, zppz)

# ============================================================
# PLOTS
# ============================================================

# H(t)
plt.figure(figsize=(8,5))
plt.plot(t, H, label="H(t)")
plt.axhline(0, color="black", linewidth=0.8)
plt.xlabel("t")
plt.ylabel("H(t)")
plt.title("Bounce background: H(t) with modified Friedmann")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("H_bounce.png", dpi=200)
plt.close()

# z''/z
plt.figure(figsize=(8,5))
plt.plot(t, zppz, label="z''/z")
plt.xlabel("t")
plt.ylabel("z''/z")
plt.title("Perturbation potential z''/z")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("zppz_bounce.png", dpi=200)
plt.close()

# v_k(t)
plt.figure(figsize=(8,5))
plt.plot(t, v_k, label=f"v_k(t), k={k_mode}")
plt.xlabel("t")
plt.ylabel("v_k(t)")
plt.title("Mukhanov–Sasaki mode across the bounce")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("vk_bounce.png", dpi=200)
plt.close()

print("Done. Generated: H_bounce.png, zppz_bounce.png, vk_bounce.png")
