import numpy as np
import matplotlib.pyplot as plt

def simulate_entropy_cycles(
    cycles=30,
    Smax0=1.0,
    Sinh0=0.3,
    alpha=1.5,   # growth of information capacity
    beta=0.9     # decay of inhomogeneous entropy
):
    Smax = [Smax0]
    Sinh = [Sinh0]
    Stot = [Smax0 + Sinh0]

    for n in range(1, cycles):
        Smax.append(Smax[-1] * alpha)
        Sinh.append(Sinh[-1] * beta)
        Stot.append(Smax[-1] + Sinh[-1])

    Smax = np.array(Smax)
    Sinh = np.array(Sinh)
    Stot = np.array(Stot)
    S_tilde = Sinh / Smax

    return Smax, Sinh, Stot, S_tilde

# ---- Run simulation ----
Smax, Sinh, Stot, S_tilde = simulate_entropy_cycles()

# ---- Generate the combined Figure 2 layout ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

# Panel 1: Absolute Entropy Growth and Capacity Limits
ax1.plot(Stot, 'b-', linewidth=2, label=r"Total entropy $S_{tot}$")
ax1.plot(Smax, 'r--', linewidth=2, label=r"Max capacity $S_{max}$")
ax1.set_xlabel("Cycle number")
ax1.set_ylabel("Entropy Scale")
ax1.set_title("Absolute Entropy Profile")
ax1.legend()
ax1.grid(True, linestyle=':')

# Panel 2: Renormalised Inhomogeneity Decay (The Attractor Flow)
ax2.plot(S_tilde, 'go-', markersize=5, linewidth=1.5, label=r"$\tilde{S} = S_{inh}/S_{max}$")
ax2.set_xlabel("Cycle number")
ax2.set_ylabel("Renormalised Inhomogeneity")
ax2.set_title(r"Decay of $\tilde{S}(n)$ Across Cycles")
ax2.legend()
ax2.grid(True, linestyle=':')

plt.suptitle("The Cosmic Renormalisation Attractor Mechanics", fontsize=12, fontweight='bold')
plt.tight_layout()

# Save as a single high-quality file for your LaTeX compiler
plt.savefig("entropy_attractor.png", dpi=300, bbox_inches='tight')
print("Successfully generated and saved: entropy_attractor.png")

plt.show()
