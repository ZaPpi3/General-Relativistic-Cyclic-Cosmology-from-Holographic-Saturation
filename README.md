# General-Relativistic Cyclic Cosmology from Holographic Saturation

This repository hosts the complete academic package for the preprint paper **"General-Relativistic Cyclic Cosmology from Holographic Saturation"**. Included here are the compiled production PDF, the raw REVTeX 4-2 LaTeX source files, numerical simulation engines, and print-quality publication figures.

The framework replaces the big-bang singularity with a General-Relativity-consistent information-saturation event, mapping cosmic evolution across sequential cyclic phases via an Israel–Darmois junction boundary condition without relying on fundamental scalar fields.

---

## 📄 Document and Manifest Assets

For quick access to the paper's contents, structural layouts, or code baselines, navigate to the following core files:

*   **`main.pdf`**: The final, production-compiled document structured in the official, two-column Physical Review D (PRD) journal format.
*   **`main.tx`**: Contains the raw LaTeX source text, document preambles, and bibliography tags.
    *   `main.tex`: The unified, consolidated TeX source file ready for local compilation or direct upload to the arXiv production servers.
*   **`Code/`**: The complete Python implementation of the numerical models supporting the theoretical framework.
    *   `eigenmodes.py`: Builds the relational quantum entanglement kernel and maps the low-lying Laplacian spectrum.
    *   `entropy_attractor.py`: Tracks the discrete renormalization-group-like flow ($\mathcal{R}$) of effective background couplings across 30 successive bounces.

---

## 🌌 Theoretical Framework Abstract

Classical general relativity predicts geodesic incompleteness under broad conditions, as established by the Penrose–Hawking singularity theorems. This motivates the search for frameworks in which the big-bang singularity is replaced by a well-defined transition. 

We present a cyclic cosmological framework in which the big-bang singularity is replaced by an information-saturation event that triggers a transition from gravitational collapse to expansion. The mechanism is formulated entirely within general relativity at the macroscopic level: the transition is implemented as an Israel–Darmois junction across a spacelike hypersurface at a critical density, interpreted as a causal boundary rather than a topology change. 

The key dynamical ingredient is an effective confinement potential that reverses sign at holographic saturation, converting net attraction into repulsion. This sign reversal is treated phenomenologically as the macroscopic signature of a microphysical phase transition in the effective gravitational degrees of freedom. Entropy is inherited across cycles but renormalised relative to the enlarged phase space activated at the throat, yielding an effectively low-entropy initial state without entropy destruction. Anisotropies are suppressed by chaotic mixing in the collapsing phase and holographic coarse-graining at saturation, producing a homogeneous FLRW initial geometry. 

We demonstrate numerically that the confinement–rupture map defines a renormalisation-group-like flow on the space of effective couplings with a stable fixed point, driving the renormalised inhomogeneity ratio exponentially to zero and constraining the dimensionless constants of physics across cycles. Numerical evidence for emergent multi-dimensional geometry from entanglement structure is presented as a concrete realisation of the underlying substrate dynamics.

---

## 🚀 Usage and Local Compilation

### Compiling the TeX Source Locally
The raw manuscript is written using the official American Physical Society **REVTeX 4-2** document class. To compile the source into the final production PDF locally, ensure the image files are kept in the same directory as the source and run:

```bash
pdflatex main.tex
```

### Executing the Simulations
To run the underlying numerical pipelines and reproduce the paper's figures, ensure you have a Python environment with `numpy` and `matplotlib` installed, then execute:

```bash
python Code/entropy_attractor.py
python Code/eigenmodes.py
```
}
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for complete documentation. All code scripts and paper text assets are open-source and free to adapt with appropriate attribution.
