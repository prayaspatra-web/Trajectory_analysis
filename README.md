# Molecular Dynamics Transport, Viscosity & Structural Analysis Suite

A high-throughput **C++17 (OpenMP)** and **Python** post-processing toolkit for calculating correlated ion transport, shear viscosity, radial distribution functions, and hydrogen-bond relaxation kinetics from large-scale classical molecular dynamics trajectories (GROMACS).

Developed to investigate concentrated chloroaluminate ionic liquid electrolytes ($[\text{C(NH}_2)_3]^+[\text{AlCl}_4]^-/[\text{Al}_2\text{Cl}_7]^-$) across Lewis-acidic compositions ($r = 1.6\text{–}2.0$).

---

## Repository Structure

```text
Trajectory_analysis/
├── README.md
├── Conductivity/
│   ├── eh_conductivity.cpp     # OpenMP Einstein-Helfand & species MSD trajectory kernel
│   └── slope.py                # Diffusive regime verification, EH/NE conductivity, Haven ratio & block convergence
├── Viscosity/
│   └── gk_viscosity.cpp        # OpenMP Green-Kubo stress-tensor autocorrelation viscosity calculator
├── Structural_analysis/
│   ├── rdf_site_200ns.cpp      # Multi-threaded site-site RDF & coordination number analyzer
│   └── hbond_lifetime.cpp      # Intermittent & continuous hydrogen-bond autocorrelation lifetime
└── examples/
    └── sample_plots/           # Validation plots (EH convergence, Haven ratio, RDF, H-bond decay)
