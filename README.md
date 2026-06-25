# orbcloud

`orbcloud` is a Python package designed to transform simulated exoplanet parameter posteriors (such as MCMC chains) into physical 3D orbital probability density clouds.

By plotting thousands of low-opacity orbital paths, the overlapping threads naturally highlight the high-probability regions of 3D orbital space, creating a beautiful and physically accurate visualization.

> [!TIP]
> In addition to visualization, `orbcloud` can be useful to rule out possible dynamical instability in the system. Visually mapping the orbital probability clouds allows researchers to quickly identify overlapping orbital regions. This helps save significant time and computational resources by avoiding expensive N-body simulations if a visual inspection already reveals that the system is most likely going to be unstable.

---

## Features

- **Vectorized Kepler Solver**: Vectorized Newton-Raphson solver to compute eccentric and true anomalies over custom phase grids.
- **Physical Star Customization**: Built-in star properties database (e.g. Vega, Barnard's Star) that automatically adjusts the size and glowing spectral color of the central star.
- **Top (2D) & Lateral (3D) Views**: Easily render orbits in 2D, 3D, or side-by-side.
- **Transparent Alpha-Clouds**: Line-by-line low opacity (`alpha=0.02`) plots that naturally map the probability clouds.
- **Robust Parameter Validation**: Informative alert messages and correction suggestions to prevent unphysical values (e.g. eccentricity $\ge 1.0$) or solver failures.

---

## Installation

```bash
pip install orbcloud
```

Dependencies: `numpy`, `matplotlib`

---

## Quickstart

Here is how to set up and render a multi-planet system:

```python
import matplotlib.pyplot as plt
from orbcloud import PlanetConfig, SystemEnsemble

# 1. Initialize the system around a customized star (e.g. M-type dwarf)
system = SystemEnsemble(star_name="Custom Star", star_mass=1.6, star_type="M")

# 2. Configure planets with mean parameters and uncertainties
planet_b = PlanetConfig(
    name="Planet b",
    P_mean=90.0, P_std=8.0,
    omega_mean_deg=60.0, omega_std_deg=40.0,
    e_mean=0.15, e_std=0.09
)

planet_c = PlanetConfig(
    name="Planet c",
    P_mean=260.0, P_std=14.0,
    omega_mean_deg=210.0, omega_std_deg=45.0,
    e_mean=0.25, e_std=0.10
)

# 3. Add planets to simulate posterior distributions and pre-compute 3D coordinates
system.add_planet(planet_b, num_samples=1000)
system.add_planet(planet_c, num_samples=1000)

# 4. Plot 2D and 3D clouds side-by-side
system.plot_system(show_reference_plane=True)

# 5. Save the result
plt.savefig("system_plot.png", dpi=150, facecolor="white", bbox_inches="tight")
plt.show()
```

---

## Development & Testing

Run unit tests via `pytest`:
```bash
python3 -m pytest -v
```

---

## License

This project is licensed under the MIT License.
