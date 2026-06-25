"""
sim.py - Simulated exoplanet posterior parameter generator without redundant RV amplitude/epoch parameters.
"""
from dataclasses import dataclass
from typing import Any, Optional
import numpy as np

# Real-world reference stars mapping spectral type, physical mass (M_sun), and visual properties
STELLAR_DATABASE: dict[str, dict[str, Any]] = {
    'theta1': {'name': 'Theta1 Orionis C', 'type': 'O', 'mass': 33.0,  'color': '#9bb0ff', 'size': 150},
    'achernar':         {'name': 'Achernar',          'type': 'B', 'mass': 6.7,   'color': '#aabfff', 'size': 120},
    'vega':             {'name': 'Vega',              'type': 'A', 'mass': 2.1,   'color': '#cad7ff', 'size': 90},
    'upsilon':{'name': 'Upsilon Andromedae','type': 'F', 'mass': 1.3,   'color': '#f8f7ff', 'size': 80},
    'sun':              {'name': 'The Sun',           'type': 'G', 'mass': 1.0,   'color': '#FFCC00', 'size': 70},
    'epsilon':  {'name': 'Epsilon Eridani',   'type': 'K', 'mass': 0.82,  'color': '#ffd2a1', 'size': 60},
    'barnard':    {'name': "Barnard's Star",    'type': 'M', 'mass': 0.144, 'color': '#ff9e9e', 'size': 45}
}

@dataclass
class PlanetConfig:
    name: str
    P_mean: float          # Period (days)
    P_std: float           # Period uncertainty (days)
    omega_mean_deg: float  # Argument of periastron (degrees)
    omega_std_deg: float   # Argument of periastron uncertainty (degrees)
    e_mean: Optional[float] = None   # Eccentricity (None defaults to Beta prior)
    e_std: Optional[float] = None    # Eccentricity uncertainty
    i_deg: float = 0.0     # Fixed inclination (degrees) - defaults to 0
    Omega_deg: float = 0.0 # Fixed longitude of ascending node (degrees) - defaults to 0

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise TypeError(
                f"Planet name must be a non-empty string. Got {type(self.name).__name__}: {self.name!r}. "
                f"Suggested: 'Planet b' or 'Kepler-186f'."
            )
            
        def _check_numeric(val, field_name):
            if not isinstance(val, (int, float, np.integer, np.floating)):
                raise TypeError(
                    f"{field_name} must be a number (float or int). Got {type(val).__name__}: {val!r}. "
                    f"Please check that you didn't pass a string, list, or dictionary."
                )

        _check_numeric(self.P_mean, "P_mean")
        _check_numeric(self.P_std, "P_std")
        _check_numeric(self.omega_mean_deg, "omega_mean_deg")
        _check_numeric(self.omega_std_deg, "omega_std_deg")
        _check_numeric(self.i_deg, "i_deg")
        _check_numeric(self.Omega_deg, "Omega_deg")

        if self.P_mean <= 0:
            raise ValueError(
                f"P_mean (period mean) must be positive and greater than 0. Got {self.P_mean}. "
                f"Suggested: 90.0 or 365.25."
            )
        if self.P_std < 0:
            raise ValueError(
                f"P_std (period uncertainty) must be non-negative. Got {self.P_std}. "
                f"Suggested: 5.0 or 0.0."
            )
        if self.omega_std_deg < 0:
            raise ValueError(
                f"omega_std_deg (omega uncertainty) must be non-negative. Got {self.omega_std_deg}. "
                f"Suggested: 10.0 or 0.0."
            )

        if (self.e_mean is None) != (self.e_std is None):
            raise ValueError(
                f"Both e_mean and e_std must be specified (or both left as None to use the default Beta prior). "
                f"Got e_mean={self.e_mean}, e_std={self.e_std}. "
                f"Please specify both (e.g., e_mean=0.15, e_std=0.05) or omit both."
            )

        if self.e_mean is not None:
            _check_numeric(self.e_mean, "e_mean")
            if not (0.0 <= self.e_mean < 1.0):
                raise ValueError(
                    f"e_mean (eccentricity) must be in the range [0, 1) for elliptic orbits. Got {self.e_mean}. "
                    f"Suggested typical values: 0.0 (circular), 0.15, or 0.3."
                )

        if self.e_std is not None:
            _check_numeric(self.e_std, "e_std")
            if self.e_std < 0:
                raise ValueError(
                    f"e_std (eccentricity uncertainty) must be non-negative. Got {self.e_std}. "
                    f"Suggested: 0.05 or 0.0."
                )



def generate_posterior_samples(config: PlanetConfig, num_samples: int = 1000) -> dict:
    """
    Simulates posterior distributions for an exoplanet's geometric parameters.

    Args:
        config (PlanetConfig object): The dataclass storing the planet's orbital parameters
        num_samples (int): The number of posterior samples to generate to simulate MCMC data (default set to 1000)

    Returns:
        dict: a dictionary of the posteriors (arrays of length = num_samples) to simulate MCMC data
            * 'P' (array): Period (days)
            * 'e' (array): Eccentricity
            * 'omega' (array): Argument of periapsis (radians)
            * 'i' (array): Inclination (radians)
            * 'Omega' (array): Longitude of the ascending node (radians)
    """
    rng = np.random.default_rng()
    
    # 1. Period (Normal distribution)
    P = rng.normal(config.P_mean, config.P_std, size=num_samples)
    P = np.maximum(P, 1e-3)  # Period must be positive
    
    # 2. Argument of periastron (Normal distribution in radians, wrapped to [0, 2pi])
    omega = np.radians(rng.normal(config.omega_mean_deg, config.omega_std_deg, size=num_samples))
    omega = omega % (2 * np.pi)
    
    # 3. Eccentricity (Beta distribution by default, or Normal if configured)
    if config.e_mean is None or config.e_std is None:
        # Standard exoplanet prior: Beta(alpha=0.867, beta=3.03) from Kipping 2013
        e = rng.beta(0.867, 3.03, size=num_samples)
    else:
        e = rng.normal(config.e_mean, config.e_std, size=num_samples)
        e = np.clip(e, 0.0, 0.99)  # Bounded for closed orbits
        
    # 4. Fixed 3D orientation parameters (i, Omega)
    i = np.full(num_samples, np.radians(config.i_deg))
    Omega = np.full(num_samples, np.radians(config.Omega_deg))
    
    return {
        'P': P,
        'e': e,
        'omega': omega,
        'i': i,
        'Omega': Omega
    }
