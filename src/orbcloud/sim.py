"""
sim.py - Simulated exoplanet posterior parameter generator without redundant RV amplitude/epoch parameters.
"""
from dataclasses import dataclass
import numpy as np

# Real-world reference stars mapping spectral type, physical mass (M_sun), and visual properties
STELLAR_DATABASE = {
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
    e_mean: float = None   # Eccentricity (None defaults to Beta prior)
    e_std: float = None    # Eccentricity uncertainty
    i_deg: float = 0.0     # Fixed inclination (degrees) - defaults to coplanar
    Omega_deg: float = 0.0 # Fixed longitude of ascending node (degrees) - defaults to coplanar


def generate_posterior_samples(config: PlanetConfig, num_samples: int = 1000) -> dict:
    """
    Simulates posterior distributions for an exoplanet's geometric parameters.
    
    Returns a dictionary of numpy arrays (each of length num_samples).
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
