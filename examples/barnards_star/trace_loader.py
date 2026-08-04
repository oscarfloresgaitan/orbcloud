import numpy as np
import arviz as az
from scipy.stats import gaussian_kde
from orbcloud.kepler_math import kepler_to_cartesian

def get_posterior_mode_kde(trace, param_name):
    if param_name not in trace.posterior.data_vars:
        return np.nan
    samples = trace.posterior[param_name].values.flatten()
    if len(np.unique(samples)) < 2:
        return samples[0] if len(samples) > 0 else np.nan
    kde = gaussian_kde(samples)
    x_range = np.linspace(samples.min(), samples.max(), 1000)
    y_density = kde(x_range)
    return x_range[np.argmax(y_density)]

def load_real_trace_to_system(system, trace_path, num_samples=1000, num_points=200, i_deg=0.0, Omega_deg=0.0, omega_deg=None, ecc=None, planet_prefix=None):
    """
    Loads real MCMC posterior samples from a NetCDF trace file into a SystemEnsemble.
    
    Parameters:
        system (SystemEnsemble): The target orbcloud system ensemble.
        trace_path (str): Path to the NetCDF trace file.
        num_samples (int): Number of posterior samples to draw (downsample).
        num_points (int): Number of points along the orbit to compute.
        i_deg (float or dict): Inclination in degrees (constant for all, or dict of planet_name: angle).
        Omega_deg (float or dict): Longitude of ascending node in degrees (constant or dict).
        omega_deg (float or dict): Fallback/constant argument of periastron in degrees.
        ecc (float or dict): Fallback/constant eccentricity values (e.g. 0.0 for circular orbits).
        planet_prefix (str): Prefix for planet names. Defaults to the system's star name.
    """
    print(f"Loading trace from {trace_path}...")
    idata = az.from_netcdf(trace_path)
    posterior = idata.posterior
    
    # Identify planet indices (e.g., P1, P2, P3...)
    vars_list = list(posterior.data_vars)
    planet_indices = sorted([int(var[1:]) for var in vars_list if var.startswith('P') and var[1:].isdigit()])
    print(f"Found {len(planet_indices)} planets with indices: {planet_indices}")
    
    # Use a fixed random seed for reproducible downsampling
    rng = np.random.default_rng(42)
    
    # Standard Mean Anomaly grid
    M_grid = np.linspace(0.0, 2 * np.pi, num_points)
    
    # Determine the prefix for planet naming (e.g., "Barnard's Star")
    prefix = planet_prefix if planet_prefix is not None else system.star_props['name']
    
    for idx in planet_indices:
        # Exoplanet letter mapping (1 -> b, 2 -> c, 3 -> d, 4 -> e)
        planet_name = f"Planet {chr(97 + idx)}"
        
        # Get variables
        p_var = f"P{idx}"
        e_var = f"ecc{idx}"
        w_var = f"omega{idx}"
        
        # Flatten chain and draw dimensions
        p_chain = posterior[p_var].values.flatten()
        
        # Determine number of available samples
        total_draws = len(p_chain)
        n_samples = min(num_samples, total_draws)
        
        # Randomly sample draws
        indices = rng.choice(total_draws, size=n_samples, replace=False)
        p_samples = p_chain[indices]
        
        # Retrieve or compute eccentricity samples
        if ecc is not None:
            p_e = ecc.get(planet_name, 0.0) if isinstance(ecc, dict) else ecc
            e_samples = np.full(n_samples, p_e)
            nominal_e = p_e
        elif e_var in posterior.data_vars:
            e_chain = posterior[e_var].values.flatten()
            e_samples = e_chain[indices]
            nominal_e = get_posterior_mode_kde(idata, e_var)
        else:
            e_samples = np.zeros(n_samples)
            nominal_e = 0.0
        
        # Retrieve or compute omega samples
        if omega_deg is not None:
            # Explicitly fixed by user
            p_w_deg = omega_deg.get(planet_name, 0.0) if isinstance(omega_deg, dict) else omega_deg
            w_samples = np.full(n_samples, np.radians(p_w_deg))
            nominal_omega = np.radians(p_w_deg)
        elif w_var in posterior.data_vars:
            # Loaded from trace
            w_chain = posterior[w_var].values.flatten()
            w_samples = w_chain[indices]
            nominal_omega = get_posterior_mode_kde(idata, w_var)
        else:
            # Fallback if trace has no omega distribution
            w_samples = np.zeros(n_samples)
            nominal_omega = 0.0
        
        # Get orientation values, falling back to 0.0 if key is missing in dict
        p_i_deg = i_deg.get(planet_name, 0.0) if isinstance(i_deg, dict) else i_deg
        p_Omega_deg = Omega_deg.get(planet_name, 0.0) if isinstance(Omega_deg, dict) else Omega_deg
        
        # Setup 3D orientation arrays (matching length of samples)
        i_samples = np.full(n_samples, np.radians(p_i_deg))
        Omega_samples = np.full(n_samples, np.radians(p_Omega_deg))
        
        # Compute coordinates using orbcloud's vectorized Kepler solver
        coords = kepler_to_cartesian(
            P=p_samples,
            e=e_samples,
            omega=w_samples,
            i=i_samples,
            Omega=Omega_samples,
            M_grid=M_grid,
            m_star=system.m_star
        )
        
        # Compute nominal coordinates
        nominal_P = np.median(p_samples)
        
        nominal_coords = kepler_to_cartesian(
            P=np.array([nominal_P]),
            e=np.array([nominal_e]),
            omega=np.array([nominal_omega]),
            i=np.array([np.radians(p_i_deg)]),
            Omega=np.array([np.radians(p_Omega_deg)]),
            M_grid=M_grid,
            m_star=system.m_star
        )[0]
        
        # Inject directly into the SystemEnsemble planets dict
        system.planets[planet_name] = {
            'coords': coords,
            'nominal_coords': nominal_coords
        }
        
        print(f"Added {planet_name}: P={nominal_P:.4f} d, e={nominal_e:.4f} (mode)")
