"""
kepler_math.py - Vectorized Kepler solvers and coordinate projections using a Mean Anomaly phase grid.
"""
import numpy as np

def solve_kepler(M: np.ndarray, e: np.ndarray, tol: float = 1e-6, max_iter: int = 100) -> np.ndarray:
    """
    Vectorized Newton-Raphson solver for Kepler's Equation: M = E - e*sin(E).
    
    Parameters:
      M: np.ndarray of shape (N_samples, N_points)
      e: np.ndarray of shape (N_samples, 1) (broadcastable to M)
    """
    e_arr = np.asarray(e)
    if np.any(e_arr < 0.0) or np.any(e_arr >= 1.0):
        raise ValueError(
            f"Eccentricity values must be in range [0, 1) for elliptic orbits. "
            f"Got range [{np.min(e_arr)}, {np.max(e_arr)}]. "
            f"Suggested typical values: 0.0 (circular), 0.15, or 0.3."
        )

    E = M.copy()
    
    for _ in range(max_iter):
        f = E - e_arr * np.sin(E) - M
        f_prime = 1.0 - e_arr * np.cos(E)
        delta = f / f_prime
        E -= delta
        
        # Check convergence
        if np.max(np.abs(delta)) < tol:
            break
            
    return E

def kepler_to_cartesian(
    P: np.ndarray,
    e: np.ndarray,
    omega: np.ndarray,
    i: np.ndarray,
    Omega: np.ndarray,
    M_grid: np.ndarray,
    m_star: float
) -> np.ndarray:
    """
    Converts Keplerian elements to 3D Cartesian coordinates (x, y, z)
    
    All inputs P, e, omega, i, Omega are 1D arrays of length N_samples.
    M_grid is a 1D array of length N_points.
    
    Returns an array of shape (N_samples, N_points, 3).
    """
    # 1. Type validation for stellar mass
    if not isinstance(m_star, (int, float, np.integer, np.floating)):
        raise TypeError(
            f"Stellar mass (m_star) must be a numeric value. Got {type(m_star).__name__}: {m_star!r}. "
            f"Suggested: 1.0 (for solar mass)."
        )
    if m_star <= 0:
        raise ValueError(
            f"Stellar mass (m_star) must be positive and greater than 0. Got {m_star}. "
            f"Suggested values: 1.0 (for solar mass), 0.14 (M dwarf), or 2.1 (A star)."
        )

    # 2. Convert and validate inputs are array-like
    P = np.asarray(P)
    e = np.asarray(e)
    omega = np.asarray(omega)
    i = np.asarray(i)
    Omega = np.asarray(Omega)
    M_grid = np.asarray(M_grid)

    if P.ndim != 1 or e.ndim != 1 or omega.ndim != 1 or i.ndim != 1 or Omega.ndim != 1:
        raise ValueError(
            f"Orbital elements (P, e, omega, i, Omega) must be 1D arrays representing samples. "
            f"Got dimensions: P={P.ndim}, e={e.ndim}, omega={omega.ndim}, i={i.ndim}, Omega={Omega.ndim}."
        )

    lengths = {
        'P': len(P),
        'e': len(e),
        'omega': len(omega),
        'i': len(i),
        'Omega': len(Omega)
    }
    if len(set(lengths.values())) > 1:
        raise ValueError(
            f"All orbital element arrays (P, e, omega, i, Omega) must have the exact same length (number of samples). "
            f"Got lengths: {lengths}. "
            f"Please verify your sample generation setup."
        )

    if len(P) == 0:
        raise ValueError("Orbital element arrays cannot be empty.")
    if len(M_grid) == 0:
        raise ValueError("M_grid (phase points grid) cannot be empty.")

    num_samples = len(P)
    
    # Deriving semi-major axis (a) in AU from Period (days) using Kepler's 3rd Law:
    P_years = P / 365.25
    a = (P_years**2 * m_star)**(1/3)
    
    # Reshape elements to (N_samples, 1) for broadcasting
    a = a[:, np.newaxis]
    e = e[:, np.newaxis]
    i = i[:, np.newaxis]
    omega = omega[:, np.newaxis]
    Omega = Omega[:, np.newaxis]
    
    # Tile the Mean Anomaly grid (shape: N_samples, N_points)
    M = np.tile(M_grid, (num_samples, 1))
    M = M % (2 * np.pi)
    
    # Solve Kepler's Equation to get Eccentric Anomaly E
    E = solve_kepler(M, e)
    
    # Position in the orbital plane
    x_orb = a * (np.cos(E) - e)
    y_orb = a * np.sqrt(1.0 - e**2) * np.sin(E)
    
    # Standard 3D rotations from orbital plane to sky plane (x, y, z)
    cos_omega, sin_omega = np.cos(omega), np.sin(omega)
    cos_Omega, sin_Omega = np.cos(Omega), np.sin(Omega)
    cos_i, sin_i = np.cos(i), np.sin(i)
    
    # Rotation transformation
    x = x_orb * (cos_omega * cos_Omega - sin_omega * sin_Omega * cos_i) - \
        y_orb * (sin_omega * cos_Omega + cos_omega * sin_Omega * cos_i)
        
    y = x_orb * (cos_omega * sin_Omega + sin_omega * cos_Omega * cos_i) - \
        y_orb * (sin_omega * sin_Omega - cos_omega * cos_Omega * cos_i)
        
    z = - x_orb * (sin_omega * sin_i) - \
        y_orb * (cos_omega * sin_i)
        
    # Stack x, y, z into a single (N_samples, N_points, 3) matrix
    coords = np.stack([x, y, z], axis=-1)
    
    return coords
