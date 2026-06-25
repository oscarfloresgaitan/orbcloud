"""
kepler_math.py - Vectorized Kepler solvers and coordinate projections using a Mean Anomaly phase grid.
"""
import numpy as np

def solve_kepler(M: np.ndarray, e: np.ndarray, tol: float = 1e-6, max_iter: int = 100) -> np.ndarray:
    """
    Vectorized Newton-Raphson solver for Kepler's Equation: M = E - e*sin(E).
    
    Args:
        M (array): Mean anomaly (units = radians, shape = N_samples, N_points)
        e (array): Eccentricity (shape = N_samples, 1)
        tol (float): Error tolerance (default set to 1e-6)
        max_iter (int): Maximum number of iterations to run the solver (default set to 100)
    
    Returns:
        E (array): Eccentric Anomaly (units = radians)
    """
    E = M.copy()
    
    for _ in range(max_iter):
        f = E - e * np.sin(E) - M
        f_prime = 1.0 - e * np.cos(E)
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

    Args:
        P (1D array): Period (unit = days, length = N_samples)
        e (1D array): Eccentricity (length = N_samples
        omega (1D array): Argument of periapsis (unit = radians, length = N_samples)
        i (1D array): Inclination (unit = radians, length = N_samples)
        Omega (1D array): Longitude of the ascending node (unit = radians, length = N_samples)
        M_grid (1D array): Mean anomaly grid (unit = radians, length = N_points)
        m_star (float): Mass of the star (unit = M_sun)
    
    Returns:
        coords (array): An array of x coordinates, y coordinates, and z coordinates of shape (N_samples, N_points, 3). 

    """
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
        
    z = x_orb * (sin_omega * sin_i) + \
        y_orb * (cos_omega * sin_i)
        
    # Stack x, y, z into a single (N_samples, N_points, 3) matrix
    coords = np.stack([x, y, z], axis=-1)
    
    return coords
