import numpy as np
import pytest
from orbcloud.kepler_math import solve_kepler, kepler_to_cartesian

def test_solve_kepler_convergence():
    # Test convergence for low, medium, and high eccentricities
    M_grid = np.linspace(0.0, 2 * np.pi, 100)
    # Shape (N_samples, N_points)
    M = np.tile(M_grid, (3, 1))
    
    # 3 samples: e = 0.0, 0.5, 0.99
    e = np.array([[0.0], [0.5], [0.99]])
    
    E = solve_kepler(M, e, tol=1e-6)
    
    # Check Kepler's Equation: M = E - e*sin(E)
    kepler_err = np.abs(E - e * np.sin(E) - M)
    assert np.max(kepler_err) < 1e-6

def test_solve_kepler_invalid_eccentricity():
    M = np.array([[0.0, 1.0]])
    
    # Eccentricity >= 1.0
    with pytest.raises(ValueError) as excinfo:
        solve_kepler(M, np.array([[1.0]]))
    assert "Eccentricity values must be in range [0, 1)" in str(excinfo.value)
    assert "Suggested" in str(excinfo.value)
    
    # Eccentricity < 0
    with pytest.raises(ValueError) as excinfo:
        solve_kepler(M, np.array([[-0.1]]))
    assert "Eccentricity values must be in range [0, 1)" in str(excinfo.value)

def test_kepler_to_cartesian_circular_orbit():
    # Test that e = 0, i = 0 yields a perfect circle in X-Y plane
    P = np.array([365.25])  # 1 year -> a = 1.0 AU around 1 solar mass star
    e = np.array([0.0])
    omega = np.array([0.0])
    i = np.array([0.0])
    Omega = np.array([0.0])
    M_grid = np.linspace(0.0, 2 * np.pi, 100)
    m_star = 1.0
    
    coords = kepler_to_cartesian(P, e, omega, i, Omega, M_grid, m_star)
    # Expected shape: (1, 100, 3)
    assert coords.shape == (1, 100, 3)
    
    x = coords[0, :, 0]
    y = coords[0, :, 1]
    z = coords[0, :, 2]
    
    # Z should be exactly 0
    np.testing.assert_allclose(z, 0.0, atol=1e-12)
    
    # Radius should be exactly 1.0 (since a = 1.0 AU)
    radii = np.sqrt(x**2 + y**2)
    np.testing.assert_allclose(radii, 1.0, rtol=1e-5)

def test_kepler_to_cartesian_coplanar():
    # Test that i = 0 yields z = 0 for any eccentricity and parameters
    P = np.array([100.0, 200.0])
    e = np.array([0.1, 0.5])
    omega = np.array([0.5, 1.2])
    i = np.array([0.0, 0.0])
    Omega = np.array([1.5, 2.0])
    M_grid = np.linspace(0.0, 2 * np.pi, 50)
    m_star = 1.2
    
    coords = kepler_to_cartesian(P, e, omega, i, Omega, M_grid, m_star)
    z = coords[:, :, 2]
    np.testing.assert_allclose(z, 0.0, atol=1e-12)

def test_kepler_to_cartesian_invalid_inputs():
    P = np.array([100.0])
    e = np.array([0.1])
    omega = np.array([0.5])
    i = np.array([0.0])
    Omega = np.array([1.5])
    M_grid = np.linspace(0.0, 2 * np.pi, 50)
    
    # 1. Invalid m_star type
    with pytest.raises(TypeError) as excinfo:
        kepler_to_cartesian(P, e, omega, i, Omega, M_grid, "invalid_mass")
    assert "Stellar mass (m_star) must be a numeric value" in str(excinfo.value)
    
    # 2. Invalid m_star value
    with pytest.raises(ValueError) as excinfo:
        kepler_to_cartesian(P, e, omega, i, Omega, M_grid, -1.0)
    assert "Stellar mass (m_star) must be positive" in str(excinfo.value)
    
    # 3. Mismatched array lengths
    with pytest.raises(ValueError) as excinfo:
        kepler_to_cartesian(np.array([100.0, 200.0]), e, omega, i, Omega, M_grid, 1.0)
    assert "must have the exact same length" in str(excinfo.value)

    # 4. Empty arrays
    with pytest.raises(ValueError) as excinfo:
        kepler_to_cartesian(np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), M_grid, 1.0)
    assert "arrays cannot be empty" in str(excinfo.value)
