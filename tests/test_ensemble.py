import pytest
import numpy as np
from orbcloud.sim import PlanetConfig
from orbcloud.ensemble import SystemEnsemble

def test_planet_config_validation():
    # 1. Valid config should construct without errors
    config = PlanetConfig(
        name="Planet b",
        P_mean=10.0, P_std=1.0,
        omega_mean_deg=90.0, omega_std_deg=10.0,
        e_mean=0.2, e_std=0.05,
        i_deg=10.0, Omega_deg=5.0
    )
    assert config.name == "Planet b"
    
    # 2. Invalid name (empty or not string)
    with pytest.raises(TypeError) as excinfo:
        PlanetConfig(name=123, P_mean=10.0, P_std=1.0, omega_mean_deg=90.0, omega_std_deg=10.0)
    assert "Planet name must be a non-empty string" in str(excinfo.value)

    # 3. Invalid P_mean
    with pytest.raises(ValueError) as excinfo:
        PlanetConfig(name="b", P_mean=0.0, P_std=1.0, omega_mean_deg=90.0, omega_std_deg=10.0)
    assert "P_mean (period mean) must be positive" in str(excinfo.value)
    assert "Suggested" in str(excinfo.value)

    # 4. Invalid P_std
    with pytest.raises(ValueError) as excinfo:
        PlanetConfig(name="b", P_mean=10.0, P_std=-1.0, omega_mean_deg=90.0, omega_std_deg=10.0)
    assert "P_std (period uncertainty) must be non-negative" in str(excinfo.value)

    # 5. Mismatched e_mean/e_std (one set, other None)
    with pytest.raises(ValueError) as excinfo:
        PlanetConfig(name="b", P_mean=10.0, P_std=1.0, omega_mean_deg=90.0, omega_std_deg=10.0, e_mean=0.1)
    assert "Both e_mean and e_std must be specified" in str(excinfo.value)

    # 6. Invalid e_mean boundary (>= 1.0)
    with pytest.raises(ValueError) as excinfo:
        PlanetConfig(name="b", P_mean=10.0, P_std=1.0, omega_mean_deg=90.0, omega_std_deg=10.0, e_mean=1.2, e_std=0.1)
    assert "e_mean (eccentricity) must be in the range [0, 1)" in str(excinfo.value)

    # 7. Invalid non-numeric type
    with pytest.raises(TypeError) as excinfo:
        PlanetConfig(name="b", P_mean="ten", P_std=1.0, omega_mean_deg=90.0, omega_std_deg=10.0)
    assert "P_mean must be a number" in str(excinfo.value)


def test_system_ensemble_init_validation():
    # 1. Valid initialization from DB
    sys_env = SystemEnsemble(star_id="vega")
    assert sys_env.star_props["name"] == "Vega"
    assert sys_env.star_props["type"] == "A"
    assert sys_env.m_star == 2.1
    
    # 2. Invalid star type
    with pytest.raises(ValueError) as excinfo:
        SystemEnsemble(star_type="Z")
    assert "Invalid star_type" in str(excinfo.value)
    
    # 3. Invalid star mass
    with pytest.raises(ValueError) as excinfo:
        SystemEnsemble(star_mass=-0.5)
    assert "star_mass must be positive" in str(excinfo.value)
    
    with pytest.raises(TypeError) as excinfo:
        SystemEnsemble(star_mass="heavy")
    assert "star_mass must be a numeric value" in str(excinfo.value)


def test_add_planet_validation():
    sys_env = SystemEnsemble(star_id="sun")
    
    # 1. Invalid config type
    with pytest.raises(TypeError):
        sys_env.add_planet("not_a_config_object")
        
    # 2. Invalid num_samples/num_points
    config = PlanetConfig(name="Planet b", P_mean=10.0, P_std=1.0, omega_mean_deg=90.0, omega_std_deg=10.0)
    
    with pytest.raises(ValueError) as excinfo:
        sys_env.add_planet(config, num_samples=0)
    assert "num_samples must be positive" in str(excinfo.value)
    
    with pytest.raises(TypeError) as excinfo:
        sys_env.add_planet(config, num_points="many")
    assert "num_points must be an integer" in str(excinfo.value)


def test_plot_system_validation_and_filtering():
    sys_env = SystemEnsemble(star_id="sun")
    config_b = PlanetConfig(name="Planet b", P_mean=50.0, P_std=2.0, omega_mean_deg=45.0, omega_std_deg=10.0)
    config_c = PlanetConfig(name="Planet c", P_mean=150.0, P_std=5.0, omega_mean_deg=120.0, omega_std_deg=15.0)
    
    sys_env.add_planet(config_b, num_samples=100, num_points=50)
    sys_env.add_planet(config_c, num_samples=100, num_points=50)
    
    # 1. Invalid plot dimension
    with pytest.raises(ValueError) as excinfo:
        sys_env.plot_system(dimension="4d")
    assert "dimension must be '2d', '3d', or 'both'" in str(excinfo.value)
    
    # 2. Non-existent planet filter
    with pytest.raises(ValueError) as excinfo:
        sys_env.plot_system(planets_to_show=["Planet d"])
    assert "Planet 'Planet d' is not in the system ensemble" in str(excinfo.value)
    assert "Available planets: ['Planet b', 'Planet c']" in str(excinfo.value)
    
    # 3. Invalid alpha ranges
    with pytest.raises(ValueError) as excinfo:
        sys_env.plot_system(alpha=1.5)
    assert "alpha must be in the range [0.0, 1.0]" in str(excinfo.value)
    
    # 4. Invalid limit padding
    with pytest.raises(ValueError) as excinfo:
        sys_env.plot_system(limit_padding=-0.1)
    assert "limit_padding must be positive" in str(excinfo.value)
