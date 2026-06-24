import sys
import os
import matplotlib.pyplot as plt

# Ensure the local src/ package is in the path
sys.path.insert(0, os.path.abspath('./src'))

from orbcloud import PlanetConfig, SystemEnsemble

def run():
    print("Initializing exoplanetary system ensemble...")
    system = SystemEnsemble(star_name='Custom Star', star_mass=1.6, star_type='M')
    
    # Planet b config (~0.4 AU, flat in X-Y plane, K and t0 dropped for simplicity)
    planet_b = PlanetConfig(
        name='Planet b',
        P_mean=90.0, P_std=8.0,
        omega_mean_deg=60.0, omega_std_deg=40.0,
        e_mean=0.15, e_std=0.09,
        i_deg=0.0, Omega_deg=0.0
    )
    
    # Planet c config (~0.8 AU, flat in X-Y plane, K and t0 dropped for simplicity)
    planet_c = PlanetConfig(
        name='Planet c',
        P_mean=260.0, P_std=14.0,
        omega_mean_deg=210.0, omega_std_deg=45.0,
        e_mean=0.25, e_std=0.10,
        i_deg=0.0, Omega_deg=0.0
    )
    
    system.add_planet(planet_b, num_samples=1000)
    system.add_planet(planet_c, num_samples=1000)
    
    # Render both 2D and 3D subplots side-by-side using the library's defaults
    print("Rendering exoplanetary system...")
    system.plot_system(show_reference_plane=True)
    
    plt.show()
    
    # Save the output image locally
    output_path = 'my_system_plot2.png'
    plt.savefig(output_path, dpi=150, facecolor='white', bbox_inches='tight')

if __name__ == '__main__':
    run()
