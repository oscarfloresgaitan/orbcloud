"""
ensemble.py - Classes representing multi-planet exoplanetary system ensembles.
"""
import numpy as np
import matplotlib.pyplot as plt
from orbcloud.sim import PlanetConfig, generate_posterior_samples, STELLAR_DATABASE
from orbcloud.kepler_math import kepler_to_cartesian

def _darken_color(hex_color: str, amount: float = 0.25) -> str:
    """Darkens a hex color by a specified amount (0.0 to 1.0) for the border.
    
    Args:
        hex_color (str): The # + 6 letter/number hexcode of the color of the orbit (e.g #F54927)
        amount (float): The amount from 0.0 to 1.0 that the color will be darkened (default set to 0.25)

    Returns:
        str: The hexcode of the darkened color in the format of #xxxxxx
    """
    hex_color = hex_color.lstrip('#')
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    r = int(r * (1.0 - amount))
    g = int(g * (1.0 - amount))
    b = int(b * (1.0 - amount))
    return f"#{r:02x}{g:02x}{b:02x}"


SPECTRAL_COLORS = {
    'O': '#9bb0ff',
    'B': '#aabfff',
    'A': '#cad7ff',
    'F': '#f8f7ff',
    'G': '#FFCC00',
    'K': '#ffd2a1',
    'M': '#ff9e9e'
}

SPECTRAL_SIZES = {
    'O': 150,
    'B': 120,
    'A': 90,
    'F': 80,
    'G': 70,
    'K': 60,
    'M': 45
}


class SystemEnsemble:
    def __init__(self, star_id: str = 'sun', star_name: str = None, star_mass: float = None, star_type: str = None):
        '''
        Args:
            star_id (str): The id of the star to access the stellar properties in STELLAR_DATABASE (default set to sun)
            star_name (str): Name of the star (default set to None)
            star_mass (float): Mass of the star (default set to None)
            star_type (str): The stellar spectral type of the star (default set to None)
        
        '''
        if star_id is not None and not isinstance(star_id, str):
            raise TypeError(
                f"star_id must be a string. Got {type(star_id).__name__}: {star_id!r}. "
                f"Suggested: 'sun', 'vega', 'barnard'."
            )
        self.star_id = star_id.lower() if star_id else None
        
        # Determine base properties
        if self.star_id in STELLAR_DATABASE:
            base_props = STELLAR_DATABASE[self.star_id]
        else:
            base_props = STELLAR_DATABASE['sun']
            
        name = star_name if star_name is not None else (base_props['name'] if self.star_id in STELLAR_DATABASE else "Custom Star")
        
        if star_mass is not None:
            if not isinstance(star_mass, (int, float, np.integer, np.floating)):
                raise TypeError(
                    f"star_mass must be a numeric value. Got {type(star_mass).__name__}: {star_mass!r}. "
                    f"Suggested: 1.0 (solar mass)."
                )
            if star_mass <= 0:
                raise ValueError(
                    f"star_mass must be positive and greater than 0. Got {star_mass}. "
                    f"Suggested typical values: 1.0 (for solar mass), 0.14 (M dwarf), or 2.1 (A star)."
                )
        mass = star_mass if star_mass is not None else base_props['mass']

        if star_type is not None:
            if not isinstance(star_type, str):
                raise TypeError(
                    f"star_type must be a string representing a spectral class. Got {type(star_type).__name__}: {star_type!r}. "
                    f"Suggested spectral classes: 'O', 'B', 'A', 'F', 'G', 'K', or 'M'."
                )
            stype = star_type.upper()
            if stype not in SPECTRAL_COLORS:
                raise ValueError(
                    f"Invalid star_type {star_type!r}. Must be one of the spectral classes: {list(SPECTRAL_COLORS.keys())}. "
                    f"Suggested: 'G' (like the Sun), 'M' (like Barnard's Star), or 'A' (like Vega)."
                )
        else:
            stype = base_props['type']
        
        # Determine color and size
        if star_type is not None:
            color = SPECTRAL_COLORS.get(stype, '#fff4e8')
            size = SPECTRAL_SIZES.get(stype, 70)
        else:
            color = base_props['color']
            size = base_props['size']
            
        self.star_props = {
            'name': name,
            'type': stype,
            'mass': mass,
            'color': color,
            'size': size
        }
        self.m_star = self.star_props['mass']
        self.planets = {} 
        
    def add_planet(self, config: PlanetConfig, num_samples: int = 1000, num_points: int = 200):
        """Simulates parameter posterior distributions and pre-computes 3D paths for a planet. This function stores the computed orbital coordinates.
        
        Args:
            config (PlanetConfig object): The dataclass storing the planet's orbital parameters
            num_samples (int): The number of posterior samples to generate (default set to 1000)
            num_points (int): The number of points to generate (default set to 200)

        Returns:
            None

        """
        # Validate inputs
        if not isinstance(config, PlanetConfig):
            raise TypeError(
                f"config must be an instance of PlanetConfig. Got {type(config).__name__}: {config!r}. "
                f"Please create a PlanetConfig instance first."
            )
        if not isinstance(num_samples, (int, np.integer)):
            raise TypeError(
                f"num_samples must be an integer. Got {type(num_samples).__name__}: {num_samples!r}. "
                f"Suggested: 500 or 1000."
            )
        if num_samples <= 0:
            raise ValueError(
                f"num_samples must be positive and greater than 0. Got {num_samples}. "
                f"Suggested: 1000."
            )
        if not isinstance(num_points, (int, np.integer)):
            raise TypeError(
                f"num_points must be an integer. Got {type(num_points).__name__}: {num_points!r}. "
                f"Suggested: 100 or 200."
            )
        if num_points <= 0:
            raise ValueError(
                f"num_points must be positive and greater than 0. Got {num_points}. "
                f"Suggested: 200."
            )

        # 1. Generate posterior samples
        samples = generate_posterior_samples(config, num_samples)
        
        # 2. Setup standard Mean Anomaly grid
        M_grid = np.linspace(0.0, 2 * np.pi, num_points)
        
        # 3. Compute (num_samples, num_points, 3) coordinate paths for the cloud
        coords = kepler_to_cartesian(
            P=samples['P'],
            e=samples['e'],
            omega=samples['omega'],
            i=samples['i'],
            Omega=samples['Omega'],
            M_grid=M_grid,
            m_star=self.m_star
        )
        
        # 4. Compute representative nominal orbit using parameter averages
        mean_i = np.mean(samples['i'])
        mean_Omega = np.mean(samples['Omega'])
        mean_e = config.e_mean if config.e_mean is not None else np.mean(samples['e'])
        
        nominal_coords = kepler_to_cartesian(
            P=np.array([config.P_mean]),
            e=np.array([mean_e]),
            omega=np.array([np.radians(config.omega_mean_deg)]),
            i=np.array([mean_i]),
            Omega=np.array([mean_Omega]),
            M_grid=M_grid,
            m_star=self.m_star
        )[0]
        
        # Store computed coordinates directly
        self.planets[config.name] = {
            'coords': coords,
            'nominal_coords': nominal_coords
        }
        
    def plot_system(
        self,
        ax=None,
        dimension: str = 'both',
        planets_to_show: list[str] = None,
        alpha: float = None,
        alpha_2d: float = 0.02,
        alpha_3d: float = 0.01,
        colors: list[str] = None,
        elev: float = 20.0,
        azim: float = -60.0,
        limit_padding: float = 1.02,
        show_reference_plane: bool = False
    ):
        """
        Plots the exoplanet system in 2D, 3D, or both side-by-side.

        Args:
            ax (Axes object): Creates a Matplotlib Axes object for plotting (default set to None)
            dimension (str): Generates a 2D and/or 3D plot of the orbits (default set to 'both')
            planets_to_show (list[str]): Identifies which planets in the system should be shown on the plot (default set to None)
            alpha (float): The opacity of the plotted data (deafult set to None)
            alpha_2d (float): The opacity of the plotted data for the 2D plot (default set to 0.02)
            alpha_3d (float): The opacity of the plotted data for the 3D plot (default set to 0.01)
            colors (list[str]): The colors to be used for the orbits (default set to None)
            elev (float): The elevation of the 3D plot for display (default set to 20.0)
            azim (float): The azimuth of the 3D plot for display (default set to -60.0)
            limit_padding (float): Sets the plot axes limits based on the maximum coordinate values (default set to 1.02)
            show_reference_plane (bool): Shows the reference (0º) plane of the orbits on the 3D plot (default set to false) 

        Returns:
            Axes object: The 2D and/or 3D plot of the orbits for the planet(s) in the system. 

        """
        if not isinstance(dimension, str):
            raise TypeError(f"dimension must be a string. Got {type(dimension).__name__}: {dimension!r}")
        dimension = dimension.lower()
        if dimension not in ['2d', '3d', 'both']:
            raise ValueError(
                f"dimension must be '2d', '3d', or 'both'. Got {dimension!r}. "
                f"Suggested: 'both' for 2D & 3D side-by-side, '2d' for top view, or '3d' for lateral view."
            )

        if planets_to_show is not None:
            if not isinstance(planets_to_show, list) or not all(isinstance(p, str) for p in planets_to_show):
                raise TypeError(
                    f"planets_to_show must be a list of planet name strings. Got {type(planets_to_show).__name__}: {planets_to_show!r}. "
                    f"Suggested: ['Planet b'] or ['Planet b', 'Planet c']."
                )
            for p in planets_to_show:
                if p not in self.planets:
                    raise ValueError(
                        f"Planet {p!r} is not in the system ensemble. "
                        f"Available planets: {list(self.planets.keys())}. "
                        f"Please add the planet first using add_planet() or check the name spelling."
                    )

        if alpha is not None:
            if not isinstance(alpha, (int, float, np.integer, np.floating)):
                raise TypeError(f"alpha must be a number. Got {type(alpha).__name__}: {alpha!r}")
            if not (0.0 <= alpha <= 1.0):
                raise ValueError(f"alpha must be in the range [0.0, 1.0]. Got {alpha}")
        
        if alpha_2d is not None:
            if not isinstance(alpha_2d, (int, float, np.integer, np.floating)):
                raise TypeError(f"alpha_2d must be a number. Got {type(alpha_2d).__name__}: {alpha_2d!r}")
            if not (0.0 <= alpha_2d <= 1.0):
                raise ValueError(f"alpha_2d must be in the range [0.0, 1.0]. Got {alpha_2d}")

        if alpha_3d is not None:
            if not isinstance(alpha_3d, (int, float, np.integer, np.floating)):
                raise TypeError(f"alpha_3d must be a number. Got {type(alpha_3d).__name__}: {alpha_3d!r}")
            if not (0.0 <= alpha_3d <= 1.0):
                raise ValueError(f"alpha_3d must be in the range [0.0, 1.0]. Got {alpha_3d}")

        if limit_padding is not None:
            if not isinstance(limit_padding, (int, float, np.integer, np.floating)):
                raise TypeError(f"limit_padding must be a number. Got {type(limit_padding).__name__}: {limit_padding!r}")
            if limit_padding <= 0:
                raise ValueError(f"limit_padding must be positive and greater than 0. Got {limit_padding}")

        
        # 1. Handle side-by-side double plot
        if dimension == 'both':
            if ax is not None:
                if isinstance(ax, (list, tuple)) and len(ax) == 2:
                    ax1, ax2 = ax
                else:
                    raise ValueError("For dimension='both', 'ax' must be a list/tuple of two axes [ax2d, ax3d].")
            else:
                fig = plt.figure(figsize=(18, 9), facecolor='white')
                ax1 = fig.add_subplot(121, facecolor='white')
                ax2 = fig.add_subplot(122, projection='3d', facecolor='white')
                
            ax1.set_title("2D Top View", fontsize=13, pad=12)
            self.plot_system(ax=ax1, dimension='2d', planets_to_show=planets_to_show, alpha=alpha, alpha_2d=alpha_2d, colors=colors, limit_padding=limit_padding)
            
            ax2.set_title("3D Lateral View", fontsize=13, pad=12)
            self.plot_system(ax=ax2, dimension='3d', planets_to_show=planets_to_show, alpha=alpha, alpha_3d=alpha_3d, colors=colors, elev=elev, azim=azim, limit_padding=limit_padding, show_reference_plane=show_reference_plane)
            
            return (ax1, ax2)

        if ax is None:
            fig = plt.figure(figsize=(10, 10), facecolor='white')
            if dimension == '3d':
                ax = fig.add_subplot(111, projection='3d', facecolor='white')
            else:
                ax = fig.add_subplot(111, facecolor='white')
            
        # Get star-specific visual properties
        star_color = self.star_props.get('color', '#fff4e8')
        star_edge = _darken_color(star_color, 0.25)

        if dimension == '3d':
            # Configure light 3D aesthetic
            ax.set_proj_type('ortho')
            ax.grid(True, linestyle=':', alpha=0.6, color='gray')
            
            ax.xaxis.pane.fill = True
            ax.yaxis.pane.fill = True
            ax.zaxis.pane.fill = True
            ax.xaxis.pane.set_facecolor('#f7f7f7')
            ax.yaxis.pane.set_facecolor('#f7f7f7')
            ax.zaxis.pane.set_facecolor('#f7f7f7')
            
            # Set viewing angle (defaults to standard perspective)
            ax.view_init(elev=elev, azim=azim)
            
            # Set axis labels
            ax.set_xlabel('Projected Distance (AU)', labelpad=10)
            ax.set_ylabel('Projected Distance (AU)', labelpad=10)
            ax.set_zlabel('Projected Distance (AU)', labelpad=10)

            # Plot glowing central star in 3D
            ax.scatter(
                [0], [0], zs=[0],
                color=star_color,
                s=self.star_props['size'] * 3, # Make star size visually prominent
                marker='o',
                edgecolors=star_edge,
                linewidths=2,
                zorder=10,
                label=self.star_props['name']
            )
        else:
            # Configure standard 2D aesthetic
            ax.grid(True, linestyle=':', alpha=0.6, color='gray')
            ax.set_xlabel('Projected Distance (AU)', labelpad=10)
            ax.set_ylabel('Projected Distance (AU)', labelpad=10)

            # Plot glowing central star in 2D
            ax.scatter(
                [0], [0],
                color=star_color,
                s=self.star_props['size'] * 3,
                marker='o',
                edgecolors=star_edge,
                linewidths=2,
                zorder=10,
                label=self.star_props['name']
            )
        
        # Determine which planets to render
        planets_list = list(self.planets.keys()) if planets_to_show is None else planets_to_show
        
        default_colors = ['#008B8B', '#C71585', '#7B68EE', '#FF8C00']
        colors = colors or default_colors
        
        # Resolve active alpha based on dimension
        if alpha is not None:
            active_alpha = alpha
        else:
            if dimension == '2d':
                active_alpha = alpha_2d
            elif dimension == '3d':
                active_alpha = alpha_3d
            else:
                active_alpha = 0.02  # Fallback
        
        # Render selected planet ensembles
        for idx, name in enumerate(planets_list):
            if name in self.planets:
                planet_color = colors[idx % len(colors)]
                p_data = self.planets[name]
                
                # 1. Plot the transparent cloud threads
                if dimension == '2d':
                    for orbit in p_data['coords']:
                        ax.plot(orbit[:, 0], orbit[:, 1], color=planet_color, alpha=active_alpha, lw=0.8)
                    # 2. Plot nominal orbit
                    ax.plot(p_data['nominal_coords'][:, 0], p_data['nominal_coords'][:, 1],
                            color=planet_color, linestyle='--', linewidth=2.5, label=name)
                else:
                    for orbit in p_data['coords']:
                        ax.plot(orbit[:, 0], orbit[:, 1], orbit[:, 2], color=planet_color, alpha=active_alpha, lw=0.8)
                    # 2. Plot nominal orbit
                    ax.plot(p_data['nominal_coords'][:, 0], p_data['nominal_coords'][:, 1], p_data['nominal_coords'][:, 2],
                            color=planet_color, linestyle='--', linewidth=2.5, label=name)
                
        # Equalize limits to maintain perfect circle projections
        all_coords = []
        for name in planets_list:
            if name in self.planets and self.planets[name]['coords'] is not None:
                all_coords.append(self.planets[name]['coords'])
                
        if len(all_coords) > 0:
            stacked = np.concatenate(all_coords, axis=0)
            max_val = np.max(np.abs(stacked))
            limit = max_val * limit_padding
            
            # Draw a dim gray plane at Z=0 to show the reference plane of the system
            if dimension == '3d' and show_reference_plane:
                x_plane = np.linspace(-limit, limit, 10)
                y_plane = np.linspace(-limit, limit, 10)
                X_plane, Y_plane = np.meshgrid(x_plane, y_plane)
                Z_plane = np.zeros_like(X_plane)
                ax.plot_surface(X_plane, Y_plane, Z_plane, color='gray', alpha=0.1, shade=False, zorder=0)
            
            ax.set_xlim(-limit, limit)
            ax.set_ylim(-limit, limit)
            if dimension == '3d':
                ax.set_zlim(-limit, limit)
                ax.set_box_aspect([1, 1, 1])
            else:
                ax.set_aspect('equal')
            
        ax.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='lightgray')
        return ax
