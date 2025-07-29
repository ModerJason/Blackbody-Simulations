import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load CSV and create a pivot table of outgoing power indexed by incident angles
def load_and_pivot(csv_path):
    df = pd.read_csv(csv_path)
    df_unique = df.drop_duplicates(subset=["IWavePhi", "IWaveTheta"])
    pivot = df_unique.pivot(index="IWaveTheta", columns="IWavePhi", values="OutgoingPower")
    return pivot

# Plot a heatmap of outgoing power from a pivot table
def plot_outgoing_power_heatmap(pivot_table, title, cmap="viridis"):
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot_table, annot=False, fmt=".2e", cmap=cmap, cbar_kws={"label": r"$P_{out}$ (W)"})
    plt.title(title)
    plt.xlabel(r"$\phi_{in}$ (degrees)")
    plt.ylabel(r"$\theta_{in}$ (degrees)")
    plt.tight_layout()

# Plot outgoing power vs the varying angle (either theta_in or phi_in) for fixed values of the other
def plot_outgoing_power_by_incoming_angle(csv_path, fixed_param, fixed_values):
    param_map = {
        r"$\theta_{in}$": "IWaveTheta",
        r"$\phi_{in}$": "IWavePhi"
    }
    if fixed_param not in param_map:
        raise ValueError("fixed_param must be r'$\\theta_{in}$' or r'$\\phi_{in}$'")
    
    col_fix = param_map[fixed_param]
    col_var = "IWavePhi" if col_fix == "IWaveTheta" else "IWaveTheta"
    xlabel_units = r"$\phi_{in}$ (degrees)" if col_fix == "IWaveTheta" else r"$\theta_{in}$ (degrees)"
    xlabel_no_units = r"$\phi_{in}$" if col_fix == "IWaveTheta" else r"$\theta_{in}$"

    df = pd.read_csv(csv_path)
    if not isinstance(fixed_values, (list, tuple)):
        fixed_values = [fixed_values]

    plt.figure(figsize=(8, 6))
    for val in fixed_values:
        df_filt = df[df[col_fix] == val]
        plt.plot(df_filt[col_var], df_filt["OutgoingPower"], marker='o', markersize=4, label=f"{fixed_param} = {val}°")

    plt.xlabel(xlabel_units)
    plt.ylabel(r"$P_{out}$ (W)")
    plt.title(rf"$P_{{out}}$ vs {xlabel_no_units}")
    plt.grid(True)
    if len(fixed_values) > 1:
        plt.legend()
    plt.tight_layout()

# Plot E-field magnitude at a fixed exit plane using incoming angles
def plot_exit_field_by_incoming_angle(csv_path, theta_in, phi_in,
                                     fixed_coord="Z", fixed_value=0.001,
                                     x_axis="X", y_axis="Y",
                                     s=10, cmap="viridis"):
    df = pd.read_csv(csv_path)
    # Filter data for the given incoming angles and fixed coordinate slice (e.g., z=0.001)
    df_filt = df[(df["IWaveTheta"] == theta_in) & 
                 (df["IWavePhi"] == phi_in) & 
                 (np.isclose(df[fixed_coord], fixed_value))]

    if df_filt.empty:
        print(f"No data for θ={theta_in}°, φ={phi_in}° at {fixed_coord.lower()} = {fixed_value}")
        return

    # Scatter plot of the E-field over the output plane
    plt.figure(figsize=(7, 6))
    sc = plt.scatter(df_filt[x_axis], df_filt[y_axis], c=df_filt["Mag_E"], cmap=cmap, s=s)
    plt.xlabel(f"{x_axis.lower()} (m)")
    plt.ylabel(f"{y_axis.lower()} (m)")
    plt.title(rf"$|E|$ at {fixed_coord.lower()} = {fixed_value}m for $\theta_{{in}}$ = {theta_in}°, $\phi_{{in}}$ = {phi_in}°")
    plt.colorbar(sc, label=r"$|E|$ (V/m)")
    plt.tight_layout()

# Plot 1D slices of far-field magnitude at a fixed outgoing angle
def plot_far_field_by_incoming_angle_fixed(csv_path,
                                    theta_in_list,
                                    phi_in_list,
                                    fixed_param,
                                    fixed_value,
                                    title_prefix="Far-field E-Field Magnitude"):
    param_map = {
        r"$\theta_{out}$": "Theta",
        r"$\phi_{out}$": "Phi"
    }
    if fixed_param not in param_map:
        raise ValueError("fixed_param must be r'$\\theta_{out}$' or r'$\\phi_{out}$'")

    fixed_col = param_map[fixed_param]
    varying_col = "Phi" if fixed_col == "Theta" else "Theta"
    xlabel_units = r"$\phi_{out}$ (degrees)" if fixed_col == "Theta" else r"$\theta_{out}$ (degrees)"

    df = pd.read_csv(csv_path)
    if not isinstance(theta_in_list, (list, tuple)):
        theta_in_list = [theta_in_list]
    if not isinstance(phi_in_list, (list, tuple)):
        phi_in_list = [phi_in_list]

    plt.figure(figsize=(10, 7))
    for theta_in in theta_in_list:
        for phi_in in phi_in_list:
            df_filt = df[(df["IWaveTheta"] == theta_in) & 
                         (df["IWavePhi"] == phi_in) & 
                         (np.isclose(df[fixed_col], fixed_value))]
            if df_filt.empty:
                continue

            # Compute total E-field magnitude
            Ephi_mag = np.sqrt(df_filt["rEphi_real"]**2 + df_filt["rEphi_imag"]**2)
            Etheta_mag = np.sqrt(df_filt["rEtheta_real"]**2 + df_filt["rEtheta_imag"]**2)
            Etot = np.sqrt(Ephi_mag**2 + Etheta_mag**2)

            label = (fr"{fixed_param} = {fixed_value}°, "
                     fr"$\theta_{{in}}$ = {theta_in}°, "
                     fr"$\phi_{{in}}$ = {phi_in}°")
            plt.plot(df_filt[varying_col], Etot, label=label)

    plt.xlabel(xlabel_units)
    plt.ylabel(r"$|E|$ (V/m)")
    plt.title(f"{title_prefix} at {fixed_param} = {fixed_value}°")
    plt.legend(fontsize='small', loc='best')
    plt.grid(True)
    plt.tight_layout()

# Plot full 2D far-field heatmap (E-field magnitude) for a specific incoming angle
def plot_far_field_by_incoming_angle(csv_path,
                                     theta_in,
                                     phi_in,
                                     cmap="viridis",
                                     title_prefix="Far-field E-Field Magnitude",
                                     tick_interval=10):
    df = pd.read_csv(csv_path)

    # Filter for specified incoming angles
    df_filt = df[(df["IWaveTheta"] == theta_in) & (df["IWavePhi"] == phi_in)]

    if df_filt.empty:
        print(f"No data for θ_in={theta_in}°, φ_in={phi_in}°")
        return

    # Compute total E-field magnitude from real/imag components
    Ephi_mag = np.sqrt(df_filt["rEphi_real"]**2 + df_filt["rEphi_imag"]**2)
    Etheta_mag = np.sqrt(df_filt["rEtheta_real"]**2 + df_filt["rEtheta_imag"]**2)
    Etot = np.sqrt(Ephi_mag**2 + Etheta_mag**2)
    df_filt = df_filt.copy()
    df_filt["|E|"] = Etot

    # Pivot to get 2D matrix: Theta_out vs Phi_out
    pivot = df_filt.pivot(index="Theta", columns="Phi", values="|E|")

    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(pivot, cmap=cmap, cbar_kws={"label": r"$|E|$ (V/m)"})

    plt.xlabel(r"$\phi_{out}$ (degrees)")
    plt.ylabel(r"$\theta_{out}$ (degrees)")
    plt.title(rf"{title_prefix} for $\theta_{{in}}$ = {theta_in}°, $\phi_{{in}}$ = {phi_in}°")

    # Set neat tick intervals on axes, even if raw data has lots of decimal places
    x_ticks = np.arange(pivot.columns.min(), pivot.columns.max() + 1, tick_interval)
    y_ticks = np.arange(pivot.index.min(), pivot.index.max() + 1, tick_interval)

    x_tick_positions = [np.abs(pivot.columns - val).argmin() for val in x_ticks]
    y_tick_positions = [np.abs(pivot.index - val).argmin() for val in y_ticks]

    ax.set_xticks(x_tick_positions)
    ax.set_xticklabels([f"{int(round(val))}" for val in x_ticks])
    ax.set_yticks(y_tick_positions)
    ax.set_yticklabels([f"{int(round(val))}" for val in y_ticks])

    plt.tight_layout()
#%%
waveguide_data = "C:/Users/Jason Wang/spyder/projects/Blackbody/Blackbody-Simulations/HFSSSimData/InfParallelPlate_cylindrical2_2000GHz_Ephi=0/waveguide.csv"
far_field_data = "C:/Users/Jason Wang/spyder/projects/Blackbody/Blackbody-Simulations/HFSSSimData/InfParallelPlate_cylindrical2_2000GHz_Ephi=0/far_field.csv"
#%%
pivot = load_and_pivot(waveguide_data)

plot_outgoing_power_heatmap(pivot, r"$P_{out}$ vs $\theta_{in}$ and $\phi_{in}$ (E$_\phi$=0)")
#%%
plot_outgoing_power_by_incoming_angle(waveguide_data, fixed_param=r"$\theta_{in}$", fixed_values=[156, 168, 180])
#%%
plot_outgoing_power_by_incoming_angle(waveguide_data, fixed_param=r"$\phi_{in}$", fixed_values=[0, 12, 24])
#%%
plot_exit_field_by_incoming_angle(csv_path=waveguide_data, theta_in =90, phi_in = 180, fixed_coord="X", x_axis="Y", y_axis="Z", fixed_value=0.0004)
#%%
plot_far_field_by_incoming_angle_fixed(
    csv_path=far_field_data,
    theta_in_list=[90, 95, 100],
    phi_in_list=[160, 175, 180],
    fixed_param=r"$\theta_{out}$",
    fixed_value=90
)
#%%
plot_far_field_by_incoming_angle(
    csv_path=far_field_data,
    theta_in=90,
    phi_in=180
)