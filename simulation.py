import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

waveguide_0_data = "C:/Users/Jason Wang/spyder/projects/Blackbody/Blackbody-Simulations/HFSSSimData/InfParallelPlate_bbsim23_500GHz_Ephi=0/waveguide.csv"
waveguide_1_data = "C:/Users/Jason Wang/spyder/projects/Blackbody/Blackbody-Simulations/HFSSSimData/InfParallelPlate_bbsim23_500GHz_Ephi=1/waveguide.csv"

far_field_0_data = "C:/Users/Jason Wang/spyder/projects/Blackbody/Blackbody-Simulations/HFSSSimData/InfParallelPlate_bbsim23_500GHz_Ephi=0/far_field.csv"
far_field_1_data = "C:/Users/Jason Wang/spyder/projects/Blackbody/Blackbody-Simulations/HFSSSimData/InfParallelPlate_bbsim23_500GHz_Ephi=1/far_field.csv"

waveguide_0_df = pd.read_csv(waveguide_0_data)
waveguide_1_df = pd.read_csv(waveguide_1_data)
far_field_0_df = pd.read_csv(far_field_0_data)
far_field_1_df = pd.read_csv(far_field_1_data)

#%%
def get_S21(phi_in, theta_in, polarization_0, polarization_1, df_0, df_1):
    """
    Retrieve the power transmission fraction |S21|^2 from HFSS data,
    weighted by the incident wave's polarization.

    Parameters:
        phi_in (float): IWavePhi angle (in degrees)
        theta_in (float): IWaveTheta angle (in degrees)
        polarization_0 (float): Component of incident E-field along polarization mode 0 (θ̂)
        polarization_1 (float): Component of incident E-field along polarization mode 1 (ϕ̂)
        df_0 (pd.DataFrame): DataFrame for polarization mode 0 (Ephi=0)
        df_1 (pd.DataFrame): DataFrame for polarization mode 1 (Ephi=1)

    Returns:
        float: Weighted and clipped |S21|^2 (P_out / P_in, capped at 1)
    """

    def lookup_power(df, phi, theta):
        row = df[(df["IWavePhi"] == phi) & (df["IWaveTheta"] == theta)]
        if row.empty:
            raise ValueError(f"No data for IWavePhi={phi}, IWaveTheta={theta}")
        P_out = row["OutgoingPower"].values[0]
        P_in = row["IngoingPower"].values[0]
        if P_in == 0:
            raise ZeroDivisionError("IngoingPower is zero.")
        return min(P_out / P_in, 1)

    transmission_0 = lookup_power(df_0, phi_in, theta_in)
    transmission_1 = lookup_power(df_1, phi_in, theta_in)

    weighted_S21_sq = (
        polarization_0**2 * transmission_0 +
        polarization_1**2 * transmission_1
    )

    return weighted_S21_sq
#%% Stage 2: Determining a probability distribution over the location of emission from the outgoing face.

def get_face_emission_distribution(phi_in, theta_in, polarization_0, polarization_1,
                              df_0, df_1):
    """
    Compute normalized emission probability distribution over the output face
    for a given incoming angle and polarization, using linear superposition of E-fields.

    Parameters:
        phi_in (float): IWavePhi in degrees
        theta_in (float): IWaveTheta in degrees
        polarization_0 (float): Amplitude of polarization mode 0 (Ephi=0)
        polarization_1 (float): Amplitude of polarization mode 1 (Ephi=1)
        df_0 (pd.DataFrame): E-field data for (Etheta=1, Ephi=0)
        df_1 (pd.DataFrame): E-field data for (Etheta=0, Ephi=1)

    Returns:
        pd.DataFrame: With columns [X, Y, Z, E2, Probability]
    """

    # Filter to the relevant angle in both datasets
    f0 = df_0[(df_0["IWavePhi"] == phi_in) & (df_0["IWaveTheta"] == theta_in)].copy()
    f1 = df_1[(df_1["IWavePhi"] == phi_in) & (df_1["IWaveTheta"] == theta_in)].copy()

    if f0.empty or f1.empty:
        raise ValueError(f"No data for IWavePhi={phi_in}, IWaveTheta={theta_in}")

    # Ensure same spatial points
    if not f0[["X", "Y", "Z"]].equals(f1[["X", "Y", "Z"]]):
        raise ValueError("Mismatch in spatial coordinates between df_0 and df_1.")

    # Reconstruct complex E-fields
    E0 = {
        "Ex": f0["Ex_real"] + 1j * f0["Ex_imag"],
        "Ey": f0["Ey_real"] + 1j * f0["Ey_imag"],
        "Ez": f0["Ez_real"] + 1j * f0["Ez_imag"],
    }

    E1 = {
        "Ex": f1["Ex_real"] + 1j * f1["Ex_imag"],
        "Ey": f1["Ey_real"] + 1j * f1["Ey_imag"],
        "Ez": f1["Ez_real"] + 1j * f1["Ez_imag"],
    }

    # Linearly combine fields according to polarization
    E_total = {
        comp: polarization_0 * E0[comp] + polarization_1 * E1[comp]
        for comp in ["Ex", "Ey", "Ez"]
    }

    # Compute |E|^2
    E2 = np.abs(E_total["Ex"])**2 + np.abs(E_total["Ey"])**2 + np.abs(E_total["Ez"])**2

    # Normalize to get probability distribution
    total_E2 = E2.sum()
    if total_E2 == 0:
        raise ValueError("Total field energy is zero — can't normalize.")

    probabilities = E2 / total_E2

    # Return results
    result = f0[["X", "Y", "Z"]].copy()
    result["E2"] = E2
    result["Probability"] = probabilities

    return result

#%% Stage 3: Determining a probability distribution over the angle of emission parametrized by (theta, phi)

def get_angular_emission_distribution(phi_in, theta_in, polarization_0, polarization_1,
                                      df_0, df_1):
    """
    Compute normalized emission probability distribution over outgoing angles (theta, phi)
    for given incoming angle and polarization, using linear superposition of E-fields
    in spherical coordinates (rEtheta, rEphi).

    Parameters:
        phi_in (float): Incoming IWavePhi in degrees
        theta_in (float): Incoming IWaveTheta in degrees
        polarization_0 (float): Amplitude of polarization mode 0 (Ephi=0)
        polarization_1 (float): Amplitude of polarization mode 1 (Ephi=1)
        df_0 (pd.DataFrame): Far-field E data for (Etheta=1, Ephi=0)
        df_1 (pd.DataFrame): Far-field E data for (Etheta=0, Ephi=1)

    Returns:
        pd.DataFrame: With columns [Theta, Phi, E2, Probability, Polarization]
                     Polarization column contains normalized complex vectors [Etheta, Ephi]
    """

    # Filter to the relevant incoming angles
    f0 = df_0[(df_0["IWavePhi"] == phi_in) & (df_0["IWaveTheta"] == theta_in)].copy()
    f1 = df_1[(df_1["IWavePhi"] == phi_in) & (df_1["IWaveTheta"] == theta_in)].copy()

    if f0.empty or f1.empty:
        raise ValueError(f"No data for IWavePhi={phi_in}, IWaveTheta={theta_in}")

    # Ensure angle match
    if not f0[["Theta", "Phi"]].equals(f1[["Theta", "Phi"]]):
        raise ValueError("Mismatch in angular coordinates between df_0 and df_1.")

    # Reconstruct spherical components
    Etheta_0 = f0["rEtheta_real"] + 1j * f0["rEtheta_imag"]
    Ephi_0   = f0["rEphi_real"]   + 1j * f0["rEphi_imag"]

    Etheta_1 = f1["rEtheta_real"] + 1j * f1["rEtheta_imag"]
    Ephi_1   = f1["rEphi_real"]   + 1j * f1["rEphi_imag"]

    # Linear superposition in spherical basis
    Etheta_total = polarization_0 * Etheta_0 + polarization_1 * Etheta_1
    Ephi_total   = polarization_0 * Ephi_0   + polarization_1 * Ephi_1

    # Compute |E|^2 = |Etheta|^2 + |Ephi|^2
    E2 = np.abs(Etheta_total)**2 + np.abs(Ephi_total)**2

    # Normalize to get probability distribution
    total_E2 = E2.sum()
    if total_E2 == 0:
        raise ValueError("Total E-field magnitude squared is zero — can't normalize.")

    probabilities = E2 / total_E2

    # Normalize polarization vectors at each point
    polarization_vectors = np.vstack([Etheta_total, Ephi_total]).T  # shape (N, 2)
    norms = np.linalg.norm(polarization_vectors, axis=1)
    # Avoid division by zero (set zero norm to 1 to avoid NaNs, those rows will have zero probability anyway)
    norms_safe = np.where(norms == 0, 1, norms)
    normalized_polarizations = polarization_vectors / norms_safe[:, None]

    # Construct result DataFrame
    result = f0[["Theta", "Phi"]].copy()
    result["E2"] = E2
    result["Probability"] = probabilities
    result["Polarization"] = [vec for vec in normalized_polarizations]

    return result
#%%
(polarization_0, polarization_1) = (1, 0) # polarization_0 corresponds to E_theta=1, E_phi=0 and polarization_1 corresponds to E_theta=0, E_phi=1
magnitude_squared = polarization_0**2 + polarization_1**2
assert np.isclose(magnitude_squared, 1.0), f"Magnitude squared is {magnitude_squared}, expected 1."

phi_in = 0
theta_in = 180
# Stage 1
power_transmission_fraction = get_S21(
    phi_in, theta_in, polarization_0, polarization_1, waveguide_0_df, waveguide_1_df
)
print(power_transmission_fraction)

# Stage 2
face_emission_distribution = get_face_emission_distribution(
    phi_in, theta_in, polarization_0, polarization_1, waveguide_0_df, waveguide_1_df
)
print(face_emission_distribution.sort_values("Probability", ascending=False).head())

# Stage 3
angular_emission_distribution = get_angular_emission_distribution(phi_in, theta_in, polarization_0, polarization_1, far_field_0_df, far_field_1_df)
print(angular_emission_distribution.sort_values("Probability", ascending=False).head())
#%%
# Sample run
def sample_emission_point(face_distribution):
    """Randomly sample a point from the outgoing face distribution."""
    probabilities = face_distribution["Probability"].values
    indices = np.arange(len(probabilities))
    sampled_index = np.random.choice(indices, p=probabilities)
    sampled_point = face_distribution.iloc[sampled_index][["X", "Y", "Z"]]
    return sampled_point, sampled_index

def sample_emission_angle(angular_distribution):
    """Randomly sample an emission angle (Theta, Phi) from the angular distribution."""
    probabilities = angular_distribution["Probability"].values
    indices = np.arange(len(probabilities))
    sampled_index = np.random.choice(indices, p=probabilities)
    sampled_angle = angular_distribution.iloc[sampled_index][["Theta", "Phi"]]
    return sampled_angle, sampled_index

def simulate_photon_emission(
    phi_in,
    theta_in,
    polarization_0,
    polarization_1,
    waveguide_0_df,
    waveguide_1_df,
    far_field_0_df,
    far_field_1_df
):
    # Stage 1: Compute |S21|^2 (power transmission)
    S21_squared = get_S21(
        phi_in, theta_in, polarization_0, polarization_1, waveguide_0_df, waveguide_1_df
    )

    # Stage 2: Get normalized spatial emission distribution on outgoing face
    face_distribution = get_face_emission_distribution(
        phi_in, theta_in, polarization_0, polarization_1, waveguide_0_df, waveguide_1_df
    )

    # Stage 3: Get normalized angular emission distribution with polarization
    angular_distribution = get_angular_emission_distribution(
        phi_in, theta_in, polarization_0, polarization_1, far_field_0_df, far_field_1_df
    )

    # Sample an emission point on the outgoing face
    sampled_point, sampled_point_index = sample_emission_point(face_distribution)

    # Sample an emission angle (Theta, Phi) and get index
    sampled_angle, sampled_angle_index = sample_emission_angle(angular_distribution)

    # Find probability of the sampled point
    prob_point = face_distribution.loc[
        (face_distribution["X"] == sampled_point["X"]) &
        (face_distribution["Y"] == sampled_point["Y"]) &
        (face_distribution["Z"] == sampled_point["Z"]),
        "Probability"
    ].values[0]

    # Find probability of the sampled angle and polarization vector
    angle_row = angular_distribution.iloc[sampled_angle_index]
    prob_angle = angle_row["Probability"]
    polarization = angle_row["Polarization"]  # complex np.array([Etheta, Ephi])

    return {
        "S21_squared": S21_squared,
        "sampled_point": sampled_point,
        "point_probability": prob_point,
        "face_distribution": face_distribution,
        "sampled_angle": sampled_angle,
        "angle_probability": prob_angle,
        "angular_distribution": angular_distribution,
        "polarization_at_sampled_angle": polarization,
    }

result = simulate_photon_emission(
    phi_in,
    theta_in,
    polarization_0,
    polarization_1,
    waveguide_0_df,
    waveguide_1_df,
    far_field_0_df,
    far_field_1_df,
)

print("Transmission |S21|^2:", result["S21_squared"])

# Extract sampled point coordinates as floats
x = result["sampled_point"]["X"]
y = result["sampled_point"]["Y"]
z = result["sampled_point"]["Z"]
print(f"Sampled emission point (X, Y, Z): {x:.6f}, {y:.6f}, {z:.6f}")

print("Probability at sampled point:", result["point_probability"])

# Extract sampled angle values as floats
theta = result["sampled_angle"]["Theta"]
phi = result["sampled_angle"]["Phi"]
print(f"Sampled emission angle (Theta, Phi): {theta:.4f}, {phi:.4f}")

print("Probability at sampled angle:", result["angle_probability"])

pol = result["polarization_at_sampled_angle"]
print("Outgoing polarization vector (Etheta, Ephi):")
print(f"Etheta = {pol[0].real:.4f} + {pol[0].imag:.4f}j")
print(f"Ephi   = {pol[1].real:.4f} + {pol[1].imag:.4f}j")

