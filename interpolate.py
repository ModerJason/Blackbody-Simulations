# -*- coding: utf-8 -*-
"""
Created on Fri Aug 15 13:08:44 2025

@author: Jason Wang
"""

import os
import pandas as pd
import numpy as np
from scipy.interpolate import RegularGridInterpolator, griddata
import matplotlib.pyplot as plt

frequencies = [500, 550, 600]
project_name = "InfParallelPlate"
design_name = "interpolate"
repo_root = os.path.dirname(os.path.abspath(__file__))
sim_data_location = os.path.join(repo_root, "HFSSSimData")

def filter_by_variables(predictor_variables, response_variables, df):
    # Keep only predictor + response columns
    df_filtered = df[predictor_variables + response_variables].drop_duplicates()
    
    return df_filtered    

def interpolate_and_save(df, predictor_variables, response_variables, output_folder, resolutions, method='linear', save_to_csv=False):
    """
    Interpolate response(s) over all predictor variables and optionally save as CSV.
    
    Parameters:
        df: pandas DataFrame with predictors + responses
        predictor_variables: list of predictor column names
        response_variables: list of response column names to interpolate
        output_folder: folder to save CSV
        resolutions: dict mapping predictor names to grid resolution (same units as df)
        method: interpolation method ('linear', 'nearest', 'cubic')
        save_to_csv: if True, save the interpolated data to CSV
    """

    # Create 1D arrays for each predictor
    grid_axes = []
    for p in predictor_variables:
        min_val = df[p].min()
        max_val = df[p].max()
        res = resolutions.get(p, 1)
        axis = np.arange(min_val, max_val + res, res)
        grid_axes.append(axis)
        print(f"{p}: {len(axis)} points from {min_val} to {max_val} with resolution {res}")


    # Create N-D grid
    mesh = np.meshgrid(*grid_axes, indexing='ij')
    mesh_points = np.stack([m.ravel() for m in mesh], axis=-1)

    # Interpolate each response
    interpolated_data = {}
    points = df[predictor_variables].values
    for resp in response_variables:
        values = df[resp].values
        grid_values = griddata(points, values, mesh_points, method=method)
        interpolated_data[resp] = grid_values

    # Build DataFrame
    export_dict = {p: mesh_points[:, i] for i, p in enumerate(predictor_variables)}
    for resp in response_variables:
        export_dict[resp] = interpolated_data[resp]

    export_df = pd.DataFrame(export_dict)

    # Save CSV if requested
    if save_to_csv:
        output_csv_path = os.path.join(output_folder, "interpolated.csv")
        export_df.to_csv(output_csv_path, index=False)
        print(f"Saved interpolated data to {output_csv_path}")

    return export_df  # return the interpolated DataFrame for further use

#%%

for freq in frequencies:
    for Ephi in [0, 1]:
        freq_sim_data_location = f"{project_name}_{design_name}_{freq}GHz_Ephi={Ephi}"
        freq_sim_data_folder = os.path.join(sim_data_location, freq_sim_data_location)

        waveguide_path = os.path.join(freq_sim_data_folder, "refined_waveguide.csv")
        far_field_path = os.path.join(freq_sim_data_folder, "refined_far_field.csv")

        if os.path.exists(waveguide_path):
            waveguide_df = pd.read_csv(waveguide_path)
            print(f"Loaded refined_waveguide.csv for freq={freq} GHz, Ephi={Ephi}")
            
            # Add a outgoing power fraction column
            waveguide_df["OutgoingPowerFraction"] = (
                waveguide_df["OutgoingPower"] / waveguide_df["IngoingPower"]
            ).clip(upper=1.0)
        else:
            print(f"Missing file: {waveguide_path}")

        if os.path.exists(far_field_path):
            far_field_df = pd.read_csv(far_field_path)
            print(f"Loaded refined_far_field.csv for freq={freq} GHz, Ephi={Ephi}")
        else:
            print(f"Missing file: {far_field_path}")
        
        S21_predictor_variables = ["IWaveTheta", "IWavePhi"]
        S21_response_variables = ["OutgoingPowerFraction"]
        S21_df = filter_by_variables(S21_predictor_variables, S21_response_variables,
                                               waveguide_df)
        
        exit_E_predictor_variables = ["IWaveTheta", "IWavePhi", "X", "Y", "Z"]
        exit_E_response_variables = ["Ex_real", "Ey_real", "Ez_real", "Ex_imag", "Ey_imag", "Ez_imag"]
        exit_E_df = filter_by_variables(exit_E_predictor_variables, exit_E_response_variables,
                                               waveguide_df)
        
        far_field_E_predictor_variables = ["IWaveTheta", "IWavePhi", "Theta", "Phi"]
        far_field_E_response_variables = ["rEphi_real", "rEphi_imag", "rEtheta_real", "rEtheta_imag"]
        far_field_E_df = filter_by_variables(far_field_E_predictor_variables, far_field_E_response_variables,
                                               far_field_df)
        
        # Dictionary of resolutions
        # resolutions = {
        #     "IWaveTheta": 0.1, # degrees
        #     "IWavePhi": 0.1, # degrees
        #     "X": 1e-5, # meters
        #     "Y": 1e-5, # meters
        #     "Z": 1e-5, # meters
        #     "Theta": 0.1, # degrees
        #     "Phi": 0.1 # degrees
        # }
        
        S21_resolutions = {
            "IWaveTheta": 0.1, # degrees
            "IWavePhi": 0.1, # degrees
        }
        
        S21_interpolated_df = interpolate_and_save(
            df=S21_df,
            predictor_variables=S21_predictor_variables,
            response_variables=S21_response_variables,
            output_folder=freq_sim_data_folder,
            resolutions=S21_resolutions,
            method='linear',
            save_to_csv = True
        )
        
        # THE CODE BELOW TAKES TOO LONG TO RUN. Need a different approach, POD + modal coefficients perhaps
        # exit_E_resolutions = {
        #     "IWaveTheta": 0.1, # degrees
        #     "IWavePhi": 0.1, # degrees
        #     "X": 1e-5, # meters
        #     "Y": 1e-5, # meters
        #     "Z": 1e-5, # meters
        # }
        
        # exit_E_interpolated_df = interpolate_and_save(
        #     df=exit_E_df,
        #     predictor_variables=exit_E_predictor_variables,
        #     response_variables=exit_E_response_variables,
        #     output_folder=freq_sim_data_folder,
        #     resolutions=exit_E_resolutions,
        #     method='linear', 
        #     save_to_csv = False
        # )
        
        # PLOT THE INTERPOLATED S21 GRID
        # plt.figure(figsize=(8,6))
        # plt.imshow(
        #     S21_grid, 
        #     extent=[Theta_grid.min(), Theta_grid.max(), Phi_grid.min(), Phi_grid.max()],
        #     origin='lower',
        #     aspect='auto',
        #     cmap='viridis'
        # )
        # plt.colorbar(label='S21 (OutgoingPowerFraction)')
        # plt.xlabel('IWaveTheta [degrees]')
        # plt.ylabel('IWavePhi [degrees]')
        # plt.title('Interpolated S21 Heatmap')