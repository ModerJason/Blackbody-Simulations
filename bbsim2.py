# -*- coding: utf-8 -*-
"""
Created on 03/20/2025

@author: Jason Wang
"""

# Notes:
# (1) The code expects an existing setup with all analysis and parametric setups deleted,
# or a fresh layout. NOTE: due to an error, it is advisable to recreate (duplicate) the project each time ANSYS is launched,
# and delete all analysis sweeps associated
# (2) The code expects many parameters to be filled in regarding the layout
# (3) If is desired, it is possible to Keyboard Interrupt in the middle of a simulation and save to csv. Then,
# it is possible to begin a new simulation with this existing data. 
# (4) The output .csv files are large, so to use the data, import directly using pandas.

# Usage:
# Steps to use the data. We seek to simulate the photon's trajectory through the waveguide and its probability distribution
# over output angles. We decouple this into 4 phases
# (1) Probability that the photon reaches the output face of the waveguide: given by |S21|, or the ratio of the
# output power to the input power. In the case where numerically the output power is greater than the input power,
# the ratio can be set to 1
# (2) Where the photon is emitted on the output face: given by the electric field at the output face, and the
# probability of emission from each individual location on the output face is proportional to the square of the
# electric field amplitude (MagE) at those discretized points. Note that the sampling in the narrow dimension of the
# waveguide may be sparse; this is because the electric field does not vary along that dimension (in the TEM mode)
# (3) The outgoing k vector for the photon: given by the far field radiation pattern, and the probability of emission
# into each outgoing k vector is proportional to power in the far field, which is proportional to |MagE|^2. |MagE|^2
# can be calculated as |MagE|^2 = (rEphi_real)^2 + (rEphi_imag)^2 + (rEtheta_real)^2 + (rEtheta_imag)^2
# (4) The polarization of the outgoing photon can be found with the far field electric field.
# For the polarization, the unit vector in the θ direction is (cosθcosϕ, cosθsinϕ, -sinθ), and the unit vector in 
# the ϕ direction is (-sinϕ, cosϕ, 0).
# Importantly, the photon will generally be in a linear combination of both polarizations, say a|0>+b|1>. When the photon is
# entering the waveguide, (i) the probability of making it to the output face is a^2*|S21 for |0>| + b^2*|S21 for |1>|
# (ii) The probability of emitting from any particular point is again weighted by a^2 and b^2, i.e.
# a^2*Prob(x,y,z) for |0> + b^2*Prob(x,y,z) for |1> (iii) same thing for the far field radiation pattern

# Packages: pyaedt, scipy, pandas, numpy, seaborn (for plotting)

import sys
import threading

# Make sure to install the package pyaedt!
from ansys.aedt.core.hfss import Hfss
from scipy import constants
import pandas as pd
import numpy as np
import os
import re
import time
from collections import defaultdict
from typing import Optional

# Physical constants
c = constants.c
mu_0 = constants.mu_0

# HFSS project setup (project name and design name)
project_name = "InfParallelPlate"
design_name = "bbsim15"

# The folder to output all output files
repo_root = os.path.dirname(os.path.abspath(__file__))
output_file_location = os.path.join(repo_root, "HFSSSimData")
os.makedirs(output_file_location, exist_ok=True)

# Whether to import waveguide and far_field_data from existing CSV (default is false)
import_from_existing_csv = False
waveguide_data_csv = os.path.join(output_file_location, "InfParallelPlate_bbsim13_500GHz_Ephi=0/waveguide.csv")
far_field_data_csv = os.path.join(output_file_location, "InfParallelPlate_bbsim13_500GHz_Ephi=0/far_field.csv")

# Design and analysis variables
Ei = 1 # strength of incident electric field [V/m]
max_delta_E = 0.02 # HFSS analysis sweep parameter
max_passes = 10 # HFSS analysis sweep parameter
num_cores = 4

# Check face id's of the plane wave ingoing face and outgoing face by clicking select object/by name
ingoing_face_id = 8
outgoing_face_id = 7

# Frequencies in GHz
freq_lower, freq_upper, freq_step = 500, 550, 100

# The following 4 variables refer to sweeps over incident plane wave. These angles are with respect to the global
# coordinate system. Symmetry can be used to make these sweeps less wide
i_theta_lower, i_theta_upper, i_phi_lower, i_phi_upper = 90, 180, 0, 90

# Define x and y directions of outgoing coordinate systems (vectors relative to global coordinate system)
# x direction points outward from face. The z direction is automatic from the right-hand rule.
# It is helpful to redefine a coordinate system so that the theta and phi sweep correspond to sweeps corresponding
# to the two length scales. The 4 angular variables refer to sweeps over the far field radiation, with respect to the user-defined CS
outgoing_face_cs_x = [0, 0, 1]
outgoing_face_cs_y = [0, 1, 0]
rad_theta_lower, rad_theta_upper, rad_phi_lower, rad_phi_upper = 0, 180, -90, 90

# a is the length scale of the dimension coinciding with the sweep over phi
# b is the length scale of the dimension coinciding with the sweep over theta
a = 10 # [um]
b = 0.05 # [um]

# Analysis variables. 1/fineness is the fraction of lambda/a swept over each radiation step in theta and phi
# Maximum coarseness is the maximum coarseness of the angular sweeps, in degrees.
# For small widths (a or b small), phi or theta will have only 1 main lobe with angular width lambda/a very high.
# The general guideline of sampling is 10 points across the narrowest feature, but we take more than 50 points to be safe
# Here, the narrowest feature is 180 degrees, and to be safe we can sample with 360/72=5 degree coarseness.
fineness = 10
minimum_coarseness = 0.1 # degrees
maximum_coarseness = 5 # degrees

# Collect radiation parameters into one variable for cleanness
rad_params = rad_theta_lower, rad_theta_upper, rad_phi_lower, rad_phi_upper, a, b, fineness, maximum_coarseness, minimum_coarseness

# Adaptive or discrete sweep. For adaptive sweep, max difference is maximum fractional difference allowed between
# any two points in the sweep, relative to the total maximum value of the outgoing power.
sweep = "adaptive"
max_difference = 0.015

# Initial step size over theta and phi (adaptive), or step size over theta and phi (discrete)
i_theta_step = 0.8
i_phi_step = 0.8

hfss = Hfss(project=project_name, design=design_name, non_graphical=False)
oDesktop = hfss.odesktop
oProject = oDesktop.SetActiveProject(project_name)
oDesign = oProject.SetActiveDesign(design_name)
oEditor = oDesign.SetActiveEditor("3D Modeler")
oEditor.SetModelUnits(
    [
        "NAME:Units Parameter",
        "Units:=", "mm",
        "Rescale:="	, True,
        "Max Model Extent:="	, 10000
    ])
# Get HFSS modules that we will use throughout
oModuleSolution = oDesign.GetModule("Solutions")
oModuleAnalysis = oDesign.GetModule("AnalysisSetup")
oModuleParametric = oDesign.GetModule("Optimetrics")
oModuleBoundary = oDesign.GetModule("BoundarySetup")
oModuleFields = oDesign.GetModule("FieldsReporter")
oModuleRad = oDesign.GetModule("RadField")

#%%

# Initialize the design variables for the project
def initialize_variables(Ei):
    hfss["Ephi"] = 0
    hfss.variable_manager.set_variable("Ei", str(Ei), sweep=False)

# Get ingoing and outgoing faces and set up radiation boundaries for ingoing and outgoing faces
# Radiation boundaries ensure that the faces act as open space / into vacuum
def get_faces_from_face_id(ingoing_face_id, outgoing_face_id):
    try:
        outgoing_face = hfss.modeler.get_face_by_id(outgoing_face_id)
        hfss.modeler.create_face_list([outgoing_face], name="outgoing")
    except Exception as e:
        raise RuntimeError(f"Error retrieving 'outgoing' facelist: {e}")

    try:
        plane_wave_face = hfss.modeler.get_face_by_id(ingoing_face_id)
    except Exception as e:
        raise RuntimeError(f"Error retrieving 'ingoing' facelist: {e}")
    hfss.assign_radiation_boundary_to_faces(assignment=[plane_wave_face.id], name="rbin")
    hfss.assign_radiation_boundary_to_faces(assignment=[outgoing_face.id], name="rbout")
    return plane_wave_face, outgoing_face

# Create coordinate system for output face
def create_local_coordinate_system(outgoing_face, outgoing_face_cs_x, outgoing_face_cs_y):
    hfss.modeler.create_coordinate_system(
        origin=outgoing_face.center,
        reference_cs="Global",
        name="outgoing_cs",
        mode="axis",
        x_pointing=outgoing_face_cs_x,
        y_pointing=outgoing_face_cs_y
    )

# Add the expression for outgoing power to the fields calculator in HFSS
def add_outgoing_power_to_calculator():
    # Add calculated expression for the outgoing power through the output rectangle
    # Note that although it should not be the case, in some cases numerically the power exiting the waveguide can be
    # larger than the power entering, for example for TEM transmission. If run long enough, the HFSS simulation
    # converges to a power ratio of 1
    oModuleFields.ClearAllNamedExpr()
    oModuleFields.CalcStack("Clear")
    oModuleFields.CopyNamedExprToStack("Vector_RealPoynting")
    oModuleFields.CalcOp("Mag")
    oModuleFields.EnterSurf("outgoing")
    oModuleFields.CalcOp("Integrate")
    oModuleFields.AddNamedExpression("outgoing_power", "Fields")

# Based on the ranges of the angles and the step sizes, calculate the number of points in the sweep in each angular direction
def get_incoming_phi_theta_num(theta_lower, theta_upper, phi_lower, phi_upper, theta_step, phi_step):
    theta_range = theta_upper - theta_lower
    phi_range = phi_upper - phi_lower
    if theta_range == 0:
        theta_num = 1
    else:
        theta_num = int(np.ceil(theta_range / theta_step)) + 1
    if phi_range == 0:
        phi_num = 1
    else:
        phi_num = int(np.ceil(phi_range / phi_step)) + 1
    return theta_num, phi_num

# Get the desired number of points swept in theta and phi
def get_radiation_phi_theta_num(theta_lower, theta_upper, phi_lower, phi_upper, a, b, freq, fineness, maximum_coarseness,
                                minimum_coarseness):
    theta_range = theta_upper - theta_lower
    phi_range = phi_upper - phi_lower
    # Find wavelength and steps (using appropriate unit conversions)
    wavelength = c / (freq * 1e9)
    # Sweep resolution is at least maximum_coarseness degrees in theta and phi
    theta_step = np.minimum(maximum_coarseness, theta_range / np.pi * wavelength / (fineness * (b * 1e-3)))
    theta_step = np.maximum(minimum_coarseness, theta_step)
    phi_step = np.minimum(maximum_coarseness, phi_range / np.pi * wavelength / (fineness * (a * 1e-3)))
    phi_step = np.maximum(minimum_coarseness, phi_step)
    theta_num = int(np.ceil(theta_range / theta_step)) + 1
    phi_num = int(np.ceil(phi_range / phi_step)) + 1
    return theta_step, phi_step, theta_num, phi_num

# Helper methods for naming conventions
def get_plane_wave_name(frequency):
    return f"plane_wave_{frequency}GHz"
def get_radiation_sphere_name(frequency):
    return f"radiation_sphere_{frequency}GHz"
def get_setup_name(frequency):
    return f"{frequency}GHz"
def get_parametric_setup_name(frequency):
    return f"E_phi_sweep_{frequency}GHz"

# Set up radiation boundaries and far field infinite sphere
def setup_radiation(rad_params, frequency):
    rad_theta_lower, rad_theta_upper, rad_phi_lower, rad_phi_upper, a, b, fineness, maximum_coarseness, minimum_coarseness = rad_params
    # Create far field infinite radiation sphere
    rad_theta_step, rad_phi_step, rad_theta_num, rad_phi_num = get_radiation_phi_theta_num(rad_theta_lower,
                                                                                           rad_theta_upper, rad_phi_lower, rad_phi_upper, a, b, frequency, fineness, maximum_coarseness, minimum_coarseness)
    sphere_name = get_radiation_sphere_name(frequency)

    hfss.insert_infinite_sphere(
        definition="Theta-Phi",
        x_start=rad_theta_lower, x_stop=rad_theta_upper, x_step=rad_theta_step,
        y_start=rad_phi_lower, y_stop=rad_phi_upper, y_step=rad_phi_step,
        custom_radiation_faces="outgoing",
        custom_coordinate_system="outgoing_cs",
        name=sphere_name
    )

# Setup plane wave
def setup_plane_wave(plane_wave_face, i_theta_lower, i_theta_upper, i_phi_lower, i_phi_upper,
                     frequency, i_theta_step, i_phi_step, refine=False):

    # Creation of plane wave is relative to Global CS (HFSS doesn't support relative CS)
    i_theta_num, i_phi_num = get_incoming_phi_theta_num(i_theta_lower, i_theta_upper, i_phi_lower,
                                                                                  i_phi_upper, i_theta_step, i_phi_step)
    plane_wave_name = get_plane_wave_name(frequency)
    if refine:
        plane_wave_name += "_refine"
    #For the polarization, the unit vector in the θ direction is (cosθcosϕ, cosθsinϕ, -sinθ), and the unit vector in 
    # the ϕ direction is (-sinϕ, cosϕ, 0).    
    hfss.plane_wave(
        assignment=[plane_wave_face],
        vector_format="Spherical",
        origin=plane_wave_face.center,
        polarization=["Ephi", "1-Ephi"],
        propagation_vector=[
            [f"{i_phi_lower}deg", f"{i_phi_upper}deg", i_phi_num],
            [f"{i_theta_lower}deg", f"{i_theta_upper}deg", i_theta_num]
        ],
        name=plane_wave_name
    )

# Set up incident plane wave and radiation, sweep over frequencies, and parametric sweep over E-field polarizations
def create_setup(frequency, max_delta_E, max_passes, previous_frequency):

    freq_str = f"{frequency}GHz"
    setup_name = get_setup_name(frequency)

    # Case 1: first frequency — full adaptive mesh
    if previous_frequency is None:
        oModuleAnalysis.InsertSetup("HfssDriven", [
            f"NAME:{setup_name}",
            "Frequency:=", freq_str,
            "MaxDeltaE:=", max_delta_E,
            "MaximumPasses:=", max_passes
        ])
    # Case 2: Reuse mesh from previous frequency
    else:
        previous_setup_name = get_setup_name(previous_frequency)
        try:
            oModuleAnalysis.InsertSetup("HfssDriven", [
                f"NAME:{setup_name}",
                "Frequency:=", freq_str,
                "MaxDeltaE:=", max_delta_E,
                "MaximumPasses:=", max_passes,
                [
                    "NAME:MeshLink",
                    "ImportMesh:=", True,
                    "Project:=", "This Project*",
                    "Product:=", "HFSS",
                    "Design:=", "This Design*",
                    "Soln:=", f"{previous_setup_name} : LastAdaptive",
                    [
                        "NAME:Params",
                        "Ephi:=", "Ephi"
                    ],
                    "ForceSourceToSolve:=", False,
                    "PreservePartnerSoln:=", False,
                    "PathRelativeTo:=", "TargetProject",
                    "ApplyMeshOp:=", False
                ]
            ])
            print(f"Setup {setup_name} created from {previous_setup_name} mesh")
        except Exception:
            oModuleAnalysis.InsertSetup("HfssDriven", [
                f"NAME:{setup_name}",
                "Frequency:=", freq_str,
                "MaxDeltaE:=", max_delta_E,
                "MaximumPasses:=", max_passes
            ])
            f"Setup {setup_name} created with default mesh"

    # Insert a parametric sweep over the two electric field polarizations
    parametric_setup_name = get_parametric_setup_name(frequency)
    oModuleParametric.InsertSetup("OptiParametric",
                                  [
                                      f"NAME:{parametric_setup_name}",
                                      "IsEnabled:=", True,
                                      [
                                          "NAME:ProdOptiSetupDataV2",
                                          "SaveFields:=", True,
                                      ],
                                      "Sim. Setups:=", [setup_name],
                                      [
                                          "NAME:Sweeps",
                                          [
                                              "NAME:SweepDefinition",
                                              "Variable:=", "Ephi",
                                              "Data:=", "LIN 0 1 1",
                                          ]
                                      ],
                                  ])

# Run analysis, including adaptive analysis if appropriate.
def run_analysis(num_cores, max_delta_E, max_passes, plane_wave_face, Ei, output_file_location, i_theta_lower, i_theta_upper, i_phi_lower, i_phi_upper,
                 frequency, i_theta_step, i_phi_step, rad_params, sweep = "adaptive", max_difference = 0.02, import_from_existing_csv = False, E_phi = None,
                 waveguide_data_csv = None, far_field_data_csv = None):

    if sweep == "discrete":
        print(f"Discrete sweep has begun for frequency {frequency}GHz")
    elif sweep == "adaptive":
        print(f"Adaptive sweep has begun for frequency {frequency}GHz")
    elif sweep == "zoom":
        print(f"Zoom sweep has begun for frequency {frequency}GHz")

    # Declare waveguide_data and far_field_data variables
    waveguide_data_0 = waveguide_data_1 = None
    far_field_data_0 = far_field_data_1 = None
    waveguide_data_from_csv = far_field_data_from_csv = None

    if not import_from_existing_csv:
        print("Proceeding with default setup")
        setup_plane_wave(
            plane_wave_face, i_theta_lower, i_theta_upper, i_phi_lower, i_phi_upper,
            frequency, i_theta_step, i_phi_step
        )

        # Change from default scattered fields to total fields formulation
        oModuleSolution.EditSources(["FieldType:=", "TotalFields"])

        hfss.analyze(cores=num_cores)

        rad_theta_lower, rad_theta_upper, rad_phi_lower, rad_phi_upper, a, b, fineness, maximum_coarseness, minimum_coarseness = rad_params
        rad_theta_step, rad_phi_step, rad_theta_num, rad_phi_num = get_radiation_phi_theta_num(rad_theta_lower, rad_theta_upper,
            rad_phi_lower, rad_phi_upper, a, b, frequency, fineness, maximum_coarseness, minimum_coarseness)

        # Extract initial waveguide and far field data (final for discrete sweep, to be adjusted for adaptive sweep)
        waveguide_data_0 = extract_waveguide_data(frequency, Ei, plane_wave_face, i_theta_lower, i_theta_upper,
            i_phi_lower, i_phi_upper, i_theta_step, i_phi_step, output_file_location, 0, csv=True, refine=False)
        far_field_data_0 = extract_far_field_data(frequency, i_theta_lower, i_theta_upper, i_phi_lower, i_phi_upper,
            i_theta_step, i_phi_step, rad_theta_step, rad_phi_step, output_file_location, 0, csv=True, refine=False)
        waveguide_data_1 = extract_waveguide_data(frequency, Ei, plane_wave_face, i_theta_lower, i_theta_upper,
            i_phi_lower, i_phi_upper, i_theta_step, i_phi_step, output_file_location, 1, csv=True, refine=False)
        far_field_data_1 = extract_far_field_data(frequency, i_theta_lower, i_theta_upper, i_phi_lower, i_phi_upper,
            i_theta_step, i_phi_step, rad_theta_step, rad_phi_step, output_file_location, 1, csv=True, refine=False)
    else:
        print("Reading in waveguide data and far-field data from existing csv files")
        waveguide_data_from_csv = pd.read_csv(waveguide_data_csv)
        far_field_data_from_csv = pd.read_csv(far_field_data_csv)

    if sweep == "adaptive" or sweep == "zoom":
        # Disable analysis and parametric setup before adaptive sweep begins, and insert new refinement setup.
        # We do not delete the setups because we will use them for better meshing for future frequencies
        parametric_setup_name = get_parametric_setup_name(frequency)
        try:
            oModuleParametric.EnableSetup(parametric_setup_name, False)
        except Exception:
            pass
        setup_name = get_setup_name(frequency)
        try:
            oModuleAnalysis.EditSetup(setup_name,
                                  [
                                      f"NAME:{setup_name}",
                                      "IsEnabled:=", False,
                                  ])
        except Exception:
            pass

        Ephi_values = [E_phi] if E_phi is not None else [0, 1]
        # Sweep over the polarizations
        for Ephi in Ephi_values:
            # Set the Ephi variable to the desired polarization
            hfss["Ephi"] = Ephi
            print(f"Beginning refinement process for frequency {frequency}GHz, Ephi={Ephi}")

            # Select correct dataset for the polarization
            if not import_from_existing_csv:
                waveguide_data = waveguide_data_0 if Ephi == 0 else waveguide_data_1
                far_field_data = far_field_data_0 if Ephi == 0 else far_field_data_1
            else:
                waveguide_data = waveguide_data_from_csv
                far_field_data = far_field_data_from_csv

            if sweep == "adaptive":
                # Repeat until converge: (1) find a new region to refine (2) create a plane wave to refine this region
                # and analyze
                incoming_power = get_incoming_power(Ei, plane_wave_face)
                converged = False
                while not converged:
                    result = find_regions_to_refine(incoming_power, waveguide_data, max_difference)
                    if result is not False:
                        refine_regions = result
                        waveguide_data, far_field_data = run_refined_plane_wave(plane_wave_face, frequency, num_cores, max_delta_E, max_passes, Ei, output_file_location,
                        Ephi, waveguide_data, far_field_data, rad_params, refine_regions, csv=False)
                    else:
                        print("Adaptive sweep has converged")
                        converged = True

                # Output two csv files at the end of the sweep for each polarization for each frequency
                print(f"Exporting waveguide and far field data to csv for frequency {frequency}GHz, Ephi={Ephi}")
                folder_name = f"{project_name}_{design_name}_{frequency}GHz_Ephi={E_phi}"
                folder_path = os.path.join(output_file_location, folder_name)
                os.makedirs(folder_path, exist_ok=True)

                waveguide_output_filename = "refined_waveguide.csv"
                full_path = os.path.join(folder_path, waveguide_output_filename)
                waveguide_data.to_csv(full_path, index=False)
                
                far_field_output_filename = "refined_far_field.csv"
                full_path = os.path.join(folder_path, far_field_output_filename)
                far_field_data.to_csv(full_path, index=False)
                
                print(f"Waveguide and far field data to csv for frequency {frequency}GHz, Ephi={Ephi} exported")
            if sweep == "zoom":
                print("Temporary filler")

# Runs refined plane wave sweeps for each region in the list of regions refine_regions. The function
# also updates the dataframes.
def run_refined_plane_wave(plane_wave_face, frequency, num_cores, max_delta_E, max_passes, Ei, output_file_location, Ephi, waveguide_data, far_field_data, rad_params,
    refine_regions, csv=False):

    rad_theta_lower, rad_theta_upper, rad_phi_lower, rad_phi_upper, a, b, fineness, maximum_coarseness, minimum_coarseness = rad_params
    rad_theta_step, rad_phi_step, rad_theta_num, rad_phi_num = get_radiation_phi_theta_num(rad_theta_lower, rad_theta_upper,
        rad_phi_lower, rad_phi_upper, a, b, frequency, fineness, maximum_coarseness, minimum_coarseness)
    key_cols = ["IWavePhi", "IWaveTheta"]

    def merge_fast(existing_df, new_dfs, name, merge_csv=False):
        print(f"[{name}] Starting merge of {len(new_dfs)} chunks...")
        t0 = time.perf_counter()

        new_df = pd.concat(new_dfs, ignore_index=True)
        existing_idx = existing_df.set_index(key_cols, drop=False)
        new_keys = new_df.set_index(key_cols, drop=False).index.unique()
        filtered_existing = existing_idx.drop(new_keys, errors="ignore").reset_index(drop=True)

        combined = pd.concat([filtered_existing, new_df], ignore_index=True)

        if merge_csv:
            print("Updating CSV file following this set of refining")
            folder_name = f"{project_name}_{design_name}_{frequency}GHz_Ephi={Ephi}"
            folder_path = os.path.join(output_file_location, folder_name)
            os.makedirs(folder_path, exist_ok=True)

            output_filename = f"refined_{name}.csv"
            full_path = os.path.join(folder_path, output_filename)
            combined.to_csv(full_path, index=False)
            
        elapsed = time.perf_counter() - t0
        print(f"[{name}] Merge took {elapsed:.3f}s")
        return combined

    print(f"Running {len(refine_regions)} refined plane wave regions at {frequency} GHz")

    waveguide_chunks = []
    far_field_chunks = []

    try:
        for idx, region in enumerate(refine_regions):
            freq_str = f"{frequency}GHz"

            print(f"\n--- Refining region {idx + 1} of {len(refine_regions)} ---")
            plane_phi_lower, plane_phi_upper = region["phi_range"]
            plane_theta_lower, plane_theta_upper = region["theta_range"]
            plane_phi_step = region["phi_step"]
            plane_theta_step = region["theta_step"]

            oModuleBoundary.DeleteAllExcitations()
            # Insert a new setup for refinement
            setup_name = get_setup_name(frequency) + "_refine"
            oModuleAnalysis.InsertSetup("HfssDriven", [
                f"NAME:{setup_name}",
                "Frequency:=", freq_str,
                "MaxDeltaE:=", max_delta_E,
                "MaximumPasses:=", max_passes,
            ])

            setup_plane_wave(
                plane_wave_face,
                plane_theta_lower, plane_theta_upper,
                plane_phi_lower, plane_phi_upper,
                frequency, plane_theta_step, plane_phi_step,
                refine=True
            )

            # Change from default scattered fields formulation to total fields formulation
            oModuleSolution.EditSources(["FieldType:=", "TotalFields"])
            hfss.analyze(cores=num_cores)

            waveguide_chunks.append(
                extract_waveguide_data(
                    frequency, Ei, plane_wave_face,
                    plane_theta_lower, plane_theta_upper,
                    plane_phi_lower, plane_phi_upper,
                    plane_theta_step, plane_phi_step,
                    output_file_location, Ephi, csv=False, refine=True
                )
            )

            far_field_chunks.append(
                extract_far_field_data(
                    frequency, plane_theta_lower, plane_theta_upper,
                    plane_phi_lower, plane_phi_upper,
                    plane_theta_step, plane_phi_step, rad_theta_step, rad_phi_step,
                    output_file_location, Ephi, csv=False, refine=True
                )
            )
            # Delete the analysis setup after refining each region
            oModuleAnalysis.DeleteSetups([setup_name])


    # Add safe exits for keyboard interruptions (just in case the user desires to export the CSV early)
    except KeyboardInterrupt:
        print(f"\n[!] User interrupted refining process")

        user_input = input(
            "Type 'csv' to save partial results to file, or press Enter to skip saving: ").strip().lower()
        if user_input == "csv":
            print("Saving last updated results to a CSV")
            csv = True
        else:
            csv = False

        waveguide_data = merge_fast(waveguide_data, waveguide_chunks, "waveguide", merge_csv=csv)
        far_field_data = merge_fast(far_field_data, far_field_chunks, "far_field", merge_csv=csv)
        if csv:
            print("Latest csv files have been successfully saved")
        sys.exit(1)

    # If no interruption, continue here
    waveguide_data = merge_fast(waveguide_data, waveguide_chunks, "waveguide", merge_csv=csv)
    far_field_data = merge_fast(far_field_data, far_field_chunks, "far_field", merge_csv=csv)
    return waveguide_data, far_field_data

# Takes in a pandas DF for the waveguide data, consisting of the outgoing power at the exit. Returns
# a dictionary of intervals that require additional sampling, merged when appropriate and ordered
# by percent power difference compared to the incoming power
def find_regions_to_refine(incoming_power, waveguide_data, max_difference):
    max_frac_diff_seen = 0.0

    print("Finding regions to refine!")

    # Focus only on unique angular combinations
    key_cols = ["IWaveTheta", "IWavePhi"]
    unique_points = waveguide_data.drop_duplicates(subset=key_cols)

    # Pivot data to form a 2D power grid: rows = theta, columns = phi
    pivot = unique_points.pivot(index="IWaveTheta", columns="IWavePhi", values="OutgoingPower")

    theta_vals = pivot.index.values
    phi_vals = pivot.columns.values
    power_grid = pivot.values
    max_power = incoming_power

    print(f"Theta values ({len(theta_vals)}): {theta_vals}")
    print(f"Phi values ({len(phi_vals)}): {phi_vals}")
    print(f"OutgoingPower max: {max_power}")

    refine_regions = []

    # Loop over each point in the grid to find high-contrast neighbors
    for i in range(power_grid.shape[0]):
        for j in range(power_grid.shape[1]):
            current_power = power_grid[i, j]
            if np.isnan(current_power):
                continue

            # Check neighbor in +φ direction
            for jj in range(j + 1, power_grid.shape[1]):
                neighbor_power = power_grid[i, jj]
                if not np.isnan(neighbor_power):
                    frac_diff = abs(current_power - neighbor_power) / max_power
                    max_frac_diff_seen = max(max_frac_diff_seen, frac_diff)

                    if frac_diff > max_difference:
                        phi1, phi2 = phi_vals[j], phi_vals[jj]
                        theta1 = theta_vals[i]

                        # Define φ-refinement region at fixed θ
                        plane_phi_lower = phi1
                        plane_phi_upper = phi2
                        plane_theta_lower = theta1
                        plane_theta_upper = theta1

                        # Break interval up until 10 distinct pieces (10 times finer)
                        plane_phi_num = 11
                        plane_phi_step = (plane_phi_upper - plane_phi_lower) / (plane_phi_num - 1) if plane_phi_num > 1 else 0.0
                        plane_theta_step = 0.0

                        refine_regions.append({
                            "phi_range": (plane_phi_lower, plane_phi_upper),
                            "theta_range": (plane_theta_lower, plane_theta_upper),
                            "phi_step": plane_phi_step,
                            "theta_step": plane_theta_step,
                            "frac_diff": frac_diff,
                            "pair": ((i, j), (i, jj))
                        })
                    break  # Only check immediate next valid neighbor

            # Check neighbor in +θ direction
            for ii in range(i + 1, power_grid.shape[0]):
                neighbor_power = power_grid[ii, j]
                if not np.isnan(neighbor_power):
                    frac_diff = abs(current_power - neighbor_power) / max_power
                    max_frac_diff_seen = max(max_frac_diff_seen, frac_diff)

                    if frac_diff > max_difference:
                        theta1, theta2 = theta_vals[i], theta_vals[ii]
                        phi1 = phi_vals[j]

                        # Define θ-refinement region at fixed φ
                        plane_phi_lower = phi1
                        plane_phi_upper = phi1
                        plane_theta_lower = theta1
                        plane_theta_upper = theta2

                        # Break interval up until 10 distinct pieces (10 times finer)
                        plane_theta_num = 11
                        plane_theta_step = (plane_theta_upper - plane_theta_lower) / (plane_theta_num - 1) if plane_theta_num > 1 else 0.0
                        plane_phi_step = 0.0

                        refine_regions.append({
                            "phi_range": (plane_phi_lower, plane_phi_upper),
                            "theta_range": (plane_theta_lower, plane_theta_upper),
                            "phi_step": plane_phi_step,
                            "theta_step": plane_theta_step,
                            "frac_diff": frac_diff,
                            "pair": ((i, j), (ii, j))
                        })
                    break  # Only check immediate next valid neighbor

    if not refine_regions:
        print("No regions found that exceed the fractional difference threshold.")
        print(f"Maximum fractional difference seen: {max_frac_diff_seen:.5f}")
        return False

    # Sort regions by strength of power contrast (descending)
    refine_regions.sort(key=lambda x: x["frac_diff"], reverse=True)

    print(f"Found {len(refine_regions)} regions to refine.")
    for region in refine_regions:
        (phi_low, phi_high) = region["phi_range"]
        (theta_low, theta_high) = region["theta_range"]
        print(f"φ∈[{phi_low:.3f}, {phi_high:.3f}] θ∈[{theta_low:.3f}, {theta_high:.3f}] "
              f"φ_step={region['phi_step']:.3f}, θ_step={region['theta_step']:.3f}, "
              f"Δ={region['frac_diff']:.5f}")

    # Merge regions along each axis
    def merge_contiguous_regions_by_group(regions, orthogonal_range_key, step_key, merge_range_key):
        if not regions:
            print("No regions to merge.")
            return []

        print(f"\nStarting merge on {len(regions)} regions grouped by '{orthogonal_range_key}'")

        # Group by orthogonal axis (e.g. θ constant for φ merge)
        groups = defaultdict(list)
        for r in regions:
            groups[r[orthogonal_range_key]].append(r)

        merged_all = []

        for ortho_val, group in sorted(groups.items()):
            print(f"\nMerging group with {orthogonal_range_key} = {ortho_val} ({len(group)} regions)")

            group_sorted = sorted(group, key=lambda reg: reg[merge_range_key][0])
            current = group_sorted[0].copy()

            for k, next_region in enumerate(group_sorted[1:], start=1):
                curr_start, curr_end = current[merge_range_key]
                next_start, next_end = next_region[merge_range_key]

                adjacent = (curr_end == next_start)
                same_step = (current[step_key] == next_region[step_key])

                if adjacent and same_step:
                    current[merge_range_key] = (curr_start, next_end)
                    current["frac_diff"] = max(current["frac_diff"], next_region["frac_diff"])
                else:
                    merged_all.append(current)
                    current = next_region.copy()

            merged_all.append(current)

        print(f"\nFinished merging. Total merged regions: {len(merged_all)}")
        return merged_all

    # Split regions by sweep direction: φ-sweeps (fixed θ) vs θ-sweeps (fixed φ)
    phi_sweep = [r for r in refine_regions if r["theta_range"][0] == r["theta_range"][1]]
    theta_sweep = [r for r in refine_regions if r["phi_range"][0] == r["phi_range"][1]]

    # Merge contiguous φ-sweep regions at constant θ
    merged_phi = merge_contiguous_regions_by_group(
        phi_sweep, orthogonal_range_key="theta_range",
        step_key="phi_step", merge_range_key="phi_range"
    )

    # Merge contiguous θ-sweep regions at constant φ
    merged_theta = merge_contiguous_regions_by_group(
        theta_sweep, orthogonal_range_key="phi_range",
        step_key="theta_step", merge_range_key="theta_range"
    )

    # Combine all merged regions and sort by importance
    merged = merged_phi + merged_theta
    merged.sort(key=lambda r: r["frac_diff"], reverse=True)

    print(f"\nMerged into {len(merged)} refined regions.")
    for region in merged:
        (phi_low, phi_high) = region["phi_range"]
        (theta_low, theta_high) = region["theta_range"]
        print(f"φ∈[{phi_low:.3f}, {phi_high:.3f}] θ∈[{theta_low:.3f}, {theta_high:.3f}] "
              f"φ_step={region['phi_step']:.3f}, θ_step={region['theta_step']:.3f}, "
              f"Δ={region['frac_diff']:.5f}")

    return merged

# Calculate incoming power
def get_incoming_power(Ei, plane_wave_face):
    # Unit conversion from mm^2 to m^2
    return 1e-6 *  hfss.modeler.get_face_area(plane_wave_face.id) * Ei ** 2 / (2 * c * mu_0)

# Create Pandas DF of (1) of the output power swept as a function of angles and electric field
# polarization and (2) electric field magnitude at output. Added timing for debugging purposes. Export time depends
# on both resolution and size of the waveguide.
def extract_waveguide_data(frequency, Ei, plane_wave_face, i_theta_lower, i_theta_upper, i_phi_lower, i_phi_upper,
                           i_theta_step, i_phi_step, output_file_location, E_phi, csv=True, refine=False):

    freq_str = f"{frequency}GHz"
    setup_name = get_setup_name(frequency) + ("_refine : LastAdaptive" if refine else " : LastAdaptive")
    exit_field_path = os.path.join(output_file_location, f"{project_name}_exitfield_{freq_str}.fld")

    i_theta_num, i_phi_num = get_incoming_phi_theta_num(
        i_theta_lower, i_theta_upper, i_phi_lower, i_phi_upper, i_theta_step, i_phi_step)
    i_phi_values = np.linspace(i_phi_lower, i_phi_upper, i_phi_num)
    i_theta_values = np.linspace(i_theta_lower, i_theta_upper, i_theta_num)
    total = len(i_phi_values) * len(i_theta_values)

    power_data = {}
    all_data = []
    count = 0
    start_time = time.perf_counter()

    print(f"Extracting waveguide data for frequency {freq_str}" + (f" and Ephi = {E_phi}" if E_phi is not None else ""))

    # First Pass: Outgoing Power
    print("Starting first pass: outgoing power")
    t1 = time.perf_counter()
    oModuleFields.CalcStack("clear")
    oModuleFields.CopyNamedExprToStack("outgoing_power")
    for i_phi in i_phi_values:
        for i_theta in i_theta_values:
            i_phi_str = f"{i_phi}deg"
            i_theta_str = f"{i_theta}deg"
            t_power = time.perf_counter()
            oModuleFields.ClcEval(setup_name,
                ["Ephi:=", E_phi, "Freq:=", freq_str, "IWavePhi:=", i_phi_str, "IWaveTheta:=", i_theta_str], "Fields")
            result = oModuleFields.GetTopEntryValue(setup_name,
                ["Ephi:=", E_phi, "Freq:=", freq_str, "IWavePhi:=", i_phi_str, "IWaveTheta:=", i_theta_str])
            oModuleFields.CalcStack("pop")
            power_data[(E_phi, i_phi, i_theta)] = float(result[0])
            count += 1
            print(f"[w1] [{count}/{total}] φ={i_phi}deg, θ={i_theta}deg, Ephi={E_phi}, {time.perf_counter() - t_power:.3f}s")

    print(f"First pass time: {(time.perf_counter() - t1):.3f}s")

    count = 0
    # Second Pass: Field Export + Read
    print("Starting second pass: waveguide exit field export")
    t2 = time.perf_counter()
    oModuleFields.CalcStack("clear")
    oModuleFields.CopyNamedExprToStack("Mag_E")
    oModuleFields.EnterSurf("outgoing")
    oModuleFields.CalcOp("Value")

    for i_phi in i_phi_values:
        for i_theta in i_theta_values:
            i_phi_str = f"{i_phi}deg"
            i_theta_str = f"{i_theta}deg"

            t_export = time.perf_counter()
            oModuleFields.CalculatorWrite(exit_field_path,
                ["Solution:=", setup_name],
                ["Ephi:=", E_phi, "Freq:=", freq_str, "IWavePhi:=", i_phi_str, "IWaveTheta:=", i_theta_str])
            print(f"Exported field in {time.perf_counter() - t_export:.3f}s")

            t_read = time.perf_counter()
            df = read_hfss_field(exit_field_path)
            print(f"Read field in {time.perf_counter() - t_read:.3f}s")

            df["Freq"] = freq_str
            df["Ephi"] = E_phi
            df["IWavePhi"] = i_phi
            df["IWaveTheta"] = i_theta
            df["OutgoingPower"] = power_data[(E_phi, i_phi, i_theta)]
            all_data.append(df)

            count += 1
            print(f"[w2] [{count}/{total}] φ={i_phi}deg, θ={i_theta}deg, Ephi={E_phi}, {time.perf_counter() - t_export:.3f}s")

    print(f"Second pass time: {(time.perf_counter() - t2):.3f}s")

    # Concatenate everything
    waveguide_df = pd.concat(all_data, ignore_index=True)
    waveguide_df["IngoingPower"] = get_incoming_power(Ei, plane_wave_face)

    priority_cols = ["Freq", "Ephi", "IWavePhi", "IWaveTheta", "OutgoingPower", "IngoingPower"]
    waveguide_df = waveguide_df[[*priority_cols, *[col for col in waveguide_df.columns if col not in priority_cols]]]

    if csv:
        folder_name = f"{project_name}_{design_name}_{frequency}GHz_Ephi={E_phi}"
        folder_path = os.path.join(output_file_location, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        waveguide_filename = "waveguide.csv"
        waveguide_df.to_csv(os.path.join(folder_path, waveguide_filename), index=False)

    print(f"[{freq_str}] Total time: {(time.perf_counter() - start_time) / 60:.3f} min")
    return waveguide_df

# Create Pandas DF of far field total electric field as a function of theta and phi. Note that this field is not normalized
# and has unit of V, not V/m, which is not very important because we care only about ratios.
# Added timing for debugging purposes. Export time depends on both resolution and size of the waveguide
def extract_far_field_data(frequency, i_theta_lower, i_theta_upper, i_phi_lower, i_phi_upper,
                           i_theta_step, i_phi_step, rad_theta_step, rad_phi_step, output_file_location, E_phi, csv=True, refine=False):

    freq_str = f"{frequency}GHz"
    setup_name = get_setup_name(frequency) + ("_refine : LastAdaptive" if refine else " : LastAdaptive")
    sphere_name = get_radiation_sphere_name(frequency)
    far_field_path = os.path.join(output_file_location, f"{project_name}_farfield_{freq_str}.ffd")

    i_theta_num, i_phi_num = get_incoming_phi_theta_num(
        i_theta_lower, i_theta_upper, i_phi_lower, i_phi_upper, i_theta_step, i_phi_step
    )
    i_phi_values = np.linspace(i_phi_lower, i_phi_upper, i_phi_num)
    i_theta_values = np.linspace(i_theta_lower, i_theta_upper, i_theta_num)
    total = len(i_phi_values) * len(i_theta_values)
    all_data = []
    count = 0
    start_time = time.perf_counter()

    print(f"Extracting far field data for frequency {freq_str}" + (f" and Ephi = {E_phi}" if E_phi is not None else ""))

    # Internal helper function to extract far field data
    def evaluate_fields_and_export(Ephi, i_phi, i_theta):
        i_phi_str = f"{i_phi}deg"
        i_theta_str = f"{i_theta}deg"
        t_export_start = time.perf_counter()

        # We add timing here for debugging purposes, monitored via threading
        file_state: dict[str, Optional[float]] = {"created": None, "written": None}

        if os.path.exists(far_field_path):
            os.remove(far_field_path)

        def monitor_file():
            # Wait for file creation
            while not os.path.exists(far_field_path):
                time.sleep(0.001)
            file_state["created"] = time.perf_counter()
            print(f"File created at {file_state['created'] - t_export_start:.3f} s")

            stable_count = 0
            last_size = -1
            started_writing = False
            sleep_interval = 0.005  # 5 ms

            while stable_count < 2:
                time.sleep(sleep_interval)
                size = os.path.getsize(far_field_path)

                if size > 0 and not started_writing:
                    file_state["started_writing"] = time.perf_counter()
                    print(
                        f"File started writing at {file_state['started_writing'] - t_export_start:.3f} s, size: {size} bytes")
                    started_writing = True

                # Only consider stable if size > 0 and unchanged
                if size == last_size and size > 0:
                    stable_count += 1
                else:
                    stable_count = 0
                    last_size = size

            file_state["written"] = time.perf_counter()
            print(f"File write stabilized at {file_state['written'] - t_export_start:.3f} s, size: {last_size} bytes")

        monitor_thread = threading.Thread(target=monitor_file)
        monitor_thread.start()

        oModuleRad.ExportFieldsToFile([
            "ExportFileName:=", far_field_path,
            "SetupName:=", sphere_name,
            "IntrinsicVariationKey:=", f"Freq='{freq_str}' IWavePhi='{i_phi_str}' IWaveTheta='{i_theta_str}'",
            "DesignVariationKey:=", f"Ephi='{Ephi}'",
            "SolutionName:=", setup_name,
            "Quantity:=", ""
        ])

        monitor_thread.join()
        t_export_end = time.perf_counter()

        print(f"Total ExportFieldsToFile() time: {(t_export_end - t_export_start):.3f} s")

        t_read = time.perf_counter()
        df = read_hfss_far_field(far_field_path, rad_theta_step, rad_phi_step)
        print(f"Read far field in {time.perf_counter() - t_read:.3f}s")

        df["Freq"] = freq_str
        df["Ephi"] = int(Ephi)
        df["IWavePhi"] = i_phi
        df["IWaveTheta"] = i_theta
        return df

    # Loop through all input combinations and evaluate the far field
    for i_phi in i_phi_values:
        for i_theta in i_theta_values:
            t1 = time.perf_counter()
            df = evaluate_fields_and_export(E_phi, i_phi, i_theta)
            all_data.append(df)
            count += 1
            print(f"[f1] [{count}/{total}] φ={i_phi}deg, θ={i_theta}deg, Ephi={E_phi}, {time.perf_counter() - t1:.3f}s")

    far_field_df = pd.concat(all_data, ignore_index=True)

    far_field_df = far_field_df[[
        "Freq", "Ephi", "IWavePhi", "IWaveTheta",
        "Phi", "Theta", "rEphi_real", "rEphi_imag", "rEtheta_real", "rEtheta_imag"
    ]]

    if csv:
        folder_name = f"{project_name}_{design_name}_{frequency}GHz_Ephi={E_phi}"
        folder_path = os.path.join(output_file_location, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        far_field_filename = "far_field.csv"
        far_field_df.to_csv(os.path.join(folder_path, far_field_filename), index=False)

    print(f"[{freq_str}] Done in {(time.perf_counter() - start_time ) /60:.3f} min")
    return far_field_df

# Regex patterns (helper method for extract_far_field_data)
def extract_var(field, variation_str):
    match = re.search(fr"{field}='([^']+)'", variation_str)
    return match.group(1) if match else None

# Takes in the .fld file for the field at the exit of the waveguide and converts it to a df
def read_hfss_field(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Skip the first 2 header lines
    data_lines = lines[2:]

    # Read numeric values into a (N, 4) NumPy array: X, Y, Z, |E|
    data = np.array([
        list(map(float, line.strip().split()))
        for line in data_lines if line.strip()
    ])

    df = pd.DataFrame(data, columns=['X', 'Y', 'Z', 'Mag_E'])

    # Delete the file from the computer for memory purposes
    try:
        os.remove(filepath)
    except FileNotFoundError:
        print(f"File not found, cannot delete: {filepath}")

    return df

# Takes in .ffd file for the far field and adds columns for rEphi_mag, rEphi_ang, rEtheta_mag, and rEtheta_ang
def read_hfss_far_field(filepath, rad_theta_step, rad_phi_step):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Parse angle metadata
    theta_start, theta_end, n_theta = map(float, lines[0].split())
    phi_start, phi_end, n_phi = map(float, lines[1].split())
    n_theta = int(n_theta)
    n_phi = int(n_phi)

    theta_vals = np.arange(theta_start, theta_end + 1e-8, rad_theta_step)
    phi_vals = np.arange(phi_start, phi_end + 1e-8, rad_phi_step)

    theta_grid, phi_grid = np.meshgrid(theta_vals, phi_vals, indexing="ij")

    # Parse field data lines (line 5 onward, due to the format of the file)
    data_lines = lines[4:]
    assert len(data_lines) == n_theta * n_phi, "Mismatch in expected vs actual data lines"

    rEtheta_real = []
    rEtheta_imag = []
    rEphi_real = []
    rEphi_imag = []

    for line in data_lines:
        parts = list(map(float, line.strip().split()))
        rEtheta_real.append(parts[0])
        rEtheta_imag.append(parts[1])
        rEphi_real.append(parts[2])
        rEphi_imag.append(parts[3])

    # Build DataFrame
    df = pd.DataFrame({
        "Phi": phi_grid.ravel(),
        "Theta": theta_grid.ravel(),
        "rEphi_real": rEphi_real,
        "rEphi_imag": rEphi_imag,
        "rEtheta_real": rEtheta_real,
        "rEtheta_imag": rEtheta_imag,
    })

    return df

# Delete excitations/far field radiation sphere and previous refinement setup following completion of each frequency
def clear_analysis(frequency, sweep):
    sphere_name = get_radiation_sphere_name(frequency)

    oModuleBoundary.DeleteAllExcitations()

    oModuleRad.DeleteSetup([sphere_name])
    setup_name = get_setup_name(frequency)

    # Disable the analysis/parametric setups if we are in a discrete sweep. If we are doing adaptive sweeps, the setups will
    # already be disabled
    if sweep == "discrete":
        oModuleAnalysis.EditSetup(setup_name,
                                  [
                                      f"NAME:{setup_name}",
                                      "IsEnabled:=", False,
                                  ])
        parametric_setup_name = get_parametric_setup_name(frequency)
        oModuleParametric.EnableSetup(parametric_setup_name, False)

# Clear existing features to start fresh
def clear_simulation():
    # Delete existing outgoing face list
    try:
        oEditor.Delete([
            "NAME:Selections",
            "Selections:=", "outgoing"
        ])
    except Exception:
        pass

    # Delete existing local coordinate system
    try:
        oEditor.Delete([
            "NAME:Selections",
            "Selections:=", "outgoing_cs"
        ])
    except Exception:
        pass

    # Delete all existing setups and parametric setups. Note: these calls require
    # the setups to be enabled in HFSS; it may be necessary to delete previous analysis sweeps
    for setup_name in oModuleAnalysis.GetSetups():
        oModuleAnalysis.DeleteSetups([setup_name])
    for parametric_setup_name in oModuleParametric.GetSetupNames():
        oModuleParametric.DeleteSetups([parametric_setup_name])

    # Revert back to initial mesh for consistency
    try:
        oModuleAnalysis.RevertAllToInitial()
    except Exception:
        pass

    # Delete existing radiation boundaries and plane wave excitations
    oModuleBoundary.DeleteAllBoundaries()
    oModuleBoundary.DeleteAllExcitations()

    # Delete far field infinite spheres (radiation)
    for sphere_name in oModuleRad.GetSetupNames("Infinite Sphere"):
        oModuleRad.DeleteSetup([sphere_name])

#%%
def main():

    # Simulation begins here
    clear_simulation()
    initialize_variables(Ei)
    plane_wave_face, outgoing_face = get_faces_from_face_id(ingoing_face_id, outgoing_face_id)
    create_local_coordinate_system(outgoing_face, outgoing_face_cs_x, outgoing_face_cs_y)
    add_outgoing_power_to_calculator()

    if not import_from_existing_csv:
        frequencies = list(range(freq_lower, freq_upper, freq_step))

        # Previous frequency mesh is used for more efficient meshing of new frequency
        previous_frequency = None

        for i, frequency in enumerate(frequencies):
            create_setup(frequency, max_delta_E, max_passes, previous_frequency)
            setup_radiation(rad_params, frequency)
            run_analysis(num_cores, max_delta_E, max_passes, plane_wave_face, Ei, output_file_location, i_theta_lower, i_theta_upper, i_phi_lower,
                         i_phi_upper, frequency, i_theta_step, i_phi_step, rad_params, sweep, max_difference)

            # Delete plane wave after each frequency, except for the last frequency. For discrete sweeps,
            # this allows the user to inspect fields.
            if i < len(frequencies) - 1:
                clear_analysis(frequency, sweep)

            previous_frequency = frequency
    else:
        # We use regex to find the Ephi and frequency from the file path
        parent_folder = os.path.basename(os.path.dirname(waveguide_data_csv))
        m = re.search(r"_(\d+)GHz_Ephi=(\d+)$", parent_folder)
        if not m:
            raise ValueError(f"Cannot parse frequency/Ephi from '{parent_folder}'")

        frequency = int(m.group(1))
        E_phi = int(m.group(2))

        hfss["Ephi"] = E_phi
        setup_radiation(rad_params, frequency)
        run_analysis(num_cores, max_delta_E, max_passes, plane_wave_face, Ei, output_file_location, i_theta_lower,
                     i_theta_upper, i_phi_lower, i_phi_upper, frequency, i_theta_step, i_phi_step, rad_params,
                     sweep, max_difference, True, E_phi, waveguide_data_csv,
                     far_field_data_csv)

    hfss.release_desktop(close_projects=False, close_desktop=False)

if __name__ == "__main__":
    main()
