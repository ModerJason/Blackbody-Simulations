# HFSS Simulations – Blackbody Radiation in Cryogenic Detector Gaps
This repository contains HFSS models and scripts for simulating how blackbody photons from warmer components propagate through small mechanical gaps ($$O(10^2–10^3 μm)$$) in the dark matter detector cryostats, especially in relation to the Cryogenic Dark Matter Search (CDMS).

These gaps behave like semi-open parallel-plate waveguides that can support low-frequency TEM modes with negligible cutoff, allowing photons to reach ultra-cold detectors and cause photoionization or leakage currents.

**We model:**
- Gap geometries and dimensions from mechanical designs
- Plane-wave excitation over blackbody-relevant frequencies
- |S21| parameters, field distributions, and far-field emission

**Workflow:**

For a guide on usage, see [here](https://github.com/ModerJason/Blackbody-Simulations/blob/main/Blackbody%20Simulations%20Usage%20Guide.pdf).
- Build or import a geometry in HFSS
- Run bbsim.py
- Post-process data: create outgoing power and electric field plots, interpolate across parameters, etc.

These results will later integrate with Geant4 open-space photon simulations for an end-to-end model.

For more details on analytical calculations, validation for HFSS simulations, and plots, see [here](https://github.com/ModerJason/Blackbody-Simulations/blob/main/Blackbody%20Simulation%20Report.pdf).

**Requirements:**  
- ANSYS HFSS 2023 R2+  
- Python packages: `numpy`, `scipy`, `matplotlib`, `pyaedt`, `seaborn`

This project is developed using the [Spyder IDE](https://www.spyder-ide.org/).


