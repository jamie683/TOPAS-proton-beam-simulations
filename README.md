PHY4004 Medical Radiation Simulation – Monte Carlo Code Repository

This repository contains the simulation input files and analysis scripts used for Assignments 1 and 2 of the PHY4004 Medical Radiation Physics module at Queen’s University Belfast.

All simulations were performed using the TOPAS Monte Carlo toolkit (Geant4-based), with Python used for analysis, optimisation, and post-processing.

The work spans both fundamental particle transport physics (Assignment 1) and applied radiotherapy treatment planning in a patient geometry (Assignment 2).

--------------------------------------------------------------------------------

Assignment 1 – Particle Transport in Water and Heterogeneous Media

Assignment 1 investigates fundamental physical processes relevant to charged particle therapy, including:

- Bragg peak formation for proton beams
- Water Equivalent Thickness (WET) of heterogeneous inserts
- Energy matching between proton and carbon ion beams
- Multiple Coulomb scattering and beam broadening
- Radial energy deposition distributions

Repository Structure (Assignment 1):

Assignment_1/

├── section_3_1/
│   Proton depth-dose simulation in water
├── section_3_2/
│   Carbon energy matching using a bisection search
├── section_3_3/
│   Water equivalent thickness (WET) calculation
├── section_3_4/
│   Stopping power comparison and interpretation
├── section_3_5/
│   Radial scattering analysis for aluminium inserts
└── section_3_6/
    Lead thickness matching and beam broadening analysis

Each folder contains:
- TOPAS parameter files (.txt)
- Python scripts for analysis and plotting

--------------------------------------------------------------------------------

Assignment 2 – Radiotherapy Treatment Planning in Patient Geometry

Assignment 2 extends these principles to clinically relevant treatment planning using a patient CT dataset and RT structure set.

Key physical and computational components include:

- CT-based patient geometry import (DICOM)
- HU → relative stopping power calibration (Schneider method)
- Photon multi-field treatment planning
- Monoenergetic proton beam modelling
- Water-equivalent path length (WEPL) calculation
- Spread-Out Bragg Peak (SOBP) construction using NNLS optimisation
- Energy spread optimisation and plateau flatness analysis
- Lateral beam-width optimisation in patient geometry
- Pencil Beam Scanning (PBS) dose modelling
- Motion and interplay effects in PBS delivery
- Neutron production in proton therapy configurations

Repository Structure (Assignment 2):

Assignment_2/

├── CTData/
│   Patient DICOM CT and RTStruct files
├── A2_1 → A2_5/
│   Geometry setup, beam alignment, DVHs, photon and monoenergetic proton plans
├── A2_6/
│   WEPL-based SOBP construction and optimisation
├── A2_7/
│   Neutron production study
├── A2_8/
│   Pencil Beam Scanning (PBS) implementation
├── A2_9/
│   Motion and interplay effects in PBS

Each section contains:
- TOPAS parameter files
- Python workflows for optimisation and analysis
- Output plots and summary tables (where applicable)

--------------------------------------------------------------------------------

Software Requirements

The simulations require:

- TOPAS Monte Carlo Toolkit
- Geant4 (via TOPAS)
- Python 3

Python libraries used:

- numpy
- matplotlib
- scipy
- pathlib
- subprocess
- csv
- re

--------------------------------------------------------------------------------

Reproducibility

The Python scripts automate the full simulation workflow by:

- Generating and modifying TOPAS parameter files
- Executing simulations via the TOPAS command line
- Parsing simulation outputs
- Computing physical quantities (e.g. Bragg peak depth, WEPL, DVH metrics)
- Performing optimisation (e.g. NNLS for SOBP, beam-width sweeps, PBS spot weighting)
- Producing plots used in the accompanying reports

--------------------------------------------------------------------------------

Associated Reports

The full scientific discussion, figures, and interpretation are presented in the Assignment 1 and Assignment 2 reports.

This repository contains only the simulation inputs and analysis code used to generate those results.

--------------------------------------------------------------------------------

Author

Jamie McAteer  
MSci Physics with Astrophysics  
Queen's University Belfast
