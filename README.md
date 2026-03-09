# PHY4004 Assignment 1 – Monte Carlo Simulation Code

This repository contains the **simulation input files and analysis scripts** used for *Assignment 1 of the PHY4004 Medical Radiation Physics module* at Queen’s University Belfast.

The simulations were performed using the **TOPAS Monte Carlo toolkit (Geant4-based)** to investigate the transport of charged particle beams in water and heterogeneous media.

The work explores several key physical phenomena relevant to particle therapy, including:

- Bragg peak formation for proton beams
- Water Equivalent Thickness (WET) of heterogeneous inserts
- Energy matching between proton and carbon ion beams
- Multiple Coulomb scattering and beam broadening
- Radial energy deposition distributions

The repository is provided to ensure **reproducibility of the simulation methodology and analysis workflow**.

---

## Repository Structure
Assignment1_TOPAS_Code/
│
├── section_3_1/
│ Proton depth-dose simulation in water
│
├── section_3_2/
│ Carbon energy matching using a bisection search
│
├── section_3_3/
│ Water equivalent thickness (WET) calculation
│
├── section_3_4/
│ Stopping power comparison and interpretation
│
├── section_3_5/
│ Radial scattering analysis for bone inserts
│
└── section_3_6/
Lead thickness matching and beam broadening analysis

Each folder contains:

- **TOPAS parameter files (.txt)** defining the simulation geometry and beam configuration  
- **Python analysis scripts** used to extract physical quantities such as Bragg peak depth, WET, and beam width metrics

---

## Software Requirements

The simulations require:

- **TOPAS Monte Carlo Toolkit**
- **Geant4 (via TOPAS)**
- **Python 3**

Python libraries used:

- numpy
- matplotlib
- pathlib
- subprocess
- re

---

## Reproducibility

The Python scripts automate the workflow by:

1. Generating modified TOPAS parameter files
2. Executing simulations via the TOPAS command line interface
3. Parsing simulation outputs
4. Computing derived physical quantities such as Bragg peak position and \(R_{80}\) beam width
5. Producing plots used in the accompanying report

---

## Associated Report

The full scientific discussion, figures, and results are presented in the assignment report.

This repository contains **only the simulation inputs and analysis code** used to generate those results.

---

## Author

Jamie McAteer  
MSci Physics with Astrophysics  
Queen's University Belfast
