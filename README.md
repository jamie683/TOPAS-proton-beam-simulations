# PHY4004 Medical Radiation Simulation

This repository contains simulation input files and analysis scripts for **Assignments 1 and 2** of the *PHY4004 Medical Radiation Physics* module at Queen’s University Belfast.

Simulations were performed using the **TOPAS Monte Carlo toolkit (Geant4-based)**, with Python used for analysis, optimisation, and post-processing.

---

## Overview

The work spans:

- **Assignment 1:** Fundamental charged particle transport physics  
- **Assignment 2:** Radiotherapy treatment planning in a patient CT geometry  

This repository is intended to support **reproducibility** of all simulations and analysis workflows used in the accompanying reports.

---

## Assignment 1 — Particle Transport Physics

This assignment investigates key physical processes relevant to particle therapy:

- Bragg peak formation (protons)
- Water Equivalent Thickness (WET)
- Proton vs carbon ion energy matching
- Multiple Coulomb scattering
- Beam broadening and radial dose distribution

### Structure

```
Assignment_1/
├── section_3_1/   # Proton depth-dose in water
├── section_3_2/   # Carbon energy matching
├── section_3_3/   # WET calculation
├── section_3_4/   # Stopping power analysis
├── section_3_5/   # Radial scattering (Al inserts)
└── section_3_6/   # Lead thickness & beam broadening
```

Each section contains:
- TOPAS parameter files (`.txt`)
- Python analysis scripts

---

## Assignment 2 — Radiotherapy Treatment Planning

Assignment 2 applies these principles to a **patient CT dataset**, including treatment planning and evaluation.

### Key components

- CT import and material calibration (HU → RSP)
- Photon multi-field treatment planning
- Monoenergetic proton beams
- WEPL-based range calculation
- SOBP construction (NNLS optimisation)
- Energy spread optimisation
- Lateral beam-width optimisation
- Pencil Beam Scanning (PBS)
- Motion & interplay effects
- Neutron production analysis

### Structure

```
Assignment_2/
├── CTData/        # Patient CT + RTStruct
├── A2_1–A2_5/     # Geometry, photons, monoenergetic protons
├── A2_6/          # SOBP construction & optimisation
├── A2_7/          # Neutron production
├── A2_8/          # PBS implementation
└── A2_9/          # Motion & interplay
```

Each section contains:
- TOPAS input files
- Python workflows
- Generated plots and DVHs

---

## Software Requirements

### Core
- TOPAS (Geant4-based)
- Geant4 data libraries

### Python
- Python 3
- numpy
- matplotlib
- scipy

---

## Reproducibility

The Python scripts automate:

- TOPAS file generation
- Simulation execution
- Output parsing
- Physical quantity extraction (Bragg peak, WEPL, DVHs)
- Optimisation (NNLS, beam-width sweeps, PBS weighting)
- Plot generation

Run scripts from the project root where possible:

```bash
python3 Assignment_2/A2_6/sobp_proton.py
```

---

## Notes

- Dose is often **normalised to maximum dose** in DVH plots
- SOBP design uses **WEPL**, not geometric depth
- PBS introduces **temporal effects** (motion interplay)
- This is research/assignment code — not clinical software

---

## Reports

This repository contains **code only**.

Full results, figures, and discussion are presented in the submitted assignment reports.

---

## Author

**Jamie McAteer**  
MSci Physics with Astrophysics  
Queen's University Belfast
