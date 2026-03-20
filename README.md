# Ocean-Model-Comparison

Analysis for the paper "Towards High-Fidelity Simulations of Coastal Submesoscale Baroclinic Instabilities with MPAS-O Part I: Idealized Experiments" by  Kyle Hinson, Dylan Schlichting, Robert D. Hetland, Darren Engwirda, Katherine Smith, Mark R. Petersen, and Kaila Uyeda

This repository contains notebooks that will reproduce the figures in the paper. Four notebooks are in the `Notebooks` directory, corresponding to the manuscript figures and based on output files in the Zenodo archive (https://doi.org/10.5281/zenodo.18868239). Additionally, the script `bichan_utils.py` contains functions that will compute strain and frontogenesis components of divergence for ROMS simulations. These velocity gradient variables are already contained within the MPAS-O output files. Grid metrics for MPAS-O simulations are contained within the `MPAS-O_Initial_*.nc` files in the Zenodo archive.

Input parameters and header files for ROMS and MPAS-O are specified in the `Code` directory, with additional information therein specifying how to run multiple resolution simulations for both models.
