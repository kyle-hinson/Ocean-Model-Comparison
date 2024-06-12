# Ocean-Model-Comparison

Analysis for the paper "Representation of submesoscale baroclinic instabilities in ROMS and MPAS-O"

This repository contains notebooks that will reproduce the figures in the paper "Representation of submesoscale baroclinic instabilities in ROMS and MPAS-O" by Kyle Hinson, Robert Hetland, Darren Engwirda, and Kat Smith, submitted to the Journal of Advances in Modeling Earth Systems.

Four notebooks are in the Notebooks directory, corresponding to the manuscript figures and based on output files in the Zenodo archive (https://doi.org/10.5281/zenodo.11404200). Additionally, the script bichan_utils.py contains functions that will compute strain and frontogenesis components of divergence for ROMS simulations. These velocity gradient variables are already contained within the MPAS-O output files. Grid metrics for MPAS-O simulations are contained within the 'mpaso_init_*.nc' files in the Zenodo archive.
