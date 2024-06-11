# Ocean-Model-Comparison

Analysis for paper "Representation of submesoscale baroclinic instabilities in ROMS and MPAS-O"

This repository contains notebooks that will reproduce the figures in the paper "Representation of submesoscale baroclinic instabilities in ROMS and MPAS-O" by Kyle Hinson, Robert Hetland, Darren Engwirda, and Kat Smith, submitted to the Journal of Advances in Modeling Earth Systems.

Five notebooks are in the Notebooks directory. The notebook Model_Resolution.ipynb will load in the idealized channel flow configurations found in the zenodo archive and reproduce the snapshot of relative vorticity for all model resolutions.

Additionally, the script bichan_utils.py contains functions that will compute strain and frontogenesis components of divergence for ROMS simulations. These variables are already contained within MPAS-O output files.
