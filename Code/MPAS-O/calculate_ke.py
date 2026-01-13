"""
Surface kinetic energy from MPAS-O history output
Outputs NetCDF files in ./energy_outputs/
"""

import xarray as xr
from xhistogram.xarray import histogram
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import os
import re
import time
from scipy import signal
import warnings
warnings.filterwarnings('ignore')
import os

mds = []
mdsg = []
mverts = []
midx = []
mnorm = []

pathso = ['/pscratch/sd/d/dylan617/bichan/mpaso/new_hres/10km/output_10000m_50_layers_gmd.nc',
         '/pscratch/sd/d/dylan617/bichan/mpaso/new_hres/5km/output_5000m_50_layers_gmd.nc',
         '/pscratch/sd/d/dylan617/bichan/mpaso/new_hres/2km/output_2000m_50_layers_gmd.nc',
         '/pscratch/sd/d/dylan617/bichan/mpaso/new_hres/1km/output_1000m_50_layers_gmd.nc',
         '/pscratch/sd/d/dylan617/bichan/mpaso/new_hres/500m/output_500m_50_layers_gmd.nc',
         '/pscratch/sd/d/dylan617/bichan/mpaso/new_hres/200m/output_200m_50_layers_gmd.nc',
         '/pscratch/sd/k/kehinson/seahorce/mpaso_channel/bichan_v202601/100m/output_100m_50_layers_gmd.nc'
         ]

pathsg = ['/pscratch/sd/d/dylan617/bichan/mpaso/new_hres/10km/channel_10000m_50_layers_init.nc',
         '/pscratch/sd/d/dylan617/bichan/mpaso/new_hres/5km/channel_5000m_50_layers_init.nc',
         '/pscratch/sd/d/dylan617/bichan/mpaso/new_hres/2km/channel_2000m_50_layers_init.nc',
         '/pscratch/sd/d/dylan617/bichan/mpaso/new_hres/1km/channel_1000m_50_layers_init.nc',
         '/pscratch/sd/d/dylan617/bichan/mpaso/new_hres/500m/channel_500m_50_layers_init.nc',
         '/pscratch/sd/d/dylan617/bichan/mpaso/new_hres/200m/channel_200m_50_layers_init.nc',
         '/pscratch/sd/d/dylan617/bichan/mpaso/new_hres/100m/channel_100m_50_layers_init.nc'
         ]

for p_out, p_grid in zip(pathso, pathsg):

    # MPAS output (time-dependent)
    ds_mpas = xr.open_dataset(p_out)

    # MPAS init / grid file
    dsg = xr.open_dataset(p_grid)
    
    mds.append(ds_mpas)
    mdsg.append(dsg)

    # --------------------------------------------------
    # Cell polygons (vertices)
    # --------------------------------------------------
    verts = np.dstack((
        dsg.xVertex.values[dsg.verticesOnCell.values - 1],
        dsg.yVertex.values[dsg.verticesOnCell.values - 1]
    ))

    # number of valid vertices per cell
    nverts = np.sum(dsg.verticesOnCell.values != 0, axis=1)

    # trim unused vertices
    verts = [v[:n] for v, n in zip(verts, nverts)]

    # select narrow cells in x (channel slice)
    idx = np.array([
        np.ptp(v[:, 0]) < 50_000 for v in verts
    ])

    midx.append(idx)
    mverts.append(np.array(verts, dtype=object)[idx])

    # color normalization
    mnorm.append(plt.matplotlib.colors.Normalize(-3, 3))

# Loop over MPAS-O outputs and calculate average KE from variable kineticEnergyCell
outdir = "energy_outputs"
os.makedirs(outdir, exist_ok=True)

# --------------------------------------------------
# Loop over MPAS-O outputs and calculate/save KE
# --------------------------------------------------
for axi, (ds_mpas, dsg, p_out) in enumerate(zip(mds, mdsg, pathso)):

    # extract resolution string from filename
    fname = os.path.basename(p_out)
    match = re.search(r"_(\d+m)_", fname)
    if match is None:
        raise ValueError(f"Could not infer resolution from {fname}")
    res = match.group(1)   # e.g. '10000m', '500m'

    # y-limits: exclude 50 km near walls
    ympas = dsg.yCell.values
    idx = np.where((ympas > 50_000) & (ympas < 250_000))[0]

    # mean KE at surface (nVertLevels = 0)
    ke = (
        ds_mpas.kineticEnergyCell
        .isel(nVertLevels=0, nCells=idx)
        .mean(dim="nCells")
    )

    # wrap as Dataset
    ke_ds = ke.to_dataset(name="meanKE")

    # save
    outfile = os.path.join(outdir, f"meanKE_{res}.nc")
    ke_ds.to_netcdf(outfile)

    print(f"Saved KE for {res} → {outfile}")

