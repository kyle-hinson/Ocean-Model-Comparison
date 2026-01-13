"""
Volume-averaged salt mixing from MPAS-O timeSeriesStatsCustom output, 
which are averages every 2 hours

Computes:
- Mnum_salt = (1/V) \iiint chiSpurSalt dV
- Mphy_salt = (1/V) \iiint chiPhyVerSalt dV

Outputs NetCDF files in ./mixing_outputs/
"""

import numpy as np
import xarray as xr
import os
import re
import warnings
warnings.filterwarnings("ignore")

# --------------------------------------------------
# Interpolate physical vertical mixing to cell centers
# --------------------------------------------------
def interp_mphy(ds):
    mphy = 0.5 * (
        ds.timeCustom_avg_chiPhyVerTracer_chiPhyVerSalt.isel(nVertLevelsP1=slice(0, -1)) +
        ds.timeCustom_avg_chiPhyVerTracer_chiPhyVerSalt.isel(nVertLevelsP1=slice(1, None))
    )
    ds["timeCustom_avg_chiPhyVerSalt"] = (
        mphy.rename({"nVertLevelsP1": "nVertLevels"})
    )

# --------------------------------------------------
# Paths (exactly as provided)
# --------------------------------------------------
pathsg = [
    "/pscratch/sd/d/dylan617/bichan/mpaso/new_hres/10km/channel_10000m_50_layers_init.nc",
    "/pscratch/sd/d/dylan617/bichan/mpaso/new_hres/5km/channel_5000m_50_layers_init.nc",
    "/pscratch/sd/d/dylan617/bichan/mpaso/new_hres/2km/channel_2000m_50_layers_init.nc",
    "/pscratch/sd/d/dylan617/bichan/mpaso/new_hres/1km/channel_1000m_50_layers_init.nc",
    "/pscratch/sd/d/dylan617/bichan/mpaso/new_hres/500m/channel_500m_50_layers_init.nc",
    "/pscratch/sd/d/dylan617/bichan/mpaso/new_hres/200m/channel_200m_50_layers_init.nc",
    "/pscratch/sd/d/dylan617/bichan/mpaso/new_hres/100m/channel_100m_50_layers_init.nc"
]

pathsd = [
    "/pscratch/sd/d/dylan617/bichan/mpaso/new_hres/10km/analysis_members/mpaso.hist.am.timeSeriesStatsCustom_10000m_50_layers_gmd*.nc",
    "/pscratch/sd/d/dylan617/bichan/mpaso/new_hres/5km/analysis_members/mpaso.hist.am.timeSeriesStatsCustom_5000m_50_layers_gmd*.nc",
    "/pscratch/sd/d/dylan617/bichan/mpaso/new_hres/2km/analysis_members/mpaso.hist.am.timeSeriesStatsCustom_2000m_50_layers_gmd*.nc",
    "/pscratch/sd/d/dylan617/bichan/mpaso/new_hres/1km/analysis_members/mpaso.hist.am.timeSeriesStatsCustom_1000m_50_layers_gmd*.nc",
    "/pscratch/sd/d/dylan617/bichan/mpaso/new_hres/500m/analysis_members/mpaso.hist.am.timeSeriesStatsCustom_500m_50_layers_gmd*.nc",
    "/pscratch/sd/d/dylan617/bichan/mpaso/new_hres/200m/analysis_members/mpaso.hist.am.timeSeriesStatsCustom_200m_50_layers_gmd*.nc",
    "/pscratch/sd/k/kehinson/seahorce/mpaso_channel/bichan_v202601/100m/analysis_members/mpaso.hist.am.timeSeriesStatsCustom_100m_50_layers_gmd*.nc"
]

# --------------------------------------------------
# Output directory
# --------------------------------------------------
outdir = "mixing_outputs"
os.makedirs(outdir, exist_ok=True)

# --------------------------------------------------
# Main loop
# --------------------------------------------------
# --------------------------------------------------
# Main loop
# --------------------------------------------------
for p_grid, p_stats in zip(pathsg, pathsd):

    # infer resolution label
    fname = os.path.basename(p_grid)
    res = re.search(r"_(\d+m)_", fname).group(1)

    outfile = os.path.join(outdir, f"mixing_vavg_{res}.nc")
    if os.path.exists(outfile):
        print(f"\nSkipping {res} (file already exists)")
        continue  # skip this resolution

    print(f"\nProcessing {res}")

    # open datasets
    dsg = xr.open_dataset(p_grid)
    dsa = xr.open_mfdataset(p_stats, combine="by_coords")

    # ensure Time coordinate
    dsa = dsa.assign_coords(Time=dsa.xtime)

    # interpolate physical mixing
    interp_mphy(dsa)

    # --------------------------------------------------
    # Spatial mask (exclude 50 km near walls)
    # --------------------------------------------------
    ycell = dsg.yCell.values
    idx = np.where((ycell > 50_000) & (ycell < 250_000))[0]

    area = dsg.areaCell.isel(nCells=idx)
    h = dsa.timeCustom_avg_layerThickness.isel(nCells=idx)

    # total volume
    V = (h * area).sum(dim=("nCells", "nVertLevels"))

    # --------------------------------------------------
    # Volume-averaged mixing
    # --------------------------------------------------
    Mnum = (
        dsa.timeCustom_avg_chiSpurTracerBR08_chiSpurSaltBR08.isel(nCells=idx)
        * h * area
    ).sum(dim=("nCells", "nVertLevels")) / V

    Mphy = (
        dsa.timeCustom_avg_chiPhyVerSalt.isel(nCells=idx)
        * h * area
    ).sum(dim=("nCells", "nVertLevels")) / V

    # --------------------------------------------------
    # Save NetCDF
    # --------------------------------------------------
    ds_out = xr.Dataset(
        {
            "Mnum_salt": Mnum,
            "Mphy_salt": Mphy,
        }
    )

    ds_out.to_netcdf(outfile)

    print(f"  Saved → {outfile}")

