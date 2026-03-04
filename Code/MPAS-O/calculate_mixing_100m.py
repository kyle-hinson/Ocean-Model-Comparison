'''
Calculates volume-averaged mixing for the 100 m mpas-o simulation.
Requires special processing to remove duplicate values from the multiple
restarts. 
'''
import numpy as np
import xarray as xr
import os
import warnings
warnings.filterwarnings("ignore")

# ------------------------------------------------------------------
# FILE PATHS (SINGLE CASE ONLY)
# ------------------------------------------------------------------
output_files = [
    '/pscratch/sd/k/kehinson/seahorce/mpaso_channel/bichan_v202601/100m/analysis_members/mpaso.hist.am.timeSeriesStatsCustom_100m_50_layers_gmd_day_0_29.nc',
    '/global/cfs/cdirs/m4572/dylan617/bichan/100m/analysis_members/mpaso.hist.am.timeSeriesStatsCustom_100m_50_layers_gmd_day_29_60.0001-01-01.nc',
    '/global/cfs/cdirs/m4572/dylan617/bichan/100m/analysis_members/mpaso.hist.am.timeSeriesStatsCustom_100m_50_layers_gmd_day_29_60.0001-02-01.nc',
    '/global/cfs/cdirs/m4572/dylan617/bichan/100m/analysis_members/mpaso.hist.am.timeSeriesStatsCustom_100m_50_layers_gmd_day_44_60.0001-02-01.nc'
]

init_file = (
    '/pscratch/sd/d/dylan617/bichan/mpaso/new_hres/100m/'
    'channel_100m_50_layers_init.nc'
)

res_key = "100m"

# ------------------------------------------------------------------
# OUTPUT DIRECTORY
# ------------------------------------------------------------------
outdir = "mixing_outputs"
os.makedirs(outdir, exist_ok=True)

# ------------------------------------------------------------------
# OPEN GRID DATASET
# ------------------------------------------------------------------
dsg = xr.open_dataset(init_file)
ycell = dsg.yCell
area  = dsg.areaCell

# Spatial mask (exclude 50 km near walls)
mask = (ycell > 50_000) & (ycell < 250_000)
idx = np.where(mask)[0]

area_idx = area.isel(nCells=idx).data  # NumPy

# ------------------------------------------------------------------
# OPEN STATS DATASET (MPAS-SAFE CONCAT)
# ------------------------------------------------------------------
print("Opening time averaged files...")

dsa = xr.open_mfdataset(
    output_files,
    combine="nested",
    concat_dim="Time",
    chunks={"Time": 1, "nCells": 60000}
)

# Ensure Time coordinate exists
if "xtime" in dsa:
    dsa = dsa.assign_coords(Time=("Time", dsa.xtime.values))

# ------------------------------------------------------------------
# REMOVE DUPLICATE TIME VALUES (RESTART-SAFE)
# ------------------------------------------------------------------
time_vals = dsa.Time.values
_, unique_idx = np.unique(time_vals, return_index=True)
unique_idx = np.sort(unique_idx)

dsa = dsa.isel(Time=unique_idx)
ntime = dsa.sizes["Time"]

print(f"{ntime} unique time steps after de-duplication")

# ------------------------------------------------------------------
# INTERPOLATE PHYSICAL VERTICAL MIXING
# ------------------------------------------------------------------
arr = dsa.timeCustom_avg_chiPhyVerTracer_chiPhyVerSalt.data
mphy = 0.5 * (arr[:, :, :-1] + arr[:, :, 1:])
dsa["timeCustom_avg_chiPhyVerSalt"] = (
    ("Time", "nCells", "nVertLevels"),
    mphy,
)

# ------------------------------------------------------------------
# SAVE MIXING FILES EVERY 24 DT
# ------------------------------------------------------------------
for t0 in range(0, ntime, 24):
    t1 = min(t0 + 24, ntime)

    outfile = f"{outdir}/recalculated_mixing_{res_key}_t{t0:05d}_t{t1-1:05d}.nc"

    if os.path.exists(outfile):
        print(f"  Exists, skipping → {outfile}")
        continue

    print(f"  Processing time chunk {t0:05d}–{t1-1:05d}")

    dsc = dsa.isel(Time=slice(t0, t1))

    # Convert required fields to NumPy
    h = dsc.timeCustom_avg_layerThickness.isel(nCells=idx).data
    chi_spur = (
        dsc.timeCustom_avg_chiSpurTracerBR08_chiSpurSaltBR08
        .isel(nCells=idx)
        .data
    )
    chi_phy = (
        dsc.timeCustom_avg_chiPhyVerSalt
        .isel(nCells=idx)
        .data
    )

    # Total volume per time
    V = np.sum(h * area_idx[:, None], axis=(1, 2))

    # Volume-averaged mixing
    Mnum = np.sum(chi_spur * h * area_idx[:, None], axis=(1, 2)) / V
    Mphy = np.sum(chi_phy * h * area_idx[:, None], axis=(1, 2)) / V
    Mtot = np.sum((chi_spur + chi_phy) * h * area_idx[:, None], axis=(1, 2)) / V

    ds_out = xr.Dataset(
        {
            "Mnum_salt": (("Time",), Mnum),
            "Mphy_salt": (("Time",), Mphy),
            "Mtot_salt": (("Time",), Mtot),
        },
        coords={"Time": dsc.Time.values},
    )

    ds_out.to_netcdf(outfile)
    print(f"    Saved → {outfile}")

print("\nAll done.")
