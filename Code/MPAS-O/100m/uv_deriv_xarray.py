import time
import numpy as np
import xarray as xr
import argparse

from msh import load_mesh
from ops import operators

# --------------------------------------------------
# Parameters
# --------------------------------------------------
DT_CHUNK = 24       # number of DTs per output file
KLEV = 0            # top layer only

# --------------------------------------------------
# Main
# --------------------------------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Compute velocity and salinity gradients and "
                    "write NetCDF files every 24 DT.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--mesh-file", dest="mesh_file", type=str, required=True,
        help="Path to MPAS mesh file."
    )

    parser.add_argument(
        "--flow-file", dest="flow_file", type=str, required=True,
        help="Path to MPAS-O flow file."
    )

    parser.add_argument(
        "--outdir", dest="outdir", type=str, required=True,
        help="Output directory for gradient NetCDF files."
    )

    args = parser.parse_args()

    # --------------------------------------------------
    # Load mesh + operators
    # --------------------------------------------------
    print("Loading mesh...")
    mesh = load_mesh(args.mesh_file)

    print("Forming operators...")
    mats = operators(mesh)

    # --------------------------------------------------
    # Open flow dataset (read-only)
    # --------------------------------------------------
    print("Opening flow file...")
    flow = xr.open_dataset(args.flow_file, chunks={"Time": 1})

    nCells = int(flow.sizes["nCells"])
    nVertLevels = int(flow.sizes["nVertLevels"])
    nTime = 240

    print(f"nCells={nCells}, nVertLevels={nVertLevels}, nTime={nTime}")
    print("Processing in 24-DT chunks...")

    # Ensure output directory exists
    import os
    os.makedirs(args.outdir, exist_ok=True)

    # --------------------------------------------------
    # Loop over time chunks
    # --------------------------------------------------
    for t0 in range(0, nTime, DT_CHUNK):
        t1 = min(t0 + DT_CHUNK, nTime)
        nt = t1 - t0

        print(f"\nProcessing timesteps {t0:05d}–{t1-1:05d}")

        # Allocate arrays (chunk-local)
        shape = (nt, nCells, nVertLevels)
        Ux   = np.zeros(shape, dtype=np.float32)
        Vy   = np.zeros(shape, dtype=np.float32)
        dUdx = np.zeros(shape, dtype=np.float32)
        dUdy = np.zeros(shape, dtype=np.float32)
        dVdx = np.zeros(shape, dtype=np.float32)
        dVdy = np.zeros(shape, dtype=np.float32)
        dSdx = np.zeros(shape, dtype=np.float32)
        dSdy = np.zeros(shape, dtype=np.float32)

        # --------------------------------------------------
        # Time loop inside chunk
        # --------------------------------------------------
        for it, step in enumerate(range(t0, t1)):
            print(f"  step {step}")

            uu_edge = flow["normalVelocity"][step, :, KLEV]

            # Velocity remap
            Ux_cell = mats.cell_lsqr_xnrm * uu_edge
            Vy_cell = mats.cell_lsqr_ynrm * uu_edge

            Ux[it, :, KLEV] = Ux_cell
            Vy[it, :, KLEV] = Vy_cell

            # Velocity gradients
            dUdn_edge = mats.edge_grad_norm * Ux_cell
            dUdx[it, :, KLEV] = mats.cell_lsqr_xnrm * dUdn_edge
            dUdy[it, :, KLEV] = mats.cell_lsqr_ynrm * dUdn_edge

            dVdn_edge = mats.edge_grad_norm * Vy_cell
            dVdx[it, :, KLEV] = mats.cell_lsqr_xnrm * dVdn_edge
            dVdy[it, :, KLEV] = mats.cell_lsqr_ynrm * dVdn_edge

            # Salinity gradients
            salt_cell = flow["salinity"][step, :, KLEV]
            dSdn_edge = mats.edge_grad_norm * salt_cell
            dSdx[it, :, KLEV] = mats.cell_lsqr_xnrm * dSdn_edge
            dSdy[it, :, KLEV] = mats.cell_lsqr_ynrm * dSdn_edge

        # --------------------------------------------------
        # Build output Dataset
        # --------------------------------------------------
        ds_out = xr.Dataset(
            {
                "Ux_cell":   (("Time", "nCells", "nVertLevels"), Ux),
                "Vy_cell":   (("Time", "nCells", "nVertLevels"), Vy),
                "dUdx_cell": (("Time", "nCells", "nVertLevels"), dUdx),
                "dUdy_cell": (("Time", "nCells", "nVertLevels"), dUdy),
                "dVdx_cell": (("Time", "nCells", "nVertLevels"), dVdx),
                "dVdy_cell": (("Time", "nCells", "nVertLevels"), dVdy),
                "dSdx_cell": (("Time", "nCells", "nVertLevels"), dSdx),
                "dSdy_cell": (("Time", "nCells", "nVertLevels"), dSdy),
            },
            coords={
                "Time": flow.Time.isel(Time=slice(t0, t1)).values,
                "nCells": flow.nCells,
                "nVertLevels": flow.nVertLevels,
            },
        )

        outfile = (
            f"{args.outdir}/output_100m_50_layers_gmd_"
            f"gradients_t{t0:05d}_t{t1-1:05d}.nc"
        )

        ds_out.to_netcdf(outfile, engine="netcdf4")
        print(f"Saved → {outfile}")

    print("\nAll chunks processed successfully.")
