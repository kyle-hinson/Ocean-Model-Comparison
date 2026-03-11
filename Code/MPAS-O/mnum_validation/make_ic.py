# Modifies the idealized submesoscale baroclinic channel described in
# Hetland et al. (2025) JPO. MPAS-O can be setup as a linear advection 
# equation using a top hat profile in the along-channel direction.

import numpy as np
import xarray
from mpas_tools.planar_hex import make_planar_hex_mesh
from mpas_tools.io import write_netcdf
from mpas_tools.mesh.conversion import convert, cull

# --- Simulation Setup ---
case_name = "channel"
num_xcell = 100#200#100
# ycell = [2*sqrt(3)] * num-xcell / 3 to yield a LX=LY domain
num_ycell = 116#232#116
len_edges = 1.0#0.5
num_layer = 1
bot_depth = 1

# --- Top-hat profile ---
def s0(x):
    return np.where((x >= 35) & (x <= 65), 1.0, 0.0)

def t0(x):
    return np.where((x >= 35) & (x <= 65), 1.0, 0.0)

# --- Create planar hex mesh ---
mesh = make_planar_hex_mesh(
    nx=num_xcell,
    ny=num_ycell,
    dc=len_edges,
    nonperiodic_x=False,
    nonperiodic_y=False)
write_netcdf(mesh, f"{case_name}_base.nc")

mesh = cull(mesh)
mesh = convert(mesh, graphInfoFileName=f"{case_name}_graph.info")
write_netcdf(mesh, f"{case_name}_cull.nc")

# --- Initialize fields ---
xCell = mesh.xCell.values
nCells = xCell.size

# z grid
z_interface = np.linspace(0, bot_depth, num_layer + 1)
z_mid = 0.5 * (z_interface[:-1] + z_interface[1:])
z_layer_thickness = z_interface[1:] - z_interface[:-1]

# Repeat vertical structure horizontally
zmid_2d = np.repeat(z_mid[np.newaxis, :], nCells, axis=0)
xmid_2d = np.repeat(xCell[:, np.newaxis], num_layer, axis=1)

# Salinity (g/kg) and Temperature (°C) with top-hat in x
salinity = s0(xmid_2d)
temperature = t0(xmid_2d)

# --- Build init dataset ---
init = mesh.copy()
init["bottomDepth"] = bot_depth * xarray.ones_like(mesh.xCell)
init["bottomDepthObserved"] = init["bottomDepth"]
init["ssh"] = (("Time", "nCells"), np.zeros((1, nCells)))
init["minLevelCell"] = xarray.ones_like(mesh.xCell, dtype=np.int32)
init["maxLevelCell"] = num_layer * xarray.ones_like(mesh.xCell, dtype=np.int32)

init["refTopDepth"] = (("nVertLevels"), z_interface[:-1])
init["refBottomDepth"] = (("nVertLevels"), z_interface[1:])
init["refZMid"] = (("nVertLevels"), z_mid)
init["refInterfaces"] = (("nVertLevelsP1"), z_interface)
init["vertCoordMovementWeights"] = (("nVertLevels"), np.ones(num_layer))

init["restingThickness"] = (("nCells", "nVertLevels"),
                             np.tile(z_layer_thickness, (nCells, 1)))
layer_thickness_3d = np.ones((1, nCells, num_layer)) * z_layer_thickness
init["layerThickness"] = (("Time", "nCells", "nVertLevels"), layer_thickness_3d)
init["zTop"] = (("Time", "nCells", "nVertLevels"),
                -np.tile(z_interface[:-1], (nCells, 1))[np.newaxis, :, :])
init["zMid"] = (("Time", "nCells", "nVertLevels"),
                -zmid_2d[np.newaxis, :, :])

init["salinity"] = (("Time", "nCells", "nVertLevels"),
                    salinity[np.newaxis, :, :])
init["temperature"] = (("Time", "nCells", "nVertLevels"),
                       temperature[np.newaxis, :, :])

# Initialize the model with a zonalVelocity of one and meridionalVelocity of zero.
# MPAS-O expects a normalVelocity, so we have to convert here 
angleEdge_np = mesh.variables["angleEdge"].values  # shape (nEdges,)

normal_velocity_edge = np.cos(angleEdge_np)  # zonal=1, meridional=0

normal_velocity_data = np.broadcast_to(
    normal_velocity_edge[np.newaxis, :, np.newaxis],
    (1, angleEdge_np.size, num_layer)
)

init["normalVelocity"] = (("Time", "nEdges", "nVertLevels"), normal_velocity_data)

# --- Save output ---
write_netcdf(init, f"{case_name}_init.nc")