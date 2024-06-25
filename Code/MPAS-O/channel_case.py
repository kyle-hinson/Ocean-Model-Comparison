
import os
import xarray
import numpy as np
import netCDF4 as nc
import argparse
import datetime

from mpas_tools.planar_hex import make_planar_hex_mesh
from mpas_tools.io import write_netcdf
from mpas_tools.mesh.conversion import convert, cull

RHO0 = 1026.0
GRAV = 9.80665

# MPAS only supports time-varying wind speed, not stress, 
# so invert the Garrat law used under the hood to give a 
# speed for:
WMAG = {0.010: 2.9364,  # tau_x = 0.010
        0.025: 4.4172,  # tau_x = 0.025
        0.050: 5.9594,  # tau_x = 0.050
        0.075: 7.0728,  # tau_x = 0.075
        0.100: 7.9729,  # tau_x = 0.100
        0.125: 8.7407,  # tau_x = 0.125
        0.150: 9.4166,  # tau_x = 0.150
        0.175: 10.0244, # tau_x = 0.175
        0.200: 10.5791  # tau_x = 0.200
        }

PERT = 0.01  # amplitude of zonal velocity perturbation

def roms_zgrid():

    zlev = np.array([
         0.0,
        -1.6667,
        -3.3333,
        -5.0000,
        -6.6667,
        -8.3333,
       -10.0000,
       -11.6667,
       -13.3333,
       -15.0000,
       -16.6667,
       -18.3333,
       -20.0000,
       -21.6667,
       -23.3333,
       -25.0000,
       -26.6667,
       -28.3333,
       -30.0000,
       -31.6667,
       -33.3333,
       -35.0000,
       -36.6667,
       -38.3333,
       -40.0000,
       -41.6667,
       -43.3333,
       -45.0000,
       -46.6667,
       -48.3333,
       -50.0000], dtype=np.float64)

    return zlev


def build_mesh(args):

#-- make a regular hex mesh
    mesh = make_planar_hex_mesh(
        nx=args.num_xcell,
        ny=args.num_ycell,
        dc=args.len_edges,
        nonperiodic_x=False,
        nonperiodic_y=True)
    write_netcdf(mesh, args.case_name + "_base.nc")

    mesh = cull(mesh)
    mesh = convert(
        mesh, graphInfoFileName=args.case_name + "_graph.info")
    write_netcdf(mesh, args.case_name + "_cull.nc")

    return mesh


def build_init(args, mesh):

#-- Form initial conditions
    init = mesh.copy()
    xcel = init.xCell
    ycel = init.yCell

    print("cell x-min. =", xcel.min().values)
    print("cell x-max. =", xcel.max().values)
    print("cell y-min. =", ycel.min().values)
    print("cell y-max. =", ycel.max().values)

    # vertical grid
    ssh = xarray.zeros_like(xcel)
    init["bottomDepth"] = args.bot_depth * xarray.ones_like(xcel)
    init["bottomDepthObserved"] = args.bot_depth * xarray.ones_like(xcel)

    init["ssh"] = (
        ("Time", "nCells"), np.zeros((1, mesh.xCell.size)))
    init["ssh"][0, :] = ssh

    if (args.num_layer <= 0):
    #-- use the same levels as roms
        zlev = -roms_zgrid()  # NB. per MPAS, these are +ve down...
        zref = -roms_zgrid()
        args.num_layer = zlev.size - 1

    #-- reset bottom - just in case
        args.bot_depth = zlev[-1]
        init["bottomDepth"] = args.bot_depth * xarray.ones_like(xcel)
        init["bottomDepthObserved"] = args.bot_depth * xarray.ones_like(xcel)

        zlev = np.reshape(zlev, (zlev.size, 1))
        zlev = np.tile(zlev, (1, ssh.size))

        zref = np.reshape(zref, (zref.size, 1))
        zref = np.tile(zref, (1, ssh.size))

    else:
    #-- build uniformly spaced layers
        zlev = np.zeros((args.num_layer+1, ssh.size), dtype=float)
        zref = np.zeros((args.num_layer+1, ssh.size), dtype=float)
    
        zlev[0, :] = ssh
        for k in range(0, args.num_layer):
            zlev[k+1, :] = zlev[k, :] + \
                (args.bot_depth + ssh) / args.num_layer
            zref[k+1, :] = zref[k, :] + \
                (args.bot_depth + 0.0) / args.num_layer

    init["minLevelCell"] = xarray.ones_like(xcel, dtype=np.int32)
    init["maxLevelCell"] = xarray.ones_like(xcel, dtype=np.int32) * args.num_layer

    init["refTopDepth"] = (
        ("nVertLevels"), np.zeros(args.num_layer))
    init["refZMid"] = (
        ("nVertLevels"), np.zeros(args.num_layer))
    init["refBottomDepth"] = (
        ("nVertLevels"), np.zeros(args.num_layer))
    init["refInterfaces"] = (
        ("nVertLevelsP1"), np.zeros(args.num_layer + 1))
    init["vertCoordMovementWeights"] = (
        ("nVertLevels"), np.ones(args.num_layer))

    init["refTopDepth"][:] = zref[:-1, 0].T
    init["refZMid"][:] = 0.5 * (zref[1:, 0].T + zref[:-1, 0].T)
    init["refBottomDepth"][:] = zref[1:, 0].T
    init["refInterfaces"][:] = zref[:, 0].T

    init["restingThickness"] = (
        ("nCells", "nVertLevels"), np.zeros((xcel.size, args.num_layer)))
    init["layerThickness"] = (
        ("Time", "nCells", "nVertLevels"), 
            np.zeros((1, mesh.xCell.size, args.num_layer)))
    init["zTop"] = (
        ("Time", "nCells", "nVertLevels"), 
            np.zeros((1, mesh.xCell.size, args.num_layer)))
    init["zMid"] = (
        ("Time", "nCells", "nVertLevels"),
            np.zeros((1, mesh.xCell.size, args.num_layer)))

    init["restingThickness"][:, :] = zref[1:, :].T - zref[:-1, :].T
    init["layerThickness"][0, :, :] = zlev[1:, :].T - zlev[:-1, :].T
    init["zTop"][0, :, :] = -zlev[:-1, :].T
    init["zMid"][0, :, :] = -0.5 * (zlev[1:, :].T + zlev[:-1, :].T)
    zmid = init["zMid"][0, :, :]

    # temperature + salinity
    temp = args.temperature0 + \
        args.n_squared / args.alpha / GRAV * RHO0 * zmid

    print("trc. T-min. =", temp.min().values)
    print("trc. T-max. =", temp.max().values)

    ymid = 0. * zmid
    for k in range(0, args.num_layer):
        ymid[:, k] = ycel.values

    half = 0.5 * (ycel.min().values + ycel.max().values)

    salt = args.salinity0 + \
        args.m_squared / args.beta / GRAV * RHO0 * (ymid - half)

    print("trc. S-min. =", salt.min().values)
    print("trc. S-max. =", salt.max().values)

    init["temperature"] = (
        ("Time", "nCells", "nVertLevels"),
            np.zeros((1, mesh.xCell.size, args.num_layer)))
    init["temperature"][0, :, :] = temp.values

    init["salinity"] = (
        ("Time", "nCells", "nVertLevels"),
            np.zeros((1, mesh.xCell.size, args.num_layer)))
    init["salinity"][0, :, :] = salt.values

    # c-grid velocity + perturbation
    init["normalVelocity"] = (
        ("Time", "nEdges", "nVertLevels"),
            np.zeros((1, mesh.xEdge.size, args.num_layer)))

    ymin = init["yVertex"].min().values

    for iEdge in range(mesh.xEdge.size):
        # balanced flow
        cel1 = mesh["cellsOnEdge"].values[iEdge, 0] - 1
        flow = 0.01 * (args.bot_depth + zmid.values[cel1, :])

        # perturbations
        pert = PERT * (
            np.sin(2. * np.pi * (
                mesh.xEdge.values[iEdge] - 0.00) / 100.E+03) *
            np.sin(2. * np.pi * (
                mesh.yEdge.values[iEdge] - ymin) / 100.E+03)
            )

        scal = np.cos(mesh.angleEdge.values[iEdge])

        init["normalVelocity"][0, iEdge, :] += (flow + pert) * scal

    # coriolis parameter
    init["fCell"] = args.coriolis * xarray.ones_like(mesh.xCell)
    init["fEdge"] = args.coriolis * xarray.ones_like(mesh.xEdge)
    init["fVertex"] = args.coriolis * xarray.ones_like(mesh.xVertex)

    write_netcdf(init, args.case_name + "_init.nc")

    return init


def build_sfrc(args, mesh):

#-- Form surface forcing data
    sfrc = nc.Dataset(
        args.case_name + "_forcing.nc", "w")

    NUM_STEPS = 31 * 24 * 6 + 1  # every 10mins over 31 days

    time = np.linspace(0., 31. * 24. * 3600., NUM_STEPS)

    sfrc.createDimension("nCells", mesh.xCell.size)
    sfrc.createDimension("StrLen", 64)
    sfrc.createDimension("Time", None)

    avar = sfrc.createVariable(
        "atmosPressure", "f8", ("Time", "nCells"))
    avar[:, :] = np.zeros(
        (NUM_STEPS, mesh.xCell.size), dtype=float)

    uvar = sfrc.createVariable(
        "windSpeedU", "f8", ("Time", "nCells"))
    uvar[:, :] = np.zeros(
        (NUM_STEPS, mesh.xCell.size), dtype=float)

    vvar = sfrc.createVariable(
        "windSpeedV", "f8", ("Time", "nCells"))
    vvar[:, :] = np.zeros(
        (NUM_STEPS, mesh.xCell.size), dtype=float)

    ref_date = datetime.datetime.strptime(
        "0001-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")

    if (args.wind_stress in WMAG):
        wmag = WMAG[args.wind_stress]
    else:
        raise ValueError("Unsupported wind stress")

    print("Wind speed (m/s):", wmag)

    tstr = []
    for iTime in range(NUM_STEPS):
        # take ^.42 of time variation to compensate for
        # wind speed rather than stress parameterisation
        # in MPAS (a Garratt et al scheme)
        wind = wmag * wind_exp(
            np.sin(0.9 * args.coriolis * time[iTime]))

        uvar[iTime, :] = wind

        date = ref_date + \
            datetime.timedelta(seconds=np.float64(time[iTime]))
        dstr = date.strftime("%Y-%m-%d_%H:%M:%S"+45*" ")
        tstr.append(dstr)
    
    tvar = sfrc.createVariable("xtime", "S1", ("Time", "StrLen"))
    tstr = np.array(tstr, "S64")
    tvar[:, :] = nc.stringtochar(tstr)

    sfrc.close()

    """
    # a constant-in-time wind stress
    sfrc["windStressZonal"] = (
        ("Time", "nCells"), 
        1. * args.wind_stress * np.ones((1, mesh.xCell.size)))

    sfrc["windStressMeridional"] = (
        ("Time", "nCells"), 
        0. * args.wind_stress * np.ones((1, mesh.xCell.size)))

    write_netcdf(sfrc, args.case_name + "_forcing.nc")
    """

    return sfrc

# exp chosen to minimise rms difference between wind parameterisations
def wind_exp(x): return np.sign(x) * (np.abs(x) ** (0.42185))


if (__name__ == "__main__"): 
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        "--case-name", dest="case_name", type=str,
        required=True, help="Base name of case.")
    parser.add_argument(
        "--num-xcell", dest="num_xcell", type=int,
        required=True, help="Number of cells in x-dir.")
    parser.add_argument(
        "--num-ycell", dest="num_ycell", type=int,
        required=True, help="Number of cells in y-dir.")
    parser.add_argument(
        "--len-edges", dest="len_edges", type=float,
        required=True, help="Length of polygon edges [m].")
    
    parser.add_argument(
        "--num-layer", dest="num_layer", type=int,
        default=0,
        required=False, help="Number of layers in z-dir.")
    parser.add_argument(
        "--bot-depth", dest="bot_depth", type=float,
        default=50.,
        required=False, help="Depth of channel [m].")

    parser.add_argument(
        "--coriolis", dest="coriolis", type=float,
        default=1.E-04,
        required=False, help="Coriolis parameter [rad/s].")
    
    parser.add_argument(
        "--temperature0", dest="temperature0", type=float,
        default=25.,
        required=False, help="Reference temperature [deg].")
    parser.add_argument(
        "--alpha", dest="alpha", type=float,
        default=0.17,
        required=False, help="Thermal expansion coefficient.")
    parser.add_argument(
        "--n-squared", dest="n_squared", type=float,
        default=1.E-04,
        required=False, help="Stratification coefficient.")
    
    parser.add_argument(
        "--salinity0", dest="salinity0", type=float,
        default=35.,
        required=False, help="Reference salinity [psu].")
    parser.add_argument(
        "--beta", dest="beta", type=float,
        default=0.76,
        required=False, help="Haline expansion coefficient.")
    parser.add_argument(
        "--m-squared", dest="m_squared", type=float,
        default=1.E-06,
        required=False, help="Stratification coefficient.")

    parser.add_argument(
        "--wind-stress", dest="wind_stress", type=float,
        default=0.010,
        required=False, help="Zonal wind stress [m^2/s].")

    args = parser.parse_args()

    print("\n")

    mesh = build_mesh(args)
    init = build_init(args, mesh)
    sfrc = build_sfrc(args, mesh)
