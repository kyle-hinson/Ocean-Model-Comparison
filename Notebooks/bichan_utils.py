import matplotlib.pyplot as plt
import cmocean.cm as cmo
import numpy as np
import os


def to_rho(var, grid=None):
    if grid is None:
        grid = var.grid
    if var.dims[-1] != 'xi_rho':
        var = grid.interp(var, 'X', to='center', boundary='extend')
    if var.dims[-2] != 'eta_rho':
        var = grid.interp(var, 'Y', to='center', boundary='extend')
    return var


def add_derivatives(ds, grid, q='salt'):
    
    qs = ds[q]
    
    #############################
    # Flow and property gradients
    
    ds['dqdx'] = to_rho(grid.derivative(qs, 'X'), grid)    # defined at rho-points
    ds['dqdy'] = to_rho(grid.derivative(qs, 'Y'), grid)    # defined at rho-points
    
    ds['dudx'] = grid.derivative(ds.u, 'X', boundary='extend')  # defined at rho-points
    ds['dvdy'] = grid.derivative(ds.v, 'Y', boundary='extend')  # defined at rho-points
    ds['dvdx'] = to_rho(grid.derivative(ds.v, 'X', boundary='extend'), grid)  # defined at rho-points
    ds['dudy'] = to_rho(grid.derivative(ds.u, 'Y', boundary='extend'), grid)  # defined at rho-points
    
    ###########################
    # Invariant flow properties
    
    # Vorticity:  v_x - u_y
    ds['zeta'] = (ds.dvdx - ds.dudy)/ds.f
    ds['zeta'].name = 'Normalized vorticity'

    # Divergence: u_x + v_y
    ds['delta'] = (ds.dudx + ds.dvdy)/ds.f
    ds['delta'].name = 'Normalized divergence'

    # Major axis of deformation
    ds['alpha'] = ( np.sqrt( (ds.dudx-ds.dvdy)**2 + (ds.dvdx+ds.dudy)**2 ) )/ds.f
    ds['alpha'].name = 'Normalized total strain'

    ##################################
    # Principle deformation components

    ds['lminor'] = 0.5 * (ds.delta - ds.alpha)
    ds['lminor'].name = 'lambda minor'

    ds['lmajor'] = 0.5 * (ds.delta + ds.alpha)
    ds['lmajor'].name = 'lambda major'
    
    #############################################
    # Along- and cross-frontal velocity gradients
    
    # angle is wrt x, so need to do arctan2(y, x)
    ds['phi_cf'] = np.arctan2(ds.dqdy, ds.dqdx)
    ds['phi_af'] = ds.phi_cf + np.pi/2.0

    ds['du_cf'] = ( ds.dudx*np.cos(ds.phi_cf)**2 + ds.dvdy*np.sin(ds.phi_cf)**2 
               + (ds.dudy + ds.dvdx)*np.sin(ds.phi_cf)*np.cos(ds.phi_cf) )/ds.f

    ds['du_af'] = ( ds.dudx*np.cos(ds.phi_af)**2 + ds.dvdy*np.sin(ds.phi_af)**2
              + (ds.dudy + ds.dvdx)*np.sin(ds.phi_af)*np.cos(ds.phi_af) )/ds.f
    
    ############################
    # The frontogenesis function
    
    # Dimensional frontogenesis function
    Dgradq_i = - ds.dudx*ds.dqdx - ds.dvdx*ds.dqdy
    Dgradq_j = - ds.dudy*ds.dqdx - ds.dvdy*ds.dqdy
    ds['Ddelq2'] = (ds.dqdx*Dgradq_i + ds.dqdy*Dgradq_j)
    ds['Ddelq2'].name = 'Frontogenesis function'

    # Density gradients squared
    ds['gradq2'] = ds.dqdx**2 + ds.dqdy**2
    ds['gradq2'].name = r'$(\nabla q)^2$'

    # Normalized frontogenesis function
    ds['nFGF'] = 0.5 * ds.Ddelq2 / (ds.gradq2 * ds.f)
    ds['nFGF'].name = r'Normalized Frontogenesis Function'
    
    return ds


def pl33tn(x, dt=1.0, T=33.0, mode="valid", t=None):
    """
    Computes low-passed series from `x` using pl33 filter, with optional
    sample interval `dt` (hours) and filter half-amplitude period T (hours)
    as input for non-hourly series.

    The PL33 filter is described on p. 21, Rosenfeld (1983), WHOI
    Technical Report 85-35.  Filter half amplitude period = 33 hrs.,
    half power period = 38 hrs.  The time series x is folded over
    and cosine tapered at each end to return a filtered time series
    xf of the same length.  Assumes length of x greater than 67.

    Can input a DataArray and use dask-supported for lazy execution. In that
    use case, dt is ignored and calculated from the input DataArray.
    cf-xarray is also required.

    Examples
    --------
    >>> from oceans.filters import pl33tn
    >>> import matplotlib.pyplot as plt
    >>> t = np.arange(500)  # Time in hours.
    >>> x = 2.5 * np.sin(2 * np.pi * t / 12.42)
    >>> x += 1.5 * np.sin(2 * np.pi * t / 12.0)
    >>> x += 0.3 * np.random.randn(len(t))
    >>> filtered_33 = pl33tn(x, dt=4.0)  # 33 hr filter
    >>> filtered_33d3 = pl33tn(x, dt=4.0, T=72.0)  # 3 day filter
    >>> fig, ax = plt.subplots()
    >>> (l1,) = ax.plot(t, x, label="original")
    >>> pad = [np.nan] * 8
    >>> (l2,) = ax.plot(t, np.r_[pad, filtered_33, pad], label="33 hours")
    >>> pad = [np.nan] * 17
    >>> (l3,) = ax.plot(t, np.r_[pad, filtered_33d3, pad], label="3 days")
    >>> legend = ax.legend()


    """

    # import cf_xarray  # noqa: F401
    # import pandas as pd
    # import xarray as xr

    if isinstance(x, xr.Dataset | pd.DataFrame):
        raise TypeError("Input a DataArray not a Dataset, or a Series not a DataFrame.")

    if isinstance(x, pd.Series) and not isinstance(
        x.index,
        pd.core.indexes.datetimes.DatetimeIndex,
    ):
        raise TypeError("Input Series needs to have parsed datetime indices.")

    # find dt in units of hours
    if isinstance(x, xr.DataArray):
        dt = (x.cf["T"][1] - x.cf["T"][0]) / np.timedelta64(
            3_600_000_000_000,
        )
    elif isinstance(x, pd.Series):
        dt = (x.index[1] - x.index[0]) / pd.Timedelta("1H")

    pl33 = np.array(
        [
            -0.00027,
            -0.00114,
            -0.00211,
            -0.00317,
            -0.00427,
            -0.00537,
            -0.00641,
            -0.00735,
            -0.00811,
            -0.00864,
            -0.00887,
            -0.00872,
            -0.00816,
            -0.00714,
            -0.00560,
            -0.00355,
            -0.00097,
            +0.00213,
            +0.00574,
            +0.00980,
            +0.01425,
            +0.01902,
            +0.02400,
            +0.02911,
            +0.03423,
            +0.03923,
            +0.04399,
            +0.04842,
            +0.05237,
            +0.05576,
            +0.05850,
            +0.06051,
            +0.06174,
            +0.06215,
            +0.06174,
            +0.06051,
            +0.05850,
            +0.05576,
            +0.05237,
            +0.04842,
            +0.04399,
            +0.03923,
            +0.03423,
            +0.02911,
            +0.02400,
            +0.01902,
            +0.01425,
            +0.00980,
            +0.00574,
            +0.00213,
            -0.00097,
            -0.00355,
            -0.00560,
            -0.00714,
            -0.00816,
            -0.00872,
            -0.00887,
            -0.00864,
            -0.00811,
            -0.00735,
            -0.00641,
            -0.00537,
            -0.00427,
            -0.00317,
            -0.00211,
            -0.00114,
            -0.00027,
        ],
    )

    _dt = np.linspace(-33, 33, 67)

    dt = float(dt) * (33.0 / T)

    filter_time = np.arange(0.0, 33.0, dt, dtype="d")
    Nt = len(filter_time)
    filter_time = np.hstack((-filter_time[-1:0:-1], filter_time))

    pl33 = np.interp(filter_time, _dt, pl33)
    pl33 /= pl33.sum()

    if isinstance(x, xr.DataArray):
        x = x.interpolate_na(dim=x.cf["T"].name)

        weight = xr.DataArray(pl33, dims=["window"])
        xf = (
            x.rolling({x.cf["T"].name: len(pl33)}, center=True)
            .construct({x.cf["T"].name: "window"})
            .dot(weight, dims="window")
        )
        # update attrs
        attrs = {
            key: f"{value}, filtered"
            for key, value in x.attrs.items()
            if key != "units"
        }
        xf.attrs = attrs

    elif isinstance(x, pd.Series):
        xf = x.to_frame().apply(np.convolve, v=pl33, mode=mode)

        # nan out edges which are not good values anyway
        if mode == "same":
            xf[: Nt - 1] = np.nan
            xf[-Nt + 2 :] = np.nan

    else:  # use numpy
        xf = np.convolve(x, pl33, mode=mode)

        # times to match xf
        if t is not None:
            # Nt = len(filter_time)
            tf = t[Nt - 1 : -Nt + 1]
            return xf, tf

        # nan out edges which are not good values anyway
        if mode == "same":
            xf[: Nt - 1] = np.nan
            xf[-Nt + 2 :] = np.nan

    return xf