from PyAstronomy import pyasl
import pandas as pd
import numpy as np

from sklearn.neighbors import KernelDensity
import astropy.units as u
from astropy import constants as const


def get_ExoplanetEU_data():
    """use PyAstronomy to access the data provided by exoplanet.eu
    (https://pyastronomy.readthedocs.io/en/latest/pyaslDoc/resBasedDoc/exoplanetEU.html)

    NOTE: returned are only planets which have an orbital period and 
    radius, as well as a period < 300 days.

    Returns:
    --------
    df_planets_ (dataframe): dataframe of the whole dataset
    period (array): array of periods
    radius (array): array of radii
    distance (array): array of semi-major axes
    """
    # Instantiate exoplanetEU2 object (download all the planets)
    v = pyasl.ExoplanetEU2()
    # Export all data as a pandas DataFrame
    df_planets = v.getAllDataPandas()

    mask_nan = ~(
        np.isnan(
            df_planets["orbital_period"]) | np.isnan(
            df_planets["radius"]))
    mask_P = df_planets["orbital_period"] < 300
    # dataframe only with planets which have P & R
    df_planets_ = df_planets[mask_nan & mask_P]

    RJ_to_RE = (const.R_jup / const.R_earth).value
    period = df_planets_["orbital_period"].values
    radius = df_planets_["radius"].values * RJ_to_RE
    distance = df_planets_["semi_major_axis"].values

    return df_planets_, period, radius, distance


def gaussian_kernel_density_estimation(
        X_data, Y_data, baseX, baseY, bandwidth=0.25):
    """
    Function takes 2 dimensional data and constructs a Gaussian kernel
    density estimate of the distribution

    Parameters:
    -----------
    X_data (array): array of x-values
    Y_data (array): array of y-values
    baseX (int): base of log in x-data
    baseY (int): base of log in y-data

    Returns:
    --------
    X_grid, Y_grid (array): two arrays which span the grid on which to
    plot the density distribution (not in log-scale!)
    prob_density (array): the corresponding probability density values

    Plot command: ax.contourf(X_grid, Y_grid, prob_density, cmap="binary",
    						  levels=15, alpha=0.5)
    """
    # transform actual data to log space before proceeding
    baseX_data = np.log(X_data) / np.log(baseX)
    baseY_data = np.log(Y_data) / np.log(baseY)

    # make a grid (in log space - to match the data)
    X = np.arange(baseX_data.min(), baseX_data.max() + 1, 1)
    Y = np.arange(baseY_data.min(), baseY_data.max() + 1, 1)
    # for now the density of the grid is fixed in here/ hardcoded!
    XX, YY = np.mgrid[X.min():X.max():200j, Y.min():Y.max():200j]

    # now I have a grid of points covering my radii and periods
    XY_sample = np.vstack([XX.ravel(), YY.ravel()]).T
    # this is my data in grid-form
    XY_train = np.vstack([baseX_data, baseY_data]).T

    # construct a Gaussian kernel density estimate of the distribution
    kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    kde.fit(XY_train)  # fit/load real dataset

    # score_samples() returns the log-likelihood of the samples
    prob_density = np.exp(kde.score_samples(XY_sample))
    # nm_sample is the range/grid over which I compute the density based
    # on the data (2-D)
    prob_density = np.reshape(prob_density, XX.shape)

    X_grid = baseX**XX
    Y_grid = baseY**YY

    return X_grid, Y_grid, prob_density
