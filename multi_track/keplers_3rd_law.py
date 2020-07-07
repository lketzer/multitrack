from astropy import constants as const
from astropy import units as u
import numpy as np


def get_period_from_a(M_star, M_pl, a_pl):
    """ Assuming a circular orbit, get period if semi-major axis,
    host star and planet mass are known.
    """
    P_sec = (4 * np.pi**2 / (const.G.cgs * (M_star * const.M_sun.cgs +
                                            M_pl * const.M_earth.cgs)) * (a_pl * const.au.cgs)**3)**(0.5)
    return P_sec.value / (86400)  # in days


def get_a_from_period(M_star, M_pl, P):
    """ Assuming a circular orbit, get semi-major axis if period,
    host star and planet mass are known.
    """
    ahoch3 = (P * 86400 * u.s)**2 * (const.G.cgs * (M_star *
                                                    const.M_sun.cgs + M_pl * const.M_earth.cgs)) / (4 * np.pi**2)
    return ((ahoch3**(1 / 3)) / const.au.cgs).value  # in days


def get_period_from_a_no_Mpl(M_star, a_pl):
    """ Assuming a circular orbit and negligible planet mass, get period
    from semi-major axis and host star mass.
    """
    P_sec = (4 * np.pi**2 / (const.G.cgs * M_star * const.M_sun.cgs)
             * (a_pl * const.au.cgs)**3)**(0.5)
    return P_sec.value / (86400)  # in days


def get_a_from_period_no_Mpl(M_star, P):
    """ Assuming a circular orbit and negligible planet mass, get
    semi-major axis from period and host star mass.
    """
    a_power3 = (P * 86400 * u.s)**2 * (const.G.cgs *
                                       M_star * const.M_sun.cgs) / (4 * np.pi**2)
    return ((a_power3**(1 / 3)) / const.au.cgs).value  # in days
