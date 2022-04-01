from astropy import constants as const
from platypos.lx_evo_and_flux import l_high_energy
from platypos.lx_evo_and_flux import undo_what_Lxuv_all_does
import platypos.mass_luminosity_relation as mlr
from platypos.lx_evo_and_flux import l_high_energy#calculate_Lx_sat


def complete_track_dict(pl,
                        track_dict,
                        Lx_calculation="Tu15", 
                        ML_rel="ZAMS_Thomas",
                        Lx1Gyr="Jackson12"):
    """
    sometimes I have only the track parameters from my distribution tracks.
    (t_sat, t_drop, Lx_drop_factor)
    -> calculate rest of params based on stellar mass and the Lx,sat formula.
    
    These give X values: "Tu15", "Jo20", "1e-3"
    
    These papers give XUV values: "OwWu17", "Kuby20", "Ribas05", "WaDa18",
                                  "LoRi17"-> for evoluition, EUV_relation NEEDS
                                  to be SanzForcada!!!
    
    Params:
    -------
    planet (class obj): planet object
    track_dict (dict): dictionary with t_sat, t_drop, Lx_drop_factor
    Lx_calculation (str): options:
                         1) "OwWu17": use L_XUV as given in Owen & Wu 2017
                             (NOTE: their XUV values seem to be very low!)
                         2) "OwWu17_X=HE": use Owen & Wu 2017 XUV saturation
                             value as X-ray saturation value
                         3) "Tu15": from Tu et al. (2015); they use
                             Lx/Lbol = 10^(-3.13)*(L_bol_star)
                         4) "1e-3": simple approximation of saturation regime:
                             Lx/Lbol ~ 10^(-3))"
                         5) "Kuby20": 
                         6) "Jo20": Rx_sat = 5.135 * 1e-4
                                    Lx_sat = Rx_sat * (Lbol * const.L_sun.cgs.value)
                         
    ML_rel (str): "ZAMS_Thomas", or "MS_Mamajeck"
    Lx1Gyr (str): default = "Jackson12" - uses the median Lx value for G-type
                  stars from the 620 Myr cluster from Jackson 2012 and the Lx-decay
                  slope from that paper to estimate the Lx value at 1 and 5 Gyr
                  (necessary for creating the tracks);
                  if something else, use the values from the Tu paper -> steeper
                  slope and lower Lx-values!
                  Jackson 2012 cluster at 620 Myr: Lx_med = 1e+29, age_med = 620.8
                  slope = -1.13
                  log10Lx = slope * (np.log10(5e3)-np.log10(age_med)) + np.log10(Lx_med)
                  Lx_5Gyr = 10**log10Lx
                  f = scipy.interpolate.interp1d([np.log10(age_med), np.log10(5e3)], [np.log10(Lx_med), np.log10(5e3)])
                  Lx_1Gyr = 10**float(f(np.log10(1e3)))
                  Lx_5Gyr = 10**float(f(np.log10(5e3)))
    
    Return:
    -------
    track (dict): full track dictionary with 9 parameters
    """

    if (Lx_calculation == "OwWu17") or (Lx_calculation == "Kuby20") \
        or (Lx_calculation == "Ribas05")  or (Lx_calculation == "WaDa18") \
        or (Lx_calculation == "LoRi17"):
        # NOTE: since platypos expects a X-ray saturation luminosity,
        # and not an XUV sat. lum, for the these calculations we need to
        # invert Lxuv_all, so that we get the corresponding Lx for all Lxuv
        # then platpos needs to be run with relationEUV="SanzForcada"!!!
        # calculate Lx_1Gyr and Lx_5Gyr based on the evolution given in one of
        # the given papers (1 stellar track)
        Lx_1Gyr = undo_what_Lxuv_all_does(l_high_energy(1e3, pl.mass_star,
                                                        paper=Lx_calculation+"XUV",
                                                        ML_rel=ML_rel))
        Lx_5Gyr = undo_what_Lxuv_all_does(l_high_energy(5e3, pl.mass_star,
                                                        paper=Lx_calculation+"XUV",
                                                        ML_rel=ML_rel))
        
    elif Lx_calculation == "Tu15":
        # need Mass-Luminosity relation to estimate L_bol based on the
        # stellar mass (NOTE: we use a MS M-R raltion and ignore any PRE-MS evo
        Lx_sat = l_high_energy(1.0, pl.mass_star, paper="Tu15", ML_rel=ML_rel)
        Lx_sat_sun = l_high_energy(1.0, 1.0, paper="Tu15", ML_rel=ML_rel)
        # for the Tu15 tracks we scale the hardcoded values for the
        # SUN at 1 & 5 Gyr up and down based on the difference in a
        # star's Lx_sat as compared to the Sun
        scaling_factor = Lx_sat / Lx_sat_sun
        
        if Lx1Gyr == "Jackson12":
            # Jackson 2012 cluster at 620 Myr
            Lx_1Gyr = 5.834924518159396e+28 * scaling_factor
            Lx_5Gyr = 9.466711397257275e+27 * scaling_factor
        else:
            # Lx value at 1 Gyr from Tu et al. (2015) model tracks (scaled)
            Lx_1Gyr = 2.10 * 10**28 * scaling_factor
            # Lx value at 5 Gyr from Tu et al. (2015) model tracks (scaled)
            Lx_5Gyr = 1.65 * 10**27 * scaling_factor

    elif Lx_calculation == "1e-3":
        # need Mass-Luminosity relation to estimate L_bol based on the
        # stellar mass (NOTE: we use a MS M-R raltion and ignore any
        # PRE-MS evolution
        Lx_sat = l_high_energy(1.0, pl.mass_star, paper="1e-3", ML_rel=ML_rel)
        Lx_sat_sun = l_high_energy(1.0, 1.0, paper="1e-3", ML_rel=ML_rel)
        scaling_factor = Lx_sat / Lx_sat_sun
        
        if Lx1Gyr == "Jackson12":
            # Jackson 2012 cluster at 620 Myr
            Lx_1Gyr = 5.834924518159396e+28 * scaling_factor
            Lx_5Gyr = 9.466711397257275e+27 * scaling_factor
        else:
            # Lx value at 1 Gyr from Tu et al. (2015) model tracks (scaled)
            Lx_1Gyr = 2.10 * 10**28 * scaling_factor
            # Lx value at 5 Gyr from Tu et al. (2015) model tracks (scaled)
            Lx_5Gyr = 1.65 * 10**27 * scaling_factor
        
    elif Lx_calculation == "Jo20":
        # need Mass-Luminosity relation to estimate L_bol based on the
        # stellar mass (NOTE: we use a MS M-R raltion and ignore any
        # PRE-MS evolution
        Lx_sat = l_high_energy(1.0, pl.mass_star, paper="Johnstone20",
                               ML_rel=ML_rel)
        Lx_sat_sun = l_high_energy(1.0, 1.0, paper="Johnstone20",
                                   ML_rel=ML_rel)
        scaling_factor = Lx_sat / Lx_sat_sun
        
        if Lx1Gyr == "Jackson12":
            # Jackson 2012 cluster at 620 Myr
            Lx_1Gyr = 5.834924518159396e+28 * scaling_factor
            Lx_5Gyr = 9.466711397257275e+27 * scaling_factor
        else:
            # Lx value at 1 Gyr from Tu et al. (2015) model tracks (scaled)
            Lx_1Gyr = 2.10 * 10**28 * scaling_factor
            # Lx value at 5 Gyr from Tu et al. (2015) model tracks (scaled)
            Lx_5Gyr = 1.65 * 10**27 * scaling_factor

    same_track_params = {
        "t_start": pl.age,
        "t_curr": 1e3,
        "t_5Gyr": 5e3,
        "Lx_curr": Lx_1Gyr,
        "Lx_5Gyr": Lx_5Gyr,
        "Lx_max": pl.Lx_age}

    # combine the same_track params with different track params for a
    # complete track dictionary and evolve planet through all tracks
    # in evo_track_dict_list

    track = same_track_params.copy()
    track.update(track_dict)
        
    return track


def complete_track_dict_noplobj(age, mass_star, track_dict,
                                Lx_calculation="Tu15", ML_rel="ZAMS_Thomas",
                                Lx1Gyr="Jackson12"):
    """ sometimes I have only the track parameters from my distribution tracks.
    (t_sat, t_drop, Lx_drop_factor)
    -> calculate rest of params based on stellar mass and the Lx,sat formula.
    
    NOTE: for the dict to be accurate, the age needs to be such that the star
    is still in its saturation phase (so <~60 Myr.)
    
    Params:
    -------
    age (float): in Myr
    mass_star (float): in solar units
    Lbol (float): in solar units
    track_dict (dict): dictionary with t_sat, t_drop, Lx_drop_factor
    Lx_calculation (str): options for calculating the *Lx_sat*:
                         1) "OwWu17": use L_XUV as given in Owen & Wu 2017
                             (NOTE: their XUV values seem to be very low!)
                         2) "OwWu17_X=HE": use Owen & Wu 2017 XUV saturation
                             value as X-ray saturation value
                         3) "Tu15": from Tu et al. (2015); they use
                             Lx/Lbol = 10^(-3.13)*(L_bol_star)
                         4) "1e-3": simple approximation of saturation regime:
                             Lx/Lbol ~ 10^(-3))"
                             
    ML_rel (str): "ZAMS_Thomas", or "MS_Mamajeck"
    Lx1Gyr (str): default = "Jackson12" - uses the median Lx value for G-type
                  stars from the 620 Myr cluster from Jackson 2012 and the Lx-decay
                  slope from that paper to estimate the Lx value at 1 and 5 Gyr
                  (necessary for creating the tracks);
                  if something else, use the values from the Tu paper -> steeper
                  slope and lower Lx-values!
                  Jackson 2012 cluster at 620 Myr: Lx_med = 1e+29, age_med = 620.8
                  slope = -1.13
                  log10Lx = slope * (np.log10(5e3)-np.log10(age_med)) + np.log10(Lx_med)
                  Lx_5Gyr = 10**log10Lx
                  f = scipy.interpolate.interp1d([np.log10(age_med), np.log10(5e3)], [np.log10(Lx_med), np.log10(5e3)])
                  Lx_1Gyr = 10**float(f(np.log10(1e3)))
                  Lx_5Gyr = 10**float(f(np.log10(5e3)))
    Return:
    -------
    track (dict): full track dictionary with 9 parameters
    """

    if (Lx_calculation == "OwWu17") or (Lx_calculation == "Kuby20") \
        or (Lx_calculation == "Ribas05")  or (Lx_calculation == "WaDa18") \
        or (Lx_calculation == "LoRi17"):
        # NOTE: since platypos expects a X-ray saturation luminosity,
        # and not an XUV sat. lum, for the these calculations we need to
        # invert Lxuv_all, so that we get the corresponding Lx for all Lxuv
        # then platpos needs to be run with relationEUV="SanzForcada"!!!
        # calculate Lx_1Gyr and Lx_5Gyr based on the evolution given in one of
        # the given papers (1 stellar track)
        Lx_sat = l_high_energy(1.0, mass_star, paper=Lx_calculation+"XUV")
        Lx_1Gyr = undo_what_Lxuv_all_does(l_high_energy(1e3, mass_star,
                                                        paper=Lx_calculation+"XUV",
                                                        ML_rel=ML_rel))
        Lx_5Gyr = undo_what_Lxuv_all_does(l_high_energy(5e3, mass_star,
                                                        paper=Lx_calculation+"XUV",
                                                        ML_rel=ML_rel))
    
#     elif Lx_calculation == "OwWu17_X=HE":
#         # OwWu 17, with Lx,sat = L_HE,sat
#         Lx_sat = l_high_energy(1.0, mass_star, paper="OwWu17_X=HE",
#                                ML_rel=ML_rel)
#         Lx_1Gyr = l_high_energy(1e3, mass_star, paper="OwWu17_X=HE",
#                                 ML_rel=ML_rel)
#         Lx_5Gyr = l_high_energy(5e3, mass_star, paper="OwWu17_X=HE",
#                                 ML_rel=ML_rel)

    elif Lx_calculation == "Tu15":
        Lx_sat = l_high_energy(1.0, mass_star, paper="Tu15", ML_rel=ML_rel)
        Lx_sat_sun = l_high_energy(1.0, 1.0, paper="Tu15", ML_rel=ML_rel)
        # for the Tu15 tracks we scale the hardcoded values for the
        # SUN at 1 & 5 Gyr up and down based on the difference in a
        # star's Lx_sat as compared to the Sun
        scaling_factor = Lx_sat / Lx_sat_sun
        
        if Lx1Gyr == "Jackson12":
            # Jackson 2012 cluster at 620 Myr
            Lx_1Gyr = 5.834924518159396e+28 * scaling_factor
            Lx_5Gyr = 9.466711397257275e+27 * scaling_factor
        else:
            # Lx value at 1 Gyr from Tu et al. (2015) model tracks (scaled)
            Lx_1Gyr = 2.10 * 10**28 * scaling_factor
            # Lx value at 5 Gyr from Tu et al. (2015) model tracks (scaled)
            Lx_5Gyr = 1.65 * 10**27 * scaling_factor

    elif Lx_calculation == "1e-3":
        Lx_sat = l_high_energy(1.0, mass_star, paper="1e-3", ML_rel=ML_rel)
        Lx_sat_sun = l_high_energy(1.0, 1.0, paper="1e-3", ML_rel=ML_rel)
        scaling_factor = Lx_sat / Lx_sat_sun
        
        if Lx1Gyr == "Jackson12":
            Lx_1Gyr = 5.834924518159396e+28 * scaling_factor
            Lx_5Gyr = 9.466711397257275e+27 * scaling_factor
        else:
            # Lx value at 1 Gyr from Tu et al. (2015) model tracks (scaled)
            Lx_1Gyr = 2.10 * 10**28 * scaling_factor
            # Lx value at 5 Gyr from Tu et al. (2015) model tracks (scaled)
            Lx_5Gyr = 1.65 * 10**27 * scaling_factor
    
    elif Lx_calculation == "Jo20":
        Lx_sat = l_high_energy(1.0, mass_star, paper="Johnstone20",
                               ML_rel=ML_rel)
        Lx_sat_sun = l_high_energy(1.0, 1.0, paper="Johnstone20",
                                  ML_rel=ML_rel)
        scaling_factor = Lx_sat / Lx_sat_sun
        
        if Lx1Gyr == "Jackson12":
            # Jackson 2012 cluster at 620 Myr
            Lx_1Gyr = 5.834924518159396e+28 * scaling_factor
            Lx_5Gyr = 9.466711397257275e+27 * scaling_factor
        else:
            # Lx value at 1 Gyr from Tu et al. (2015) model tracks (scaled)
            Lx_1Gyr = 2.10 * 10**28 * scaling_factor
            # Lx value at 5 Gyr from Tu et al. (2015) model tracks (scaled)
            Lx_5Gyr = 1.65 * 10**27 * scaling_factor

    same_track_params = {"t_start": age,
                         "t_curr": 1e3,
                         "t_5Gyr": 5e3,
                         "Lx_curr": Lx_1Gyr,
                         "Lx_5Gyr": Lx_5Gyr,
                         "Lx_max": Lx_sat}
    
    # combine the same_track params with different track params for a
    # complete track dictionary and evolve planet through all tracks
    # in evo_track_dict_list

    track = same_track_params.copy()
    track.update(track_dict)
        
    return track
