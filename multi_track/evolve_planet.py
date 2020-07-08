import os
import numpy as np
import time
import sys
from astropy import constants as const
import multiprocessing as mp
import multiprocessing

from platypos import Planet_LoFo14
from platypos import Planet_Ot20
from platypos.planet_LoFo14_PAPER import Planet_LoFo14_PAPER
from platypos.planet_Ot20_PAPER import Planet_Ot20_PAPER
import platypos.planet_models_LoFo14 as plmoLoFo14

from platypos.lx_evo_and_flux import l_high_energy
from platypos.lx_evo_and_flux import undo_what_Lxuv_all_does
import platypos.mass_luminosity_relation as mlr


def evolve_one_planet(pl_folder_pair,
                      t_final, initial_step_size,
                      epsilon, K_on, beta_on,
                      evo_track_dict_list, path_for_saving):
    """
    Function evolves one planet (pl) at a time, but through all stellar
    evolutionary tracks specified in 'evo_track_dict_list' (see below
    for details).
    It also makes sure the results are saved in the correct
    folder-subfolder structure which must have been previously created!
    All calculations for one planet belong to each other (same host star
    initial parameters), but the planets themselves are independent of
    each other. Each time this function is called represents an
    independent process (important for multiprocessing).

    Parameters:
    -----------
    pl_folder_pair (list): a list with two elements, the first is the
                           folder name, the second the planet object
                           (e.g. ["planet1", pl_object1])

    t_final (float): end time of the integration (e.g. 3, 5, 10 Gyr)

    initial_step_size (float): Initial step size for the integration
                               (e.g. 0.1 Myr)

    epsilon (float): Evaporation efficiency (constant); should be a
                     value between 0 and 1 in the platypos framework

    K_on (str): "yes" to use K-estimation, or "no" to set K=1
                (see Beta_K_functions in platypos_package)

    beta_on (str): "yes" to use beta-estimation, or "no" to set beta=1
                   (see Beta_K_functions in platypos_package)

    evo_track_dict_list (list): list with track parameters (can be one
                                track only); either all 9 parameters per
                                track are specified, or a list of track
                                parameters which are different is passed.
                                This is for a sample where each planet
                                has a different host and the tracks have
                                to be tailored to the host star
                                mass/saturation luminosity.
    NOTE: there are TWO Options:
        (1) track dictionary contains all 9 parameters
            (e.g. if host star is the same for all planets in sample)
        (2) the track dictionary contains only 3 parameters (t_sat,\
            dt_drop, Lx_drop_factor) and the remaining parameters have
            to be tailored to each host star (t_start, t_curr, t_5Gyr,
            Lx_curr, Lx_5Gyr, Lx_max)

    path_for_saving (str): file path to master folder in which the
                           individual planet folders lie

    Returns:
    --------
    None
    """

    # Option 1
    if len(evo_track_dict_list[0]) == 9:
        # get planet object and folder to save results in
        pl = pl_folder_pair[1]
        folder = pl_folder_pair[0]
        # evolve the planet through all the complete tracks in
        # evo_track_dict_list
        for track in evo_track_dict_list:
            # set planet name based on specified track
            pl.set_name(
                t_final,
                initial_step_size,
                epsilon,
                K_on,
                beta_on,
                evo_track_dict=track)
            # generate a useful planet name for saving the results
            pl_file_name = folder + "_track" + \
                pl.planet_id.split("_track")[1] + ".txt"

            # check if result exists, if not evolve planet
            if not os.path.isdir(path_for_saving + pl_file_name):
                pl.evolve_forward_and_create_full_output(
                    t_final,
                    initial_step_size,
                    epsilon,
                    K_on,
                    beta_on,
                    evo_track_dict=track,
                    path_for_saving=path_for_saving,
                    planet_folder_id=folder)

                # calculate radius at t_final if planet would undergo ONLY
                # thermal contraction and save to file
                try: 
                    R_thermal = plmoLoFo14.calculate_planet_radius(
                        pl.core_mass, pl.fenv, t_final, pl.flux, pl.metallicity)
                    filename = folder + "_thermal_contr.txt"

                    if not os.path.exists(path_for_saving + filename):
                        with open(path_for_saving + filename, "w") as t:
                            file_content = "t_final,R_th\n" \
                                + str(t_final) + "," \
                                + str(R_thermal)
                        t.write(file_content)
                except:
                    pass


    # Option 2
    else:
        # calculate missing track parameters based on individual planet
        # (e.g. in a sample where host star mass is different for each planet)
        # Need to set Lx_1Gyr & Lx_5Gyr based on the Lx_sat approximation used
        # We scale the parameters at 1 & 5 Gyr up and down for non-solar mass
        # stars, based on the spread in Lx_sat compared to the one for the Sun
        # NOTE: in this scenario, t_curr and t_5Gyr are hardcoded to 1 & 5 Gyr!
        # If you want a track with these values different, you need to pass a
        # complete dictionary

        pl = pl_folder_pair[1]
        folder = pl_folder_pair[0]

        # here I need to select how Lx_sat was calculated. This info should be
        # already stored in pl.Lx_sat_info
        # (this is needed to scale Lx_1Gyr and Lx_5Gyr correctly)
        # There are currently 4 options:
        #     1) "OwWu17": use L_XUV as given in Owen & Wu 2017
        #                  (NOTE: their XUV values seem to be very low!)
        #     2) "OwWu17_X=HE": use Owen & Wu 2017 XUV saturation value
        #                       as X-ray saturation value
        #     3) "Tu15": from Tu et al. (2015); they use
        #                               Lx/Lbol = 10^(-3.13)*(L_bol_star)
        #     4) "1e-3": simple approximation of saturation regime:
        #                               Lx/Lbol ~ 10^(-3))

        try:
            Lx_calculation = pl.Lx_sat_info
        except:
            raise ValueError("No info about how to calculate Lx_sat specified!")

        if Lx_calculation == "OwWu17":
            # OwWu 17
            # NOTE: since platypos expects a X-ray saturation luminosity,
            # and not an XUV sat. lum, for the OwWu17 formula we need to
            # invert Lxuv_all, so that we get the corresponding Lx for all Lxuv
            Lx_age_OW17 = undo_what_Lxuv_all_does(
                L_high_energy(pl.age, pl.mass_star))
            Lx_1Gyr = undo_what_Lxuv_all_does(L_high_energy(1e3, pl.mass_star))
            Lx_5Gyr = undo_what_Lxuv_all_does(L_high_energy(5e3, pl.mass_star))
            same_track_params = {
                "t_start": pl.age,
                "t_curr": 1e3,
                "t_5Gyr": 5e3,
                "Lx_curr": Lx_1Gyr,
                "Lx_5Gyr": Lx_5Gyr,
                "Lx_max": pl.Lx_age}

        elif Lx_calculation == "OwWu17_X=HE":
            # OwWu 17, with Lx,sat = L_HE,sat
            Lx_age_OW17_HE = L_high_energy(pl.age, pl.mass_star)
            Lx_1Gyr = L_high_energy(1e3, pl.mass_star)
            Lx_5Gyr = L_high_energy(5e3, pl.mass_star)
            same_track_params = {
                "t_start": pl.age,
                "t_curr": 1e3,
                "t_5Gyr": 5e3,
                "Lx_curr": Lx_1Gyr,
                "Lx_5Gyr": Lx_5Gyr,
                "Lx_max": pl.Lx_age}

        elif Lx_calculation == "Tu15":
            # need Mass-Luminosity relation to estimate L_bol based on the
            # stellar mass (NOTE: we use a MS M-R raltion and ignore any
            # PRE-MS evolution
            mass_luminosity_relation = mlr.mass_lum_relation_mamajek()
            Lbol = 10**mass_luminosity_relation(pl.mass_star)
            Lx_sat_Tu15 = 10**(-3.13) * Lbol * const.L_sun.cgs.value
            # for the Tu15 tracks we scale the hardcoded values for the
            # SUN at 1 & 5 Gyr up and down based on the difference in a
            # star's Lx_sat as compared to the Sun
            scaling_factor = Lx_sat_Tu15 / \
                (10**(-3.13) * (1.0) * const.L_sun.cgs.value)
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

        elif Lx_calculation == "1e-3":
            # need Mass-Luminosity relation to estimate L_bol based on the
            # stellar mass (NOTE: we use a MS M-R raltion and ignore any
            # PRE-MS evolution
            mass_luminosity_relation = mlr.mass_lum_relation_mamajek()
            Lbol = 10**mass_luminosity_relation(pl.mass_star)
            Lx_age_1e3 = 10**(-3.0) * Lbol * const.L_sun.cgs.value
            scaling_factor = Lx_age_1e3 / \
                (10**(-3) * (1.0) * const.L_sun.cgs.value)
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
        for t in evo_track_dict_list:
            track = same_track_params.copy()
            track.update(t)

            # set planet name based on specified track
            pl.set_name(t_final, initial_step_size, epsilon,
                        K_on, beta_on, evo_track_dict=track)
            # generate a useful planet name for saving the results
            pl_file_name = folder + "_track" + \
                pl.planet_id.split("_track")[1] + ".txt"

            # check if result exists, if not evolve planet
            if not os.path.isdir(path_for_saving + pl_file_name):
                pl.evolve_forward_and_create_full_output(
                    t_final,
                    initial_step_size,
                    epsilon,
                    K_on,
                    beta_on,
                    evo_track_dict=track,
                    path_for_saving=path_for_saving,
                    planet_folder_id=folder)

                try:
                    # calculate radius at t_final if planet would undergo ONLY
                    # thermal contraction and save to file
                    R_thermal = plmoLoFo14.calculate_planet_radius(
                        pl.core_mass, pl.fenv, t_final, pl.flux, pl.metallicity)
                    filename = folder + "_thermal_contr.txt"
                    if not os.path.exists(path_for_saving + filename):
                        with open(path_for_saving + filename, "w") as t:
                            file_content = "t_final,R_th\n"\
                                + str(t_final) + ","\
                                + str(R_thermal)
                            t.write(file_content)
                except:
                    pass


def evolve_one_planet_along_one_track(folder_planet_track,
                                      t_final, initial_step_size,
                                      epsilon, K_on, beta_on,
                                      path_for_saving):
    """
    Function evolves one planet (pl) at a time through one stellar
    evolutionary track.
    It also makes sure the results are saved in the correct
    folder-subfolder structure  which must have been previously created!
    Each time this function is called represents an independent process
    (important for multiprocessing).

    Parameters:
    -----------
    folder_planet_track (list): a list with three elements, the first is
                                the folder name, the second the planet
                                object, the third is a track dictionary
                                (e.g. ["planet1", pl_object1, track_dict1])

    NOTE: track_dict is a dictionary with either all 9 parameters per
          track are specified, or a list of track parameters which are
          different is passed. This is for a sample where each planet
          has a different host and the tracks have to be tailored to the
          host star mass/saturation luminosity.
          There are TWO Options:
                (1) track dictionary contains all 9 parameters
                (e.g. if host star is the same for all planets in sample)
                (2) the track dictionary contains only 3 parameters (t_sat,
                dt_drop, Lx_drop_factor) and the remaining parameters
                have to be tailored to each host star (t_start, t_curr,
                t_5Gyr,Lx_curr, Lx_5Gyr, Lx_max)

    t_final (float): end time of the integration (e.g. 3, 5, 10 Gyr)

    initial_step_size (float): Initial step size for the integration
                                                   (e.g. 0.1 Myr)

    eepsilon (float): Evaporation efficiency (constant); should be a
                     value between 0 and 1 in the platypos framework

    K_on (str): "yes" to use K-estimation, or "no" to set K=1
                (see Beta_K_functions in platypos_package)

    beta_on (str): "yes" to use beta-estimation, or "no" to set beta=1
                   (see Beta_K_functions in platypos_package)

    path_for_saving (str): file path to master folder in which the
                                           individual planet folders lie

    Returns:
    --------
    None
    """

    # get planet object and folder to save results in, as well as the track
    # dictionary to evolve the star along
    folder = folder_planet_track[0]
    planet = folder_planet_track[1]
    track_dict = folder_planet_track[2]

    # Option 1
    if len(track_dict) == 9:
        # set planet name based on track params
        planet.set_name(
            t_final,
            initial_step_size,
            epsilon,
            K_on,
            beta_on,
            evo_track_dict=track_dict)
        # generate a useful planet name for saving the results
        pl_file_name = folder + "_track" + \
            planet.planet_id.split("_track")[1] + ".txt"

        # check if result exists, if not evolve planet
        if not os.path.isdir(path_for_saving + pl_file_name):
            planet.evolve_forward_and_create_full_output(
                t_final,
                initial_step_size,
                epsilon,
                K_on,
                beta_on,
                evo_track_dict=track_dict,
                path_for_saving=path_for_saving,
                planet_folder_id=folder)

            try:
                # calculate radius at t_final if planet would undergo ONLY
                # thermal contraction and save to file
                R_thermal = plmoLoFo14.calculate_planet_radius(planet.core_mass,
                                                               planet.fenv,
                                                               t_final,
                                                               planet.flux,
                                                               planet.metallicity)
                filename = folder + "_thermal_contr.txt"
                if not os.path.exists(path_for_saving + filename):
                    with open(path_for_saving + filename, "w") as t:
                        file_content = "t_final,R_th\n"\
                            + str(t_final) + ","\
                            + str(R_thermal)
                        t.write(file_content)
            except:
                pass

    # Option 2
    else:
        # calculate missing track parameters based on individual planet
        # (e.g. in a sample where host star mass is different for each planet)
        # Need to set Lx_1Gyr & Lx_5Gyr based on the Lx_sat approximation used
        # We scale the parameters at 1 & 5 Gyr up and down for non-solar mass
        # stars, based on the spread in Lx_sat compared to the one for the Sun
        # NOTE: in this scenario, t_curr and t_5Gyr are hardcoded to 1 & 5 Gyr!
        # If you want a track with these values different, you need to pass a
        # complete dictionary

        # here I need to select how Lx_sat was calculated. This info should be
        # already stored in pl.Lx_sat_info
        # (this is needed to scale Lx_1Gyr and Lx_5Gyr correctly)
        # There are currently 4 options:
        #     1) "OwWu17": use L_XUV as given in Owen & Wu 2017
        #                  (NOTE: their XUV values seem to be very low!)
        #     2) "OwWu17_X=HE": use Owen & Wu 2017 XUV saturation value
        #                       as X-ray saturation value
        #     3) "Tu15": from Tu et al. (2015); they use
        #                               Lx/Lbol = 10^(-3.13)*(L_bol_star)
        #     4) "1e-3": simple approximation of saturation regime:
        #                               Lx/Lbol ~ 10^(-3))

        try:
            Lx_calculation = planet.Lx_sat_info
        except:
            raise ValueError("No info about how to calculate Lx_sat specified!")
            
        if Lx_calculation == "OwWu17":
            # OwWu 17
            # NOTE: since platypos expects a X-ray saturation luminosity,
            # and not an XUV sat. lum, for the OwWu17 formula we need to
            # invert Lxuv_all, so that we get the corresponding Lx for all Lxuv
            Lx_age_OW17 = undo_what_Lxuv_all_does(
                L_high_energy(planet.age, planet.mass_star))
            Lx_1Gyr = undo_what_Lxuv_all_does(
                L_high_energy(1e3, planet.mass_star))
            Lx_5Gyr = undo_what_Lxuv_all_does(
                L_high_energy(5e3, planet.mass_star))
            same_track_params = {
                "t_start": planet.age,
                "t_curr": 1e3,
                "t_5Gyr": 5e3,
                "Lx_curr": Lx_1Gyr,
                "Lx_5Gyr": Lx_5Gyr,
                "Lx_max": planet.Lx_age}

        elif Lx_calculation == "OwWu17_X=HE":
            # OwWu 17, with Lx,sat = L_HE,sat
            Lx_age_OW17_HE = L_high_energy(planet.age, planet.mass_star)
            Lx_1Gyr = L_high_energy(1e3, planet.mass_star)
            Lx_5Gyr = L_high_energy(5e3, planet.mass_star)
            same_track_params = {
                "t_start": planet.age,
                "t_curr": 1e3,
                "t_5Gyr": 5e3,
                "Lx_curr": Lx_1Gyr,
                "Lx_5Gyr": Lx_5Gyr,
                "Lx_max": planet.Lx_age}

        elif Lx_calculation == "Tu15":
            # need Mass-Luminosity relation to estimate L_bol based on the
            # stellar mass (NOTE: we use a MS M-R raltion and ignore any
            # PRE-MS evolution
            mass_luminosity_relation = mlr.mass_lum_relation_mamajek()
            Lbol = 10**mass_luminosity_relation(planet.mass_star)
            Lx_sat_Tu15 = 10**(-3.13) * Lbol * const.L_sun.cgs.value
            # for the Tu15 tracks we scale the hardcoded values for the
            # SUN at 1 & 5 Gyr up and down based on the difference in a
            # star's Lx_sat as compared to the Sun
            scaling_factor = Lx_sat_Tu15 / \
                (10**(-3.13) * (1.0) * const.L_sun.cgs.value)
            # Lx value at 1 Gyr from Tu et al. (2015) model tracks (scaled)
            Lx_1Gyr = 2.10 * 10**28 * scaling_factor
            # Lx value at 5 Gyr from Tu et al. (2015) model tracks (scaled)
            Lx_5Gyr = 1.65 * 10**27 * scaling_factor
            same_track_params = {
                "t_start": planet.age,
                "t_curr": 1e3,
                "t_5Gyr": 5e3,
                "Lx_curr": Lx_1Gyr,
                "Lx_5Gyr": Lx_5Gyr,
                "Lx_max": planet.Lx_age}

        elif Lx_calculation == "1e-3":
            # need Mass-Luminosity relation to estimate L_bol based on the
            # stellar mass (NOTE: we use a MS M-R raltion and ignore any
            # PRE-MS evolution
            mass_luminosity_relation = mlr.mass_lum_relation_mamajek()
            Lbol = 10**mass_luminosity_relation(planet.mass_star)
            Lx_age_1e3 = 10**(-3.0) * Lbol * const.L_sun.cgs.value
            # we scale the hardcoded values for the SUN at 1 & 5 Gyr up and
            # down based on the difference in a star's Lx_sat as compared
            # to the Sun
            scaling_factor = Lx_age_1e3 / \
                (10**(-3) * (1.0) * const.L_sun.cgs.value)
            # Lx value at 1 Gyr from Tu et al. (2015) model tracks (scaled)
            Lx_1Gyr = 2.10 * 10**28 * scaling_factor
            # Lx value at 5 Gyr from Tu et al. (2015) model tracks (scaled)
            Lx_5Gyr = 1.65 * 10**27 * scaling_factor
            same_track_params = {
                "t_start": planet.age,
                "t_curr": 1e3,
                "t_5Gyr": 5e3,
                "Lx_curr": Lx_1Gyr,
                "Lx_5Gyr": Lx_5Gyr,
                "Lx_max": planet.Lx_age}

        # combine the same_track params with different track params
        # for a complete track dictionary and evolve planet
        track_dict_complete = same_track_params.copy()
        track_dict_complete.update(track_dict)
        # print(track_dict_complete)

        # set planet name based on specified track
        planet.set_name(
            t_final,
            initial_step_size,
            epsilon,
            K_on,
            beta_on,
            evo_track_dict=track_dict_complete)
        # generate a useful planet name for saving the results
        pl_file_name = folder + "_track" + \
            planet.planet_id.split("_track")[1] + ".txt"

        # check if result exists, if not evolve planet
        if not os.path.isdir(path_for_saving + folder + "/" + pl_file_name):
            planet.evolve_forward_and_create_full_output(
                t_final,
                initial_step_size,
                epsilon,
                K_on,
                beta_on,
                evo_track_dict=track_dict_complete,
                path_for_saving=path_for_saving,
                planet_folder_id=folder)

            try:
                # calculate radius at t_final if planet would undergo ONLY thermal
                # contraction and save to file
                R_thermal = plmoLoFo14.calculate_planet_radius(planet.core_mass,
                                                               planet.fenv,
                                                               t_final,
                                                               planet.flux,
                                                               planet.metallicity)
                filename = folder + "_thermal_contr.txt"
                if not os.path.exists(path_for_saving + filename):
                    with open(path_for_saving + filename, "w") as t:
                        file_content = "t_final,R_th\n"\
                            + str(t_final) + ","\
                            + str(R_thermal)
                        t.write(file_content)
            except:
                pass

        #print("Finished: ", planet.planet_id.rstrip(".txt"))


def next_chunk(list, chunk_size):
    """ function splits the input list into chuncks of size chunk_size.
    NOTE: what this function does is create and return a generator object.
    Generators are iterators, but a kind of iterable you can only iterate
    over once. So generators do not store all the values in memory, they
    generate the values on the fly.

    Parameters:
    -----------
    list: list which to split into smaller chuncks

    chunck_size: length of each sublist

    Return:
    -------
    generator object (i.e. a list of sublists which can be iterated over once)
    """

    for start in range(0, len(list), chunk_size):
        yield list[start:start + chunk_size]
