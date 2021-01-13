import os
import pandas as pd
import numpy as np

import astropy.units as u
from astropy import constants as const

import platypos.planet_models_LoFo14 as plmoLoFo14
from platypos.mass_loss_rate_function import mass_loss_rate_forward_LO14_shortshort
from platypos.lx_evo_and_flux import l_xuv_all
from platypos.lx_evo_and_flux import flux_at_planet
from platypos.beta_K_functions import beta_fct
import multitrack.keplers_3rd_law as kepler3


def read_results_file(path, filename):
    """Function to read in the results file for an individual track. """

    df = pd.read_csv(path + filename, float_precision='round_trip')
    # NOTE: set float_precision to avoid pd doing sth. weird to the last digit
    # Pandas uses a dedicated dec 2 bin converter that compromises accuracy
    # in preference to speed. Passing float_precision='round_trip' to read_csv
    # fixes this.
    return df


def read_in_PLATYPOS_results_fullEvo(path_to_results, N_tracks):
    """ Call this function to read in ALL the results. Need to specify
    how many tracks have been evolved. This allows you to also run the
    function while the planet ensemble is still evolving. (e.g. if you
    want to check how much is left).
    Returns a dictionary with FULL evolution results (all tracks)! 

    Parameters:
    -----------
    path_to_results (str): path to the master folder which containts all
                           the results for a single run (i.e. all the
                           subfolders for the individual planets)

    N_tracks (int): total number of tracks specified when running PLATYPOS;
                    only planets with complete output are used (this i
                    mainly so that I can read in and look at the results,
                    even though Platypos is still running)

    Returns:
    --------
    planet_df_dict (dict): dictionary of results (final parameters),
                           with planet names as keys, and corresponding
                           results-dataframes as values (results-dataframes
                           have N_tracks*4 columns [t1,M1,R1,Lx1, etc..])

    tracks_dict (dict): dictionary with track infos; keys are planet
                        names, values are a list (of length N_tracks)
                        with the track parameters [track_number, full
                        track name, evolved_off flag (True of False)
                        NOTE: the last parameter is only important for
                        MESA planets]

    planet_init_dict (dict): dictionary of initial planet parameters,
                             with planet names are keys, and the intial
                             parameters values (intial planet parameters
                             for LoFo14 planets are: semi-major axis - a,
                             M_init: mass, R_init: radius,
                             M_core: mass_core, age0: age, where a is in
                             AU, mass, radius and core mass in Earth
                             units, age in Myr)
    """

    files = os.listdir(path_to_results)
    print("Total # of planet folders = ", len(files))
    # check for empty folders (where maybe sth went wrong, or where planet has
    # not evolved yet)
    non_empty_folders = []
    for f in files:
        if len(os.listdir(path_to_results + f)) == 0:
            pass
        elif len([file for file in os.listdir(path_to_results + f) if f in file and "track" in file]) == 0:
            # this means no output file has been produced by PLATYPOS for any
            #  of the tracks (the filenames start with f and contain "track)")
            pass
        else:
            non_empty_folders.append(f)
    print("Non-empty folders: ", len(non_empty_folders))

    # Next, read in the results for each planet dictionary of results, with 
    # planet names as keys, and corresponding results-dataframes as values
    planet_df_dict = {}
    tracks_dict = {}  # dictionary with planet names as keys, and list of 
                      # track infos for each planet folder as values
    planet_init_dict = {}  # dictionary of initial planet parameters with 
                           # planet names are keys, parameters the values

    for i, f in enumerate(non_empty_folders):
        # f: folder name
        # get all files in one planet directory
        all_files_in_f = [
            f for f in os.listdir(
                path_to_results +
                f) if not f.startswith('.')]
        # NOTE: I had an instance where there were hidden files in a folder,
        # so this is to make sure hidden files (starting with ".") are ignored!

        # make list of only files which contain calculation results
        # (they start with the planet-folder name & contain "track", and do not
        # contain these other terms -> those are other files in the directory)
        result_files = [file for file in all_files_in_f if ("track" in file)
                        and ("track_params" not in file)
                        and ("evolved_off" not in file)
                        and ("final" not in file)]

        # sort result files first by t_sat, then by dt_drop, then by 
        # Lx_drop_factor; this is to have some order in the results
        # e.g. planet_001_track_10.0_240.0_5000.0_2e+30_0.0_0.0.txt
        # planet_name    t_start t_sat t_fina  Lx_sat  Lx_drop  Lx_drop_factor
        result_files_sorted = sorted(sorted(sorted(result_files,
                                                   key=lambda x: float(x.rstrip(".txt").split("_")[-1])),
                                            key=lambda x: float(x.rstrip(".txt").split("_")[-2])),
                                     key=lambda x: float(x.rstrip(".txt").split("_")[-5]))

        # skip planets which do not have all tracks available! (important for
        # when reading in results while Platypos is still running)
        # number of tracks for which results are available
        N_tracks_subfolder = len(result_files_sorted)
        # only read in results in a single planet folder if all track-results
        # are available
        if N_tracks_subfolder == N_tracks:
            # get file with initial planet params (name: f+".txt")
            df_pl = pd.read_csv(
                path_to_results + f + "/" + f + ".txt",
                float_precision='round_trip')
            planet_init_dict[f] = df_pl.values[0]  # add to dictionary

            # build dataframe with results from all tracks
            df_all_tracks = pd.DataFrame()
            # read in the results file for each track one by one and build up
            # one single (one-row) dataframe per planet
            for file in result_files_sorted:
                df_i = read_results_file(path_to_results, f + "/" + file)
                df_all_tracks = pd.concat([df_all_tracks, df_i], axis=1)
                # df.reset_index(level=0)

            # set the column names of the new dataframe (i goes from 1 to
            # N_tracks+1)
            col_names = []
            for i in range(1, int(len(df_all_tracks.columns) / 4) + 1):
                col_names.append("t" + str(i))
                col_names.append("M" + str(i))
                col_names.append("R" + str(i))
                col_names.append("Lx" + str(i))
            df_all_tracks.columns = col_names

            # now I have a dataframe for each planet, which contains all the
            # final parameters for each evolutionary track
            # add to dictionary with planet name as key
            planet_df_dict[f] = df_all_tracks

            # NEXT, read in track names in case I need this info later, for 
            # example for knowing the properties of each track
            # track number corresponds to t1,2,3, etc..
            track_dict = {}
            # flag planets which have moved off (only important for MESA
            # planets for now)
            list_planets_evolved_off = [
                "track" +
                file.rstrip(".txt").split("track")[1] for file in all_files_in_f if "evolved_off" in file]
            track_info_list = []
            for i, file in enumerate(result_files_sorted):
                if "track" + \
                        file.rstrip(".txt").split("track")[1] in list_planets_evolved_off:
                    # if track name in list of planets which has evolved off,
                    # set flag to True
                    track_evooff = True
                else:
                    track_evooff = False
                # contains all the info for each track
                track_info_list.append(
                    (str(i + 1), file[len(f + "_"):].rstrip(".txt"), track_evooff))
            # add track_dictionary for each planet to a master-track dictionary
            tracks_dict[f] = track_info_list

        else:
            print(
                "Tracks for ",
                f,
                " avaiable: ",
                str(N_tracks_subfolder) +
                "/" +
                str(N_tracks))

    # Lastly, convert the planet_init_dict to a dataframe
    planet_init_df = pd.DataFrame.from_dict(
        planet_init_dict, orient='index', columns=df_pl.columns)

    # now I can easily access the planet name, semi-major axies, core mass,
    # initial mass and radius -> all info is stored in a beautiful dataframe
    print("\nTotal number of planets to analyze: ", len(planet_init_df))

    return planet_df_dict, planet_init_df, tracks_dict


def read_in_PLATYPOS_results_final(path_to_results, N_tracks):
    """ 
    Call this function to read in the final results. Need to specify
    how many tracks have been evolved. This allows you to also run the
    function while the planet ensemble is still evolving. (e.g. if you
    want to check how much is left)
    
    NOTE: This is in principle the same as read_in_PLATYPOS_results, but 
    instead of reading in the whole radius & mass evolution file, it
    reads in the "final"-file, which contains only the last parameters
    of the whole mass & radius evolution-file. (faster!) 

    Parameters:
    -----------
    path_to_results (str): path to the master folder which containts all
                           the results for a single run (i.e. all the
                           subfolders for the individual planets)

    N_tracks (int): total number of tracks specified when running PLATYPOS;
                    only planets with complete output are used (this i
                    mainly so that I can read in and look at the results,
                    even though Platypos is still running)

    Returns:
    --------
    planet_all (dataframe): dataframe of initial & final parameters,
                            with planet names as indices

    tracks_dict (dict): dictionary with track infos; keys are planet
                        names, values are a list (of length N_tracks)
                        with the track parameters [track_number, full
                        track name, evolved_off flag (True of False)
                        NOTE: the last parameter is only important for
                        MESA planets]
    """
    
    files = os.listdir(path_to_results)
    print("Total # of planet folders = ", len(files))
    # check for empty folders (where maybe sth went wrong, or where planet has
    # not evolved yet)
    non_empty_folders = []
    for f in files:
        if len(os.listdir(path_to_results + f)) == 0:
            pass
        elif len([file for file in os.listdir(path_to_results + f) if f in file and "track" in file]) == 0:
            # this means no output file has been produced by PLATYPOS for any
            #  of the tracks (the filenames start with f and contain "track)")
            pass
        else:
            non_empty_folders.append(f)
    print("Non-empty folders: ", len(non_empty_folders))

    # Next, read in the results for each planet dictionary of results, with 
    # planet names as keys, and corresponding results-dataframes as values
    planet_df_dict = {}
    tracks_dict = {}  # dictionary with planet names as keys, and list of 
                      # track infos for each planet folder as values
    # dictionary of initial planet parameters with planet names are keys,
    # parameters the values
    planet_init_dict = {}

    for i, f in enumerate(non_empty_folders):
        # f: folder name
        # get all files in one planet directory
        all_files_in_f = [
            f for f in os.listdir(
                path_to_results +
                f) if not f.startswith('.')]
        # NOTE: I had an instance where there were hidden files in a folder,
        # so this is to make sure hidden files (starting with ".") are ignored!

        # make list of only files which contain calculation results
        # (they start with the planet-folder name & contain "track", and do not
        # contain these other terms -> those are other files in the directory)
        result_files = [file for file in all_files_in_f if ("final" in file) and ("track_params" not in file)]

        # sort result files first by t_sat, then by dt_drop, then by 
        # Lx_drop_factor; this is to have some order in the results
        # e.g. planet_001_track_10.0_240.0_5000.0_2e+30_0.0_0.0.txt
        # planet_name    t_start t_sat t_fina  Lx_sat  Lx_drop  Lx_drop_factor
        result_files_sorted = sorted(sorted(sorted(result_files,
                                                   key=lambda x: float(x.rstrip("_final.txt").split("_")[-1])),
                                            key=lambda x: float(x.rstrip("_final.txt").split("_")[-2])),
                                     key=lambda x: float(x.rstrip("_final.txt").split("_")[-5]))

        # skip planets which do not have all tracks available! (important for
        # when reading in results while Platypos is still running)
        # number of tracks for which results are available
        N_tracks_subfolder = len(result_files_sorted)
        # only read in results in a single planet folder if all track-results
        # are available
        if N_tracks_subfolder == N_tracks:
            # get file with initial planet params (name: f+".txt")
            df_pl = pd.read_csv(
                path_to_results + f + "/" + f + ".txt",
                float_precision='round_trip')
            planet_init_dict[f] = df_pl.values[0]  # add to dictionary

            # build dataframe with results from all tracks
            df_final = pd.DataFrame()
            # read in the results file for each track one by one and build up
            # one single (one-row) dataframe per planet
            for file in result_files_sorted:
                df_i = pd.read_csv(path_to_results + f + "/" + file, float_precision='round_trip')
                df_i.drop(["a", "core_mass", "metallicity", "track"], axis=1, inplace=True)
                df_final = pd.concat([df_final, df_i], axis=1)
                #df_final.reset_index(level=0)
                
            # set the column names of the new dataframe (i goes from 1 to
            # N_tracks+1)
            col_names = []
            for i in range(1, int(len(df_final.columns) / 4) + 1):
                col_names.append("time" + str(i))
                col_names.append("fenv" + str(i))
                col_names.append("mass" + str(i))
                col_names.append("radius" + str(i))
                #col_names.append("Lx" + str(i))
            df_final.columns = col_names

            # now I have a dataframe for each planet, which contains all the
            # final parameters for each evolutionary track
            # add to dictionary with planet name as key
            planet_df_dict[f] = df_final.loc[0]

            # NEXT, read in track names in case I need this info later, for 
            # example for knowing the properties of each track
            # track number corresponds to t1,2,3, etc..
            track_dict = {}
            # flag planets which have moved off (only important for MESA
            # planets for now)
            list_planets_evolved_off = [
                "track" +
                file.rstrip("_final.txt").split("track")[1] for file in all_files_in_f if "evolved_off" in file]
            track_info_list = []
            for i, file in enumerate(result_files_sorted):
                if "track" + \
                        file.rstrip("_final.txt").split("track")[1] in list_planets_evolved_off:
                    # if track name in list of planets which has evolved off,
                    # set flag to True
                    track_evooff = True
                else:
                    track_evooff = False
                # contains all the info for each track
                track_info_list.append(
                    (str(i + 1), file[len(f + "_"):].rstrip("_final.txt"), track_evooff))
            # add track_dictionary for each planet to a master-track dictionary
            tracks_dict[f] = track_info_list

        else:
            print(
                "Tracks for ",
                f,
                " avaiable: ",
                str(N_tracks_subfolder) +
                "/" +
                str(N_tracks))

    # Lastly, convert the planet_init_dict to a dataframe
    planet_init_df = pd.DataFrame.from_dict(
        planet_init_dict, orient='index', columns=df_pl.columns)
    
    planet_final_df = pd.DataFrame.from_dict(
        planet_df_dict, orient='index', columns=df_final.columns)

    # merge initial and final dataframe into one
    planet_all = pd.merge(planet_init_df, planet_final_df, left_index=True, right_index=True, how='outer')
    
    # now I can easily access the planet name, semi-major axies, core mass,
    # initial mass and radius -> all info is stored in a beautiful dataframe
    print("\nTotal number of planets to analyze: ", len(planet_init_df))

    return planet_all, tracks_dict


def read_in_host_star_parameters(path_to_results):
    """ Function to read in the initial stellar parameters for each
    planet in the population.

    Parameter:
    ----------
    path_to_results (str): path to the master folder which containts all
                           the results for a single run

    Returns:
    --------
    star_df (dataframe): dataframe with planet id's as indices, and all
                         initial host star parameters as columns
    """

    files = os.listdir(path_to_results)

    # only use non-empty folders
    non_empty_folders = []
    for f in files:
        if len(os.listdir(path_to_results + f)) == 0:
            pass
        elif len([file for file in os.listdir(path_to_results + f) if f in file and "track" in file]) == 0:
            # this means no output file has been produced by PLATYPOS for any
            # of the tracks
            pass
        else:
            non_empty_folders.append(f)
    print("Non-empty folders: ", len(non_empty_folders))

    # add all host star parameters to a dictionary (with planet names
    # as keys, and star parameters as values)
    star_dict = {}
    for i, f in enumerate(non_empty_folders):
        # f: folder name
        # get all files in one planet directory
        all_files_in_f = [
            f for f in os.listdir(
                path_to_results +
                f) if not f.startswith('.')]
        df_star = pd.read_csv(
            path_to_results +
            f +
            "/" +
            "host_star_properties" +
            ".txt",
            float_precision='round_trip')
        star_dict[f] = df_star.values[0]

    # convert the planet_init_dict to a dataframe
    star_df = pd.DataFrame.from_dict(
        star_dict, orient='index', columns=df_star.columns)

    return star_df


def read_in_thermal_contraction_radius(path_to_results):
    """ Function to read in the radius at the end time of the
    simulation if planet undergoes only thermal contraction.
    (No evaporation!)

    Parameter:
    ----------
    path_to_results (str): path to the master folder which containts all
                           the results for a single run

    Returns:
    --------
    df_thermal (dataframe): dataframe with planet id's as indices, and all
                            thermal contraction radii
    """

    files = os.listdir(path_to_results)

    # only use non-empty folders
    non_empty_folders = []
    for f in files:
        if len(os.listdir(path_to_results + f)) == 0:
            pass
        elif len([file for file in os.listdir(path_to_results + f) if f in file and "thermal" in file]) == 0:
            # this means no output file has been produced by PLATYPOS for any
            # of the tracks
            pass
        else:
            non_empty_folders.append(f)
    print("Non-empty folders: ", len(non_empty_folders))

    # add all host star parameters to a dictionary (with planet names
    # as keys, and star parameters as values)
    pl_dict = {}
    for i, f in enumerate(non_empty_folders):
        # f: folder name
        # get all files in one planet directory
        all_files_in_f = [
            f for f in os.listdir(
                path_to_results +
                f) if not f.startswith('.')]
        df_thermal = pd.read_csv(
            path_to_results +
            f +
            "/" +
            f +
            "_thermal_contr.txt",
            float_precision='round_trip')
        pl_dict[f] = df_thermal.values[0]

    # convert the planet_init_dict to a dataframe
    df_thermal = pd.DataFrame.from_dict(
        pl_dict, orient='index', columns=df_thermal.columns)

    return df_thermal


def read_in_PLATYPOS_results_dataframe(path_to_results, N_tracks,
                                       return_second_to_last):
    """
    Calls read_in_PLATYPOS_results_fullEvo & then does some more 
    re-aranging to the data to make it easier to handle.

    Parameters:
    -----------
    path_to_results (str): path to the master folder which containts all
                           the results for a single run (i.e. all the 
                           subfolders for the individual planets)

    N_tracks (int): total number of tracks specified when running PLATYPOS;
                    only planets with complete output are used (this is 
                    mainly so that I can read in and look at the results,
                    even though Platypos is still running)

    Returns:
    --------
    planet_all_df (dataframe): dataframe with planet id's as indices and
                               initial and final parameters (for all the
                               tracks) as columns

    tracks_dict (dict): dictionary with planet id's as keys, and list of
                        all tracks and their parameters as values
                        
    if return_second_to_last == True:
    
    planet_all_df_2nd_to_last (dataframe): dataframe with planet id's as indices
                                           and
                               initial and second to last final parameters (for all the
                               tracks) as columns
    """
    # call read_in_PLATYPOS_results
    planet_df_dict, planet_init_df, tracks_dict = read_in_PLATYPOS_results_fullEvo(
        path_to_results, N_tracks)

    # create a master dictionary which contains all final (!) parameters
    planet_final_dict = {}
    planet_final_dict_2nd = {}
    for key_pl, df_pl in planet_df_dict.items():
        # number of tracks for which there is a result file available
        N_tracks = int(len(df_pl.columns) / 4)

        # now I need to check for each track what the index of the last 
        # non-nan value is (if planet has moved outside of grid, or has
        # reached the stopping condition, PLATYPOS terminates & returnes
        # the final planetary parameters. Problem: This might not always
        # be at the same time step for each track! So when I read in all
        # the tracks into one dataframe, the remaining rows will be filled
        # with NaNs)

        df_final = pd.DataFrame()
        for i in range(1, N_tracks + 1):
            # return index for last non-NA/null value
            final_index = df_pl["M" + str(i)].last_valid_index()
            # get corresponding final time, mass & radius, add to df_final
            df_final.at[0, "t" + str(i)] = df_pl["t" + str(i)].loc[final_index]
            df_final.at[0, "R" + str(i)] = df_pl["R" + str(i)].loc[final_index]
            df_final.at[0, "M" + str(i)] = df_pl["M" + str(i)].loc[final_index]
            df_final.at[0, "Lx" + str(i)] = df_pl["Lx" + str(i)].loc[final_index]
        df_final.reset_index(drop=True)
        planet_final_dict[key_pl] = df_final.values[0]
        
        if return_second_to_last == True:
            df_final_2nd = pd.DataFrame()
            for i in range(1, N_tracks + 1):
                # return index for last non-NA/null value
                final_index = df_pl["M" + str(i)].last_valid_index()
                # get corresponding final time, mass & radius, add to df_final
                df_final_2nd.at[0, "t" + str(i)] = df_pl["t" +
                                                     str(i)].loc[final_index - 1]
                df_final_2nd.at[0, "R" + str(i)] = df_pl["R" +
                                                     str(i)].loc[final_index - 1]
                df_final_2nd.at[0, "M" + str(i)] = df_pl["M" +
                                                     str(i)].loc[final_index - 1]
                df_final_2nd.at[0, "Lx" + str(i)] = df_pl["Lx" +
                                                     str(i)].loc[final_index - 1]
            df_final_2nd.reset_index(drop=True)
            planet_final_dict_2nd[key_pl] = df_final_2nd.values[0]

    # convert master dictionary to dataframe with planet id's as indices
    planet_final_df = pd.DataFrame.from_dict(
        planet_final_dict, orient='index', columns=df_final.columns)

    # concatenate planet_init_df and planet_final_df into one master dataframe
    planet_all_df = pd.concat(
        [planet_init_df, planet_final_df], axis=1, sort=False)
    # together with tracks_dict this allows me to select, filter, analyze any
    # track and any planet

    # For MESA planets only: to make life easier, add evolved_off info to 
    # my master dataframe
    # add N_tracks more columns to dataframe planet_all_df which indicate
    # whether the planet on given track has evolved off or not
    for key_pl in planet_all_df.index.values:
        track_number = [a_tuple[0] for a_tuple in tracks_dict[key_pl]]
        track_evooff = [a_tuple[2] for a_tuple in tracks_dict[key_pl]]
        for number, evooff in zip(track_number, track_evooff):
            planet_all_df.at[key_pl, "track" + str(number)] = evooff
    
    if return_second_to_last == True:
            
        # convert master dictionary to dataframe with planet id's as indices
        planet_final_df_2nd = pd.DataFrame.from_dict(
            planet_final_dict_2nd, orient='index', columns=df_final_2nd.columns)

        # concatenate planet_init_df and planet_final_df into one master dataframe
        planet_all_df_2nd = pd.concat(
            [planet_init_df, planet_final_df_2nd], axis=1, sort=False)
        # together with tracks_dict this allows me to select, filter, analyze any
        # track and any planet

        # For MESA planets only: to make life easier, add evolved_off info to 
        # my master dataframe
        # add N_tracks more columns to dataframe planet_all_df which indicate
        # whether the planet on given track has evolved off or not
        for key_pl in planet_all_df_2nd.index.values:
            track_number = [a_tuple[0] for a_tuple in tracks_dict[key_pl]]
            track_evooff = [a_tuple[2] for a_tuple in tracks_dict[key_pl]]
            for number, evooff in zip(track_number, track_evooff):
                planet_all_df_2nd.at[key_pl, "track" + str(number)] = evooff

        return planet_all_df, planet_all_df_2nd, tracks_dict
    
    else:
        return planet_all_df, tracks_dict

    
def read_in_PLATYPOS_results_dataframe_second_to_last_time(
        path_to_results, N_tracks):
    """
    OBSOLETE!
    The same as read_in_PLATYPOS_results_dataframe, but returns a dataframe 
    with the final parameters one time step before Platypos terminated! 
    This is to analyze the behavior of my code.
    """

    # call read_in_PLATYPOS_results
    planet_df_dict, planet_init_df, tracks_dict = read_in_PLATYPOS_results_fullEvo(
        path_to_results, N_tracks)

    # create a master dictionary which contains all final (!) parameters
    planet_final_dict = {}
    for key_pl, df_pl in planet_df_dict.items():
        # number of tracks for which there is a result file available
        N_tracks = int(len(df_pl.columns) / 4)

        # now I need to check for each track what the index of the last 
        # non-nan value is (if planet has moved outside of grid, or has
        # reached the stopping condition, PLATYPOS terminates & returnes
        # the final planetary parameters. Problem: This might not always
        # be at the same time step for each track! So when I read in all
        # the tracks into one dataframe, the remaining rows will be filled
        # with NaNs)

        df_final = pd.DataFrame()
        for i in range(1, N_tracks + 1):
            # return index for last non-NA/null value
            final_index = df_pl["M" + str(i)].last_valid_index()
            # get corresponding final time, mass & radius, add to df_final
            df_final.at[0, "t" + str(i)] = df_pl["t" +
                                                 str(i)].loc[final_index - 1]
            df_final.at[0, "R" + str(i)] = df_pl["R" +
                                                 str(i)].loc[final_index - 1]
            df_final.at[0, "M" + str(i)] = df_pl["M" +
                                                 str(i)].loc[final_index - 1]
        df_final.reset_index(drop=True)
        planet_final_dict[key_pl] = df_final.values[0]

    # convert master dictionary to dataframe with planet id's as indices
    planet_final_df = pd.DataFrame.from_dict(
        planet_final_dict, orient='index', columns=df_final.columns)

    # concatenate planet_init_df and planet_final_df into one master dataframe
    planet_all_df = pd.concat(
        [planet_init_df, planet_final_df], axis=1, sort=False)
    # together with tracks_dict this allows me to select, filter, analyze any
    # track and any planet

    # For MESA planets only: to make life easier, add evolved_off info to my
    # master dataframe
    # add N_tracks more columns to dataframe planet_all_df which indicate
    # whether the planet on given track has evolved off or not
    for key_pl in planet_all_df.index.values:
        track_number = [a_tuple[0] for a_tuple in tracks_dict[key_pl]]
        track_evooff = [a_tuple[2] for a_tuple in tracks_dict[key_pl]]
        for number, evooff in zip(track_number, track_evooff):
            planet_all_df.at[key_pl, "track" + str(number)] = evooff

    return planet_all_df, tracks_dict


def get_complete_results(path_to_results, N_tracks):
    """ Combine everything into one dataframe (star & planet parameters), 
    and calculate some additional parameters, which might be handy to have.
    NOTE: two dataframes are returned, one with the parameters at the simulation
    end time, and an equivalent one but at one time step before.
    
    Returns:
    df_run, df_run_2nd, tracks_dict
    """
    planet_all_df, planet_all_df_2nd, tracks_dict = read_in_PLATYPOS_results_dataframe(
                                                                path_to_results, N_tracks, 
                                                                return_second_to_last=True)
    track_list = ["track"+str(i) for i in range(1, N_tracks+1)] 
    planet_all_df.drop(labels=track_list, axis=1, inplace=True)
    planet_all_df_2nd.drop(labels=track_list, axis=1, inplace=True)
    
    star_df = read_in_host_star_parameters(path_to_results)
    star_df.drop(["age"], axis=1, inplace=True)
    # more than one track:
    for i in range(1, N_tracks+1): 
        # calculate initial and final envelope mass
        M_env_init = planet_all_df["mass"] - planet_all_df["core_mass"] 
        M_env_final = planet_all_df["M"+str(i)] - planet_all_df["core_mass"]
        # calculate final envelope mass fraction
        planet_all_df["fenv_final"+str(i)] = (M_env_final / planet_all_df["M"+str(i)]) * 100
        # calculate fraction of envelope lost
        planet_all_df["frac_of_env_lost"+str(i)] = (M_env_init - M_env_final) / M_env_init
        
        # calculate initial and final envelope mass (for 2nd-to-last dataframe)
        M_env_init_2nd = planet_all_df_2nd["mass"] - planet_all_df_2nd["core_mass"] 
        M_env_final_2nd = planet_all_df_2nd["M"+str(i)] - planet_all_df_2nd["core_mass"]
        # calculate final envelope mass fraction
        planet_all_df_2nd["fenv_final"+str(i)] = (M_env_final_2nd / planet_all_df_2nd["M"+str(i)]) * 100
        # calculate fraction of envelope lost
        planet_all_df_2nd["frac_of_env_lost"+str(i)] = (M_env_init_2nd - M_env_final_2nd) / M_env_init_2nd
    
    # add core radius and period to dataframe
    planet_all_df["core_radius"] = plmoLoFo14.calculate_core_radius(planet_all_df["core_mass"])    
    planet_all_df["period"] = kepler3.get_period_from_a(
                                    star_df["mass_star"][planet_all_df.index].values,
                                    planet_all_df["mass"].values, planet_all_df["a"].values)
    # combine planet- and host-star dataframe & add info about planetary incident flux
    df_run = pd.concat([planet_all_df, star_df], axis=1, sort=False)#, join='inner')
    FLUX_AT_EARTH = 1373. * 1e7 * (u.erg/u.s) / (100*u.cm*100*u.cm) # erg/s/cm^2
    df_run["flux"] = df_run["Lbol"] * const.L_sun.cgs.value\
                        / (4 * np.pi * (df_run["a"] * const.au.cgs.value)**2)
    df_run["flux_EARTH"] = df_run["Lbol"]*const.L_sun.cgs.value\
                        / (4 * np.pi * (df_run["a"] * const.au.cgs.value)**2)\
                        / FLUX_AT_EARTH
    
    # add core radius and period to dataframe
    planet_all_df_2nd["core_radius"] = plmoLoFo14.calculate_core_radius(planet_all_df_2nd["core_mass"])    
    planet_all_df_2nd["period"] = kepler3.get_period_from_a(
                                    star_df["mass_star"][planet_all_df_2nd.index].values,
                                    planet_all_df_2nd["mass"].values, planet_all_df_2nd["a"].values)
    # combine planet- and host-star dataframe & add info about planetary incident flux
    df_run_2nd = pd.concat([planet_all_df_2nd, star_df], axis=1, sort=False)#, join='inner')
    FLUX_AT_EARTH = 1373. * 1e7 * (u.erg/u.s) / (100*u.cm*100*u.cm) # erg/s/cm^2
    df_run_2nd["flux"] = df_run_2nd["Lbol"] * const.L_sun.cgs.value\
                        / (4 * np.pi * (df_run_2nd["a"] * const.au.cgs.value)**2)
    df_run_2nd["flux_EARTH"] = df_run_2nd["Lbol"]*const.L_sun.cgs.value\
                        / (4 * np.pi * (df_run_2nd["a"] * const.au.cgs.value)**2)\
                        / FLUX_AT_EARTH
    
    return df_run, df_run_2nd, tracks_dict


def calculate_mass_loss_rate_at_initial_and_final_timestep(
                                              run,
                                              epsilon,
                                              beta_on="yes",
                                              beta_cutoff=True,
                                              K_on="yes",
                                              relation_EUV='Linsky'):
    """ Takes a run-dataframe (from get_complete_results())
    and returns the same dataframe but with mass-loss rate at 
    initial and final timestep, and the mass-loss timescale at
    the last time step added as new columns.
    
    Input:
    ------
    run (dataframe): input is dataframe returned by get_complete_results() 
    epsilon (float): evaporation efficiency; use same value as used in
                     the simulation run
    beta_on (str):   beta calculation on or off ("yes"/"no"); use same value 
                     as used in the simulation run
    beta_cutoff (boolean): beta_cutoff on or off (True/False); use same value 
                     as used in the simulation run
    K_on (str):      K calculation on or off ("yes"/"no"); use same value 
                     as used in the simulation run
    relation_EUV (str): specify EUV relation for calculating L_XUV; use same
                     value as used in the simulation
    
    Returns:
    --------
    run (dataframe): same as imput but with new columns added.
                     - Mdot_final+track# [g/s], [M_earth/yr], [M_earth/Gyr]
                     - t_ML_final+track# [Myr], [Gyr]
                     - Mdot_init+track# [g/s], [M_earth/yr], [M_earth/Gyr]
    """
    # get number of tracks (can be e.g. 1 or 10)
    number_of_tracks = len([c for c in run.columns if c[0]=="R"])
    for track in range(1, number_of_tracks+1):
        
        # calculate the mass loss rate with initial/final parameters 
        # (at the first/last recorded time step) for each track
        Mdot_init_track = []
        Mdot_final_track = []
        for index, row in run.iterrows():
            
            Mdot_init = mass_loss_rate_forward_LO14_shortshort(
                                        t_=run.loc[index]["age"],
                                        epsilon=epsilon,
                                        K_on=K_on,
                                        beta_on=beta_on,
                                        distance=run.loc[index]["a"],
                                        R_p=run.loc[index]["radius"],
                                        M_p=run.loc[index]["mass"],
                                        f_env=run.loc[index]["fenv"],
                                        Lx_at_t_=run.loc[index]["Lx_age"],
                                        mass_star=run.loc[index]["mass_star"],
                                        beta_cutoff=beta_cutoff,
                                        relation_EUV=relation_EUV)
            Mdot_init_track.append(Mdot_init) # in g/s
            
            Mdot_final = mass_loss_rate_forward_LO14_shortshort(
                                        t_=run.loc[index]["t"+str(track)],
                                        epsilon=epsilon,
                                        K_on=K_on,
                                        beta_on=beta_on,
                                        distance=run.loc[index]["a"],
                                        R_p=run.loc[index]["R"+str(track)],
                                        M_p=run.loc[index]["M"+str(track)],
                                        f_env=run.loc[index]["fenv_final"+str(track)],
                                        Lx_at_t_=run.loc[index]["Lx"+str(track)],
                                        mass_star=run.loc[index]["mass_star"],
                                        beta_cutoff=beta_cutoff,
                                        relation_EUV=relation_EUV)
            Mdot_final_track.append(Mdot_final) # in g/s
        
        # append to initial dataframe [g/s and M_earth/Gyr]
        run["Mdot_init"+str(track)] = Mdot_init_track  
        run["Mdot_init"+str(track)+"M_earth_per_Gyr"] =\
                        run["Mdot_init"+str(track)] / (const.M_earth.cgs.value)\
                        * (86400.*365*1e9)
        run["Mdot_init"+str(track)+"M_earth_per_yr"] =\
                        run["Mdot_init"+str(track)] / (const.M_earth.cgs.value)\
                        * (86400.*365)
        
        run["Mdot_final"+str(track)] = Mdot_final_track  
        run["Mdot_final"+str(track)+"M_earth_per_Gyr"] =\
                        run["Mdot_final"+str(track)] / (const.M_earth.cgs.value)\
                        * (86400.*365*1e9)
        run["Mdot_final"+str(track)+"M_earth_per_yr"] =\
                        run["Mdot_final"+str(track)] / (const.M_earth.cgs.value)\
                        * (86400.*365)

        # final mass-loss timescale = final envelope mass divided by mass loss rate
        final_mass_loss_timescale = ((run["M"+str(track)] - run["core_mass"])\
                                     * const.M_earth.cgs.value)\
                                     / (-run["Mdot_final"+str(track)])
        run["t_ML_final"+str(track)+"_Myr"] = final_mass_loss_timescale\
                                              / (86400.*365*1e6) # million years
        run["t_ML_final"+str(track)+"_Gyr"] = final_mass_loss_timescale\
                                              / (86400.*365*1e9) # giga years
        
    return run


def calculate_init_final_beta(run,
                              beta_cutoff=True,
                              relation_EUV="Linsky"):
    """ calculate initial and final beta value 
    for all the tracks in the dataframe. """
    # get number of tracks (can be e.g. 1 or 10)
    number_of_tracks = len([c for c in run.columns if c[0]=="R"]) 

    for track in range(1, number_of_tracks+1):
        # calculate beta with initial and final parameters

        beta_init_track = []
        beta_final_track = []
        for index, row in run.iterrows():
            # get initial and final XUV flux
            F_XUV_init = flux_at_planet(
                                l_xuv_all(run.loc[index]["Lx_age"], relation_EUV),
                                run.loc[index]["a"]
                                       )
            F_XUV_final = flux_at_planet(
                                l_xuv_all(run.loc[index]["Lx"+str(track)], relation_EUV),
                                run.loc[index]["a"]
                                       )
            # calculate beta based on XUV flux and planetary mass and radius
            beta_init = beta_fct(run.loc[index]["mass"],
                                 F_XUV_init,
                                 run.loc[index]["radius"],
                                 cutoff=beta_cutoff)
            beta_final = beta_fct(run.loc[index]["M"+str(track)],
                                  F_XUV_init,
                                  run.loc[index]["R"+str(track)],
                                  cutoff=beta_cutoff)

            beta_init_track.append(beta_init)
            beta_final_track.append(beta_final)
        
        # append to initial dataframe
        run["beta_init"+str(track)] = beta_init_track
        run["beta_final"+str(track)] = beta_final_track
        
    return run


def calculate_additional_parameters(run,
                                    run_2nd,
                                    epsilon,
                                    beta_on="yes",
                                    beta_cutoff=True,
                                    K_on="yes",
                                    relation_EUV='Linsky'):
    
    """ calculate mass-loss rate at beginning and end,
    the mass-loss timescale at the last time step,
    beta parameter at first and last time step. 
    Look at the individual functions for details."""
    
    # calculate initial & final mass-loss rates
    run = calculate_mass_loss_rate_at_initial_and_final_timestep(
                                               run,
                                               epsilon,
                                               beta_on,
                                               beta_cutoff,
                                               K_on,
                                               relation_EUV)
    
    run_2nd = calculate_mass_loss_rate_at_initial_and_final_timestep(
                                               run_2nd,
                                               epsilon,
                                               beta_on,
                                               beta_cutoff,
                                               K_on,
                                               relation_EUV)
    
    # calculate initial and final beta value
    run = calculate_init_final_beta(run,
                                    beta_cutoff,
                                    relation_EUV)
    run_2nd = calculate_init_final_beta(run_2nd,
                                        beta_cutoff,
                                        relation_EUV)

    return run, run_2nd
