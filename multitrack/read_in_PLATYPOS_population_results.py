import os
import pandas as pd
import numpy as np
import datatable as dt

from scipy import interpolate
from scipy.interpolate import CubicSpline
import random
import astropy.units as u
from astropy import constants as const

import platypos.planet_models_LoFo14 as plmoLoFo14
import platypos.planet_models_ChRo16 as plmoChRo16
import platypos.planet_model_Ot20 as plmoOt20
from platypos.mass_loss_rate_function import mass_loss_rate_noplanetobj
from platypos.lx_evo_and_flux import l_xuv_all
from platypos.lx_evo_and_flux import flux_at_planet
from platypos.beta_K_functions import beta_fct, beta_calc
import multitrack.keplers_3rd_law as kepler3

# old runs only returned 4 columns (t, M, R, Lx)
# new runs return 5 columns (t, M, R, Lx, Mdot_info)
COLUMNS = 5


def read_results_file(path, filename):
    """Function to read in the results file for an individual track. """

    df = dt.fread(path + filename).to_pandas()
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
    
    planet_init_dict (dict): dictionary of initial planet parameters,
                             with planet names are keys, and the intial
                             parameters values (intial planet parameters
                             - for LoFo14 planets are: semi-major axis - a,
                             M_init: mass, R_init: radius,
                             M_core: mass_core, age0: age, where a is in
                             AU, mass, radius and core mass in Earth
                             units, age in Myr)

    tracks_dict (dict): dictionary with track infos; keys are planet
                        names, values are a list (of length N_tracks)
                        with the track parameters [track_number, full
                        track name, evolved_off flag (True of False)
                        NOTE: the last parameter is only important for
                        MESA planets]
    """
    print(path_to_results)
    files = os.listdir(path_to_results)
    files = [f for f in files if ".json" not in f]
    print("Total # of planet folders = ", len(files))
    # check for empty folders (where maybe sth went wrong, or where planet has
    # not evolved yet)
    non_empty_folders = []
    for f in files:
        if len(os.listdir(os.path.join(path_to_results, f))) == 0:
            pass
        elif len([file for file in os.listdir(os.path.join(path_to_results, f)) \
                  if f in file and "track" in file]) == 0:
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
        all_files_in_f = [f for f in os.listdir(os.path.join(path_to_results, f)) \
                          if not f.startswith('.')]
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
        # NOT REALLY GREAT PRACTICE BUT WELL..
        # e.g. planet_001_track_10.0_240.0_5000.0_2e+30_0.0_0.0.txt
        # planet_name    t_start t_sat t_fina  Lx_sat  Lx_drop  Lx_drop_factor
        result_files_sorted = \
                sorted(sorted(sorted(result_files,
                                     key=lambda x: float(x.rstrip(".txt").split("_")[-1])),
                              key=lambda x: float(x.rstrip(".txt").split("_")[-2])),
                       key=lambda x: float(x.rstrip(".txt").split("_")[-5]))

        # skip planets which do not have all tracks available! (important for
        # when reading in results while Platypos is still running)
        # number of tracks for which results are available
        N_tracks_subfolder = len(result_files_sorted)

        # only read in results in a single planet folder if ALL track-results
        # are available
        if N_tracks_subfolder == N_tracks:
            # get file with initial planet params (name: f+".txt")
            #path_to_results + f + "/" + f + ".txt"
            df_pl = pd.read_csv(os.path.join(path_to_results, f, f + ".txt"),
                                float_precision='round_trip')
            planet_init_dict[f] = df_pl.values[0]  # add to dictionary
            # build dataframe with results from all tracks
            df_all_tracks = pd.DataFrame()
            # read in the results file for each track one by one and build up
            # one single (one-row) dataframe per planet
            for file in result_files_sorted:
                #path_to_results + f + "/" + file
                df_i = dt.fread(os.path.join(path_to_results, f, file)).to_pandas()
                df_all_tracks = pd.concat([df_all_tracks, df_i], axis=1)
                # df.reset_index(level=0)

            # set the column names of the new dataframe (i goes from 1 to
            # N_tracks+1)
            col_names = []
            for i in range(1, int(len(df_all_tracks.columns) / COLUMNS) + 1):
                col_names.append("t" + str(i))
                col_names.append("M" + str(i))
                col_names.append("R" + str(i))
                col_names.append("Lx" + str(i))
                if COLUMNS == 5:
                    col_names.append("Mdot_info" + str(i))
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
    files = [f for f in files if ".json" not in f]
    print("Total # of planet folders = ", len(files))
    # check for empty folders (where maybe sth went wrong, or where planet has
    # not evolved yet)
    non_empty_folders = []
    for f in files:
        if len(os.listdir(path_to_results + f)) == 0:
            pass
        elif len([file for file in os.listdir(path_to_results + f) \
                  if f in file and "track" in file]) == 0:
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
        result_files = [file for file in all_files_in_f \
                        if ("final" in file) and ("track_params" not in file)]

        # sort result files first by t_sat, then by dt_drop, then by 
        # Lx_drop_factor; this is to have some order in the results
        # e.g. planet_001_track_10.0_240.0_5000.0_2e+30_0.0_0.0.txt
        # planet_name    t_start t_sat t_fina  Lx_sat  Lx_drop  Lx_drop_factor
        result_files_sorted = \
            sorted(sorted(sorted(result_files,
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
            df_pl = pd.read_csv(path_to_results + f + "/" + f + ".txt",
                                float_precision='round_trip')
            df_pl_contr = pd.read_csv(path_to_results + f + "/" + f + ".txt",
                                float_precision='round_trip')
            planet_init_dict[f] = df_pl.values[0]  # add to dictionary

            # build dataframe with results from all tracks
            df_final = pd.DataFrame()
            # read in the results file for each track one by one and build up
            # one single (one-row) dataframe per planet
            for file in result_files_sorted:
                df_i = dt.fread(path_to_results + f + "/" + file).to_pandas()
                try:
                    df_i["metallicity"]
                    df_i.drop(["a", "core_mass", "metallicity", "track"],
                              axis=1, inplace=True)
                except:
                    try:
                        df_i["core_comp"]
                        df_i.drop(["a", "core_mass", "core_comp", "track"],
                                  axis=1, inplace=True)
                    except:
                        df_i.drop(["a", "core_mass", "track"],
                                  axis=1, inplace=True)
                df_final = pd.concat([df_final, df_i], axis=1)
                #df_final.reset_index(level=0)
                
            # set the column names of the new dataframe (i goes from 1 to
            # N_tracks+1)
            col_names = []
            for i in range(1, int(len(df_final.columns) / COLUMNS) + 1):
                col_names.append("time" + str(i))
                col_names.append("fenv" + str(i))
                col_names.append("mass" + str(i))
                col_names.append("radius" + str(i))
                col_names.append("Lx" + str(i))
                if COLUMNS == 5:
                    col_names.append("Mdot_info" + str(i))
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
            list_planets_evolved_off = \
                ["track" + file.rstrip("_final.txt").split("track")[1] \
                 for file in all_files_in_f if "evolved_off" in file]
            track_info_list = []
            for i, file in enumerate(result_files_sorted):
                if "track" + file.rstrip("_final.txt").split("track")[1] in list_planets_evolved_off:
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
    planet_all = pd.merge(planet_init_df, planet_final_df,
                          left_index=True, right_index=True, how='outer')
    
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
    files = [f for f in files if ".json" not in f]
    # only use non-empty folders
    non_empty_folders = []
    for f in files:
        if len(os.listdir(path_to_results + f)) == 0:
            pass
        elif len([file for file in os.listdir(path_to_results + f) \
                  if f in file and "track" in file]) == 0:
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
    files = [f for f in files if ".json" not in f]
    # only use non-empty folders
    non_empty_folders = []
    for f in files:
        if len(os.listdir(path_to_results + f)) == 0:
            pass
        elif len([file for file in os.listdir(path_to_results + f) \
                  if f in file and "thermal" in file]) == 0:
            # this means no output file has been produced by PLATYPOS for any
            # of the tracks
            pass
        else:
            non_empty_folders.append(f)
    print("Non-empty folders: ", len(non_empty_folders))

    if len(non_empty_folders) > 0:
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
                path_to_results + f + "/" + f + "_thermal_contr.txt",
                float_precision='round_trip')
            pl_dict[f] = df_thermal.values[0]

        # convert the planet_init_dict to a dataframe
        df_thermal = pd.DataFrame.from_dict(
            pl_dict, orient='index', columns=df_thermal.columns)

        return df_thermal
    else:
        raise ValueError("No th. contr. radius files created!")


def read_in_PLATYPOS_results_dataframe(path_to_results, N_tracks,
                                       return_second_to_last=False):
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
                                           and initial and second to last final
                                           parameters (for all the tracks) as
                                           columns
    """
    
    # call read_in_PLATYPOS_results
    planet_df_dict, planet_init_df, tracks_dict = read_in_PLATYPOS_results_fullEvo(path_to_results, N_tracks)

    # create a master dictionary which contains all final (!) parameters
    planet_final_dict = {}
    planet_final_dict_2nd = {}
    for key_pl, df_pl in planet_df_dict.items():
        # number of tracks for which there is a result file available
        N_tracks = int(len(df_pl.columns) / COLUMNS)

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

    

    return planet_all_df, tracks_dict


def create_snapshots_table(pl_df_dict, snapshot_times):
    """ funtion returns a table with snapshots of planet population (t, M, R, Lx)
    at the specified snapshot-times (can be initial or final time too.)
    It takes the full M/R/Lx evolution and interpolates to get the values at 
    the exact input time.
    Parameters:
    -----------
    pl_df_dict (dict): dictionary with pandas tables for each planet (this is
                       what read_in_PLATYPOS_results_fullEvo() returns)
    snapshot_times (list): list of times for which you want to know the mass,
                           radius and Lx
    Returns:
    --------
    planet_snap_df (DataFrame): pandas table, indices are the planet id's,
                                columns are the snapshot time with corresponding
                                M, R, Lx (so len(snapshot_times)*4 columns)
    e.g. 2 tracks: t1_st1, R1_st1,..., t2_st1, R2_st1,...,t1_st2, R1_st2,...
    """
           
    # create a master dictionary which contains all final (!) parameters
    pl_snapshot_dict = {}
    # generate column names (t1_st1, M1_st1,..., t2_st1, M2_st1, ..., t1_st2, M1_st2...)
    columns_one_pl = []
    N_tracks = int(len(random.choice(list(pl_df_dict.values())).columns) / COLUMNS)
    for i in range(1, N_tracks+1):
        for t in snapshot_times:
            # make header-column names
            for j in ["t"+str(i)+"_", "R"+str(i)+"_", "M"+str(i)+"_", "Lx"+str(i)+"_"]:
                columns_one_pl.append(j+"{:.1e}".format(t))

    for key_pl, df_pl in pl_df_dict.items():
        # number of tracks for which there is a result file available
        N_tracks = int(len(df_pl.columns) / COLUMNS)
        # now I (possibly) need to interpolate to get the values at the
        # provided snapshot_time

        values_one_pl = []
        for i in range(1, N_tracks + 1):
            # create interpolation function
            lvi = df_pl["t"+str(i)].last_valid_index()
            t_lvi = df_pl["t"+str(i)].loc[lvi]

            M_interp = CubicSpline(df_pl["t"+str(i)].loc[:lvi],
                                   df_pl["M"+str(i)].loc[:lvi], extrapolate=True)
            R_interp = CubicSpline(df_pl["t"+str(i)].loc[:lvi],
                                   df_pl["R"+str(i)].loc[:lvi], extrapolate=True)
            Lx_interp = CubicSpline(df_pl["t"+str(i)].loc[:lvi],
                                   df_pl["Lx"+str(i)].loc[:lvi], extrapolate=True)

            masses_interp = [float(M_interp(snap_t)) if (snap_t < t_lvi) else \
                             df_pl["M"+str(i)].loc[lvi] for snap_t in snapshot_times]
            radii_interp = [float(R_interp(snap_t)) if (snap_t < t_lvi) else \
                            df_pl["R"+str(i)].loc[lvi] for snap_t in snapshot_times]
            Lxs_interp = [float(Lx_interp(snap_t)) if (snap_t < t_lvi) else \
                          df_pl["Lx"+str(i)].loc[lvi] for snap_t in snapshot_times]
            # now I have arrays of len(snapshot_times), which contain the mass,
            # radius and Lx at the corresponding time value
            # this info I can now store in a dataframe (times as columns)

            # now I want to make a seperate dataframe for each snapshot time!
            # where each df contains the R & M values for all the tracks at that
            # snapshot time

            for values in list(zip(snapshot_times, radii_interp, \
                                   masses_interp, Lxs_interp)):
                for v in values:
                    values_one_pl.append(v)

        pl_snapshot_dict[key_pl] = values_one_pl

    # convert master dictionary to dataframe with planet id's as indices
    planet_snap_df = pd.DataFrame.from_dict(pl_snapshot_dict, orient='index', \
                                            columns=columns_one_pl)
    return planet_snap_df


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
                                              beta_settings,
                                              K_on,
                                              mass_loss_calc,
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
                     
    beta_settings (dict):
    
    K_on (str):      K calculation on or off ("yes"/"no"); use same value 
                     as used in the simulation run
                     
    mass_loss_calc (str):
                     
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
            Mdot_init = mass_loss_rate_noplanetobj(
                                    t_=run.loc[index]["age"],
                                    distance=run.loc[index]["a"],
                                    R_p_at_t_=run.loc[index]["radius"],
                                    M_p_at_t_=run.loc[index]["mass"],
                                    Lx_at_t_=run.loc[index]["Lx_age"],
                                    epsilon=epsilon, K_on=K_on,
                                    beta_settings=beta_settings,
                                    mass_star=run.loc[index]["mass_star"],
                                    Lbol_solar=run.loc[index]["Lbol"],
                                    mass_loss_calc=mass_loss_calc,
                                    relation_EUV=relation_EUV)
            Mdot_init_track.append(Mdot_init) # in g/s
            
            Mdot_final = mass_loss_rate_noplanetobj(
                                    t_=run.loc[index]["t"+str(track)],
                                    distance=run.loc[index]["a"],
                                    R_p_at_t_=run.loc[index]["R"+str(track)],
                                    M_p_at_t_=run.loc[index]["M"+str(track)],
                                    Lx_at_t_=run.loc[index]["Lx"+str(track)],
                                    epsilon=epsilon, K_on=K_on,
                                    beta_settings=beta_settings,
                                    mass_star=run.loc[index]["mass_star"],
                                    Lbol_solar=run.loc[index]["Lbol"],
                                    mass_loss_calc=mass_loss_calc,
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
                              beta_settings,
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
                            l_xuv_all(run.loc[index]["Lx_age"],
                                      relation_EUV, run.loc[index]["mass_star"]),
                            run.loc[index]["a"]
                                       )
            F_XUV_final = flux_at_planet(
                            l_xuv_all(run.loc[index]["Lx"+str(track)],
                                      relation_EUV, run.loc[index]["mass_star"]),
                            run.loc[index]["a"]
                                       )
            # calculate beta based on XUV flux and planetary mass and radius
            beta_init = beta_calc(run.loc[index]["mass"],
                                  run.loc[index]["radius"],
                                  F_XUV_init,
                                  beta_settings,
                                  distance=run.loc[index]["a"],
                                  M_star=run.loc[index]["mass_star"],
                                  Lbol_solar=run.loc[index]["Lbol"])
            beta_final = beta_calc(run.loc[index]["M"+str(track)],
                                   run.loc[index]["R"+str(track)],
                                   F_XUV_final,
                                   beta_settings,
                                   distance=run.loc[index]["a"],
                                   M_star=run.loc[index]["mass_star"],
                                   Lbol_solar=run.loc[index]["Lbol"])

            beta_init_track.append(beta_init)
            beta_final_track.append(beta_final)
        
        # append to initial dataframe
        run["beta_init"+str(track)] = beta_init_track
        run["beta_final"+str(track)] = beta_final_track
        
    return run


def calculate_additional_parameters(run,
                                    run_2nd,
                                    epsilon,
                                    beta_settings,
                                    K_on,
                                    mass_loss_calc,
                                    relation_EUV='Linsky'):
    
    """ calculate mass-loss rate at beginning and end,
    the mass-loss timescale at the last time step,
    beta parameter at first and last time step. 
    Look at the individual functions for details.
    
    Parameters:
    -----------
    
    Returns:
    --------
    run (DataFrame):
    run2 (DataFrame):
    """
    
    # calculate initial & final mass-loss rates
    run = calculate_mass_loss_rate_at_initial_and_final_timestep(
                                               run,
                                               epsilon,
                                               beta_settings,
                                               K_on,
                                               mass_loss_calc,
                                               relation_EUV)
    
    run_2nd = calculate_mass_loss_rate_at_initial_and_final_timestep(
                                               run_2nd,
                                               epsilon,
                                               beta_settings,
                                               K_on,
                                               mass_loss_calc,
                                               relation_EUV)
    
    # calculate initial and final beta value
    run = calculate_init_final_beta(run, beta_settings, relation_EUV)
    run_2nd = calculate_init_final_beta(run_2nd, beta_settings, relation_EUV)

    return run, run_2nd


################################################################################
# new stuff -> replace old ones
################################################################################

def read_in_PLATYPOS_results_df_new(path_to_results, N_tracks,
                                    return_second_to_last,
                                    snapshot_times=None,
                                    MESA=False):
    """
    Calls read_in_PLATYPOS_results_fullEvo & then does some more 
    re-aranging to the data to make it easier to handle.
    - gets initial parameters,
    - extracts final parameters (e.g. t, R, M, Lx)
    - can extract the parameters at the second-to-last timestep
    -can extract the parameters at any given snapshot time

    Parameters:
    -----------
    path_to_results (str): path to the master folder which containts all
                           the results for a single run (i.e. all the 
                           subfolders for the individual planets)

    N_tracks (int): total number of tracks specified when running PLATYPOS;
                    only planets with complete output are used (this is 
                    mainly so that I can read in and look at the results,
                    even though Platypos is still running)
    
    return_second_to_last (bool): if True, values at second-to-last time step
                                  of the simulation are returned
                                  
    snapshot_times (list of float): list of snapshot times for which you want to
                                    extract all the planetary parameters
                                    
    MESA (bool): default=False (ignore for now!)

    Returns:
    --------
    planet_all_df (dataframe): dataframe with planet id's as indices and
                               initial and final parameters (for all the
                               tracks) as columns

    tracks_dict (dict): dictionary with planet id's as keys, and list of
                        all tracks and their parameters as values
                        
    if return_second_to_last == True:
    planet_all_df_2nd_to_last (DataFrame): dataframe with planet id's as indices
                                           and initial and second to last final
                                           parameters (for all the tracks) as
                                           columns
                                           
    if snapshot_times == list of times:
    planet_all_df_snap (DataFrame): dataframe with planet id's as indices
                                    and initial parameters plus parameters at
                                    all the snapshot times (for all the tracks!)
                                    as columns
    
    RETURN ORDER:
    planet_all_df, (planet_all_df_2nd_to_last), (planet_all_df_snap), tracks_dict
    """
    
    # call read_in_PLATYPOS_results
    planet_df_dict, planet_init_df, tracks_dict = read_in_PLATYPOS_results_fullEvo(path_to_results, N_tracks)

    # create a master dictionary which contains all final (!) parameters
    planet_final_dict = {}
    planet_final_dict_2nd = {}
    
    # generate column names (t1, M1,..., t2, M2, ...)
    columns_all_tracks = []
    for i in range(1, N_tracks+1):
        # make header-column names
        columns_all_tracks.append("t"+str(i))
        columns_all_tracks.append("R"+str(i))
        columns_all_tracks.append("M"+str(i))
        columns_all_tracks.append("Lx"+str(i))
        columns_all_tracks.append("Lx0_"+str(i))
    
    if (snapshot_times != None) & (type(snapshot_times) == list):
        # create a master dictionary which contains all final (!) parameters
        pl_snapshot_dict = {}
        # generate column names (t1_st1, M1_st1,..., t2_st1, M2_st1, ..., t1_st2, M1_st2...)
        columns_one_pl = []
        for i in range(1, N_tracks+1):
            for t in snapshot_times:
                # make header-column names
                for j in ["t"+str(i)+"_", "R"+str(i)+"_", "M"+str(i)+"_", "Lx"+str(i)+"_", "Lx0_"+str(i)+"_"]:
                    columns_one_pl.append(j+"{:.1e}".format(t))

    for key_pl, df_pl in planet_df_dict.items():
        # number of tracks for which there is a result file available
        N_tracks = int(len(df_pl.columns) / COLUMNS)

        # now I need to check for each track what the index of the last 
        # non-nan value is (if planet has moved outside of grid, or has
        # reached the stopping condition, PLATYPOS terminates & returnes
        # the final planetary parameters. Problem: This might not always
        # be at the same time step for each track! So when I read in all
        # the tracks into one dataframe, the remaining rows will be filled
        # with NaNs)

        # extract final values
        final_values_one = []
        for i in range(1, N_tracks + 1):
            # return index for last non-NA/null value
            final_index = df_pl["M" + str(i)].last_valid_index()
            first_index = df_pl["M" + str(i)].first_valid_index()
            # get corresponding final time, mass & radius, add to df_final
            for v in [df_pl["t" + str(i)].loc[final_index], \
                      df_pl["R" + str(i)].loc[final_index], \
                      df_pl["M" + str(i)].loc[final_index], \
                      df_pl["Lx" + str(i)].loc[final_index], \
                      df_pl["Lx" + str(i)].loc[first_index]]:
                final_values_one.append(v)
            
        planet_final_dict[key_pl] = final_values_one

        # extract second-to-last values
        final_2nd_values_one = []
        if return_second_to_last == True:
            final_values_one = []
            for i in range(1, N_tracks + 1):
                # return index for last non-NA/null value
                final_index = df_pl["M" + str(i)].last_valid_index()
                first_index = df_pl["M" + str(i)].first_valid_index()
                # get corresponding final time, mass & radius, add to df_final
                for v in [df_pl["t" + str(i)].loc[final_index-1], \
                          df_pl["R" + str(i)].loc[final_index-1], \
                          df_pl["M" + str(i)].loc[final_index-1], \
                          df_pl["Lx" + str(i)].loc[final_index-1], \
                          df_pl["Lx" + str(i)].loc[first_index]]:
                    final_2nd_values_one.append(v)
            planet_final_dict_2nd[key_pl] = final_2nd_values_one
            
        # extract snapshot values
        if (snapshot_times != None) & (type(snapshot_times) == list):
            values_one_pl = []
            for i in range(1, N_tracks + 1):
                # create interpolation function
                lvi = df_pl["t"+str(i)].last_valid_index()
                fvi = df_pl["t"+str(i)].first_valid_index()
                t_lvi = df_pl["t"+str(i)].loc[lvi]

                M_interp = CubicSpline(df_pl["t"+str(i)].loc[:lvi],
                                       df_pl["M"+str(i)].loc[:lvi], extrapolate=True)
                R_interp = CubicSpline(df_pl["t"+str(i)].loc[:lvi],
                                       df_pl["R"+str(i)].loc[:lvi], extrapolate=True)
                Lx_interp = CubicSpline(df_pl["t"+str(i)].loc[:lvi],
                                       df_pl["Lx"+str(i)].loc[:lvi], extrapolate=True)

                masses_interp = [float(M_interp(snap_t)) if (snap_t < t_lvi) else \
                                 df_pl["M"+str(i)].loc[lvi] for snap_t in snapshot_times]
                radii_interp = [float(R_interp(snap_t)) if (snap_t < t_lvi) else \
                                df_pl["R"+str(i)].loc[lvi] for snap_t in snapshot_times]
                Lxs_interp = [float(Lx_interp(snap_t)) if (snap_t < t_lvi) else \
                              df_pl["Lx"+str(i)].loc[lvi] for snap_t in snapshot_times]
                # now I have arrays of len(snapshot_times), which contain the mass,
                # radius and Lx at the corresponding time value
                # this info I can now store in a dataframe (times as columns)
                Lx0s = [df_pl["Lx"+str(i)].loc[fvi] for snap_t in snapshot_times]

                # now I want to make a seperate dataframe for each snapshot time!
                # where each df contains the R & M values for all the tracks at that
                # snapshot time

                for values in list(zip(snapshot_times, radii_interp, \
                                       masses_interp, Lxs_interp, Lx0s)):
                    for v in values:
                        values_one_pl.append(v)
            pl_snapshot_dict[key_pl] = values_one_pl


    # convert master dictionary to dataframe with planet id's as indices
    planet_final_df = pd.DataFrame.from_dict(
        planet_final_dict, orient='index', columns=columns_all_tracks)

    # concatenate planet_init_df and planet_final_df into one master dataframe
    planet_all_df = pd.concat(
        [planet_init_df, planet_final_df], axis=1, sort=False)
    # together with tracks_dict this allows me to select, filter, analyze any
    # track and any planet
    
    if MESA == True:
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
            planet_final_dict_2nd, orient='index', columns=columns_all_tracks)

        # concatenate planet_init_df and planet_final_df into one master dataframe
        planet_all_df_2nd = pd.concat(
            [planet_init_df, planet_final_df_2nd], axis=1, sort=False)
        # together with tracks_dict this allows me to select, filter, analyze any
        # track and any planet
        
        if MESA == True:
            # For MESA planets only: to make life easier, add evolved_off info to 
            # my master dataframe
            # add N_tracks more columns to dataframe planet_all_df which indicate
            # whether the planet on given track has evolved off or not
            for key_pl in planet_all_df_2nd.index.values:
                track_number = [a_tuple[0] for a_tuple in tracks_dict[key_pl]]
                track_evooff = [a_tuple[2] for a_tuple in tracks_dict[key_pl]]
                for number, evooff in zip(track_number, track_evooff):
                    planet_all_df_2nd.at[key_pl, "track" + str(number)] = evooff
                    
    if (snapshot_times != None) & (type(snapshot_times) == list):
        # convert master dictionary to dataframe with planet id's as indices
        planet_snap_df = pd.DataFrame.from_dict(pl_snapshot_dict, orient='index', \
                                                columns=columns_one_pl)
        # concatenate planet_init_df and planet_final_df into one master dataframe
        planet_all_df_snap = pd.concat(
            [planet_init_df, planet_snap_df], axis=1, sort=False)
        # together with tracks_dict this allows me to select, filter, analyze any
        # track and any planet               
    
    if (return_second_to_last == True) & (snapshot_times == None):
        return planet_all_df, planet_all_df_2nd, tracks_dict
    elif (return_second_to_last == True) & (snapshot_times != None):
        return planet_all_df, planet_all_df_2nd, planet_all_df_snap, tracks_dict
    elif (return_second_to_last == False) & (snapshot_times != None):
        return planet_all_df, planet_all_df_snap, tracks_dict
    else:
        return planet_all_df, tracks_dict
    

def get_complete_results_new(planet_df, path_to_results, N_tracks):
    """ Combine everything into one DataFrame (star & planet parameters), 
    and calculate some additional parameters, which might be handy to have.
    NOTE: two dataframes are returned, one with the parameters at the simulation
    end time, and an equivalent one but at one time step before.
    
    Parameters:
    -----------
    Returns:
    --------
    df_run, df_run_2nd, tracks_dict
    """
    L_SUN = 3.828e+33 #const.L_sun.cgs.value
    AU = 14959787070000.0 #const.au.cgs

    #N_tracks = len(random.choice(list(tracks_dict.values())))
    try:
        track_list = ["track"+str(i) for i in range(1, N_tracks+1)] 
        planet_df.drop(labels=track_list, axis=1, inplace=True)
    except:
        pass
        
    star_df = read_in_host_star_parameters(path_to_results)
    star_df.drop(["age"], axis=1, inplace=True)

    columns = [c for c in planet_df.columns if ((c[0]=="t") or (c[0]=="R") or \
                                             (c[0]=="M") or (c[0:2]=="Lx")) \
                                              and (c != "t_eq")]
    #print(columns)
    for i in range(1, N_tracks+1):
        # calculate envelope mass fractions & the fraction of envelope lost at given
        # time (can be final time, 2nd-to-last or snapshot times)
        M_env_init = planet_df["mass"] - planet_df["core_mass"]
        M_columns = [c for c in columns if c[0]=="M"]
        for M_col in M_columns:
            M_env_final = planet_df[M_col] - planet_df["core_mass"]
            # calculate final envelope mass fraction
            planet_df["fenv_final"+M_col[1:]] = (M_env_final / planet_df[M_col]) * 100
            # calculate fraction of envelope lost
            planet_df["frac_env_lost"+M_col[1:]] = (M_env_init - M_env_final) / M_env_init

    # add core radius and period to dataframe
    try:
        planet_df["metallicity"] # LoFo14 planet
        planet_df["core_radius"] = plmoLoFo14.calculate_core_radius(planet_df["core_mass"])
    except:
        try:
            planet_df["core_comp"] # ChRo16 planet
            planet_df["core_radius"] = plmoChRo16.calculate_core_radius(planet_df["core_mass"])
        except:
            planet_df["core_radius"] = 2.15 # hardcoded (same as in mass_evo_RK4_forward for Ot20 planets)

    planet_df["period"] = kepler3.get_period_from_a(
                            star_df["mass_star"][planet_df.index].values,
                            planet_df["mass"].values, planet_df["a"].values)

    # combine planet- and host-star dataframe & add info about planetary incident flux
    df_run = pd.concat([planet_df, star_df], axis=1, sort=False)#, join='inner')
    FLUX_AT_EARTH = 1373. * 1e7 / (100*100) # erg/s/cm^2
    df_run["flux"] = df_run["Lbol"] * L_SUN / (4 * np.pi * (df_run["a"] * AU)**2)
    df_run["flux_EARTH"] = df_run["flux"] / FLUX_AT_EARTH

    return df_run


def calculate_mass_loss_rate_new(run, add_parameters):
    """ Takes a run-dataframe (from get_complete_results_new())
    and returns the same dataframe but with mass-loss rate at 
    initial and final timestep, and the mass-loss timescale at
    the last time step added as new columns. (or if run is the snapshots table, 
    these values are calculated for the particular timesteps)
    
    Input:
    ------
    run (dataframe): input is dataframe returned by get_complete_results() 
    add_parameters (dict): dictionary containing all info about the run 
                           e.g. add_parameters = {"epsilon": 0.1, 
                           "beta_settings": {'beta_calc': 'Lopez17', 'RL_cut': True},
                           "K_on": "yes", "mass_loss_calc": "Elim_and_RRlim",
                           "relation_EUV": "Linsky"}
    
    Returns:
    --------
    run (dataframe): same as imput but with new columns added.
                     - Mdot_final+track# [g/s], [M_earth/yr], [M_earth/Gyr]
                     - t_ML_final+track# [Myr], [Gyr]
                     - Mdot_init+track# [g/s], [M_earth/yr], [M_earth/Gyr]
    """
    M_EARTH = 5.972364730419773e+27 #const.M_earth.cgs.value
    
    try:
        epsilon = add_parameters["epsilon"]
        beta_settings = add_parameters["beta_settings"]
        K_on = add_parameters["K_on"]
        mass_loss_calc = add_parameters["mass_loss_calc"]
        relation_EUV = add_parameters["relation_EUV"]
    except:
        raise ValueError("add_parameters needs: epsilon, beta_settings, " + \
                         "K_on, mass_loss_calc, and relation_EUV" )
    
    columns = [c for c in run.columns if ((c[0]=="t") or (c[0]=="R") or \
                                         (c[0]=="M") or (c[0:2]=="Lx")) \
                                          and (c != "t_eq") and (c != "Lx_age")]
    Lx_columns = [c for c in columns if (c[0:2]=="Lx") and (c!="Lx_age") and ("Lx0" not in c)]
    Lx0_columns = [c for c in columns if ("Lx0" in c)]
    
    # calculate beta at t0 (initial time)
    for Lx0_col in Lx0_columns:
        Mdot_init_track = []
        for index, row in run.iterrows():
            Mdot_init = mass_loss_rate_noplanetobj(
                                    t_=run.loc[index]["age"],
                                    distance=run.loc[index]["a"],
                                    R_p_at_t_=run.loc[index]["radius"],
                                    M_p_at_t_=run.loc[index]["mass"],
                                    Lx_at_t_=run.loc[index][Lx0_col],
                                    epsilon=epsilon, K_on=K_on,
                                    beta_settings=beta_settings,
                                    mass_star=run.loc[index]["mass_star"],
                                    Lbol_solar=run.loc[index]["Lbol"],
                                    mass_loss_calc=mass_loss_calc,
                                    relation_EUV=relation_EUV)

            Mdot_init_track.append(-Mdot_init)
    
        # append to initial dataframe [g/s and M_earth/Gyr]
        run["Mdot"+Lx0_col[2:]] = Mdot_init_track  
        run["Mdot"+Lx0_col[2:]+"M_earth_per_Gyr"] =\
                        run["Mdot"+Lx0_col[2:]] / M_EARTH \
                        * (86400.*365*1e9)
        run["Mdot"+Lx0_col[2:]+"M_earth_per_yr"] =\
                        run["Mdot"+Lx0_col[2:]] / M_EARTH \
                        * (86400.*365)
    
    for Lx_col in Lx_columns:
        Mdot_track = []
        for index, row in run.iterrows():
            Mdot = mass_loss_rate_noplanetobj(
                                    t_=run.loc[index]["t"+Lx_col[2:]],
                                    distance=run.loc[index]["a"],
                                    R_p_at_t_=run.loc[index]["R"+Lx_col[2:]],
                                    M_p_at_t_=run.loc[index]["M"+Lx_col[2:]],
                                    Lx_at_t_=run.loc[index][Lx_col],
                                    epsilon=epsilon, K_on=K_on,
                                    beta_settings=beta_settings,
                                    mass_star=run.loc[index]["mass_star"],
                                    Lbol_solar=run.loc[index]["Lbol"],
                                    mass_loss_calc=mass_loss_calc,
                                    relation_EUV=relation_EUV)

            Mdot_track.append(-Mdot)
        
        run["Mdot"+Lx_col[2:]] = Mdot_track  
        run["Mdot"+Lx_col[2:]+"M_earth_per_Gyr"] = \
                        run["Mdot"+Lx_col[2:]] / M_EARTH \
                        * (86400.*365*1e9)
        run["Mdot"+Lx_col[2:]+"M_earth_per_yr"] = \
                        run["Mdot"+Lx_col[2:]] / M_EARTH \
                        * (86400.*365)

        # final mass-loss timescale = final envelope mass divided by mass loss rate
        mass_loss_timescale = ((run["M"+Lx_col[2:]] - run["core_mass"])\
                                     * M_EARTH)\
                                     / (run["Mdot"+Lx_col[2:]])
        run["t_ML"+Lx_col[2:]+"_Myr"] = mass_loss_timescale \
                                              / (86400.*365*1e6) # million years
        run["t_ML"+Lx_col[2:]+"_Gyr"] = mass_loss_timescale \
                                              / (86400.*365*1e9) # giga years

    return run


def calculate_beta(run,
                   beta_settings,
                   relation_EUV="Linsky"):
    """ calculate beta value for all the tracks in the dataframe. """

    columns = [c for c in run.columns if ((c[0]=="t") or (c[0]=="R") or \
                                         (c[0]=="M") or (c[0:2]=="Lx")) \
                                          and (c != "t_eq") and (c != "Lx_age")]
    Lx_columns = [c for c in columns if (c[0:2]=="Lx") and (c!="Lx_age") and ("Lx0" not in c)]
    Lx0_columns = [c for c in columns if ("Lx0" in c)]

    
    # calculate beta at t0 (initial time)
    for Lx0_col in Lx0_columns:
        beta_track = []
        for index, row in run.iterrows():
            F_XUV_init = flux_at_planet(
                            l_xuv_all(run.loc[index][Lx0_col],
                                      relation_EUV, run.loc[index]["mass_star"]),
                            run.loc[index]["a"])
            
            
            beta = beta_calc(run.loc[index]["mass"],
                           run.loc[index]["radius"],
                           F_XUV_init,
                           beta_settings,
                           distance=run.loc[index]["a"],
                           M_star=run.loc[index]["mass_star"],
                           Lbol_solar=run.loc[index]["Lbol"])

            beta_track.append(beta)

        run["beta"+Lx0_col[2:]] = beta_track

    # calculate beta either at final times or the snapshot times (for all tracks)
    for Lx_col in Lx_columns:
        beta_track = []
        for index, row in run.iterrows():
            F_XUV_final = flux_at_planet(
                            l_xuv_all(run.loc[index][Lx_col],
                                      relation_EUV, run.loc[index]["mass_star"]),
                            run.loc[index]["a"])

            beta = beta_calc(run.loc[index]["M"+Lx_col[2:]],
                           run.loc[index]["R"+Lx_col[2:]],
                           F_XUV_final,
                           beta_settings,
                           distance=run.loc[index]["a"],
                           M_star=run.loc[index]["mass_star"],
                           Lbol_solar=run.loc[index]["Lbol"])

            beta_track.append(beta)

        run["beta"+Lx_col[2:]] = beta_track

    return run


def calculate_additional_parameters_new(run, add_parameters):
    """ Function to calculate the mass-loss rate at beginning and end of the
    simulation, the mass-loss timescale at the last time step, and the beta
    parameters at first and last time step. OR if the input-run is the snapshot
    one, then calculate these parameters at the snapshot times.
    Look at the individual functions for more details.
    
    Parameters:
    -----------
    run (DataFrame): output pandas table from read_in_PLATYPOS_results_df_new()
                        -> can be the final dataframe or the snapshots df
    add_parameters (dict): dictionary containing all info about the run 
                           e.g. add_parameters = {"epsilon": 0.1, 
                           "beta_settings": {'beta_calc': 'Lopez17', 'RL_cut': True},
                           "K_on": "yes","mass_loss_calc": "Elim_and_RRlim",
                           "relation_EUV": "Linsky"}
    
    Returns:
    --------
    run (DataFrame): pd-table eith mass-loss rates and beta values calculated
    
    """
    try:
        epsilon = add_parameters["epsilon"]
        beta_settings = add_parameters["beta_settings"]
        K_on = add_parameters["K_on"]
        mass_loss_calc = add_parameters["mass_loss_calc"]
        relation_EUV = add_parameters["relation_EUV"]
    except:
        raise ValueError("add_parameters needs: epsilon, beta_settings, " + \
                         "K_on, mass_loss_calc, and relation_EUV")
        
    # calculate initial & final mass-loss rates
    run = calculate_mass_loss_rate(run, add_parameters)
    
    # calculate initial and final beta value
    run = calculate_beta(run, add_parameters["beta_settings"],
                         add_parameters["relation_EUV"])

    return run


def create_master_table(df_all, path_to_results, N_tracks, add_parameters):
    """ Function creates a master table with all infos one might want.
    Extracts/calculates: 
        - initial parameters
        - final parameters of the sim. run (e.g. M, R, t)
        - host star parameters
        - initial and final beta
        - initial and final mass-loss rates
        
    Parameters:
    -----------
    df_all (DataFrame): output pandas table from read_in_PLATYPOS_results_df_new()
                        -> can be the final dataframe or the snapshots df
    path_to_results (str): path to the results folder
    N_tracks (int): number of tracks in the simulation run
    add_parameters (dict): dictionary containing all info about the run 
                           e.g. add_parameters = {"epsilon": 0.1, 
                           "beta_settings": {'beta_calc': 'Lopez17', 'RL_cut': True},
                           "K_on": "yes","mass_loss_calc": "Elim_and_RRlim",
                           "relation_EUV": "Linsky"}
    
    Returns:
    --------
    df_all_more (DataFrame): giant pandas table with all the parameters you
                             never knew you needed.
    """
    try:
        epsilon = add_parameters["epsilon"]
        beta_settings = add_parameters["beta_settings"]
        K_on = add_parameters["K_on"]
        mass_loss_calc = add_parameters["mass_loss_calc"]
        relation_EUV = add_parameters["relation_EUV"]
    except:
        raise ValueError("add_parameters needs: epsilon, beta_settings, " + \
                         "K_on, mass_loss_calc, and relation_EUV" )
        
    df_all_more = get_complete_results_new(df_all, path_to_results, N_tracks)

    df_all_more = calculate_beta(df_all_more,
                                 add_parameters["beta_settings"],
                                 add_parameters["relation_EUV"])

    df_all_more = calculate_mass_loss_rate_new(df_all_more, add_parameters)
    
    return df_all_more