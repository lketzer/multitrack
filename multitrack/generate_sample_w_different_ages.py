import os
import pandas as pd
import numpy as np
import scipy.stats
import datatable as dt


def generate_different_ages_sample(run, ages=None, path_save=None, filename=None,
                                   age_probablilities_file="./age_probabilities_for_my_logspaced_ages.csv"):
    """ 
    Use my previously generated age probability distribution to generate a
    planetary sample consisting of planets at different ages. Each planet has
    one age drawn (based on a given probability distribution) and the 
    respective final parameters for this age are stored in a table.
    
    Input:
    ------
    run(dataframe): previously read in snapshot dataframe
                    (skew/norm dataframe or the one with all tracks)
    ages (dataframe): if ages is None, draw an age from the age distro;
                      else: ages should be a pandas series with the already 
                      drawn planetary ages (this way I can generate the same
                      age distribution for the mixed track sample and the
                      single-track sample)
                    
    path_save (str): if specified, can save the resulting table to file
    filename (str): if specified, can save the resulting table to file
    
    Returns:
    --------
    run_mix_age: dataframe with init & final params for the age-distro sample
    
    """

    if path_save is not None and filename is not None:
        if os.path.exists(os.path.join(path_save, filename+"_age.csv")):
            print("exists.")
            return dt.fread(os.path.join(path_save, filename+"_age.csv")).to_pandas().set_index("C0")
    
    
    # check if supplied run is a mixed track df or includes all tracks
    times = np.array([c.lstrip("R2").lstrip("_") for c in run.columns if 'R2' in c])
    #print(times)
    
    try:
        times = times.astype(np.float64)
    except:
        raise Error("Need to supply the snapshots dataframe!")

    # read in the files with the probabilities
    df_age = pd.read_csv(age_probablilities_file)

    # create the discrete probability mass functions
    custm_age = scipy.stats.rv_discrete(name='custm_age',
                                   values=(df_age["Age_Myr"],
                                           df_age["probability"]))

    # create dataframe with final parameters [easier for plotting]
    try:
        run["metallicity"]
        planet_model = "LoFo14"
    except:
        planet_model = "ChRo16"

    
    # create the columns of the final dataframe
    # if input is the mixed-track sample
    if len(times) == 0:
        columns = ['a', 'period', 'core_mass', 'core_radius', 'fenv', 'mass',
                   'radius', 'age', 't_eq', 't_i', 'R_i', 'M_i', 'Lx0',
                   'Lx_i', 'fenv_i', 'frac_env_lost', "t_ML_i_Gyr",
                   "Mdot_i", "Mdot_i_M_earth_per_Gyr", "Mdot_i_M_earth_per_yr",
                   "Mdot_init", "Mdot_init_M_earth_per_Gyr", "Mdot_init_M_earth_per_yr",
                   "beta_init", "beta_i", 'mass_star', 'Lbol', 'Lx_age', 'flux',
                   'flux_EARTH', 'track#']
    # if input is the dataframe with all tracks
    else:
        # determine number of tracks
        colsR = [c for c in run.columns if 'R' in c and 'EARTH' not in c]
        number_tracks = len(np.unique(np.array([c.split('_')[0] for c in colsR])))
        columns = ['a', 'period', 'core_mass', 'core_radius', 'fenv', 'mass',
                   'radius', 'age', 't_eq']
        for i in range(number_tracks):
            i += 1
            cols_tracks = [f't{i}_i', f'R{i}_i', f'M{i}_i', f'Lx0_{i}', f'Lx{i}_i',
                           f'fenv_final{i}_i', f'frac_env_lost{i}_i', f"t_ML{i}_i_Gyr",
                           f"Mdot{i}_i", f"Mdot{i}_i_M_earth_per_Gyr",
                           f"Mdot{i}_i_M_earth_per_yr", f"Mdot{i}_init", 
                           f"Mdot{i}_init_M_earth_per_Gyr", f"Mdot{i}_init_M_earth_per_yr",
                           f"beta{i}_init", f"beta{i}_i"]
            columns += cols_tracks
        cols_add = [ 'mass_star', 'Lbol', 'Lx_age', 'flux', 'flux_EARTH']
        columns += cols_add

    if planet_model == "LoFo14":
        columns.append('metallicity')
    elif planet_model == "ChRo16":
        columns.append('core_comp')

    run_mix_age = pd.DataFrame(columns=columns)
    # now draw age and extract all the necessary values
    
    age_mix = []
    for index, row in run.iterrows():
        if ages is None:
            # draw age
            age = custm_age.rvs()
        else:
            age = ages.loc[index]
            age = float(age)

        if len(times) == 0:
            age_str = f'_{age:.1e}'
            age_mix.append(age_str[1:]) # add age
            tr_num = age_str

            # get corresponding parameters
            cols = ['a', 'period', 'core_mass', 'core_radius', 'fenv', 'mass',
                    'radius', 'age', 't_eq', 
                    f't{tr_num}', f'R{tr_num}', f'M{tr_num}', f'Lx0{tr_num}', 
                    f'Lx{tr_num}', f'fenv{tr_num}', f'frac_env_lost{tr_num}',
                    f"t_ML{tr_num}_Gyr", f"Mdot{tr_num}", f"Mdot{tr_num}M_earth_per_Gyr",
                    f"Mdot{tr_num}M_earth_per_yr", f"Mdot0{tr_num}",
                    f"Mdot0{tr_num}M_earth_per_Gyr", f"Mdot0{tr_num}M_earth_per_yr",
                    f"beta0{tr_num}", f"beta{tr_num}",
                    'mass_star', 'Lbol', 'Lx_age', 'flux', 'flux_EARTH', 'track#']

            if planet_model == "LoFo14":
                cols.append('metallicity')
            elif planet_model == "ChRo16":
                cols.append('core_comp')

            values = row[cols].values
            run_mix_age.loc[index] = values # add track params to new dataframe

        else:
            age_mix.append(f'{age:.1e}') # add age

            # get corresponding parameters
            cols = ['a', 'period', 'core_mass', 'core_radius', 'fenv', 'mass',\
                    'radius', 'age', 't_eq']
            for i in range(number_tracks):
                i += 1
                tr_age = f'{i}_{age:.1e}'
                cols_tr = [f't{tr_age}', f'R{tr_age}', f'M{tr_age}', f'Lx0_{tr_age}', 
                           f'Lx{tr_age}', f'fenv_final{tr_age}', f'frac_env_lost{tr_age}',
                           f"t_ML{tr_age}_Gyr", f"Mdot{tr_age}", f"Mdot{tr_age}M_earth_per_Gyr",
                           f"Mdot{tr_age}M_earth_per_yr", f"Mdot0_{tr_age}",
                           f"Mdot0_{tr_age}M_earth_per_Gyr", f"Mdot0_{tr_age}M_earth_per_yr",
                           f"beta0_{tr_age}", f"beta{tr_age}"]
                cols += cols_tr
            cols += ['mass_star', 'Lbol', 'Lx_age', 'flux', 'flux_EARTH']

            if planet_model == "LoFo14":
                cols.append('metallicity')
            elif planet_model == "ChRo16":
                cols.append('core_comp')

            values = row[cols].values
            run_mix_age.loc[index] = values # add track params to new dataframe
        
    # add age to dataframe
    run_mix_age["age"] = age_mix

    if (path_save != None) and (filename != None):
        run_mix_age.to_csv(os.path.join(path_save, filename+"_age.csv"))

    return run_mix_age