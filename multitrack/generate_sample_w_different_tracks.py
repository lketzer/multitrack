import pandas as pd
import scipy.stats


def generate_different_tracks_sample(run, new_tracks=True, tsat100=True):
    """ 
    Use my previously generated track probability distribution to generate a
    planetary sample consisting of host stars evolving through different 
    activity tracks. Each planet has one track drawn (based on a given 
    probability distribution) and the respective final parameter for this track
    is stored in a table.
    
    Input:
    ------
    run(dataframe): previously read in
    tsat100: if True, I likely have a run with 10 tracks (9 tracks from my 
             distributions plus a 10th track which has a t_sat
             of 100 Myr, which I don't want included in my mixed sample)
             -> make sure that the right tracks are chosen
             (e.g. in my track list tsat=100 will be track #7, which is normally
             a track number from my distro tracks (#1-9))
    
    Returns:
    --------
    run_mix_skew: dataframe with init & final params for the skewed track distro 
    run_mix_norm: dataframe with init & final params for the normal track distro
    
    NOTE: works of course only for runs with more than one track!
    """
    
    # read in the files with the probabilities
    if new_tracks == True:
        df_norm = pd.read_csv("./track_probabilities_9tracks_normally_distributed_tsat_new.csv")
        df_skew = pd.read_csv("./track_probabilities_9tracks_skew_distributed_tsat_new.csv")
    else:
        df_norm = pd.read_csv("./track_probabilities_9tracks_normally_distributed_tsat.csv")
        df_skew = pd.read_csv("./track_probabilities_9tracks_skew_distributed_tsat.csv")
        
    # create the discrete probability mass functions
    custm_skew = scipy.stats.rv_discrete(name='custm_skew',
                                   values=(df_skew["track"],
                                           df_skew["probability"]))
    custm_norm = scipy.stats.rv_discrete(name='custm_norm',
                                   values=(df_norm["track"],
                                           df_norm["probability"]))
    
    # create dataframe with final parameters (mixed tracks) [easier for plotting]
    track_mix_skew = []
    track_mix_norm = []
    
    try:
        run["metallicity"]
        planet_model = "LoFo14"
    except:
        planet_model = "ChRo16"
    
    columns = ['a', 'period', 'core_mass', 'core_radius', 'fenv', 'mass',\
               'radius', 'age', 't_eq', 't_final', 'R_final', 'M_final', 'Lx0',\
               'Lx_final', 'fenv_final', 'frac_env_lost',\
               "t_ML_final_Gyr",\
               "Mdot_final", \
               "Mdot_final_M_earth_per_Gyr",\
               "Mdot_final_M_earth_per_yr",\
               "Mdot_init",\
               "Mdot_init_M_earth_per_Gyr",\
               "Mdot_init_M_earth_per_yr",\
               "beta_init", "beta_final",\
               'mass_star', 'Lbol', 'Lx_age', 'flux', 'flux_EARTH']
    
    if planet_model == "LoFo14":
        columns.append('metallicity')
    elif planet_model == "ChRo16":
        columns.append('core_comp')

    run_mix_skew = pd.DataFrame(columns=columns)
    run_mix_norm = pd.DataFrame(columns=columns)

    for index, row in run.iterrows():
        # draw track number from skewed distro (1 to 9)
        tr_num = track_no_true = custm_skew.rvs()
        if tsat100 == True:
            # track 7,8,9 are actually 8,9,10 in the run dataframe
            if (tr_num == 7) or (tr_num == 8) or (tr_num == 9):
                tr_num += 1
        track_mix_skew.append(track_no_true) # add original track number (from 1-9 only)
        # get corresponding parameters

        tr_num = str(tr_num)
        cols = ['a', 'period', 'core_mass', 'core_radius', 'fenv', 'mass',\
                'radius', 'age', 't_eq', ''.join(['t', tr_num]), \
                ''.join(['R', tr_num]),  ''.join(['M', tr_num]),\
                ''.join(['Lx0_', tr_num]), ''.join(['Lx', tr_num]),\
                ''.join(['fenv_final', tr_num]),\
                ''.join(['frac_env_lost', tr_num]),\
                "t_ML"+tr_num+"_Gyr",\
                "Mdot"+tr_num,\
                "Mdot"+tr_num+"M_earth_per_Gyr",\
                "Mdot"+tr_num+"M_earth_per_yr",\
                "Mdot0_"+tr_num,\
                "Mdot0_"+tr_num+"M_earth_per_Gyr",\
                "Mdot0_"+tr_num+"M_earth_per_yr",\
                "beta0_"+tr_num, "beta"+tr_num,\
                'mass_star', 'Lbol', 'Lx_age', 'flux', 'flux_EARTH']
                
        if planet_model == "LoFo14":
            cols.append('metallicity')
        elif planet_model == "ChRo16":
            cols.append('core_comp')
    
        values = row[cols].values
        run_mix_skew.loc[index] = values # add track params to new dataframe

        # repeat same process for normal distro
        tr_num = track_no_true = custm_norm.rvs()
        if tsat100 == True:
            # track 7,8,9 are actually 8,9,10 in the run dataframe
            if (tr_num == 7) or (tr_num == 8) or (tr_num == 9):
                tr_num += 1
        track_mix_norm.append(track_no_true)

        tr_num = str(tr_num)
        cols = ['a', 'period', 'core_mass', 'core_radius', 'fenv', 'mass',\
                'radius', 'age', 't_eq', ''.join(['t', tr_num]), \
                ''.join(['R', tr_num]),  ''.join(['M', tr_num]),\
                ''.join(['Lx0_', tr_num]), ''.join(['Lx', tr_num]),\
                ''.join(['fenv_final', tr_num]),\
                ''.join(['frac_env_lost', tr_num]),\
                "t_ML"+tr_num+"_Gyr",\
                "Mdot"+tr_num,\
                "Mdot"+tr_num+"M_earth_per_Gyr",\
                "Mdot"+tr_num+"M_earth_per_yr",\
                "Mdot0_"+tr_num,\
                "Mdot0_"+tr_num+"M_earth_per_Gyr",\
                "Mdot0_"+tr_num+"M_earth_per_yr",\
                "beta0_"+tr_num, "beta"+tr_num,\
                'mass_star', 'Lbol', 'Lx_age', 'flux', 'flux_EARTH']
                
        if planet_model == "LoFo14":
            cols.append('metallicity')
        elif planet_model == "ChRo16":
            cols.append('core_comp')
            
        values = row[cols].values
        run_mix_norm.loc[index] = values

    # add track number to dataframe
    run_mix_skew["track#"] = track_mix_skew
    run_mix_norm["track#"] = track_mix_norm
    
    return run_mix_skew, run_mix_norm


def generate_different_tracks_sample_snapshots(run_st, run_mix):
    """
    Creates a dataframe which contains all the the snapshots
    for the previously generated mixed track sample.
    
    Input:
    ------
    run_st (df): snapshots dataframe
    run_mix (df): dataframe from generate_different_tracks_sample
                  (either skew or normal)
    Returns:
    --------
    run_st_mix (df): dataframe which contains the snapshots for the
                     previously generated mixed track sample
                     
    """
    
    # create starting dataframe with all initial values from run_mix
    cols_common = [c for c in run_mix.columns if "final" not in c]
    run_st_mix = run_mix[cols_common]
    
    # extract the snapshot times from the dataframe
    snapshot_times = []
    cols = [c for c in run_st.columns if "e+" in c]
    for c in cols:
        st = c.split("_")[-1]
        if (st not in snapshot_times) and ("yr" not in st):
            snapshot_times.append(st)  

    # based on the track for each panet specified in run_mix, extract all the
    # values for all of the times in snapshot times and apend to the starting df
    for st in snapshot_times:
        # do the following for each snapshot time
        cols = ['_'.join(['t', st]), '_'.join(['R', st]),  '_'.join(['M', st]), \
                '_'.join(['Lx', st]), '_'.join(['Lx0', st]), '_'.join(['fenv', st]), \
                '_'.join(['frac_env_lost', st]), '_'.join(['beta0', st]), \
                '_'.join(['beta', st]), '_'.join(['Mdot0', st]), \
                'Mdot0_'+st+'M_earth_per_Gyr', 'Mdot0_'+st+'M_earth_per_yr', \
                "Mdot_"+st, "Mdot_"+st+"M_earth_per_Gyr", "Mdot_"+st+"M_earth_per_yr", \
                "t_ML_"+st+"_Myr", "t_ML_"+st+"_Gyr"]
        
        # build up dataframe for one snapshot time
        run_one_st_mix = pd.DataFrame(columns=cols, index=run_mix.index)
        
        columns = run_st.columns
        for planet, row in run_mix.iterrows():
            # get all columns which contain track + "_" + st 
            track = str(row["track#"])
            track_columns = [c for c in columns if (track + "_" + st) in c]
            run_one_st_mix.loc[planet] = run_st.loc[planet, track_columns].values
            
            #track_columns = [c for c in run_st.loc[planet].index if "_".join([track, st]) in c]
            #run_one_st_mix.loc[planet] = run_st.loc[planet][track_columns].values
        # append snapshot time df to final df
        run_st_mix = pd.concat([run_st_mix, run_one_st_mix], axis=1)    
    
    return run_st_mix
