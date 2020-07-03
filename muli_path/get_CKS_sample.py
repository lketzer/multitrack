

def read_in_CKS_sample(path_to_directory):
    """Get dataset from CKS (California-Kepler-Survey): https://github.com/California-Planet-Search/cks-website/blob/master/readme.md
    Column explanations: https://www.astro.caltech.edu/~howard/cks/column-definitions.txt)"""
    
    df_cks_orig = pd.read_csv(path_to_directory)
    print(len(df_cks_orig))
    df_cks = df_cks_orig.copy()
    # sample selection based on Fulton 2017
    mask_confirmed = df_cks["koi_disposition"] == "CONFIRMED" # only select confirmed planets
    df_cks = df_cks[mask_confirmed]
    print(len(df_cks))
    # restrict sample to only the magnitudelimited portion of the larger CKS sample (Kp <14.2):
    mask_magnitude = df_cks["kic_kmag"] < 14.2
    df_cks = df_cks[mask_magnitude]
    print(len(df_cks))
    # planet-to-star radius ratio (R_pl/R_star) becomes uncertain at high impact parameters (b) due to degeneracies with limbdarkening. 
    # excluded KOIs with b > 0.7 to minimize the impact of grazing geometries.
    mask_impactparam = df_cks["koi_impact"] <= 0.7
    df_cks = df_cks[mask_impactparam]
    print(len(df_cks))
    # remove planets with orbital periods longer than 100 days in order to avoid domains of low completeness 
    # (especially for planets smaller than about 4 R_earth) and low transit probability.
    mask_period = df_cks["koi_period"] <= 100.
    df_cks = df_cks[mask_period]
    print(len(df_cks))
    # also excised planets orbiting evolved stars since they have somewhat lower detectability and less certain radii. 
    # implemented using an ad hoc temperature-dependent stellar radius filter:
    mask_evolved = df_cks["koi_srad"] <= 10**(0.00025*(df_cks["koi_steff"]-5500)+0.20)
    df_cks = df_cks[mask_evolved]
    print(len(df_cks))
    # also restrict sample to planets orbiting stars within the temperature range where we can extract precise stellar parameters 
    # from our high-resolution optical spectra (6500–4700 K).
    mask_temperature = (df_cks["koi_steff"] >= 4700) & (df_cks["koi_steff"] <= 6500)
    df_cks = df_cks[mask_temperature]
    print(len(df_cks))

    # drop columns which have missing stellar mass, planetary radius, semi-major axis or period
    df_cks = df_cks.dropna(axis=0, how="any", subset=["koi_sma", "koi_period", "koi_smass", "koi_prad"])
    df_cks.reset_index(inplace=True)

    return df_cks # final filtered sample

    # # inflate uncertainties on the histogram bin heights by the scaling factors to account for completeness
    # # corrections -> stellar properties of planet-hosting stars come from a different source and have higher 
    # # precision than the stellar properties for the full set of Kepler stars
    # df_R_bin_scaling = pd.read_csv("../supplementary_files/Fulton_Radius_correction.csv", sep="\s+", header=None, names=["bin", "factor"])
    # df_R_bin_scaling["bin_left"] = np.zeros(len(df_R_bin_scaling))
    # df_R_bin_scaling["bin_right"] = np.zeros(len(df_R_bin_scaling))

    # for index in df_R_bin_scaling.index:
    #     # Select rpw by index position using iloc[]
    #     bin_range = df_R_bin_scaling["bin"].iloc[index].split("–")
    #     df_R_bin_scaling.loc[index, "bin_left"] = float(bin_range[0])
    #     df_R_bin_scaling.loc[index, "bin_right"] = float(bin_range[1])
    #     factor = df_R_bin_scaling["factor"].iloc[index]

    # df_R_bin_scaling.head()    