import numpy as np

import platypos.planet_models_LoFo14 as plmoLoFo14
import platypos.planet_models_ChRo16 as plmoChRo16
  

def write_file_thermal_contraction_radius(planet, t_final, 
                                          path_for_saving, folder):
    """ Get radius of planets only considering thermal contraction.
        Works for LoFo14 & ChRo16. """
    
    # calculate radius at t_final if planet would undergo ONLY thermal
    # contraction and save to file
    if planet.type == "LoFo14":
        R_thermal = plmoLoFo14.calculate_planet_radius(planet.core_mass,
                                                       planet.fenv,
                                                       t_final,
                                                       planet.flux,
                                                       planet.metallicity)
    elif planet.type == "ChRo16":
        R_thermal = plmoChRo16.calculate_planet_radius(planet.core_mass,
                                                       planet.fenv,
                                                       planet.flux,
                                                       t_final,
                                                       planet.core_comp)
    filename = folder + "_thermal_contr.txt"
    path = os.path.join(path_for_saving, filename)
    if not os.path.exists(path):
        with open(path, "w") as t:
            file_content = "t_final,R_th\n"\
                + str(t_final) + ","\
                + str(R_thermal)
            t.write(file_content)
