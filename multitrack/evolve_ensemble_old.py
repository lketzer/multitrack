def evolve_ensamble(planets_chunks,
                    t_final, initial_step_size,
                    epsilon, K_on, beta_on,
                    evo_track_dict_list, path_save):
    """
    Function to evolve an ensemble of planets. This function parallelizes the multiprocessing_func (i.e. evolve_one_planet).
    The N planets in each planets_chunks sublist are parallelized and run at the same time (at least I think this is the code
    is doing...). Once one "chunck" of planets is done, the code moves on to the next.
    (Source: https://medium.com/@urban_institute/using-multiprocessing-to-make-python-code-faster-23ea5ef996ba)

    Parameters:
    -----------
    planets_chunks (list of list): it is a list of sublists of [folder, planet] pairs.
                                   Each sublist (i.e. a list of planets) in the list will be run seperately.

    t_final (float): end time of the integration (e.g. 3, 5, 10 Gyr)

    initial_step_size (float): Initial step size for the integration (e.g. 0.1 Myr)

    epsilon (float): Evaporation efficiency (constant); should be a value between 0 and 1 in our interpretation

    K_on (str): "yes" to use K-estimation, or "no" to set K=1 (see Beta_K_functions in platypos_package)

    beta_on (str): "yes" to use beta-estimation, or "no" to set beta=1 (see Beta_K_functions in platypos_package)

    evo_track_dict_list (list): list with track parameters (can be one track only); either all 9 parameters per track are specified,
                                or a list of track parameters which are different is passed.
                                This is for a sample where each planet has a different host and the tracks have to be tailored to the
                                host star mass/saturation luminosity.

    path_for_saving (str): file path to master folder in which the individual planet folders lie

    Returns:
    --------
    None


    INFO on how multi-processing in Python works:
    In case of multi-processing, each process runs on different core depending upon the number of cores on the machine.
    Hence there’s no need of a GIL while doing multi-processing since all processes are independent. Thus multi-processing
    is actually providing us the actual parallelism in python. But it works only when done right. If we have 12 tasks and
    a 4 core machine then one process per core would be the ideal case giving the true parallelism we want. If we spawn
    more processes than the number of cores each process will compete for resources, this would lead to context switching
    and thus concurrency instead of parallelism.
    (https://towardsdatascience.com/python-multi-threading-vs-multi-processing-1e2561eb8a24)
    """

    for i in range(len(planets_chunks)):
        # here is where I use the multiprocessing package
        starttime = time.time()

        # Create a list of processes to run (so all the planets in each
        # chunck/sublist)
        processes = []
        for j in range(len(planets_chunks[i])):
            pl_folder_pair = planets_chunks[i][j]  # get folder-planet pair
            path_for_saving = path_save + \
                planets_chunks[i][j][0] + "/"  # get corresponding folder
            # create process and append to list (# instantiating process with
            # arguments)
            p = multiprocessing.Process(
                target=evolve_one_planet, args=(pl_folder_pair, t_final,
                                                initial_step_size, epsilon,
                                                K_on, beta_on,
                                                evo_track_dict_list,
                                                path_for_saving))
            processes.append(p)
            # Start the process’s activity.
            p.start()

        # complete the processes by terminating each single one
        for process in processes:
            # "join() says that the code in __main__ must wait until all our tasks are complete before continuing!"
            # Make sure Python waits for the process to terminate and then
            # exits the completed processes
            process.join()

        t = (time.time() - starttime) / 60
        print('That took {} minutes'.format(t))
