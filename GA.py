from deap import algorithms, base, creator, tools
import random
import numpy as np
import open3d as o3d
import copy 
import pandas as pd
import os
from datetime import date

def colour_map(intensity_array,cmap):
    """ Converts the scalar intensity values into RGB values for visualisation. 
    
    Parameters:
    ----------
    intensity_array: list
        List of length 2 containing unnormalised intensity values of the imaging color channel
    cmap: string
        Name of the colour map
    
    Returns:
    ----------
    rgb_array: np.array
        n x 3 array where n is the number of points in the point cloud
    lut: matplotlib.cm.ScalarMappable object 
    """
    
    min_intensity=np.amin(intensity_array)
    max_intensity=np.amax(intensity_array)
    norm = col.Normalize(vmin=min_intensity, vmax=max_intensity) #create scale from highest to lowest value
    col_map = cm.get_cmap(cmap) #load colourmap/LUT
    lut = cm.ScalarMappable(norm=norm, cmap=col_map) #this combines both
    rgb_array= np.empty((len(intensity_array),4)) #create empty array for RBG values

    #loop over every intensity converting it to RBG value in the colour map
    for index in range(0,len(intensity_array)):
        intensity_value=intensity_array[index,0]
        rgb_array[index,:]=lut.to_rgba(intensity_value)
    
    rgb_array=rgb_array[:,0:3] #removes 4th column this is 'alpha', open3d only takes RBG
    
    return rgb_array, lut

def min_max_normalisation(intensity_values):
    """ Helper function to return the min-max normalised intensity values for a color channel
    
    Parameters: 
    ----------
    intensity_values: np.array 
        Unnormalised color channel values
    
    Return:
    result: np.array
        Min-max normalised intensity values.
    ----------
    
    """
    min_val = np.amin(intensity_values)
    max_val = np.amax(intensity_values)
    output = (intensity_values - min_val) / (max_val - min_val) *100 
    return output


def mae(source_val, target_val):
    """Helper function to calculate the mean absolute error between the source and target color intensity channels
    
    Parameters:
    ----------
    source_val: np.array
        Source intensity values
    target_va;: np.array
        Target intensity values
    
    Return:
    ---------
    mae: float
        Mean absolute error 
    """
    
    try:
        assert source_val.shape == target_val.shape   
        
    except:
        print("source and target intensity values must be of the same shape")
    
    else:
        try:
            mae = (sum(abs(source_val - target_val))) / (source_val.shape[0])
        except ZeroDivisionError:
            mae = [10]
    return mae


def execute_global_registration(source_processed, target_processed, source_fpfh,
                                target_fpfh, voxel_size = 10, ransac_dist_modifier=1.5, ransac_edge_length=0.9, 
                                ransac_mutual_filter = True, verbose = False):
    """ Implements the RANSAC registration based on feature matching and returns a registration.RegistrationResult object.
    
    Source: Adapted from open3d global registration documentation: http://www.open3d.org/docs/release/tutorial/pipelines/global_registration.html
    
    Parameters:
    ----------
    source_processed: geometry.PointCloud
        Source point cloud after downsampling (if downsample=True) and normal estimation
    target_processed: geometry.PointCloud
        Target point cloud after downsampling (if downsample=True) and normal estimation
    source_fpfh: registration.Feature
        Source point cloud fpfh information
    target_fpfh: registration.Feature
        Target point cloud fpfh information
    voxel_size: float, optional
        Multiplied with the ransac_dist_modifier to yield the distance threshold used by CorrespondenceCheckerBasedOnDistance
    ransac_dist_modifier:float, optional
        Multiplied with the voxel_size to yield the distance threshold used by CorrespondenceCheckerBasedOnDistance
    ransac_edge_length: float, optional
        Input to CorrespondenceCheckerBasedOnEdgeLength
    
    Return:
    ----------
    result: registration.RegistrationResult
        Result of RANSAC alignment
    """
    
    distance_threshold = voxel_size * ransac_dist_modifier
    
    if verbose:
        print(":: RANSAC registration on point clouds.")
        print("   Since the  voxel size is %.3f," % voxel_size)
        print("   we use a liberal distance threshold %.3f." % distance_threshold)
        
    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_processed, target_processed, source_fpfh, target_fpfh, ransac_mutual_filter, distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 4, [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(ransac_edge_length),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                    distance_threshold)
            ], o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))
    
    return result_ransac

def icp_registration(source_processed, target_processed, source_fpfh, target_fpfh, ransac_transform, voxel_size = 10, icp_dist_check = 1):  
    """ Implements the Point-to-Plane ICP registration algorithm and returns a registration.RegistrationResult object.
    
    Source: Adapted from open3d ICP registration documentation: http://www.open3d.org/docs/release/tutorial/pipelines/icp_registration.html?highlight=point%20plane%20icp%20registration
    
     Parameters:
    ----------
    source_processed: geometry.PointCloud
        Source point cloud after downsampling (if downsample=True) and normal estimation
    target_processed: geometry.PointCloud
        Target point cloud after downsampling (if downsample=True) and normal estimation
    source_fpfh: registration.Feature
        Source point cloud fpfh information
    target_fpfh: registration.Feature
        Target point cloud fpfh information
    ransac_transform: np.array of dimensions 4x4
        Initial transformation estimation. We use the RANSAC global alignment as the rough initial alignment.
    voxel_size: float, optional
        Multiplied with icp_dist_check to determine the maximum correspondence points-pair distance.
    icp_dist_check: float, optional
        Multiplied with voxel_size to determine the maximum correspondence points-pair distance.

    Return:
    ----------
    result: registration.RegistrationResult
        Result of Point-to-Plane ICP alignment
    """
    
    distance_threshold = voxel_size * icp_dist_check

    result_icp = o3d.pipelines.registration.registration_icp(
            source_processed, target_processed, distance_threshold, ransac_transform.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration = 2000))
    
    return result_icp

def colored_icp(source_pcd, target_pcd, ransac_transform, voxel_radius, coloredICP_maxnn, downsample = False):
    """ Implements the Colored ICP registration algorithm and returns a registration.RegistrationResult object.
    
    Source: Adapted from open3d ICP registration documentation:http://www.open3d.org/docs/release/tutorial/pipelines/colored_pointcloud_registration.html
    
     Parameters:
    ----------
    source_pcd: geometry.PointCloud
        Source point cloud 
    target_pcd: geometry.PointCloud
        Target point cloud 
    ransac_transform: np.array of dimensions 4x4
        Initial transformation estimation. We use the RANSAC global alignment as the rough initial alignment.
    voxel_radius: float
        Used in the estimate normals function, and acts as the max correspondence distance in the registration_colored_icp function
    max_iter: float
        Max iteration in the ICPconvergence algorithm
    downsample: boolean, optional
        Determines whether downsampling should be performed or not

    Return:
    ----------
    result: registration.RegistrationResult
        Result of Colored ICP alignment
    """
    
    current_transformation = ransac_transform

    source_colorreg = copy.deepcopy(source_pcd)
    target_colorreg = copy.deepcopy(target_pcd)

    if downsample:
        source_colorreg = source_colorreg.voxel_down_sample(voxel_radius)
        target_colorreg = target_colorreg.voxel_down_sample(voxel_radius)
        
    source_colorreg.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_radius * 2, max_nn=coloredICP_maxnn))
    
    target_colorreg.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_radius * 2, max_nn=coloredICP_maxnn))
    
    result_icp_colored = o3d.pipelines.registration.registration_colored_icp(
            source_colorreg, target_colorreg, voxel_radius, current_transformation,
            o3d.pipelines.registration.TransformationEstimationForColoredICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                    relative_rmse=1e-6,
                                                    max_iteration=100))
    
    current_transformation = result_icp_colored.transformation
        
    return result_icp_colored

def calculate_mae(source_color, target_color, registration_result):
    """Returns the mean absolute error between the source and target color intensity channels
    
    Parameters:
    ----------
    source_color: np.array
        Source color channel (raw) intensity values
    target_color: np.array
        Target color channel (raw) intensity values
    registration_result: registration.RegistrationResult
        Registration result of alignment
    
    Return:
    ---------
    mae: float
        Mean absolute error 
    """
    
    norm_colors_source = min_max_normalisation(source_color)
    norm_colors_target = min_max_normalisation(target_color)
    corr_result_ransac = np.array(registration_result.correspondence_set)
    source_indices_ransac = corr_result_ransac[:,0]
    target_indices_ransac = corr_result_ransac[:,1]
    source_color_norm = norm_colors_source[source_indices_ransac]
    target_color_norm = norm_colors_target[target_indices_ransac]

    return mae(source_color_norm, target_color_norm)[0]
 

def evaluation_ransac(individual,source_processed, target_processed, source_fpfh, target_fpfh):
    
    """ Perform RANSAC registration for a particular combination of hyperparameters, and return the fitness, inlier_mse and mae values in a 3-element tuple."""
        
    result_ransac = execute_global_registration(source_processed, target_processed,
                                            source_fpfh, target_fpfh,
                                            individual[0], individual[5],individual[6])
    
    corr_result_ransac = np.array(result_ransac.correspondence_set)
    
    # If the registration fails, no registration result is returned. Penalise the combination of hyperparameters.
    
    if len(corr_result_ransac) == 0:
        result_ransac.inlier_rmse = 10000
        result_mae_ransac = 10000
    
    else:
        result_mae_ransac = calculate_mae(source_color, target_color, result_ransac)
        
    return (result_ransac.fitness,result_ransac.inlier_rmse, result_mae_ransac)

def evaluation_icp(individual,source_processed, target_processed, source_fpfh, target_fpfh):
    
    """ Perform ICP registration for a particular combination of hyperparameters, and return the fitness, inlier_mse and mae values in a 3-element tuple.
    """
    
    result_ransac = execute_global_registration(source_processed, target_processed,
                                            source_fpfh, target_fpfh,
                                            individual[0], individual[6],individual[7])
    
    result_icp= icp_registration(source_processed, target_processed, source_fpfh, target_fpfh, result_ransac.transformation, individual[0], individual[8])
    
    corr_result_icp = np.array(result_icp.correspondence_set)
    
    # If the registration fails, no registration result is returned. Penalise the combination of hyperparameters.
    
    if len(corr_result_icp) == 0:
        result_icp.inlier_rmse = 10000
        result_mae_icp = 10000
    
    else:
        result_mae_icp = calculate_mae(source_color, target_color, result_icp)
    
    return (result_icp.fitness,result_icp.inlier_rmse, result_mae_icp)

def evaluation_colored_icp(individual, source_processed, target_processed, source_fpfh, target_fpfh, source_color, target_color):
    
    """
    Perform colored ICP registration for a particular combination of hyperparameters, and return the fitness, inlier_mse and mae values in a 3-element tuple.
    """  
    
    result_ransac = execute_global_registration(source_processed, target_processed, source_fpfh, target_fpfh, individual[0], individual[5],individual[6],verbose = False)
    
    result_colored_icp = colored_icp(source_processed, target_processed, result_ransac.transformation, individual[0], individual[7])

    corr_result_col_icp = np.array(result_colored_icp.correspondence_set)
    
    # If the registration fails, no registration result is returned. Penalise the combination of hyperparameters.
    
    if len(corr_result_col_icp) == 0:
        result_colored_icp.inlier_rmse = 10000
        result_mae_col_icp = 10000
    
    else:
        result_mae_col_icp = calculate_mae(source_color, target_color, result_colored_icp)
    
    return (result_colored_icp.fitness,result_colored_icp.inlier_rmse, result_mae_col_icp, corr_result_col_icp)

def mutation(individual, indpb):
    """ Definition of the mutation operator """
    
    size = len(individual)
    
    for i, p in zip(range(size),indpb):
        if random.random() < p: #If the mutation occurs:
            current_val = individual[i]
            random_num = random.random()  #Generate a random number 
            # The hyperparameter value is incremented/decremented with equiprobability. 
            if random_num >= 0.5:
                individual[i] -= 1
                if individual[i] <= 0:
                    individual[i] = current_val # In the event that the mutation causes the val to be <=0, we keep the original value.
            else:
                individual[i] +=1       
    return [individual,]



# We define a generator operator, with the alias `generator`, which generates the values for the individuals.

def generator():
    """Describe how to generate the range of values for each parameter.
    
    Returns:
    ---------
    voxel size: float
    norm_radius_modifier: float, optional
        Multiplied with voxel_size to compute the KDTree search radius for estimate_normals
    norm_maxnn: float 
        Max number of neighbors in the KDTree search in estimate_normals
    fpfh_radius_modifier: float 
        Multiplied with voxel_size to compute the KDTree search radius for compute_fpfh_feature
    fpfh_maxnn: float 
        Max number of neighbors in the KDTree search in compute_fpfh_feature
    ransac_dist_modifier:float
        Multiplied with the voxel_size to yield the distance threshold used by CorrespondenceCheckerBasedOnDistance
    ransac_edge_length: float
        Input to CorrespondenceCheckerBasedOnEdgeLength
    
    """
    voxel_size = random.choice(range(1,100,5))
    norm_radius_modifier= random.choice(range(1,100,5))
    norm_maxnn= random.choice(range(1,100,5))
    fpfh_radius_modifier = random.choice(range(1,100,5))
    fpfh_maxnn= random.choice(range(1,100,5))
    ransac_dist_modifier= random.choice(range(1,100,5))
    ransac_edge_length= random.choice(range(1,100,5))
    coloredICP_maxnn = random.choice(range(1,100,5))

    return  [voxel_size, norm_radius_modifier, norm_maxnn, fpfh_radius_modifier, fpfh_maxnn, ransac_dist_modifier, ransac_edge_length, coloredICP_maxnn] 

def GA(source_processed, target_processed, source_fpfh, target_fpfh, source_color, target_color, 
       NDIM = 10, NOBJ = 3, p = 12, pop_size = 30, max_gen = 50, CXPB = 1.0, MUTPB = 1.0, 
       iter_ = 0, run_name = "./test"):
    
    # results.fitness (MAX), results.inlier_mse (MIN), mae (MIN)
    creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0, -1.0)) 

    # A list-type individual with a fitness attribute
    creator.create("Individual", list, fitness=creator.FitnessMulti)

    # Instantiate a Toolbox to register all the evolutionary operators.
    toolbox = base.Toolbox()
    
    NDIM = NDIM
    NOBJ = NOBJ
    p = p
    toolbox.pop_size = pop_size
    toolbox.max_gen = max_gen
    CXPB = CXPB
    MUTPB = MUTPB
    
    ref_points = tools.uniform_reference_points(NOBJ, p)

    """ Registering operators""" 
    toolbox.register("generator", generator)

    # Structure initializers
    # define 'individual' to be a single individual taking up the values generated by the toolbox.generator. So we don't 
    # need to repeat the toolbox.generator function. 
    # This gives us flexibilty to define each parameter with its unique distribution, instead of keeping the distribution
    # the same and applying it repeatedly across each parameter.

    toolbox.register("individual", tools.initIterate, creator.Individual, 
        toolbox.generator)

    # define the population to be a list of individuals # We don't define n here, but in the main body to give flexibility to num individuals.
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Operator registration
    #----------
    # register the goal / fitness function
    toolbox.register("evaluate", evaluation_colored_icp, source_processed = source_processed, 
                     target_processed = target_processed, source_fpfh = source_fpfh, target_fpfh = target_fpfh,
                    source_color = source_color, target_color = target_color)

    # register the crossover operator
    toolbox.register("mate", tools.cxTwoPoint)

    # register a mutation operator 

    toolbox.register("mutate", mutation, indpb = [0.2]*NDIM)

    toolbox.register("select", tools.selNSGA3, ref_points = ref_points )


    random.seed(iter_ )
    
    # Initialize statistics object
     ## Creating logbook for recording statistics
        
    stats_o3d_fitness = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats_o3d_rmse = tools.Statistics(lambda ind: ind.fitness.values[1])
    stats_o3d_mae = tools.Statistics(lambda ind: ind.fitness.values[2])
    mstats = tools.MultiStatistics(o3d_fitness=stats_o3d_fitness, 
                                   o3d_rmse =stats_o3d_rmse,
                                   o3d_mae = stats_o3d_mae)
    
    mstats.register("mean", lambda ind: round(sum(ind)/len(pop),3))
    mstats.register("max", lambda ind: round(np.max(ind),3))
    mstats.register("min", lambda ind: round(np.min(ind),3))
    
    logbook = tools.Logbook()
    logbook.header = ["gen", "evals", "o3d_fitness", "o3d_rmse", "o3d_mae"]
    logbook.chapters["o3d_fitness"].header = ["mean", "max"]
    logbook.chapters["o3d_rmse"].header = ["mean", "min"]
    logbook.chapters["o3d_mae"].header = ["mean", "min"]
    
    pop = toolbox.population(n= toolbox.pop_size)
        
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit[0:3]

    #Compile statistics about the population
    record = mstats.compile(pop)
    logbook.record(gen=0, evals=len(invalid_ind), **record)
    print(logbook.stream)
    
    # Manual logging results with corr maps
    corr_map_results = []
    
    for gen in range(1, toolbox.max_gen):

        
        #Apply crossover and mutation to generate new offsprings. Return a list of varied individuals that are independent of their parents.
        offspring = algorithms.varAnd(pop, toolbox, CXPB, MUTPB)
        fitnesses_offspring = [ind.fitness for ind in offspring if ind.fitness.valid]

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit[0:3]
            if fit[3].size: #If a correspondence map is present:
                corr_map_info = {"gen" : gen, "individual" : ind, "correspondence_set" : fit[3].tolist(), "fitness":ind.fitness.values[0],
                                 "inlier_rmse": ind.fitness.values[1], "mae": ind.fitness.values[2]}
                corr_map_results.append(corr_map_info)

        # Select the next generation population from parents and offspring
        pop = toolbox.select(pop + offspring, toolbox.pop_size)
      
        # Compile statistics about the new population
        record = mstats.compile(pop)
        logbook.record(gen=gen, evals=len(pop), **record)
        print(logbook.stream)
        
    corr_map_results_df = pd.DataFrame(corr_map_results)
    
    ## Saving best individuals and logbook
    newpath = run_name

    if not os.path.exists(newpath):
        os.makedirs(newpath)
           
    #np.save(file = f"{newpath}/result_ransac_df.csv", arr = best_k_ind_df)
    
    
    corr_map_results_df.to_csv(f'{newpath}/corrmapresults_coloredICP_run{iter_}.csv', index=False)
    
    results = {"gen" : logbook.select("gen"),
                "eval" : logbook.select("evals"),
                "o3d_fitness_mean" : logbook.chapters['o3d_fitness'].select("mean"),
                "o3d_fitness_max" : logbook.chapters['o3d_fitness'].select("max"),
                "o3d_rmse_mean" : logbook.chapters['o3d_rmse'].select("mean"),
                "o3d_rmse_min" : logbook.chapters['o3d_rmse'].select("min"),
                "o3d_mae_mean" : logbook.chapters['o3d_mae'].select("mean"),
                "o3d_mae_min" : logbook.chapters['o3d_mae'].select("min")}
    
    df_log = pd.DataFrame.from_dict(results) 
    df_log.to_csv(f'{newpath}/logbook_run{iter_}.csv', index=False) # Writing to a CSV file
    
