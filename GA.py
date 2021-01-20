from deap import algorithms, base, creator, tools
from align_func import execute_global_registration, icp_registration, colored_icp, calculate_mae
import random
import numpy as np

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

