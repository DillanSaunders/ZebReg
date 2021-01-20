from data_processing import colour_map
import open3d as o3d
import numpy as np
import copy

def execute_global_registration(source_processed, target_processed, source_fpfh,
                                target_fpfh, voxel_size = 10, ransac_dist_modifier=1.5, ransac_edge_length=0.9, verbose = False):
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
        
    result = o3d.registration.registration_ransac_based_on_feature_matching(
        source_processed, target_processed, source_fpfh, target_fpfh, distance_threshold,
        o3d.registration.TransformationEstimationPointToPoint(False), 4, [
            o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(ransac_edge_length),
            o3d.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.registration.RANSACConvergenceCriteria(4000000, 500))
    
    return result

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

    result = o3d.registration.registration_icp(
        source_processed, target_processed, distance_threshold, ransac_transform,
        o3d.registration.TransformationEstimationPointToPlane(),
        o3d.registration.ICPConvergenceCriteria(max_iteration = 2000))
    
    return result

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
    
    result_icp_colored = o3d.registration.registration_colored_icp(
        source_colorreg, target_colorreg, voxel_radius, current_transformation,
        o3d.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                relative_rmse=1e-6,
                                                max_iteration=100))
    
    current_transformation = result_icp_colored.transformation
        
    return result_icp_colored

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
        mae = (sum(abs(source_val - target_val))) / (source_val.shape[0])
    return mae

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
 
def obtain_registration_metrics(target, source_color, target_color, registration_result):
    """ For a particular registration result, displays the fitness, inlier RMSE and MAE estimate. Also describes the correspondence 
    map properties. 
    
    Parameters:
    ----------
    target: geometry.PointCloud
        Target point cloud
    source_color: np.array
        Source color channel (raw) intensity values
    target_color: np.array
        Target color channel (raw) intensity values
    registration_result: registration.RegistrationResult
        Registration result of alignment
    """
    
    print("--- Registration results --- ")
    print(f"Fitness: {registration_result.fitness*100:.2f}%")
    print(f"Inlier RMSE: {registration_result.inlier_rmse}")
    print(f"MAE: {calculate_mae(source_color, target_color, registration_result):.2f}\n---------------------------------------")    
    
    corr_map = np.array(registration_result.correspondence_set)
    source_indices = corr_map[:,0]
    target_indices = corr_map[:,1]
                
    target_new = copy.deepcopy(target)
    num_target = np.array(target.points).shape[0]
    target_range = np.arange(0, num_target)
    
    unmapped_targets =np.where(np.invert(np.in1d(target_range, target_indices)))[0]
    target_repeats = {i:list(target_indices).count(i) for i in target_indices if list(target_indices).count(i) > 1}
    unique_target_indices = [x for x in target_indices if x not in target_repeats]

    print("--- Correspondence map properties --- ")
    print(f"{len(unmapped_targets)} ({(len(unmapped_targets)/ num_target)*100:.3f}%) unmapped targets.")
    print(f"{len(target_repeats)} ({(len(target_repeats)/ num_target)*100:.3f}%) targets that are mapped by multiple source points.")
    print(f"{len(unique_target_indices)} ({(len(unique_target_indices)/ num_target)*100:.3f}%) targets that are uniquely mapped by a single source point.")
    
    if len(unmapped_targets) + len(target_repeats) + len(unique_target_indices) == len(target.points):
        print(f"All {len(target.points)} target points are accounted for.")
             
def map_source2target(source, target, source_color, target_color, registration_result, method = "median", verbose = False):
    """ Returns the registered target point cloud, where the color intensity values of multiply mapped target points are imputed by
    a chosen averaging method.
    
    Parameters:
    ----------
    source: geometry.PointCloud
        Source point cloud
    target: geometry.PointCloud
        Target point cloud
    source_color: np.array
        Source color channel intensity values
    target_color: np.array
        Target color channel intensity values
    registration_result: registration.RegistrationResult
        Registration result of alignment
    method: str
        A choice between "mean" or "median" averaging for imputing the intensity of multiply mapped target points
    verbose: boolean
        Prints the color numpy arrays before and after the mapping is performed.
        
    Return:
    --------
    target_new: geometry.PointCloud
        Updated target point cloud
    mapped_col_range: matplotlib.cm.ScalarMappable object 
        
    """
    corr_map = np.array(registration_result.correspondence_set)
    source_indices = corr_map[:,0]
    target_indices = corr_map[:,1]
            
    target_new = copy.deepcopy(target)
    color_list = np.zeros(shape=(target_color.shape))
    
    target_repeats = {i:list(target_indices).count(i) for i in target_indices if list(target_indices).count(i) > 1}
    unique_target_indices = [x for x in target_indices if x not in target_repeats]
    
    if method == "median" or method == "Median": 
        print("Using median averaging")
        for ind in target_repeats:
            bool_mask = target_indices == ind
            source_indices_repeat = source_indices[bool_mask]
            color_list[ind] = np.median(source_color[source_indices_repeat])

        for ind in unique_target_indices:
            bool_mask = target_indices == ind
            source_indices_unique = source_indices[bool_mask]
            color_list[ind] = source_color[source_indices_unique]
            
    elif method == "mean" or method == "Mean" or method == "average" or method == "Average":
        print("Using mean averaging")
        for ind in target_repeats:
            bool_mask = target_indices == ind
            source_indices_repeat = source_indices[bool_mask]
            color_list[ind] = np.mean(source_color[source_indices_repeat])

        for ind in unique_target_indices:
            bool_mask = target_indices == ind
            source_indices_unique = source_indices[bool_mask]
            color_list[ind] = source_color[source_indices_unique]
    else:
        raise Exception("Unrecognised method used. Only mean/average or median functions are permitted.") 
    
    if verbose:
        print("before assignment", np.array(target_new.colors))
    
    mapped_rgb, mapped_col_range=colour_map(color_list,"viridis")
            
    target_new.colors =o3d.utility.Vector3dVector(mapped_rgb)
            
    if verbose:
        print("after assignment", np.array(target_new.colors))
        print(np.all([mapped_rgb, np.array(target_new.colors)]))
    
    return (target_new,mapped_col_range)



    
