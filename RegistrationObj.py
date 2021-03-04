from utils import colour_map, min_max_normalisation, mae
from open3d import JVisualizer
from xlrd import XLRDError
from sklearn import neighbors

import pandas as pd
import numpy as np
import open3d as o3d
import copy
import matplotlib.pyplot as plt
import tifffile.tifffile as tiff


class RegistrationObj():
    
    def __init__(self, pos_path_source, pos_path_target, color_path_source, color_path_target, 
                 algorithm, **kwargs):
        self.pos_path_source= pos_path_source
        self.source_pcd = o3d.geometry.PointCloud()
        self.pos_path_target= pos_path_target
        self.target_pcd = o3d.geometry.PointCloud()
        self.color_path_source= color_path_source
        self.source_color = None
        self.norm_source_color = None
        self.color_path_target= color_path_target
        self.target_color = None
        self.norm_target_color = None
        self.algorithm = algorithm
        assert (self.algorithm=='colored_icp'or self.algorithm=='icp' or self.algorithm=='ransac'), 'Invalid algorithm specified. Choose from: \n ransac -> runs only RANSAC global registration \n icp -> runs RANSAC and then local ICP registration \n colored_icp -> runs RANSAC and then local colored ICP registration'
        self.preprocessing = False
        self.result = None
        self.registration_result = o3d.pipelines.registration.RegistrationResult()
        self.mae = None
        self.registered_color = None
        self.registered_target = None
        self.mode = "knn"
        self.method = "Median"
        self.other_registered_channels = []
        self.manual_corr_map = None #To insert corr map obtained from GA. Will override corr_map from registration result
        
        
        """Setting the arguments for create_pcd"""
        self.pos_skiprows = None
        self.pos_usecols = None
        self.pos_header = 0
        self.color_skiprows = None
        self.color_usecols = None
        self.color_header = 0
        
        self.__dict__.update(kwargs)

        """ Setting the values of registration parameters"""
        self.voxel_size =  kwargs.get('voxel_size', 10)
        self.downsampling =  kwargs.get('downsampling', False)
        self.norm_radius_modifier = kwargs.get('norm_radius_modifier', 2)
        self.norm_maxnn = kwargs.get('norm_maxnn', 30)
        self.fpfh_radius_modifier = kwargs.get('fpfh_radius_modifier', 5)
        self.fpfh_maxnn = kwargs.get('fpfh_maxnn', 100)
        self.ransac_dist_modifier = kwargs.get('ransac_dist_modifier', 1.5)
        self.ransac_edge_length = kwargs.get('ransac_edge_length', 0.9)
        self.ransac_mutual_filter = kwargs.get('ransac_mutual_filter', True)
        self.icp_dist_check = kwargs.get('icp_dist_check', 1)
        self.coloredICP_maxnn = kwargs.get('coloredICP_maxnn', 50)
        self.other_source_channels = kwargs.get('other_source_channels', [])
        self.other_target_channels = kwargs.get('other_target_channels', []) 
        self.n_neighbors = kwargs.get('n_neighbors', 5)
        self.weights = kwargs.get('weights', "distance")
        
    def __str__(self):
        try:
            string = """--- Registration Object--- \nAlgorithm used = {0} \nPreprocessing performed = {1} \nRegistration performed = {2} 
            \nFitness = {3:.2f} \nInlier RMSE = {4:.2f} \nScaled inlier RMSE = {5:.2f} \nMAE = {6:.2f} 
            \nRegistered color = {7}""". format(self.algorithm, self.preprocessing, 
                                                          self.result, 
                                                         self.registration_result.fitness,
                                                         self.registration_result.inlier_rmse, 
                                                          (self.registration_result.inlier_rmse/self.registration_result.fitness),
                                                         self.mae, self.registered_color)
            return string
        
        except ZeroDivisionError:
            """ Occurs if registration hasn't been performed as scaled_inlier_rmse divides by zero fitness"""
            
            string = """--- Registration Object--- \nAlgorithm used = {0} \nPreprocessing performed = {1} \nRegistration performed = {2}""" .format(self.algorithm, self.preprocessing, self.result) 
        return string
    
    def __repr__(self):
        return self.__str__()

    def preprocessing_status_update(self):
        self.preprocessing = True
        
    def result_status_update(self):
        self.result = True
                 
    def registration_result_update(self, registration_result):
        self.registration_result = registration_result
        
    def update_source_color(self, source_color):
        self.source_color = source_color
    
    def update_norm_source_color(self, norm_source_color):
        self.norm_source_color = norm_source_color
    
    def update_target_color(self, target_color):
        self.target_color = target_color
    
    def update_norm_target_color(self, norm_target_color):
        self.norm_target_color = norm_target_color
        
    def update_source_pcd(self, source_pcd):
        self.source_pcd = source_pcd
        
    def update_target_pcd(self, target_pcd):
        self.target_pcd = target_pcd
    
    def update_mae(self, mae):
        self.mae = mae
    
    def update_registered_color(self, registered_color):
        self.registered_color = registered_color
    
    def update_registered_target(self, registered_target):
        self.registered_target = registered_target  
        
    def update_other_registered_channels(self, other_registered_channels):
        self.other_registered_channels = other_registered_channels 

    def create_pcd(self, print_filenames = False):
        """ Converts excel (.xls or .csv) files containing xyz coordinates into geometry.PointCloud object. 
        Returns the filenames as a list if return_filenames=True. """

        filenames = []
        
        """ Handling .xls files"""
        try: 
            source_df = pd.read_excel(self.pos_path_source, skiprows = self.pos_skiprows, usecols = self.pos_usecols, header = self.pos_header)
            target_df = pd.read_excel(self.pos_path_target, skiprows = self.pos_skiprows, usecols = self.pos_usecols, header = self.pos_header)
            source_color_df = pd.read_excel(self.color_path_source, skiprows = self.color_skiprows, usecols = self.color_usecols, header = self.color_header)
            target_color_df = pd.read_excel(self.color_path_target, skiprows = self.color_skiprows, usecols = self.color_usecols, header = self.color_header)       
            
            source_pcd = o3d.geometry.PointCloud() 
            source_np = np.asarray(source_df)
            assert (len(source_np.shape)==2 and source_np.shape[1]==3), 'Source points file has incorrect dimensions. \n Source points file must be a 2-dimensional array with "n" rows and 3 columns'
            assert (source_np.dtype=='float64'), 'Source points contain non-float values. \n This could mean column headings have been read incorrectly from file. Try altering pos_skiprows, pos_header, pos_usecols attributes.'
            
            source_color = np.asarray(source_color_df)
            assert (len(source_color.shape)==2 and source_color.shape[1]==1), 'Source colors file has incorrect dimensions. \n Source colors file must be a 2-dimensional array with "n" rows and 1 column'
            assert (source_color.dtype=='float64'), 'Source colors contain non-float values. \n This could mean column headings have been read incorrectly from file. Try altering color_skiprows, color_header, color_usecols attributes.'
            self.update_source_color(source_color)
            norm_colors_source = min_max_normalisation(source_color)
            self.update_norm_source_color(norm_colors_source)
            
            source_rgb, _ = colour_map(source_color,"viridis")
            source_pcd.points=o3d.utility.Vector3dVector(source_np)
            source_pcd.colors=o3d.utility.Vector3dVector(source_rgb)
            o3d.io.write_point_cloud(self.pos_path_source.split('.xls')[0] + '.pcd', source_pcd)
            self.update_source_pcd(source_pcd)
            filenames.append(self.pos_path_source.split('.xls')[0] + '.pcd')
            
            target_pcd = o3d.geometry.PointCloud() 
            target_np = np.asarray(target_df)
            assert (len(target_np.shape)==2 and target_np.shape[1]==3), 'Target points file has incorrect dimensions. \n Target points file must be a 2-dimensional array with "n" rows and 3 columns'
            assert (target_np.dtype=='float64'), 'Target points contain non-float values. \n This could mean column headings have been read incorrectly from file. Try altering pos_skiprows, pos_header attributes.'
            
            target_color = np.asarray(target_color_df)
            assert (len(target_color.shape)==2 and target_color.shape[1]==1), 'Target colors file has incorrect dimensions. \n Target colors file must be a 2-dimensional array with "n" rows and 1 column'
            assert (target_color.dtype=='float64'), 'Target colors contain non-float values. \n This could mean column headings have been read incorrectly from file. Try altering color_skiprows, color_header, color_usecols attributes.'
            norm_colors_target = min_max_normalisation(target_color)
            self.update_target_color(target_color)
            self.update_norm_target_color(norm_colors_target)
            
            target_rgb, _ = colour_map(target_color,"viridis")
            target_pcd.points=o3d.utility.Vector3dVector(target_np)
            target_pcd.colors=o3d.utility.Vector3dVector(target_rgb)
            o3d.io.write_point_cloud(self.pos_path_target.split('.xls')[0] + '.pcd', target_pcd)
            self.update_target_pcd(target_pcd)
            filenames.append(self.pos_path_target.split('.xls')[0] + '.pcd')
            
            return source_pcd, target_pcd

        except XLRDError:
     
            """ Handling .csv files"""
            try:
                source_df = pd.read_csv(self.pos_path_source, skiprows = self.pos_skiprows, usecols = self.pos_usecols, header = self.pos_header)
                target_df = pd.read_csv(self.pos_path_target, skiprows = self.pos_skiprows, usecols = self.pos_usecols, header = self.pos_header)
                source_color_df = pd.read_csv(self.color_path_source, skiprows = self.color_skiprows, usecols = self.color_usecols, header = self.color_header)
                target_color_df = pd.read_csv(self.color_path_target, skiprows = self.color_skiprows, usecols = self.color_usecols, header = self.color_header)

                source_pcd = o3d.geometry.PointCloud() 
                source_np = np.asarray(source_df)
                assert (len(source_np.shape)==2 and source_np.shape[1]==3), 'Source points file has incorrect dimensions. \n Source points file must be a 2-dimensional array with "n" rows and 3 columns'
                assert (source_np.dtype=='float64' or source_np.dtype == 'int64'), 'Source points contain non-float or int values. \n This could mean column headings have been read incorrectly from file. Try altering pos_skiprows, pos_header, pos_usecols attributes.'
            
                source_color = np.asarray(source_color_df)
                assert (len(source_color.shape)==2 and source_color.shape[1]==1), 'Source colors file has incorrect dimensions. \n Source colors file must be a 2-dimensional array with "n" rows and 1 column'
                assert (source_color.dtype=='float64' or source_color.dtype == 'int64'), 'Source colors contain non-float or int values. \n This could mean column headings have been read incorrectly from file. Try altering color_skiprows, color_header, color_usecols attributes.'
                
                norm_colors_source = min_max_normalisation(source_color)
                self.update_source_color(source_color)
                self.update_norm_source_color(norm_colors_source)
                
                source_rgb, _ = colour_map(source_color,"viridis")
                source_pcd.points=o3d.utility.Vector3dVector(source_np)
                source_pcd.colors=o3d.utility.Vector3dVector(source_rgb)
                o3d.io.write_point_cloud(self.pos_path_source.split('csv')[0] + '.pcd', source_pcd)
                self.update_source_pcd(source_pcd)
                filenames.append(self.pos_path_source.split('.csv')[0] + '.pcd')

                target_pcd = o3d.geometry.PointCloud() 
                target_np = np.asarray(target_df)
                assert (len(target_np.shape)==2 and target_np.shape[1]==3), 'Target points file has incorrect dimensions. \n Target points file must be a 2-dimensional array with "n" rows and 3 columns'
                assert (target_np.dtype=='float64' or target_np.dtype == "int64"), 'Target points contain non-float or int values. \n This could mean column headings have been read incorrectly from file. Try altering pos_skiprows, pos_header attributes.'
            
                target_color = np.asarray(target_color_df)
                assert (len(target_color.shape)==2 and target_color.shape[1]==1), 'Target colors file has incorrect dimensions. \n Target colors file must be a 2-dimensional array with "n" rows and 1 column'
                assert (target_color.dtype=='float64' or target_np.dtype == "int64"), 'Target colors contain non-float or int values. \n This could mean column headings have been read incorrectly from file. Try altering color_skiprows, color_header, color_usecols attributes.'
                norm_colors_target = min_max_normalisation(target_color)
                self.update_target_color(target_color)
                self.update_norm_target_color(norm_colors_target)
                
                target_rgb, _ = colour_map(target_color,"viridis")
                target_pcd.points=o3d.utility.Vector3dVector(target_np)
                target_pcd.colors=o3d.utility.Vector3dVector(target_rgb)
                o3d.io.write_point_cloud(self.pos_path_target.split('.csv')[0] + '.pcd', target_pcd)
                self.update_target_pcd(target_pcd)
                filenames.append(self.pos_path_target.split('.csv')[0] + '.pcd')
            
                return source_pcd, target_pcd

            except ValueError as v:
                
                print("Input excel files should be of extension .xls or .csv, with UTF-8 encoding. Please convert both files to either one of these formats.")
                print(v)
                
        finally:
            if print_filenames:
                print (filenames)
  
        
    def preprocessing_func(self, verbose = True):
            
        """ Down sample the point cloud, estimate normals, then compute a FPFH feature for each point. 
        Returns the processed PointCloud object and an open3d.registration.Feature class object."""

        source_pcd, target_pcd = self.create_pcd()   

        if self.downsampling:
            source_processed = source_pcd.voxel_down_sample(self.voxel_size)
            target_processed = target_pcd.voxel_down_sample(self.voxel_size)

            if verbose:
                print(f":: Downsample with a voxel size {voxel_size}")
        else:
            source_processed = source_pcd
            target_processed = target_pcd

            if verbose:
                print(":: Point Cloud was not downsampled")

        radius_normal = self.voxel_size * self.norm_radius_modifier
        radius_feature = self.voxel_size * self.fpfh_radius_modifier

        if verbose:
            print(f":: Estimate normal with search radius {radius_normal}.")
            print(f":: Compute FPFH feature with search radius {radius_feature}.\n---------------------------------------")

        source_processed.estimate_normals(
            search_param = 
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=self.norm_maxnn),
        fast_normal_computation = True)

        source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            source_processed,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=self.fpfh_maxnn))

        target_processed.estimate_normals(
            search_param = 
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=self.norm_maxnn),
        fast_normal_computation = True)

        target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            target_processed,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=self.fpfh_maxnn))
        
        self.preprocessing_status_update()
        
        return source_processed, target_processed, source_fpfh, target_fpfh
    
    def global_ransac_registration(self, verbose = True):
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

        distance_threshold = self.voxel_size * self.ransac_dist_modifier

        if verbose:
            print(":: RANSAC registration on point clouds.")
            print("   Since the  voxel size is %.3f," % self.voxel_size)
            print("   we use a liberal distance threshold %.3f." % distance_threshold)
            
        source_processed, target_processed, source_fpfh, target_fpfh = self.preprocessing_func(verbose = verbose)

        result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_processed, target_processed, source_fpfh, target_fpfh, self.ransac_mutual_filter, distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 4, [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(self.ransac_edge_length),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                    distance_threshold)
            ], o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))
        
        self.registration_result_update(result_ransac)
        self.result_status_update()
        self.calculate_mae()
        
        return result_ransac
    
    def icp_registration(self, verbose = True):  
        """ Implements the Point-to-Plane ICP registration algorithm and returns a registration.RegistrationResult object.
        """

        ransac_transform = self.global_ransac_registration(verbose = verbose)
        
        source_processed, target_processed, source_fpfh, target_fpfh = self.preprocessing_func(verbose = verbose)

        distance_threshold = self.voxel_size * self.icp_dist_check

        result_icp = o3d.pipelines.registration.registration_icp(
            source_processed, target_processed, distance_threshold, ransac_transform.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration = 2000))
                        

        self.registration_result_update(result_icp)
        self.result_status_update()
        self.calculate_mae()
        
        return result_icp


    def colored_icp(self, verbose = True):
        """ Implements the Colored ICP registration algorithm and returns a registration.RegistrationResult object.

        Source: Adapted from open3d ICP registration documentation:http://www.open3d.org/docs/release/tutorial/pipelines/colored_pointcloud_registration.html
        """
        
        ransac_transform = self.global_ransac_registration(verbose = verbose)
        
        source_processed, target_processed, source_fpfh, target_fpfh = self.preprocessing_func(verbose = verbose)

        current_transformation = ransac_transform.transformation

        source_colorreg = copy.deepcopy(source_processed)
        target_colorreg = copy.deepcopy(target_processed)

        if self.downsampling:
            source_colorreg = source_colorreg.voxel_down_sample(self.voxel_size)
            target_colorreg = target_colorreg.voxel_down_sample(self.voxel_size)

        source_colorreg.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size * 2, max_nn=self.coloredICP_maxnn))

        target_colorreg.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size * 2, max_nn=self.coloredICP_maxnn))

        result_icp_colored = o3d.pipelines.registration.registration_colored_icp(
            source_colorreg, target_colorreg, self.voxel_size, current_transformation,
            o3d.pipelines.registration.TransformationEstimationForColoredICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                    relative_rmse=1e-6,
                                                    max_iteration=100))

        current_transformation = result_icp_colored.transformation
        
        self.registration_result_update(result_icp_colored)
        self.result_status_update()
        self.calculate_mae()

        return result_icp_colored
    
    def perform_registration(self, verbose = True):
        if self.algorithm == "ransac":
            return self.global_ransac_registration(verbose = verbose)
            
        elif self.algorithm == "icp":
            return self.icp_registration(verbose = verbose)

        elif self.algorithm == "colored_icp":
            return self.colored_icp(verbose = verbose)
        
        else:
            print("Only 'ransac', 'icp' and 'colored_icp' are available algorithms to choose from.")
            return None
        
    def calculate_mae(self):
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
        corr_result = np.array(self.registration_result.correspondence_set)
        source_indices = corr_result[:,0]
        target_indices = corr_result[:,1]
        source_color_norm = self.norm_source_color[source_indices]
        target_color_norm = self.norm_target_color[target_indices]
        
        self.update_mae(mae(source_color_norm, target_color_norm)[0])
    
    def obtain_registration_metrics(self):
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
        print(f"Fitness: {self.registration_result.fitness*100:.2f}%")
        print(f"Inlier RMSE: {self.registration_result.inlier_rmse:.2f}")
        print(f"MAE: {self.mae:.2f}\n---------------------------------------")    

        corr_map = np.array(self.registration_result.correspondence_set)
        source_indices = corr_map[:,0]
        target_indices = corr_map[:,1]

        num_target = np.array(self.target_pcd.points).shape[0]
        target_range = np.arange(0, num_target)

        unmapped_targets =np.where(np.invert(np.in1d(target_range, target_indices)))[0]
        target_repeats = {i:list(target_indices).count(i) for i in target_indices if list(target_indices).count(i) > 1}
        unique_target_indices = [x for x in target_indices if x not in target_repeats]

        print("--- Correspondence map properties --- ")
        print(f"{len(unmapped_targets)} ({(len(unmapped_targets)/ num_target)*100:.3f}%) unmapped targets.")
        print(f"{len(target_repeats)} ({(len(target_repeats)/ num_target)*100:.3f}%) targets that are mapped by multiple source points.")
        print(f"{len(unique_target_indices)} ({(len(unique_target_indices)/ num_target)*100:.3f}%) targets that are uniquely mapped by a single source point.")

        if len(unmapped_targets) + len(target_repeats) + len(unique_target_indices) == len(self.target_pcd.points):
            print(f"All {len(self.target_pcd.points)} target points are accounted for.")
            
    def transform_source(self):
        transformation = self.registration_result.transformation
        source_transformed= copy.deepcopy(self.source_pcd)
        source_transformed.transform(transformation)
        return source_transformed

    def map_source2target(self, verbose = False):
        """ Returns the registered target point cloud, where the color intensity values of multiply mapped target points are imputed by
        a chosen averaging method. Currently, unmapped points in the target cloud retain their original intensities.

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
        color_list: np.array
            Updated color intensity array of the target

        """
        if self.manual_corr_map :
            corr_map = self.manual_corr_map
        else:
            corr_map = np.array(self.registration_result.correspondence_set)
            
        source_indices = corr_map[:,0]
        target_indices = corr_map[:,1]
        
        target_new = copy.deepcopy(self.target_pcd)
        
        target_repeats = {i:list(target_indices).count(i) for i in target_indices if list(target_indices).count(i) > 1}
        unique_target_indices = [x for x in target_indices if x not in target_repeats]
        
        if self.mode == "complete":
            color_list = copy.deepcopy(self.target_color)
            
        elif self.mode == "null" or self.mode == "knn":
            color_list = np.zeros(shape=(self.target_color.shape))
            
            
        ### Dealing with all other color channels
    
        if self.other_source_channels:
            other_registered_channels = []
            
            for i in range(len(self.other_source_channels)):
                print(f"{i} - Processing Other Source and Target channels")
                try:
                    source_other_color = pd.read_excel(self.other_source_channels[i], skiprows = self.color_skiprows, usecols = self.color_usecols, header = self.color_header)
                
                except XLRDError:
     
                    """ Handling .csv files"""
                    source_other_color = pd.read_csv(self.other_source_channels[i], skiprows = self.color_skiprows, usecols = self.color_usecols, header = self.color_header)
                    
                source_color_list = np.asarray(source_other_color)
                
                if self.mode == "knn":
                    source_transformed = self.transform_source()
                    X = np.asarray(source_transformed.points)
                    y = source_color_list
                    knn = neighbors.KNeighborsRegressor(self.n_neighbors, weights= self.weights)
                    target_color_list = knn.fit(X, y).predict(np.asarray(target_new.points))
                    print(target_color_list)
                    other_registered_channels.append(target_color_list)
                    continue 
                
                elif self.mode == "complete":
                    try:
                        target_other_color = pd.read_excel(self.other_target_channels[i], skiprows = self.color_skiprows, usecols = self.color_usecols, header = self.color_header)
                    
                    except: 
                        target_other_color = pd.read_csv(self.other_target_channels[i], skiprows = self.color_skiprows, usecols = self.color_usecols, header = self.color_header)
                    
                    target_color_list = np.asarray(target_other_color)
                    
                elif self.mode == "null":
                    target_color_list = np.zeros(shape=(self.target_color.shape))
                    
                    
                if self.method == "median" or self.method == "Median": 
                    print("Using median averaging")
                    for ind in target_repeats:
                        bool_mask = target_indices == ind
                        source_indices_repeat = source_indices[bool_mask]
                        target_color_list[ind] = np.median(source_color_list[source_indices_repeat])

                    for ind in unique_target_indices:
                        bool_mask = target_indices == ind
                        source_indices_unique = source_indices[bool_mask]
                        target_color_list[ind] = source_color_list[source_indices_unique]

                elif self.method == "mean" or self.method == "Mean" or self.method == "average" or self.method == "Average":
                    print("Using mean averaging")
                    
                    for ind in target_repeats:
                        bool_mask = target_indices == ind
                        source_indices_repeat = source_indices[bool_mask]
                        target_color_list[ind] = np.mean(source_color_list[source_indices_repeat])

                    for ind in unique_target_indices:
                        bool_mask = target_indices == ind
                        source_indices_unique = source_indices[bool_mask]
                        target_color_list[ind] = source_color_list[source_indices_unique]
                        
                other_registered_channels.append(target_color_list)
        
            self.update_other_registered_channels(other_registered_channels)
            
        ### Dealing with the color channel used for registration 
                        
        if self.mode == "knn":
            print("Entering knn for sox2")
            source_transformed = self.transform_source()
            X = np.asarray(source_transformed.points)
            y = self.source_color
            knn = neighbors.KNeighborsRegressor(self.n_neighbors, weights= self.weights)
            color_list = knn.fit(X, y).predict(np.asarray(target_new.points))
                    
        else:
            if self.method == "median" or self.method == "Median": 
                print("Using median averaging")
                for ind in target_repeats:
                    bool_mask = target_indices == ind
                    source_indices_repeat = source_indices[bool_mask]
                    color_list[ind] = np.median(self.source_color[source_indices_repeat])

                for ind in unique_target_indices:
                    bool_mask = target_indices == ind
                    source_indices_unique = source_indices[bool_mask]
                    color_list[ind] = self.source_color[source_indices_unique]

            elif self.method == "mean" or self.method == "Mean" or self.method == "average" or self.method == "Average":
                print("Using mean averaging")
                for ind in target_repeats:
                    bool_mask = target_indices == ind
                    source_indices_repeat = source_indices[bool_mask]
                    color_list[ind] = np.mean(self.source_color[source_indices_repeat])

                for ind in unique_target_indices:
                    bool_mask = target_indices == ind
                    source_indices_unique = source_indices[bool_mask]
                    color_list[ind] = self.source_color[source_indices_unique]
            else:
                raise Exception("Unrecognised method used. Only mean/average or median functions are permitted.") 

        if verbose:
            print("before assignment", np.array(target_new.colors))

        mapped_rgb, mapped_col_range=colour_map(color_list,"viridis")

        target_new.colors =o3d.utility.Vector3dVector(mapped_rgb)
        
        self.update_registered_color(color_list)
        self.update_registered_target(target_new)

        if verbose:
            print("after assignment", np.array(target_new.colors))
            print(np.all([mapped_rgb, np.array(target_new.colors)]))

        return (target_new,mapped_col_range, color_list)
    
class RegistrationObj_noisy(RegistrationObj):
    
    def __init__(self, pos_path_source, pos_path_target, color_path_source, color_path_target, 
                 algorithm = "colored_icp",**kwargs):
        super().__init__(pos_path_source, pos_path_target, color_path_source, color_path_target, **kwargs)
        self.noisy_registrationObj = []
        self.noisy_pcd_list = []
        self.noisy_pcd_temp = o3d.geometry.PointCloud()
        
    def update_noisy_pcd_list(self, noisy_pcd_list):
        self.noisy_pcd_list = noisy_pcd_list
        
    def update_noisy_registrationObj(self, noisy_registrationObj):
        self.noisy_registrationObj = noisy_registrationObj
        
    def global_ransac_registration(self, verbose = True):

        distance_threshold = self.voxel_size * self.ransac_dist_modifier

        if verbose:
            print(":: RANSAC registration on point clouds.")
            print("   Since the  voxel size is %.3f," % self.voxel_size)
            print("   we use a liberal distance threshold %.3f." % distance_threshold)
            
        source_processed, target_processed, source_fpfh, target_fpfh = self.preprocessing_func(verbose = verbose)

        result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            self.noisy_pcd_temp, target_processed, source_fpfh, target_fpfh, self.ransac_mutual_filter, distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 4, [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(self.ransac_edge_length),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                    distance_threshold)
            ], o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))

        self.registration_result_update(result_ransac)
        print(self.registration_result)
        self.result_status_update()

        return result_ransac

    def icp_registration(self, verbose = True):  
        """ Implements the Point-to-Plane ICP registration algorithm and returns a registration.RegistrationResult object.
        """

        ransac_transform = self.global_ransac_registration(verbose = verbose)
        
        source_processed, target_processed, source_fpfh, target_fpfh = self.preprocessing_func(verbose = verbose)

        distance_threshold = self.voxel_size * self.icp_dist_check

        result_icp = o3d.pipelines.registration.registration_icp(
            self.noisy_pcd_temp, target_processed, distance_threshold, ransac_transform.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration = 2000))
                        

        self.registration_result_update(result_icp)
        self.result_status_update()
        self.calculate_mae()
        
        return result_icp


    def colored_icp(self, verbose = True):
        """ Implements the Colored ICP registration algorithm and returns a registration.RegistrationResult object.

        Source: Adapted from open3d ICP registration documentation:http://www.open3d.org/docs/release/tutorial/pipelines/colored_pointcloud_registration.html
        """
        print("Entering into colored_ICP")
        ransac_transform = self.global_ransac_registration(verbose = verbose)
        
        source_processed, target_processed, source_fpfh, target_fpfh = self.preprocessing_func(verbose = verbose)

        current_transformation = ransac_transform.transformation

        source_colorreg = copy.deepcopy(self.noisy_pcd_temp)
        target_colorreg = copy.deepcopy(target_processed)

        if self.downsampling:
            source_colorreg = source_colorreg.voxel_down_sample(self.voxel_size)
            target_colorreg = target_colorreg.voxel_down_sample(self.voxel_size)

        source_colorreg.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size * 2, max_nn=self.coloredICP_maxnn))

        target_colorreg.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size * 2, max_nn=self.coloredICP_maxnn))

        result_icp_colored = o3d.pipelines.registration.registration_colored_icp(
            source_colorreg, target_colorreg, self.voxel_size, current_transformation,
            o3d.pipelines.registration.TransformationEstimationForColoredICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                    relative_rmse=1e-6,
                                                    max_iteration=100))

        current_transformation = result_icp_colored.transformation
        
        self.registration_result_update(result_icp_colored)
        self.result_status_update()
        self.calculate_mae()

        return result_icp_colored
        
    def simulate_noise(self, sd_range = 50, sd_interval = 3, sim_num = 2, ax = None, legend = True,
                  title = None, x_label = "Noise (standard deviation)", y_label = "Metric results",
                  results_only = False, verbose = False):
        
        self.create_pcd()

        """ Noise simulation"""
        noise_sd = np.linspace(0,sd_range,sd_interval)
        len_noise = len(noise_sd)
        size_data = np.asarray(self.source_pcd.points).shape

        noisy_pcd_list = []
        results_list = []
        noisy_registrationObj = []

        for j in range(sim_num):
            np.random.seed(j)
            print(f"----------Simulation num: {j+1}----------")
            for i in range(len_noise):
                source_pcd = copy.deepcopy(self.source_pcd)
                noise = np.random.normal(0,noise_sd[i],size_data)
                combined_noise = np.asarray(source_pcd.points) + noise
                self.noisy_pcd_temp.points=o3d.utility.Vector3dVector(np.asarray(combined_noise))
                
                noisy_pcd_temp = copy.deepcopy(self.noisy_pcd_temp)
                noisy_pcd_list.append(noisy_pcd_temp)
                myRegObj = self.perform_registration(verbose = verbose)
                MyRegObjCopy =copy.deepcopy(myRegObj)
                noisy_registrationObj.append(MyRegObjCopy)

                results_list.append([self.registration_result.fitness, self.registration_result.inlier_rmse, 
                                     (self.registration_result.inlier_rmse)/(self.registration_result.fitness), 
                                     self.mae, (self.mae)/(self.registration_result.fitness), np.asarray(self.registration_result.correspondence_set).shape[0]])

        self.update_noisy_pcd_list(noisy_pcd_list)
        self.update_noisy_registrationObj(noisy_registrationObj)
        result_array = np.array(results_list)
        result = np.reshape(result_array, (sim_num,sd_interval,6))
        result_mean = np.mean(result, axis = 0)

        fitness = [mylist[0] for mylist in result_mean]
        inlier_rmse = [mylist[1] for mylist in result_mean]
        scaled_inlier_rmse = [mylist[2] for mylist in result_mean]
        mae = [mylist[3] for mylist in result_mean]
        scaled_inlier_mae = [mylist[4] for mylist in result_mean]
        corr_num = [mylist[5] for mylist in result_mean]


        if results_only:
            return np.asarray([fitness, inlier_rmse, scaled_inlier_rmse, mae, scaled_inlier_mae, corr_num])

        else:
            """ Plotting"""

            if ax is None:
                ax = plt.gca()
            ax.plot(noise_sd, fitness, label='Fitness')
            ax.plot(noise_sd, inlier_rmse, label='Inlier RMSE')
            ax.plot(noise_sd, mae, label='MAE')

            if legend:
                ax.legend(loc = "upper left")

            if title:
                ax.set_title(label = title)

            if x_label:
                ax.set_xlabel(x_label)

            if y_label:
                ax.set_ylabel(y_label)

            return ax

class IterativePairwise():
    
    def __init__(self, pos_path_source_list, pos_path_target, color_path_source_list, color_path_target, 
                 algorithm = "colored_icp", **kwargs):
        
        self.pos_path_source_list= pos_path_source_list
        self.pos_path_target = pos_path_target
        self.source_pcd = []
        self.target_pcd = None
        self.color_path_source_list = color_path_source_list
        self.color_path_target = color_path_target
        self.algorithm = algorithm
        self.source_length = len(self.pos_path_source_list)
        self.combined_results = []
        self.registered_color = None
        self.registration_obj = []
        self.other_registered_channels = []
        self.mode = "knn"
        self.method = "Median"
        self.n_neighbors = kwargs.get('n_neighbors', 5)
        self.weights = kwargs.get('weights', "distance")
                
        """Setting the arguments for RegistrationObj.create_pcd"""
        self.pos_skiprows = None
        self.pos_usecols = None
        self.pos_header = 0
        self.color_skiprows = None
        self.color_usecols = None
        self.color_header = 0
        
        self.__dict__.update(kwargs)
        
        self.other_source_channels = kwargs.get('other_source_channels', [])
        self.other_target_channels = kwargs.get('other_target_channels', [])
        
    def __str__(self):
        string = """--- Iterative Pairwise Object--- \nAlgorithm used = {0}\nSource file names : {1}\nNumber of source files : {2}\nTarget file name: {3}""". format(self.algorithm, self.pos_path_source_list, self.source_length, self.pos_path_target) 
        return string
    
    def __repr__(self):
        return self.__str__()
    
    def __getitem__(self, i):
        return self.combined_results[i]
    
    def update_combined_results(self, combined_results):
        self.combined_results = combined_results
        
    def update_registration_obj(self, registration_obj):
        self.registration_obj = registration_obj
        
    def update_registered_color(self, registered_color):
        self.registered_color = registered_color
        
    def update_other_registered_channels(self, other_registered_channels):
        self.other_registered_channels = other_registered_channels
        
    def update_source_pcd(self, source_pcd):
        self.source_pcd = source_pcd
    
    def update_target_pcd(self, target_pcd):
        self.target_pcd = target_pcd
    
    def iterative_registration(self, jupyter_visualise = True, verbose = True):
        assert (self.source_length == len(self.color_path_source_list)), "Length of source positions list is different from the source color intensities list."
        assert (type(self.pos_path_target) == str), "Target position path should be a string." 
        
        result_color_list = []
        result_list = []
        source_pcd = []
        registration_obj = []
        other_registration_channels = []
        
        for i in range(self.source_length):
            print(f"--- Registering Source dataset {i}")
            if not len(self.other_source_channels):
                print(f"--No other source channels detected")
                myObj = RegistrationObj(self.pos_path_source_list[i], self.pos_path_target, 
                                        self.color_path_source_list[i], self.color_path_target, algorithm = "colored_icp",
                            pos_skiprows = self.pos_skiprows, pos_usecols = self.pos_usecols, 
                                        color_skiprows = self.color_skiprows, color_usecols = self.color_usecols,
                                       mode = self.mode, method = self.method, n_neighbors = self.n_neighbors, 
                                       weights = self.weights)
                myObj.perform_registration(verbose = verbose)
                target_new, _, color_list = myObj.map_source2target()
                result_color_list.append(color_list)
                other_registration_channels.append(myObj.other_registered_channels)
                result_list.append([myObj.registration_result.fitness, myObj.registration_result.inlier_rmse,
                                   (myObj.registration_result.inlier_rmse/myObj.registration_result.fitness), myObj.mae])
                source_pcd.append(myObj.source_pcd)
                registration_obj.append(myObj.registration_result)
            else:
                myObj = RegistrationObj(self.pos_path_source_list[i], self.pos_path_target, 
                                        self.color_path_source_list[i], self.color_path_target, algorithm = "colored_icp",
                            pos_skiprows = self.pos_skiprows, pos_usecols = self.pos_usecols, 
                                        color_skiprows = self.color_skiprows, color_usecols = self.color_usecols,
                                       other_source_channels = self.other_source_channels[i],
                                       other_target_channels = self.other_target_channels,
                                       mode = self.mode, method = self.method, n_neighbors = self.n_neighbors, 
                                       weights = self.weights)
                myObj.perform_registration(verbose = verbose)
                target_new, _, color_list = myObj.map_source2target()
                result_color_list.append(color_list)
                other_registration_channels.append(myObj.other_registered_channels)
                result_list.append([myObj.registration_result.fitness, myObj.registration_result.inlier_rmse,
                                   (myObj.registration_result.inlier_rmse/myObj.registration_result.fitness), myObj.mae])
                source_pcd.append(myObj.source_pcd)
                registration_obj.append(myObj.registration_result)
                
        
        self.update_combined_results(result_list)
        self.update_source_pcd(source_pcd)
        self.update_target_pcd(myObj.target_pcd)
        self.update_registration_obj(registration_obj)
        
        result_color_list = np.asarray(result_color_list)
        color_median = np.median(result_color_list, axis = 0)
        self.update_registered_color(color_median)
        
        other_registration_channels = np.asarray(other_registration_channels)
        other_color_median = np.median(other_registration_channels, axis = 0)
        self.update_other_registered_channels(other_color_median)
        
        image_rgb_final, _ = colour_map(color_median,"viridis")
        target_final =  copy.deepcopy(myObj.target_pcd)
        target_final.colors=o3d.utility.Vector3dVector(image_rgb_final)
        
        if jupyter_visualise:
            visualizer = JVisualizer()
            visualizer.add_geometry(target_final)
            visualizer.show()
        
        return target_final
            
    
