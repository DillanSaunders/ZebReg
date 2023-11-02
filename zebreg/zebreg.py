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


class Registration_Obj():
    
    def __init__(self, points, intensities, target_index, intensity_index=None, **kwargs):
        
        self.points = points
        self.intensities = intensities
        self.target_index = target_index
        self.intensity_index = intensity_index
        
        self.source_pcd = None
        self.source_fpfh = None
        self.target_pcd = None
        self.target_fpfh = None
        self.result = False #change this parameter so that it says which algorithm used
        self.preprocessing = False
        self.registration_result = [o3d.pipelines.registration.RegistrationResult()]*len(points)
        self.registration_type = None
        
        self.reg_points = []
        
        self.__dict__.update(kwargs)

        """ Setting the values of registration parameters"""
        self.voxel_size =  kwargs.get('voxel_size', 25)
        self.downsampling =  kwargs.get('downsampling', False)
        self.norm_radius_modifier = kwargs.get('norm_radius_modifier', 2)
        self.norm_maxnn = kwargs.get('norm_maxnn', 30)
        self.fpfh_radius_modifier = kwargs.get('fpfh_radius_modifier', 5)
        self.fpfh_maxnn = kwargs.get('fpfh_maxnn', 100)
        self.ransac_dist_modifier = kwargs.get('ransac_dist_modifier', 1.5)
        self.ransac_edge_length = kwargs.get('ransac_edge_length', 0.9)
        self.ransac_mutual_filter = kwargs.get('ransac_mutual_filter', True) # not present in version o3d 0.11.1
        self.ransac_n_correspondences = kwargs.get('ransac_n_correspondences',4)
        self.convergence_max_iterations = kwargs.get('convergence_max_iterations',400000)
        self.convergence_max_validation = kwargs.get('convergence_max_validation', 500) # from o3d version 0.12.0 this parameter is a confidence interval float between 0-1
        self.icp_dist_check = kwargs.get('icp_dist_check', 1)
        self.coloredICP_maxnn = kwargs.get('coloredICP_maxnn', 50)
        self.lambda_geometric=kwargs.get('lamba_geometric', 0.96800)
        self.n_neighbors = kwargs.get('n_neighbors', 5)
        self.weights = kwargs.get('weights', "distance")
        
    def result_status_update(self):
        self.result = True
        
    def create_pcd(self, verbose = True):
        """ Converts excel (.xls or .csv) files containing xyz coordinates into geometry.PointCloud object. 
        Returns the filenames as a list if return_filenames=True. """
        #assert points must be a list
        target_points = self.points[self.target_index]
        target_pcd = o3d.geometry.PointCloud()
        target_pcd.points=o3d.utility.Vector3dVector(target_points)
        
        target_color = self.intensities[self.target_index][:,self.intensity_index]
        target_color = target_color.reshape(len(target_color),1)
        target_rgb, _ = color_map(target_color,'viridis')
        target_pcd.colors=o3d.utility.Vector3dVector(target_rgb)
        
        source_points = self.points
        source_pcd = []
        for source_index in range(len(source_points)):
            single_source_pcd = o3d.geometry.PointCloud()
            single_source_pcd.points = o3d.utility.Vector3dVector(source_points[source_index])
            source_color = self.intensities[source_index][:,self.intensity_index]
            source_color = source_color.reshape(len(source_color),1)
            source_rgb, _ = color_map(source_color,'viridis')
            single_source_pcd.colors = o3d.utility.Vector3dVector(source_rgb)
            source_pcd.append(single_source_pcd)
        
        if verbose:
            print(f":: Target point cloud will be taken from position {self.target_index} in the data list")
            print(f":: Point clouds will be colored using position {self.intensity_index} in the intensities list")
        
        radius_normal = self.voxel_size * self.norm_radius_modifier
        radius_feature = self.voxel_size * self.fpfh_radius_modifier

        if verbose:
            print(f":: Estimate normal with search radius {radius_normal}.")
            print(f":: Compute FPFH feature with search radius {radius_feature}.\n---------------------------------------")
            
        # Preprocess target point cloud    
        if self.downsampling:
            target_pcd = target_pcd.voxel_down_sample(self.voxel_size)
            if verbose:
                print(f":: Target Point Cloud downsampled with a voxel size {self.voxel_size}")
        else:
            if verbose:
                print(":: Target Point Cloud was not downsampled")

        target_pcd.estimate_normals(
            search_param = 
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=self.norm_maxnn),
        fast_normal_computation = True)

        target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(target_pcd,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=self.fpfh_maxnn))
        
        # Preprocess source point cloud(s)
        
        source_fpfh = []
        for source_index in range(len(source_pcd)):
            if self.downsampling:
                source_pcd[source_index] = source_pcd[source_index].voxel_down_sample(self.voxel_size)
                if verbose:
                    print(f":: Source Point Cloud(s) downsampled with a voxel size {self.voxel_size}")
            else:
                if verbose:
                    print(":: Source Point Cloud(s) was not downsampled")

            source_pcd[source_index].estimate_normals(
                search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=self.norm_maxnn),
                fast_normal_computation = True)

            source_fpfh.append(o3d.pipelines.registration.compute_fpfh_feature(
                source_pcd[source_index],
                o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=self.fpfh_maxnn)))

        
        if verbose:
            print(f":: Normals estimated")
            print(f":: FPFH computed")
            print(f":: Preprocessing completed")
        
        self.source_pcd = source_pcd
        self.target_pcd = target_pcd
        self.source_fpfh = source_fpfh 
        self.target_fpfh = target_fpfh
        self.preprocessing = True
       
        return 

                  
    def ransac_registration(self, verbose = True):
        """ Implements Global RANSAC registration based on feature matching and returns a registration.RegistrationResult object.

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
        
        self.create_pcd(verbose = verbose)

        distance_threshold = self.voxel_size * self.ransac_dist_modifier

        if verbose:
            print(":: RANSAC registration on point clouds.")
            print("   Since the  voxel size is %.3f," % self.voxel_size)
            print("   we use a liberal distance threshold %.3f." % distance_threshold)
                    
        for source_index in range(len(self.source_pcd)):

            result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                self.source_pcd[source_index], self.target_pcd, self.source_fpfh[source_index], self.target_fpfh, distance_threshold,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(False), self.ransac_n_correspondences, [
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(self.ransac_edge_length),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                        distance_threshold)
            ], o3d.pipelines.registration.RANSACConvergenceCriteria(self.convergence_max_iterations,self.convergence_max_validation))
            
            self.registration_result[source_index] = result_ransac
        
        
        self.result_status_update()
        #how to include mae in registration result - create zebreg specific reg result?
        #self.calculate_mae()
        self.registration_type = 'RANSAC'
        print(f":: Completed Global Ransac Registration \n {self.registration_result}")
        
        return
    
    def icp_registration(self, verbose = True):  
        """ Implements the Point-to-Plane ICP registration algorithm and returns a registration.RegistrationResult object.
        """

        #ransac_transform = self.ransac_registration(verbose = verbose)
        
        #source_processed, target_processed, source_fpfh, target_fpfh = self.preprocessing_func(verbose = verbose)
        
        self.ransac_registration(verbose=verbose)

        distance_threshold = self.voxel_size * self.icp_dist_check
        
        for source_index in range(len(self.source_pcd)):
            result_icp = o3d.pipelines.registration.registration_icp(
                self.source_pcd[source_index], self.target_pcd, distance_threshold,
                self.registration_result[source_index].transformation,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration = 2000))
            
            self.registration_result[source_index] = result_icp
                  

        #self.registration_result_update(result_icp)
        self.result_status_update()
        #self.calculate_mae()
        self.registration_type = 'ICP'
        print(f":: Completed Local ICP Registration \n {self.registration_result}")
        
        return


    def colored_icp_registration(self, verbose = True):
        """ Implements the Colored ICP registration algorithm and returns a registration.RegistrationResult object.

        Source: Adapted from open3d ICP registration documentation:http://www.open3d.org/docs/release/tutorial/pipelines/colored_pointcloud_registration.html
        """
        self.ransac_registration(verbose=verbose)
        distance_threshold = self.voxel_size * self.icp_dist_check
        
        for source_index in range(len(self.source_pcd)):
            result_colicp = o3d.pipelines.registration.registration_colored_icp(
                self.source_pcd[source_index], self.target_pcd, self.voxel_size,
                self.registration_result[source_index].transformation,
                o3d.pipelines.registration.TransformationEstimationForColoredICP(self.lambda_geometric),
                o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                                  relative_rmse=1e-6,
                                                                  max_iteration=100))
            self.registration_result[source_index] = result_colicp
                  

        self.result_status_update()
        #self.calculate_mae()
        self.registration_type = 'Colored ICP'
        print(f":: Completed Local Colored ICP Registration \n {self.registration_result}")

        return 
                  
    def register_pcd(self, registration_algorithm, verbose = True):
        
        if registration_algorithm == 'ransac':
            self.ransac_registration(verbose = verbose)
            
        elif registration_algorithm == 'icp':
            self.icp_registration(verbose = verbose)
            
        elif registration_algorithm == 'colored_icp':
            self.colored_icp_registration(verbose = verbose)
        
        else:
            print('Must specifiy a valid registration algorithm from "ransac", "icp", "colored_icp"')
                  
    def transform_pcd(self):
        
        for source_index in range(len(self.source_pcd)):
            source_transformed = self.source_pcd[source_index].transform(self.registration_result[source_index].transformation)
            self.reg_points.append(np.asarray(source_transformed.points))
        
        return
    
    def average_intensities(self, verbose = False):
        
        av_pcd = self.target_pcd
        
        #for source_index in range(len(self.source_pcd)):
            
        return
    
    def registration_statistics(self, corr_set = False):
        
        if self.registration_type == 'RANSAC' or self.registration_type == 'ICP':
            
            statistics = np.empty((len(self.registration_result),2))
                
            for i in range(len(self.registration_result)):
                statistics[i] = [self.registration_result[i].fitness,self.registration_result[i].inlier_rmse]
                    
        elif self.registration_type == 'Colored ICP':
            
            statistics = np.empty((len(self.registration_result),3))
            
            target_color_norm = min_max_normalisation(self.intensities[self.target_index][:,0])
            for i in range(len(self.registration_result)):
                corr_result = np.array(self.registration_result[i].correspondence_set)
                source_indices = corr_result[:,0]
                target_indices = corr_result[:,1]
                source_color_norm = min_max_normalisation(self.intensities[i])
                source_corr_int = source_color_norm[source_indicies]
                target_corr_int = target_color_norm[target_indicies]
                
                mae_value = mae(source_corr_int, target_corr_int)[0]
                
                statistics[i] = [self.registration_result[i].fitness,self.registration_resul[i].inlier_rmse, mae_value]
                
        if corr_set == False:
            
            return statistics
        
        elif corr_set == True:
            
            corr_list = []
            
            for i in range(len(self.registration_result)):
                
                corr_list.append(np.array(self.registration_result[i].correspondence_set))
            
            return statistics, corr_list
        
    def map_pcd(self, method,unmapped='knn',knn_radius=5, verbose = False):
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
        
        corr_map = [np.asarray(reg_ems.correspondence_set,dtype=np.uint32) for reg_ems in self.registration_result]

        dataframes = [pd.DataFrame(self.intensities[reg_ems],index = corr_map[reg_ems][:,1]) for reg_ems in range(len(corr_map))]

        combined_dataframes = pd.concat(dataframes, axis=0)
        
        if unmapped =='knn':
            unmapped_points = []
            for i in range(self.intensities[self.target_index].shape[0]):
                if len(combined_dataframes.loc[i].shape) == 1:
                    unmapped_points.append(i)

            mapped_pcd = o3d.geometry.PointCloud()
            mapped_pcd.points = o3d.utility.Vector3dVector(np.concatenate(self.reg_points,axis=0))
            mapped_tree = o3d.geometry.KDTreeFlann(mapped_pcd)

            nn_idx = []
            for ump in unmapped_points:
                k,idx,_ = mapped_tree.search_radius_vector_3d(self.reg_points[self.target_index][ump], knn_radius) 
                nn_idx.append(np.array(idx))

            all_ints = np.concatenate(self.intensities,axis=0)
            av_nn = []
            for i in range(len(nn_idx)):
                if method == 'mean':
                    av_nn.append(np.mean(all_ints[nn_idx[i]], axis=0))
                if method == 'median':
                    av_nn.append(np.median(all_ints[nn_idx[i]], axis=0))

            combined_dataframes.loc[unmapped_points] = av_nn


        merged_intensities = np.empty(self.intensities[self.target_index].shape)
        for i in range(len(merged_intensities)):
            if len(combined_dataframes.loc[i].shape) > 1:
                if method == 'mean':
                    merged_intensities[i] = np.nanmean(combined_dataframes.loc[i],axis=0)
                elif method == 'median':
                    merged_intensities[i] = np.nanmedian(combined_dataframes.loc[i],axis=0)
            else:
                merged_intensities[i] = np.array(combined_dataframes.loc[i])

        return self.points[self.target_index],pd.DataFrame(merged_intensities)

    
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

