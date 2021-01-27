import open3d as o3d
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as col

def open_points_xls(files_path, cols=[0,1,2],row_skip=0, header=0):
    """Opens excel files containing columns of XYZ coordinates of nuclei centers as Pandas DataFrames. These are stored in a list. 
    Parameters
    ----------
    files_path: string
        Location of the files to be loaded into python. Use '*' as the wildcard. The excel files should all be in the same format. 
    cols: list
        Number of columns of excel sheet to load into python. 
    row_skip: int or None
        Does not load these rows of the head of the excel sheet into python.
    header: int or None
        The column containing the header for the columns of the excel sheet.
        
    Returns
    ---------
    files_dataframe:
        List of Pandas DataFrames the same length as the number of excel files. 
    """
    
    files_list = list(np.sort(glob.glob(pts_path)))
    files_dataframe = [pd.read_excel(file, skiprows = [row_skip], header = header, usecols = cols) for file in files_list]
    
    return files_dataframe, files_list

def open_intensities_xls(files_path, cols=[0],row_skip=0, header=0):
    """Opens excel files containing a single column of mean gene expression intensity at a given point as Pandas DataFrames. These are stored in a list. 
    Parameters
    ----------
    files_path: string
        Location of the files to be loaded into python. Use '*' as the wildcard. The excel files should all be in the same format. 
    cols: list
        Number of columns of excel sheet to load into python. 
    row_skip: int or None
        Does not load these rows of the head of the excel sheet into python.
    header: int or None
        The column containing the header for the columns of the excel sheet.
        
    Returns
    ---------
    files_dataframe:
        List of Pandas DataFrames the same length as the number of excel files. 
    """
    
    files_list = list(np.sort(glob.glob(pts_path)))
    files_dataframe = [pd.read_excel(file, skiprows = [row_skip], header = header, usecols = cols) for file in files_list]
    
    return files_dataframe, files_list

def excel_to_pcd(excel_list, names_list, return_filenames = False):
    """ Converts excel files containing xyz coordinates into geometry.PointCloud object. Returns the filenames as a list if return_filenames=True"""
    
    filenames = []
    for index in range(len(excel_list)):
        excel_np = excel_list[index].to_numpy(dtype= "float64")
        excel_pcd=o3d.geometry.PointCloud() #Creating the PointCloud object constructor 
        excel_pcd.points=o3d.utility.Vector3dVector(excel_np)
        o3d.io.write_point_cloud(names_list[index].split('.xls')[0] + '.pcd', excel_pcd)
        filenames.append(names_list[index].split('.xls')[0] + '.pcd')
        
    if return_filenames:
        return filenames

def preprocess_point_cloud(pcd, voxel_size =10, downsampling= False, norm_radius_modifier=2,norm_maxnn=30,fpfh_radius_modifier=5,fpfh_maxnn=100):
    
    """ Down sample the point cloud, estimate normals, then compute a FPFH feature for each point. 
    Returns the processed PointCloud object and an open3d.registration.Feature class object."""
    
    if downsampling:
        print(f":: Downsample with a voxel size {voxel_size}")
        pcd_processed = pcd.voxel_down_sample(voxel_size)
    else:
        print(":: Point Cloud was not downsampled")
        pcd_processed=pcd

    radius_normal = voxel_size * norm_radius_modifier
    print(f":: Estimate normal with search radius {radius_normal}.")
    pcd_processed.estimate_normals(
        search_param = 
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=norm_maxnn),
    fast_normal_computation = True)

    radius_feature = voxel_size * fpfh_radius_modifier
    print(f":: Compute FPFH feature with search radius {radius_feature}.\n---------------------------------------")
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_processed,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=fpfh_maxnn))
    
    return pcd_processed, pcd_fpfh

def prepare_dataset(excel_list, names_list, voxel_size = 10, downsampling=False,norm_radius_modifier=2,norm_maxnn=30,fpfh_radius_modifier=5,fpfh_maxnn=100):
    
    """Returns 6 objects corresponding to the original and processed source and target PointClouds, and the source and target fpfh Feature objects.
    
    Parameters
    ----------
    excel_list : list
        List of length 2 containing pandas.DataFrames of the source and target files
    names_list: list
        List of the corresponding file names of the pandas.DataFrames source and target files
    voxel_size: float, optional
        Used to determine the voxel size to downsample to (if downsampling = True), for calculating the KDTree search radius to estimate normals and for calculating the KDTree search radius to compute the FPFH features. 
    downsampling: bool, optional
        Determines whether downsampling should be performed or not.
    norm_radius_modifier: float, optional
        Multiplied with voxel_size to compute the KDTree search radius for estimate_normals
    norm_maxnn: int, optional
        Max number of neighbors in the KDTree search in estimate_normals
    fpfh_radius_modifier: float, optional
        Multiplied with voxel_size to compute the KDTree search radius for compute_fpfh_feature
    fpfh_maxnn: int, optional
        Max number of neighbors in the KDTree search in compute_fpfh_feature
    
    Returns
    ----------
    source : geometry.PointCloud
        Source point cloud before processing
    target : geometry.PointCloud
        Target point cloud before processing
    source_processed: geometry.PointCloud
        Source point cloud after downsampling (if downsample=True) and normal estimation.
    target_processed: geometry.PointCloud
        Target point cloud after downsampling (if downsample=True) and normal estimation.
    source_fpfh: registration.Feature
        Source point cloud fpfh information
    target_fpfh: registration.Feature
        Target point cloud fpfh information
    """
    
    pcd_list = excel_to_pcd(excel_list, names_list, return_filenames = True)
    source = o3d.io.read_point_cloud(pcd_list[0]) #sample.pcd is the source
    target = o3d.io.read_point_cloud(pcd_list[1]) #reference.pcd is the target
    source_processed, source_fpfh = preprocess_point_cloud(source, voxel_size, downsampling,norm_radius_modifier,norm_maxnn,fpfh_radius_modifier,fpfh_maxnn)
    target_processed, target_fpfh = preprocess_point_cloud(target, voxel_size, downsampling,norm_radius_modifier,norm_maxnn,fpfh_radius_modifier,fpfh_maxnn)
    return source, target, source_processed, target_processed, source_fpfh, target_fpfh

def process_images(positions_paths,color_paths, skiprows, cmap = "viridis"):
    """ Returns separate lists of point cloud objects, fpfh registration feature objects and the RGB array
    
    Parameters:
    ----------
    positions_paths: list
        Contains the filepaths to the source and target position coordinates (.xls format).
    color_path: list
        Contains the filepaths to the source and target color intensity values (.xls format)
    cmap: string
        Name of the colour map
        
    Returns:
    ----------
    pcd_list: list
        List containing the source and target pcd objects (with color channel appended)
    fpfh_list: list
        List containing the fpfh registration feature objects
    image_rgb: np.array
            
    """
    pcd_list = []
    fpfh_list = []
    image_rgb_list = []
    
    for position,color in zip(positions_paths, color_paths):
        image_excel = [pd.read_excel(position, skiprows = skiprows, header = 0, usecols = [0,1,2])]
        image_pcd = excel_to_pcd(image_excel, [position], return_filenames = True)
        image = o3d.io.read_point_cloud(image_pcd[0]) 
        image_processed, image_fpfh = preprocess_point_cloud(image)
        
        color_excel = [pd.read_excel(color, skiprows = skiprows, header = 0, usecols = [0])]
        image_color = color_excel[0].to_numpy(dtype='float64')
        image_rgb, image_col_range = colour_map(image_color,cmap)

        image_processed.colors=o3d.utility.Vector3dVector(image_rgb)
        pcd_list.append(image_processed)
        fpfh_list.append(image_fpfh)
        image_rgb_list.append(image_rgb)
        
    return pcd_list, fpfh_list, image_rgb_list  


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
        To be used to create a Matplotlib scale bar

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

def assign_colors(source_pcd, target_pcd, color_list, cmap):
    """ Assigns colors to source and target point clouds.
    
    Parameters:
    ----------
    source_pcd: geometry.PointCloud
        Source point cloud 
    target_pcd: geometry.PointCloud
        Target point cloud
    color_list: list
         List of length 2 containing unnormalised intensity values of the imaging color channel
    cmap: string
        Name of the colour map
    """
    
    source_color=color_list[0].to_numpy(dtype='float64')
    target_color=color_list[1].to_numpy(dtype='float64')
    source_rgb, source_col_range=colour_map(source_color,cmap)
    target_rgb, target_col_range=colour_map(target_color,cmap)
    
    if target_rgb.shape[0] == np.asarray(target_pcd.points).shape[0] and source_rgb.shape[0] == np.asarray(source_pcd.points).shape[0]:
        target_pcd.colors=o3d.utility.Vector3dVector(target_rgb)
        source_pcd.colors=o3d.utility.Vector3dVector(source_rgb)
        
    else:
        print("Warning: Dimensions of the source or target point cloud does not match the dimensions of the color channel. Attempting to append colors.")
        target_pcd.colors=o3d.utility.Vector3dVector(target_rgb)
        source_pcd.colors=o3d.utility.Vector3dVector(source_rgb)
        
        
    print(":: Assigned colors to point clouds")
    
    return None

def colour_region(total_points, region_points, cmap):
    
    """" Adds RGB colour to a point cloud, and creates the corresponding intensity array, using a subset of the point cloud to highlight that region
    
    Parameters:
    -----------
    total_points: geometry.PointCloud
        Complete tailbud point cloud to be coloured
    region_points: geometry.PointCloud
        Subset of total_points which denotes a specifc region e.g. ablation, in the structure
    cmap: string
        Name of Matplotlib color map used to visualise the RGB colour
        
    Returns:
    --------
    rgb_array: np.array
        n x 3 array where n is the number of points in the point cloud
    lut: matplotlib.cm.ScalarMappable object 
       To be used to create a Matplotlib scale bar
    region_intensity: Numpy Array
        Numpy array of length total_points with intensity values corresponding to each xyz point
    """
    
    
    total_points=total_points.to_numpy(dtype="float64")
    region_points=region_points.to_numpy(dtype="float64")

    region_intensity= np.zeros([len(total_points),1])

    for index in range(len(region_points)):

        corresponding_points=np.where(total_points[:,0]==region_points[index,0])
        if len(list(corresponding_points)[0]) > 1:
            print('there is more than 1 corresponding point')
            
            for j in range(len(list(corresponding_points)[0])):
                position_index=list(corresponding_points)[0][j]
                if total_points[position_index,1]== region_points[index,1]:
                    region_intensity[position_index]=1.0

        else:
            region_intensity[list(corresponding_points)[0][0]]=1.0
    
    rgb_array, lut = colormap(region_intensity, cmap) 
            
    return rgb_array, lut, region_intensity

def pcd_to_tiff(point_cloud, intensity_array, xdim_px, ydim_px, zdim_px, x_res, y_res, z_res, spot_diameter, image_name):
    
    """ Takes each point of a point cloud and creates spots of a given pixel intensity given by intensity array. This is then exported as a tiff file. 
    Parameters:
    -----------
    
    """
    export_pcd=np.asarray(point_cloud.points)
    export_image=np.zeros([zdim_px,xdim_px,ydim_px])
    
    export_x_location=np.round((export_pcd[:,0]-np.min(export_pcd[:,0]))*x_resolution).astype('int')
    export_y_location=np.round((export_pcd[:,1]-np.min(export_pcd[:,1]))*y_resolution).astype('int')
    export_z_location=np.round((export_pcd[:,2]-np.min(export_pcd[:,2]))*z_resolution).astype('int')
    
    pixel_radius=int(np.round(x_resolution*(spot_diameter/2)))
    
    for i in range(len(export_pcd)):
        print(len(export_pcd)-i,'remaining')
        center_point=export_x_location[i],export_y_location[i],export_z_location[i]
        for x in range(center_point[0]-pixel_radius,center_point[0]+pixel_radius+1):
            for y in range(center_point[1]-pixel_radius, center_point[1]+pixel_radius+1):
                for z in range(center_point[2]-pixel_radius,center_point[2]+pixel_radius+1):
                    length=np.sqrt((x-center_point[0])**2+(y-center_point[1])**2+(z-center_point[2])**2)
                
                if length<=int(np.round(pixel_radius)):
                    export_image[z,x,y]=intensity_array[i]
                    
    tiff.imwrite(image_name,export_image.astype("uint8"), compression='jpeg')
    print('Saved',image_name,'as Tiff')
    
    return None
    
