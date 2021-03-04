import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as col
from scipy.interpolate import interp1d
from skimage.morphology import dilation
from scipy.interpolate import interp1d
import tifffile.tifffile as tiff
import matplotlib.pyplot as plt

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


def mae_permutation(num_permute, color_array):
    mae_list = []
   
    for i in range(num_permute):
        np.random.seed(i)
        x = np.random.permutation(color_array)
        x = np.reshape(x, color_array.shape)
        mae_val = (sum(abs(x - color_array))) / (x.shape[0])
        mae_val_rounded = round(float(mae_val),3) 
        mae_list.append(mae_val_rounded)
    return mae_list


def pcd_to_tif(pcd,intensity_array, width, height, depth, filename = "test.tif", dtype = "int", selem_color = np.ones((3,6,6)), selem_DAPI = np.ones((3,6,6)), return_DAPI = True, verbose = True):
    pcd_points = np.asarray(pcd.points)
    x_lim = [np.min(pcd_points[:,0]), np.max(pcd_points[:,0]) ]
    y_lim = [np.min(pcd_points[:,1]), np.max(pcd_points[:,1]) ]
    z_lim = [np.min(pcd_points[:,2]), np.max(pcd_points[:,2])]

    x_interp = interp1d(x_lim, [0,width-1])
    y_interp = interp1d(y_lim, [0,height-1])
    z_interp = interp1d(z_lim, [0,depth-1])

    data_x_interp = np.round(x_interp(pcd_points[:,0]))
    data_y_interp = np.round(y_interp(pcd_points[:,1]))
    data_z_interp = np.round(z_interp(pcd_points[:,2]))
    
    combined_zyxc = np.vstack([data_z_interp.squeeze(), data_x_interp.squeeze(), data_y_interp.squeeze(), intensity_array.squeeze()]).astype(dtype)
    export_image=np.zeros([depth, width, height])
    dapi_image = np.zeros([depth, width, height])
    
    if return_DAPI:
        for i in range(combined_zyxc.shape[1]):
            export_image[combined_zyxc[0][i], combined_zyxc[1][i], combined_zyxc[2][i]] = combined_zyxc[3][i]
            dapi_image[combined_zyxc[0][i], combined_zyxc[1][i], combined_zyxc[2][i]] = np.max(combined_zyxc[3])
        dilated = dilation(export_image, selem = selem_color)
        dilated_dapi = dilation(dapi_image, selem = selem_DAPI)
        
        tiff.imwrite(filename, dilated.astype("uint8"), compression='jpeg')
        if verbose:
            print(f"Saving color channel as {filename}")
            print("Saving DAPI channel as {}".format("DAPI_"+filename))
        tiff.imwrite(("DAPI_"+filename), dilated_dapi.astype("uint8"), compression='jpeg')
                        

    else:        
        for i in range(combined_zyxc.shape[1]):
            export_image[combined_zyxc[0][i], combined_zyxc[1][i], combined_zyxc[2][i]] = combined_zyxc[3][i]
            dilated = dilation(export_image, selem = selem_color)
        tiff.imwrite(filename, dilated.astype("uint8"), compression='jpeg')
    
    return None

def visualise_pcd(pcd1, figsize = (10,10), color1 = 'k', s = 50, pcd2=None, color2='r'):
    fig = plt.figure(figsize = figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    if pcd2==None:
        pcd_points = np.asarray(pcd1.points)
        try:
            pcd_color = np.asarray(pcd1.colors)
            ax.scatter(xs = pcd_points[:,0], ys = pcd_points[:,1], zs = pcd_points[:,2], color= pcd_color, s = s)
        except:
            ax.scatter(xs = pcd_points[:,0], ys = pcd_points[:,1], zs = pcd_points[:,2], color= color1, s = s)
            
    else:
        pcd1_points=np.asarray(pcd1.points)
        pcd2_points=np.asarray(pcd2.points)
        
        try:
            pcd1_color=np.asarray(pcd1.colors)
            pcd2_color=np.asarray(pcd2.colors)
            ax.scatter(xs = pcd1_points[:,0], ys = pcd1_points[:,1], zs = pcd1_points[:,2], color= pcd1_color, s = s)
            ax.scatter(xs = pcd2_points[:,0], ys = pcd2_points[:,1], zs = pcd2_points[:,2], color= pcd2_color, s = s)
        except:
            ax.scatter(xs = pcd1_points[:,0], ys = pcd1_points[:,1], zs = pcd1_points[:,2], color= color1, s = s)
            ax.scatter(xs = pcd2_points[:,0], ys = pcd2_points[:,1], zs = pcd2_points[:,2], color= color2, s = s)
