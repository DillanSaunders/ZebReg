import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as col

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