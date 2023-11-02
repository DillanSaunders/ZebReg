import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def draw_scatter_pcd(pcd, ax = None, manual_color = False, **plt_kwargs):
    """Plotting positions of points"""
    
    pcd_pts = np.asarray(pcd.points)
    
    if ax is None:
        ax = plt.gca()
        
    if manual_color == False:
        try:
            pcd.colors

        except:
            print("Colors not found for point cloud object. Check if assign_colors function has been run.")
        
        ax.scatter(xs = pcd_pts[:,0], ys = pcd_pts[:,1], zs = pcd_pts[:,2], color=pcd.colors, **plt_kwargs)

    else:
        ax.scatter(xs = pcd_pts[:,0], ys = pcd_pts[:,1], zs = pcd_pts[:,2], **plt_kwargs)

    
    
