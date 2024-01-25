# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 16:18:59 2023

@author: lopez
"""

# Imports useful libraries.
from skimage import io, img_as_float, filters
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar


def calculate_Pearsons(image1,image2):
    """Calculates the Pearson's coefficient."""
    X = image1.reshape(-1) # Transforms the images from matrix of data into a row of data. The same as .flatten()
    Y = image2.reshape(-1)
    X_bar = np.average(X) # Calculates the average pixel value in the first image (channel 1).
    Y_bar = np.average(Y) # Calculates the average pixel value in the first image (channel 2).
    R = np.sum((X-X_bar)*(Y-Y_bar))/(np.sqrt(np.sum((X-X_bar)**2)*np.sum((Y-Y_bar)**2))) # Calculates the Pearson's function. 
    return R

# Imports the image.
img = img_as_float(io.imread('C:/Users/lopez/Desktop/Paper Christine/convallaria_tiff.tif'))

# Selects a subset of the image, img_1 is Fast Green and img_2 is Safranin.  
img_1 = img[:,0,200:300,200:300]
img_2 = img[:,1,200:300,200:300]

# Creates binary images and intersection maps.
thresh_1 = filters.threshold_otsu(img_1)
thresh_2 = filters.threshold_otsu(img_2)
img_1_binary = img_1 > thresh_1
img_2_binary = img_2 > thresh_2
intersection_and = img_1_binary & img_2_binary
intersection_or = img_1_binary & img_2_binary

# Creates an RGB image for display.
rgb = np.zeros((251,100,100,3))
rgb[:,:,:,0] = img_1
rgb[:,:,:,2] = img_1
rgb[:,:,:,1] = img_2

# Initialises the slice variables. 
slice_initial = 115
slice_final = 135

# Creates empty lists to store the Pearson's coefficients and number of slices.
Rs_all = []
Rs_and = []
Rs_or = []
num_slices = []

# Implements a loop to populate the lists above.
for i in range(116):
    slices = slice_final-slice_initial
    num_slices.append(slices)
    img1 = img_1[slice_initial:slice_final,:,:]
    img2 = img_2[slice_initial:slice_final,:,:]
    int_and = intersection_and[slice_initial:slice_final,:,:]
    int_or = intersection_or[slice_initial:slice_final,:,:]
    R_all = calculate_Pearsons(img1,img2)
    Rs_all.append(R_all)
    R_and = calculate_Pearsons(img1[int_and],img2[int_and])
    Rs_and.append(R_and)
    R_or = calculate_Pearsons(img1[int_or],img2[int_or])
    Rs_or.append(R_or)
    slice_initial -= 1 
    slice_final += 1
    

# Plots the figure.
fig = plt.figure()
fig.set_figheight(16.9)
fig.set_figwidth(16.9)
ax1 = plt.subplot2grid(shape=(2, 3), loc=(0, 0), rowspan=2, colspan=1)
ax2 = plt.subplot2grid(shape=(2, 3), loc=(0, 1), rowspan=1, colspan=2)
ax3 = plt.subplot2grid(shape=(2, 3), loc=(1, 1), rowspan=1, colspan=2)
ax1.imshow(rgb[:,50,:]*5)
ax1.axhline(115,color='red',linestyle='dashed',linewidth=3)
ax1.axhline(135,color='red',linestyle='dashed',linewidth=3)
ax1.text(2,15,'A',color='white',size=45)
ax1.tick_params(axis='both', which='major', labelsize=22)
ax1.set_ylabel('Z',size=28)
ax1.set_xlabel('Y',size=28)
scalebar = AnchoredSizeBar(ax1.transData, 22, label='', loc='lower right', pad=0.1, borderpad=1, color='white', frameon=False, size_vertical=3) # The scale bar is 50 microns in lenght (pixel size is 2.28 microns).
ax1.add_artist(scalebar)
ax2.imshow(rgb[125,:,:]*5)
ax2.text(1,7,'B',color='white',size=45)
ax2.set_ylabel('Y',size=28)
ax2.set_xlabel('X',size=28)
ax2.tick_params(axis='both', which='major', labelsize=22)
scalebar = AnchoredSizeBar(ax2.transData, 22, label='', loc='lower right', pad=0.1, borderpad=1, color='white', frameon=True, size_vertical=1) # The scale bar is 50 microns in lenght (pixel size is 2.28 microns).
ax2.add_artist(scalebar)
ax3.scatter(num_slices,Rs_all,s=100,marker='o',edgecolors='r',facecolor='white', label="$ PCC_{ALL}$")
ax3.set_ylabel("$ PCC_{ALL}$",size=28, color='red')
ax3.set_xlabel('Z-stack size / slices',size=28)
ax3.text(10,0.418,'C',color='black',size=45)
ax3.tick_params(axis='y', which='major', labelsize=22, color='red',labelcolor='red')
ax3.tick_params(axis='x', which='major', labelsize=22, color='black',labelcolor='black')
ax3.legend(loc='lower right', bbox_to_anchor=(0.95, 0.05), fontsize=28, frameon=False, labelcolor='red')
ax4 = ax3.twinx()
ax4.set_ylabel("$ PCC_{OR}$", color='blue', size=28)
ax4.scatter(num_slices,Rs_or,s=100,marker='o',edgecolors='blue',facecolor='white', label="$ PCC_{OR}$")
ax4.tick_params(axis='y', which='major', labelsize=22, color='blue',labelcolor='blue')
ax4.legend(loc='lower right', bbox_to_anchor=(0.95, 0.15), fontsize=28, frameon=False, labelcolor='blue')
plt.tight_layout()
plt.savefig('Figure_3.tiff',dpi=300)
plt.show()





    
    
    

    
