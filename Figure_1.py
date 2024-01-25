# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 13:35:47 2023

@author: lopez
"""

# Imports useful libraries.
import numpy as np
from math import sqrt, exp
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage import img_as_float, filters
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar


# Functions designed to create 2D Gaussian curves at different locations.
def distance(point1,point2):
    """This function calculates the Euclidean distance between two points."""
    return sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

def gaussian_add(image,diameter,center,intensity):
    """This function adds the Gaussian. The input image must be an image with a background of 0."""
    base = np.zeros(image.shape[:2])
    rows, cols = image.shape[:2]
    for x in range(cols):
        for y in range(rows):
            base[y,x] = exp(((-distance((y,x),center)**2)/(2*(diameter**2))))
    image_processed = image + base*intensity
    return image_processed

def calculate_Pearsons(image1,image2):
    """Calculates the Pearson's coefficient."""
    X = image1.reshape(-1) # Transforms the images from matrix of data into a row of data. The same as .flatten()
    Y = image2.reshape(-1)
    X_bar = np.average(X) # Calculates the average pixel value in the first image (channel 1).
    Y_bar = np.average(Y) # Calculates the average pixel value in the first image (channel 2).
    R = np.sum((X-X_bar)*(Y-Y_bar))/(np.sqrt(np.sum((X-X_bar)**2)*np.sum((Y-Y_bar)**2))) # Calculates the Pearson's function. 
    return R

def calculate_Pearsons_biological(image1,image2,show=False):
    """Calculates the Pearsons' coefficient for an experimental image using the Otsu thresholding algorithm.
    It returns the ALL, AND, and OR coefficients in that order and, finally, the RGB image."""
    img1 = img_as_float(image1)
    img2 = img_as_float(image2)
    # The ALL criterion.
    all_R = calculate_Pearsons(img1,img2)
    # The AND criterion.
    img1_thresh = filters.threshold_otsu(img1)
    img2_thresh = filters.threshold_otsu(img2)
    img1_mask = img1 > img1_thresh
    img2_mask = img2 > img2_thresh
    intersection_and = img1_mask & img2_mask
    and_R = calculate_Pearsons(img1[intersection_and],img2[intersection_and])
    # The OR criterion.
    intersection_or = img1_mask | img2_mask
    or_R = calculate_Pearsons(img1[intersection_or],img2[intersection_or])
    # Displays the RGB image.
    full_RGB = generate_RGB(img1,img2)
    if show:
        thresh_RGB = generate_RGB(img1_mask,img2_mask)
        fig,ax = plt.subplots(1,2,figsize=(16,8))
        ax[0].imshow(full_RGB)
        ax[0].axis('off')
        ax[1].imshow(thresh_RGB)
        ax[1].axis('off')
        plt.tight_layout()
        plt.show()
    return all_R, and_R, or_R, full_RGB, intersection_and, intersection_or
    
def generate_RGB(image1,image2):
    """This function creates an RGB image out of two images."""
    assert image1.shape == image2.shape
    if len(image1.shape) >= 3:
        rgb = np.zeros((image1.shape[0],image1.shape[1],image1.shape[2],3))
        rgb[0,:,:,0] = image1[0,:,:] 
        rgb[0,:,:,1] = image2[0,:,:]
        rgb[0,:,:,2] = image1[0,:,:] 
    else:
        rgb = np.zeros((image1.shape[0],image1.shape[1],3))
        rgb[:,:,0] = image1
        rgb[:,:,1] = image2
        rgb[:,:,2] = image1
    return rgb 

def generate_mock_spheres(shape=(1000,1000),diameter=50,center1=[(600,400)],center2=[(700,300)],intensity=0.5):
    """This function generates mock images and calculates their Pearson's coefficient in 
    three different way."""
    img1 = np.zeros(shape)
    img2 = np.zeros(shape)
    for i in range(len(center1)):
        img1 = gaussian_add(img1,diameter=diameter,center=center1[i],intensity=intensity)
    for i in range(len(center2)):
        img2 = gaussian_add(img2,diameter=diameter,center=center2[i],intensity=intensity)
    # The All criterion.
    img1 = np.clip(img1,0.0,1.0)
    img2 = np.clip(img2,0.0,1.0)
    All_R = calculate_Pearsons(img1,img2)
    rgb = generate_RGB(img1,img2)
    # The AND criterion.
    thresh_1 = filters.threshold_otsu(img1)
    thresh_2 = filters.threshold_otsu(img2)
    img1_mask = img1 > thresh_1
    img2_mask = img2 > thresh_2
    intersection_AND = img1_mask & img2_mask
    AND_R = calculate_Pearsons(img1[intersection_AND],img2[intersection_AND])
    # The OR criterion.
    intersection_OR = img1_mask | img2_mask
    OR_R = calculate_Pearsons(img1[intersection_OR],img2[intersection_OR])
    return All_R, OR_R, AND_R, intersection_OR, intersection_AND, rgb


# Creates list of coordinates for the green and red spheres.
center1 = [(100,100),(400,150),(800,750),(500,500),(100,900)]
center2 = [(900,450),(700,200),(200,550),(500,500),(500,900)]
All_R, OR_R, AND_R, intersection_OR, intersection_AND, rgb = generate_mock_spheres(center1=center1,center2=center2)

# Imports the biological image.
img = imread('C:/Users/lopez/Desktop/Paper Christine/Posible biological images/Candidate 1/fig2e1.tif') 
img1 = img[:,:,0]
img2 = img[:,:,1]

# Gets the coefficients for the biological image.
all_R, and_R, or_R, full_RGB, intersection_and, intersection_or = calculate_Pearsons_biological(img1,img2)

# Displays the figure. 
fig, ax = plt.subplots(3,2,figsize=(16.9,25.35))
ax[0,0].imshow(rgb*2)
ax[0,0].axis('off')
ax[0,0].text(10,980,'A',color='white',size=45)
ax[0,0].text(620,850,"$ PCC_{ALL}$ = "+str(round(All_R,2)),color='white',size=28)
ax[0,0].text(620,910,"$ PCC_{AND}$ = "+str(round(AND_R,2))+str(0),color='white',size=28)
ax[0,0].text(620,970,"$ PCC_{OR} = $"+str(round(OR_R,2)),color='white',size=28)
ax[0,1].imshow(intersection_AND, cmap='Greys_r')
ax[0,1].axis('off')
ax[0,1].text(10,980,'B',color='white',size=45)
ax[0,1].text(570,960,"Mask for $ PCC_{AND}$",color='white',size=28)
ax[1,0].imshow(intersection_OR, cmap='Greys_r')
ax[1,0].axis('off')
ax[1,0].text(10,980,'C',color='white',size=45)
ax[1,0].text(600,960,"Mask for $ PCC_{OR}$",color='white',size=28)
ax[1,1].imshow(full_RGB)
ax[1,1].axis('off')
ax[1,1].text(10,2095,'D',color='white',size=45)
ax[1,1].text(30,110,"$ PCC_{ALL}$ = "+str(round(all_R,2)),color='white',size=28)
ax[1,1].text(30,240,"$ PCC_{AND}$ = "+str(round(and_R,2))+str(0),color='white',size=28)
ax[1,1].text(30,370,"$ PCC_{OR} = $"+str(round(or_R,2)),color='white',size=28)
scalebar = AnchoredSizeBar(ax[1,1].transData, int(full_RGB.shape[0]*0.27), label='', loc='lower right', pad=0.1, borderpad=1, color='white', frameon=False, size_vertical=20)
ax[1,1].add_artist(scalebar)
ax[2,0].imshow(intersection_and, cmap='Greys_r')
ax[2,0].axis('off')
ax[2,0].text(10,2095,'E',color='white',size=45)
ax[2,0].text(1640,2000,"Mask for $ PCC_{AND}$",color='black',size=28,ha="center", va="center", bbox=dict(boxstyle="round", facecolor='grey', alpha=1.0))
ax[2,1].imshow(intersection_or, cmap='Greys_r')
ax[2,1].axis('off')
ax[2,1].text(10,2095,'F',color='white',size=45)
ax[2,1].text(1670,2000,"Mask for $ PCC_{OR}$",color='black',size=28,ha="center", va="center", bbox=dict(boxstyle="round", facecolor='grey', alpha=1.0))
plt.tight_layout()
plt.savefig('Figure_1.tiff',dpi=300)
plt.show()

# Prints the coefficients.
print('The ALL, AND, and OR coefficients for the biological example are {},{},{}.'.format(all_R,and_R,or_R))
print('The ALL, AND, and OR coefficients for the spheres are {},{},{}.'.format(All_R,AND_R,OR_R))


