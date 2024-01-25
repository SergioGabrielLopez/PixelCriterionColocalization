# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 12:54:23 2023

@author: lopez
"""

# Imports useful libraries.
import matplotlib.pyplot as plt
import numpy as np
from skimage import util
from math import sqrt, exp

# Functions designed to create 2D Gaussian curves at different locations.
def distance(point1,point2):
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

def generate_RGB(image1,image2):
    """This function creates an RGB image out of two images."""
    assert image1.shape == image2.shape
    rgb = np.zeros((image1.shape[0],image1.shape[1],3))
    rgb[:,:,0] = image1
    rgb[:,:,2] = image1
    rgb[:,:,1] = image2
    return rgb

def generate_mock(shape=(1000,1000),square1=[400,500],square2=[450,550],value=0.8,threshold=0.1, Noise=False):
    """This function generates mock images and calculates their Pearson's coefficient in 
    three different way."""
    if Noise == False:
        img1 = np.zeros(shape)
        img2 = np.zeros(shape)
        img1[square1[0]:square1[1],square1[0]:square1[1]] = value
        img2[square2[0]:square2[1],square2[0]:square2[1]] = value
        # The AND criterion.
        img1_mask = img1 > threshold
        img2_mask = img2 > threshold
        intersection = img1_mask & img2_mask
        AND_R = calculate_Pearsons(img1[intersection],img2[intersection])
        AND_RGB = generate_RGB(img1,img2)
        # The OR criterion.
        intersection_OR = img1_mask | img2_mask
        OR_R = calculate_Pearsons(img1[intersection_OR],img2[intersection_OR])
        # The ALL criterion.
        ALL_R = calculate_Pearsons(img1,img2)
    else:
        img1 = np.zeros(shape)
        img2 = np.zeros(shape)
        img1[square1[0]:square1[1],square1[0]:square1[1]] = value
        img2[square2[0]:square2[1],square2[0]:square2[1]] = value
        img1 = np.clip(util.random_noise(img1,mean=0.05,var=0.2),0,1)
        img2 = np.clip(util.random_noise(img2,mean=0.05,var=0.2),0,1)
        # The AND criterion.
        img1_mask = img1 > threshold
        img2_mask = img2 > threshold
        intersection = img1_mask & img2_mask
        AND_R = calculate_Pearsons(img1[intersection],img2[intersection])
        AND_RGB = generate_RGB(img1,img2)
        # The OR criterion.
        intersection_OR = img1_mask | img2_mask
        OR_R = calculate_Pearsons(img1[intersection_OR],img2[intersection_OR])
        # The ALL criterion.
        ALL_R = calculate_Pearsons(img1,img2)
    return ALL_R,AND_R,OR_R,AND_RGB


# Creates a set of lists to be populated.
ANDs_R = []
ORs_R = []
images = []
ALLs = []

# Creates a list of coordinates to use in the loop.
coor1 = [400,400,400]
coor2 = [480,560,720]
coor3 = [440,480,560]
coor4 = [520,640,880]

# Populates the lists.
for i in range(len(coor1)):    
    ALL_R,AND_R,OR_R,AND_RGB = generate_mock(square1=[coor1[i],coor2[i]],square2=[coor3[i],coor4[i]])
    ANDs_R.append(AND_R)
    ORs_R.append(OR_R)
    images.append(AND_RGB)
    ALLs.append(ALL_R)
    
# Creates a plot of R vs ratio of area covered by signal to total area.
x_1 = 2
y_1 = 2
Rs = []
area_single_square = []
images_simulation = []
for i in range(330):
    area_single_square.append(x_1*y_1)
    img_1_plot = np.zeros((1000,1000))
    img_2_plot = np.zeros((1000,1000))
    img_1_plot[0:int(x_1),0:int(y_1)] = 1
    img_2_plot[int(x_1/2):int(x_1+x_1/2),int(y_1/2):int(y_1+y_1/2)] = 1
    R = calculate_Pearsons(img_1_plot,img_2_plot)
    Rs.append(R)
    x_1 += 2
    y_1 += 2
    rgb = generate_RGB(img_1_plot,img_2_plot)
    images_simulation.append(rgb)
    
total_area = 1000*1000
area_double_square = np.array(area_single_square)*2    
ratio = area_double_square/total_area
Rs_normalised = np.array(Rs)/Rs[0]
    
# Displays the images.
fig,ax = plt.subplots(2,2,figsize=(16.9,15))
ax[0,0].imshow(images[0])
ax[0,0].text(550,60,"$ PCC_{AND} = $"+"%0.2f" % (1.00),color='w',fontsize=28)
ax[0,0].text(550,120,"$ PCC_{OR} = $"+"%0.2f" % (-0.75),color='w',fontsize=28)
ax[0,0].text(550,180,"$ PCC_{ALL} = $"+"%0.2f" % (ALLs[0]),color='w',fontsize=28)
ax[0,0].text(10,80,'A',color='white',size=45)
ax[0,0].axis('off') 
ax[0,1].imshow(images[1])
ax[0,1].text(550,60,"$ PCC_{AND} = $"+"%0.2f" % (1.00),color='w',fontsize=28)
ax[0,1].text(550,120,"$ PCC_{OR} = $"+"%0.2f" % (-0.75),color='w',fontsize=28)
ax[0,1].text(550,180,"$ PCC_{ALL} = $"+"%0.2f" % (ALLs[1]),color='w',fontsize=28)
ax[0,1].text(10,80,'B',color='white',size=45)
ax[0,1].axis('off')
ax[1,0].imshow(images[2])
ax[1,0].text(550,60,"$ PCC_{AND} = $"+"%0.2f" % (1.00),color='w',fontsize=28)
ax[1,0].text(550,120,"$ PCC_{OR} = $"+"%0.2f" % (-0.75),color='w',fontsize=28)
ax[1,0].text(550,180,"$ PCC_{ALL} = $"+"%0.2f" % (ALLs[2]),color='w',fontsize=28)
ax[1,0].text(10,80,'C',color='white',size=45)
ax[1,0].axis('off')
ax[1,1].set_xlabel('Object area / image area',size=28)
ax[1,1].set_ylabel("Normalised "+"$ PCC_{ALL}$",size=28)
ax[1,1].plot(ratio,Rs_normalised,'r--', linewidth=3)
ax[1,1].tick_params(labelsize=20)
ax[1,1].text(0.845,0.9,'D',color='black',size=45)
plt.tight_layout()
plt.savefig('Figure_2.tiff',dpi=300)
plt.show() 