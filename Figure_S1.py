# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 10:58:56 2023

@author: lopez
"""

# Imports useful libraries.
import matplotlib.pyplot as plt
import numpy as np
from skimage import img_as_float, io, filters
from scipy.stats import gaussian_kde


def generate_cyto(img1,img2,method='all',thresh1=None,thresh2=None):
    """This function calculates the probability density function for the cytofluorogram of imaages 1 and 2."""
    im1_flat = img1.flatten()
    im2_flat = img2.flatten()
    if method == 'all': # implements the PCCall pixel selection criterion.
        list_pixels = list(zip(im1_flat,im2_flat))[::50]
        x = [i[0] for i in list_pixels]
        y = [i[1] for i in list_pixels]
        xy = np.vstack([x,y])
        z = gaussian_kde(xy)(xy)
        return x,y,z
    elif method =='or': # implements the PCCor pixel selection criterion.
        list_pixels = list(zip(im1_flat,im2_flat))[::10]
        list_cleaned = [i for i in list_pixels if i[0] > thresh1 or i[1] > thresh2]
        x = [i[0] for i in list_cleaned]
        y = [i[1] for i in list_cleaned]
        xy = np.vstack([x,y])
        z = gaussian_kde(xy)(xy)
        return x,y,z
    elif method =='and': # implements the PCCor pixel selection criterion.
        list_pixels = list(zip(im1_flat,im2_flat))
        list_cleaned = [i for i in list_pixels if i[0] > thresh1 and i[1] > thresh2]
        x = [i[0] for i in list_cleaned]
        y = [i[1] for i in list_cleaned]
        xy = np.vstack([x,y])
        z = gaussian_kde(xy)(xy)
        return x,y,z
    
# We import the images.
img = img_as_float(io.imread('C:/Users/lopez/Desktop/Paper Christine/Posible biological images/Candidate 1/fig2e1.tif'))
img1 = img[:,:,0]
img2 = img[:,:,1]

# We obtain the thresholds.
thresh1 = filters.threshold_otsu(img1)
thresh2 = filters.threshold_otsu(img2)

# We calculate the probability density functions and the linear fits for PCCall.
x_all,y_all,z_all = generate_cyto(img1,img2)
coef_all = np.polyfit(x_all,y_all,1)
poly1d_fn_all = np.poly1d(coef_all) 

# We calculate the probability density functions and the linear fits for PCCor.
x_or, y_or, z_or = generate_cyto(img1,img2,method='or',thresh1=thresh1,thresh2=thresh2)
coef_or = np.polyfit(x_or,y_or,1)
poly1d_fn_or = np.poly1d(coef_or) 

# We calculate the probability density functions and the linear fits for PCCand.
x_and, y_and, z_and = generate_cyto(img1,img2,method='and',thresh1=thresh1,thresh2=thresh2)
coef_and = np.polyfit(x_and,y_and,1)
poly1d_fn_and = np.poly1d(coef_and) 

# Creates an X-axis to extend the fit so as to match the figure axes.
x_full_all = np.linspace(0.0,1.0,len(x_all))
x_full_and = np.linspace(0.0,1.0,len(x_and))
x_full_or = np.linspace(0.0,1.0,len(x_or))

# We plot the figure.
fig = plt.figure()
fig.set_figheight(15)
fig.set_figwidth(16.9)
ax1 = plt.subplot2grid((2,4),(0,0),colspan=2) 
ax2 = plt.subplot2grid((2,4),(0,2),colspan=2) 
ax3 = plt.subplot2grid((2,4),(1,1),colspan=2)
ax1.scatter(x_all,y_all,c=z_all,s=1,cmap='hot')
ax1.plot(x_full_all, poly1d_fn_all(x_full_all), '--b',linewidth=2)
ax1.set_xlim(0.0,1.0)
ax1.set_ylim(0.0,1.0)
ax1.text(0.03,0.92,'$ PCC_{ALL}$',color='black',size=28,bbox=dict(boxstyle="round",color='grey'))
ax1.text(0.93,0.91,'A',color='black',size=45)
ax1.tick_params(axis='both', which='major', labelsize=22)
ax1.set_ylabel('BRCA1 (green)',size=28)
ax1.set_xlabel('Smad3/p53 (magenta)',size=28)
ax2.scatter(x_or[::5],y_or[::5],c=z_or[::5],s=1,cmap='hot')
ax2.plot(x_full_or, poly1d_fn_or(x_full_or), '--b',linewidth=2)
ax2.set_xlim(0.0,1.0)
ax2.set_ylim(0.0,1.0)
ax2.text(0.03,0.92,'$ PCC_{OR}$',color='black',size=28,bbox=dict(boxstyle="round",color='grey'))
ax2.text(0.93,0.91,'B',color='black',size=45)
ax2.set_ylabel('BRCA1 (green)',size=28)
ax2.set_xlabel('Smad3/p53 (magenta)',size=28)
ax2.tick_params(axis='both', which='major', labelsize=22)
ax3.scatter(x_and[::5],y_and[::5],c=z_and[::5],s=1,cmap='hot')
ax3.plot(x_full_and, poly1d_fn_and(x_full_and), '--b',linewidth=2)
ax3.set_xlim(0.0,1.0)
ax3.set_ylim(0.0,1.0)
ax3.set_ylabel("BRCA1 (green)",size=28, color='black')
ax3.set_xlabel('Smad3/p53 (magenta)',size=28)
ax3.text(0.03,0.92,'$ PCC_{AND}$',color='black',size=28,bbox=dict(boxstyle="round",color='grey'))
ax3.text(0.93,0.91,'C',color='black',size=45)
ax3.tick_params(axis='y', which='major', labelsize=22, color='red',labelcolor='black')
ax3.tick_params(axis='x', which='major', labelsize=22, color='black',labelcolor='black')
plt.tight_layout()
plt.savefig('Figure_S1.tiff',dpi=300)
plt.show()








    

    
    